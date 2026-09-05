"""In-process MCP server exposing Hermes tools to the Claude Agent SDK loop.

Ported from PR #56413 (akshaynexus/hermes-agent, feat/claude-agent-sdk-inference,
`agent/claude_agent_sdk_adapter.py::_build_hybrid_mcp_server`, L496-559) onto
this branch's fcava-based `claude-agent-sdk` provider. Rationale: fcava
(PR #65982) exposes only the stdio ``hermes-tools`` subprocess (~25 curated
tools) — third-party MCPs configured in Hermes are invisible to the SDK.
akshaynexus solves this via an in-process MCP that routes every Hermes tool
through ``agent_runtime_helpers.invoke_tool``, so the SDK sees the full
registry + proxified MCPs.

Kept as its own module (rather than added to ``claude_agent_sdk_session.py``)
to make the port trivially removable: when either PR's approach lands in
upstream, delete this file and drop the wire-up.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent.transports.hermes_tool_exposure import (
    looks_like_tool_error,
    normalize_tool_spec,
)

logger = logging.getLogger(__name__)

# MCP server name used to expose the "extras" bucket in-process (proxified
# third-party MCPs + agent-level tools). Distinct from ``hermes-tools`` so
# the legacy stdio-parity bucket keeps operator grants intact.
HYBRID_SERVER = "hermes-hybrid"

# MCP server name preserved for backward compatibility with operator grants
# in ``~/.claude/settings.json`` that key on ``mcp__hermes-tools__<tool>``.
# When the hybrid bridge is active, the tools listed in
# ``HERMES_TOOLS_LEGACY_NAMES`` are exposed under this name so those grants
# keep matching without a migration step.
HERMES_TOOLS_SERVER = "hermes-tools"


def build_hybrid_mcp_server(
    agent,
    tools: List[dict],
    *,
    server_name: str = HYBRID_SERVER,
    only_names: Optional[Iterable[str]] = None,
    exclude_names: Optional[Iterable[str]] = None,
) -> Any:
    """Build an in-process MCP server exposing Hermes tools to the SDK loop.

    Each Hermes tool becomes an ``@tool`` whose handler routes through Hermes'
    unified single-tool entry ``invoke_tool`` — which runs the tool_request
    middleware + plugin block hooks and handles BOTH registry tools and
    agent-level tools (todo/memory/clarify/delegate_task/read_terminal/
    session_search) with live agent state — then wraps the result in the
    ``<untrusted_tool_result>`` promptware defense used by the native path.

    ``server_name`` sets the MCP server name Claude sees
    (``mcp__<server_name>__<tool>``). The bridge uses two buckets — see the
    call site in ``claude_agent_sdk_session.py`` — one under
    ``hermes-tools`` (stdio-legacy names, keeps existing operator grants
    intact) and one under ``hermes-hybrid`` (everything else).

    ``only_names`` (optional) restricts the exposed tools to that set,
    matched on the raw Hermes registry name (no ``mcp__`` prefix). Used to
    partition the registry between the two buckets.

    ``exclude_names`` (optional) drops the listed tools from this server.
    Applied AFTER ``only_names``. Used to honour the operator's
    ``agent.claude_agent_sdk.hybrid_mcp_bridge_exclude`` config: the stdio
    wrapper's curation was a security choice, and this list lets an
    operator keep the wide bridge for proxified MCPs without inheriting
    high-blast tools (delegate_task, cron_*, read_terminal, terminal).
    """
    # Lazy imports so this module stays cheap to import at package load.
    import claude_agent_sdk as sdk
    from agent.tool_dispatch_helpers import _maybe_wrap_untrusted
    from agent.agent_runtime_helpers import invoke_tool

    effective_task_id = (
        getattr(agent, "task_id", None) or getattr(agent, "_task_id", None) or ""
    )
    guardrails = getattr(agent, "_tool_guardrails", None)

    def _make_handler(tool_name: str):
        async def _handler(args: Dict[str, Any]) -> Dict[str, Any]:
            call_args = args or {}
            if guardrails is not None and hasattr(guardrails, "before_call"):
                try:
                    decision = guardrails.before_call(tool_name, call_args)
                    if decision is not None and not getattr(
                        decision, "allows_execution", True
                    ):
                        msg = getattr(decision, "message", None) or (
                            "Blocked by Hermes tool guardrail policy"
                        )
                        return {
                            "content": [{"type": "text", "text": msg}],
                            "is_error": True,
                        }
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "hybrid guardrail error for %s: %s", tool_name, exc
                    )
            try:
                raw = await asyncio.to_thread(
                    invoke_tool,
                    agent,
                    tool_name,
                    call_args,
                    effective_task_id,
                    tool_call_id=f"cas_{uuid.uuid4().hex[:12]}",
                )
            except Exception as exc:  # noqa: BLE001
                return {
                    "content": [{"type": "text", "text": f"Tool failed: {exc}"}],
                    "is_error": True,
                }
            text = raw if isinstance(raw, str) else json.dumps(raw, default=str)
            is_error = looks_like_tool_error(text)
            text = _maybe_wrap_untrusted(tool_name, text)
            return {
                "content": [{"type": "text", "text": text}],
                "is_error": is_error,
            }

        return _handler

    # Normalize + sort tools deterministically BEFORE SDK registration.
    # Rationale: the SDK bakes the registered tool order into the request's
    # `tools:` block sent to Anthropic. Any run-to-run reorder (dict iteration
    # order across restarts, plugin discovery order, etc.) invalidates the
    # prompt cache prefix — cache-hit rate collapses from ~90% to ~0. Sorting
    # by name gives a stable byte-identical block across restarts of the same
    # tool set. Idea ported from OpenClaw's cache-hygiene playbook (they
    # observed the same effect at scale, hence their deterministic ordering).
    # ``agent.tools`` (the OpenAI-format schemas from
    # ``get_tool_definitions``) is the internal registry format, NOT an
    # OAuth-wire payload — MCP tools are natively keyed as
    # ``mcp__<server>__<tool>`` here (see ``mcp_prefixed_tool_name``).
    # Passing ``strip_mcp_prefix=True`` would mangle those names into
    # unresolvable strings that ``invoke_tool``'s registry lookup rejects
    # silently — every third-party MCP call would return an "unknown tool"
    # JSON error while the SDK reports success (is_error=True content).
    only_set: Optional[set] = set(only_names) if only_names is not None else None
    exclude_set: set = set(exclude_names or ())

    normalized_specs: List[Tuple[str, str, Dict[str, Any]]] = []
    for spec in tools or []:
        normalized = normalize_tool_spec(spec, strip_mcp_prefix=False)
        if normalized is None:
            continue
        name = normalized[0]
        if only_set is not None and name not in only_set:
            continue
        if name in exclude_set:
            continue
        normalized_specs.append(normalized)
    normalized_specs.sort(key=lambda t: t[0])

    sdk_tools = []
    for name, description, input_schema in normalized_specs:
        try:
            decorated = sdk.tool(name, description, input_schema)(
                _make_handler(name)
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("hybrid: skipping tool %s (%s)", name, exc)
            continue
        sdk_tools.append(decorated)

    return sdk.create_sdk_mcp_server(
        name=server_name, version="1.0.0", tools=sdk_tools
    )
