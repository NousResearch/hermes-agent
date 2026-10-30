"""One-shot in-process agent runs: a peer ``AIAgent`` conversation, closed before returning.

The parameter surface is the contract plugins see through ``ctx.agent_loop`` (and the model sees
through the ``agent`` tool): only what a caller can know — the user message, an optional
context block, an optional system message, and the overrides it owns (model, provider, api_mode,
toolsets, iteration budget, history). Everything else is resolved here the way any host surface
resolves a run:

* model          — ``model.default``/``model.model`` from config, else the provider's default;
* credentials    — :func:`hermes_cli.runtime_provider.resolve_runtime_provider` for the requested
                   route (the main route when no provider is pinned), never a borrowed key;
* reasoning      — ``resolve_reasoning_config(config, model)`` for the *final* model;
* the agent      — one ``AIAgent`` per run, ``platform="subagent"`` (a peer is unattended and
                   hidden from human session pickers), every run closed in ``finally``.

Every run also appends to its native history (:mod:`agent.run_transcript`)::

    <HERMES_HOME>/sessions/<agent>/<session>.jsonl

where ``<agent>`` is ``layer`` (or ``meta["agent"]``) and ``<session>`` is ``history_session``
(the conversation that dispatched the run), falling back to the run's own ``session_id``.
``record_history=False`` opts out; the caller then owns persistence.

The defaults are a peer's posture: no host memory, no project context files, no background review
fork and no operator soul identity. A custom-agent caller flips each one explicitly.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from . import run_transcript

logger = logging.getLogger(__name__)

#: Tool-call budget for one peer run.
MAX_ITERATIONS = 10

#: Runtime keys forwarded from the resolved route to ``AIAgent`` — the same set the gateway
#: forwards from ``_resolve_runtime_agent_kwargs()``, minus ``model`` (resolved separately).
_RUNTIME_KEYS = ("api_key", "base_url", "api_mode", "credential_pool", "request_overrides",
                 "requested_provider", "capabilities", "command", "args")

#: Per-turn counters copied from the host result; names match the turn finalizer.
_USAGE_KEYS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "reasoning_tokens", "prompt_tokens", "completion_tokens", "total_tokens",
    "estimated_cost_usd", "cost_status", "cost_source",
)

#: Tools a read-only peer must not hold: it reads and advises, it does not mutate.
READ_ONLY_DROPPED_TOOLS = frozenset({
    "write_file", "patch", "agent", "skill_manage",
})

#: Two runs can land in the same millisecond for the same agent; run ids must still differ.
_SEQUENCE = itertools.count(1)


@dataclass(frozen=True)
class AgentRunResult:
    """One run's answer, how it was produced, and where its history landed."""

    answer: str
    api_calls: int
    model: str
    provider: str
    toolsets: tuple[str, ...]
    #: The conversation loop's messages, tool calls included. Empty when the host returned none.
    messages: tuple[dict[str, Any], ...] = ()
    #: Token/cost counters the host attached to the turn result; empty when none were reported.
    usage: dict[str, Any] = field(default_factory=dict)
    transcript_path: str = ""
    transcript_written: bool = False

    def as_dict(self) -> dict[str, Any]:
        """The tool/plugin JSON face: ``messages`` stay out (they are history, not an answer)."""
        return {"answer": self.answer, "api_calls": self.api_calls, "model": self.model,
                "provider": self.provider, "toolsets": list(self.toolsets),
                "usage": dict(self.usage), "transcript_path": self.transcript_path,
                "transcript_written": self.transcript_written}


def _config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        return load_config_readonly() or {}
    except Exception:  # health: allow BLE001 -- an unreadable config must not break a run
        logger.debug("agent_run: config unreadable", exc_info=True)
        return {}


def _resolve_route(model: str, provider: str) -> tuple[str, dict[str, Any]]:
    """``(model, runtime)`` for the final route; a pinned provider resolves its own credentials."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    requested = str(provider or "").strip() or None
    chosen = str(model or "").strip()
    runtime = dict(resolve_runtime_provider(requested=requested,
                                            target_model=chosen or None) or {})
    if not chosen:
        model_cfg = _config().get("model") or {}
        chosen = str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
    if not chosen:
        chosen = str(runtime.get("model") or "").strip()
    if not chosen and runtime.get("provider"):
        from hermes_cli.models import get_default_model_for_provider

        chosen = get_default_model_for_provider(str(runtime["provider"])) or ""
    return chosen, runtime


def _agent_kwargs(runtime: Mapping[str, Any], model: str) -> dict[str, Any]:
    """Credential/provider kwargs for the final route, plus its reasoning config."""
    from hermes_constants import resolve_reasoning_config

    knobs = {key: runtime.get(key) for key in _RUNTIME_KEYS if key in runtime}
    knobs["reasoning_config"] = resolve_reasoning_config(_config(), model)
    return knobs


def _prompt(message: str, context: str = "") -> str:
    return message if not context else f"{message}\n\n--- context ---\n{context}"


def _system_segments(system_message: Any) -> list[str]:
    """Non-empty system segments; a legacy string is one segment."""
    if isinstance(system_message, str):
        raw = [system_message]
    elif isinstance(system_message, Sequence) and not isinstance(system_message, (bytes, bytearray)):
        raw = [str(part) for part in system_message]
    else:
        return []
    return [part.strip() for part in raw if str(part).strip()]


def _usage(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: result[key] for key in _USAGE_KEYS if result.get(key) is not None}


def _transcript(result: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """The host's messages as plain dicts; anything that is not a message shape is skipped."""
    rows: list[dict[str, Any]] = []
    for item in result.get("messages") or ():
        if isinstance(item, Mapping):
            rows.append(dict(item))
            continue
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            converted = to_dict()
            if isinstance(converted, Mapping):
                rows.append(dict(converted))
    return tuple(rows)


def _with_reasoning(rows: tuple[dict[str, Any], ...],
                    reasoning: Any) -> tuple[dict[str, Any], ...]:
    """Attach a top-level ``reasoning_content`` to the run's last assistant message."""
    text = str(reasoning or "").strip()
    if not text:
        return rows
    for index in range(len(rows) - 1, -1, -1):
        row = dict(rows[index])
        if str(row.get("role") or "") != "assistant":
            continue
        if not (row.get("reasoning_content") or row.get("reasoning")):
            row["reasoning_content"] = text
        return rows[:index] + (row,) + rows[index + 1:]
    return rows + ({"role": "reasoning", "content": text},)


def apply_read_only(agent: Any) -> list[str]:
    """Strip mutating tools from an already-built agent; returns what was removed."""
    dropped: list[str] = []
    tools = getattr(agent, "tools", None)
    if isinstance(tools, list):
        kept: list[Any] = []
        for tool in tools:
            name = ""
            if isinstance(tool, Mapping):
                name = str((tool.get("function") or {}).get("name") or "")
            if name in READ_ONLY_DROPPED_TOOLS:
                dropped.append(name)
                continue
            kept.append(tool)
        agent.tools = kept
    names = getattr(agent, "valid_tool_names", None)
    if isinstance(names, (set, frozenset, list)):
        agent.valid_tool_names = {name for name in names if name not in READ_ONLY_DROPPED_TOOLS}
    return sorted(set(dropped))


def _configure_tool_guardrails(agent: Any, settings: Mapping[str, Any]) -> None:
    """Give one peer run its own loop guard; a host without the controller leaves the run alone."""
    try:
        from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController

        agent._tool_guardrails = ToolCallGuardrailController(
            ToolCallGuardrailConfig.from_mapping(dict(settings), platform="subagent"))
    except Exception:  # health: allow BLE001 -- the guard is an improvement, never a precondition
        logger.debug("agent_run: tool guardrails not applied", exc_info=True)


def _account(session_id: str, task: str, result: AgentRunResult) -> None:
    """Bill the run to the conversation its caller scoped, under the caller's task name."""
    if not session_id or not task or not result.usage:
        return
    try:
        from hermes_state_registry import acquire, release_or_close

        db = acquire()
    except Exception:  # health: allow BLE001 -- no ledger available is not a run failure
        logger.debug("agent_run: no session store for accounting", exc_info=True)
        return
    try:
        db.record_auxiliary_usage(
            session_id, task, model=result.model, billing_provider=result.provider,
            input_tokens=int(result.usage.get("input_tokens") or result.usage.get("prompt_tokens") or 0),
            output_tokens=int(result.usage.get("output_tokens") or result.usage.get("completion_tokens") or 0),
            cache_read_tokens=int(result.usage.get("cache_read_tokens") or 0),
            cache_write_tokens=int(result.usage.get("cache_write_tokens") or 0),
            reasoning_tokens=int(result.usage.get("reasoning_tokens") or 0),
            estimated_cost_usd=result.usage.get("estimated_cost_usd"),
            api_call_count=max(1, int(result.api_calls or 1)))
    except Exception:  # health: allow BLE001 -- accounting is best-effort by contract
        logger.debug("agent_run: accounting write failed", exc_info=True)
    finally:
        release_or_close(db)


def _record_history(agent_name: str, history_session: str, messages: tuple[dict[str, Any], ...],
                    *, run_id: str, model: str, provider: str, system: str,
                    usage: Mapping[str, Any]) -> tuple[str, bool]:
    """Write this run's history; a write failure is reported, never raised into a finished run."""
    path = run_transcript.path_for(agent_name, history_session)
    try:
        info = run_transcript.record(agent_name, history_session, messages, run_id=run_id,
                                     model=model, provider=provider, system_message=system,
                                     usage=usage)
        return str(info["path"]), bool(info["written"])
    except Exception:  # health: allow BLE001 -- the answer stands; only the audit trail failed
        logger.warning("agent_run: history write failed for %s/%s", agent_name,
                       history_session, exc_info=True)
        return str(path), False


def run(message: str, *, context: str = "", model: str = "", provider: str = "",
        api_mode: str = "", toolsets: Optional[Sequence[str]] = None,
        system_message: str | Sequence[str] = "", max_iterations: int = MAX_ITERATIONS,
        project_dir: Optional[str] = None, all_tools: bool = False, read_only: bool = False,
        history: Optional[Sequence[Mapping[str, Any]]] = None, session_id: str = "",
        layer: str = "", history_session: str = "", account_task: str = "",
        skip_memory: bool = True, skip_context_files: bool = True,
        skip_background_review: bool = True, load_soul_identity: bool = False,
        extra_tools: Optional[Sequence[Mapping[str, Any]]] = None, cache_scope: str = "",
        tool_guardrails: Optional[Mapping[str, Any]] = None,
        meta: Optional[Mapping[str, Any]] = None, record_history: bool = True) -> AgentRunResult:
    """Run one agent loop on *message* and return its answer. Never leaves an agent open.

    ``history`` is the prior conversation for a continuing peer; system rows are dropped (the
    host owns the system prompt). ``session_id`` keeps this run's host session distinct from the
    caller's; an empty value lets the host mint one. ``layer`` names the run for history and
    ``meta`` carries identifiers only. ``all_tools`` passes every host tool; otherwise
    ``toolsets`` is the exact surface (an empty list is "no tools").
    """
    from run_agent import AIAgent  # first: the managed-environment bootstrap replaces sys.path

    final_model, runtime = _resolve_route(model, provider)
    knobs = _agent_kwargs(runtime, final_model)
    if str(api_mode or "").strip():
        knobs["api_mode"] = str(api_mode).strip()
    final_provider = str(knobs.get("provider") or provider or runtime.get("provider") or "")
    tools = [str(tool) for tool in (toolsets or [])]
    conversation = [dict(row) for row in (history or ())
                    if isinstance(row, Mapping) and str(row.get("role") or "") != "system"]
    segments = _system_segments(system_message)
    system = "\n\n".join(segments)
    agent_name = str(layer or (meta or {}).get("agent") or "agent").strip() or "agent"
    history_key = str(history_session or session_id or "").strip()
    run_id = f"{agent_name}-{int(time.time() * 1000)}-{next(_SEQUENCE)}"
    agent = AIAgent(
        model=final_model,
        **knobs,
        session_id=str(session_id or "").strip() or None,
        enabled_toolsets=None if all_tools else tools,
        max_iterations=max(1, int(max_iterations or MAX_ITERATIONS)),
        quiet_mode=True,
        platform="subagent",
        cwd=str(project_dir) if project_dir else None,
        skip_context_files=bool(skip_context_files),
        skip_memory=bool(skip_memory),
        skip_background_review=bool(skip_background_review),
        load_soul_identity=bool(load_soul_identity),
        extra_tools=[dict(tool) for tool in (extra_tools or ())
                     if isinstance(tool, Mapping)],
        cache_scope=str(cache_scope or "").strip(),
    )
    try:
        if tool_guardrails:
            _configure_tool_guardrails(agent, tool_guardrails)
        if read_only:
            apply_read_only(agent)
        result = agent.run_conversation(
            _prompt(message, context), system_message=system or None,
            conversation_history=conversation or None)
        messages = _with_reasoning(
            _transcript(result), result.get("reasoning_content") or result.get("reasoning"))
        if not messages:
            # The host normally returns the loop's messages; keep at least the prompt and the
            # answer so the history still answers "what did this agent do?".
            messages = ({"role": "user", "content": message},
                        {"role": "assistant",
                         "content": str(result.get("final_response") or "")})
        usage = _usage(result)
        peer = AgentRunResult(
            answer=(result.get("final_response") or "").strip(),
            api_calls=int(result.get("api_calls") or 0),
            model=final_model,
            provider=final_provider,
            toolsets=("*",) if all_tools else tuple(tools),
            messages=messages,
            usage=usage,
        )
        if record_history and history_key:
            path, written = _record_history(agent_name, history_key, messages, run_id=run_id,
                                            model=final_model, provider=final_provider,
                                            system=system, usage=usage)
            peer = AgentRunResult(
                answer=peer.answer, api_calls=peer.api_calls, model=peer.model,
                provider=peer.provider, toolsets=peer.toolsets, messages=peer.messages,
                usage=peer.usage, transcript_path=path, transcript_written=written)
        _account(history_key, account_task, peer)
        return peer
    finally:
        # One agent per run in a long-lived host: close() releases tool subprocesses and HTTP
        # clients, otherwise every run leaks them.
        with contextlib.suppress(Exception):
            agent.close()


async def run_async(message: str, **kwargs: Any) -> AgentRunResult:
    """Await :func:`run` off the event loop (the agent loop is synchronous by design)."""
    return await asyncio.to_thread(run, message, **kwargs)


__all__ = [
    "MAX_ITERATIONS",
    "READ_ONLY_DROPPED_TOOLS",
    "AgentRunResult",
    "apply_read_only",
    "run",
    "run_async",
]
