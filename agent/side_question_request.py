"""One text-only request on a detached cache-parity fork of the live session agent, for plugins
(``ctx.side_question``).

The fork replays the turn snapshot captured at finalization (system prompt, ``tools[]``,
messages) verbatim plus one user prompt, so the request reads the parent's warm prompt cache.
It runs the turn loop's own request phases (assembly, the payload build with its LLM
middleware, ``perform_api_call``) exactly once: no tool dispatch (tool calls in the response
are dropped), no retries, no fallback model, no compression, no persistence. Like any other
side call it reports through ``pre/post_auxiliary_call`` (``aux_task="plugin_side_question"``
plus the asking ``plugin_id``), never the turn-scoped ``pre/post_api_request``, and its usage is
billed to the session the token was issued for as ``plugin_side_question:<plugin_id>`` (``/btw``
keeps ``side_question``).
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

SIDE_QUESTION_ORIGIN = "side_question"
# Hook/usage task for plugin side questions, distinct from /btw's ``side_question``.
PLUGIN_SIDE_QUESTION_TASK = "plugin_side_question"

# Tool entries the client executes; any other ``type`` (Responses ``web_search``, Anthropic
# ``web_search_20250305``, ...) runs server-side and could act without Hermes seeing it.
_CLIENT_TOOL_TYPES = frozenset({"function", "custom"})
# OpenRouter request plugins that only pick the model; anything else (``web``, ...) executes.
_ROUTING_ONLY_OPENROUTER_PLUGINS = frozenset({"pareto-router"})


@dataclass(frozen=True)
class SideQuestionCapture:
    """The parent's request inputs frozen at turn finalization (the live parent moves on).
    ``parent_ref`` is weak: a lease must not keep an evicted session's agent alive."""

    parent_ref: Any  # weakref.ref[AIAgent]
    messages: list[dict[str, Any]]
    system_prompt: Optional[str]
    system_prompt_static: Optional[str]
    tools: list[dict[str, Any]]
    session_id: str
    turn_id: str
    runtime: tuple  # runtime_key(parent) at capture
    context: contextvars.Context = field(repr=False)


def runtime_key(agent: Any) -> tuple:
    """What decides which model the fork talks to and how: a turn routed elsewhere (voice chat)
    restores the main runtime after ``post_llm_call``, and the captured prompt must not go out
    on it."""
    return tuple(getattr(agent, name, None) for name in ("model", "provider", "base_url", "api_mode", "reasoning_config"))


@dataclass
class _Audit:
    plugin_id: str
    started: float = field(default_factory=time.monotonic)
    outcome: str = "error"
    usage: dict[str, Any] = field(default_factory=dict)


def _agent_provider_refusal(agent: Any) -> Optional[str]:
    """Why ``agent``'s runtime cannot serve a text-only side request, or None.

    MoA fans out to advisor models; an agent-as-provider (ACP CLI shim, external-process
    provider, codex app-server, a client declaring ``SUPPORTS_HERMES_TOOL_CALLS = False``)
    runs its own tools on the remote side, which no client-side drop can stop."""
    from agent.background_review import _parent_can_emit_tool_calls
    from hermes_cli.runtime_provider_backends import _is_external_process_provider

    if agent.provider == "moa":
        return "moa"
    if (
        str(agent.base_url or "").lower().startswith(("acp://", "acp+tcp://"))
        or agent.api_mode == "codex_app_server"
        or _is_external_process_provider(agent.provider)
        or not _parent_can_emit_tool_calls(agent)
    ):
        return "agent_provider"
    return None


def has_server_side_tools(api_kwargs: dict[str, Any]) -> bool:
    """True when the FINAL request asks the provider to execute something itself: a tool entry
    with a non-client ``type`` (transports rewrite client tools, e.g. Codex/xAI swap a client
    ``web_search`` for ``{"type": "web_search"}``), OpenAI ``web_search_options``, or an
    OpenRouter plugin other than a routing-only one."""
    extra_body = api_kwargs.get("extra_body") if isinstance(api_kwargs.get("extra_body"), dict) else {}
    # ``request_overrides`` land in extra_body, whose ``tools`` the SDK merge sends over the top level.
    for tool in [*(api_kwargs.get("tools") or ()), *(extra_body.get("tools") or ())]:
        tool_type = tool.get("type") if isinstance(tool, dict) else None
        if tool_type and tool_type not in _CLIENT_TOOL_TYPES:
            return True
    if "web_search_options" in api_kwargs or "web_search_options" in extra_body:
        return True
    plugins = [*(api_kwargs.get("plugins") or ()), *(extra_body.get("plugins") or ())]
    return any(not (isinstance(p, dict) and p.get("id") in _ROUTING_ONLY_OPENROUTER_PLUGINS) for p in plugins)


def _build_fork(parent: Any, capture: SideQuestionCapture) -> Any:
    """A never-routed cache-parity fork pinned to the captured request inputs."""
    from agent.background_review import build_cache_parity_fork

    # task_cfg={} = never route: the side question must hit the parent's warm cache.
    fork, _rt, _routed = build_cache_parity_fork(
        parent, {}, max_iterations=1, write_origin=SIDE_QUESTION_ORIGIN,
    )
    fork._cached_system_prompt = capture.system_prompt
    # The static tier decides where the system-prompt cache breakpoint splits; rebuilding it on
    # the fork re-reads prompt files and can land the marker elsewhere.
    fork._cached_system_prompt_static = capture.system_prompt_static
    fork.tools = capture.tools
    fork.valid_tool_names = {tool["function"]["name"] for tool in capture.tools}
    fork.session_id = capture.session_id
    return fork


def _send(fork: Any, capture: SideQuestionCapture, prompt: str, audit: _Audit) -> Any:
    """Assemble, build and perform ONE request on the fork; the normalized reply, or None on
    refusal / a cancelled call."""
    from agent import relay_llm
    from agent.auxiliary_hooks import finish_aux_call_hooks, start_aux_call_hooks
    from agent.side_question import trim_snapshot_for_fork
    from agent.turn_api_call import perform_api_call
    from agent.turn_api_request import build_request_payload
    from agent.turn_context import _PER_TURN_RESET_STATE
    from agent.turn_request_assembly import assemble_api_request
    from agent.turn_retry_state import TurnRetryState
    from agent.turn_truncation import normalize_response_for_agent
    from agent.turn_usage import record_response_usage

    turn_id = f"{capture.turn_id}:{SIDE_QUESTION_ORIGIN}"
    task_id = str(uuid.uuid4())
    api_request_id = f"{turn_id}:api:1"
    # Turn state a fork that never ran a turn lacks: the per-turn counters the request phases
    # read, the replay-canonicalization clock (build_api_messages), the ids middleware and relay
    # key on, and a non-streaming transport.
    for name, value in _PER_TURN_RESET_STATE:
        setattr(fork, name, value)
    fork._current_turn_timestamp = time.time()
    fork._current_turn_id, fork._current_task_id, fork._current_api_request_id = turn_id, task_id, api_request_id
    fork._disable_streaming = True

    messages = trim_snapshot_for_fork(capture.messages) + [{"role": "user", "content": prompt}]
    assembled = assemble_api_request(
        fork, messages=messages, current_turn_user_idx=len(messages) - 1, _ext_prefetch_cache=None,
        _plugin_user_context=None, moa_config=None, active_system_prompt=fork._cached_system_prompt,
        original_user_message=prompt, pending_moa_prepared_request=None, request_logger=logger,
    )
    built = build_request_payload(
        fork, api_messages=assembled.api_messages, _moa_prepared_request=None,
        tools_for_api=assembled.tools_for_api, system_message=None, api_call_count=1,
        api_request_id=api_request_id, effective_task_id=task_id, turn_id=turn_id,
    )
    if has_server_side_tools(built.api_kwargs):
        audit.outcome = "refused:server_side_tools"
        return None
    hooks = start_aux_call_hooks(
        aux_task=PLUGIN_SIDE_QUESTION_TASK, metadata={"api_request_id": api_request_id, "plugin_id": audit.plugin_id},
        client=fork.client,
        kwargs=built.api_kwargs, provider=str(fork.provider or ""), model=str(fork.model or ""),
        api_mode=str(fork.api_mode or ""), streaming=False,
    )
    api_start_time = time.time()
    # One hook bracket around the whole attempt: a reply that fails after the provider returned
    # (accounting, normalization) still closes the observer's pre_auxiliary_call with its error.
    try:
        verdict = perform_api_call(
            fork, api_kwargs=built.api_kwargs, _original_api_kwargs=built._original_api_kwargs,
            _llm_middleware_trace=built._llm_middleware_trace, _moa_prepared_request=None,
            _retry=TurnRetryState(), thinking_spinner=None, retry_count=0, api_call_count=1,
            api_request_id=api_request_id, effective_task_id=task_id, turn_id=turn_id, interrupted=False,
        )
        response = None if verdict.action == "break" else verdict.response
        if response is not None:
            record_response_usage(
                fork, response, messages=messages, api_call_count=1,
                api_duration=time.time() - api_start_time, compression_attempts=0, max_compression_attempts=0,
            )
            assistant_message = normalize_response_for_agent(fork, response)
            # The reply is in the transport's native shape: observers get the normalized view.
            normalized = (assistant_message, assistant_message.finish_reason,
                          fork._usage_summary_for_api_request_hook(response))
    except BaseException as exc:
        finish_aux_call_hooks(hooks, error=exc)
        relay_llm.complete_logical_call(api_request_id, outcome="failed")
        raise
    if response is None:
        finish_aux_call_hooks(hooks, error=InterruptedError("side question cancelled"))
        relay_llm.complete_logical_call(api_request_id, outcome="cancelled")
        audit.outcome = "error"
        return None
    relay_llm.complete_logical_call(api_request_id, outcome="success")
    finish_aux_call_hooks(hooks, response, normalized=normalized)
    return assistant_message


def _answer_text(fork: Any, assistant_message: Any) -> Optional[str]:
    """Visible text of the reply; its ``tool_calls`` are dropped, never dispatched."""
    from agent.message_content import flatten_message_text

    content = assistant_message.content
    text = content if isinstance(content, str) else flatten_message_text(content)
    return fork._strip_think_blocks(text or "").strip() or None


def _log_audit(audit: _Audit) -> None:
    usage = audit.usage
    logger.info(
        "side_question plugin=%s outcome=%s calls=%d in=%d cache_read=%d cache_write=%d out=%d latency=%.1fs",
        audit.plugin_id or "?", audit.outcome,
        *(int(usage.get(key) or 0) for key in (
            "api_calls", "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens")),
        time.monotonic() - audit.started,
    )


def ask_text_only(
    capture: SideQuestionCapture, prompt: str, *, plugin_id: str = "", timeout: float = 15.0,
) -> Optional[str]:
    """Ask the session's own agent ``prompt`` after the captured turn; its text answer or None.

    None on refusal (MoA, agent-as-provider, server-side tools in the final request), timeout,
    provider error or an empty / tool-call-only reply. The request parameters (tools, reasoning,
    max_tokens, tool_choice) are the parent's verbatim: changing any of them changes the cache
    key, and an inherited thinking budget 400s against a smaller max_tokens."""
    from agent.background_review import release_cache_parity_fork

    audit = _Audit(plugin_id)
    parent = capture.parent_ref()
    refusal = "parent_gone" if parent is None else _agent_provider_refusal(parent)
    if refusal is not None:
        audit.outcome = f"refused:{refusal}"
        _log_audit(audit)
        return None
    fork = None
    built: list = []  # the fork once built, for the timer
    timed_out = threading.Event()

    def _on_timeout() -> None:
        timed_out.set()
        # Soft interrupt with a system reason: aborts the in-flight request without being
        # recorded as a user Stop.
        for target in built:
            target.interrupt(None, tool_reason="side question timed out")

    # The budget starts before the build: runtime resolution (credentials, client setup) is part
    # of the caller's wait. A build cannot be interrupted, so one that outlasts the budget is
    # caught right after it and sends nothing.
    timer = threading.Timer(timeout, _on_timeout)
    timer.daemon = True
    timer.start()
    try:
        fork = _build_fork(parent, capture)
        built.append(fork)
        if timed_out.is_set():
            audit.outcome = "timeout"
            return None
        if runtime_key(parent) != capture.runtime:
            audit.outcome = "refused:runtime_changed"
            return None
        assistant_message = _send(fork, capture, prompt, audit)
        if assistant_message is None:
            return None
        answer = _answer_text(fork, assistant_message)
        audit.outcome = "ok" if answer else "empty"
        return answer
    except Exception:
        # Boundary: a plugin's side question returns None, it never raises into the plugin.
        audit.outcome = "timeout" if timed_out.is_set() else "error"
        logger.warning("side question %s (plugin=%s)", audit.outcome, plugin_id or "?", exc_info=True)
        return None
    finally:
        timer.cancel()
        if fork is not None:
            # Billed to the session the token was issued for: the parent's id may have rotated
            # (``/compress``, a session reset) since the turn ended.
            audit.usage = release_cache_parity_fork(
                parent, fork, task=f"{PLUGIN_SIDE_QUESTION_TASK}:{plugin_id or '?'}",
                session_id=capture.session_id)
        _log_audit(audit)
