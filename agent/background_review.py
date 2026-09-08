"""Background memory/skill review — fork the agent to evaluate the turn. After every turn
``AIAgent.run_conversation`` may spawn a daemon thread that replays the conversation snapshot in a
forked :class:`AIAgent` and asks "should any skill/memory be saved or updated?". Writes go
straight to the memory + skill stores; the main conversation and prompt cache are never touched.
The fork inherits the parent's live runtime (provider, model, credentials, cached system prompt)
so it hits the same prefix cache, and runs under a dispatch-side tool whitelist."""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from agent.thread_scoped_output import thread_scoped_silence

logger = logging.getLogger(__name__)

_BACKGROUND_REVIEW_CANCEL_TIMEOUT_SECONDS = 2.0


class _BackgroundReviewRun:
    """Per-review cancellation and request-completion handshake."""

    def __init__(self) -> None:
        self.cancel_requested = threading.Event()
        self.request_done = threading.Event()
        self._lock = threading.Lock()
        self._review_agent = None
        self._request_finished = self._cancel_dispatched = False

    def begin_request(self, review_agent: Any) -> bool:
        """Atomically admit the first provider-capable review phase."""
        with self._lock:
            if self.cancel_requested.is_set() or self._request_finished:
                return False
            self._review_agent = review_agent
            return True

    def cancel(self) -> Any:
        """Fence startup and return the running fork, if one was admitted."""
        with self._lock:
            self.cancel_requested.set()
            if self._review_agent is None or self._cancel_dispatched:
                return None
            self._cancel_dispatched = True
            return self._review_agent

    def mark_request_finished(self) -> bool:
        """Latch request completion once; the caller publishes the event."""
        with self._lock:
            if self._request_finished:
                return False
            self._request_finished, self._review_agent = True, None
            return True


@contextmanager
def _optional_lock(agent: Any, attr: str) -> Iterator[None]:
    """``with`` over a lock attribute that may be absent (direct test stubs)."""
    lock = getattr(agent, attr, None)
    if lock is None:
        yield
        return
    with lock:
        yield


def prepare_background_review_run(agent: Any) -> Optional[_BackgroundReviewRun]:
    """Install a unique run token on the parent before ``Thread.start()``."""
    run = _BackgroundReviewRun()
    try:
        lock = getattr(agent, "_background_review_lock", None)
        if lock is None:
            lock = agent._background_review_lock = threading.Lock()
        with lock:
            current = getattr(agent, "_background_review_run", None)
            if current is not None and not current.request_done.is_set():
                return None
            agent._background_review_run = run
    except (AttributeError, TypeError):
        return None
    return run


def finish_background_review_run(agent: Any, run: Optional[_BackgroundReviewRun]) -> None:
    """Publish one run's request exit without clearing a successor (ABA-safe)."""
    if run is None or not run.mark_request_finished():
        return
    with _optional_lock(agent, "_background_review_lock"):
        if getattr(agent, "_background_review_run", None) is run:
            agent._background_review_run = None
    run.request_done.set()


def _interrupt_background_review(review_agent: Any) -> None:
    """Request abort off-thread so a wedged abort hook cannot stall the live turn (the bounded
    ``request_done`` wait in the canceller relies on this returning fast)."""
    def _interrupt() -> None:
        try:
            from agent.interrupt_compat import request_hard_interrupt

            request_hard_interrupt(
                review_agent, "superseded by a new live turn", tool_reason="background review superseded"
            )
        except Exception:
            logger.debug("Failed to cancel in-flight background review for a new turn", exc_info=True)

    try:
        threading.Thread(target=_interrupt, daemon=True, name="bg-review-cancel").start()
    except Exception:
        logger.debug("Failed to start background-review cancellation thread", exc_info=True)


def cancel_background_review_for_live_turn(agent: Any) -> None:
    """Cancel the current review and await its request-phase acknowledgement. Foreground priority:
    past the bounded deadline, warn and let the live turn proceed — self-improvement work must
    never block a user-facing turn.

    Foreground priority is preserved: if the review does not acknowledge within the bounded deadline, a
    warning is logged and the live turn proceeds anyway. See #84423.
    """
    with _optional_lock(agent, "_background_review_lock"):
        run = getattr(agent, "_background_review_run", None)
        legacy_agent = getattr(agent, "_background_review_agent", None)
    review_agent = legacy_agent if run is None else run.cancel()
    if review_agent is not None:
        _interrupt_background_review(review_agent)
    if run is None:
        return
    if not run.request_done.wait(timeout=_BACKGROUND_REVIEW_CANCEL_TIMEOUT_SECONDS):
        logger.warning(
            "Background review did not acknowledge cancellation within %.1fs; "
            "proceeding with foreground live turn",
            _BACKGROUND_REVIEW_CANCEL_TIMEOUT_SECONDS,
        )


_REVIEW_MAX_ITERATIONS = 16
_REVIEW_MAX_INPUT_TOKENS_DEFAULT = 600_000


def _task_block(cfg: Any) -> Dict[str, Any]:
    aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
    task = aux.get("background_review", {})
    return task if isinstance(task, dict) else {}


def _background_review_task_config(task_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if task_cfg is not None:
        return task_cfg if isinstance(task_cfg, dict) else {}
    try:
        from hermes_cli.config import load_config_readonly
        return _task_block(load_config_readonly())
    except Exception:
        return {}


def _review_input_token_budget(task_cfg: Optional[Dict[str, Any]] = None) -> Optional[int]:
    raw = _background_review_task_config(task_cfg).get("max_input_tokens", _REVIEW_MAX_INPUT_TOKENS_DEFAULT)
    try:
        budget = int(raw)
    except (TypeError, ValueError):
        budget = _REVIEW_MAX_INPUT_TOKENS_DEFAULT
    return budget if budget > 0 else None


def load_background_review_settings() -> tuple[bool, Dict[str, Any]]:
    try:
        from hermes_cli.config import load_config_readonly
        from utils import is_truthy_value
        task = _task_block(load_config_readonly())
        return is_truthy_value(task.get("enabled"), default=True), task
    except Exception:
        logger.warning(
            "Failed to read background_review.enabled; leaving automatic "
            "review enabled (fail-open)",
            exc_info=True,
        )
        return True, {}


def is_background_review_enabled(task_cfg: Optional[Dict[str, Any]] = None) -> bool:
    if task_cfg is not None:
        try:
            from utils import is_truthy_value
            return is_truthy_value(task_cfg.get("enabled"), default=True)
        except Exception:
            logger.warning(
                "Failed to interpret background_review.enabled; leaving "
                "automatic review enabled (fail-open)",
                exc_info=True,
            )
            return True
    enabled, _ = load_background_review_settings()
    return enabled


def _resolve_review_runtime(agent: Any, task_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    parent_runtime = agent._current_main_runtime()
    parent_api_mode = parent_runtime.get("api_mode") or None
    parent = {
        "provider": agent.provider, "model": agent.model,
        "api_key": parent_runtime.get("api_key") or None, "base_url": parent_runtime.get("base_url") or None,
        "api_mode": "codex_responses" if parent_api_mode == "codex_app_server" else parent_api_mode,
        "credential_pool": getattr(agent, "_credential_pool", None),
        "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        "max_tokens": getattr(agent, "max_tokens", None), "command": getattr(agent, "acp_command", None),
        "args": list(getattr(agent, "acp_args", []) or []), "routed": False,
    }
    task = _background_review_task_config(task_cfg)
    task_provider, task_model, task_base_url, task_api_key = (
        str(task.get(key, "")).strip() or None for key in ("provider", "model", "base_url", "api_key")
    )
    if not (task_provider and task_provider != "auto" and task_model) or (
        task_provider == (agent.provider or "") and task_model == (agent.model or "")
    ):
        return parent
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        rp = resolve_runtime_provider(
            requested=task_provider, target_model=task_model,
            explicit_api_key=task_api_key, explicit_base_url=task_base_url,
        )
        return {
            "provider": rp.get("provider") or task_provider, "model": rp.get("model") or task_model,
            **{key: rp.get(key) for key in ("api_key", "base_url", "api_mode", "credential_pool", "command")},
            "request_overrides": dict(rp.get("request_overrides") or {}),
            "args": list(rp.get("args") or []), "routed": True,
        }
    except Exception as e:
        logger.debug("background-review aux routing failed (%s); using main model", e)
        return parent


def _parent_can_emit_tool_calls(agent: Any) -> bool:
    client = getattr(agent, "client", None)
    for candidate in (client, type(client) if client is not None else None):
        supported = getattr(candidate, "SUPPORTS_HERMES_TOOL_CALLS", None)
        if candidate is not None and supported is not None:
            return bool(supported)
    return True


def _msg_text(m: Dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
    return c.strip() if isinstance(c, str) else ""


def _digest_history(messages_snapshot: List[Dict], tail: int = 24) -> List[Dict]:
    msgs = list(messages_snapshot or [])
    while len(msgs) > tail:
        keep = msgs[-tail:]
        if not (isinstance(keep[0], dict) and keep[0].get("role") == "tool"):
            break
        tail += 1
    else:
        return msgs
    lines: List[str] = []
    for m in msgs[:-len(keep)]:
        if not isinstance(m, dict):
            continue
        role, text = m.get("role"), _msg_text(m).replace("\n", " ")
        if role == "user" and text:
            lines.append(f"USER: {text[:300]}")
        elif role == "assistant":
            if m.get("tool_calls"):
                names = [(tc.get("function") or {}).get("name", "?") for tc in m["tool_calls"] if isinstance(tc, dict)]
                lines.append(f"ASSISTANT[tools: {', '.join(names)}]")
            if text:
                lines.append(f"ASSISTANT: {text[:200]}")
    digest = (
        "[Earlier conversation digest — older turns summarised to bound the "
        "review's cold-write cost on the routed aux model. Recent turns "
        "follow verbatim below.]\n" + "\n".join(lines)
    )
    return [{"role": "user", "content": digest}] + keep


_MEMORY_REVIEW_PROMPT = (
    "Review the conversation above and consider saving to memory if appropriate.\n\n"
    "Focus on:\n"
    "1. Has the user revealed things about themselves — their persona, desires, preferences, or "
    "personal details worth remembering?\n"
    "2. Has the user expressed expectations about how you should behave, their work style, or ways "
    "they want you to operate?\n\n"
    "If something stands out, save it using the memory tool. If nothing is worth saving, just say "
    "'Nothing to save.' and stop."
)

_SKILL_REVIEW_PROMPT = "Review the conversation above and update the skill library as appropriate. If nothing stands out, say 'Nothing to save.' and stop."

_COMBINED_REVIEW_PROMPT = "Review the conversation above and update memory and skills as appropriate. If nothing stands out, say 'Nothing to save.' and stop."


def summarize_background_review_actions(
    review_messages: List[Dict], prior_snapshot: List[Dict], notification_mode: str = "on"
) -> List[str]:
    mode = str(notification_mode or "on").lower()
    if mode == "off":
        return []
    existing_ids = {m.get("tool_call_id") for m in prior_snapshot or [] if isinstance(m, dict) and m.get("role") == "tool" and m.get("tool_call_id")}
    actions: List[str] = []
    for msg in review_messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if tcid and tcid in existing_ids:
            continue
        try:
            data = json.loads(msg.get("content", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict) or not data.get("success"):
            continue
        message = data.get("message", "")
        if message:
            actions.append(message)
    return actions


def build_memory_write_metadata(
    agent: Any, *, write_origin: Optional[str] = None, execution_context: Optional[str] = None,
    task_id: Optional[str] = None, tool_call_id: Optional[str] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "write_origin": write_origin or getattr(agent, "_memory_write_origin", "assistant_tool"),
        "execution_context": execution_context or getattr(agent, "_memory_write_context", "foreground"),
        "session_id": agent.session_id or "",
        "parent_session_id": agent._parent_session_id or "",
        "platform": agent.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
        "tool_name": "memory",
        "task_id": task_id or None,
        "tool_call_id": tool_call_id or None,
    }
    return {k: v for k, v in metadata.items() if v not in {None, ""}}


def _snapshot_review_usage(review_agent: Any) -> Dict[str, Any]:
    _USAGE_COUNTERS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "api_calls")
    return {
        **{key: getattr(review_agent, key, None) for key in ("model", "provider", "base_url")},
        **{key: int(getattr(review_agent, f"session_{key}", 0) or 0) for key in _USAGE_COUNTERS},
        "estimated_cost_usd": getattr(review_agent, "session_estimated_cost_usd", None),
    }


def _record_review_usage_to_parent(parent_agent: Any, usage: Dict[str, Any]) -> None:
    try:
        session_db = getattr(parent_agent, "_session_db", None)
        session_id = getattr(parent_agent, "session_id", None)
        counts = {key: int(usage.get(key) or 0) for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "api_calls")}
        if session_db is None or not session_id or not any(counts.values()):
            return
        session_db.record_auxiliary_usage(
            session_id, task="background_review", model=usage.get("model"),
            billing_provider=usage.get("provider"), billing_base_url=usage.get("base_url"),
            estimated_cost_usd=usage.get("estimated_cost_usd"),
            api_call_count=counts.pop("api_calls"), **counts,
        )
    except Exception as e:
        logger.debug("Background review usage recording failed (non-fatal): %s", e)


def _log_review_completion(usage: Dict[str, Any], result: str) -> None:
    logger.info(
        "Background review complete: thread=bg-review calls=%d in=%d out=%d cache_read=%d result=%s",
        int(usage.get("api_calls") or 0), int(usage.get("input_tokens") or 0),
        int(usage.get("output_tokens") or 0), int(usage.get("cache_read_tokens") or 0), result,
    )


def _same_model_parity_kwargs(agent: Any) -> Dict[str, Any]:
    _PROVIDER_PIN_ATTRS = (
        "providers_allowed", "providers_ignored", "providers_order", "provider_sort",
        "provider_require_parameters", "provider_data_collection",
    )
    kwargs: Dict[str, Any] = {
        "reasoning_config": getattr(agent, "reasoning_config", None),
        "ephemeral_system_prompt": getattr(agent, "ephemeral_system_prompt", None),
        **{attr: val for attr in _PROVIDER_PIN_ATTRS if (val := getattr(agent, attr, None))},
    }
    if parent_prefill := copy.deepcopy(getattr(agent, "prefill_messages", None) or []):
        kwargs["prefill_messages"] = parent_prefill
    return kwargs


def _detach_fork_compression(review_agent: Any) -> None:
    bind = getattr(getattr(review_agent, "context_compressor", None), "bind_session_state", None)
    detached = False
    if callable(bind):
        try:
            bind(session_db=None, session_id="")
            detached = True
        except Exception:
            logger.warning(
                "background-review compressor detachment failed; keeping compression DISABLED on this review fork",
                exc_info=True,
            )
    review_agent.compression_in_place = True
    review_agent.compression_enabled = detached
    if detached:
        review_agent._review_defer_compaction_before_first_response = True


def _fork_init_kwargs(agent: Any, rt: Dict[str, Any], routed: bool, max_iterations: int) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": rt.get("model") or agent.model, "max_iterations": max_iterations, "quiet_mode": True,
        "platform": agent.platform, "provider": rt.get("provider") or agent.provider,
        "api_mode": rt.get("api_mode"), "base_url": rt.get("base_url") or None,
        "api_key": rt.get("api_key") or None, "credential_pool": rt.get("credential_pool"),
        "request_overrides": rt.get("request_overrides") or {}, "parent_session_id": agent.session_id,
        "enabled_toolsets": getattr(agent, "enabled_toolsets", None),
        "disabled_toolsets": getattr(agent, "disabled_toolsets", None), "skip_memory": True,
    }
    if isinstance(rt.get("max_tokens"), int):
        kwargs["max_tokens"] = rt["max_tokens"]
    if isinstance(rt.get("command"), str) and rt["command"]:
        kwargs.update(acp_command=rt["command"], acp_args=rt.get("args") or [])
    if not routed:
        kwargs.update(_same_model_parity_kwargs(agent))
    return kwargs


def build_cache_parity_fork(
    agent: Any, task_cfg: Optional[Dict[str, Any]] = None, *, max_iterations: int,
    write_origin: str = "background_review",
) -> Tuple[Any, Dict[str, Any], bool]:
    from run_agent import AIAgent
    _rt = _resolve_review_runtime(agent, task_cfg)
    _routed = bool(_rt.get("routed"))
    review_agent = AIAgent(**_fork_init_kwargs(agent, _rt, _routed, max_iterations))
    review_agent._memory_write_origin = review_agent._memory_write_context = write_origin
    review_agent._memory_store = agent._memory_store
    review_agent._memory_enabled = agent._memory_enabled
    review_agent._user_profile_enabled = agent._user_profile_enabled
    review_agent._memory_nudge_interval = review_agent._skill_nudge_interval = 0
    review_agent._skip_mcp_refresh = review_agent._persist_disabled = review_agent.suppress_status_output = True
    review_agent._end_session_on_close = False
    review_agent._session_db = None
    review_agent.session_id = agent.session_id
    if not _routed:
        review_agent._cached_system_prompt = agent._cached_system_prompt
        review_agent.session_start = agent.session_start
    _detach_fork_compression(review_agent)
    review_agent._review_input_token_budget = _review_input_token_budget(task_cfg)
    return review_agent, _rt, _routed


def _bg_review_auto_deny(command, description, **kwargs):
    logger.warning("Background review auto-denied dangerous command: %s (%s)", command, description)
    return "deny"


def _set_thread_approval_callback(callback: Any) -> None:
    from tools.terminal_tool import set_approval_callback
    with suppress(Exception):
        set_approval_callback(callback)


def _track_review_fork(agent: Any, review_agent: Any, *, register: bool) -> None:
    if review_agent is None:
        return
    if hasattr(agent, "_background_review_agent"):
        with _optional_lock(agent, "_background_review_lock"):
            if register:
                agent._background_review_agent = review_agent
            elif agent._background_review_agent is review_agent:
                agent._background_review_agent = None
    if hasattr(agent, "_active_children"):
        with _optional_lock(agent, "_active_children_lock"):
            if register:
                agent._active_children.append(review_agent)
            else:
                with suppress(ValueError, AttributeError):
                    agent._active_children.remove(review_agent)


def _review_tool_whitelist(
    review_agent: Any, task_cfg: Optional[Dict[str, Any]], review_memory: bool = True,
) -> Tuple[set, set]:
    """``(whitelist, configured_extra_tools)`` for the review fork — DISPATCH-side only, so the
    advertised ``tools[]`` stays byte-identical to the parent's (prompt-cache parity).

    ``review_memory`` scopes the memory toolset to the trigger that actually justified it: a
    review spawned ONLY by the skills nudge (``review_memory=False``) must not receive
    ``add``/``replace``/``remove`` access to MEMORY.md, even when memory is enabled for the
    profile. Without this, an unattended skill-only review that happens to hit the memory tool's
    near-limit consolidation prompt can silently delete standing memory entries the user never
    asked it to touch — see #105921."""
    from model_tools import get_tool_definitions
    memory_on = review_memory and (review_agent._memory_enabled or review_agent._user_profile_enabled)
    review_toolsets = ["memory", "skills"] if memory_on else ["skills"]
    whitelist = {t["function"]["name"] for t in get_tool_definitions(enabled_toolsets=review_toolsets, quiet_mode=True)}
    whitelist |= {"read_file", "search_files"}
    configured_extra_tools: set = set()
    try:
        extra_raw = _background_review_task_config(task_cfg).get("extra_tools", [])
        if isinstance(extra_raw, list):
            configured_extra_tools = {name.strip() for name in extra_raw if isinstance(name, str) and name.strip()}
    except Exception:
        logger.debug("background_review extra_tools parse failed", exc_info=True)
    return whitelist | configured_extra_tools, configured_extra_tools


@dataclass
class _ReviewForkState:
    review_agent: Any = None
    review_messages: List[Dict] = field(default_factory=list)
    review_usage: Dict[str, Any] = field(default_factory=dict)


def _release_fork_clients(review_agent: Any) -> None:
    with suppress(Exception):
        review_agent.release_clients()


def _run_review_fork(
    agent: Any, messages_snapshot: List[Dict], prompt: str, task_cfg: Optional[Dict[str, Any]],
    review_run: Optional[_BackgroundReviewRun], st: _ReviewForkState, review_memory: bool = True,
) -> None:
    st.review_agent, _rt, _routed = build_cache_parity_fork(agent, task_cfg, max_iterations=_REVIEW_MAX_ITERATIONS)
    _track_review_fork(agent, st.review_agent, register=True)
    from hermes_cli.plugins import set_thread_tool_whitelist, clear_thread_tool_whitelist
    review_whitelist, configured_extra_tools = _review_tool_whitelist(st.review_agent, task_cfg, review_memory)
    extra_list = ", ".join(sorted(configured_extra_tools))
    deny_extra = f" Configured extra tools also allowed: {extra_list}." if configured_extra_tools else ""
    prompt_extra = f" Exception — these configured tools are also allowed: {extra_list}." if configured_extra_tools else ""
    set_thread_tool_whitelist(
        review_whitelist,
        deny_msg_fmt=(
            "Background review denied non-whitelisted tool: "
            "{tool_name}. Allowed here: skill_view/skills_list/read_file/search_files to read, "
            "skill_manage(action='patch'|...) to change skills, and "
            "memory for notes." + deny_extra + " Do not retry {tool_name}."
        ),
    )
    with suppress(Exception):
        from tools.skill_manager_guards import _reset_background_review_read_marks
        _reset_background_review_read_marks()
    try:
        if review_run is None or review_run.begin_request(st.review_agent):
            st.review_agent.run_conversation(
                user_message=(
                    prompt + "\n\nYou can only call memory and skill "
                    "management tools. Other tools will be denied "
                    "at runtime — do not attempt them." + prompt_extra
                ),
                conversation_history=_digest_history(messages_snapshot) if _routed else messages_snapshot,
            )
    finally:
        clear_thread_tool_whitelist()
        if st.review_agent is not None:
            st.review_usage.update(_snapshot_review_usage(st.review_agent))
            _record_review_usage_to_parent(agent, st.review_usage)
        _track_review_fork(agent, st.review_agent, register=False)
        finish_background_review_run(agent, review_run)
    st.review_messages = list(getattr(st.review_agent, "_session_messages", []))
    _release_fork_clients(st.review_agent)
    st.review_agent = None


def _publish_review_summary(agent: Any, actions: List[str]) -> None:
    summary = " · ".join(dict.fromkeys(actions))
    agent._safe_print(f"  💾 Self-improvement review: {summary}")
    if agent.background_review_callback:
        with suppress(Exception):
            agent.background_review_callback(f"💾 Self-improvement review: {summary}")


def _run_review_in_thread(
    agent: Any, messages_snapshot: List[Dict], prompt: str,
    task_cfg: Optional[Dict[str, Any]] = None, review_run: Optional[_BackgroundReviewRun] = None,
    review_memory: bool = True,
) -> None:
    """``review_memory`` records whether this review was actually triggered for memory review
    (vs. skills-only) so the fork's tool whitelist can scope memory write access accordingly
    (#105921)."""
    if review_run is not None and review_run.cancel_requested.is_set():
        finish_background_review_run(agent, review_run)
        return
    _set_thread_approval_callback(_bg_review_auto_deny)
    if not _parent_can_emit_tool_calls(agent) and not _resolve_review_runtime(agent, task_cfg).get("routed"):
        logger.warning(
            "Background review skipped: provider %r cannot emit Hermes tool calls, "
            "so the review fork could not write memories or skills. Set "
            "auxiliary.background_review.{provider,model} to route the review to a normal model.",
            getattr(agent, "provider", "?"),
        )
        _set_thread_approval_callback(None)
        return
    st = _ReviewForkState()
    try:
        with thread_scoped_silence():
            _run_review_fork(agent, messages_snapshot, prompt, task_cfg, review_run, st, review_memory)
        try:
            actions = summarize_background_review_actions(
                st.review_messages, messages_snapshot,
                notification_mode=getattr(agent, "memory_notifications", "on"),
            )
        except Exception as e:
            logger.warning(
                "summarize_background_review_actions returned partial results after exception "
                "(treating as empty): %s", e,
            )
            actions = []
        _log_review_completion(st.review_usage, "skill" if actions else "none")
        if actions:
            _publish_review_summary(agent, actions)
    except Exception as e:
        logger.warning("Background memory/skill review failed: %s", e)
        if st.review_usage:
            _log_review_completion(st.review_usage, "error")
        agent._emit_auxiliary_failure("background review", e)
    finally:
        _track_review_fork(agent, st.review_agent, register=False)
        finish_background_review_run(agent, review_run)
        if st.review_agent is not None:
            with suppress(Exception), thread_scoped_silence():
                _release_fork_clients(st.review_agent)
        _set_thread_approval_callback(None)


_PROMPT_NAME_BY_SCOPE = {
    (True, True): "_COMBINED_REVIEW_PROMPT", (True, False): "_MEMORY_REVIEW_PROMPT",
    (False, True): "_SKILL_REVIEW_PROMPT", (False, False): "_SKILL_REVIEW_PROMPT",
}


def spawn_background_review_thread(
    agent: Any, messages_snapshot: List[Dict], review_memory: bool = False,
    review_skills: bool = False, focus: Optional[str] = None,
    task_cfg: Optional[Dict[str, Any]] = None, review_run: Optional[_BackgroundReviewRun] = None,
):
    """``focus`` (``/refine [instructions]``) may explicitly ask about memory even when
    ``review_memory`` is False; once the user directly drives the review, honor their intent
    over the automatic-trigger scoping (#105921)."""
    if task_cfg is None:
        task_cfg = _background_review_task_config()
    name = _PROMPT_NAME_BY_SCOPE[(review_memory, review_skills)]
    prompt = getattr(agent, name, globals()[name])
    if focus := (focus or "").strip():
        prompt = (
            f"{prompt}\n\nThe user explicitly requested this review with the following "
            f"focus — prioritize it over the general instructions above:\n{focus}"
        )
    effective_review_memory = review_memory or bool(focus)

    def _target() -> None:
        _run_review_in_thread(
            agent, messages_snapshot, prompt, task_cfg=task_cfg, review_run=review_run,
            review_memory=effective_review_memory,
        )

    return _target, prompt


__all__ = [
    "_MEMORY_REVIEW_PROMPT", "_SKILL_REVIEW_PROMPT", "_COMBINED_REVIEW_PROMPT",
    "is_background_review_enabled", "load_background_review_settings",
    "spawn_background_review_thread", "summarize_background_review_actions", "build_memory_write_metadata",
]
