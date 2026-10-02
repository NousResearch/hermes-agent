"""Plugin runtime dispatch contracts and shared execution metadata.

Hook, event, middleware, and prompt-section execution are runtime-owned. This module also
owns runtime-neutral contracts, limits, helpers, and observer schema metadata.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import inspect
import logging
import queue
import re
import threading
import types
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Set, Union

from plugin_runtime.config_bridge import read_hook_callback_timeout_seconds
from plugin_runtime.registration import PluginRegistration


logger = logging.getLogger("hermes_cli.plugins")

_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS = 30.0


def resolve_plugin_command_result(result: Any) -> Any:
    """Resolve a plugin command result, awaiting async handlers: ``asyncio.run`` when no loop is
    running, else a helper thread with its own loop (30s bound so a hung handler cannot wedge the
    terminal)."""
    if not inspect.isawaitable(result):
        return result
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)
    outcome: Dict[str, Any] = {}
    failure: Dict[str, BaseException] = {}
    done = threading.Event()

    def _runner() -> None:
        try:
            outcome["value"] = asyncio.run(result)
        except BaseException as exc:  # pragma: no cover - re-raised below
            failure["exc"] = exc
        finally:
            done.set()

    # copy_context: the helper thread must see the caller's profile/secret scope, else an
    # async hook under a running loop reads the default HERMES_HOME and get_secret raises.
    threading.Thread(target=contextvars.copy_context().run, args=(_runner,),
                     name="hermes-plugin-command-await", daemon=True).start()
    if not done.wait(timeout=_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS):
        raise TimeoutError("Plugin command async handler did not complete within "
                           f"{_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS:.0f}s")
    if "exc" in failure:
        raise failure["exc"]
    return outcome.get("value")

OBSERVER_SCHEMA_VERSION = "hermes.observer.v1"

TOOL_REQUEST_MIDDLEWARE = "tool_request"
TOOL_EXECUTION_MIDDLEWARE = "tool_execution"
LLM_REQUEST_MIDDLEWARE = "llm_request"
LLM_EXECUTION_MIDDLEWARE = "llm_execution"

VALID_MIDDLEWARE: set[str] = {
    TOOL_REQUEST_MIDDLEWARE,
    TOOL_EXECUTION_MIDDLEWARE,
    LLM_REQUEST_MIDDLEWARE,
    LLM_EXECUTION_MIDDLEWARE,
}

VALID_HOOKS: Set[str] = {
    "pre_tool_call", "post_tool_call", "transform_terminal_output", "transform_tool_result",
    # transform_llm_output: return a replacement string (first non-None wins) or None.
    "transform_llm_output", "pre_llm_call", "post_llm_call",
    # Streaming observers (agent.plugin_stream_hooks), off the token path; payloads are immutable
    # normalized text/lifecycle and cannot transform the stream.
    "on_stream_start", "on_stream_delta", "on_stream_end", "on_interim_message",
    # pre_verify: once per turn when the agent edited code and is about to verify/finish. Return
    # {"action": "continue", "message"} (or Claude-Code Stop {"decision": "block", "reason"}) to keep
    # going; anything else finishes. Bounded by agent.max_verify_nudges.
    "pre_verify", "pre_api_request", "post_api_request", "api_request_error",
    # pre/post_auxiliary_call: once per physical provider attempt of an auxiliary LLM call
    # (agent/auxiliary_hooks.py — titling, compression, MoA, vision, approval, ...). Same payload
    # shape as pre/post_api_request plus ``aux_task``; distinct events so turn-scoped
    # ``*_api_request`` subscribers never receive auxiliary traffic (#79733). Observers; fail-open.
    "pre_auxiliary_call", "post_auxiliary_call",
    # transform_api_error_classification: once per failed API call BEFORE
    # agent/error_classifier.classify_api_error(). Kwargs: provider, model, status_code, error_type,
    # error_code, error_message, error_body, error, approx_tokens, context_length, num_messages.
    # Return None or {"reason": <FailoverReason name> (required), "retryable"/"should_compress"/
    # "should_rotate_credential"/"should_fallback": bool, "message": str, "error_context": dict}.
    # Run-all-then-pick-first (see get_plugin_error_classification). Privacy: error_message/
    # error_body may be unredacted.
    "transform_api_error_classification", "on_session_start", "on_session_end",
    "on_session_finalize", "on_session_reset",
    # on_skill_lifecycle: successful skill lifecycle facts (local skill name visible to plugins).
    "on_skill_lifecycle", "subagent_start", "subagent_stop",
    # pre_gateway_dispatch: once per incoming MessageEvent, after the internal-event guard, BEFORE
    # auth/pairing and dispatch. Kwargs: event, gateway, session_store. Return {"action": "skip",
    # "reason"} -> drop; {"action": "rewrite", "text"} -> replace event.text; "allow"/None -> normal.
    "pre_gateway_dispatch",
    # agent_loop_stopped: an agent turn was interrupted mid-run (/stop, or the running-agent
    # fast-path of /new; see gateway/run.py::_interrupt_and_clear_session). Kwargs: session_key,
    # platform, reason, invalidation_reason. Return values are ignored.
    "agent_loop_stopped",
    # Approval observers (tools/approval.py); returns ignored — plugins cannot veto or pre-answer
    # (use pre_tool_call). Kwargs: command, description, pattern_key, pattern_keys, session_key,
    # surface: "cli"|"gateway"|"smart"; post_approval_response adds choice ("once"|"session"|
    # "always"|"deny"|"timeout"|"smart_approve"|"smart_deny") and decided_by.
    "pre_approval_request", "post_approval_response",
    # on_room_member_activity: a hosted Group Chat member's live runtime events (tool.started/completed,
    # request.opened, message.delta, reasoning.delta, turn.error, ...) stamped with room_id, thread_id,
    # member_id, turn_id, task_id, execution_generation. Observer, queued per consumer off the token
    # path (agent.plugin_stream_hooks); never written to the durable room log. Kwargs: those
    # coordinates + kind, seq, payload (the client-safe session event payload, approvals redacted).
    "on_room_member_activity",
    # pre_transcription: after provider resolution, BEFORE any backend runs. Kwargs: file_path,
    # provider, model, language, prompt, source. Return None or a dict mutating prompt/language/
    # model (registration order, last-writer-wins; file_path is read-only).
    "pre_transcription",
    # Kanban task observers (hermes_cli.kanban_db), fired AFTER the DB commit so a slow plugin never
    # holds the SQLite write lock; returns ignored. claimed fires in the DISPATCHER right before
    # spawn; completed/blocked fire in the WORKER (or whichever process drove it). Kwargs: task_id,
    # board, assignee, run_id, profile_name; completed adds summary, blocked adds reason.
    "kanban_task_claimed", "kanban_task_completed", "kanban_task_blocked",
    # Kanban worker/mutation/tick observers; returns ignored; fire sites short-circuit on
    # has_hook(). Kwargs: task_id, profile_name, board, assignee, run_id plus, per hook:
    # worker_spawned (DISPATCHER, after PID persisted, inside the dispatch lock — stay fast):
    #   worker_pid, workspace_path (privacy: project layout/usernames).
    # worker_exited (tick-derived on dead-PID reclaim): worker_pid, exit_kind ("clean_exit" |
    #   "rate_limited" | "nonzero_exit" | "signaled" | "unknown"), exit_code, outcome, retry_status.
    # worker_stale_claim (TTL-expired claim reclaimed; live-PID extensions do NOT fire):
    #   worker_pid, heartbeat_stale, retry_status.
    # task_updated (committed task-row write outside claim/complete/block, in whichever process
    #   committed it): changed_fields — field NAMES only, never values.
    # dispatch_tick (once per dispatch_once, strictly AFTER the dispatch lock is released): board,
    #   profile_name, dry_run, outcome ("ok"|"skipped_locked"|"idle"), result: DispatchResult
    #   (privacy: task ids, assignees, workspace paths).
    "on_kanban_worker_spawned", "on_kanban_worker_exited", "on_kanban_worker_stale_claim",
    "on_kanban_task_updated", "on_kanban_dispatch_tick",
    # gateway_platform_event: normalized envelopes only, never raw SDK objects or adapter handles.
    # Kwargs: platform, event_type, payload (event_type-local; see hooks.md). New event types land
    # only together with real fire-sites.
    # on_kanban_dispatch_tick fires once per dispatcher tick in dispatch_once, strictly AFTER the board's
    # single-writer dispatch lock has been released (the #56066 original fired inside the lock — the #64231
    # disposition mandates the post-lock re-port), so a slow subscriber can never extend the writer critical
    # section. Kwargs: board: str | None, profile_name: str, dry_run: bool, outcome: "ok" | "skipped_locked"
    # | "idle", result: hermes_cli.kanban_db.DispatchResult (spawned, reclaimed, promoted,
    # reconciled_orphans, crashed, stale, timed_out, auto_blocked, rate_limited, auto_assigned_default,
    # respawn_guarded, skipped_per_profile_capped, skipped_unassigned, skipped_nonspawnable,
    # skipped_locked). Privacy: result carries task ids, assignees, and workspace paths.
    # Gateway platform-boundary observer hooks (#64176). Observer-only; each callback isolated by
    # invoke_hook. This surface grants no adapter handles or platform actions. Fired today: Telegram
    # "reaction" + "message_edited"; Discord "message_edited", "message_deleted", "thread_created",
    # "thread_renamed". Each event type carries its own event-local additive payload contract (see
    # hooks.md). Other event types and hook names land here only together with real fire-sites and payload
    # contracts; no inert VALID_HOOKS surface is registered ahead of implementation.
    "gateway_platform_event",
    # pre_command: BEFORE a recognized slash command's handler on CLI and gateway canonical dispatch;
    # returns IGNORED in v1. Deliberately NOT fired for the gateway's running-agent intercept path
    # (/stop, /approve, busy_policy) — a slow/hostile plugin must not touch the operator's escape
    # hatches. Kwargs: surface, command (canonical), alias_used, args_raw, session_key, platform.
    # Slash-command dispatch observer (#64204, observer-first per #64182 ground rule 3). Return values are
    # IGNORED in v1 — a plugin returning a directive-shaped dict gets a debug log so future block/rewrite
    # adopters are discoverable once the middleware variant ships against the #64231 taxonomy.
    "pre_command",
}

# Hooks whose directive the shell-hook response parser has no channel for. VALID_HOOKS doubles as
# the shell-hook allow-list, so these are refused loudly instead of having output silently ignored.
SHELL_UNSUPPORTED_HOOKS: Set[str] = {"transform_api_error_classification"}

# Allowlist of agent-turn hot-path hooks bounded by plugins.hook_callback_timeout (fail-open:
# abandon without join — joining reintroduced a shutdown hang). Unlisted hooks run synchronously.
_HOOK_TIMEOUT_BOUNDED_HOOKS: Set[str] = {
    "post_tool_call", "transform_terminal_output", "transform_tool_result", "transform_llm_output",
    "pre_llm_call", "post_llm_call", "pre_api_request", "post_api_request", "api_request_error",
    "pre_auxiliary_call", "post_auxiliary_call", "pre_verify", "on_session_start", "on_session_end",
}

# Policy hooks: timeout / still-running must fail closed (block the tool).
_HOOK_TIMEOUT_FAIL_CLOSED_HOOKS: Set[str] = {"pre_tool_call"}
# Documented parent-thread serialization contract — never run on a timeout worker.
_HOOK_CALLER_THREAD_HOOKS: Set[str] = {"subagent_stop"}
# After a timeout, suppress the same callback this long so a hung hook cannot pile up threads.
_HOOK_TIMEOUT_SUPPRESSION_SECONDS = 60.0
# Live workers a hung callback may accumulate before it is skipped outright.
_HOOK_MAX_ABANDONED_WORKERS = 3
_PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE = "pre_tool_call plugin callback timed out or is still running"


def _policy_error_block_directive(
    hook_name: str, cb: Callable, exc: BaseException,
) -> Dict[str, str]:
    """Build the fail-closed directive for a policy hook callback failure."""
    callback_name = getattr(cb, "__name__", repr(cb))
    return {
        "action": "block",
        "message": (
            f"{hook_name} plugin callback {callback_name} raised "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        ),
    }


# System-prompt sections are tightly bounded: they become high-trust prompt bytes charged every turn.
SYSTEM_PROMPT_SECTION_POSITIONS = frozenset({"after_memory"})
DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS = 4_000
MAX_SYSTEM_PROMPT_SECTION_CHARS = 4_000
MAX_SYSTEM_PROMPT_SECTIONS = 32
MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS = 8_000
_SYSTEM_PROMPT_SECTION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SYSTEM_PROMPT_SECTION_HEADING_PREFIX = "## Plugin Context: "
PLUGIN_SECTIONS_START = "<!-- hermes-plugin-sections:start -->"
PLUGIN_SECTIONS_END = "<!-- hermes-plugin-sections:end -->"


def is_valid_system_prompt_section_id(value: Any) -> bool:
    """Return whether value is a stable, heading-safe section identifier."""
    return isinstance(value, str) and bool(_SYSTEM_PROMPT_SECTION_ID_RE.fullmatch(value))


def format_system_prompt_section(section_id: str, content: str) -> str:
    """Render an auditable, length-framed block recoverable from the full prompt."""
    return (
        f"{_SYSTEM_PROMPT_SECTION_HEADING_PREFIX}{section_id}\n"
        f"<!-- hermes-plugin-section-chars:{len(content)} -->\n\n{content}"
    )


def format_system_prompt_sections(sections: list) -> str:
    """Render the canonical container used for persistence recovery."""
    if not sections:
        return ""
    blocks = [format_system_prompt_section(item.id, item.content) for item in sections]
    return f"{PLUGIN_SECTIONS_START}\n" + "\n\n".join(blocks) + f"\n{PLUGIN_SECTIONS_END}"


# Reserved event namespace prefix — only core may publish hermes:<event>.
HERMES_EVENT_NAMESPACE = "hermes"
# Event recursion depth cap; over-deep emits are dropped with a warning.
_EVENT_EMIT_DEPTH_CAP = 8
# Max queued + running events per manager generation; emit never waits — a full budget drops.
_EVENT_PENDING_CAP = 64
_EVENT_WORKER_STOP = object()


@dataclass(frozen=True)
class PluginSystemPromptSection:
    """A plugin-owned section rendered once for each new session."""

    id: str
    content: Union[str, Callable[[Mapping[str, Any]], str]]
    position: str
    max_chars: int
    plugin: str


@dataclass(frozen=True)
class RenderedPluginSystemPromptSection:
    """Validated prompt bytes frozen on the owning AIAgent."""

    id: str
    content: str
    position: str
    plugin: str


@dataclass(frozen=True)
class _EventSubscription:
    """Host-owned subscription ledger entry."""

    owner: str
    callback: Callable


@dataclass(frozen=True)
class _QueuedPluginEvent:
    """Immutable dispatch envelope consumed by the event worker."""

    event: str
    payload: Dict[str, Any]
    subscriptions: tuple[_EventSubscription, ...]
    depth: int
    generation: int
    context: contextvars.Context


# Hook callback timeout (non-blocking abandon). Default cap per Python hook callback.
_HOOK_CALLBACK_TIMEOUT_SECS = 30.0
_MAX_HOOK_CALLBACK_TIMEOUT_SECS = 600.0
_HOOK_SKIPPED = object()


def _resolve_hook_callback_timeout() -> float:
    """Resolve the effective hook callback timeout from runtime configuration."""
    default = _HOOK_CALLBACK_TIMEOUT_SECS
    raw_timeout = read_hook_callback_timeout_seconds()
    if raw_timeout is None:
        return default
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError):
        logger.warning("plugins.hook_callback_timeout is not a number; using default %gs", default)
        return default
    if timeout < 0:
        logger.warning(
            "plugins.hook_callback_timeout=%g is negative; using default %gs", timeout, default,
        )
        return default
    if timeout > _MAX_HOOK_CALLBACK_TIMEOUT_SECS:
        logger.warning(
            "plugins.hook_callback_timeout=%g exceeds max %gs; clamping",
            timeout,
            _MAX_HOOK_CALLBACK_TIMEOUT_SECS,
        )
        return _MAX_HOOK_CALLBACK_TIMEOUT_SECS
    return timeout


def _hook_call_identity(kwargs: Dict[str, Any]) -> Optional[str]:
    """Identity of the call this callback fires for, or None when the event has none."""
    for field in ("tool_call_id", "turn_id"):
        value = kwargs.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _hook_uses_callback_timeout(hook_name: str, timeout: float) -> bool:
    """Whether hook_name should run under the non-blocking timeout path."""
    if timeout <= 0 or hook_name in _HOOK_CALLER_THREAD_HOOKS:
        return False
    return hook_name in _HOOK_TIMEOUT_BOUNDED_HOOKS or hook_name in _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS

class PluginHookDispatchHost(Protocol):
    """Structural host contract required by runtime hook execution."""

    _hooks: Dict[str, List[Callable]]
    _hook_running_callbacks: Dict[tuple, object]
    _hook_abandoned: Dict[tuple, set]
    _hook_timeout_suppressed_until: Dict[tuple, float]
    _hook_timeout_lock: Any
    _hook_timeout_suppression_seconds: float
    _hook_failures_reported: set

    @staticmethod
    def _plugin_dispatch_safe_worker_enabled() -> bool: ...

    @staticmethod
    def _plugin_dispatch_resolve_result(result: Any) -> Any: ...


class PluginHookDispatchMixin:
    """Canonical sync/async plugin hook execution runtime."""
    @staticmethod
    def _hook_callback_kwargs(callback: Callable, payload: Dict[str, Any]) -> Dict[str, Any]:
        """The slice of *payload* a callback accepts: everything for ``**kwargs`` (or
        un-introspectable) callbacks, only declared names for narrow legacy signatures."""
        try:
            parameters = inspect.signature(callback).parameters
        except (TypeError, ValueError):
            return dict(payload)  # no introspectable signature
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return dict(payload)
        keyword_kinds = {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        return {
            name: value for name, value in payload.items()
            if name in parameters and parameters[name].kind in keyword_kinds
        }

    def _invoke_hook_callback(self, callback: Callable, payload: Dict[str, Any]) -> Any:
        """Invoke a hook while withholding additive fields from narrow legacy callbacks.

        An ``async def`` callback returns a coroutine; resolve it the way plugin slash commands
        are (loop-safe), otherwise the bare coroutine object is appended to the results and the
        plugin's body never runs (#12449).
        """
        return self._plugin_dispatch_resolve_result(
            callback(**self._hook_callback_kwargs(callback, payload))
        )

    def invoke_hook(self, hook_name: str, **kwargs: Any) -> List[Any]:
        """Call all callbacks for *hook_name*; return their non-``None`` results.

        Payloads evolve additively: ``**kwargs`` callbacks get everything, narrow signatures only
        what they declare. Each callback is isolated. Bounded hooks and ``pre_tool_call`` run under
        ``plugins.hook_callback_timeout`` (worker abandoned, never joined); ``pre_tool_call`` fails
        closed with a block directive, others skip. ``_HOOK_CALLER_THREAD_HOOKS`` always run on the
        caller thread. ``pre_llm_call`` may return ``{"context": "..."}`` (or a str) to inject.
        """
        if self._plugin_dispatch_safe_worker_enabled():
            return []
        # Gateway platform events define event-local envelopes; a bus-wide version here would turn
        # unrelated adapter payloads into one monolithic compatibility contract.
        if hook_name != "gateway_platform_event":
            kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
        results: List[Any] = []
        timeout = _resolve_hook_callback_timeout()
        use_timeout = _hook_uses_callback_timeout(hook_name, timeout)
        fail_closed = hook_name in _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS
        for cb in self._hooks.get(hook_name, []):
            try:
                if use_timeout:
                    ret = self._run_hook_callback_bounded(hook_name, cb, kwargs, timeout)
                    if ret is _HOOK_SKIPPED:
                        if fail_closed:  # policy hook: fail closed with a block directive
                            results.append({"action": "block", "message": _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE})
                        continue
                else:
                    ret = self._invoke_hook_callback(cb, kwargs)
                if ret is not None:
                    results.append(ret)
            except (Exception, SystemExit) as exc:
                self._report_hook_failure(hook_name, cb, kwargs, exc)
                if fail_closed:  # a guard that raised made no decision: same veto as a timeout
                    results.append(_policy_error_block_directive(hook_name, cb, exc))
        return results

    def _report_hook_failure(
        self, hook_name: str, cb: Callable, kwargs: Dict[str, Any], exc: BaseException, *, surface: str = "Hook"
    ) -> None:
        """One WARNING per distinct (hook, callback, error); identical repeats at DEBUG.

        A callback whose signature names a parameter the hook never sends (``tool_data`` instead
        of ``tool_name``/``args``) fails identically on every tool call — ~1700 WARNING lines an
        hour that bury real signals (#111922). The first report names the fields the hook does
        provide so the plugin author can fix the signature. The key names the callback by
        module/qualname (not ``id()``, which CPython recycles across plugin reloads) and
        truncates the message so a hook that embeds tool args in its error cannot grow the set
        per call; the set is cleared on unload alongside the timeout-suppression map.
        """
        callback_name = getattr(cb, "__name__", repr(cb))
        key = (hook_name, getattr(cb, "__module__", ""), getattr(cb, "__qualname__", callback_name),
               type(exc).__name__, str(exc)[:200])
        if key in self._hook_failures_reported:
            logger.debug("%s '%s' callback %s raised again: %s", surface, hook_name, callback_name, exc)
            return
        self._hook_failures_reported.add(key)
        logger.warning(
            "%s '%s' callback %s raised: %s (%s provides: %s; identical failures are logged at DEBUG from now on)",
            surface, hook_name, callback_name, exc, surface.lower(), ", ".join(sorted(kwargs)) or "no fields")

    def _run_hook_callback_bounded(
        self, hook_name: str, cb: Callable, kwargs: Dict[str, Any], timeout: float
    ) -> Any:
        """Run one callback on a daemon worker with a wall-clock cap; ``_HOOK_SKIPPED`` when
        suppressed, still running for this call id, over the abandoned-worker cap, timed out
        (worker abandoned, never joined), or the worker could not be started. Exceptions
        propagate."""
        callback_name = getattr(cb, "__name__", repr(cb))
        # Suppression is a fact about the CALLBACK — a hung one must keep its back-off —
        # so that key stays coarse. The gate must instead tell CONCURRENT CALLS apart.
        suppression_key = (hook_name, id(cb))
        gate_key = (*suppression_key, _hook_call_identity(kwargs))
        token = object()
        with self._hook_timeout_lock:
            suppressed_until = self._hook_timeout_suppressed_until.get(suppression_key)
            if (gate_key in self._hook_running_callbacks
                    or (suppressed_until is not None and suppressed_until > time.monotonic())):
                logger.warning(
                    "Hook '%s' callback %s skipped after previous "
                    "timeout or while still running", hook_name, callback_name)
                return _HOOK_SKIPPED
            # Workers abandoned on timeout still hold threads. Once the suppression window has
            # passed, a fresh call id may start a new worker (a hung guard must not fail every
            # later tool call closed until restart, #105223), but only up to a small cap per
            # callback — expiring the bookkeeping while the hung worker lives must not leak a
            # thread per call (#98382). At the cap the callback keeps being skipped (fail-closed
            # for pre_tool_call) until one of its workers finishes and releases its slot.
            abandoned = self._hook_abandoned.get(suppression_key)
            if abandoned and len(abandoned) >= _HOOK_MAX_ABANDONED_WORKERS:
                logger.warning(
                    "Hook '%s' callback %s (%s) skipped: %d abandoned worker(s) still running — "
                    "the plugin is hung; fix or disable it (retried when a worker finishes)",
                    hook_name, callback_name, getattr(cb, "__module__", "unknown plugin"), len(abandoned))
                return _HOOK_SKIPPED
            if suppressed_until is not None:
                self._hook_timeout_suppressed_until.pop(suppression_key, None)
            self._hook_running_callbacks[gate_key] = token

        context = contextvars.copy_context()
        done = threading.Event()
        outcome: Dict[str, Any] = {}
        failure: Dict[str, BaseException] = {}

        def _release_token() -> None:
            with self._hook_timeout_lock:
                if self._hook_running_callbacks.get(gate_key) is token:
                    self._hook_running_callbacks.pop(gate_key, None)
                    abandoned = self._hook_abandoned.get(suppression_key)
                    if abandoned is not None:
                        abandoned.discard(gate_key)
                        if not abandoned:
                            self._hook_abandoned.pop(suppression_key, None)

        def _runner() -> None:
            try:
                outcome["value"] = context.run(self._invoke_hook_callback, cb, kwargs)
            except BaseException as exc:
                failure["exc"] = exc
            finally:
                _release_token()
                done.set()

        thread = threading.Thread(target=_runner, name=f"hermes-hook-{callback_name}"[:40], daemon=True)
        try:
            thread.start()
        except RuntimeError as exc:
            _release_token()  # the runner's finally never runs when OS thread creation fails
            logger.warning(
                "Hook '%s' callback %s worker failed to start: %s — skipping",
                hook_name, callback_name, exc)
            return _HOOK_SKIPPED
        if not done.wait(timeout=timeout):  # do not join — that would reintroduce the hang
            with self._hook_timeout_lock:
                # See #6622.
                self._hook_timeout_suppressed_until[suppression_key] = (
                    time.monotonic() + self._hook_timeout_suppression_seconds)
                # The worker may have finished (and released its token) between the wait
                # expiring and this lock; recording it as abandoned then would block the
                # callback for that call id until reload with no thread behind it.
                if self._hook_running_callbacks.get(gate_key) is token:
                    self._hook_abandoned.setdefault(suppression_key, set()).add(gate_key)
            logger.warning(
                "Hook '%s' callback %s timed out after %gs — skipping", hook_name, callback_name, timeout)
            return _HOOK_SKIPPED
        if "exc" in failure:
            raise failure["exc"]
        return outcome.get("value")

    def has_hook(self, hook_name: str) -> bool:
        """Return True when at least one callback is registered for a hook."""
        if self._plugin_dispatch_safe_worker_enabled():
            return False
        return bool(self._hooks.get(hook_name))

    async def ainvoke_hook(self, hook_name: str, **kwargs: Any) -> List[Any]:
        """:meth:`invoke_hook` for callers that are already on an event loop.

        Same payload narrowing, per-callback isolation and result contract. The difference is
        where an ``async def`` callback runs: here it is awaited on the caller's own loop, so a
        callback that awaits anything scheduled on that loop can make progress. Through the
        sync path it runs on a helper thread while the caller blocks in ``done.wait()`` — on the
        gateway that stalls the whole event loop for the callback's duration. Sync callbacks
        run inline. Bounded hooks keep ``plugins.hook_callback_timeout`` via ``asyncio.wait_for``
        (the coroutine is cancelled, not abandoned); a timed-out ``pre_tool_call`` fails closed.
        """
        if hook_name != "gateway_platform_event":
            kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
        results: List[Any] = []
        timeout = _resolve_hook_callback_timeout()
        use_timeout = _hook_uses_callback_timeout(hook_name, timeout)
        fail_closed = hook_name in _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS
        for cb in self._hooks.get(hook_name, []):
            callback_name = getattr(cb, "__name__", repr(cb))
            try:
                ret = cb(**self._hook_callback_kwargs(cb, kwargs))
                if inspect.isawaitable(ret):
                    ret = await (asyncio.wait_for(ret, timeout) if use_timeout else ret)
                if ret is not None:
                    results.append(ret)
            except asyncio.TimeoutError:
                logger.warning("Hook '%s' callback %s timed out after %.0fs", hook_name, callback_name, timeout)
                if fail_closed:  # policy hook: fail closed with a block directive
                    results.append({"action": "block", "message": _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE})
            except (Exception, SystemExit) as exc:
                # Same isolation + failure contract as the sync path (#111922 warn-once, #109624
                # a raising policy guard fails closed).
                self._report_hook_failure(hook_name, cb, kwargs, exc)
                if fail_closed:
                    results.append(_policy_error_block_directive(hook_name, cb, exc))
        return results

    def iter_hook_callbacks(self, hook_name: str) -> tuple[Callable, ...]:
        """Return a stable snapshot of callbacks registered for a hook."""
        if self._plugin_dispatch_safe_worker_enabled():
            return ()
        return tuple(self._hooks.get(hook_name, ()))

class PluginEventDispatchHost(PluginHookDispatchHost, Protocol):
    """Structural host contract required by runtime event dispatch."""

    _subscriptions: Dict[str, List[_EventSubscription]]
    _event_lock: Any
    _event_idle: Any
    _event_generation: int
    _event_pending_by_generation: Dict[int, int]
    _event_queue: queue.Queue[Any]
    _event_worker: Optional[threading.Thread]
    _emit_depth: Any
    _ownership_ledger: Dict[str, List[PluginRegistration]]

    def _track_owner_registration(
        self, plugin_key: str, kind: str, key: str, release: Callable[[], None], *,
        persistent: bool = False,
    ) -> PluginRegistration: ...

    def _dispose_registrations(self, registrations: List[PluginRegistration]) -> None: ...

    def _forget_registrations(self, registrations: List[PluginRegistration]) -> None: ...


class PluginEventDispatchMixin(PluginHookDispatchMixin):
    """Canonical plugin event subscription and asynchronous delivery runtime."""
    def _subscribe_event(self, owner: str, event: str, callback: Callable) -> PluginRegistration:
        """Add an owner-tagged event subscription in registration order and ledger-track its cleanup."""
        if not callable(callback):
            raise TypeError("Event subscriber callback must be callable")
        subscription = _EventSubscription(owner, callback)
        with self._event_lock:
            self._subscriptions.setdefault(event, []).append(subscription)

        def _release() -> None:
            with self._event_lock:
                entries = self._subscriptions.get(event)
                if not entries:
                    return
                index = next((i for i in range(len(entries) - 1, -1, -1) if entries[i] is subscription), None)
                if index is None:
                    return
                del entries[index]
                if not entries:
                    self._subscriptions.pop(event, None)

        return self._track_owner_registration(owner, "event_subscription", event, _release)

    def _remove_plugin_subscriptions(self, owner: str) -> int:
        """Dispose every ledger-owned event subscription for *owner*; return the count.
        Queued envelopes re-check membership per callback, so disposal also cancels already-snapshotted
        deliveries."""
        registrations = [
            registration for registration in self._ownership_ledger.get(owner, [])
            if registration.kind == "event_subscription" and registration.active
        ]
        self._dispose_registrations(registrations)
        self._forget_registrations(registrations)
        return len(registrations)

    def _ensure_event_worker_locked(self) -> None:
        worker = self._event_worker
        if worker is not None and worker.is_alive():
            return
        worker = threading.Thread(
            target=self._event_worker_loop, args=(self._event_queue,), name="hermes-plugin-events",
            daemon=True,
        )
        self._event_worker = worker
        worker.start()

    def _event_worker_loop(self, dispatch_queue: queue.Queue[Any]) -> None:
        while True:
            item = dispatch_queue.get()
            try:
                if item is _EVENT_WORKER_STOP:
                    return
                self._deliver_event(item)
            finally:
                if item is not _EVENT_WORKER_STOP:
                    self._mark_event_done(item.generation)
                dispatch_queue.task_done()

    def _mark_event_done(self, generation: int) -> None:
        with self._event_idle:
            pending = self._event_pending_by_generation.get(generation, 0)
            if pending > 0:
                self._event_pending_by_generation[generation] = pending - 1
            self._event_idle.notify_all()

    def _deliver_event(self, item: _QueuedPluginEvent) -> None:
        """Deliver one queued event on the host-owned worker thread."""
        with self._event_lock:
            if item.generation != self._event_generation:
                return
        previous_depth = getattr(self._emit_depth, "value", 0)
        self._emit_depth.value = item.depth
        try:
            for subscription in item.subscriptions:
                with self._event_lock:
                    if item.generation != self._event_generation:
                        break
                    # Owner unload may have removed this entry after the event was queued.
                    if not any(cur is subscription for cur in self._subscriptions.get(item.event, [])):
                        continue
                callback = subscription.callback
                try:
                    # Fresh deep copy per subscriber: no callback can mutate what the next sees.
                    self._plugin_dispatch_resolve_result(
                        item.context.copy().run(callback, **copy.deepcopy(item.payload)))
                except (Exception, SystemExit) as exc:
                    # A subscriber that fails identically on every emit is reported once (#111922).
                    self._report_hook_failure(item.event, callback, item.payload, exc, surface="Event")
        finally:
            self._emit_depth.value = previous_depth

    def _wait_for_event_dispatch(self, timeout: float = 2.0) -> bool:
        """Wait for the current event generation to become idle (test helper)."""
        with self._event_idle:
            generation = self._event_generation
            return self._event_idle.wait_for(
                lambda: self._event_pending_by_generation.get(generation, 0) == 0, timeout=timeout)

    def _dispatch_event(self, event: str, payload: Dict[str, Any]) -> int:
        """Queue *event* without blocking; return the subscriber count scheduled. Pending work is
        bounded per generation so a blocking subscriber costs one worker and later emits drop."""
        depth = getattr(self._emit_depth, "value", 0)
        if depth >= _EVENT_EMIT_DEPTH_CAP:
            logger.warning(
                "Event bus recursion cap (%d) exceeded while dispatching '%s' "
                "— dropping this emit to prevent an infinite loop", _EVENT_EMIT_DEPTH_CAP, event)
            return 0
        budget_msg = "Event bus pending budget (%d) exhausted while dispatching '%s' — dropping this emit"
        with self._event_lock:
            subscriptions = tuple(self._subscriptions.get(event, []))
            if not subscriptions:
                return 0
            generation = self._event_generation
            pending = self._event_pending_by_generation.get(generation, 0)
            if pending >= _EVENT_PENDING_CAP:
                logger.warning(budget_msg, _EVENT_PENDING_CAP, event)
                return 0
            item = _QueuedPluginEvent(
                event=event, payload=dict(payload), subscriptions=subscriptions, depth=depth + 1,
                generation=generation, context=contextvars.copy_context())
            try:
                self._event_queue.put_nowait(item)
            except queue.Full:
                logger.warning(budget_msg, _EVENT_PENDING_CAP, event)
                return 0
            self._event_pending_by_generation[generation] = pending + 1
            self._ensure_event_worker_locked()
            return len(subscriptions)

class PluginDispatchHost(PluginEventDispatchHost, Protocol):
    """Structural host contract for complete runtime dispatch execution."""

    _middleware: Dict[str, List[Callable]]
    _system_prompt_sections: Dict[str, PluginSystemPromptSection]


class PluginDispatchMixin(PluginEventDispatchMixin):
    """Canonical plugin hook, event, middleware, and prompt-section execution runtime."""

    def _init_dispatch_runtime_state(self) -> None:
        """Initialize manager-local state used exclusively by runtime dispatch execution."""
        # Event bus: owner-tagged subscriptions, one non-blocking worker, generation-local
        # pending accounting, and emitter context used for recursion-depth propagation.
        self._subscriptions: Dict[str, List[_EventSubscription]] = {}
        self._event_lock = threading.RLock()
        self._event_idle = threading.Condition(self._event_lock)
        self._event_generation = 0
        self._event_pending_by_generation: Dict[int, int] = {0: 0}
        self._event_queue: queue.Queue[Any] = queue.Queue(maxsize=_EVENT_PENDING_CAP)
        self._event_worker: Optional[threading.Thread] = None
        self._emit_depth = threading.local()

        # In-flight / recently timed-out hook callbacks and warn-once failure bookkeeping.
        self._hook_running_callbacks: Dict[tuple, object] = {}
        self._hook_abandoned: Dict[tuple, set] = {}
        self._hook_timeout_suppressed_until: Dict[tuple, float] = {}
        self._hook_timeout_lock = threading.Lock()
        self._hook_timeout_suppression_seconds = _HOOK_TIMEOUT_SUPPRESSION_SECONDS
        self._hook_failures_reported: set = set()

    def render_system_prompt_sections(
        self, session_info: Mapping[str, Any]
    ) -> List[RenderedPluginSystemPromptSection]:
        """Render all registered sections deterministically and fail open."""
        if self._plugin_dispatch_safe_worker_enabled():
            return []
        frozen_info = types.MappingProxyType(dict(session_info))
        rendered: List[RenderedPluginSystemPromptSection] = []
        total_chars = len(PLUGIN_SECTIONS_START) + len(PLUGIN_SECTIONS_END) + 2
        for _section_id, section in sorted(self._system_prompt_sections.items()):
            if len(rendered) >= MAX_SYSTEM_PROMPT_SECTIONS:
                logger.warning(
                    "Plugin system prompt section %s exceeded the section-count "
                    "budget (%d) and was skipped", section.id, MAX_SYSTEM_PROMPT_SECTIONS)
                continue
            text = self._render_prompt_section_text(section, frozen_info)
            if text is None:
                continue
            rendered_chars = len(format_system_prompt_section(section.id, text))
            if rendered:
                rendered_chars += 2  # canonical ``\n\n`` separator
            if total_chars + rendered_chars > MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS:
                logger.warning(
                    "Plugin system prompt section %s (%s) exceeded the aggregate "
                    "session budget (%d chars) and was skipped", section.id, section.plugin,
                    MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS)
                continue
            rendered.append(
                RenderedPluginSystemPromptSection(
                    id=section.id, content=text, position=section.position, plugin=section.plugin))
            total_chars += rendered_chars
            logger.info(
                "Session plugin prompt section: id=%s plugin=%s position=%s chars=%d", section.id,
                section.plugin, section.position, len(text))
        return rendered

    @staticmethod
    def _render_prompt_section_text(
        section: PluginSystemPromptSection, frozen_info: Mapping[str, Any]
    ) -> Optional[str]:
        """Evaluate one section; return its stripped text or None (with a warning) when skipped."""
        def _skip(detail: str, *args: Any) -> None:
            logger.warning(
                "Plugin system prompt section %s (%s) " + detail, section.id, section.plugin, *args)

        try:
            value = section.content(frozen_info) if callable(section.content) else section.content
        except (Exception, SystemExit) as exc:
            _skip("raised and was skipped: %s", exc)
            return None
        if not isinstance(value, str):
            _skip("returned %s, not str; skipped", type(value).__name__)
            return None
        text = value.strip()
        if not text:
            return None
        if PLUGIN_SECTIONS_START in text or PLUGIN_SECTIONS_END in text:
            _skip("contained a reserved persistence marker and was skipped")
            return None
        if len(text) > section.max_chars:
            _skip("exceeded max_chars (%d > %d) and was skipped", len(text), section.max_chars)
            return None
        return text

    def has_middleware(self, kind: str) -> bool:
        """Return True when at least one callback is registered for middleware."""
        if self._plugin_dispatch_safe_worker_enabled():
            return False
        return bool(self._middleware.get(kind))

    def invoke_middleware(self, kind: str, **kwargs: Any) -> List[Any]:
        """Call middleware callbacks for *kind* (each isolated); return non-``None`` results."""
        if self._plugin_dispatch_safe_worker_enabled():
            return []
        results: List[Any] = []
        for cb in self._middleware.get(kind, []):
            try:
                ret = cb(**kwargs)
                if ret is not None:
                    results.append(ret)
            except (Exception, SystemExit) as exc:
                # Runs once per tool call like a hook, so a mis-declared callback floods identically.
                self._report_hook_failure(kind, cb, kwargs, exc, surface="Middleware")
        return results
