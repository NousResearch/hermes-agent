"""Plugin session forks (``ctx.fork_session``, capability ``llm.session_fork``).

The background-review pattern, for plugins: after a turn, a plugin redeems the turn's lease
(``post_llm_call``'s ``side_question_token``, :mod:`agent.side_question_lease`) to fork the live
session agent (same system prompt, ``tools[]``, memory and warm prompt cache) and run one bounded
multi-step prompt in the background. Guardrails are the review's own: a dispatch-side tool
whitelist of the tools the plugin names (already advertised to the session), dangerous-command
approvals auto-denied, an aggregate input budget, an iteration cap, cancellation when the parent
starts its next live turn, idle deferral on the managed local runtime, persistence fully detached.
Usage is billed to the session as ``plugin_fork:<plugin id>``; each loop request reports through
``pre/post_auxiliary_call`` (``aux_task="plugin_fork"`` + ``plugin_id``), never the turn-scoped
``pre/post_api_request``. Tool handlers called inside the fork receive
``turn_origin="plugin_fork:<plugin id>"``.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

PLUGIN_FORK_TASK = "plugin_fork"
DEFAULT_MAX_ITERATIONS = 8
MAX_ITERATIONS = 16  # the background review's own cap
MAX_PROMPT_CHARS = 8192
MAX_TIMEOUT_SECONDS = 1800.0

_TURN_ORIGIN: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("hermes_turn_origin", default=None)


def current_turn_origin() -> Optional[str]:
    """``"plugin_fork:<plugin id>"`` while a plugin session fork runs on this context, else None."""
    return _TURN_ORIGIN.get()


class PluginForkHandle:
    """What ``ctx.fork_session`` returns: progress and outcome of the fork, never the fork itself.

    ``outcome``: ``deferred`` (queued for machine idle), ``running``, then one of ``ok``,
    ``cancelled``, ``superseded`` (a newer deferred fork of the same plugin replaced it),
    ``timeout``, ``budget_exhausted``, ``error``, ``refused:<reason>``."""

    def __init__(self, plugin_id: str, cancel: Callable[[], None]):
        self.plugin_id = plugin_id
        self._cancel = cancel
        self._done = threading.Event()
        self._outcome = "running"
        self._result: Optional[str] = None
        self._actions: tuple[str, ...] = ()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def cancelled(self) -> bool:
        return self._outcome in ("cancelled", "superseded")

    @property
    def outcome(self) -> str:
        return self._outcome

    @property
    def result(self) -> Optional[str]:
        """The fork's final text (None until done, or when it ended without one)."""
        return self._result

    @property
    def actions(self) -> tuple[str, ...]:
        """Names of the whitelisted tools the fork called, in order."""
        return self._actions

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._done.wait(timeout)

    def cancel(self) -> None:
        self._cancel()

    def _settle(self, outcome: str, result: Optional[str] = None, actions: tuple[str, ...] = ()) -> None:
        if self._done.is_set():
            return
        self._outcome, self._result, self._actions = outcome, result, actions
        self._done.set()

    def __repr__(self) -> str:
        return f"PluginForkHandle(plugin_id={self.plugin_id!r}, outcome={self._outcome!r})"


@dataclass
class _ForkJob:
    plugin_id: str
    capture: Any  # SideQuestionCapture
    prompt: str
    tools: frozenset
    max_iterations: int
    input_budget: Optional[int]
    task_cfg: dict
    timeout: Optional[float]
    run: Any  # background_review._BackgroundReviewRun
    handle: PluginForkHandle = field(init=False)

    @property
    def origin(self) -> str:
        return f"plugin_fork:{self.plugin_id}"


# Launched forks live on their parent as ``_plugin_fork_runs`` (a set of runs, guarded by
# ``_lock``): the next live turn cancels them (background_review).
# Deferred (queued) jobs per idle-queue key: a newer one supersedes the queued one.
_queued: dict[str, _ForkJob] = {}
_lock = threading.Lock()


def _bounded_int(value: Any, low: int, high: Optional[int] = None) -> Optional[int]:
    """``value`` clamped to ``[low, high]``; None when it is not an int."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return max(low, value if high is None else min(value, high))


def _clamped_timeout(timeout: Any) -> Optional[float]:
    try:
        value = float(timeout)
    except (TypeError, ValueError):
        return None
    return min(value, MAX_TIMEOUT_SECONDS) if value > 0 else None  # NaN/<=0: no wall clock


def _resolve_task_cfg(task: Optional[str], owned_tasks: Iterable[str]) -> Optional[dict]:
    """The fork's routing block: ``{}`` (never routed) without ``task``; the plugin's own auxiliary
    task (inheritance resolved) when it names one; None for a task the plugin does not own."""
    if task is None:
        return {}
    if task not in set(owned_tasks):
        return None
    from agent.auxiliary_task_config import _get_auxiliary_task_config

    return dict(_get_auxiliary_task_config(task))


def start_session_fork(
    token: Any, prompt: Any, *, plugin_id: str, home: Any, tools: Any = (), max_iterations: Any = DEFAULT_MAX_ITERATIONS,
    input_budget: Any = None, task: Optional[str] = None, timeout: Any = None, owned_tasks: Iterable[str] = (),
) -> Optional[PluginForkHandle]:
    """Spend the lease behind ``token`` on one background fork; its handle, or None (bad token,
    expired lease, another profile's lease, a tool the session does not advertise, a task the
    plugin does not own). The lease is consumed by every call except another profile's."""
    from agent.background_review import _BackgroundReviewRun
    from agent.side_question_lease import take_lease

    lease = take_lease(token, prompt, plugin_id=plugin_id, home=home, kind="session_fork",
                       max_prompt_chars=MAX_PROMPT_CHARS)
    if lease is None:
        return None
    capture = lease.capture
    names = frozenset(tools) if isinstance(tools, (list, tuple, set, frozenset)) else None
    advertised = {t.get("function", {}).get("name") for t in capture.tools if isinstance(t, dict)}
    if names is None or not all(isinstance(n, str) for n in names) or not names <= advertised:
        logger.info("session_fork plugin=%s outcome=rejected:tools_not_in_session %s", plugin_id,
                    sorted(map(str, (names or set()) - advertised)) if names is not None else tools)
        return None
    task_cfg = capture.context.copy().run(_resolve_task_cfg, task, owned_tasks)
    if task_cfg is None:
        logger.info("session_fork plugin=%s outcome=rejected:task_not_owned (%r)", plugin_id, task)
        return None
    run = _BackgroundReviewRun()
    job = _ForkJob(plugin_id, capture, prompt, names, _bounded_int(max_iterations, 1, MAX_ITERATIONS) or DEFAULT_MAX_ITERATIONS,
                   _bounded_int(input_budget, 1), task_cfg, _clamped_timeout(timeout), run)
    job.handle = PluginForkHandle(plugin_id, lambda: _cancel_job(job))
    parent = capture.parent_ref()
    if parent is not None and capture.context.copy().run(_should_defer, parent, task_cfg):
        _defer(job, parent)
    else:
        _launch(job)
    return job.handle


def _should_defer(parent: Any, task_cfg: dict) -> bool:
    from agent.review_idle_queue import defer_mode, review_targets_managed_local

    return defer_mode(task_cfg) == "auto" and review_targets_managed_local(parent, task_cfg)


def _defer(job: _ForkJob, parent: Any) -> None:
    from agent.review_idle_queue import QUEUE

    key = f"{getattr(parent, 'session_id', None) or id(parent)}:{job.origin}"

    def _dispatch() -> None:
        with _lock:
            if _queued.get(key) is job:
                del _queued[key]
        _launch(job)

    with _lock:
        previous, _queued[key] = _queued.get(key), job
    if previous is not None:
        previous.handle._settle("superseded")
    job.handle._outcome = "deferred"
    QUEUE.enqueue(parent, key, {"task_cfg": job.task_cfg}, dispatch=_dispatch)


def _launch(job: _ForkJob) -> None:
    from agent.memory_provider import spawn_context_thread

    parent = job.capture.parent_ref()
    if parent is None or job.run.cancel_requested.is_set():
        job.handle._settle("refused:parent_gone" if parent is None else "cancelled")
        return
    with _lock:
        runs = getattr(parent, "_plugin_fork_runs", None)
        if runs is None:
            runs = parent._plugin_fork_runs = set()
        runs.add(job.run)
    job.handle._outcome = "running"
    # Spawned inside the captured turn context, so the worker binds that profile + secret scope.
    job.capture.context.copy().run(lambda: spawn_context_thread(
        _run_job, name=f"plugin-fork:{job.plugin_id}", args=(job,)).start())


def _cancel_job(job: _ForkJob) -> None:
    from agent.background_review import _interrupt_background_review

    fork = job.run.cancel()
    if fork is not None:
        _interrupt_background_review(fork, message="plugin fork cancelled", tool_reason="plugin fork cancelled")
    if job.handle.outcome == "deferred":
        job.handle._settle("cancelled")


def interrupt_plugin_forks(agent: Any, *, message: str, tool_reason: str) -> list:
    """Fence/interrupt every launched fork of ``agent``; the runs to await (``request_done``)."""
    from agent.background_review import _interrupt_background_review

    with _lock:
        runs = list(getattr(agent, "_plugin_fork_runs", None) or ())
    for run in runs:
        fork = run.cancel()
        if fork is not None:
            _interrupt_background_review(fork, message=message, tool_reason=tool_reason)
    return runs


def _pin_capture(fork: Any, capture: Any) -> None:
    """Same-model fork: replay the captured request inputs byte-for-byte (cache parity)."""
    import copy

    fork._cached_system_prompt = capture.system_prompt
    fork._cached_system_prompt_static = capture.system_prompt_static
    fork.tools = copy.deepcopy(capture.tools)
    fork.valid_tool_names = {tool["function"]["name"] for tool in fork.tools}
    fork._prompt_cache_fork_tag = PLUGIN_FORK_TASK


def _tool_calls_after(messages: list, snapshot: list, allowed: frozenset) -> tuple[str, ...]:
    seen = {tc.get("id") for m in snapshot if isinstance(m, dict) for tc in (m.get("tool_calls") or ()) if isinstance(tc, dict)}
    return tuple(
        name for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or ()) if isinstance(tc, dict) and tc.get("id") not in seen
        for name in [(tc.get("function") or {}).get("name")] if name in allowed
    )


def _outcome(job: _ForkJob, fork: Any, result: Any, timed_out: threading.Event) -> str:
    from agent.conversation_loop import _review_input_budget_exhausted

    if timed_out.is_set():
        return "timeout"
    if job.run.cancel_requested.is_set():
        return "cancelled"
    result = result if isinstance(result, dict) else {}
    if result and not result.get("failed"):
        return "ok"
    # The loop stops on the budget between iterations; the finalizer then reports the open tool
    # round, so the fork's own counter is the witness.
    return "budget_exhausted" if _review_input_budget_exhausted(fork) else "error"


def _run_job(job: _ForkJob) -> None:
    """Worker body, in the captured turn context (profile + secret scope)."""
    from agent.background_review import (
        _bg_review_auto_deny, _digest_history, _interrupt_background_review, _set_thread_approval_callback,
        build_cache_parity_fork, finish_background_review_run, release_cache_parity_fork,
    )
    from agent.auxiliary_hooks import end_loop_aux_attempt
    from agent.side_question import trim_snapshot_for_fork
    from agent.side_question_request import _agent_provider_refusal, runtime_key
    from agent.thread_scoped_output import thread_scoped_silence
    from hermes_cli.plugins import clear_thread_tool_whitelist, set_thread_tool_whitelist

    capture, started = job.capture, time.monotonic()
    parent = capture.parent_ref()
    fork = result = timer = None
    outcome, actions, usage = "error", (), {}
    timed_out = threading.Event()
    origin_token = _TURN_ORIGIN.set(job.origin)
    _set_thread_approval_callback(_bg_review_auto_deny)
    try:
        if parent is None:
            outcome = "refused:parent_gone"
            return
        with thread_scoped_silence():
            fork, _rt, routed = build_cache_parity_fork(parent, job.task_cfg, max_iterations=job.max_iterations,
                                                        write_origin=job.origin)
            refusal = None if routed else _agent_provider_refusal(parent)
            if refusal is None and not routed and runtime_key(parent) != capture.runtime:
                refusal = "runtime_changed"
            if refusal is not None:
                outcome = f"refused:{refusal}"
                return
            if not routed:
                _pin_capture(fork, capture)
            fork.session_id = capture.session_id
            budget = fork._review_input_token_budget
            if job.input_budget is not None:
                budget = min(job.input_budget, budget) if budget else job.input_budget
            fork._review_input_token_budget = budget
            fork._loop_aux_task = (PLUGIN_FORK_TASK, {"plugin_id": job.plugin_id})
            _track_child(parent, fork, register=True)
            set_thread_tool_whitelist(set(job.tools), deny_msg_fmt=(
                "This background task may only call: " + (", ".join(sorted(job.tools)) or "no tools")
                + ". {tool_name} was denied; do not retry it."))
            if job.timeout is not None:
                def _on_timeout() -> None:
                    timed_out.set()
                    _interrupt_background_review(fork, message="plugin fork timed out", tool_reason="plugin fork timed out")
                timer = threading.Timer(job.timeout, _on_timeout)
                timer.daemon = True
                timer.start()
            snapshot = trim_snapshot_for_fork(capture.messages)
            if job.run.begin_request(fork):
                result = fork.run_conversation(
                    user_message=job.prompt, conversation_history=_digest_history(snapshot) if routed else snapshot)
                actions = _tool_calls_after((result or {}).get("messages") or [], snapshot, job.tools)
        outcome = _outcome(job, fork, result, timed_out)
    except Exception:
        # Boundary: a plugin's fork settles its handle, it never raises into the plugin.
        outcome = "timeout" if timed_out.is_set() else "cancelled" if job.run.cancel_requested.is_set() else "error"
        logger.warning("session fork %s (plugin=%s)", outcome, job.plugin_id, exc_info=True)
    finally:
        if timer is not None:
            timer.cancel()
        clear_thread_tool_whitelist()
        if fork is not None:
            end_loop_aux_attempt(fork, error=InterruptedError(f"plugin fork {outcome}"))
            usage = release_cache_parity_fork(parent, fork, task=f"{PLUGIN_FORK_TASK}:{job.plugin_id}",
                                              session_id=capture.session_id)
            _track_child(parent, fork, register=False)
        finish_background_review_run(parent, job.run)
        with _lock:
            getattr(parent, "_plugin_fork_runs", set()).discard(job.run)
        _TURN_ORIGIN.reset(origin_token)
        _set_thread_approval_callback(None)
        text = (result or {}).get("final_response") if isinstance(result, dict) else None
        job.handle._settle(outcome, (text or "").strip() or None, actions)
        logger.info(
            "session_fork plugin=%s outcome=%s tools=%s calls=%d in=%d cache_read=%d out=%d latency=%.1fs",
            job.plugin_id, outcome, ",".join(actions) or "-",
            *(int(usage.get(k) or 0) for k in ("api_calls", "input_tokens", "cache_read_tokens", "output_tokens")),
            time.monotonic() - started)


def _track_child(parent: Any, fork: Any, *, register: bool) -> None:
    """Parent interrupt fan-out (``_active_children``); never the review's own slot."""
    if parent is None or not hasattr(parent, "_active_children"):
        return
    lock = getattr(parent, "_active_children_lock", None) or threading.Lock()
    with lock:
        if register:
            parent._active_children.append(fork)
        elif fork in parent._active_children:
            parent._active_children.remove(fork)
