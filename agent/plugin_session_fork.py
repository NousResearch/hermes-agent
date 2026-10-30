"""Plugin session forks (``ctx.fork_session``, capability ``llm.session_fork``).

The background-review pattern, for plugins: after a turn, a plugin redeems the turn's lease
(``post_llm_call``'s ``side_question_token``, :mod:`agent.side_question_lease`) to fork the live
session agent (same system prompt, ``tools[]``, memory and warm prompt cache) and run one bounded
multi-step prompt in the background. Guardrails are the review's own: a dispatch-side tool
whitelist of the tools the plugin names (its own, already advertised to the session or
deferred there behind Tool Search, then reached through the session's ``tool_call`` bridge),
dangerous-command approvals auto-denied, an aggregate input budget, an iteration cap, cancellation
when the parent starts its next live turn, idle deferral on the managed local runtime, persistence
fully detached.
Usage is billed to the session as ``plugin_fork:<plugin id>``; each loop request reports through
``pre/post_auxiliary_call`` (``aux_task="plugin_fork"`` + ``plugin_id``), never the turn-scoped
``pre/post_api_request``. Tool handlers called inside the fork receive
``turn_origin="plugin_fork:<plugin id>"``.
"""

from __future__ import annotations

import contextvars
import json
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

# The one place a core tool is marked fork-safe. A plugin fork holds only the requesting plugin's
# own tools plus these; every other tool (another plugin's, any MCP tool, any unmarked core tool)
# is refused, because a handler may start work (a subagent, a remote agent task, a cron run, a
# detached worker, a server-initiated sampling call) that runs outside the fork thread's dispatch
# guard, keeping neither its tool whitelist nor its auto-denied approvals. Admission: the handler
# only reads, needs no approval, and starts no process, agent loop, remote task or worker. Empty:
# the read-only candidates fail it (``skill_view`` can render a skill's inline shell).
FORK_SAFE_CORE_TOOLS: frozenset[str] = frozenset()


def fork_tool_allowlist(own_tools: Iterable[Any], *agents: Any) -> frozenset[str]:
    """The tools a plugin fork may hold: the ``own_tools`` (names whose live registration is the
    requesting plugin's) that dispatch to that registration's handler, plus
    :data:`FORK_SAFE_CORE_TOOLS`. A name the dispatcher serves before the registry
    (:func:`agent.inline_tool_executors.pre_registry_route`: an agent-level tool such as
    ``delegate_task``, a legacy alias, a Tool Search bridge tool, a connector name, and for each of
    ``agents`` its context-engine and memory-provider tools) is never the plugin's own: calling it
    runs connector, provider or core code, not the plugin's."""
    from agent.inline_tool_executors import pre_registry_route

    routers = agents or (None,)
    return frozenset(
        n for n in own_tools
        if isinstance(n, str) and all(pre_registry_route(n, agent) is None for agent in routers)
    ) | FORK_SAFE_CORE_TOOLS


def current_turn_origin() -> Optional[str]:
    """``"plugin_fork:<plugin id>"`` while a plugin session fork runs on this context, else None."""
    return _TURN_ORIGIN.get()


class PluginForkHandle:
    """What ``ctx.fork_session`` returns: progress and outcome of the fork, never the fork itself.

    ``outcome``: ``deferred`` (queued for machine idle), ``running``, then one of ``ok``,
    ``cancelled``, ``superseded`` (a newer deferred fork of the same plugin replaced it),
    ``timeout``, ``budget_exhausted``, ``error``, ``error:spawn`` (its worker thread could not
    start), ``refused:<reason>``."""

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


@dataclass(eq=False)  # identity: a job is a member of its parent's registry set
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
    # The requesting plugin's own tool names, read at request time and again at dispatch.
    own_tools: Callable[[], Iterable[str]] = frozenset
    handle: PluginForkHandle = field(init=False)
    # Guarded by ``_lock``: the idle-queue slot while deferred; ``launched`` once ``_launch``
    # admitted the job, after which only its worker settles the handle.
    queue_key: Optional[str] = field(default=None, init=False)
    dispatch: Optional[Callable[[], None]] = field(default=None, init=False)
    launched: bool = field(default=False, init=False)

    @property
    def origin(self) -> str:
        return f"plugin_fork:{self.plugin_id}"


# Every fork of a parent, deferred, launching or running, is registered on the parent as
# ``_plugin_fork_jobs`` (a set guarded by ``_lock``) until its worker exits or a fence settles it,
# so the next live turn fences all of them (background_review). ``_plugin_forks_fenced_turn`` is
# the turn the last interrupt fenced: a fork of that turn still on its way to registration is
# refused when it gets there. Deferred jobs are also indexed per idle-queue key: a newer one
# supersedes the queued one.
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
    own_tools: Callable[[], Iterable[str]] = frozenset,
) -> Optional[PluginForkHandle]:
    """Spend the lease behind ``token`` on one background fork; its handle, or None (bad token,
    expired lease, another profile's lease, a tool outside :func:`fork_tool_allowlist`, a tool the
    session neither advertises nor defers behind Tool Search, a task the plugin does not own).
    When a named tool is deferred, one line appended to ``prompt`` says to call it through
    ``tool_call`` (:func:`_deferred_call_note`); ``tools[]`` stays the session's. The lease is consumed by every
    call except another profile's and one naming a tool outside the allowlist. ``own_tools``
    returns the plugin's own tool names; it is read here and again when the fork starts. Never
    raises: a fork whose worker thread cannot start returns its handle settled ``error:spawn``."""
    from agent.background_review import _BackgroundReviewRun
    from agent.side_question_lease import lease_parent, take_lease

    names = frozenset(tools) if isinstance(tools, (list, tuple, set, frozenset)) else None
    refused = names - fork_tool_allowlist(own_tools(), lease_parent(token, home)) if names is not None else None
    if refused:
        # A request no session could satisfy: refused before the lease, so the token stays usable.
        logger.info("session_fork plugin=%s outcome=rejected:tool_not_fork_safe %s", plugin_id,
                    sorted(map(str, refused)))
        return None
    lease = take_lease(token, prompt, plugin_id=plugin_id, home=home, kind="session_fork",
                       max_prompt_chars=MAX_PROMPT_CHARS)
    if lease is None:
        return None
    capture = lease.capture
    deferred, unreached = _session_reach(capture, names)
    if unreached is not None:
        logger.info("session_fork plugin=%s outcome=rejected:tools_not_in_session %s", plugin_id,
                    sorted(map(str, unreached)) if names is not None else tools)
        return None
    if deferred:
        prompt = prompt + _deferred_call_note(deferred)
    task_cfg = capture.context.copy().run(_resolve_task_cfg, task, owned_tasks)
    if task_cfg is None:
        logger.info("session_fork plugin=%s outcome=rejected:task_not_owned (%r)", plugin_id, task)
        return None
    run = _BackgroundReviewRun()
    job = _ForkJob(plugin_id, capture, prompt, names, _bounded_int(max_iterations, 1, MAX_ITERATIONS) or DEFAULT_MAX_ITERATIONS,
                   _bounded_int(input_budget, 1), task_cfg, _clamped_timeout(timeout), run, own_tools)
    job.handle = PluginForkHandle(plugin_id, lambda: _cancel_job(job))
    parent = capture.parent_ref()
    if parent is not None and capture.context.copy().run(_should_defer, parent, task_cfg):
        _defer(job, parent)
    else:
        _launch(job)
    return job.handle


def _session_reach(capture: Any, names: Optional[frozenset]) -> tuple[frozenset, Optional[frozenset]]:
    """``(deferred, unreached)`` for the requested ``names`` in the captured session. A name is
    reachable when the captured ``tools[]`` advertises it, or when the session defers it behind
    Tool Search: those ``tools[]`` carry the ``tool_call`` bridge and the name is in the session's
    deferred catalog, read with the classification the session's own ``tool_call`` unwrap uses
    (:func:`agent.tool_executor._tool_search_scoped_names`). ``unreached`` is None when every name
    is reachable; ``deferred`` holds the names reached only through the bridge."""
    from tools.tool_search_catalog import TOOL_CALL_NAME

    if names is None or not all(isinstance(n, str) for n in names):
        return frozenset(), frozenset(names or ())
    advertised = {t.get("function", {}).get("name") for t in capture.tools if isinstance(t, dict)}
    rest = names - advertised
    if not rest:
        return frozenset(), None
    in_catalog = (capture.context.copy().run(_session_deferred_tools, capture.parent_ref())
                  if TOOL_CALL_NAME in advertised else frozenset())
    unreached = rest - in_catalog
    return (frozenset(rest), None) if not unreached else (frozenset(), unreached)


def _session_deferred_tools(parent: Any) -> frozenset:
    if parent is None:
        return frozenset()
    from agent.tool_executor import _tool_search_scoped_names

    return _tool_search_scoped_names(parent)


def _deferred_call_note(names: Iterable[str]) -> str:
    """The line core appends to a fork's prompt when Tool Search defers a requested tool: the
    fork's ``tools[]`` (the session's, byte for byte) holds only the bridge, so the model is told
    to reach the tool through ``tool_call`` and given the parameters ``tool_call`` validates
    against. ``tool_describe`` and ``tool_search`` are therefore never needed, and stay refused."""
    from tools.registry import registry
    from tools.tool_search_catalog import TOOL_CALL_NAME

    def _parameters(name: str) -> str:
        schema = registry.get_schema(name) or {}
        fn = schema.get("function") if schema.get("type") == "function" else schema
        params = fn.get("parameters") if isinstance(fn, dict) else None
        return json.dumps(params if isinstance(params, dict) else {}, ensure_ascii=False,
                          separators=(",", ":"), sort_keys=True)

    return (f"\n\nCall {', '.join(sorted(names))} through {TOOL_CALL_NAME} with name=<tool> and arguments "
            "matching its parameters: " + "; ".join(f"{n} {_parameters(n)}" for n in sorted(names)) + ".")


def _should_defer(parent: Any, task_cfg: dict) -> bool:
    from agent.review_idle_queue import defer_mode, review_targets_managed_local

    return defer_mode(task_cfg) == "auto" and review_targets_managed_local(parent, task_cfg)


def _defer(job: _ForkJob, parent: Any) -> None:
    from agent.review_idle_queue import QUEUE

    key = f"{getattr(parent, 'session_id', None) or id(parent)}:{job.origin}"

    def _dispatch() -> None:
        with _lock:
            if _queued.get(key) is not job:  # fenced or superseded while queued
                return
            del _queued[key]
        _launch(job)

    previous = fenced = None
    with _lock:
        refusal = _launch_refusal(parent, job)
        if refusal is None:
            previous = _queued.get(key)
            if previous is not None:
                fenced = _fence_locked(previous, "superseded")
            job.queue_key, job.dispatch, _queued[key] = key, _dispatch, job
            _register_locked(parent, job)
            job.handle._outcome = "deferred"
            QUEUE.enqueue(parent, key, {"task_cfg": job.task_cfg}, dispatch=_dispatch)
    if refusal is not None:
        job.handle._settle(refusal)
    if fenced is not None:
        _after_fence(previous, *fenced, message="plugin fork superseded", tool_reason="plugin fork superseded")


def _launch(job: _ForkJob) -> None:
    from agent.background_review import finish_background_review_run
    from agent.memory_provider import spawn_context_thread

    parent = job.capture.parent_ref()
    with _lock:
        # Check and registration under one lock: an interrupt either ran first (the check sees its
        # fence) or runs after (it finds the job registered and fences it).
        refusal = _launch_refusal(parent, job)
        if refusal is None:
            _register_locked(parent, job)
            job.launched = True
            job.handle._outcome = "running"
        else:
            _unregister_locked(parent, job)
    if refusal is not None:
        job.handle._settle(refusal)
        finish_background_review_run(parent, job.run)
        return
    try:
        # Built inside the captured turn context, so the worker binds that profile + secret scope.
        worker = job.capture.context.copy().run(
            spawn_context_thread, _run_job, name=f"plugin-fork:{job.plugin_id}", args=(job,))
        worker.start()
    except Exception:
        # No worker will settle this admitted job: unregister it, publish its request exit (so the
        # next turn's cancellation does not wait on it) and settle it here, never raising.
        logger.warning("session fork error:spawn (plugin=%s)", job.plugin_id, exc_info=True)
        with _lock:
            _unregister_locked(parent, job)
        finish_background_review_run(parent, job.run)
        job.handle._settle("error:spawn")


def _launch_refusal(parent: Any, job: _ForkJob) -> Optional[str]:
    """Why ``job`` may not start now (caller holds ``_lock``), or None. A fork of a turn an
    interrupt already fenced, or of a turn the parent has moved past, is cancelled: that interrupt
    ran before the job was registered, so it could not fence the job itself."""
    if parent is None:
        return "refused:parent_gone"
    turn = job.capture.turn_id
    if (job.run.cancel_requested.is_set() or getattr(parent, "_current_turn_id", turn) != turn
            or getattr(parent, "_plugin_forks_fenced_turn", None) == turn):
        return "cancelled"
    return None


def _register_locked(parent: Any, job: _ForkJob) -> None:
    jobs = getattr(parent, "_plugin_fork_jobs", None)
    if jobs is None:
        jobs = parent._plugin_fork_jobs = set()
    jobs.add(job)


def _unregister_locked(parent: Any, job: _ForkJob) -> None:
    (getattr(parent, "_plugin_fork_jobs", None) or set()).discard(job)


def _fence_locked(job: _ForkJob, outcome: str) -> tuple[Any, bool]:
    """Cancel ``job`` (caller holds ``_lock``): ``(admitted fork to interrupt or None, never
    launched)``. No worker will ever settle a job that never launched, so it is settled with
    ``outcome`` and unregistered here, and its queued dispatch becomes a no-op."""
    fork = job.run.cancel()
    if job.launched:
        return fork, False
    if job.queue_key is not None and _queued.get(job.queue_key) is job:
        del _queued[job.queue_key]
    _unregister_locked(job.capture.parent_ref(), job)
    job.handle._settle(outcome)
    return None, True


def _after_fence(job: _ForkJob, fork: Any, never_launched: bool, *, message: str, tool_reason: str) -> None:
    """The fence's unlocked half: interrupt an admitted fork; for a job that never launched, free
    its idle-queue slot and publish its request exit (so the canceller does not wait on it)."""
    from agent.background_review import _interrupt_background_review, finish_background_review_run
    from agent.review_idle_queue import QUEUE

    if fork is not None:
        _interrupt_background_review(fork, message=message, tool_reason=tool_reason)
    if never_launched:
        if job.queue_key is not None:
            QUEUE.discard(job.queue_key, job.dispatch)
        finish_background_review_run(job.capture.parent_ref(), job.run)


def _cancel_job(job: _ForkJob) -> None:
    with _lock:
        fork, never_launched = _fence_locked(job, "cancelled")
    _after_fence(job, fork, never_launched, message="plugin fork cancelled", tool_reason="plugin fork cancelled")


def launched_fork_runs(agent: Any) -> list:
    """The runs of ``agent``'s forks that are past launch (a deferred fork has not started)."""
    with _lock:
        return [job.run for job in (getattr(agent, "_plugin_fork_jobs", None) or ()) if job.launched]


def interrupt_plugin_forks(agent: Any, *, message: str, tool_reason: str) -> list:
    """Fence every fork of ``agent`` (deferred, launching or running) and refuse any fork of the
    current turn not registered yet; the runs to await (``request_done``)."""
    with _lock:
        agent._plugin_forks_fenced_turn = getattr(agent, "_current_turn_id", None)
        jobs = list(getattr(agent, "_plugin_fork_jobs", None) or ())
        fenced = [(job, *_fence_locked(job, "cancelled")) for job in jobs]
    for job, fork, never_launched in fenced:
        _after_fence(job, fork, never_launched, message=message, tool_reason=tool_reason)
    return [job.run for job in jobs]


def _pin_capture(fork: Any, capture: Any) -> None:
    """Same-model fork: replay the captured request inputs byte-for-byte (cache parity)."""
    import copy

    fork._cached_system_prompt = capture.system_prompt
    fork._cached_system_prompt_static = capture.system_prompt_static
    fork.tools = copy.deepcopy(capture.tools)
    fork.valid_tool_names = {tool["function"]["name"] for tool in fork.tools}
    fork._prompt_cache_fork_tag = PLUGIN_FORK_TASK


def _called_tool(tc: dict) -> Optional[str]:
    """The tool a call runs: its own name, or a ``tool_call`` carrier's single target (the only
    form the executor unwraps; the dispatch guard then checks the target, never the bridge)."""
    from tools.tool_search_catalog import TOOL_CALL_NAME
    from tools.tool_search_validation import normalize_tool_call_entries

    fn = tc.get("function") or {}
    name = fn.get("name")
    if name != TOOL_CALL_NAME:
        return name
    try:
        args = json.loads(fn.get("arguments") or "{}")
        entries, err = normalize_tool_call_entries(args if isinstance(args, dict) else {})
    except (TypeError, ValueError):
        return None
    return entries[0]["name"] if not err and len(entries) == 1 else None


def _tool_calls_after(messages: list, snapshot: list, allowed: frozenset) -> tuple[str, ...]:
    seen = {tc.get("id") for m in snapshot if isinstance(m, dict) for tc in (m.get("tool_calls") or ()) if isinstance(tc, dict)}
    return tuple(
        name for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or ()) if isinstance(tc, dict) and tc.get("id") not in seen
        for name in [_called_tool(tc)] if name in allowed
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
            # The allowlist is read again here, against the live session agent and the fork that
            # dispatches: the guard holds only tools that are still the plugin's own and still
            # reach its handler (or are marked fork-safe), however the job was built. It never
            # holds a Tool Search bridge: a ``tool_call`` reaches the guard unwrapped to its
            # target (after the fork's own scope check), so the bridge is only a carrier for
            # these names, and a call the executor does not unwrap is denied as ``tool_call``.
            allowed = set(job.tools) & fork_tool_allowlist(job.own_tools(), parent, fork)
            set_thread_tool_whitelist(allowed, deny_msg_fmt=(
                "This background task may only call: " + (", ".join(sorted(allowed)) or "no tools")
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
                actions = _tool_calls_after((result or {}).get("messages") or [], snapshot, frozenset(allowed))
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
            fork._loop_aux_task = None
            usage = release_cache_parity_fork(parent, fork, task=f"{PLUGIN_FORK_TASK}:{job.plugin_id}",
                                              session_id=capture.session_id)
            _track_child(parent, fork, register=False)
        finish_background_review_run(parent, job.run)
        with _lock:
            _unregister_locked(parent, job)
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
