"""``async_rewake`` shell hooks (Claude Code ``asyncRewake``): each firing runs the hook in its own
process on a background thread, the event that fired it never waits, and an exit code 2 wakes the
session that fired it with the hook's stderr (stdout when stderr is empty).

The wake is a ``hook_wake`` event on ``process_registry.completion_queue``, routed like a background
process's ``watch_match`` (the firing turn's session key and chat route). Every surface that already
wakes an idle session for a background process delivers it: the classic CLI drain, the TUI/Desktop
poller and the messaging gateway's idle watcher. So it runs between turns as a labelled user-role
notice (the one cache-safe channel), never inside a running turn and never ahead of queued user input.

One live run per hook and session: a new firing replaces the previous run (a hook on ``post_llm_call``
must not pile up one waiting process per turn). Runs end at their own exit, at their timeout, when
their hook is removed or revoked, when their session is finalized or reset, and at interpreter exit.
A delegated child never starts one (it has no idle loop to wake), nor does a session that cannot
receive a completion after its turn (one-shot runs, cron, Kanban workers, stateless endpoints)."""

from __future__ import annotations

import atexit
import logging
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

WAKE_EXIT_CODE = 2
WAKE_OUTPUT_LIMIT = 4000
# SIGTERM first, so a hook can deregister itself from whatever it waits on; SIGKILL after this.
_STOP_GRACE_SECONDS = 2.0
# Event payload field -> the session var that carries it (same route a background process stamps).
_ROUTE_ENV = (
    ("chat_id", "HERMES_SESSION_CHAT_ID"), ("user_id", "HERMES_SESSION_USER_ID"),
    ("user_name", "HERMES_SESSION_USER_NAME"), ("thread_id", "HERMES_SESSION_THREAD_ID"),
    ("message_id", "HERMES_SESSION_MESSAGE_ID"),
)


@dataclass(eq=False)
class _Run:
    key: tuple
    session_ids: frozenset
    proc: Optional[subprocess.Popen] = None
    stopped: threading.Event = field(default_factory=threading.Event)


_runs: dict[tuple, _Run] = {}
_runs_lock = threading.Lock()
_atexit_installed = False
_REAPER_MARK = "shell_hook_rewake_reaper"


def start(spec: Any, hook_key: Optional[tuple], kwargs: dict[str, Any], stdin_json: str) -> None:
    """Start one background run of ``spec`` for the firing in ``kwargs``; returns at once. Reads the
    session route here, in the firing turn's context, and hands that context to the run thread so the
    hook gets the routed profile's environment, as a synchronous hook would."""
    from agent.delegation_context import is_delegated_child_context
    from gateway.session_context import async_delivery_supported, get_session_env
    if is_delegated_child_context() or not async_delivery_supported():
        logger.debug("async_rewake hook %s -> %s not started: this session cannot be woken later",
                     spec.event, spec.command)
        return
    fired_for = str(kwargs.get("session_id") or kwargs.get("parent_session_id") or "")
    session_id = get_session_env("HERMES_SESSION_ID", "") or fired_for
    session_key = get_session_env("HERMES_SESSION_KEY", "")
    event = _wake_event(spec, session_id, session_key, get_session_env)
    # Both ids: compression can rotate the agent's id away from the one bound for the turn.
    run = _Run(key=(hook_key or (spec.event, spec.matcher, spec.command), session_key or fired_for),
               session_ids=frozenset(filter(None, (session_id, fired_for))))
    with _runs_lock:
        previous, _runs[run.key] = _runs.get(run.key), run
    if previous is not None:
        _stop(previous)
    _install_atexit()
    from agent.memory_provider import spawn_context_thread
    spawn_context_thread(_run, name=f"shell-hook-rewake[{spec.event}]", args=(spec, run, stdin_json, event)).start()


def _wake_event(spec: Any, session_id: str, session_key: str, get_session_env) -> dict[str, Any]:
    """The ``hook_wake`` event minus its output. ``session_id`` is the producer id, as a background
    process's ``proc_...`` is; the Hermes session rides ``parent_session_id`` and ``session_key``."""
    event_id = uuid.uuid4().hex
    platform = get_session_env("HERMES_SESSION_PLATFORM", "")
    return {
        "type": "hook_wake", "event_id": event_id, "session_id": f"hook_{event_id[:12]}",
        "session_key": session_key, "parent_session_id": session_id,
        "command": spec.command, "hook_event": spec.event, "started_at": time.time(),
        "platform": platform,
        **{name: (get_session_env(var, "") if platform else "") for name, var in _ROUTE_ENV},
    }


def _run(spec: Any, run: _Run, stdin_json: str, event: dict[str, Any]) -> None:
    from agent.shell_hooks import _spawn
    try:
        result = _spawn(spec, stdin_json, on_spawn=lambda proc: _attach(run, proc))
    finally:
        with _runs_lock:
            if _runs.get(run.key) is run:
                del _runs[run.key]
    if run.stopped.is_set():
        return
    if result["error"]:
        logger.warning("async_rewake hook failed (event=%s command=%s): %s", spec.event, spec.command, result["error"])
        return
    if result["timed_out"]:
        logger.info("async_rewake hook stopped at its %ds timeout without waking (event=%s command=%s)",
                    spec.timeout, spec.event, spec.command)
        return
    if result["returncode"] != WAKE_EXIT_CODE:
        if result["returncode"] != 0:
            logger.warning("async_rewake hook exited %s (event=%s command=%s); no wake", result["returncode"],
                           spec.event, spec.command)
        return
    from tools.process_registry import process_registry
    logger.info("async_rewake hook exited %d: waking session %s (event=%s command=%s)", WAKE_EXIT_CODE,
                event["session_key"] or event["parent_session_id"] or "<unrouted>", spec.event, spec.command)
    process_registry.completion_queue.put({**event, "output": wake_output(result),
                                           "elapsed": int(result["elapsed_seconds"])})


def wake_output(result: dict[str, Any]) -> str:
    """What the woken session reads: stderr, else stdout, capped to its newest characters."""
    text = (result.get("stderr") or "").strip() or (result.get("stdout") or "").strip()
    return text[-WAKE_OUTPUT_LIMIT:]


def _attach(run: _Run, proc: subprocess.Popen) -> None:
    run.proc = proc
    if run.stopped.is_set():  # stopped between registration and spawn
        _kill(proc)


def _stop(run: _Run, *, grace: float = _STOP_GRACE_SECONDS) -> None:
    """End ``run`` without a wake; never blocks the caller (an agent turn or a session teardown)."""
    run.stopped.set()
    proc = run.proc
    if proc is not None and proc.poll() is None:
        from agent.memory_provider import spawn_context_thread
        spawn_context_thread(_kill, name="shell-hook-rewake-stop", args=(proc, grace)).start()


def _kill(proc: subprocess.Popen, grace: float = 0.0) -> None:
    from hermes_cli._subprocess_compat import kill_process_tree
    kill_process_tree(proc, grace=grace)


def _stop_where(predicate) -> int:
    with _runs_lock:
        doomed = [run for run in _runs.values() if predicate(run)]
        for run in doomed:
            del _runs[run.key]
    for run in doomed:
        _stop(run)
    return len(doomed)


def cancel_hooks(hook_keys: Iterable[tuple]) -> None:
    """Stop the runs of hooks that were unregistered (removed from config or revoked)."""
    keys = set(hook_keys)
    if keys:
        _stop_where(lambda run: run.key[0] in keys)


def cancel_session(session_ids: Iterable[Any]) -> int:
    """Stop every run fired for one of ``session_ids``. Ids only: a teardown can run under another
    session's bound key (a gateway sweep), so the key would name the wrong conversation."""
    ids = {str(s) for s in session_ids if s}
    return _stop_where(lambda run: bool(run.session_ids & ids))


def cancel_all() -> None:
    """Stop every run now (interpreter exit, tests)."""
    with _runs_lock:
        doomed = list(_runs.values())
        _runs.clear()
    for run in doomed:
        run.stopped.set()
        if run.proc is not None and run.proc.poll() is None:
            _kill(run.proc)


def live_run_count() -> int:
    with _runs_lock:
        return len(_runs)


def _install_atexit() -> None:
    global _atexit_installed
    if not _atexit_installed:
        _atexit_installed = True
        atexit.register(cancel_all)  # hooks lead their own process group: nothing else would stop them


def _reap_session(**kwargs: Any) -> None:
    cancel_session((kwargs.get("session_id"), kwargs.get("old_session_id")))


def install_session_reaper(manager: Any) -> None:
    """Stop a session's runs when it is finalized or reset, so a hook waiting on an ended session
    neither outlives it nor wakes its successor. Idempotent per manager; a plugin force-reload clears
    ``_hooks`` and re-registration installs it again."""
    for event in ("on_session_finalize", "on_session_reset"):
        callbacks = manager._hooks.setdefault(event, [])
        if not any(getattr(cb, _REAPER_MARK, False) for cb in callbacks):
            setattr(_reap_session, _REAPER_MARK, True)
            callbacks.append(_reap_session)
