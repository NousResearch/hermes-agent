"""``async_rewake`` hooks (Claude Code ``asyncRewake``): each firing runs the hook in its own process on
a background thread, the event that fired it never waits, and an exit code 2 wakes the session that
fired it with the hook's stderr (stdout when stderr is empty).

The wake is a ``hook_wake`` event on ``process_registry.completion_queue``, routed like a background
process's ``watch_match`` (the firing turn's session key and chat route), so every surface that wakes
an idle session for a background process delivers it between turns: the classic CLI drain, the
TUI/Desktop poller and the messaging gateway's idle watcher.

Guards, each answering a way a waiting hook goes wrong:

- Started only from a turn of a session that can be woken later (``spec.REWAKE_EVENTS``; not in a
  delegated child, not where ``async_delivery_supported()`` is false) and only when the bound session
  is the one the event names, so the route is provable.
- One live run per hook and session: a new firing replaces the previous run, so a hook on a per-turn
  event never piles up one waiting process per turn. A replaced run that already exited 2 still wakes.
- A checkpoint after ``max_consecutive_wakes`` wakes with no message from the user: that wake asks the
  model to summarize for the user and ask them to confirm or correct, and the hook is not started again
  until the user replies, so whatever it waits on stays at its source. A run already waiting at the
  checkpoint still delivers what it took (at most one per hook and session). A hook that re-fires on a
  condition the model does not clear, or two agents waking each other, cannot run on unattended.
- Runs stop without waking when their session is finalized, reset or left (``cancel_session``), when
  the hook is removed or revoked, and on every exit path (``stop_all``: ``atexit`` and the hard-exit sweep
  ``kill_live_foreground_processes`` the gateway, CLI and TUI already run before ``os._exit``).
- No run starts once a path about to ``os._exit`` has swept them (``stop_all(fence=True)``): a turn
  finishing after the sweep must not leave a hook behind ``os._exit``.
- At most ``MAX_LIVE_RUNS`` per profile, only the newest output is kept while a hook runs, and
  whatever a hook left in its process group ends with it."""

from __future__ import annotations

import atexit
import logging
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from agent import deadline
from agent.conversation_compression_scaffolding import is_user_role_scaffolding
from agent.delegation_context import is_delegated_child_context
from agent.memory_provider import spawn_context_thread
from agent.shell_hooks.spawn import spawn_hook
from agent.skill_commands import skills_loaded_by_scaffold, slugify_skill_name
from gateway.message_timestamps import strip_leading_message_timestamps
from gateway.session_context import async_delivery_supported, get_session_env
from hermes_cli import _subprocess_compat
from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE
from hermes_cli.heartbeat import HEARTBEAT_PROMPT_TEMPLATE
from hermes_cli.loops import WAKEUP_PROMPT_TEMPLATE, load_loop

logger = logging.getLogger(__name__)

WAKE_EXIT_CODE = 2
WAKE_OUTPUT_LIMIT = 4000
MAX_LIVE_RUNS = 64
# SIGTERM first, so a hook can hand back what it took from its source; SIGKILL after this.
_STOP_GRACE_SECONDS = 2.0
# Event payload field -> the session var that carries it (the route a background process stamps).
_ROUTE_ENV = (
    ("chat_id", "HERMES_SESSION_CHAT_ID"), ("user_id", "HERMES_SESSION_USER_ID"),
    ("user_name", "HERMES_SESSION_USER_NAME"), ("thread_id", "HERMES_SESSION_THREAD_ID"),
    ("message_id", "HERMES_SESSION_MESSAGE_ID"),
)


@dataclass(eq=False)
class _Run:
    key: tuple  # (hook key, route)
    session_ids: set
    proc: Optional[subprocess.Popen] = None
    stopped: bool = False  # guarded by _lock: a stopped run never wakes


_lock = threading.Lock()
_runs: dict[tuple, _Run] = {}
# (hook key, route) -> (wakes queued since the last turn of the user, the session ids it counts for)
_streaks: dict[tuple, tuple[int, frozenset]] = {}
_stopping: set = set()  # processes in their SIGTERM grace, so an exit can still SIGKILL them
_closed = False  # set by a fencing stop_all: the process is exiting
_quiet_skips: set = set()  # hook keys whose firing under another session's binding was logged at info
_loop_ticks_taken: dict[str, int] = {}  # session id -> the last /loop tick a turn was attributed to
# A tick fired before this process started had its turn in the process that fired it.
_STARTED_AT = time.time()
_atexit_installed = False
_HOUSEKEEPER_MARK = "shell_hook_rewake_housekeeper"


def start(spec: Any, hook_key: Optional[tuple], kwargs: dict[str, Any], stdin_json: str) -> None:
    """Start one background run of ``spec`` for the firing in ``kwargs``; returns at once. The route
    is read here, in the firing turn's context, and that context is handed to the run thread, so the
    hook gets the routed profile's environment as a synchronous hook would."""
    route = _route(kwargs)
    if route is None:
        _log_unroutable(spec, hook_key)
        return
    session_id, session_key = route
    run = _Run(key=(hook_key or (spec.event, spec.matcher, spec.command), session_key or session_id),
               session_ids={session_id})
    with _lock:
        previous = _runs.get(run.key)
        level, refusal = _refusal(spec, run, hook_key, previous)
        if refusal:
            logger.log(level, "async_rewake hook %s -> %s not started: %s", spec.event, spec.command, refusal)
            return
        _runs[run.key] = run
        replaced = previous is not None and _mark_stopped(previous, keep_finished_wake=True)
    if replaced:
        _terminate(previous.proc)
    _install_atexit()
    event = _wake_event(spec, session_id, session_key, get_session_env)
    spawn_context_thread(_run, name=f"shell-hook-rewake[{spec.event}]", args=(spec, run, stdin_json, event)).start()


def _refusal(spec: Any, run: _Run, hook_key: Optional[tuple], previous: Optional[_Run]) -> tuple[int, str]:
    """``(log level, why)`` ``run`` may not start; ``why`` is ``""`` when it may. Caller holds _lock."""
    if _closed:
        return logging.INFO, "the process is exiting"
    limit = spec.max_consecutive_wakes
    if limit and _streaks.get(run.key, (0,))[0] >= limit:  # said once at info, when the checkpoint was reached
        return logging.DEBUG, "waiting at its checkpoint for the user to reply"
    home = run.key[0][0] if hook_key else ""
    if previous is None and sum(1 for r in list(_runs.values()) if r.key[0][:1] == (home,)) >= MAX_LIVE_RUNS:
        return logging.INFO, f"{MAX_LIVE_RUNS} runs already wait for profile {home}"
    return logging.DEBUG, ""


def _log_unroutable(spec: Any, hook_key: Optional[tuple]) -> None:
    """A delegated child or a session that cannot be woken later is expected (debug). A firing under
    another session's binding is said once per hook at info: a hook on a per-turn event that silently
    stops being armed would look like a dead integration."""
    if is_delegated_child_context() or not async_delivery_supported():
        logger.debug("async_rewake hook %s -> %s not started: a delegated child or a session that cannot be "
                     "woken later", spec.event, spec.command)
        return
    with _lock:
        first = hook_key not in _quiet_skips
        _quiet_skips.add(hook_key)
    logger.log(logging.INFO if first else logging.DEBUG,
               "async_rewake hook %s -> %s not started: no session, or another session than the event names, is "
               "bound to this firing", spec.event, spec.command)


def _route(kwargs: dict[str, Any]) -> Optional[tuple[str, str]]:
    """``(session_id, session_key)`` of the session this firing may wake, or None. The bound session
    must be the one the event names: an event fired outside its turn can run under another
    session's binding, and a wake routed from there would reach the wrong conversation."""
    if is_delegated_child_context() or not async_delivery_supported():
        return None
    named = str(kwargs.get("session_id") or kwargs.get("parent_session_id") or "")  # subagent_stop names its parent
    bound = get_session_env("HERMES_SESSION_ID", "")
    if not bound or (named and named != bound):
        return None
    return bound, get_session_env("HERMES_SESSION_KEY", "")


def _wake_event(spec: Any, session_id: str, session_key: str, get_session_env) -> dict[str, Any]:
    """The ``hook_wake`` event minus its output. ``session_id`` is the producer id, as a background
    process's ``proc_...`` is; the Hermes session rides ``parent_session_id`` and ``session_key``. A
    session with no key (the in-process CLI) is keyed by its id, so the drain demands ownership proof
    (compression-lineage aware) instead of handing the wake to whichever session drains first."""
    event_id = uuid.uuid4().hex
    platform = get_session_env("HERMES_SESSION_PLATFORM", "")
    return {
        "type": "hook_wake", "event_id": event_id, "session_id": f"hook_{event_id[:12]}",
        "session_key": session_key or session_id, "parent_session_id": session_id,
        "command": spec.command, "hook_event": spec.event, "started_at": time.time(),
        "platform": platform,
        **{name: (get_session_env(var, "") if platform else "") for name, var in _ROUTE_ENV},
    }


def _run(spec: Any, run: _Run, stdin_json: str, event: dict[str, Any]) -> None:
    """The only thread that waits on the hook (a second waiter can reap it first and leave this one
    an exit status of 0), so the stop path only signals."""
    result = spawn_hook(spec, stdin_json, on_spawn=lambda proc: _attach(run, proc), output_limit=WAKE_OUTPUT_LIMIT)
    with _lock:
        if _runs.get(run.key) is run:
            del _runs[run.key]
        stopped = run.stopped
        number = _count_wake(spec, run) if not stopped and result["returncode"] == WAKE_EXIT_CODE else 0
    _kill_group_leftovers(run.proc)  # a helper the hook left behind would outlive its session
    if number:
        _deliver(spec, {**event, "wake_number": number, "wake_limit": spec.max_consecutive_wakes}, result)
    elif not stopped:
        _log_quiet_end(spec, result)


def _count_wake(spec: Any, run: _Run) -> int:
    """Count one wake: its number since the user's last message. A run that was already waiting when the
    checkpoint was reached still wakes (it took what it waited on), past the limit and with the same
    checkpoint; none starts after it. Caller holds _lock."""
    number = _streaks.get(run.key, (0, frozenset()))[0] + 1
    _streaks[run.key] = (number, frozenset(run.session_ids))
    if number == spec.max_consecutive_wakes:
        logger.info("async_rewake hook %s -> %s reached its checkpoint (%d wakes with no message from the user): "
                    "not started again until the user replies", spec.event, spec.command, number)
    return number


def _deliver(spec: Any, event: dict[str, Any], result: dict[str, Any]) -> None:
    from tools.process_registry import process_registry  # ~320 ms a process that never wakes never needs
    logger.info("async_rewake hook exited %d: waking session %s (event=%s command=%s, wake %d)", WAKE_EXIT_CODE,
                event["session_key"], spec.event, spec.command, event["wake_number"])
    process_registry.completion_queue.put({**event, "output": wake_output(result),
                                           "elapsed": int(result["elapsed_seconds"])})


def _log_quiet_end(spec: Any, result: dict[str, Any]) -> None:
    if result["error"]:
        logger.warning("async_rewake hook failed (event=%s command=%s): %s", spec.event, spec.command, result["error"])
    elif result["timed_out"]:
        logger.info("async_rewake hook stopped at its %ds timeout without waking (event=%s command=%s)",
                    spec.timeout, spec.event, spec.command)
    elif result["returncode"] not in (0, WAKE_EXIT_CODE):
        logger.warning("async_rewake hook exited %s (event=%s command=%s); no wake", result["returncode"],
                       spec.event, spec.command)


def wake_output(result: dict[str, Any]) -> str:
    """What the woken session reads: stderr, else stdout, capped to its newest characters."""
    text = (result.get("stderr") or "").strip() or (result.get("stdout") or "").strip()
    return text[-WAKE_OUTPUT_LIMIT:]


def _attach(run: _Run, proc: subprocess.Popen) -> None:
    with _lock:
        run.proc = proc
        stopped = run.stopped
    if stopped:  # stopped between registration and spawn
        _kill(proc)


def _mark_stopped(run: _Run, *, keep_finished_wake: bool = False) -> bool:
    """Mark ``run`` as never waking; False when it is left to wake (a replaced run whose hook already
    exited 2 delivers what it found). Caller holds _lock."""
    if keep_finished_wake and run.proc is not None and run.proc.returncode == WAKE_EXIT_CODE:
        return False
    run.stopped = True
    return True


def _terminate(proc: Optional[subprocess.Popen]) -> None:
    """SIGTERM the hook's tree now and SIGKILL it after the grace if it is still running, without
    blocking the caller (an agent turn or a session teardown) and without waiting on the process."""
    if proc is None or proc.returncode is not None:
        return
    with _lock:
        _stopping.add(proc)
    deadline.kill_process_tree(proc.pid, sig=signal.SIGTERM)
    spawn_context_thread(_escalate, name="shell-hook-rewake-stop", args=(proc,)).start()


def _escalate(proc: subprocess.Popen) -> None:
    until = time.monotonic() + _STOP_GRACE_SECONDS
    while proc.returncode is None and time.monotonic() < until:  # the run thread reaps it
        time.sleep(0.05)
    try:
        if proc.returncode is None:  # not reaped yet, so its pid and group are still ours
            _kill(proc)
    finally:
        with _lock:
            _stopping.discard(proc)


def _kill_group_leftovers(proc: Optional[subprocess.Popen]) -> None:
    """After the hook was reaped: SIGKILL what is left of the group it led (a helper it started, or
    one that ignored SIGTERM). The group id stays reserved while any member lives, so this reaches
    only it."""
    if proc is None or os.name != "posix":
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # windows-footgun: ok (POSIX-only branch)
    except (ProcessLookupError, PermissionError):
        pass


def _kill(proc: subprocess.Popen) -> None:
    _subprocess_compat.kill_process_tree(proc)


def _stop_where(predicate, streak_ids: frozenset = frozenset()) -> int:
    """Stop the runs ``predicate`` picks and forget their wake streaks (and those counted for
    ``streak_ids``, sessions that ended with no run waiting)."""
    with _lock:
        doomed = [run for run in _runs.values() if predicate(run)]
        for run in doomed:
            del _runs[run.key]
            run.stopped = True
        keys = {run.key for run in doomed}
        for key in [k for k, (_n, ids) in _streaks.items() if k in keys or ids & streak_ids]:
            del _streaks[key]
    for run in doomed:
        _terminate(run.proc)
    return len(doomed)


def cancel_hooks(hook_keys: Iterable[tuple]) -> None:
    """Stop the runs of hooks that were unregistered (removed from config or revoked)."""
    keys = set(hook_keys)
    if keys:
        _stop_where(lambda run: run.key[0] in keys)


def cancel_session(session_ids: Iterable[Any]) -> int:
    """Stop every run fired for one of ``session_ids``. Ids only: a teardown can run under another
    session's bound key (a gateway sweep), so the key would name the wrong conversation."""
    ids = frozenset(str(s) for s in session_ids if s)
    return _stop_where(lambda run: bool(run.session_ids & ids), ids)


def stop_all(*, fence: bool = True) -> int:
    """Stop every run now, SIGKILL included (interpreter exit, exit sweeps): nothing may outlive the
    process. ``fence`` also refuses every later start (a hard exit; the graceful sweep leaves it off,
    like the foreground fence). Returns how many processes were signalled. Never blocks on the lock:
    a signal handler can run this on a thread that holds it."""
    global _closed
    locked = _lock.acquire(timeout=1)
    try:
        _closed = _closed or fence
        doomed = list(_runs.values())
        _runs.clear()
        for run in doomed:
            run.stopped = True
        procs = {run.proc for run in doomed if run.proc is not None} | set(_stopping)
        _stopping.clear()
    finally:
        if locked:
            _lock.release()
    live = [proc for proc in procs if proc.returncode is None]
    for proc in live:
        _kill(proc)
    return len(live)


def live_run_count() -> int:
    with _lock:
        return len(_runs)


def reset_for_tests() -> None:
    global _closed
    stop_all()
    with _lock:
        _streaks.clear()
        _quiet_skips.clear()
        _loop_ticks_taken.clear()
        _closed = False


def _install_atexit() -> None:
    global _atexit_installed
    if not _atexit_installed:
        _atexit_installed = True
        atexit.register(stop_all)  # hooks lead their own process group: nothing else would stop them


# --- Session lifecycle: internal observers on the hook manager -----------------------------------

def _on_finalize(**kwargs: Any) -> None:
    keep = kwargs.get("new_session_id")
    cancel_session(s for s in (kwargs.get("session_id"), kwargs.get("old_session_id")) if s != keep)


def _on_reset(**kwargs: Any) -> None:
    # ``session_id`` is the new session here (gateway and CLI alike): only the old one ended.
    cancel_session((kwargs.get("old_session_id"),))


def _on_compaction(**kwargs: Any) -> None:
    """A rotation continues the conversation under a new id: its runs follow, so a teardown of the
    new id stops them and the next firing replaces them."""
    old, new = kwargs.get("old_session_id"), kwargs.get("session_id")
    if not old or not new or old == new:
        return
    displaced = []
    with _lock:
        for run in [run for run in _runs.values() if old in run.session_ids]:
            run.session_ids.add(new)
            if run.key[1] == old:
                _runs.pop(run.key, None)
                run.key = (run.key[0], new)
                if (taken := _runs.get(run.key)) is not None:  # a firing under the new id came first: it wins
                    _mark_stopped(run)
                    displaced.append(run)
                    _runs[run.key] = taken
                else:
                    _runs[run.key] = run
        for key in [key for key in _streaks if old in _streaks[key][1]]:
            streak, ids = _streaks.pop(key)
            _streaks[(key[0], new) if key[1] == old else key] = (streak, ids | {new})
        if old in _loop_ticks_taken:  # the loop migrates with the session, ticks counted on
            _loop_ticks_taken[new] = _loop_ticks_taken.pop(old)
    for run in displaced:
        _terminate(run.proc)


def _on_turn_start(**kwargs: Any) -> None:
    """A turn the user started ends every wake streak of its session. Hermes' own turns (a heartbeat,
    a background-process or delegation notice, another hook wake) do not: they are no sign of a
    person, and a heartbeat would otherwise keep the budget from ever stopping a loop."""
    text = _turn_text(str(kwargs.get("user_message") or ""))
    if not _from_the_user(text):
        return
    route = _route(kwargs)
    if route is None or _taken_by_loop_tick(route[0], text):
        return
    routes = {route[1] or route[0], route[0]}
    with _lock:
        for key in [key for key in _streaks if key[1] in routes]:
            del _streaks[key]


def _turn_text(text: str) -> str:
    """A turn's input without a gateway's rendered message timestamp and the ``[System note: ...]``
    Hermes puts ahead of a message (the gateway's restart and auto-continue notes)."""
    text = strip_leading_message_timestamps(text)[0].strip()
    while (note := re.match(r"\[System note:[^\]]*\]", text, re.IGNORECASE)) is not None:
        text = text[note.end():].strip()
    return text


def _taken_by_loop_tick(session_id: str, text: str) -> bool:
    """Whether this turn is the one a ``/loop 10m /<skill>`` tick injected, which reads exactly like
    the user's /skill. Every loop driver counts the tick (``ticks_fired``) before it injects the bare
    command, whenever it then completes the tick, and runs its turn in the process that fired it, so
    a turn loading that skill while a tick of this process has no turn yet is the loop's, one turn
    per tick. Kept per session for the life of the process, so leaving and resuming a session does
    not count its ticks again."""
    skills = {slugify_skill_name(name) for name in skills_loaded_by_scaffold(text)}
    loop = load_loop(session_id) if skills else None
    words = loop.prompt.split() if loop is not None and loop.status != "cleared" else []
    command = words[1] if words[:1] == ["/skill"] and len(words) > 1 else words[0][1:] if words[:1] and words[0][:1] == "/" else ""
    if not command or slugify_skill_name(command) not in skills:
        return False
    with _lock:
        before_us = loop.ticks_fired if (loop.last_fired_at or 0) < _STARTED_AT else 0
        if loop.ticks_fired <= _loop_ticks_taken.setdefault(session_id, before_us):
            return False
        _loop_ticks_taken[session_id] = loop.ticks_fired
    return True


def _from_the_user(text: str) -> bool:
    """Whether a turn's input (after ``_turn_text``) reads as a message the user wrote. A /skill the
    user typed, or their message under a channel's auto-loaded skill, is theirs. Not theirs: an empty
    turn, user-role scaffolding, a turn that opens with one of Hermes' own control frames
    (``CONTROL_FRAME_OPENERS``: every ``[IMPORTANT: ...]`` notice, runtime notes, the steer marker,
    compaction handoffs, delegation results), or one of its recurring prompts (/heartbeat, /loop,
    /goal). Text is all this hook sees, so a producer that sends plain text as the user (a plugin's
    ``inject_message``) counts as the user."""
    from agent.prompt_builder import CONTROL_FRAME_OPENERS  # ~120 ms a process with no async_rewake hook never needs
    if skills_loaded_by_scaffold(text):
        return True
    control_frame = r"\[(?:" + "|".join(opener.replace("]", r"\]") for opener in CONTROL_FRAME_OPENERS) + ")"
    recurring = tuple(t.split("{", 1)[0] for t in (HEARTBEAT_PROMPT_TEMPLATE, WAKEUP_PROMPT_TEMPLATE,
                                                   CONTINUATION_PROMPT_TEMPLATE))
    return bool(text) and not (is_user_role_scaffolding({"role": "user", "content": text})
                               or re.match(control_frame, text, re.IGNORECASE) or text.startswith(recurring))


_HOUSEKEEPING = {"on_session_finalize": _on_finalize, "on_session_reset": _on_reset,
                 "post_compaction": _on_compaction, "pre_llm_call": _on_turn_start}


def install_housekeeping(manager: Any) -> None:
    """Hang the lifecycle observers on ``manager`` once (a plugin force-reload clears ``_hooks`` and
    the next registration installs them again)."""
    for event, callback in _HOUSEKEEPING.items():
        callbacks = manager._hooks.setdefault(event, [])
        if not any(getattr(cb, _HOUSEKEEPER_MARK, False) for cb in callbacks):
            setattr(callback, _HOUSEKEEPER_MARK, True)
            callbacks.append(callback)
