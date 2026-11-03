"""Windows gateway hang watchdog: an OS-scheduled liveness probe that lives OUTSIDE the gateway.

A Scheduled Task (``Hermes_Watchdog[_<profile>]``, registered by ``hermes gateway install`` beside the
gateway task) runs ``python -m hermes_cli.gateway_windows_watchdog --once`` every minute while the user
is signed in, from registration and after each logon.
Each run is a fresh process: it shares no cron, event loop or process with the gateway, so a gateway
that is wedged, frozen or gone cannot take its own watchdog down with it.

One run: classify the gateway serving this ``HERMES_HOME`` -> update the persisted failure streak ->
restart only on the third consecutive failure, within a restart cap and an exponential backoff after
restarts that did not bring the gateway back. Every failure, restart attempt and outcome is a JSON line
in ``logs/gateway-watchdog.log`` built from fixed codes only (never messages, argv or environment), so
nothing secret can reach it.

Operator stop wins: ``hermes gateway stop`` writes a fresh token to ``state/gateway-watchdog.operator-stop``,
a file only it writes. Every decision compares against that token, never against anything in this
module's own state file, so no interleaving of the two writers can turn a stop back into a restart.

Outage notice: right before a restart the window (last healthy check, first failed check) is queued
in the planned-restart marker, and the recovered gateway sends it to its home channels, asking for
requests from that window to be resent (``gateway/watchdog_outage.py``). This process holds no
messaging credentials and sends nothing itself.

Responsiveness, not process existence: a live gateway is judged by
``hermes_cli.gateway.probe_gateway_loop_liveness`` — the loop-tick witness the event loop answers itself
plus the heartbeat file, the same contract ``hermes update``/``restart`` trust before escalating. The
control pipe's ``identify`` is NOT used: native worker threads answer it while the loop is frozen. A
gateway that hangs before its first heartbeat stays ``unknown`` and is left alone (no evidence, no kill).

Responsibility boundaries (this module only decides WHEN to restart a hung or vanished gateway):
- ``gateway/shutdown_watchdog.py``: in-process threads that hard-exit a wedged shutdown/loop. They
  share the gateway's fate; this probe covers what they cannot (a frozen or dead process).
- Gateway task ``RestartOnFailure``: relaunches only when the task action fails. The action is a
  wscript launcher that exits once python starts, so a live-but-hung gateway is invisible to it.
- systemd ``WatchdogSec`` (``gateway.systemd_watchdog_seconds``): the Linux equivalent, untouched.
  This module is wired only into the Windows backend; Linux/macOS service paths do not import it.
- Delivery queue / restart resilience inside the gateway: own what survives a restart. The watchdog
  restarts through the gateway's own Scheduled Task (``_spawn_detached`` when none is registered)
  and relies on them.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional

ALIVE, WEDGED, DOWN, UNKNOWN, STOPPED, UPDATING = "alive", "wedged", "down", "unknown", "stopped", "updating"
STARTUP_FAILED = "startup_failed"
_FAILING = frozenset({WEDGED, DOWN})

ACTION_NONE = "none"
ACTION_RESTART = "restart"
ACTION_SUPPRESSED_BACKOFF = "suppressed_backoff"
ACTION_SUPPRESSED_LIMIT = "suppressed_limit"
REASON_UPDATE_IN_PROGRESS = "update_in_progress"  # yielded before touching the gateway
REASON_UPDATE_AFTER_STOP = "update_in_progress_after_stop"  # yielded after stopping it: still owed
REASON_OPERATOR_STOPPED = "operator_stopped"  # `hermes gateway stop` landed mid-restart: never start
REASON_SPAWN_LOCK_UNAVAILABLE = "spawn_lock_unavailable"  # cannot sync with a stop: fail closed

FAILURE_THRESHOLD = 3
RESTART_LIMIT = 3
RESTART_WINDOW_S = 3600
BACKOFF_BASE_S = 300
BACKOFF_MAX_S = 3600
# Gateway boot (adapter connects) precedes the first heartbeat; a slow boot must not read as failure.
RECOVERY_TIMEOUT_S = 180.0
RECOVERY_POLL_S = 5.0
# Spawn section: the start plus the wait until a stop's PID sweep can see the new process.
SPAWN_VISIBLE_TIMEOUT_S = 60.0
# A stop drains (<= 30 s) inside the section; the watchdog then gives up rather than start blind.
SPAWN_LOCK_TIMEOUT_S = 120.0
# Longer than the watchdog ever holds the section (schtasks /Run 15 s + visibility wait).
STOP_LOCK_TIMEOUT_S = 180.0

TASK_PREFIX = "Hermes_Watchdog"
_TASK_DESCRIPTION = "Hermes Agent Gateway hang watchdog"
_TASK_INTERVAL = "PT1M"
# Starts after the gateway task's own 30 s logon delay.
_TASK_LOGON_DELAY = "PT2M"
_TASK_TIME_LIMIT = "PT10M"
_MODULE = "hermes_cli.gateway_windows_watchdog"

LOG_FIELDS = ("ts", "event", "verdict", "action", "reason", "result", "pid", "consecutive_failures", "backoff_until")
_LOG_MAX_BYTES = 1024 * 1024


# ── Probe

def classify(
    home: Path, *, live_pid: Optional[Callable[[Path], Optional[int]]] = None,
    read_runtime: Optional[Callable[[Path], Any]] = None, loop_probe: Optional[Callable[..., str]] = None,
    retained_state: Optional[Callable[[Any], str]] = None,
) -> tuple[str, Optional[int]]:
    """``(verdict, pid)`` for the gateway serving ``home``. The keyword seams default to the canonical
    readers, resolved late so importing this module stays cheap for the task action.

    A missing gateway is ``down`` only when an in-process watchdog stamped the hard exit of a wedged
    loop (``retained_gateway_state`` -> ``degraded``). Anything else is ``stopped``: Ctrl+C on a
    foreground run and a force-stopped hung gateway both leave ``gateway_state: running`` behind, and
    Windows ``gateway stop`` records no ``desired_state``, so a leftover "running" proves no crash."""
    if live_pid is None or read_runtime is None or retained_state is None:
        import gateway.status as gateway_status
        live_pid = live_pid or gateway_status.live_gateway_pid_for_home
        read_runtime = read_runtime or gateway_status.read_runtime_status
        retained_state = retained_state or gateway_status.retained_gateway_state
    pid = live_pid(home)
    if pid is None:
        retained = retained_state(read_runtime(home / "gateway_state.json"))
        return {"degraded": DOWN, "startup_failed": STARTUP_FAILED}.get(retained, STOPPED), None
    if loop_probe is None:
        from hermes_cli.gateway import probe_gateway_loop_liveness as loop_probe
    # ALIVE/WEDGED share probe_gateway_loop_liveness's values; anything else is ambiguity.
    loop = loop_probe(pid, home=home)
    return (loop if loop in (ALIVE, WEDGED) else UNKNOWN), pid


def update_in_progress() -> bool:
    """``hermes update`` pauses the gateway and swaps the venv; a respawn then would hold the files
    the update replaces. Fails closed: an unanswerable probe is treated as an update."""
    try:
        from hermes_cli.update_lock import update_in_progress as live_update
        return live_update()
    except Exception:
        return True


# ── Decision

def _optional_number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _normalized(state: Any) -> dict[str, Any]:
    raw = state if isinstance(state, dict) else {}
    times = raw.get("restart_times")
    return {
        "consecutive_failures": int(raw.get("consecutive_failures") or 0),
        "failed_restarts": int(raw.get("failed_restarts") or 0),
        "backoff_until": float(raw.get("backoff_until") or 0.0),
        "restart_times": [float(t) for t in times] if isinstance(times, list) else [],
        "last_verdict": raw.get("last_verdict"),
        # The stop token current when this watchdog took the gateway down (None: nothing owed).
        "owed_token": raw.get("owed_token") if isinstance(raw.get("owed_token"), str) else None,
        # The stop token the previous tick saw (None: no tick yet).
        "seen_token": raw.get("seen_token") if isinstance(raw.get("seen_token"), str) else None,
        "last_alive_at": _optional_number(raw.get("last_alive_at")),
        "first_failed_at": _optional_number(raw.get("first_failed_at")),
    }


def decide(state: Any, verdict: str, now: float, stop_token: str = "") -> tuple[dict[str, Any], str]:
    """Pure transition: ``unknown`` neither counts nor resets the streak (ambiguity is no evidence).

    A gateway this watchdog took down (``owed_token``) counts as failing while it is absent: its
    leftover state alone would read as an operator stop. The debt ends at a successful restart, at
    any live process (whatever it does next is judged on its own), at a start that failed on its own
    (``startup_failed``: restarting a broken config cannot help) and at ``hermes gateway stop``.

    The operator's stop is not part of this state: it is ``stop_token``, read from the marker only
    ``hermes gateway stop`` writes. A debt binds the token it was taken under, and a token that
    moved since the last tick clears the streak, so a stale state file — whatever overwrote what —
    can never turn an operator stop back into a restart."""
    state = _normalized(state)
    if state["seen_token"] is not None and state["seen_token"] != stop_token:
        state.update(consecutive_failures=0, first_failed_at=None)
    state.update(seen_token=stop_token, last_verdict=verdict)
    if state["owed_token"] != stop_token:
        state["owed_token"] = None
    if verdict == ALIVE:
        state.update(consecutive_failures=0, failed_restarts=0, backoff_until=0.0, owed_token=None,
                     last_alive_at=now, first_failed_at=None)
        return state, ACTION_NONE
    if verdict in (WEDGED, UNKNOWN, STARTUP_FAILED):
        state["owed_token"] = None
    failing = verdict in _FAILING or (verdict == STOPPED and state["owed_token"] is not None)
    if verdict in (STOPPED, UPDATING, STARTUP_FAILED) and not failing:
        state.update(consecutive_failures=0, first_failed_at=None)
        return state, ACTION_NONE
    if not failing:
        return state, ACTION_NONE
    state["consecutive_failures"] += 1
    if state["first_failed_at"] is None:
        state["first_failed_at"] = now
    if state["consecutive_failures"] < FAILURE_THRESHOLD:
        return state, ACTION_NONE
    if now < state["backoff_until"]:
        return state, ACTION_SUPPRESSED_BACKOFF
    state["restart_times"] = [t for t in state["restart_times"] if now - t < RESTART_WINDOW_S]
    if len(state["restart_times"]) >= RESTART_LIMIT:
        return state, ACTION_SUPPRESSED_LIMIT
    return state, ACTION_RESTART


def _record_restart_result(state: dict[str, Any], ok: bool, now: float) -> dict[str, Any]:
    state["consecutive_failures"] = 0
    if ok:
        state.update(failed_restarts=0, backoff_until=0.0, owed_token=None, first_failed_at=None)
    else:
        state["failed_restarts"] += 1
        state["backoff_until"] = now + min(BACKOFF_BASE_S * 2 ** (state["failed_restarts"] - 1), BACKOFF_MAX_S)
    return state


# ── State + log (both under HERMES_HOME; neither carries anything but codes and numbers)

def _state_path(home: Path) -> Path:
    return home / "state" / "gateway-watchdog.json"


def _log_path(home: Path) -> Path:
    return home / "logs" / "gateway-watchdog.log"


def load_state(home: Path) -> dict[str, Any]:
    try:
        return _normalized(json.loads(_state_path(home).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):  # a corrupt file must not stop every future tick
        return _normalized({})


def _stop_marker_path(home: Path) -> Path:
    return home / "state" / "gateway-watchdog.operator-stop"


_STOP_WRITE_ATTEMPTS = 5


def forget_owed_restart(home: Path) -> None:
    """``hermes gateway stop``: record a fresh stop token in a marker ONLY this path writes.

    The watchdog's own state file is never touched here, so no read/modify/replace of it — however
    it interleaves with this stop — can lose the stop: every decision compares against the token
    (``decide``, ``_stopped``), and any debt or streak from before it is void. Retries a replace the
    watchdog's brief read holds open; never raises, so the stop itself always proceeds."""
    path = _stop_marker_path(home)
    token = f"{time.time():.6f}-{uuid.uuid4().hex}"
    for attempt in range(_STOP_WRITE_ATTEMPTS):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(token)
            try:
                os.replace(tmp, path)
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
            return
        except OSError:
            if attempt + 1 < _STOP_WRITE_ATTEMPTS:
                time.sleep(0.05)


def read_stop_token(home: Path) -> str:
    """The latest operator stop token; ``""`` when no stop was ever recorded. Fails closed: an
    unreadable marker yields a token nothing else equals, so it reads as a stop just recorded and
    nothing is restarted until the marker can be read again."""
    try:
        return _stop_marker_path(home).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except (OSError, ValueError):
        return f"unreadable-{uuid.uuid4().hex}"


def _stopped(home: Path, tick_token: str) -> bool:
    return read_stop_token(home) != tick_token


def owed_restart_pending(home: Path) -> bool:
    """True while this watchdog still owes a restart no operator stop has cancelled."""
    owed = load_state(home)["owed_token"]
    return owed is not None and owed == read_stop_token(home)


def _save_state(home: Path, state: dict[str, Any]) -> None:
    path = _state_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(path)


def _log(home: Path, event: str, **fields: Any) -> None:
    """Append one JSON line. A full or locked disk must not stop the probe, so write errors are dropped."""
    record = {"ts": datetime.now(UTC).isoformat(), "event": event}
    record.update({k: v for k, v in fields.items() if k in LOG_FIELDS and v is not None})
    path = _log_path(home)
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > _LOG_MAX_BYTES:
            path.replace(path.with_suffix(".log.1"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


# ── Restart

def wait_for_recovery(
    home: Path, *, timeout: float = RECOVERY_TIMEOUT_S, interval: float = RECOVERY_POLL_S,
    sleep: Callable[[float], None] = time.sleep, monotonic: Callable[[], float] = time.monotonic,
    probe: Callable[[Path], tuple[str, Optional[int]]] = classify,
) -> bool:
    """True once the gateway answers as ``alive`` within ``timeout``."""
    deadline = monotonic() + timeout
    while True:
        if probe(home)[0] == ALIVE:
            return True
        if monotonic() >= deadline:
            return False
        sleep(interval)


def start_gateway(
    *, home: Path, timeout: float = SPAWN_VISIBLE_TIMEOUT_S, sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """Start through the gateway's own Scheduled Task when it is registered, so the new gateway
    descends from that task's launcher (no execution limit), exactly as at logon, rather than from
    this watchdog run and its 10-minute limit; else the same detached spawn as ``hermes gateway start``.

    Returns once the new process is visible to ``_gateway_pids`` — the scan ``hermes gateway stop``
    sweeps — or after ``timeout`` (False). Runs inside the spawn section (``spawn_lock``), so a stop
    waiting on that section always finds the gateway this started."""
    from hermes_cli import gateway_windows as gw
    if not (gw.is_task_registered() and gw._exec_schtasks(["/Run", "/TN", gw.get_task_name()])[0] == 0):
        gw._spawn_detached(home=home)
    deadline = monotonic() + timeout
    while not gw._gateway_pids():
        if monotonic() >= deadline:
            return False
        sleep(0.25)
    return True


@contextlib.contextmanager
def spawn_lock(home: Path, *, timeout: float = SPAWN_LOCK_TIMEOUT_S):
    """Cross-process exclusive section shared by the watchdog's final-check+start and the operator
    stop's sweep (``state/gateway-watchdog.spawn.lock``; 1-byte ``msvcrt.locking`` range on Windows,
    ``flock`` on POSIX, as elsewhere in Hermes). Yields True when held, False when it could not be
    taken within ``timeout`` or the lock file could not be opened — callers decide how to fail closed."""
    try:
        handle = _open_lock_file(home)
    except OSError:
        yield False
        return
    held = False
    try:
        deadline = time.monotonic() + timeout
        while not (held := _try_lock(handle)) and time.monotonic() < deadline:
            time.sleep(0.05)
        yield held
    finally:
        if held:
            with contextlib.suppress(OSError):
                _unlock(handle)
        handle.close()


def _open_lock_file(home: Path):
    path = home / "state" / "gateway-watchdog.spawn.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")  # closed by spawn_lock
    if handle.seek(0, os.SEEK_END) == 0:
        handle.write(b"\0")  # msvcrt.locking needs a byte to lock
        handle.flush()
    return handle


def _try_lock(handle) -> bool:
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def operator_stop(home: Path, sweep: Callable[[], Any], *, lock_timeout: float = STOP_LOCK_TIMEOUT_S) -> bool:
    """``hermes gateway stop``'s half of the protocol: persist the stop intent FIRST, then sweep
    inside the spawn section. A watchdog start either had not reached its final check when the token
    moved (it then starts nothing) or completed its whole section — gateway started AND visible —
    before this sweep looked. The sweep always runs: a section that cannot be taken (a wedged holder,
    an unopenable lock) never skips the operator's stop. Returns whether the section was held."""
    forget_owed_restart(home)
    with spawn_lock(home, timeout=lock_timeout) as held:
        sweep()
    return held


def announce_outage(home: Path, outage: dict[str, Any]) -> None:
    """Queue the recovered gateway's home-channel outage notice (``gateway/watchdog_outage.py``):
    only the window's timestamps are written; the gateway sends it through its own adapters."""
    from gateway.watchdog_outage import record_outage
    record_outage(home, last_healthy_at=outage.get("last_healthy_at"), detected_at=outage["detected_at"])


def restart_gateway(
    home: Path, pid: Optional[int], *, wait: Callable[[Path], bool] = wait_for_recovery,
    stop: Optional[Callable[[int], bool]] = None, spawn: Callable[..., Any] = start_gateway,
    updating: Callable[[], bool] = update_in_progress, stop_token: Optional[str] = None,
    outage: Optional[dict[str, Any]] = None, announce: Callable[[Path, dict[str, Any]], None] = announce_outage,
    lock_timeout: float = SPAWN_LOCK_TIMEOUT_S,
) -> tuple[bool, str]:
    """Stop the wedged ``pid`` (``_escalate_wedged_gateway``: terminate, then a start-time-guarded
    force kill), start a fresh gateway, confirm it answers. An update or an operator stop that began
    since the probe wins: both are re-checked before each step that touches the gateway.
    ``stop_token`` is the stop token the tick started from, so a stop landing before this call counts.

    The outage notice is queued right before the start — never for a restart that did not happen —
    and a failure to queue it is logged (type only) and never holds the restart back.

    The final stop check and the start run as ONE section under ``spawn_lock``, the lock the operator
    stop sweeps under after persisting its token (``operator_stop``): no stop can slip between the
    check and the start. A section that cannot be locked fails closed — nothing is started."""
    if stop_token is None:
        stop_token = read_stop_token(home)
    if pid is not None:
        if updating():
            return False, REASON_UPDATE_IN_PROGRESS
        if _stopped(home, stop_token):
            return False, REASON_OPERATOR_STOPPED
        if stop is None:
            from hermes_cli.gateway import _escalate_wedged_gateway as stop
        if not stop(pid):
            return False, "stop_failed"  # a second gateway beside a live one would fight it for the lock
    if updating():
        return False, REASON_UPDATE_IN_PROGRESS if pid is None else REASON_UPDATE_AFTER_STOP
    if _stopped(home, stop_token):
        return False, REASON_OPERATOR_STOPPED
    if outage is not None:
        try:
            announce(home, outage)
        except Exception as exc:  # message text may carry anything; only the type is logged
            _log(home, "notice_failed", reason=f"error:{type(exc).__name__}")
    with spawn_lock(home, timeout=lock_timeout) as held:
        if not held:
            return False, REASON_SPAWN_LOCK_UNAVAILABLE
        if _stopped(home, stop_token):  # final check: a stop up to now, the notice write included
            return False, REASON_OPERATOR_STOPPED
        spawn(home=home)
    return (True, "alive") if wait(home) else (False, "not_recovered")


def run_once(
    home: Path, *, probe: Callable[[Path], tuple[str, Optional[int]]] = classify,
    restart: Callable[..., tuple[bool, str]] = restart_gateway,
    clock: Callable[[], float] = time.time, updating: Callable[[], bool] = update_in_progress,
) -> str:
    """One watchdog tick; returns the action taken. The stop token is read once, before anything
    else, and every later step compares against it."""
    home = Path(home)
    tick_token = read_stop_token(home)
    previous = load_state(home)
    verdict, pid = (UPDATING, None) if updating() else probe(home)
    state, action = decide(previous, verdict, clock(), tick_token)
    counted = state["consecutive_failures"] > previous["consecutive_failures"]
    if counted or verdict != previous["last_verdict"]:
        _log(home, "probe", verdict=verdict, pid=pid, consecutive_failures=state["consecutive_failures"])
    if action == ACTION_RESTART:
        state = _attempt_restart(home, state, verdict, pid, restart, clock, tick_token)
    elif action != ACTION_NONE:
        _log(home, "restart_suppressed", action=action, consecutive_failures=state["consecutive_failures"],
             backoff_until=state["backoff_until"] or None)
    # No reconciliation is needed before saving: a stop that lands at any point from here on moves
    # the token, which voids this state's debt and streak at the next tick.
    _save_state(home, state)
    return action


def _cancel_for_stop(state: dict[str, Any]) -> dict[str, Any]:
    state.update(owed_token=None, consecutive_failures=0)
    return state


def _attempt_restart(
    home: Path, state: dict[str, Any], verdict: str, pid: Optional[int],
    restart: Callable[..., tuple[bool, str]], clock: Callable[[], float], tick_token: str,
) -> dict[str, Any]:
    if _stopped(home, tick_token):
        # The operator stopped the hung gateway while we probed it: nothing to restart.
        _log(home, "restart_result", result="cancelled_by_stop", reason=REASON_OPERATOR_STOPPED)
        return _cancel_for_stop(state)
    owed_before = state["owed_token"]
    outage = {"last_healthy_at": state["last_alive_at"], "detected_at": state["first_failed_at"] or clock()}
    # Persist the attempt first: a watchdog killed mid-restart must still count it against the cap.
    state["restart_times"].append(clock())
    state["owed_token"] = tick_token  # void as soon as any stop moves the token
    _save_state(home, state)
    _log(home, "restart_begin", reason=verdict, pid=pid, consecutive_failures=state["consecutive_failures"])
    try:
        ok, reason = restart(home, pid, stop_token=tick_token, outage=outage)
    except Exception as exc:  # message text may carry anything; only the type is logged
        ok, reason = False, f"error:{type(exc).__name__}"
    if _stopped(home, tick_token):
        # The operator stopped it mid-restart (restart_gateway then starts nothing) or while we
        # waited for the new gateway: their stop, not our failure.
        state, result = _cancel_for_stop(state), "cancelled_by_stop"
    elif reason in (REASON_UPDATE_IN_PROGRESS, REASON_UPDATE_AFTER_STOP):
        # Yielded to an update: not an attempt, not a failure; owed only if we already stopped it.
        state["restart_times"].pop()
        state["consecutive_failures"] = 0
        state["owed_token"] = tick_token if reason == REASON_UPDATE_AFTER_STOP else owed_before
        result = "skipped"
    else:
        state = _record_restart_result(state, ok, clock())
        result = "recovered" if ok else "failed"
    _log(home, "restart_result", result=result, reason=reason, backoff_until=state["backoff_until"] or None)
    return state


# ── Scheduled Task (registered/removed/refreshed by the gateway service paths in gateway_windows)

def task_name() -> str:
    """``Hermes_Watchdog[_<suffix>]``: the gateway task's profile suffix, but outside the
    ``Hermes_Gateway*`` namespace every gateway-task scan (``registered_task_launchers``) matches."""
    from hermes_cli import gateway_windows as gw
    return TASK_PREFIX + gw.get_task_name()[len(gw._TASK_NAME_DEFAULT):]


def launcher_path() -> Path:
    from hermes_cli import gateway_windows as gw
    return gw._hermes_home() / "gateway-service" / f"{gw._sanitize_filename(task_name())}.vbs"


def write_launcher() -> Path:
    """Hidden-console launcher sharing the gateway's interpreter/env rendering; waits for python so
    the task instance spans the run and ``IgnoreNew`` keeps probes from overlapping."""
    from hermes_cli import gateway_windows as gw
    path = launcher_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    script = gw._build_gateway_vbs_script(*gw._launcher_settings(), module=_MODULE, args=["--once"], wait=True)
    gw._atomic_write(path, script, path.with_name(path.name + ".tmp"), encoding="utf-16")
    return path


def build_task_xml(launcher: Path, user: Optional[str], *, start_boundary: Optional[str] = None) -> str:
    """Two triggers repeating every minute: a TimeTrigger from registration (a LogonTrigger alone
    first fires at the NEXT logon, leaving a fresh install unwatched) and the LogonTrigger, whose
    delay lets the gateway task start first after sign-in. ``InteractiveToken`` runs either only
    while the user is signed in, like the gateway itself."""
    from xml.sax.saxutils import escape
    boundary = start_boundary or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")  # local time, as Task Scheduler writes it
    user_principal = f"\n      <UserId>{escape(user)}</UserId>" if user else ""
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>{escape(_TASK_DESCRIPTION)}</Description>
  </RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <Enabled>true</Enabled>
      <StartBoundary>{escape(boundary)}</StartBoundary>
      <Repetition>
        <Interval>{_TASK_INTERVAL}</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
    </TimeTrigger>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <Delay>{_TASK_LOGON_DELAY}</Delay>
      <Repetition>
        <Interval>{_TASK_INTERVAL}</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">{user_principal}
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>{_TASK_TIME_LIMIT}</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>wscript.exe</Command>
      <Arguments>//B //Nologo "{escape(str(launcher))}"</Arguments>
    </Exec>
  </Actions>
</Task>
"""


def is_registered(name: str) -> bool:
    from hermes_cli import gateway_windows as gw
    return gw._exec_schtasks(["/Query", "/TN", name])[0] == 0


def install_task() -> tuple[bool, str]:
    """Create or replace the watchdog task; ``(ok, detail)``."""
    from hermes_cli import gateway_windows as gw
    launcher = write_launcher()
    return gw._install_scheduled_task(task_name(), launcher, task_xml=build_task_xml(launcher, gw._resolve_task_user()))


def uninstall_task() -> list[str]:
    """Remove the watchdog task (never elevates) and its launcher; printable messages."""
    from hermes_cli import gateway_windows as gw
    messages: list[str] = []
    name = task_name()
    if is_registered(name):
        code, _out, err = gw._exec_schtasks(["/Delete", "/F", "/TN", name])
        messages.append(f"✓ Removed gateway watchdog Scheduled Task {name!r}" if code == 0
                        else f"⚠ Gateway watchdog Scheduled Task {name!r} remains (schtasks code {code}): {err.strip()}")
    path = launcher_path()
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
        messages.append(f"✓ Removed gateway watchdog launcher: {path}")
    return messages


# Never StartBoundary: it is the registration time and differs on every install.
_DRIFT_LEAVES = (
    "Task/Triggers/TimeTrigger/Repetition/Interval",
    "Task/Triggers/LogonTrigger/Delay",
    "Task/Triggers/LogonTrigger/Repetition/Interval",
    "Task/Settings/ExecutionTimeLimit",
    "Task/Settings/MultipleInstancesPolicy",
    "Task/Actions/Exec/Arguments",
)


def task_drift(registered_xml: Optional[str], template_xml: str) -> list[str]:
    """Template leaves the registered task lacks or differs on; empty when unqueryable (fail open)."""
    from hermes_cli import gateway_windows as gw
    live = gw._task_xml_leaf_values(registered_xml) if registered_xml else None
    want = gw._task_xml_leaf_values(template_xml)
    if live is None or want is None:
        return []
    return [leaf.rsplit("/", 1)[-1] for leaf in _DRIFT_LEAVES if live.get(leaf) != want.get(leaf)]


def refresh_if_installed() -> list[str]:
    """``hermes update``: rewrite an installed watchdog's launcher and re-register a drifted task.
    Never enables the watchdog where it is not installed (that is ``hermes gateway install``)."""
    from hermes_cli import gateway_windows as gw
    name = task_name()
    if not is_registered(name):
        return []
    launcher = write_launcher()
    messages = [f"✓ Refreshed gateway watchdog launcher: {launcher}"]
    template = build_task_xml(launcher, gw._resolve_task_user())
    drift = task_drift(gw._query_scheduled_task_xml(name), template)
    if drift:
        ok, detail = gw._install_scheduled_task(name, launcher, task_xml=template)
        messages.append(f"{'✓' if ok else '⚠'} Repaired gateway watchdog task ({', '.join(drift)}): {detail}")
    return messages


# ── Entry point (the Scheduled Task action)

def _status(home: Path) -> int:
    print(json.dumps({"task": task_name(), "state": load_state(home)}, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m {_MODULE}", description=_TASK_DESCRIPTION)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="probe once and restart a hung gateway if due")
    mode.add_argument("--status", action="store_true", help="print the watchdog task name and state")
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        print("gateway_windows_watchdog is Windows-only", file=sys.stderr)
        return 2
    from hermes_cli.config import get_hermes_home
    home = Path(get_hermes_home())
    if args.status:
        return _status(home)
    run_once(home)
    return 0


if __name__ == "__main__":
    sys.exit(main())
