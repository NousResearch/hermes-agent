"""``hermes update --force``: stop every gateway this update owns at once, without a drain.

The default pre-swap pause (``update_cmd_posix_pause._stop_gateways``) asks each gateway to finish
its in-flight work first and waits up to the restart budget (30 min by default). ``--force`` is the
explicit "stop everything now" the user asked for: it names what each gateway was running, stops it
immediately, kills anything still alive after a short bound, then lets the update and the normal
restart run.

A supervised gateway (systemd unit, launchd job) is stopped through its supervisor from the start:
that stop is what keeps ``Restart=``/``KeepAlive`` from bringing it back on the old tree, and it is
the path that carries the privilege (``sudo -n``) a system unit needs. A bare gateway takes the
force contract (``pause-for-update {"force": true}``) when it answers it, else the planned-stop
marker + SIGTERM. Either way the gateway enters ``stop()`` at once: running turns are interrupted,
tool and background processes killed (``process_registry.kill_all``) and background delegations
interrupted as resumable (``async_delegation.interrupt_all``). Gateways are stopped concurrently,
each against its own deadline measured from its own stop request.
"""

from __future__ import annotations

import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

#: How long a gateway gets to run its own immediate stop before it is killed.
FORCE_KILL_AFTER_S = 12.0
#: After the kill: how long to wait for the process (and its supervisor's stop job) to settle.
_SETTLE_S = 5.0


def _live_work(entry: dict) -> list | None:
    """What the gateway is running right now: the live ``active-work`` verb, else its state file."""
    from hermes_cli.update_cmd_drain_report import read_active_work
    if not entry.get("home"):
        return None
    home = Path(entry["home"])
    with suppress(Exception):
        from gateway.control_socket import query_gateway_control
        answer = query_gateway_control(home, "active-work", timeout=3.0)
        if isinstance(answer, dict) and isinstance(answer.get("active_work"), list):
            return list(answer["active_work"])
    return read_active_work(home)


def _ask_force_stop(entry: dict) -> bool:
    """True when the gateway took a force stop over its control socket (``ack["force"] is True``).
    An older gateway either rejects the parameter or answers with a drain ACK without the key;
    neither counts, so the signal path stops it."""
    if not entry.get("home"):
        return False
    from gateway.control_socket import query_gateway_control
    with suppress(Exception):
        ack = query_gateway_control(Path(entry["home"]), "pause-for-update", params={"force": True})
        return bool(isinstance(ack, dict) and ack.get("force") is True
                    and (ack.get("pausing") or ack.get("already_stopping")))
    return False


def _signal_stop(entry: dict) -> None:
    """Planned-stop marker + SIGTERM: the gateway's immediate ``stop()``, read as a planned stop."""
    from hermes_cli.update_cmd_windows import _write_update_planned_stop_marker
    if entry.get("home"):
        _write_update_planned_stop_marker(Path(entry["home"]), entry["pid"])
    with suppress(ProcessLookupError, PermissionError):
        os.kill(entry["pid"], signal.SIGTERM)


def _kill(entry: dict) -> None:
    """SIGKILL the gateway: through systemd for a unit (its privileged path reaches a system
    unit another account runs), else directly."""
    from hermes_cli.update_cmd_posix_pause import _manage_cmd
    if entry.get("kind") == "systemd" and (cmd := _manage_cmd(entry["scope"])) is not None:
        from hermes_cli.update_cmd_fleet import _systemctl
        _systemctl(cmd + ["kill", "--signal=SIGKILL", entry["unit"]], timeout=10)
        return
    with suppress(ProcessLookupError, PermissionError):
        os.kill(entry["pid"], getattr(signal, "SIGKILL", signal.SIGTERM))


def _describe(entry: dict, work: list | None) -> str:
    from hermes_cli.update_cmd_drain_report import describe_active_work_unit
    from hermes_cli.update_cmd_posix_pause import _names
    name = _names([entry])
    if not work:
        return f"  ⏹ {name}: stopped by --force (no active work reported)"
    home = Path(entry["home"]) if entry.get("home") else None
    units = "; ".join(describe_active_work_unit(u, home) for u in work if isinstance(u, dict))
    return f"  ⏹ {name}: stopped by --force, interrupting {len(work)} unit(s): {units}"


def _stop_one(token: dict, entry: dict, lock: threading.Lock) -> str | None:
    """Stop *entry* within its own deadline. ``None`` when it is gone (and, if supervised, its
    supervisor is stopped), else why it is not."""
    from hermes_cli import update_pause_record as pause_record
    from hermes_cli.update_cmd_posix_pause import _stop_at_once, _wait_gone
    work = _live_work(entry)
    with lock:  # the pause record is one file shared by every entry
        pause_record.mark_stop_sent(token, entry["pid"])  # before the request: a crash keeps it owed
    supervisor, failed = None, []
    if entry.get("kind"):
        def _supervisor_stop() -> None:
            try:
                _stop_at_once(entry)
            except Exception as exc:  # health: allow BLE001 -- reported as this entry's refusal below
                failed.append(str(exc))
        supervisor = threading.Thread(target=_supervisor_stop, name=f"force-stop-{entry['pid']}", daemon=True)
        supervisor.start()
    elif not _ask_force_stop(entry):
        _signal_stop(entry)
    print(_describe(entry, work), flush=True)
    born = {entry["pid"]: entry.get("ct")}
    if _wait_gone([entry["pid"]], FORCE_KILL_AFTER_S, born):
        print(f"  ⚠ gateway PID {entry['pid']} still running {FORCE_KILL_AFTER_S:.0f}s after the stop — killing it",
              flush=True)
        _kill(entry)
        if _wait_gone([entry["pid"]], _SETTLE_S, born):
            return f"gateway PID {entry['pid']} did not stop"
    if supervisor is not None:
        supervisor.join(_SETTLE_S)
        if supervisor.is_alive():
            return f"the supervisor of gateway PID {entry['pid']} did not finish stopping it"
    return failed[0] if failed else None


def force_stop_gateways(token: dict, entries: list[dict]) -> None:
    """Stop every entry now, concurrently; kill each survivor ``FORCE_KILL_AFTER_S`` after its own
    stop request. Raises ``PauseRefused`` naming every entry that is not stopped."""
    from hermes_cli.update_cmd_posix_pause import PauseRefused
    if not entries:
        return
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=len(entries), thread_name_prefix="update-force") as pool:
        refusals = [r for r in pool.map(lambda e: _stop_one(token, e, lock), entries) if r]
    if refusals:
        raise PauseRefused("; ".join(refusals))
