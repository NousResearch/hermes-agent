"""``hermes update --force``: stop every gateway this update owns at once, without a drain.

The default pre-swap pause (``update_cmd_posix_pause._stop_gateways``) asks each gateway to finish
its in-flight work first and waits up to the restart budget (30 min by default). ``--force`` is the
explicit "stop everything now" the user asked for: it names what each gateway was running, asks it
to stop immediately, SIGKILLs anything still alive after a short bound, then lets the update and the
normal restart run.

The immediate stop is the gateway's own: a gateway that implements the force contract answers
``pause-for-update {"force": true}``; any other gets the planned-stop marker + SIGTERM, which enters
``stop()`` at once (``agent.restart_drain_timeout``, 0 by default), interrupts the running turns,
kills its tool and background processes (``process_registry.kill_all``) and interrupts background
delegations (``async_delegation.interrupt_all``) as resumable.
"""

from __future__ import annotations

import os
import signal
from contextlib import suppress
from pathlib import Path

#: How long a gateway gets to run its own immediate stop before it is SIGKILLed.
FORCE_KILL_AFTER_S = 12.0


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
    """True when the gateway took a force stop over its control socket (Lane B contract:
    ``ack["force"] is True``). An older gateway either rejects the parameter or answers with a
    drain ACK without the key; neither counts, so the signal path below stops it."""
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


def _describe(entry: dict, work: list | None) -> str:
    from hermes_cli.update_cmd_drain_report import describe_active_work_unit
    from hermes_cli.update_cmd_posix_pause import _names
    name = _names([entry])
    if not work:
        return f"  ⏹ {name}: stopped by --force (no active work reported)"
    home = Path(entry["home"]) if entry.get("home") else None
    units = "; ".join(describe_active_work_unit(u, home) for u in work if isinstance(u, dict))
    return f"  ⏹ {name}: stopped by --force, interrupting {len(work)} unit(s): {units}"


def force_stop_gateways(token: dict, entries: list[dict]) -> None:
    """Stop every entry now; SIGKILL survivors after ``FORCE_KILL_AFTER_S``; then keep each
    supervisor from respawning it on the old tree (the pause's resume starts it on the new one)."""
    from hermes_cli import update_pause_record as pause_record
    from hermes_cli.update_cmd_posix_pause import PauseRefused, _alive, _disarm, _wait_gone
    for entry in entries:
        work = _live_work(entry)
        pause_record.mark_stop_sent(token, entry["pid"])  # before the request: a crash keeps it owed
        if not _ask_force_stop(entry):
            _signal_stop(entry)
        print(_describe(entry, work))
    born = {e["pid"]: e.get("ct") for e in entries}
    left = _wait_gone([e["pid"] for e in entries], FORCE_KILL_AFTER_S, born)
    for pid in sorted(left):
        print(f"  ⚠ gateway PID {pid} still running {FORCE_KILL_AFTER_S:.0f}s after the stop — SIGKILL")
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    if still := _wait_gone(left, 5.0, born):
        raise PauseRefused("gateway PID(s) " + ", ".join(map(str, sorted(still))) + " did not stop")
    for entry in entries:
        if not _alive(entry["pid"], entry.get("ct")):
            _disarm(entry)
