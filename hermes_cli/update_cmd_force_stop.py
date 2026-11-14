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

A gateway that had to be killed never ran that ``kill_all``: its background processes live in
their own systemd scopes or sessions, so they outlive it. :func:`_sweep_left_behind` stops what its
process checkpoints still name. A stop that does not settle (the supervisor's stop job still
running, or the gateway still alive after the kill) raises ``StopUnsettled``, which aborts
the update before the checkout moves: a rollback start would queue behind that stop job.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

#: How long a gateway gets to run its own immediate stop before it is killed.
FORCE_KILL_AFTER_S = 12.0
#: After the kill: how long to wait for the process (and its supervisor's stop job) to settle.
_SETTLE_S = 5.0
#: After the kill: how long the supervisor's stop job gets to settle once its unit is empty.
_STOP_JOB_SETTLE_S = 30.0


def _checkpoint_homes(entry: dict) -> list[Path]:
    """Homes whose ``processes.json`` this gateway's processes are recorded in: its own and, for the
    host multiplexer, every profile it serves (a served profile's turn checkpoints in that home).
    Read while the gateway is alive: the host record names only a live owner."""
    homes = [Path(entry["home"])] if entry.get("home") else []
    with suppress(Exception):
        from gateway.host_topology import host_gateway_topology
        from hermes_cli.profiles import get_profile_dir
        topology = host_gateway_topology()
        if topology is not None and topology.pid == entry["pid"]:
            homes += [get_profile_dir(name) for name in topology.profiles]
    unique: dict[str, Path] = {}
    for home in homes:
        unique.setdefault(os.path.realpath(home), home)
    return list(unique.values())


def _sweep_one(record: dict) -> bool:
    """Stop one checkpointed process (its tree, then its scope); True when it was ours and is gone.
    A PID whose start time does not match the recorded one was recycled and is never signalled."""
    from tools.process_registry import ProcessRegistry, _stop_systemd_unit
    pid, born = record.get("pid"), record.get("host_start_time")
    ours = isinstance(pid, int) and born is not None and ProcessRegistry._host_pid_is_ours(pid, born)
    if ours:
        ProcessRegistry._terminate_host_pid(pid, born)
    if record.get("systemd_unit"):
        _stop_systemd_unit(record["systemd_unit"])  # reaps double-forked descendants in the scope
    return ours and not ProcessRegistry._host_pid_is_ours(pid, born)


def _sweep_left_behind(homes: list[Path]) -> int:
    """What the killed gateway's ``stop()`` would have killed (``kill_all(source="gateway_shutdown")``:
    persisted jobs included): every host process its checkpoints name. How many were stopped."""
    records = []
    for home in homes:
        with suppress(OSError, ValueError):
            listed = json.loads((home / "processes.json").read_text(encoding="utf-8-sig"))
            records += [r for r in listed if isinstance(r, dict) and (r.get("pid_scope") or "host") == "host"]
    if not records:
        return 0
    with ThreadPoolExecutor(max_workers=min(8, len(records)), thread_name_prefix="update-force-sweep") as pool:
        return sum(pool.map(_sweep_one, records))


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


def _stop_one(token: dict, entry: dict, lock: threading.Lock) -> tuple[bool, str] | None:
    """Stop *entry* within its own deadline. ``None`` when it is gone (and, if supervised, its
    supervisor is stopped), else ``(unsettled, why)``: *unsettled* when its stop is still running."""
    from hermes_cli import update_pause_record as pause_record
    from hermes_cli.update_cmd_posix_pause import _names, _stop_at_once, _wait_gone
    work = _live_work(entry)
    homes = _checkpoint_homes(entry)
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
    # One deadline for the whole service: the gateway exiting is not enough while its supervisor's
    # stop job still waits on other processes in the unit (a tool child ignoring SIGTERM).
    deadline = time.monotonic() + FORCE_KILL_AFTER_S
    born = {entry["pid"]: entry.get("ct")}
    alive = bool(_wait_gone([entry["pid"]], FORCE_KILL_AFTER_S, born))
    if supervisor is not None:
        supervisor.join(max(0.0, deadline - time.monotonic()))
    if alive or (supervisor is not None and supervisor.is_alive()):
        print(f"  ⚠ gateway PID {entry['pid']} still stopping {FORCE_KILL_AFTER_S:.0f}s after the stop — killing it",
              flush=True)
        _kill(entry)  # a unit: SIGKILL to every process in its cgroup, so the stop job can settle
        if swept := _sweep_left_behind(homes):  # its own scopes/sessions: outside that cgroup
            print(f"  ⏹ {_names([entry])}: also stopped {swept} background process(es) it left behind", flush=True)
        if _wait_gone([entry["pid"]], _SETTLE_S, born):
            return True, (f"gateway PID {entry['pid']} is still running after SIGKILL "
                          f"(check `ps -o pid,stat,cmd -p {entry['pid']}`; a D-state process waits on I/O)")
    if supervisor is not None:
        # Keep ownership of the stop job until it settles: a rollback start would queue behind it.
        supervisor.join(_STOP_JOB_SETTLE_S)
        if supervisor.is_alive():
            return True, f"{_supervisor_name(entry)} is still stopping gateway PID {entry['pid']} ({_check_hint(entry)})"
    return (False, failed[0]) if failed else None


def _supervisor_name(entry: dict) -> str:
    return f"systemd's stop job for {entry['unit']}" if entry.get("kind") == "systemd" \
        else f"launchd's bootout of {entry.get('label')}"


def _check_hint(entry: dict) -> str:
    if entry.get("kind") == "systemd":
        user = " --user" if entry.get("scope") == "user" else ""
        return f"check `systemctl{user} status {entry['unit']}` and `systemctl{user} list-jobs`"
    return f"check `launchctl print {entry.get('domain')}/{entry.get('label')}`"


def force_stop_gateways(token: dict, entries: list[dict]) -> None:
    """Stop every entry now, concurrently; kill each survivor ``FORCE_KILL_AFTER_S`` after its own
    stop request. Raises ``StopUnsettled`` when any stop is still running (the update must not
    restart or swap past it), else ``PauseRefused`` naming every entry that is not stopped."""
    from hermes_cli.update_cmd_posix_pause import PauseRefused, StopUnsettled
    if not entries:
        return
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=len(entries), thread_name_prefix="update-force") as pool:
        refusals = [r for r in pool.map(lambda e: _stop_one(token, e, lock), entries) if r]
    if unsettled := [why for still_running, why in refusals if still_running]:
        raise StopUnsettled("; ".join(unsettled))
    if refusals:
        raise PauseRefused("; ".join(why for _, why in refusals))
