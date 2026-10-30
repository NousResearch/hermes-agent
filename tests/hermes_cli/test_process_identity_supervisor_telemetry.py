"""Telemetry from the owner-death supervisor: reap records, restart and spawn-failure events, and the rule
that recording never costs the supervisor. The supervisor itself is tested in test_process_identity_supervisor.py,
whose harness this file shares."""

import os
import subprocess
import sys
import threading

import pytest

from hermes_cli import process_identity
from tests.hermes_cli.test_process_identity_supervisor import (
    _VICTIM,
    _kill,
    _reset_client_state as _reset_client_state,  # noqa: PLC0414 -- autouse fixture: the import registers it here
    _supervisor_argv,
    _wait_exited,
    all_groups_alive as all_groups_alive,  # noqa: PLC0414 -- the self-alias suppresses F811 for the fixture parameter
)
from tests.waiting import wait_until


@pytest.mark.parametrize("ignores_term", [False, True])
def test_a_reap_leaves_one_content_free_record_while_a_gateway_spools_for_this_home(tmp_path, ignores_term):
    """The reap happens after its owner died, where no exporter runs: it leaves a record in the spool the
    exporting gateway drains (how many groups, and whether SIGKILL was needed). Nothing without a spool."""
    from agent.monitoring import managed_processes

    managed_processes.set_spooling([tmp_path], True)
    code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('armed', flush=True); time.sleep(300)" \
        if ignores_term else "import time; print('armed', flush=True); time.sleep(300)"
    victims = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, start_new_session=True)
               for _ in range(2)]
    supervisor = subprocess.Popen([*_supervisor_argv(os.getpgid(0)), str(managed_processes.spool_dir(tmp_path))],
                                  stdin=subprocess.PIPE, text=True, start_new_session=True)
    try:
        for victim in victims:
            assert victim.stdout.readline().strip() == "armed"
            supervisor.stdin.write(f"register {victim.pid}\n")
        # Reap them the moment they die, as init does for a dead owner's children: an unreaped zombie keeps
        # its group answering `kill -0` and would turn every reap into an escalation to SIGKILL.
        for victim in victims:
            threading.Thread(target=victim.wait, daemon=True).start()
        supervisor.stdin.close()
        assert _wait_exited(supervisor, timeout=20)
        assert all(_wait_exited(victim) for victim in victims)
        (event,) = managed_processes.drain_spool([tmp_path])
        assert (event.subsystem, event.action, event.outcome, event.actor, event.count) == (
            "owned_children", "orphan_reap", "killed" if ignores_term else "terminated", "supervisor", 2)
        assert event.ts_ns > 1_600_000_000 * 10**9
    finally:
        for victim in victims:
            _kill(victim.pid)
            victim.wait(timeout=10)
        _kill(supervisor.pid)
        supervisor.wait(timeout=10)


def test_a_registered_child_that_already_exited_is_not_reported_as_a_reap(tmp_path):
    """Registrations are pruned only at the next change, so a child that exited on its own can still be
    registered at the owner's death: TERMing an empty group is not a reap, and must not raise an alert."""
    from agent.monitoring import managed_processes

    managed_processes.set_spooling([tmp_path], True)
    gone = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    gone.wait(timeout=10)
    supervisor = subprocess.Popen([*_supervisor_argv(os.getpgid(0)), str(managed_processes.spool_dir(tmp_path))],
                                  stdin=subprocess.PIPE, text=True, start_new_session=True)
    supervisor.stdin.write(f"register {gone.pid}\n")
    supervisor.stdin.close()
    assert _wait_exited(supervisor)
    assert managed_processes.drain_spool([tmp_path]) == []


@pytest.mark.parametrize("filler", ["{i}-1.json", ".record-{i}.tmp"], ids=["records", "abandoned_temp_files"])
def test_a_reap_into_a_full_spool_writes_nothing(tmp_path, filler):
    """Nothing drains a spool whose gateway died: the shell writer stops at the same cap as the Python one."""
    from agent.monitoring import managed_processes

    managed_processes.set_spooling([tmp_path], True)
    spool = managed_processes.spool_dir(tmp_path)
    for i in range(managed_processes._MAX_SPOOLED):
        (spool / filler.format(i=i)).write_text("{}", encoding="utf-8")
    victim = subprocess.Popen(_VICTIM, start_new_session=True)
    try:
        supervisor = subprocess.Popen([*_supervisor_argv(os.getpgid(0)), str(spool)], stdin=subprocess.PIPE,
                                      text=True, start_new_session=True)
        threading.Thread(target=victim.wait, daemon=True).start()
        supervisor.stdin.write(f"register {victim.pid}\n")
        supervisor.stdin.close()
        assert _wait_exited(supervisor) and _wait_exited(victim), "a full spool must not cost the reap"
        assert len(list(spool.iterdir())) == managed_processes._MAX_SPOOLED
    finally:
        _kill(victim.pid)
        victim.wait(timeout=10)


def test_a_reap_records_nothing_without_a_spool_or_with_nothing_registered(tmp_path):
    victim = subprocess.Popen(_VICTIM, start_new_session=True)
    absent = tmp_path / "monitoring" / "spool"
    try:
        supervisor = subprocess.Popen([*_supervisor_argv(os.getpgid(0)), str(absent)], stdin=subprocess.PIPE,
                                      text=True, start_new_session=True)
        supervisor.stdin.write(f"register {victim.pid}\n")
        supervisor.stdin.close()
        assert _wait_exited(supervisor) and _wait_exited(victim)
        assert not absent.exists(), "the supervisor must never create the spool: export is off for this home"
        absent.mkdir(parents=True)
        empty = subprocess.Popen([*_supervisor_argv(os.getpgid(0)), str(absent)], stdin=subprocess.PIPE, text=True,
                                 start_new_session=True)
        empty.stdin.close()  # a release with nothing registered is not a reap
        assert _wait_exited(empty)
        assert list(absent.iterdir()) == []
    finally:
        _kill(victim.pid)
        victim.wait(timeout=10)


def test_a_supervisor_that_cannot_spawn_is_reported(monkeypatch, all_groups_alive):
    """Losing the owner-death safety net is silent for the user; operators see it as an event."""
    from agent.monitoring import managed_processes
    seen = []
    monkeypatch.setattr(managed_processes, "record", lambda *a, **k: seen.append((a, k)))

    def no_sh(*_a, **_k):
        raise OSError(2, "No such file or directory")

    monkeypatch.setattr(process_identity.subprocess, "Popen", no_sh)
    monkeypatch.setattr(process_identity, "_reported_unavailable", False)
    process_identity.own_process_groups([111])
    process_identity.own_process_groups([222])  # a broken sh fails every registration: reported once
    assert seen == [(("owned_children", "supervisor_unavailable", "spawn_failed"), {"actor": "owner", "count": None})]


def test_the_real_supervisor_is_told_this_homes_spool_and_records_its_reap(monkeypatch):
    """Wiring: the supervisor spawned for an owned child gets this home's spool, so a reap after the owner
    died is reported once a gateway exports for this home."""
    from agent.monitoring import managed_processes

    home = managed_processes.spool_dir().parents[1]
    managed_processes.set_spooling([home], True)
    child = subprocess.Popen(_VICTIM, start_new_session=True)
    try:
        process_identity.own_process_groups([child.pid])
        proc = process_identity._supervisor
        assert proc.args[-1] == str(managed_processes.spool_dir())
        threading.Thread(target=child.wait, daemon=True).start()  # reaped at death, as init would (no zombie)
        proc.stdin.close()  # the EOF the owner's death sends
        assert _wait_exited(child)
        wait_until(lambda: any(managed_processes.spool_dir().glob("*.json")), timeout=10, what="the reap record")
        (event,) = managed_processes.drain_spool([home])
        assert (event.action, event.outcome, event.count) == ("orphan_reap", "terminated", 1)
    finally:
        _kill(child.pid)
        child.wait(timeout=10)


def test_a_hung_telemetry_write_never_holds_up_a_supervisor_restart(monkeypatch):
    """A record can write a file (a stalled disk, a hung network home): it runs after the supervisor lock is
    released, so the replacement starts and other children still register meanwhile."""
    from agent.monitoring import managed_processes

    release = threading.Event()
    entered = threading.Event()

    def hung_record(*_a, **_k):
        entered.set()
        release.wait(20)

    monkeypatch.setattr(managed_processes, "record", hung_record)
    child = subprocess.Popen(_VICTIM, start_new_session=True)
    other = subprocess.Popen(_VICTIM, start_new_session=True)
    try:
        process_identity.own_process_groups([child.pid])
        first = process_identity._supervisor
        first.kill()
        assert entered.wait(10), "the supervisor exit was not reported"
        assert process_identity._supervisor not in (None, first), "the replacement waited for the telemetry write"
        registered = threading.Thread(target=process_identity.own_process_groups, args=([other.pid],), daemon=True)
        registered.start()
        registered.join(5)
        assert not registered.is_alive(), "another child's registration waited for the telemetry write"
    finally:
        release.set()
        for proc in (child, other):
            _kill(proc.pid)
            proc.wait(timeout=10)


def test_supervisor_telemetry_can_never_cost_the_supervisor(monkeypatch):
    """The spool path is resolved inside the supervisor spawn: a broken monitoring import must cost the
    record, not the safety net."""
    import builtins
    real_import = builtins.__import__

    def broken(name, *a, **k):
        if name.startswith("agent.monitoring"):
            raise ImportError("monitoring unavailable")
        return real_import(name, *a, **k)

    child = subprocess.Popen(_VICTIM, start_new_session=True)
    try:
        with monkeypatch.context() as patched:
            patched.setattr(builtins, "__import__", broken)
            process_identity.own_process_groups([child.pid])
        proc = process_identity._supervisor
        assert proc is not None and proc.poll() is None, "the supervisor did not start without telemetry"
        proc.stdin.close()  # and it still reaps
        assert _wait_exited(child), "the supervisor started without telemetry did not reap"
    finally:
        _kill(child.pid)
        child.wait(timeout=10)


def test_a_restart_that_cannot_spawn_is_reported_as_not_restarted(monkeypatch):
    """Operators must not see a recovery while the children are uncovered."""
    from agent.monitoring import managed_processes

    seen = []
    monkeypatch.setattr(managed_processes, "record", lambda subsystem, action, outcome, **k: seen.append((action, outcome)))
    monkeypatch.setattr(process_identity, "_reported_unavailable", False)
    child = subprocess.Popen(_VICTIM, start_new_session=True)
    try:
        process_identity.own_process_groups([child.pid])
        first = process_identity._supervisor
        monkeypatch.setattr(process_identity, "_spawn_supervisor", lambda: None)
        first.kill()
        wait_until(lambda: ("supervisor_exit", "not_restarted") in seen, timeout=10, what="the exit reported")
        assert ("supervisor_exit", "restarted") not in seen
    finally:
        _kill(child.pid)
        child.wait(timeout=10)


def test_a_supervisor_whose_pipe_breaks_at_once_is_reported_unavailable(monkeypatch, all_groups_alive):
    from agent.monitoring import managed_processes

    class _BrokenPipe:
        def write(self, _data):
            raise BrokenPipeError(32, "Broken pipe")

        flush = close = lambda self: None

    seen = []
    monkeypatch.setattr(managed_processes, "record", lambda subsystem, action, outcome, **k: seen.append((action, outcome)))
    monkeypatch.setattr(process_identity, "_reported_unavailable", False)
    monkeypatch.setattr(process_identity, "_spawn_supervisor", lambda: type("P", (), {"stdin": _BrokenPipe(), "pid": 1})())
    process_identity.own_process_groups([111])
    assert ("supervisor_unavailable", "registration_failed") in seen
