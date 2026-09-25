"""``gateway start/restart --all`` and ``gateway run`` mean "the ONE host multiplexer".

Base behaviour: ``--all`` SIGTERMed every gateway process on the host — including a multiplexer
serving other profiles — and started a single gateway in its place, so a per-profile command
caused a host-wide outage that left exactly one profile served.

Also covered here, because each was proven dead or wrong on the PR head:

* a supervised attach exits 75 (retry), never 78 (park) — the observation is transient;
* ``--replace`` is not eaten by the CLI guard, so ``start_gateway`` can signal the owner;
* ``restart --all`` retracts the stopped owner's record and re-enters with ``replace=True``,
  instead of attaching to the corpse it just stopped;
* ``-p X gateway restart --all`` reaches the ``--all``-aware branch.

The record is written as raw JSON so the file collects against a tree without the ``home`` field.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import host_attach
from gateway import host_rendezvous as hr
from hermes_cli import gateway as gw


def _reset_probe_memo() -> None:
    """Drop the host-gateway memo when the tree HAS one.

    ``getattr`` on purpose: a tree without the memo must still run these tests and fail on the
    OUTCOME, never error at fixture setup — an import tripwire proves nothing about behaviour.
    """
    getattr(host_attach, "invalidate_host_gateway_cache", lambda: None)()


@pytest.fixture(autouse=True)
def _no_memo():
    """The probe is memoized for a couple of seconds; each arrangement starts from cold."""
    _reset_probe_memo()
    yield
    _reset_probe_memo()


def _write_record(pid: int, home: Path, profiles: tuple[str, ...]) -> None:
    payload = {
        "role": hr.ROLE_GATEWAY, "home": str(home), "pid": pid,
        "createTime": hr.process_create_time(pid), "host": "", "port": None,
        "protocolVersion": hr.HOST_PROTOCOL_VERSION, "tokenFingerprint": "",
        "profiles": list(profiles), "updatedAt": "",
    }
    path = hr.record_path(hr.ROLE_GATEWAY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def host_owner(tmp_path, monkeypatch):
    """A REAL live process published as the host gateway, answering identify for three profiles."""
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    home = tmp_path / "root"
    _write_record(child.pid, home, ("default", "ops", "coder"))
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway",
        lambda dialled, **kw: {"pid": child.pid, "hermes_home": str(home),
                               "served_profiles": ["default", "ops", "coder"]})
    try:
        yield SimpleNamespace(pid=child.pid, home=home, proc=child)
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_start_all_never_sweeps_a_live_host_multiplexer(host_owner, monkeypatch, capsys):
    def _never(*a, **k):
        raise AssertionError("start --all SIGTERMed the host multiplexer")

    monkeypatch.setattr(gw, "kill_gateway_processes", _never)
    monkeypatch.setattr(gw, "_guard_named_profile_under_multiplexer", lambda **k: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda verb: False)
    monkeypatch.setattr(gw, "_service_backend", lambda: None)
    monkeypatch.setattr(gw, "find_gateway_pids", lambda **k: [])

    gw._cmd_start(SimpleNamespace(system=False, all=True, force=False))

    out = capsys.readouterr().out
    assert "already running" in out
    # The served set survives the verb: all three profiles are still reported as served.
    for profile in ("default", "ops", "coder"):
        assert profile in out


def test_restart_all_refuses_to_sweep_a_host_multiplexer_owned_by_another_profile(
        host_owner, monkeypatch, tmp_path):
    def _never(*a, **k):
        raise AssertionError("restart --all SIGTERMed another profile's host multiplexer")

    monkeypatch.setattr(gw, "kill_gateway_processes", _never)
    monkeypatch.setattr(gw, "_stop_installed_service", lambda system: False)
    monkeypatch.setattr("gateway.status._get_process_hermes_home",
                        lambda: Path(tmp_path / "root" / "profiles" / "ops"))

    with pytest.raises(SystemExit) as exc:
        gw._restart_all(system=False)
    assert exc.value.code == gw.GATEWAY_FATAL_CONFIG_EXIT_CODE


def test_restart_all_does_not_attach_to_the_process_it_just_stopped(
        host_owner, monkeypatch, tmp_path):
    """After a confirmed stop the record is retracted and the re-entry carries ``replace``.

    Without both, the re-entered ``gateway run`` read the corpse's record, decided ATTACH and
    exited 0 — a restart that silently left the host with no gateway at all.
    """
    monkeypatch.setattr("gateway.status._get_process_hermes_home", lambda: host_owner.home)
    monkeypatch.setattr(gw, "_stop_installed_service", lambda system: False)
    monkeypatch.setattr(gw, "kill_gateway_processes", lambda **k: 1)
    monkeypatch.setattr(gw, "_wait_for_api_server_port_free", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_print_unfolded_gateway_note", lambda owner: None)

    def _stopped(*a, **k):
        # The owner really is gone by the time the re-entry happens.
        host_owner.proc.kill()
        host_owner.proc.wait(timeout=10)
        _reset_probe_memo()

    monkeypatch.setattr(gw, "_wait_for_gateway_exit", _stopped)
    reentry: list[dict] = []
    monkeypatch.setattr(gw, "run_gateway", lambda **kw: reentry.append(kw))

    gw._restart_all(system=False)

    assert hr.read_record(hr.ROLE_GATEWAY, include_stale=True) is None, \
        "the stopped owner's record must be retracted"
    assert reentry and reentry[0].get("replace") is True


def test_named_profile_restart_all_reaches_the_all_aware_branch(monkeypatch):
    """The generic guard ran first and printed a bare `gateway restart` (no --all)."""
    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_dispatch_all_via_service_manager_if_s6", lambda verb: False)
    monkeypatch.setattr(gw, "_guard_named_profile_under_multiplexer",
                        lambda **k: pytest.fail("the generic guard pre-empted the --all branch"))
    called: list[bool] = []
    monkeypatch.setattr(gw, "_restart_all", lambda system: called.append(True))

    gw._cmd_restart(SimpleNamespace(system=False, all=True, force=False))

    assert called == [True]


def test_a_supervised_attach_is_retried_not_parked(monkeypatch, capsys):
    """78 parks the unit for good; "someone serves me right now" is a transient observation."""
    owner = host_attach.HostGateway(4321, Path("/somewhere"), ("default", "other"))
    monkeypatch.setattr(
        "gateway.host_attach.decide",
        lambda home, replace=False: host_attach.HostAttachDecision(
            host_attach.ATTACH, "attached", owner, transient=True))
    monkeypatch.setattr(gw, "_running_under_gateway_supervisor", lambda: True)

    with pytest.raises(SystemExit) as exc:
        gw._attach_to_host_gateway_or_guard(force=False)

    assert exc.value.code == gw.GATEWAY_SERVICE_RESTART_EXIT_CODE
    assert exc.value.code != gw.GATEWAY_FATAL_CONFIG_EXIT_CODE


def test_a_refusal_lands_in_the_profile_logs_not_only_on_stdout(monkeypatch, caplog):
    """The refusal's stdout goes to the supervisor's unit log; launchd then maps 78 to a clean exit and
    parks the unit. Nothing in the profile's own logs said why (field report on #118097)."""
    owner = host_attach.HostGateway(4321, Path("/somewhere"), ("default",))
    monkeypatch.setattr(
        "gateway.host_attach.decide",
        lambda home, replace=False: host_attach.HostAttachDecision(
            host_attach.REFUSE, host_attach._refuse_message(owner, "nous"), owner))

    with caplog.at_level("WARNING", logger="hermes_cli.gateway"), pytest.raises(SystemExit) as exc:
        gw._attach_to_host_gateway_or_guard(force=False)

    assert exc.value.code == gw.GATEWAY_FATAL_CONFIG_EXIT_CODE
    warned = [r for r in caplog.records if r.levelno >= 30 and "migrate --multiplex" in r.getMessage()]
    assert warned, "a refusal must leave the remedy in the profile's own log"


def test_replace_is_not_eaten_by_the_cli_guard(monkeypatch):
    """The guard exited before ``start_gateway`` ever saw ``--replace``, so nothing was replaced."""
    owner = host_attach.HostGateway(4321, Path("/somewhere"), ("default", "other"))
    seen: list[bool] = []

    def _decide(home, replace=False):
        seen.append(replace)
        return host_attach.HostAttachDecision(
            host_attach.REPLACE_HOST if replace else host_attach.ATTACH, "", owner)

    monkeypatch.setattr("gateway.host_attach.decide", _decide)
    monkeypatch.setattr(gw, "_guard_named_profile_under_multiplexer",
                        lambda **k: pytest.fail("--replace must not fall through to the guard"))

    gw._attach_to_host_gateway_or_guard(force=False, replace=True)  # must not raise SystemExit

    assert seen == [True], "the CLI guard must pass --replace into the host decision"
