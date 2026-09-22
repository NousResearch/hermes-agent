"""The gateway lifecycle verbs mean "the ONE host process", not "this profile's gateway".

Invariants, all against a REAL live process standing in for the owner:

* ATTACH requires a LIVE ``identify`` answer. A record on its own proves only that an owner
  exists — believing its ``profiles`` list parked a second profile's supervised unit against a
  served set that process had not committed to (and, with multiplex off, never would).
* ``--replace`` signals the owner instead of standing down.
* ``--force`` starts without asking the owner anything.
* The claim-time record carries NO served set; it is filled in once the channel answers.

The record is written as raw JSON, not via ``HostRecord(...)``, so these tests collect and RUN
against a tree without the ``home`` field and fail on the OUTCOME rather than on a TypeError.
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gateway import host_attach, host_rendezvous as hr
from gateway import run as gateway_run


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


@pytest.fixture
def owner_pid(tmp_path, monkeypatch):
    """A REAL live process standing in for the host gateway (liveness is proved, not stubbed)."""
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield child.pid
    finally:
        child.terminate()
        child.wait(timeout=10)


def _publish(pid: int, home: Path, profiles: tuple[str, ...]) -> None:
    payload = {
        "role": hr.ROLE_GATEWAY,
        "home": str(home),
        "pid": pid,
        "createTime": hr.process_create_time(pid),
        "host": "",
        "port": None,
        "protocolVersion": hr.HOST_PROTOCOL_VERSION,
        "tokenFingerprint": "",
        "profiles": list(profiles),
        "updatedAt": "",
    }
    path = hr.record_path(hr.ROLE_GATEWAY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _answer_identify(monkeypatch, pid: int, home: Path, served: list[str]) -> None:
    """Make the owner's control socket answer ``identify`` — the ONLY source of a served set."""
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway",
        lambda dialled, **kw: (
            {"pid": pid, "hermes_home": str(home), "served_profiles": served}
            if Path(dialled) == home else None))


def test_a_record_only_owner_never_produces_attach(tmp_path, monkeypatch, owner_pid):
    """Owner present, control socket silent: the served set is UNKNOWN, so never ATTACH.

    The claim-time record lands before the socket binds, so during a boot race a second profile's
    supervised unit read "I am already served" from a process that had committed to nothing — and
    exited 78, which parks the unit for good.
    """
    ours = tmp_path / "root" / "profiles" / "other"
    _publish(owner_pid, tmp_path / "root", ("default", "other"))
    monkeypatch.setattr(host_attach, "ATTACH_CHANNEL_WAIT_S", 0.0)
    monkeypatch.setattr("gateway.control_socket.identify_gateway", lambda *a, **k: None)
    monkeypatch.setattr("gateway.control_socket.rescan_gateway_profiles", lambda *a, **k: None)

    decision = host_attach.decide(ours)

    assert decision.outcome == host_attach.REFUSE
    # ...and it is a RUNTIME observation, so a supervisor must retry rather than park the unit.
    assert decision.transient is True
    assert decision.owner is not None and decision.owner.served_known is False


def test_attach_needs_a_live_identify_answer(tmp_path, monkeypatch, owner_pid):
    owner_home = tmp_path / "root"
    ours = owner_home / "profiles" / "other"
    _publish(owner_pid, owner_home, ())  # the record claims nothing; the socket answers
    _answer_identify(monkeypatch, owner_pid, owner_home, ["default", "other"])
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: ours)

    def _never(*a, **k):
        raise AssertionError("a second gateway was started for an already-served profile")

    monkeypatch.setattr(gateway_run, "_start_gateway_replace_existing_instance", _never)

    assert asyncio.run(gateway_run._host_attach_or_none(replace=False)) is True


def test_run_for_an_unserved_profile_rescans_then_attaches(tmp_path, monkeypatch, owner_pid):
    owner_home = tmp_path / "root"
    ours = owner_home / "profiles" / "other"
    _publish(owner_pid, owner_home, ("default",))
    _answer_identify(monkeypatch, owner_pid, owner_home, ["default"])
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: ours)
    asked: list[Path] = []

    def _rescan(home, *, timeout=8.0):
        asked.append(Path(home))
        return {"multiplex": True, "served_profiles": ["default", "other"]}

    monkeypatch.setattr("gateway.control_socket.rescan_gateway_profiles", _rescan)

    assert asyncio.run(gateway_run._host_attach_or_none(replace=False)) is True
    assert asked == [owner_home], "the rescan must reach the OWNER's home, not ours"


def test_host_gateway_refuses_when_it_will_not_serve_the_profile(tmp_path, monkeypatch, owner_pid):
    """A MULTIPLEXING owner whose roster still excludes us after a rescan is the permanent refusal."""
    owner_home = tmp_path / "root"
    _publish(owner_pid, owner_home, ("default",))
    _answer_identify(monkeypatch, owner_pid, owner_home, ["default"])
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: owner_home / "profiles" / "other")
    monkeypatch.setattr("gateway.control_socket.rescan_gateway_profiles",
                        lambda home, timeout=8.0: {"multiplex": True, "served_profiles": ["default"]})

    assert asyncio.run(gateway_run._host_attach_or_none(replace=False)) is False


def test_a_standalone_owner_is_the_per_profile_topology_not_a_refusal(tmp_path, monkeypatch, owner_pid, caplog):
    """The owner answers ``multiplex: False``: it is a per-profile gateway, not a multiplexer that
    excluded us. Refusing here (exit 78 → launchd parks the unit) took every other profile's
    supervised gateway down at boot on a one-process-per-profile fleet. Start as before."""
    owner_home = tmp_path / "root" / "profiles" / "tank"
    _publish(owner_pid, owner_home, ("tank",))
    _answer_identify(monkeypatch, owner_pid, owner_home, ["tank"])
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: tmp_path / "root" / "profiles" / "nous")
    monkeypatch.setattr("gateway.control_socket.rescan_gateway_profiles",
                        lambda home, timeout=8.0: {"multiplex": False, "served_profiles": ["tank"]})

    with caplog.at_level("WARNING", logger="gateway.host_attach"):
        assert host_attach.decide(tmp_path / "root" / "profiles" / "nous").outcome == host_attach.START
    assert any("migrate --multiplex" in r.getMessage() for r in caplog.records), "the converge hint is logged"
    assert asyncio.run(gateway_run._host_attach_or_none(replace=False)) is None


def test_replace_does_not_target_another_profiles_standalone_gateway(tmp_path, monkeypatch, owner_pid):
    """A persisted service ``--replace`` must not turn a valid per-profile fleet into a restart loop.

    The other profile cannot satisfy our start, but ownership protection correctly prevents us from
    signalling it. Degrade to the same side-by-side START used without ``--replace``.
    """
    owner_home = tmp_path / "root" / "profiles" / "youtube-studio"
    our_home = tmp_path / "root"
    _publish(owner_pid, owner_home, ("youtube-studio",))
    _answer_identify(monkeypatch, owner_pid, owner_home, ["youtube-studio"])
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: our_home)
    monkeypatch.setattr(
        "gateway.control_socket.rescan_gateway_profiles",
        lambda home, timeout=8.0: {
            "multiplex": False,
            "served_profiles": ["youtube-studio"],
        },
    )
    monkeypatch.setattr(
        gateway_run,
        "_start_gateway_replace_existing_instance",
        lambda *a, **k: pytest.fail("another profile's gateway must not be signalled"),
    )

    assert host_attach.decide(our_home, replace=True).outcome == host_attach.START
    assert asyncio.run(gateway_run._host_attach_or_none(replace=True)) is None


def test_replace_signals_the_owner_instead_of_standing_down(tmp_path, monkeypatch, owner_pid):
    """``gateway run --replace`` against a live owner must REACH it — the branch was dead code."""
    owner_home = tmp_path / "root"
    _publish(owner_pid, owner_home, ("default", "other"))
    _answer_identify(monkeypatch, owner_pid, owner_home, ["default", "other"])
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: owner_home / "profiles" / "other")
    signalled: list[int] = []

    async def _replace(pid, replace):
        signalled.append(pid)
        return True

    monkeypatch.setattr(gateway_run, "_start_gateway_replace_existing_instance", _replace)

    assert asyncio.run(gateway_run._host_attach_or_none(replace=True)) is None
    assert signalled == [owner_pid]


def test_force_starts_without_consulting_the_owner(tmp_path, monkeypatch, owner_pid):
    """``--force`` printed the starting banner and then attached anyway. It must START."""
    owner_home = tmp_path / "root"
    _publish(owner_pid, owner_home, ("default", "other"))
    _answer_identify(monkeypatch, owner_pid, owner_home, ["default", "other"])
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: owner_home / "profiles" / "other")

    assert asyncio.run(gateway_run._host_attach_or_none(replace=False, force=True)) is None


def test_the_claim_time_record_publishes_no_served_set(tmp_path, monkeypatch):
    """Claim time is too early to know the served set; publishing a guess strands other profiles."""
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: tmp_path / "root")
    monkeypatch.setattr(hr, "served_profiles",
                        lambda **kw: pytest.fail("the claim must not guess a served set"))
    try:
        gateway_run._claim_host_gateway_role()
        record = hr.read_record(hr.ROLE_GATEWAY)
        assert record is not None and record.pid == os.getpid()
        assert record.profiles == ()
    finally:
        hr.release_host_lock(hr.ROLE_GATEWAY)
        hr.clear_record(hr.ROLE_GATEWAY)


def test_served_profiles_ignores_the_retired_opt_out_but_honours_an_explicit_argument(monkeypatch):
    """``served_profiles()`` forced ``multiplex=True`` and claimed profiles it would never serve,
    so it learned to read ``gateway.multiplex_profiles``. That key is now RETIRED as a topology
    opt-out: reading it here was the last place an explicit ``false`` still narrowed the record,
    which is why the CLI reported "standalone, serving default" while the runtime multiplexed.
    The caller's explicit argument — the RUNTIME verdict — still decides."""
    asked: list[bool] = []

    def _roster(*, multiplex):
        asked.append(multiplex)
        return ([("default", Path("/x")), ("other", Path("/y"))] if multiplex
                else [("default", Path("/x"))])

    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", _roster)
    monkeypatch.setattr(
        "hermes_cli.gateway_multiplex_mode.explicit_multiplex_flag", lambda home: False)

    assert hr.served_profiles() == ("default", "other")
    assert hr.served_profiles(multiplex=False) == ("default",)
    assert asked == [True, False]


@pytest.mark.skipif(sys.platform == "win32",
                    reason="POSIX ownership check; Windows has no st_uid to compare")
def test_a_foreign_record_is_not_a_record(tmp_path, monkeypatch, owner_pid):
    """A record this OS user did not write must never decide our lifecycle (forgery/DoS)."""
    _publish(owner_pid, tmp_path / "root", ("default", "other"))
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)

    assert hr.read_record(hr.ROLE_GATEWAY) is None
    assert host_attach.host_gateway() is None


def test_the_default_profile_arriving_second_starts_beside_a_standalone_named_owner(tmp_path, monkeypatch, owner_pid):
    """The field shape of #118282: after a fleet restart a NAMED standalone unit claimed the host first and
    the DEFAULT gateway arrived second. Refusing it exited 78 and its system unit crash-looped; the default
    profile is a peer in a per-profile fleet, not a latecomer to a multiplexer."""
    root = tmp_path / "root"
    owner_home = root / "profiles" / "agent-ops"
    _publish(owner_pid, owner_home, ("agent-ops",))
    _answer_identify(monkeypatch, owner_pid, owner_home, ["agent-ops"])
    monkeypatch.setattr(gateway_run, "get_hermes_home", lambda: root)
    monkeypatch.setattr("gateway.control_socket.rescan_gateway_profiles",
                        lambda home, timeout=8.0: {"multiplex": False, "served_profiles": ["agent-ops"]})

    decision = host_attach.decide(root)
    assert host_attach.profile_name_for_home(root) == "default"
    assert decision.outcome == host_attach.START
    assert asyncio.run(gateway_run._host_attach_or_none(replace=False)) is None
