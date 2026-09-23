"""Every host restart actor is leased, arbitrated, or exempt — with the reason written down.

Task ``t_559d31fb`` (gap **G1** of ``build/gateway_restart/POST-UPDATE-GATEWAY-RESTART-DESIGN.md``).
The parent implementation (``t_5d0ba21a``) leased the update run and the manual-gateway drain; two
actors still restarted the ONE host gateway with no lease, no arbitration and no record. That is
exactly how 2026-09-23 lost its drain: SIGTERM 19:28:46 then SIGUSR1 19:28:53 from a second actor
(``~/.hermes/logs/gateway.log:4545-4555``) and the earliest signal won the semantics, amputating an
in-flight cron job.

Encoded against that failure:

* the CLI actor ``hermes gateway restart`` takes the lease, and its launchctl-kickstart path runs
  inside it;
* it waits ≤``LEASE_WAIT_S`` on a live holder and then exits rc=0 ``already-in-progress`` WITHOUT
  signalling — the second actor of 19:28:53;
* it publishes its drain deadline on the lease BEFORE signalling, so a second actor's SIGTERM is
  refused and the requesting actor is named in one structured line — the 19:28:46 SIGTERM;
* the dashboard/Desktop relaunch inherits that lease through the command it spawns, and the raw
  operator ``launchctl kickstart -k`` plus the update's abort-recovery child are explicitly exempt;
* a dead holder is taken over, not waited on; an unwritable lease path fails open.

Nothing here touches the live host gateway: the lease lives under a per-test
``HERMES_GATEWAY_LOCK_DIR``, and every signal / service call / ``launchctl`` invocation is a fake.
"""

from __future__ import annotations

import importlib
import json
import os
import signal as signal_mod
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_restart_orchestrator as orch

PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.hermes.gateway</string>
    <key>ThrottleInterval</key>
    <integer>30</integer>
    <key>ExitTimeOut</key>
    <integer>60</integer>
</dict>
</plist>
"""


@pytest.fixture(autouse=True)
def host_state_dir(tmp_path, monkeypatch):
    """Isolate the host rendezvous dir, and never inherit a recovery-child marker."""
    lock_dir = tmp_path / "gateway-locks"
    lock_dir.mkdir()
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(lock_dir))
    monkeypatch.delenv(orch.RECOVERY_ENV, raising=False)
    return lock_dir


@pytest.fixture
def plist(tmp_path, monkeypatch):
    """This host's real launchd contract, wired into every budget probe."""
    path = tmp_path / "ai.hermes.gateway.plist"
    path.write_text(PLIST, encoding="utf-8")
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(gateway_mod, "get_launchd_plist_path", lambda: path, raising=False)
    monkeypatch.setattr(gateway_mod, "launchd_gateway_labels_for_install", lambda: [], raising=False)
    return path


def lease_file() -> dict:
    path = orch.lease_path()
    assert path is not None
    return json.loads(path.read_text(encoding="utf-8"))


def lease_file_or_none() -> dict | None:
    path = orch.lease_path()
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class Harness:
    """Everything a driven ``hermes gateway restart`` did, recorded rather than hoped for."""

    def __init__(self) -> None:
        self.service_calls: list[tuple[str, str]] = []
        self.signals: list[tuple[int, int]] = []
        self.kickstarts: list[list[str]] = []
        #: ``(lease_held, open_drain_pid)`` sampled at the moment the restart was dispatched.
        self.lease_at_dispatch: list[tuple[bool, int | None]] = []

    def sigusr1(self) -> int:
        return sum(1 for _pid, sig in self.signals if sig == signal_mod.SIGUSR1)

    def sigterms(self) -> int:
        return sum(1 for _pid, sig in self.signals if sig == signal_mod.SIGTERM)

    def dispatch_sample(self) -> tuple[bool, int | None]:
        drain = lease_file_or_none() or {}
        entry = drain.get("drain") or {}
        return orch.restart_in_progress(), entry.get("pid")


@pytest.fixture
def harness(monkeypatch, plist):
    """Drive the CLI restart actor with every destructive edge faked (never the live gateway)."""
    harness = Harness()
    import hermes_cli.gateway as gw
    from gateway import status as gateway_status

    monkeypatch.setattr(orch, "_send_signal", lambda pid, sig: harness.signals.append((pid, sig)),
                        raising=False)
    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_guard_named_profile_under_multiplexer", lambda **k: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_dispatch_all_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda windows: "launchd")
    monkeypatch.setattr(gw, "_get_restart_exit_wait_budget", lambda: 1815.0)
    monkeypatch.setattr(gw, "get_launchd_label", lambda: "ai.hermes.gateway", raising=False)
    monkeypatch.setattr(gw, "_graceful_restart_via_sigusr1",
                        lambda pid, budget, **k: harness.signals.append((pid, signal_mod.SIGUSR1)) or False)
    monkeypatch.setattr(gateway_status, "get_running_pid", lambda: 4321)
    monkeypatch.setattr(gw, "_wait_for_api_server_port_free", lambda *a, **k: None)

    def _fake_run(argv, **kwargs):
        harness.kickstarts.append(list(argv))
        return subprocess.CompletedProcess(args=list(argv), returncode=0)

    monkeypatch.setattr("subprocess.run", _fake_run)
    return harness


def _dispatch_into_gateway(harness: Harness, gcalls: dict) -> None:
    """``_service_call`` stand-in: sample the lease, then run the real ``launchd_restart``."""
    import hermes_cli.gateway as gw

    harness.lease_at_dispatch.append(harness.dispatch_sample())
    gcalls["launchd_restart"]()


def drive_cli_restart(harness: Harness, monkeypatch, *, gcalls: dict | None = None, **kwargs):
    """Run ``_cmd_restart`` to completion against the fixture host."""
    import hermes_cli.gateway as gw

    gcalls = gcalls or {}
    gcalls.setdefault("launchd_restart", gw.launchd_restart)

    def _service_call(backend, verb, system=False):
        harness.service_calls.append((backend, verb))
        if backend == "launchd" and verb == "restart":
            _dispatch_into_gateway(harness, gcalls)

    monkeypatch.setattr(gw, "_service_call", _service_call)
    args = SimpleNamespace(system=False, all=False, force=False, **kwargs)
    gw._cmd_restart(args)
    return args


# --------------------------------------------------------------------------- #
# Actor census: the §3.1 table, machine-readable
# --------------------------------------------------------------------------- #

class TestActorCensus:
    """G1 named four actors; every entry point that can restart the host gateway is now classified."""

    def test_every_g1_actor_has_a_disposition(self):
        dispositions = orch.actor_dispositions()
        assert set(dispositions) == {
            "update-run",
            "manual-gateway-drain",
            "cli-gateway-restart",
            "dashboard-desktop-relaunch",
            "launchctl-kickstart-raw",
            "update-abort-recovery",
        }
        assert dispositions["update-run"] == "leased"
        assert dispositions["manual-gateway-drain"] == "arbitrated"
        assert dispositions["cli-gateway-restart"] == "leased"
        assert dispositions["dashboard-desktop-relaunch"] == "inherits-lease"
        assert dispositions["launchctl-kickstart-raw"] == "exempt"
        assert dispositions["update-abort-recovery"] == "exempt"

    def test_every_disposition_carries_a_written_reason(self):
        for actor in orch.RESTART_ACTORS:
            assert actor.reason.strip(), f"{actor.name} is classified without a reason"
            assert actor.disposition in {
                "leased", "arbitrated", "inherits-lease", "exempt",
            }, f"{actor.name} has an unknown disposition {actor.disposition!r}"

    @pytest.mark.parametrize("actor", [a for a in orch.RESTART_ACTORS if "(" not in a.entry_point],
                             ids=lambda a: a.name)
    def test_in_code_entry_points_exist(self, actor):
        """A classification naming a symbol that does not exist would be prose, not coverage."""
        module_name, _, attr = actor.entry_point.rpartition(".")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attr)), actor.entry_point

    def test_the_exempt_actors_are_exactly_the_out_of_process_ones(self):
        exempt = {a.name for a in orch.RESTART_ACTORS if a.disposition == "exempt"}
        assert exempt == {"launchctl-kickstart-raw", "update-abort-recovery"}

    def test_recovery_marker_is_read_not_assumed(self):
        assert orch.in_update_restart_recovery({}) is False
        assert orch.in_update_restart_recovery({orch.RECOVERY_ENV: "1"}) is True
        assert orch.in_update_restart_recovery({orch.RECOVERY_ENV: "0"}) is False

    def test_the_marker_the_orchestrator_reads_is_the_one_recovery_sets(self):
        """The exemption is only real if both ends spell the env var the same way."""
        from hermes_cli import update_restart_recovery as recovery

        assert orch.RECOVERY_ENV == recovery._RECOVERY_ENV
        assert recovery._child_environment("default")[orch.RECOVERY_ENV] == "1"


# --------------------------------------------------------------------------- #
# The CLI actor takes the lease
# --------------------------------------------------------------------------- #

class TestCliRestartActorTakesTheLease:
    def test_restart_runs_under_the_lease_and_releases_it_afterwards(self, harness, monkeypatch):
        drive_cli_restart(harness, monkeypatch)

        assert harness.service_calls == [("launchd", "restart")]
        held, drain_pid = harness.lease_at_dispatch[0]
        assert held is True, "the launchctl kickstart path must run under the host restart lease"
        assert drain_pid == 4321
        assert lease_file_or_none() is None, "the actor must release the lease when it is done"
        assert orch.restart_in_progress() is False

    def test_the_lease_names_the_actor_and_its_drain_deadline(self, harness, monkeypatch):
        seen: dict = {}

        def _sample(harness=harness):
            seen.update(lease_file())
            harness.lease_at_dispatch.append(harness.dispatch_sample())

        import hermes_cli.gateway as gw

        monkeypatch.setattr(gw, "_service_call", lambda backend, verb, system=False: _sample())
        gw._cmd_restart(SimpleNamespace(system=False, all=False, force=False))

        requestor = seen["requestor"]
        assert requestor["trigger"] == "hermes gateway restart"
        assert requestor["pid"] == os.getpid()
        # G9's whole point: the durable record names WHO, not just that something happened.
        assert requestor["argv"] and all(isinstance(part, str) for part in requestor["argv"])
        assert requestor["cwd"]

        drain = seen["drains"]["4321"]
        assert drain["deadline_ts"] > drain["at"]
        # The deadline is the DERIVED drain budget (ExitTimeOut 60 − 5), never the CLI's 1815 s
        # patience: under launchd the supervisor SIGKILLs at ExitTimeOut (G6).
        assert 50.0 <= drain["deadline_ts"] - drain["at"] <= 56.0

    def test_the_launchctl_kickstart_happens_inside_that_lease(self, harness, monkeypatch):
        """The SIGUSR1 path is the happy path; this drives the kickstart fallback of the same actor."""
        drive_cli_restart(harness, monkeypatch)  # _graceful_restart_via_sigusr1 returns False

        assert harness.kickstarts, "the fallback kickstart must be reached"
        assert harness.kickstarts[0][:3] == ["launchctl", "kickstart", "-k"]
        assert harness.lease_at_dispatch[0][0] is True

    def test_a_dead_holder_is_taken_over_not_waited_on(self, harness, monkeypatch):
        orch.acquire_restart_lease(
            sha="b" * 40, runtime_ids=["gateway:default"], trigger="actor-A",
            requestor=orch.Requestor(pid=999_999, argv=("hermes", "update"), cwd="/",
                                     hermes_home="/tmp", trigger="actor-A"))

        drive_cli_restart(harness, monkeypatch)

        assert harness.service_calls == [("launchd", "restart")]
        assert lease_file_or_none() is None

    def test_a_cli_restart_that_raises_still_releases_the_lease(self, harness, monkeypatch):
        import hermes_cli.gateway as gw

        monkeypatch.setattr(gw, "_service_call",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("launchctl died")))
        with pytest.raises(RuntimeError):
            gw._cmd_restart(SimpleNamespace(system=False, all=False, force=False))

        assert lease_file_or_none() is None, "a crash must not leave the host leased forever"


# --------------------------------------------------------------------------- #
# The second actor of 19:28:53: wait, then defer, signalling nothing
# --------------------------------------------------------------------------- #

class TestSecondActorWaitsThenDefers:
    def _hold_lease(self, trigger: str = "hermes update") -> None:
        outcome = orch.acquire_restart_lease(
            sha="b" * 40, runtime_ids=["gateway:default"], trigger=trigger,
            requestor=orch.Requestor(pid=os.getpid(), argv=("hermes", "update"), cwd="/",
                                     hermes_home="/tmp", trigger=trigger))
        assert outcome.acquired is True

    def test_deferral_exits_clean_without_signalling_or_restarting(self, harness, monkeypatch, capsys):
        import hermes_cli.gateway as gw

        monkeypatch.setattr(orch, "LEASE_WAIT_S", 0.05, raising=False)
        self._hold_lease()
        monkeypatch.setattr(gw, "_service_call",
                            lambda *a, **k: pytest.fail("a deferring actor must not restart anything"))

        gw._cmd_restart(SimpleNamespace(system=False, all=False, force=False))  # rc=0, no SystemExit

        out = capsys.readouterr().out
        assert "already-in-progress" in out
        assert "hermes update" in out, "the holder must be named, not just detected"
        assert harness.signals == [] and harness.kickstarts == []

    def test_the_holder_keeps_its_lease_and_its_drain(self, harness, monkeypatch):
        import hermes_cli.gateway as gw

        monkeypatch.setattr(orch, "LEASE_WAIT_S", 0.05, raising=False)
        self._hold_lease()
        orch.mark_drain_started(4321, deadline_ts=1e9, actor="update-drain:default", label="default")

        gw._cmd_restart(SimpleNamespace(system=False, all=False, force=False))

        record = lease_file()
        assert record["requestor"]["trigger"] == "hermes update"
        assert record["drains"]["4321"]["actor"] == "update-drain:default"

    def test_the_deferral_reads_as_a_no_op_to_the_fleet_watchdog(self, harness, monkeypatch, capsys):
        """``gateway_watchdog.sh`` greps a CLI restart's output for ``refus`` to spot a no-op.

        A deferral restored nothing, so it must match: the watchdog then logs NO-OP RECOVERY
        instead of recording a plausible ``cli restart``. This pins that contract without the repo
        test depending on a script that lives outside it.
        """
        import hermes_cli.gateway as gw

        monkeypatch.setattr(orch, "LEASE_WAIT_S", 0.05, raising=False)
        self._hold_lease()

        gw._cmd_restart(SimpleNamespace(system=False, all=False, force=False))

        out = capsys.readouterr().out.lower()
        assert "refus" in out, "the deferral must match GW_CLI_REFUSAL_MATCH"
        assert "already-in-progress" in out

    def test_an_older_sha_request_defers_to_the_later_update(self, harness, monkeypatch, capsys):
        import hermes_cli.gateway as gw

        monkeypatch.setattr(orch, "lease_request_is_stale", lambda sha, checkout_sha=None: True,
                            raising=False)
        monkeypatch.setattr(gw, "_service_call",
                            lambda *a, **k: pytest.fail("an older revision must not restart"))

        gw._cmd_restart(SimpleNamespace(system=False, all=False, force=False))

        assert "already-in-progress" in capsys.readouterr().out
        assert harness.signals == []

    def test_an_unwritable_lease_path_fails_open(self, harness, monkeypatch):
        """A missing host state dir must not turn into a gateway that cannot be restarted at all."""
        import hermes_cli.gateway as gw

        monkeypatch.setattr(orch, "lease_path", lambda: None, raising=False)

        gw._cmd_restart(SimpleNamespace(system=False, all=False, force=False))

        assert harness.signals or harness.kickstarts, "the restart must still happen"
        assert lease_file_or_none() is None


# --------------------------------------------------------------------------- #
# The 19:28:46 SIGTERM: refused and named inside the CLI's open drain
# --------------------------------------------------------------------------- #

class TestCliDrainIsDefended:
    def test_a_second_actors_sigterm_inside_the_deadline_is_refused_and_named(
            self, harness, monkeypatch, caplog):
        import hermes_cli.gateway as gw

        observed: list[orch.SignalOutcome] = []

        def _seconds_actor(backend, verb, system=False):
            # The 19:28:46 actor: a destructive SIGTERM for the pid the CLI is draining.
            observed.append(orch.restart_signal_gate(4321, "SIGTERM", actor="actor-B"))

        monkeypatch.setattr(gw, "_service_call", _seconds_actor)

        with caplog.at_level("WARNING", logger="hermes_cli.update_cmd"):
            gw._cmd_restart(SimpleNamespace(system=False, all=False, force=False))

        assert observed and observed[0].sent is False
        assert observed[0].reason == "drain-deadline-not-reached"
        assert observed[0].deadline_ts is not None, "the deadline must be durable, not process-local"

        record = caplog.records[-1].getMessage()
        assert "restart_signal_refused" in record
        assert "actor-B" in record and "drain-deadline-not-reached" in record
        assert harness.sigterms() == 0, "the offending SIGTERM must never reach the gateway"


# --------------------------------------------------------------------------- #
# The exempt actors
# --------------------------------------------------------------------------- #

class TestExemptActorsDoNotQueue:
    def test_an_update_abort_recovery_child_acts_on_the_holders_lease(
            self, harness, monkeypatch):
        """The updater that spawned it is blocked waiting for it — queuing here would deadlock."""
        monkeypatch.setenv(orch.RECOVERY_ENV, "1")
        orch.acquire_restart_lease(
            sha="b" * 40, runtime_ids=["gateway:default"], trigger="hermes update",
            requestor=orch.Requestor(pid=os.getpid(), argv=("hermes", "update"), cwd="/",
                                     hermes_home="/tmp", trigger="hermes update"))
        monkeypatch.setattr(orch, "LEASE_WAIT_S", 600.0, raising=False)

        drive_cli_restart(harness, monkeypatch)

        assert harness.service_calls == [("launchd", "restart")], "a recovery child must not defer"
        assert lease_file()["requestor"]["trigger"] == "hermes update", (
            "a recovery child must never release (or replace) the holder's lease"
        )

    def test_the_dashboard_relaunch_spawns_the_leased_cli_command(self, monkeypatch):
        """``inherits-lease`` is a claim about the child, so pin the child's argv."""
        from hermes_cli import web_server, web_server_gateway

        spawned: list[tuple] = []
        monkeypatch.setattr(web_server._gateway_mod, "_spawn_hermes_action",
                            lambda argv, name: spawned.append((tuple(argv), name))
                            or SimpleNamespace(pid=4242))
        import hermes_cli.gateway as gw

        monkeypatch.setattr(gw, "_reap_unsupervised_gateway_orphans", lambda: None, raising=False)
        monkeypatch.setattr(web_server, "_LAST_GATEWAY_RESTART", None, raising=False)

        web_server._spawn_gateway_restart(None)

        argv, name = spawned[0]
        assert name == "gateway-restart"
        assert list(argv[-2:]) == ["gateway", "restart"], (
            "the dashboard actor must spawn the leased CLI command, not a bare service call"
        )
        assert web_server_gateway._gateway_subcommand(None, "restart")[-2:] == ["gateway", "restart"]

    def test_the_dashboard_endpoint_funnels_into_the_classified_spawner(self, monkeypatch):
        """Desktop → ``POST /api/gateway/restart`` → ``actions.restart_gateway`` → the classified spawner.

        The Desktop's own client posts that path (``apps/desktop/src/api/system.ts``), so this is the
        link that makes ``inherits-lease`` true for the *Desktop* actor too, not just the browser UI.
        """
        import asyncio

        from hermes_cli import web_server
        from hermes_cli.web_routers import actions

        seen: list = []

        def _fake_spawn(profile=None):
            seen.append(profile)
            return SimpleNamespace(pid=7), False

        monkeypatch.setattr(web_server, "_spawn_gateway_restart", _fake_spawn, raising=False)

        asyncio.run(actions.restart_gateway(None))

        assert seen == [None], "the endpoint must reach the classified `inherits-lease` spawner"

    def test_the_raw_operator_kickstart_is_documented_as_unleasable(self):
        """No Hermes code runs in an operator's shell command; the reason must say so."""
        actor = next(a for a in orch.RESTART_ACTORS if a.name == "launchctl-kickstart-raw")
        assert actor.disposition == "exempt"
        assert "cannot be leased" in actor.reason
        assert "catch-up" in actor.reason

        import hermes_cli.gateway_launchd as gateway_launchd

        doc = gateway_launchd._launchd_kickstart.__doc__ or ""
        assert "exempt" in doc and "RESTART_ACTORS" in doc, (
            "the in-code kickstart seam must point at the classification"
        )


# --------------------------------------------------------------------------- #
# The context manager's own contract
# --------------------------------------------------------------------------- #

class TestRestartActorLeaseContract:
    def test_a_held_lease_is_a_deferral_and_is_not_released(self, monkeypatch):
        orch.acquire_restart_lease(
            sha="b" * 40, runtime_ids=["gateway:default"], trigger="actor-A",
            requestor=orch.Requestor(pid=os.getpid(), argv=("hermes", "update"), cwd="/",
                                     hermes_home="/tmp", trigger="actor-A"))
        monkeypatch.setattr(orch, "LEASE_WAIT_S", 0.05, raising=False)

        with orch.restart_actor_lease(trigger="actor-B") as outcome:
            assert outcome.must_defer is True
            assert outcome.state == "held"

        assert lease_file()["requestor"]["trigger"] == "actor-A"

    def test_an_acquired_lease_is_released_on_the_way_out(self):
        with orch.restart_actor_lease(trigger="actor-A") as outcome:
            assert outcome.must_defer is False
            assert outcome.state in ("acquired", "takeover")
            assert lease_file()["requestor"]["trigger"] == "actor-A"

        assert lease_file_or_none() is None

    def test_unavailable_is_not_a_deferral(self, monkeypatch):
        monkeypatch.setattr(orch, "lease_path", lambda: None, raising=False)

        with orch.restart_actor_lease(trigger="actor-A") as outcome:
            assert outcome.acquired is False
            assert outcome.must_defer is False, (
                "no writable host state dir must fail open, not block every restart"
            )

    def test_checkout_sha_is_never_guessed(self, monkeypatch):
        import hermes_cli.update_cmd_fleet as fleet

        monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: "c" * 40, raising=False)
        assert orch.checkout_restart_sha() == "c" * 40

        monkeypatch.setattr(fleet, "_current_checkout_sha", lambda: None, raising=False)
        assert orch.checkout_restart_sha() == "", "an unresolvable identity proves nothing"


# --------------------------------------------------------------------------- #
# Regression guard: the leased CLI restart keeps its existing behaviours
# --------------------------------------------------------------------------- #

class TestCliRestartStillHasItsOwnBehaviour:
    def test_external_supervisor_handback_still_reaches_the_supervisor(self, harness, monkeypatch):
        """The lease must not shadow the external-supervisor handback (t_110637's fix)."""
        import hermes_cli.gateway as gw
        from hermes_cli import gateway_supervised_restart as supervised

        calls: list[int] = []
        monkeypatch.setattr(gw, "_installed_service_kind_for", lambda windows: None)
        monkeypatch.setattr(gw, "stop_profile_gateway", lambda: pytest.fail("must not stop it"))
        monkeypatch.setattr(supervised, "restart_externally_supervised_gateway",
                            lambda pid: calls.append(pid))
        monkeypatch.setattr(supervised, "gateway_declares_external_supervisor", lambda pid: True)

        gw._cmd_restart(SimpleNamespace(system=False, all=False, force=False))

        assert calls == [4321]
        assert harness.signals == []
        assert lease_file_or_none() is None

    def test_path_of_the_plist_used_for_budgets(self, plist):
        assert Path(plist).exists()
