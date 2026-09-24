"""A CONVERGED host is a no-op for ``hermes gateway migrate --multiplex``, run after run.

Driven through the REAL topology path, not stubbed per-profile PIDs: a live host rendezvous
record + the per-profile ``gateway_state.json`` the multiplexer stamps, so
``gateway.status.live_gateway_pid_for_home`` genuinely answers with the HOST gateway's PID for
every secondary (that is ``_host_gateway_serves_home`` working as designed). Stubbing PIDs by
profile NAME is what let the "half-migrated converged host" defect hide behind green CI: the
fixture simply never produced the value the reporting path produces.

Only the WIRE is faked (the owner's ``identify`` answer over its control socket); every predicate
under test — ownership, the plan's topology verdict, what the apply signals — is the real one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import hermes_constants
from hermes_cli import gateway_migrate as gm


@pytest.fixture
def converged_host(tmp_path, monkeypatch):
    """default + coder + ops, all served by ONE live host gateway (this process).

    Returns the root home. ``signals`` on the returned object records every service operation and
    process stop the migration would perform — a converged host must produce none.
    """
    root = tmp_path / "home" / ".hermes"
    for sub in ("profiles/coder", "profiles/ops"):
        (root / sub).mkdir(parents=True)
        (root / sub / "SOUL.md").write_text("x\n", encoding="utf-8")
    (root / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    assert str(hermes_constants.get_default_hermes_root()).startswith(str(tmp_path))

    served = ["default", "coder", "ops"]
    pid = os.getpid()
    # What a live multiplexer leaves on disk: its own pid file + runtime record in the launch
    # home, and a runtime record stamped into EVERY served profile's home.
    (root / "gateway.pid").write_text(json.dumps({"pid": pid, "hermes_home": str(root)}))
    (root / "gateway_state.json").write_text(json.dumps(
        {"pid": pid, "hermes_home": str(root), "gateway_state": "running", "served_profiles": served}))
    for name in ("coder", "ops"):
        home = root / "profiles" / name
        (home / "gateway_state.json").write_text(json.dumps(
            {"pid": pid, "hermes_home": str(home), "gateway_state": "running"}))

    import gateway.status as status
    from gateway import host_attach, host_rendezvous as hr
    monkeypatch.setattr(status, "_read_process_cmdline", lambda p: "hermes gateway run")
    # The WIRE only: the owner's control socket answer. Everything that reads it is real.
    monkeypatch.setattr("gateway.control_socket.identify_gateway",
                        lambda home: {"pid": pid, "hermes_home": str(root), "served_profiles": served})
    hr.publish_record(hr.ROLE_GATEWAY, profiles=tuple(served), home=str(root))
    host_attach.invalidate_host_gateway_cache()

    signals: list[tuple] = []
    monkeypatch.setattr(gm, "_service_op",
                        lambda kind, system, verb, home, **kw: signals.append(("service", verb, str(home))))
    monkeypatch.setattr(gm, "_stop_gateway_process", lambda home: signals.append(("stop", str(home))))
    monkeypatch.setattr(gm, "_spawn_detached_gateway", lambda home: signals.append(("spawn", str(home))) or True)
    monkeypatch.setattr(gm, "_host_supports_migration", lambda: None)
    yield type("Fixture", (), {"root": root, "signals": signals, "pid": pid, "served": served})
    hr.clear_record(hr.ROLE_GATEWAY)


def test_the_host_gateway_pid_is_reported_for_every_secondary(converged_host):
    """Premise of the defect, asserted so the test cannot silently stop exercising it: the
    topology reporter DOES hand the host gateway's PID to every profile it serves."""
    root = converged_host.root
    for name in ("coder", "ops"):
        assert gm._live_gateway_pid(root / "profiles" / name) == converged_host.pid


def test_a_converged_host_is_already_multiplexed_and_owns_nothing(converged_host):
    plan = gm.build_migration_plan()
    assert [p.name for p in plan.standalone_secondaries] == []
    assert all(p.pid is None for p in plan.secondaries), "a served profile does not OWN the host gateway"
    assert plan.default.pid == converged_host.pid, "the launch home does own it"
    assert plan.already_multiplexed and not plan.interrupted
    assert not plan.eligible_for_migration()


def test_three_consecutive_runs_signal_nothing(converged_host, capsys):
    """The update hook and the explicit command, three times over, on an untouched host."""
    from types import SimpleNamespace
    for _ in range(3):
        gm.maybe_auto_migrate_after_update()
        gm.cmd_migrate(SimpleNamespace(multiplex=True, dry_run=False, yes=True))  # returns, never exits
    out = capsys.readouterr().out
    assert converged_host.signals == [], f"a converged host was signalled: {converged_host.signals}"
    assert "Half-migrated" not in out and "SIGTERM" not in out
    assert "already multiplexing" in out
