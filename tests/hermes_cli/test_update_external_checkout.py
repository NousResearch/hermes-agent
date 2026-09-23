"""Regression coverage for separate-checkout gateways (#117449)."""

from pathlib import Path
from types import SimpleNamespace

from hermes_cli import update_cmd_fleet, update_inventory, update_receipt


def _checkout(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    module = root / "hermes_cli" / "main.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    return root


def test_different_checkout_is_external_but_same_checkout_sha_skew_is_stale(tmp_path):
    updated = _checkout(tmp_path, "updated")
    separate = _checkout(tmp_path, "separate")

    external = update_receipt._fleet_row(
        "pinned", 1, "old", "1", "new",
        code_root=separate, expected_root=updated, served_profiles=["pinned", "coder", "coder"],
    )
    stale = update_receipt._fleet_row(
        "default", 2, "old", "1", "new",
        code_root=updated, expected_root=updated,
    )

    assert external["state"] == "external"
    assert external["served_profiles"] == ["pinned", "coder"]
    assert stale["state"] == "stale"


def test_unresolved_checkout_fails_closed_to_sha_comparison():
    row = update_receipt._fleet_row("default", 1, "old", "1", "new")
    assert row["state"] == "stale"


def test_code_root_requires_absolute_path(tmp_path, monkeypatch):
    _checkout(tmp_path, "updated")
    monkeypatch.chdir(tmp_path / "updated")
    assert update_receipt._code_root_for_path("-m") is None
    assert update_receipt._code_root_for_path("hermes_cli/main.py") is None


def test_gateway_root_uses_pid_guarded_status_argv(tmp_path, monkeypatch):
    root = _checkout(tmp_path, "pinned")
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda _path: {"pid": 42, "argv": [str(root / "hermes_cli" / "main.py")]},
    )
    assert update_receipt._gateway_code_root(42, tmp_path / "home") == root.resolve()


def test_gateway_root_uses_virtualenv_when_module_path_is_absent(tmp_path, monkeypatch):
    root = _checkout(tmp_path, "pinned")
    venv = root / "venv"
    monkeypatch.setattr("gateway.status.read_runtime_status", lambda _path: {"pid": 42, "argv": ["-m"]})
    fake_process = SimpleNamespace(
        environ=lambda: {"VIRTUAL_ENV": str(venv)},
        exe=lambda: "/usr/bin/python",
        cmdline=lambda: ["python", "-m", "hermes_cli.main"],
    )
    monkeypatch.setattr("psutil.Process", lambda _pid: fake_process)
    assert update_receipt._gateway_code_root(42, tmp_path / "home") == root.resolve()


def test_external_receipt_row_does_not_report_runtime_skew():
    receipt = {"fleet": [{"profile": "pinned", "state": "external", "code_sha": "foreign"}]}
    assert update_cmd_fleet._receipt_reports_stale_runtime(receipt, "updated") is False


def test_external_gateway_does_not_block_pending_restart_discharge(monkeypatch):
    rows = [
        {"profile": "default", "state": "current", "code_sha": "updated"},
        {
            "profile": "pinned", "state": "external", "code_sha": "foreign",
            "served_profiles": ["pinned", "coder"],
        },
    ]
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda: rows)
    owed = {("gateway", "default"), ("gateway", "pinned"), ("gateway", "coder")}
    assert update_cmd_fleet._live_fleet_covers_receipt("updated", {}, owed) is True


def test_external_gateway_does_not_fail_matrix(capsys):
    failed = update_receipt.print_fleet_version_matrix([
        {
            "profile": "pinned", "pid": 42, "code_sha": "foreign",
            "state": "external", "code_root": "/srv/hermes-pinned",
        }
    ])
    output = capsys.readouterr().out
    assert failed is False
    assert "separate checkout" in output
    assert "/srv/hermes-pinned" in output


def test_collect_fleet_versions_classifies_separate_checkout_gateway(tmp_path, monkeypatch):
    """The production entry point resolves the gateway's code root and reports ``external``.

    A pid-guarded ``gateway_state.json`` whose argv points at a pinned checkout must
    classify as ``external`` (never ``stale``) even though its sha is foreign.
    """
    import json

    pinned = _checkout(tmp_path, "pinned")
    home = tmp_path / "fleet_home"
    home.mkdir()
    (home / "gateway_state.json").write_text(json.dumps({
        "gateway_state": "running", "kind": "hermes-gateway", "pid": 4242,
        "argv": [str(pinned / "hermes_cli" / "main.py"), "gateway", "run"],
        "code_sha": "f" * 40, "code_version": "0.0.1",
    }), encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.build_info.get_code_identity",
        lambda refresh=False: {"sha": "a" * 40, "short_sha": "a" * 8, "version": "1.0", "source": "git"},
    )
    monkeypatch.setattr("hermes_cli.profiles._get_default_hermes_home", lambda: home)
    monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "no_profiles")
    monkeypatch.setattr(update_receipt, "_socket_identity", lambda _home: None)
    monkeypatch.setattr("gateway.status.live_gateway_pid_for_home", lambda _home: 4242)

    fleet = update_receipt.collect_fleet_versions()

    assert [row["state"] for row in fleet] == ["external"]
    assert fleet[0]["code_root"] == str(pinned.resolve())
    assert update_receipt.print_fleet_version_matrix(fleet) is False  # matrix does not fail the update


def test_plan_reconciliation_skips_only_proven_external_gateway(tmp_path, monkeypatch, capsys):
    """A separate install's gateway is recorded but is not this checkout's restart debt."""
    external = _checkout(tmp_path, "external")
    own = update_receipt._updater_code_root()
    assert own is not None
    homes = [("default", tmp_path / "default"), ("work", tmp_path / "work")]
    records = {
        homes[0][1]: {"pid": 41, "argv": [str(own / "hermes_cli" / "main.py")]},
        homes[1][1]: {"pid": 42, "argv": [str(external / "hermes_cli" / "main.py")]},
    }
    monkeypatch.setattr(update_inventory, "_supervisor_classifier", lambda: lambda _pid: "launchd")
    monkeypatch.setattr(update_receipt, "_socket_identity", lambda _home: None)
    monkeypatch.setattr("gateway.status.live_gateway_pid_for_home", lambda home: records[home]["pid"])
    monkeypatch.setattr("gateway.status.read_runtime_status", lambda path: records[path.parent])
    monkeypatch.setattr("hermes_cli.gateway.find_profile_gateway_processes", lambda: [])
    plan = update_inventory.UpdatePlan()
    update_inventory._collect_gateway_runtimes(plan, homes, set())
    restored = update_inventory.UpdatePlan.from_dict(plan.to_dict())
    outcomes = update_inventory.match_runtime_outcomes(
        restored, restarted_services=["ai.hermes.gateway"], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    assert [(row["profile"], row["outcome"]) for row in outcomes] == [
        ("default", "restarted"), ("work", "external"),
    ]
    assert update_inventory.report_unaccounted_runtimes(outcomes) is False
    update_inventory.print_update_plan(restored)
    assert "external checkout" in capsys.readouterr().out

    # An older pre-swap updater did not stamp ownership into the plan. The
    # post-swap fleet probe still proves this exact gateway PID is external.
    legacy = update_inventory.UpdatePlan.from_dict(plan.to_dict())
    legacy.runtimes[1].detail = {}
    recovered = update_inventory.match_runtime_outcomes(
        legacy, restarted_services=["ai.hermes.gateway"], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
        external_gateway_pids={42},
    )
    assert [row["outcome"] for row in recovered] == ["restarted", "external"]


def test_unknown_gateway_checkout_remains_restart_debt(tmp_path, monkeypatch):
    home = tmp_path / "work"
    monkeypatch.setattr(update_inventory, "_supervisor_classifier", lambda: lambda _pid: "launchd")
    monkeypatch.setattr(update_receipt, "_socket_identity", lambda _home: None)
    monkeypatch.setattr(update_receipt, "_gateway_code_root", lambda _pid, _home: None)
    monkeypatch.setattr("gateway.status.live_gateway_pid_for_home", lambda _home: 42)
    monkeypatch.setattr("gateway.status.read_runtime_status", lambda _path: {"pid": 42})
    monkeypatch.setattr("hermes_cli.gateway.find_profile_gateway_processes", lambda: [])
    plan = update_inventory.UpdatePlan()
    update_inventory._collect_gateway_runtimes(plan, [("work", home)], set())
    outcomes = update_inventory.match_runtime_outcomes(
        plan, restarted_services=[], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    assert outcomes[0]["outcome"] == "unaccounted"
    assert update_inventory.report_unaccounted_runtimes(outcomes) is True


def test_external_gateway_is_not_a_manual_restart_target(tmp_path, monkeypatch):
    import subprocess
    import sys

    import hermes_cli.gateway as gateway

    external_root = _checkout(tmp_path, "external")
    external_home = tmp_path / "external-home"
    external_home.mkdir()
    profiles = tmp_path / "root-home" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "work").symlink_to(external_home, target_is_directory=True)
    own_root = update_receipt._updater_code_root()
    assert own_root is not None
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        monkeypatch.setattr(update_receipt, "_profile_homes", lambda: [("work", profiles / "work")])
        monkeypatch.setattr("gateway.status.live_gateway_pid_for_home", lambda _home: process.pid)
        monkeypatch.setattr(update_receipt, "_gateway_code_root", lambda _pid, _home: external_root.resolve())
        plan = update_inventory.UpdatePlan(runtimes=[update_inventory._runtime(
            "gateway", "work", process.pid, "manual", detail={"code_root": str(external_root.resolve())},
        )])
        external_pids = update_cmd_fleet._verified_external_gateway_pids(plan)
        assert external_pids == {process.pid}
        monkeypatch.setattr(gateway, "_get_service_pids", lambda **_kw: set())
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda **_kw: [process.pid])
        monkeypatch.setattr(gateway, "find_profile_gateway_processes", lambda **_kw: [])
        outcome = update_cmd_fleet._GatewayRestartOutcome(
            incomplete=False, phase_errors=[], pre_restart_gateway_pids=[], restarted_services=[],
            failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set(),
        )
        update_cmd_fleet._restart_manual_gateways(outcome, 0, external_pids=external_pids)
        assert process.poll() is None
        assert outcome.killed_pids == set()
        # A missing ownership proof must not exempt this process from the restart sweep.
        monkeypatch.setattr(update_receipt, "_gateway_code_root", lambda _pid, _home: None)
        assert update_cmd_fleet._verified_external_gateway_pids(plan) == set()
        assert own_root != external_root.resolve()
    finally:
        process.terminate()
        process.wait(timeout=5)
