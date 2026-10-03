"""A symlinked profile that is really another install must not fail `hermes update` (#120240).

``~/.hermes/profiles/work -> ~/.hermes-work`` where ``~/.hermes-work`` has its own checkout and
gateway: the plan inventories that gateway, the restart phase correctly leaves it alone, and the
fleet matrix already classifies it ``external``. Reconciliation must agree instead of reporting it
``unaccounted`` and exiting 1.
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd_fleet as fleet
from hermes_cli import update_receipt
from hermes_cli.update_inventory import (
    RuntimeRecord,
    UpdatePlan,
    match_runtime_outcomes,
    report_unaccounted_runtimes,
)

SHA = "a" * 40


def _checkout(root: Path) -> Path:
    (root / "hermes_cli").mkdir(parents=True)
    (root / "hermes_cli" / "main.py").write_text("", encoding="utf-8")
    return root


def _gateway_state(home: Path, pid: int, entrypoint: Path, sha: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "gateway_state.json").write_text(json.dumps({
        "gateway_state": "running", "kind": "hermes-gateway", "pid": pid,
        "argv": [str(entrypoint), "gateway", "run"], "code_sha": sha, "code_version": "1.0",
    }), encoding="utf-8")


def _gateway(profile: str, pid: int) -> RuntimeRecord:
    return RuntimeRecord(kind="gateway", profile=profile, pid=pid, supervisor="launchd", restart_via="launchd")


@pytest.fixture
def external_profile_host(tmp_path, monkeypatch):
    """Root install with its own gateway plus ``profiles/work`` symlinked to a separate install."""
    root = tmp_path / "root_home"
    updater_main = update_receipt._updater_code_root() / "hermes_cli" / "main.py"
    _gateway_state(root, 1111, updater_main, SHA)

    other_checkout = _checkout(tmp_path / "other_install" / "hermes-agent")
    other_home = tmp_path / "other_install" / "home"
    _gateway_state(other_home, 4242, other_checkout / "hermes_cli" / "main.py", "f" * 40)
    (root / "profiles").mkdir()
    (root / "profiles" / "work").symlink_to(other_home, target_is_directory=True)

    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(update_receipt, "_code_identity", lambda refresh=False: {"sha": SHA, "version": "1.0"})
    monkeypatch.setattr(update_receipt, "_socket_identity", lambda _home: None)
    live = {root.resolve(): 1111, other_home.resolve(): 4242}
    monkeypatch.setattr("gateway.status.live_gateway_pid_for_home", lambda home: live.get(Path(home).resolve()))
    monkeypatch.setattr(fleet, "_time", SimpleNamespace(sleep=lambda _s: None, monotonic=time.monotonic))
    monkeypatch.setattr(fleet, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr("hermes_cli.update_cmd._finish_dashboard_update_cleanup", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.gateway_migrate.maybe_auto_migrate_after_update", lambda: None)
    return SimpleNamespace(other_checkout=other_checkout)


def _restart_outcome() -> "fleet._GatewayRestartOutcome":
    # The root's own LaunchAgent was restarted; the other install's gateway was left alone.
    return fleet._GatewayRestartOutcome(
        False, [], [1111, 4242], ["ai.hermes.gateway"], [], [], [], set(),
    )


def test_symlinked_external_profile_gateway_does_not_fail_update(external_profile_host, capsys):
    plan = UpdatePlan(runtimes=[_gateway("default", 1111), _gateway("work", 4242)])
    update_receipt.begin_update_receipt()

    fleet._verify_fleet_after_update(
        _restart_outcome(), _pre_update_plan=plan, _windows_gateway_resume=None, update_complete=True,
    )

    receipt = update_receipt.read_latest_receipt()
    assert receipt["outcome"] == "success"
    assert {o["profile"]: o["outcome"] for o in receipt["runtime_outcomes"]} == {
        "default": "restarted", "work": "external",
    }
    out = capsys.readouterr().out
    assert "never touched" not in out
    assert "gateway [work] pid 4242" in out  # still surfaced, informationally


def test_external_profile_does_not_mask_a_missed_own_gateway(external_profile_host):
    """The root's own gateway missed by the restart phase is still an owed restart."""
    plan = UpdatePlan(runtimes=[_gateway("default", 1111), _gateway("work", 4242)])
    restart = _restart_outcome()
    restart.restarted_services.clear()
    update_receipt.begin_update_receipt()

    # Contract C3: no longer SystemExit(1); the miss is flagged and owed instead.
    fleet._verify_fleet_after_update(
        restart, _pre_update_plan=plan, _windows_gateway_resume=None, update_complete=True,
    )

    assert restart.incomplete
    receipt = update_receipt.read_latest_receipt()
    outcomes = {o["profile"]: o["outcome"] for o in receipt["runtime_outcomes"]}
    assert outcomes == {"default": "unaccounted", "work": "external"}
    assert [f["step"] for f in receipt["followups"]] == ["gateway_restart"]


def test_external_requires_verified_pid_evidence():
    """Fail closed: no external evidence, or evidence for another pid, stays unaccounted."""
    plan = UpdatePlan(runtimes=[_gateway("work", 4242)])
    common = dict(
        restarted_services=[], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids=set(), failed_units=[],
    )
    assert match_runtime_outcomes(plan, **common)[0]["outcome"] == "unaccounted"
    assert match_runtime_outcomes(plan, **common, external_gateway_pids={9999})[0]["outcome"] == "unaccounted"
    outcomes = match_runtime_outcomes(plan, **common, external_gateway_pids={4242})
    assert outcomes[0]["outcome"] == "external"
    assert report_unaccounted_runtimes(outcomes) is False


def test_external_evidence_never_covers_a_serve_runtime():
    plan = UpdatePlan(runtimes=[RuntimeRecord(kind="serve", profile="work", pid=4242, supervisor="manual-serve")])
    outcomes = match_runtime_outcomes(
        plan, restarted_services=[], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids=set(), failed_units=[], external_gateway_pids={4242},
    )
    assert outcomes[0]["outcome"] == "unaccounted"
