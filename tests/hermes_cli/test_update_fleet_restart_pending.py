"""Interrupted-update fleet-restart obligation (#95294 parts 1+2).

A ``hermes update`` killed after git pull advanced HEAD but before the
fleet restart left running gateways on stale code. The next update said
"Already up to date" and skipped restart. These tests cover:

- ``fleet_restart_pending`` marker written after HEAD advances, cleared
  after a successful (or no-op) fleet restart
- interrupt between pull and restart leaves the marker
- next ``hermes update`` with git already up to date still runs the
  pending restart when the marker OR a skewed unfinished latest.json is
  present

No live gateway, no network. Git and restart are mocked.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_cli import main as hermes_main
import hermes_cli.main_web_build as main_web_build
import hermes_cli.main_install_repair as main_install_repair
from hermes_cli import update_cmd
import hermes_cli.update_cmd_fleet as update_cmd_fleet
import hermes_cli.update_cmd_deps as update_cmd_deps
from hermes_cli.update_receipt import COMMAND_BOUNDARY_STOP_REASON
from hermes_constants import get_hermes_home
import hermes_cli.update_host_obligation as host_obligation
from gateway import host_rendezvous


def _make_head_moved_side_effect(pre_sha="abc123", post_sha="def456"):
    """Simulate git commands where HEAD advances from pre_sha to post_sha."""
    calls = {"n": 0}

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)

        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")

        if "rev-list" in joined:
            return SimpleNamespace(returncode=0, stdout="3\n", stderr="")

        if joined.endswith("rev-parse HEAD"):
            if calls["n"] == 0:
                calls["n"] += 1
                return SimpleNamespace(returncode=0, stdout=f"{pre_sha}\n", stderr="")
            return SimpleNamespace(returncode=0, stdout=f"{post_sha}\n", stderr="")

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return side_effect


def _make_up_to_date_side_effect(sha="abc123"):
    """Simulate git commands where origin is already at HEAD."""

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)

        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")

        if "rev-list" in joined:
            return SimpleNamespace(returncode=0, stdout="0\n", stderr="")

        if joined.endswith("rev-parse HEAD"):
            return SimpleNamespace(returncode=0, stdout=f"{sha}\n", stderr="")

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return side_effect


def _patch_update_deps(monkeypatch, tmp_path, run_side_effect):
    """Patch ``_cmd_update_impl`` helpers. Mirrors test_update_head_moved_gate."""
    monkeypatch.setattr(hermes_main.subprocess, "run", run_side_effect)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "_resolve_update_branch", lambda args: "main")
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: False)
    monkeypatch.setattr(
        update_cmd, "_restart_macos_launchd_gateways", lambda *a, **k: None
    )
    monkeypatch.setattr(
        update_cmd_fleet, "_restart_macos_launchd_gateways", lambda *a, **k: None
    )
    monkeypatch.setattr(
        hermes_main,
        "_get_origin_url",
        lambda *a, **k: "https://github.com/NousResearch/hermes-agent.git",
    )
    monkeypatch.setattr(update_cmd, "_is_fork", lambda *a, **k: False)
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed", lambda *a, **k: None
    )
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", lambda *a, **k: 0)
    monkeypatch.setattr(
        hermes_main, "_record_bytecode_fingerprint", lambda *a, **k: None
    )
    monkeypatch.setattr(
        main_web_build, "_record_bytecode_fingerprint", lambda *a, **k: None
    )
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", lambda *a, **k: None)
    monkeypatch.setattr(
        hermes_main, "_pause_windows_gateways_for_update", lambda: None
    )
    monkeypatch.setattr(
        hermes_main, "_resume_windows_gateways_after_update", lambda *a, **k: None
    )
    monkeypatch.setattr(hermes_main, "_write_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(hermes_main, "_clear_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(main_install_repair, "_clear_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(update_cmd, "_finish_dashboard_update_cleanup", lambda *a, **k: None
    )
    monkeypatch.setattr(
        update_cmd, "_finish_dashboard_update_cleanup", lambda *a, **k: None
    )
    monkeypatch.setattr(hermes_main, "_build_web_ui", lambda *a, **k: None)
    monkeypatch.setattr(main_web_build, "_build_web_ui", lambda *a, **k: None)
    monkeypatch.setattr(
        update_cmd, "_venv_core_imports_healthy", lambda: (True, "")
    )
    monkeypatch.setattr(update_cmd, "_update_node_dependencies", lambda: [])
    monkeypatch.setattr(update_cmd_deps, "_update_node_dependencies", lambda: [])

    import hermes_cli.gateway as hermes_gateway

    monkeypatch.setattr(
        hermes_gateway, "find_gateway_pids", lambda **_kwargs: []
    )
    monkeypatch.setattr(hermes_gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(
        hermes_gateway, "find_profile_gateway_processes", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda **k: [],
    )
    monkeypatch.setattr(
        "hermes_cli.update_inventory.collect_runtime_inventory",
        lambda: SimpleNamespace(runtimes=[], to_dict=lambda: {}),
    )


def _update_args():
    return SimpleNamespace(branch=None, yes=False, force=False, force_venv=False)


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------


def test_obligation_round_trip_is_host_scoped():
    """The obligation is one record per HOST (beside the host rendezvous record), not per home."""
    path = host_obligation.host_obligation_path()
    assert path.parent == host_rendezvous.host_state_dir()
    assert not path.exists()

    update_cmd._write_fleet_restart_pending_marker(expected_sha="abc123")
    assert update_cmd_fleet._fleet_restart_obligation_armed()
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["expected_sha"] == "abc123"
    assert record["pid"] and record["started"]
    assert not (get_hermes_home() / "fleet_restart_pending").exists()

    update_cmd._clear_fleet_restart_pending_marker()
    assert not update_cmd_fleet._fleet_restart_obligation_armed()


def test_pending_needed_when_marker_exists():
    update_cmd._write_fleet_restart_pending_marker()
    assert update_cmd._pending_fleet_restart_needed() is True
    update_cmd._clear_fleet_restart_pending_marker()
    assert update_cmd._pending_fleet_restart_needed() is False


def test_pending_needed_when_unfinished_receipt_runtime_sha_skews(monkeypatch):
    disk_sha = "e" * 40
    old_sha = "7" * 40
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: disk_sha)
    monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: disk_sha)

    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(
        json.dumps(
            {
                "exit_code": 1,
                "stop_reason": "KeyboardInterrupt: ",
                "outcome": "failed",
                "plan": {
                    "expected_sha": disk_sha,
                    "runtimes": [
                        {
                            "kind": "gateway",
                            "profile": "default",
                            "pid": 2111768,
                            "supervisor": "systemd",
                            "code_sha": old_sha,
                            "restart_via": "systemd",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    assert update_cmd._pending_fleet_restart_needed() is True


def test_successful_receipt_with_pre_update_plan_shas_does_not_retrigger(
    monkeypatch,
):
    """A completed update's plan.runtimes are pre-pull SHAs — not a catch-up."""
    disk_sha = "n" * 40
    old_sha = "o" * 40
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: disk_sha)
    monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: disk_sha)

    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(
        json.dumps(
            {
                "exit_code": 0,
                "outcome": "success",
                "plan": {
                    "expected_sha": old_sha,
                    "runtimes": [
                        {
                            "kind": "gateway",
                            "profile": "default",
                            "pid": 1,
                            "code_sha": old_sha,
                        }
                    ],
                },
                "fleet": [
                    {
                        "profile": "default",
                        "pid": 2,
                        "code_sha": disk_sha,
                        "state": "current",
                    }
                ],
                "gateway_restart": {"incomplete": False},
            }
        ),
        encoding="utf-8",
    )

    assert update_cmd._pending_fleet_restart_needed() is False


def test_successful_command_boundary_receipt_without_fleet_does_not_retrigger(
    monkeypatch,
):
    """A normal command-boundary stop is not an interrupted update."""
    disk_sha = "n" * 40
    old_sha = "o" * 40
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: disk_sha)
    monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: disk_sha)

    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(
        json.dumps(
            {
                "exit_code": 0,
                "outcome": "success",
                "stop_reason": COMMAND_BOUNDARY_STOP_REASON,
                "plan": {
                    "expected_sha": old_sha,
                    "runtimes": [
                        {
                            "kind": "gateway",
                            "profile": "default",
                            "pid": 1,
                            "code_sha": old_sha,
                        }
                    ],
                },
                "fleet": [],
                "gateway_restart": {},
            }
        ),
        encoding="utf-8",
    )

    assert update_cmd._pending_fleet_restart_needed() is False


@pytest.mark.parametrize(
    ("receipt", "unfinished"),
    [
        pytest.param({"outcome": "success", "exit_code": 0, "stop_reason": "sys.exit(0)"}, False, id="success-sys-exit-0"),
        pytest.param({"outcome": "success", "stop_reason": "KeyboardInterrupt: "}, False, id="success-no-exit-code"),
        pytest.param({"exit_code": 0, "stop_reason": "sys.exit(0)"}, False, id="exit-0-no-outcome"),
        # update_contract writes {"outcome": "refused", "stop_reason": <code>} with no exit_code;
        # the stop_reason clause is what keeps that receipt unfinished.
        pytest.param({"outcome": "refused", "stop_reason": "not_updatable_in_place"}, True, id="refused-stop-reason-only"),
        pytest.param({"outcome": "failed", "exit_code": 1, "stop_reason": "KeyboardInterrupt: "}, True, id="failed-interrupt"),
    ],
)
def test_stop_reason_only_marks_unfinished_when_nothing_vouches_for_success(receipt, unfinished):
    assert update_cmd._receipt_looks_unfinished(receipt) is unfinished


def test_stale_fleet_matrix_on_latest_receipt_is_pending(monkeypatch):
    disk_sha = "n" * 40
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: disk_sha)
    monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: disk_sha)

    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(
        json.dumps(
            {
                "outcome": "partial",
                "exit_code": 1,
                "fleet": [
                    {
                        "profile": "default",
                        "pid": 9,
                        "code_sha": "s" * 40,
                        "state": "stale",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert update_cmd._pending_fleet_restart_needed() is True


def test_run_pending_restart_true_when_no_gateways(monkeypatch, capsys):
    monkeypatch.setattr(
        "hermes_cli.gateway.find_gateway_pids", lambda **k: []
    )

    # An empty PID scan is insufficient; every supervisor scope must answer empty.
    monkeypatch.setattr(update_cmd_fleet, "_systemd_gateway_unit_listings", lambda: [
        (scope, cmd, SimpleNamespace(returncode=0, stdout=""))
        for scope, cmd in update_cmd_fleet._SYSTEMD_SCOPES
    ])
    # The launchd scope too: a developer machine with a live fleet would otherwise
    # drain its real units and report the restart incomplete (#110701).
    monkeypatch.setattr(
        update_cmd_fleet, "_restart_macos_launchd_gateways", lambda *a, **k: None
    )
    # And the Windows scope: an installed Windows gateway service would be restarted for real.
    monkeypatch.setattr("hermes_cli.gateway_windows.is_installed", lambda: False)
    assert update_cmd._run_pending_fleet_restart() is True
    assert "Pending fleet restart completed" in capsys.readouterr().out


def test_run_pending_restart_skips_gateways_already_on_checkout_code(monkeypatch, capsys):
    """A gateway the update itself cold-started seconds earlier is already current on the checkout
    SHA: the catch-up must not stop it (the Windows stop/start pair then printed "No gateway was
    running" plus a second spawn, #117051) and must report nothing to restart."""
    sha = "c" * 40
    monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: sha)
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **k: [48096])
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda: [{"profile": "default", "pid": 48096, "code_sha": sha, "state": "current"}],
    )
    stopped = []
    monkeypatch.setattr("hermes_cli.gateway.kill_gateway_processes", lambda **k: stopped.append(k))
    monkeypatch.setattr("hermes_cli.gateway_windows.restart", lambda: stopped.append("windows-restart"))
    monkeypatch.setattr("hermes_cli.gateway_windows.is_installed", lambda: True)
    monkeypatch.setattr(update_cmd_fleet, "_restart_macos_launchd_gateways", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd_fleet, "_systemd_gateway_unit_listings", lambda: [])

    assert update_cmd._run_pending_fleet_restart() is True
    assert stopped == []
    assert "nothing to restart" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# cmd_update integration (mocked git / restart)
# ---------------------------------------------------------------------------


def test_marker_written_after_pull_cleared_after_successful_restart(
    monkeypatch, tmp_path, capsys
):
    args = _update_args()
    _patch_update_deps(monkeypatch, tmp_path, _make_head_moved_side_effect())

    wrote = []
    orig = update_cmd._write_fleet_restart_pending_marker

    def _spy(*, expected_sha="", runtimes=None):
        orig(expected_sha=expected_sha, runtimes=runtimes)
        wrote.append(update_cmd_fleet._fleet_restart_obligation_armed())

    monkeypatch.setattr(update_cmd, "_write_fleet_restart_pending_marker", _spy)

    hermes_main.cmd_update(args)

    assert wrote == [True], "marker must exist immediately after HEAD advances"
    assert not update_cmd_fleet._fleet_restart_obligation_armed()
    out = capsys.readouterr().out
    assert "✓ Code updated!" in out


def test_clean_update_warns_about_surviving_pre_update_serve_runtime(
    monkeypatch, tmp_path, capsys
):
    """The successful update path must surface an inventoried stale serve."""
    args = _update_args()
    _patch_update_deps(monkeypatch, tmp_path, _make_head_moved_side_effect())
    monkeypatch.setattr(
        update_cmd,
        "_surviving_pre_update_serve_runtimes",
        lambda _plan: [
            {
                "pid": 5555,
                "kind": "serve",
                "profile": "default",
                "supervisor": "manual-serve",
            }
        ],
    )

    hermes_main.cmd_update(args)

    out = capsys.readouterr().out
    assert "pid 5555" in out
    assert "serve" in out
    assert "pre-update code" in out


def test_clean_update_escalates_surviving_serve_as_unaccounted(
    monkeypatch, tmp_path, capsys
):
    """#100479 end to end: the plan inventoried a gateway (restarted through
    ``hermes-gateway.service``) and an unmanaged ``serve`` on the same
    default profile. The serve survives the update as the SAME process, so
    the update must (1) warn, (2) reconcile it as ``unaccounted`` instead of
    borrowing the gateway's restart, and (3) exit 1 with a ``partial``
    receipt — not print a clean success."""
    from hermes_cli.update_inventory import (
        RuntimeRecord, UpdatePlan, _restart_mechanism,
    )
    import hermes_cli.update_inventory as ui

    args = _update_args()
    _patch_update_deps(monkeypatch, tmp_path, _make_head_moved_side_effect())

    plan = UpdatePlan()
    plan.runtimes = [
        RuntimeRecord(kind="gateway", profile="default", pid=4444,
                      supervisor="systemd",
                      restart_via=_restart_mechanism("systemd", "default")),
        RuntimeRecord(kind="serve", profile="default", pid=5555,
                      supervisor="manual-serve",
                      restart_via=_restart_mechanism("manual-serve", "default"),
                      detail={"create_time": 1000.0}),
    ]
    monkeypatch.setattr(ui, "collect_runtime_inventory", lambda: plan)
    # The restart phase's own bookkeeping says the gateway unit restarted
    # (systemd branch is stubbed off in _patch_update_deps, so feed it here).
    real_match = ui.match_runtime_outcomes

    def _match(p, **kw):
        kw["restarted_services"] = list(kw.get("restarted_services") or []) + [
            "hermes-gateway.service"
        ]
        return real_match(p, **kw)

    monkeypatch.setattr(ui, "match_runtime_outcomes", _match)
    # Real survivor probe semantics against a fake ledger: pid 5555 is still
    # the same incarnation the plan recorded.
    import hermes_cli.process_identity as pi

    monkeypatch.setattr(
        pi, "ledger_entries",
        lambda **_k: [{"pid": 5555, "purpose": "serve", "create_time": 1000.0}],
    )

    with pytest.raises(SystemExit) as excinfo:
        hermes_main.cmd_update(args)
    assert excinfo.value.code == 1

    out = capsys.readouterr().out
    assert "pid 5555" in out and "pre-update code" in out
    assert "Planned runtimes the restart phase never touched" in out
    assert "serve [default] pid 5555" in out

    latest = get_hermes_home() / "logs" / "update_receipts" / "latest.json"
    receipt = json.loads(latest.read_text(encoding="utf-8"))
    assert receipt["outcome"] == "partial"
    by_pid = {o["pid"]: o["outcome"] for o in receipt["runtime_outcomes"]}
    assert by_pid == {4444: "restarted", 5555: "unaccounted"}


def test_clean_update_defers_desktop_owned_serve_and_clears_marker(
    monkeypatch, tmp_path, capsys
):
    """#111494 end to end: the only survivor is the Desktop app's own ``serve``
    backend. The restart phase is forbidden to restart it, so reconciliation must
    not count it as a missed restart either — otherwise every update with the
    Desktop open ends ``partial``/exit 1 and re-arms ``fleet_restart_pending``
    with nothing that could ever discharge it. It is surfaced (``deferred``,
    relaunch hint) rather than dropped."""
    from hermes_cli.update_inventory import (
        RuntimeRecord, UpdatePlan, _restart_mechanism,
    )
    import hermes_cli.update_inventory as ui
    import hermes_cli.process_identity as pi

    args = _update_args()
    _patch_update_deps(monkeypatch, tmp_path, _make_head_moved_side_effect())

    plan = UpdatePlan()
    plan.runtimes = [
        RuntimeRecord(kind="gateway", profile="default", pid=4444,
                      supervisor="systemd",
                      restart_via=_restart_mechanism("systemd", "default")),
        RuntimeRecord(kind="serve", profile="default", pid=6161,
                      supervisor="desktop",
                      restart_via=_restart_mechanism("desktop", "default"),
                      detail={"create_time": 1000.0}),
    ]
    monkeypatch.setattr(ui, "collect_runtime_inventory", lambda: plan)
    real_match = ui.match_runtime_outcomes

    def _match(p, **kw):
        kw["restarted_services"] = list(kw.get("restarted_services") or []) + [
            "hermes-gateway.service"
        ]
        return real_match(p, **kw)

    monkeypatch.setattr(ui, "match_runtime_outcomes", _match)
    # The gateway leg is healthy on the new code; only the Desktop serve is left.
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda **_k: [{"profile": "default", "pid": 4444, "code_sha": "def456",
                       "code_version": "0.21.0", "state": "current"}],
    )
    # Same incarnation still alive: the Desktop serve genuinely survived on pre-update code.
    monkeypatch.setattr(
        pi, "ledger_entries",
        lambda **_k: [{"pid": 6161, "purpose": "serve", "create_time": 1000.0}],
    )

    hermes_main.cmd_update(args)  # no SystemExit(1)

    out = capsys.readouterr().out
    assert "pid 6161" in out and "pre-update code" in out
    assert "relaunch the Desktop app" in out
    assert "Planned runtimes the restart phase never touched" not in out
    assert not update_cmd_fleet._fleet_restart_obligation_armed()

    latest = get_hermes_home() / "logs" / "update_receipts" / "latest.json"
    receipt = json.loads(latest.read_text(encoding="utf-8"))
    assert receipt["outcome"] == "success"
    by_pid = {o["pid"]: o["outcome"] for o in receipt["runtime_outcomes"]}
    assert by_pid == {4444: "restarted", 6161: "deferred"}


def test_interrupt_between_pull_and_restart_leaves_marker(
    monkeypatch, tmp_path
):
    args = _update_args()
    _patch_update_deps(monkeypatch, tmp_path, _make_head_moved_side_effect())

    def _interrupt(*_a, **_k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", _interrupt)

    with pytest.raises(KeyboardInterrupt):
        hermes_main.cmd_update(args)

    assert update_cmd_fleet._fleet_restart_obligation_armed()
    record = json.loads(host_obligation.host_obligation_path().read_text(encoding="utf-8"))
    assert record["expected_sha"] == "def456"


def test_already_up_to_date_runs_pending_restart_when_marker_present(
    monkeypatch, tmp_path, capsys
):
    args = _update_args()
    _patch_update_deps(monkeypatch, tmp_path, _make_up_to_date_side_effect())
    monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: "abc123")
    update_cmd._write_fleet_restart_pending_marker(expected_sha="abc123", runtimes=[{"kind": "gateway", "profile": "default"}])

    seen = {"ran": False}
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda **k: [{"profile": "default", "state": "current", "code_sha": "abc123"}] if seen["ran"] else [],
    )

    def _restart():
        seen["ran"] = True
        return True

    monkeypatch.setattr(update_cmd, "_run_pending_fleet_restart", _restart)
    monkeypatch.setattr(update_cmd_fleet, "_run_pending_fleet_restart", _restart)

    hermes_main.cmd_update(args)

    assert seen["ran"] is True
    assert not update_cmd_fleet._fleet_restart_obligation_armed()
    out = capsys.readouterr().out
    assert "did not restart running gateways" in out


def test_already_up_to_date_runs_pending_restart_when_receipt_skewed(
    monkeypatch, tmp_path, capsys
):
    args = _update_args()
    _patch_update_deps(monkeypatch, tmp_path, _make_up_to_date_side_effect())

    disk_sha = "e" * 40
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: disk_sha)
    monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: disk_sha)
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(
        json.dumps(
            {
                "exit_code": 1,
                "stop_reason": "KeyboardInterrupt: ",
                "outcome": "failed",
                "plan": {
                    "expected_sha": disk_sha,
                    "runtimes": [
                        {
                            "kind": "gateway",
                            "profile": "default",
                            "pid": 42,
                            "code_sha": "7" * 40,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    seen = {"ran": False}
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda **k: [{"profile": "default", "state": "current", "code_sha": disk_sha}] if seen["ran"] else [],
    )
    monkeypatch.setattr(
        update_cmd,
        "_run_pending_fleet_restart",
        lambda: seen.__setitem__("ran", True) or True,
    )
    monkeypatch.setattr(
        update_cmd_fleet,
        "_run_pending_fleet_restart",
        lambda: seen.__setitem__("ran", True) or True,
    )

    hermes_main.cmd_update(args)

    assert seen["ran"] is True
    out = capsys.readouterr().out
    assert "did not restart running gateways" in out


def test_already_up_to_date_skips_restart_when_nothing_pending(
    monkeypatch, tmp_path, capsys
):
    args = _update_args()
    _patch_update_deps(monkeypatch, tmp_path, _make_up_to_date_side_effect())

    seen = {"ran": False}
    monkeypatch.setattr(
        update_cmd,
        "_run_pending_fleet_restart",
        lambda: seen.__setitem__("ran", True) or True,
    )
    monkeypatch.setattr(
        update_cmd_fleet,
        "_run_pending_fleet_restart",
        lambda: seen.__setitem__("ran", True) or True,
    )

    hermes_main.cmd_update(args)

    assert seen["ran"] is False
    assert "did not restart running gateways" not in capsys.readouterr().out


def test_startup_warn_prints_when_marker_present(capsys):
    update_cmd._write_fleet_restart_pending_marker()
    update_cmd._warn_pending_fleet_restart_on_startup()
    err = capsys.readouterr().err
    assert "did not restart running gateways" in err
    assert "hermes gateway restart" in err


def test_startup_warn_silent_when_nothing_pending(capsys):
    update_cmd._warn_pending_fleet_restart_on_startup()
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


# ── Self-heal: marker left behind by a supervisor-level restart (#105417 / #111272) ──
#
# `systemctl --user restart hermes-gateway` never runs this module's clear path, and an update
# whose fleet probe answered empty exits before clearing — so the marker survives a restart
# that DID bring the fleet to the pulled code, and every later CLI call warns forever. The
# marker is discharged when (and only when) the fleet provably serves expected_sha.


def _patch_marker_sha(monkeypatch, disk_sha):
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: disk_sha)
    monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: disk_sha)


def test_startup_warn_discharged_when_fleet_current(monkeypatch, capsys):
    disk_sha = "e" * 40
    update_cmd._write_fleet_restart_pending_marker(expected_sha=disk_sha, runtimes=[{"kind": "gateway", "profile": "default"}])
    _patch_marker_sha(monkeypatch, disk_sha)
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda **kwargs: [
            {"profile": "default", "pid": 42, "code_sha": disk_sha, "code_version": "0.21.0", "state": "current"}
        ],
    )

    update_cmd._warn_pending_fleet_restart_on_startup()

    assert capsys.readouterr().err == ""
    assert not update_cmd_fleet._fleet_restart_obligation_armed()


def test_startup_warn_discharged_when_multiplexer_covers_owed_profiles(monkeypatch, capsys):
    """A current multiplexer discharges every profile named in its live record (#113350)."""
    disk_sha = "e" * 40
    # The marker owns its inventory (two gateways owed); an inventory-less marker
    # discharges on live-fleet evidence alone (#115638).
    update_cmd._write_fleet_restart_pending_marker(
        expected_sha=disk_sha,
        runtimes=[{"kind": "gateway", "profile": p} for p in ("default", "coder")],
    )
    _patch_marker_sha(monkeypatch, disk_sha)
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(
        json.dumps(
            {
                "outcome": "partial",
                "exit_code": 1,
                "plan": {
                    "runtimes": [
                        {"kind": "gateway", "profile": profile, "pid": pid}
                        for profile, pid in (("default", 42), ("coder", 43))
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda **kwargs: [
            {
                "profile": "default",
                "pid": 42,
                "code_sha": disk_sha,
                "code_version": "0.21.0",
                "state": "current",
                "served_profiles": ["default", "coder"],
            }
        ],
    )

    update_cmd._warn_pending_fleet_restart_on_startup()

    assert capsys.readouterr().err == ""
    assert not update_cmd_fleet._fleet_restart_obligation_armed()
    # The same live multiplexer coverage also discharges the receipt fallback
    # after an operator has already removed the marker.
    assert update_cmd._pending_fleet_restart_needed() is False


@pytest.mark.parametrize("supervisor", ["desktop", "launchd"])
def test_startup_warn_discharged_when_inventory_holds_supervised_serve(monkeypatch, capsys, supervisor):
    """A supervised serve/dashboard row in the marker's inventory is its supervisor's to
    restart (#115090 for receipts, #111494 for the Desktop backend) — it must not make the
    gateway warning permanently undischargeable once every gateway serves the pulled SHA.

    Only ``desktop`` and ``launchd`` are parametrized: those are the two supervisor values the
    inventory writer can actually put on a serve/dashboard row (``update_inventory``'s ledger
    pass emits exactly launchd, desktop or manual-serve). The systemd/windows-service/service
    members of ``_SUPERVISOR_OWNED_SERVE_BACKENDS`` only ever appear on gateway rows.
    """
    disk_sha = "e" * 40
    update_cmd._write_fleet_restart_pending_marker(
        expected_sha=disk_sha,
        runtimes=[
            {"kind": "gateway", "profile": "default", "pid": 42, "supervisor": "systemd"},
            {"kind": "serve", "profile": "default", "pid": 6161, "supervisor": supervisor,
             "detail": {"create_time": 1000.0}},
        ],
    )
    _patch_marker_sha(monkeypatch, disk_sha)
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda **kwargs: [
            {"profile": "default", "pid": 42, "code_sha": disk_sha, "code_version": "0.21.0", "state": "current"}
        ],
    )

    update_cmd._warn_pending_fleet_restart_on_startup()

    assert capsys.readouterr().err == ""
    assert not update_cmd_fleet._fleet_restart_obligation_armed()


def test_startup_warn_kept_when_inventory_holds_unclassified_serve(monkeypatch, capsys):
    """Fail-closed stays: a serve row no supervisor owns is still unsettled evidence."""
    disk_sha = "e" * 40
    update_cmd._write_fleet_restart_pending_marker(
        expected_sha=disk_sha,
        runtimes=[
            {"kind": "gateway", "profile": "default", "pid": 42, "supervisor": "systemd"},
            {"kind": "serve", "profile": "default", "pid": 6161, "supervisor": "manual"},
        ],
    )
    _patch_marker_sha(monkeypatch, disk_sha)
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda **kwargs: [
            {"profile": "default", "pid": 42, "code_sha": disk_sha, "code_version": "0.21.0", "state": "current"}
        ],
    )

    update_cmd._warn_pending_fleet_restart_on_startup()

    assert "did not restart running gateways" in capsys.readouterr().err
    assert update_cmd_fleet._fleet_restart_obligation_armed()


@pytest.mark.parametrize(
    "disk_sha, fleet",
    [
        ("e" * 40, [{"profile": "default", "pid": 42, "code_sha": "7" * 40, "code_version": "0.20.0", "state": "stale"}]),
        ("e" * 40, []),  # probe answered empty: no proof either way
        ("e" * 40, [{"profile": "default", "pid": 42, "code_sha": None, "code_version": None, "state": "unknown"}]),
        (None, [{"profile": "default", "code_sha": "e" * 40, "state": "current"}]),
        # checkout advanced past the marker: a newer pull owns a fresh obligation
        ("f" * 40, [{"profile": "default", "pid": 42, "code_sha": "e" * 40, "code_version": None, "state": "current"}]),
    ],
    ids=["stale-row", "empty-probe", "unknown-identity", "unknown-checkout", "checkout-moved"],
)
def test_startup_warn_kept_without_positive_evidence(monkeypatch, capsys, disk_sha, fleet):
    update_cmd._write_fleet_restart_pending_marker(expected_sha="e" * 40, runtimes=[{"kind": "gateway", "profile": "default"}])
    _patch_marker_sha(monkeypatch, disk_sha)
    monkeypatch.setattr("hermes_cli.update_receipt.collect_fleet_versions", lambda **kwargs: fleet)

    update_cmd._warn_pending_fleet_restart_on_startup()

    assert "did not restart running gateways" in capsys.readouterr().err
    assert update_cmd_fleet._fleet_restart_obligation_armed()


def test_startup_warn_kept_when_receipt_owed_gateway_is_down(monkeypatch, capsys):
    """A sibling the restart phase killed yields no startup row; the marker still owns it."""
    disk_sha = "e" * 40
    update_cmd._write_fleet_restart_pending_marker(expected_sha=disk_sha, runtimes=[{"kind": "gateway", "profile": p} for p in ("alpha", "beta")])
    _patch_marker_sha(monkeypatch, disk_sha)
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(
        json.dumps(
            {
                "outcome": "partial",
                "exit_code": 1,
                "plan": {"runtimes": [{"kind": "gateway", "profile": p, "code_sha": "o" * 40, "pid": 1} for p in ("alpha", "beta")]},
                "fleet": [
                    {"profile": "alpha", "pid": 42, "code_sha": disk_sha, "state": "current"},
                    {"profile": "beta", "pid": 43, "code_sha": None, "state": "down"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda **kwargs: [
            {"profile": "alpha", "pid": 42, "code_sha": disk_sha, "code_version": "0.21.0", "state": "current"}
        ],
    )

    update_cmd._warn_pending_fleet_restart_on_startup()

    assert "did not restart running gateways" in capsys.readouterr().err
    assert update_cmd_fleet._fleet_restart_obligation_armed()


def test_startup_warn_silent_when_failed_receipt_already_restarted_fleet(monkeypatch, capsys):
    """#112604 aftermath: the update pulled ``pulled``, restarted every gateway onto it, then a
    post-restart step crashed (receipt ``failed``, empty ``fleet`` matrix). Later a manual
    ``git pull`` moved the checkout again. The startup hint must not blame that update for a
    restart it performed; ``hermes update``'s catch-up still owes the fleet the checkout."""
    pre, pulled, checkout = "a" * 40, "b" * 40, "c" * 40
    _patch_marker_sha(monkeypatch, checkout)
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(
        json.dumps(
            {
                "outcome": "failed", "exit_code": 1,
                "stop_reason": "AttributeError: module 'hermes_cli.main_dashboard' has no attribute 'x'",
                "pre_update": {"sha": pre}, "post_update": {"sha": pulled},
                "gateway_restart": {
                    "restarted_services": ["hermes-gateway"], "relaunched_profiles": [],
                    "externally_supervised_profiles": [], "killed_pids": [], "failed_units": [],
                    "incomplete": False, "phase_error": "",
                },
                "fleet": [],
                "plan": {"runtimes": [{"kind": "gateway", "profile": "default", "code_sha": pre, "pid": 1}]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda **kwargs: [
            {"profile": "default", "pid": 42, "code_sha": pulled, "code_version": "0.21.3", "state": "stale"}
        ],
    )

    update_cmd._warn_pending_fleet_restart_on_startup()

    assert capsys.readouterr().err == ""
    assert update_cmd_fleet._pending_fleet_restart_needed() is True


def test_startup_warn_silent_when_completed_update_fleet_restarted_onto_moved_checkout(monkeypatch, capsys):
    """The remedy the warning names must clear it: after a completed update, a manual ``git pull``
    plus ``hermes gateway restart`` leaves every owed gateway on today's checkout — newer than the
    update's ``post_update.sha`` — which is nothing that update still owes (#113350 steps 3–4)."""
    pre, pulled, checkout = "a" * 40, "b" * 40, "c" * 40
    _patch_marker_sha(monkeypatch, checkout)
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(
        json.dumps(
            {
                "outcome": "success", "exit_code": 0,
                "pre_update": {"sha": pre}, "post_update": {"sha": pulled},
                "gateway_restart": {
                    "restarted_services": ["hermes-gateway"], "relaunched_profiles": [],
                    "externally_supervised_profiles": [], "killed_pids": [], "failed_units": [],
                    "incomplete": False, "phase_error": "",
                },
                "fleet": [{"profile": "default", "pid": 7, "code_sha": pulled, "state": "current"}],
                "plan": {"runtimes": [{"kind": "gateway", "profile": "default", "code_sha": pre, "pid": 1}]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda **kwargs: [
            {"profile": "default", "pid": 42, "code_sha": checkout, "code_version": "0.21.3", "state": "current"}
        ],
    )

    update_cmd._warn_pending_fleet_restart_on_startup()

    assert capsys.readouterr().err == ""


# ── Inventory-less markers discharge on live-fleet evidence (#115638) ──
#
# A marker written without an inventory line (legacy markers, or a pull that raced
# a missing pre-update plan) records no owed set, so the inventory check stays
# fail-closed forever and every CLI call warns. With no recorded obligation the
# marker discharges when the live fleet provably serves expected_sha — same
# evidence bar as an inventoried marker, minus the owed-coverage check.


def test_startup_warn_discharged_when_inventory_less_marker_fleet_current(monkeypatch, capsys):
    disk_sha = "e" * 40
    update_cmd._write_fleet_restart_pending_marker(expected_sha=disk_sha)
    assert "inventory" not in host_obligation.read_host_obligation()
    _patch_marker_sha(monkeypatch, disk_sha)
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda **kwargs: [
            {"profile": "default", "pid": 42, "code_sha": disk_sha, "code_version": "0.21.0", "state": "current"}
        ],
    )

    update_cmd._warn_pending_fleet_restart_on_startup()

    assert capsys.readouterr().err == ""
    assert not update_cmd_fleet._fleet_restart_obligation_armed()


def test_startup_warn_kept_when_inventory_less_marker_fleet_stale(monkeypatch, capsys):
    disk_sha = "e" * 40
    update_cmd._write_fleet_restart_pending_marker(expected_sha=disk_sha)
    _patch_marker_sha(monkeypatch, disk_sha)
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda **kwargs: [
            {"profile": "default", "pid": 42, "code_sha": "7" * 40, "code_version": "0.20.0", "state": "stale"}
        ],
    )

    update_cmd._warn_pending_fleet_restart_on_startup()

    assert "did not restart running gateways" in capsys.readouterr().err
    assert update_cmd_fleet._fleet_restart_obligation_armed()

# ── Empty-inventory marker: a pull that recorded no gateway owes nothing (#115311) ──

def _write_marker_with_inventory(expected_sha, runtimes):
    marker = update_cmd_fleet._fleet_restart_pending_marker_path()
    marker.write_text(
        f"started=0\npid=1\nexpected_sha={expected_sha}\n"
        f"inventory={json.dumps({'version': 1, 'runtimes': runtimes})}\n",
        encoding="utf-8",
    )
    return marker


def test_empty_inventory_does_not_arm_marker():
    """A pre-update plan with zero runtimes must never arm the marker: on a no-gateway
    (Desktop-hosted) install every later update would otherwise hit the unbeatable
    'Fleet restart incomplete' exit 1 (#115311)."""
    update_cmd._write_fleet_restart_pending_marker(expected_sha="e" * 40, runtimes=[])
    assert not update_cmd_fleet._fleet_restart_obligation_armed()


def test_pending_fleet_restart_cleared_instead_of_exit_1(monkeypatch, tmp_path):
    """Repro: an already-up-to-date host carrying an empty-inventory marker must exit 0 with
    no restart run — not print 'Fleet restart incomplete' and exit 1 (#115311)."""
    args = _update_args()
    _patch_update_deps(monkeypatch, tmp_path, _make_up_to_date_side_effect())
    marker = _write_marker_with_inventory("abc123", [])

    seen = {"ran": False}
    monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: "abc123")
    monkeypatch.setattr(
        update_cmd,
        "_run_pending_fleet_restart",
        lambda: seen.__setitem__("ran", True) or True,
    )
    monkeypatch.setattr(
        update_cmd_fleet,
        "_run_pending_fleet_restart",
        lambda: seen.__setitem__("ran", True) or True,
    )

    hermes_main.cmd_update(args)

    assert seen["ran"] is False
    assert not marker.exists()


def test_catchup_settles_failed_receipt_from_live_fleet_instead_of_exit_1(monkeypatch, capsys):
    """A failed receipt whose plan rows carry pre-pull SHAs and an unidentifiable profile can never
    be matched to a live gateway, so every up-to-date `hermes update` ran the restart, printed
    "completed" then "incomplete", exited 1 and wrote another failed receipt — even after a manual
    `hermes gateway restart` put the fleet on the checkout code (#117051). With every live row
    current at the checkout SHA, the catch-up settles the receipt from that matrix and exits 0."""
    sha = "d" * 40
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: sha)
    monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: sha)
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    latest = receipt_dir / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "outcome": "failed",
                "exit_code": 1,
                "stop_reason": "sys.exit(1)",
                "gateway_restart": {},
                "fleet": [],
                "plan": {
                    "expected_sha": sha,
                    "runtimes": [{"kind": "gateway", "profile": "unknown", "pid": 42, "code_sha": "0" * 40}],
                },
            }
        ),
        encoding="utf-8",
    )
    live = [{"profile": "default", "pid": 48096, "code_sha": sha, "state": "current"}]
    monkeypatch.setattr("hermes_cli.update_receipt.collect_fleet_versions", lambda **k: list(live))
    monkeypatch.setattr(update_cmd, "_run_pending_fleet_restart", lambda: True)
    assert update_cmd._pending_fleet_restart_needed() is True

    update_cmd._apply_pending_fleet_restart_catchup()

    out = capsys.readouterr().out
    assert "incomplete" not in out and "still off the checkout code" not in out
    settled = json.loads(latest.read_text(encoding="utf-8"))
    assert settled["fleet"] == live
    assert settled["gateway_restart"]["incomplete"] is False
    assert update_cmd._pending_fleet_restart_needed() is False
    assert update_cmd_fleet._update_owes_fleet_restart() is False
