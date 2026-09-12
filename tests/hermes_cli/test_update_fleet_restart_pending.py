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
import os
import time
from types import SimpleNamespace

import pytest

from hermes_cli import main as hermes_main
import hermes_cli.main_web_build as main_web_build
import hermes_cli.main_install_repair as main_install_repair
from hermes_cli import update_cmd
import hermes_cli.update_cmd_fleet as update_cmd_fleet
from hermes_cli import update_receipt
import hermes_cli.update_cmd_deps as update_cmd_deps
from hermes_cli.update_receipt import COMMAND_BOUNDARY_STOP_REASON
from hermes_constants import get_hermes_home


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
    monkeypatch.setattr(update_cmd, "_purge_stale_hermes_modules", lambda: None)
    monkeypatch.setattr(hermes_main, "_purge_stale_hermes_modules", lambda: None)

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


def test_marker_round_trip_under_hermes_home():
    path = update_cmd._fleet_restart_pending_marker_path()
    assert path.parent == get_hermes_home()
    assert path.name == "fleet_restart_pending"
    assert not path.exists()

    update_cmd._write_fleet_restart_pending_marker(expected_sha="abc123")
    assert path.is_file()
    body = path.read_text(encoding="utf-8")
    assert "started=" in body
    assert "pid=" in body
    assert "expected_sha=abc123" in body

    update_cmd._clear_fleet_restart_pending_marker()
    assert not path.exists()


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
    monkeypatch.setattr(hermes_main, "_purge_stale_hermes_modules", lambda: None)

    # An empty PID scan is insufficient; both supervisor scopes must answer empty.
    monkeypatch.setattr(update_cmd_fleet, "_systemd_gateway_unit_listings", lambda: [
        (scope, cmd, SimpleNamespace(returncode=0, stdout=""))
        for scope, cmd in update_cmd_fleet._SYSTEMD_SCOPES
    ])
    assert update_cmd._run_pending_fleet_restart() is True
    assert "Pending fleet restart completed" in capsys.readouterr().out


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

    def _spy(*, expected_sha=""):
        orig(expected_sha=expected_sha)
        wrote.append(update_cmd._fleet_restart_pending_marker_path().is_file())

    monkeypatch.setattr(update_cmd, "_write_fleet_restart_pending_marker", _spy)

    hermes_main.cmd_update(args)

    assert wrote == [True], "marker must exist immediately after HEAD advances"
    assert not update_cmd._fleet_restart_pending_marker_path().exists()
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

    marker = update_cmd._fleet_restart_pending_marker_path()
    assert marker.is_file()
    assert "expected_sha=def456" in marker.read_text(encoding="utf-8")


def test_already_up_to_date_runs_pending_restart_when_marker_present(
    monkeypatch, tmp_path, capsys
):
    args = _update_args()
    _patch_update_deps(monkeypatch, tmp_path, _make_up_to_date_side_effect())
    update_cmd._write_fleet_restart_pending_marker(expected_sha="def456")

    seen = {"ran": False}

    def _restart():
        seen["ran"] = True
        return True

    monkeypatch.setattr(update_cmd, "_run_pending_fleet_restart", _restart)
    monkeypatch.setattr(update_cmd_fleet, "_run_pending_fleet_restart", _restart)

    hermes_main.cmd_update(args)

    assert seen["ran"] is True
    assert not update_cmd._fleet_restart_pending_marker_path().exists()
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


# ---------------------------------------------------------------------------
# Lease-with-verification: a stranded marker must yield to live fleet evidence.
# Markers are written DIRECTLY (bypassing the writer) so their pid is dead — the
# "update in flight" state; the writer-alive guard is exercised separately.
# ---------------------------------------------------------------------------

_DEAD_PID = 999_999_998


def _stranded_marker(expected_sha: str, *, age_seconds: float = 0.0) -> None:
    path = update_cmd._fleet_restart_pending_marker_path()
    started = time.time() - age_seconds
    path.write_text(
        f"started={started}\npid={_DEAD_PID}\nexpected_sha={expected_sha}\n", encoding="utf-8"
    )


def test_marker_with_live_fleet_at_expected_sha_discharges(monkeypatch):
    _stranded_marker("a" * 40)
    monkeypatch.setattr(
        update_receipt, "collect_fleet_versions",
        lambda: [{"profile": "neo", "pid": 5, "state": "current", "code_sha": "a" * 40}],
    )
    try:
        assert update_cmd_fleet._pending_fleet_restart_needed() is False
        assert not update_cmd._fleet_restart_pending_marker_path().exists()
    finally:
        update_cmd._clear_fleet_restart_pending_marker()


def test_dead_pid_marker_with_current_fleet_discharges(monkeypatch):
    _stranded_marker("b" * 40, age_seconds=26 * 86400)  # the 26-day neo shape
    monkeypatch.setattr(
        update_receipt, "collect_fleet_versions",
        lambda: [
            {"profile": "herc", "pid": 6, "state": "current", "code_sha": "b" * 40},
            {"profile": "nvx", "pid": 7, "state": "current", "code_sha": "b" * 40},
        ],
    )
    try:
        assert update_cmd_fleet._pending_fleet_restart_needed() is False
    finally:
        update_cmd._clear_fleet_restart_pending_marker()


def test_marker_with_stale_fleet_still_pends(monkeypatch):
    _stranded_marker("c" * 40)
    monkeypatch.setattr(
        update_receipt, "collect_fleet_versions",
        lambda: [{"profile": "neo", "pid": 8, "state": "stale", "code_sha": "old"}],
    )
    try:
        assert update_cmd_fleet._pending_fleet_restart_needed() is True
        assert update_cmd._fleet_restart_pending_marker_path().is_file()  # fail-closed: kept
    finally:
        update_cmd._clear_fleet_restart_pending_marker()


def test_marker_with_unprobeable_fleet_pends(monkeypatch):
    _stranded_marker("d" * 40)
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda: [])
    try:
        assert update_cmd_fleet._pending_fleet_restart_needed() is True  # no evidence ≠ discharge
        assert update_cmd._fleet_restart_pending_marker_path().is_file()
    finally:
        update_cmd._clear_fleet_restart_pending_marker()


def test_marker_writer_alive_keeps_pending():
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text(
        f"started={time.time()}\npid={os.getpid()}\nexpected_sha={'e' * 40}\n", encoding="utf-8"
    )
    try:
        assert update_cmd_fleet._pending_fleet_restart_needed() is True  # update in flight: don't race
        assert marker.is_file()
    finally:
        update_cmd._clear_fleet_restart_pending_marker()


def test_marker_without_expected_sha_discharges_only_via_age_ceiling(monkeypatch):
    path = update_cmd._fleet_restart_pending_marker_path()
    path.write_text(f"started={time.time()}\npid={_DEAD_PID}\n", encoding="utf-8")
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda: [])
    try:
        # young unverifiable marker: keep warning
        assert update_cmd_fleet._pending_fleet_restart_needed() is True
        assert path.is_file()
        # past the 30-day ceiling: clear
        path.write_text(
            f"started={time.time() - 31 * 86400}\npid={_DEAD_PID}\n", encoding="utf-8"
        )
        assert update_cmd_fleet._pending_fleet_restart_needed() is False
        assert not path.exists()
    finally:
        if path.exists():
            update_cmd._clear_fleet_restart_pending_marker()


def test_catchup_is_noop_when_fleet_already_current(monkeypatch):
    _stranded_marker("f" * 40)
    monkeypatch.setattr(
        update_receipt, "collect_fleet_versions",
        lambda: [{"profile": "neo", "pid": 9, "state": "current", "code_sha": "f" * 40}],
    )
    called = {"ran": False}

    def _must_not_run():
        called["ran"] = True
        return True

    monkeypatch.setattr(update_cmd, "_run_pending_fleet_restart", _must_not_run)
    monkeypatch.setattr(update_cmd_fleet, "_run_pending_fleet_restart", _must_not_run)
    update_cmd_fleet._apply_pending_fleet_restart_catchup()
    assert called["ran"] is False
    assert not update_cmd._fleet_restart_pending_marker_path().exists()


def test_startup_warn_silent_when_fleet_current_despite_marker(monkeypatch, capsys):
    _stranded_marker("1" * 40)
    monkeypatch.setattr(
        update_receipt, "collect_fleet_versions",
        lambda: [{"profile": "xbook", "pid": 10, "state": "current", "code_sha": "1" * 40}],
    )
    try:
        update_cmd._warn_pending_fleet_restart_on_startup()
        captured = capsys.readouterr()
        assert "did not restart running gateways" not in captured.err
        assert not update_cmd._fleet_restart_pending_marker_path().exists()
    finally:
        if update_cmd._fleet_restart_pending_marker_path().exists():
            update_cmd._clear_fleet_restart_pending_marker()


# ---------------------------------------------------------------------------
# Pre-commit bundle (FUP-20260910-11): A1 partial-probe fail-open, M1 stale-vs-
# unknown forever-pin, atomic write, pid+writer_start pairing, 7d ceiling.
# ---------------------------------------------------------------------------

def _fleet_rows(*specs):
    """Build fleet rows: (state, code_sha) tuples -> row dicts."""
    return [
        {"profile": f"p{i}", "pid": 100 + i, "state": state, "code_sha": sha}
        for i, (state, sha) in enumerate(specs)
    ]


def test_probe_abort_midloop_never_clears_marker(monkeypatch, capsys):
    """T1 (A1 fault-injection, MANDATORY): a sweep that raises on profile k of N returns
    None; verdict is unknown; the marker is NOT cleared AND the warning persists."""
    _stranded_marker("a" * 40)
    calls = {"n": 0}

    def _aborting_probe():
        # A1 fault-injection at the collect contract layer: an aborted sweep returns
        # None (per-profile isolation catches the raise inside collect_fleet_versions
        # and converts it to None) — NEVER partial rows, even when the rows gathered
        # before the abort all match expected_sha. First call exercises the None
        # contract; later calls raise outright to prove the except path too.
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("profile k exploded mid-sweep")
        return None  # aborted sweep — even though rows seen before the abort matched

    monkeypatch.setattr(update_receipt, "collect_fleet_versions", _aborting_probe)
    try:
        # Not cleared...
        assert update_cmd_fleet._pending_fleet_restart_needed() is True
        assert update_cmd._fleet_restart_pending_marker_path().is_file()
        # ...and still warning (fail-closed = not cleared AND still warning).
        update_cmd._warn_pending_fleet_restart_on_startup()
        assert "did not restart running gateways" in capsys.readouterr().err
    finally:
        update_cmd._clear_fleet_restart_pending_marker()

def test_probe_error_does_not_start_grace_clock(monkeypatch):
    """T2 (neo's caution): consecutive probe errors past the idle-grace window do NOT
    clear — errors route to the age ceiling only, never the grace path."""
    _stranded_marker("b" * 40, age_seconds=3600)  # 1h old: way past the 600s grace
    monkeypatch.setattr(
        update_receipt, "collect_fleet_versions",
        lambda: None,  # aborted sweep (None), not a clean empty sweep
    )
    try:
        assert update_cmd_fleet._pending_fleet_restart_needed() is True
        assert update_cmd._fleet_restart_pending_marker_path().is_file()
    finally:
        update_cmd._clear_fleet_restart_pending_marker()


def test_unknown_sha_gateway_clears_at_age_ceiling_not_forever(monkeypatch):
    """T3 (M1): a legacy pre-stamp gateway (state=unknown / code_sha=None) yields verdict
    unknown, which routes to the age ceiling — the marker does NOT pin forever."""
    _stranded_marker("c" * 40, age_seconds=1 * 3600)
    monkeypatch.setattr(
        update_receipt, "collect_fleet_versions",
        lambda: [{"profile": "legacy", "pid": 11, "state": "unknown", "code_sha": None}],
    )
    path = update_cmd._fleet_restart_pending_marker_path()
    try:
        # young unverifiable: keep warning (fail-closed)
        assert update_cmd_fleet._pending_fleet_restart_needed() is True
        assert path.is_file()
        # past the 7d ceiling: clear (not forever)
        path.write_text(
            f"started={time.time() - 8 * 86400}\npid={_DEAD_PID}\nexpected_sha={'c' * 40}\n",
            encoding="utf-8",
        )
        assert update_cmd_fleet._pending_fleet_restart_needed() is False
        assert not path.exists()
    finally:
        if path.exists():
            update_cmd._clear_fleet_restart_pending_marker()


def test_stale_with_present_sha_keeps_pending_indefinitely(monkeypatch):
    """T4 (M1 inverse): a PROVABLY stale row (code_sha present && != expected) holds the
    marker past ANY age — the cure is a restart, not time."""
    _stranded_marker("d" * 40, age_seconds=100 * 86400)
    monkeypatch.setattr(
        update_receipt, "collect_fleet_versions",
        lambda: [{"profile": "neo", "pid": 12, "state": "stale", "code_sha": "olds" + "h" * 36}],
    )
    try:
        assert update_cmd_fleet._pending_fleet_restart_needed() is True
        assert update_cmd._fleet_restart_pending_marker_path().is_file()
    finally:
        update_cmd._clear_fleet_restart_pending_marker()


def test_recycled_writer_pid_does_not_block_reconcile(monkeypatch):
    """T5 (Change 5): a live pid whose process start time DIFFERS from the recorded
    writer_start is a recycled PID, not the writer — reconciliation proceeds."""
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text(
        f"started={time.time()}\npid={os.getpid()}\nwriter_start=1234567890.0\n"
        f"expected_sha={'e' * 40}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        update_cmd_fleet, "_fleet_restart_verdict", lambda expected: "match"
    )
    try:
        assert update_cmd_fleet._marker_writer_alive(
            {"pid": str(os.getpid()), "writer_start": "1234567890.0"}
        ) is True  # our OWN pid still counts as in-flight regardless of pairing
        # A different live pid with a mismatched start time does not count as the writer.
        def _other_process_start(pid):
            return 9999999999.0 if pid == 987654 else None
        from gateway import status as gw_status
        monkeypatch.setattr(gw_status, "get_process_start_time", _other_process_start)
        monkeypatch.setattr(update_cmd_fleet.os, "kill", lambda pid, sig: None)  # pid "exists"
        assert update_cmd_fleet._marker_writer_alive(
            {"pid": "987654", "writer_start": "1234567890.0"}
        ) is False
    finally:
        update_cmd._clear_fleet_restart_pending_marker()


def test_writer_alive_with_matching_start_time_blocks(monkeypatch):
    """T6 (Change 5 inverse): a live pid with a MATCHING start time is the writer —
    write-lock holds, reconciliation is suppressed."""
    live_start = 1700000000.0
    from gateway import status as gw_status
    monkeypatch.setattr(gw_status, "get_process_start_time", lambda pid: live_start)
    monkeypatch.setattr(update_cmd_fleet.os, "kill", lambda pid, sig: None)
    assert update_cmd_fleet._marker_writer_alive(
        {"pid": "987654", "writer_start": str(live_start)}
    ) is True


def test_truncated_marker_write_is_atomic(monkeypatch):
    """T7 (M3): the writer uses temp+os.replace — no torn-marker parse window; the tmp
    file never survives a successful write."""
    update_cmd._write_fleet_restart_pending_marker(expected_sha="7" * 40)
    path = update_cmd._fleet_restart_pending_marker_path()
    try:
        assert path.is_file()
        assert not path.with_name(path.name + ".tmp").exists()
        fields = update_cmd_fleet._read_fleet_restart_pending_marker()
        assert fields.get("expected_sha") == "7" * 40
        assert fields.get("writer_start")  # pairing field present on fresh writes
    finally:
        update_cmd._clear_fleet_restart_pending_marker()


def test_unverifiable_write_is_loud_not_refused(monkeypatch, caplog):
    """Change 4 (wizred variant): expected_sha="" markers are still written (interrupt-
    recovery net) with a LOUD warning — never silently, never refused."""
    import logging
    with caplog.at_level(logging.WARNING, logger="hermes_cli.update_cmd"):
        update_cmd._write_fleet_restart_pending_marker(expected_sha="")
    path = update_cmd._fleet_restart_pending_marker_path()
    try:
        assert path.is_file()
        assert any("WITHOUT expected_sha" in rec.message for rec in caplog.records)
    finally:
        update_cmd._clear_fleet_restart_pending_marker()


def test_error_vs_idle_distinction(monkeypatch):
    """T8 (item 3 merged with A1): an aborted sweep with zero good rows -> verdict
    'unknown', NOT 'idle' — the grace window must not run on a broken probe."""
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda: None)
    assert update_cmd_fleet._fleet_restart_verdict("8" * 40) == "unknown"


def test_reconcile_contract_may_clear_marker():
    """T9 (M2 guard): the reconcile path is documented as a may-clear operation reached
    through the _pending_fleet_restart_needed predicate — pin the wiring so a future
    refactor cannot silently drop the side effect."""
    import inspect
    reconcile_doc = update_cmd_fleet._reconcile_fleet_restart_pending.__doc__ or ""
    assert "MAY CLEAR" in reconcile_doc
    # The predicate routes through the reconciler (the side-effecting path).
    src = inspect.getsource(update_cmd_fleet._pending_fleet_restart_needed)
    assert "_reconcile_fleet_restart_pending()" in src
