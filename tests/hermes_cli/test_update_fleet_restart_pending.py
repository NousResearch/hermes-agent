"""Interrupted-update fleet-restart obligation (#95294 parts 1+2).

A ``hermes update`` killed after git pull advanced HEAD but before the
fleet restart left running gateways on stale code. The next update said
"Already up to date" and skipped restart. These tests cover:

- ``fleet_restart_pending`` marker written after HEAD advances, then settled
  with generation-bound completion evidence after a successful restart
- interrupt between pull and restart leaves the marker
- next ``hermes update`` with git already up to date still runs the
  pending restart when the marker OR a skewed unfinished latest.json is
  present

No live gateway, no network. Git and restart are mocked.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from hermes_cli import main as hermes_main
import hermes_cli.main_web_build as main_web_build
import hermes_cli.main_install_repair as main_install_repair
from hermes_cli import update_cmd
import hermes_cli.update_cmd_fleet as update_cmd_fleet
import hermes_cli.update_cmd_deps as update_cmd_deps
from hermes_cli.update_receipt import COMMAND_BOUNDARY_STOP_REASON
import hermes_cli.update_receipt as update_receipt
from hermes_constants import get_hermes_home


_UPDATE_ID = "0123456789abcdef0123456789abcdef"


def _marker_body(*, started, expected_sha, pid=99999999, update_id=_UPDATE_ID):
    return f"started={started}\npid={pid}\nupdate_id={update_id}\nexpected_sha={expected_sha}\n"


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
        lambda: SimpleNamespace(runtimes=[], inventory_errors=[], to_dict=lambda: {}),
    )


def _update_args():
    return SimpleNamespace(branch=None, yes=False, force=False, force_venv=False)


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------


def test_marker_round_trip_under_hermes_home(monkeypatch):
    path = update_cmd._fleet_restart_pending_marker_path()
    assert path.parent == get_hermes_home()
    assert path.name == "fleet_restart_pending"
    assert not path.exists()

    monkeypatch.setattr(update_receipt, "current_update_id", lambda: _UPDATE_ID)
    monkeypatch.setattr(update_receipt, "checkpoint_update_receipt", lambda _sha: True)
    update_cmd._write_fleet_restart_pending_marker(expected_sha="abc123")
    assert path.is_file()
    body = path.read_text(encoding="utf-8")
    assert "started=" in body
    assert "pid=" in body
    assert f"update_id={_UPDATE_ID}" in body
    assert "expected_sha=abc123" in body

    update_cmd._clear_fleet_restart_pending_marker()
    assert not path.exists()


def test_pending_needed_when_marker_exists():
    update_cmd._write_fleet_restart_pending_marker()
    assert update_cmd._pending_fleet_restart_needed() is True
    update_cmd._clear_fleet_restart_pending_marker()
    assert update_cmd._pending_fleet_restart_needed() is False


def test_verified_successor_fulfills_exact_marker_generation(monkeypatch):
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
    ).strip()
    started = psutil.Process().create_time() - 1
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text(_marker_body(started=started, expected_sha=sha), encoding="utf-8")
    original = marker.read_bytes()
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(json.dumps({
        "pid": 99999999,
        "update_id": _UPDATE_ID,
        "post_update": {"sha": sha},
        "plan": {"inventory_complete": True, "runtimes": [
            {"kind": "gateway", "profile": "default", "pid": 99999998}
        ]},
    }), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [("default", get_hermes_home())])
    monkeypatch.setattr(
        "hermes_cli.update_receipt._socket_identity",
        lambda _home: (os.getpid(), {"profile": "default", "code_sha": sha}),
    )
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **_kwargs: [os.getpid()])
    monkeypatch.setattr("hermes_cli.process_identity.ledger_entries", lambda **_kwargs: [])
    monkeypatch.setattr("hermes_cli.dashboard_procs._scan_dashboard_processes", lambda **_kwargs: [])

    assert update_cmd._pending_fleet_restart_needed() is False
    assert marker.read_bytes() == original
    completed = json.loads(update_cmd_fleet._fleet_restart_completion_path().read_text(encoding="utf-8"))
    assert completed["expected_sha"] == sha
    assert completed["update_id"] == _UPDATE_ID

    marker.write_text(
        _marker_body(started=started + 1, pid=99999997, expected_sha=sha), encoding="utf-8"
    )
    assert update_cmd._pending_fleet_restart_needed() is True


def test_live_old_generation_prevents_successor_from_fulfilling_marker(monkeypatch):
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
    ).strip()
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text(
        _marker_body(started=psutil.Process().create_time() - 1, expected_sha=sha),
        encoding="utf-8",
    )
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(json.dumps({
        "pid": 99999999,
        "update_id": _UPDATE_ID,
        "post_update": {"sha": sha},
        "plan": {"inventory_complete": True, "runtimes": [
            {"kind": "gateway", "profile": "default", "pid": os.getpid()}
        ]},
    }), encoding="utf-8")
    successor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [("default", get_hermes_home())])
        monkeypatch.setattr(
            "hermes_cli.update_receipt._socket_identity",
            lambda _home: (successor.pid, {"profile": "default", "code_sha": sha}),
        )
        assert update_cmd._pending_fleet_restart_needed() is True
    finally:
        successor.terminate()
        successor.wait(timeout=10)


def test_vanished_successor_cannot_fulfill_marker(monkeypatch):
    marker = {"started": 1.0, "expected_sha": "abc123"}
    receipt = {
        "plan": {
            "inventory_complete": True,
            "runtimes": [{"kind": "gateway", "profile": "default", "pid": 41}],
        }
    }
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [("default", get_hermes_home())])
    monkeypatch.setattr(
        "hermes_cli.update_receipt._socket_identity",
        lambda _home: (42, {"profile": "default", "code_sha": "abc123"}),
    )
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **_kwargs: [42])
    monkeypatch.setattr("hermes_cli.process_identity.ledger_entries", lambda **_kwargs: [])
    monkeypatch.setattr("hermes_cli.dashboard_procs._scan_dashboard_processes", lambda **_kwargs: [])

    def vanished(_pid):
        raise psutil.NoSuchProcess(_pid)

    monkeypatch.setattr(psutil, "Process", vanished)

    assert update_cmd_fleet._marker_obligation_is_fulfilled(marker, receipt) is False


def test_reused_pid_and_target_sha_cannot_select_historical_receipt(monkeypatch):
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
    ).strip()
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text(_marker_body(started=psutil.Process().create_time() - 1, expected_sha=sha), encoding="utf-8")
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(json.dumps({
        "pid": 99999999,
        "update_id": "fedcba9876543210fedcba9876543210",
        "post_update": {"sha": sha},
        "plan": {"inventory_complete": True, "runtimes": [
            {"kind": "gateway", "profile": "default", "pid": 99999998}
        ]},
    }), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [("default", get_hermes_home())])
    monkeypatch.setattr(
        "hermes_cli.update_receipt._socket_identity",
        lambda _home: (os.getpid(), {"profile": "default", "code_sha": sha}),
    )

    assert update_cmd._pending_fleet_restart_needed() is True
    assert not update_cmd_fleet._fleet_restart_completion_path().exists()


def test_multiple_profiles_require_identity_matched_successors(monkeypatch):
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
    ).strip()
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text(_marker_body(started=psutil.Process().create_time() - 1, expected_sha=sha), encoding="utf-8")
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    receipt = {
        "pid": 99999999,
        "update_id": _UPDATE_ID,
        "post_update": {"sha": sha},
        "plan": {"inventory_complete": True, "runtimes": [
            {"kind": "gateway", "profile": "default", "pid": 99999998},
            {"kind": "gateway", "profile": "work", "pid": 99999997},
        ]},
    }
    (receipt_dir / "latest.json").write_text(json.dumps(receipt), encoding="utf-8")
    homes = [("default", get_hermes_home()), ("work", get_hermes_home() / "profiles" / "work")]
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: homes)
    monkeypatch.setattr(
        "hermes_cli.update_receipt._socket_identity",
        lambda home: (
            os.getpid(),
            {"profile": "default" if home == homes[0][1] else "work", "code_sha": sha},
        ),
    )
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **_kwargs: [os.getpid()])
    monkeypatch.setattr("hermes_cli.process_identity.ledger_entries", lambda **_kwargs: [])
    monkeypatch.setattr("hermes_cli.dashboard_procs._scan_dashboard_processes", lambda **_kwargs: [])

    assert update_cmd._pending_fleet_restart_needed() is False


def test_unrecorded_stale_gateway_keeps_marker_pending(monkeypatch):
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text(_marker_body(started=1, expected_sha="new-sha"), encoding="utf-8")
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(json.dumps({
        "pid": 99999999,
        "update_id": _UPDATE_ID,
        "post_update": {"sha": "new-sha"},
        "plan": {"inventory_complete": True, "runtimes": []},
    }), encoding="utf-8")
    homes = [("default", get_hermes_home())]
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: homes)
    monkeypatch.setattr(
        "hermes_cli.update_receipt._socket_identity",
        lambda _home: (42, {"profile": "default", "code_sha": "old-sha"}),
    )
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **_kwargs: [42])

    assert update_cmd._pending_fleet_restart_needed() is True


def test_unidentified_gateway_keeps_marker_pending(monkeypatch):
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text(_marker_body(started=1, expected_sha="new-sha"), encoding="utf-8")
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(json.dumps({
        "pid": 99999999,
        "update_id": _UPDATE_ID,
        "post_update": {"sha": "new-sha"},
        "plan": {"inventory_complete": True, "runtimes": []},
    }), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [])
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **_kwargs: [42])

    assert update_cmd._pending_fleet_restart_needed() is True


def test_unledgered_serve_process_keeps_marker_pending(monkeypatch):
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text(_marker_body(started=1, expected_sha="abc123"), encoding="utf-8")
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(json.dumps({
        "pid": 99999999,
        "update_id": _UPDATE_ID,
        "post_update": {"sha": "abc123"},
        "plan": {"inventory_complete": True, "runtimes": []},
    }), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [])
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **_kwargs: [])
    monkeypatch.setattr("hermes_cli.process_identity.ledger_entries", lambda **_kwargs: [])
    monkeypatch.setattr(
        "hermes_cli.dashboard_procs._scan_dashboard_processes",
        lambda **_kwargs: [(77, "hermes --profile work serve")],
    )

    assert update_cmd._pending_fleet_restart_needed() is True


def test_legacy_marker_settles_from_strict_live_fleet_evidence(monkeypatch):
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text("started=1\npid=99999999\nexpected_sha=abc123\n", encoding="utf-8")
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [("default", get_hermes_home())])
    monkeypatch.setattr(
        "hermes_cli.update_receipt._socket_identity",
        lambda _home: (42, {"profile": "default", "code_sha": "abc123"}),
    )
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **_kwargs: [42])
    monkeypatch.setattr("hermes_cli.process_identity.ledger_entries", lambda **_kwargs: [])
    monkeypatch.setattr("hermes_cli.dashboard_procs._scan_dashboard_processes", lambda **_kwargs: [])

    assert update_cmd._pending_fleet_restart_needed() is False
    completion = json.loads(update_cmd_fleet._fleet_restart_completion_path().read_text(encoding="utf-8"))
    assert completion["update_id"] is None


def test_incomplete_strict_gateway_scan_keeps_marker_pending(monkeypatch):
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text(_marker_body(started=1, expected_sha="abc123"), encoding="utf-8")
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(json.dumps({
        "pid": 99999999,
        "update_id": _UPDATE_ID,
        "post_update": {"sha": "abc123"},
        "plan": {"inventory_complete": True, "runtimes": []},
    }), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [])
    monkeypatch.setattr(
        "hermes_cli.gateway.find_gateway_pids",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("process listing unavailable")),
    )

    assert update_cmd._pending_fleet_restart_needed() is True


@pytest.mark.parametrize(("inventory_complete", "pending"), [(True, False), (None, True)])
def test_empty_worklist_requires_completed_inventory(monkeypatch, inventory_complete, pending):
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text(_marker_body(started=1, expected_sha="abc123"), encoding="utf-8")
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    plan = {"runtimes": []}
    if inventory_complete is not None:
        plan["inventory_complete"] = inventory_complete
    (receipt_dir / "latest.json").write_text(json.dumps({
        "pid": 99999999,
        "update_id": _UPDATE_ID,
        "post_update": {"sha": "abc123"},
        "plan": plan,
    }), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [])
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **_kwargs: [])
    monkeypatch.setattr("hermes_cli.process_identity.ledger_entries", lambda **_kwargs: [])
    monkeypatch.setattr("hermes_cli.dashboard_procs._scan_dashboard_processes", lambda **_kwargs: [])

    assert update_cmd._pending_fleet_restart_needed() is pending


def test_catchup_rechecks_generation_before_clearing_marker(monkeypatch):
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text(_marker_body(started=1, expected_sha="abc123"), encoding="utf-8")
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    receipt = {
        "pid": 99999999,
        "update_id": _UPDATE_ID,
        "post_update": {"sha": "abc123"},
        "plan": {
            "inventory_complete": True,
            "runtimes": [{"kind": "gateway", "profile": "default", "pid": 42}],
        },
    }
    (receipt_dir / "latest.json").write_text(json.dumps(receipt), encoding="utf-8")
    pending = iter([True, False])
    original = marker.read_bytes()
    seen = []
    monkeypatch.setattr(update_cmd_fleet, "_pending_fleet_restart_needed", lambda: next(pending))
    monkeypatch.setattr(
        update_cmd,
        "_run_pending_fleet_restart",
        lambda *, receipt=None: seen.append(receipt) or True,
    )

    update_cmd_fleet._apply_pending_fleet_restart_catchup()

    assert seen == [receipt]
    assert marker.read_bytes() == original


@pytest.mark.parametrize(
    ("target", "payload"),
    [
        pytest.param("completion", [], id="completion-list"),
        pytest.param("receipt", [], id="receipt-list"),
        pytest.param(
            "receipt",
            {"pid": 99999999, "update_id": _UPDATE_ID, "post_update": []},
            id="post-update-list",
        ),
        pytest.param(
            "receipt",
            {
                "pid": 99999999,
                "update_id": _UPDATE_ID,
                "post_update": {"sha": "abc123"},
                "plan": [],
            },
            id="plan-list",
        ),
        pytest.param(
            "receipt",
            {
                "pid": 99999999,
                "update_id": _UPDATE_ID,
                "post_update": {"sha": "abc123"},
                "plan": {"runtimes": {}},
            },
            id="runtimes-object",
        ),
        pytest.param(
            "receipt",
            {
                "pid": 99999999,
                "update_id": _UPDATE_ID,
                "post_update": {"sha": "abc123"},
                "plan": {"runtimes": [{"kind": "gateway", "profile": [], "pid": 99999998}]},
            },
            id="profile-list",
        ),
        pytest.param(
            "receipt",
            {
                "pid": 99999999,
                "update_id": _UPDATE_ID,
                "post_update": {"sha": "abc123"},
                "plan": {"runtimes": [{"kind": "gateway", "profile": {}, "pid": 99999998}]},
            },
            id="profile-object",
        ),
    ],
)
def test_wrong_persisted_container_shapes_remain_pending(target, payload):
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text(_marker_body(started=1, expected_sha="abc123"), encoding="utf-8")
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    path = (
        update_cmd_fleet._fleet_restart_completion_path()
        if target == "completion"
        else receipt_dir / "latest.json"
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert update_cmd._pending_fleet_restart_needed() is True


@pytest.mark.parametrize("started", ["nan", "inf", "-inf"])
def test_non_finite_marker_timestamp_remains_pending(monkeypatch, started):
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
    ).strip()
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text(
        _marker_body(started=started, expected_sha=sha),
        encoding="utf-8",
    )
    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(
        json.dumps(
            {
                "pid": 99999999,
                "post_update": {"sha": sha},
                "plan": {
                    "runtimes": [
                        {"kind": "gateway", "profile": "default", "pid": 99999998}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "hermes_cli.update_receipt._profile_homes",
        lambda: [("default", get_hermes_home())],
    )
    monkeypatch.setattr(
        "hermes_cli.update_receipt._socket_identity",
        lambda _home: (os.getpid(), {"profile": "default", "code_sha": sha}),
    )

    assert update_cmd._pending_fleet_restart_needed() is True


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


@pytest.mark.parametrize(("dashboard_gone", "pending"), [(False, True), (True, False)])
def test_stale_receipt_settles_after_recorded_dashboard_incarnation_is_gone(
    monkeypatch, dashboard_gone, pending
):
    disk_sha = "n" * 40
    old_sha = "o" * 40
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: disk_sha)
    monkeypatch.setattr(
        update_cmd_fleet,
        "_recorded_incarnation_is_gone",
        lambda _runtime: dashboard_gone,
    )
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda: [
            {
                "profile": "default",
                "pid": 3,
                "code_sha": disk_sha,
                "state": "current",
            }
        ],
    )
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **_kwargs: [3])

    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(
        json.dumps(
            {
                "outcome": "success",
                "exit_code": 0,
                "plan": {
                    "runtimes": [
                        {
                            "kind": "dashboard",
                            "profile": "default",
                            "pid": 1,
                            "detail": {"create_time": 1.0},
                        }
                    ]
                },
                "fleet": [
                    {
                        "profile": "default",
                        "pid": 2,
                        "code_sha": old_sha,
                        "state": "current",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert update_cmd._pending_fleet_restart_needed() is pending


def test_stale_receipt_does_not_settle_when_strict_scan_finds_omitted_gateway(
    monkeypatch,
):
    monkeypatch.setattr(
        "hermes_cli.update_receipt.read_latest_receipt",
        lambda: {
            "plan": {"runtimes": [{"kind": "gateway", "profile": "default"}]},
            "fleet": [{"profile": "default"}],
        },
    )
    monkeypatch.setattr(
        "hermes_cli.update_receipt.collect_fleet_versions",
        lambda: [{"profile": "default", "pid": 3, "code_sha": "new", "state": "current"}],
    )
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **_kwargs: [3, 4])

    assert update_cmd_fleet._live_fleet_covers_receipt("new") is False


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


@pytest.mark.parametrize("kind", ["serve", "dashboard"])
def test_pending_restart_rejects_unsupported_receipt_runtime(monkeypatch, kind):
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **_kwargs: [])
    monkeypatch.setattr("hermes_cli.gateway.supports_systemd_services", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway.is_macos", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway.is_windows", lambda: False)
    receipt = {"plan": {"runtimes": [{"kind": kind, "profile": "default", "pid": 42}]}}

    assert update_cmd._run_pending_fleet_restart(receipt=receipt) is False


@pytest.mark.parametrize("kind", ["serve", "dashboard"])
def test_pending_restart_accepts_gone_recorded_runtime(monkeypatch, kind):
    monkeypatch.setattr(hermes_main, "_purge_stale_hermes_modules", lambda: None)
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **_kwargs: [])
    monkeypatch.setattr("hermes_cli.gateway.supports_systemd_services", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway.is_macos", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway.is_windows", lambda: False)
    receipt = {"plan": {"runtimes": [{
        "kind": kind, "profile": "default", "pid": 99999999,
        "detail": {"create_time": 1.0},
    }]}}

    assert update_cmd._run_pending_fleet_restart(receipt=receipt) is True


def test_pending_restart_accepts_systemd_managed_dashboard_worklist(monkeypatch):
    monkeypatch.setattr(hermes_main, "_purge_stale_hermes_modules", lambda: None)
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **_kwargs: [])
    monkeypatch.setattr("hermes_cli.gateway.supports_systemd_services", lambda: True)
    monkeypatch.setattr("hermes_cli.gateway.is_macos", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway.is_windows", lambda: False)
    monkeypatch.setattr(update_cmd_fleet, "_systemd_gateway_unit_listings", lambda: [
        (scope, cmd, SimpleNamespace(returncode=0, stdout=""))
        for scope, cmd in update_cmd_fleet._SYSTEMD_SCOPES
    ])
    receipt = {"plan": {"runtimes": [{
        "kind": "dashboard", "profile": "default", "pid": 42, "supervisor": "systemd",
        "detail": {"create_time": 1.0},
    }]}}

    assert update_cmd._run_pending_fleet_restart(receipt=receipt) is True


@pytest.mark.parametrize("failure", ["kill", "wait"])
def test_pending_restart_fails_when_old_gateway_cannot_be_stopped(monkeypatch, failure):
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text(_marker_body(started=1, expected_sha="abc123"), encoding="utf-8")
    original = marker.read_bytes()
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **_kwargs: [42])
    monkeypatch.setattr("hermes_cli.gateway.supports_systemd_services", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway.is_macos", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway.is_windows", lambda: False)
    if failure == "kill":
        monkeypatch.setattr(
            "hermes_cli.gateway.kill_gateway_processes",
            lambda **_kwargs: (_ for _ in ()).throw(OSError("denied")),
        )
        monkeypatch.setattr("hermes_cli.gateway._wait_for_gateway_exit", lambda **_kwargs: True)
    else:
        monkeypatch.setattr("hermes_cli.gateway.kill_gateway_processes", lambda **_kwargs: 1)
        monkeypatch.setattr("hermes_cli.gateway._wait_for_gateway_exit", lambda **_kwargs: False)

    with pytest.raises(SystemExit) as excinfo:
        update_cmd_fleet._apply_pending_fleet_restart_catchup()

    assert excinfo.value.code == 1
    assert marker.read_bytes() == original


def test_pending_restart_fails_when_one_supervisor_scope_fails(monkeypatch):
    marker = update_cmd._fleet_restart_pending_marker_path()
    marker.write_text(_marker_body(started=1, expected_sha="abc123"), encoding="utf-8")
    original = marker.read_bytes()
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **_kwargs: [])
    monkeypatch.setattr("hermes_cli.gateway.supports_systemd_services", lambda: True)
    monkeypatch.setattr("hermes_cli.gateway.is_macos", lambda: False)
    monkeypatch.setattr("hermes_cli.gateway.is_windows", lambda: False)
    listings = [
        ("user", ["systemctl", "--user"], SimpleNamespace(returncode=1, stdout="")),
        ("system", ["systemctl"], SimpleNamespace(returncode=0, stdout="")),
    ]
    monkeypatch.setattr(update_cmd_fleet, "_systemd_gateway_unit_listings", lambda: listings)

    with pytest.raises(SystemExit) as excinfo:
        update_cmd_fleet._apply_pending_fleet_restart_catchup()

    assert excinfo.value.code == 1
    assert marker.read_bytes() == original


# ---------------------------------------------------------------------------
# cmd_update integration (mocked git / restart)
# ---------------------------------------------------------------------------


def test_marker_written_after_pull_settled_after_successful_restart(
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
    assert update_cmd._fleet_restart_pending_marker_path().is_file()
    assert update_cmd._pending_fleet_restart_needed() is False
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


def test_clean_update_preserves_marker_when_runtime_inventory_is_incomplete(
    monkeypatch, tmp_path, capsys
):
    from hermes_cli.update_inventory import UpdatePlan
    import hermes_cli.update_inventory as ui

    args = _update_args()
    _patch_update_deps(monkeypatch, tmp_path, _make_head_moved_side_effect())
    plan = UpdatePlan()
    plan.inventory_errors.append("Serve/dashboard ledger inventory")
    monkeypatch.setattr(ui, "collect_runtime_inventory", lambda: plan)

    with pytest.raises(SystemExit) as excinfo:
        hermes_main.cmd_update(args)

    assert excinfo.value.code == 1
    assert update_cmd._fleet_restart_pending_marker_path().is_file()
    assert not update_cmd_fleet._fleet_restart_completion_path().exists()
    receipt = json.loads(
        (get_hermes_home() / "logs" / "update_receipts" / "latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["outcome"] == "partial"
    assert "runtime inventory was incomplete" in capsys.readouterr().out


def test_completion_write_failure_finalizes_partial_receipt(monkeypatch, tmp_path):
    args = _update_args()
    _patch_update_deps(monkeypatch, tmp_path, _make_head_moved_side_effect())
    monkeypatch.setattr(
        update_cmd_fleet, "_record_fleet_restart_completion", lambda *_args: False
    )

    with pytest.raises(SystemExit) as excinfo:
        hermes_main.cmd_update(args)

    assert excinfo.value.code == 1
    assert update_cmd._fleet_restart_pending_marker_path().is_file()
    receipt = json.loads(
        (get_hermes_home() / "logs" / "update_receipts" / "latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["outcome"] == "partial"


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
    marker_fields = dict(
        line.split("=", 1) for line in marker.read_text(encoding="utf-8").splitlines()
    )
    assert marker_fields["expected_sha"] == "def456"
    receipt = json.loads((get_hermes_home() / "logs" / "update_receipts" / "latest.json").read_text(encoding="utf-8"))
    assert marker_fields["update_id"] == receipt["update_id"]
    assert receipt["plan"]["inventory_complete"] is True


def test_already_up_to_date_runs_pending_restart_when_marker_present(
    monkeypatch, tmp_path, capsys
):
    args = _update_args()
    _patch_update_deps(monkeypatch, tmp_path, _make_up_to_date_side_effect())
    update_cmd._write_fleet_restart_pending_marker(expected_sha="def456")
    pending = iter([True, False])
    monkeypatch.setattr(update_cmd_fleet, "_pending_fleet_restart_needed", lambda: next(pending))

    seen = {"ran": False}

    def _restart():
        seen["ran"] = True
        return True

    monkeypatch.setattr(update_cmd, "_run_pending_fleet_restart", _restart)
    monkeypatch.setattr(update_cmd_fleet, "_run_pending_fleet_restart", _restart)

    hermes_main.cmd_update(args)

    assert seen["ran"] is True
    assert update_cmd._fleet_restart_pending_marker_path().is_file()
    out = capsys.readouterr().out
    assert "unverified fleet-restart obligation" in out


def test_already_up_to_date_runs_pending_restart_when_receipt_skewed(
    monkeypatch, tmp_path, capsys
):
    args = _update_args()
    _patch_update_deps(monkeypatch, tmp_path, _make_up_to_date_side_effect())
    pending = iter([True, False])
    monkeypatch.setattr(update_cmd_fleet, "_pending_fleet_restart_needed", lambda: next(pending))

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
    assert "unverified fleet-restart obligation" in out


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
    assert "unverified fleet-restart obligation" not in capsys.readouterr().out


def test_startup_warn_prints_when_marker_present(capsys):
    update_cmd._write_fleet_restart_pending_marker()
    update_cmd._warn_pending_fleet_restart_on_startup()
    err = capsys.readouterr().err
    assert "unverified fleet-restart obligation" in err
    assert "hermes gateway restart" in err


def test_startup_warn_silent_when_nothing_pending(capsys):
    update_cmd._warn_pending_fleet_restart_on_startup()
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""
