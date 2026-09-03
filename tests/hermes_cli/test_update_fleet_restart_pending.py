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
from hermes_cli import update_cmd
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
    monkeypatch.setattr(
        hermes_main,
        "_get_origin_url",
        lambda *a, **k: "https://github.com/NousResearch/hermes-agent.git",
    )
    monkeypatch.setattr(hermes_main, "_is_fork", lambda *a, **k: False)
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed", lambda *a, **k: None
    )
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", lambda *a, **k: 0)
    monkeypatch.setattr(
        hermes_main, "_record_bytecode_fingerprint", lambda *a, **k: None
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
    monkeypatch.setattr(
        hermes_main, "_finish_dashboard_update_cleanup", lambda *a, **k: None
    )
    monkeypatch.setattr(
        update_cmd, "_finish_dashboard_update_cleanup", lambda *a, **k: None
    )
    monkeypatch.setattr(hermes_main, "_build_web_ui", lambda *a, **k: None)
    monkeypatch.setattr(
        update_cmd, "_venv_core_imports_healthy", lambda: (True, "")
    )
    monkeypatch.setattr(update_cmd, "_update_node_dependencies", lambda: [])
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


def test_stale_fleet_matrix_on_latest_receipt_is_pending(monkeypatch):
    disk_sha = "n" * 40
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: disk_sha)

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

    assert update_cmd._run_pending_fleet_restart() is True
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


def test_stale_unfinished_receipt_dismissed_after_catchup(monkeypatch):
    """An interrupted-update receipt re-triggers pending infinitely.
    _dismiss_stale_receipt_trigger must rename it so the next check
    sees nothing pending (#98022)."""
    disk_sha = "e" * 40
    stale_sha = "7" * 40
    monkeypatch.setattr(update_cmd, "_current_checkout_sha", lambda: disk_sha)

    receipt_dir = get_hermes_home() / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    receipt_json = json.dumps(
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
                        "code_sha": stale_sha,
                    }
                ],
            },
        }
    )
    receipt_path = receipt_dir / "latest.json"
    receipt_path.write_text(receipt_json, encoding="utf-8")

    assert update_cmd._pending_fleet_restart_needed() is True

    update_cmd._dismiss_stale_receipt_trigger()

    assert update_cmd._pending_fleet_restart_needed() is False
    assert not receipt_path.exists()
    assert receipt_dir.joinpath("latest.json.bak").read_text(encoding="utf-8") == receipt_json
