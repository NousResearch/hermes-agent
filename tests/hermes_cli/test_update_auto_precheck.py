"""Read-only no-op admission must preserve canonical recovery obligations."""

import json
from pathlib import Path

import pytest

from hermes_cli import update_auto_precheck as precheck
from hermes_cli.update_auto_state import AutoUpdateContext


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    context = AutoUpdateContext(tmp_path / "install", home, home / "logs" / "update_receipts")
    marker, directory, pause = home / "pending", home / "serve_restart_pending", home / "paused.json"
    monkeypatch.setattr(precheck, "_paths_in_scope", lambda c: ([marker, pause], [directory], pause))
    return context, marker, directory, pause


def test_first_clean_no_update_needs_no_previous_auto_receipt(evidence):
    context, *_ = evidence
    before = list(context.home.rglob("*"))
    assert precheck.recovery_needed(context, "current") is False
    assert list(context.home.rglob("*")) == before


@pytest.mark.parametrize("kind", ["marker", "serve", "claim"])
def test_pending_artifact_keeps_the_canonical_recovery_path(evidence, kind):
    context, marker, directory, pause = evidence
    if kind == "marker":
        marker.write_text("owed")
    elif kind == "serve":
        directory.mkdir()
        (directory / "pending.json").write_text("{}")
    else:
        pause.with_name(pause.name + ".worker.claim").write_text("claimed")
    assert precheck.recovery_needed(context, "current") is True


def test_empty_old_serve_directory_and_lock_are_not_debt(evidence):
    context, _, directory, pause = evidence
    directory.mkdir()
    pause.with_name(pause.name + ".lock").write_text("")
    pause.with_name(pause.name + ".retired").write_text("")
    assert precheck.recovery_needed(context, "current") is False


@pytest.mark.parametrize("change", [{"outcome": "failed"}, {"outcome": "running"}, {"outcome": "partial"},
                                   {"followups": [{"step": "build"}]}, {"user_action": "restore stash"},
                                   {"pending_manual_serves": [{"pid": 1}]}, {"carried_manual_serves": [{"pid": 1}]},
                                   {"gateway_restart": {"incomplete": True}}, {"exit_code": 1},
                                   {"post_update": {"sha": "older"}}, {"fleet": [{"state": "stale"}]},
                                   {"finished_at": None}])
def test_terminal_receipt_debt_is_not_hidden_by_current_code(evidence, change):
    context, *_ = evidence
    context.receipt_directory.mkdir(parents=True)
    receipt = {"outcome": "success", "finished_at": "now", "post_update": {"sha": "current"}, **change}
    (context.receipt_directory / "latest.json").write_text(json.dumps(receipt))
    assert precheck.recovery_needed(context, "current") is True


def test_clean_receipt_allows_check_only(evidence):
    context, *_ = evidence
    context.receipt_directory.mkdir(parents=True)
    receipt = {"outcome": "success", "finished_at": "now", "post_update": {"sha": "current"},
               "fleet": [{"state": "current"}, {"state": "external"}]}
    (context.receipt_directory / "latest.json").write_text(json.dumps(receipt))
    assert precheck.recovery_needed(context, "current") is False


@pytest.mark.parametrize("content", ["not-json", "[]", "null"])
def test_corrupt_receipt_is_unknown_not_clean(evidence, content):
    context, *_ = evidence
    context.receipt_directory.mkdir(parents=True)
    (context.receipt_directory / "latest.json").write_text(content)
    assert precheck.recovery_needed(context, "current") is True


@pytest.mark.parametrize("error", [PermissionError("unreadable"), ValueError("foreign scope")])
def test_unknown_scope_or_read_error_keeps_recovery(evidence, monkeypatch, error):
    context, *_ = evidence

    def fail(c):
        raise error

    monkeypatch.setattr(precheck, "_paths_in_scope", fail)
    assert precheck.recovery_needed(context, "current") is True


def test_real_artifact_owners_are_passive_and_include_named_profiles(tmp_path, monkeypatch):
    from hermes_cli.venv_sync import completion_pending_path

    home = tmp_path / "home"
    named = home / "profiles" / "work"
    named.mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n")
    (named / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "host-state"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    install = Path(precheck.__file__).resolve().parents[1]
    context = AutoUpdateContext(install, home, home / "logs" / "update_receipts")
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    paths, directories, pause = precheck._paths_in_scope(context)
    assert completion_pending_path(install) in paths
    assert home / "fleet_restart_pending" in paths
    assert named / "fleet_restart_pending" in paths
    assert named / "serve_restart_pending" in directories
    assert pause.parent == home
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


@pytest.mark.platforms("posix")
def test_dangling_pending_marker_is_not_absence(evidence):
    context, marker, *_ = evidence
    marker.symlink_to(context.home / "missing")
    assert precheck.recovery_needed(context, "current") is True
