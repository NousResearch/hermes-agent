"""Canonical updater invocation, exact receipts, and plan target parity."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import update_auto_run as runner
from hermes_cli.update_auto_state import AutoUpdateContext, default_status, read_status


@pytest.fixture
def context(tmp_path, monkeypatch):
    home = tmp_path / "home"
    root = tmp_path / "install"
    home.mkdir(); root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return AutoUpdateContext(root, home, home / "logs" / "update_receipts")


def test_command_pins_same_installation_home_from_every_profile(context, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "installation_command", lambda root, args, **kw: calls.append((root, args, kw)) or args)
    runner.command(context, ["update", "auto", "run-scheduled"])
    assert calls[0] == (context.install, ["--profile", "default", "update", "auto", "run-scheduled"], {"home": context.home})
    named = AutoUpdateContext(context.install, context.home / "profiles" / "work", context.receipt_directory)
    runner.command(named, ["update"])
    assert calls[1][1][:2] == ["--profile", "default"]
    assert calls[1][2] == {"home": context.home}


@pytest.mark.parametrize("receipt,code,expected", [
    ({"outcome": "success", "pre_update": {"sha": "same"}, "post_update": {"sha": "same"}}, 0, ("up_to_date", 0)),
    ({"outcome": "success", "pre_update": {"sha": "old"}, "post_update": {"sha": "new"}}, 0, ("success", 0)),
    ({"outcome": "success", "followups": [{"step": "restart"}]}, 0, ("followup_required", 14)),
    ({"outcome": "success", "exit_code": 1}, 0, ("followup_required", 14)),
    ({"outcome": "success"}, 130, ("followup_required", 14)),
    ({"outcome": "partial", "user_action": {"kind": "stash"}}, 1, ("followup_required", 14)),
    ({"outcome": "failed"}, 11, ("backup_failed", 11)),
    ({"outcome": "failed"}, 0, ("update_failed", 12)),
    ({"outcome": "interrupted"}, 130, ("update_failed", 130)),
    ({"outcome": "refused"}, 2, ("update_failed", 2)),
])
def test_receipt_semantics_follow_current_commit_contract(receipt, code, expected):
    assert runner.receipt_result(receipt, code) == expected


def test_receipt_matching_ignores_latest_and_incomplete(context):
    directory = context.receipt_directory
    directory.mkdir(parents=True)
    terminal = {"correlation_id": "our-run", "finished_at": "now", "outcome": "success"}
    (directory / "latest.json").write_text(json.dumps(terminal))
    (directory / "update_foreign.json").write_text(json.dumps({**terminal, "correlation_id": "someone-else"}))
    (directory / "update_incomplete.json").write_text(json.dumps({**terminal, "finished_at": None}))
    assert runner.find_receipt(directory, "our-run") is None
    expected = directory / "update_ours.json"
    expected.write_text(json.dumps(terminal))
    assert runner.find_receipt(directory, "our-run") == (expected, terminal)
    (directory / "update_duplicate.json").write_text(json.dumps(terminal))
    assert runner.find_receipt(directory, "our-run") is None


@pytest.mark.parametrize("terminal,returncode,expected", [(True, 0, "success"), (False, 0, "unverified"),
                                                        (False, 2, "unverified")])
@pytest.mark.live_system_guard_bypass
def test_real_child_boundary_requires_its_own_receipt(context, monkeypatch, terminal, returncode, expected):
    # Only our throwaway stub executes, never hermes_cli.main or an updater.
    monkeypatch.setattr(runner, "require_source_install", lambda c: None)
    child = context.install / "fake_updater.py"
    child.write_text(
        "import json,os,pathlib,sys\n"
        "home=pathlib.Path(os.environ['HERMES_HOME'])\n"
        "(home/'seen.json').write_text(json.dumps({'args':sys.argv[1:],'home':str(home)}))\n"
        "print('canonical child output')\n"
        + ("directory=home/'logs'/'update_receipts'; directory.mkdir(parents=True,exist_ok=True)\n"
           "(directory/'update_ours.json').write_text(json.dumps({'correlation_id':os.environ['HERMES_UPDATE_CORRELATION_ID'],"
           "'finished_at':'now','outcome':'success','pre_update':{'sha':'old'},'post_update':{'sha':'new'}}))\n" if terminal else "")
        + f"sys.exit({returncode})\n"
    )
    monkeypatch.setattr(runner, "command", lambda c, argv: [sys.executable, str(child), *argv])
    status = default_status(context)
    code = runner.run_update(context, status, SimpleNamespace(branch="main", channel=None))
    persisted = read_status(context)
    assert persisted["status"] == expected
    assert code == (0 if terminal else 12)
    seen = json.loads((context.home / "seen.json").read_text())
    assert seen["args"] == ["update", "--yes", "--require-backup", "--branch", "main"]
    assert seen["home"] == str(context.home)
    assert "canonical child output" in context.log_path.read_text()


def test_failed_spawn_leaves_durable_failure(context, monkeypatch):
    monkeypatch.setattr(runner, "require_source_install", lambda c: None)
    monkeypatch.setattr(runner, "command", lambda c, a: [str(context.install / "missing")])
    assert runner.run_update(context, default_status(context), SimpleNamespace()) == 12
    assert read_status(context)["status"] == "update_failed"


@pytest.mark.parametrize("channel,target_branch,explicit_branch,expected", [
    ("main", "main", None, "main"), ("stable", None, None, None),
    ("preview", "next", None, "next"), ("stable", None, "feature", "feature"),
])
def test_plan_target_matches_apply_not_current_parked_branch(context, monkeypatch, channel, target_branch, explicit_branch, expected):
    from hermes_cli import config, source_check, source_releases, update_installation

    monkeypatch.setattr(runner, "require_source_install", lambda c: None)
    monkeypatch.setattr(config, "require_readable_config_before_write", lambda p: {})
    monkeypatch.setattr(update_installation, "resolve_install_channel", lambda *a, **kw: channel)
    monkeypatch.setattr(source_releases, "resolve_source_target", lambda *a: SimpleNamespace(branch=target_branch))
    calls = []
    monkeypatch.setattr(source_check, "check_for_updates", lambda **kw: calls.append(kw) or {
        "supported": True, "updateAvailable": True, "currentSha": "a", "targetSha": "b", "branch": kw["branch"]})
    runner.check_update(context, SimpleNamespace(branch=explicit_branch, channel=None))
    assert calls[0]["branch"] == expected
    assert calls[0]["channel"] == ("main" if explicit_branch else channel)
    assert calls[0]["force"] is True
