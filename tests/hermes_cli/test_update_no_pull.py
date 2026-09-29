"""--no-pull completes the updater at the current HEAD without touching git (#124382)."""
import argparse
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd
from hermes_cli.subcommands.update import build_update_parser


def _make_args(no_pull=True):
    return argparse.Namespace(
        gateway=False, check=False, plan=False, list_venv_holders=False,
        no_backup=False, backup=False, yes=False, keep_stash=False,
        branch=None, switch_branch=False, force=False, force_venv=False,
        set_channel=None, install_id=False, channel=None,
        no_gateway_restart=False, no_pull=no_pull)


def _patch_impl(monkeypatch):
    """Freeze the pre-completion scaffolding via the ``_m`` seam the impl reads through."""
    resumed = []
    fake_m = SimpleNamespace(
        _pause_windows_gateways_for_update=lambda: None,
        _resume_windows_gateways_after_update=lambda token: resumed.append(token),
        _run_pre_update_backup=lambda args: None,
        _desktop_packaged_executable=lambda d: None,
        _desktop_dist_exists=lambda d: False,
        _resolve_update_branch=lambda args: "main",
        PROJECT_ROOT=Path("/tmp/fake-hermes-root"),
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_m)
    monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda args, gm: SimpleNamespace(
        pre_update_version=None, gw_input_fn=None, assume_yes=True, keep_stash=False,
        switch_branch=False, discard_local_changes=False, no_gateway_restart=False))
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", lambda args: None)
    monkeypatch.setattr(update_cmd, "_record_pre_update_backup_outcome", lambda args, sid: None)
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda: (False, ["git"], False))
    monkeypatch.setattr(update_cmd, "_source_completion_request", lambda *a, **k: {"schema": 1})
    return resumed


def test_no_pull_flag_parses():
    parser = argparse.ArgumentParser()
    build_update_parser(parser.add_subparsers(), cmd_update=lambda a: None)
    assert parser.parse_args(["update", "--no-pull"]).no_pull is True
    assert parser.parse_args(["update"]).no_pull is False


def test_no_pull_completes_at_current_head_without_git_movement(monkeypatch, capsys):
    _patch_impl(monkeypatch)

    captured = {}
    completed = {}

    def _fake_complete(request):
        completed["ran"] = True
        captured.update(request)

    monkeypatch.setattr(update_cmd, "_complete_source_update", _fake_complete)

    git_calls = []

    def _spy_run(git_cmd, args, cwd=None, check=False, network=False):
        git_calls.append(" ".join(str(a) for a in args))
        if args[:1] == ["status"]:
            return subprocess.CompletedProcess(["git"] + args, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(["git"] + args, 0, stdout="", stderr="")

    monkeypatch.setattr(update_cmd, "_git_run", _spy_run)
    monkeypatch.setattr(update_cmd, "_current_branch_name", lambda git_cmd, check=False: "pinned")
    monkeypatch.setattr(update_cmd, "_capture_head_sha", lambda git_cmd, cwd: "a" * 40)

    update_cmd._cmd_update_impl(_make_args(no_pull=True), gateway_mode=False)

    assert completed["ran"]
    assert captured["expected_sha"] == "a" * 40
    assert captured["completion_message"] == "✓ Completed at aaaaaaaaaa (pinned); no code was pulled."
    # No fetch / merge / reset / checkout / pull / stash: git may only be inspected.
    forbidden = ("fetch", "merge", "reset", "checkout", "pull", "stash")
    moved = [call for call in git_calls if call.split()[0] in forbidden]
    assert not moved, f"--no-pull moved the checkout: {moved}"


def test_no_pull_refuses_dirty_tree(monkeypatch, capsys):
    resumed = _patch_impl(monkeypatch)
    monkeypatch.setattr(
        update_cmd, "_git_run",
        lambda git_cmd, args, cwd=None, check=False, network=False:
        subprocess.CompletedProcess(["git"] + args, 0, stdout=" M file.py\n", stderr=""))

    with pytest.raises(SystemExit) as exc:
        update_cmd._cmd_update_impl(_make_args(no_pull=True), gateway_mode=False)

    assert exc.value.code == 1
    assert resumed == [None], "dirty refusal must resume Windows gateways"
    out = capsys.readouterr().out
    assert "refused" in out and "--no-pull" in out
