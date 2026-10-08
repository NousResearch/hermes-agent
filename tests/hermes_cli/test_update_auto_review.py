"""Integration boundaries for unattended updater scope and launch failures."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import update_auto_run as runner, update_auto_state as state


@pytest.fixture
def context(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return state.AutoUpdateContext(tmp_path / "install", home, home / "logs" / "update_receipts")


def test_failure_to_open_output_does_not_leave_a_phantom_running_update(context, monkeypatch):
    monkeypatch.setattr(runner, "require_source_install", lambda context: None)
    monkeypatch.setattr(runner, "command", lambda context, args: ["/unused/updater"])

    def denied(*args, **kwargs):
        raise PermissionError("log destination is read-only")

    monkeypatch.setattr(runner, "append_log", denied)
    with state.operation_lock(context):
        try:
            runner.run_update(context, state.default_status(context), SimpleNamespace())
        except OSError:
            pass
    assert state.read_status(context)["status"] == "update_failed"


@pytest.mark.platforms("posix")
def test_symlinked_profile_cannot_retarget_installation_home(context, tmp_path, monkeypatch):
    from hermes_cli import config

    external = tmp_path / "another-install" / "home"
    external.mkdir(parents=True)
    (external / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    profile = context.home / "profiles" / "work"
    profile.parent.mkdir()
    profile.symlink_to(external, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr(config, "get_project_root", lambda: context.install)
    actual = state.AutoUpdateContext.current()
    assert actual.home == context.home
    assert actual.receipt_directory == context.home / "logs" / "update_receipts"
    monkeypatch.setattr(runner, "installation_command", lambda root, args, **kwargs: args)
    assert runner.command(actual, ["update"]) == ["--profile", "default", "update"]


def test_generation_console_script_uses_owning_checkout(context, monkeypatch):
    from hermes_cli import config
    from pm.environments import install_key

    checkout = context.install
    (checkout / "hermes_cli").mkdir(parents=True)
    (checkout / "hermes_cli" / "main.py").write_text("", encoding="utf-8")
    install_state = context.home / "installs" / install_key(checkout)
    generation = install_state / "environments" / "generation"
    (generation / "venv").mkdir(parents=True)
    workspace = generation / "workspace"
    workspace.mkdir()
    (install_state / "inputs").mkdir()
    (install_state / "inputs" / ".project-root").write_text(str(checkout), encoding="utf-8")
    monkeypatch.setattr(sys, "prefix", str(generation / "venv"))
    monkeypatch.setattr(sys, "base_prefix", str(context.home / "python"))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(config, "get_project_root", lambda: workspace)
    assert state.AutoUpdateContext.current().install == checkout


def test_plan_refuses_remote_branch_fallback_that_apply_would_reject(context, monkeypatch):
    from hermes_cli import config, source_check

    monkeypatch.setattr(runner, "require_source_install", lambda context: None)
    monkeypatch.setattr(config, "require_readable_config_before_write", lambda path: {})
    monkeypatch.setattr(source_check, "check_for_updates", lambda **kwargs: {
        "supported": True, "updateAvailable": True, "currentSha": "old", "targetSha": "main-tip",
        "branch": "main",
    })
    with pytest.raises(ValueError):
        runner.check_update(context, SimpleNamespace(branch="deleted-feature", channel=None))


def test_reconciliation_uses_canonical_failure_code_when_parent_lost_it(context):
    context.receipt_directory.mkdir(parents=True)
    receipt = {"correlation_id": "our-run", "finished_at": "now", "outcome": "failed", "exit_code": 11}
    (context.receipt_directory / "update_ours.json").write_text(json.dumps(receipt), encoding="utf-8")
    status = {**state.default_status(context), "status": "running", "runPending": True,
              "correlationId": "our-run", "exitCode": None}
    with state.operation_lock(context):
        assert runner.reconcile_run(context, status) is False
    assert status["status"] == "backup_failed"
    assert status["exitCode"] == 11
