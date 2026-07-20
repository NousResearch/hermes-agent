"""Tests for git-based gateway deployment orchestration."""

from pathlib import Path

import pytest

from hermes_cli.deploy_gateway import DeployOptions, GatewayDeployer, parse_args


def _make_options(tmp_path: Path, **overrides) -> DeployOptions:
    data = {
        "host": "root@example.com",
        "project_dir": tmp_path / "repo",
        "remote_dir": "/srv/hermes-agent",
        "remote_hermes_home": "/srv/.hermes",
        "remote_python": "/srv/hermes-agent/venv/bin/python",
        "ref": "HEAD",
        "tail_lines": 40,
        "sync": True,
        "restart": True,
        "rollback": False,
        "system_service": True,
    }
    data.update(overrides)
    data["project_dir"].mkdir(parents=True, exist_ok=True)
    return DeployOptions(**data)


def test_run_sync_release_installs_restarts_and_verifies(monkeypatch, tmp_path):
    options = _make_options(tmp_path)
    deployer = GatewayDeployer(options)
    calls = []

    monkeypatch.setattr(deployer, "resolve_target_commit", lambda previous_state=None: "abc123")
    monkeypatch.setattr(deployer, "ensure_clean_worktree", lambda: calls.append("clean"))
    monkeypatch.setattr(deployer, "read_remote_release_state", lambda: {"current_commit": "old111", "previous_commit": "older000"})
    monkeypatch.setattr(deployer, "create_git_bundle", lambda target_commit: tmp_path / f"{target_commit}.bundle")
    monkeypatch.setattr(deployer, "sync_release_bundle", lambda bundle_path, target_commit: calls.append(("sync", bundle_path.name, target_commit)))
    monkeypatch.setattr(deployer, "ensure_system_service", lambda: calls.append("install"))
    monkeypatch.setattr(deployer, "restart_system_service", lambda: calls.append("restart"))
    monkeypatch.setattr(deployer, "verify_readiness", lambda: {"main_pid": 4242})
    monkeypatch.setattr(deployer, "write_remote_release_state", lambda state: calls.append(("state", state["previous_commit"], state["current_commit"])))

    result = deployer.run()

    assert calls == [
        "clean",
        ("sync", "abc123.bundle", "abc123"),
        "install",
        "restart",
        ("state", "old111", "abc123"),
    ]
    assert result["target_commit"] == "abc123"
    assert result["main_pid"] == 4242


def test_run_restart_only_skips_sync(monkeypatch, tmp_path):
    options = _make_options(tmp_path, sync=False)
    deployer = GatewayDeployer(options)
    calls = []

    monkeypatch.setattr(deployer, "read_remote_release_state", lambda: {"current_commit": "old111"})
    monkeypatch.setattr(deployer, "ensure_system_service", lambda: calls.append("install"))
    monkeypatch.setattr(deployer, "restart_system_service", lambda: calls.append("restart"))
    monkeypatch.setattr(deployer, "verify_readiness", lambda: {"main_pid": 7})

    result = deployer.run()

    assert calls == ["install", "restart"]
    assert result["target_commit"] == "old111"


def test_rollback_uses_previous_commit(monkeypatch, tmp_path):
    options = _make_options(tmp_path, rollback=True)
    deployer = GatewayDeployer(options)

    assert deployer.resolve_target_commit({"current_commit": "new222", "previous_commit": "old111"}) == "old111"


def test_rollback_without_previous_commit_fails(tmp_path):
    options = _make_options(tmp_path, rollback=True)
    deployer = GatewayDeployer(options)

    with pytest.raises(RuntimeError, match="No previous deployed commit"):
        deployer.resolve_target_commit({"current_commit": "new222"})


def test_dirty_worktree_rejected_for_sync(monkeypatch, tmp_path):
    options = _make_options(tmp_path)
    deployer = GatewayDeployer(options)

    monkeypatch.setattr(deployer, "run_local", lambda *args, **kwargs: " M gateway/run.py\n")

    with pytest.raises(RuntimeError, match="working tree is dirty"):
        deployer.ensure_clean_worktree()


def test_build_remote_gateway_command_runs_from_remote_dir(tmp_path):
    options = _make_options(tmp_path)
    deployer = GatewayDeployer(options)

    command = deployer.build_remote_gateway_command("restart", "--system")

    assert "cd /srv/hermes-agent" in command
    assert "/srv/hermes-agent/venv/bin/python -m hermes_cli.main gateway restart --system" in command


def test_build_remote_gateway_command_exports_remote_hermes_home(tmp_path):
    options = _make_options(tmp_path, remote_hermes_home="/srv/custom-hermes")
    deployer = GatewayDeployer(options)

    command = deployer.build_remote_gateway_command("restart", "--system")

    assert "HERMES_HOME=/srv/custom-hermes" in command


def test_build_release_state_tracks_previous_commit(tmp_path):
    options = _make_options(tmp_path)
    deployer = GatewayDeployer(options)

    state = deployer.build_release_state("old111", "new222")

    assert state["current_commit"] == "new222"
    assert state["previous_commit"] == "old111"
    assert state["source_ref"] == "HEAD"


def test_parse_args_rejects_legacy_file_arguments():
    with pytest.raises(SystemExit):
        parse_args(["gateway/run.py"])


def test_parse_args_rejects_rollback_without_sync():
    with pytest.raises(SystemExit):
        parse_args(["--rollback", "--no-sync"])


def test_parse_args_defaults_remote_python_to_remote_venv():
    options = parse_args(["--remote-dir", "/srv/hermes-agent"])

    assert options.remote_python == "/srv/hermes-agent/venv/bin/python"
