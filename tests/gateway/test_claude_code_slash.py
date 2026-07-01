from pathlib import Path
from unittest import mock

from gateway.claude_code_command import find_repo, launch_claude_code, workspace_roots
from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command


def test_cc_registered_as_gateway_only_command():
    cmd = resolve_command("cc")

    assert cmd is not None
    assert cmd.name == "cc"
    assert cmd.gateway_only is True
    assert "cc" in GATEWAY_KNOWN_COMMANDS


def test_workspace_roots_use_gateway_claude_code_config(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    missing = tmp_path / "missing"
    config = {
        "gateway": {
            "claude_code": {
                "workspace_roots": [str(root), str(missing)],
            }
        }
    }

    assert workspace_roots(config) == [root.resolve()]


def test_find_repo_by_name_under_configured_root(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    repo = root / "demo"
    repo.mkdir(parents=True)
    config = {"gateway": {"claude_code": {"workspace_roots": [str(root)]}}}

    monkeypatch.setattr(
        "gateway.claude_code_command._git_root",
        lambda path: repo.resolve() if Path(path).name == "demo" else None,
    )

    found, error = find_repo("demo", config)

    assert error is None
    assert found == repo.resolve()


def test_find_repo_reports_ambiguous_name(tmp_path, monkeypatch):
    one = tmp_path / "one" / "demo"
    two = tmp_path / "two" / "demo"
    one.mkdir(parents=True)
    two.mkdir(parents=True)
    config = {"gateway": {"claude_code": {"workspace_roots": [str(tmp_path)]}}}

    def fake_git_root(path):
        path = Path(path)
        if path == one:
            return one.resolve()
        if path == two:
            return two.resolve()
        return None

    monkeypatch.setattr("gateway.claude_code_command._git_root", fake_git_root)

    found, error = find_repo("demo", config)

    assert found is None
    assert error is not None
    assert "ambiguous" in error
    assert str(one.resolve()) in error
    assert str(two.resolve()) in error


def test_launch_claude_code_starts_tmux_with_remote_control(tmp_path, monkeypatch):
    repo = tmp_path / "demo"
    repo.mkdir()
    monkeypatch.setattr("gateway.claude_code_command._git_root", lambda path: repo.resolve())
    monkeypatch.setattr("gateway.claude_code_command.shutil.which", lambda name: f"/usr/bin/{name}")

    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["tmux", "has-session"]:
            return mock.Mock(returncode=1, stdout="", stderr="")
        if args[:3] == ["tmux", "new-session", "-d"]:
            return mock.Mock(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr("gateway.claude_code_command.subprocess.run", fake_run)

    result = launch_claude_code(repo, "demo")

    assert result.tmux_session == "cc-demo"
    assert result.remote_name == "demo"
    assert result.repo_path == repo.resolve()
    new_session = calls[-1]
    assert new_session[:3] == ["tmux", "new-session", "-d"]
    assert "-c" in new_session
    assert str(repo.resolve()) in new_session
    assert new_session[-1] == "claude --dangerously-skip-permissions --remote-control demo"
