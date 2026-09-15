from pathlib import Path

from hermes_cli.oneshot import run_oneshot


def test_local_oneshot_pins_tools_to_launch_directory(monkeypatch, tmp_path):
    stale = tmp_path / "stale"
    stale.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setenv("TERMINAL_CWD", str(stale))

    observed = {}

    def fake_run_agent(*_args, **_kwargs):
        observed["cwd"] = Path(__import__("os").environ["TERMINAL_CWD"])
        return "ok", {}

    monkeypatch.setattr("hermes_cli.oneshot._run_agent", fake_run_agent)

    assert run_oneshot("do the work") == 0
    assert observed["cwd"] == workspace


def test_remote_oneshot_preserves_backend_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CWD", "/workspace")

    observed = {}

    def fake_run_agent(*_args, **_kwargs):
        observed["cwd"] = __import__("os").environ["TERMINAL_CWD"]
        return "ok", {}

    monkeypatch.setattr("hermes_cli.oneshot._run_agent", fake_run_agent)

    assert run_oneshot("do the work") == 0
    assert observed["cwd"] == "/workspace"
