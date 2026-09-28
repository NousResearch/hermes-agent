from pathlib import Path
from types import SimpleNamespace

from hermes_cli import uninstall


def test_dry_run_prints_plan_without_mutating(monkeypatch, tmp_path, capsys):
    project_root = tmp_path / "hermes-agent"
    hermes_home = tmp_path / ".hermes"
    project_root.mkdir()
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("model: {}\n")

    called = False

    def _fail_if_called(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(uninstall, "get_project_root", lambda: project_root)
    monkeypatch.setattr(uninstall, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(uninstall, "_is_default_hermes_home", lambda home: False)
    monkeypatch.setattr(uninstall, "_discover_named_profiles", lambda: [])
    monkeypatch.setattr(uninstall, "_perform_uninstall", _fail_if_called)

    uninstall.run_uninstall(SimpleNamespace(dry_run=True, yes=True, full=True))

    output = capsys.readouterr().out
    assert called is False
    assert "Dry run" in output
    assert str(project_root) in output
    assert str(hermes_home) in output
    assert project_root.exists()
    assert hermes_home.exists()


def test_build_uninstall_parser_accepts_dry_run():
    import argparse
    from hermes_cli.subcommands.uninstall import build_uninstall_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_uninstall_parser(subparsers, cmd_uninstall=lambda args: args)

    args = parser.parse_args(["uninstall", "--dry-run", "--full"])

    assert args.dry_run is True
    assert args.full is True


def test_uninstall_profile_uses_python_m_invocation(monkeypatch, tmp_path):
    """Regression for #76705: _uninstall_profile must use sys.executable -m hermes_cli.main
    so a broken/half-removed console script shim does not break profile teardown."""
    runs = []

    def _fake_run(cmd, **kwargs):
        runs.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(uninstall.subprocess, "run", _fake_run)
    profile = SimpleNamespace(name="worker1", path=tmp_path / "worker1")
    (tmp_path / "worker1").mkdir()

    uninstall._uninstall_profile(profile)

    assert len(runs) >= 2
    for cmd in runs:
        assert cmd[0] == uninstall.sys.executable
        assert cmd[1:3] == ["-m", "hermes_cli.main"]
        assert "--profile" in cmd
        assert cmd[cmd.index("--profile") + 1] == "worker1"
