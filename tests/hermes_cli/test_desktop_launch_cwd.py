import argparse
from pathlib import Path

from hermes_cli.main import _desktop_launch_cwd


def test_desktop_launch_cwd_defaults_to_process_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert _desktop_launch_cwd(argparse.Namespace(cwd=None)) == str(tmp_path.resolve())


def test_desktop_launch_cwd_allows_explicit_override(tmp_path):
    workspace = tmp_path / "Math"
    workspace.mkdir()

    assert _desktop_launch_cwd(argparse.Namespace(cwd=str(workspace))) == str(workspace.resolve())


def test_desktop_launch_cwd_expands_user(monkeypatch, tmp_path):
    home = tmp_path / "home"
    math = home / "Math"
    math.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    assert _desktop_launch_cwd(argparse.Namespace(cwd="~/Math")) == str(Path(home, "Math").resolve())
