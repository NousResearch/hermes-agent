"""`hermes config edit` runs a $EDITOR that carries flags as argv, not as one executable name (#156214)."""
import pytest

from hermes_cli import config_edit


@pytest.fixture
def run_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(config_edit.subprocess, "run", lambda cmd, **k: calls.append(cmd))
    return calls


@pytest.mark.parametrize("editor, argv", [
    ("nano", ["nano"]),
    ("omarchy-launch-editor --inline", ["omarchy-launch-editor", "--inline"]),
    ('"/opt/My Editor/bin/edit" --wait', ["/opt/My Editor/bin/edit", "--wait"]),
])
def test_editor_with_flags_runs_as_argv(tmp_path, monkeypatch, run_calls, editor, argv):
    monkeypatch.setenv("EDITOR", editor)
    monkeypatch.setattr(config_edit.sys, "platform", "linux")
    config_edit.edit_config()
    assert run_calls == [[*argv, str(tmp_path / "config.yaml")]]


@pytest.mark.parametrize("editor", ['vim "unbalanced', "   "])
def test_unrunnable_editor_opens_nothing(monkeypatch, capsys, run_calls, editor):
    monkeypatch.setenv("EDITOR", editor)
    monkeypatch.setattr(config_edit.sys, "platform", "linux")
    config_edit.edit_config()
    assert run_calls == []
    assert "Cannot run $EDITOR" in capsys.readouterr().out


def test_windows_editor_path_stays_whole(tmp_path, monkeypatch, run_calls):
    monkeypatch.setenv("EDITOR", r"C:\Windows\notepad.exe")
    monkeypatch.setattr(config_edit.sys, "platform", "win32")
    config_edit.edit_config()
    assert run_calls == [[r"C:\Windows\notepad.exe", str(tmp_path / "config.yaml")]]
