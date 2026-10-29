import os
from pathlib import Path

import pytest

from hermes_cli.session_recap import build_recap


@pytest.mark.parametrize("at_root", [True, False])
def test_recap_shows_files_relative_to_native_cwd(tmp_path, monkeypatch, at_root):
    cwd = Path(tmp_path.anchor) if at_root else tmp_path / "workspace"
    if not at_root:
        cwd.mkdir()
    monkeypatch.chdir(cwd)
    target = cwd / "notes" / "recap-note.txt"
    messages = [{"role": "assistant", "tool_calls": [{"function": {
        "name": "write_file", "arguments": {"path": str(target)},
    }}]}]

    recap = build_recap(messages)

    expected = os.path.join("notes", "recap-note.txt")
    assert f"  Files touched: {expected}" in recap.splitlines()


def test_recap_shortens_files_beneath_native_root_home(tmp_path, monkeypatch):
    root = Path(tmp_path.anchor)
    monkeypatch.setenv("HOME", str(root))
    monkeypatch.setenv("USERPROFILE", str(root))
    monkeypatch.chdir(tmp_path)
    target = root / "recap-note.txt"
    messages = [{"role": "assistant", "tool_calls": [{"function": {
        "name": "write_file", "arguments": {"path": str(target)},
    }}]}]

    recap = build_recap(messages)

    assert "  Files touched: ~/recap-note.txt" in recap.splitlines()
