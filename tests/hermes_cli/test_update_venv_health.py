"""Live lifecycle discovery still identifies Windows venv processes.

These tests exercise the retained scan, not the retired update admission gate.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import main as cli_main
from hermes_cli import update_cmd_windows


def _proc(pid: int, exe: str, name: str, cmdline: list[str] | None = None, cwd: str = ""):
    proc = MagicMock()
    proc.info = {
        "pid": pid,
        "exe": exe,
        "name": name,
    }
    proc.cmdline.return_value = cmdline or []
    proc.cwd.return_value = cwd
    return proc


@pytest.mark.platforms("windows")
def test_detect_venv_python_excludes_self_and_ancestors(tmp_path):
    import os as _os

    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    parent = MagicMock()
    parent.pid = 555
    me = MagicMock()
    me.parents.return_value = [parent]
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter(
            [
                _proc(_os.getpid(), venv_py, "python.exe"),
                _proc(555, venv_py, "hermes.exe"),
            ]
        ),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        assert update_cmd_windows._detect_venv_python_processes() == []


@pytest.mark.platforms("windows")
def test_detect_venv_python_prefetches_only_cheap_process_fields(tmp_path):
    venv_py = str(tmp_path / "venv" / "Scripts" / "python.exe")
    holder = _proc(101, venv_py, "python.exe", [venv_py, "-m", "hermes_cli.main", "serve"])
    unrelated = _proc(102, r"C:\Program Files\Browser\browser.exe", "browser.exe")
    unrelated.cmdline.side_effect = AssertionError("unrelated cmdline must stay lazy")
    unrelated.cwd.side_effect = AssertionError("unrelated cwd must stay lazy")
    attrs_seen = []
    me = MagicMock()
    me.parents.return_value = []

    def process_iter(attrs):
        attrs_seen.append(attrs)
        return iter([unrelated, holder])

    fake_psutil = types.SimpleNamespace(
        process_iter=process_iter,
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = update_cmd_windows._detect_venv_python_processes()

    assert attrs_seen == [["pid", "exe", "name"]]
    assert [match[0] for match in matches] == [101]
    holder.cmdline.assert_called_once_with()
    holder.cwd.assert_not_called()
    unrelated.cmdline.assert_not_called()
    unrelated.cwd.assert_not_called()


@pytest.mark.platforms("windows")
def test_detect_venv_python_keeps_external_interpreter_fallback(tmp_path):
    external = _proc(
        103,
        r"C:\Python311\python.exe",
        "python.exe",
        ["python.exe", "-m", "hermes_cli.main", "serve"],
        str(tmp_path),
    )
    me = MagicMock()
    me.parents.return_value = []
    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: iter([external]),
        Process=lambda *a, **k: me,
    )
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.dict(
        sys.modules, {"psutil": fake_psutil}
    ):
        matches = update_cmd_windows._detect_venv_python_processes()

    assert [match[0] for match in matches] == [103]
    external.cmdline.assert_called_once_with()
    external.cwd.assert_called_once_with()
