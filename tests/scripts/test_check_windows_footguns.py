"""Tests for the console-binary-spawn rule in scripts/check-windows-footguns.py.

The checker is a hyphenated script, so it's loaded by path. Fixtures are passed
to ``find_console_spawn_matches`` as source strings (the test file itself is in
the checker's EXCLUDED_FILES so its fixtures don't self-trigger the rule).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check-windows-footguns.py"
_spec = importlib.util.spec_from_file_location("check_windows_footguns", _SCRIPT)
cwf = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
# Register before exec: the module's @dataclass uses a string annotation, and
# dataclass resolution looks the module up in sys.modules by name.
sys.modules["check_windows_footguns"] = cwf
_spec.loader.exec_module(cwf)


def _hit_lines(src: str) -> list[int]:
    return [lineno for lineno, _line, _fg in cwf.find_console_spawn_matches(src)]


def test_flags_single_line_raw_spawn():
    assert _hit_lines('subprocess.run(["git", "status"])\n') == [1]


def test_flags_multiline_raw_spawn_at_call_start():
    src = 'subprocess.run(\n    ["gh", "auth", "token"],\n    capture_output=True,\n)\n'
    assert _hit_lines(src) == [1]


def test_flags_popen_and_check_output():
    assert _hit_lines('subprocess.Popen(["gh", "pr", "list"])\n') == [1]
    assert _hit_lines('subprocess.check_output(["git", "rev-parse"])\n') == [1]


def test_ignores_explicit_creationflags():
    src = 'subprocess.run(["git"], creationflags=windows_hide_flags())\n'
    assert _hit_lines(src) == []


def test_ignores_hidden_helper():
    assert _hit_lines('run_hidden(["git", "status"])\n') == []
    assert _hit_lines('popen_hidden(["gh", "auth", "token"])\n') == []


def test_ignores_inline_suppression_after_paren():
    assert _hit_lines('subprocess.run(["git"])  # windows-footgun: ok\n') == []


def test_ignores_suppression_on_multiline_close():
    src = 'subprocess.run(\n    ["git", "diff"],\n)  # windows-footgun: ok\n'
    assert _hit_lines(src) == []


def test_ignores_variable_program():
    assert _hit_lines("subprocess.run(cmd, capture_output=True)\n") == []


def test_ignores_non_console_binary():
    assert _hit_lines('subprocess.run(["python", "-c", "x"])\n') == []


def test_ignores_docstring_example():
    assert _hit_lines('"""\nsubprocess.run(["git", "status"])\n"""\n') == []


def test_strips_path_and_exe_from_argv0():
    src = 'subprocess.run(["C:\\\\Program Files\\\\Git\\\\bin\\\\git.exe", "x"])\n'
    assert _hit_lines(src) == [1]


def test_console_rule_excluded_from_all_line_run():
    # The repo-wide --all gate must NOT include this contextual rule (it would
    # flag dozens of legitimate CLI-interactive call sites).
    assert cwf.CONSOLE_SPAWN_FOOTGUN not in cwf.FOOTGUNS
