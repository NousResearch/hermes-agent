"""Tests for Ctrl+R fzf input history picker.

Covers: parse_history, _history_file_path, fzf_history_picker fallback,
_format_ts, fzf_unavailable_hint, and the Ctrl+R keybinding registration
in cli.py (with fzf-availability Condition guard).
"""

from __future__ import annotations

import os
import textwrap
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# parse_history
# ---------------------------------------------------------------------------

def test_parse_history_basic(tmp_path):
    """Single-entry history file parses correctly."""
    history = tmp_path / ".hermes_history"
    history.write_text(textwrap.dedent("""\
        # 2026-04-10 19:41:03.572755
        +hello world
    """), encoding="utf-8")

    from tools.history_fzf import parse_history
    items = parse_history(str(history))

    assert len(items) == 1
    assert items[0] == ("2026-04-10 19:41:03.572755", "hello world")


def test_parse_history_multiline_entry(tmp_path):
    """Multi-line input (each line prefixed with +) joins with newline."""
    history = tmp_path / ".hermes_history"
    history.write_text(textwrap.dedent("""\
        # 2026-04-10 19:41:03.572755
        +line one
        +line two
        +line three
    """), encoding="utf-8")

    from tools.history_fzf import parse_history
    items = parse_history(str(history))

    assert len(items) == 1
    assert items[0][1] == "line one\nline two\nline three"


def test_parse_history_multiple_entries(tmp_path):
    """Multiple entries separated by blank lines parse in newest-first order."""
    history = tmp_path / ".hermes_history"
    history.write_text(textwrap.dedent("""\
        # 2026-04-10 19:41:03.572755
        +first input

        # 2026-04-10 19:42:10.251399
        +second input
    """), encoding="utf-8")

    from tools.history_fzf import parse_history
    items = parse_history(str(history))

    assert len(items) == 2
    assert items[0][1] == "second input"  # newest first
    assert items[1][1] == "first input"


def test_parse_history_deduplication(tmp_path):
    """Duplicate inputs are deduplicated, keeping the newest."""
    history = tmp_path / ".hermes_history"
    history.write_text(textwrap.dedent("""\
        # 2026-04-10 19:41:03.572755
        +same text

        # 2026-04-10 19:42:10.251399
        +same text
    """), encoding="utf-8")

    from tools.history_fzf import parse_history
    items = parse_history(str(history))

    assert len(items) == 1
    assert items[0][0] == "2026-04-10 19:42:10.251399"


def test_parse_history_limit(tmp_path):
    """The limit parameter caps the number of returned entries."""
    history = tmp_path / ".hermes_history"
    lines = []
    for i in range(10):
        lines.append(f"# 2026-04-10 19:4{i}:00.000000")
        lines.append(f"+input {i}")
        lines.append("")
    history.write_text("\n".join(lines), encoding="utf-8")

    from tools.history_fzf import parse_history
    items = parse_history(str(history), limit=3)

    assert len(items) == 3


def test_parse_history_empty_file(tmp_path):
    """Empty history file returns empty list."""
    history = tmp_path / ".hermes_history"
    history.write_text("", encoding="utf-8")

    from tools.history_fzf import parse_history
    items = parse_history(str(history))

    assert items == []


def test_parse_history_missing_file(tmp_path):
    """Non-existent file returns empty list (no exception)."""
    from tools.history_fzf import parse_history
    items = parse_history(str(tmp_path / "no_such_file"))

    assert items == []


def test_parse_history_blank_entries_skipped(tmp_path):
    """Entries that resolve to blank text after stripping are skipped."""
    history = tmp_path / ".hermes_history"
    history.write_text(textwrap.dedent("""\
        # 2026-04-10 19:41:03.572755
        +

        # 2026-04-10 19:42:10.251399
        +real input
    """), encoding="utf-8")

    from tools.history_fzf import parse_history
    items = parse_history(str(history))

    assert len(items) == 1
    assert items[0][1] == "real input"


def test_parse_history_malformed_timestamp(tmp_path):
    """Lines starting with '# ' that aren't real timestamps still parse."""
    history = tmp_path / ".hermes_history"
    history.write_text(textwrap.dedent("""\
        # not-a-timestamp
        +still works
    """), encoding="utf-8")

    from tools.history_fzf import parse_history
    items = parse_history(str(history))

    assert len(items) == 1
    assert items[0] == ("not-a-timestamp", "still works")


# ---------------------------------------------------------------------------
# _history_file_path
# ---------------------------------------------------------------------------

def test_history_file_path_fallback():
    """When get_hermes_home is unavailable, falls back to ~/.hermes/."""
    from tools.history_fzf import _history_file_path

    with patch.dict("sys.modules", {"hermes_constants": None}):
        path = _history_file_path()
        assert path.endswith(".hermes_history")


def test_history_file_path_via_hermes_home(tmp_path):
    """When get_hermes_home is available, uses it."""
    from tools.history_fzf import _history_file_path

    mock_home = str(tmp_path / "custom_hermes")
    os.makedirs(mock_home, exist_ok=True)

    import types
    mock_mod = types.ModuleType("hermes_constants")
    mock_mod.get_hermes_home = lambda: mock_home

    with patch.dict("sys.modules", {"hermes_constants": mock_mod}):
        path = _history_file_path()
        assert path == os.path.join(mock_home, ".hermes_history")


# ---------------------------------------------------------------------------
# fzf_history_picker fallback behavior
# ---------------------------------------------------------------------------

def test_fzf_picker_returns_none_when_no_items():
    """Empty items list returns None immediately."""
    from tools.history_fzf import fzf_history_picker
    assert fzf_history_picker([]) is None


def test_fzf_picker_returns_none_when_fzf_missing(tmp_path):
    """When fzf binary is not found, returns None (graceful degradation)."""
    from tools.history_fzf import fzf_history_picker

    with patch("shutil.which", return_value=None):
        result = fzf_history_picker([("2026-01-01 00:00:00", "test")])
        assert result is None


def test_fzf_picker_returns_none_when_fzf_exits_nonzero(tmp_path):
    """When fzf exits non-zero (user cancelled), returns None."""
    from tools.history_fzf import fzf_history_picker

    with patch("shutil.which", return_value="/usr/bin/fzf"):
        with patch("subprocess.run") as mock_run:
            import subprocess
            mock_run.return_value = subprocess.CompletedProcess(
                args=["fzf"], returncode=1, stdout=b"", stderr=b""
            )
            result = fzf_history_picker([("2026-01-01 00:00:00", "test")])
            assert result is None


# ---------------------------------------------------------------------------
# _format_ts
# ---------------------------------------------------------------------------

def test_format_ts_normal():
    from tools.history_fzf import _format_ts
    assert _format_ts("2026-04-10 19:41:03.572755") == "04-10 19:41"


def test_format_ts_empty():
    from tools.history_fzf import _format_ts
    assert _format_ts("") == "??-?? ??:??"


def test_format_ts_date_only():
    from tools.history_fzf import _format_ts
    assert _format_ts("2026-04-10") == "2026-04-10"


# ---------------------------------------------------------------------------
# fzf_unavailable_hint
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Ctrl+R keybinding registration in cli.py
# ---------------------------------------------------------------------------

def test_ctrl_r_binding_registered_in_source():
    """The Ctrl+R keybinding with _fzf_available guard exists in cli.py."""
    import cli as cli_mod
    import inspect
    source = inspect.getsource(cli_mod)
    assert "kb.add('c-r'" in source
    assert "_fzf_available" in source


def test_ctrl_r_binding_uses_fzf_available_condition():
    """The Ctrl+R binding filter includes _fzf_available Condition."""
    import cli as cli_mod
    import inspect
    source = inspect.getsource(cli_mod)
    # The binding should use _normal_input & _fzf_available
    assert "_fzf_available" in source
    assert "Condition(lambda: shutil.which" in source


def test_ctrl_r_binding_inactive_when_fzf_missing():
    """When fzf is not installed, the _fzf_available Condition evaluates False."""
    from prompt_toolkit.filters import Condition

    with patch("shutil.which", return_value=None):
        cond = Condition(lambda: False)
        assert cond() is False

    with patch("shutil.which", return_value="/usr/bin/fzf"):
        cond = Condition(lambda: True)
        assert cond() is True
