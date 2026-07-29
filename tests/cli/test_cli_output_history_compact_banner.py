from collections import deque
import os
from types import SimpleNamespace
from typing import Any, cast

import pytest

import cli as cli_mod
from cli import HermesCLI


class _HistoryConsole:
    def clear(self):
        pass

    def print(self, *args, **kwargs):
        text = " ".join(str(arg) for arg in args)
        cli_mod._cprint(text)


class _NullOutput:
    def erase_screen(self):
        pass

    def cursor_goto(self, *_args):
        pass

    def flush(self):
        pass


@pytest.fixture(autouse=True)
def restore_output_history(monkeypatch):
    original = {
        "_OUTPUT_HISTORY_ENABLED": cli_mod._OUTPUT_HISTORY_ENABLED,
        "_OUTPUT_HISTORY_REPLAYING": cli_mod._OUTPUT_HISTORY_REPLAYING,
        "_OUTPUT_HISTORY_SUPPRESSED": cli_mod._OUTPUT_HISTORY_SUPPRESSED,
        "_OUTPUT_HISTORY_MAX_LINES": cli_mod._OUTPUT_HISTORY_MAX_LINES,
        "_OUTPUT_HISTORY": cli_mod._OUTPUT_HISTORY,
    }
    monkeypatch.setattr(cli_mod, "_OUTPUT_HISTORY_ENABLED", True)
    monkeypatch.setattr(cli_mod, "_OUTPUT_HISTORY_REPLAYING", False)
    monkeypatch.setattr(cli_mod, "_OUTPUT_HISTORY_SUPPRESSED", False)
    monkeypatch.setattr(cli_mod, "_OUTPUT_HISTORY_MAX_LINES", 200)
    monkeypatch.setattr(cli_mod, "_OUTPUT_HISTORY", deque(maxlen=200))
    monkeypatch.setattr(cli_mod, "_pt_print", lambda *args, **kwargs: None)
    yield
    for name, value in original.items():
        setattr(cli_mod, name, value)


def _history_text() -> str:
    rendered: list[str] = []
    for entry in cli_mod._OUTPUT_HISTORY:
        if callable(entry):
            value = entry()
            if isinstance(value, str):
                rendered.extend(value.splitlines())
            else:
                rendered.extend(str(line) for line in cast(Any, value))
        else:
            rendered.append(str(entry))
    return "\n".join(rendered)


def _make_cli(*, app_active: bool = False) -> HermesCLI:
    cli_obj = cast(Any, HermesCLI.__new__(HermesCLI))
    cli_obj.console = _HistoryConsole()
    cli_obj.compact = True
    cli_obj.agent = None
    cli_obj.enabled_toolsets = []
    cli_obj.session_id = "test-session"
    cli_obj.model = "test-model"
    cli_obj.provider = "test-provider"
    cli_obj.base_url = ""
    cli_obj._app = SimpleNamespace(output=_NullOutput()) if app_active else None
    cli_obj._pending_resume_sessions = None
    cli_obj._confirm_destructive_slash = lambda *args, **kwargs: True
    cli_obj.new_session = lambda *args, **kwargs: None
    cli_obj._show_status = lambda: None
    cli_obj._show_tool_availability_warnings = lambda: None
    return cli_obj


def test_show_banner_compact_skips_output_history_but_later_output_records(monkeypatch):
    marker = "COMPACT_SHOW_BANNER_HISTORY_MARKER"
    ordinary = "ordinary output after compact show banner"
    cli_obj = _make_cli()

    monkeypatch.setattr(cli_mod, "_build_compact_banner", lambda: marker)
    monkeypatch.setattr(
        cli_mod.shutil,
        "get_terminal_size",
        lambda *args, **kwargs: os.terminal_size((120, 24)),
    )

    cli_obj.show_banner()

    assert marker not in _history_text()

    cli_obj._console_print(ordinary)

    history = _history_text()
    assert marker not in history
    assert ordinary in history


def test_clear_tui_compact_clears_history_and_skips_banner_but_later_output_records(monkeypatch):
    marker = "COMPACT_CLEAR_TUI_HISTORY_MARKER"
    ordinary = "ordinary output after compact tui clear"
    cli_obj = _make_cli(app_active=True)
    cli_mod._record_output_history("old history entry")

    monkeypatch.setattr(cli_mod, "_build_compact_banner", lambda: marker)
    monkeypatch.setattr(
        cli_mod.shutil,
        "get_terminal_size",
        lambda *args, **kwargs: os.terminal_size((120, 24)),
    )
    monkeypatch.setattr(cli_mod, "get_random_tip", lambda: "test tip", raising=False)

    cli_obj.process_command("/clear")

    history_after_clear = _history_text()
    assert "old history entry" not in history_after_clear
    assert marker not in history_after_clear

    cli_mod._cprint(ordinary)

    history = _history_text()
    assert "old history entry" not in history
    assert marker not in history
    assert ordinary in history
