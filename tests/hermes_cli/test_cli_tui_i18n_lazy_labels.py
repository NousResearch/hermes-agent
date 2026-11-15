"""TUI approval labels and modal hints are catalog lookups at render time, not import-time text.

A language pack registered (or ``display.language`` changed) after ``hermes_cli.cli_tui_mixin``
was imported must still reach the approval panel and the hint row; a module-level dict of
English strings would freeze the first language forever.
"""
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import i18n
from cli import HermesCLI


@pytest.fixture
def swapped_catalog(tmp_path, monkeypatch):
    """Point the catalog loader at a temp ``en.yaml`` whose TUI copy differs from the bundled one."""
    fake = tmp_path / "locales"
    fake.mkdir()
    (fake / "en.yaml").write_text(
        "cli:\n"
        "  tui:\n"
        "    approval_once: 'ONCE-SWAPPED'\n"
        "    approval_deny: 'DENY-SWAPPED'\n"
        "    approval_title: 'TITLE-SWAPPED'\n"
        "    hint_approval: 'HINT-SWAPPED'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(i18n, "_locales_dir", lambda: fake)
    i18n.reset_language_cache()
    try:
        yield
    finally:
        i18n.reset_language_cache()


def _cli_with_approval():
    cli = HermesCLI.__new__(HermesCLI)
    cli._approval_lock = threading.Lock()
    cli._sudo_state = cli._secret_state = cli._slash_confirm_state = cli._connection_state = None
    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._command_running = False
    cli._approval_deadline = time.monotonic() + 30
    cli._approval_state = {
        "command": "rm -rf build",
        "description": "recursive delete",
        "choices": ["once", "deny"],
        "selected": 0,
        "show_full": False,
        "response_queue": queue.Queue(),
    }
    cli._app = SimpleNamespace(invalidate=MagicMock())
    return cli


def test_approval_panel_and_hint_follow_the_live_catalog(swapped_catalog, monkeypatch):
    import shutil

    monkeypatch.setattr("cli.shutil.get_terminal_size", lambda *_a, **_k: shutil.os.terminal_size((100, 40)))
    cli = _cli_with_approval()

    rendered = "".join(text for _style, text in cli._get_approval_display_fragments())
    assert "TITLE-SWAPPED" in rendered
    assert "1. ONCE-SWAPPED" in rendered and "2. DENY-SWAPPED" in rendered
    assert "Allow once" not in rendered

    hint = "".join(text for _style, text in cli._tui_hint_text())
    assert hint.startswith("  HINT-SWAPPED")
    assert hint.rstrip().endswith("s)")  # countdown row still appended after the localized hint


def test_unknown_choice_id_renders_verbatim():
    from hermes_cli.cli_tui_mixin import _approval_choice_label

    assert _approval_choice_label("custom-id") == "custom-id"
    assert _approval_choice_label("deny") == i18n.t("cli.tui.approval_deny")
