"""Tests for hermes_cli.terminal_title — OSC tab-title emission."""
from __future__ import annotations

import io
import os
from pathlib import Path
from unittest import mock

import pytest

from hermes_cli import terminal_title


# ---------------------------------------------------------------------------
# Sanitisation
# ---------------------------------------------------------------------------

def test_clean_strips_control_characters():
    raw = "Plan\x07with\x1b]bad\x00stuff\nand newlines"
    cleaned = terminal_title._clean(raw)
    for bad in ("\x07", "\x1b", "\x00", "\n"):
        assert bad not in cleaned


def test_clean_truncates_to_max_len():
    cleaned = terminal_title._clean("x" * 500, max_len=50)
    assert len(cleaned) == 50
    assert cleaned.endswith("\u2026")


def test_clean_handles_empty():
    assert terminal_title._clean("") == ""
    assert terminal_title._clean(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Output is gated by env / TTY
# ---------------------------------------------------------------------------

def test_set_tab_title_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("HERMES_DISABLE_TAB_TITLE", "1")
    # The disable flag is honored inside _tty_stream(); confirm no stream is
    # opened. If no stream, the real _write() short-circuits without emitting.
    assert terminal_title._tty_stream() is None
    # End-to-end: nothing should be written anywhere.
    terminal_title.set_tab_title("Helm", "anything")  # must not raise


def test_set_tab_title_noop_when_term_dumb(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.delenv("HERMES_DISABLE_TAB_TITLE", raising=False)
    assert terminal_title._tty_stream() is None


def test_set_tab_title_noop_when_no_tty(monkeypatch):
    monkeypatch.delenv("HERMES_DISABLE_TAB_TITLE", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(os, "open", mock.Mock(side_effect=OSError))
    fake_stderr = io.StringIO()
    monkeypatch.setattr("sys.stderr", fake_stderr)
    # Force the open("/dev/tty") path to fail, and stderr is not a TTY.
    with mock.patch("builtins.open", side_effect=OSError):
        assert terminal_title._tty_stream() is None


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------

def test_set_tab_title_emits_osc0(monkeypatch):
    captured: list[str] = []
    monkeypatch.setattr(terminal_title, "_write", lambda s: captured.append(s))
    terminal_title.set_tab_title("Helm", "Plan Hermes console")
    assert captured == ["\x1b]0;Helm: Plan Hermes console\x07"]


def test_set_tab_title_persona_only_when_no_title(monkeypatch):
    captured: list[str] = []
    monkeypatch.setattr(terminal_title, "_write", lambda s: captured.append(s))
    terminal_title.set_tab_title("Helm", "")
    assert captured == ["\x1b]0;Helm\x07"]


def test_set_tab_title_falls_back_to_default_persona(monkeypatch):
    captured: list[str] = []
    monkeypatch.setattr(terminal_title, "_write", lambda s: captured.append(s))
    terminal_title.set_tab_title(None, "x")
    assert captured == ["\x1b]0;Hermes: x\x07"]


def test_set_tab_title_truncates_long_title(monkeypatch):
    captured: list[str] = []
    monkeypatch.setattr(terminal_title, "_write", lambda s: captured.append(s))
    terminal_title.set_tab_title("Helm", "x" * 500)
    payload = captured[0]
    # Frame + persona prefix + truncated body, all under reasonable bound.
    assert payload.startswith("\x1b]0;Helm: ")
    assert payload.endswith("\x07")
    assert len(payload) <= len("\x1b]0;") + 1 + terminal_title._MAX_TITLE_LEN + len("\x07")


def test_set_cwd_emits_osc7_with_file_url(monkeypatch, tmp_path):
    captured: list[str] = []
    monkeypatch.setattr(terminal_title, "_write", lambda s: captured.append(s))
    terminal_title.set_cwd(tmp_path)
    assert captured, "expected one OSC 7 emission"
    seq = captured[0]
    assert seq.startswith("\x1b]7;file://")
    assert str(tmp_path) in seq
    assert seq.endswith("\x07")


# ---------------------------------------------------------------------------
# Persona resolution
# ---------------------------------------------------------------------------

def test_resolve_persona_prefers_explicit_config_value(tmp_path):
    soul = tmp_path / "SOUL.md"
    soul.write_text("# Helm\nName: Helm\n", encoding="utf-8")
    assert terminal_title.resolve_persona_name(soul, "Override") == "Override"


def test_resolve_persona_reads_name_line(tmp_path):
    soul = tmp_path / "SOUL.md"
    soul.write_text("# Helm\n## Identity\n- Name: Helm\n", encoding="utf-8")
    assert terminal_title.resolve_persona_name(soul) == "Helm"


def test_resolve_persona_falls_back_to_heading(tmp_path):
    soul = tmp_path / "SOUL.md"
    soul.write_text("# Atlas\nSome body without a name field.\n",
                    encoding="utf-8")
    assert terminal_title.resolve_persona_name(soul) == "Atlas"


def test_resolve_persona_default_when_missing(tmp_path):
    assert terminal_title.resolve_persona_name(tmp_path / "missing.md") == "Hermes"


def test_resolve_persona_default_when_empty(tmp_path):
    soul = tmp_path / "SOUL.md"
    soul.write_text("", encoding="utf-8")
    assert terminal_title.resolve_persona_name(soul) == "Hermes"
