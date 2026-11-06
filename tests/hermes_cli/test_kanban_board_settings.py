"""Unit tests for ``hermes_cli.kanban_board_settings`` (per-board resolution).

Pure-function coverage with a fake ``kanban`` config dict — no filesystem, no
config load. The pin-collapse case needs the env pin + a board path, so it uses
a temp ``HERMES_HOME`` and ``pin_first_board_resolution`` to reproduce the
machine-flow topology.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from hermes_cli import kanban_board_settings as kbs


def _cfg(boards):
    return {"boards": boards}


# --- normalize_board_slug ---------------------------------------------------

def test_normalize_board_slug_lowercases():
    assert kbs.normalize_board_slug("TSA-Mgmt") == "tsa-mgmt"


@pytest.mark.parametrize("bad", ["", "  ", "has space", "a/b", "UPPER OK?"])
def test_normalize_board_slug_invalid_is_none(bad):
    assert kbs.normalize_board_slug(bad) is None


def test_normalize_board_slug_none():
    assert kbs.normalize_board_slug(None) is None


# --- board_override ---------------------------------------------------------

def test_board_override_reads_stripped_value():
    cfg = _cfg({"tsa-mgmt": {"default_assignee": "  tsa-worker  "}})
    assert kbs.board_override(cfg, "default_assignee", "tsa-mgmt") == "tsa-worker"


def test_board_override_unknown_board_is_none():
    cfg = _cfg({"tsa-mgmt": {"default_assignee": "x"}})
    assert kbs.board_override(cfg, "default_assignee", "svs") is None


def test_board_override_empty_string_is_unset():
    cfg = _cfg({"tsa-mgmt": {"default_assignee": ""}})
    assert kbs.board_override(cfg, "default_assignee", "tsa-mgmt") is None


def test_board_override_without_boards_key():
    assert kbs.board_override({}, "default_assignee", "tsa-mgmt") is None
    assert kbs.board_override("not-a-dict", "default_assignee", "tsa-mgmt") is None


def test_board_override_case_insensitive_slug_and_key():
    cfg = _cfg({"TSA-Mgmt": {"Default_Assignee": "worker"}})
    assert kbs.board_override(cfg, "default_assignee", "tsa-mgmt") == "worker"


def test_board_override_non_string_value_warns_and_ignores(caplog):
    cfg = _cfg({"tsa-mgmt": {"default_assignee": ["worker"]}})
    with caplog.at_level(logging.WARNING):
        assert kbs.board_override(cfg, "default_assignee", "tsa-mgmt") is None
    assert any("expected a string" in rec.message for rec in caplog.records)


def test_board_override_invalid_slug_warns_and_ignores(caplog):
    cfg = _cfg({"a/b": {"default_assignee": "worker"}})
    with caplog.at_level(logging.WARNING):
        assert kbs.board_override(cfg, "default_assignee", "a/b") is None
    assert any("invalid board slug" in rec.message for rec in caplog.records)


def test_board_override_missing_key():
    cfg = _cfg({"tsa-mgmt": {"orchestrator_profile": "planner"}})
    assert kbs.board_override(cfg, "default_assignee", "tsa-mgmt") is None


# --- malformed configured board keys (review #135651, item 3) ----------------

def test_board_override_malformed_configured_key_ignored(caplog):
    """A hand-written key that can never name a board is skipped, and a sibling
    valid board is unaffected."""
    kbs._MALFORMED_BOARD_KEY_WARNED.clear()
    cfg = _cfg({"bad/slug": {"default_assignee": "worker"}, "tsa-mgmt": {"default_assignee": "ok"}})
    with caplog.at_level(logging.WARNING):
        assert kbs.board_override(cfg, "default_assignee", "tsa-mgmt") == "ok"
    assert any("malformed board key" in rec.message and "bad/slug" in rec.message for rec in caplog.records)


def test_board_override_malformed_key_warns_once(caplog):
    kbs._MALFORMED_BOARD_KEY_WARNED.clear()
    cfg = _cfg({"bad/slug": {"default_assignee": "worker"}})
    with caplog.at_level(logging.WARNING):
        kbs.board_override(cfg, "default_assignee", "tsa-mgmt")
        kbs.board_override(cfg, "default_assignee", "tsa-mgmt")
    assert sum("malformed board key" in rec.message for rec in caplog.records) == 1


def test_board_override_non_string_configured_key_warns(caplog):
    kbs._MALFORMED_BOARD_KEY_WARNED.clear()
    cfg = {"boards": {7: {"default_assignee": "worker"}}}
    with caplog.at_level(logging.WARNING):
        assert kbs.board_override(cfg, "default_assignee", "tsa-mgmt") is None
    assert any("malformed board key" in rec.message for rec in caplog.records)


# --- effective_setting ------------------------------------------------------

def test_effective_setting_board_beats_fallback():
    cfg = _cfg({"tsa-mgmt": {"default_assignee": "board-worker"}})
    assert kbs.effective_setting(cfg, "default_assignee", "tsa-mgmt", fallback="global") == "board-worker"


def test_effective_setting_unset_board_uses_caller_global():
    cfg = _cfg({"svs": {"default_assignee": "svs-worker"}})
    assert kbs.effective_setting(cfg, "default_assignee", "tsa-mgmt", fallback="global") == "global"


def test_effective_setting_never_reads_global_from_cfg():
    """The daemon passes no fallback; a cfg-level global must NOT leak in."""
    cfg = {"default_assignee": "global-from-cfg", "boards": {}}
    assert kbs.effective_setting(cfg, "default_assignee", "tsa-mgmt", fallback=None) is None


def test_effective_setting_no_board_uses_fallback():
    cfg = _cfg({"tsa-mgmt": {"default_assignee": "board-worker"}})
    assert kbs.effective_setting(cfg, "default_assignee", None, fallback="global") == "global"


def test_effective_setting_blank_fallback_is_none():
    assert kbs.effective_setting({}, "default_assignee", "tsa-mgmt", fallback="  ") is None


def test_keys_tuple_is_the_two_v1_knobs():
    assert kbs.KANBAN_BOARD_SETTING_KEYS == ("orchestrator_profile", "default_assignee")


# --- pin collapse (HERMES_KANBAN_DB) ---------------------------------------

def test_pin_collapse_suppresses_board_override(tmp_path, monkeypatch, caplog):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    pin = tmp_path / "pinned.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pin))

    from hermes_cli import kanban_db

    cfg = _cfg({"tsa-mgmt": {"default_assignee": "board-worker"}})
    with kanban_db.pin_first_board_resolution(), caplog.at_level(logging.WARNING):
        assert kbs.board_pin_suppressed("tsa-mgmt") is True
        assert kbs.effective_setting(cfg, "default_assignee", "tsa-mgmt", fallback="global") == "global"
    assert any("HERMES_KANBAN_DB pins" in rec.message for rec in caplog.records)


def test_pin_collapse_warns_once(tmp_path, monkeypatch, caplog):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "pinned.db"))
    kbs._PIN_SUPPRESSED_WARNED.clear()

    from hermes_cli import kanban_db

    with kanban_db.pin_first_board_resolution(), caplog.at_level(logging.WARNING):
        kbs.board_pin_suppressed("tsa-mgmt")
        kbs.board_pin_suppressed("tsa-mgmt")
    assert sum("HERMES_KANBAN_DB pins" in rec.message for rec in caplog.records) == 1


def test_no_pin_means_no_suppression(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    assert kbs.board_pin_suppressed("tsa-mgmt") is False
