"""Tests for Telegram update-ID deduplication (persistent re-delivery guard).

The adapter tracks processed ``update_id`` values in memory and on disk so
Telegram re-delivered updates (webhook/polling retries, reconnect windows) are
not processed twice. These tests exercise the guard + persistence helpers
directly; they use ``object.__new__`` (like the group-gating tests) so no real
adapter init or network is involved.
"""

import json
import os

import pytest

from plugins.platforms.telegram.adapter import TelegramAdapter


def _bare_adapter(tmp_path, hermes_home):
    """Build an adapter with the dedup helpers wired but no full init."""
    adapter = object.__new__(TelegramAdapter)
    adapter._processed_update_ids = set()
    adapter._processed_update_ids_path = hermes_home / ".telegram_processed_updates.json"
    return adapter


def test_is_processed_false_when_empty(tmp_path):
    adapter = _bare_adapter(tmp_path, tmp_path)
    assert adapter._is_processed_update(1001) is False
    assert adapter._is_processed_update(None) is False


def test_mark_then_is_processed(tmp_path):
    adapter = _bare_adapter(tmp_path, tmp_path)
    adapter._mark_update_processed(1001)
    assert adapter._is_processed_update(1001) is True
    assert adapter._is_processed_update(1002) is False


def test_mark_none_is_noop(tmp_path):
    adapter = _bare_adapter(tmp_path, tmp_path)
    adapter._mark_update_processed(None)
    assert adapter._is_processed_update(None) is False


def test_persistence_round_trip(tmp_path):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    adapter = _bare_adapter(tmp_path, hermes_home)
    for uid in (1, 2, 3):
        adapter._mark_update_processed(uid)
    adapter._save_processed_update_ids()

    # A fresh adapter (simulating a gateway restart) reloads from disk.
    adapter2 = _bare_adapter(tmp_path, hermes_home)
    adapter2._load_processed_update_ids()
    assert adapter2._is_processed_update(1) is True
    assert adapter2._is_processed_update(2) is True
    assert adapter2._is_processed_update(3) is True
    assert adapter2._is_processed_update(4) is False


def test_save_caps_at_10000(tmp_path):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    adapter = _bare_adapter(tmp_path, hermes_home)
    for uid in range(15000):
        adapter._mark_update_processed(uid)
    adapter._save_processed_update_ids()

    data = json.loads((hermes_home / ".telegram_processed_updates.json").read_text())
    assert len(data) == 10000
    assert data[0] == 5000  # oldest 5000 dropped
    assert data[-1] == 14999


def test_load_ignores_corrupt_file(tmp_path):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    (hermes_home / ".telegram_processed_updates.json").write_text("not json{")
    adapter = _bare_adapter(tmp_path, hermes_home)
    adapter._load_processed_update_ids()  # should not raise
    assert adapter._is_processed_update(1) is False


def test_missing_attribute_is_noop(tmp_path):
    # Adapters built via object.__new__ without the dedup attributes set
    # (as in the group-gating tests) must not crash the handlers.
    adapter = object.__new__(TelegramAdapter)
    assert adapter._is_processed_update(1001) is False
    adapter._mark_update_processed(1001)  # no-op, no raise
