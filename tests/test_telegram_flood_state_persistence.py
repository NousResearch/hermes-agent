"""Flood bans must survive a gateway restart.

Regression: the hub adapter kept ``_flood_block_until`` in memory only. A
restart during a multi-hour Telegram ban dropped every deadline, so the first
send after boot called the API inside the ban and Telegram extended it. Messages
then fell through to the failover bot for hours. The Mac adapter already
persisted this state; the hub did not.
"""

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

ADAPTER_PATH = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "platforms"
    / "telegram"
    / "adapter.py"
)


def _load_adapter_cls():
    spec = importlib.util.spec_from_file_location("_tg_adapter_flood", ADAPTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TelegramAdapter


@pytest.fixture()
def adapter_cls(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    cls = _load_adapter_cls()

    class _Named(cls):
        name = "telegram-test"

    return _Named


def _fresh(adapter_cls):
    obj = object.__new__(adapter_cls)
    obj._flood_block_until = {}
    # Present on the Mac adapter, absent on the hub's: set it either way so this
    # test measures flood-deadline persistence, not adapter feature drift.
    obj._failover_dead_threads = set()
    return obj


def test_flood_deadline_survives_restart(adapter_cls, tmp_path):
    first = _fresh(adapter_cls)
    first._arm_flood_block(1335137548, 900)

    state = json.loads(
        (tmp_path / "state" / "telegram-flood-state.json").read_text(encoding="utf-8")
    )
    assert "1335137548" in state["flood_block_until"]

    after_restart = _fresh(adapter_cls)
    after_restart._load_flood_state()
    remaining = after_restart._flood_block_until["1335137548"] - time.monotonic()
    # Upper bound has slack: the deadline round-trips through wall-clock time,
    # so restore can land a few ms above the original monotonic window.
    assert 870 < remaining < 905


def test_expired_ban_is_not_restored(adapter_cls, tmp_path):
    path = tmp_path / "state" / "telegram-flood-state.json"
    path.write_text(
        json.dumps({"flood_block_until": {"999": time.time() - 10}}), encoding="utf-8"
    )
    obj = _fresh(adapter_cls)
    obj._load_flood_state()
    assert obj._flood_block_until == {}


def test_missing_state_file_is_not_an_error(adapter_cls):
    obj = _fresh(adapter_cls)
    obj._load_flood_state()
    assert obj._flood_block_until == {}


def test_arming_never_shortens_an_open_window(adapter_cls):
    obj = _fresh(adapter_cls)
    obj._arm_flood_block(42, 900)
    long_deadline = obj._flood_block_until["42"]
    obj._arm_flood_block(42, 5)
    assert obj._flood_block_until["42"] == long_deadline
