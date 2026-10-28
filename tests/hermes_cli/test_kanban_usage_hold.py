"""Kanban dispatch waits out a provider usage limit (hermes_cli/kanban_dispatch_usage_hold.py).

Contract: a ready card whose worker would call a held provider is not claimed or spawned and
costs no failure; the board gets one ``usage_held`` event per hold however many ticks pass; the
card dispatches on the first tick after the reset. A fallback chain lets it run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agent.usage_hold import record_hold
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch, all_assignees_spawnable):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    (home / "config.yaml").write_text(json.dumps({"model": {"provider": "openai-codex"}}), encoding="utf-8")
    return home


def _tick(spawns):
    with kbc.connect() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=lambda task, ws, board=None: spawns.append(task.id) or 42)
        events = [r["kind"] for r in conn.execute("SELECT kind FROM task_events ORDER BY id")]
    return res, events


def test_held_provider_parks_cards_until_the_reset(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="write the report", assignee="default")
    record_hold("openai-codex", until=time.time() + 1800, source="provider_error")

    spawns: list = []
    for _ in range(3):
        res, events = _tick(spawns)
        assert res.skipped_usage_held == [tid]
    assert spawns == []
    assert events.count("usage_held") == 1
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "ready" and not task.consecutive_failures

    # The reset passes: the next tick dispatches it.
    later = time.time() + 1900
    monkeypatch.setattr(time, "time", lambda: later)
    res, _ = _tick(spawns)
    assert spawns == [tid] and not res.skipped_usage_held


def test_fallback_chain_or_other_provider_still_dispatches(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="default")
    record_hold("anthropic", until=time.time() + 1800, source="provider_error")
    spawns: list = []
    _tick(spawns)
    assert spawns == [tid]  # a hold on another provider never parks this profile's cards

    with kbc.connect() as conn:
        tid2 = kb.create_task(conn, title="t2", assignee="default")
    record_hold("openai-codex", until=time.time() + 1800, source="provider_error")
    (kanban_home / "config.yaml").write_text(json.dumps({
        "model": {"provider": "openai-codex"}, "fallback_providers": [{"provider": "nous", "model": "x"}]}), encoding="utf-8")
    _tick(spawns)
    assert tid2 in spawns
