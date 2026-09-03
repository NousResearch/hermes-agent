"""E2E: unattached kanban_create → home-channel fallback subscription →
gateway notifier delivery through a real RecordingAdapter.

Root-cause scenario: a board full of agent/background-created tasks and
zero kanban_notify_subs rows meant terminal events never reached the
user's home channel. This test drives the whole chain:
kanban_create (no session origin) → subscribe_task_to_configured_homes
→ GatewayRunner._kanban_notifier_watcher → adapter.send().
"""

from __future__ import annotations

import asyncio

import pytest


class RecordingAdapter:
    """Minimal adapter that records sends (same shape as
    tests/gateway/test_kanban_notifier.py's helper)."""

    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


@pytest.fixture
def discord_home_env(monkeypatch, tmp_path):
    """Isolated HERMES_HOME with a single enabled Discord home channel."""
    from pathlib import Path as _Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "disc_fake")
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "status-channel")
    return home


@pytest.mark.asyncio
async def test_unattached_create_reaches_discord_home_via_notifier(
    monkeypatch,
    discord_home_env,
):
    """Full chain: background/CLI kanban_create with homes configured but no
    session origin → subscription row exists → watcher claims the terminal
    event and delivers it through the Discord recording adapter."""
    import yaml

    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from hermes_cli import kanban_db as kb

    # Pin an explicit config so load_gateway_config sees exactly one
    # ENABLED platform (discord) with a home channel.
    cfg_home = discord_home_env / "config.yaml"
    cfg_home.write_text(
        yaml.safe_dump({"platforms": {"discord": {"enabled": True}}}),
        encoding="utf-8",
    )

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    # Create the task exactly as an unattached agent would: no platform /
    # chat_id / session key in the environment.
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "default")

    from tools import kanban_tools as kt

    out = kt._handle_create({
        "title": "swarm decomposition follow-up",
        "assignee": "worker1",
    })
    import json as _json

    d = _json.loads(out)
    assert d["ok"] is True, d

    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, d["task_id"])
        assert any(s["platform"] == "discord" for s in subs), (
            "unattached create must land a discord home subscription"
        )
        kb.block_task(conn, d["task_id"], reason="needs human input")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.DISCORD: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    runner._owns_kanban_dispatcher_lock = lambda: True
    # The default profile owns the home-fallback row (stamped at create).
    monkeypatch.setattr(
        GatewayRunner,
        "_active_profile_name",
        lambda self: "default",
        raising=True,
    )
    monkeypatch.setattr(
        GatewayRunner,
        "_authorization_adapter",
        lambda self, plat, profile=None: (
            adapter if plat == Platform.DISCORD else None
        ),
        raising=True,
    )

    await _run_one_notifier_tick(monkeypatch, runner)

    assert adapter.sent, (
        "terminal event for the unattached-created task must be delivered "
        "to the discord home channel"
    )
    assert all(m["chat_id"] == "status-channel" for m in adapter.sent)
    assert any("needs human input" in m["text"] for m in adapter.sent)
