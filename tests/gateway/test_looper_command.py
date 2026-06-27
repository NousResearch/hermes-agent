from __future__ import annotations

import asyncio
import json

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tools import slash_confirm


class _FakeSessionEntry:
    session_id = "sid-looper-gateway"


class _FakeSessionStore:
    def get_or_create_session(self, source):
        return self.entry

    def __init__(self):
        self.entry = _FakeSessionEntry()


def test_gateway_looper_returns_preview_and_resolves_on_approval(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    runner.adapters = {}
    runner._queued_events = {}

    event = MessageEvent(
        text="/looper Improve the OddsEdge Picks review flow and keep it review-gated.",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-looper",
            chat_type="group",
            user_id="user-looper",
            thread_id="6",
        ),
        message_id="msg-looper",
    )

    preview = asyncio.run(GatewayRunner._handle_looper_command(runner, event))
    assert "Looper preview" in preview
    assert "/goal" in preview
    assert "Approve Once" in preview

    session_key = runner._session_key_for_source(event.source)
    pending = slash_confirm.get_pending(session_key)
    assert pending is not None
    assert pending["command"] == "looper"

    resolved = asyncio.run(slash_confirm.resolve(session_key, pending["confirm_id"], "once"))
    assert resolved is not None
    assert resolved.startswith("✅ ready")
    assert "Final /goal prompt:" in resolved

    run_root = home / "looper" / "sid-looper-gateway"
    assert run_root.exists()
    run_dirs = [p for p in run_root.iterdir() if p.is_dir()]
    assert run_dirs, "expected at least one looper run directory"
    run_dir = run_dirs[0]
    assert (run_dir / "loop.resolved.json").exists()

    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "ready"
    assert state["approved"] is True
