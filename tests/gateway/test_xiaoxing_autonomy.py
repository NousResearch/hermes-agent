import asyncio
import json
from datetime import datetime

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
import gateway.xiaoxing_autonomy as xiaoxing_autonomy
from gateway.xiaoxing_autonomy import (
    build_trigger_prompt,
    sanitize_autonomy_response,
    should_skip_autonomy_send,
)
from plugins.platforms.napcat.adapter import NapCatAdapter


def test_xiaoxing_autonomy_prompt_requires_silent_or_public_text():
    prompt = build_trigger_prompt("daytime_random")

    assert "XIAOXING_AUTONOMY_TRIGGER" in prompt
    assert "[SILENT]" in prompt
    assert "不要输出触发说明" in prompt


def test_xiaoxing_autonomy_sanitizes_internal_leak_to_public_tail():
    leaked = """[XIAOXING_AUTONOMY_TRIGGER]
判断：应该发送。
工具调用：不要泄漏。

爸，今天先轻轻冒个泡。"""

    assert sanitize_autonomy_response(leaked) == "爸，今天先轻轻冒个泡。"
    assert should_skip_autonomy_send("[SILENT")
    assert should_skip_autonomy_send("判断：工具调用")


def test_gateway_prefers_milky_home_channel_for_xiaoxing_autonomy():
    config = GatewayConfig()
    milky = Platform("milky")
    napcat = Platform("napcat")
    config.platforms[milky] = PlatformConfig(
        enabled=True,
        home_channel=HomeChannel(platform=milky, chat_id="490008192", name="Dad QQ Milky"),
    )
    config.platforms[napcat] = PlatformConfig(
        enabled=True,
        home_channel=HomeChannel(platform=napcat, chat_id="490008192", name="Dad QQ"),
    )
    runner = GatewayRunner(config)
    runner.adapters = {napcat: object(), milky: object()}

    target = runner._xiaoxing_autonomy_home_target()

    assert target is not None
    assert target.platform == milky
    assert target.home.chat_id == "490008192"


def test_gateway_autonomy_trigger_sends_agent_text_to_home_channel():
    config = GatewayConfig()
    milky = Platform("milky")
    config.platforms[milky] = PlatformConfig(
        enabled=True,
        home_channel=HomeChannel(platform=milky, chat_id="490008192", name="Dad QQ Milky"),
    )
    runner = GatewayRunner(config)

    class FakeAdapter:
        def __init__(self):
            self.sent = []

        async def send(self, chat_id, content, metadata=None):
            self.sent.append((chat_id, content, metadata))
            return SendResult(success=True, message_id="1")

    adapter = FakeAdapter()
    runner.adapters = {milky: adapter}

    async def fake_handle_message(event):
        assert event.internal is True
        assert event.source.platform == milky
        assert event.source.chat_id == "490008192"
        assert event.raw_message["internal_trigger"] is True
        return "爸，我今天只是来轻轻打个招呼。"

    runner._handle_message = fake_handle_message

    sent = asyncio.run(
        runner._fire_xiaoxing_autonomy_trigger(
            {"id": "2026-06-24:daytime_random_1", "kind": "daytime_random", "at": "2026-06-24T12:00:00"}
        )
    )

    assert sent is True
    assert adapter.sent == [
        (
            "490008192",
            "爸，我今天只是来轻轻打个招呼。",
            {
                "xiaoxing_autonomy_trigger": True,
                "internal_trigger": True,
                "trigger": {
                    "id": "2026-06-24:daytime_random_1",
                    "kind": "daytime_random",
                    "at": "2026-06-24T12:00:00",
                },
            },
        )
    ]


def test_gateway_initial_autonomy_plan_does_not_backfill_due_messages(monkeypatch, tmp_path):
    state_path = tmp_path / "xiaoxing_autonomy.json"
    monkeypatch.setenv("HERMES_XIAOXING_AUTONOMY_TRIGGER_STATE", str(state_path))
    monkeypatch.setattr(
        xiaoxing_autonomy,
        "build_daily_trigger_plan",
        lambda day: [
            {"id": "past", "kind": "daytime_random", "at": "2026-06-24T12:00:00"},
            {"id": "future", "kind": "bedtime_chat", "at": "2026-06-24T22:00:00"},
        ],
    )
    runner = GatewayRunner(GatewayConfig())

    fired = asyncio.run(runner._run_xiaoxing_autonomy_tick(datetime(2026, 6, 24, 18, 0, 0)))

    assert fired == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["fired"] == ["past"]


def test_napcat_autonomy_loop_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("NAPCAT_XIAOXING_TRIGGERS", raising=False)
    adapter = NapCatAdapter(PlatformConfig(enabled=True, extra={"allowed_users": "490008192"}))

    assert adapter.autonomy_triggers_enabled is False
    assert adapter._trigger_task is None
