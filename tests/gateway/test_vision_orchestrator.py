import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageAttachment, MessageEvent, MessageType
from gateway.session import SessionSource


@pytest.mark.asyncio
async def test_image_only_turn_waits_for_analysis_result():
    from gateway.vision_orchestrator import VisionOrchestrator

    orchestrator = VisionOrchestrator(config_loader=lambda: {})

    async def _analyze(**_kwargs):
        await asyncio.sleep(0.01)
        return {"success": True, "analysis": "图里是一张后台报错截图"}

    analyze = AsyncMock(side_effect=_analyze)

    event = MessageEvent(
        text="",
        message_type=MessageType.PHOTO,
        source=SessionSource(platform=Platform.QQ_NAPCAT, chat_id="1", chat_type="group"),
        attachments=[
            MessageAttachment(
                kind="image",
                mime_type="image/png",
                local_path="/tmp/demo.png",
                analysis_ref="/tmp/demo.png",
            )
        ],
    )

    outcome = await orchestrator.prepare_turn(
        event=event,
        user_text="",
        analyze_image=analyze,
    )

    assert outcome.direct_reply is None
    assert "后台报错截图" in outcome.enriched_text


@pytest.mark.asyncio
async def test_image_only_turn_returns_degraded_note_when_analysis_fails():
    from gateway.vision_orchestrator import VisionOrchestrator

    orchestrator = VisionOrchestrator(config_loader=lambda: {})
    analyze = AsyncMock(return_value={"success": False, "analysis": "", "error": "vision timeout"})

    event = MessageEvent(
        text="",
        message_type=MessageType.PHOTO,
        source=SessionSource(platform=Platform.QQ_NAPCAT, chat_id="1", chat_type="group"),
        attachments=[
            MessageAttachment(
                kind="image",
                mime_type="image/png",
                local_path="/tmp/demo.png",
                analysis_ref="/tmp/demo.png",
            )
        ],
    )

    outcome = await orchestrator.prepare_turn(
        event=event,
        user_text="",
        analyze_image=analyze,
    )

    assert outcome.direct_reply is None
    assert outcome.enriched_text == "[Image attached; no verified image description is available for this turn.]"


@pytest.mark.asyncio
async def test_text_plus_image_turn_does_not_block_on_pending_analysis(monkeypatch):
    from gateway.vision_orchestrator import VisionOrchestrator

    orchestrator = VisionOrchestrator(config_loader=lambda: {})

    async def _slow_analyze(**_kwargs):
        await asyncio.sleep(0.05)
        return {"success": True, "analysis": "图里是一张产品界面"}

    event = MessageEvent(
        text="看下这个界面",
        message_type=MessageType.PHOTO,
        source=SessionSource(platform=Platform.QQ_NAPCAT, chat_id="1", chat_type="group"),
        attachments=[
            MessageAttachment(
                kind="image",
                mime_type="image/png",
                local_path="/tmp/demo.png",
                analysis_ref="/tmp/demo.png",
            )
        ],
    )

    monkeypatch.setattr(
        "gateway.vision_orchestrator._DEFAULT_INLINE_WAIT_SECONDS",
        0.005,
        raising=False,
    )
    monkeypatch.setattr(
        "gateway.vision_orchestrator._DEFAULT_GROUP_INLINE_WAIT_SECONDS",
        0.005,
        raising=False,
    )

    outcome = await orchestrator.prepare_turn(
        event=event,
        user_text="看下这个界面",
        analyze_image=_slow_analyze,
    )

    assert outcome.direct_reply is None
    assert outcome.enriched_text.endswith("看下这个界面")
    assert "[Image attached; no verified image description is available yet.]" in outcome.enriched_text
    assert "产品界面" not in outcome.enriched_text
