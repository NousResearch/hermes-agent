"""T4 regression tests for Feishu multi-card batch throttling."""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import FinalDeliveryState
from plugins.platforms.feishu import adapter as feishu_adapter
from plugins.platforms.feishu.adapter import FeishuAdapter


class _FakeResponse:
    code = 0
    msg = "ok"

    def __init__(self, message_id: str = "om_1") -> None:
        self.data = SimpleNamespace(message_id=message_id)

    def success(self) -> bool:
        return True


class _FakeFailureResponse:
    code = 999
    msg = "card rejected"
    data = None

    def success(self) -> bool:
        return False


def _make_adapter() -> FeishuAdapter:
    adapter = FeishuAdapter(
        PlatformConfig(
            enabled=True,
            token="fake-token",
            extra={"final_response_format": "card"},
        )
    )
    adapter._client = object()
    return adapter


@pytest.mark.asyncio
async def test_multicard_boundary_delay_is_minimum_plus_injected_jitter():
    delays = []
    jitter_bounds = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    def fake_jitter(low: float, high: float) -> float:
        jitter_bounds.append((low, high))
        return 0.017

    await feishu_adapter._sleep_between_feishu_multicard_parts(
        sleep=fake_sleep,
        jitter=fake_jitter,
    )

    assert feishu_adapter._FEISHU_MULTICARD_MIN_INTERVAL_SECONDS == 0.22
    assert jitter_bounds == [
        (0.0, feishu_adapter._FEISHU_MULTICARD_JITTER_MAX_SECONDS)
    ]
    assert delays == [pytest.approx(0.237)]


@pytest.mark.asyncio
async def test_send_three_cards_throttles_only_the_two_part_boundaries(monkeypatch):
    adapter = _make_adapter()
    events = []
    logical_time = 0.0

    monkeypatch.setattr(
        feishu_adapter,
        "build_feishu_card_v2_payloads",
        lambda *args, **kwargs: ["card-1", "card-2", "card-3"],
    )

    async def fake_send(**kwargs):
        events.append(("send", kwargs["payload"], logical_time))
        return _FakeResponse(kwargs["payload"])

    async def fake_boundary_sleep():
        nonlocal logical_time
        events.append(("sleep", None, logical_time))
        logical_time += 0.227

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)
    monkeypatch.setattr(
        feishu_adapter,
        "_sleep_between_feishu_multicard_parts",
        fake_boundary_sleep,
    )

    result = await adapter.send(
        "oc_target",
        "content",
        metadata={"hermes_final_response": True},
    )

    assert result.success is True
    assert events == [
        ("send", "card-1", 0.0),
        ("sleep", None, 0.0),
        ("send", "card-2", 0.227),
        ("sleep", None, 0.227),
        ("send", "card-3", 0.454),
    ]


@pytest.mark.asyncio
async def test_send_two_cards_has_real_minimum_boundary_interval(monkeypatch):
    adapter = _make_adapter()
    send_times = []

    monkeypatch.setattr(
        feishu_adapter,
        "build_feishu_card_v2_payloads",
        lambda *args, **kwargs: ["card-1", "card-2"],
    )
    monkeypatch.setattr(feishu_adapter.random, "uniform", lambda low, high: 0.0)

    async def fake_send(**kwargs):
        send_times.append(asyncio.get_running_loop().time())
        return _FakeResponse(kwargs["payload"])

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)

    result = await adapter.send(
        "oc_target",
        "content",
        metadata={"hermes_final_response": True},
    )

    assert result.success is True
    assert len(send_times) == 2
    assert send_times[1] - send_times[0] >= 0.22


@pytest.mark.asyncio
async def test_send_single_card_does_not_sleep(monkeypatch):
    adapter = _make_adapter()
    sends = []
    sleeps = []

    monkeypatch.setattr(
        feishu_adapter,
        "build_feishu_card_v2_payloads",
        lambda *args, **kwargs: ["only-card"],
    )

    async def fake_send(**kwargs):
        sends.append(kwargs["payload"])
        return _FakeResponse()

    async def fake_boundary_sleep():
        sleeps.append(True)

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)
    monkeypatch.setattr(
        feishu_adapter,
        "_sleep_between_feishu_multicard_parts",
        fake_boundary_sleep,
    )

    result = await adapter.send(
        "oc_target",
        "content",
        metadata={"hermes_final_response": True},
    )

    assert result.success is True
    assert sends == ["only-card"]
    assert sleeps == []


@pytest.mark.asyncio
async def test_send_second_card_failure_stops_without_trailing_sleep(monkeypatch):
    adapter = _make_adapter()
    events = []

    monkeypatch.setattr(
        feishu_adapter,
        "build_feishu_card_v2_payloads",
        lambda *args, **kwargs: ["card-1", "card-2", "card-3"],
    )

    async def fake_send(**kwargs):
        payload = kwargs["payload"]
        events.append(("send", payload))
        if payload == "card-2":
            return _FakeFailureResponse()
        return _FakeResponse(payload)

    async def fake_boundary_sleep():
        events.append(("sleep", None))

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)
    monkeypatch.setattr(
        feishu_adapter,
        "_sleep_between_feishu_multicard_parts",
        fake_boundary_sleep,
    )

    result = await adapter.send(
        "oc_target",
        "content",
        metadata={"hermes_final_response": True},
    )

    assert result.success is False
    assert result.delivery_state is FinalDeliveryState.PARTIALLY_DELIVERED
    assert events == [
        ("send", "card-1"),
        ("sleep", None),
        ("send", "card-2"),
    ]


@pytest.mark.asyncio
async def test_rich_two_cards_throttles_the_single_part_boundary(monkeypatch):
    adapter = _make_adapter()
    events = []

    monkeypatch.setattr(
        feishu_adapter,
        "build_feishu_card_v2_payloads_from_document",
        lambda *args, **kwargs: ["rich-1", "rich-2"],
    )

    async def fake_upload(path: str) -> str:
        return "img-key"

    async def fake_send(**kwargs):
        events.append(("send", kwargs["payload"]))
        return _FakeResponse(kwargs["payload"])

    async def fake_boundary_sleep():
        events.append(("sleep", None))

    monkeypatch.setattr(adapter, "_upload_image_for_card", fake_upload)
    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)
    monkeypatch.setattr(
        feishu_adapter,
        "_sleep_between_feishu_multicard_parts",
        fake_boundary_sleep,
    )

    result = await adapter.try_send_final_rich_response(
        chat_id="oc_target",
        original_response="content\nMEDIA:/tmp/image.png",
        text_content="content",
        images=[],
        media_files=[("/tmp/image.png", False)],
        local_files=[],
        force_document_attachments=False,
        reply_to=None,
        metadata={"hermes_final_response": True},
    )

    assert result is not None and result.success is True
    assert events == [
        ("send", "rich-1"),
        ("sleep", None),
        ("send", "rich-2"),
    ]
