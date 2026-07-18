"""Runtime tests for Feishu final response card delivery."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import FinalDeliveryState
from plugins.platforms.feishu.adapter import FeishuAdapter


class _FakeResponse:
    def __init__(self, message_id="om_1"):
        self.code = 0
        self.msg = "ok"
        self.data = SimpleNamespace(message_id=message_id)

    def success(self):
        return True


class _FakeFailureResponse:
    code = 999
    msg = "card rejected"
    data = SimpleNamespace(message_id=None)

    def success(self):
        return False


class _FakeImageData:
    image_key = "img_test"


class _FakeImageResponse:
    code = 0
    msg = "ok"
    data = _FakeImageData()

    def success(self):
        return True


class _FakeImageApi:
    def create(self, request):
        return _FakeImageResponse()


class _FakeClient:
    im = SimpleNamespace(v1=SimpleNamespace(image=_FakeImageApi()))


def _make_adapter(*, final_response_format="card", markdown_tables="table") -> FeishuAdapter:
    cfg = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={
            "final_response_format": final_response_format,
            "markdown_tables": markdown_tables,
            "card_schema": "2.0",
        },
    )
    adapter = FeishuAdapter(cfg)
    adapter._client = object()
    return adapter


@pytest.mark.asyncio
async def test_feishu_upload_image_for_card_returns_image_key(tmp_path):
    adapter = _make_adapter(final_response_format="card")
    adapter._client = _FakeClient()

    image = tmp_path / "a.png"
    image.write_bytes(b"fake png")

    image_key = await adapter._upload_image_for_card(str(image))

    assert image_key == "img_test"


@pytest.mark.asyncio
async def test_feishu_final_response_card_mode_sends_interactive_card(monkeypatch):
    adapter = _make_adapter(final_response_format="card")
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        return _FakeResponse()

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)

    result = await adapter.send(
        "oc_123",
        "# 标题\n\n| A | B |\n| --- | --- |\n| 1 | 2 |",
        metadata={
            "thread_id": "t1",
            "reply_to_message_id": "om_anchor",
            "hermes_final_response": True,
        },
    )

    assert result.success is True
    assert len(calls) == 1
    assert calls[0]["msg_type"] == "interactive"
    card = json.loads(calls[0]["payload"])
    assert card["schema"] == "2.0"
    assert card["body"]["elements"][0] == {
        "tag": "markdown",
        "content": "标题",
        "text_size": "heading",
    }
    assert any(element.get("tag") == "table" for element in card["body"]["elements"])


@pytest.mark.asyncio
async def test_feishu_final_response_long_card_sends_multiple_interactive_cards(monkeypatch):
    adapter = _make_adapter(final_response_format="card")
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        return _FakeResponse(f"om_{len(calls)}")

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)
    long_text = "\n\n".join(
        f"第 {i:03d} 段：" + ("长内容不会被塞进单张卡片导致截断。" * 20)
        for i in range(1, 101)
    )

    result = await adapter.send(
        "oc_123",
        long_text,
        metadata={
            "thread_id": "t1",
            "reply_to_message_id": "om_anchor",
            "hermes_final_response": True,
        },
    )

    assert result.success is True
    assert len(calls) > 1
    assert all(call["msg_type"] == "interactive" for call in calls)
    titles = [json.loads(call["payload"])["header"]["title"]["content"] for call in calls]
    assert titles[0].startswith("Hermes 1/")
    assert titles[-1].startswith(f"Hermes {len(calls)}/")


@pytest.mark.asyncio
async def test_feishu_non_final_response_uses_legacy_payload(monkeypatch):
    adapter = _make_adapter(final_response_format="card")
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        return _FakeResponse()

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)

    await adapter.send("oc_123", "# status", metadata={"thread_id": "t1"})

    assert len(calls) == 1
    assert calls[0]["msg_type"] != "interactive"


@pytest.mark.asyncio
async def test_feishu_final_response_auto_mode_keeps_legacy_when_media_tag_present(monkeypatch, tmp_path):
    media = tmp_path / "out.png"
    media.write_bytes(b"fake")
    adapter = _make_adapter(final_response_format="auto")
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        return _FakeResponse()

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)

    await adapter.send(
        "oc_123",
        f"结果\nMEDIA:{media}",
        metadata={"thread_id": "t1", "hermes_final_response": True},
    )

    assert len(calls) == 1
    assert calls[0]["msg_type"] != "interactive"


@pytest.mark.asyncio
async def test_feishu_final_response_partial_multi_card_failure_does_not_duplicate_legacy(monkeypatch):
    adapter = _make_adapter(final_response_format="card")
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        if kwargs["msg_type"] == "interactive" and len(calls) == 2:
            return _FakeFailureResponse()
        return _FakeResponse(f"om_{len(calls)}")

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)
    long_text = "\n\n".join(
        f"第 {i:03d} 段：" + ("长内容会拆成多张卡片。" * 20)
        for i in range(1, 101)
    )

    result = await adapter.send(
        "oc_123",
        long_text,
        metadata={
            "thread_id": "t1",
            "reply_to_message_id": "om_anchor",
            "hermes_final_response": True,
        },
    )

    assert result.success is False
    assert [call["msg_type"] for call in calls] == ["interactive", "interactive"]


@pytest.mark.asyncio
async def test_feishu_final_rich_response_partial_multi_card_failure_does_not_fallback(monkeypatch, tmp_path):
    adapter = _make_adapter(final_response_format="card")
    image = tmp_path / "a.png"
    image.write_bytes(b"fake png")
    calls = []

    async def fake_upload(path):
        return "img_test"

    async def fake_send(**kwargs):
        calls.append(kwargs)
        if kwargs["msg_type"] == "interactive" and len(calls) == 2:
            return _FakeFailureResponse()
        return _FakeResponse(f"om_{len(calls)}")

    monkeypatch.setattr(adapter, "_upload_image_for_card", fake_upload)
    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)
    text = "\n\n".join(
        f"第 {i:03d} 段：" + ("长内容会拆成多张卡片。" * 20)
        for i in range(1, 101)
    )

    result = await adapter.try_send_final_rich_response(
        chat_id="oc_123",
        original_response=f"{text}\nMEDIA:{image}",
        text_content=text,
        images=[],
        media_files=[(str(image), False)],
        local_files=[],
        force_document_attachments=False,
        reply_to=None,
        metadata={"hermes_final_response": True},
        is_ephemeral_response=False,
    )

    assert result is not None and result.success is False
    assert result.delivery_state is FinalDeliveryState.PARTIALLY_DELIVERED
    assert [call["msg_type"] for call in calls] == ["interactive", "interactive"]


@pytest.mark.asyncio
async def test_feishu_final_response_card_failure_falls_back_to_legacy(monkeypatch):
    adapter = _make_adapter(final_response_format="card")
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        if kwargs["msg_type"] == "interactive":
            raise RuntimeError("card rejected")
        return _FakeResponse("fallback")

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)

    result = await adapter.send(
        "oc_123",
        "**hello**",
        metadata={
            "thread_id": "t1",
            "reply_to_message_id": "om_anchor",
            "hermes_final_response": True,
        },
    )

    assert result.success is True
    assert [call["msg_type"] for call in calls] == ["interactive", "post"]


@pytest.mark.asyncio
async def test_feishu_final_rich_response_embeds_local_image_media(monkeypatch, tmp_path):
    adapter = _make_adapter(final_response_format="card")
    image = tmp_path / "a.png"
    image.write_bytes(b"fake png")
    calls = []

    async def fake_upload(path):
        assert path == str(image)
        return "img_test"

    async def fake_send(**kwargs):
        calls.append(kwargs)
        return _FakeResponse("rich")

    monkeypatch.setattr(adapter, "_upload_image_for_card", fake_upload)
    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)

    result = await adapter.try_send_final_rich_response(
        chat_id="oc_123",
        original_response=f"正文\nMEDIA:{image}",
        text_content="正文",
        images=[],
        media_files=[(str(image), False)],
        local_files=[],
        force_document_attachments=False,
        reply_to=None,
        metadata={"hermes_final_response": True},
        is_ephemeral_response=False,
    )

    assert result is not None and result.success is True
    assert calls[0]["msg_type"] == "interactive"
    card = json.loads(calls[0]["payload"])
    assert any(
        element.get("tag") == "img" and element.get("img_key") == "img_test"
        for element in card["body"]["elements"]
    )
    assert "MEDIA:" not in calls[0]["payload"]


@pytest.mark.asyncio
async def test_feishu_final_rich_response_upload_failure_falls_back(monkeypatch, tmp_path):
    adapter = _make_adapter(final_response_format="card")
    image = tmp_path / "a.png"
    image.write_bytes(b"fake png")

    async def fake_upload(path):
        raise RuntimeError("upload failed")

    fake_send = AsyncMock()
    monkeypatch.setattr(adapter, "_upload_image_for_card", fake_upload)
    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)

    result = await adapter.try_send_final_rich_response(
        chat_id="oc_123",
        original_response=f"正文\nMEDIA:{image}",
        text_content="正文",
        images=[],
        media_files=[(str(image), False)],
        local_files=[],
        force_document_attachments=False,
        reply_to=None,
        metadata={"hermes_final_response": True},
        is_ephemeral_response=False,
    )

    assert result is None
    fake_send.assert_not_awaited()


@pytest.mark.asyncio
async def test_feishu_final_rich_response_skips_non_image_media(tmp_path):
    adapter = _make_adapter(final_response_format="card")
    document = tmp_path / "a.pdf"
    document.write_bytes(b"fake pdf")

    result = await adapter.try_send_final_rich_response(
        chat_id="oc_123",
        original_response=f"正文\nMEDIA:{document}",
        text_content="正文",
        images=[],
        media_files=[(str(document), False)],
        local_files=[],
        force_document_attachments=False,
        reply_to=None,
        metadata={"hermes_final_response": True},
        is_ephemeral_response=False,
    )

    assert result is None


@pytest.mark.asyncio
async def test_feishu_final_rich_response_respects_as_document(tmp_path):
    adapter = _make_adapter(final_response_format="card")
    image = tmp_path / "a.png"
    image.write_bytes(b"fake png")

    result = await adapter.try_send_final_rich_response(
        chat_id="oc_123",
        original_response=f"[[as_document]]\nMEDIA:{image}",
        text_content="正文",
        images=[],
        media_files=[(str(image), False)],
        local_files=[],
        force_document_attachments=True,
        reply_to=None,
        metadata={"hermes_final_response": True},
        is_ephemeral_response=False,
    )

    assert result is None


@pytest.mark.asyncio
async def test_feishu_final_response_thread_without_reply_anchor_uses_legacy_thread_create(monkeypatch):
    adapter = _make_adapter(final_response_format="card")
    captured = {}

    def fake_create_body(**kwargs):
        captured["body"] = kwargs
        return kwargs

    def fake_create_request(receive_id_type, body):
        captured["receive_id_type"] = receive_id_type
        captured["request_body"] = body
        return {"receive_id_type": receive_id_type, "body": body}

    class _Messages:
        def create(self, request):
            captured["request"] = request
            return _FakeResponse("om_thread")

    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=_Messages())))
    monkeypatch.setattr(adapter, "_build_create_message_body", fake_create_body)
    monkeypatch.setattr(adapter, "_build_create_message_request", fake_create_request)

    result = await adapter.send(
        "oc_parent",
        "**thread reply**",
        metadata={"thread_id": "omt_thread", "hermes_final_response": True},
    )

    assert result.success is True
    assert captured["receive_id_type"] == "thread_id"
    assert captured["body"]["receive_id"] == "omt_thread"
    assert captured["body"]["msg_type"] != "interactive"


@pytest.mark.asyncio
async def test_feishu_final_rich_response_thread_without_reply_anchor_skips_before_image_upload(
    monkeypatch,
    tmp_path,
):
    adapter = _make_adapter(final_response_format="card")
    image = tmp_path / "a.png"
    image.write_bytes(b"fake png")
    fake_upload = AsyncMock(return_value="img_test")
    fake_send = AsyncMock(return_value=_FakeResponse("rich"))
    monkeypatch.setattr(adapter, "_upload_image_for_card", fake_upload)
    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)

    result = await adapter.try_send_final_rich_response(
        chat_id="oc_parent",
        original_response=f"正文\nMEDIA:{image}",
        text_content="正文",
        images=[],
        media_files=[(str(image), False)],
        local_files=[],
        force_document_attachments=False,
        reply_to=None,
        metadata={"thread_id": "omt_thread", "hermes_final_response": True},
        is_ephemeral_response=False,
    )

    assert result is None
    fake_upload.assert_not_awaited()
    fake_send.assert_not_awaited()


def test_feishu_invalid_markdown_tables_config_defaults_to_table():
    adapter = _make_adapter(final_response_format="card", markdown_tables="bogus")

    assert adapter._final_response_table_policy() == "table"


def test_gateway_platforms_feishu_extra_loads_final_card_settings(tmp_path, monkeypatch):
    from gateway.config import load_gateway_config, Platform

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "platforms:\n"
        "  feishu:\n"
        "    extra:\n"
        "      final_response_format: auto\n"
        "      markdown_tables: table\n"
        "      card_schema: '2.0'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    cfg = load_gateway_config()

    extra = cfg.platforms[Platform.FEISHU].extra
    assert extra["final_response_format"] == "auto"
    assert extra["markdown_tables"] == "table"
    assert extra["card_schema"] == "2.0"


def test_nested_gateway_platforms_feishu_extra_loads_final_card_settings(tmp_path, monkeypatch):
    from gateway.config import load_gateway_config, Platform

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "gateway:\n"
        "  platforms:\n"
        "    feishu:\n"
        "      extra:\n"
        "        final_response_format: card\n"
        "        markdown_tables: markdown\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    cfg = load_gateway_config()

    extra = cfg.platforms[Platform.FEISHU].extra
    assert extra["final_response_format"] == "card"
    assert extra["markdown_tables"] == "markdown"


# ---------------------------------------------------------------------------
# T10: Regression tests locking already-correct behavior
# ---------------------------------------------------------------------------


class _EmptyImageKeyData:
    image_key = None


class _EmptyImageKeyResponse:
    code = 230002
    msg = "image upload returned no key"
    data = _EmptyImageKeyData()

    def success(self):
        return True  # API "succeeded" but no image_key


class _EmptyImageKeyApi:
    def create(self, request):
        return _EmptyImageKeyResponse()


class _EmptyImageKeyClient:
    im = SimpleNamespace(v1=SimpleNamespace(image=_EmptyImageKeyApi()))


@pytest.mark.asyncio
async def test_empty_image_key_raises_runtime_error(tmp_path):
    """_upload_image_for_card must raise RuntimeError when image_key is empty."""
    adapter = _make_adapter(final_response_format="card")
    adapter._client = _EmptyImageKeyClient()

    image = tmp_path / "a.png"
    image.write_bytes(b"fake png")

    with pytest.raises(RuntimeError, match="missing image_key"):
        await adapter._upload_image_for_card(str(image))


@pytest.mark.asyncio
async def test_auto_mode_with_media_tag_keeps_legacy_path(monkeypatch, tmp_path):
    """auto mode + MEDIA tag: _should_send_final_response_as_card returns False."""
    adapter = _make_adapter(final_response_format="auto")
    media = tmp_path / "out.png"
    media.write_bytes(b"fake")

    result = adapter._should_send_final_response_as_card(
        f"结果\nMEDIA:{media}",
        metadata={"hermes_final_response": True},
    )
    assert result is False

    # Also verify auto mode without media sends as card
    result_no_media = adapter._should_send_final_response_as_card(
        "普通文本",
        metadata={"hermes_final_response": True},
    )
    assert result_no_media is True


def test_real_sdk_create_and_reply_builders_accept_uuid():
    """Real lark-oapi builders must accept uuid parameter."""
    try:
        from lark_oapi.api.im.v1 import (
            CreateMessageRequestBody,
            ReplyMessageRequestBody,
        )
    except ImportError:
        pytest.skip("lark-oapi not installed; skipping SDK builder test")

    create_body = (
        CreateMessageRequestBody.builder()
        .receive_id("oc_test")
        .msg_type("interactive")
        .content('{"schema":"2.0"}')
        .uuid("test-uuid-12345")
        .build()
    )
    assert create_body is not None

    reply_body = (
        ReplyMessageRequestBody.builder()
        .content('{"schema":"2.0"}')
        .msg_type("interactive")
        .reply_in_thread(False)
        .uuid("test-uuid-67890")
        .build()
    )
    assert reply_body is not None


@pytest.mark.asyncio
async def test_interactive_thread_reply_failure_does_not_create_top_level_duplicate(monkeypatch):
    """When interactive reply fails in thread with fallback code, should not create a new top-level message."""
    adapter = _make_adapter(final_response_format="card")

    send_calls = []  # track all _send_raw_message invocations

    class _ThreadReplyFailResponse:
        code = 230011
        msg = "reply target withdrawn"
        data = SimpleNamespace(message_id=None)

        def success(self):
            return False

    class _FakeReplyApi:
        def reply(self, request):
            return _ThreadReplyFailResponse()

        def create(self, request):
            # This should NOT be called for thread messages
            send_calls.append("create_called")
            return _FakeResponse("om_new_top_level")

    adapter._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=_FakeReplyApi()))
    )

    response = await adapter._feishu_send_with_retry(
        chat_id="oc_parent",
        msg_type="interactive",
        payload=json.dumps({"schema": "2.0", "body": {"elements": []}}),
        reply_to="om_reply_target",
        metadata={"thread_id": "omt_thread", "hermes_final_response": True},
    )

    # The failed reply response should be returned directly (not retried as create)
    assert not response.success()
    assert response.code == 230011
    # Verify create was never called — no top-level duplicate
    assert send_calls == [], "message.create should not be called for thread reply failure"
