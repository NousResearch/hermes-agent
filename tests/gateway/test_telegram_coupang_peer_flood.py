"""쿠팡 URL 비활성화와 Telegram PEER_FLOOD 회로 차단 회귀 시험."""

import asyncio
from types import SimpleNamespace

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import (
    TelegramAdapter,
    _deactivate_coupang_urls,
)


@pytest.fixture(autouse=True)
def _isolated_persistent_circuit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.platforms.telegram.outbound_circuit._db_path",
        lambda: tmp_path / "state.db",
    )


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    return adapter


def test_deactivate_coupang_urls_preserves_product_path_and_other_urls():
    content = (
        "상품 https://www.coupang.com/vp/products/123456?itemId=789 와 "
        "https://example.com/coupang.com 안내"
    )

    assert _deactivate_coupang_urls(content) == (
        "상품 https://www.coupang[.]com/vp/products/123456?itemId=789 와 "
        "https://example.com/coupang.com 안내"
    )


def test_deactivate_coupang_urls_defangs_inline_and_fenced_code_too():
    content = (
        "실제 https://link.coupang.com/a/ABC\n"
        "`curl https://www.coupang.com/vp/products/111`\n"
        "```python\nurl = 'https://coupang.com/vp/products/222'\n```"
    )

    assert _deactivate_coupang_urls(content) == (
        "실제 https://link.coupang[.]com/a/ABC\n"
        "`curl https://www.coupang[.]com/vp/products/111`\n"
        "```python\nurl = 'https://coupang[.]com/vp/products/222'\n```"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("coupang.com/item", "coupang[.]com/item"),
        ("www.coupang.com:443/item", "www.coupang[.]com:443/item"),
        ("shop.coupang.com./item", "shop.coupang[.]com./item"),
        ("COUPANG.COM:8443/item", "coupang[.]com:8443/item"),
        ("[상품](coupang.com/item)", "[상품](coupang[.]com/item)"),
        ("[상품](HTTPS://WWW.COUPANG.COM.:443/item)", "[상품](HTTPS://WWW.coupang[.]com.:443/item)"),
        ("example-coupang.com/item", "example-coupang.com/item"),
        ("https://example.com/coupang.com/item", "https://example.com/coupang.com/item"),
    ],
)
def test_deactivate_coupang_urls_handles_host_variants_without_false_positives(
    source, expected
):
    assert _deactivate_coupang_urls(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("[상품](//coupang.com/item)", "[상품](//coupang[.]com/item)"),
        ("//www.coupang.com/item", "//www.coupang[.]com/item"),
        ("https://EXAMPLE.com/?next=//coupang.com/item", "https://EXAMPLE.com/?next=//coupang.com/item"),
        ("https://example.com/#https://coupang.com/item", "https://example.com/#https://coupang.com/item"),
        ("notcoupang.com/item", "notcoupang.com/item"),
        ("coupang.com.evil.test/item", "coupang.com.evil.test/item"),
    ],
)
def test_deactivate_coupang_urls_handles_protocol_relative_without_query_damage(
    source, expected
):
    assert _deactivate_coupang_urls(source) == expected


@pytest.mark.asyncio
async def test_send_defangs_coupang_url_immediately_before_transport(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "format_message", lambda value: value)
    adapter._bot.send_message.return_value.message_id = 10

    result = await adapter.send(
        "123", "상품 https://www.coupang.com/vp/products/123456", metadata={"notify": True}
    )

    assert result.success
    assert adapter._bot.send_message.await_args.kwargs["text"] == (
        "상품 https://www.coupang[.]com/vp/products/123456"
    )


@pytest.mark.asyncio
async def test_edit_defangs_coupang_url_immediately_before_transport():
    adapter = _make_adapter()

    result = await adapter.edit_message(
        "123", "10", "상품 https://link.coupang.com/a/ABC", finalize=False
    )

    assert result.success
    assert adapter._bot.edit_message_text.await_args.kwargs["text"] == (
        "상품 https://link.coupang[.]com/a/ABC"
    )


class PeerFloodError(Exception):
    def __init__(self, retry_after=None):
        super().__init__("Telegram server error: PEER_FLOOD")
        self.retry_after = retry_after


@pytest.mark.asyncio
async def test_peer_flood_opens_per_chat_circuit_for_send_edit_and_typing(monkeypatch):
    adapter = _make_adapter()
    now = {"value": 100.0}
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.asyncio.get_running_loop",
        lambda: type("Loop", (), {"time": lambda self: now["value"]})(),
    )
    adapter._bot.send_message.side_effect = PeerFloodError()

    first = await adapter.send("123", "첫 발송", metadata={"notify": True})
    second = await adapter.send("123", "반복 발송", metadata={"notify": True})
    edit = await adapter.edit_message("123", "10", "수정", finalize=False)
    await adapter.send_typing("123")

    assert not first.success and first.error_kind == "peer_flood"
    assert not first.retryable
    assert not second.success and second.error_kind == "peer_flood"
    assert not edit.success and edit.error_kind == "peer_flood"
    assert adapter._bot.send_message.await_count == 1
    adapter._bot.edit_message_text.assert_not_awaited()
    adapter._bot.send_chat_action.assert_not_awaited()


@pytest.mark.asyncio
async def test_peer_flood_circuit_honors_explicit_retry_after(monkeypatch):
    adapter = _make_adapter()
    now = {"value": 100.0}
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.asyncio.get_running_loop",
        lambda: type("Loop", (), {"time": lambda self: now["value"]})(),
    )
    monkeypatch.setattr(
        "plugins.platforms.telegram.outbound_circuit.time.time",
        lambda: now["value"],
    )
    adapter._bot.send_message.side_effect = [
        PeerFloodError(retry_after=42.0),
        type("Message", (), {"message_id": 11})(),
    ]

    failed = await adapter.send("123", "첫 발송", metadata={"notify": True})
    now["value"] = 141.9
    blocked = await adapter.send("123", "아직 차단", metadata={"notify": True})
    now["value"] = 142.1
    recovered = await adapter.send("123", "차단 해제", metadata={"notify": True})

    assert failed.retry_after == pytest.approx(42.0)
    assert not blocked.success
    assert recovered.success
    assert adapter._bot.send_message.await_count == 2


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [
        (float("nan"), 300.0),
        (float("inf"), 300.0),
        (float("-inf"), 300.0),
        (0.0, 300.0),
        (-1.0, 300.0),
        (900.0, 300.0),
        (42.0, 42.0),
    ],
)
@pytest.mark.asyncio
async def test_peer_flood_retry_after_uses_only_finite_positive_capped_values(
    retry_after, expected
):
    adapter = _make_adapter()

    result = adapter._open_peer_flood_circuit("123", PeerFloodError(retry_after))

    assert result.retry_after == pytest.approx(expected)
    remaining = adapter._peer_flood_remaining("123")
    assert remaining is not None
    assert expected - 1.0 <= remaining <= expected


@pytest.mark.asyncio
async def test_rich_send_peer_flood_opens_same_chat_circuit(monkeypatch):
    adapter = _make_adapter()
    now = {"value": 100.0}
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.asyncio.get_running_loop",
        lambda: type("Loop", (), {"time": lambda self: now["value"]})(),
    )
    adapter._bot.do_api_request.side_effect = PeerFloodError(retry_after=60.0)

    result = await adapter._try_send_rich("123", "표 | 내용", None, None)
    blocked = await adapter.send("123", "반복", metadata={"notify": True})

    assert result is not None and result.error_kind == "peer_flood"
    assert not blocked.success and blocked.error_kind == "peer_flood"
    adapter._bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_final_edit_peer_flood_does_not_plain_text_retry():
    adapter = _make_adapter()
    adapter._bot.edit_message_text.side_effect = PeerFloodError(retry_after=30.0)

    result = await adapter.edit_message("123", "10", "최종 응답", finalize=True)

    assert not result.success and result.error_kind == "peer_flood"
    assert adapter._bot.edit_message_text.await_count == 1


@pytest.mark.asyncio
async def test_overflow_edit_peer_flood_opens_same_chat_circuit(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "MAX_MESSAGE_LENGTH", 20)
    adapter._bot.edit_message_text.side_effect = PeerFloodError(retry_after=30.0)

    result = await adapter.edit_message(
        "123", "10", "긴 최종 응답 " * 20, finalize=True
    )
    blocked = await adapter.send("123", "반복", metadata={"notify": True})

    assert not result.success and result.error_kind == "peer_flood"
    assert not blocked.success and blocked.error_kind == "peer_flood"
    adapter._bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_open_peer_flood_circuit_blocks_all_outbound_send_entrypoints(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.asyncio.get_running_loop",
        lambda: type("Loop", (), {"time": lambda self: 100.0})(),
    )
    adapter._telegram_peer_flood_until["123"] = 200.0

    results = [
        await adapter.send_voice("123", "missing.ogg"),
        await adapter.send_image_file("123", "missing.png"),
        await adapter.send_document("123", "missing.txt"),
        await adapter.send_video("123", "missing.mp4"),
        await adapter.send_image("123", "not-a-url"),
        await adapter.send_animation("123", "not-a-url"),
        await adapter.send_update_prompt("123", "update?"),
        await adapter.send_exec_approval("123", "cmd", "session"),
        await adapter.send_slash_confirm("123", "title", "body", "session", "confirm"),
        await adapter.send_clarify("123", "question", ["yes"], "clarify", "session"),
        await adapter.send_model_picker("123", [], "model", "provider", "session", None),
        await adapter.send_choice_picker("123", "title", [{"value": "x"}], "session", None),
    ]
    await adapter.send_multiple_images("123", [("https://example.com/a.png", "a")])

    assert all(result.error_kind == "peer_flood" for result in results)
    assert all(
        getattr(adapter._bot, method).await_count == 0
        for method in (
            "send_message",
            "send_voice",
            "send_audio",
            "send_media_group",
            "send_photo",
            "send_document",
            "send_video",
            "send_animation",
        )
    )


@pytest.mark.asyncio
async def test_peer_flood_from_control_prompt_opens_circuit_without_fallback():
    adapter = _make_adapter()
    adapter._bot.send_message.side_effect = PeerFloodError(retry_after=60.0)

    result = await adapter.send_exec_approval("123", "cmd", "session")
    blocked = await adapter.send_choice_picker(
        "123", "title", [{"value": "x"}], "session", None
    )

    assert result.error_kind == "peer_flood"
    assert blocked.error_kind == "peer_flood"
    assert adapter._bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_dm_topic_typing_fallback_peer_flood_opens_circuit():
    adapter = _make_adapter()
    adapter._bot.send_chat_action.side_effect = [
        RuntimeError("message thread not found"),
        PeerFloodError(retry_after=60.0),
    ]

    await adapter.send_typing(
        "123",
        metadata={"thread_id": "99", "telegram_dm_topic_reply_fallback": True},
    )
    await adapter.send_typing("123")

    circuit = adapter._peer_flood_circuit_result("123")
    assert circuit is not None and circuit.error_kind == "peer_flood"
    assert adapter._bot.send_chat_action.await_count == 2


@pytest.mark.asyncio
async def test_peer_flood_from_each_media_send_opens_circuit_without_fallback(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter._probe_voice_duration_seconds",
        lambda _path: 1,
    )
    files = {}
    for suffix in ("ogg", "png", "txt", "mp4"):
        path = tmp_path / f"media.{suffix}"
        path.write_bytes(b"test")
        files[suffix] = str(path)

    cases = (
        ("send_voice", (files["ogg"],), "send_voice"),
        ("send_multiple_images", ([("https://example.com/a.png", "a")],), "send_media_group"),
        ("send_image_file", (files["png"],), "send_photo"),
        ("send_document", (files["txt"],), "send_document"),
        ("send_video", (files["mp4"],), "send_video"),
        ("send_image", ("https://example.com/a.png",), "send_photo"),
        ("send_animation", ("https://example.com/a.gif",), "send_animation"),
    )

    for index, (entrypoint, args, bot_method) in enumerate(cases):
        adapter = _make_adapter()
        getattr(adapter._bot, bot_method).side_effect = PeerFloodError(retry_after=60.0)
        chat_id = str(123 + index)

        await asyncio.wait_for(getattr(adapter, entrypoint)(chat_id, *args), timeout=1.0)

        circuit = adapter._peer_flood_circuit_result(chat_id)
        assert circuit is not None and circuit.error_kind == "peer_flood", entrypoint
        assert getattr(adapter._bot, bot_method).await_count == 1, entrypoint


@pytest.mark.asyncio
async def test_rich_draft_peer_flood_does_not_fall_back_to_legacy(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_should_attempt_rich_draft", lambda _content: True)
    adapter._bot.do_api_request.side_effect = PeerFloodError(retry_after=60.0)

    result = await adapter.send_draft("123", 7, "표 | 내용")

    assert not result.success and result.error_kind == "peer_flood"
    adapter._bot.send_message_draft.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_draft_checks_open_circuit_and_defangs_legacy_text(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_should_attempt_rich_draft", lambda _content: False)
    adapter._bot.send_message_draft.return_value = True

    delivered = await adapter.send_draft("123", 8, "//www.coupang.com/item")
    adapter._telegram_peer_flood_until["123"] = asyncio.get_running_loop().time() + 60
    blocked = await adapter.send_draft("123", 9, "blocked")

    assert delivered.success
    assert adapter._bot.send_message_draft.await_args_list[0].kwargs["text"] == (
        r"//www\.coupang\[\.\]com/item"
    )
    assert not blocked.success and blocked.error_kind == "peer_flood"
    assert adapter._bot.send_message_draft.await_count == 1


@pytest.mark.asyncio
async def test_open_peer_flood_keeps_later_local_expiry():
    adapter = _make_adapter()
    now = asyncio.get_running_loop().time()
    adapter._telegram_peer_flood_until["123"] = now + 120.0

    adapter._open_peer_flood_circuit("123", PeerFloodError(retry_after=10.0))

    remaining = adapter._peer_flood_remaining("123")
    assert remaining is not None
    assert 119.0 <= remaining <= 120.0


@pytest.mark.asyncio
async def test_adapter_reads_circuit_opened_by_other_process(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.persistent_peer_flood_remaining_async",
        AsyncMock(return_value=55.0),
    )

    result = await adapter.send("123", "blocked", metadata={"notify": True})

    assert not result.success and result.error_kind == "peer_flood"
    assert 54.0 <= result.retry_after <= 55.0
    adapter._bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dm_topic_seed_and_reaction_peer_flood_open_circuit(monkeypatch):
    adapter = _make_adapter()
    adapter._dm_topics_config = [{"chat_id": 123, "topics": [{"name": "General"}]}]
    adapter._dm_topics = {}
    monkeypatch.setattr(adapter, "_persist_dm_topic_thread_id", lambda *args, **kwargs: None)
    adapter._bot.create_forum_topic.return_value.message_thread_id = 9
    adapter._bot.send_message.side_effect = PeerFloodError(retry_after=60.0)

    await adapter._setup_dm_topics()
    reacted = await adapter._set_reaction("123", "10", "👍")

    assert not reacted
    assert adapter._peer_flood_circuit_result("123") is not None
    adapter._bot.set_message_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_callback_edit_peer_flood_blocks_callback_fallback_and_answer(monkeypatch):
    adapter = _make_adapter()
    adapter._choice_picker_state["123"] = {
        "choices": [{"value": "x"}],
        "on_choice_selected": AsyncMock(return_value="done"),
    }
    monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *args, **kwargs: True)
    query = AsyncMock()
    query.data = "cp:0"
    query.message.chat_id = 123
    query.message.chat.type = "private"
    query.message.message_thread_id = None
    query.from_user.id = 123
    query.from_user.first_name = "User"
    query.edit_message_text.side_effect = PeerFloodError(retry_after=60.0)
    update = type("Update", (), {"callback_query": query})()

    await adapter._handle_callback_query(update, None)

    assert query.edit_message_text.await_count == 1
    query.answer.assert_not_awaited()
    assert adapter._peer_flood_circuit_result("123") is not None


@pytest.mark.asyncio
async def test_delete_message_checks_and_opens_peer_flood_circuit():
    adapter = _make_adapter()
    adapter._bot.delete_message.side_effect = PeerFloodError(retry_after=60.0)

    assert not await adapter.delete_message("123", "10")
    assert not await adapter.delete_message("123", "11")

    assert adapter._bot.delete_message.await_count == 1
    assert adapter._peer_flood_circuit_result("123") is not None


@pytest.mark.asyncio
async def test_ensure_forum_commands_checks_and_opens_peer_flood_circuit():
    adapter = _make_adapter()
    adapter._forum_lock = asyncio.Lock()
    adapter._forum_command_registered = set()
    adapter._bot.set_my_commands.side_effect = PeerFloodError(retry_after=60.0)
    message = SimpleNamespace(chat=SimpleNamespace(is_forum=True, id=123))

    await adapter._ensure_forum_commands(message)
    await adapter._ensure_forum_commands(message)

    assert adapter._bot.set_my_commands.await_count == 1
    assert adapter._peer_flood_circuit_result("123") is not None


@pytest.mark.parametrize(
    ("entrypoint", "suffix", "bot_method"),
    [
        ("send_voice", "mp3", "send_audio"),
        ("send_image_file", "png", "send_photo"),
        ("send_document", "txt", "send_document"),
        ("send_video", "mp4", "send_video"),
        ("send_image", None, "send_photo"),
        ("send_animation", None, "send_animation"),
        ("send_multiple_images", None, "send_media_group"),
    ],
)
@pytest.mark.asyncio
async def test_every_media_caption_uses_shared_policy(
    entrypoint, suffix, bot_method, tmp_path, monkeypatch
):
    adapter = _make_adapter()
    getattr(adapter._bot, bot_method).return_value = SimpleNamespace(message_id=10)
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter._probe_voice_duration_seconds",
        lambda _path: 1,
    )
    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda _url: True)

    class _InputMediaPhoto(dict):
        def __init__(self, media, caption=None):
            super().__init__(media=media, caption=caption)

    monkeypatch.setattr("telegram.InputMediaPhoto", _InputMediaPhoto)
    caption = "https://www.coupang.com/item/1 " + ("😀" * 700)

    if entrypoint == "send_multiple_images":
        await adapter.send_multiple_images(
            "123", [("https://example.com/image.png", caption)]
        )
        media = adapter._bot.send_media_group.await_args.kwargs["media"]
        sent_caption = media[0]["caption"]
    elif entrypoint in {"send_image", "send_animation"}:
        await getattr(adapter, entrypoint)(
            "123", "https://example.com/image.png", caption=caption
        )
        sent_caption = getattr(adapter._bot, bot_method).await_args.kwargs["caption"]
    else:
        media_path = tmp_path / f"media.{suffix}"
        media_path.write_bytes(b"test")
        await getattr(adapter, entrypoint)("123", str(media_path), caption=caption)
        sent_caption = getattr(adapter._bot, bot_method).await_args.kwargs["caption"]

    assert "coupang.com" not in sent_caption.lower(), entrypoint
    assert "coupang[.]com" in sent_caption.lower(), entrypoint
    assert len(sent_caption.encode("utf-16-le")) // 2 <= 1024, entrypoint


@pytest.mark.asyncio
async def test_async_peer_flood_open_uses_async_persistence_wrapper(monkeypatch):
    adapter = _make_adapter()
    adapter._bot.send_message.side_effect = PeerFloodError(retry_after=60.0)
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.persist_peer_flood_circuit",
        lambda *_args: (_ for _ in ()).throw(AssertionError("sync DB call")),
    )
    async_open = AsyncMock(return_value=60.0)
    monkeypatch.setattr(
        "plugins.platforms.telegram.outbound_circuit.open_circuit_async",
        async_open,
    )

    result = await adapter.send("123", "hello", metadata={"notify": True})

    assert result.error_kind == "peer_flood"
    async_open.assert_awaited_once_with("123", 60.0)


@pytest.mark.asyncio
async def test_send_typing_uses_async_circuit_read(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.persistent_peer_flood_remaining",
        lambda *_args: (_ for _ in ()).throw(AssertionError("sync DB call")),
    )
    async_remaining = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.persistent_peer_flood_remaining_async",
        async_remaining,
    )

    await adapter.send_typing("123")

    async_remaining.assert_awaited_once_with("123")
    adapter._bot.send_chat_action.assert_awaited_once()


@pytest.mark.asyncio
async def test_global_command_batch_stops_after_first_peer_flood():
    from plugins.platforms.telegram import outbound_circuit

    adapter = _make_adapter()
    adapter._bot.set_my_commands.side_effect = PeerFloodError(retry_after=60.0)
    adapter._set_status_indicator = AsyncMock()
    adapter._setup_dm_topics = AsyncMock()

    await adapter._run_post_connect_housekeeping()
    await adapter._run_post_connect_housekeeping()

    assert adapter._bot.set_my_commands.await_count == 1
    assert outbound_circuit.remaining(outbound_circuit.GLOBAL_CIRCUIT_KEY) is not None


@pytest.mark.asyncio
async def test_status_indicator_checks_and_opens_global_circuit():
    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="test-token", extra={"status_indicator": True})
    )
    adapter._bot = AsyncMock()
    adapter._bot.set_my_short_description.side_effect = PeerFloodError(60.0)

    await adapter._set_status_indicator(online=True)
    await adapter._set_status_indicator(online=False)

    assert adapter._bot.set_my_short_description.await_count == 1


@pytest.mark.asyncio
async def test_inline_answer_uses_normalized_user_circuit_key():
    adapter = _make_adapter()
    adapter._is_callback_user_authorized = lambda *_args, **_kwargs: False
    first = SimpleNamespace(
        query="",
        offset="",
        from_user=SimpleNamespace(id=42, username="tester"),
        answer=AsyncMock(side_effect=PeerFloodError(60.0)),
    )
    second = SimpleNamespace(
        query="",
        offset="",
        from_user=SimpleNamespace(id="042", username="tester"),
        answer=AsyncMock(),
    )

    await adapter._handle_inline_query(SimpleNamespace(inline_query=first), None)
    await adapter._handle_inline_query(SimpleNamespace(inline_query=second), None)

    first.answer.assert_awaited_once()
    second.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_command_scope_error_log_redacts_bot_token(caplog):
    adapter = _make_adapter()
    secret = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
    adapter._bot.set_my_commands.side_effect = RuntimeError(f"request token {secret}")
    adapter._set_status_indicator = AsyncMock()
    adapter._setup_dm_topics = AsyncMock()

    with caplog.at_level("WARNING"):
        await adapter._run_post_connect_housekeeping()

    assert secret not in caplog.text


@pytest.mark.asyncio
async def test_adapter_write_failure_blocks_next_adapter_call(monkeypatch):
    from plugins.platforms.telegram import outbound_circuit

    first = _make_adapter()
    first._bot.send_message.side_effect = PeerFloodError(60.0)
    real_connection = outbound_circuit._connection

    def broken_connection():
        raise OSError("disk unavailable")

    async def fail_persist(chat_id, delay):
        outbound_circuit._connection = broken_connection
        try:
            return await outbound_circuit.open_circuit_async(chat_id, delay)
        finally:
            outbound_circuit._connection = real_connection

    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.persist_peer_flood_circuit_async",
        fail_persist,
    )
    failed = await first.send("123", "first", metadata={"notify": True})
    second = _make_adapter()

    blocked = await second.send("123", "second", metadata={"notify": True})

    assert failed.error_kind == "peer_flood"
    assert blocked.error_kind == "peer_flood"
    second._bot.send_message.assert_not_awaited()
