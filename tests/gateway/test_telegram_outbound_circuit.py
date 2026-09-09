"""Telegram 프로세스 간 PEER_FLOOD 발송 회로 시험."""

import asyncio
import sqlite3
import threading
import time

import pytest

from plugins.platforms.telegram import outbound_circuit


def test_open_circuit_persists_only_chat_key_and_expiry(tmp_path, monkeypatch):
    monkeypatch.setattr(outbound_circuit, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(outbound_circuit.time, "time", lambda: 100.0)

    remaining = outbound_circuit.open_circuit("123", 42.0)

    assert remaining == 42.0
    assert outbound_circuit.remaining("123") == 42.0
    with sqlite3.connect(tmp_path / "state.db") as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(telegram_peer_flood_circuits)")]
        row = conn.execute("SELECT * FROM telegram_peer_flood_circuits").fetchone()
    assert columns == ["chat_key", "expires_at"]
    assert row == ("123", 142.0)


def test_open_circuit_preserves_later_existing_expiry(tmp_path, monkeypatch):
    monkeypatch.setattr(outbound_circuit, "_db_path", lambda: tmp_path / "state.db")
    now = {"value": 100.0}
    monkeypatch.setattr(outbound_circuit.time, "time", lambda: now["value"])

    outbound_circuit.open_circuit("123", 60.0)
    now["value"] = 110.0
    outbound_circuit.open_circuit("123", 10.0)

    assert outbound_circuit.remaining("123") == 50.0


def test_expired_circuit_is_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(outbound_circuit, "_db_path", lambda: tmp_path / "state.db")
    now = {"value": 100.0}
    monkeypatch.setattr(outbound_circuit.time, "time", lambda: now["value"])
    outbound_circuit.open_circuit("123", 10.0)

    now["value"] = 110.1

    assert outbound_circuit.remaining("123") is None
    with sqlite3.connect(tmp_path / "state.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM telegram_peer_flood_circuits").fetchone()[0] == 0


def test_invalid_delay_uses_bounded_default(tmp_path, monkeypatch):
    monkeypatch.setattr(outbound_circuit, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(outbound_circuit.time, "time", lambda: 100.0)

    assert outbound_circuit.open_circuit("123", float("nan")) == 300.0
    assert outbound_circuit.open_circuit("456", 900.0) == 300.0


@pytest.mark.parametrize("chat_id", [123, "123", " 123 "])
def test_chat_key_normalization_is_shared(chat_id):
    assert outbound_circuit.chat_key(chat_id) == "123"


def test_locked_database_returns_finite_safe_block_with_short_latency(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(outbound_circuit, "_db_path", lambda: db_path)
    outbound_circuit.open_circuit("seed", 1.0)
    lock = sqlite3.connect(db_path, timeout=0)
    lock.execute("PRAGMA journal_mode=DELETE")
    lock.execute("BEGIN EXCLUSIVE")
    try:
        started = time.monotonic()
        result = outbound_circuit.remaining("123")
        elapsed = time.monotonic() - started
    finally:
        lock.rollback()
        lock.close()

    assert result is not None
    assert 0 < result <= outbound_circuit.DB_ERROR_COOLDOWN_SECONDS
    assert elapsed < 0.75


def test_repeated_db_read_error_has_finite_block_then_one_release(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    clock = {"value": 100.0}
    monkeypatch.setattr(outbound_circuit, "_db_path", lambda: db_path)
    monkeypatch.setattr(outbound_circuit.time, "monotonic", lambda: clock["value"])

    def broken_connection():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(outbound_circuit, "_connection", broken_connection)

    assert outbound_circuit.remaining("123") == outbound_circuit.DB_ERROR_COOLDOWN_SECONDS
    clock["value"] += 2.0
    assert outbound_circuit.remaining("123") == pytest.approx(3.0)
    clock["value"] += 3.1
    assert outbound_circuit.remaining("123") is None


def test_open_write_error_fails_closed_for_target_chat(tmp_path, monkeypatch):
    clock = {"value": 100.0}
    monkeypatch.setattr(outbound_circuit, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(outbound_circuit.time, "monotonic", lambda: clock["value"])

    real_connection = outbound_circuit._connection

    def broken_connection():
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(outbound_circuit, "_connection", broken_connection)

    assert outbound_circuit.open_circuit("00123", 60.0) == 60.0
    monkeypatch.setattr(outbound_circuit, "_connection", real_connection)
    blocked = outbound_circuit.remaining("123")
    assert blocked == pytest.approx(outbound_circuit.DB_ERROR_COOLDOWN_SECONDS)
    clock["value"] += outbound_circuit.DB_ERROR_COOLDOWN_SECONDS + 0.1
    assert outbound_circuit.remaining("123") is None


@pytest.mark.parametrize(
    ("sync_name", "async_name", "args"),
    [
        ("remaining", "remaining_async", ("123",)),
        ("open_circuit", "open_circuit_async", ("123", 60.0)),
    ],
)
@pytest.mark.asyncio
async def test_async_wrappers_keep_event_loop_ticking_with_mocked_db_lock(
    monkeypatch, sync_name, async_name, args
):
    release = threading.Event()
    entered = threading.Event()

    def slow_operation(*_args):
        entered.set()
        release.wait(timeout=1.0)
        return None

    monkeypatch.setattr(outbound_circuit, sync_name, slow_operation)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while not entered.is_set():
            await asyncio.sleep(0.001)
        for _ in range(5):
            await asyncio.sleep(0.01)
            ticks += 1
        release.set()

    result, _ = await asyncio.wait_for(
        asyncio.gather(getattr(outbound_circuit, async_name)(*args), ticker()),
        timeout=1.5,
    )

    assert result is None
    assert ticks == 5


@pytest.mark.asyncio
async def test_adapter_db_lock_check_does_not_stop_event_loop(tmp_path, monkeypatch):
    import asyncio

    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    db_path = tmp_path / "state.db"
    monkeypatch.setattr(outbound_circuit, "_db_path", lambda: db_path)
    outbound_circuit.open_circuit("seed", 1.0)
    lock = sqlite3.connect(db_path, timeout=0)
    lock.execute("PRAGMA journal_mode=DELETE")
    lock.execute("BEGIN EXCLUSIVE")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = type("Bot", (), {})()
    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(20):
            await asyncio.sleep(0.005)
            ticks += 1

    try:
        started = time.monotonic()
        result, _ = await asyncio.gather(adapter.send("123", "blocked"), ticker())
        elapsed = time.monotonic() - started
    finally:
        lock.rollback()
        lock.close()

    assert not result.success and result.error_kind == "peer_flood"
    assert ticks == 20
    assert elapsed < 0.75


@pytest.mark.asyncio
async def test_single_gateway_applies_policy_to_text_caption_and_edit(monkeypatch):
    from plugins.platforms.telegram.outbound_policy import TelegramOutboundGateway

    calls = []

    async def operation(**kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.setattr(outbound_circuit, "remaining_async", lambda _key: asyncio.sleep(0, result=None))
    gateway = TelegramOutboundGateway()

    assert await gateway.write(
        "00123",
        "send_photo",
        operation,
        caption="https://www.coupang.com/vp/products/1",
    ) == "ok"
    assert await gateway.write(
        "123",
        "edit_message_text",
        operation,
        text="see coupang.com now",
    ) == "ok"
    assert "coupang[.]com" in calls[0]["caption"]
    assert "coupang[.]com" in calls[1]["text"]


@pytest.mark.asyncio
async def test_gateway_defangs_nested_raw_payload_without_mutating_caller(monkeypatch):
    from copy import deepcopy

    from plugins.platforms.telegram.outbound_policy import TelegramOutboundGateway

    received = []
    payload = {
        "rich_message": {
            "text": "https://www.coupang.com/p?next=https://example.com/#ok",
            "rows": ["//link.coupang.com/a/X", ("https://example.com/coupang.com", 7)],
        }
    }
    original = deepcopy(payload)

    async def operation(*args, **kwargs):
        received.append((args, kwargs))
        return "ok"

    monkeypatch.setattr(outbound_circuit, "remaining_async", lambda _key: asyncio.sleep(0, result=None))

    result = await TelegramOutboundGateway().write(
        "123", "do_api_request", operation, "sendRichMessage", api_kwargs=payload
    )

    assert result == "ok"
    sent = received[0][1]["api_kwargs"]
    assert sent["rich_message"]["text"].startswith("https://www.coupang[.]com/")
    assert sent["rich_message"]["rows"][0] == "//link.coupang[.]com/a/X"
    assert sent["rich_message"]["rows"][1] == ("https://example.com/coupang.com", 7)
    assert payload == original
    assert sent is not payload


@pytest.mark.asyncio
async def test_gateway_rebuilds_supported_ptb_payloads_without_mutating_caller():
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import asyncio
        from telegram import InlineQueryResultArticle, InputMediaPhoto, InputTextMessageContent
        from plugins.platforms.telegram import outbound_circuit
        from plugins.platforms.telegram.outbound_policy import TelegramOutboundGateway

        async def main():
            outbound_circuit.remaining_async = lambda key: asyncio.sleep(0, result=None)
            media = InputMediaPhoto("file-id", caption="https://coupang.com/media?x=1#f")
            article = InlineQueryResultArticle(
                "id", "title", InputTextMessageContent("https://link.coupang.com/a/inline")
            )
            received = []

            async def operation(*args, **kwargs):
                received.append((args, kwargs))
                return True

            await TelegramOutboundGateway().write(
                "123", "answer", operation, [article], api_kwargs={"media": [media]}
            )
            sent_article = received[0][0][0][0]
            sent_media = received[0][1]["api_kwargs"]["media"][0]
            assert sent_article.input_message_content.message_text == (
                "https://link.coupang[.]com/a/inline"
            )
            assert sent_media.caption == "https://coupang[.]com/media?x=1#f"
            assert article.input_message_content.message_text == "https://link.coupang.com/a/inline"
            assert media.caption == "https://coupang.com/media?x=1#f"
            assert sent_article is not article
            assert sent_media is not media

        asyncio.run(main())
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_gateway_defangs_default_value_string_before_ptb_serialization():
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import asyncio
        from telegram import LinkPreviewOptions
        from telegram._utils.defaultvalue import DefaultValue
        from plugins.platforms.telegram import outbound_circuit
        from plugins.platforms.telegram.outbound_policy import TelegramOutboundGateway

        async def main():
            outbound_circuit.remaining_async = lambda key: asyncio.sleep(0, result=None)
            preview = LinkPreviewOptions(
                url=DefaultValue("https://link.coupang.com/a/BYPASS")
            )
            received = []

            async def operation(**kwargs):
                received.append(kwargs)
                return True

            await TelegramOutboundGateway().write(
                "123",
                "send_message",
                operation,
                text="safe",
                link_preview_options=preview,
            )
            sent = received[0]["link_preview_options"]
            assert sent.to_dict()["url"] == "https://link.coupang[.]com/a/BYPASS"
            assert preview.to_dict()["url"] == "https://link.coupang.com/a/BYPASS"

        asyncio.run(main())
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_gateway_defangs_telegram_subclass_dict_before_ptb_serialization():
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import asyncio
        from telegram import InlineKeyboardButton, TelegramObject
        from plugins.platforms.telegram import outbound_circuit
        from plugins.platforms.telegram.outbound_policy import TelegramOutboundGateway

        class CustomMarkup(TelegramObject):
            pass

        async def main():
            outbound_circuit.remaining_async = lambda key: asyncio.sleep(0, result=None)
            markup = CustomMarkup()
            markup.inline_keyboard = [
                [InlineKeyboardButton("open", url="https://link.coupang.com/a/BYPASS")]
            ]
            received = []

            async def operation(**kwargs):
                received.append(kwargs)
                return True

            await TelegramOutboundGateway().write(
                "123", "send_message", operation, text="safe", reply_markup=markup
            )
            sent = received[0]["reply_markup"]
            assert sent.to_dict()["inline_keyboard"][0][0]["url"] == (
                "https://link.coupang[.]com/a/BYPASS"
            )
            assert markup.to_dict()["inline_keyboard"][0][0]["url"] == (
                "https://link.coupang.com/a/BYPASS"
            )
            assert sent.inline_keyboard is not markup.inline_keyboard

        asyncio.run(main())
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.asyncio
async def test_gateway_handles_recursive_payload_cycle_and_depth_fail_closed(monkeypatch):
    from plugins.platforms.telegram.outbound_policy import (
        TelegramOutboundGateway,
        TelegramUnsafeContentBlocked,
    )

    received = []
    cycle = ["https://coupang.com/cycle"]
    cycle.append(cycle)

    async def operation(*args, **kwargs):
        received.append((args, kwargs))
        return True

    monkeypatch.setattr(outbound_circuit, "remaining_async", lambda _key: asyncio.sleep(0, result=None))
    gateway = TelegramOutboundGateway()

    await gateway.write("123", "send_media_group", operation, media=cycle)
    sent_cycle = received[0][1]["media"]
    assert sent_cycle[0] == "https://coupang[.]com/cycle"
    assert sent_cycle[1] is sent_cycle
    assert cycle[0] == "https://coupang.com/cycle"
    assert cycle[1] is cycle

    deep = "https://coupang.com/deep"
    for _ in range(gateway._MAX_CONTENT_DEPTH + 1):
        deep = [deep]
    with pytest.raises(TelegramUnsafeContentBlocked):
        await gateway.write("123", "send_media_group", operation, media=deep)
    assert len(received) == 1


@pytest.mark.asyncio
async def test_callback_and_control_text_enter_same_policy_gateway(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setattr(outbound_circuit, "_db_path", lambda: tmp_path / "state.db")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._bot.send_message.return_value = SimpleNamespace(message_id=7)

    control = await adapter.send_exec_approval(
        "123",
        "open https://www.coupang.com/item/1",
        "session",
    )
    query = SimpleNamespace(
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
        from_user=SimpleNamespace(id=123),
        message=SimpleNamespace(chat_id=123, text="callback"),
    )
    guarded_query = adapter._guard_callback_query(query, "123")
    await guarded_query.edit_message_text(
        text="https://link.coupang.com/a/ABC"
    )

    assert control.success
    assert "coupang.com" not in adapter._bot.send_message.await_args.kwargs["text"].lower()
    assert "coupang[.]com" in adapter._bot.send_message.await_args.kwargs["text"].lower()
    callback_text = query.edit_message_text.await_args.kwargs["text"].lower()
    assert "coupang.com" not in callback_text
    assert "coupang[.]com" in callback_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("from_user", "expected_key"),
    [
        (type("User", (), {"id": 321, "first_name": "User"})(), "321"),
        (None, "telegram:callback:unknown"),
    ],
)
async def test_message_less_callback_answer_enters_gateway_once(
    tmp_path, monkeypatch, from_user, expected_key
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setattr(outbound_circuit, "_db_path", lambda: tmp_path / "state.db")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    query = SimpleNamespace(
        answer=AsyncMock(),
        data="ea:once:not-an-integer",
        from_user=from_user,
        inline_message_id="inline-42",
        message=None,
    )
    gateway = adapter._outbound_gateway
    original_write = gateway.write
    gateway.write = AsyncMock(side_effect=original_write)

    await adapter._handle_callback_query(SimpleNamespace(callback_query=query), None)

    query.answer.assert_awaited_once_with(text="Invalid approval data.")
    gateway.write.assert_awaited_once()
    assert gateway.write.await_args.args[:2] == (expected_key, "answer")


@pytest.mark.asyncio
async def test_inline_callback_answer_and_edit_each_enter_gateway_once(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from tools import approval

    monkeypatch.setattr(outbound_circuit, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(approval, "resolve_gateway_approval", lambda *_args: 1)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._approval_state[7] = "session"
    monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *_args, **_kwargs: True)
    query = SimpleNamespace(
        answer=AsyncMock(),
        data="ea:once:7",
        edit_message_text=AsyncMock(),
        from_user=SimpleNamespace(id=321, first_name="User"),
        inline_message_id="inline-42",
        message=None,
    )
    gateway = adapter._outbound_gateway
    original_write = gateway.write
    gateway.write = AsyncMock(side_effect=original_write)

    await adapter._handle_callback_query(SimpleNamespace(callback_query=query), None)

    assert query.answer.await_count == 1
    assert query.edit_message_text.await_count == 1
    assert [call.args[:2] for call in gateway.write.await_args_list] == [
        ("321", "answer"),
        ("321", "edit_message_text"),
    ]


@pytest.mark.asyncio
async def test_single_gateway_opens_immediately_and_stops_same_batch(monkeypatch):
    from plugins.platforms.telegram.outbound_policy import (
        TelegramOutboundGateway,
        TelegramPeerFloodBlocked,
    )

    opened = []
    attempts = 0

    async def remaining(key):
        return 30.0 if opened and key == "123" else None

    async def open_async(key, delay):
        opened.append((key, delay))
        return delay

    async def peer_flood_operation(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("Telegram says PEER_FLOOD retry_after=60")

    monkeypatch.setattr(outbound_circuit, "remaining_async", remaining)
    monkeypatch.setattr(outbound_circuit, "open_circuit_async", open_async)
    gateway = TelegramOutboundGateway()

    with pytest.raises(TelegramPeerFloodBlocked):
        await gateway.write("00123", "edit_message_text", peer_flood_operation, text="x")
    with pytest.raises(TelegramPeerFloodBlocked):
        await gateway.write("123", "send_message", peer_flood_operation, text="fallback")

    assert attempts == 1
    assert opened and opened[0][0] == "123"


@pytest.mark.asyncio
async def test_adapter_does_not_record_gateway_peer_flood_twice(monkeypatch):
    from unittest.mock import AsyncMock

    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._bot.send_message.side_effect = RuntimeError("PEER_FLOOD")
    opened = AsyncMock(return_value=30.0)
    monkeypatch.setattr(outbound_circuit, "open_circuit_async", opened)
    monkeypatch.setattr(outbound_circuit, "remaining_async", AsyncMock(return_value=None))

    result = await adapter.send("123", "hello", metadata={"notify": True})

    assert not result.success and result.error_kind == "peer_flood"
    assert result.retry_after == pytest.approx(30.0)
    opened.assert_awaited_once_with("123", 300.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [
        {"type": "chat", "chat_id": "00123"},
        {"type": "chat", "chat_id": 123},
    ],
)
async def test_chat_command_scope_peer_flood_is_persistent_and_not_global(
    tmp_path, monkeypatch, scope
):
    from plugins.platforms.telegram.outbound_policy import (
        GuardedTelegramTarget,
        TelegramOutboundGateway,
        TelegramPeerFloodBlocked,
    )

    monkeypatch.setattr(outbound_circuit, "_db_path", lambda: tmp_path / "state.db")
    calls = []

    class FirstBot:
        async def set_my_commands(self, *_args, **_kwargs):
            calls.append("first-chat")
            error = RuntimeError("PEER_FLOOD")
            error.retry_after = 42
            raise error

    with pytest.raises(TelegramPeerFloodBlocked) as opened:
        await GuardedTelegramTarget(FirstBot(), TelegramOutboundGateway()).set_my_commands(
            [], scope=scope
        )
    assert opened.value.retry_after == pytest.approx(42.0, abs=1.0)

    class NextBot:
        async def set_my_commands(self, *_args, **_kwargs):
            calls.append("next")
            return True

    next_target = GuardedTelegramTarget(NextBot(), TelegramOutboundGateway())
    with pytest.raises(TelegramPeerFloodBlocked):
        await next_target.set_my_commands([], scope={"chat_id": "123"})
    assert await next_target.set_my_commands([], scope={"type": "default"}) is True

    assert calls == ["first-chat", "next"]
    assert outbound_circuit.remaining("123") == pytest.approx(42.0, abs=1.0)
    assert outbound_circuit.remaining(outbound_circuit.GLOBAL_CIRCUIT_KEY) is None


def test_db_read_error_isolated_by_db_path_and_chat_key(tmp_path, monkeypatch):
    clock = {"value": 100.0}
    monkeypatch.setattr(outbound_circuit, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(outbound_circuit.time, "monotonic", lambda: clock["value"])
    outbound_circuit._READ_ERROR_UNTIL.clear()

    real_connection = outbound_circuit._connection

    def broken_connection():
        raise sqlite3.OperationalError("read failed for https://secret.invalid/token")

    monkeypatch.setattr(outbound_circuit, "_connection", broken_connection)
    assert outbound_circuit.remaining("123") == outbound_circuit.DB_ERROR_COOLDOWN_SECONDS
    monkeypatch.setattr(outbound_circuit, "_connection", real_connection)

    assert outbound_circuit.remaining("123") == pytest.approx(5.0)
    assert outbound_circuit.remaining("456") is None


def test_production_telegram_writes_use_only_the_policy_gateway():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    targets = (
        root / "plugins/platforms/telegram/adapter.py",
        root / "tools/send_message_tool.py",
        root / "gateway/run.py",
    )
    write_methods = {
        "answer", "create_forum_topic", "delete_message", "do_api_request",
        "edit_forum_topic", "edit_message_text", "pin_chat_message",
        "send_audio", "send_animation", "send_chat_action", "send_document",
        "send_media_group", "send_message",
        "send_message_draft", "send_photo", "send_video", "send_voice",
        "set_message_reaction", "set_my_commands", "set_my_short_description",
    }
    violations = []
    forbidden_receivers = {"self._bot", "bot", "inline_query", "self._query"}
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr not in write_methods:
                continue
            receiver = ast.unparse(node.value)
            if receiver in forbidden_receivers:
                violations.append(f"{path.relative_to(root)}:{node.lineno}:{receiver}.{node.attr}")

    assert violations == []
