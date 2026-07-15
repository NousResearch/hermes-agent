import asyncio
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_adapter_module  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


def _public_channel(**attributes):
    default_role = object()
    guild = SimpleNamespace(id=1, default_role=default_role)

    def permissions_for(role):
        assert role is default_role
        return SimpleNamespace(view_channel=True)

    return SimpleNamespace(
        guild=guild,
        permissions_for=permissions_for,
        **attributes,
    )


def _mark_public(channel):
    default_role = object()
    channel.guild = SimpleNamespace(id=1, default_role=default_role)
    channel.permissions_for = lambda role: SimpleNamespace(
        view_channel=role is default_role
    )
    return channel


@pytest.mark.asyncio
async def test_writer_policy_blocks_standalone_discord_rest_egress(monkeypatch):
    monkeypatch.setattr(
        discord_adapter_module,
        "_discord_public_only_policy_required",
        lambda: True,
    )

    result = await discord_adapter_module._standalone_send(
        SimpleNamespace(token="test-token", extra={}),
        "555",
        "must use live adapter",
    )

    assert "standalone REST egress is disabled" in result["error"]


@pytest.mark.asyncio
async def test_live_media_sender_requires_public_target(tmp_path, monkeypatch):
    monkeypatch.setattr(
        discord_adapter_module,
        "_discord_public_only_policy_required",
        lambda: True,
    )
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    media_path = tmp_path / "proof.txt"
    media_path.write_text("proof", encoding="utf-8")
    private_channel = SimpleNamespace(
        id=555,
        guild=SimpleNamespace(id=1, default_role=object()),
        permissions_for=lambda _role: SimpleNamespace(view_channel=False),
        send=AsyncMock(),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _channel_id: private_channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send_media_files(
        "555",
        "blocked",
        [(str(media_path), False)],
    )

    assert result.success is False
    assert "not publicly visible" in (result.error or "")
    private_channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_media_batches_stop_after_public_visibility_is_revoked(
    tmp_path,
    monkeypatch,
):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    state = {"public": True}
    default_role = object()
    channel = SimpleNamespace(
        id=555,
        guild=SimpleNamespace(id=1, default_role=default_role),
        permissions_for=lambda role: SimpleNamespace(
            view_channel=state["public"] and role is default_role
        ),
    )

    async def _send_first_batch(**_kwargs):
        state["public"] = False
        return SimpleNamespace(id=700)

    channel.send = AsyncMock(side_effect=_send_first_batch)
    adapter._client = SimpleNamespace(
        get_channel=lambda _channel_id: channel,
        fetch_channel=AsyncMock(),
    )
    monkeypatch.setattr(
        discord_adapter_module,
        "_discord_public_only_policy_required",
        lambda: True,
    )
    media_files = []
    for index in range(11):
        path = tmp_path / f"proof-{index}.txt"
        path.write_text("proof", encoding="utf-8")
        media_files.append((str(path), False))

    result = await adapter.send_media_files("555", "caption", media_files)

    assert result.success is False
    assert "not publicly visible" in (result.error or "")
    assert result.raw_response == {"message_ids": ["700"]}
    assert channel.send.await_count == 1


@pytest.mark.asyncio
async def test_image_download_rechecks_public_proof_before_send(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    channel = SimpleNamespace(id=555, send=AsyncMock())
    adapter._client = SimpleNamespace(
        get_channel=lambda _channel_id: channel,
        fetch_channel=AsyncMock(),
    )
    response = MagicMock()
    response.status = 200
    response.headers = {"content-type": "image/png"}
    response.read = AsyncMock(return_value=b"image")
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get.return_value = response
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "plugins.platforms.discord.adapter.is_safe_url",
        return_value=True,
    ), patch(
        "plugins.platforms.discord.adapter._discord_policy_public_target_error",
        side_effect=[None, "public proof revoked"],
    ), patch(
        "aiohttp.ClientSession",
        return_value=session_context,
    ), patch("plugins.platforms.discord.adapter.discord.File", return_value=MagicMock()):
        result = await adapter.send_image("555", "https://example.test/image.png")

    assert result.success is False
    assert result.error == "public proof revoked"
    channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_chunked_text_stops_after_public_visibility_is_revoked():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    state = {"public": True}
    default_role = object()
    channel = SimpleNamespace(
        id=555,
        guild=SimpleNamespace(id=1, default_role=default_role),
        permissions_for=lambda role: SimpleNamespace(
            view_channel=state["public"] and role is default_role
        ),
    )

    async def _send_first_chunk(**_kwargs):
        state["public"] = False
        return SimpleNamespace(id=701)

    channel.send = AsyncMock(side_effect=_send_first_chunk)
    adapter._client = SimpleNamespace(
        get_channel=lambda _channel_id: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("555", "x" * 3000)

    assert result.success is False
    assert "not publicly visible" in (result.error or "")
    assert result.raw_response == {"message_ids": ["701"]}
    assert channel.send.await_count == 1


@pytest.mark.asyncio
async def test_verify_public_message_receipt_checks_bot_author_and_content():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    message = SimpleNamespace(
        id=1234,
        author=SimpleNamespace(id=42),
        content="verified delivery",
    )
    channel = _public_channel(
        id=555,
        fetch_message=AsyncMock(return_value=message),
    )
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=42),
        get_channel=lambda _channel_id: channel,
        fetch_channel=AsyncMock(),
    )
    digest = hashlib.sha256(message.content.encode()).hexdigest()

    receipt = await adapter.verify_public_message_receipt(
        channel_id="555",
        message_id="1234",
        expected_content_sha256=digest,
    )

    assert receipt["verified"] is True
    assert receipt["content_sha256"] == digest


@pytest.mark.asyncio
async def test_verify_public_message_receipt_rejects_content_mismatch():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    message = SimpleNamespace(id=1234, author=SimpleNamespace(id=42), content="actual")
    channel = _public_channel(
        id=555,
        fetch_message=AsyncMock(return_value=message),
    )
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=42),
        get_channel=lambda _channel_id: channel,
        fetch_channel=AsyncMock(),
    )

    with pytest.raises(RuntimeError, match="content hash mismatch"):
        await adapter.verify_public_message_receipt(
            channel_id="555",
            message_id="1234",
            expected_content_sha256="a" * 64,
        )


@pytest.mark.asyncio
async def test_send_retries_without_reference_when_reply_target_is_system_message():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    reference_obj = object()
    ref_msg = SimpleNamespace(id=99, to_reference=MagicMock(return_value=reference_obj))
    sent_msg = SimpleNamespace(id=1234)
    send_calls = []

    async def fake_send(*, content, reference=None):
        send_calls.append({"content": content, "reference": reference})
        if len(send_calls) == 1:
            raise RuntimeError(
                "400 Bad Request (error code: 50035): Invalid Form Body\n"
                "In message_reference: Cannot reply to a system message"
            )
        return sent_msg

    channel = _public_channel(
        fetch_message=AsyncMock(return_value=ref_msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("555", "hello", reply_to="99")

    assert result.success is True
    assert result.message_id == "1234"
    assert channel.fetch_message.await_count == 1
    assert channel.send.await_count == 2
    ref_msg.to_reference.assert_called_once_with(fail_if_not_exists=False)
    assert send_calls[0]["reference"] is reference_obj
    assert send_calls[1]["reference"] is None


@pytest.mark.asyncio
async def test_send_retries_without_reference_when_reply_target_is_deleted():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    reference_obj = object()
    ref_msg = SimpleNamespace(id=99, to_reference=MagicMock(return_value=reference_obj))
    sent_msgs = [SimpleNamespace(id=1001), SimpleNamespace(id=1002)]
    send_calls = []

    async def fake_send(*, content, reference=None):
        send_calls.append({"content": content, "reference": reference})
        if len(send_calls) == 1:
            raise RuntimeError(
                "400 Bad Request (error code: 10008): Unknown Message"
            )
        return sent_msgs[len(send_calls) - 2]

    channel = _public_channel(
        fetch_message=AsyncMock(return_value=ref_msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    long_text = "A" * (adapter.MAX_MESSAGE_LENGTH + 50)
    result = await adapter.send("555", long_text, reply_to="99")

    assert result.success is True
    assert result.message_id == "1001"
    assert channel.fetch_message.await_count == 1
    assert channel.send.await_count == 3
    ref_msg.to_reference.assert_called_once_with(fail_if_not_exists=False)
    assert send_calls[0]["reference"] is reference_obj
    assert send_calls[1]["reference"] is None
    assert send_calls[2]["reference"] is None


@pytest.mark.asyncio
async def test_send_does_not_retry_on_unrelated_errors():
    """Regression guard: errors unrelated to the reply reference (e.g. 50013
    Missing Permissions) must NOT trigger the no-reference retry path — they
    should propagate out of the per-chunk loop and surface as a failed
    SendResult so the caller sees the real problem instead of a silent retry.
    """
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    reference_obj = object()
    ref_msg = SimpleNamespace(id=99, to_reference=MagicMock(return_value=reference_obj))
    send_calls = []

    async def fake_send(*, content, reference=None):
        send_calls.append({"content": content, "reference": reference})
        raise RuntimeError(
            "403 Forbidden (error code: 50013): Missing Permissions"
        )

    channel = _public_channel(
        fetch_message=AsyncMock(return_value=ref_msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("555", "hello", reply_to="99")

    # Outer except in adapter.send() wraps propagated errors as SendResult.
    assert result.success is False
    assert "50013" in (result.error or "")
    # Only the first attempt happens — no reference-retry replay.
    assert channel.send.await_count == 1
    assert send_calls[0]["reference"] is reference_obj


@pytest.mark.asyncio
async def test_single_receipt_send_rejects_format_expansion_before_any_post():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    channel = _public_channel(
        type=0,
        send=AsyncMock(return_value=SimpleNamespace(id=1234)),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    content = (
        "| Column Alpha | Column Beta | Column Gamma |\n"
        "|---|---|---|\n"
        + "".join(
            f"| row{index} | value{index} | detail{index} |\n"
            for index in range(50)
        )
    )
    assert len(content) < adapter.MAX_MESSAGE_LENGTH
    assert len(adapter.format_message(content)) > adapter.MAX_MESSAGE_LENGTH

    result = await adapter.send(
        "555",
        content,
        metadata={"require_single_public_receipt": True},
    )

    assert result.success is False
    assert "exactly one formatted public message" in (result.error or "")
    channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_single_receipt_send_allows_exactly_one_formatted_public_post():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    channel = _public_channel(
        type=0,
        send=AsyncMock(return_value=SimpleNamespace(id=1234)),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send(
        "555",
        "One receipt-bound post",
        metadata={"require_single_public_receipt": True},
    )

    assert result.success is True
    assert result.message_id == "1234"
    channel.send.assert_awaited_once_with(
        content="One receipt-bound post",
        reference=None,
    )


@pytest.mark.asyncio
async def test_public_thread_visible_to_everyone_allows_send_and_receipt():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    default_role = object()
    bot_user = SimpleNamespace(id=42)
    sent_message = SimpleNamespace(id=1234, author=bot_user, content="public thread")
    thread = SimpleNamespace(
        id=555,
        type=SimpleNamespace(value=11, name="public_thread"),
        guild=SimpleNamespace(id=1, default_role=default_role),
        permissions_for=MagicMock(
            return_value=SimpleNamespace(view_channel=True)
        ),
        send=AsyncMock(return_value=sent_message),
        fetch_message=AsyncMock(return_value=sent_message),
    )
    adapter._client = SimpleNamespace(
        user=bot_user,
        get_channel=lambda _channel_id: thread,
        fetch_channel=AsyncMock(),
    )

    sent = await adapter.send("555", "public thread")
    receipt = await adapter.verify_public_message_receipt(
        channel_id="555",
        message_id="1234",
        expected_content_sha256=hashlib.sha256(b"public thread").hexdigest(),
    )

    assert sent.success is True
    assert receipt["verified"] is True
    assert receipt["channel_id"] == "555"
    # Initial send + pre-POST + receipt pre-read + receipt post-read proofs.
    assert thread.permissions_for.call_count == 4
    assert all(
        call.args == (default_role,)
        for call in thread.permissions_for.call_args_list
    )


@pytest.mark.asyncio
async def test_guild_channel_hidden_from_everyone_fails_send_and_receipt_closed():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    default_role = object()
    channel = SimpleNamespace(
        id=555,
        type=0,
        guild=SimpleNamespace(id=1, default_role=default_role),
        permissions_for=MagicMock(
            return_value=SimpleNamespace(view_channel=False)
        ),
        send=AsyncMock(),
        fetch_message=AsyncMock(),
    )
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=42),
        get_channel=lambda _channel_id: channel,
        fetch_channel=AsyncMock(),
    )

    sent = await adapter.send("555", "must stay blocked")

    assert sent.success is False
    assert "@everyone/default role" in (sent.error or "")
    channel.send.assert_not_awaited()
    with pytest.raises(RuntimeError, match="@everyone/default role"):
        await adapter.verify_public_message_receipt(
            channel_id="555",
            message_id="1234",
        )
    channel.fetch_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_receipt_fails_if_public_visibility_is_revoked_during_readback():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    state = {"public": True}
    default_role = object()
    bot_user = SimpleNamespace(id=42)
    message = SimpleNamespace(id=1234, author=bot_user, content="receipt")

    async def _fetch(_message_id):
        state["public"] = False
        return message

    channel = SimpleNamespace(
        id=555,
        guild=SimpleNamespace(id=1, default_role=default_role),
        permissions_for=lambda role: SimpleNamespace(
            view_channel=state["public"] and role is default_role
        ),
        fetch_message=AsyncMock(side_effect=_fetch),
    )
    adapter._client = SimpleNamespace(
        user=bot_user,
        get_channel=lambda _channel_id: channel,
        fetch_channel=AsyncMock(),
    )

    with pytest.raises(RuntimeError, match="lost public"):
        await adapter.verify_public_message_receipt(
            channel_id="555",
            message_id="1234",
        )


@pytest.mark.asyncio
async def test_missing_default_role_permission_data_fails_closed():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    channel = SimpleNamespace(
        id=555,
        type=0,
        guild=SimpleNamespace(id=1),
        permissions_for=MagicMock(
            return_value=SimpleNamespace(view_channel=True)
        ),
        send=AsyncMock(),
    )
    adapter._client = SimpleNamespace(
        user=SimpleNamespace(id=42),
        get_channel=lambda _channel_id: channel,
        fetch_channel=AsyncMock(),
    )

    sent = await adapter.send("555", "missing public proof")

    assert sent.success is False
    assert "@everyone/default role" in (sent.error or "")
    channel.permissions_for.assert_not_called()
    channel.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# Forum channel tests
# ---------------------------------------------------------------------------

import discord as _discord_mod  # noqa: E402 — imported after _ensure_discord_mock


class TestIsForumParent:
    def test_none_returns_false(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        assert adapter._is_forum_parent(None) is False

    def test_forum_channel_class_instance(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        forum_cls = getattr(_discord_mod, "ForumChannel", None)
        if forum_cls is None:
            # Re-create a type for the mock
            forum_cls = type("ForumChannel", (), {})
            _discord_mod.ForumChannel = forum_cls
        ch = forum_cls()
        assert adapter._is_forum_parent(ch) is True

    def test_type_value_15(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        ch = SimpleNamespace(type=15)
        assert adapter._is_forum_parent(ch) is True

    def test_regular_channel_returns_false(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        ch = SimpleNamespace(type=0)
        assert adapter._is_forum_parent(ch) is False

    def test_thread_returns_false(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        ch = SimpleNamespace(type=11)  # public thread
        assert adapter._is_forum_parent(ch) is False


@pytest.mark.asyncio
async def test_single_receipt_send_rejects_forum_parent_before_thread_creation():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 999
    _mark_public(forum_channel)
    forum_channel.create_thread = AsyncMock()
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: forum_channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send(
        "999",
        "Receipt-bound route-back",
        metadata={"require_single_public_receipt": True},
    )

    assert result.success is False
    assert "forum parents are not supported" in (result.error or "")
    forum_channel.create_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_to_forum_creates_thread_post():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    # thread object has no 'send' so _send_to_forum uses thread.thread
    thread_ch = SimpleNamespace(id=555, send=AsyncMock(return_value=SimpleNamespace(id=600)))
    thread = SimpleNamespace(
        id=555,
        message=SimpleNamespace(id=500),
        thread=thread_ch,
    )
    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 999
    forum_channel.name = "ideas"
    _mark_public(forum_channel)
    forum_channel.create_thread = AsyncMock(return_value=thread)
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: forum_channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("999", "Hello forum!")

    assert result.success is True
    assert result.message_id == "500"
    forum_channel.create_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_to_forum_sends_remaining_chunks():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    # Force a small max message length so the message splits
    adapter.MAX_MESSAGE_LENGTH = 20

    chunk_msg_1 = SimpleNamespace(id=500)
    chunk_msg_2 = SimpleNamespace(id=501)
    thread_ch = SimpleNamespace(
        id=555,
        send=AsyncMock(return_value=chunk_msg_2),
    )
    # thread object has no 'send' so _send_to_forum uses thread.thread
    thread = SimpleNamespace(
        id=555,
        message=chunk_msg_1,
        thread=thread_ch,
    )
    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 999
    forum_channel.name = "ideas"
    _mark_public(forum_channel)
    forum_channel.create_thread = AsyncMock(return_value=thread)
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: forum_channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("999", "A" * 50)

    assert result.success is True
    assert result.message_id == "500"
    # Should have sent at least one follow-up chunk
    assert thread_ch.send.await_count >= 1


@pytest.mark.asyncio
async def test_send_to_forum_create_thread_failure():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 999
    forum_channel.name = "ideas"
    _mark_public(forum_channel)
    forum_channel.create_thread = AsyncMock(side_effect=Exception("rate limited"))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: forum_channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("999", "Hello forum!")

    assert result.success is False
    assert "rate limited" in result.error



# ---------------------------------------------------------------------------
# Forum follow-up chunk failure reporting + media on forum paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_to_forum_follow_up_chunk_failures_collected_as_warnings():
    """Partial-send chunk failures surface in raw_response['warnings']."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter.MAX_MESSAGE_LENGTH = 20

    chunk_msg_1 = SimpleNamespace(id=500)
    # Every follow-up chunk fails — we should collect a warning per failure
    thread_ch = SimpleNamespace(
        id=555,
        send=AsyncMock(side_effect=Exception("rate limited")),
    )
    thread = SimpleNamespace(id=555, message=chunk_msg_1, thread=thread_ch)
    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 999
    forum_channel.name = "ideas"
    _mark_public(forum_channel)
    forum_channel.create_thread = AsyncMock(return_value=thread)
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: forum_channel,
        fetch_channel=AsyncMock(),
    )

    # Long enough to produce multiple chunks
    result = await adapter.send("999", "A" * 60)

    # Starter message (first chunk) was delivered via create_thread, so send is
    # successful overall — but follow-up chunks all failed and are reported.
    assert result.success is True
    assert result.message_id == "500"
    warnings = (result.raw_response or {}).get("warnings") or []
    assert len(warnings) >= 1
    assert all("rate limited" in w for w in warnings)


@pytest.mark.asyncio
async def test_forum_post_file_creates_thread_with_attachment():
    """_forum_post_file routes file-bearing sends to create_thread with file kwarg."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    thread_ch = SimpleNamespace(id=777, send=AsyncMock())
    thread = SimpleNamespace(id=777, message=SimpleNamespace(id=800), thread=thread_ch)
    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 999
    forum_channel.name = "ideas"
    _mark_public(forum_channel)
    forum_channel.create_thread = AsyncMock(return_value=thread)

    # discord.File is a real class; build a MagicMock that looks like one
    fake_file = SimpleNamespace(filename="photo.png")

    result = await adapter._forum_post_file(
        forum_channel,
        content="here is a photo",
        file=fake_file,
    )

    assert result.success is True
    assert result.message_id == "800"
    forum_channel.create_thread.assert_awaited_once()
    call_kwargs = forum_channel.create_thread.await_args.kwargs
    assert call_kwargs["file"] is fake_file
    assert call_kwargs["content"] == "here is a photo"
    # Thread name derived from content's first line
    assert call_kwargs["name"] == "here is a photo"


@pytest.mark.asyncio
async def test_forum_post_file_uses_filename_when_no_content():
    """Thread name falls back to file.filename when no content is provided."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    thread = SimpleNamespace(id=1, message=SimpleNamespace(id=2), thread=SimpleNamespace(id=1, send=AsyncMock()))
    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 10
    forum_channel.name = "forum"
    _mark_public(forum_channel)
    forum_channel.create_thread = AsyncMock(return_value=thread)

    fake_file = SimpleNamespace(filename="voice-message.ogg")
    result = await adapter._forum_post_file(forum_channel, content="", file=fake_file)

    assert result.success is True
    call_kwargs = forum_channel.create_thread.await_args.kwargs
    # Content was empty → thread name derived from filename
    assert call_kwargs["name"] == "voice-message.ogg"


@pytest.mark.asyncio
async def test_forum_post_file_creation_failure():
    """_forum_post_file returns a failed SendResult when create_thread raises."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 999
    _mark_public(forum_channel)
    forum_channel.create_thread = AsyncMock(side_effect=Exception("missing perms"))

    result = await adapter._forum_post_file(
        forum_channel,
        content="hi",
        file=SimpleNamespace(filename="x.png"),
    )

    assert result.success is False
    assert "missing perms" in (result.error or "")


# ---------------------------------------------------------------------------
# Typing indicator task lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_typing_loop_stops_when_public_visibility_is_revoked(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    state = {"public": True}
    default_role = object()
    channel = SimpleNamespace(
        guild=SimpleNamespace(id=1, default_role=default_role),
        permissions_for=lambda role: SimpleNamespace(
            view_channel=state["public"] and role is default_role
        ),
    )
    adapter._client = MagicMock()
    adapter._client.get_channel.return_value = channel

    async def _request(_route):
        state["public"] = False

    adapter._client.http.request = AsyncMock(side_effect=_request)
    adapter._typing_tasks = {}
    original_sleep = asyncio.sleep

    async def _no_wait(_delay):
        return None

    monkeypatch.setattr(
        discord_adapter_module,
        "_discord_public_only_policy_required",
        lambda: True,
    )
    with patch(
        "plugins.platforms.discord.adapter.asyncio.sleep",
        new=_no_wait,
    ):
        await adapter.send_typing("12345")
        await original_sleep(0)
        await original_sleep(0)

    assert adapter._client.http.request.await_count == 1
    assert "12345" not in adapter._typing_tasks


@pytest.mark.asyncio
async def test_typing_task_removed_after_api_error():
    """When typing API call fails, stale task must be removed so typing can restart."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = MagicMock()
    adapter._client.http = MagicMock()
    adapter._client.http.request = AsyncMock(side_effect=Exception("rate limited"))
    adapter._typing_tasks = {}

    await adapter.send_typing("12345")
    await asyncio.sleep(0.1)

    assert "12345" not in adapter._typing_tasks, \
        "Stale task should be removed after API error"


@pytest.mark.asyncio
async def test_typing_restartable_after_error():
    """After a typing error, send_typing should start a new task (not blocked by stale entry)."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = MagicMock()
    adapter._client.http = MagicMock()
    adapter._typing_tasks = {}

    # First call fails
    adapter._client.http.request = AsyncMock(side_effect=Exception("503"))
    await adapter.send_typing("12345")
    await asyncio.sleep(0.1)

    # Second call should work
    adapter._client.http.request = AsyncMock()
    await adapter.send_typing("12345")

    assert "12345" in adapter._typing_tasks, \
        "Should restart typing after previous failure"


@pytest.mark.asyncio
async def test_typing_stop_cleans_up():
    """stop_typing should remove the task from _typing_tasks."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = MagicMock()
    adapter._client.http = MagicMock()
    adapter._client.http.request = AsyncMock()
    adapter._typing_tasks = {}

    await adapter.send_typing("12345")
    assert "12345" in adapter._typing_tasks

    await adapter.stop_typing("12345")
    assert "12345" not in adapter._typing_tasks
