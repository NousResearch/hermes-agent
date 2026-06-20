"""Tests for EphemeralReply — system-notice auto-delete in gateway adapters.

Slash-command handlers in ``gateway/run.py`` can return an
``EphemeralReply`` wrapper to request auto-deletion of the reply message
after a TTL.  The base adapter unwraps the sentinel before sending and
schedules a detached delete task when the platform supports
``delete_message``.

Covered:

1. ``_unwrap_ephemeral`` returns text + ttl for EphemeralReply, and
   passes plain strings through unchanged.
2. TTL is zeroed on platforms that don't override ``delete_message``
   (silent degrade — message stays in place).
3. TTL is honored on platforms that DO override ``delete_message``.
4. ``_schedule_ephemeral_delete`` invokes ``delete_message`` after the
   configured delay with the correct chat_id / message_id.
5. ``_process_message_background`` sends the unwrapped text (not the
   sentinel object) and schedules deletion when appropriate.
6. The two busy-session bypass paths also unwrap + schedule.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    EphemeralReply,
    SecureReply,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource


class _NoDeleteAdapter(BasePlatformAdapter):
    """Adapter that does NOT override delete_message (silent degrade)."""

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, content="", **kwargs):
        return SendResult(success=True, message_id="m-1")

    async def get_chat_info(self, chat_id):
        return {}


class _DeleteCapableAdapter(BasePlatformAdapter):
    """Adapter that overrides delete_message (TTL honored)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.deleted: list[tuple[str, str]] = []

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, content="", **kwargs):
        return SendResult(success=True, message_id="m-2")

    async def get_chat_info(self, chat_id):
        return {}

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        self.deleted.append((chat_id, message_id))
        return True


def _no_delete_adapter():
    return _NoDeleteAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM
    )


def _delete_adapter():
    return _DeleteCapableAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM
    )


def _make_event(text="/stop", chat_id="42"):
    return MessageEvent(
        text=text,
        message_id="msg-1",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            user_id="u-1",
        ),
        message_type=MessageType.TEXT,
    )


# ---------------------------------------------------------------------------
# _unwrap_ephemeral
# ---------------------------------------------------------------------------


def test_unwrap_plain_string_is_passthrough():
    adapter = _delete_adapter()
    text, ttl = adapter._unwrap_ephemeral("hello")
    assert text == "hello"
    assert ttl == 0


def test_unwrap_none_is_passthrough():
    adapter = _delete_adapter()
    text, ttl = adapter._unwrap_ephemeral(None)
    assert text is None
    assert ttl == 0


def test_unwrap_ephemeral_explicit_ttl_on_capable_adapter():
    adapter = _delete_adapter()
    text, ttl = adapter._unwrap_ephemeral(EphemeralReply("bye", ttl_seconds=60))
    assert text == "bye"
    assert ttl == 60


def test_unwrap_ephemeral_zeros_ttl_on_incapable_adapter():
    """Platforms without delete_message should silently degrade to normal send."""
    adapter = _no_delete_adapter()
    text, ttl = adapter._unwrap_ephemeral(EphemeralReply("bye", ttl_seconds=60))
    assert text == "bye"
    assert ttl == 0  # forced to 0 — message will stay in place


def test_unwrap_ephemeral_default_ttl_from_config():
    adapter = _delete_adapter()
    with patch.object(adapter, "_get_ephemeral_system_ttl_default", return_value=120):
        text, ttl = adapter._unwrap_ephemeral(EphemeralReply("bye"))
    assert text == "bye"
    assert ttl == 120


def test_unwrap_ephemeral_default_ttl_zero_disables():
    """Config default of 0 (the shipped default) means the feature is off."""
    adapter = _delete_adapter()
    with patch.object(adapter, "_get_ephemeral_system_ttl_default", return_value=0):
        text, ttl = adapter._unwrap_ephemeral(EphemeralReply("bye"))
    assert text == "bye"
    assert ttl == 0


def test_unwrap_ephemeral_handles_unreadable_config():
    adapter = _delete_adapter()
    with patch.object(
        adapter,
        "_get_ephemeral_system_ttl_default",
        side_effect=RuntimeError("boom"),
    ):
        text, ttl = adapter._unwrap_ephemeral(EphemeralReply("bye"))
    # Fall back to 0 rather than crashing the handler pipeline.
    assert text == "bye"
    assert ttl == 0


# ---------------------------------------------------------------------------
# _schedule_ephemeral_delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_ephemeral_delete_calls_delete_after_ttl():
    adapter = _delete_adapter()
    # Use a very short TTL to keep the test fast — the implementation
    # floors sleeps at 1s via ``max(1, int(ttl_seconds))``.  Patch asyncio.sleep
    # inside the module under test; the test body uses the real one for
    # scheduler pumping.
    import gateway.platforms.base as base_module

    sleeps: list[float] = []
    _real_sleep = base_module.asyncio.sleep

    async def _fake_sleep(duration):
        sleeps.append(duration)
        # Yield control so the rest of the task body can run.
        await _real_sleep(0)

    with patch.object(base_module.asyncio, "sleep", _fake_sleep):
        adapter._schedule_ephemeral_delete(
            chat_id="42", message_id="m-2", ttl_seconds=5
        )
        # Let the spawned task run.
        for _ in range(5):
            await _real_sleep(0)

    # Only the ttl sleep shows up — the test pump uses the real sleep.
    assert 5 in sleeps
    assert adapter.deleted == [("42", "m-2")]


@pytest.mark.asyncio
async def test_schedule_ephemeral_delete_swallows_errors():
    adapter = _delete_adapter()

    async def _boom(*a, **kw):
        raise RuntimeError("permission denied")

    adapter.delete_message = _boom  # type: ignore[assignment]
    with patch("gateway.platforms.base.asyncio.sleep", AsyncMock()):
        adapter._schedule_ephemeral_delete(
            chat_id="42", message_id="m-2", ttl_seconds=1
        )
        # No exception should propagate even though delete_message raised.
        for _ in range(5):
            await asyncio.sleep(0)


def test_schedule_ephemeral_delete_outside_event_loop_is_noop():
    """No running loop → no crash, silently drops the request."""
    adapter = _delete_adapter()
    # No pytest.mark.asyncio → no loop.  Must not raise.
    adapter._schedule_ephemeral_delete(
        chat_id="42", message_id="m-2", ttl_seconds=1
    )
    assert adapter.deleted == []


# ---------------------------------------------------------------------------
# _process_message_background unwraps EphemeralReply before send
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_message_unwraps_ephemeral_before_send():
    """The adapter must send the wrapper's .text, never the wrapper object."""
    adapter = _delete_adapter()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="sent-1")
    )

    async def _handler(evt):
        return EphemeralReply("⚡ Stopped.", ttl_seconds=5)

    adapter.set_message_handler(_handler)

    sleeps: list[float] = []

    async def _fake_sleep(duration):
        sleeps.append(duration)

    event = _make_event()
    session_key = "agent:main:telegram:private:42"
    with patch("gateway.platforms.base.asyncio.sleep", _fake_sleep), patch.object(
        adapter, "_keep_typing", new=AsyncMock()
    ):
        await adapter._process_message_background(event, session_key)
        # Pump until the detached delete task completes.
        for _ in range(10):
            await asyncio.sleep(0)

    # Sent text is the unwrapped string, NOT repr(EphemeralReply(...))
    adapter._send_with_retry.assert_called_once()
    sent_text = adapter._send_with_retry.call_args.kwargs["content"]
    assert sent_text == "⚡ Stopped."
    # Auto-delete scheduled using the returned message_id
    assert ("42", "sent-1") in adapter.deleted


@pytest.mark.asyncio
async def test_process_message_ephemeral_reply_does_not_auto_upload_bare_paths(tmp_path):
    """Tips/system notices may mention local paths; they must remain text."""
    adapter = _delete_adapter()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="sent-1")
    )
    adapter.send_document = AsyncMock(
        return_value=SendResult(success=True, message_id="doc-1")
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  provider: test\n", encoding="utf-8")
    reply_text = f"Tip: hermes chat --ignore-user-config skips {config_path}"

    async def _handler(evt):
        return EphemeralReply(reply_text, ttl_seconds=0)

    adapter.set_message_handler(_handler)

    event = _make_event(text="/new")
    session_key = "agent:main:telegram:private:42"
    with patch("gateway.platforms.base.asyncio.sleep", AsyncMock()), patch.object(
        adapter, "_keep_typing", new=AsyncMock()
    ):
        await adapter._process_message_background(event, session_key)

    adapter._send_with_retry.assert_called_once()
    assert adapter._send_with_retry.call_args.kwargs["content"] == reply_text
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_message_incapable_platform_does_not_schedule_delete():
    adapter = _no_delete_adapter()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="sent-1")
    )

    async def _handler(evt):
        return EphemeralReply("⚡ Stopped.", ttl_seconds=5)

    adapter.set_message_handler(_handler)

    # Spy on delete_message to confirm it is NOT invoked.
    delete_calls: list = []

    async def _spy_delete(chat_id, message_id):
        delete_calls.append((chat_id, message_id))
        return False

    adapter.delete_message = _spy_delete  # type: ignore[assignment]

    event = _make_event()
    session_key = "agent:main:telegram:private:42"
    with patch("gateway.platforms.base.asyncio.sleep", AsyncMock()), patch.object(
        adapter, "_keep_typing", new=AsyncMock()
    ):
        await adapter._process_message_background(event, session_key)
        for _ in range(10):
            await asyncio.sleep(0)

    # Send happened with the unwrapped text...
    adapter._send_with_retry.assert_called_once()
    assert adapter._send_with_retry.call_args.kwargs["content"] == "⚡ Stopped."
    # ...but delete was never scheduled because the capability check skipped
    # the schedule call (TTL was zeroed in _unwrap_ephemeral).
    # Note: the capability gate on _unwrap_ephemeral checks for
    # ``type(adapter).delete_message is BasePlatformAdapter.delete_message``.
    # Monkeypatching the instance does NOT change the class, so this test
    # verifies the gate uses the class method to detect capability.
    assert delete_calls == []


@pytest.mark.asyncio
async def test_process_message_plain_string_behaves_unchanged():
    adapter = _delete_adapter()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="sent-1")
    )

    async def _handler(evt):
        return "plain reply"

    adapter.set_message_handler(_handler)

    event = _make_event()
    session_key = "agent:main:telegram:private:42"
    with patch("gateway.platforms.base.asyncio.sleep", AsyncMock()), patch.object(
        adapter, "_keep_typing", new=AsyncMock()
    ):
        await adapter._process_message_background(event, session_key)
        for _ in range(5):
            await asyncio.sleep(0)

    adapter._send_with_retry.assert_called_once()
    assert adapter._send_with_retry.call_args.kwargs["content"] == "plain reply"
    assert adapter.deleted == []  # no auto-delete for plain replies


class _EditCapableAdapter(_DeleteCapableAdapter):
    """Adapter that can edit first, delete as fallback."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.edited: list[tuple[str, str, str]] = []

    async def edit_message(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False) -> SendResult:
        self.edited.append((chat_id, message_id, content))
        return SendResult(success=True, message_id=message_id)


def _edit_adapter():
    return _EditCapableAdapter(
        PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM
    )


def test_unwrap_secure_reply_uses_explicit_metadata():
    adapter = _delete_adapter()
    text, meta = adapter._unwrap_secure(
        SecureReply("pw=abc", ttl_seconds=45, redacted_text="pw=[expired]")
    )
    assert text == "pw=abc"
    assert meta == {
        "ttl_seconds": 45,
        "redacted_text": "pw=[expired]",
        "protect_content": True,
        "spoiler": True,
    }


def test_unwrap_secure_marker_uses_config_defaults():
    adapter = _delete_adapter()
    with patch.object(
        adapter,
        "_get_secure_message_config",
        return_value={
            "ttl_seconds": 12,
            "redacted_text": "[gone]",
            "protect_content": False,
            "spoiler": False,
            "marker_start": "[[secure]]",
            "marker_end": "[[/secure]]",
        },
    ):
        text, meta = adapter._unwrap_secure("[[secure]]token=abc[[/secure]]")
    assert text == "token=abc"
    assert meta["ttl_seconds"] == 12
    assert meta["redacted_text"] == "[gone]"
    assert meta["protect_content"] is False
    assert meta["spoiler"] is False


def test_render_secure_content_spoiler_for_telegram_and_neutralizes_nested_delimiters():
    adapter = _delete_adapter()
    rendered = adapter._render_secure_content("alpha || beta", spoiler=True)
    assert rendered.startswith("||") and rendered.endswith("||")
    assert "|\u200b|" in rendered


@pytest.mark.asyncio
async def test_schedule_secure_redaction_prefers_edit_over_delete():
    adapter = _edit_adapter()
    import gateway.platforms.base as base_module

    _real_sleep = base_module.asyncio.sleep

    async def _fake_sleep(duration):
        await _real_sleep(0)

    with patch.object(base_module.asyncio, "sleep", _fake_sleep):
        adapter._schedule_secure_redaction(
            chat_id="42",
            message_id="m-2",
            ttl_seconds=5,
            redacted_text="[expired]",
        )
        for _ in range(5):
            await _real_sleep(0)

    assert adapter.edited == [("42", "m-2", "[expired]")]
    assert adapter.deleted == []


@pytest.mark.asyncio
async def test_process_message_secure_reply_wraps_and_schedules_redaction():
    adapter = _edit_adapter()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="sent-secure")
    )

    async def _handler(evt):
        return SecureReply("Password: abc", ttl_seconds=5, redacted_text="Password: [expired]")

    adapter.set_message_handler(_handler)
    event = _make_event()
    session_key = "agent:main:telegram:private:42"

    import gateway.platforms.base as base_module
    _real_sleep = base_module.asyncio.sleep

    async def _fake_sleep(duration):
        await _real_sleep(0)

    with patch.object(base_module.asyncio, "sleep", _fake_sleep), patch.object(
        adapter, "_keep_typing", new=AsyncMock()
    ):
        await adapter._process_message_background(event, session_key)
        for _ in range(10):
            await asyncio.sleep(0)

    adapter._send_with_retry.assert_called_once()
    kwargs = adapter._send_with_retry.call_args.kwargs
    assert kwargs["content"] == "||Password: abc||"
    assert kwargs["metadata"]["secure_message"]["protect_content"] is True
    assert ("42", "sent-secure", "Password: [expired]") in adapter.edited


@pytest.mark.asyncio
async def test_secure_reply_media_directive_is_not_uploaded_as_permanent_attachment():
    adapter = _edit_adapter()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="sent-secure")
    )
    adapter.send_multiple_images = AsyncMock()
    adapter.send_document = AsyncMock()

    async def _handler(evt):
        return SecureReply("MEDIA:/tmp/secret.png", ttl_seconds=5)

    adapter.set_message_handler(_handler)
    event = _make_event()
    session_key = "agent:main:telegram:private:42"

    import gateway.platforms.base as base_module
    _real_sleep = base_module.asyncio.sleep

    async def _fake_sleep(duration):
        await _real_sleep(0)

    with patch.object(base_module.asyncio, "sleep", _fake_sleep), patch.object(
        adapter, "_keep_typing", new=AsyncMock()
    ):
        await adapter._process_message_background(event, session_key)
        for _ in range(10):
            await asyncio.sleep(0)

    adapter._send_with_retry.assert_called_once()
    sent = adapter._send_with_retry.call_args.kwargs["content"]
    assert "MEDIA:" not in sent
    assert "secure attachment omitted" in sent
    adapter.send_multiple_images.assert_not_called()
    adapter.send_document.assert_not_called()
    assert any(edit[2] == "[redacted — secure message expired]" for edit in adapter.edited)


@pytest.mark.asyncio
async def test_secure_reply_bare_local_file_path_is_not_uploaded_as_permanent_attachment():
    adapter = _edit_adapter()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="sent-secure")
    )
    adapter.send_multiple_images = AsyncMock()
    adapter.send_document = AsyncMock()

    async def _handler(evt):
        return SecureReply("/tmp/secret.png", ttl_seconds=5)

    adapter.set_message_handler(_handler)
    event = _make_event()
    session_key = "agent:main:telegram:private:42"

    import gateway.platforms.base as base_module
    _real_sleep = base_module.asyncio.sleep

    async def _fake_sleep(duration):
        await _real_sleep(0)

    with patch.object(base_module.asyncio, "sleep", _fake_sleep), patch.object(
        adapter, "_keep_typing", new=AsyncMock()
    ):
        await adapter._process_message_background(event, session_key)
        for _ in range(10):
            await asyncio.sleep(0)

    adapter._send_with_retry.assert_called_once()
    sent = adapter._send_with_retry.call_args.kwargs["content"]
    assert "/tmp/secret.png" in sent
    adapter.send_multiple_images.assert_not_called()
    adapter.send_document.assert_not_called()
    assert any(edit[2] == "[redacted — secure message expired]" for edit in adapter.edited)
