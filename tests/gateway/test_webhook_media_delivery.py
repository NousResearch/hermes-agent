"""Webhook cross-platform delivery must handle ``MEDIA:`` tags like cron deliveries do.

``_deliver_cross_platform`` previously sent the raw rendered prompt to ``adapter.send()``,
so a route script (or rendered event payload) emitting ``MEDIA:/path`` reached Telegram /
Discord as literal prose — cron deliveries got media extraction + path-policy filtering
(``cron/scheduler_delivery.py``), webhook deliveries never did. These tests pin the
sibling-path behavior: unwrap the tag, enforce the shared media-path policy (denylist /
allow-dirs / strict), and send each attachment through the adapter's native media lane.

Security: webhook routes are attacker-influenceable by design (github/svix/linear/... events
render into prompts), so a ``MEDIA:`` tag must pass the same
``validate_media_delivery_path`` gate as cron before anything egresses — without it an
injected tag is an arbitrary-file-exfiltration vector into a chat.
"""

import os

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter


class _FakeAdapter:
    """Fake delivery adapter with the media lanes the real adapters implement."""

    def __init__(self):
        self.sent_text = []
        self.sent_media = []

    async def send(self, chat_id, content, metadata=None, **kwargs) -> SendResult:
        self.sent_text.append((chat_id, content))
        return SendResult(success=True)

    async def send_voice(
        self, chat_id, audio_path, metadata=None, **kwargs
    ) -> SendResult:
        self.sent_media.append(("send_voice", chat_id, audio_path))
        return SendResult(success=True)

    async def send_video(
        self, chat_id, video_path, metadata=None, **kwargs
    ) -> SendResult:
        self.sent_media.append(("send_video", chat_id, video_path))
        return SendResult(success=True)

    async def send_image_file(
        self, chat_id, image_path, metadata=None, **kwargs
    ) -> SendResult:
        self.sent_media.append(("send_image_file", chat_id, image_path))
        return SendResult(success=True)

    async def send_document(
        self, chat_id, file_path, metadata=None, **kwargs
    ) -> SendResult:
        self.sent_media.append(("send_document", chat_id, file_path))
        return SendResult(success=True)


def _make_delivery_adapter(fake) -> WebhookAdapter:
    from types import SimpleNamespace

    runner = SimpleNamespace(
        _adapters_for_profile=lambda profile=None: {Platform.DISCORD: fake},
        _authorization_adapter=lambda platform, profile=None: fake,
        config=None,
    )
    adapter = WebhookAdapter.__new__(WebhookAdapter)
    adapter.gateway_runner = runner
    return adapter


@pytest.mark.asyncio
async def test_media_tag_unwraps_and_sends_audio(tmp_path):
    """A delivered prompt containing ``MEDIA:<audio>`` posts cleaned text and plays audio."""
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake-mp3")
    fake = _FakeAdapter()
    adapter = _make_delivery_adapter(fake)

    result = await adapter._deliver_cross_platform(
        "discord",
        f"🎙️ New from @TheQuartering\nMEDIA:{audio}",
        {"deliver_extra": {"chat_id": "12345"}},
    )

    assert result.success is True
    # The literal tag must not reach the user; the text rides along clean.
    assert fake.sent_text == [("12345", "🎙️ New from @TheQuartering")]
    # Audio goes down the adapter's media lane (send_voice — cron's sibling lane), not raw prose.
    assert fake.sent_media == [("send_voice", "12345", str(audio))]


@pytest.mark.asyncio
async def test_denied_media_path_is_dropped_not_delivered(tmp_path, monkeypatch):
    """Under strict + allowlist media policy, a MEDIA path outside the allow root is dropped.

    The tag is stripped from the delivered text too — an allowlisted-deployment miss must not
    smuggle the literal path into the chat any more than it egresses the file.
    """
    safe_dir = tmp_path / "allowed"
    safe_dir.mkdir()
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    safe = safe_dir / "ok.mp3"
    safe.write_bytes(b"fake-mp3")
    denied = other_dir / "denied.mp3"
    denied.write_bytes(b"fake-mp3")

    # Gate the deployment: strict mode, no recency trust, only safe_dir allowed.
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(safe_dir))

    fake = _FakeAdapter()
    adapter = _make_delivery_adapter(fake)

    result = await adapter._deliver_cross_platform(
        "discord",
        f"MEDIA:{denied}\nMEDIA:{safe}",
        {"deliver_extra": {"chat_id": "12345"}},
    )

    assert result.success is True
    # denied.mp3 (valid extension, inside tmp but outside the allow root) never egresses;
    # safe.mp3 under the allow root does.
    assert fake.sent_media == [("send_voice", "12345", str(safe))]
    # Both tags stripped from text; nothing but the safe media file is delivered
    # (no empty text message).
    assert fake.sent_text == []


@pytest.mark.asyncio
async def test_no_media_leaves_legacy_send_untouched(tmp_path):
    """Routes without MEDIA tags keep the exact pre-existing single-send path (no regression)."""
    fake = _FakeAdapter()
    adapter = _make_delivery_adapter(fake)

    result = await adapter._deliver_cross_platform(
        "discord",
        "Alert: server is on fire!",
        {"deliver_extra": {"chat_id": "12345"}},
    )

    assert result.success is True
    assert fake.sent_text == [("12345", "Alert: server is on fire!")]
    assert fake.sent_media == []
