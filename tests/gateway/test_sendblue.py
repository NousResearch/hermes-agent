"""Tests for the Sendblue iMessage gateway adapter."""
import asyncio
import hmac
import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter(monkeypatch, **extra):
    monkeypatch.setenv("SENDBLUE_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("SENDBLUE_API_SECRET", "test-secret")
    monkeypatch.setenv("SENDBLUE_NUMBER", "+15555550100")
    monkeypatch.setenv("SENDBLUE_WEBHOOK_SECRET", "test-webhook-secret")
    from gateway.platforms.sendblue import SendblueAdapter

    cfg = PlatformConfig(
        enabled=True,
        extra={
            "api_key_id": "test-key-id",
            "api_secret": "test-secret",
            "sendblue_number": "+15555550100",
            "webhook_secret": "test-webhook-secret",
            **extra,
        },
    )
    return SendblueAdapter(cfg)


class _MockRequest:
    """Minimal aiohttp.Request surface for _handle_webhook tests.

    Exposes .headers, .json() async, and .remote. Enough for the handler
    to do signature check, body parse, and logging — no more. Pass
    json_error to make .json() raise that exception (used to test the
    JSONDecodeError → 400 path).
    """
    def __init__(self, body=None, headers=None, remote="127.0.0.1", json_error=None, content_length=0):
        self._body = body
        self.headers = headers or {}
        self.remote = remote
        self._json_error = json_error
        self.content_length = content_length

    async def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._body


async def _drain_background_tasks(adapter):
    """Wait for fire-and-forget asyncio.create_task() to settle.

    _handle_webhook does asyncio.create_task(self.handle_message(event))
    and returns immediately. To assert on handle_message side effects,
    we need to yield the event loop once for the task to actually run.
    """
    if adapter._background_tasks:
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)
    else:
        await asyncio.sleep(0)


class _MockHttpxResponse:
    """Minimal httpx.Response surface for media download tests.

    Provides .content (bytes payload) and .raise_for_status() (no-op for
    2xx, raises HTTPStatusError for 4xx/5xx). Enough for the helper to
    do its happy path and to test error branches.
    """
    def __init__(self, content=b"", status_code=200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=httpx.Request("GET", "http://test"),
                response=self,
            )


class TestSendblueSignatureVerification:
    def test_correct_secret_passes(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_secret="abc123")
        assert adapter._verify_signature("abc123") is True

    def test_wrong_secret_fails(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_secret="abc123")
        assert adapter._verify_signature("wrong") is False

    def test_empty_header_with_configured_secret_fails(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_secret="abc123")
        assert adapter._verify_signature("") is False

    def test_no_secret_configured_fails_closed(self, monkeypatch):
        # Default behavior: missing secret rejects every webhook.
        adapter = _make_adapter(monkeypatch)
        adapter.webhook_secret = ""
        assert adapter.disable_signature_check is False
        assert adapter._verify_signature("whatever") is False

    def test_no_secret_with_explicit_disable_passes(self, monkeypatch):
        # Operator opt-in: disable_signature_check=True bypasses verification.
        adapter = _make_adapter(monkeypatch)
        adapter.webhook_secret = ""
        adapter.disable_signature_check = True
        assert adapter._verify_signature("whatever") is True

    def test_uses_constant_time_compare(self, monkeypatch):
        """Regression guard: _verify_signature must use hmac.compare_digest,
        not plain '==', to avoid timing-attack leakage on the webhook
        signing secret."""
        import gateway.platforms.sendblue as sb
        adapter = _make_adapter(monkeypatch, webhook_secret="abc123")
        called = {"hit": False}
        real = hmac.compare_digest

        def spy(a, b):
            called["hit"] = True
            return real(a, b)

        import hmac as _hmac_module
        monkeypatch.setattr(sb, "hmac", type("M", (), {"compare_digest": spy}))
        assert adapter._verify_signature("abc123") is True
        assert called["hit"] is True


class TestSendblueWebhookBodyCap:
    """Reject oversized webhook bodies with 413 before doing any work.

    Sendblue webhook payloads are small JSON (a few KB at most — media is
    referenced by URL, not inlined). A 1 MiB cap is generous and protects
    against DoS if the signing secret leaks (or signature verification is
    disabled).
    """
    @pytest.mark.asyncio
    async def test_oversized_body_returns_413_before_signature_check(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        # Content-Length over the cap, no signature header — should still
        # reject with 413 (size check fires before signature verification).
        request = _MockRequest(
            body={"is_outbound": False, "from_number": "+17706768883", "content": "x"},
            headers={},
            content_length=10 * 1024 * 1024,  # 10 MiB
        )
        response = await adapter._handle_webhook(request)
        assert response.status == 413
        assert adapter.handle_message.call_count == 0

    @pytest.mark.asyncio
    async def test_body_at_cap_is_accepted(self, monkeypatch):
        import gateway.platforms.sendblue as sb
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        # Exactly the cap is allowed; size > cap is rejected.
        request = _MockRequest(
            body={
                "is_outbound": False,
                "sendblue_number": "+15555550100",
                "from_number": "+17706768883",
                "content": "hi",
            },
            headers={"sb-signing-secret": "test-webhook-secret"},
            content_length=sb.MAX_WEBHOOK_BODY_BYTES,
        )
        response = await adapter._handle_webhook(request)
        await _drain_background_tasks(adapter)
        assert response.status == 200
        assert adapter.handle_message.call_count == 1


class TestSendblueWebhookDedup:
    """Sendblue retries webhooks on 5xx/timeout. The adapter must dedupe
    by message_handle so a single inbound never dispatches twice.
    """
    @pytest.mark.asyncio
    async def test_duplicate_message_handle_only_dispatches_once(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        payload = {
            "is_outbound": False,
            "sendblue_number": "+15555550100",
            "from_number": "+17706768883",
            "content": "hello",
            "message_handle": "handle-abc",
        }
        headers = {"sb-signing-secret": "test-webhook-secret"}
        first = await adapter._handle_webhook(_MockRequest(body=payload, headers=headers))
        second = await adapter._handle_webhook(_MockRequest(body=payload, headers=headers))
        await _drain_background_tasks(adapter)
        assert first.status == 200
        assert second.status == 200
        assert adapter.handle_message.call_count == 1

    @pytest.mark.asyncio
    async def test_distinct_handles_both_dispatch(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        base = {
            "is_outbound": False,
            "sendblue_number": "+15555550100",
            "from_number": "+17706768883",
            "content": "hello",
        }
        headers = {"sb-signing-secret": "test-webhook-secret"}
        await adapter._handle_webhook(_MockRequest(
            body={**base, "message_handle": "handle-1"}, headers=headers,
        ))
        await adapter._handle_webhook(_MockRequest(
            body={**base, "message_handle": "handle-2"}, headers=headers,
        ))
        await _drain_background_tasks(adapter)
        assert adapter.handle_message.call_count == 2

    @pytest.mark.asyncio
    async def test_missing_handle_is_not_deduped(self, monkeypatch):
        """Some Sendblue events omit message_handle; we must not collapse
        them all into the empty-string bucket."""
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        payload = {
            "is_outbound": False,
            "sendblue_number": "+15555550100",
            "from_number": "+17706768883",
            "content": "hello",
            # no message_handle
        }
        headers = {"sb-signing-secret": "test-webhook-secret"}
        await adapter._handle_webhook(_MockRequest(body=payload, headers=headers))
        await adapter._handle_webhook(_MockRequest(body=payload, headers=headers))
        await _drain_background_tasks(adapter)
        assert adapter.handle_message.call_count == 2

    def test_dedup_set_is_bounded(self, monkeypatch):
        import gateway.platforms.sendblue as sb
        adapter = _make_adapter(monkeypatch)
        # Fill past capacity; oldest entries must be evicted.
        for i in range(sb.DEDUP_CAPACITY + 50):
            assert adapter._is_duplicate(f"h-{i}") is False
        assert len(adapter._seen_handles) == sb.DEDUP_CAPACITY
        # Oldest 50 should have been evicted; their handles are no
        # longer in the set, so re-presenting them reads as fresh.
        assert adapter._is_duplicate("h-0") is False
        # Most-recent is still remembered.
        assert adapter._is_duplicate(f"h-{sb.DEDUP_CAPACITY + 49}") is True


class TestSendblueWebhookTypingIndicator:
    """Typing-indicator webhooks must not be dispatched as messages.

    Sendblue posts typing_indicator events with an `is_typing` field and
    no message content. Forwarding them as MessageEvents would feed the
    agent an empty turn.
    """
    @pytest.mark.asyncio
    async def test_typing_indicator_returns_200_without_dispatch(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        request = _MockRequest(
            body={
                "from_number": "+17706768883",
                "sendblue_number": "+15555550100",
                "is_typing": True,
            },
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        response = await adapter._handle_webhook(request)
        await _drain_background_tasks(adapter)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_typing_indicator_does_not_call_handle_message(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        request = _MockRequest(
            body={
                "from_number": "+17706768883",
                "sendblue_number": "+15555550100",
                "is_typing": False,  # stopped-typing event also short-circuits
            },
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        await adapter._handle_webhook(request)
        await _drain_background_tasks(adapter)
        assert adapter.handle_message.call_count == 0


class TestSendblueSendEmptyChunks:
    @pytest.mark.asyncio
    async def test_multibubble_blank_paragraphs_returns_failure(self, monkeypatch):
        """multi_bubble_split + content that's only blank lines yields
        zero chunks. Must return success=False, not silent success."""
        adapter = _make_adapter(monkeypatch, multi_bubble_split=True)
        adapter._sendblue_api_post = AsyncMock()
        # format_message(strip_markdown) collapses whitespace differently per
        # input; use a value that survives format_message but splits empty.
        # Stub format_message to return a known empty-paragraph string so the
        # test is independent of strip_markdown's behavior.
        adapter.format_message = lambda c: "\n\n   \n\n"
        result = await adapter.send("+17706768883", "anything")
        assert result.success is False
        assert "empty" in (result.error or "").lower()
        adapter._sendblue_api_post.assert_not_called()


class TestSendblueWebhookRouting:
    @pytest.mark.asyncio
    async def test_routes_message_for_configured_number(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)  # SENDBLUE_NUMBER = +15555550100
        adapter.handle_message = AsyncMock()
        payload = {
            "is_outbound": False,
            "sendblue_number": "+15555550100",
            "from_number": "+17706768883",
            "content": "hello",
        }
        request = _MockRequest(
            body=payload,
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        response = await adapter._handle_webhook(request)
        await _drain_background_tasks(adapter)
        assert response.status == 200
        assert adapter.handle_message.call_count == 1

    @pytest.mark.asyncio
    async def test_silently_drops_message_for_other_number(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)  # SENDBLUE_NUMBER = +15555550100
        adapter.handle_message = AsyncMock()
        payload = {
            "is_outbound": False,
            "sendblue_number": "+15555559999",  # NOT our number
            "from_number": "+17706768883",
            "content": "hello",
        }
        request = _MockRequest(
            body=payload,
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        response = await adapter._handle_webhook(request)
        await _drain_background_tasks(adapter)
        assert response.status == 200  # silent — 200 even though dropped
        assert adapter.handle_message.call_count == 0

    @pytest.mark.asyncio
    async def test_no_number_configured_processes_all(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.sendblue_number = ""  # disable number filter
        adapter.handle_message = AsyncMock()
        payload = {
            "is_outbound": False,
            "sendblue_number": "+15555559999",  # any number
            "from_number": "+17706768883",
            "content": "hello",
        }
        request = _MockRequest(
            body=payload,
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        response = await adapter._handle_webhook(request)
        await _drain_background_tasks(adapter)
        assert response.status == 200
        assert adapter.handle_message.call_count == 1


class TestSendblueWebhookParsing:
    @pytest.mark.asyncio
    async def test_single_object_normalized_to_list(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        payload = {
            "is_outbound": False,
            "sendblue_number": "+15555550100",
            "from_number": "+17706768883",
            "content": "hello",
        }
        request = _MockRequest(
            body=payload,  # dict, not list
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        response = await adapter._handle_webhook(request)
        await _drain_background_tasks(adapter)
        assert response.status == 200
        assert adapter.handle_message.call_count == 1

    @pytest.mark.asyncio
    async def test_array_processed_as_list(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        item = {
            "is_outbound": False,
            "sendblue_number": "+15555550100",
            "from_number": "+17706768883",
            "content": "hello",
        }
        request = _MockRequest(
            body=[item, item],  # array of two valid items
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        response = await adapter._handle_webhook(request)
        await _drain_background_tasks(adapter)
        assert response.status == 200
        assert adapter.handle_message.call_count == 2

    @pytest.mark.asyncio
    async def test_malformed_json_returns_400(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        request = _MockRequest(
            json_error=json.JSONDecodeError("expecting value", "", 0),
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        response = await adapter._handle_webhook(request)
        await _drain_background_tasks(adapter)
        assert response.status == 400
        assert adapter.handle_message.call_count == 0

    @pytest.mark.asyncio
    async def test_missing_from_number_skipped(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        payload = {
            "is_outbound": False,
            "sendblue_number": "+15555550100",
            # no from_number
            "content": "hello",
        }
        request = _MockRequest(
            body=payload,
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        response = await adapter._handle_webhook(request)
        await _drain_background_tasks(adapter)
        # Item skipped at required-fields check, but batch still succeeds
        assert response.status == 200
        assert adapter.handle_message.call_count == 0

    @pytest.mark.asyncio
    async def test_missing_content_skipped(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        payload = {
            "is_outbound": False,
            "sendblue_number": "+15555550100",
            "from_number": "+17706768883",
            # no content, no media_url — no text source at all
        }
        request = _MockRequest(
            body=payload,
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        response = await adapter._handle_webhook(request)
        await _drain_background_tasks(adapter)
        assert response.status == 200
        assert adapter.handle_message.call_count == 0


class TestSendblueMediaDownload:
    @pytest.mark.asyncio
    async def test_image_url_cached_with_correct_ext(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.client = AsyncMock()
        adapter.client.get = AsyncMock(
            return_value=_MockHttpxResponse(content=b"fake-image-bytes")
        )
        cache_mock = Mock(return_value="/cache/foo.jpg")
        monkeypatch.setattr(
            "gateway.platforms.sendblue.cache_image_from_bytes",
            cache_mock,
        )
        local_path, mime_type = await adapter._download_and_cache_media(
            "https://cdn.sendblue.com/img/test.jpg"
        )
        assert local_path == "/cache/foo.jpg"
        assert mime_type == "image/jpeg"
        cache_mock.assert_called_once_with(b"fake-image-bytes", ".jpg")

    @pytest.mark.asyncio
    async def test_audio_url_cached_via_audio_helper(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.client = AsyncMock()
        adapter.client.get = AsyncMock(
            return_value=_MockHttpxResponse(content=b"fake-audio-bytes")
        )
        cache_audio = Mock(return_value="/cache/voice.caf")
        cache_image = Mock(side_effect=AssertionError("image helper called for audio"))
        cache_doc = Mock(side_effect=AssertionError("doc helper called for audio"))
        monkeypatch.setattr(
            "gateway.platforms.sendblue.cache_audio_from_bytes", cache_audio,
        )
        monkeypatch.setattr(
            "gateway.platforms.sendblue.cache_image_from_bytes", cache_image,
        )
        monkeypatch.setattr(
            "gateway.platforms.sendblue.cache_document_from_bytes", cache_doc,
        )
        local_path, mime_type = await adapter._download_and_cache_media(
            "https://cdn.sendblue.com/audio/voice.caf"
        )
        assert local_path == "/cache/voice.caf"
        assert mime_type == "audio/x-caf"
        cache_audio.assert_called_once_with(b"fake-audio-bytes", ".caf")

    @pytest.mark.asyncio
    async def test_video_url_cached_via_document_helper(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.client = AsyncMock()
        adapter.client.get = AsyncMock(
            return_value=_MockHttpxResponse(content=b"fake-video-bytes")
        )
        cache_doc = Mock(return_value="/cache/clip.mp4")
        monkeypatch.setattr(
            "gateway.platforms.sendblue.cache_document_from_bytes", cache_doc,
        )
        local_path, mime_type = await adapter._download_and_cache_media(
            "https://cdn.sendblue.com/video/clip.mp4"
        )
        assert local_path == "/cache/clip.mp4"
        assert mime_type == "video/mp4"
        cache_doc.assert_called_once()
        # First positional arg is the bytes, second is the filename
        assert cache_doc.call_args[0][0] == b"fake-video-bytes"
        assert cache_doc.call_args[0][1] == "clip.mp4"

    @pytest.mark.asyncio
    async def test_document_url_cached_via_document_helper(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.client = AsyncMock()
        adapter.client.get = AsyncMock(
            return_value=_MockHttpxResponse(content=b"%PDF-fake")
        )
        cache_doc = Mock(return_value="/cache/report.pdf")
        monkeypatch.setattr(
            "gateway.platforms.sendblue.cache_document_from_bytes", cache_doc,
        )
        local_path, mime_type = await adapter._download_and_cache_media(
            "https://cdn.sendblue.com/doc/report.pdf"
        )
        assert local_path == "/cache/report.pdf"
        assert mime_type == "application/octet-stream"
        assert cache_doc.call_args[0][1] == "report.pdf"

    @pytest.mark.asyncio
    async def test_unknown_extension_falls_to_document(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.client = AsyncMock()
        adapter.client.get = AsyncMock(
            return_value=_MockHttpxResponse(content=b"mystery")
        )
        cache_doc = Mock(return_value="/cache/file.xyz")
        monkeypatch.setattr(
            "gateway.platforms.sendblue.cache_document_from_bytes", cache_doc,
        )
        local_path, mime_type = await adapter._download_and_cache_media(
            "https://cdn.sendblue.com/raw/file.xyz"
        )
        assert local_path == "/cache/file.xyz"
        assert mime_type == "application/octet-stream"

    @pytest.mark.asyncio
    async def test_webhook_routes_audio_to_voice_message_type(self, monkeypatch):
        """End-to-end: .caf webhook → handle_message gets MessageType.VOICE
        with media_urls/media_types populated."""
        from gateway.platforms.base import MessageType
        adapter = _make_adapter(monkeypatch)
        adapter.client = AsyncMock()
        adapter.client.get = AsyncMock(
            return_value=_MockHttpxResponse(content=b"audio-bytes")
        )
        monkeypatch.setattr(
            "gateway.platforms.sendblue.cache_audio_from_bytes",
            lambda data, ext: "/cache/voice.caf",
        )
        adapter.handle_message = AsyncMock()
        request = _MockRequest(
            body={
                "is_outbound": False,
                "sendblue_number": "+15555550100",
                "from_number": "+17706768883",
                "media_url": "https://cdn.sendblue.com/audio/voice.caf",
            },
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        response = await adapter._handle_webhook(request)
        assert response.status == 200
        await _drain_background_tasks(adapter)
        event = adapter.handle_message.call_args[0][0]
        assert event.message_type == MessageType.VOICE
        assert event.media_urls == ["/cache/voice.caf"]
        assert event.media_types == ["audio/x-caf"]

    @pytest.mark.asyncio
    async def test_download_failure_returns_none(self, monkeypatch, caplog):
        adapter = _make_adapter(monkeypatch)
        adapter.client = AsyncMock()
        adapter.client.get = AsyncMock(
            side_effect=httpx.RequestError(
                "connection refused",
                request=httpx.Request("GET", "http://test"),
            )
        )
        with caplog.at_level("WARNING"):
            local_path, mime_type = await adapter._download_and_cache_media(
                "https://cdn.sendblue.com/img/missing.jpg"
            )
        assert local_path is None
        assert mime_type is None
        assert any(
            "media download failed" in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_media_only_with_failure_dropped_in_handler(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        # Shallow mock: skip the network and cache layers entirely
        adapter._download_and_cache_media = AsyncMock(return_value=(None, None))
        payload = {
            "is_outbound": False,
            "sendblue_number": "+15555550100",
            "from_number": "+17706768883",
            # no content
            "media_url": "https://cdn.sendblue.com/img/test.jpg",
        }
        request = _MockRequest(
            body=payload,
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        response = await adapter._handle_webhook(request)
        await _drain_background_tasks(adapter)
        assert response.status == 200  # batch succeeds, item silently dropped
        assert adapter.handle_message.call_count == 0


class TestSendblueOutboundSend:
    @pytest.mark.asyncio
    async def test_send_makes_correct_api_call(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, '{"message_handle": "abc"}')
        )
        result = await adapter.send("+17706768883", "hello")
        assert result.success is True
        assert result.message_id == "abc"
        adapter._sendblue_api_post.assert_called_once_with(
            "send-message",
            {
                "number": "+17706768883",
                "from_number": "+15555550100",
                "content": "hello",
            },
        )

    @pytest.mark.asyncio
    async def test_truncates_content_over_max_length(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, '{"message_handle": "abc"}')
        )
        # 20000 chars > 18996 MAX_MESSAGE_LENGTH — inherited truncate_message
        # should split into multiple chunks, each POSTed separately
        result = await adapter.send("+17706768883", "X" * 20000)
        assert result.success is True
        assert adapter._sendblue_api_post.call_count > 1

    @pytest.mark.asyncio
    async def test_empty_content_returns_failure(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock()
        result = await adapter.send("+17706768883", "")
        assert result.success is False
        assert "non-empty" in result.error
        adapter._sendblue_api_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_network_error_returns_retryable_failure(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        # status=0 is the transport-error convention from _sendblue_api_post
        adapter._sendblue_api_post = AsyncMock(
            return_value=(0, "connection error")
        )
        result = await adapter.send("+17706768883", "hello")
        assert result.success is False
        assert result.retryable is True

    @pytest.mark.asyncio
    async def test_4xx_returns_non_retryable_failure(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(
            return_value=(400, '{"error": "bad request"}')
        )
        result = await adapter.send("+17706768883", "hello")
        assert result.success is False
        assert result.retryable is False


class TestSendblueSendImage:
    def test_public_image_url_truth_table(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        # Public HTTPS image URLs -> True
        assert adapter._is_public_image_url("https://cdn.example.com/img.jpg") is True
        assert adapter._is_public_image_url("https://cdn.example.com/img.png") is True
        assert adapter._is_public_image_url("https://cdn.example.com/img.JPG") is True
        assert adapter._is_public_image_url("https://cdn.example.com/img.heic") is True
        assert adapter._is_public_image_url("https://cdn.example.com/img.jpg?token=abc") is True

    def test_non_public_url_truth_table(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        # Non-HTTPS or no/wrong extension -> False
        assert adapter._is_public_image_url("http://cdn.example.com/img.jpg") is False
        assert adapter._is_public_image_url("file:///tmp/img.jpg") is False
        assert adapter._is_public_image_url("https://cdn.example.com/img.txt") is False
        assert adapter._is_public_image_url("https://cdn.example.com/noext") is False
        assert adapter._is_public_image_url("") is False
        assert adapter._is_public_image_url("not-a-url") is False

    @pytest.mark.asyncio
    async def test_public_image_sends_with_media_url(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, '{"message_handle": "img-abc"}')
        )
        result = await adapter.send_image(
            "+17706768883",
            "https://cdn.example.com/img.jpg",
            caption=None,
        )
        assert result.success is True
        assert result.message_id == "img-abc"
        adapter._sendblue_api_post.assert_called_once_with(
            "send-message",
            {
                "number": "+17706768883",
                "from_number": "+15555550100",
                "media_url": "https://cdn.example.com/img.jpg",
            },
        )

    @pytest.mark.asyncio
    async def test_public_image_with_caption_includes_content(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, '{"message_handle": "img-xyz"}')
        )
        result = await adapter.send_image(
            "+17706768883",
            "https://cdn.example.com/img.jpg",
            caption="look at this",
        )
        assert result.success is True
        adapter._sendblue_api_post.assert_called_once_with(
            "send-message",
            {
                "number": "+17706768883",
                "from_number": "+15555550100",
                "media_url": "https://cdn.example.com/img.jpg",
                "content": "look at this",
            },
        )

    @pytest.mark.asyncio
    async def test_non_public_url_falls_back_to_base_class(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock()  # should not be called
        super_send_image = AsyncMock()
        monkeypatch.setattr(
            "gateway.platforms.base.BasePlatformAdapter.send_image",
            super_send_image,
        )
        await adapter.send_image(
            "+17706768883",
            "http://cdn.example.com/img.jpg",  # http, not https
            caption="hi",
        )
        super_send_image.assert_called_once()
        adapter._sendblue_api_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_4xx_returns_non_retryable_failure(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(
            return_value=(400, '{"error": "invalid media_url"}')
        )
        result = await adapter.send_image(
            "+17706768883",
            "https://cdn.example.com/img.jpg",
        )
        assert result.success is False
        assert result.retryable is False


class TestSendblueQuotaCommand:
    def test_format_zero_outbound(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        out = adapter._format_quota_response({"outbound": 0, "inbound": 5})
        assert out.startswith("📊 Sendblue (since 3am EST)")
        assert "↑ 0 sent / 200 daily cap" in out
        assert "[░░░░░░░░] 0%" in out
        assert "↓ 5 received" in out

    def test_format_at_cap(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        out = adapter._format_quota_response({"outbound": 200, "inbound": 0})
        assert "[████████] 100%" in out
        assert "↑ 200 sent / 200 daily cap" in out

    def test_format_custom_cap(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, sendblue_daily_cap=500)
        out = adapter._format_quota_response({"outbound": 250, "inbound": 0})
        assert "/ 500 daily cap" in out
        assert "50%" in out
        assert "[████░░░░]" in out

    @pytest.mark.asyncio
    async def test_fetch_usage_happy_path(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_get = AsyncMock(side_effect=[
            (200, {"pagination": {"total": 42}}),
            (200, {"pagination": {"total": 17}}),
        ])
        result = await adapter._fetch_sendblue_usage()
        assert result["outbound"] == 42
        assert result["inbound"] == 17
        assert result["source"] == "Sendblue API"
        assert adapter._sendblue_api_get.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_usage_cache_hit(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_get = AsyncMock(side_effect=[
            (200, {"pagination": {"total": 5}}),
            (200, {"pagination": {"total": 3}}),
        ])
        first = await adapter._fetch_sendblue_usage()
        second = await adapter._fetch_sendblue_usage()
        assert second["outbound"] == 5
        assert second["source"] == "cache"
        assert adapter._sendblue_api_get.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_usage_api_error_no_cache(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_get = AsyncMock(return_value=(500, "server error"))
        result = await adapter._fetch_sendblue_usage()
        assert result["source"] == "error"
        assert "500" in (result.get("error") or "")
        assert result["outbound"] == 0

    @pytest.mark.asyncio
    async def test_quota_intercept_short_circuits_agent(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._fetch_sendblue_usage = AsyncMock(return_value={
            "outbound": 10, "inbound": 5, "day_key": "x",
            "source": "Sendblue API", "error": None,
        })
        adapter.send = AsyncMock()
        adapter.handle_message = AsyncMock()
        request = _MockRequest(
            body={
                "is_outbound": False,
                "sendblue_number": "+15555550100",
                "from_number": "+17706768883",
                "content": "/quota",
                "message_handle": "abc",
            },
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        response = await adapter._handle_webhook(request)
        assert response.status == 200
        await _drain_background_tasks(adapter)
        adapter._fetch_sendblue_usage.assert_called_once()
        adapter.send.assert_called_once()
        sent_content = adapter.send.call_args[0][1]
        assert "📊 Sendblue" in sent_content
        adapter.handle_message.assert_not_called()


class TestSendblueReadReceiptsAndTyping:
    @pytest.mark.asyncio
    async def test_mark_read_posts_correct_payload(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(return_value=(200, {}))
        ok = await adapter.mark_read("+17706768883")
        assert ok is True
        adapter._sendblue_api_post.assert_called_once_with(
            "mark-read",
            {"number": "+17706768883", "from_number": "+15555550100"},
            timeout=5.0,
        )

    @pytest.mark.asyncio
    async def test_mark_read_disabled_when_flag_false(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        adapter._sendblue_api_post = AsyncMock()
        ok = await adapter.mark_read("+17706768883")
        assert ok is False
        adapter._sendblue_api_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_mark_read_failure_returns_false(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(return_value=(400, "bad"))
        ok = await adapter.mark_read("+17706768883")
        assert ok is False

    @pytest.mark.asyncio
    async def test_send_typing_posts_correct_payload(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(return_value=(200, {}))
        result = await adapter.send_typing("+17706768883")
        assert result is None
        adapter._sendblue_api_post.assert_called_once_with(
            "send-typing-indicator",
            {"number": "+17706768883", "from_number": "+15555550100"},
            timeout=5.0,
        )

    @pytest.mark.asyncio
    async def test_webhook_dispatches_mark_read(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        adapter.mark_read = AsyncMock(return_value=True)
        request = _MockRequest(
            body={
                "is_outbound": False,
                "sendblue_number": "+15555550100",
                "from_number": "+17706768883",
                "content": "hello",
            },
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        response = await adapter._handle_webhook(request)
        assert response.status == 200
        await _drain_background_tasks(adapter)
        adapter.mark_read.assert_called_once_with("+17706768883")

    @pytest.mark.asyncio
    async def test_webhook_skips_mark_read_for_sms(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        adapter.mark_read = AsyncMock(return_value=True)
        request = _MockRequest(
            body={
                "is_outbound": False,
                "sendblue_number": "+15555550100",
                "from_number": "+17706768883",
                "content": "hello",
                "service": "sms",
            },
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        response = await adapter._handle_webhook(request)
        assert response.status == 200
        await _drain_background_tasks(adapter)
        adapter.mark_read.assert_not_called()

    @pytest.mark.asyncio
    async def test_webhook_skips_mark_read_for_rcs(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        adapter.mark_read = AsyncMock(return_value=True)
        request = _MockRequest(
            body={
                "is_outbound": False,
                "sendblue_number": "+15555550100",
                "from_number": "+17706768883",
                "content": "hello",
                "service": "RCS",
            },
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        await adapter._handle_webhook(request)
        await _drain_background_tasks(adapter)
        adapter.mark_read.assert_not_called()

    @pytest.mark.asyncio
    async def test_webhook_dispatches_mark_read_when_service_missing(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        adapter.mark_read = AsyncMock(return_value=True)
        request = _MockRequest(
            body={
                "is_outbound": False,
                "sendblue_number": "+15555550100",
                "from_number": "+17706768883",
                "content": "hello",
                # no `service` field — older payloads, forward-compat default
            },
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        await adapter._handle_webhook(request)
        await _drain_background_tasks(adapter)
        adapter.mark_read.assert_called_once_with("+17706768883")


class TestSendblueSendStyle:
    def test_normalize_valid_style_lowercased(self, monkeypatch):
        from gateway.platforms.sendblue import _normalize_send_style
        assert _normalize_send_style("Confetti") == "confetti"
        assert _normalize_send_style("  GENTLE  ") == "gentle"

    def test_normalize_none_and_empty(self, monkeypatch):
        from gateway.platforms.sendblue import _normalize_send_style
        assert _normalize_send_style(None) is None
        assert _normalize_send_style("") is None
        assert _normalize_send_style("   ") is None

    def test_normalize_invalid_returns_none_and_warns(self, monkeypatch, caplog):
        from gateway.platforms.sendblue import _normalize_send_style
        with caplog.at_level("WARNING"):
            assert _normalize_send_style("nuclear") is None
        assert any("invalid send_style" in r.message for r in caplog.records)

    def test_default_style_from_env(self, monkeypatch):
        monkeypatch.setenv("SENDBLUE_DEFAULT_SEND_STYLE", "confetti")
        adapter = _make_adapter(monkeypatch)
        assert adapter.default_send_style == "confetti"

    def test_default_style_from_extra_overrides_env(self, monkeypatch):
        monkeypatch.setenv("SENDBLUE_DEFAULT_SEND_STYLE", "confetti")
        adapter = _make_adapter(monkeypatch, sendblue_default_send_style="slam")
        assert adapter.default_send_style == "slam"

    def test_default_style_invalid_env_is_dropped(self, monkeypatch, caplog):
        monkeypatch.setenv("SENDBLUE_DEFAULT_SEND_STYLE", "nuclear")
        with caplog.at_level("WARNING"):
            adapter = _make_adapter(monkeypatch)
        assert adapter.default_send_style is None

    @pytest.mark.asyncio
    async def test_send_includes_default_style(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, sendblue_default_send_style="confetti")
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, json.dumps({"message_handle": "h1"}))
        )
        await adapter.send("+17706768883", "hi")
        sent_payload = adapter._sendblue_api_post.call_args[0][1]
        assert sent_payload["send_style"] == "confetti"

    @pytest.mark.asyncio
    async def test_send_no_style_when_default_unset(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, json.dumps({"message_handle": "h1"}))
        )
        await adapter.send("+17706768883", "hi")
        sent_payload = adapter._sendblue_api_post.call_args[0][1]
        assert "send_style" not in sent_payload

    @pytest.mark.asyncio
    async def test_send_metadata_overrides_default(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, sendblue_default_send_style="confetti")
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, json.dumps({"message_handle": "h1"}))
        )
        await adapter.send("+17706768883", "hi", metadata={"send_style": "slam"})
        sent_payload = adapter._sendblue_api_post.call_args[0][1]
        assert sent_payload["send_style"] == "slam"

    @pytest.mark.asyncio
    async def test_send_metadata_none_suppresses_default(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, sendblue_default_send_style="confetti")
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, json.dumps({"message_handle": "h1"}))
        )
        await adapter.send("+17706768883", "hi", metadata={"send_style": None})
        sent_payload = adapter._sendblue_api_post.call_args[0][1]
        assert "send_style" not in sent_payload

    @pytest.mark.asyncio
    async def test_send_invalid_metadata_style_dropped(self, monkeypatch, caplog):
        adapter = _make_adapter(monkeypatch, sendblue_default_send_style="confetti")
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, json.dumps({"message_handle": "h1"}))
        )
        with caplog.at_level("WARNING"):
            await adapter.send("+17706768883", "hi", metadata={"send_style": "nuclear"})
        sent_payload = adapter._sendblue_api_post.call_args[0][1]
        assert "send_style" not in sent_payload

    @pytest.mark.asyncio
    async def test_send_image_includes_style(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, sendblue_default_send_style="balloons")
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, json.dumps({"message_handle": "h1"}))
        )
        await adapter.send_image(
            "+17706768883", "https://example.com/a.png", caption="cap"
        )
        sent_payload = adapter._sendblue_api_post.call_args[0][1]
        assert sent_payload["send_style"] == "balloons"
        assert sent_payload["media_url"] == "https://example.com/a.png"
        assert sent_payload["content"] == "cap"


class _MockUploadResponse:
    """Minimal httpx.Response surface for upload_file tests."""
    def __init__(self, status_code=200, body=None, raise_on_json=False):
        self.status_code = status_code
        self._body = body or {}
        self._raise_on_json = raise_on_json
        self.text = json.dumps(self._body) if not raise_on_json else "not json"

    def json(self):
        if self._raise_on_json:
            raise ValueError("not json")
        return self._body


class TestSendblueMediaUpload:
    @pytest.mark.asyncio
    async def test_upload_file_happy_path(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch)
        f = tmp_path / "pic.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        adapter.client = AsyncMock()
        adapter.client.post = AsyncMock(
            return_value=_MockUploadResponse(
                200, {"media_url": "https://cdn.sb/abc.png"}
            )
        )
        media_url = await adapter._upload_file_to_sendblue(str(f))
        assert media_url == "https://cdn.sb/abc.png"
        # Verify multipart was used (files= kwarg present)
        _, kwargs = adapter.client.post.call_args
        assert "files" in kwargs
        assert kwargs["headers"]["sb-api-key-id"] == "test-key-id"
        # Content-Type NOT injected (httpx sets multipart boundary itself)
        assert "Content-Type" not in kwargs["headers"]

    @pytest.mark.asyncio
    async def test_upload_file_missing_returns_none(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.client = AsyncMock()
        result = await adapter._upload_file_to_sendblue("/no/such/file.png")
        assert result is None
        adapter.client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_file_http_error_returns_none(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch)
        f = tmp_path / "pic.png"
        f.write_bytes(b"data")
        adapter.client = AsyncMock()
        adapter.client.post = AsyncMock(
            return_value=_MockUploadResponse(500, {"error": "boom"})
        )
        assert await adapter._upload_file_to_sendblue(str(f)) is None

    @pytest.mark.asyncio
    async def test_upload_file_no_media_url_returns_none(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch)
        f = tmp_path / "pic.png"
        f.write_bytes(b"data")
        adapter.client = AsyncMock()
        adapter.client.post = AsyncMock(
            return_value=_MockUploadResponse(200, {"status": "OK"})  # no media_url
        )
        assert await adapter._upload_file_to_sendblue(str(f)) is None

    @pytest.mark.asyncio
    async def test_upload_file_timeout_returns_none(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch)
        f = tmp_path / "pic.png"
        f.write_bytes(b"data")
        adapter.client = AsyncMock()
        adapter.client.post = AsyncMock(side_effect=httpx.TimeoutException("slow"))
        assert await adapter._upload_file_to_sendblue(str(f)) is None

    @pytest.mark.asyncio
    async def test_send_image_file_uploads_then_sends(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch)
        f = tmp_path / "pic.png"
        f.write_bytes(b"data")
        adapter._upload_file_to_sendblue = AsyncMock(
            return_value="https://cdn.sb/pic.png"
        )
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, json.dumps({"message_handle": "h1"}))
        )
        result = await adapter.send_image_file("+17706768883", str(f), caption="cap")
        assert result.success
        sent_payload = adapter._sendblue_api_post.call_args[0][1]
        assert sent_payload["media_url"] == "https://cdn.sb/pic.png"
        assert sent_payload["content"] == "cap"

    @pytest.mark.asyncio
    async def test_send_image_file_upload_failure_propagates(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch)
        f = tmp_path / "pic.png"
        f.write_bytes(b"data")
        adapter._upload_file_to_sendblue = AsyncMock(return_value=None)
        adapter._sendblue_api_post = AsyncMock()
        result = await adapter.send_image_file("+17706768883", str(f))
        assert not result.success
        assert result.retryable is True
        adapter._sendblue_api_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_voice_caf_no_warning(self, monkeypatch, tmp_path, caplog):
        adapter = _make_adapter(monkeypatch)
        f = tmp_path / "memo.caf"
        f.write_bytes(b"data")
        adapter._upload_file_to_sendblue = AsyncMock(
            return_value="https://cdn.sb/memo.caf"
        )
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, json.dumps({"message_handle": "h1"}))
        )
        with caplog.at_level("DEBUG"):
            result = await adapter.send_voice("+17706768883", str(f))
        assert result.success
        assert not any("not .caf" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_send_voice_non_caf_logs_debug(self, monkeypatch, tmp_path, caplog):
        adapter = _make_adapter(monkeypatch)
        f = tmp_path / "memo.mp3"
        f.write_bytes(b"data")
        adapter._upload_file_to_sendblue = AsyncMock(
            return_value="https://cdn.sb/memo.mp3"
        )
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, json.dumps({"message_handle": "h1"}))
        )
        with caplog.at_level("DEBUG"):
            await adapter.send_voice("+17706768883", str(f))
        assert any("not .caf" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_send_video_uploads_then_sends(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch)
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"data")
        adapter._upload_file_to_sendblue = AsyncMock(
            return_value="https://cdn.sb/clip.mp4"
        )
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, json.dumps({"message_handle": "h1"}))
        )
        result = await adapter.send_video("+17706768883", str(f))
        assert result.success
        assert adapter._sendblue_api_post.call_args[0][1]["media_url"].endswith(".mp4")

    @pytest.mark.asyncio
    async def test_send_document_uploads_then_sends(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch)
        f = tmp_path / "report.pdf"
        f.write_bytes(b"data")
        adapter._upload_file_to_sendblue = AsyncMock(
            return_value="https://cdn.sb/report.pdf"
        )
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, json.dumps({"message_handle": "h1"}))
        )
        result = await adapter.send_document(
            "+17706768883", str(f), caption="see attached"
        )
        assert result.success
        payload = adapter._sendblue_api_post.call_args[0][1]
        assert payload["content"] == "see attached"
        assert payload["media_url"].endswith(".pdf")

    @pytest.mark.asyncio
    async def test_send_animation_delegates_to_send_image(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, json.dumps({"message_handle": "h1"}))
        )
        result = await adapter.send_animation(
            "+17706768883", "https://example.com/dance.gif", caption="lol"
        )
        assert result.success
        payload = adapter._sendblue_api_post.call_args[0][1]
        assert payload["media_url"] == "https://example.com/dance.gif"
        assert payload["content"] == "lol"

    @pytest.mark.asyncio
    async def test_media_send_inherits_default_send_style(self, monkeypatch, tmp_path):
        adapter = _make_adapter(monkeypatch, sendblue_default_send_style="confetti")
        f = tmp_path / "pic.png"
        f.write_bytes(b"data")
        adapter._upload_file_to_sendblue = AsyncMock(
            return_value="https://cdn.sb/pic.png"
        )
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, json.dumps({"message_handle": "h1"}))
        )
        await adapter.send_image_file("+17706768883", str(f))
        sent_payload = adapter._sendblue_api_post.call_args[0][1]
        assert sent_payload["send_style"] == "confetti"


class TestSendblueGroupChat:
    def test_is_group_chat_id_truth_table(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        # E.164 phone numbers = DM
        assert adapter._is_group_chat_id("+17706768883") is False
        assert adapter._is_group_chat_id("+1234567890") is False
        # Non-phone-prefixed strings = group
        assert adapter._is_group_chat_id("group_abc123") is True
        assert adapter._is_group_chat_id("550e8400-e29b-41d4-a716-446655440000") is True
        # Edge cases
        assert adapter._is_group_chat_id("") is False
        assert adapter._is_group_chat_id(None) is False

    @pytest.mark.asyncio
    async def test_get_chat_info_distinguishes_dm_and_group(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        dm_info = await adapter.get_chat_info("+17706768883")
        assert dm_info["type"] == "dm"
        group_info = await adapter.get_chat_info("group_abc123")
        assert group_info["type"] == "group"

    @pytest.mark.asyncio
    async def test_webhook_group_message_routes_to_group_id(self, monkeypatch):
        """Group webhook → MessageEvent.source.chat_id is group_id,
        chat_type is "group", user_id is the sender's phone."""
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        adapter.mark_read = AsyncMock()
        request = _MockRequest(
            body={
                "is_outbound": False,
                "sendblue_number": "+15555550100",
                "from_number": "+17706768883",
                "content": "hello group",
                "group_id": "550e8400-e29b-41d4-a716-446655440000",
                "group_display_name": "Family Chat",
                "participants": ["+17706768883", "+15551234567"],
            },
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        response = await adapter._handle_webhook(request)
        assert response.status == 200
        await _drain_background_tasks(adapter)
        event = adapter.handle_message.call_args[0][0]
        assert event.source.chat_id == "550e8400-e29b-41d4-a716-446655440000"
        assert event.source.chat_type == "group"
        assert event.source.chat_name == "Family Chat"
        assert event.source.user_id == "+17706768883"
        # Read receipts not sent for group messages
        adapter.mark_read.assert_not_called()

    @pytest.mark.asyncio
    async def test_webhook_dm_unchanged_by_group_plumbing(self, monkeypatch):
        """Regression guard: regular DM webhooks still route to from_number
        with chat_type='dm' after the group support refactor."""
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        request = _MockRequest(
            body={
                "is_outbound": False,
                "sendblue_number": "+15555550100",
                "from_number": "+17706768883",
                "content": "hi",
                "group_id": "",  # empty string = DM per docs
            },
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        response = await adapter._handle_webhook(request)
        assert response.status == 200
        await _drain_background_tasks(adapter)
        event = adapter.handle_message.call_args[0][0]
        assert event.source.chat_id == "+17706768883"
        assert event.source.chat_type == "dm"

    @pytest.mark.asyncio
    async def test_send_to_group_uses_group_endpoint(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, json.dumps({"message_handle": "h1"}))
        )
        result = await adapter.send("550e8400-e29b-41d4-a716-446655440000", "hello")
        assert result.success
        endpoint, payload = adapter._sendblue_api_post.call_args[0][:2]
        assert endpoint == "send-group-message"
        assert payload["group_id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert "number" not in payload

    @pytest.mark.asyncio
    async def test_send_to_dm_uses_regular_endpoint(self, monkeypatch):
        """Regression guard for DM path after group routing landed."""
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, json.dumps({"message_handle": "h1"}))
        )
        await adapter.send("+17706768883", "hi")
        endpoint, payload = adapter._sendblue_api_post.call_args[0][:2]
        assert endpoint == "send-message"
        assert payload["number"] == "+17706768883"
        assert "group_id" not in payload

    @pytest.mark.asyncio
    async def test_send_image_to_group_uses_group_endpoint(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(
            return_value=(200, json.dumps({"message_handle": "h1"}))
        )
        await adapter.send_image(
            "550e8400-e29b-41d4-a716-446655440000",
            "https://example.com/a.png",
        )
        endpoint, payload = adapter._sendblue_api_post.call_args[0][:2]
        assert endpoint == "send-group-message"
        assert payload["group_id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert payload["media_url"] == "https://example.com/a.png"
        assert "number" not in payload

    @pytest.mark.asyncio
    async def test_mark_read_skips_groups(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock()
        ok = await adapter.mark_read("550e8400-e29b-41d4-a716-446655440000")
        assert ok is False
        adapter._sendblue_api_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_typing_skips_groups(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock()
        await adapter.send_typing("550e8400-e29b-41d4-a716-446655440000")
        adapter._sendblue_api_post.assert_not_called()


class TestSendblueQuotaDayKey:
    def test_day_key_is_utc_zulu_format(self, monkeypatch):
        """_get_sendblue_day_key must return a Z-suffixed UTC string,
        not an offset-bearing local ISO. Sendblue's created_at_gte
        parsing is not guaranteed to handle offsets — UTC is unambiguous."""
        from gateway.platforms.sendblue import SendblueAdapter
        key = SendblueAdapter._get_sendblue_day_key()
        assert key.endswith("Z"), key
        # No offset like +HH:MM or -HH:MM should appear
        assert "+" not in key
        assert key.count("-") == 2  # only the two date separators
        # Parses cleanly as UTC ISO-8601
        parsed = datetime.fromisoformat(key.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None

    def test_day_key_is_3am_eastern_boundary(self, monkeypatch):
        """The cutoff hour is always 3am America/New_York. After UTC
        conversion that's 07:00Z (EST) or 08:00Z (EDT) — minutes/
        seconds must be zero either way."""
        from gateway.platforms.sendblue import SendblueAdapter
        key = SendblueAdapter._get_sendblue_day_key()
        # Hour is 07 (EST = winter) or 08 (EDT = summer); minutes+seconds zero
        assert key[11:13] in ("07", "08"), key
        assert key[14:16] == "00", key
        assert key[17:19] == "00", key


class TestSendblueMarkReadTaskTracking:
    @pytest.mark.asyncio
    async def test_webhook_mark_read_task_is_tracked(self, monkeypatch):
        """The fire-and-forget mark_read in STEP 3i must register with
        self._background_tasks so adapter shutdown can drain it
        cleanly. Regression guard against the prior bare
        asyncio.create_task that left tasks GC-cancellable."""
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        # Slow mark_read so the task is still pending when we inspect
        # the set (otherwise the discard callback would have fired).
        mark_read_started = asyncio.Event()
        mark_read_release = asyncio.Event()

        async def slow_mark_read(_chat_id):
            mark_read_started.set()
            await mark_read_release.wait()
            return True

        adapter.mark_read = slow_mark_read
        request = _MockRequest(
            body={
                "is_outbound": False,
                "sendblue_number": "+15555550100",
                "from_number": "+17706768883",
                "content": "hi",
            },
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        await adapter._handle_webhook(request)
        await mark_read_started.wait()
        # mark_read task is pending; handle_message task may also be in the
        # set. We just assert at least one of the tracked tasks is the
        # mark_read coroutine.
        tracked_coros = [
            t.get_coro().__qualname__ for t in adapter._background_tasks
        ]
        assert any("slow_mark_read" in q for q in tracked_coros), tracked_coros
        mark_read_release.set()
        await _drain_background_tasks(adapter)
        # After completion, the discard callback should have removed it
        assert all(not t.get_coro().__qualname__.endswith("slow_mark_read")
                   for t in adapter._background_tasks)


class TestSendblueReactions:
    """Reactions stack: per-chat last-inbound cache populated from
    webhooks + send_reaction POST shape + targeting fallback.
    """
    @pytest.mark.asyncio
    async def test_webhook_populates_last_inbound_handle(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        request = _MockRequest(
            body={
                "is_outbound": False,
                "sendblue_number": "+15555550100",
                "from_number": "+17706768883",
                "content": "hi",
                "message_handle": "handle-xyz",
            },
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        await adapter._handle_webhook(request)
        await _drain_background_tasks(adapter)
        assert adapter._last_inbound_handle["+17706768883"] == "handle-xyz"

    @pytest.mark.asyncio
    async def test_webhook_caches_group_handle_by_group_id(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter.handle_message = AsyncMock()
        group_id = "550e8400-e29b-41d4-a716-446655440000"
        request = _MockRequest(
            body={
                "is_outbound": False,
                "sendblue_number": "+15555550100",
                "from_number": "+17706768883",
                "content": "hi",
                "message_handle": "group-handle",
                "group_id": group_id,
            },
            headers={"sb-signing-secret": "test-webhook-secret"},
        )
        await adapter._handle_webhook(request)
        await _drain_background_tasks(adapter)
        assert adapter._last_inbound_handle[group_id] == "group-handle"

    @pytest.mark.asyncio
    async def test_send_reaction_posts_correct_payload(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(return_value=(200, {}))
        adapter._last_inbound_handle["+17706768883"] = "handle-abc"
        ok = await adapter.send_reaction("+17706768883", "love")
        assert ok is True
        adapter._sendblue_api_post.assert_called_once_with(
            "send-reaction",
            {
                "from_number": "+15555550100",
                "message_handle": "handle-abc",
                "reaction": "love",
            },
            timeout=5.0,
        )

    @pytest.mark.asyncio
    async def test_send_reaction_accepts_explicit_handle(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(return_value=(202, {}))
        # Cache empty, but caller passes the handle directly.
        ok = await adapter.send_reaction(
            "+17706768883", "Laugh", message_handle="explicit-handle",
        )
        assert ok is True
        called = adapter._sendblue_api_post.call_args
        assert called.args[1]["message_handle"] == "explicit-handle"
        # Reaction is lowercased to match Sendblue's enum.
        assert called.args[1]["reaction"] == "laugh"

    @pytest.mark.asyncio
    async def test_send_reaction_rejects_invalid_reaction(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock()
        adapter._last_inbound_handle["+17706768883"] = "handle-abc"
        ok = await adapter.send_reaction("+17706768883", "wave")
        assert ok is False
        adapter._sendblue_api_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_reaction_no_handle_available(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock()
        # Cache empty, no explicit handle → fail without API call.
        ok = await adapter.send_reaction("+17706768883", "like")
        assert ok is False
        adapter._sendblue_api_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_reaction_api_failure_returns_false(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(return_value=(500, "boom"))
        adapter._last_inbound_handle["+17706768883"] = "handle-abc"
        ok = await adapter.send_reaction("+17706768883", "like")
        assert ok is False


class TestSendblueAutoMarkReadConfig:
    """auto_mark_read is the canonical key; send_read_receipts is a
    back-compat alias. Both should resolve to the same internal flag."""

    def test_default_is_true(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.auto_mark_read is True
        assert adapter.send_read_receipts is True

    def test_auto_mark_read_false_disables(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, auto_mark_read=False)
        assert adapter.auto_mark_read is False
        assert adapter.send_read_receipts is False

    def test_legacy_send_read_receipts_alias(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        assert adapter.auto_mark_read is False
        assert adapter.send_read_receipts is False

    def test_auto_mark_read_wins_over_legacy(self, monkeypatch):
        # If both are set, the canonical key wins.
        adapter = _make_adapter(
            monkeypatch, auto_mark_read=True, send_read_receipts=False,
        )
        assert adapter.auto_mark_read is True
        assert adapter.send_read_receipts is True


class TestSendblueStatusCallback:
    """status_callback_url propagates into outbound payloads when set,
    and is absent when unset."""

    @pytest.mark.asyncio
    async def test_send_text_no_callback_by_default(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._sendblue_api_post = AsyncMock(return_value=(200, "{}"))
        await adapter.send("+17706768883", "hi")
        payload = adapter._sendblue_api_post.call_args[0][1]
        assert "status_callback" not in payload

    @pytest.mark.asyncio
    async def test_send_text_includes_callback_when_configured(self, monkeypatch):
        url = "https://example.com/sendblue/status"
        adapter = _make_adapter(monkeypatch, status_callback_url=url)
        adapter._sendblue_api_post = AsyncMock(return_value=(200, "{}"))
        await adapter.send("+17706768883", "hi")
        payload = adapter._sendblue_api_post.call_args[0][1]
        assert payload.get("status_callback") == url

    def test_env_var_populates_callback_url(self, monkeypatch):
        url = "https://env.example.com/cb"
        monkeypatch.setenv("SENDBLUE_STATUS_CALLBACK_URL", url)
        adapter = _make_adapter(monkeypatch)
        assert adapter.status_callback_url == url


class TestSendbluePollingConfig:
    """Config field defaults + bounds for polling fallback."""

    def test_defaults_disabled(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.polling_enabled is False
        assert adapter.polling_interval_seconds == 60
        assert adapter.polling_lookback_seconds == 300

    def test_enable_via_extra(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            polling_enabled=True,
            polling_interval_seconds=30,
            polling_lookback_seconds=120,
        )
        assert adapter.polling_enabled is True
        assert adapter.polling_interval_seconds == 30
        assert adapter.polling_lookback_seconds == 120

    def test_minimum_clamps(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            polling_interval_seconds=1,
            polling_lookback_seconds=10,
        )
        assert adapter.polling_interval_seconds == 10
        assert adapter.polling_lookback_seconds == 60


class TestSendbluePollMessagesOnce:
    """_poll_messages_once should fetch, dispatch new messages,
    skip already-seen handles, and advance the cursor."""

    @pytest.mark.asyncio
    async def test_seeds_cursor_on_first_tick(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, polling_enabled=True)
        adapter._sendblue_api_get = AsyncMock(
            return_value=(200, {"status": "OK", "data": []})
        )
        assert adapter._polling_cursor_iso == ""
        await adapter._poll_messages_once()
        assert adapter._polling_cursor_iso.endswith("Z")
        params = adapter._sendblue_api_get.call_args[0][1]
        assert params["is_outbound"] == "false"
        assert "created_at_gte" in params

    @pytest.mark.asyncio
    async def test_dispatches_new_messages(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, polling_enabled=True)
        # Pre-seed cursor older than the test message so the advance
        # check exercises the real comparison.
        adapter._polling_cursor_iso = "2026-05-21T21:00:00.000Z"
        adapter._sendblue_api_get = AsyncMock(return_value=(200, {
            "status": "OK",
            "data": [
                {
                    "from_number": "+17706768883",
                    "content": "hello from poll",
                    "message_handle": "h-new-1",
                    "date_sent": "2026-05-21T22:00:00.000Z",
                    "sendblue_number": "+15555550100",
                    "service": "iMessage",
                },
            ],
        }))
        adapter._process_inbound_item = AsyncMock()
        n = await adapter._poll_messages_once()
        assert n == 1
        adapter._process_inbound_item.assert_awaited_once()
        assert adapter._polling_cursor_iso == "2026-05-21T22:00:00.000Z"

    @pytest.mark.asyncio
    async def test_skips_already_seen_handles(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, polling_enabled=True)
        adapter._seen_handles["h-old"] = None
        adapter._sendblue_api_get = AsyncMock(return_value=(200, {
            "status": "OK",
            "data": [
                {
                    "from_number": "+17706768883",
                    "content": "dup",
                    "message_handle": "h-old",
                    "date_sent": "2026-05-21T22:00:00.000Z",
                },
            ],
        }))
        adapter._process_inbound_item = AsyncMock()
        n = await adapter._poll_messages_once()
        assert n == 0
        adapter._process_inbound_item.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_fetch_returns_zero_no_cursor_advance(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, polling_enabled=True)
        adapter._polling_cursor_iso = "2026-05-21T21:00:00.000Z"
        adapter._sendblue_api_get = AsyncMock(return_value=(500, "boom"))
        adapter._process_inbound_item = AsyncMock()
        n = await adapter._poll_messages_once()
        assert n == 0
        adapter._process_inbound_item.assert_not_awaited()
        assert adapter._polling_cursor_iso == "2026-05-21T21:00:00.000Z"

    @pytest.mark.asyncio
    async def test_unexpected_shape_returns_zero(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, polling_enabled=True)
        adapter._sendblue_api_get = AsyncMock(
            return_value=(200, {"data": "not-a-list"})
        )
        adapter._process_inbound_item = AsyncMock()
        n = await adapter._poll_messages_once()
        assert n == 0
        adapter._process_inbound_item.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_legacy_messages_key_shape_still_works(self, monkeypatch):
        # Forward/back-compat: tolerate {"messages": [...]} as well.
        adapter = _make_adapter(monkeypatch, polling_enabled=True)
        adapter._polling_cursor_iso = "2026-05-21T21:00:00.000Z"
        adapter._sendblue_api_get = AsyncMock(return_value=(200, {
            "messages": [
                {
                    "from_number": "+17706768883",
                    "content": "legacy shape",
                    "message_handle": "h-legacy",
                    "created_at": "2026-05-21T22:00:00.000Z",
                    "sendblue_number": "+15555550100",
                },
            ],
        }))
        adapter._process_inbound_item = AsyncMock()
        n = await adapter._poll_messages_once()
        assert n == 1
        assert adapter._polling_cursor_iso == "2026-05-21T22:00:00.000Z"


class TestSendbluePollingLoop:
    """_polling_loop sleeps between ticks, swallows iteration errors,
    and cancels cleanly."""

    @pytest.mark.asyncio
    async def test_cancellation_propagates(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch, polling_enabled=True, polling_interval_seconds=60,
        )
        adapter._poll_messages_once = AsyncMock(return_value=0)
        task = asyncio.create_task(adapter._polling_loop())
        await asyncio.sleep(0)  # let the loop run one iteration
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        adapter._poll_messages_once.assert_awaited()

    @pytest.mark.asyncio
    async def test_iteration_exception_does_not_kill_loop(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch, polling_enabled=True, polling_interval_seconds=60,
        )
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient")
            return 0

        adapter._poll_messages_once = flaky
        task = asyncio.create_task(adapter._polling_loop())
        # Yield enough times for the flaky iteration to fire and the
        # loop to enter sleep.
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert call_count >= 1


class TestSendblueConfigLoading:
    """Env-var precedence + canonical-key/alias handling.

    Targets the `_apply_env_overrides` blank-out trap (env vars take
    precedence over `extra` when present), and the auto_mark_read /
    send_read_receipts alias resolution. These are the config gotchas
    most likely to bite operators.
    """

    def test_extra_takes_precedence_over_env_when_both_set(self, monkeypatch):
        from gateway.platforms.sendblue import SendblueAdapter
        monkeypatch.setenv("SENDBLUE_API_KEY_ID", "env-key")
        monkeypatch.setenv("SENDBLUE_API_SECRET", "env-secret")
        cfg = PlatformConfig(
            enabled=True,
            extra={"api_key_id": "extra-key", "api_secret": "extra-secret"},
        )
        adapter = SendblueAdapter(cfg)
        assert adapter.api_key_id == "extra-key"
        assert adapter.api_secret == "extra-secret"

    def test_env_used_when_extra_missing(self, monkeypatch):
        from gateway.platforms.sendblue import SendblueAdapter
        monkeypatch.setenv("SENDBLUE_API_KEY_ID", "env-key")
        monkeypatch.setenv("SENDBLUE_API_SECRET", "env-secret")
        monkeypatch.setenv("SENDBLUE_NUMBER", "+15555550199")
        cfg = PlatformConfig(enabled=True, extra={})
        adapter = SendblueAdapter(cfg)
        assert adapter.api_key_id == "env-key"
        assert adapter.api_secret == "env-secret"
        assert adapter.sendblue_number == "+15555550199"

    def test_auto_mark_read_canonical_key_wins(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            auto_mark_read=False,
            send_read_receipts=True,
        )
        assert adapter.auto_mark_read is False
        assert adapter.send_read_receipts is False

    def test_send_read_receipts_alias_when_canonical_missing(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, send_read_receipts=False)
        assert adapter.auto_mark_read is False
        assert adapter.send_read_receipts is False

    def test_auto_mark_read_defaults_true(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.auto_mark_read is True
        assert adapter.send_read_receipts is True

    def test_disable_signature_check_defaults_false(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.disable_signature_check is False

    def test_disable_signature_check_from_env(self, monkeypatch):
        monkeypatch.setenv("SENDBLUE_DISABLE_SIGNATURE_CHECK", "true")
        adapter = _make_adapter(monkeypatch)
        assert adapter.disable_signature_check is True

    def test_polling_interval_floor_enforced(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, polling_interval_seconds=1)
        assert adapter.polling_interval_seconds == 10

    def test_webhook_path_normalized_to_leading_slash(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_path="sendblue/hook")
        assert adapter.webhook_path == "/sendblue/hook"


class TestSendblueConnectLifecycle:
    """Connect/disconnect preconditions and polling-task lifecycle.

    Covers the silent-startup failure modes: missing credentials,
    polling task not spawned/cancelled cleanly. These would otherwise
    surface only as "inbound is dead but service started green."
    """

    @pytest.mark.asyncio
    async def test_connect_fails_without_credentials(self, monkeypatch):
        from gateway.platforms.sendblue import SendblueAdapter
        cfg = PlatformConfig(enabled=True, extra={})
        adapter = SendblueAdapter(cfg)
        assert await adapter.connect() is False

    @staticmethod
    def _patch_connect_io(monkeypatch, adapter):
        """Stub the IO inside connect() so the test stays hermetic."""
        adapter._sendblue_api_get = AsyncMock(return_value=(200, {"webhooks": []}))
        adapter._register_webhook = AsyncMock(return_value=True)
        adapter._unregister_webhook = AsyncMock(return_value=True)

        class _NoOpRunner:
            async def setup(self):
                pass

            async def cleanup(self):
                pass

        class _NoOpSite:
            def __init__(self, *_a, **_kw):
                pass

            async def start(self):
                pass

        import aiohttp.web as _web
        monkeypatch.setattr(_web, "AppRunner", lambda *_a, **_kw: _NoOpRunner())
        monkeypatch.setattr(_web, "TCPSite", _NoOpSite)

    @pytest.mark.asyncio
    async def test_polling_task_spawned_when_enabled(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch, polling_enabled=True, polling_interval_seconds=60,
        )
        self._patch_connect_io(monkeypatch, adapter)
        adapter._poll_messages_once = AsyncMock(return_value=0)

        try:
            assert await adapter.connect() is True
            assert adapter._polling_task is not None
            assert not adapter._polling_task.done()
        finally:
            await adapter.disconnect()
            assert adapter._polling_task is None

    @pytest.mark.asyncio
    async def test_polling_task_not_spawned_when_disabled(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, polling_enabled=False)
        self._patch_connect_io(monkeypatch, adapter)

        try:
            assert await adapter.connect() is True
            assert adapter._polling_task is None
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_cancels_running_polling_task(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch, polling_enabled=True, polling_interval_seconds=60,
        )
        adapter._poll_messages_once = AsyncMock(return_value=0)
        adapter._unregister_webhook = AsyncMock(return_value=True)

        adapter._polling_task = asyncio.create_task(adapter._polling_loop())
        await asyncio.sleep(0)  # let loop start
        assert not adapter._polling_task.done()

        await adapter.disconnect()
        assert adapter._polling_task is None
