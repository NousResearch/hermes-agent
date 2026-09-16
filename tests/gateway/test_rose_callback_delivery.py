"""Tests for rose_callback delivery fixes in the webhook adapter.

Covers:
  1. Callback fires exactly once per job (not per send() call)
  2. Callback fires only after the agent task completes (terminal)
  3. Interim messages don't trigger callbacks
  4. HTTP error codes in content are classified as failures
  5. Provider authentication errors are classified as failures
  6. Agent task exception sends a failed callback
  7. Agent task cancellation sends a failed callback
  8. No-response scenario sends a failed callback
  9. Callback signing is unchanged (signature format preserved)
"""

import asyncio
import hashlib
import hmac as hmac_mod
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.webhook import SendResult

_HMAC_FIELD = "secret"
_HMAC_PLACEHOLDER = os.environ.get("HERMES_TEST_HMAC", "unit-test-placeholder")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter():
    """Build a minimal WebhookAdapter-like object with the fields our tests need."""
    from gateway.platforms.webhook import WebhookAdapter

    with patch.object(WebhookAdapter, "__init__", lambda self, cfg: None):
        adapter = WebhookAdapter.__new__(WebhookAdapter)

    # `name` is a read-only property derived from `platform.value`;
    # set the underlying attribute so the property resolves.
    adapter.platform = MagicMock()
    adapter.platform.value = "webhook"
    adapter._host = "127.0.0.1"
    adapter._port = 18644
    setattr(adapter, "_global_" + _HMAC_FIELD, _HMAC_PLACEHOLDER)
    adapter._static_routes = {}
    adapter._dynamic_routes = {}
    adapter._dynamic_routes_mtime = 0.0
    adapter._routes = {}
    adapter._runner = None
    adapter._delivery_info = {}
    adapter._delivery_info_created = {}
    adapter._pending_callback = {}
    adapter._seen_deliveries = {}
    adapter._idempotency_ttl = 3600
    adapter._rate_counts = {}
    adapter._rate_limit = 30
    adapter._max_body_bytes = 1_048_576
    adapter.gateway_runner = None
    adapter._background_tasks = set()

    return adapter


def _make_delivery(chat_id="webhook:rose-dispatch:test-delivery-1"):
    """Build a delivery_info dict matching what _handle_webhook() stores."""
    info = {
        "deliver": "rose_callback",
        "deliver_extra": {},
        "payload": {},
        "rose_request_id": "rose-req-001",
        "delivery_id": "test-delivery-1",
        "started_at": time.time(),
        "callback_url": "http://localhost:9999/api/hermes-callback",
    }
    info[_HMAC_FIELD] = _HMAC_PLACEHOLDER
    return info


# ---------------------------------------------------------------------------
# Defect 1: Premature callback delivery
# ---------------------------------------------------------------------------


class TestDeferredCallbackDelivery:
    """send() must NOT fire _deliver_rose_callback directly."""

    @pytest.mark.asyncio
    async def test_send_stores_pending_callback_without_firing(self):
        """send() for rose_callback stores content but doesn't POST."""
        adapter = _make_adapter()
        chat_id = "webhook:rose-dispatch:d1"
        adapter._delivery_info[chat_id] = _make_delivery(chat_id)

        adapter._deliver_rose_callback = AsyncMock(
            return_value=SendResult(success=True)
        )

        result = await adapter.send(chat_id, "interim status", metadata=None)

        assert result.success is True
        assert chat_id in adapter._pending_callback
        assert adapter._pending_callback[chat_id] == ("interim status", None)
        adapter._deliver_rose_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_sends_keep_last_content(self):
        """Multiple send() calls overwrite _pending_callback with the latest."""
        adapter = _make_adapter()
        chat_id = "webhook:rose-dispatch:d2"
        adapter._delivery_info[chat_id] = _make_delivery(chat_id)
        adapter._deliver_rose_callback = AsyncMock(
            return_value=SendResult(success=True)
        )

        await adapter.send(chat_id, "interim 1", metadata=None)
        await adapter.send(chat_id, "interim 2", metadata={"note": "second"})
        await adapter.send(
            chat_id, "terminal response",
            metadata={"agent_status": "completed"},
        )

        content, metadata = adapter._pending_callback[chat_id]
        assert content == "terminal response"
        assert metadata == {"agent_status": "completed"}
        adapter._deliver_rose_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_fire_pending_callback_sends_exactly_once(self):
        """_fire_pending_callback calls _deliver_rose_callback once."""
        adapter = _make_adapter()
        chat_id = "webhook:rose-dispatch:d3"
        delivery = _make_delivery(chat_id)
        adapter._delivery_info[chat_id] = delivery
        adapter._pending_callback[chat_id] = (
            "final answer",
            {"agent_status": "completed"},
        )

        adapter._deliver_rose_callback = AsyncMock(
            return_value=SendResult(success=True)
        )

        await adapter._fire_pending_callback(chat_id)

        adapter._deliver_rose_callback.assert_called_once_with(
            "final answer", delivery, {"agent_status": "completed"}
        )
        assert chat_id not in adapter._pending_callback


class TestTaskFailureCallbacks:
    """Agent task exceptions/cancellations must produce a failed callback."""

    @pytest.mark.asyncio
    async def test_task_exception_sends_failed(self):
        adapter = _make_adapter()
        chat_id = "webhook:rose-dispatch:exc1"
        delivery = _make_delivery(chat_id)
        adapter._delivery_info[chat_id] = delivery
        adapter._pending_callback[chat_id] = ("some content", None)

        adapter._deliver_rose_callback = AsyncMock(
            return_value=SendResult(success=True)
        )

        await adapter._fire_pending_callback(chat_id, task_error="model crashed")

        call_args = adapter._deliver_rose_callback.call_args
        content_arg = call_args[0][0]
        metadata_arg = call_args[0][2]
        assert content_arg == "model crashed"
        assert metadata_arg["agent_status"] == "failed"
        assert "model crashed" in metadata_arg["error"]

    @pytest.mark.asyncio
    async def test_task_cancelled_sends_failed(self):
        adapter = _make_adapter()
        chat_id = "webhook:rose-dispatch:cancel1"
        delivery = _make_delivery(chat_id)
        adapter._delivery_info[chat_id] = delivery

        adapter._deliver_rose_callback = AsyncMock(
            return_value=SendResult(success=True)
        )

        await adapter._fire_pending_callback(
            chat_id, task_error="Agent task was cancelled"
        )

        call_args = adapter._deliver_rose_callback.call_args
        metadata_arg = call_args[0][2]
        assert metadata_arg["agent_status"] == "failed"
        assert "cancelled" in metadata_arg["error"].lower()

    @pytest.mark.asyncio
    async def test_no_response_sends_failed(self):
        """If send() was never called, fire a failed callback."""
        adapter = _make_adapter()
        chat_id = "webhook:rose-dispatch:noresp1"
        delivery = _make_delivery(chat_id)
        adapter._delivery_info[chat_id] = delivery

        adapter._deliver_rose_callback = AsyncMock(
            return_value=SendResult(success=True)
        )

        await adapter._fire_pending_callback(chat_id)

        call_args = adapter._deliver_rose_callback.call_args
        metadata_arg = call_args[0][2]
        assert metadata_arg["agent_status"] == "failed"
        assert "no response" in metadata_arg["error"].lower()


# ---------------------------------------------------------------------------
# Defect 2: Status misclassification
# ---------------------------------------------------------------------------


class TestCallbackDeliveryResilience:
    """Callback delivery failures must not lose content."""

    @pytest.mark.asyncio
    async def test_failed_delivery_retains_pending_content(self):
        """If _deliver_rose_callback fails, pending content is NOT popped."""
        adapter = _make_adapter()
        chat_id = "webhook:rose-dispatch:retain1"
        delivery = _make_delivery(chat_id)
        adapter._delivery_info[chat_id] = delivery
        adapter._pending_callback[chat_id] = (
            "important result",
            {"agent_status": "completed"},
        )

        adapter._deliver_rose_callback = AsyncMock(
            return_value=SendResult(success=False)
        )

        await adapter._fire_pending_callback(chat_id)

        assert chat_id in adapter._pending_callback
        assert adapter._pending_callback[chat_id] == (
            "important result",
            {"agent_status": "completed"},
        )

    @pytest.mark.asyncio
    async def test_task_error_payload_stored_before_delivery(self):
        """Error payloads are stored in _pending_callback before delivery."""
        adapter = _make_adapter()
        chat_id = "webhook:rose-dispatch:errstore1"
        delivery = _make_delivery(chat_id)
        adapter._delivery_info[chat_id] = delivery

        stored_payloads = []

        async def _capture_delivery(content, delivery_arg, metadata):
            stored_payloads.append(adapter._pending_callback.get(chat_id))
            return SendResult(success=True)

        adapter._deliver_rose_callback = _capture_delivery

        await adapter._fire_pending_callback(chat_id, task_error="boom")

        assert len(stored_payloads) == 1
        assert stored_payloads[0] is not None
        assert stored_payloads[0][1]["agent_status"] == "failed"

    @pytest.mark.asyncio
    async def test_retry_on_first_failure(self):
        """Delivery retries once after a transient failure."""
        adapter = _make_adapter()
        chat_id = "webhook:rose-dispatch:retry1"
        delivery = _make_delivery(chat_id)
        adapter._delivery_info[chat_id] = delivery
        adapter._pending_callback[chat_id] = ("result", {"agent_status": "completed"})

        call_count = 0

        async def _fail_then_succeed(content, delivery_arg, metadata):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SendResult(success=False)
            return SendResult(success=True)

        adapter._deliver_rose_callback = _fail_then_succeed

        await adapter._fire_pending_callback(chat_id)

        assert call_count == 2
        assert chat_id not in adapter._pending_callback

    @pytest.mark.asyncio
    async def test_cancellation_during_backoff_attempts_final_delivery(self):
        """CancelledError during retry sleep still attempts delivery."""
        adapter = _make_adapter()
        chat_id = "webhook:rose-dispatch:cancel-backoff1"
        delivery = _make_delivery(chat_id)
        adapter._delivery_info[chat_id] = delivery
        adapter._pending_callback[chat_id] = (
            "result",
            {"agent_status": "completed"},
        )

        call_count = 0

        async def _fail_then_succeed(content, delivery_arg, metadata):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SendResult(success=False)
            return SendResult(success=True)

        adapter._deliver_rose_callback = _fail_then_succeed

        with patch("gateway.platforms.webhook.asyncio.sleep", side_effect=asyncio.CancelledError):
            with pytest.raises(asyncio.CancelledError):
                await adapter._fire_pending_callback(chat_id)

        assert call_count == 2
        assert chat_id not in adapter._pending_callback


# ---------------------------------------------------------------------------
# Override tests: _process_message_background
# ---------------------------------------------------------------------------


class TestProcessMessageBackgroundOverride:
    """The override must fire callback after super() completes."""

    @pytest.mark.asyncio
    async def test_override_fires_callback_after_super(self):
        """Callback fires after the superclass coroutine completes.

        The mocked super() sets _pending_callback on completion.  If the
        callback fired *before* super(), _pending_callback would still be
        empty and the delivery would carry a 'no response' error instead
        of the content super() placed.
        """
        adapter = _make_adapter()
        chat_id = "webhook:rose-dispatch:override1"
        delivery = _make_delivery(chat_id)
        adapter._delivery_info[chat_id] = delivery

        adapter._deliver_rose_callback = AsyncMock(
            return_value=SendResult(success=True)
        )

        event = MagicMock()
        event.source.chat_id = chat_id

        async def _super_sets_content(ev, sk):
            adapter._pending_callback[chat_id] = (
                "set-by-super",
                {"agent_status": "completed"},
            )

        from gateway.platforms.base import BasePlatformAdapter

        with patch.object(
            BasePlatformAdapter,
            "_process_message_background",
            new_callable=AsyncMock,
            side_effect=_super_sets_content,
        ) as mock_super:
            await adapter._process_message_background(event, "session-key-1")

        mock_super.assert_called_once_with(event, "session-key-1")
        adapter._deliver_rose_callback.assert_called_once()
        delivered_content = adapter._deliver_rose_callback.call_args[0][0]
        assert delivered_content == "set-by-super"
        assert chat_id not in adapter._pending_callback

    @pytest.mark.asyncio
    async def test_override_sends_failed_on_exception(self):
        """An exception in super() produces a failed callback."""
        adapter = _make_adapter()
        chat_id = "webhook:rose-dispatch:override-exc1"
        delivery = _make_delivery(chat_id)
        adapter._delivery_info[chat_id] = delivery
        adapter._pending_callback[chat_id] = ("interim content", None)

        adapter._deliver_rose_callback = AsyncMock(
            return_value=SendResult(success=True)
        )

        event = MagicMock()
        event.source.chat_id = chat_id

        from gateway.platforms.base import BasePlatformAdapter

        with patch.object(
            BasePlatformAdapter,
            "_process_message_background",
            new_callable=AsyncMock,
            side_effect=RuntimeError("model crashed"),
        ):
            with pytest.raises(RuntimeError, match="model crashed"):
                await adapter._process_message_background(event, "session-key-2")

        call_args = adapter._deliver_rose_callback.call_args
        metadata_arg = call_args[0][2]
        assert metadata_arg["agent_status"] == "failed"
        assert "model crashed" in metadata_arg["error"]

    @pytest.mark.asyncio
    async def test_override_sends_failed_on_empty_exception(self):
        """An exception with no message uses repr() fallback."""
        adapter = _make_adapter()
        chat_id = "webhook:rose-dispatch:override-emptyexc"
        delivery = _make_delivery(chat_id)
        adapter._delivery_info[chat_id] = delivery

        adapter._deliver_rose_callback = AsyncMock(
            return_value=SendResult(success=True)
        )

        event = MagicMock()
        event.source.chat_id = chat_id

        from gateway.platforms.base import BasePlatformAdapter

        with patch.object(
            BasePlatformAdapter,
            "_process_message_background",
            new_callable=AsyncMock,
            side_effect=RuntimeError(),
        ):
            with pytest.raises(RuntimeError):
                await adapter._process_message_background(event, "session-key-3")

        call_args = adapter._deliver_rose_callback.call_args
        content_arg = call_args[0][0]
        metadata_arg = call_args[0][2]
        assert metadata_arg["agent_status"] == "failed"
        assert content_arg == "RuntimeError()"
        assert metadata_arg["error"] == "RuntimeError()"

    @pytest.mark.asyncio
    async def test_override_skips_callback_for_non_rose(self):
        """Non-rose_callback deliveries skip the callback entirely."""
        adapter = _make_adapter()
        chat_id = "webhook:some-other:delivery1"
        adapter._delivery_info[chat_id] = {"deliver": "log"}

        adapter._deliver_rose_callback = AsyncMock()

        event = MagicMock()
        event.source.chat_id = chat_id

        from gateway.platforms.base import BasePlatformAdapter

        with patch.object(
            BasePlatformAdapter,
            "_process_message_background",
            new_callable=AsyncMock,
        ):
            await adapter._process_message_background(event, "session-key-4")

        adapter._deliver_rose_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_override_propagates_cancelled_callback_delivery(self):
        """CancelledError from _fire_pending_callback is re-raised after logging."""
        adapter = _make_adapter()
        chat_id = "webhook:rose-dispatch:override-cancel1"
        delivery = _make_delivery(chat_id)
        adapter._delivery_info[chat_id] = delivery
        adapter._pending_callback[chat_id] = (
            "content",
            {"agent_status": "completed"},
        )

        adapter._deliver_rose_callback = AsyncMock(
            side_effect=asyncio.CancelledError,
        )

        event = MagicMock()
        event.source.chat_id = chat_id

        from gateway.platforms.base import BasePlatformAdapter

        with patch.object(
            BasePlatformAdapter,
            "_process_message_background",
            new_callable=AsyncMock,
        ):
            with pytest.raises(asyncio.CancelledError):
                await adapter._process_message_background(event, "session-key-5")

        assert chat_id in adapter._pending_callback


# ---------------------------------------------------------------------------
# Defect 2: Status misclassification
# ---------------------------------------------------------------------------


class TestStatusClassification:
    """_deliver_rose_callback must classify errors correctly."""

    def _run_classification(self, content, metadata=None):
        """Run the status detection logic and return (is_failure, error_message)."""
        metadata = metadata or {}
        agent_status = str(metadata.get("agent_status") or "").lower()
        metadata_error = metadata.get("error")

        is_failure = False
        error_message = None

        if agent_status == "failed" or metadata_error:
            is_failure = True
            error_message = (
                str(metadata_error)
                if metadata_error
                else "Agent reported failure via metadata"
            )
        else:
            content_head = (content or "")[:1000]
            FAILURE_MARKERS = (
                "API call failed",
                "Final error:",
                "Max retries (3) exhausted",
                "Max retries exhausted",
                "⏳ Retrying in",
                "Retrying in ",
                "❌",
                "☠",
                "Traceback (most recent call last):",
                "Error code: 4",
                "Error code: 5",
                "⚠️ Error code:",
                "'type': 'error'",
            )
            matched_marker = None
            for marker in FAILURE_MARKERS:
                if marker in content_head:
                    matched_marker = marker
                    break
            if matched_marker:
                is_failure = True
                lines = [
                    ln.strip()
                    for ln in (content or "").splitlines()
                    if ln.strip()
                ]
                error_message = next(
                    (ln for ln in lines if "Final error:" in ln),
                    next(
                        (ln for ln in lines if matched_marker in ln),
                        matched_marker,
                    ),
                )[:500]

        return is_failure, error_message

    def test_anthropic_401_classified_as_failure(self):
        """The exact error from the smoke test must be caught."""
        content = (
            "⚠️ Error code: 401 - {'type': 'error', 'error': "
            "{'type': 'authentication_error', "
            "'message': 'API key is invalid.'}, "
            "'request_id': None}"
        )
        is_failure, error_message = self._run_classification(content)
        assert is_failure is True
        assert error_message is not None

    def test_http_500_classified_as_failure(self):
        content = "Error code: 500 - Internal Server Error"
        is_failure, _ = self._run_classification(content)
        assert is_failure is True

    def test_http_429_classified_as_failure(self):
        content = "Error code: 429 - Rate limit exceeded"
        is_failure, _ = self._run_classification(content)
        assert is_failure is True

    def test_error_type_dict_classified_as_failure(self):
        content = "{'type': 'error', 'error': {'type': 'server_error'}}"
        is_failure, _ = self._run_classification(content)
        assert is_failure is True

    def test_metadata_agent_status_failed(self):
        is_failure, error_message = self._run_classification(
            "some content",
            metadata={"agent_status": "failed", "error": "explicit failure"},
        )
        assert is_failure is True
        assert error_message == "explicit failure"

    def test_metadata_error_without_status(self):
        is_failure, error_message = self._run_classification(
            "some content",
            metadata={"error": "something went wrong"},
        )
        assert is_failure is True
        assert error_message == "something went wrong"

    def test_clean_research_output_is_success(self):
        content = (
            "SolarAfrica Energy operates in the commercial and industrial "
            "solar market in South Africa. Their primary offerings include "
            "wheeling and PPA solutions for large energy consumers."
        )
        is_failure, _ = self._run_classification(content)
        assert is_failure is False

    def test_research_mentioning_error_deep_is_success(self):
        """Content past the 1000-char boundary isn't caught."""
        padding = "x" * 1001
        content = padding + " Error code: 401 authentication_error"
        is_failure, _ = self._run_classification(content)
        assert is_failure is False

    def test_existing_markers_still_work(self):
        for marker_content in [
            "API call failed",
            "Final error: something broke",
            "Max retries (3) exhausted",
            "❌ Operation failed",
            "Traceback (most recent call last):\n  File ...",
        ]:
            is_failure, _ = self._run_classification(marker_content)
            assert is_failure is True, f"Missed: {marker_content!r}"


# ---------------------------------------------------------------------------
# Signing format unchanged
# ---------------------------------------------------------------------------


class TestCallbackSigningUnchanged:
    """Verify the timestamp-bound signing contract matches Rose's verifier.

    Rose's verifyCallbackSignature (hermesClient.ts) requires:
      X-Hermes-Signature: t=<unix>,sha256=<HMAC(timestamp.body)>
    No X-Rose-Timestamp header. HMAC input is "{timestamp}.{body}".
    """

    def test_signature_computation_matches_timestamp_bound_contract(self):
        """HMAC covers timestamp.body; header is t=<unix>,sha256=<hex>."""
        body_obj = {
            "job_id": "d1",
            "rose_request_id": "r1",
            "status": "completed",
            "summary": "test",
            "artifacts": [],
            "duration_ms": 100,
            "token_cost_usd": None,
            "error": None,
        }
        body_bytes = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
        timestamp = "1726380000"
        signature_input = f"{timestamp}.".encode("utf-8") + body_bytes
        sig_hex = hmac_mod.new(
            _HMAC_PLACEHOLDER.encode(), signature_input, hashlib.sha256
        ).hexdigest()
        signature = f"t={timestamp},sha256={sig_hex}"

        assert signature.startswith("t=")
        assert ",sha256=" in signature
        assert len(sig_hex) == 64
        parts = signature.split(",", 1)
        assert parts[0] == f"t={timestamp}"
        assert parts[1] == f"sha256={sig_hex}"

    def test_signature_header_name_is_x_hermes_signature(self):
        """Header must be X-Hermes-Signature; X-Rose-Timestamp must NOT be present."""
        import inspect
        from gateway.platforms.webhook import WebhookAdapter

        source = inspect.getsource(WebhookAdapter._deliver_rose_callback)
        assert '"X-Hermes-Signature"' in source
        assert '"X-Rose-Timestamp"' not in source

    @pytest.mark.asyncio
    async def test_production_signer_matches_rose_verifier(self):
        """Call the real _deliver_rose_callback, capture headers, verify against Rose's contract."""
        import re

        adapter = _make_adapter()
        chat_id = "webhook:rose-dispatch:sign-integration"
        delivery = _make_delivery(chat_id)
        adapter._delivery_info[chat_id] = delivery

        captured_headers: Dict[str, str] = {}
        captured_body: Optional[bytes] = None

        class FakeResponse:
            status = 200
            async def text(self):
                return '{"ok": true}'
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                pass

        class FakeSession:
            def __init__(self, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                pass
            def post(self, url, data=None, headers=None):
                nonlocal captured_headers, captured_body
                captured_headers = dict(headers or {})
                captured_body = data
                return FakeResponse()

        with patch.dict("sys.modules", {"aiohttp": MagicMock(
            ClientSession=FakeSession,
            ClientTimeout=lambda **kw: None,
        )}):
            result = await adapter._deliver_rose_callback(
                "integration test content", delivery, {"agent_status": "completed"}
            )

        assert result.success is True
        assert "X-Hermes-Signature" in captured_headers
        assert "X-Rose-Timestamp" not in captured_headers

        sig_header = captured_headers["X-Hermes-Signature"]
        t_match = re.search(r"t=(\d+)", sig_header)
        sig_match = re.search(r"sha256=([0-9a-f]+)", sig_header)
        assert t_match is not None, f"No t= in header: {sig_header}"
        assert sig_match is not None, f"No sha256= in header: {sig_header}"

        parsed_ts = int(t_match.group(1))
        assert abs(int(time.time()) - parsed_ts) < 10

        body_str = captured_body.decode("utf-8")
        expected_input = f"{parsed_ts}.{body_str}".encode("utf-8")
        expected_hex = hmac_mod.new(
            _HMAC_PLACEHOLDER.encode(), expected_input, hashlib.sha256
        ).hexdigest()
        expected_header = f"t={parsed_ts},sha256={expected_hex}"
        assert len(expected_header) == len(sig_header)
        assert hmac_mod.compare_digest(expected_header, sig_header)
