"""Tests for bounded WhatsApp bridge delivery attempts.

Task 2 of the delivery-reliability plan:
- max 3 attempts, backoff schedule [1, 5] with injectable sleep, jitter off in tests
- one Idempotency-Key per logical delivery, reused across attempts
- auth headers preserved on every attempt
- 2xx ends attempts; permanent or ambiguous failure stops immediately
- optional policy hook can veto a delivery before the first attempt
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from plugins.platforms.whatsapp import delivery_ledger
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
from plugins.platforms.whatsapp.delivery_reliability import (
    AMBIGUOUS,
    NON_RETRYABLE,
    retry_backoff_delay,
    send_with_retries,
    set_delivery_policy_hook,
)

TOKEN = "test-bridge-token-123"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(autouse=True)
def _clear_policy_hook():
    """Each test starts and ends with no delivery policy hook registered."""
    set_delivery_policy_hook(None)
    yield
    set_delivery_policy_hook(None)


# ---------------------------------------------------------------------------
# Backoff schedule
# ---------------------------------------------------------------------------

class TestRetryBackoffDelay:
    def test_schedule_without_jitter(self):
        assert retry_backoff_delay(2, jitter=False) == 1.0
        assert retry_backoff_delay(3, jitter=False) == 5.0

    def test_jitter_bounds(self):
        delays = [retry_backoff_delay(2, jitter=True) for _ in range(50)]
        assert all(1.0 <= d <= 1.5 for d in delays)


# ---------------------------------------------------------------------------
# send_with_retries loop
# ---------------------------------------------------------------------------

def _attempt_fn(script):
    """Async attempt callable fed by a list of (status, payload) or exceptions.

    Records the headers of every attempt in ``calls``.
    """
    calls = []

    async def attempt(headers):
        calls.append(dict(headers))
        step = script[min(len(calls) - 1, len(script) - 1)]
        if isinstance(step, BaseException):
            raise step
        return step

    attempt.calls = calls
    return attempt


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


class TestSendWithRetries:
    def test_success_first_attempt(self):
        attempt = _attempt_fn([(200, {"messageId": "m1"})])
        sleep = AsyncMock()
        outcome = _run(send_with_retries(attempt, base_headers=AUTH, sleep=sleep, jitter=False))
        assert outcome.ok
        assert outcome.attempts == 1
        assert outcome.data == {"messageId": "m1"}
        sleep.assert_not_awaited()

    def test_503_exhausts_three_attempts_with_backoff(self):
        attempt = _attempt_fn([(503, "unavailable")])
        sleep = AsyncMock()
        outcome = _run(send_with_retries(attempt, base_headers=AUTH, sleep=sleep, jitter=False))
        assert not outcome.ok
        assert outcome.attempts == 3
        assert outcome.failure.category == "http_503"
        assert [c.args[0] for c in sleep.await_args_list] == [1.0, 5.0]

    def test_connection_refused_is_bounded(self):
        attempt = _attempt_fn([ConnectionRefusedError()])
        outcome = _run(send_with_retries(attempt, sleep=AsyncMock(), jitter=False))
        assert not outcome.ok
        assert outcome.attempts == 3
        assert outcome.failure.category == "connection_refused"

    def test_429_then_success(self):
        attempt = _attempt_fn([(429, "slow down"), (200, {"messageId": "m2"})])
        outcome = _run(send_with_retries(attempt, sleep=AsyncMock(), jitter=False))
        assert outcome.ok
        assert outcome.attempts == 2

    def test_timeout_is_ambiguous_and_never_retried(self):
        attempt = _attempt_fn([asyncio.TimeoutError()])
        sleep = AsyncMock()
        outcome = _run(send_with_retries(attempt, sleep=sleep, jitter=False))
        assert not outcome.ok
        assert outcome.attempts == 1
        assert outcome.failure.decision == AMBIGUOUS
        sleep.assert_not_awaited()

    @pytest.mark.parametrize("status", [400, 401])
    def test_permanent_status_stops_immediately(self, status):
        attempt = _attempt_fn([(status, "denied")])
        sleep = AsyncMock()
        outcome = _run(send_with_retries(attempt, sleep=sleep, jitter=False))
        assert not outcome.ok
        assert outcome.attempts == 1
        assert outcome.failure.decision == NON_RETRYABLE
        sleep.assert_not_awaited()

    def test_idempotency_key_reused_across_attempts(self):
        attempt = _attempt_fn([(503, "unavailable")])
        outcome = _run(send_with_retries(attempt, base_headers=AUTH, sleep=AsyncMock(), jitter=False))
        keys = {c["Idempotency-Key"] for c in attempt.calls}
        assert len(attempt.calls) == 3
        assert keys == {outcome.idempotency_key}
        assert outcome.idempotency_key

    def test_distinct_logical_deliveries_get_distinct_keys(self):
        first = _run(send_with_retries(_attempt_fn([(200, {})]), jitter=False))
        second = _run(send_with_retries(_attempt_fn([(200, {})]), jitter=False))
        assert first.idempotency_key != second.idempotency_key

    def test_auth_headers_preserved_on_every_attempt(self):
        attempt = _attempt_fn([(503, "unavailable")])
        _run(send_with_retries(attempt, base_headers=AUTH, sleep=AsyncMock(), jitter=False))
        for headers in attempt.calls:
            assert headers["Authorization"] == AUTH["Authorization"]


# ---------------------------------------------------------------------------
# Policy hook (profile-owned outreach policy, e.g. Sawi DDD19/30-day rules)
# ---------------------------------------------------------------------------

class TestPolicyHook:
    def test_veto_blocks_before_first_attempt(self):
        seen = {}

        def hook(context):
            seen.update(context)
            return False

        set_delivery_policy_hook(hook)
        attempt = _attempt_fn([(200, {})])
        outcome = _run(send_with_retries(
            attempt, policy_context={"platform": "whatsapp", "route": "send"},
        ))
        assert not outcome.ok
        assert outcome.attempts == 0
        assert outcome.failure.category == "policy_blocked"
        assert attempt.calls == []
        assert seen["route"] == "send"

    def test_hook_approval_allows_delivery(self):
        set_delivery_policy_hook(lambda context: True)
        outcome = _run(send_with_retries(_attempt_fn([(200, {})]), jitter=False))
        assert outcome.ok

    def test_hook_exception_fails_closed(self):
        def hook(context):
            raise RuntimeError("profile hook bug")

        set_delivery_policy_hook(hook)
        attempt = _attempt_fn([(200, {})])
        outcome = _run(send_with_retries(attempt))
        assert not outcome.ok
        assert outcome.failure.category == "policy_blocked"
        assert attempt.calls == []


# ---------------------------------------------------------------------------
# Dead-letter ledger integration (Task 3)
# ---------------------------------------------------------------------------

class TestSendWithRetriesDeadLetter:
    def test_exhausted_retries_record_dead_letter_when_ledger_enabled(self, tmp_path, monkeypatch):
        path = tmp_path / "ledger.jsonl"
        monkeypatch.setenv(delivery_ledger._LEDGER_ENABLED_ENV, "1")
        monkeypatch.setenv(delivery_ledger._LEDGER_PATH_ENV, str(path))

        attempt = _attempt_fn([(503, "unavailable")])
        outcome = _run(send_with_retries(
            attempt, base_headers=AUTH, sleep=AsyncMock(), jitter=False,
            platform="whatsapp", route="/send",
        ))

        assert outcome.dead_letter_ref
        entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(entries) == 1
        assert entries[0]["platform"] == "whatsapp"
        assert entries[0]["route"] == "/send"
        assert entries[0]["attempts"] == 3
        assert entries[0]["category"] == "http_503"

    def test_ambiguous_timeout_records_dead_letter(self, tmp_path, monkeypatch):
        path = tmp_path / "ledger.jsonl"
        monkeypatch.setenv(delivery_ledger._LEDGER_ENABLED_ENV, "1")
        monkeypatch.setenv(delivery_ledger._LEDGER_PATH_ENV, str(path))

        attempt = _attempt_fn([asyncio.TimeoutError()])
        outcome = _run(send_with_retries(
            attempt, sleep=AsyncMock(), jitter=False, platform="whatsapp", route="/send",
        ))

        assert outcome.dead_letter_ref
        entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert entries[0]["category"] == "timeout"
        assert entries[0]["attempts"] == 1

    def test_success_never_records_dead_letter(self, tmp_path, monkeypatch):
        path = tmp_path / "ledger.jsonl"
        monkeypatch.setenv(delivery_ledger._LEDGER_ENABLED_ENV, "1")
        monkeypatch.setenv(delivery_ledger._LEDGER_PATH_ENV, str(path))

        attempt = _attempt_fn([(429, "slow down"), (200, {"messageId": "m2"})])
        outcome = _run(send_with_retries(
            attempt, sleep=AsyncMock(), jitter=False, platform="whatsapp", route="/send",
        ))

        assert outcome.ok
        assert outcome.dead_letter_ref is None
        assert not path.exists()

    def test_no_dead_letter_ref_when_ledger_disabled(self):
        attempt = _attempt_fn([(503, "unavailable")])
        outcome = _run(send_with_retries(
            attempt, sleep=AsyncMock(), jitter=False, platform="whatsapp", route="/send",
        ))
        assert not outcome.ok
        assert outcome.dead_letter_ref is None

    def test_policy_veto_never_records_dead_letter(self, tmp_path, monkeypatch):
        path = tmp_path / "ledger.jsonl"
        monkeypatch.setenv(delivery_ledger._LEDGER_ENABLED_ENV, "1")
        monkeypatch.setenv(delivery_ledger._LEDGER_PATH_ENV, str(path))
        set_delivery_policy_hook(lambda context: False)

        attempt = _attempt_fn([(200, {})])
        outcome = _run(send_with_retries(
            attempt, platform="whatsapp", route="/send",
        ))

        assert outcome.dead_letter_ref is None
        assert not path.exists()


# ---------------------------------------------------------------------------
# Adapter integration: mutating bridge POSTs go through the bounded helper
# ---------------------------------------------------------------------------

def _make_adapter():
    """Create a WhatsAppAdapter with test attributes (bypass __init__)."""
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter._bridge_port = 3000
    adapter._session_path = MagicMock()
    adapter._bridge_log_fh = None
    adapter._bridge_log = None
    adapter._bridge_process = None
    adapter._reply_prefix = None
    adapter._running = True
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._background_tasks = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._message_queue = asyncio.Queue()
    adapter._http_session = MagicMock()
    adapter._mention_patterns = []
    adapter._dm_policy = "open"
    adapter._allow_from = set()
    adapter._group_policy = "open"
    adapter._group_allow_from = set()
    adapter._bridge_token = TOKEN
    adapter._retry_sleep = AsyncMock()
    adapter._retry_jitter = False
    return adapter


class _AsyncCM:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _response(status=200, payload=None, text=""):
    resp = MagicMock(status=status)
    resp.json = AsyncMock(return_value=payload or {"messageId": "msg1"})
    resp.text = AsyncMock(return_value=text)
    return resp


class TestAdapterBoundedDelivery:
    @pytest.mark.asyncio
    async def test_send_retries_503_three_times_with_stable_key(self):
        adapter = _make_adapter()
        adapter._http_session.post = MagicMock(
            return_value=_AsyncCM(_response(status=503, text="unavailable"))
        )

        result = await adapter.send("chat1", "hello")
        assert not result.success
        calls = adapter._http_session.post.call_args_list
        assert len(calls) == 3
        keys = {c.kwargs["headers"]["Idempotency-Key"] for c in calls}
        assert len(keys) == 1
        for c in calls:
            assert c.kwargs["headers"]["Authorization"] == AUTH["Authorization"]

    @pytest.mark.asyncio
    async def test_send_timeout_never_retries(self):
        adapter = _make_adapter()
        adapter._http_session.post = MagicMock(side_effect=asyncio.TimeoutError())

        result = await adapter.send("chat1", "hello")
        assert not result.success
        assert len(adapter._http_session.post.call_args_list) == 1

    @pytest.mark.asyncio
    async def test_send_401_never_retries(self):
        adapter = _make_adapter()
        adapter._http_session.post = MagicMock(
            return_value=_AsyncCM(_response(status=401, text="unauthorized"))
        )

        result = await adapter.send("chat1", "hello")
        assert not result.success
        assert len(adapter._http_session.post.call_args_list) == 1

    @pytest.mark.asyncio
    async def test_send_success_carries_idempotency_key(self):
        adapter = _make_adapter()
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(_response()))

        result = await adapter.send("chat1", "hello")
        assert result.success
        headers = adapter._http_session.post.call_args.kwargs["headers"]
        assert headers.get("Idempotency-Key")

    @pytest.mark.asyncio
    async def test_media_send_retries_connection_refused(self, tmp_path):
        adapter = _make_adapter()
        adapter._http_session.post = MagicMock(side_effect=ConnectionRefusedError())
        media = tmp_path / "pic.png"
        media.write_bytes(b"png")

        result = await adapter._send_media_to_bridge("chat1", str(media), "image")
        assert not result.success
        assert len(adapter._http_session.post.call_args_list) == 3
