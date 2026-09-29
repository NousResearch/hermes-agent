"""Tests for POST /api/bot-chat/send — the internal topic-subscription -> Bot Chat delivery
endpoint used by callers like a Windmill agent_mail fan-out flow (e.g. prime -> dev-accounts).

Covers:
- Auth enforcement (401 with no/invalid Bearer token, same API_SERVER_KEY as every other route)
- Input validation (missing/oversized content, missing/oversized dedupe_key, invalid topic/source)
- Cron module unavailability (501 when _CRON_AVAILABLE is False)
- Status mapping (settled/suppressed -> 200, queued/claimed -> 202, error -> 502)
- dedupe_key becomes the delivery's idempotency key (job execution_id) end-to-end
- Never logs raw content — only sanitized metadata (topic/source/status/delivery_id)
"""

import logging

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from unittest.mock import patch

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware

_MOD = "gateway.platforms.api_server"
_DELIVERY_MOD = "cron.scheduler_delivery"

SECRET_CONTENT_MARKER = "sk-super-secret-raw-mail-body-should-never-be-logged"


def _make_adapter(api_key: str = "sk-secret") -> APIServerAdapter:
    config = PlatformConfig(enabled=True, extra={"key": api_key} if api_key else {})
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/bot-chat/send", adapter._handle_bot_chat_send)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


def _settled_delivery(status="settled", delivery_id="deadbeef" * 4, error=None):
    """A fake ``_deliver_to_bot_chat`` that mimics the real receipt-recording contract."""

    def fake(job, content, profile):
        job.setdefault("_bot_chat_delivery_receipts", {})["bot-chat:(own)"] = {
            "status": status, "delivery_id": delivery_id,
        }
        return None if status in ("settled", "suppressed") else (error or f"{status}: not settled")

    return fake


class TestAuth:
    @pytest.mark.asyncio
    async def test_rejects_unauthenticated(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True):
                resp = await cli.post("/api/bot-chat/send", json={"content": "hi", "dedupe_key": "k1"})
                assert resp.status == 401

    @pytest.mark.asyncio
    async def test_rejects_wrong_key(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer wrong-key"},
                    json={"content": "hi", "dedupe_key": "k1"})
                assert resp.status == 401

    @pytest.mark.asyncio
    async def test_valid_key_admitted(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), \
                 patch(f"{_DELIVERY_MOD}._deliver_to_bot_chat", side_effect=_settled_delivery()):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer sk-secret"},
                    json={"content": "hi", "dedupe_key": "k1"})
                assert resp.status == 200


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_content_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer sk-secret"},
                    json={"dedupe_key": "k1"})
                assert resp.status == 400

    @pytest.mark.asyncio
    async def test_blank_content_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer sk-secret"},
                    json={"content": "   ", "dedupe_key": "k1"})
                assert resp.status == 400

    @pytest.mark.asyncio
    async def test_content_too_long_400(self, adapter):
        from gateway.platforms.api_server import MAX_BOT_CHAT_SEND_CONTENT_LENGTH
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer sk-secret"},
                    json={"content": "x" * (MAX_BOT_CHAT_SEND_CONTENT_LENGTH + 1), "dedupe_key": "k1"})
                assert resp.status == 400

    @pytest.mark.asyncio
    async def test_missing_dedupe_key_400(self, adapter):
        """dedupe_key must be required, never inferred, so two distinct messages that render
        identically can never be silently collapsed into one delivery."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer sk-secret"},
                    json={"content": "hi"})
                assert resp.status == 400

    @pytest.mark.asyncio
    async def test_dedupe_key_too_long_400(self, adapter):
        from gateway.platforms.api_server import MAX_BOT_CHAT_SEND_DEDUPE_KEY_LENGTH
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer sk-secret"},
                    json={"content": "hi", "dedupe_key": "k" * (MAX_BOT_CHAT_SEND_DEDUPE_KEY_LENGTH + 1)})
                assert resp.status == 400

    @pytest.mark.asyncio
    async def test_invalid_topic_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer sk-secret"},
                    json={"content": "hi", "dedupe_key": "k1", "topic": "not valid! topic"})
                assert resp.status == 400

    @pytest.mark.asyncio
    async def test_invalid_json_body_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer sk-secret", "Content-Type": "application/json"},
                    data=b"not json")
                assert resp.status == 400

    @pytest.mark.asyncio
    async def test_non_object_body_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer sk-secret"},
                    json=["not", "an", "object"])
                assert resp.status == 400


class TestCronUnavailable:
    @pytest.mark.asyncio
    async def test_cron_unavailable_501(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", False):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer sk-secret"},
                    json={"content": "hi", "dedupe_key": "k1"})
                assert resp.status == 501


class TestStatusMapping:
    @pytest.mark.asyncio
    async def test_settled_returns_200(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), \
                 patch(f"{_DELIVERY_MOD}._deliver_to_bot_chat",
                       side_effect=_settled_delivery(status="settled", delivery_id="abc123")):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer sk-secret"},
                    json={"content": "hi", "dedupe_key": "k1", "topic": "dev-accounts", "source": "agent_mail"})
                assert resp.status == 200
                data = await resp.json()
                assert data == {"status": "settled", "delivery_id": "abc123"}

    @pytest.mark.asyncio
    async def test_suppressed_returns_200(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), \
                 patch(f"{_DELIVERY_MOD}._deliver_to_bot_chat",
                       side_effect=_settled_delivery(status="suppressed", delivery_id="abc123")):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer sk-secret"},
                    json={"content": "hi", "dedupe_key": "k1"})
                assert resp.status == 200
                assert (await resp.json())["status"] == "suppressed"

    @pytest.mark.asyncio
    async def test_queued_returns_202(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), \
                 patch(f"{_DELIVERY_MOD}._deliver_to_bot_chat",
                       side_effect=_settled_delivery(status="queued", delivery_id="q1",
                                                      error="bot-chat:(own) queued (receipt q1): completion unverified; do not resend")):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer sk-secret"},
                    json={"content": "hi", "dedupe_key": "k1"})
                assert resp.status == 202
                assert (await resp.json())["status"] == "queued"

    @pytest.mark.asyncio
    async def test_error_returns_502(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), \
                 patch(f"{_DELIVERY_MOD}._deliver_to_bot_chat",
                       side_effect=_settled_delivery(status="ambiguous", delivery_id="e1", error="boom")):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer sk-secret"},
                    json={"content": "hi", "dedupe_key": "k1"})
                assert resp.status == 502

    @pytest.mark.asyncio
    async def test_raising_delivery_returns_502_not_500(self, adapter):
        """Discovery/admission uncertainty inside the delivery path must never surface as a
        500 or, worse, retry into a second-writer fallback — same fail-closed contract as the
        underlying ``_deliver_to_bot_chat``."""
        def boom(job, content, profile):
            raise RuntimeError("unexpected")

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), \
                 patch(f"{_DELIVERY_MOD}._deliver_to_bot_chat", side_effect=boom):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer sk-secret"},
                    json={"content": "hi", "dedupe_key": "k1"})
                assert resp.status == 502


class TestIdempotencyWiring:
    @pytest.mark.asyncio
    async def test_dedupe_key_becomes_job_execution_id(self, adapter):
        """dedupe_key must reach ``_deliver_to_bot_chat`` as the job's execution_id — the exact
        field its hash-based idempotency key is derived from — so a resend of the same
        dedupe_key can never create a second Bot Chat turn."""
        seen = {}

        def fake(job, content, profile):
            seen["execution_id"] = job.get("execution_id")
            seen["content"] = content
            seen["profile"] = profile
            job.setdefault("_bot_chat_delivery_receipts", {})["bot-chat:(own)"] = {
                "status": "settled", "delivery_id": "d1"}
            return None

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(f"{_MOD}._CRON_AVAILABLE", True), \
                 patch(f"{_DELIVERY_MOD}._deliver_to_bot_chat", side_effect=fake):
                resp = await cli.post(
                    "/api/bot-chat/send",
                    headers={"Authorization": "Bearer sk-secret"},
                    json={"content": "hello", "dedupe_key": "msg-id-<abc@example.com>"})
                assert resp.status == 200
        assert seen["execution_id"] == "msg-id-<abc@example.com>"
        assert seen["content"] == "hello"
        # profile="" delivers to THIS request's own (multiplex-scoped) profile.
        assert seen["profile"] == ""


class TestSanitizedLogging:
    @pytest.mark.asyncio
    async def test_content_never_logged(self, adapter, caplog):
        """The endpoint must record sanitized metadata only — never the message content —
        even on success, so a mail-derived digest containing anything sensitive never lands
        in the gateway's own logs."""
        def fake(job, content, profile):
            job.setdefault("_bot_chat_delivery_receipts", {})["bot-chat:(own)"] = {
                "status": "settled", "delivery_id": "d1"}
            return None

        app = _create_app(adapter)
        with caplog.at_level(logging.DEBUG):
            async with TestClient(TestServer(app)) as cli:
                with patch(f"{_MOD}._CRON_AVAILABLE", True), \
                     patch(f"{_DELIVERY_MOD}._deliver_to_bot_chat", side_effect=fake):
                    resp = await cli.post(
                        "/api/bot-chat/send",
                        headers={"Authorization": "Bearer sk-secret"},
                        json={"content": SECRET_CONTENT_MARKER, "dedupe_key": "k1",
                              "topic": "dev-accounts", "source": "agent_mail"})
                    assert resp.status == 200
        for record in caplog.records:
            assert SECRET_CONTENT_MARKER not in record.getMessage()

    @pytest.mark.asyncio
    async def test_content_never_logged_on_error(self, adapter, caplog):
        def boom(job, content, profile):
            raise RuntimeError(f"failure near content {content[:5]}...")  # never do this upstream

        app = _create_app(adapter)
        with caplog.at_level(logging.DEBUG):
            async with TestClient(TestServer(app)) as cli:
                with patch(f"{_MOD}._CRON_AVAILABLE", True), \
                     patch(f"{_DELIVERY_MOD}._deliver_to_bot_chat", side_effect=boom):
                    resp = await cli.post(
                        "/api/bot-chat/send",
                        headers={"Authorization": "Bearer sk-secret"},
                        json={"content": SECRET_CONTENT_MARKER, "dedupe_key": "k1"})
                    assert resp.status == 502
        for record in caplog.records:
            assert SECRET_CONTENT_MARKER not in record.getMessage()
