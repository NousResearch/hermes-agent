"""Integration tests for the ActivePieces platform adapter.

Exercises the AP-specific edges over the inherited webhook pipeline:
1. Bearer-authenticated AP-shaped POST → agent MessageEvent (right platform,
   right session prefix, skills lane attached)
2. Wrong / missing bearer → 401; secretless flow fails closed
3. ``deliver: activepieces`` POSTs the agent reply {delivery_id, flow, reply}
   to the flow's reply URL
4. Flow validation: deliver=activepieces without an http(s) URL is rejected
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.platforms.activepieces import ActivePiecesAdapter

AP_SECRET = "ap-flow-secret"
#: The envelope AP flows POST (contract shared with cold-start/Twilio consumers).
AP_PAYLOAD = {
    "source": "activepieces",
    "piece": "gmail",
    "trigger": "new_email_received",
    "project_id": 42,
    "event": {
        "subject": "Invoice attached",
        "from": "billing@example.com",
        "body": "See the attached invoice for September.",
    },
}


def _make_adapter(flows, **extra_kw) -> ActivePiecesAdapter:
    extra = {"host": "127.0.0.1", "port": 0, "flows": flows}
    extra.update(extra_kw)
    return ActivePiecesAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: ActivePiecesAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post("/activepieces/{route_name}", adapter._handle_webhook)
    return app


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


class TestAPEventReachesAgent:

    @pytest.mark.asyncio
    async def test_bearer_post_triggers_agent_with_skills(self):
        """A bearer-authenticated AP envelope wakes the agent run with the
        route's skill injected and the ActivePieces source identity."""
        flows = {
            "gmail_new_email": {
                "secret": AP_SECRET,
                "prompt": "New email from {event.from}: {event.subject}",
                "skills": ["email-triage"],
                "deliver": "log",
            }
        }
        adapter = _make_adapter(flows)
        captured: list[MessageEvent] = []

        async def _capture(event: MessageEvent):
            captured.append(event)

        adapter.handle_message = _capture
        skill_content = "TRIAGE SKILL — work the email: New email from billing@example.com: Invoice attached"

        with patch(
            "agent.skill_commands.build_skill_invocation_message",
            return_value=skill_content,
        ) as mock_build, patch(
            "agent.skill_commands.get_skill_commands",
            return_value={"/email-triage": {"name": "email-triage"}},
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/activepieces/gmail_new_email",
                    data=json.dumps(AP_PAYLOAD),
                    headers={**_bearer(AP_SECRET), "X-Request-ID": "ap-delivery-001"},
                )
                assert resp.status == 202
                data = await resp.json()
                assert data["status"] == "accepted"
                assert data["route"] == "gmail_new_email"

            await asyncio.sleep(0.05)
            mock_build.assert_called_once()

        assert len(captured) == 1
        event = captured[0]
        # The skills lane: the prompt is the skill invocation, not the raw render.
        assert "TRIAGE SKILL" in event.text
        assert "Invoice attached" in event.text
        assert event.source.platform == Platform.ACTIVEPIECES
        assert event.source.chat_type == "activepieces"
        assert event.source.chat_id.startswith("activepieces:gmail_new_email:")
        assert event.message_id == "ap-delivery-001"

    @pytest.mark.asyncio
    async def test_global_secret_bearer_accepted(self):
        """A flow without its own secret uses the platform-level secret."""
        adapter = _make_adapter(
            {"webhook_in": {"deliver": "log"}}, secret=AP_SECRET)
        captured: list[MessageEvent] = []

        async def _capture(event: MessageEvent):
            captured.append(event)

        adapter.handle_message = _capture
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/activepieces/webhook_in", json=AP_PAYLOAD, headers=_bearer(AP_SECRET))
            assert resp.status == 202
        await asyncio.sleep(0.05)
        assert len(captured) == 1

    @pytest.mark.asyncio
    async def test_wrong_bearer_rejected(self):
        adapter = _make_adapter({"gmail_new_email": {"secret": AP_SECRET, "deliver": "log"}})
        adapter.handle_message = AsyncMock()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/activepieces/gmail_new_email", json=AP_PAYLOAD, headers=_bearer("wrong-secret"))
            assert resp.status == 401
            resp2 = await cli.post("/activepieces/gmail_new_email", json=AP_PAYLOAD)
            assert resp2.status == 401  # no auth header at all → fail closed
        await asyncio.sleep(0.05)
        assert adapter.handle_message.await_count == 0


class TestSecretlessFlowFailsClosed:

    def test_secretless_flow_rejected_at_validation(self):
        adapter = _make_adapter({"open_flow": {"deliver": "log"}})
        with pytest.raises(ValueError, match="no HMAC secret"):
            adapter._validate_route("open_flow", adapter._routes["open_flow"])

    def test_defaults_are_loopback_and_own_port(self):
        adapter = ActivePiecesAdapter(PlatformConfig(enabled=True, extra={"flows": {}}))
        assert adapter._host == "127.0.0.1"
        assert adapter._port == 8647
        assert adapter._source_label == "activepieces"


class TestAPReplyDelivery:

    @pytest.mark.asyncio
    async def test_reply_posted_to_flow_url(self):
        """deliver=activepieces POSTs {delivery_id, flow, reply} to the reply URL."""
        received: list[dict] = []
        received_headers: list[dict] = []

        async def _reply_target(request: "web.Request") -> "web.Response":
            received.append(await request.json())
            received_headers.append(dict(request.headers))
            return web.json_response({"ok": True})

        target = web.Application()
        target.router.add_post("/reply", _reply_target)
        async with TestClient(TestServer(target)) as target_cli:
            reply_url = str(target_cli.server.make_url("/reply"))
            flows = {
                "gmail_new_email": {
                    "secret": AP_SECRET,
                    "deliver": "activepieces",
                    "deliver_extra": {"url": reply_url, "headers": {"X-AP-Flow": "gmail_new_email"}},
                }
            }
            adapter = _make_adapter(flows)
            adapter.handle_message = AsyncMock()

            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/activepieces/gmail_new_email", json=AP_PAYLOAD,
                    headers={**_bearer(AP_SECRET), "X-Request-ID": "ap-delivery-77"})
                assert resp.status == 202

            chat_id = "activepieces:gmail_new_email:ap-delivery-77"
            assert chat_id in adapter._delivery_info
            assert adapter._delivery_info[chat_id]["deliver"] == "activepieces"

            result = await adapter.send(chat_id, "Filed the invoice and drafted a reply.")
            assert result.success is True

        assert len(received) == 1
        assert received[0] == {
            "delivery_id": "ap-delivery-77",
            "flow": "gmail_new_email",
            "reply": "Filed the invoice and drafted a reply.",
        }
        assert received_headers[0].get("X-AP-Flow") == "gmail_new_email"

    @pytest.mark.asyncio
    async def test_reply_target_down_reports_failure(self):
        flows = {
            "gmail_new_email": {
                "secret": AP_SECRET,
                "deliver": "activepieces",
                "deliver_extra": {"url": "http://127.0.0.1:9/nope"},
            }
        }
        adapter = _make_adapter(flows)
        result = await adapter._deliver_activepieces(
            "hello", {"deliver_extra": flows["gmail_new_email"]["deliver_extra"], "route": "gmail_new_email"},
            "activepieces:gmail_new_email:d1")
        assert result.success is False
        assert result.error

    def test_deliver_activepieces_without_url_rejected(self):
        adapter = _make_adapter({"gmail_new_email": {"secret": AP_SECRET, "deliver": "activepieces"}})
        with pytest.raises(ValueError, match="http\\(s\\) URL"):
            adapter._validate_route("gmail_new_email", adapter._routes["gmail_new_email"])
