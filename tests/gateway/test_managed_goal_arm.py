"""Jean-Billie managed background missions — fork C1 F1/F2/F4.

Covers the loopback arming surface and its invariants WITHOUT a live LLM:

* F1  — ``POST /v1/message`` arms a mission: 202 + ``goalId``, and injects a synthetic
        ``/goal <text>`` turn on an API_SERVER source keyed by ``conversationId``.
* F1  — idempotent re-arm: a second POST for an already-active mission returns 202
        WITHOUT re-arming (turn budget / goal text preserved).
* F2  — stable anchor: the goal is keyed by ``conversationId`` (``_goal_anchor_for_source``),
        so it stays reachable as ``goal:<conversationId>`` even when the session_id rotates at
        compression; API_SERVER turns are authorized by the API-key boundary.
* F4  — "rien ne part sans accord": any outbound tool a mission turn might call is PROPOSED
        (or fail-closed BLOCKED), never silently executed — the autonomous loop runs through the
        same ``jb_outbound`` middleware as every other turn.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


# ──────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so goal ``state_meta`` writes never touch the real DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _arm_app(adapter: APIServerAdapter) -> web.Application:
    """Minimal aiohttp app exposing only the arming route (mirrors _create_app in test_api_server)."""
    app = web.Application()
    app.router.add_post("/v1/message", adapter._handle_arm_message)
    return app


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra=extra))
    # Stub the gateway loop entrypoint so arming never spins a real agent turn; we assert the
    # synthetic event the handler injects instead.
    adapter.handle_message = AsyncMock()
    return adapter


# ──────────────────────────────────────────────────────────────────────
# F1 — arming endpoint
# ──────────────────────────────────────────────────────────────────────


class TestArmEndpoint:
    @pytest.mark.asyncio
    async def test_arm_returns_202_and_injects_goal_event(self, hermes_home):
        adapter = _make_adapter()
        async with TestClient(TestServer(_arm_app(adapter))) as cli:
            resp = await cli.post(
                "/v1/message",
                json={"text": "relancer les devis en attente", "conversationId": "mission:abc"},
            )
            assert resp.status == 202
            data = await resp.json()
            assert data["goalId"] == "mission:abc"
            assert data["status"] == "armed"

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "/goal relancer les devis en attente"
        assert event.message_type == MessageType.TEXT
        assert event.source.platform == Platform.API_SERVER
        assert event.source.chat_id == "mission:abc"

    @pytest.mark.asyncio
    async def test_missing_text_is_400(self, hermes_home):
        adapter = _make_adapter()
        async with TestClient(TestServer(_arm_app(adapter))) as cli:
            resp = await cli.post("/v1/message", json={"conversationId": "mission:abc"})
            assert resp.status == 400
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_blank_text_is_400(self, hermes_home):
        adapter = _make_adapter()
        async with TestClient(TestServer(_arm_app(adapter))) as cli:
            resp = await cli.post("/v1/message", json={"text": "   ", "conversationId": "mission:abc"})
            assert resp.status == 400
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_conversation_id_is_400(self, hermes_home):
        adapter = _make_adapter()
        async with TestClient(TestServer(_arm_app(adapter))) as cli:
            resp = await cli.post("/v1/message", json={"text": "do X"})
            assert resp.status == 400
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_json_is_400(self, hermes_home):
        adapter = _make_adapter()
        async with TestClient(TestServer(_arm_app(adapter))) as cli:
            resp = await cli.post("/v1/message", data="not json", headers={"Content-Type": "application/json"})
            assert resp.status == 400
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auth_required_when_key_set(self, hermes_home):
        adapter = _make_adapter(api_key="sk-secret")
        async with TestClient(TestServer(_arm_app(adapter))) as cli:
            resp = await cli.post(
                "/v1/message", json={"text": "do X", "conversationId": "mission:abc"}
            )
            assert resp.status == 401
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auth_ok_with_bearer(self, hermes_home):
        adapter = _make_adapter(api_key="sk-secret")
        async with TestClient(TestServer(_arm_app(adapter))) as cli:
            resp = await cli.post(
                "/v1/message",
                json={"text": "do X", "conversationId": "mission:abc"},
                headers={"Authorization": "Bearer sk-secret"},
            )
            assert resp.status == 202
        adapter.handle_message.assert_awaited_once()


# ──────────────────────────────────────────────────────────────────────
# F1 — idempotent re-arm
# ──────────────────────────────────────────────────────────────────────


class TestIdempotentRearm:
    @pytest.mark.asyncio
    async def test_active_goal_not_rearmed(self, hermes_home):
        from hermes_cli.goals import GoalManager

        GoalManager(session_id="mission:abc").set("travail initial de la mission")
        before = GoalManager(session_id="mission:abc").state
        assert before is not None and before.status == "active"

        adapter = _make_adapter()
        async with TestClient(TestServer(_arm_app(adapter))) as cli:
            resp = await cli.post(
                "/v1/message",
                json={"text": "consigne differente", "conversationId": "mission:abc"},
            )
            assert resp.status == 202
            data = await resp.json()
            assert data["goalId"] == "mission:abc"
            assert data["status"] == "already_active"

        # No re-arm: the loop was never entered and the goal text + turn budget are untouched.
        adapter.handle_message.assert_not_awaited()
        after = GoalManager(session_id="mission:abc").state
        assert after.goal == "travail initial de la mission"
        assert after.turns_used == before.turns_used


# ──────────────────────────────────────────────────────────────────────
# F2 — stable anchor + API_SERVER authorization
# ──────────────────────────────────────────────────────────────────────


class TestGoalAnchor:
    def test_api_server_source_anchors_to_conversation_id(self):
        runner = GatewayRunner(GatewayConfig())
        src = SessionSource(platform=Platform.API_SERVER, chat_id="mission:abc")
        assert runner._goal_anchor_for_source(src) == "mission:abc"

    def test_other_platforms_have_no_anchor(self):
        runner = GatewayRunner(GatewayConfig())
        src = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", user_id="u1")
        assert runner._goal_anchor_for_source(src) is None

    def test_none_source_has_no_anchor(self):
        runner = GatewayRunner(GatewayConfig())
        assert runner._goal_anchor_for_source(None) is None

    def test_api_server_authorized_without_allowlist(self):
        """API_SERVER turns are authorized by the API-key boundary — no user allowlist needed."""
        runner = GatewayRunner(GatewayConfig())
        src = SessionSource(platform=Platform.API_SERVER, chat_id="mission:abc", user_id="jb-managed")
        assert runner._is_user_authorized(src) is True

    def test_telegram_still_denied_without_allowlist(self, monkeypatch):
        """Anchoring/auth change must not loosen other platforms."""
        for var in ("TELEGRAM_ALLOWED_USERS", "TELEGRAM_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS"):
            monkeypatch.delenv(var, raising=False)
        runner = GatewayRunner(GatewayConfig())
        src = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", user_id="stranger")
        assert runner._is_user_authorized(src) is False

    def test_goal_survives_session_rotation(self, hermes_home):
        """The mission goal lives at goal:<conversationId>; a rotated session_id can't see it, the anchor can."""
        from hermes_cli.goals import GoalManager

        GoalManager(session_id="mission:abc").set("travail de fond de la mission")

        # A compression-rotated session_id (timestamp-style) is a different key — orphaned.
        assert GoalManager(session_id="20260620_120000_deadbeef").is_active() is False
        # The stable anchor still resolves the active goal.
        assert GoalManager(session_id="mission:abc").is_active() is True

    def test_arm_seam_keys_goal_by_anchor_not_gateway_session_id(self, hermes_home):
        """Seam contract (pre-rebase): the /goal turn injected by POST /v1/message
        arms through ``_get_goal_manager_for_event`` — for an API_SERVER event the
        manager MUST be keyed by the conversationId ANCHOR, never by the gateway
        session_id that compression rotates. If upstream restructures the goal
        keying (v0.18 completion contracts), this fails at the seam."""
        from gateway.platforms.base import MessageEvent
        from hermes_cli.goals import GoalManager

        runner = GatewayRunner(GatewayConfig())
        source = SessionSource(
            platform=Platform.API_SERVER,
            chat_id="mission:rot",
            chat_name="Jean-Billie mission",
            chat_type="dm",
            user_id="jb-managed",
            user_name="Jean-Billie",
        )
        event = MessageEvent(text="/goal relancer les impayés", message_type=MessageType.TEXT, source=source)

        mgr, session_entry = runner._get_goal_manager_for_event(event)
        assert mgr is not None and session_entry is not None
        # Keyed by the stable anchor…
        assert mgr.session_id == "mission:rot"
        # …which is NOT the gateway session_id (that one rotates at compression).
        assert session_entry.session_id != "mission:rot"

        mgr.set("relancer les impayés")
        # Reachable via the anchor; invisible from the rotatable session_id.
        assert GoalManager(session_id="mission:rot").is_active() is True
        assert GoalManager(session_id=session_entry.session_id).is_active() is False


# ──────────────────────────────────────────────────────────────────────
# F4 — "rien ne part sans accord" (egress stays gated for mission turns)
# ──────────────────────────────────────────────────────────────────────


class TestOutboundInvariant:
    """A managed mission's autonomous turns run through the SAME jb_outbound middleware as any
    other turn — there is no privileged path. An outbound tool is PROPOSED (or fail-closed
    BLOCKED), never silently executed. (Execution-vs-propose round-trip is covered exhaustively
    in plugins/jb_outbound/test_jb_outbound*.py; here we assert the classification contract the
    mission loop relies on.)"""

    @staticmethod
    def _classify():
        plugins_dir = Path(__file__).resolve().parents[2] / "plugins"
        if str(plugins_dir) not in sys.path:
            sys.path.insert(0, str(plugins_dir))
        import jb_outbound.classify as classify  # noqa: WPS433

        return classify

    def test_outbound_tools_are_proposed_not_sent(self):
        classify = self._classify()
        # Native send + Composio third-party egress → PROPOSE (dashboard approval required).
        assert classify.classify("send_message") == classify.PROPOSE
        assert classify.classify("mcp_composio_GMAIL_SEND_EMAIL") == classify.PROPOSE
        assert classify.classify("mcp_composio_LINKEDIN_CREATE_POST") == classify.PROPOSE

    def test_reads_pass_and_unknown_egress_fails_closed(self):
        classify = self._classify()
        assert classify.classify("mcp_composio_GMAIL_FETCH_MESSAGES") == classify.PASS
        # An unclassified Composio action is BLOCKED, never auto-sent.
        assert classify.classify("mcp_composio_UNCLASSIFIED_THING") == classify.BLOCK
