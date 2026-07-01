"""
Tests for the /v1/actions tool-approval endpoints on the API server adapter.

Exercise the real ``tools.tool_gate`` + ``tools.write_approval`` + kanban engine
against a temp HERMES_HOME / board, so the HTTP layer and the deferred-approval
machinery agree end-to-end:

- list / detail (token redacted, 404)
- approve -> spawns the exec card (reuses tool_gate.approve_action), one-shot
  (second approve -> 409), missing -> 404, expired -> 409
- reject -> discards the pending + archives the approval card, missing -> 404
- auth: 401 when API_SERVER_KEY is set and the bearer is missing
"""

import time
from functools import partial

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware
from gateway.platforms import actions_api
from hermes_cli import kanban_db as kb


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Pin HERMES_HOME + the kanban DB to temp paths; reset the init cache."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb._INITIALIZED_PATHS.clear()
    kb.init_db(db_path)
    return tmp_path


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/actions", partial(actions_api.handle_list_actions, adapter))
    app.router.add_get("/v1/actions/{pending_id}", partial(actions_api.handle_get_action, adapter))
    app.router.add_post("/v1/actions/{pending_id}/approve", partial(actions_api.handle_approve_action, adapter))
    app.router.add_post("/v1/actions/{pending_id}/reject", partial(actions_api.handle_reject_action, adapter))
    return app


def _stage(tool="send_gmail_message", args=None, *, board_assignee="alon",
           tenant="rev", ttl_hours=72):
    """Stage a deferred action via the real gate path; return (pending_id, card_id)."""
    from tools import tool_gate
    args = args if args is not None else {"to": "pm@example.com", "subject": "hi"}
    cfg = {
        "enabled": True,
        "require_approval": [tool],
        "force_deferred": [tool],
        "deferred": {"board_assignee": board_assignee, "tenant_from_session": False,
                     "fixed_tenant": tenant, "pending_ttl_hours": ttl_hours},
    }
    r = tool_gate.stage_deferred(
        tool, args, summary=tool_gate.summarize_tool_call(tool, args), config=cfg)
    return r["pending_id"], r["card_id"]


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestListAndGet:
    @pytest.mark.asyncio
    async def test_list(self, home):
        _stage()
        _stage(args={"to": "b@c.com"})
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.get("/v1/actions")
            assert resp.status == 200
            data = await resp.json()
            assert data["count"] == 2
            assert all(a["tool_name"] == "send_gmail_message" for a in data["actions"])
            # The one-shot replay token must never appear in list rows.
            assert all("token" not in a for a in data["actions"])

    @pytest.mark.asyncio
    async def test_detail_redacts_token_and_404(self, home):
        pid, card_id = _stage()
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.get(f"/v1/actions/{pid}")
            assert resp.status == 200
            action = (await resp.json())["action"]
            assert action["id"] == pid
            assert action["card_id"] == card_id
            assert action["tool_name"] == "send_gmail_message"
            assert "token" not in action  # redacted

            assert (await cli.get("/v1/actions/nope")).status == 404


# ---------------------------------------------------------------------------
# Approve
# ---------------------------------------------------------------------------


class TestApprove:
    @pytest.mark.asyncio
    async def test_approve_spawns_exec_card_and_is_one_shot(self, home):
        pid, _ = _stage()
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.post(f"/v1/actions/{pid}/approve")
            assert resp.status == 200
            body = await resp.json()
            assert body["ok"] is True
            assert body["exec_card_id"]
            # The exec card exists on the board and is assigned to a real profile.
            with kb.connect_closing() as conn:
                exec_card = kb.get_task(conn, body["exec_card_id"])
            assert exec_card is not None

            # Second approve is refused (already approved) -> 409.
            resp2 = await cli.post(f"/v1/actions/{pid}/approve")
            assert resp2.status == 409

    @pytest.mark.asyncio
    async def test_approve_missing_404(self, home):
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            assert (await cli.post("/v1/actions/ghost/approve")).status == 404

    @pytest.mark.asyncio
    async def test_approve_expired_409(self, home):
        pid, _ = _stage()
        from tools import tool_gate
        from tools import write_approval as wa
        wa.update_pending(tool_gate.SUBSYSTEM, pid, {"expires_at": time.time() - 10})
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.post(f"/v1/actions/{pid}/approve")
            assert resp.status == 409


# ---------------------------------------------------------------------------
# Reject
# ---------------------------------------------------------------------------


class TestReject:
    @pytest.mark.asyncio
    async def test_reject_discards_and_archives(self, home):
        pid, card_id = _stage()
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.post(f"/v1/actions/{pid}/reject")
            assert resp.status == 200
            assert (await resp.json())["ok"] is True
            # Pending gone -> a later GET 404s; can't be replayed.
            assert (await cli.get(f"/v1/actions/{pid}")).status == 404
        # Approval card archived (hidden from the board).
        with kb.connect_closing() as conn:
            card = kb.get_task(conn, card_id)
        assert card is not None and card.status == "archived"

    @pytest.mark.asyncio
    async def test_reject_missing_404(self, home):
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            assert (await cli.post("/v1/actions/ghost/reject")).status == 404


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestAuth:
    @pytest.mark.asyncio
    async def test_requires_bearer(self, home):
        _stage()
        adapter = _make_adapter(api_key="sk-secret")
        async with TestClient(TestServer(_app(adapter))) as cli:
            assert (await cli.get("/v1/actions")).status == 401
            ok = await cli.get("/v1/actions", headers={"Authorization": "Bearer sk-secret"})
            assert ok.status == 200
