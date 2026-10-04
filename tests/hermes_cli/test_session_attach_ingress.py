"""Serve attachment reuses its existing token and exact owning lease across profiles."""
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from hermes_state import SessionDB
from tests.tui_gateway.test_queued_prompt_persistence import _desktop_session
from tui_gateway import server, shared_runtime


@pytest.mark.asyncio
@pytest.mark.parametrize("gated", [False, True])
async def test_serve_handshake_uses_existing_endpoint_and_profile_lease(tmp_path, monkeypatch, gated):
    from hermes_cli import web_server
    from agent import secret_scope

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr(web_server.app.state, "session_attach_origin", "http://127.0.0.1:8765", raising=False)
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "EXISTING-TOKEN")
    monkeypatch.setattr(web_server.app.state, "auth_required", gated, raising=False)
    monkeypatch.setattr(shared_runtime, "_origin", None)
    records = []
    try:
        for name in ("a", "b"):
            home = tmp_path / name
            monkeypatch.setenv("HERMES_HOME", str(home))
            db = SessionDB(db_path=home / "state.db")
            sid, key = _desktop_session(monkeypatch, db)
            session = server._sessions[sid]
            session["profile_home"] = str(home)
            with server._session_profile_runtime_scope(session):
                server._ensure_session_db_row(session)
                assert server._ensure_active_session_slot(sid, session) is None
            records.append((home, db, sid, key, session["active_session_lease"]))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web_server.app, client=("127.0.0.1", 5000)),
                                     base_url="http://127.0.0.1:8765") as client:
            for home, db, sid, key, lease in (records[0], records[1], records[0]):
                from hermes_cli.active_sessions import active_session_registry_snapshot
                assert active_session_registry_snapshot(home)[0]["metadata"]["shared_runtime_url"] == "http://127.0.0.1:8765"
                params = {"session_id": key, "lease_id": lease.lease_id, "profile_home": str(home.resolve())}
                response = await client.get("/api/session-attach", params=params)
                assert response.status_code == 200, response.text
                url = urlsplit(response.json()["websocket_url"])
                assert (url.scheme, url.netloc, url.path) == ("ws", "127.0.0.1:8765", "/api/ws")
                if gated:
                    from hermes_cli.dashboard_auth.ws_tickets import consume_ticket, INTERNAL_PROVIDER
                    assert consume_ticket(parse_qs(url.query)["ticket"][0])["provider"] == INTERNAL_PROVIDER
                else:
                    assert parse_qs(url.query) == {"token": ["EXISTING-TOKEN"]}
                for field in params:
                    response = await client.get("/api/session-attach", params={**params, field: "wrong"})
                    assert response.status_code == 409
                    assert "EXISTING-TOKEN" not in response.text
                for headers in ({"Origin": "https://example.com"}, {"Host": "example.com"}):
                    response = await client.get("/api/session-attach", params=params, headers=headers)
                    assert response.status_code == 403
    finally:
        for home, db, sid, key, lease in records:
            server._sessions.pop(sid, None)
            lease.release()
            db.close()
