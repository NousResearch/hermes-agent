"""Hard-deleted gateway sessions must not resurrect through the routing index (#42422).

A hard delete (Desktop remote mode -> ``DELETE /api/sessions/<id>``, or ``hermes
sessions delete`` / TUI deletes out of process) removes the state.db rows but used to
leave the gateway's durable channel->session routing index intact. The next inbound
Discord/Telegram message then resolved the SAME session id, and run_agent's INSERT OR
IGNORE re-created the row — the deleted conversation reappeared in the session list
with its prior content.

Two invariants pin the fix:

* the gateway's own delete endpoint drops every routing entry for the deleted id in
  BOTH durable copies (the ``gateway_routing`` table and the legacy ``sessions.json``
  mirror) and scrubs the on-disk transcript artifacts, so the next inbound message
  mints a fresh session;
* a delete made OUT of the gateway process self-heals at routing time: an entry whose
  session row is gone from a readable DB is dropped like an ended one instead of
  resurrecting the deleted id.
"""

import json
from datetime import datetime

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import hermes_constants
import hermes_state
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.session import SessionEntry, SessionSource, SessionStore


def _source(user_id: str = "user-1") -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="chan-1",
        chat_name="chan",
        chat_type="dm",
        user_id=user_id,
    )


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so the store, its DB and the mirror all land in tmp."""
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    return tmp_path


@pytest.fixture
def store(home) -> SessionStore:
    return SessionStore(sessions_dir=home / "sessions", config=GatewayConfig())


def _routing_rows(store: SessionStore) -> dict:
    return store._routing_db.load_gateway_routing_entries(scope=store._routing_scope())


def _mirror_entries(store: SessionStore) -> dict:
    import json
    mirror = store.sessions_dir / "sessions.json"
    if not mirror.exists():
        return {}
    return {k: v for k, v in json.loads(mirror.read_text()).items() if not k.startswith("_")}


class TestRemoveBySessionId:
    def test_drops_every_route_for_the_id_and_persists_both_copies(self, store):
        sid = "20260926_120000_deadbeef"
        entry_kwargs = dict(created_at=datetime.now(), updated_at=datetime.now(),
                            platform=Platform.DISCORD, chat_type="dm")
        for key in ("agent:main:discord:chan-1:user-1", "agent:main:discord:chan-1:user-2"):
            store._entries[key] = SessionEntry(session_key=key, session_id=sid, **entry_kwargs)
        store._entries["agent:main:discord:chan-2:user-3"] = SessionEntry(
            session_key="agent:main:discord:chan-2:user-3", session_id="other-sid", **entry_kwargs)

        dropped = store.remove_by_session_id(sid)

        assert dropped == 2
        assert not [k for k, e in store._entries.items() if e.session_id == sid]
        assert store._entries["agent:main:discord:chan-2:user-3"].session_id == "other-sid"
        for key in ("agent:main:discord:chan-1:user-1", "agent:main:discord:chan-1:user-2"):
            assert key not in _routing_rows(store)
            assert key not in _mirror_entries(store)
        assert "agent:main:discord:chan-2:user-3" in _routing_rows(store)
        assert "agent:main:discord:chan-2:user-3" in _mirror_entries(store)

    def test_unknown_id_is_a_noop(self, store):
        assert store.remove_by_session_id("no-such-sid") == 0


class TestOutOfProcessDeleteSelfHeals:
    def test_hard_deleted_row_yields_a_fresh_session_not_a_resurrected_id(self, store):
        entry = store.get_or_create_session(_source())
        key = entry.session_key
        db = store._db_for_key(key)
        assert db.get_session(entry.session_id) is not None

        # Out-of-process hard delete (CLI `hermes sessions delete`, TUI): only the
        # durable row goes; the gateway's in-memory index is untouched.
        assert db.delete_session(entry.session_id, sessions_dir=store.sessions_dir)

        routed = store.get_or_create_session(_source())

        assert routed.session_key == key
        assert routed.session_id != entry.session_id
        # The stale route is gone from both durable copies: the key now maps to the FRESH id.
        rows, mirror = _routing_rows(store), _mirror_entries(store)
        for snapshot in (rows, mirror):
            route = snapshot[key]
            if isinstance(route, str):  # state.db rows are JSON strings
                route = json.loads(route)
            assert route["session_id"] == routed.session_id
            assert route["session_id"] != entry.session_id
        # And the fresh route points at a live row.
        assert store._db_for_key(key).get_session(routed.session_id) is not None


class TestDeleteEndpointClearsRouting:
    @pytest.mark.asyncio
    async def test_delete_scrubs_row_files_and_routing(self, store, home):
        entry = store.get_or_create_session(_source())
        sid = entry.session_id
        db = store._db_for_key(entry.session_key)
        transcript = store.sessions_dir / f"{sid}.json"
        dump = store.sessions_dir / f"request_dump_{sid}_0.json"
        store.sessions_dir.mkdir(parents=True, exist_ok=True)
        transcript.write_text("{}")
        dump.write_text("{}")

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        adapter._session_db = db
        app = web.Application()
        app["gateway_runner"] = type("Runner", (), {"session_store": store})()
        app.router.add_delete("/api/sessions/{session_id}", adapter._handle_delete_session)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.delete(f"/api/sessions/{sid}")
            assert resp.status == 200
            assert (await resp.json())["deleted"] is True

        # DB row and on-disk artifacts are gone.
        assert db.get_session(sid) is None
        assert not transcript.exists()
        assert not dump.exists()
        # Routing entries for the deleted id are gone from memory and both durable copies.
        assert store.peek_session_id(entry.session_key) is None
        assert entry.session_key not in _routing_rows(store)
        assert entry.session_key not in _mirror_entries(store)
        # The next inbound message mints a fresh session instead of resurrecting.
        routed = store.get_or_create_session(_source())
        assert routed.session_id != sid
