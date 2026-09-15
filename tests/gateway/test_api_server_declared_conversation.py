"""Declared-conversation identity on the API server (#96811).

A client that manages its own history has no ``previous_response_id`` chain,
so ``/v1/responses`` and ``/v1/runs`` used to mint a throwaway physical
session id per request even when the request declared its conversation with
``X-Hermes-Session-Key``.  Every conversation-affinity hint Hermes sends is
derived from that physical id — ``prompt_cache_key`` on both OpenAI-wire
transports, the OpenRouter/Nous sticky ``session_id``, and xAI's
``x-grok-conv-id`` — so all four re-keyed on every single reply.

These tests pin the identity contract itself rather than the four consumers:
the declared key resolves to one live session, the resolution is fenced by
the durable conversation boundaries already recorded in
``sessions.end_reason``, and nothing that does not declare a key changes.
"""

import asyncio
import types
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)
from hermes_state import SessionDB

KEY = "agent:main:api_server:room-42:member-7"
OTHER_KEY = "agent:main:api_server:room-42:member-8"
SOURCE = "api_server"


@pytest.fixture
def adapter(tmp_path):
    """An adapter whose lazy SessionDB is pinned to a scratch state.db."""
    a = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "key": "k"})
    )
    db = SessionDB(tmp_path / "state.db")
    a._session_db = db
    try:
        yield a, db
    finally:
        db.close()


def _seed(db, session_id, *, key=KEY, source=SOURCE):
    db.create_session(session_id=session_id, source=source, model="m")
    if key:
        db.record_gateway_session_peer(session_id, source=source, session_key=key)


class TestDeclaredConversationResolution:
    def test_declared_key_resolves_the_live_session(self, adapter):
        a, db = adapter
        _seed(db, "sess-live")
        assert a._declared_conversation_session(KEY) == "sess-live"

    def test_replies_land_on_one_session(self, adapter):
        """The defect in one line: same key, three replies, one identity."""
        a, db = adapter
        _seed(db, "sess-live")
        assert {a._declared_conversation_session(KEY) for _ in range(3)} == {"sess-live"}

    def test_no_declared_key_resolves_nothing(self, adapter):
        """Undeclared clients keep today's per-request identity."""
        a, db = adapter
        _seed(db, "sess-live")
        assert a._declared_conversation_session(None) is None
        assert a._declared_conversation_session("") is None
        assert a._declared_conversation_session("   ") is None

    def test_unknown_key_resolves_nothing(self, adapter):
        a, db = adapter
        _seed(db, "sess-live")
        assert a._declared_conversation_session(OTHER_KEY) is None

    def test_distinct_keys_stay_isolated(self, adapter):
        a, db = adapter
        _seed(db, "sess-mine", key=KEY)
        _seed(db, "sess-theirs", key=OTHER_KEY)
        assert a._declared_conversation_session(KEY) == "sess-mine"
        assert a._declared_conversation_session(OTHER_KEY) == "sess-theirs"

    def test_another_platforms_key_is_not_adopted(self, adapter):
        """The source filter keeps a telegram chat key out of the API server."""
        a, db = adapter
        _seed(db, "sess-telegram", key=KEY, source="telegram")
        assert a._declared_conversation_session(KEY) is None

    def test_db_failure_degrades_to_a_fresh_id(self, adapter):
        a, _ = adapter

        class _Boom:
            def find_latest_gateway_session_for_peer(self, **_kw):
                raise RuntimeError("db down")

        a._session_db = _Boom()
        assert a._declared_conversation_session(KEY) is None

    def test_missing_db_degrades_to_a_fresh_id(self, adapter, monkeypatch):
        a, _ = adapter
        monkeypatch.setattr(a, "_ensure_session_db", lambda: None)
        assert a._declared_conversation_session(KEY) is None


class TestConversationBoundariesRotate:
    """The boundary the recovery fence honours is durable in end_reason.

    #79017/#86733's contract: an affinity scope stays warm across
    continuation and compression rotation, and goes cold on a new
    conversation.  ``_RESET_END_REASONS`` is that boundary set, and the
    recovery fence honours every member of it — including the idle/daily
    policy resets, not just ``/new``.
    """

    @pytest.mark.parametrize(
        "reason",
        ["session_reset", "session_switch", "idle", "daily", "suspended",
         "resume_pending_expired"],
    )
    def test_boundary_rotates_the_conversation(self, adapter, reason):
        a, db = adapter
        _seed(db, "sess-old")
        db.end_session("sess-old", reason)
        assert a._declared_conversation_session(KEY) is None

    def test_boundary_cannot_be_reached_behind(self, adapter):
        """A later row wins, and the retired one never comes back (no ABA)."""
        a, db = adapter
        _seed(db, "sess-gen1")
        db.end_session("sess-gen1", "session_reset")
        _seed(db, "sess-gen2")
        assert a._declared_conversation_session(KEY) == "sess-gen2"
        db.end_session("sess-gen2", "session_reset")
        assert a._declared_conversation_session(KEY) is None

    def test_accidental_end_stays_resumable(self, adapter):
        """An accidental close is not a conversation boundary."""
        a, db = adapter
        _seed(db, "sess-live")
        db.end_session("sess-live", "agent_close")
        assert a._declared_conversation_session(KEY) == "sess-live"


class TestBindDeclaredConversation:
    def test_bind_makes_the_row_resolvable(self, adapter):
        """Without the bind the row is written unkeyed and is invisible."""
        a, db = adapter
        db.create_session(session_id="sess-new", source=SOURCE, model="m")
        assert a._declared_conversation_session(KEY) is None

        a._bind_declared_conversation("sess-new", KEY)
        assert a._declared_conversation_session(KEY) == "sess-new"

    def test_bind_is_a_noop_without_a_key(self, adapter):
        a, db = adapter
        db.create_session(session_id="sess-new", source=SOURCE, model="m")
        a._bind_declared_conversation("sess-new", None)
        a._bind_declared_conversation("sess-new", "  ")
        assert db.get_session("sess-new").get("session_key") in (None, "")

    def test_bind_is_a_noop_without_a_session(self, adapter):
        a, _ = adapter
        a._bind_declared_conversation(None, KEY)
        a._bind_declared_conversation("", KEY)
        assert a._declared_conversation_session(KEY) is None

    def test_bind_survives_a_db_failure(self, adapter):
        a, _ = adapter

        class _Boom:
            def record_gateway_session_peer(self, *_a, **_kw):
                raise RuntimeError("db down")

        a._session_db = _Boom()
        a._bind_declared_conversation("sess-new", KEY)  # must not raise

    def test_bind_follows_a_compression_rotation(self, adapter):
        """The turn binds the row it ended on; the retired parent follows.

        ``include_compression_ancestors`` keys the whole compression lineage,
        so the next reply resolves the live child rather than its parent.
        """
        a, db = adapter
        db.create_session(session_id="sess-parent", source=SOURCE, model="m")
        db.end_session("sess-parent", "compression")
        db.create_session(
            session_id="sess-child",
            source=SOURCE,
            model="m",
            parent_session_id="sess-parent",
        )
        a._bind_declared_conversation("sess-child", KEY)
        assert a._declared_conversation_session(KEY) == "sess-child"


class TestOtherStructuresUnaffected:
    """The bind uses the same routing-peer record every native platform uses.

    These pin the two places a newly keyed row could leak into: the channel
    directory (contact lists) and the SessionStore's own routing table.
    """

    def test_channel_directory_skips_declared_api_rows(self, adapter):
        """Rows carry no chat_id/origin, so the directory has no entry to build."""
        from gateway import channel_directory

        a, db = adapter
        _seed(db, "sess-live")
        row = db.get_session("sess-live")
        origin = {
            "chat_id": row.get("chat_id"),
            "thread_id": row.get("thread_id"),
            "chat_name": row.get("display_name"),
        }
        assert channel_directory._session_entry_id(origin) is None

    def test_session_store_routing_table_is_untouched(self, adapter):
        """SessionStore loads from gateway_routing, not from sessions.session_key."""
        a, db = adapter
        _seed(db, "sess-live")
        assert db.load_gateway_routing_entries() == {}


class TestHandlerWiring:
    """The precedence the two handlers apply, isolated from aiohttp."""

    @staticmethod
    def _resolve_responses(adapter, *, stored, key):
        # gateway/platforms/api_server.py::_handle_responses
        return stored or adapter._declared_conversation_session(key) or "minted-uuid"

    @staticmethod
    def _resolve_runs(adapter, *, body_id, stored, key):
        # gateway/platforms/api_server.py::_handle_runs
        return (
            (body_id or stored)
            or adapter._declared_conversation_session(key)
            or "minted-run-id"
        )

    def test_response_chain_still_outranks_the_declared_key(self, adapter):
        a, db = adapter
        _seed(db, "sess-live")
        assert self._resolve_responses(a, stored="sess-chained", key=KEY) == "sess-chained"

    def test_declared_key_outranks_a_minted_id(self, adapter):
        a, db = adapter
        _seed(db, "sess-live")
        assert self._resolve_responses(a, stored=None, key=KEY) == "sess-live"

    def test_undeclared_request_still_mints(self, adapter):
        a, _ = adapter
        assert self._resolve_responses(a, stored=None, key=None) == "minted-uuid"

    def test_runs_body_session_id_still_wins(self, adapter):
        a, db = adapter
        _seed(db, "sess-live")
        assert self._resolve_runs(a, body_id="explicit", stored=None, key=KEY) == "explicit"

    def test_runs_declared_key_outranks_the_run_id(self, adapter):
        a, db = adapter
        _seed(db, "sess-live")
        assert self._resolve_runs(a, body_id=None, stored=None, key=KEY) == "sess-live"

    def test_runs_undeclared_still_uses_the_run_id(self, adapter):
        a, _ = adapter
        assert self._resolve_runs(a, body_id=None, stored=None, key=None) == "minted-run-id"


class TestRunAgentOptIn:
    """Only the two routes that resolve a declared id record one."""

    def test_bind_targets_the_rotated_session(self, adapter, monkeypatch):
        """The finally block binds ``agent.session_id``, not the id it started on."""
        a, _ = adapter
        calls = []
        monkeypatch.setattr(
            a, "_bind_declared_conversation", lambda sid, key: calls.append((sid, key))
        )
        agent = types.SimpleNamespace(session_id="sess-rotated")
        a._bind_declared_conversation(
            getattr(agent, "session_id", None) or "sess-initial", KEY
        )
        assert calls == [("sess-rotated", KEY)]

    def test_bind_falls_back_when_the_agent_never_started(self, adapter, monkeypatch):
        a, _ = adapter
        calls = []
        monkeypatch.setattr(
            a, "_bind_declared_conversation", lambda sid, key: calls.append((sid, key))
        )
        agent = types.SimpleNamespace(session_id=None)
        a._bind_declared_conversation(
            getattr(agent, "session_id", None) or "sess-initial", KEY
        )
        assert calls == [("sess-initial", KEY)]


class TestBindFollowsPrecedence:
    """Recording is gated on the declared key actually selecting the session.

    `record_gateway_session_peer` performs `SET session_key = ?`, so binding a
    session that the response chain or an explicit body id selected would
    rewrite THAT conversation's routing key to this request's header: the
    original conversation could no longer be recovered by its own key, and the
    header key would recover it instead (@andrexibiza on #98811).
    """

    @staticmethod
    def _responses_gate(*, stored, key):
        # gateway/platforms/api_server.py::_handle_responses
        return not stored and bool(key)

    @staticmethod
    def _runs_gate(*, body_id, key):
        # gateway/platforms/api_server.py::_handle_runs
        return not body_id and bool(key)

    def test_chained_session_does_not_record_the_header_key(self, adapter):
        assert self._responses_gate(stored="sess-chained", key=KEY) is False

    def test_explicit_body_session_does_not_record_the_header_key(self, adapter):
        assert self._runs_gate(body_id="explicit", key=KEY) is False

    def test_declared_or_minted_session_records(self, adapter):
        assert self._responses_gate(stored=None, key=KEY) is True
        assert self._runs_gate(body_id=None, key=KEY) is True

    def test_undeclared_request_records_nothing(self, adapter):
        assert self._responses_gate(stored=None, key=None) is False
        assert self._runs_gate(body_id=None, key=None) is False

    def test_a_foreign_binding_is_never_overwritten(self, adapter):
        """Defence in depth behind the gate, at the DB layer."""
        a, db = adapter
        _seed(db, "sess-A", key=KEY)

        a._bind_declared_conversation("sess-A", OTHER_KEY)

        assert db.get_session("sess-A")["session_key"] == KEY
        assert a._declared_conversation_session(KEY) == "sess-A"
        assert a._declared_conversation_session(OTHER_KEY) is None

    def test_rebinding_the_same_key_is_idempotent(self, adapter):
        a, db = adapter
        _seed(db, "sess-A", key=KEY)
        a._bind_declared_conversation("sess-A", KEY)
        assert db.get_session("sess-A")["session_key"] == KEY
        assert a._declared_conversation_session(KEY) == "sess-A"

    def test_an_unbound_row_still_binds(self, adapter):
        a, db = adapter
        db.create_session(session_id="sess-new", source=SOURCE, model="m")
        a._bind_declared_conversation("sess-new", KEY)
        assert a._declared_conversation_session(KEY) == "sess-new"

    def test_the_original_conversation_stays_recoverable(self, adapter):
        """The end-to-end shape of the defect: A must survive a B-keyed turn."""
        a, db = adapter
        _seed(db, "sess-A", key=KEY)
        # A request carrying A's chain plus header key B: the gate refuses to
        # record, and the DB guard refuses even if something else tried.
        assert self._responses_gate(stored="sess-A", key=OTHER_KEY) is False
        a._bind_declared_conversation("sess-A", OTHER_KEY)

        assert a._declared_conversation_session(KEY) == "sess-A"
        assert a._declared_conversation_session(OTHER_KEY) is None


API_KEY = "test-api-key"


@pytest.fixture
def live(tmp_path):
    """A real adapter behind real routes, with a scratch state.db attached."""
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "key": API_KEY})
    )
    db = SessionDB(tmp_path / "state.db")
    adapter._session_db = db

    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    try:
        yield adapter, db, app
    finally:
        db.close()


def _spy_run_agent(adapter, seen):
    """Stand in for _run_agent, faithful to its bind contract.

    Production records the peer in _run_agent's ``finally`` when the caller
    opted in; mirroring that here keeps the assertion on the real handler's
    decision AND on the row it produces, instead of on a gate expression
    restated inside the test.
    """

    async def _fake(**kwargs):
        seen.append(kwargs)
        # AIAgent._ensure_db_session() creates the row during the turn.
        sid = kwargs.get("session_id")
        db = adapter._ensure_session_db()
        if sid and db is not None and db.get_session(sid) is None:
            db.create_session(session_id=sid, source=SOURCE, model="m")
        if kwargs.get("bind_declared_conversation"):
            adapter._bind_declared_conversation(
                kwargs.get("session_id"), kwargs.get("gateway_session_key")
            )
        return (
            {"final_response": "ok", "messages": [], "api_calls": 1},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

    return _fake


def _headers(session_key=None):
    h = {"Authorization": f"Bearer {API_KEY}"}
    if session_key:
        h["X-Hermes-Session-Key"] = session_key
    return h


class TestResponsesHandlerPrecedence:
    """POST /v1/responses driven end to end, not a restated gate."""

    @pytest.mark.asyncio
    async def test_declared_key_selects_and_records_the_conversation(self, live):
        adapter, db, app = live
        seen = []
        adapter._run_agent = _spy_run_agent(adapter, seen)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={"model": "hermes-agent", "input": "hi"},
                headers=_headers(KEY),
            )
            assert resp.status == 200

        minted = seen[0]["session_id"]
        assert seen[0]["bind_declared_conversation"] is True
        assert db.get_session(minted)["session_key"] == KEY

    @pytest.mark.asyncio
    async def test_a_second_reply_lands_on_the_same_conversation(self, live):
        adapter, db, app = live
        seen = []
        adapter._run_agent = _spy_run_agent(adapter, seen)

        async with TestClient(TestServer(app)) as cli:
            for _ in range(3):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi"},
                    headers=_headers(KEY),
                )
                assert resp.status == 200

        assert len({k["session_id"] for k in seen}) == 1

    @pytest.mark.asyncio
    async def test_undeclared_request_keeps_a_per_request_id(self, live):
        adapter, db, app = live
        seen = []
        adapter._run_agent = _spy_run_agent(adapter, seen)

        async with TestClient(TestServer(app)) as cli:
            for _ in range(2):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi"},
                    headers=_headers(),
                )
                assert resp.status == 200

        assert len({k["session_id"] for k in seen}) == 2
        assert all(k["bind_declared_conversation"] is False for k in seen)

    @pytest.mark.asyncio
    async def test_the_response_chain_outranks_the_header_and_records_nothing(self, live):
        """The blocker: a chained turn carrying a foreign key must not rebind."""
        adapter, db, app = live
        seen = []
        adapter._run_agent = _spy_run_agent(adapter, seen)

        # Conversation A already belongs to KEY.
        _seed(db, "sess-A", key=KEY)
        adapter._response_store.put(
            "resp_A",
            {
                "conversation_history": [],
                "session_id": "sess-A",
                "instructions": None,
            },
        )

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": "hi",
                    "previous_response_id": "resp_A",
                },
                headers=_headers(OTHER_KEY),
            )
            assert resp.status == 200

        assert seen[0]["session_id"] == "sess-A"
        assert seen[0]["bind_declared_conversation"] is False
        # A keeps its own key; the header key cannot recover it.
        assert db.get_session("sess-A")["session_key"] == KEY
        assert adapter._declared_conversation_session(KEY) == "sess-A"
        assert adapter._declared_conversation_session(OTHER_KEY) is None


class TestRunsHandlerPrecedence:
    """POST /v1/runs driven end to end.

    /v1/runs owns its agent lifecycle rather than routing through _run_agent,
    so the session it settled on is captured where it builds the agent.
    """

    @staticmethod
    def _capture_agent(adapter, seen):
        real = adapter._create_agent

        def _spy(*a, **kw):
            seen.append(kw)
            agent = types.SimpleNamespace(
                session_id=kw.get("session_id"),
                session_prompt_tokens=0,
                session_completion_tokens=0,
                session_total_tokens=0,
                run_conversation=lambda **_kw: {"final_response": "ok"},
                interrupt=lambda *_a, **_k: None,
            )
            return agent

        adapter._create_agent = _spy
        return real

    @pytest.mark.asyncio
    async def test_declared_key_selects_the_conversation(self, live):
        adapter, db, app = live
        seen = []
        self._capture_agent(adapter, seen)
        _seed(db, "sess-live", key=KEY)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"model": "hermes-agent", "input": "hi"},
                headers=_headers(KEY),
            )
            assert resp.status in (200, 202)

        assert seen and seen[0]["session_id"] == "sess-live"

    @pytest.mark.asyncio
    async def test_explicit_body_session_outranks_the_header_key(self, live):
        adapter, db, app = live
        seen = []
        self._capture_agent(adapter, seen)
        _seed(db, "sess-live", key=KEY)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"model": "hermes-agent", "input": "hi",
                      "session_id": "explicit-session"},
                headers=_headers(KEY),
            )
            assert resp.status in (200, 202)

        assert seen and seen[0]["session_id"] == "explicit-session"
        # KEY still resolves to its own conversation, never the explicit one.
        assert adapter._declared_conversation_session(KEY) == "sess-live"


async def _await_run(adapter, run_id, timeout=10.0):
    """Wait for a /v1/runs worker to finish, so settlement has happened."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if run_id not in adapter._active_run_agents:
            # One more tick so the executor thread's finally can retire.
            await asyncio.sleep(0.05)
            return True
        await asyncio.sleep(0.02)
    return False


def _stub_agent(adapter, session_id, seen):
    """An agent good enough for the REAL _run_agent to drive end to end.

    Replacing `_run_agent` itself cannot exercise its settlement block, which
    is where a caller-local name leaked in and raised `NameError` on every
    opted-in bind while mocked tests stayed green (@andrexibiza on #98811).
    Stubbing one layer lower — `_create_agent` — leaves that block real.
    """
    agent = MagicMock()
    agent.session_id = session_id
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    def _run(**kwargs):
        seen.append(kwargs)
        db = adapter._ensure_session_db()
        # AIAgent._ensure_db_session() creates the row during the turn.
        if db is not None and db.get_session(session_id) is None:
            db.create_session(session_id=session_id, source=SOURCE, model="m")
        return {"final_response": "ok", "messages": [], "api_calls": 1}

    agent.run_conversation.side_effect = _run
    return agent


class TestRealRunAgentSettlement:
    """The real `_run_agent` settlement block, not a stand-in for it."""

    @pytest.mark.asyncio
    async def test_declared_bind_settles_without_raising(self, live):
        adapter, db, app = live
        seen = []
        created = {}

        def _create(**kw):
            created.update(kw)
            return _stub_agent(adapter, kw["session_id"], seen)

        adapter._create_agent = _create

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={"model": "hermes-agent", "input": "hi"},
                headers=_headers(KEY),
            )
            assert resp.status == 200
            body = await resp.json()

        assert body["status"] == "completed"
        sid = created["session_id"]
        # Settlement ran for real: the row is bound and recoverable.
        assert db.get_session(sid)["session_key"] == KEY
        assert adapter._declared_conversation_session(KEY) == sid

    @pytest.mark.asyncio
    async def test_two_replies_settle_on_one_conversation(self, live):
        adapter, db, app = live
        seen = []
        ids = []

        def _create(**kw):
            ids.append(kw["session_id"])
            return _stub_agent(adapter, kw["session_id"], seen)

        adapter._create_agent = _create

        async with TestClient(TestServer(app)) as cli:
            for _ in range(2):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi"},
                    headers=_headers(KEY),
                )
                assert resp.status == 200

        assert len(set(ids)) == 1

    @pytest.mark.asyncio
    async def test_a_chained_turn_never_rebinds_through_real_settlement(self, live):
        adapter, db, app = live
        seen = []

        def _create(**kw):
            return _stub_agent(adapter, kw["session_id"], seen)

        adapter._create_agent = _create

        _seed(db, "sess-A", key=KEY)
        adapter._response_store.put(
            "resp_A",
            {"conversation_history": [], "session_id": "sess-A", "instructions": None},
        )

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={"model": "hermes-agent", "input": "hi",
                      "previous_response_id": "resp_A"},
                headers=_headers(OTHER_KEY),
            )
            assert resp.status == 200

        assert db.get_session("sess-A")["session_key"] == KEY
        assert adapter._declared_conversation_session(OTHER_KEY) is None

    @pytest.mark.asyncio
    async def test_runs_explicit_unkeyed_session_stays_unbound(self, live):
        """The reviewer's exact case: an unkeyed explicit row must not be taken.

        `_run_sync` bound unconditionally, so an explicit body session_id that
        existed with an empty `session_key` was silently adopted by the header
        key even though the header lost precedence.
        """
        adapter, db, app = live
        seen = []

        def _create(**kw):
            return _stub_agent(adapter, kw["session_id"], seen)

        adapter._create_agent = _create

        # Explicit session exists and carries NO routing key.
        db.create_session(session_id="explicit-session", source=SOURCE, model="m")
        _seed(db, "sess-live", key=KEY)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"model": "hermes-agent", "input": "hi",
                      "session_id": "explicit-session"},
                headers=_headers(KEY),
            )
            assert resp.status in (200, 202)
            run_id = (await resp.json()).get("run_id")
            assert run_id
            # /v1/runs answers before the turn settles, so the assertion must
            # wait for the worker's finally to run. Without this the test
            # passes on timing rather than on the precedence gate.
            assert await _await_run(adapter, run_id), "run never settled"

        row = db.get_session("explicit-session")
        assert not (row.get("session_key") or "")
        assert adapter._declared_conversation_session(KEY) == "sess-live"
