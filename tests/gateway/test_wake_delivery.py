"""Tests for gateway/wake.py — background wake delivery.

Two strategies:
* push-capable adapters keep the synthetic MessageEvent / handle_message path;
* the stateless API server (supports_async_delivery=False) self-POSTs
  /v1/chat/completions with the RAW session id in X-Hermes-Session-Id, so the
  wake turn resumes the REAL session instead of a parallel invisible one
  keyed by build_session_key().
"""

import asyncio

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.wake import deliver_wake, adapter_supports_push


class PushAdapter:
    """Default adapter shape — no supports_async_delivery attribute."""

    def __init__(self):
        self.handled = []

    async def handle_message(self, event):
        self.handled.append(event)


class ApiServerLikeAdapter:
    supports_async_delivery = False

    def __init__(self, host="0.0.0.0", port=0, key="test-key", model="hermes"):
        self._host = host
        self._port = port
        self._api_key = key
        self._model_name = model

    async def handle_message(self, event):  # pragma: no cover — must NOT be hit
        raise AssertionError("non-push adapter must not receive handle_message wakes")


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="group",
    )


def test_adapter_supports_push_default_true():
    assert adapter_supports_push(PushAdapter()) is True
    assert adapter_supports_push(ApiServerLikeAdapter()) is False


async def _serve(handler):
    """Spin an in-process aiohttp server on an ephemeral loopback port."""
    from aiohttp import web

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def test_deliver_wake_non_push_self_posts_raw_session_id(monkeypatch):
    """The self-post carries the RAW session id header + bearer auth and a
    single user message with stream=false — the exact entry point real
    gateway turns use."""
    from aiohttp import web

    seen = {}

    async def handler(request):
        seen["session_id"] = request.headers.get("X-Hermes-Session-Id")
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = await request.json()
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(host="0.0.0.0", port=port, key="sekrit")
            await deliver_wake(adapter, text="task done — wake", session_id="raw-sid-42")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert seen["session_id"] == "raw-sid-42"
    assert seen["auth"] == "Bearer sekrit"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"] == [
        {"role": "user", "content": "task done — wake"}
    ]


def test_deliver_wake_retries_429_then_succeeds(monkeypatch):
    """HTTP 429 (max_concurrent_runs cap) is transient — retried with backoff."""
    from aiohttp import web

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return web.json_response({"error": "busy"}, status=429)
        return web.json_response({"choices": []})

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port)
            await deliver_wake(adapter, text="x", session_id="sid")
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert calls["n"] == 2


def test_persist_delegation_delivery_appends_delivery_row(tmp_path):
    """#85957: the delegation completion lands in the session transcript as a
    display_kind=async_delegation_complete delivery row (real SessionDB), and
    NO self-post / agent turn is involved."""
    from pathlib import Path

    from gateway.wake import persist_delegation_delivery
    from hermes_state import SessionDB

    db = SessionDB(db_path=Path(tmp_path) / "state.db")
    sid = "raw-hq-sid"
    db.create_session(sid, source="api_server")
    db.append_message(sid, "user", content="please confirm before writing")
    db.append_message(sid, "assistant", content="awaiting confirmation",
                      finish_reason="stop")

    class DbAdapter(ApiServerLikeAdapter):
        def _ensure_session_db(self):
            return db

    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_x",
        "results": [{"status": "completed"}, {"status": "failed"}],
        "total_duration_seconds": 12.5,
    }
    asyncio.run(persist_delegation_delivery(
        DbAdapter(), text="[ASYNC DELEGATION BATCH COMPLETE — deleg_x]",
        session_id=sid, evt=evt,
    ))

    rows = db.get_messages(sid)
    assert len(rows) == 3
    delivery = rows[-1]
    assert delivery["role"] == "user"
    assert delivery["display_kind"] == "async_delegation_complete"
    meta = delivery["display_metadata"]
    assert meta["delegation_id"] == "deleg_x"
    assert meta["task_count"] == 2
    assert meta["failed_count"] == 1
    assert meta["duration_seconds"] == 12.5


def test_persist_delegation_delivery_raises_without_db():
    """DB unavailable must RAISE so the durable claim is released for retry."""
    from gateway.wake import persist_delegation_delivery

    class NoDbAdapter(ApiServerLikeAdapter):
        def _ensure_session_db(self):
            return None

    with pytest.raises(RuntimeError, match="SessionDB unavailable"):
        asyncio.run(persist_delegation_delivery(
            NoDbAdapter(), text="x", session_id="sid",
        ))



# ---------------------------------------------------------------------------
# Durable wake self-post: idempotency + persist receipt on the DEFAULT-profile
# HTTP path and the SECONDARY-profile in-process path.
# ---------------------------------------------------------------------------


def test_ac_gov_f25_c_7(monkeypatch):
    """DEFAULT-profile HTTP branch: deliver_wake sends the opt-in headers and treats a 2xx without the persist ack as undelivered; no owner_profile."""
    import inspect

    from aiohttp import web

    from gateway import wake

    # signature contract: opt-in idempotency + persist ack, and NO owner_profile (secondary
    # routing reuses the existing profile= parameter).
    params = inspect.signature(wake.deliver_wake).parameters
    assert "idempotency_key" in params and "require_persist_ack" in params
    assert "owner_profile" not in params
    assert "profile" in params

    monkeypatch.setattr(wake, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    state = {"persisted": None, "seen_headers": {}}

    async def handler(request):
        state["seen_headers"] = dict(request.headers)
        resp_headers = {}
        if state["persisted"] is not None:
            resp_headers["X-Hermes-Turn-Persisted"] = state["persisted"]
        return web.json_response({"choices": [{"message": {"content": "ok"}}]}, headers=resp_headers)

    async def run():
        runner, port = await _serve(handler)
        try:
            adapter = ApiServerLikeAdapter(port=port, key="sekrit")

            # 2xx WITHOUT the server-generated persist ack -> undelivered -> raises (the durable
            # caller RETAINS its fence rather than settling on an unpersisted turn).
            state["persisted"] = None
            with pytest.raises(Exception):
                await wake.deliver_wake(
                    adapter, text="t", session_id="s",
                    idempotency_key="idem-1", require_persist_ack=True)
            # the self-post carried the opt-in idempotency + persist-require headers
            assert state["seen_headers"].get("Idempotency-Key") == "idem-1"
            assert state["seen_headers"].get("X-Hermes-Require-Persist") == "1"

            # 2xx WITH X-Hermes-Turn-Persisted:true -> delivered (no raise)
            state["persisted"] = "true"
            await wake.deliver_wake(
                adapter, text="t", session_id="s",
                idempotency_key="idem-1", require_persist_ack=True)
        finally:
            await runner.cleanup()

    asyncio.run(run())


class _InProcAdapter:
    """A non-push adapter whose run_internal_session_turn is the REAL in-process route
    (api_server_runs.run_internal_session_turn) with only the turn-execution machinery stubbed to a
    counter — so the opt-in persisted-result de-dup and the persist-return contract are exercised,
    not re-implemented. No HTTP is ever made."""

    supports_async_delivery = False

    def __init__(self):
        self._model_name = "hermes"
        self.turn_runs = 0
        self.http_calls = 0
        self.ran_in_process = False
        self._persisted = None

    # knobs the test drives
    def set_turn_persisted(self, value):
        self._persisted = value

    def reset_counts(self):
        self.turn_runs = 0
        self.http_calls = 0
        self.ran_in_process = False

    # seams the real _run_once_persist consumes (all cheap / hermetic)
    def _draining_response(self):
        return None

    def _concurrency_limited_response(self):
        return None

    async def _ensure_session_db_async(self):
        return None  # -> _resolve_live_session_id falls open to the raw session id

    async def _get_existing_session_or_404(self, session_id):
        from types import SimpleNamespace
        return SimpleNamespace(id=session_id), None

    async def _conversation_history_for_session(self, session_id):
        return []

    def _select_request_route(self, body, *, session_id, gateway_session_key, model_alias):
        return {}, {}, None

    async def _run_agent(self, **kwargs):
        self.turn_runs += 1
        self.ran_in_process = True
        return {"turn_persisted": self._persisted}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async def handle_message(self, event):  # pragma: no cover — must NOT be hit
        raise AssertionError("non-push adapter must not receive handle_message wakes")

    async def run_internal_session_turn(self, *, session_id, text, profile,
                                        notification_category="result", idempotency_key=None):
        from gateway.platforms import api_server as _api
        from gateway.platforms import api_server_runs
        return await api_server_runs.run_internal_session_turn(
            self, session_id=session_id, text=text, profile=profile,
            notification_category=notification_category, idempotency_key=idempotency_key,
            _api_server=_api)


def test_ac_gov_f25_c_8(monkeypatch):
    """SECONDARY-profile in-process branch: deliver_wake gates on run_internal_session_turn's persist result AND dedups by idempotency_key (no HTTP, no header)."""
    from gateway import wake
    from gateway.platforms.api_server import _IdempotencyCache

    # isolate the process-global cache the in-process de-dup reuses
    monkeypatch.setattr("gateway.platforms.api_server._idem_cache", _IdempotencyCache())
    monkeypatch.setattr(wake, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    adapter = _InProcAdapter()

    # not persisted -> raise (caller retains the fence); an unpersisted same-key retry RE-RUNS
    adapter.set_turn_persisted(False)
    for _ in range(2):
        with pytest.raises(Exception):
            asyncio.run(wake.deliver_wake(
                adapter, text="t", session_id="s", profile="owner-b",
                idempotency_key="idem-2", require_persist_ack=True))
    assert adapter.ran_in_process is True
    assert adapter.http_calls == 0
    assert adapter.turn_runs == 2  # unpersisted retry re-ran

    # persisted -> returns; a persisted same-key retry is deduped (turn runs once)
    adapter.reset_counts()
    adapter.set_turn_persisted(True)
    for _ in range(2):
        asyncio.run(wake.deliver_wake(
            adapter, text="t", session_id="s", profile="owner-b",
            idempotency_key="idem-3", require_persist_ack=True))
    assert adapter.turn_runs == 1  # persisted same-key retry deduped


def test_in_process_wake_fingerprints_content(monkeypatch):
    """The in-process de-dup keys on the wake content, not a constant: a same-key retry with
    DIFFERENT text re-runs (never silently served the earlier receipt); an identical retry dedups."""
    from gateway import wake
    from gateway.platforms.api_server import _IdempotencyCache

    monkeypatch.setattr("gateway.platforms.api_server._idem_cache", _IdempotencyCache())
    monkeypatch.setattr(wake, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    adapter = _InProcAdapter()
    adapter.set_turn_persisted(True)

    asyncio.run(wake.deliver_wake(adapter, text="first", session_id="s", profile="owner-b",
                                  idempotency_key="idem-x", require_persist_ack=True))
    asyncio.run(wake.deliver_wake(adapter, text="second", session_id="s", profile="owner-b",
                                  idempotency_key="idem-x", require_persist_ack=True))
    assert adapter.turn_runs == 2

    adapter.reset_counts()
    for _ in range(2):
        asyncio.run(wake.deliver_wake(adapter, text="same", session_id="s", profile="owner-b",
                                      idempotency_key="idem-y", require_persist_ack=True))
    assert adapter.turn_runs == 1
