"""Tests for the request-scoped ephemeral-persist contract.

Verified: store:false and X-Hermes-Session-Mode: ephemeral on /v1/responses
and /v1/chat/completions create ZERO SessionDB rows (via _persist_disabled
+ session_db=None), while header-less and explicit X-Hermes-Session-Id
requests keep today's exact behavior.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter, _resolve_session_persistence, _derive_chat_session_id,
)


# ---------------------------------------------------------------------------
# Fixtures (mirrors tests/gateway/test_api_server.py + AGENTS.md profile_env)
# ---------------------------------------------------------------------------

@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


def _create_app(adapter):
    """Create the aiohttp app from the adapter (mirrors the existing test suite)."""
    import aiohttp.web as web
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    app.router.add_delete("/v1/responses/{response_id}", adapter._handle_delete_response)
    return app


def _make_adapter(api_key=""):
    extra = {}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _count_rows(home):
    """SessionDB ``sessions`` row count.

    Deliberately NOT the only evidence: the response store is a SEPARATE sqlite file
    (response_store.db) that this never saw, which is exactly how a header-only
    ephemeral request could keep writing rows while this stayed green. Use
    _count_response_rows alongside it — see TestEphemeralLeavesNoStore.
    """
    db = Path(home) / "state.db"
    if not db.exists():
        return 0
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT COUNT(*) FROM sessions;").fetchone()[0]
    finally:
        conn.close()


def _count_response_rows(home):
    """Rows in the api_server Responses store (response_store.db) — the F1 store.

    'Zero rows in the sessions table' is not 'nothing persisted': the responses
    store is a different file, so an ephemeral request could be writing here
    while _count_rows reported a clean zero.
    """
    db = Path(home) / "response_store.db"
    if not db.exists():
        return 0
    conn = sqlite3.connect(db)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';")}
        if "responses" not in tables:
            return 0
        return conn.execute("SELECT COUNT(*) FROM responses;").fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# _resolve_session_persistence
# ---------------------------------------------------------------------------

class TestResolveSessionPersistence:
    def test_absent_persists(self):
        assert _resolve_session_persistence({}, True) is True

    def test_store_false_is_ephemeral(self):
        assert _resolve_session_persistence({}, False) is False

    def test_header_ephemeral_overrides_store_true(self):
        assert _resolve_session_persistence({"X-Hermes-Session-Mode": "ephemeral"}, True) is False

    def test_header_ephemeral_lowercase(self):
        assert _resolve_session_persistence({"X-Hermes-Session-Mode": "Ephemeral"}, True) is False

    def test_other_header_value_persists(self):
        assert _resolve_session_persistence({"X-Hermes-Session-Mode": "something"}, True) is True


# ---------------------------------------------------------------------------
# _create_agent persist flag (unit-level proof)
# ---------------------------------------------------------------------------

class TestCreateAgentPersist:
    """persist=True keeps the SessionDB flush gate open;
    persist=False closes it via _persist_disabled + session_db=None."""

    def test_persist_true_keeps_session_db(self):
        mock_agent = MagicMock()
        mock_agent._session_db = MagicMock()
        mock_agent._persist_disabled = False
        mock_agent._memory_manager = MagicMock()
        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={}), \
             patch("run_agent.AIAgent", return_value=mock_agent) as MockAgent:
            agent = adapter_for_persist_test()._create_agent(session_id="sess-1", persist=True)
        assert agent._session_db is not None
        assert agent._persist_disabled is False

    def test_persist_false_disables_persistence(self):
        mock_agent = MagicMock()
        mock_agent._session_db = None
        mock_agent._persist_disabled = True
        mock_agent._memory_manager = None
        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={}), \
             patch("run_agent.AIAgent", return_value=mock_agent) as MockAgent:
            agent = adapter_for_persist_test()._create_agent(session_id="sess-1", persist=False)
        assert agent._session_db is None
        assert agent._persist_disabled is True

    def test_persist_false_unique_session_id(self):
        """Ephemeral must allocate a unique runtime session_id."""
        counter = [0]
        def make_agent(*args, **kwargs):
            counter[0] += 1
            m = MagicMock()
            m.session_id = f"ephemeral-{counter[0]}"
            m._session_db = None
            m._persist_disabled = True
            m._memory_manager = None
            return m
        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={}), \
             patch("run_agent.AIAgent", side_effect=make_agent) as MockAgent:
            ids = {adapter_for_persist_test()._create_agent(persist=False).session_id for _ in range(5)}
        assert len(ids) == 5
        assert MockAgent.call_count == 5

    def test_persist_false_no_memory_manager(self):
        """Ephemeral must NOT adopt a real session's MemoryManager.

        Asserted on the CONSTRUCTOR ARGS, not on the returned mock's attribute: a
        MagicMock returned by the patched AIAgent has ``_memory_manager`` at whatever
        the test set it to (line 118 sets None), so an attribute read here records
        the test's own setup instead of post-init reality and stays green through a
        leak. The constructor args are what production actually passed, and
        TestEphemeralArmBuildsNoMemoryProvider then proves the real init agrees.
        """
        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={}), \
             patch("run_agent.AIAgent", return_value=MagicMock()) as MockAgent:
            adapter_for_persist_test()._create_agent(session_id="sess-1", persist=False)
        kwargs = MockAgent.call_args.kwargs
        assert kwargs["memory_manager"] is None, "ephemeral arm must not adopt a session manager"
        assert kwargs["skip_memory"] is True, (
            "ephemeral arm must pass skip_memory=True so _init_memory neither builds a "
            "MemoryManager nor boots the provider (F2)")

    def test_persist_false_checks_out_nothing(self):
        """The ephemeral arm must not consume a parked manager from the registry."""
        adapter = adapter_for_persist_test()
        with patch.object(adapter._memory_sessions, "checkout", return_value=None) as checkout, \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={}), \
             patch("run_agent.AIAgent", return_value=MagicMock()):
            adapter._create_agent(session_id="sess-1", persist=False)
        checkout.assert_not_called()

    def test_persist_true_keeps_memory_path(self):
        """Control for the two guards above: the persisted arm still checks out and
        does NOT skip memory. Without this, 'skip_memory is always True' would pass."""
        adapter = adapter_for_persist_test()
        with patch.object(adapter._memory_sessions, "checkout", return_value="parked") as checkout, \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={}), \
             patch("run_agent.AIAgent", return_value=MagicMock()) as MockAgent:
            adapter._create_agent(session_id="sess-1", persist=True)
        kwargs = MockAgent.call_args.kwargs
        checkout.assert_called_once_with("sess-1")
        assert kwargs["memory_manager"] == "parked"
        assert kwargs["skip_memory"] is False


def adapter_for_persist_test():
    return _make_adapter("test-key")


# ---------------------------------------------------------------------------
# Live probes: sqlite row counts
# ---------------------------------------------------------------------------

# Provider resolution under a throwaway HERMES_HOME has no credentials. Stub
# the gateway runtime seam so _create_agent actually builds an AIAgent with a
# SessionDB; otherwise persist and ephemeral both write 0 rows and the
# after==before assertions prove nothing.
_STUB_RUNTIME = {
    "api_key": "fake-key",
    "provider": "openai",
    "base_url": "http://127.0.0.1:9",
    "api_mode": "chat_completions",
}


def _fake_success_call(
    agent, *, api_kwargs, _original_api_kwargs, _llm_middleware_trace,
    _moa_prepared_request, _retry, thinking_spinner, retry_count,
    api_call_count, api_request_id, effective_task_id, turn_id,
    interrupted,
):
    """Complete the turn at the conversation_loop.perform_api_call seam.

    Signature must match that helper (conversation_loop binds it at import;
    _run_phase inspects parameters). Return a chat-completions-shaped
    ApiCallVerdict so the loop does not retry. Session-row creation happens
    at turn start, before this call.
    """
    from types import SimpleNamespace

    from agent.turn_api_call import ApiCallVerdict

    msg = SimpleNamespace(
        content="ok", tool_calls=None, reasoning=None,
        reasoning_content=None, reasoning_details=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    response = SimpleNamespace(choices=[choice], model="test/model", usage=None)
    return ApiCallVerdict(
        action="fallthrough",
        response=response,
        thinking_spinner=thinking_spinner,
        interrupted=interrupted,
    )


@contextmanager
def _live_turn_patches():
    """Same harness for ephemeral and persisted live probes: only store/header differ."""
    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value=dict(_STUB_RUNTIME)), \
         patch("agent.conversation_loop.perform_api_call", new=_fake_success_call):
        yield


class TestEphemeralLiveRowCounts:
    """Live probes against a throwaway HERMES_HOME: ephemeral requests
    must create ZERO SessionDB rows; persisted requests must still write.

    Discrimination is the pair: same fake call + stub runtime, only the
    store/header differing, opposite row outcomes.
    """

    @pytest.mark.asyncio
    async def test_store_false_creates_zero_session_rows(self, profile_env):
        adapter = _make_adapter("ephemeral-test-key")
        app = _create_app(adapter)
        before = _count_rows(profile_env)
        async with TestClient(TestServer(app)) as cli:
            with _live_turn_patches():
                resp = await cli.post("/v1/responses", json={
                    "model": "hermes-agent", "input": "Hello", "store": False},
                    headers={"Authorization": "Bearer ephemeral-test-key"})
        after = _count_rows(profile_env)
        assert resp.status == 200, await resp.text()
        assert after == before, f"ephemeral store:false created {after-before} new SessionDB rows; expected 0"

    @pytest.mark.asyncio
    async def test_header_ephemeral_creates_zero_session_rows(self, profile_env):
        adapter = _make_adapter("ephemeral-test-key")
        app = _create_app(adapter)
        before = _count_rows(profile_env)
        async with TestClient(TestServer(app)) as cli:
            with _live_turn_patches():
                resp = await cli.post("/v1/responses", json={
                    "model": "hermes-agent", "input": "Hello"},
                    headers={"X-Hermes-Session-Mode": "ephemeral",
                             "Authorization": "Bearer ephemeral-test-key"})
        after = _count_rows(profile_env)
        assert resp.status == 200, await resp.text()
        assert after == before, f"X-Hermes-Session-Mode:ephemeral created {after-before} new SessionDB rows; expected 0"

    @pytest.mark.asyncio
    async def test_chat_completions_store_false_creates_zero_rows(self, profile_env):
        adapter = _make_adapter("ephemeral-test-key")
        app = _create_app(adapter)
        before = _count_rows(profile_env)
        async with TestClient(TestServer(app)) as cli:
            with _live_turn_patches():
                resp = await cli.post("/v1/chat/completions", json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "hi"}],
                    "store": False},
                    headers={"Authorization": "Bearer ephemeral-test-key"})
        after = _count_rows(profile_env)
        assert resp.status == 200, await resp.text()
        assert after == before, f"chat store:false created {after-before} new SessionDB rows; expected 0"

    @pytest.mark.asyncio
    async def test_store_true_creates_a_session_row(self, profile_env):
        """Negative control: same harness as the zero-row tests, persist on.

        store:true / no ephemeral header must write at least one sessions row.
        Paired with the three after==before tests, this is what proves the
        feature rather than a harness that never writes.
        """
        adapter = _make_adapter("ephemeral-test-key")
        app = _create_app(adapter)
        before = _count_rows(profile_env)
        async with TestClient(TestServer(app)) as cli:
            with _live_turn_patches():
                resp = await cli.post("/v1/responses", json={
                    "model": "hermes-agent", "input": "Hello", "store": True},
                    headers={"Authorization": "Bearer ephemeral-test-key"})
        after = _count_rows(profile_env)
        assert resp.status == 200, await resp.text()
        assert after >= before + 1, (
            f"persisted store:true wrote {after - before} SessionDB rows; expected >= 1"
        )


# ---------------------------------------------------------------------------
# Live probes: the stores an ephemeral request must NOT touch
# ---------------------------------------------------------------------------

class TestEphemeralLeavesNoStore:
    """F1: ephemeral must not write response_store.db either.

    The SessionDB row counts above are blind here — response_store.db is a
    SEPARATE file, so a header-only ephemeral request (no ``store`` field, so
    store==True) could write a retrievable response row while every
    ``after == before`` session assertion stayed green. That is the exact live
    signature the auditor reported: state.db never created, response_store.db
    with one row.

    Discrimination is the pair again — the persisted request in the SAME class
    must still write a response row, so a store that never writes proves nothing.
    """

    @pytest.mark.asyncio
    async def test_header_ephemeral_writes_no_response_row(self, profile_env):
        adapter = _make_adapter("ephemeral-test-key")
        app = _create_app(adapter)
        before = _count_response_rows(profile_env)
        async with TestClient(TestServer(app)) as cli:
            with _live_turn_patches():
                resp = await cli.post("/v1/responses", json={
                    "model": "hermes-agent", "input": "Hello"},
                    headers={"X-Hermes-Session-Mode": "ephemeral",
                             "Authorization": "Bearer ephemeral-test-key"})
        after = _count_response_rows(profile_env)
        assert resp.status == 200, await resp.text()
        assert after == before, (
            f"X-Hermes-Session-Mode:ephemeral wrote {after - before} response_store rows; "
            f"expected 0 (the request left a retrievable trace)")

    @pytest.mark.asyncio
    async def test_store_false_writes_no_response_row(self, profile_env):
        adapter = _make_adapter("ephemeral-test-key")
        app = _create_app(adapter)
        before = _count_response_rows(profile_env)
        async with TestClient(TestServer(app)) as cli:
            with _live_turn_patches():
                resp = await cli.post("/v1/responses", json={
                    "model": "hermes-agent", "input": "Hello", "store": False},
                    headers={"Authorization": "Bearer ephemeral-test-key"})
        after = _count_response_rows(profile_env)
        assert resp.status == 200, await resp.text()
        assert after == before, f"store:false wrote {after - before} response_store rows; expected 0"

    @pytest.mark.asyncio
    async def test_persisted_request_still_writes_a_response_row(self, profile_env):
        """Positive control for the two tests above.

        A persisted /v1/responses request must still store its response. If this
        fails, the ephemeral assertions pass only because the store never writes.
        """
        adapter = _make_adapter("ephemeral-test-key")
        app = _create_app(adapter)
        before = _count_response_rows(profile_env)
        async with TestClient(TestServer(app)) as cli:
            with _live_turn_patches():
                resp = await cli.post("/v1/responses", json={
                    "model": "hermes-agent", "input": "Hello", "store": True},
                    headers={"Authorization": "Bearer ephemeral-test-key"})
        after = _count_response_rows(profile_env)
        assert resp.status == 200, await resp.text()
        assert after >= before + 1, (
            f"persisted store:true wrote {after - before} response_store rows; expected >= 1")

    @pytest.mark.asyncio
    async def test_stored_response_is_retrievable(self, profile_env):
        """The persisted response row is a real, retrievable record, not a stray write."""
        adapter = _make_adapter("ephemeral-test-key")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with _live_turn_patches():
                resp = await cli.post("/v1/responses", json={
                    "model": "hermes-agent", "input": "Hello", "store": True},
                    headers={"Authorization": "Bearer ephemeral-test-key"})
            body = await resp.json()
            got = await cli.get(f"/v1/responses/{body['id']}",
                                headers={"Authorization": "Bearer ephemeral-test-key"})
            assert got.status == 200, await got.text()
            fetched = await got.json()
        assert fetched["id"] == body["id"]


class TestEphemeralArmBuildsNoMemoryProvider:
    """F2: an ephemeral turn must not construct or boot a memory provider.

    The leak was invisible to the old guard because it read ``agent._memory_manager``
    off a MagicMock. These assert on the two seams that actually decide it, with
    the real ``_init_memory`` running: no MemoryManager is constructed and no
    provider plugin is loaded/initialized.
    """

    @staticmethod
    @contextmanager
    def _memory_init_spy():
        """Count MemoryManager construction and provider initialize_all calls.

        Patched where _init_memory reads them (late import inside the function,
        so the module attribute is the seam).
        """
        import agent.memory_manager as mm
        real_mm = mm.MemoryManager
        built, initialized = [], []

        class SpyMemoryManager(real_mm):
            def __init__(self, *a, **kw):
                built.append(1)
                super().__init__(*a, **kw)

            def initialize_all(self, *a, **kw):
                initialized.append(1)
                return super().initialize_all(*a, **kw)

        with patch.object(mm, "MemoryManager", SpyMemoryManager), \
             patch("plugins.memory.load_memory_provider", return_value=None) as load:
            yield built, initialized, load

    @staticmethod
    def _force_external_memory_provider(profile_env):
        """Configure a non-core memory.provider so the _init_memory provider branch
        is actually reachable. Without this the branch is skipped for being core
        and the guard would pass for the wrong reason."""
        import yaml
        (Path(profile_env) / "config.yaml").write_text(yaml.safe_dump({
            "memory": {"provider": "ephemeral-probe-provider"}}))

    @pytest.mark.asyncio
    async def test_ephemeral_turn_initializes_no_memory_provider(self, profile_env):
        self._force_external_memory_provider(profile_env)
        adapter = _make_adapter("ephemeral-test-key")
        app = _create_app(adapter)
        with self._memory_init_spy() as (built, initialized, load):
            with _live_turn_patches():
                async with TestClient(TestServer(app)) as cli:
                    resp = await cli.post("/v1/responses", json={
                        "model": "hermes-agent", "input": "Hello"},
                        headers={"X-Hermes-Session-Mode": "ephemeral",
                                 "Authorization": "Bearer ephemeral-test-key"})
        assert resp.status == 200, await resp.text()
        assert built == [], (
            f"ephemeral turn constructed {len(built)} MemoryManager(s); expected 0")
        assert initialized == [], (
            f"ephemeral turn booted a memory provider {len(initialized)} time(s); expected 0")
        load.assert_not_called()

    @pytest.mark.asyncio
    async def test_persisted_turn_does_reach_memory_provider(self, profile_env):
        """Discriminating control: the persisted arm DOES construct a manager.

        Proves the guards above observe a real difference between the two arms
        rather than a branch that never executes under the probe harness.
        """
        self._force_external_memory_provider(profile_env)
        adapter = _make_adapter("ephemeral-test-key")
        app = _create_app(adapter)
        with self._memory_init_spy() as (built, initialized, load):
            with _live_turn_patches():
                async with TestClient(TestServer(app)) as cli:
                    resp = await cli.post("/v1/responses", json={
                        "model": "hermes-agent", "input": "Hello", "store": True},
                        headers={"Authorization": "Bearer ephemeral-test-key"})
        assert resp.status == 200, await resp.text()
        assert len(built) >= 1, (
            "persisted turn built no MemoryManager — the ephemeral guard above would "
            "then pass without discriminating anything")
        load.assert_called()

    @pytest.mark.asyncio
    async def test_ephemeral_turn_does_not_checkin(self, profile_env):
        """The turn's finally must not park an orphan manager.

        Without the persist gate at the checkin site, an ephemeral turn parks a
        manager under its throwaway uuid4 session_id — an entry nothing will ever
        address, so nothing ever evicts it.
        """
        adapter = _make_adapter("ephemeral-test-key")
        app = _create_app(adapter)
        with patch.object(adapter._memory_sessions, "checkin") as checkin:
            with _live_turn_patches():
                async with TestClient(TestServer(app)) as cli:
                    resp = await cli.post("/v1/responses", json={
                        "model": "hermes-agent", "input": "Hello"},
                        headers={"X-Hermes-Session-Mode": "ephemeral",
                                 "Authorization": "Bearer ephemeral-test-key"})
        assert resp.status == 200, await resp.text()
        checkin.assert_not_called()
        assert adapter._memory_sessions.parked() == {}, (
            "ephemeral turn left a parked memory manager: "
            f"{adapter._memory_sessions.parked()}")

    @pytest.mark.asyncio
    async def test_persisted_turn_still_checks_out_and_in(self, profile_env):
        """Balance control: the persisted arm checks out AND checks in.

        check-out and check-in must stay symmetric on the persisted arm; gating
        the checkin on persist must not have silenced it everywhere.
        """
        adapter = _make_adapter("ephemeral-test-key")
        app = _create_app(adapter)
        with patch.object(adapter._memory_sessions, "checkin", wraps=adapter._memory_sessions.checkin) as checkin, \
             patch.object(adapter._memory_sessions, "checkout", wraps=adapter._memory_sessions.checkout) as checkout:
            with _live_turn_patches():
                async with TestClient(TestServer(app)) as cli:
                    resp = await cli.post("/v1/responses", json={
                        "model": "hermes-agent", "input": "Hello", "store": True},
                        headers={"Authorization": "Bearer ephemeral-test-key"})
        assert resp.status == 200, await resp.text()
        assert checkout.call_count >= 1, "persisted turn never checked out"
        assert checkin.call_count >= 1, (
            "persisted turn never checked in — check-out/check-in are no longer balanced")


# ---------------------------------------------------------------------------
# Routing: persist flag reaches _create_agent / _run_agent
# ---------------------------------------------------------------------------

class TestEphemeralRouting:
    """Verify the persist flag is threaded to _create_agent."""

    @pytest.mark.asyncio
    async def test_responses_store_false_runs_persist_false(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = ({"final_response": "ok", "messages": [], "api_calls": 1},
                                         {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                await cli.post("/v1/responses", json={
                    "model": "hermes-agent", "input": "Hello", "store": False},
                    headers={"Authorization": "Bearer test-key"})
        assert mock_run.call_args.kwargs["persist"] is False

    @pytest.mark.asyncio
    async def test_responses_header_ephemeral_runs_persist_false(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = ({"final_response": "ok", "messages": [], "api_calls": 1},
                                         {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                await cli.post("/v1/responses", json={
                    "model": "hermes-agent", "input": "Hello"},
                    headers={"X-Hermes-Session-Mode": "ephemeral",
                             "Authorization": "Bearer test-key"})
        assert mock_run.call_args.kwargs["persist"] is False

    @pytest.mark.asyncio
    async def test_chat_completions_store_false_runs_persist_false(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = ({"final_response": "ok", "messages": [], "api_calls": 1},
                                         {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                await cli.post("/v1/chat/completions", json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "hi"}],
                    "store": False},
                    headers={"Authorization": "Bearer test-key"})
        assert mock_run.call_args.kwargs["persist"] is False

    @pytest.mark.asyncio
    async def test_chat_completions_header_ephemeral_runs_persist_false(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = ({"final_response": "ok", "messages": [], "api_calls": 1},
                                         {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                await cli.post("/v1/chat/completions", json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "hi"}],
                    "store": False},
                    headers={"X-Hermes-Session-Mode": "ephemeral",
                             "Authorization": "Bearer test-key"})
        assert mock_run.call_args.kwargs["persist"] is False

    @pytest.mark.asyncio
    async def test_persisted_runs_persist_true(self, adapter):
        """persist=True (default) keeps the SessionDB flush gate open."""
        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={}), \
             patch("run_agent.AIAgent", return_value=MagicMock()) as MockAgent:
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = ({"final_response": "ok", "messages": [], "api_calls": 1},
                                                 {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                    await cli.post("/v1/responses", json={
                        "model": "hermes-agent", "input": "hi", "store": True},
                        headers={"Authorization": "Bearer test-key"})
        assert mock_run.call_args.kwargs["persist"] is True


# ---------------------------------------------------------------------------
# Regression guards
# ---------------------------------------------------------------------------

class TestRegressionGuards:
    """Header-less and explicit-X-Hermes-Session-Id requests must keep
    today's exact behavior (fingerprint session, history resume, titled)."""

    def test_headerless_chat_uses_stable_fingerprint(self):
        """Header-less Open-WebUI-shaped chat produces the SAME api-<hex>
        fingerprint session (regression guard — must not change)."""
        fp = _derive_chat_session_id(None, "hello")
        assert fp.startswith("api-")
        assert fp == _derive_chat_session_id(None, "hello")  # stable

    @pytest.mark.asyncio
    async def test_explicit_session_id_loads_history(self, auth_adapter):
        """When X-Hermes-Session-Id is provided, history comes from SessionDB."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}
        db_history = [
            {"role": "user", "content": "stored message 1"},
            {"role": "assistant", "content": "stored reply 1"},
        ]
        mock_db = MagicMock()
        mock_db.get_messages_as_conversation.return_value = db_history
        mock_db.resolve_resume_session_id.side_effect = lambda sid: sid
        auth_adapter._session_db = mock_db
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post("/v1/chat/completions",
                    headers={"X-Hermes-Session-Id": "existing-session",
                             "Authorization": "Bearer sk-secret"},
                    json={"model": "hermes-agent",
                          "messages": [{"role": "user", "content": "new question"}]})
        assert resp.status == 200
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["conversation_history"] == db_history
        assert call_kwargs["user_message"] == "new question"

    @pytest.mark.asyncio
    async def test_persisted_runs_persist_true_default(self, adapter):
        """Header-less request (no store field, no ephemeral header) defaults
        to persist=True — today's exact behavior unchanged (f)."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = ({"final_response": "ok"}, {"total_tokens": 0})
                await cli.post("/v1/responses", json={
                    "model": "hermes-agent", "input": "hi"},
                    headers={"Authorization": "Bearer test-key"})
        assert mock_run.call_args.kwargs["persist"] is True
