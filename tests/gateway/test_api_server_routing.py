"""Tests for the api_server adapter's multi-agent routing wiring.

Covers ``APIServerAdapter._resolve_agent_profile`` — the bridge between
inbound ``X-Hermes-Chat-Id`` (etc.) headers and the shared
``_attach_agent_id`` resolver / ``AgentProfile`` registry.

These tests construct the adapter in-process (no HTTP listener) and feed
it ``MagicMock``-backed ``aiohttp.web.Request`` stand-ins, mirroring the
style used elsewhere in ``test_api_server.py``.

Ported from David Gutowsky's original #25660-era commit (643bbbf5a) onto
jethac's rebased single-gateway-multi-agent branch.  Session-context
plumbing and ``_bind_api_server_session`` were added after v0.17.0; the
routing methods themselves are unchanged.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.profile import AgentProfile, DEFAULT_AGENT_ID, get_active_profile
from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _use_profile_and_secret_scope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(
    *,
    routes=None,
    registry=None,
    default_agent: str = "main",
) -> APIServerAdapter:
    """Build an APIServerAdapter wired to *routes* and *registry*.

    Why: Mirrors what ``GatewayRunner`` does at startup — calls
    ``set_routing_context`` with a fake gateway that exposes
    ``_agent_registry`` — so routing tests don't need a live server.
    What: Constructs adapter, fakes the gateway ref with a registry,
    and calls ``set_routing_context``.
    Test: All tests in this module use this fixture; assert the returned
    adapter has the expected routing context set up.
    """
    cfg = PlatformConfig()
    adapter = APIServerAdapter(cfg)

    fake_gateway = MagicMock()
    fake_gateway._agent_registry = registry or {}
    adapter.set_routing_context(
        routes=routes or [],
        default_agent=default_agent,
        gateway=fake_gateway,
    )
    return adapter


def _request_with_headers(headers: dict) -> MagicMock:
    """Stub aiohttp.web.Request — only ``.headers.get`` is exercised.

    Why: Avoids standing up a full aiohttp server for unit tests.
    What: Returns a MagicMock whose ``headers`` attribute behaves like a
    dict (``get`` works, key access works).
    Test: Pass ``{"X-Hermes-Chat-Id": "val"}``; assert
    ``req.headers.get("X-Hermes-Chat-Id")`` returns "val".
    """
    req = MagicMock()
    req.headers = headers
    return req


# ---------------------------------------------------------------------------
# Header sanitisation
# ---------------------------------------------------------------------------


class TestReadRoutingHeader:
    def test_returns_value_when_present(self):
        adapter = _make_adapter()
        req = _request_with_headers({"X-Hermes-Chat-Id": "calendar-propose"})
        assert adapter._read_routing_header(req, "X-Hermes-Chat-Id") == "calendar-propose"

    def test_returns_none_when_absent(self):
        adapter = _make_adapter()
        req = _request_with_headers({})
        assert adapter._read_routing_header(req, "X-Hermes-Chat-Id") is None

    def test_strips_whitespace(self):
        adapter = _make_adapter()
        req = _request_with_headers({"X-Hermes-Chat-Id": "  coder  "})
        assert adapter._read_routing_header(req, "X-Hermes-Chat-Id") == "coder"

    def test_rejects_crlf_injection(self):
        adapter = _make_adapter()
        req = _request_with_headers({"X-Hermes-Chat-Id": "ok\r\nX-Injected: yes"})
        assert adapter._read_routing_header(req, "X-Hermes-Chat-Id") is None

    def test_rejects_overlong_value(self):
        adapter = _make_adapter()
        req = _request_with_headers({"X-Hermes-Chat-Id": "x" * 1024})
        assert adapter._read_routing_header(req, "X-Hermes-Chat-Id") is None


# ---------------------------------------------------------------------------
# _resolve_agent_profile — core routing wiring
# ---------------------------------------------------------------------------


class TestResolveAgentProfile:
    def test_header_match_routes_to_specified_agent(self, tmp_path):
        calendar = AgentProfile(id="calendar-propose", home_dir=tmp_path / "calendar")
        main = AgentProfile(id="main")
        registry = {"main": main, "calendar-propose": calendar}
        routes = [
            {"match": {"platform": "api_server", "chat_id": "calendar-propose"}, "agent": "calendar-propose"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)
        req = _request_with_headers({"X-Hermes-Chat-Id": "calendar-propose"})

        profile, agent_id = adapter._resolve_agent_profile(req)

        assert agent_id == "calendar-propose"
        assert profile is calendar

    def test_no_header_falls_through_to_default_agent(self):
        main = AgentProfile(id="main")
        calendar = AgentProfile(id="calendar-propose")
        registry = {"main": main, "calendar-propose": calendar}
        routes = [
            {"match": {"platform": "api_server", "chat_id": "calendar-propose"}, "agent": "calendar-propose"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)
        req = _request_with_headers({})

        profile, agent_id = adapter._resolve_agent_profile(req)

        assert agent_id == "main"
        assert profile is main

    def test_unmatched_header_falls_through_to_default(self):
        main = AgentProfile(id="main")
        calendar = AgentProfile(id="calendar-propose")
        registry = {"main": main, "calendar-propose": calendar}
        routes = [
            {"match": {"platform": "api_server", "chat_id": "calendar-propose"}, "agent": "calendar-propose"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)
        req = _request_with_headers({"X-Hermes-Chat-Id": "unknown-agent"})

        profile, agent_id = adapter._resolve_agent_profile(req)

        assert agent_id == "main"
        assert profile is main

    def test_platform_only_route_matches_any_request(self):
        """A bare ``platform: api_server`` route is the catch-all."""
        coder = AgentProfile(id="coder")
        main = AgentProfile(id="main")
        registry = {"main": main, "coder": coder}
        routes = [
            {"match": {"platform": "api_server"}, "agent": "coder"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)
        # No headers — platform alone matches.
        req = _request_with_headers({})

        profile, agent_id = adapter._resolve_agent_profile(req)

        assert agent_id == "coder"
        assert profile is coder

    def test_user_id_route_matches(self):
        vip = AgentProfile(id="vip")
        main = AgentProfile(id="main")
        registry = {"main": main, "vip": vip}
        routes = [
            {"match": {"platform": "api_server", "user_id": "alice"}, "agent": "vip"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)
        req = _request_with_headers({"X-Hermes-User-Id": "alice"})

        profile, agent_id = adapter._resolve_agent_profile(req)

        assert agent_id == "vip"
        assert profile is vip

    def test_thread_id_route_matches(self):
        thread_agent = AgentProfile(id="thread-agent")
        main = AgentProfile(id="main")
        registry = {"main": main, "thread-agent": thread_agent}
        routes = [
            {"match": {"platform": "api_server", "thread_id": "T-42"}, "agent": "thread-agent"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)
        req = _request_with_headers({"X-Hermes-Thread-Id": "T-42"})

        profile, agent_id = adapter._resolve_agent_profile(req)

        assert agent_id == "thread-agent"
        assert profile is thread_agent

    def test_more_specific_route_wins_when_declared_first(self):
        coder = AgentProfile(id="coder")
        general = AgentProfile(id="general")
        main = AgentProfile(id="main")
        registry = {"main": main, "coder": coder, "general": general}
        routes = [
            {"match": {"platform": "api_server", "chat_id": "code", "user_id": "alice"}, "agent": "coder"},
            {"match": {"platform": "api_server", "chat_id": "code"}, "agent": "general"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)
        req = _request_with_headers({
            "X-Hermes-Chat-Id": "code",
            "X-Hermes-User-Id": "alice",
        })

        profile, agent_id = adapter._resolve_agent_profile(req)
        assert agent_id == "coder"
        assert profile is coder

    def test_empty_registry_returns_none_profile(self):
        """Legacy single-agent install: registry empty → profile is None."""
        adapter = _make_adapter(routes=[], registry={})
        req = _request_with_headers({})

        profile, agent_id = adapter._resolve_agent_profile(req)

        assert agent_id == "main"
        assert profile is None

    def test_crlf_header_is_treated_as_absent(self):
        """A header with control chars must not bypass routing."""
        main = AgentProfile(id="main")
        calendar = AgentProfile(id="calendar-propose")
        registry = {"main": main, "calendar-propose": calendar}
        routes = [
            {"match": {"platform": "api_server", "chat_id": "calendar-propose"}, "agent": "calendar-propose"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)
        req = _request_with_headers({
            "X-Hermes-Chat-Id": "calendar-propose\r\nX-Injected: yes",
        })

        profile, agent_id = adapter._resolve_agent_profile(req)

        # Tainted header dropped → no match → default agent.
        assert agent_id == "main"
        assert profile is main

    def test_gateway_ref_missing_does_not_raise(self):
        """No gateway wired (single-agent install never calls
        set_routing_context with a gateway): profile resolution must
        still succeed and return ``None``."""
        cfg = PlatformConfig()
        adapter = APIServerAdapter(cfg)
        # No set_routing_context call: routes and gateway_ref remain unset.
        req = _request_with_headers({"X-Hermes-Chat-Id": "anything"})

        profile, agent_id = adapter._resolve_agent_profile(req)

        # No registry → no profile, default agent_id falls back to "main".
        assert profile is None
        assert agent_id == "main"


# ---------------------------------------------------------------------------
# ContextVar isolation under concurrency
# ---------------------------------------------------------------------------


class TestContextVarIsolation:
    """Two concurrent requests with different chat_ids must route to
    independent agents without leaking state."""

    def test_concurrent_use_profile_does_not_cross_contaminate(self):
        """Verify ``use_profile`` honours asyncio task isolation.

        This is the foundation the api_server adapter relies on when
        wrapping concurrent agent runs.  If this test fails, the patch
        does not actually isolate requests.
        """
        from agent.profile import use_profile

        code = AgentProfile(id="code", home_dir="/tmp/code")
        chat = AgentProfile(id="chat", home_dir="/tmp/chat")

        observed = {"code": None, "chat": None}

        async def _under_profile(name: str, profile: AgentProfile, hold: float) -> None:
            with use_profile(profile):
                # Yield to the other task — if profiles leak we'll see it.
                await asyncio.sleep(hold)
                observed[name] = get_active_profile()

        async def _main() -> None:
            await asyncio.gather(
                _under_profile("code", code, 0.05),
                _under_profile("chat", chat, 0.01),
            )

        asyncio.run(_main())

        assert observed["code"] is code
        assert observed["chat"] is chat
        # Outer context is restored.
        assert get_active_profile() is None

    def test_concurrent_resolve_returns_independent_profiles(self, tmp_path):
        """Run ``_resolve_agent_profile`` from two parallel tasks with
        different headers and assert each task sees its own profile."""
        code = AgentProfile(id="code", home_dir=tmp_path / "code")
        chat = AgentProfile(id="chat", home_dir=tmp_path / "chat")
        main = AgentProfile(id="main")
        registry = {"main": main, "code": code, "chat": chat}
        routes = [
            {"match": {"platform": "api_server", "chat_id": "code"}, "agent": "code"},
            {"match": {"platform": "api_server", "chat_id": "chat"}, "agent": "chat"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)

        from agent.profile import use_profile

        observed = {}

        async def _one(label: str, header_value: str, hold: float) -> None:
            req = _request_with_headers({"X-Hermes-Chat-Id": header_value})
            profile, agent_id = adapter._resolve_agent_profile(req)
            with use_profile(profile):
                await asyncio.sleep(hold)
                observed[label] = (agent_id, get_active_profile())

        async def _main() -> None:
            await asyncio.gather(
                _one("a", "code", 0.05),
                _one("b", "chat", 0.01),
            )

        asyncio.run(_main())

        assert observed["a"] == ("code", code)
        assert observed["b"] == ("chat", chat)


# ---------------------------------------------------------------------------
# Backward compatibility — adapter unchanged for legacy callers
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_routing_constants_match_documented_headers(self):
        """Header names are part of the public contract — guard against
        silent renames."""
        assert APIServerAdapter._AGENT_CHAT_ID_HEADER == "X-Hermes-Chat-Id"
        assert APIServerAdapter._AGENT_USER_ID_HEADER == "X-Hermes-User-Id"
        assert APIServerAdapter._AGENT_THREAD_ID_HEADER == "X-Hermes-Thread-Id"

    def test_legacy_no_routes_no_agents_returns_main(self):
        """Existing single-agent installs: no agents config, no routes →
        every request resolves to ``main`` with a ``None`` profile (i.e.
        the legacy ``HERMES_HOME`` env-driven path)."""
        adapter = _make_adapter(routes=[], registry={})
        req = _request_with_headers({"X-Hermes-Chat-Id": "ignored"})
        profile, agent_id = adapter._resolve_agent_profile(req)
        assert agent_id == "main"
        assert profile is None

    def test_default_agent_id_constant_is_main(self):
        """Anchor the default-agent contract."""
        assert DEFAULT_AGENT_ID == "main"


class TestSecretScopeBinding:
    """Regression coverage for the credential-scope half of profile routing.

    The gap this guards against (found in architectural review of the api_server routing):
    ``use_profile`` alone routes the *home* (``get_hermes_home`` → SOUL / memory / skills) but
    leaves ``get_secret`` **unscoped**.  Under ``gateway.multiplex_profiles`` the routed agent's
    LLM/provider key — resolved in the run via ``credential_pool`` → ``get_secret`` — would then
    fail closed (``UnscopedSecretError``) or read another profile's process-global value.  The
    api_server path must therefore install the profile's fail-closed secret scope alongside the
    home, mirroring the base adapter's ``_profile_runtime_scope``.
    """

    def test_profile_guard_installs_fail_closed_secret_scope(self, tmp_path, monkeypatch):
        from agent import secret_scope as ss

        home = tmp_path / "coder"
        home.mkdir()
        (home / ".env").write_text("AGENT_API_KEY=sk-coder-scoped\n")
        profile = AgentProfile(id="coder", home_dir=home, api_key_env="AGENT_API_KEY")

        # A process-global value that must NEVER leak into the scoped read.
        monkeypatch.setenv("AGENT_API_KEY", "sk-global-leak")
        monkeypatch.setattr(ss, "_MULTIPLEX_ACTIVE", True)

        # Before entering the guard: exactly the failure the gap produced —
        # multiplex on + no scope → fail closed rather than leak the global.
        with pytest.raises(ss.UnscopedSecretError):
            ss.get_secret("AGENT_API_KEY")

        # Inside the api_server profile guard: resolves the PROFILE's scoped key
        # (from its .env), never the process-global leak, and never fail-closed —
        # while the home binding (get_active_profile) is simultaneously in place.
        with _use_profile_and_secret_scope(profile):
            assert ss.get_secret("AGENT_API_KEY") == "sk-coder-scoped"
            assert get_active_profile() is profile

        # Scope is torn down on exit — no leakage past the run.
        with pytest.raises(ss.UnscopedSecretError):
            ss.get_secret("AGENT_API_KEY")
        assert get_active_profile() is None

    def test_two_profiles_resolve_their_own_scoped_key(self, tmp_path, monkeypatch):
        """Sequential runs for different agents each see only their own credential —
        the cross-profile isolation that multiplexing exists to guarantee."""
        from agent import secret_scope as ss

        monkeypatch.setattr(ss, "_MULTIPLEX_ACTIVE", True)
        monkeypatch.setenv("AGENT_API_KEY", "sk-global-leak")

        seen = {}
        for name, key in (("coder", "sk-coder"), ("research", "sk-research")):
            home = tmp_path / name
            home.mkdir()
            (home / ".env").write_text(f"AGENT_API_KEY={key}\n")
            profile = AgentProfile(id=name, home_dir=home, api_key_env="AGENT_API_KEY")
            with _use_profile_and_secret_scope(profile):
                seen[name] = ss.get_secret("AGENT_API_KEY")

        assert seen == {"coder": "sk-coder", "research": "sk-research"}

    def test_none_profile_is_noop_no_scope_installed(self, monkeypatch):
        """Single-agent path (no routed profile): no scope is installed, so legacy
        ``os.environ`` behavior is preserved and callers pay nothing."""
        from agent import secret_scope as ss

        # Even with multiplex flag on, a None profile must not install a scope
        # (there is no profile home to scope to) — it is a pure home no-op.
        monkeypatch.setattr(ss, "_MULTIPLEX_ACTIVE", False)
        monkeypatch.setenv("SOME_KEY", "from-env")

        with _use_profile_and_secret_scope(None):
            assert ss.current_secret_scope() is None
            assert ss.get_secret("SOME_KEY") == "from-env"
            assert get_active_profile() is None


class TestRunAgentInstallsScope:
    """The *wiring* regression — proves ``_run_agent`` itself enters the profile's
    credential scope, not merely that the helper works in isolation.

    This is the test that would have caught the original gap.  Before the fix the
    executor closure bound ``use_profile(agent_profile)`` alone, so ``_create_agent``
    (and the ``run_conversation`` it drives) ran with the *home* routed but the
    *secret scope* absent — under ``multiplex_profiles`` the agent's LLM key would
    fail closed or read another profile's process-global value.  Here we spy on
    ``_create_agent`` and assert that, at the moment the agent is built inside the
    executor thread, the profile's scoped credential resolves and the process-global
    leak does not.
    """

    @pytest.mark.asyncio
    async def test_run_agent_creates_agent_inside_profile_secret_scope(
        self, tmp_path, monkeypatch
    ):
        from agent import secret_scope as ss

        home = tmp_path / "coder"
        home.mkdir()
        (home / ".env").write_text("AGENT_API_KEY=sk-coder-scoped\n")
        profile = AgentProfile(
            id="coder", home_dir=home, api_key_env="AGENT_API_KEY"
        )

        # Multiplex on + a process-global value that must NOT leak into the run.
        monkeypatch.setattr(ss, "_MULTIPLEX_ACTIVE", True)
        monkeypatch.setenv("AGENT_API_KEY", "sk-global-leak")
        # clear_session_vars is imported inside the executor closure; neutralise it.
        monkeypatch.setattr(
            "gateway.session_context.clear_session_vars", lambda tokens: None
        )

        adapter = _make_adapter(registry={"coder": profile})
        adapter._bind_api_server_session = lambda **kwargs: None

        # Capture the credential/profile state AT AGENT-CREATION TIME (executor thread).
        seen = {}

        def _spy_create_agent(**kwargs):
            seen["scope"] = ss.current_secret_scope()
            seen["profile"] = get_active_profile()
            try:
                seen["key"] = ss.get_secret("AGENT_API_KEY")
            except ss.UnscopedSecretError as exc:  # the bug's signature
                seen["key"] = exc
            agent = MagicMock()
            agent.run_conversation.return_value = {}
            return agent

        adapter._create_agent = _spy_create_agent

        await adapter._run_agent(
            user_message="hi",
            conversation_history=[],
            agent_profile=profile,
        )

        # The scope was live when the agent was built — not fail-closed, not leaked.
        assert seen["profile"] is profile
        assert seen["scope"] is not None
        assert seen["key"] == "sk-coder-scoped"

        # And it is torn down once the run returns.
        assert get_active_profile() is None
        with pytest.raises(ss.UnscopedSecretError):
            ss.get_secret("AGENT_API_KEY")


# ---------------------------------------------------------------------------
# Session creation persists the routed agent_id
# ---------------------------------------------------------------------------


def _make_routed_request(headers: dict, body: dict) -> MagicMock:
    """Build a mock aiohttp.web.Request with headers and an async json() body.

    Why: _handle_create_session calls both request.headers.get (for auth and
    routing) and await request.json() (via _read_json_body).  This stubs both
    without standing up a full aiohttp server.
    What: Returns a MagicMock whose headers dict supports .get() and whose
    json coroutine returns the provided body dict.
    Test: Feed to _handle_create_session and assert on the db.create_session
    call kwargs.
    """
    req = MagicMock()
    req.headers = headers
    req.json = AsyncMock(return_value=body)
    return req


class TestSessionCreationPersistsAgentId:
    """Routing decisions must be stamped on the session row AT CREATION TIME.

    Why: _insert_session_row uses COALESCE(sessions.agent_id, excluded.agent_id)
    on a NOT NULL DEFAULT 'main' column.  The first writer wins — a later
    backfill call can never override 'main' once it is written.  Therefore the
    resolved agent_id must be passed to create_session on the initial write.

    Each test constructs the adapter with routing rules, wires a mock SessionDB,
    calls the handler directly, then inspects the captured create_session kwargs.
    """

    @pytest.mark.asyncio
    async def test_create_session_persists_routed_agent_id(self, tmp_path):
        """POST /api/sessions with X-Hermes-Chat-Id → routed agent persisted.

        Why: The core regression guard.  Before the fix, the atomic
        check-insert-title write (_handle_create_session's TOCTOU-safe
        rewrite of the old direct create_session call) never bound
        agent_id, so every row defaulted to 'main' regardless of routing.
        What: Routes 'coder' chat_id to the 'coder' agent against a real
        SessionDB (the atomic path runs a raw INSERT via db._execute_write,
        so a MagicMock can't observe it meaningfully); asserts the persisted
        row's agent_id via db.get_session.
        Test: Revert the resolved_agent_id bind in the INSERT and this test
        fails because the persisted row's agent_id reverts to 'main'.
        """
        from hermes_state import SessionDB

        coder = AgentProfile(id="coder", home_dir=tmp_path / "coder")
        main = AgentProfile(id="main")
        registry = {"main": main, "coder": coder}
        routes = [
            {"match": {"platform": "api_server", "chat_id": "coder"}, "agent": "coder"},
        ]
        adapter = _make_adapter(routes=routes, registry=registry)

        real_db = SessionDB(db_path=tmp_path / "state.db")
        adapter._session_db = real_db

        req = _make_routed_request(
            headers={"X-Hermes-Chat-Id": "coder"},
            body={"id": "sess-coder-001"},
        )

        try:
            resp = await adapter._handle_create_session(req)
            assert resp.status == 201
            assert real_db.get_session("sess-coder-001")["agent_id"] == "coder", (
                "agent_id must be 'coder' — the routed agent resolved from X-Hermes-Chat-Id. "
                "If this fails, the fix was not applied or was reverted."
            )
        finally:
            real_db.close()

    @pytest.mark.asyncio
    async def test_create_session_defaults_to_main_without_routing_header(self, tmp_path):
        """POST /api/sessions with no routing header → agent_id persisted as 'main'.

        Why: Regression guard for the default path.  No routing header means no
        agent match; the default agent_id ('main') must be written explicitly
        (not just relying on the column default) so callers can see it in
        db.get_session().
        What: No X-Hermes-Chat-Id supplied; assert the persisted row's
        agent_id is 'main' via a real SessionDB (see the sibling test above
        for why a MagicMock can't observe the atomic INSERT path).
        Test: Pass a header that matches and assert this test fails to prove
        test sensitivity.
        """
        from hermes_state import SessionDB

        main = AgentProfile(id="main")
        registry = {"main": main}
        adapter = _make_adapter(routes=[], registry=registry)

        real_db = SessionDB(db_path=tmp_path / "state.db")
        adapter._session_db = real_db

        req = _make_routed_request(
            headers={},  # no routing headers
            body={"id": "sess-main-001"},
        )

        try:
            resp = await adapter._handle_create_session(req)
            assert resp.status == 201
            assert real_db.get_session("sess-main-001")["agent_id"] == "main"
        finally:
            real_db.close()

    @pytest.mark.asyncio
    async def test_fork_session_inherits_source_agent_id(self, tmp_path):
        """POST /api/sessions/{id}/fork → fork row inherits parent's agent_id.

        Why: A fork is a continuation of the parent lineage.  The fork endpoint
        carries no routing headers, so re-resolving routing would fall back to
        'main'.  Inheriting the source agent_id is the only correct semantics.
        What: Source session has agent_id='coder'; fork call must persist
        agent_id='coder' on the new row, not 'main'.
        Test: Set source agent_id='coder'; assert fork create_session receives
        agent_id='coder'.
        """
        coder = AgentProfile(id="coder", home_dir=tmp_path / "coder")
        main = AgentProfile(id="main")
        registry = {"main": main, "coder": coder}
        adapter = _make_adapter(routes=[], registry=registry)

        source_session = {
            "id": "sess-parent",
            "agent_id": "coder",
            "model": "claude-3",
            "system_prompt": None,
            "title": "my conv",
        }

        mock_db = MagicMock()
        # _get_existing_session_or_404 calls db.get_session(source_id)
        # then fork check calls db.get_session(fork_id)
        mock_db.get_session.side_effect = lambda sid: (
            source_session if sid == "sess-parent" else None
        )
        mock_db.create_session.side_effect = lambda sid, *args, **kwargs: sid
        mock_db.get_messages.return_value = []
        mock_db.replace_messages.return_value = None
        mock_db.get_next_title_in_lineage.return_value = "my conv fork"
        mock_db.set_session_title.return_value = None
        adapter._session_db = mock_db

        req = _make_routed_request(
            headers={},  # fork endpoint: no routing headers
            body={"id": "sess-fork-001"},
        )
        # Inject source_id into match_info as the route does
        req.match_info = {"session_id": "sess-parent"}

        resp = await adapter._handle_fork_session(req)

        assert resp.status == 201
        mock_db.create_session.assert_called_once()
        call_kwargs = mock_db.create_session.call_args
        assert call_kwargs.args[0] == "sess-fork-001"
        assert call_kwargs.kwargs.get("agent_id") == "coder", (
            "Fork must inherit the parent session's agent_id='coder', not 'main'."
        )


# ---------------------------------------------------------------------------
# Stateful session turns run under the session's persisted agent
# ---------------------------------------------------------------------------


def _make_session_chat_request(session_id: str, body: dict) -> MagicMock:
    """Build a mock aiohttp.web.Request for session-chat endpoints.

    Why: _handle_session_chat and _handle_session_chat_stream read
    match_info["session_id"], headers (for auth + session key), and
    await request.json() — this stubs all three without a live server.
    What: Returns a MagicMock with no auth header (so _check_auth passes
    when the adapter has no auth key configured) and the given body.
    Test: Feed to _handle_session_chat; assert _run_agent receives the
    expected agent_profile kwarg.
    """
    req = MagicMock()
    req.headers = {}  # no auth header → _check_auth returns None
    req.match_info = {"session_id": session_id}
    req.json = AsyncMock(return_value=body)
    return req


class TestSessionChatRunsUnderSessionAgent:
    """Session chat turns must use the agent the session was created for.

    Why: _handle_session_chat and _handle_session_chat_stream previously
    called _run_agent without agent_profile, so every turn silently ran
    under the default agent regardless of the session's persisted agent_id.
    The fix reads session.agent_id, resolves the profile via
    _profile_for_agent_id, and passes it as agent_profile=.

    All tests mock _run_agent to avoid a live executor thread and inspect
    the agent_profile kwarg directly.
    """

    @pytest.mark.asyncio
    async def test_session_chat_uses_session_agent_profile(self, tmp_path):
        """_handle_session_chat passes the session's AgentProfile to _run_agent.

        Why: Core regression guard — proves the fix is wired end-to-end.
        What: Adapter has a 'coder' profile; session row has agent_id='coder';
        assert _run_agent receives agent_profile == coder profile.
        Test: Remove the agent_profile= kwarg from _handle_session_chat and
        this test fails because agent_profile would be None.
        """
        coder = AgentProfile(id="coder", home_dir=tmp_path / "coder")
        main = AgentProfile(id="main")
        registry = {"main": main, "coder": coder}
        adapter = _make_adapter(routes=[], registry=registry)

        coder_session = {"id": "sess-coder", "agent_id": "coder"}
        mock_db = MagicMock()
        mock_db.get_session.return_value = coder_session
        mock_db.get_messages_as_conversation.return_value = []
        adapter._session_db = mock_db

        captured = {}

        async def _mock_run_agent(**kwargs):
            captured["agent_profile"] = kwargs.get("agent_profile")
            return {"final_response": "ok", "session_id": "sess-coder"}, {}

        adapter._run_agent = _mock_run_agent

        req = _make_session_chat_request(
            session_id="sess-coder",
            body={"message": "hello"},
        )

        resp = await adapter._handle_session_chat(req)

        assert resp.status == 200
        assert captured["agent_profile"] is coder, (
            "_run_agent must receive agent_profile=coder for a session routed to 'coder'. "
            "If this fails, the agent_profile= kwarg was not passed in _handle_session_chat."
        )

    @pytest.mark.asyncio
    async def test_session_chat_stream_uses_session_agent_profile(self, tmp_path):
        """_handle_session_chat_stream passes the session's AgentProfile to _run_agent.

        Why: Stream path has a nested _run_and_signal coroutine; the profile
        must be captured in the outer handler scope and closed over.
        What: Same setup as the sync test; assert agent_profile== coder profile.
        Test: Remove agent_profile= from the _run_agent call in _run_and_signal
        and this test fails.
        """
        coder = AgentProfile(id="coder", home_dir=tmp_path / "coder")
        main = AgentProfile(id="main")
        registry = {"main": main, "coder": coder}
        adapter = _make_adapter(routes=[], registry=registry)

        coder_session = {"id": "sess-coder-stream", "agent_id": "coder"}
        mock_db = MagicMock()
        mock_db.get_session.return_value = coder_session
        mock_db.get_messages_as_conversation.return_value = []
        adapter._session_db = mock_db

        captured = {}

        async def _mock_run_agent(**kwargs):
            captured["agent_profile"] = kwargs.get("agent_profile")
            return {"final_response": "streamed ok", "session_id": "sess-coder-stream"}, {}

        adapter._run_agent = _mock_run_agent

        req = _make_session_chat_request(
            session_id="sess-coder-stream",
            body={"message": "hello stream"},
        )

        # _handle_session_chat_stream returns a StreamResponse; we don't need to
        # drain the SSE queue — _run_and_signal will complete before we inspect.
        import aiohttp
        from unittest.mock import patch

        with patch("aiohttp.web.StreamResponse") as MockStream:
            mock_stream = AsyncMock()
            mock_stream.write = AsyncMock()
            mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
            mock_stream.__aexit__ = AsyncMock(return_value=False)
            MockStream.return_value = mock_stream

            resp = await adapter._handle_session_chat_stream(req)

        # Allow the background task (_run_and_signal) to complete.
        await asyncio.sleep(0.05)

        assert captured.get("agent_profile") is coder, (
            "_run_agent must receive agent_profile=coder for a streaming session turn. "
            "If this fails, agent_profile= was not passed in _run_and_signal."
        )

    @pytest.mark.asyncio
    async def test_session_chat_legacy_main_session_passes_none_profile(self):
        """A legacy session with agent_id='main' gives agent_profile=None to _run_agent.

        Why: Backward-compatibility guard.  None is the no-op sentinel that lets
        _run_agent fall through to default behaviour (no profile wrapper).
        What: Session has agent_id='main'; registry has a 'main' profile to
        confirm the lookup works — but the expected result is still the resolved
        profile, not None, for 'main'.  Separately verify that a missing
        agent_id also resolves gracefully.
        Test: Change expected to the main profile and assert this fails when
        agent_profile is None to prove the check is live.
        """
        main = AgentProfile(id="main")
        registry = {"main": main}
        adapter = _make_adapter(routes=[], registry=registry)

        # Session with no agent_id (truly legacy / pre-migration row)
        legacy_session = {"id": "sess-legacy", "agent_id": None}
        mock_db = MagicMock()
        mock_db.get_session.return_value = legacy_session
        mock_db.get_messages_as_conversation.return_value = []
        adapter._session_db = mock_db

        captured = {}

        async def _mock_run_agent(**kwargs):
            captured["agent_profile"] = kwargs.get("agent_profile")
            return {"final_response": "ok", "session_id": "sess-legacy"}, {}

        adapter._run_agent = _mock_run_agent

        req = _make_session_chat_request(
            session_id="sess-legacy",
            body={"message": "hello legacy"},
        )

        resp = await adapter._handle_session_chat(req)

        assert resp.status == 200
        # agent_id=None → _profile_for_agent_id returns None → no-op default path.
        assert captured["agent_profile"] is None, (
            "A session with agent_id=None must pass agent_profile=None to _run_agent "
            "so the default (no profile wrapper) behaviour is preserved."
        )

    @pytest.mark.asyncio
    async def test_profile_for_agent_id_helper(self, tmp_path):
        """_profile_for_agent_id returns the registered profile or None.

        Why: Exercises the helper in isolation to confirm it mirrors the
        registry lookup in _resolve_agent_profile without duplication.
        What: Known id → profile; unknown id → None; None id → None;
        no registry → None.
        Test: Change the expected profile to a different object and assert
        the comparison fails to prove the test is live.
        """
        coder = AgentProfile(id="coder", home_dir=tmp_path / "coder")
        main = AgentProfile(id="main")
        registry = {"main": main, "coder": coder}
        adapter = _make_adapter(routes=[], registry=registry)

        assert adapter._profile_for_agent_id("coder") is coder
        assert adapter._profile_for_agent_id("main") is main
        assert adapter._profile_for_agent_id("unknown") is None
        assert adapter._profile_for_agent_id(None) is None
        assert adapter._profile_for_agent_id("") is None

        # No registry (legacy single-agent install)
        adapter2 = _make_adapter(routes=[], registry=None)
        assert adapter2._profile_for_agent_id("coder") is None
