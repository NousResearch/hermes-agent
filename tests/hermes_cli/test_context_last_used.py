"""Tests for hermes_cli/context_usage.py and the endpoints that surface it.

Covers the "last used" derivation from state.db (MCP tool calls, channel
sessions, best-effort key attribution) plus the /api/context/last-used,
/api/mcp/servers, /api/messaging/platforms, and /api/env wiring. See the
module docstring in context_usage.py for the "never fabricate a timestamp"
contract these tests are pinning down.
"""

import time

import pytest


def _write_state_db(db_path, now, *, include_hyphenated_mcp: bool = False):
    """Seed a temp state.db with sessions/messages at known timestamps."""
    import hermes_state

    db = hermes_state.SessionDB(db_path=db_path)
    db.create_session("s1", "slack")
    db.append_message(
        "s1",
        "assistant",
        tool_calls=[
            {"id": "c1", "function": {"name": "mcp__github__create_issue", "arguments": "{}"}}
        ],
        timestamp=now - 20,
    )
    db.append_message(
        "s1",
        "assistant",
        tool_calls=[
            {"id": "c2", "function": {"name": "mcp__github__list_issues", "arguments": "{}"}}
        ],
        timestamp=now - 10,
    )
    db.append_message(
        "s1",
        "assistant",
        content="just talking, no tools",
        timestamp=now - 9,
    )
    db.append_message(
        "s1",
        "assistant",
        tool_calls=[{"id": "c3", "function": {"name": "elevenlabs_tts", "arguments": "{}"}}],
        timestamp=now - 5,
    )
    db.append_message(
        "s1",
        "assistant",
        tool_calls=[{"id": "c4", "function": {"name": "web_search", "arguments": "{}"}}],
        timestamp=now - 3,
    )
    if include_hyphenated_mcp:
        # Registration sanitizes configured "github-enterprise" → github_enterprise
        # in the persisted tool name (tools.mcp_tool.sanitize_mcp_name_component).
        db.append_message(
            "s1",
            "assistant",
            tool_calls=[
                {
                    "id": "c5",
                    "function": {
                        "name": "mcp__github_enterprise__list_repos",
                        "arguments": "{}",
                    },
                }
            ],
            timestamp=now - 2,
        )
    db.close()
    return db


class TestComputeContextLastUsed:
    def test_mcp_tool_call_maps_to_server_max_timestamp(self, tmp_path):
        from hermes_cli.context_usage import compute_context_last_used

        now = time.time()
        _write_state_db(tmp_path / "state.db", now)

        result = compute_context_last_used(home=tmp_path)

        assert result["mcp"]["github"] == pytest.approx(now - 10)

    def test_hyphenated_configured_name_matches_sanitized_tool_call(self, tmp_path):
        """Configured ``github-enterprise`` must match persisted
        ``mcp__github_enterprise__…`` tool names."""
        from hermes_cli.context_usage import compute_context_last_used

        now = time.time()
        _write_state_db(tmp_path / "state.db", now, include_hyphenated_mcp=True)

        result = compute_context_last_used(
            home=tmp_path, mcp_server_names=["github-enterprise", "github"]
        )

        assert result["mcp"]["github-enterprise"] == pytest.approx(now - 2)
        assert result["mcp"]["github"] == pytest.approx(now - 10)
        assert "github_enterprise" not in result["mcp"]

    def test_non_mcp_tools_ignored_for_mcp_map(self, tmp_path):
        from hermes_cli.context_usage import compute_context_last_used

        now = time.time()
        _write_state_db(tmp_path / "state.db", now)

        result = compute_context_last_used(home=tmp_path)

        assert set(result["mcp"].keys()) == {"github"}

    def test_unique_provider_tool_attributes_to_its_key(self, tmp_path):
        from hermes_cli.context_usage import compute_context_last_used

        now = time.time()
        _write_state_db(tmp_path / "state.db", now)

        result = compute_context_last_used(home=tmp_path)

        assert result["keys"]["ELEVENLABS_API_KEY"] == pytest.approx(now - 5)

    def test_ambiguous_provider_tool_is_omitted_from_keys(self, tmp_path):
        """web_search is served by several possible providers (Tavily, Exa,
        Firecrawl, ...) — with no way to tell which one actually ran, no key
        should claim it."""
        from hermes_cli.context_usage import compute_context_last_used

        now = time.time()
        _write_state_db(tmp_path / "state.db", now)

        result = compute_context_last_used(home=tmp_path)

        assert "FIRECRAWL_API_KEY" not in result["keys"]
        assert "TAVILY_API_KEY" not in result["keys"]
        assert "EXA_API_KEY" not in result["keys"]

    def test_channel_last_used_from_session_source(self, tmp_path):
        from hermes_cli.context_usage import compute_context_last_used

        now = time.time()
        _write_state_db(tmp_path / "state.db", now)

        result = compute_context_last_used(home=tmp_path)

        assert result["channels"]["slack"] is not None

    def test_empty_db_returns_empty_maps_without_throwing(self, tmp_path):
        import hermes_state
        from hermes_cli.context_usage import compute_context_last_used

        hermes_state.SessionDB(db_path=tmp_path / "state.db").close()

        result = compute_context_last_used(home=tmp_path)

        assert result == {
            "mcp": {},
            "channels": {},
            "keys": {},
            "computed_at": result["computed_at"],
        }

    def test_missing_db_returns_empty_maps_without_throwing(self, tmp_path):
        from hermes_cli.context_usage import compute_context_last_used

        result = compute_context_last_used(home=tmp_path)

        assert result["mcp"] == {}
        assert result["channels"] == {}
        assert result["keys"] == {}

    def test_mcp_server_names_stops_scan_once_all_found(self, tmp_path):
        """Passing the configured server set lets the scan stop early once
        every server already has a hit — verified indirectly: it must still
        find the newest (not an older, pre-early-exit) timestamp."""
        from hermes_cli.context_usage import compute_context_last_used

        now = time.time()
        _write_state_db(tmp_path / "state.db", now)

        result = compute_context_last_used(home=tmp_path, mcp_server_names=["github"])

        assert result["mcp"]["github"] == pytest.approx(now - 10)

    def test_lookback_limit_bounds_the_scan(self, tmp_path):
        """A tiny lookback window means older tool calls fall outside it and
        are honestly omitted rather than scanned in full every time."""
        import hermes_state
        from hermes_cli.context_usage import compute_context_last_used

        now = time.time()
        db_path = tmp_path / "state.db"
        db = hermes_state.SessionDB(db_path=db_path)
        db.create_session("s1", "slack")
        # Many filler messages without tool_calls, then one MCP call buried
        # near the oldest end.
        db.append_message(
            "s1",
            "assistant",
            tool_calls=[
                {"id": "c0", "function": {"name": "mcp__github__create_issue", "arguments": "{}"}}
            ],
            timestamp=now - 1000,
        )
        for i in range(20):
            db.append_message("s1", "user", content=f"msg {i}", timestamp=now - 900 + i)
        db.close()

        result = compute_context_last_used(home=tmp_path, lookback_messages=5)

        assert result["mcp"] == {}

    def test_opens_state_db_via_tracked_connection(self, tmp_path, monkeypatch):
        """Review regression: must not raw-sqlite3.connect state.db."""
        import hermes_cli.sqlite_safe_read as safe
        from hermes_cli import context_usage

        now = time.time()
        _write_state_db(tmp_path / "state.db", now)

        calls = []
        real = safe.connect_tracked

        def _spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        # Imported inside _connect_state_db_ro — patch the module attribute.
        monkeypatch.setattr(safe, "connect_tracked", _spy)

        result = context_usage.compute_context_last_used(home=tmp_path)

        assert result["mcp"]["github"] == pytest.approx(now - 10)
        assert calls, "expected connect_tracked to open state.db"
        assert any(
            kwargs.get("tracking_path") is not None or "mode=ro" in str(args)
            for args, kwargs in calls
        )


def _client():
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    hermes_state.DEFAULT_DB_PATH = get_hermes_home() / "state.db"
    return client


class TestContextLastUsedEndpoint:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client = _client()

    def test_empty_state_returns_200_with_empty_maps(self):
        response = self.client.get("/api/context/last-used")

        assert response.status_code == 200
        body = response.json()
        assert body["mcp"] == {}
        assert body["channels"] == {}
        assert body["keys"] == {}
        assert "computed_at" in body

    def test_reflects_configured_mcp_server_usage(self):
        from hermes_constants import get_hermes_home

        self.client.post(
            "/api/mcp/servers", json={"name": "github", "url": "https://x/mcp"}
        )

        now = time.time()
        _write_state_db(get_hermes_home() / "state.db", now)

        response = self.client.get("/api/context/last-used")

        assert response.status_code == 200
        assert response.json()["mcp"]["github"] == pytest.approx(now - 10)

    def test_hyphenated_server_name_on_batch_endpoint(self):
        from hermes_constants import get_hermes_home

        self.client.post(
            "/api/mcp/servers",
            json={"name": "github-enterprise", "url": "https://x/mcp"},
        )

        now = time.time()
        _write_state_db(
            get_hermes_home() / "state.db", now, include_hyphenated_mcp=True
        )

        body = self.client.get("/api/context/last-used").json()

        assert body["mcp"]["github-enterprise"] == pytest.approx(now - 2)


class TestMcpServersLastUsedField:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client = _client()

    def test_server_summary_includes_null_last_used_when_never_called(self):
        self.client.post(
            "/api/mcp/servers", json={"name": "unused-server", "url": "https://x/mcp"}
        )

        servers = self.client.get("/api/mcp/servers").json()["servers"]

        assert servers[0]["name"] == "unused-server"
        assert servers[0]["last_used_at"] is None

    def test_server_summary_includes_real_last_used_when_called(self):
        from hermes_constants import get_hermes_home

        self.client.post(
            "/api/mcp/servers", json={"name": "github", "url": "https://x/mcp"}
        )

        now = time.time()
        _write_state_db(get_hermes_home() / "state.db", now)

        servers = self.client.get("/api/mcp/servers").json()["servers"]

        assert servers[0]["last_used_at"] == pytest.approx(now - 10)

    def test_hyphenated_server_name_matches_sanitized_tool_calls(self):
        from hermes_constants import get_hermes_home

        self.client.post(
            "/api/mcp/servers",
            json={"name": "github-enterprise", "url": "https://x/mcp"},
        )

        now = time.time()
        _write_state_db(
            get_hermes_home() / "state.db", now, include_hyphenated_mcp=True
        )

        servers = self.client.get("/api/mcp/servers").json()["servers"]
        assert servers[0]["name"] == "github-enterprise"
        assert servers[0]["last_used_at"] == pytest.approx(now - 2)


class TestChannelAndKeyLastUsedFields:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client = _client()

    def test_messaging_platforms_includes_channel_last_used(self):
        from hermes_constants import get_hermes_home

        now = time.time()
        _write_state_db(get_hermes_home() / "state.db", now)

        platforms = self.client.get("/api/messaging/platforms").json()["platforms"]
        by_id = {p["id"]: p for p in platforms}

        assert "slack" in by_id
        assert by_id["slack"]["last_used_at"] is not None
        # Platforms never used stay None rather than fabricated.
        unused = next(p for p in platforms if p["id"] != "slack")
        assert unused["last_used_at"] is None

    def test_env_includes_unique_key_last_used(self):
        from hermes_constants import get_hermes_home

        now = time.time()
        _write_state_db(get_hermes_home() / "state.db", now)

        env = self.client.get("/api/env").json()

        assert env["ELEVENLABS_API_KEY"]["last_used_at"] == pytest.approx(now - 5)
        # Ambiguous multi-provider tools must not stamp any single key.
        for key in ("FIRECRAWL_API_KEY", "TAVILY_API_KEY", "EXA_API_KEY"):
            if key in env:
                assert env[key]["last_used_at"] is None
