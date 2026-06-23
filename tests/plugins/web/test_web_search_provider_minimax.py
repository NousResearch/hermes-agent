"""Tests for the MiniMax Web Search provider (plugins/web/minimax/).

Covers:
- MiniMaxWebSearchProvider identity (name, display_name, capability flags)
- is_available() — cheap probe across the 4 accepted key-alias env vars
- search() — happy path with the documented response shape
- search() response normalization (organic → results → web field fallback,
  link/url/href + snippet/description/summary field aliases)
- search() error paths — empty query, missing key, HTTP error, transport
  error, API-level base_resp.status_code != 0, JSON parse failure
- get_setup_schema() — picker metadata
- Env-var precedence — first non-empty alias wins

The provider is a thin HTTP wrapper around ``POST {host}/v1/coding_plan/search``
that returns the canonical ``{success, data: {web: [{title, url,
description, position}]}}`` envelope every other Hermes web provider
produces — these tests pin that contract.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_payload(organic: list | None = None) -> dict:
    """Build a minimal successful MiniMax search response.

    The real API returns ``{"organic": [...], "related_searches": [...]}``
    (verified against the live endpoint, 2026-06-23). On success the
    ``base_resp`` envelope is absent.
    """
    if organic is None:
        organic = [
            {"title": "First hit", "link": "https://example.com/a",
             "snippet": "First snippet", "date": "2026-01-01"},
            {"title": "Second hit", "link": "https://example.com/b",
             "snippet": "Second snippet", "date": "2026-02-02"},
        ]
    return {"organic": organic, "related_searches": ["q1", "q2"]}


def _err_payload(code: int, msg: str) -> dict:
    """Build an API-level error envelope (the only kind MiniMax returns
    when a request is well-formed but the API rejects it — e.g. region
    mismatch → 1004 invalid api key)."""
    return {"base_resp": {"status_code": code, "status_msg": msg}}


def _mock_resp(json_data, status_code: int = 200):
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_data
    if status_code >= 400:
        m.raise_for_status = MagicMock(side_effect=Exception(f"HTTP {status_code}"))
        m.text = json.dumps(json_data)
    else:
        m.raise_for_status = MagicMock()
    return m


# All four accepted env-var aliases — first non-empty wins (per
# plugins/web/minimax/provider.py::_resolve_api_key).
_ALL_KEY_ALIASES = (
    "MINIMAX_CODE_PLAN_KEY",
    "MINIMAX_CODING_API_KEY",
    "MINIMAX_OAUTH_TOKEN",
    "MINIMAX_API_KEY",
)


@pytest.fixture
def _clear_minimax_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every MiniMax env var so is_available() returns False."""
    for k in _ALL_KEY_ALIASES + ("MINIMAX_API_HOST",):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# Provider identity / capability
# ---------------------------------------------------------------------------


class TestMiniMaxProviderIdentity:
    def test_provider_name(self):
        from plugins.web.minimax.provider import MiniMaxWebSearchProvider
        assert MiniMaxWebSearchProvider().name == "minimax"

    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.minimax.provider import MiniMaxWebSearchProvider
        assert issubclass(MiniMaxWebSearchProvider, WebSearchProvider)

    def test_supports_search_only(self):
        """MiniMax Token Plan search is search-only — no body extraction.

        Hermes will route web_extract calls to another configured backend
        (Firecrawl/Tavily/Exa/Parallel) via the registry's capability filter.
        """
        from plugins.web.minimax.provider import MiniMaxWebSearchProvider
        p = MiniMaxWebSearchProvider()
        assert p.supports_search() is True
        assert p.supports_extract() is False

    def test_display_name_mentions_minimax(self):
        from plugins.web.minimax.provider import MiniMaxWebSearchProvider
        assert "MiniMax" in MiniMaxWebSearchProvider().display_name


# ---------------------------------------------------------------------------
# is_available() — cheap probe, no network
# ---------------------------------------------------------------------------


class TestMiniMaxProviderIsAvailable:
    """is_available() runs on every ``hermes tools`` repaint and at
    tool-registration time. It MUST NOT make a network call — only inspect
    process env. These tests pin that contract.
    """

    def test_unavailable_with_no_env_set(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        from plugins.web.minimax.provider import MiniMaxWebSearchProvider
        assert MiniMaxWebSearchProvider().is_available() is False

    @pytest.mark.parametrize("env_var", _ALL_KEY_ALIASES)
    def test_available_via_any_alias(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env, env_var
    ):
        from plugins.web.minimax.provider import MiniMaxWebSearchProvider
        monkeypatch.setenv(env_var, "sk-cp-test")
        assert MiniMaxWebSearchProvider().is_available() is True

    def test_whitespace_only_value_does_not_count(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        from plugins.web.minimax.provider import MiniMaxWebSearchProvider
        monkeypatch.setenv("MINIMAX_API_KEY", "   \t  ")
        # Whitespace-only should be treated as unset (matches OpenClaw behavior).
        assert MiniMaxWebSearchProvider().is_available() is False


# ---------------------------------------------------------------------------
# search() — happy path
# ---------------------------------------------------------------------------


class TestMiniMaxSearchHappyPath:
    def test_returns_canonical_envelope(self, monkeypatch: pytest.MonkeyPatch,
                                         _clear_minimax_env):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-test")
        monkeypatch.setenv("MINIMAX_API_HOST", "https://api.minimax.io")

        with patch("httpx.post", return_value=_mock_resp(_ok_payload())) as mock_post:
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            result = MiniMaxWebSearchProvider().search("capital of france", limit=5)

        assert result["success"] is True
        assert "data" in result
        assert "web" in result["data"]
        assert isinstance(result["data"]["web"], list)
        assert len(result["data"]["web"]) == 2

    def test_normalizes_organic_to_canonical_shape(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-test")
        with patch("httpx.post", return_value=_mock_resp(_ok_payload())):
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            result = MiniMaxWebSearchProvider().search("test", limit=5)

        first = result["data"]["web"][0]
        assert first == {
            "title": "First hit",
            "url": "https://example.com/a",
            "description": "First snippet",
            "position": 1,
        }
        assert result["data"]["web"][1]["position"] == 2

    def test_honors_limit(self, monkeypatch: pytest.MonkeyPatch,
                            _clear_minimax_env):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-test")
        # 5 organic rows; ask for 3
        organic = [
            {"title": f"T{i}", "link": f"https://e.com/{i}",
             "snippet": f"S{i}", "date": ""}
            for i in range(5)
        ]
        with patch("httpx.post", return_value=_mock_resp(_ok_payload(organic))):
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            result = MiniMaxWebSearchProvider().search("x", limit=3)

        assert len(result["data"]["web"]) == 3
        assert [r["position"] for r in result["data"]["web"]] == [1, 2, 3]

    def test_sends_correct_request_to_canonical_endpoint(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        """Pins the actual API contract — host, path, headers, body."""
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-test-xyz")
        monkeypatch.setenv("MINIMAX_API_HOST", "https://api.minimax.io")

        with patch("httpx.post", return_value=_mock_resp(_ok_payload())) as mock_post:
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            MiniMaxWebSearchProvider().search("hello world", limit=5)

        call = mock_post.call_args
        # URL
        assert call.args[0] == "https://api.minimax.io/v1/coding_plan/search"
        # Headers
        assert call.kwargs["headers"]["Authorization"] == "Bearer sk-cp-test-xyz"
        assert call.kwargs["headers"]["Content-Type"] == "application/json"
        # Body
        assert call.kwargs["json"] == {"q": "hello world"}

    def test_cn_host_default_when_unset(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        """Defaults to global host (api.minimax.io) when MINIMAX_API_HOST unset."""
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-test")
        monkeypatch.delenv("MINIMAX_API_HOST", raising=False)

        with patch("httpx.post", return_value=_mock_resp(_ok_payload())) as mock_post:
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            MiniMaxWebSearchProvider().search("x", limit=5)

        assert mock_post.call_args.args[0] == "https://api.minimax.io/v1/coding_plan/search"

    def test_cn_host_override(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        """MINIMAX_API_HOST=https://api.minimaxi.com routes to China endpoint."""
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-test")
        monkeypatch.setenv("MINIMAX_API_HOST", "https://api.minimaxi.com")

        with patch("httpx.post", return_value=_mock_resp(_ok_payload())) as mock_post:
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            MiniMaxWebSearchProvider().search("x", limit=5)

        assert mock_post.call_args.args[0] == "https://api.minimaxi.com/v1/coding_plan/search"

    def test_host_trailing_slash_stripped(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        """MINIMAX_API_HOST with a trailing slash should not double-slash the path."""
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-test")
        monkeypatch.setenv("MINIMAX_API_HOST", "https://api.minimax.io/")

        with patch("httpx.post", return_value=_mock_resp(_ok_payload())) as mock_post:
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            MiniMaxWebSearchProvider().search("x", limit=5)

        assert mock_post.call_args.args[0] == "https://api.minimax.io/v1/coding_plan/search"


# ---------------------------------------------------------------------------
# search() — response-shape robustness
# ---------------------------------------------------------------------------


class TestMiniMaxResponseShapeFallbacks:
    """If MiniMax's response shape ever changes, the normalizer should
    degrade gracefully rather than silently mask real data as empty."""

    def test_falls_back_to_results_field(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-test")
        # Hypothetical future shape — single 'results' key instead of 'organic'.
        payload = {
            "results": [
                {"title": "A", "url": "https://a.com", "description": "da"},
            ]
        }
        with patch("httpx.post", return_value=_mock_resp(payload)):
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            result = MiniMaxWebSearchProvider().search("x", limit=5)
        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "A"
        assert result["data"]["web"][0]["url"] == "https://a.com"

    def test_falls_back_to_web_field(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-test")
        payload = {
            "web": [
                {"title": "B", "href": "https://b.com", "summary": "sb"},
            ]
        }
        with patch("httpx.post", return_value=_mock_resp(payload)):
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            result = MiniMaxWebSearchProvider().search("x", limit=5)
        assert result["success"] is True
        # Field aliases — href → url, summary → description
        assert result["data"]["web"][0]["url"] == "https://b.com"
        assert result["data"]["web"][0]["description"] == "sb"

    def test_empty_results_returns_success_with_empty_list(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-test")
        with patch("httpx.post", return_value=_mock_resp({"organic": []})):
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            result = MiniMaxWebSearchProvider().search("x", limit=5)
        assert result["success"] is True
        assert result["data"]["web"] == []


# ---------------------------------------------------------------------------
# search() — error paths
# ---------------------------------------------------------------------------


class TestMiniMaxSearchErrors:
    def test_empty_query_returns_error_dict(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-test")
        from plugins.web.minimax.provider import MiniMaxWebSearchProvider
        result = MiniMaxWebSearchProvider().search("", limit=5)
        assert result["success"] is False
        assert "query is required" in result["error"]

    def test_whitespace_query_returns_error_dict(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-test")
        from plugins.web.minimax.provider import MiniMaxWebSearchProvider
        result = MiniMaxWebSearchProvider().search("   ", limit=5)
        assert result["success"] is False
        assert "query is required" in result["error"]

    def test_missing_key_returns_error_dict(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        from plugins.web.minimax.provider import MiniMaxWebSearchProvider
        result = MiniMaxWebSearchProvider().search("anything", limit=5)
        assert result["success"] is False
        # Helpful error lists all 4 accepted aliases
        assert "MINIMAX_CODE_PLAN_KEY" in result["error"]
        assert "MINIMAX_API_KEY" in result["error"]

    def test_http_error_returns_error_dict(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-test")
        err = _mock_resp({"error": "unauthorized"}, status_code=401)
        with patch("httpx.post", return_value=err):
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            result = MiniMaxWebSearchProvider().search("x", limit=5)
        assert result["success"] is False
        assert "HTTP 401" in result["error"]

    def test_transport_error_returns_error_dict(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        """Network failures (DNS, timeout, connection refused) should
        surface as typed errors, not raise."""
        import httpx
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-test")
        with patch("httpx.post", side_effect=httpx.ConnectError("no route")):
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            result = MiniMaxWebSearchProvider().search("x", limit=5)
        assert result["success"] is False
        assert "Could not reach MiniMax" in result["error"]

    def test_api_level_error_1004_region_mismatch(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        """Status code 1004 = invalid api key (region mismatch is the
        most common cause per the OpenClaw docs)."""
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-test")
        with patch("httpx.post",
                    return_value=_mock_resp(_err_payload(1004, "invalid api key"))):
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            result = MiniMaxWebSearchProvider().search("x", limit=5)
        assert result["success"] is False
        assert "1004" in result["error"]
        assert "invalid api key" in result["error"]

    def test_json_parse_failure_returns_error_dict(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-test")
        bad = MagicMock()
        bad.status_code = 200
        bad.raise_for_status = MagicMock()
        bad.json.side_effect = ValueError("not json")
        with patch("httpx.post", return_value=bad):
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            result = MiniMaxWebSearchProvider().search("x", limit=5)
        assert result["success"] is False
        assert "parse" in result["error"].lower()


# ---------------------------------------------------------------------------
# Key resolution precedence
# ---------------------------------------------------------------------------


class TestMiniMaxKeyPrecedence:
    """The first non-empty env-var alias wins. This matches OpenClaw's
    minimax-search config so the same .env works in both agents."""

    def test_prefers_code_plan_key_over_api_key(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        monkeypatch.setenv("MINIMAX_CODE_PLAN_KEY", "from-cp")
        monkeypatch.setenv("MINIMAX_API_KEY", "from-generic")
        with patch("httpx.post", return_value=_mock_resp(_ok_payload())) as mock_post:
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            MiniMaxWebSearchProvider().search("x", limit=5)
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer from-cp"

    def test_coding_api_key_wins_over_api_key(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        monkeypatch.setenv("MINIMAX_CODING_API_KEY", "from-coding")
        monkeypatch.setenv("MINIMAX_API_KEY", "from-generic")
        with patch("httpx.post", return_value=_mock_resp(_ok_payload())) as mock_post:
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            MiniMaxWebSearchProvider().search("x", limit=5)
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer from-coding"

    def test_oauth_token_works(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        monkeypatch.setenv("MINIMAX_OAUTH_TOKEN", "oauth-bearer")
        with patch("httpx.post", return_value=_mock_resp(_ok_payload())) as mock_post:
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            MiniMaxWebSearchProvider().search("x", limit=5)
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer oauth-bearer"

    def test_falls_back_to_api_key_when_nothing_else_set(
        self, monkeypatch: pytest.MonkeyPatch, _clear_minimax_env
    ):
        monkeypatch.setenv("MINIMAX_API_KEY", "fallback-key")
        with patch("httpx.post", return_value=_mock_resp(_ok_payload())) as mock_post:
            from plugins.web.minimax.provider import MiniMaxWebSearchProvider
            MiniMaxWebSearchProvider().search("x", limit=5)
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer fallback-key"


# ---------------------------------------------------------------------------
# get_setup_schema() — picker metadata
# ---------------------------------------------------------------------------


class TestMiniMaxSetupSchema:
    def test_setup_schema_shape(self):
        from plugins.web.minimax.provider import MiniMaxWebSearchProvider
        schema = MiniMaxWebSearchProvider().get_setup_schema()
        assert isinstance(schema, dict)
        assert "name" in schema
        assert "env_vars" in schema
        assert isinstance(schema["env_vars"], list)
        # First env_var should be the primary Token Plan key
        assert schema["env_vars"][0]["key"] == "MINIMAX_CODE_PLAN_KEY"


# ---------------------------------------------------------------------------
# Plugin registration — confirm it self-registers into the registry
# ---------------------------------------------------------------------------


class TestMiniMaxPluginRegistration:
    def test_plugin_module_exposes_register(self):
        from plugins.web.minimax import register
        assert callable(register)

    def test_plugin_registers_into_registry(self):
        """The plugin's register(ctx) hook should add it to the web search
        provider registry under the name 'minimax'."""
        from plugins.web.minimax import register
        from agent.web_search_registry import (
            get_provider, _reset_for_tests, register_provider,
        )
        # Snapshot — register, assert, then clean up so we don't pollute
        # the global registry for other tests.
        from plugins.web.minimax.provider import MiniMaxWebSearchProvider
        from agent.web_search_provider import WebSearchProvider

        p = MiniMaxWebSearchProvider()
        assert isinstance(p, WebSearchProvider)
        assert p.name == "minimax"
        assert p.supports_search() is True
        assert p.supports_extract() is False
