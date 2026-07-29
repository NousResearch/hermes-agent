"""Tests for the bundled BackSearch plugin (point-in-time web search/fetch).

Real imports from the plugin module — no mocking of the handlers
themselves. HTTP is stubbed at the httpx layer; no live network calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from plugins.backsearch.tools import (
    BACKFETCH_SCHEMA,
    BACKSEARCH_SCHEMA,
    DEFAULT_BASE_URL,
    _validate_as_of,
    check_backsearch_available,
    handle_backfetch,
    handle_backsearch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int = 200, payload: dict | None = None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload or {}
    if status_code >= 400:
        import httpx

        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


SEARCH_HIT = {
    "url": "https://www.example.com/article",
    "title": "UAE Central Bank cuts interest rates",
    "snippet": "The UAE Central Bank on Wednesday lowered...",
    "crawl_date": "2025-12-10T20:24:28Z",
    "publish_date": "2025-12-10T00:00:00Z",
    "host": "www.example.com",
}


@pytest.fixture(autouse=True)
def _key_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENREWARD_API_KEY", "or_test_key")
    monkeypatch.delenv("OPENREWARD_SEARCH_URL", raising=False)


# ---------------------------------------------------------------------------
# Availability gating
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_available_with_key(self):
        assert check_backsearch_available() is True

    def test_unavailable_without_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENREWARD_API_KEY", raising=False)
        # get_env_value may read ~/.hermes/.env; force the plain-os path
        with patch(
            "plugins.backsearch.tools._get_env", return_value=""
        ):
            assert check_backsearch_available() is False

    def test_search_without_key_returns_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        with patch("plugins.backsearch.tools._get_env", return_value=""):
            out = json.loads(
                handle_backsearch({"query": "rates", "as_of": "2026-01-15"})
            )
        assert "OPENREWARD_API_KEY" in out["error"]


# ---------------------------------------------------------------------------
# as_of validation
# ---------------------------------------------------------------------------


class TestAsOfValidation:
    def test_valid(self):
        assert _validate_as_of("2026-01-15") == "2026-01-15"

    @pytest.mark.parametrize("bad", ["", None, "01/15/2026", "2026-1-5", "jan 15"])
    def test_invalid(self, bad):
        with pytest.raises(ValueError):
            _validate_as_of(bad)

    def test_handler_surfaces_as_of_error(self):
        out = json.loads(handle_backsearch({"query": "x", "as_of": "not-a-date"}))
        assert "as_of" in out["error"]

    def test_missing_query(self):
        out = json.loads(handle_backsearch({"as_of": "2026-01-15"}))
        assert "query" in out["error"]


# ---------------------------------------------------------------------------
# Search behaviour
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_success_shape(self):
        with patch(
            "httpx.post",
            return_value=_mock_response(200, {"mode": "hybrid", "hits": [SEARCH_HIT]}),
        ) as post:
            out = json.loads(
                handle_backsearch({"query": "central bank", "as_of": "2026-01-15"})
            )
        assert out["success"] is True
        assert out["as_of"] == "2026-01-15"
        hit = out["hits"][0]
        assert hit["url"] == SEARCH_HIT["url"]
        assert hit["crawl_date"] == SEARCH_HIT["crawl_date"]
        assert hit["publish_date"] == SEARCH_HIT["publish_date"]
        # request went to the right endpoint with the auth header
        url = post.call_args.args[0] if post.call_args.args else post.call_args.kwargs["url"]
        assert url == f"{DEFAULT_BASE_URL}/search"
        assert post.call_args.kwargs["headers"]["x-api-key"] == "or_test_key"
        body = post.call_args.kwargs["json"]
        assert body["as_of"] == "2026-01-15"
        assert body["query"] == "central bank"

    def test_empty_hits_appends_archive_window_hint(self):
        with patch("httpx.post", return_value=_mock_response(200, {"hits": []})):
            out = json.loads(
                handle_backsearch({"query": "anything", "as_of": "2026-01-15"})
            )
        assert out["success"] is True
        assert out["hits"] == []
        assert "preview archive" in out["note"]

    def test_allowed_and_blocked_domains_mutually_exclusive(self):
        out = json.loads(
            handle_backsearch(
                {
                    "query": "x",
                    "as_of": "2026-01-15",
                    "allowed_domains": ["a.com"],
                    "blocked_domains": ["b.com"],
                }
            )
        )
        assert "never both" in out["error"]

    def test_k_is_clamped(self):
        with patch(
            "httpx.post", return_value=_mock_response(200, {"hits": []})
        ) as post:
            handle_backsearch({"query": "x", "as_of": "2026-01-15", "k": 500})
        assert post.call_args.kwargs["json"]["k"] == 20

    def test_domain_list_accepts_comma_string(self):
        with patch(
            "httpx.post", return_value=_mock_response(200, {"hits": []})
        ) as post:
            handle_backsearch(
                {
                    "query": "x",
                    "as_of": "2026-01-15",
                    "allowed_domains": "a.com, b.com",
                }
            )
        assert post.call_args.kwargs["json"]["allowed_domains"] == ["a.com", "b.com"]

    def test_402_returns_balance_error(self):
        with patch("httpx.post", return_value=_mock_response(402)):
            out = json.loads(
                handle_backsearch({"query": "x", "as_of": "2026-01-15"})
            )
        assert "balance" in out["error"].lower()

    def test_401_returns_key_error(self):
        with patch("httpx.post", return_value=_mock_response(401)):
            out = json.loads(
                handle_backsearch({"query": "x", "as_of": "2026-01-15"})
            )
        assert "OPENREWARD_API_KEY" in out["error"]

    def test_network_error_is_soft(self):
        with patch("httpx.post", side_effect=OSError("boom")):
            out = json.loads(
                handle_backsearch({"query": "x", "as_of": "2026-01-15"})
            )
        assert "failed" in out["error"].lower()


# ---------------------------------------------------------------------------
# Fetch behaviour
# ---------------------------------------------------------------------------


class TestFetch:
    def test_fetch_success(self):
        with patch(
            "httpx.post",
            return_value=_mock_response(
                200,
                {
                    "text": "The article body.",
                    "title": "Headline",
                    "crawl_date": "2025-12-10T20:24:28Z",
                },
            ),
        ) as post:
            out = json.loads(
                handle_backfetch(
                    {"url": "https://example.com/a", "as_of": "2026-01-15"}
                )
            )
        assert out["success"] is True
        assert out["text"] == "The article body."
        assert out["title"] == "Headline"
        url = post.call_args.args[0] if post.call_args.args else post.call_args.kwargs["url"]
        assert url == f"{DEFAULT_BASE_URL}/fetch"

    def test_fetch_404_no_capture_is_soft_error(self):
        with patch("httpx.post", return_value=_mock_response(404)):
            out = json.loads(
                handle_backfetch(
                    {"url": "https://example.com/a", "as_of": "2026-01-15"}
                )
            )
        assert "capture" in out["error"].lower()

    def test_fetch_missing_url(self):
        out = json.loads(handle_backfetch({"as_of": "2026-01-15"}))
        assert "url" in out["error"]

    def test_fetch_prompt_sets_summarize(self):
        with patch(
            "httpx.post", return_value=_mock_response(200, {"text": "summary"})
        ) as post:
            handle_backfetch(
                {
                    "url": "https://example.com/a",
                    "as_of": "2026-01-15",
                    "prompt": "what rate?",
                }
            )
        body = post.call_args.kwargs["json"]
        assert body["prompt"] == "what rate?"
        assert body["summarize"] is True

    def test_fetch_no_prompt_omits_summarize(self):
        with patch(
            "httpx.post", return_value=_mock_response(200, {"text": "t"})
        ) as post:
            handle_backfetch(
                {"url": "https://example.com/a", "as_of": "2026-01-15"}
            )
        body = post.call_args.kwargs["json"]
        assert "summarize" not in body
        assert "prompt" not in body

    def test_fetch_long_text_truncated_with_note(self):
        from plugins.backsearch.tools import _FETCH_TEXT_CAP

        with patch(
            "httpx.post",
            return_value=_mock_response(200, {"text": "x" * (_FETCH_TEXT_CAP + 100)}),
        ):
            out = json.loads(
                handle_backfetch(
                    {"url": "https://example.com/a", "as_of": "2026-01-15"}
                )
            )
        assert out["truncated"] is True
        assert len(out["text"]) == _FETCH_TEXT_CAP

    def test_base_url_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENREWARD_SEARCH_URL", "http://localhost:9999/")
        with patch(
            "httpx.post", return_value=_mock_response(200, {"text": "t"})
        ) as post:
            handle_backfetch(
                {"url": "https://example.com/a", "as_of": "2026-01-15"}
            )
        url = post.call_args.args[0] if post.call_args.args else post.call_args.kwargs["url"]
        assert url == "http://localhost:9999/fetch"


# ---------------------------------------------------------------------------
# Registration + toolset wiring (behavioural contracts, not snapshots)
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_plugin_registers_both_tools(self):
        import plugins.backsearch as plugin_mod

        registered = []

        class _Ctx:
            def register_tool(self, **kwargs):
                registered.append(kwargs)

        plugin_mod.register(_Ctx())
        names = {r["name"] for r in registered}
        assert names == {"backsearch", "backfetch"}
        for r in registered:
            assert r["toolset"] == "backsearch"
            assert r["check_fn"] is check_backsearch_available
            assert r["requires_env"] == ["OPENREWARD_API_KEY"]
            assert callable(r["handler"])

    def test_toolset_defined_with_plugin_tools(self):
        from toolsets import TOOLSETS

        ts = TOOLSETS["backsearch"]
        assert set(ts["tools"]) == {"backsearch", "backfetch"}

    def test_schemas_declare_required_params(self):
        assert set(BACKSEARCH_SCHEMA["parameters"]["required"]) == {"query", "as_of"}
        assert set(BACKFETCH_SCHEMA["parameters"]["required"]) == {"url", "as_of"}
        # schema names match the registered tool names
        assert BACKSEARCH_SCHEMA["name"] == "backsearch"
        assert BACKFETCH_SCHEMA["name"] == "backfetch"

    def test_configurable_in_hermes_tools_and_default_off(self):
        from hermes_cli.tools_config import (
            _DEFAULT_OFF_TOOLSETS,
            CONFIGURABLE_TOOLSETS,
            TOOL_CATEGORIES,
        )

        keys = {ts for ts, _, _ in CONFIGURABLE_TOOLSETS}
        assert "backsearch" in keys
        assert "backsearch" in _DEFAULT_OFF_TOOLSETS
        env_keys = [
            ev["key"]
            for prov in TOOL_CATEGORIES["backsearch"]["providers"]
            for ev in prov["env_vars"]
        ]
        assert "OPENREWARD_API_KEY" in env_keys

    def test_env_var_documented_in_optional_env_vars(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        meta = OPTIONAL_ENV_VARS["OPENREWARD_API_KEY"]
        assert meta["password"] is True
        assert meta["category"] == "tool"

    def test_auto_enable_helper_reflects_key_presence(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from hermes_cli.tools_config import _backsearch_credentials_present

        assert _backsearch_credentials_present() is True
        monkeypatch.delenv("OPENREWARD_API_KEY", raising=False)
        with patch("plugins.backsearch.tools._get_env", return_value=""):
            assert _backsearch_credentials_present() is False
