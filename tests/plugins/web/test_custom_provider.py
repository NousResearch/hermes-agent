"""Tests for the custom OpenAI-compatible web search plugin.

Covers the plugin form of the custom backend (search + extract), its env
vs config.yaml resolution helpers, and the search-result extraction
priority (search_results[] → citations[] → answer text). All HTTP calls
are stubbed via ``monkeypatch`` on ``httpx.post`` so the suite stays
offline.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest


_CUSTOM_ENV_KEYS = (
    "CUSTOM_SEARCH_API_KEY",
    "CUSTOM_SEARCH_BASE_URL",
    "CUSTOM_SEARCH_MODEL",
)


@pytest.fixture(autouse=True)
def _clear_custom_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every CUSTOM_SEARCH_* env var so tests start clean."""
    for k in _CUSTOM_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def _empty_web_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``load_config()`` return an empty dict so the config fallback misses."""
    from hermes_cli import config as _config_mod

    monkeypatch.setattr(_config_mod, "load_config", lambda: {})


def _ensure_plugins_loaded() -> None:
    from hermes_cli.plugins import _ensure_plugins_discovered

    _ensure_plugins_discovered()


def _get_custom_provider():
    from agent.web_search_registry import get_provider

    _ensure_plugins_loaded()
    return get_provider("custom")


# ---------------------------------------------------------------------------
# Registration + capability flags
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_plugin_registers_with_name_custom(self) -> None:
        provider = _get_custom_provider()
        assert provider is not None
        assert provider.name == "custom"

    def test_display_name_is_human_readable(self) -> None:
        provider = _get_custom_provider()
        assert provider is not None
        assert provider.display_name == "Custom (OpenAI-compatible)"

    def test_capability_flags(self) -> None:
        provider = _get_custom_provider()
        assert provider is not None
        assert provider.supports_search() is True
        assert provider.supports_extract() is True
        assert provider.supports_crawl() is False

    def test_setup_schema_exposes_three_env_vars(self) -> None:
        provider = _get_custom_provider()
        assert provider is not None
        schema = provider.get_setup_schema()
        assert isinstance(schema, dict)
        keys = [v["key"] for v in schema["env_vars"]]
        assert keys == [
            "CUSTOM_SEARCH_API_KEY",
            "CUSTOM_SEARCH_BASE_URL",
            "CUSTOM_SEARCH_MODEL",
        ]


# ---------------------------------------------------------------------------
# is_available()
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_false_when_no_key_anywhere(self, _empty_web_config: None) -> None:
        provider = _get_custom_provider()
        assert provider is not None
        assert provider.is_available() is False

    def test_true_when_env_key_set(
        self, monkeypatch: pytest.MonkeyPatch, _empty_web_config: None
    ) -> None:
        provider = _get_custom_provider()
        assert provider is not None
        monkeypatch.setenv("CUSTOM_SEARCH_API_KEY", "real")
        assert provider.is_available() is True

    def test_true_when_config_key_set_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config-only api_key (no env) should still light up the provider."""
        from hermes_cli import config as _config_mod

        monkeypatch.setattr(
            _config_mod,
            "load_config",
            lambda: {"web": {"custom_api_key": "from-config"}},
        )
        provider = _get_custom_provider()
        assert provider is not None
        assert provider.is_available() is True


# ---------------------------------------------------------------------------
# Resolution helpers — env vs config priority
# ---------------------------------------------------------------------------


class TestResolution:
    def test_api_key_env_wins_over_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hermes_cli import config as _config_mod
        from plugins.web.custom.provider import _resolve_api_key

        monkeypatch.setenv("CUSTOM_SEARCH_API_KEY", "from-env")
        monkeypatch.setattr(
            _config_mod,
            "load_config",
            lambda: {"web": {"custom_api_key": "from-config"}},
        )
        assert _resolve_api_key() == "from-env"

    def test_api_key_falls_back_to_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hermes_cli import config as _config_mod
        from plugins.web.custom.provider import _resolve_api_key

        monkeypatch.setattr(
            _config_mod,
            "load_config",
            lambda: {"web": {"custom_api_key": "from-config"}},
        )
        assert _resolve_api_key() == "from-config"

    def test_base_url_strips_trailing_slash(
        self, monkeypatch: pytest.MonkeyPatch, _empty_web_config: None
    ) -> None:
        from plugins.web.custom.provider import _resolve_base_url

        monkeypatch.setenv("CUSTOM_SEARCH_BASE_URL", "https://api.example.com/")
        assert _resolve_base_url() == "https://api.example.com"

    def test_base_url_env_wins_over_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hermes_cli import config as _config_mod
        from plugins.web.custom.provider import _resolve_base_url

        monkeypatch.setenv("CUSTOM_SEARCH_BASE_URL", "https://env.example.com")
        monkeypatch.setattr(
            _config_mod,
            "load_config",
            lambda: {"web": {"custom_base_url": "https://cfg.example.com"}},
        )
        assert _resolve_base_url() == "https://env.example.com"

    def test_model_defaults_to_sonar(self, _empty_web_config: None) -> None:
        from plugins.web.custom.provider import _resolve_model

        assert _resolve_model() == "sonar"

    def test_model_env_wins_over_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hermes_cli import config as _config_mod
        from plugins.web.custom.provider import _resolve_model

        monkeypatch.setenv("CUSTOM_SEARCH_MODEL", "env-model")
        monkeypatch.setattr(
            _config_mod,
            "load_config",
            lambda: {"web": {"custom_model": "cfg-model"}},
        )
        assert _resolve_model() == "env-model"

    def test_bearer_header_constructed(self) -> None:
        from plugins.web.custom.provider import _build_headers

        headers = _build_headers("sk-abc")
        assert headers["Authorization"] == "Bearer sk-abc"
        assert headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# search() — result extraction priority
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


def _stub_chat(monkeypatch: pytest.MonkeyPatch, payload: Dict[str, Any]) -> dict:
    """Stub httpx.post to return ``payload`` and record the call args."""
    captured: dict = {}

    def _fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResponse(payload)

    import httpx
    monkeypatch.setattr(httpx, "post", _fake_post)
    return captured


@pytest.fixture
def _configured_custom(monkeypatch: pytest.MonkeyPatch, _empty_web_config: None) -> None:
    """Set the minimum env vars so the provider can make a (stubbed) call."""
    monkeypatch.setenv("CUSTOM_SEARCH_API_KEY", "test-key")
    monkeypatch.setenv("CUSTOM_SEARCH_BASE_URL", "https://api.example.com")


class TestSearch:
    def test_prefers_search_results(
        self, monkeypatch: pytest.MonkeyPatch, _configured_custom: None
    ) -> None:
        _stub_chat(monkeypatch, {
            "search_results": [
                {"title": "T1", "url": "https://a", "snippet": "S1"},
                {"title": "T2", "url": "https://b", "content": "C2"},
            ],
            "citations": ["https://ignored"],
            "choices": [{"message": {"content": "ignored"}}],
        })
        provider = _get_custom_provider()
        result = provider.search("q", limit=5)
        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 2
        assert web[0] == {
            "title": "T1", "url": "https://a", "description": "S1", "position": 1,
        }
        assert web[1]["description"] == "C2"

    def test_falls_back_to_citations_strings(
        self, monkeypatch: pytest.MonkeyPatch, _configured_custom: None
    ) -> None:
        _stub_chat(monkeypatch, {
            "citations": ["https://a", "https://b"],
            "choices": [{"message": {"content": "ignored"}}],
        })
        provider = _get_custom_provider()
        result = provider.search("q", limit=5)
        web = result["data"]["web"]
        assert len(web) == 2
        assert web[0]["url"] == "https://a"
        assert web[0]["title"] == ""
        assert web[1]["position"] == 2

    def test_falls_back_to_citations_dicts(
        self, monkeypatch: pytest.MonkeyPatch, _configured_custom: None
    ) -> None:
        _stub_chat(monkeypatch, {
            "citations": [
                {"title": "T", "url": "https://x", "snippet": "S"},
            ],
        })
        provider = _get_custom_provider()
        result = provider.search("q", limit=5)
        web = result["data"]["web"]
        assert web[0] == {
            "title": "T", "url": "https://x", "description": "S", "position": 1,
        }

    def test_last_resort_answer_text(
        self, monkeypatch: pytest.MonkeyPatch, _configured_custom: None
    ) -> None:
        _stub_chat(monkeypatch, {
            "choices": [{"message": {"content": "the answer"}}],
        })
        provider = _get_custom_provider()
        result = provider.search("q", limit=5)
        web = result["data"]["web"]
        assert len(web) == 1
        assert web[0]["title"] == "Search Answer"
        assert web[0]["description"] == "the answer"

    def test_empty_response_returns_empty_results(
        self, monkeypatch: pytest.MonkeyPatch, _configured_custom: None
    ) -> None:
        _stub_chat(monkeypatch, {})
        provider = _get_custom_provider()
        result = provider.search("q", limit=5)
        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_limit_enforced_on_search_results(
        self, monkeypatch: pytest.MonkeyPatch, _configured_custom: None
    ) -> None:
        _stub_chat(monkeypatch, {
            "search_results": [
                {"title": f"T{i}", "url": f"https://{i}", "snippet": ""}
                for i in range(10)
            ],
        })
        provider = _get_custom_provider()
        result = provider.search("q", limit=3)
        assert len(result["data"]["web"]) == 3

    def test_missing_api_key_returns_error_dict(
        self, _empty_web_config: None
    ) -> None:
        provider = _get_custom_provider()
        result = provider.search("q")
        assert result["success"] is False
        assert "API key" in result["error"]

    def test_missing_base_url_returns_error_dict(
        self, monkeypatch: pytest.MonkeyPatch, _empty_web_config: None
    ) -> None:
        monkeypatch.setenv("CUSTOM_SEARCH_API_KEY", "key-only")
        provider = _get_custom_provider()
        result = provider.search("q")
        assert result["success"] is False
        assert "base URL" in result["error"]

    def test_search_uses_resolved_model_and_url(
        self, monkeypatch: pytest.MonkeyPatch, _empty_web_config: None
    ) -> None:
        monkeypatch.setenv("CUSTOM_SEARCH_API_KEY", "k")
        monkeypatch.setenv("CUSTOM_SEARCH_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("CUSTOM_SEARCH_MODEL", "my-model")
        captured = _stub_chat(monkeypatch, {"search_results": []})
        provider = _get_custom_provider()
        provider.search("hello")
        assert captured["url"] == "https://api.example.com/chat/completions"
        assert captured["kwargs"]["json"]["model"] == "my-model"
        assert captured["kwargs"]["json"]["messages"][0]["content"] == "hello"
        assert captured["kwargs"]["headers"]["Authorization"] == "Bearer k"


# ---------------------------------------------------------------------------
# extract() — per-URL isolation
# ---------------------------------------------------------------------------


class TestExtract:
    def test_extract_is_sync(self) -> None:
        """Custom extract is synchronous — async is not required."""
        import inspect

        provider = _get_custom_provider()
        assert inspect.iscoroutinefunction(provider.extract) is False

    def test_extract_returns_documents(
        self, monkeypatch: pytest.MonkeyPatch, _configured_custom: None
    ) -> None:
        _stub_chat(monkeypatch, {
            "choices": [{"message": {"content": "# markdown content"}}],
        })
        provider = _get_custom_provider()
        results = provider.extract(["https://a", "https://b"])
        assert len(results) == 2
        assert results[0]["url"] == "https://a"
        assert results[0]["content"] == "# markdown content"
        assert results[0]["raw_content"] == "# markdown content"
        assert results[0]["metadata"]["sourceURL"] == "https://a"

    def test_extract_empty_list_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch, _configured_custom: None
    ) -> None:
        called = {"n": 0}

        def _fake_post(*a: Any, **kw: Any) -> Any:
            called["n"] += 1
            raise AssertionError("should not be called on empty list")

        import httpx
        monkeypatch.setattr(httpx, "post", _fake_post)

        provider = _get_custom_provider()
        assert provider.extract([]) == []
        assert called["n"] == 0

    def test_extract_per_url_failures_isolated(
        self, monkeypatch: pytest.MonkeyPatch, _configured_custom: None
    ) -> None:
        """One failing URL doesn't poison the rest."""
        seen: list[str] = []

        def _fake_post(url: str, **kwargs: Any) -> _FakeResponse:
            content = kwargs["json"]["messages"][0]["content"]
            target = content.split(": ", 1)[1].splitlines()[0]
            seen.append(target)
            if target == "https://bad":
                raise RuntimeError("boom")
            return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

        import httpx
        monkeypatch.setattr(httpx, "post", _fake_post)

        provider = _get_custom_provider()
        results = provider.extract(["https://good", "https://bad", "https://also-good"])
        assert len(results) == 3
        assert results[0]["content"] == "ok"
        assert "error" in results[1]
        assert "boom" in results[1]["error"]
        assert results[2]["content"] == "ok"
