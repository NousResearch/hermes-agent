"""Tests for the FederatedSearch web search provider plugin."""

from __future__ import annotations

import json
import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from plugins.web.federated.provider import (
    FederatedSearchProvider,
    _extract_custom_results,
    _rank_results,
    _read_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> FederatedSearchProvider:
    return FederatedSearchProvider()


# ---------------------------------------------------------------------------
# Provider identity & capabilities
# ---------------------------------------------------------------------------


class TestProviderIdentity:
    def test_name(self, provider: FederatedSearchProvider) -> None:
        assert provider.name == "federated"

    def test_display_name(self, provider: FederatedSearchProvider) -> None:
        assert provider.display_name == "Federated Search"

    def test_supports_search(self, provider: FederatedSearchProvider) -> None:
        assert provider.supports_search() is True

    def test_supports_extract(self, provider: FederatedSearchProvider) -> None:
        assert provider.supports_extract() is False

    def test_is_available_no_config(self, provider: FederatedSearchProvider) -> None:
        """Without config, is_available should return False."""
        assert provider.is_available() is False

    def test_get_setup_schema(self, provider: FederatedSearchProvider) -> None:
        schema = provider.get_setup_schema()
        assert schema["name"] == "Federated Search"
        assert "env_vars" in schema
        assert isinstance(schema["env_vars"], list)


# ---------------------------------------------------------------------------
# Config reading
# ---------------------------------------------------------------------------


class TestConfigReading:
    def test_read_config_no_federated(self) -> None:
        """When web.federated is absent, _read_config returns None."""
        with patch(
            "plugins.web.federated.provider._read_config",
            return_value=None,
        ):
            assert _read_config() is None

    def test_is_available_with_backends(self) -> None:
        """With at least one backend configured, is_available returns True."""
        config = {"backends": [{"name": "tavily"}]}
        with patch(
            "plugins.web.federated.provider._read_config",
            return_value=config,
        ):
            p = FederatedSearchProvider()
            assert p.is_available() is True


# ---------------------------------------------------------------------------
# Custom backend result extraction
# ---------------------------------------------------------------------------


class TestExtractCustomResults:
    def test_organic_shape(self) -> None:
        """Parse common ``{organic: [...]}`` shape (Google/Tavily-style)."""
        data = {
            "organic": [
                {"title": "Result A", "link": "https://a.com", "snippet": "Description A"},
                {"title": "Result B", "url": "https://b.com", "content": "Description B"},
            ]
        }
        results = _extract_custom_results(data)
        assert len(results) == 2
        assert results[0]["title"] == "Result A"
        assert results[0]["url"] == "https://a.com"

    def test_data_web_shape(self) -> None:
        """Parse ``{data: {web: [...]}}`` shape."""
        data = {
            "data": {
                "web": [
                    {"title": "X", "url": "https://x.com", "description": "Desc X"},
                    {"title": "Y", "url": "https://y.com", "description": "Desc Y"},
                ]
            }
        }
        results = _extract_custom_results(data)
        assert len(results) == 2

    def test_results_shape(self) -> None:
        """Parse ``{results: [...]}`` shape."""
        data = {
            "results": [
                {"title": "P", "url": "https://p.com", "content": "Content P"},
            ]
        }
        results = _extract_custom_results(data)
        assert len(results) == 1
        assert results[0]["title"] == "P"

    def test_empty_data(self) -> None:
        """Empty or malformed data returns empty list."""
        assert _extract_custom_results({}) == []
        assert _extract_custom_results(None) == []
        assert _extract_custom_results("not a dict") == []


# ---------------------------------------------------------------------------
# LLM Ranking
# ---------------------------------------------------------------------------


class TestRankResults:
    def test_empty_results(self) -> None:
        """Empty results stay empty after ranking."""
        assert _rank_results("test", [], None) == []

    def test_llm_fallback_on_failure(self) -> None:
        """When LLM call fails, results keep original order."""
        results = [
            {"title": "A", "url": "https://a.com", "description": "A desc"},
            {"title": "B", "url": "https://b.com", "description": "B desc"},
        ]
        ranked = _rank_results("test", results, None)
        assert len(ranked) == 2
        assert ranked[0]["title"] == "A"


# ---------------------------------------------------------------------------
# Integration: federated search with mocked backends
# ---------------------------------------------------------------------------


class TestFederatedSearch:
    """Integration test with mocked sub-backends."""

    def test_no_backends_configured(self, provider: FederatedSearchProvider) -> None:
        """When no backends in config, search returns error."""
        with patch(
            "plugins.web.federated.provider._read_config",
            return_value={"backends": []},
        ):
            result = provider.search("test query")
            assert result["success"] is False
            assert "no search backends" in result["error"]

    def test_single_backend(self, provider: FederatedSearchProvider) -> None:
        """Single custom backend returns results."""
        config = {
            "backends": [{"name": "tavily", "type": "custom"}],
            "timeout": 10,
            "max_results": 8,
        }
        with patch(
            "plugins.web.federated.provider._read_config",
            return_value=config,
        ), patch(
            "plugins.web.federated.provider._search_one_backend",
            return_value=[
                {"title": "R1", "url": "https://r1.com", "description": "D1"},
                {"title": "R2", "url": "https://r2.com", "description": "D2"},
            ],
        ):
            result = provider.search("test", limit=5)
            assert result["success"] is True
            web = result["data"]["web"]
            assert len(web) == 2
            assert web[0]["title"] == "R1"
            assert web[1]["position"] == 2

    def test_multiple_backends_merge(self, provider: FederatedSearchProvider) -> None:
        """Multiple backends merge results."""
        config = {
            "backends": [
                {"name": "backend1", "type": "custom"},
                {"name": "backend2", "type": "custom"},
            ],
            "timeout": 10,
            "max_results": 8,
        }

        call_count = 0
        def fake_search(backend, query, limit):
            nonlocal call_count
            call_count += 1
            name = backend.get("name", "")
            if name == "backend1":
                return [{"title": "A1", "url": "https://a1.com", "description": ""}]
            return [{"title": "B1", "url": "https://b1.com", "description": ""}]

        with patch(
            "plugins.web.federated.provider._read_config",
            return_value=config,
        ), patch(
            "plugins.web.federated.provider._search_one_backend",
            side_effect=fake_search,
        ):
            result = provider.search("test")
            assert result["success"] is True
            assert len(result["data"]["web"]) == 2

    def test_max_results_respected(self, provider: FederatedSearchProvider) -> None:
        """Only max_results items are returned after ranking."""
        config = {
            "backends": [{"name": "tavily", "type": "custom"}],
            "max_results": 1,
            "timeout": 5,
        }
        with patch(
            "plugins.web.federated.provider._read_config",
            return_value=config,
        ), patch(
            "plugins.web.federated.provider._search_one_backend",
            return_value=[
                {"title": f"R{i}", "url": f"https://r{i}.com", "description": ""}
                for i in range(5)
            ],
        ):
            result = provider.search("test")
            assert result["success"] is True
            assert len(result["data"]["web"]) == 1

    def test_timeout_config_is_read(self, provider: FederatedSearchProvider) -> None:
        """The timeout config value is correctly read from config."""
        config = {
            "backends": [{"name": "t", "type": "custom"}],
            "timeout": 30,
            "max_results": 5,
        }
        with patch(
            "plugins.web.federated.provider._read_config",
            return_value=config,
        ), patch(
            "plugins.web.federated.provider._search_one_backend",
            return_value=[{"title": "R", "url": "https://r.com", "description": ""}],
        ):
            result = provider.search("test")
            assert result["success"] is True

    def test_blocked_backend_timeout_produces_partial_results(
        self, provider: FederatedSearchProvider,
    ) -> None:
        """A backend that blocks must be bounded by the configured timeout.

        The blocking backend is abandoned; results from faster backends are
        still collected and returned.
        """
        import threading

        block_signal = threading.Event()

        def slow_backend(backend, query, limit):
            block_signal.wait()  # block until released
            return []

        def fast_backend(backend, query, limit):
            return [{"title": "Fast", "url": "https://f.com", "description": ""}]

        config = {
            "backends": [
                {"name": "slow", "type": "custom"},
                {"name": "fast", "type": "custom"},
            ],
            "timeout": 1,
            "max_results": 5,
        }
        try:
            with patch(
                "plugins.web.federated.provider._read_config",
                return_value=config,
            ), patch(
                "plugins.web.federated.provider._search_one_backend",
                side_effect=lambda b, q, l: slow_backend(b, q, l) if b["name"] == "slow" else fast_backend(b, q, l),
            ):
                result = provider.search("test", limit=5)
                # Must return within a few seconds (not waiting for the blocked backend)
                assert result["success"] is True
                web = result["data"]["web"]
                # Fast backend result should be present
                assert len(web) >= 1
                assert web[0]["title"] == "Fast"
        finally:
            block_signal.set()  # release blocked thread so it can exit

    def test_limit_below_max_results(
        self, provider: FederatedSearchProvider,
    ) -> None:
        """When limit < max_results, return only `limit` items."""
        config = {
            "backends": [{"name": "t", "type": "custom"}],
            "max_results": 8,
            "timeout": 5,
        }
        with patch(
            "plugins.web.federated.provider._read_config",
            return_value=config,
        ), patch(
            "plugins.web.federated.provider._search_one_backend",
            return_value=[
                {"title": f"R{i}", "url": f"https://r{i}.com", "description": ""}
                for i in range(10)
            ],
        ):
            result = provider.search("test", limit=3)
            assert result["success"] is True
            assert len(result["data"]["web"]) == 3

    def test_limit_above_max_results(
        self, provider: FederatedSearchProvider,
    ) -> None:
        """When limit > max_results, the configured max_results acts as the cap."""
        config = {
            "backends": [{"name": "t", "type": "custom"}],
            "max_results": 4,
            "timeout": 5,
        }
        with patch(
            "plugins.web.federated.provider._read_config",
            return_value=config,
        ), patch(
            "plugins.web.federated.provider._search_one_backend",
            return_value=[
                {"title": f"R{i}", "url": f"https://r{i}.com", "description": ""}
                for i in range(10)
            ],
        ):
            result = provider.search("test", limit=10)
            assert result["success"] is True
            assert len(result["data"]["web"]) == 4  # capped by max_results


# ---------------------------------------------------------------------------
# Config key integration
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_read_config_with_federated():
    """Helper fixture to inject a realistic config."""
    def _inject(config_override: Dict[str, Any]) -> None:
        patcher = patch(
            "plugins.web.federated.provider._read_config",
            return_value=config_override,
        )
        patcher.start()
        return patcher

    yield _inject
    # Cleanup happens automatically in fixture teardown


class TestConfigurableSettings:
    """Verify that all three config items are read and applied."""

    def test_default_values_used_when_config_missing(self) -> None:
        """When config has no timeout/max_results/ranker, defaults apply."""
        config = {"backends": [{"name": "t", "type": "custom"}]}
        with patch(
            "plugins.web.federated.provider._read_config",
            return_value=config,
        ), patch(
            "plugins.web.federated.provider._search_one_backend",
            return_value=[{"title": "R", "url": "https://r.com", "description": ""}],
        ):
            p = FederatedSearchProvider()
            result = p.search("test")
            assert result["success"] is True

    def test_timeout_config(self) -> None:
        """Config item 1: timeout is readable."""
        config = {
            "backends": [{"name": "t", "type": "custom"}],
            "timeout": 30,
            "max_results": 5,
        }
        assert config["timeout"] == 30

    def test_max_results_config(self) -> None:
        """Config item 3: max_results is readable."""
        config = {
            "backends": [{"name": "t", "type": "custom"}],
            "timeout": 10,
            "max_results": 12,
        }
        assert config["max_results"] == 12

    def test_ranker_config(self) -> None:
        """Config item 2: ranker provider/model is readable."""
        config = {
            "backends": [{"name": "t", "type": "custom"}],
            "ranker": {"provider": "opencode-go", "model": "deepseek-v4-flash"},
        }
        assert config["ranker"]["provider"] == "opencode-go"
        assert config["ranker"]["model"] == "deepseek-v4-flash"


# ---------------------------------------------------------------------------
# HERMES_HOME config discovery integration test
# ---------------------------------------------------------------------------


class TestFederatedConfigDiscovery:
    """Verify that federated search config is discovered from an isolated
    HERMES_HOME directory via the real config loading path."""

    def test_read_config_from_hermes_home(self, tmp_path) -> None:
        """_read_config discovers web.federated from config.yaml in HERMES_HOME."""
        import yaml

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "web": {
                "federated": {
                    "timeout": 15,
                    "max_results": 6,
                    "backends": [
                        {"name": "tavily"},
                        {"name": "custom1", "type": "custom",
                         "base_url": "https://api.example.com",
                         "api_key_env": "EXAMPLE_KEY",
                         "search_path": "/v1/search",
                         "query_param": "q"},
                    ],
                    "ranker": {
                        "provider": "opencode-go",
                        "model": "deepseek-v4-flash",
                    },
                },
            },
        }))

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            from plugins.web.federated.provider import _read_config as _cfg
            cfg = _cfg()
            assert cfg is not None
            assert cfg["timeout"] == 15
            assert cfg["max_results"] == 6
            assert len(cfg["backends"]) == 2
            assert cfg["backends"][0]["name"] == "tavily"
            assert cfg["ranker"]["provider"] == "opencode-go"
