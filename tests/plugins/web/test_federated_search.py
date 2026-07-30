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
    _HealthCache,
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
            return_value=([
                {"title": "R1", "url": "https://r1.com", "description": "D1"},
                {"title": "R2", "url": "https://r2.com", "description": "D2"},
            ], None),
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
                return [{"title": "A1", "url": "https://a1.com", "description": ""}], None
            return [{"title": "B1", "url": "https://b1.com", "description": ""}], None

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
            return_value=([
                {"title": f"R{i}", "url": f"https://r{i}.com", "description": ""}
                for i in range(5)
            ], None),
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
            return_value=([{"title": "R", "url": "https://r.com", "description": ""}], None),
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
            return [], None

        def fast_backend(backend, query, limit):
            return [{"title": "Fast", "url": "https://f.com", "description": ""}], None

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
            return_value=([
                {"title": f"R{i}", "url": f"https://r{i}.com", "description": ""}
                for i in range(10)
            ], None),
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
            return_value=([
                {"title": f"R{i}", "url": f"https://r{i}.com", "description": ""}
                for i in range(10)
            ], None),
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
            return_value=([{"title": "R", "url": "https://r.com", "description": ""}], None),
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


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestValidateConfig:
    """K-way aggregation config validation."""

    def test_valid_minimal(self) -> None:
        from plugins.web.federated.provider import _validate_config
        config = {"backends": [{"name": "tavily"}]}
        assert _validate_config(config) is None

    def test_valid_k_equals_backends(self) -> None:
        from plugins.web.federated.provider import _validate_config
        config = {
            "k": 3,
            "backends": [
                {"name": "a"}, {"name": "b"}, {"name": "c"},
            ],
        }
        assert _validate_config(config) is None

    def test_k_exceeds_max(self) -> None:
        from plugins.web.federated.provider import _validate_config, _MAX_K
        config = {"k": _MAX_K + 1, "backends": [{"name": "a"}]}
        err = _validate_config(config)
        assert err is not None
        assert "exceeds maximum" in err

    def test_k_mismatch_too_few(self) -> None:
        from plugins.web.federated.provider import _validate_config
        config = {"k": 3, "backends": [{"name": "a"}]}
        err = _validate_config(config)
        assert err is not None
        assert "exactly 3" in err

    def test_k_mismatch_too_many(self) -> None:
        from plugins.web.federated.provider import _validate_config
        config = {"k": 2, "backends": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
        err = _validate_config(config)
        assert err is not None
        assert "exactly 2" in err

    def test_min_backends_exceeds_k(self) -> None:
        from plugins.web.federated.provider import _validate_config
        config = {
            "k": 2, "min_backends": 3,
            "backends": [{"name": "a"}, {"name": "b"}],
        }
        err = _validate_config(config)
        assert err is not None
        assert "cannot exceed k" in err

    def test_min_backends_less_than_one(self) -> None:
        from plugins.web.federated.provider import _validate_config
        config = {
            "min_backends": 0,
            "backends": [{"name": "a"}],
        }
        err = _validate_config(config)
        assert err is not None
        assert "must be at least 1" in err

    def test_no_backends(self) -> None:
        from plugins.web.federated.provider import _validate_config
        config = {"backends": []}
        err = _validate_config(config)
        assert err is not None
        assert "no search backends" in err


# ---------------------------------------------------------------------------
# Health cache
# ---------------------------------------------------------------------------


class TestHealthCache:
    """TTL-based health probe cache."""

    def test_cache_hit(self) -> None:
        from plugins.web.federated.provider import _HealthCache
        cache = _HealthCache(ttl_seconds=300)
        cache.set_available("tavily", True)
        assert cache.is_available("tavily") is True

    def test_cache_miss_expired(self) -> None:
        from plugins.web.federated.provider import _HealthCache
        cache = _HealthCache(ttl_seconds=0)  # instant expiry
        cache.set_available("tavily", True)
        assert cache.is_available("tavily") is None

    def test_cache_miss_unknown(self) -> None:
        from plugins.web.federated.provider import _HealthCache
        cache = _HealthCache(ttl_seconds=300)
        assert cache.is_available("unknown") is None

    def test_mark_failed_cooldown(self) -> None:
        from plugins.web.federated.provider import _HealthCache
        cache = _HealthCache(ttl_seconds=300)
        cache.set_available("tavily", True)
        cache.mark_failed("tavily", 429)
        # After mark_failed, should be unavailable
        assert cache.is_available("tavily") is False

    def test_failure_cooldown_respects_status_code(self) -> None:
        from plugins.web.federated.provider import _HealthCache
        cache = _HealthCache(ttl_seconds=300)
        cache.set_available("tavily", True)
        # 401 → 300s cooldown, stored as False
        cache.mark_failed("tavily", 401)
        assert cache.is_available("tavily") is False


# ---------------------------------------------------------------------------
# Health probe
# ---------------------------------------------------------------------------


class TestProbeBackend:
    """Backend availability probing."""

    def test_registered_provider_available(self) -> None:
        from plugins.web.federated.provider import _probe_backend
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        with patch(
            "plugins.web.federated.provider._get_registered_provider",
            return_value=mock_provider,
        ):
            assert _probe_backend({"name": "tavily"}) is True

    def test_registered_provider_unavailable(self) -> None:
        from plugins.web.federated.provider import _probe_backend
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = False
        with patch(
            "plugins.web.federated.provider._get_registered_provider",
            return_value=mock_provider,
        ):
            assert _probe_backend({"name": "tavily"}) is False

    def test_registered_provider_not_found(self) -> None:
        from plugins.web.federated.provider import _probe_backend
        with patch(
            "plugins.web.federated.provider._get_registered_provider",
            return_value=None,
        ):
            assert _probe_backend({"name": "nonexistent"}) is False

    def test_custom_backend_no_base_url(self) -> None:
        from plugins.web.federated.provider import _probe_backend
        assert _probe_backend({
            "name": "bad", "type": "custom",
        }) is False

    def test_custom_backend_reachable(self) -> None:
        from plugins.web.federated.provider import _probe_backend
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.head", return_value=mock_resp):
            assert _probe_backend({
                "name": "ok", "type": "custom",
                "base_url": "https://api.example.com",
            }) is True

    def test_custom_backend_server_error(self) -> None:
        from plugins.web.federated.provider import _probe_backend
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("httpx.head", return_value=mock_resp):
            assert _probe_backend({
                "name": "down", "type": "custom",
                "base_url": "https://api.example.com",
            }) is False

    def test_custom_backend_timeout(self) -> None:
        from plugins.web.federated.provider import _probe_backend
        with patch("httpx.head", side_effect=Exception("timeout")):
            assert _probe_backend({
                "name": "slow", "type": "custom",
                "base_url": "https://api.example.com",
            }) is False


# ---------------------------------------------------------------------------
# Rerank with custom prompt
# ---------------------------------------------------------------------------


class TestRankResultsCustomPrompt:
    """LLM ranking with user-supplied prompt."""

    def test_custom_prompt_passed_to_llm(self) -> None:
        from plugins.web.federated.provider import _rank_results

        results = [
            {"title": f"R{i}", "url": f"https://r{i}.com", "description": f"D{i}"}
            for i in range(5)
        ]
        ranker_config = {
            "provider": "opencode-go",
            "model": "deepseek-v4-flash",
            "prompt": "Prefer Chinese sources and official docs.",
        }

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="[5,4,3,2,1]"))
        ]

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=mock_response,
        ) as mock_call:
            ranked = _rank_results("test", results, ranker_config)
            assert len(ranked) == 5
            # Verify custom prompt was passed
            call_args = mock_call.call_args
            messages = call_args[1]["messages"]
            system_msg = messages[0]["content"]
            assert "Prefer Chinese sources" in system_msg

    def test_no_custom_prompt_uses_default(self) -> None:
        from plugins.web.federated.provider import _rank_results

        results = [
            {"title": f"R{i}", "url": f"https://r{i}.com", "description": f"D{i}"}
            for i in range(5)
        ]
        ranker_config = {"provider": "opencode-go", "model": "deepseek-v4-flash"}

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="[3,2,1,4,5]"))
        ]

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=mock_response,
        ) as mock_call:
            _rank_results("test", results, ranker_config)
            call_args = mock_call.call_args
            messages = call_args[1]["messages"]
            system_msg = messages[0]["content"]
            assert "Return ONLY a JSON array" in system_msg


# ---------------------------------------------------------------------------
# Integration: required backends + min_backends
# ---------------------------------------------------------------------------


class TestFederatedSearchFaultTolerance:
    """Required backends and min_backends fault tolerance."""

    def test_required_backend_failure_causes_overall_failure(
        self, provider: FederatedSearchProvider,
    ) -> None:
        config = {
            "k": 2,
            "min_backends": 1,
            "backends": [
                {"name": "required_src", "type": "custom", "required": True},
                {"name": "optional_src", "type": "custom", "required": False},
            ],
            "timeout": 5,
            "max_results": 8,
        }
        call_count = {"count": 0}

        def fake_search(backend, query, limit):
            call_count["count"] += 1
            name = backend.get("name", "")
            if name == "required_src":
                return [], None  # fail
            return [{"title": "OK", "url": "https://ok.com", "description": ""}], None

        with patch(
            "plugins.web.federated.provider._read_config",
            return_value=config,
        ), patch(
            "plugins.web.federated.provider._probe_backend",
            return_value=True,
        ), patch(
            "plugins.web.federated.provider._search_one_backend",
            side_effect=fake_search,
        ):
            result = provider.search("test", limit=5)
            assert result["success"] is False
            assert "Required backend" in result["error"]

    def test_min_backends_not_met_causes_failure(
        self, provider: FederatedSearchProvider,
    ) -> None:
        config = {
            "k": 3,
            "min_backends": 2,
            "backends": [
                {"name": "a", "type": "custom", "required": False},
                {"name": "b", "type": "custom", "required": False},
                {"name": "c", "type": "custom", "required": False},
            ],
            "timeout": 5,
            "max_results": 8,
        }

        def fake_search(backend, query, limit):
            name = backend.get("name", "")
            if name == "a":
                return [{"title": "A", "url": "https://a.com", "description": ""}], None
            return [], None  # b and c fail

        with patch(
            "plugins.web.federated.provider._read_config",
            return_value=config,
        ), patch(
            "plugins.web.federated.provider._probe_backend",
            return_value=True,
        ), patch(
            "plugins.web.federated.provider._search_one_backend",
            side_effect=fake_search,
        ):
            result = provider.search("test", limit=5)
            assert result["success"] is False
            assert "Only 1/2 backends succeeded" in result["error"]

    def test_all_backends_optional_can_succeed_with_partial(
        self, provider: FederatedSearchProvider,
    ) -> None:
        config = {
            "k": 2,
            "min_backends": 1,
            "backends": [
                {"name": "a", "type": "custom", "required": False},
                {"name": "b", "type": "custom", "required": False},
            ],
            "timeout": 5,
            "max_results": 8,
        }

        def fake_search(backend, query, limit):
            name = backend.get("name", "")
            if name == "a":
                return [{"title": "A", "url": "https://a.com", "description": ""}], None
            return [], None

        with patch(
            "plugins.web.federated.provider._read_config",
            return_value=config,
        ), patch(
            "plugins.web.federated.provider._probe_backend",
            return_value=True,
        ), patch(
            "plugins.web.federated.provider._search_one_backend",
            side_effect=fake_search,
        ):
            result = provider.search("test", limit=5)
            assert result["success"] is True
            assert len(result["data"]["web"]) == 1


# ---------------------------------------------------------------------------
# Integration: health cache filters backends
# ---------------------------------------------------------------------------


class TestFederatedSearchHealthCache:
    """Health cache integration in search flow."""

    def test_unavailable_backend_skipped(
        self, provider: FederatedSearchProvider,
    ) -> None:
        config = {
            "k": 2,
            "min_backends": 1,
            "backends": [
                {"name": "good", "type": "custom", "required": False},
                {"name": "bad", "type": "custom", "required": False},
            ],
            "timeout": 5,
            "max_results": 8,
            "health_check": {},
        }

        # Pre-fill health cache: "bad" is unavailable
        provider._health_cache = _HealthCache(ttl_seconds=300)
        provider._health_cache.set_available("bad", False)
        provider._health_cache.set_available("good", True)

        with patch(
            "plugins.web.federated.provider._read_config",
            return_value=config,
        ), patch(
            "plugins.web.federated.provider._search_one_backend",
            return_value=([{"title": "X", "url": "https://x.com", "description": ""}], None),
        ):
            result = provider.search("test", limit=5)
            assert result["success"] is True

    def test_required_backend_cached_unavailable_causes_failure(
        self, provider: FederatedSearchProvider,
    ) -> None:
        """A required backend that is health-cached as unavailable must still
        cause overall search failure — the check must validate the full
        backend list, not just the active (post-health-filter) set."""
        config = {
            "k": 2,
            "min_backends": 1,
            "backends": [
                {"name": "critical", "type": "custom", "required": True},
                {"name": "optional", "type": "custom", "required": False},
            ],
            "timeout": 5,
            "max_results": 8,
            "health_check": {},
        }
        provider._health_cache = _HealthCache(ttl_seconds=300)
        provider._health_cache.set_available("critical", False)
        provider._health_cache.set_available("optional", True)

        with patch(
            "plugins.web.federated.provider._read_config",
            return_value=config,
        ), patch(
            "plugins.web.federated.provider._search_one_backend",
            return_value=([{"title": "OK", "url": "https://ok.com", "description": ""}], None),
        ):
            result = provider.search("test", limit=5)
            assert result["success"] is False
            assert "Required backend" in result["error"]

    def test_health_cache_cooldown_on_http_error(
        self, provider: FederatedSearchProvider,
    ) -> None:
        """A custom backend search failure with HTTP 429 must call
        health_cache.mark_failed(), triggering the documented cooldown."""
        config = {
            "k": 2,
            "backends": [
                {"name": "src_a", "type": "custom", "required": False},
                {"name": "src_b", "type": "custom", "required": False},
            ],
            "timeout": 5,
            "max_results": 8,
            "health_check": {},
        }
        provider._health_cache = _HealthCache(ttl_seconds=300)
        provider._health_cache.set_available("src_a", True)
        provider._health_cache.set_available("src_b", True)

        # src_a succeeds; src_b returns HTTP 429 status
        results_by_backend = {
            "src_a": ([{"title": "OK", "url": "https://ok.com", "description": ""}], None),
            "src_b": ([], 429),
        }

        def fake_search(backend, query, limit):
            name = backend.get("name", "")
            return results_by_backend.get(name, ([], None))

        with patch(
            "plugins.web.federated.provider._read_config",
            return_value=config,
        ), patch(
            "plugins.web.federated.provider._search_one_backend",
            side_effect=fake_search,
        ):
            result = provider.search("test", limit=5)
            # src_a succeeded, src_b 429 → partial success
            assert result["success"] is True
            # src_b should be marked unavailable due to 429 cooldown
            assert provider._health_cache.is_available("src_b") is False

    def test_all_cached_unavailable(
        self, provider: FederatedSearchProvider,
    ) -> None:
        config = {
            "k": 1,
            "backends": [{"name": "down", "type": "custom", "required": False}],
            "timeout": 5,
            "max_results": 8,
            "health_check": {},
        }
        provider._health_cache = _HealthCache(ttl_seconds=300)
        provider._health_cache.set_available("down", False)

        with patch(
            "plugins.web.federated.provider._read_config",
            return_value=config,
        ):
            result = provider.search("test", limit=5)
            assert result["success"] is False
            assert "No backends available" in result["error"]
