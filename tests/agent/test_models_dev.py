"""Tests for agent.models_dev — models.dev registry integration."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock

import pytest

from agent.models_dev import (
    PROVIDER_TO_MODELS_DEV,
    _extract_context,
    _default_model_override,
    _explicit_model_override,
    _override_context_window,
    _override_for,
    _NotModified,
    _validate_registry,
    fetch_models_dev,
    get_model_capabilities,
    get_model_info,
    get_provider_info,
    lookup_models_dev_context,
)


SAMPLE_REGISTRY = {
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic",
        "models": {
            "claude-opus-4-6": {
                "id": "claude-opus-4-6",
                "limit": {"context": 1000000, "output": 128000},
            },
            "claude-sonnet-4-6": {
                "id": "claude-sonnet-4-6",
                "limit": {"context": 1000000, "output": 64000},
            },
            "claude-sonnet-4-0": {
                "id": "claude-sonnet-4-0",
                "limit": {"context": 200000, "output": 64000},
            },
        },
    },
    "github-copilot": {
        "id": "github-copilot",
        "name": "GitHub Copilot",
        "models": {
            "claude-opus-4.6": {
                "id": "claude-opus-4.6",
                "limit": {"context": 128000, "output": 32000},
            },
        },
    },
    "xai": {
        "id": "xai",
        "name": "xAI",
        "models": {
            "grok-build-0.1": {
                "id": "grok-build-0.1",
                "limit": {"context": 256000, "output": 64000},
            },
        },
    },
    "kilo": {
        "id": "kilo",
        "name": "Kilo Gateway",
        "models": {
            "anthropic/claude-sonnet-4.6": {
                "id": "anthropic/claude-sonnet-4.6",
                "limit": {"context": 1000000, "output": 128000},
            },
        },
    },
    "deepseek": {
        "id": "deepseek",
        "name": "DeepSeek",
        "models": {
            "deepseek-chat": {
                "id": "deepseek-chat",
                "limit": {"context": 128000, "output": 8192},
            },
        },
    },
    "audio-only": {
        "id": "audio-only",
        "models": {
            "tts-model": {
                "id": "tts-model",
                "limit": {"context": 0, "output": 0},
            },
        },
    },
}


class TestProviderMapping:
    def test_all_mapped_providers_are_strings(self):
        for hermes_id, mdev_id in PROVIDER_TO_MODELS_DEV.items():
            assert isinstance(hermes_id, str)
            assert isinstance(mdev_id, str)

    def test_known_providers_mapped(self):
        assert PROVIDER_TO_MODELS_DEV["anthropic"] == "anthropic"
        assert PROVIDER_TO_MODELS_DEV["copilot"] == "github-copilot"
        assert PROVIDER_TO_MODELS_DEV["stepfun"] == "stepfun"
        assert PROVIDER_TO_MODELS_DEV["kilocode"] == "kilo"
        assert PROVIDER_TO_MODELS_DEV["ai-gateway"] == "vercel"

    def test_xai_oauth_uses_xai_catalog(self):
        assert PROVIDER_TO_MODELS_DEV["xai"] == "xai"
        assert PROVIDER_TO_MODELS_DEV["xai-oauth"] == "xai"

    def test_unmapped_provider_not_in_dict(self):
        assert "nous" not in PROVIDER_TO_MODELS_DEV



class TestExtractContext:
    def test_valid_entry(self):
        assert _extract_context({"limit": {"context": 128000}}) == 128000




    def test_non_dict_returns_none(self):
        assert _extract_context("not a dict") is None



class TestLookupModelsDevContext:
    @patch("agent.models_dev.fetch_models_dev")
    def test_exact_match(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert lookup_models_dev_context("anthropic", "claude-opus-4-6") == 1000000






    @patch("agent.models_dev.fetch_models_dev")
    def test_zero_context_filtered(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        # audio-only is not a mapped provider, but test the filtering directly
        data = SAMPLE_REGISTRY["audio-only"]["models"]["tts-model"]
        assert _extract_context(data) is None



class TestFetchModelsDev:
    @pytest.fixture(autouse=True)
    def _reset_fetch_state(self):
        import agent.models_dev as md

        md._models_dev_cache = {}
        md._models_dev_cache_time = 0
        md._models_dev_retry_after = 0
        md._models_dev_refresh_in_flight = False
        yield
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0
        md._models_dev_retry_after = 0
        md._models_dev_refresh_in_flight = False

    def _mock_response(self, data, etag="", status_code=200):
        """Build a MagicMock response with optional ETag header."""
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = data
        resp.headers = {"ETag": etag} if etag else {}
        resp.raise_for_status = MagicMock()
        return resp






    @patch("agent.models_dev.requests.get")
    def test_stale_disk_cache_returns_without_foreground_network(self, mock_get):
        """#35838: stale disk cache should not wait on models.dev timeout."""
        import agent.models_dev as md
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0

        with patch.object(md, "_disk_cache_age_seconds",
                          return_value=md._MODELS_DEV_CACHE_TTL + 60), \
             patch.object(md, "_load_disk_cache", return_value=SAMPLE_REGISTRY), \
             patch.object(md, "_load_etag", return_value=""), \
             patch.object(md, "_start_background_refresh_models_dev") as mock_refresh:
            result = fetch_models_dev()

        mock_get.assert_not_called()
        mock_refresh.assert_called_once()
        assert "anthropic" in result



    @patch("agent.models_dev.requests.get")
    def test_stale_cache_failure_enters_backoff_and_suppresses_retry(self, mock_get):
        import agent.models_dev as md

        mock_get.side_effect = OSError("models.dev unreachable")
        md._models_dev_cache = SAMPLE_REGISTRY
        md._models_dev_cache_time = time.time() - md._MODELS_DEV_CACHE_TTL - 1

        with patch.object(
            md,
            "_disk_cache_age_seconds",
            return_value=md._MODELS_DEV_CACHE_TTL + 60,
        ), patch.object(md, "_load_disk_cache", return_value=SAMPLE_REGISTRY), \
           patch.object(md, "_load_etag", return_value=""):
            first = fetch_models_dev()
            # Join the background refresh worker so its failure backoff is
            # observable and requests.get stays patched for its lifetime.
            for worker in threading.enumerate():
                if worker.name == "models-dev-refresh":
                    worker.join(timeout=5)
                    assert not worker.is_alive()

        assert first == SAMPLE_REGISTRY
        assert not md._models_dev_refresh_in_flight
        assert md._models_dev_retry_after > time.time()
        mock_get.assert_called_once()

        # A subsequent stale-cache hit inside the backoff window must not
        # spawn another refresh worker (in_flight is set synchronously
        # before the worker thread starts, so False proves no spawn).
        md._models_dev_cache_time = time.time() - md._MODELS_DEV_CACHE_TTL - 1
        second = fetch_models_dev()
        assert second == SAMPLE_REGISTRY
        assert not md._models_dev_refresh_in_flight
        mock_get.assert_called_once()

    @patch("agent.models_dev.requests.get")
    def test_background_refresh_success_commits_registry(self, mock_get):
        """The bg worker must save disk + swap mem cache + clear backoff."""
        import agent.models_dev as md

        response = self._mock_response(SAMPLE_REGISTRY, etag='"abc123"')
        mock_get.return_value = response

        md._models_dev_cache = {"stale": {}}
        md._models_dev_cache_time = 0
        md._models_dev_retry_after = time.time() - 1

        with patch.object(md, "_save_disk_cache") as mock_save, \
             patch.object(md, "_load_etag", return_value=""), \
             patch.object(md, "_save_etag") as mock_save_etag:
            # Run the worker synchronously — deterministic, no thread.
            md._models_dev_refresh_in_flight = True
            md._background_refresh_models_dev()

        # ETag is committed together with the cache body so the sidecar
        # can never get ahead of the data it vouches for.
        mock_save.assert_called_once_with(SAMPLE_REGISTRY, '"abc123"')
        mock_save_etag.assert_not_called()
        assert md._models_dev_cache == SAMPLE_REGISTRY
        assert md._models_dev_cache_time > 0
        assert md._models_dev_retry_after == 0
        assert not md._models_dev_refresh_in_flight


    @patch("agent.models_dev.requests.get")
    def test_concurrent_refreshes_share_one_network_request(self, mock_get):
        import agent.models_dev as md

        request_started = threading.Event()
        release_request = threading.Event()
        response = self._mock_response(SAMPLE_REGISTRY)

        def blocking_get(*_args, **_kwargs):
            request_started.set()
            assert release_request.wait(timeout=5)
            return response

        mock_get.side_effect = blocking_get
        with patch.object(md, "_disk_cache_age_seconds", return_value=None), patch.object(
            md, "_save_disk_cache"
        ), patch.object(md, "_load_etag", return_value=""), \
             patch.object(md, "_save_etag"), \
             ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(fetch_models_dev) for _ in range(6)]
            assert request_started.wait(timeout=2)
            release_request.set()
            results = [future.result(timeout=5) for future in futures]

        assert results == [SAMPLE_REGISTRY] * 6
        mock_get.assert_called_once()

    @patch("agent.models_dev.requests.get")
    def test_force_refresh_bypasses_failure_backoff(self, mock_get):
        import agent.models_dev as md

        response = self._mock_response(SAMPLE_REGISTRY)
        mock_get.side_effect = [OSError("models.dev unreachable"), response]

        with patch.object(md, "_disk_cache_age_seconds", return_value=None), patch.object(
            md, "_load_disk_cache", return_value={}
        ), patch.object(md, "_save_disk_cache"), \
             patch.object(md, "_load_etag", return_value=""), \
             patch.object(md, "_save_etag"):
            assert fetch_models_dev() == {}
            assert fetch_models_dev(force_refresh=True) == SAMPLE_REGISTRY

        assert mock_get.call_count == 2
        assert md._models_dev_retry_after == 0

    @pytest.mark.parametrize(
        ("cache", "cache_time", "disk_data", "expected"),
        [
            (SAMPLE_REGISTRY, lambda md: time.time(), {}, SAMPLE_REGISTRY),
            (
                SAMPLE_REGISTRY,
                lambda md: time.time() - md._MODELS_DEV_CACHE_TTL - 1,
                {},
                SAMPLE_REGISTRY,
            ),
            ({}, lambda _md: 0, {}, {}),
        ],
        ids=["fresh-memory", "stale-memory", "missing"],
    )
    @patch("agent.models_dev.requests.get")
    def test_network_disabled_never_fetches(
        self, mock_get, cache, cache_time, disk_data, expected
    ):
        import agent.models_dev as md

        md._models_dev_cache = cache
        md._models_dev_cache_time = cache_time(md)
        with patch.object(md, "_load_disk_cache", return_value=disk_data):
            result = fetch_models_dev(allow_network=False)

        assert result == expected
        mock_get.assert_not_called()







# ---------------------------------------------------------------------------
# ETag conditional GET
# ---------------------------------------------------------------------------


class TestETagConditionalGet:
    """Tests for ETag-based conditional GET (If-None-Match / 304 handling)."""

    @pytest.fixture(autouse=True)
    def _reset_fetch_state(self):
        import agent.models_dev as md
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0
        md._models_dev_retry_after = 0
        md._models_dev_refresh_in_flight = False
        yield
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0
        md._models_dev_retry_after = 0
        md._models_dev_refresh_in_flight = False

    @patch("agent.models_dev.requests.get")
    def test_etag_sent_when_cached(self, mock_get):
        """If-None-Match header is sent when a cached ETag exists."""
        import agent.models_dev as md

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = SAMPLE_REGISTRY
        response.headers = {"ETag": '"v2"'}
        response.raise_for_status = MagicMock()
        mock_get.return_value = response

        # Conditional GET requires a servable in-memory registry — an
        # If-None-Match without one invites a 304 against nothing.
        md._models_dev_cache = SAMPLE_REGISTRY
        md._models_dev_cache_time = 0

        with patch.object(md, "_disk_cache_age_seconds", return_value=None), \
             patch.object(md, "_load_disk_cache", return_value={}), \
             patch.object(md, "_save_disk_cache"), \
             patch.object(md, "_load_etag", return_value='"v1"'), \
             patch.object(md, "_save_etag"):
            fetch_models_dev(force_refresh=True)

        call_kwargs = mock_get.call_args
        headers = call_kwargs.kwargs.get("headers", {})
        assert headers.get("If-None-Match") == '"v1"'

    @patch("agent.models_dev.requests.get")
    def test_304_reconfirms_cache_freshness(self, mock_get):
        """A 304 Not Modified re-confirms the existing cache without download."""
        import agent.models_dev as md

        response = MagicMock()
        response.status_code = 304
        mock_get.return_value = response

        md._models_dev_cache = SAMPLE_REGISTRY
        md._models_dev_cache_time = 0
        md._models_dev_retry_after = time.time() + 100  # backoff was armed

        with patch.object(md, "_load_etag", return_value='"v1"'), \
             patch.object(md, "_save_etag"):
            # Run the background worker synchronously
            md._models_dev_refresh_in_flight = True
            md._background_refresh_models_dev()

        # Cache content unchanged
        assert md._models_dev_cache == SAMPLE_REGISTRY
        # Freshness timestamp advanced
        assert md._models_dev_cache_time > 0
        # Backoff cleared
        assert md._models_dev_retry_after == 0
        assert not md._models_dev_refresh_in_flight
        # response.json() was never called — no body to parse
        response.json.assert_not_called()

    @patch("agent.models_dev.requests.get")
    def test_foreground_304_returns_existing_cache(self, mock_get):
        """Foreground fetch with 304 returns the existing cache."""
        import agent.models_dev as md

        response = MagicMock()
        response.status_code = 304
        mock_get.return_value = response

        md._models_dev_cache = SAMPLE_REGISTRY
        md._models_dev_cache_time = 0
        md._models_dev_retry_after = 0

        with patch.object(md, "_disk_cache_age_seconds", return_value=None), \
             patch.object(md, "_load_disk_cache", return_value={}), \
             patch.object(md, "_load_etag", return_value='"v1"'), \
             patch.object(md, "_save_etag"):
            result = fetch_models_dev(force_refresh=True)

        assert result == SAMPLE_REGISTRY
        assert md._models_dev_cache_time > 0

    @patch("agent.models_dev.requests.get")
    def test_new_etag_persisted_after_successful_fetch(self, mock_get):
        """A successful fetch with an ETag in the response persists it."""
        import agent.models_dev as md

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = SAMPLE_REGISTRY
        response.headers = {"ETag": '"new-etag"'}
        response.raise_for_status = MagicMock()
        mock_get.return_value = response

        with patch.object(md, "_disk_cache_age_seconds", return_value=None), \
             patch.object(md, "_load_disk_cache", return_value={}), \
             patch.object(md, "_save_disk_cache") as mock_save, \
             patch.object(md, "_load_etag", return_value=""), \
             patch.object(md, "_save_etag") as mock_save_etag:
            fetch_models_dev()

        # ETag rides along with the cache body into _save_disk_cache.
        mock_save.assert_called_once_with(SAMPLE_REGISTRY, '"new-etag"')
        mock_save_etag.assert_not_called()

    @patch("agent.models_dev.requests.get")
    def test_no_etag_header_sent_without_cached_etag(self, mock_get):
        """No If-None-Match header when no cached ETag exists."""
        import agent.models_dev as md

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = SAMPLE_REGISTRY
        response.headers = {}
        response.raise_for_status = MagicMock()
        mock_get.return_value = response

        with patch.object(md, "_disk_cache_age_seconds", return_value=None), \
             patch.object(md, "_load_disk_cache", return_value={}), \
             patch.object(md, "_save_disk_cache"), \
             patch.object(md, "_load_etag", return_value=""), \
             patch.object(md, "_save_etag"):
            fetch_models_dev()

        call_kwargs = mock_get.call_args
        headers = call_kwargs.kwargs.get("headers", {})
        assert "If-None-Match" not in headers


# ---------------------------------------------------------------------------
# Corrupt / invalid cache rejection
# ---------------------------------------------------------------------------


class TestCorruptCacheRejection:
    """A corrupt or empty disk cache must be rejected, not served as {}."""

    def test_validate_registry_rejects_empty_dict(self):
        assert not _validate_registry({})

    def test_validate_registry_rejects_non_dict(self):
        assert not _validate_registry("not a dict")
        assert not _validate_registry(None)
        assert not _validate_registry([])

    def test_validate_registry_accepts_populated_dict(self):
        assert _validate_registry({"anthropic": {}})

    def test_corrupt_json_on_disk_rejected_with_warning(self, tmp_path, caplog):
        """Invalid JSON in a REAL cache file is rejected with a warning."""
        import logging

        import agent.models_dev as md

        cache = tmp_path / "models_dev_cache.json"
        cache.write_text("not json{{{", encoding="utf-8")
        with patch.object(md, "_get_cache_path", return_value=cache), \
             patch.object(md, "_get_etag_path", return_value=tmp_path / "models_dev_cache.etag"):
            with caplog.at_level(logging.WARNING):
                result = md._load_disk_cache()

        assert result == {}
        assert any("disk cache" in r.message for r in caplog.records)

    def test_empty_dict_on_disk_rejected_with_warning(self, tmp_path, caplog):
        """A REAL cache file containing {} is rejected with a warning."""
        import logging

        import agent.models_dev as md

        cache = tmp_path / "models_dev_cache.json"
        cache.write_text("{}", encoding="utf-8")
        with patch.object(md, "_get_cache_path", return_value=cache), \
             patch.object(md, "_get_etag_path", return_value=tmp_path / "models_dev_cache.etag"):
            with caplog.at_level(logging.WARNING):
                result = md._load_disk_cache()

        assert result == {}
        assert any("corrupt or empty" in r.message for r in caplog.records)

    def test_corrupt_cache_clears_etag_sidecar(self, tmp_path):
        """Rejecting a corrupt cache must drop the ETag sidecar (#35838 loop).

        If the sidecar outlives the registry it vouches for, the next
        conditional GET draws a 304 against nothing and the process serves
        {} forever. Clearing the sidecar forces an unconditional refetch.
        """
        import agent.models_dev as md

        cache = tmp_path / "models_dev_cache.json"
        etag = tmp_path / "models_dev_cache.etag"
        cache.write_text("corrupt!!", encoding="utf-8")
        etag.write_text("stale-etag", encoding="utf-8")

        with patch.object(md, "_get_cache_path", return_value=cache), \
             patch.object(md, "_get_etag_path", return_value=etag):
            result = md._load_disk_cache()

        assert result == {}
        assert not etag.exists()
        # The corrupt file is quarantined (renamed), so the rejection is
        # a one-time event instead of a re-parse + warning per call.
        assert not cache.exists()
        assert cache.with_suffix(".json.corrupt").exists()

    def test_conditional_get_skipped_without_servable_cache(self):
        """No If-None-Match header when the process holds no registry.

        A conditional GET without a servable cache invites a 304 that
        leaves the process with no data at all — the permanent
        empty-registry loop. The header is only sent when _models_dev_cache
        is populated.
        """
        import agent.models_dev as md

        captured: dict = {}

        def fake_get(url, headers=None, timeout=None):
            captured["headers"] = dict(headers or {})
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"anthropic": {"models": {}}}
            resp.headers = {"ETag": "fresh"}
            return resp

        with patch.object(md.requests, "get", side_effect=fake_get), \
             patch.object(md, "_load_etag", return_value="stale-etag"), \
             patch.object(md, "_models_dev_cache", {}):
            data, etag = md._fetch_models_dev_from_network()

        assert "If-None-Match" not in captured["headers"]
        assert data == {"anthropic": {"models": {}}}
        assert etag == "fresh"

    def test_304_with_empty_cache_arms_backoff_and_clears_etag(self, tmp_path):
        """Defense in depth: a 304 landing on an empty registry must not
        mark {} as fresh — it clears the sidecar and arms the backoff."""
        import agent.models_dev as md

        etag = tmp_path / "models_dev_cache.etag"
        etag.write_text("stale", encoding="utf-8")

        with patch.object(md, "_get_etag_path", return_value=etag), \
             patch.object(md, "_models_dev_cache", {}):
            before = md._models_dev_retry_after
            try:
                md._confirm_cache_not_modified(where="test")
                assert not etag.exists()
                assert md._models_dev_retry_after > time.time() - 1
            finally:
                md._models_dev_retry_after = before


# ---------------------------------------------------------------------------
# Mirror URL override via config
# ---------------------------------------------------------------------------


class TestMirrorUrlOverride:
    """models_dev.url config key overrides the API endpoint."""

    @pytest.fixture(autouse=True)
    def _reset_fetch_state(self):
        import agent.models_dev as md
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0
        md._models_dev_retry_after = 0
        md._models_dev_refresh_in_flight = False
        yield
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0
        md._models_dev_retry_after = 0
        md._models_dev_refresh_in_flight = False

    @patch("agent.models_dev.requests.get")
    def test_mirror_url_used_when_configured(self, mock_get):
        """When config has models_dev.url, requests.get hits that URL."""
        import agent.models_dev as md

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = SAMPLE_REGISTRY
        response.headers = {}
        response.raise_for_status = MagicMock()
        mock_get.return_value = response

        fake_config = {"models_dev": {"url": "https://mirror.example.com/api.json"}}

        with patch.object(md, "_disk_cache_age_seconds", return_value=None), \
             patch.object(md, "_load_disk_cache", return_value={}), \
             patch.object(md, "_save_disk_cache"), \
             patch.object(md, "_load_etag", return_value=""), \
             patch.object(md, "_save_etag"), \
             patch("hermes_cli.config.load_config_readonly", return_value=fake_config):
            fetch_models_dev()

        call_args = mock_get.call_args
        assert "mirror.example.com" in call_args.args[0]

    @patch("agent.models_dev.requests.get")
    def test_default_url_used_when_not_configured(self, mock_get):
        """Without config override, the default models.dev URL is used."""
        import agent.models_dev as md

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = SAMPLE_REGISTRY
        response.headers = {}
        response.raise_for_status = MagicMock()
        mock_get.return_value = response

        with patch.object(md, "_disk_cache_age_seconds", return_value=None), \
             patch.object(md, "_load_disk_cache", return_value={}), \
             patch.object(md, "_save_disk_cache"), \
             patch.object(md, "_load_etag", return_value=""), \
             patch.object(md, "_save_etag"), \
             patch("hermes_cli.config.load_config_readonly", return_value={}):
            fetch_models_dev()

        call_args = mock_get.call_args
        assert "models.dev" in call_args.args[0]

    @patch("agent.models_dev.requests.get")
    def test_empty_url_falls_back_to_default(self, mock_get):
        """An empty string URL in config falls back to the default."""
        import agent.models_dev as md

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = SAMPLE_REGISTRY
        response.headers = {}
        response.raise_for_status = MagicMock()
        mock_get.return_value = response

        fake_config = {"models_dev": {"url": ""}}

        with patch.object(md, "_disk_cache_age_seconds", return_value=None), \
             patch.object(md, "_load_disk_cache", return_value={}), \
             patch.object(md, "_save_disk_cache"), \
             patch.object(md, "_load_etag", return_value=""), \
             patch.object(md, "_save_etag"), \
             patch("hermes_cli.config.load_config_readonly", return_value=fake_config):
            fetch_models_dev()

        call_args = mock_get.call_args
        assert "models.dev" in call_args.args[0]


# ---------------------------------------------------------------------------
# No-network-on-hot-paths invariant
# ---------------------------------------------------------------------------


class TestNoNetworkOnHotPaths:
    """Query functions must default to allow_network=False on hot paths."""

    @patch("agent.models_dev.requests.get")
    def test_get_model_capabilities_default_no_network(self, mock_get):
        """get_model_capabilities defaults to allow_network=False."""
        with patch("agent.models_dev.fetch_models_dev") as mock_fetch:
            mock_fetch.return_value = CAPS_REGISTRY
            get_model_capabilities("anthropic", "claude-sonnet-4")
        # fetch_models_dev was called with allow_network=False
        mock_fetch.assert_called_once_with(allow_network=False)

    @patch("agent.models_dev.requests.get")
    def test_get_model_info_default_no_network(self, mock_get):
        """get_model_info defaults to allow_network=False."""
        with patch("agent.models_dev.fetch_models_dev") as mock_fetch:
            mock_fetch.return_value = SAMPLE_REGISTRY
            get_model_info("anthropic", "claude-opus-4-6")
        mock_fetch.assert_called_once_with(allow_network=False)

    @patch("agent.models_dev.requests.get")
    def test_lookup_models_dev_context_default_no_network(self, mock_get):
        """lookup_models_dev_context defaults to allow_network=False."""
        with patch("agent.models_dev.fetch_models_dev") as mock_fetch:
            mock_fetch.return_value = SAMPLE_REGISTRY
            lookup_models_dev_context("anthropic", "claude-opus-4-6")
        mock_fetch.assert_called_once_with(allow_network=False)

    @patch("agent.models_dev.requests.get")
    def test_get_model_capabilities_explicit_network(self, mock_get):
        """get_model_capabilities can opt into network."""
        with patch("agent.models_dev.fetch_models_dev") as mock_fetch:
            mock_fetch.return_value = CAPS_REGISTRY
            get_model_capabilities("anthropic", "claude-sonnet-4", allow_network=True)
        # allow_network=True uses the zero-arg call shape so the dozens of
        # test sites that monkeypatch fetch_models_dev with zero-arg
        # lambdas keep working.
        mock_fetch.assert_called_once_with()


# ---------------------------------------------------------------------------
# get_model_capabilities — vision via modalities.input
# ---------------------------------------------------------------------------


CAPS_REGISTRY = {
    "google": {
        "id": "google",
        "models": {
            "gemma-4-31b-it": {
                "id": "gemma-4-31b-it",
                "attachment": False,
                "tool_call": True,
                "modalities": {"input": ["text", "image"]},
                "limit": {"context": 128000, "output": 8192},
            },
            "gemma-3-1b": {
                "id": "gemma-3-1b",
                "tool_call": True,
                "limit": {"context": 32000, "output": 8192},
            },
            "text-only-with-stale-attachment": {
                "id": "text-only-with-stale-attachment",
                "attachment": True,
                "tool_call": True,
                "modalities": {"input": ["text"]},
                "limit": {"context": 128000, "output": 8192},
            },
        },
    },
    "anthropic": {
        "id": "anthropic",
        "models": {
            "claude-sonnet-4": {
                "id": "claude-sonnet-4",
                "attachment": True,
                "tool_call": True,
                "limit": {"context": 200000, "output": 64000},
            },
        },
    },
}


class TestGetModelCapabilities:
    """Tests for get_model_capabilities vision detection."""

    def test_vision_from_attachment_flag(self):
        """Models with attachment=True and no modalities should report supports_vision=True."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("anthropic", "claude-sonnet-4")
        assert caps is not None
        assert caps.supports_vision is True




    def test_modalities_non_dict_handled(self):
        """Non-dict modalities field should not crash."""
        registry = {
            "google": {"id": "google", "models": {
                "weird-model": {
                    "id": "weird-model",
                    "modalities": "text",  # not a dict
                    "limit": {"context": 200000, "output": 8192},
                },
            }},
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=registry):
            caps = get_model_capabilities("gemini", "weird-model")
        assert caps is not None
        assert caps.supports_vision is False


# ---------------------------------------------------------------------------
# Per-model metadata overrides (model_overrides config)
# ---------------------------------------------------------------------------


class TestModelOverrides:
    """Tests for the model_overrides config system."""

    def _setup_overrides(self, overrides_dict):
        """Patch _load_model_overrides to return the given dict."""
        import agent.models_dev as md
        return patch.object(md, "_load_model_overrides", return_value=overrides_dict)

    # --- override resolution ---

    def test_per_provider_model_override(self):
        """Per-provider+model override is found first."""
        overrides = {
            "upstage": {
                "solar-pro4": {"context_window": 524288},
            },
        }
        with self._setup_overrides(overrides):
            result = _explicit_model_override("upstage", "solar-pro4")
        assert result is not None
        assert result["context_window"] == 524288

    def test_explicit_override_case_insensitive_model(self):
        """Model ids match case-insensitively, mirroring catalog lookup."""
        overrides = {
            "upstage": {
                "Solar-Pro4": {"context_window": 524288},
            },
        }
        with self._setup_overrides(overrides):
            result = _explicit_model_override("upstage", "solar-pro4")
        assert result is not None
        assert result["context_window"] == 524288

    def test_provider_key_accepts_either_id_space(self):
        """Override keyed by Hermes id resolves for models.dev id and back."""
        overrides = {
            "copilot": {
                "my-model": {"context_window": 111111},
            },
        }
        with self._setup_overrides(overrides):
            # Caller passes the models.dev id; config keyed by Hermes id.
            result = _explicit_model_override("github-copilot", "my-model")
        assert result is not None
        assert result["context_window"] == 111111

        overrides = {
            "github-copilot": {
                "my-model": {"context_window": 222222},
            },
        }
        with self._setup_overrides(overrides):
            # Caller passes the Hermes id; config keyed by models.dev id.
            result = _explicit_model_override("copilot", "my-model")
        assert result is not None
        assert result["context_window"] == 222222

    def test_default_fills_gap_for_unknown_model(self):
        """_default applies to models the catalog does not know."""
        overrides = {
            "upstage": {
                "_default": {"context_window": 128000},
            },
        }
        with self._setup_overrides(overrides):
            result = _override_for("upstage", "unknown-model", catalog_hit=False)
        assert result is not None
        assert result["context_window"] == 128000

    def test_default_does_not_clamp_catalog_known_model(self):
        """FILL-GAP semantics: _default never displaces catalog data.

        A `_default: {context_window: 128000}` must not clamp every
        catalog-known model of the provider — it only fills catalog misses.
        """
        overrides = {
            "upstage": {
                "_default": {"context_window": 128000},
            },
            "_default": {"context_window": 65536},
        }
        with self._setup_overrides(overrides):
            result = _override_for("upstage", "known-model", catalog_hit=True)
        assert result is None

    def test_global_default_fallback(self):
        """Global _default is used when provider has no section."""
        overrides = {
            "_default": {"context_window": 65536},
        }
        with self._setup_overrides(overrides):
            result = _default_model_override("unknown-provider")
        assert result is not None
        assert result["context_window"] == 65536

    def test_no_override_returns_none(self):
        with self._setup_overrides({}):
            assert _explicit_model_override("anthropic", "claude-sonnet-4") is None
            assert _default_model_override("anthropic") is None

    def test_explicit_beats_default(self):
        """Per-provider+model wins over per-provider _default."""
        overrides = {
            "upstage": {
                "solar-pro4": {"context_window": 524288},
                "_default": {"context_window": 128000},
            },
        }
        with self._setup_overrides(overrides):
            result = _override_for("upstage", "solar-pro4", catalog_hit=False)
        assert result is not None
        assert result["context_window"] == 524288

    def test_per_provider_default_beats_global(self):
        overrides = {
            "upstage": {
                "_default": {"context_window": 128000},
            },
            "_default": {"context_window": 65536},
        }
        with self._setup_overrides(overrides):
            result = _default_model_override("upstage")
        assert result is not None
        assert result["context_window"] == 128000

    # --- _override_context_window (explicit-only, early-chain) ---

    def test_override_context_window_returns_value(self):
        overrides = {
            "upstage": {
                "syn-pro": {"context_window": 65536},
            },
        }
        with self._setup_overrides(overrides):
            ctx = _override_context_window("upstage", "syn-pro")
        assert ctx == 65536

    def test_override_context_window_returns_none_when_missing(self):
        with self._setup_overrides({}):
            ctx = _override_context_window("upstage", "syn-pro")
        assert ctx is None

    def test_override_context_window_rejects_zero(self):
        overrides = {
            "upstage": {
                "bad-model": {"context_window": 0},
            },
        }
        with self._setup_overrides(overrides):
            ctx = _override_context_window("upstage", "bad-model")
        assert ctx is None

    def test_override_context_window_ignores_default(self):
        """Early-chain lookup is explicit-only: a _default must not preempt
        more specific sources (custom_providers, live probes)."""
        overrides = {
            "upstage": {
                "_default": {"context_window": 128000},
            },
        }
        with self._setup_overrides(overrides):
            ctx = _override_context_window("upstage", "syn-pro")
        assert ctx is None

    def test_malformed_context_window_warns_once(self, caplog):
        """Garbage values are rejected with a one-shot warning, not silence."""
        import logging

        import agent.models_dev as md
        md._OVERRIDE_WARNED_KEYS.clear()
        overrides = {
            "upstage": {
                "bad-model": {"context_window": "512k"},
            },
        }
        with self._setup_overrides(overrides), caplog.at_level(logging.WARNING):
            assert _override_context_window("upstage", "bad-model") is None
            assert _override_context_window("upstage", "bad-model") is None
        warnings = [r for r in caplog.records if "model_overrides" in r.message]
        assert len(warnings) == 1

    # --- get_model_capabilities with overrides ---

    def test_caps_override_unknown_model(self):
        """Override provides capabilities for a model NOT in the catalog (#8731)."""
        overrides = {
            "custom:my-vllm": {
                "my-llava-model": {
                    "context_window": 8192,
                    "supports_vision": True,
                    "supports_reasoning": False,
                    "supports_tools": True,
                },
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value={}):
            caps = get_model_capabilities("custom:my-vllm", "my-llava-model")
        assert caps is not None
        assert caps.context_window == 8192
        assert caps.supports_vision is True
        assert caps.supports_reasoning is False
        assert caps.supports_tools is True

    def test_caps_override_patches_existing_catalog_entry(self):
        """Explicit override patches specific fields on a known entry (#84482)."""
        overrides = {
            "anthropic": {
                "claude-sonnet-4": {
                    "context_window": 500000,
                },
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("anthropic", "claude-sonnet-4")
        assert caps is not None
        # Override wins
        assert caps.context_window == 500000
        # Non-overridden fields preserved from catalog
        assert caps.supports_vision is True
        assert caps.supports_tools is True

    def test_caps_default_does_not_clamp_catalog_model(self):
        """A _default must not displace catalog data for known models."""
        overrides = {
            "anthropic": {
                "_default": {"context_window": 1000},
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = get_model_capabilities("anthropic", "claude-sonnet-4")
        assert caps is not None
        assert caps.context_window != 1000

    def test_caps_no_override_no_catalog_returns_none(self):
        with self._setup_overrides({}), \
             patch("agent.models_dev.fetch_models_dev", return_value={}):
            caps = get_model_capabilities("anthropic", "unknown-model")
        assert caps is None

    def test_caps_override_default_for_unknown_model(self):
        """Per-provider _default provides capabilities for unknown models."""
        overrides = {
            "custom:my-vllm": {
                "_default": {
                    "context_window": 32768,
                    "supports_tools": True,
                },
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value={}):
            caps = get_model_capabilities("custom:my-vllm", "some-new-model")
        assert caps is not None
        assert caps.context_window == 32768
        assert caps.supports_tools is True

    # --- lookup_models_dev_context with overrides ---

    def test_context_lookup_override_wins_over_catalog(self):
        overrides = {
            "anthropic": {
                "claude-opus-4-6": {"context_window": 500000},
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value=SAMPLE_REGISTRY):
            ctx = lookup_models_dev_context("anthropic", "claude-opus-4-6")
        assert ctx == 500000

    def test_context_lookup_override_for_unknown_provider(self):
        overrides = {
            "upstage": {
                "solar-pro4": {"context_window": 524288},
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value={}):
            ctx = lookup_models_dev_context("upstage", "solar-pro4")
        assert ctx == 524288

    def test_context_lookup_default_fills_catalog_miss(self):
        """_default supplies context for a model the catalog lacks."""
        overrides = {
            "anthropic": {
                "_default": {"context_window": 77777},
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value=SAMPLE_REGISTRY):
            ctx = lookup_models_dev_context("anthropic", "model-not-in-catalog")
        assert ctx == 77777

    def test_context_lookup_default_does_not_clamp_catalog(self):
        """_default must not beat a catalog-known model's real context."""
        overrides = {
            "anthropic": {
                "_default": {"context_window": 1000},
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value=SAMPLE_REGISTRY):
            ctx = lookup_models_dev_context("anthropic", "claude-opus-4-6")
        assert ctx == 1000000  # catalog value, not the _default

    # --- get_model_info with overrides (canonical schema) ---

    def test_model_info_override_for_unknown_model(self):
        """Canonical-schema override provides metadata for an unknown model.

        Same key space as every other consumer — context_window,
        max_output_tokens, supports_* — NOT the internal catalog shape.
        """
        overrides = {
            "custom:my-vllm": {
                "my-llava-model": {
                    "model_family": "llava",
                    "supports_reasoning": False,
                    "supports_tools": True,
                    "context_window": 8192,
                    "max_output_tokens": 4096,
                },
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value={}):
            info = get_model_info("custom:my-vllm", "my-llava-model")
        assert info is not None
        assert info.family == "llava"
        assert info.context_window == 8192
        assert info.max_output == 4096
        assert info.tool_call is True
        assert info.reasoning is False

    def test_model_info_override_merges_with_catalog(self):
        """Override patches context without clobbering the catalog's output.

        The limit sub-dict is MERGED: an override setting only
        context_window preserves the catalog's limit.output.
        """
        overrides = {
            "anthropic": {
                "claude-sonnet-4-6": {
                    "context_window": 500000,
                },
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value=SAMPLE_REGISTRY):
            info = get_model_info("anthropic", "claude-sonnet-4-6")
        assert info is not None
        # Override wins for the field it sets
        assert info.context_window == 500000
        # Sub-dict merge: catalog's limit.output survives
        assert info.max_output == 64000
        # Non-overridden fields preserved from catalog
        assert info.name == "claude-sonnet-4-6"

    def test_model_info_default_does_not_clamp_catalog(self):
        """_default fills gaps only — known models keep catalog metadata."""
        overrides = {
            "anthropic": {
                "_default": {"context_window": 1000},
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value=SAMPLE_REGISTRY):
            info = get_model_info("anthropic", "claude-sonnet-4-6")
        assert info is not None
        assert info.context_window == 1000000

    # --- e2e config plumbing (real config.yaml, no _load_model_overrides mock) ---

    def test_e2e_overrides_load_from_real_config_yaml(self, tmp_path, monkeypatch):
        """The real config path works end-to-end: config.yaml on disk ->
        load_config_readonly -> cfg_get -> override applied.

        Every other test mocks _load_model_overrides; this one exercises
        the actual wiring (key name, cfg accessor, cache invalidation).
        """
        import importlib

        import agent.models_dev as md
        import hermes_cli.config as hc

        home = tmp_path / "hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            "model_overrides:\n"
            "  upstage:\n"
            "    solar-pro4:\n"
            "      context_window: 524288\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(home))

        # Reset caches that memoize config paths (the override layer has
        # no local cache — it rides load_config_readonly's mtime cache).
        hc_cache = getattr(hc, "_LOAD_CONFIG_CACHE", None)
        if isinstance(hc_cache, dict):
            hc_cache.clear()
        raw_cache = getattr(hc, "_RAW_CONFIG_CACHE", None)
        if isinstance(raw_cache, dict):
            raw_cache.clear()
        importlib.reload  # no-op guard: modules stay loaded, caches cleared

        with patch("agent.models_dev.fetch_models_dev", return_value={}):
            ctx = lookup_models_dev_context("upstage", "solar-pro4")
        assert ctx == 524288

    def test_suffix_keyed_model_counts_as_catalog_hit(self):
        """A suffix-keyed catalog model (kimi-k2.6:cloud) is KNOWN: a
        _default must not displace its capabilities."""
        registry = {
            "ollama-cloud": {
                "id": "ollama-cloud",
                "models": {
                    "kimi-k2.6:cloud": {
                        "id": "kimi-k2.6:cloud",
                        "tool_call": True,
                        "limit": {"context": 262144, "output": 8192},
                    },
                },
            },
        }
        overrides = {
            "ollama-cloud": {
                "_default": {"context_window": 1000, "supports_tools": False},
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value=registry):
            caps = get_model_capabilities("ollama-cloud", "kimi-k2.6")
        assert caps is not None
        assert caps.context_window == 262144  # catalog, not the _default
        assert caps.supports_tools is True

    def test_model_info_unknown_model_gets_safe_defaults(self):
        """get_model_info's unknown-model path seeds the same safe
        defaults as get_model_capabilities (200K/tools-on), so a partial
        override doesn't yield ctx=0/tools-off."""
        overrides = {
            "custom:my-vllm": {
                "my-model": {"supports_reasoning": True},
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value={}):
            info = get_model_info("custom:my-vllm", "my-model")
        assert info is not None
        assert info.context_window == 200000
        assert info.max_output == 8192
        assert info.tool_call is True
        assert info.reasoning is True

    def test_model_info_vision_override_sets_input_modality(self):
        """supports_vision: true surfaces as an image input modality."""
        overrides = {
            "custom:my-vllm": {
                "my-model": {
                    "supports_vision": True,
                    "context_window": 8192,
                },
            },
        }
        with self._setup_overrides(overrides), \
             patch("agent.models_dev.fetch_models_dev", return_value={}):
            info = get_model_info("custom:my-vllm", "my-model")
        assert info is not None
        assert "image" in info.input_modalities
        assert info.attachment is True
