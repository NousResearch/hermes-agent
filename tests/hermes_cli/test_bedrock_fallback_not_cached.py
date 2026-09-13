"""Bedrock's static fallback model list must never be persisted to
``provider_models_cache.json`` with the live TTL (#74151).

When live discovery fails (expired AWS SSO, throttle, no creds yet),
``provider_model_ids("bedrock")`` returns the curated emergency stub. The old
cache layer could not tell that stub from a live result, so it wrote the stub
to disk with the full 1h TTL — and because Bedrock declares no API-key env vars
(``api_key_env_vars=()`` in ``PROVIDER_REGISTRY``) and AWS SSO tokens live
outside ``$HERMES_HOME``, the cache fingerprint (built from a provider's API-key
env vars, base URL, and the mtimes of ``$HERMES_HOME`` auth files) is unchanged
by an SSO refresh, so re-auth did not bust the persisted stub. Result:
``/model`` showed only the ~10 offline entries for an hour, with
current-generation models absent.

The fix threads a ``_provenance`` flag from the Bedrock branch so the cache
layer serves the fallback in-memory only, letting the next picker open retry.
"""

import json

import hermes_cli.models as models


def _cache_file(tmp_path):
    return tmp_path / "provider_models_cache.json"


def _patch_home(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "_provider_models_cache_path", lambda: _cache_file(tmp_path))


class TestProvenanceFlag:
    def test_fallback_sets_provenance(self, monkeypatch):
        monkeypatch.setattr(
            "agent.bedrock_adapter.bedrock_model_ids_or_none", lambda: None
        )
        prov: dict = {}
        ids = models.provider_model_ids("bedrock", _provenance=prov)
        # The curated stub is returned...
        assert ids == list(models._PROVIDER_MODELS.get("bedrock", []))
        assert ids  # non-empty curated list exists
        # ...and flagged as a fallback.
        assert prov.get("fallback") is True

    def test_live_result_leaves_provenance_clean(self, monkeypatch):
        live = ["us.anthropic.claude-sonnet-5", "eu.anthropic.claude-opus-5"]
        monkeypatch.setattr(
            "agent.bedrock_adapter.bedrock_model_ids_or_none", lambda: live
        )
        prov: dict = {}
        ids = models.provider_model_ids("bedrock", _provenance=prov)
        assert ids == live
        assert prov.get("fallback") is not True

    def test_provenance_is_optional(self, monkeypatch):
        # Callers that don't pass _provenance must keep working unchanged.
        monkeypatch.setattr(
            "agent.bedrock_adapter.bedrock_model_ids_or_none", lambda: None
        )
        assert models.provider_model_ids("bedrock") == list(
            models._PROVIDER_MODELS.get("bedrock", [])
        )


class TestCacheDoesNotPersistFallback:
    def test_failed_discovery_is_not_written_to_disk(self, monkeypatch, tmp_path):
        _patch_home(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "agent.bedrock_adapter.bedrock_model_ids_or_none", lambda: None
        )
        result = models.cached_provider_model_ids("bedrock")
        # Right result returned in-memory...
        assert result == list(models._PROVIDER_MODELS.get("bedrock", []))
        # ...but nothing pinned to disk.
        assert not _cache_file(tmp_path).exists()

    def test_live_discovery_is_cached(self, monkeypatch, tmp_path):
        _patch_home(monkeypatch, tmp_path)
        live = ["us.anthropic.claude-sonnet-5", "eu.anthropic.claude-opus-5"]
        monkeypatch.setattr(
            "agent.bedrock_adapter.bedrock_model_ids_or_none", lambda: live
        )
        result = models.cached_provider_model_ids("bedrock")
        assert result == live
        cache = json.loads(_cache_file(tmp_path).read_text(encoding="utf-8"))
        assert cache["bedrock"]["models"] == live

    def test_discovery_retries_after_a_transient_failure(self, monkeypatch, tmp_path):
        """A failure must not cap the catalog: once creds recover, the live
        list wins on the very next call (nothing stale was frozen)."""
        _patch_home(monkeypatch, tmp_path)
        live = ["us.anthropic.claude-sonnet-5", "eu.anthropic.claude-opus-5"]

        monkeypatch.setattr(
            "agent.bedrock_adapter.bedrock_model_ids_or_none", lambda: None
        )
        first = models.cached_provider_model_ids("bedrock")
        assert first == list(models._PROVIDER_MODELS.get("bedrock", []))

        monkeypatch.setattr(
            "agent.bedrock_adapter.bedrock_model_ids_or_none", lambda: live
        )
        second = models.cached_provider_model_ids("bedrock")
        assert second == live, "a transient discovery failure hid the live catalog"
