"""OpenRouter account presets in the picker (``@preset/<slug>``).

Covers the account-scoped source: slugging and the system-prompt exclusion, the two cache layers
with their credential-fingerprint guard, the retry suppression and stale fallback that keep a failing
account cheap, and the deliberate separation between the catalog list (detection, silent default) and
the picker row presets are injected into.
"""

import time
from unittest.mock import patch

import hermes_cli.models as _models_mod
from hermes_cli.models import (
    OPENROUTER_MODELS,
    _openrouter_preset_fingerprint,
    _read_openrouter_presets_disk,
    _write_openrouter_presets_disk,
    fetch_openrouter_models,
    fetch_openrouter_presets,
)

_PRESET_PAYLOAD = {
    "data": [
        {
            "slug": "cheap-glm",
            "name": "cheap-glm",
            "designated_version": {
                "system_prompt": None,
                "config": {"model": "z-ai/glm-5.3-flash", "provider": {"order": ["deepinfra"]}},
            },
        },
        {
            "slug": "with-prompt",
            "name": "with-prompt",
            "designated_version": {"system_prompt": "You are a terse reviewer."},
        },
        {"slug": "cheap-glm", "name": "cheap-glm v2", "designated_version": {"system_prompt": None}},
        {"slug": "", "name": "no slug", "designated_version": {}},
        "not-a-dict",
    ]
}


class TestOpenRouterPresetListing:
    def test_slugs_are_prefixed_and_system_prompt_presets_are_skipped(self):
        """A preset carrying a system prompt is not selectable: the wire id is expanded server-side."""
        with patch.object(_models_mod, "_get_json", return_value=_PRESET_PAYLOAD):
            entries = _models_mod._fetch_openrouter_presets_remote("key", "", 8.0)

        assert entries == [("@preset/cheap-glm", "cheap-glm")]

    def test_unusable_account_response_means_no_presets(self):
        with patch.object(_models_mod, "_get_json", return_value={"nope": True}):
            assert _models_mod._fetch_openrouter_presets_remote("key", "", 8.0) is None

    def test_picker_row_leads_with_presets(self):
        """The row leads with them: every surface caps or collapses the catalog's tail."""
        from hermes_cli.model_switch_providers import _finalize_picker_rows

        rows = [
            {"slug": "openrouter", "is_current": False, "models": ["anthropic/claude-opus-5"], "total_models": 1},
            {"slug": "deepseek", "is_current": True, "models": ["deepseek-flash"], "total_models": 1},
        ]
        with patch.object(_models_mod, "fetch_openrouter_presets", return_value=[("@preset/cheap-glm", "cheap-glm")]):
            out = _finalize_picker_rows(rows, {}, "")

        row = next(r for r in out if r["slug"] == "openrouter")
        assert row["models"][0] == "@preset/cheap-glm"
        assert row["total_models"] == 2


class TestOpenRouterPresetsStayOutOfTheCatalogList:
    def test_catalog_list_never_carries_presets(self):
        """The catalog list feeds model-name detection and the silent default, so account presets
        must not appear in it and cannot shadow a catalog model."""
        with patch.object(_models_mod, "fetch_openrouter_presets", return_value=[("@preset/cheap-glm", "cheap-glm")]), \
             patch("hermes_cli.model_catalog.get_curated_openrouter_models", return_value=None), \
             patch("hermes_cli.models._urlopen_model_catalog_request", side_effect=OSError("boom")):
            models = fetch_openrouter_models(force_refresh=True)

        assert models == OPENROUTER_MODELS
        assert not any(mid.startswith("@preset/") for mid, _ in models)


class TestOpenRouterPresetsInThePickerRowVariant:
    """``list_picker_providers`` rebuilds the OpenRouter row from the catalog; presets must survive."""

    def test_presets_are_kept_when_the_row_is_rebuilt(self):
        from hermes_cli.model_switch_providers import list_picker_providers

        rows = [{"slug": "openrouter", "is_current": False, "is_user_defined": False,
                 "models": ["@preset/cheap-glm", "anthropic/claude-opus-5"], "total_models": 2}]
        with patch("hermes_cli.model_switch.list_authenticated_providers", return_value=rows), \
             patch.object(_models_mod, "fetch_openrouter_models",
                          return_value=[("anthropic/claude-opus-5", "")]):
            out = list_picker_providers()

        row = next(r for r in out if r["slug"] == "openrouter")
        assert row["models"] == ["@preset/cheap-glm", "anthropic/claude-opus-5"]

    def test_presets_are_kept_when_the_catalog_fetch_fails(self):
        from hermes_cli.model_switch_providers import list_picker_providers

        rows = [{"slug": "openrouter", "is_current": False, "is_user_defined": False,
                 "models": ["@preset/cheap-glm"], "total_models": 1}]
        with patch("hermes_cli.model_switch.list_authenticated_providers", return_value=rows), \
             patch.object(_models_mod, "fetch_openrouter_models", side_effect=RuntimeError("boom")):
            out = list_picker_providers()

        assert next(r for r in out if r["slug"] == "openrouter")["models"] == ["@preset/cheap-glm"]


class TestOpenRouterPresetFailSoft:
    def test_account_endpoint_failure_yields_no_presets(self, monkeypatch):
        monkeypatch.setattr(_models_mod, "_openrouter_presets_failed_at", None)
        with patch.object(_models_mod, "_openrouter_preset_credential", return_value=("key", "")), \
             patch.object(_models_mod, "_fetch_openrouter_presets_remote", side_effect=RuntimeError("401")):
            assert fetch_openrouter_presets(force_refresh=True) == []

    def test_missing_credential_never_calls_the_account_endpoint(self):
        with patch.object(_models_mod, "_openrouter_preset_credential", return_value=("", "")), \
             patch.object(_models_mod, "_fetch_openrouter_presets_remote") as remote:
            assert fetch_openrouter_presets(force_refresh=True) == []
        remote.assert_not_called()

    def test_failed_refresh_is_suppressed_instead_of_retried(self, monkeypatch):
        """A 401/offline account must not pay the fetch timeout on every picker open."""
        monkeypatch.setattr(_models_mod, "_openrouter_presets_cache", None)
        monkeypatch.setattr(_models_mod, "_openrouter_presets_failed_at", None)
        with patch.object(_models_mod, "_openrouter_preset_credential", return_value=("key", "")), \
             patch.object(_models_mod, "_fetch_openrouter_presets_remote", side_effect=RuntimeError("offline")) as remote:
            assert fetch_openrouter_presets() == []
            assert fetch_openrouter_presets() == []

        assert remote.call_count == 1

    def test_suppression_is_per_credential(self, monkeypatch):
        """Another account's failure must not silence this key's first fetch."""
        monkeypatch.setattr(_models_mod, "_openrouter_presets_cache", None)
        monkeypatch.setattr(_models_mod, "_openrouter_presets_failed_at",
                            (_openrouter_preset_fingerprint("other-key"), time.monotonic()))
        with patch.object(_models_mod, "_openrouter_preset_credential", return_value=("key", "")), \
             patch.object(_models_mod, "_fetch_openrouter_presets_remote",
                          return_value=[("@preset/fresh", "fresh")]) as remote:
            assert fetch_openrouter_presets() == [("@preset/fresh", "fresh")]

        assert remote.call_count == 1

    def test_failed_refresh_serves_the_cached_presets(self, monkeypatch):
        cached = [("@preset/cheap-glm", "cheap-glm")]
        monkeypatch.setattr(_models_mod, "_openrouter_presets_cache",
                            (_openrouter_preset_fingerprint("key"), cached))
        monkeypatch.setattr(_models_mod, "_openrouter_presets_failed_at", None)
        with patch.object(_models_mod, "_openrouter_preset_credential", return_value=("key", "")), \
             patch.object(_models_mod, "_fetch_openrouter_presets_remote", side_effect=RuntimeError("boom")):
            assert fetch_openrouter_presets(force_refresh=True) == cached


class TestOpenRouterPresetCacheScope:
    def test_disk_cache_is_dropped_when_the_credential_changes(self):
        _write_openrouter_presets_disk("fingerprint-a", [("@preset/cheap-glm", "cheap-glm")])

        assert _read_openrouter_presets_disk("fingerprint-b") is None
        assert _read_openrouter_presets_disk("fingerprint-a") == [("@preset/cheap-glm", "cheap-glm")]

    def test_expired_disk_cache_is_a_miss(self):
        from hermes_cli.models import _write_json_cache

        _write_json_cache(
            _models_mod._openrouter_presets_disk_path(),
            {"fetched_at": time.time() - 10_000, "fingerprint": "fingerprint-a",
             "presets": [["@preset/cheap-glm", "cheap-glm"]]})

        assert _read_openrouter_presets_disk("fingerprint-a") is None

    def test_explicit_refresh_drops_the_preset_cache(self, monkeypatch):
        """`hermes model --refresh` / `/model --refresh` must not hide a freshly created preset."""
        _write_openrouter_presets_disk("fingerprint-a", [("@preset/cheap-glm", "cheap-glm")])
        monkeypatch.setattr(_models_mod, "_openrouter_presets_cache", ("fingerprint-a", [("@preset/old", "old")]))
        monkeypatch.setattr(_models_mod, "_openrouter_presets_failed_at", ("fingerprint-a", time.monotonic()))

        _models_mod.clear_provider_models_cache("openrouter")

        assert _models_mod._openrouter_presets_cache is None
        assert _models_mod._openrouter_presets_failed_at is None
        assert _read_openrouter_presets_disk("fingerprint-a") is None

    def test_process_slot_from_another_credential_is_ignored(self, monkeypatch):
        """A rotated key must not keep serving the previous account's preset names."""
        monkeypatch.setattr(_models_mod, "_openrouter_presets_cache",
                            (_openrouter_preset_fingerprint("old-key"), [("@preset/stale", "stale")]))
        monkeypatch.setattr(_models_mod, "_openrouter_presets_failed_at", None)
        with patch.object(_models_mod, "_openrouter_preset_credential", return_value=("new-key", "")), \
             patch.object(_models_mod, "_fetch_openrouter_presets_remote", return_value=[("@preset/fresh", "fresh")]):
            assert fetch_openrouter_presets() == [("@preset/fresh", "fresh")]
