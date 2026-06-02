"""Tests for hermes_cli.model_catalog — remote manifest fetch + cache + fallback."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Isolate HERMES_HOME + reset any module-level catalog cache per test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Force a fresh catalog module state for each test.
    import importlib
    from hermes_cli import model_catalog
    importlib.reload(model_catalog)
    yield home
    model_catalog.reset_cache()


def _valid_manifest() -> dict:
    return {
        "version": 1,
        "updated_at": "2026-04-25T22:00:00Z",
        "metadata": {"source": "test"},
        "providers": {
            "openrouter": {
                "metadata": {"display_name": "OpenRouter"},
                "models": [
                    {"id": "anthropic/claude-opus-4.7", "description": "recommended"},
                    {"id": "openai/gpt-5.4", "description": ""},
                    {"id": "openrouter/elephant-alpha", "description": "free"},
                ],
            },
            "nous": {
                "metadata": {"display_name": "Nous Portal"},
                "models": [
                    {"id": "anthropic/claude-opus-4.7"},
                    {"id": "moonshotai/kimi-k2.6"},
                ],
            },
        },
    }


class TestValidation:
    def test_accepts_well_formed_manifest(self, isolated_home):
        from hermes_cli.model_catalog import _validate_manifest
        assert _validate_manifest(_valid_manifest()) is True

    def test_rejects_non_dict(self, isolated_home):
        from hermes_cli.model_catalog import _validate_manifest
        assert _validate_manifest("string") is False
        assert _validate_manifest([]) is False
        assert _validate_manifest(None) is False

    def test_rejects_missing_version(self, isolated_home):
        from hermes_cli.model_catalog import _validate_manifest
        m = _valid_manifest()
        del m["version"]
        assert _validate_manifest(m) is False

    def test_rejects_future_version(self, isolated_home):
        from hermes_cli.model_catalog import _validate_manifest
        m = _valid_manifest()
        m["version"] = 999
        assert _validate_manifest(m) is False

    def test_rejects_missing_providers(self, isolated_home):
        from hermes_cli.model_catalog import _validate_manifest
        m = _valid_manifest()
        del m["providers"]
        assert _validate_manifest(m) is False

    def test_rejects_malformed_model_entry(self, isolated_home):
        from hermes_cli.model_catalog import _validate_manifest
        m = _valid_manifest()
        m["providers"]["openrouter"]["models"][0] = {"id": ""}  # empty id
        assert _validate_manifest(m) is False

    def test_rejects_non_string_model_id(self, isolated_home):
        from hermes_cli.model_catalog import _validate_manifest
        m = _valid_manifest()
        m["providers"]["openrouter"]["models"][0] = {"id": 42}
        assert _validate_manifest(m) is False


class TestFetchSuccess:
    def test_fetch_and_cache_writes_disk(self, isolated_home):
        from hermes_cli import model_catalog
        manifest = _valid_manifest()
        with patch.object(
            model_catalog, "_fetch_manifest", return_value=manifest
        ) as fetch:
            result = model_catalog.get_catalog(force_refresh=True)

        assert result == manifest
        assert fetch.called

        cache_file = model_catalog._cache_path()
        assert cache_file.exists()
        with open(cache_file) as fh:
            assert json.load(fh) == manifest

    def test_second_call_uses_in_process_cache(self, isolated_home):
        from hermes_cli import model_catalog
        manifest = _valid_manifest()
        with patch.object(
            model_catalog, "_fetch_manifest", return_value=manifest
        ) as fetch:
            model_catalog.get_catalog(force_refresh=True)
            model_catalog.get_catalog()  # should not hit network again
        assert fetch.call_count == 1

    def test_force_refresh_always_refetches(self, isolated_home):
        from hermes_cli import model_catalog
        manifest = _valid_manifest()
        with patch.object(
            model_catalog, "_fetch_manifest", return_value=manifest
        ) as fetch:
            model_catalog.get_catalog(force_refresh=True)
            model_catalog.get_catalog(force_refresh=True)
        assert fetch.call_count == 2


class TestFetchFailure:
    def test_network_failure_returns_empty_when_no_cache(self, isolated_home):
        from hermes_cli import model_catalog
        with patch.object(model_catalog, "_fetch_manifest", return_value=None):
            result = model_catalog.get_catalog(force_refresh=True)
        assert result == {}

    def test_network_failure_falls_back_to_disk_cache(self, isolated_home):
        from hermes_cli import model_catalog
        # Prime disk cache with a fresh copy.
        manifest = _valid_manifest()
        with patch.object(model_catalog, "_fetch_manifest", return_value=manifest):
            model_catalog.get_catalog(force_refresh=True)

        # Now wipe in-process cache and simulate network failure on refetch.
        model_catalog.reset_cache()
        with patch.object(model_catalog, "_fetch_manifest", return_value=None):
            result = model_catalog.get_catalog(force_refresh=True)

        assert result == manifest

    def test_fetch_failure_falls_back_to_stale_cache(self, isolated_home):
        from hermes_cli import model_catalog
        manifest = _valid_manifest()
        # Write stale cache directly (mtime in the past).
        cache = model_catalog._cache_path()
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as fh:
            json.dump(manifest, fh)
        old = time.time() - 30 * 24 * 3600  # 30 days ago
        import os as _os
        _os.utime(cache, (old, old))

        with patch.object(model_catalog, "_fetch_manifest", return_value=None):
            result = model_catalog.get_catalog()

        # Stale cache is better than nothing.
        assert result == manifest


class TestFallbackChain:
    """``_fetch_manifest_with_fallback`` walks ``DEFAULT_CATALOG_FALLBACK_URLS``
    when the primary URL fails. Regression: the Docusaurus site behind Vercel
    occasionally returns HTTP 403 + x-vercel-mitigated: challenge for urllib;
    without a fallback URL the user's disk cache freezes and new model
    releases (opus 4.8, etc.) never reach the picker.
    """

    PRIMARY = "https://hermes-agent.nousresearch.com/docs/api/model-catalog.json"
    FALLBACK = (
        "https://raw.githubusercontent.com/NousResearch/hermes-agent"
        "/main/website/static/api/model-catalog.json"
    )

    def test_uses_primary_when_it_succeeds(self, isolated_home):
        from hermes_cli import model_catalog
        calls: list[str] = []

        def fake_fetch(url, timeout):
            calls.append(url)
            return _valid_manifest()

        with patch.object(model_catalog, "_fetch_manifest", side_effect=fake_fetch):
            result = model_catalog._fetch_manifest_with_fallback(self.PRIMARY, 5.0)

        assert result is not None
        assert calls == [self.PRIMARY], "fallback URLs must not be touched on primary success"

    def test_falls_through_to_raw_github_on_primary_failure(self, isolated_home):
        from hermes_cli import model_catalog
        calls: list[str] = []

        def fake_fetch(url, timeout):
            calls.append(url)
            if url == self.PRIMARY:
                return None  # simulate Vercel 403
            return _valid_manifest()

        with patch.object(model_catalog, "_fetch_manifest", side_effect=fake_fetch):
            result = model_catalog._fetch_manifest_with_fallback(self.PRIMARY, 5.0)

        assert result is not None
        assert calls == [self.PRIMARY, self.FALLBACK]

    def test_returns_none_when_all_urls_fail(self, isolated_home):
        from hermes_cli import model_catalog

        with patch.object(model_catalog, "_fetch_manifest", return_value=None) as fetch:
            result = model_catalog._fetch_manifest_with_fallback(self.PRIMARY, 5.0)

        assert result is None
        # Primary + every fallback URL was attempted exactly once.
        assert fetch.call_count == 1 + len(model_catalog.DEFAULT_CATALOG_FALLBACK_URLS)

    def test_dedupes_when_primary_equals_fallback(self, isolated_home):
        """Operator who configured ``model_catalog.url`` to the raw GitHub URL
        should not get a duplicate fetch from the fallback list."""
        from hermes_cli import model_catalog

        with patch.object(model_catalog, "_fetch_manifest", return_value=None) as fetch:
            model_catalog._fetch_manifest_with_fallback(self.FALLBACK, 5.0)

        assert fetch.call_count == 1, f"expected 1 call, got {fetch.call_count}"

    def test_get_catalog_uses_fallback_chain(self, isolated_home):
        """End-to-end: ``get_catalog`` routes through the fallback helper so
        a primary URL failure transparently produces a working catalog."""
        from hermes_cli import model_catalog
        manifest = _valid_manifest()
        calls: list[str] = []

        def fake_fetch(url, timeout):
            calls.append(url)
            if url == self.PRIMARY:
                return None
            return manifest

        with patch.object(model_catalog, "_fetch_manifest", side_effect=fake_fetch):
            result = model_catalog.get_catalog(force_refresh=True)

        assert result == manifest
        assert self.FALLBACK in calls


class TestCuratedAccessors:
    def test_openrouter_returns_tuples(self, isolated_home):
        from hermes_cli import model_catalog
        with patch.object(
            model_catalog, "_fetch_manifest", return_value=_valid_manifest()
        ):
            result = model_catalog.get_curated_openrouter_models()
        assert result == [
            ("anthropic/claude-opus-4.7", "recommended"),
            ("openai/gpt-5.4", ""),
            ("openrouter/elephant-alpha", "free"),
        ]

    def test_nous_returns_ids(self, isolated_home):
        from hermes_cli import model_catalog
        with patch.object(
            model_catalog, "_fetch_manifest", return_value=_valid_manifest()
        ):
            result = model_catalog.get_curated_nous_models()
        assert result == ["anthropic/claude-opus-4.7", "moonshotai/kimi-k2.6"]

    def test_openrouter_returns_none_when_catalog_empty(self, isolated_home):
        from hermes_cli import model_catalog
        with patch.object(model_catalog, "_fetch_manifest", return_value=None):
            assert model_catalog.get_curated_openrouter_models() is None

    def test_nous_returns_none_when_catalog_empty(self, isolated_home):
        from hermes_cli import model_catalog
        with patch.object(model_catalog, "_fetch_manifest", return_value=None):
            assert model_catalog.get_curated_nous_models() is None


class TestDisabled:
    def test_disabled_config_short_circuits(self, isolated_home):
        from hermes_cli import model_catalog
        with patch.object(
            model_catalog,
            "_load_catalog_config",
            return_value={
                "enabled": False,
                "url": "http://ignored",
                "ttl_hours": 24.0,
                "providers": {},
            },
        ):
            with patch.object(model_catalog, "_fetch_manifest") as fetch:
                result = model_catalog.get_catalog()
        assert result == {}
        fetch.assert_not_called()


class TestProviderOverride:
    def test_override_url_takes_precedence(self, isolated_home):
        from hermes_cli import model_catalog

        override_payload = {
            "version": 1,
            "providers": {
                "openrouter": {
                    "models": [
                        {"id": "override/model", "description": "custom"},
                    ]
                }
            },
        }

        def fake_fetch(url, timeout):
            if "override" in url:
                return override_payload
            return _valid_manifest()

        with patch.object(
            model_catalog,
            "_load_catalog_config",
            return_value={
                "enabled": True,
                "url": "http://master",
                "ttl_hours": 24.0,
                "providers": {"openrouter": {"url": "http://override"}},
            },
        ):
            with patch.object(model_catalog, "_fetch_manifest", side_effect=fake_fetch):
                result = model_catalog.get_curated_openrouter_models()

        assert result == [("override/model", "custom")]


class TestIntegrationWithModelsModule:
    """Exercise the fallback paths via the real callers in hermes_cli.models."""

    def test_curated_nous_ids_falls_back_to_hardcoded_on_empty_catalog(
        self, isolated_home
    ):
        from hermes_cli import model_catalog
        from hermes_cli.models import get_curated_nous_model_ids, _PROVIDER_MODELS

        with (
            patch.object(model_catalog, "_fetch_manifest", return_value=None),
            patch("urllib.request.urlopen", side_effect=Exception("net down")),
        ):
            result = get_curated_nous_model_ids()

        assert result == list(_PROVIDER_MODELS["nous"])

    def test_curated_nous_ids_prefers_manifest(self, isolated_home):
        from hermes_cli import model_catalog
        from hermes_cli.models import get_curated_nous_model_ids

        with (
            patch.object(
                model_catalog, "_fetch_manifest", return_value=_valid_manifest()
            ),
            patch("urllib.request.urlopen", side_effect=Exception("net down")),
        ):
            result = get_curated_nous_model_ids()

        assert result == ["anthropic/claude-opus-4.7", "moonshotai/kimi-k2.6"]

    def test_picker_nous_row_uses_curated_list(self, tmp_path, monkeypatch):
        """The /model picker surfaces the curated ``_PROVIDER_MODELS["nous"]``
        list in curated order — matching the ``hermes model`` CLI — not the live
        ``/v1/models`` catalog or the manifest. Portal free/paid recommendations
        are unioned in when reachable; offline (as here, with the Portal calls
        stubbed out) it's exactly the curated list.
        """
        # We deliberately do NOT use the ``isolated_home`` fixture here:
        # that fixture monkeypatches ``Path.home`` to ``tmp_path``, which
        # trips the auth-store seat-belt in ``_auth_file_path()`` because
        # ``HERMES_HOME / auth.json`` then resolves to the same path the
        # seat-belt thinks is the "real" user store. Use the autouse
        # ``_hermetic_environment`` HERMES_HOME directly instead.
        import importlib
        from hermes_cli import model_catalog
        from hermes_cli.models import get_curated_nous_model_ids
        importlib.reload(model_catalog)
        try:
            from hermes_cli.model_switch import list_picker_providers

            active_home = Path(os.environ["HERMES_HOME"])
            (active_home / "auth.json").write_text(
                json.dumps(
                    {
                        "providers": {"nous": {"access_token": "fake"}},
                        "credential_pool": {},
                    }
                )
            )

            # Stub the Portal recommendation union so the row is deterministic
            # (the curated list alone) and never touches the network. ``expected``
            # is computed from the same source the picker uses internally
            # (``curated["nous"] = get_curated_nous_model_ids()``), so the test
            # stays an invariant — it can't rot as the curated/manifest list grows.
            with (
                patch.object(
                    model_catalog, "_fetch_manifest", return_value=_valid_manifest()
                ),
                patch("urllib.request.urlopen", side_effect=Exception("net down")),
                patch("hermes_cli.models.check_nous_free_tier", return_value=False),
                patch(
                    "hermes_cli.models.union_with_portal_free_recommendations",
                    side_effect=lambda ids, *a, **k: (ids, {}),
                ),
                patch(
                    "hermes_cli.models.union_with_portal_paid_recommendations",
                    side_effect=lambda ids, *a, **k: (ids, {}),
                ),
            ):
                expected = get_curated_nous_model_ids()
                picker = list_picker_providers(
                    current_provider="nous", max_models=99
                )
        finally:
            model_catalog.reset_cache()

        nous_row = next((r for r in picker if r["slug"] == "nous"), None)
        assert nous_row is not None, "nous row must appear when authed"
        assert nous_row["models"] == expected


# -----------------------------------------------------------------------------
# Drift guard — prevent the in-repo curated lists from going out of sync with
# the docs-hosted manifest at website/static/api/model-catalog.json.
#
# History: qwen/qwen3.6-plus was added to _PROVIDER_MODELS["nous"] in commit
# 9dd6e5510 but website/static/api/model-catalog.json was not regenerated for
# weeks, so free-tier users on a new install fetched a stale manifest and the
# free-tier picker showed "No free models currently available." even though
# the Portal was serving qwen/qwen3.6-plus as free. CI must catch this.
# -----------------------------------------------------------------------------


class TestManifestMatchesInRepoLists:
    """Fail if the on-disk manifest is out of date relative to in-repo lists."""

    @staticmethod
    def _strip_volatile(catalog: dict) -> dict:
        """Drop fields that always change (timestamps) for diff comparison."""
        out = dict(catalog)
        out.pop("updated_at", None)
        return out

    def test_in_repo_lists_match_manifest(self):
        """``scripts/build_model_catalog.py`` output must match the committed file.

        If this fails, run ``python scripts/build_model_catalog.py`` and
        commit the regenerated ``website/static/api/model-catalog.json``.
        """
        # Resolve the repo root from this test file's location.
        repo_root = Path(__file__).resolve().parents[2]
        manifest_path = repo_root / "website" / "static" / "api" / "model-catalog.json"

        if not manifest_path.exists():
            pytest.skip(f"manifest missing at {manifest_path}")

        # Build expected catalog using the same script CI would.
        import importlib.util
        script_path = repo_root / "scripts" / "build_model_catalog.py"
        spec = importlib.util.spec_from_file_location("_build_model_catalog", script_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        expected = mod.build_catalog()

        with open(manifest_path, encoding="utf-8") as fh:
            actual = json.load(fh)

        assert self._strip_volatile(actual) == self._strip_volatile(expected), (
            "website/static/api/model-catalog.json is out of sync with "
            "_PROVIDER_MODELS['nous'] / OPENROUTER_MODELS. "
            "Run: python scripts/build_model_catalog.py && "
            "git add website/static/api/model-catalog.json"
        )


class TestNousLiveProbe:
    """Tests for the live Nous inference API probe in ``hermes_cli.models``."""

    def test_probe_merges_and_caches(self, isolated_home):
        """Live API models get merged in (minus Hermes 3/4), and the 120s
        cache prevents redundant network calls on subsequent probes."""
        from hermes_cli.models import get_curated_nous_model_ids
        from hermes_cli import model_catalog
        import hermes_cli.models as m

        m._nous_live_cache = None

        body = json.dumps({"data": [
            {"id": "anthropic/claude-opus-4.7"},
            {"id": "deepseek/deepseek-v4-flash:free"},
            {"id": "NousResearch/hermes-4"},
            {"id": "nousresearch/hermes-3-llama-3.1-70b"},
        ]})
        fake_resp = MagicMock()
        fake_resp.read.return_value = body.encode()
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = lambda s, *a: None

        call_count = 0

        def fake_urlopen(*a, **kw):
            nonlocal call_count
            call_count += 1
            return fake_resp

        with (
            patch.object(model_catalog, "_fetch_manifest", return_value=None),
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            r1 = get_curated_nous_model_ids()
            r2 = get_curated_nous_model_ids()
            r3 = get_curated_nous_model_ids()

        # Network was hit exactly once — subsequent calls used the cache
        assert call_count == 1
        # Live-only model discovered
        assert "deepseek/deepseek-v4-flash:free" in r1
        # Nous Hermes foundation models filtered out
        assert "NousResearch/hermes-4" not in r1
        assert "nousresearch/hermes-3-llama-3.1-70b" not in r1
        # Cache hit returned the same result
        assert r1 == r2 == r3
