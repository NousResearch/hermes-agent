"""Model-list provenance (#110055): which source a provider's list came from, and why.

Covers the judgement rules (each source → its line), the fallback REASON propagation out of the
resolution layer (``hermes_cli.models`` / ``model_catalog``), and the picker row payload the CLI /
uidrivers render.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import hermes_cli.providers as providers_mod
from hermes_cli import model_list_provenance as prov


@pytest.fixture(autouse=True)
def _clean_journal():
    """The journal is process-global: every test starts from nothing recorded."""
    prov.reset()
    yield
    prov.reset()


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so disk caches / manifests never touch the real one."""
    home = tmp_path / ".hermes"
    (home / "cache").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_manifest(home: Path, *, age_seconds: float = 0.0) -> Path:
    path = home / "cache" / "model_catalog.json"
    path.write_text(json.dumps({
        "version": 1,
        "providers": {"nous": {"models": [{"id": "nous/hermes-4"}]}},
    }), encoding="utf-8")
    if age_seconds:
        old = time.time() - age_seconds
        import os
        os.utime(path, (old, old))
    return path


class TestLabelRules:
    def test_unknown_provider_has_no_provenance(self):
        assert prov.provider_provenance("nobody-resolved-this") == {}

    def test_bundled_fallback_carries_its_reason(self):
        prov.record("copilot", prov.BUNDLED, reason="GitHub catalog token not found", count=7)
        entry = prov.provider_provenance("copilot")
        assert entry["label"] == "bundled fallback — GitHub catalog token not found"
        assert entry["degraded"] is True

    def test_plain_bundled_list_is_not_degraded(self):
        prov.record("kimi-coding", prov.BUNDLED, count=4)
        entry = prov.provider_provenance("kimi-coding")
        assert entry["label"] == "bundled in-repo list"
        assert entry["degraded"] is False

    def test_live_list_shows_count_and_age(self):
        prov.record("openrouter", prov.LIVE, count=12, at=time.time() - 120)
        assert prov.provider_provenance("openrouter")["label"] == "live · 12 models · 2m ago"

    def test_discovered_list_names_its_endpoint(self):
        entry = prov.describe(prov.DISCOVERED, count=55, detail="/models on http://box:8000/v1")
        assert entry["label"] == "/models on http://box:8000/v1 · 55 models"

    def test_hosted_catalog_snapshot_dates_itself(self):
        entry = prov.describe(prov.CATALOG, at=time.time() - 3600)
        assert entry["label"].startswith("hosted catalog · snapshot ")
        assert "stale" not in entry["label"]

    def test_stale_hosted_catalog_says_to_refresh(self):
        entry = prov.describe(prov.CATALOG, at=time.time() - 40 * 86400, stale=True)
        assert entry["label"].endswith("— stale, run /model --refresh")

    def test_age_labels_coarsen(self):
        assert prov.age_label(5) == "just now"
        assert prov.age_label(600) == "10m ago"
        assert prov.age_label(7200) == "2h ago"
        assert prov.age_label(3 * 86400) == "3d ago"


class TestEndpointRules:
    def test_discovery_result_wins(self):
        entry = prov.endpoint_provenance("http://box:8000/v1", discovered=True,
                                         discovery_allowed=True, count=55)
        assert entry["source"] == prov.DISCOVERED
        assert "http://box:8000/v1" in entry["label"]

    def test_pinned_discovery_reads_as_configured(self):
        entry = prov.endpoint_provenance("http://box:8000/v1", discovered=None,
                                        discovery_allowed=False)
        assert entry["label"] == "configured models: list (discover_models: false)"

    def test_unanswered_probe_says_so(self):
        entry = prov.endpoint_provenance("http://box:8000/v1", discovered=None,
                                         discovery_allowed=True)
        assert entry["label"] == "configured models: list — live /models unavailable"


class TestCatalogStatus:
    def test_missing_manifest_reports_unavailable(self, isolated_home):
        from hermes_cli.model_catalog import catalog_status
        assert catalog_status()["origin"] == "unavailable"

    def test_disk_manifest_is_the_hosted_snapshot(self, isolated_home):
        _write_manifest(isolated_home)
        from hermes_cli.model_catalog import catalog_status
        status = catalog_status()
        assert status["origin"] == "hosted" and status["fresh"] is True
        assert prov.catalog_provenance()["source"] == prov.CATALOG

    def test_day_old_snapshot_is_flagged_stale(self, isolated_home):
        _write_manifest(isolated_home, age_seconds=3 * 86400)
        assert prov.catalog_provenance()["stale"] is True


class TestResolutionRecords:
    def test_copilot_without_token_reports_the_bundled_fallback(self, monkeypatch):
        from hermes_cli import models

        monkeypatch.setattr(models, "_resolve_copilot_catalog_api_key", lambda: "")
        monkeypatch.setattr(models, "_fetch_github_models", lambda *_a, **_k: None)

        ids = models.provider_model_ids("copilot")

        entry = prov.provider_provenance("copilot")
        assert ids, "bundled copilot list still has to be returned"
        assert entry["source"] == prov.BUNDLED
        assert entry["reason"] == "GitHub catalog token not found"

    def test_copilot_with_a_live_catalog_is_live(self, monkeypatch):
        from hermes_cli import models

        monkeypatch.setattr(models, "_resolve_copilot_catalog_api_key", lambda: "gho_x")
        monkeypatch.setattr(models, "_fetch_github_models", lambda *_a, **_k: ["gpt-6-astra"])

        assert models.provider_model_ids("copilot") == ["gpt-6-astra"]
        assert prov.provider_provenance("copilot")["source"] == prov.LIVE

    def test_missing_api_key_reports_the_profile_fallback_reason(self, monkeypatch):
        from hermes_cli import models

        class _Profile:
            auth_type = "api_key"
            base_url = "https://api.example/v1"
            fallback_models = ("example-1",)

        monkeypatch.setattr("providers.get_provider_profile", lambda _name: _Profile())
        monkeypatch.setattr(models, "_api_key_credentials", lambda _n: ("", ""))

        assert models.provider_model_ids("example") == ["example-1"]
        entry = prov.provider_provenance("example")
        assert entry["degraded"] is True
        assert entry["reason"] == "no API key configured"

    def test_disk_cache_row_carries_provenance_for_the_next_process(self, isolated_home, monkeypatch):
        from hermes_cli import models

        monkeypatch.setattr(models, "_credential_fingerprint", lambda _p: "fp")
        monkeypatch.setattr(models, "provider_model_ids", lambda _p, **_k: ["m1", "m2"])
        # What the resolution layer journals right before the row is written.
        prov.record("openrouter", prov.LIVE, count=2)

        assert models.cached_provider_model_ids("openrouter") == ["m1", "m2"]
        row = json.loads((isolated_home / "provider_models_cache.json").read_text())["openrouter"]
        assert row["source"] == prov.LIVE and row["models"] == ["m1", "m2"]

    def test_cache_hit_reports_the_row_age_not_now(self, isolated_home, monkeypatch):
        from hermes_cli import models

        monkeypatch.setattr(models, "_credential_fingerprint", lambda _p: "fp")
        monkeypatch.setattr(
            models, "_load_provider_models_cache",
            lambda: {"openrouter": {"fp": "fp", "at": time.time() - 600, "models": ["m1"]}})

        assert models.cached_provider_model_ids("openrouter") == ["m1"]
        entry = prov.provider_provenance("openrouter")
        assert entry["label"] == "live · 1 model · 10m ago"


class TestPickerRows:
    def _custom(self, **extra):
        entry = {"name": "Box", "base_url": "http://box:8000/v1", "model": "m1"}
        entry.update(extra)
        return entry

    def _rows(self, monkeypatch, custom_providers, discovered):
        from hermes_cli.model_switch import list_authenticated_providers

        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
        monkeypatch.setattr(
            "hermes_cli.model_switch_providers._discover_endpoint_models",
            lambda *_a, **_k: (discovered, False))
        return list_authenticated_providers(
            current_provider="custom", current_base_url="http://box:8000/v1",
            user_providers={}, custom_providers=custom_providers)

    def _box_row(self, rows):
        return next(r for r in rows if str(r["slug"]).endswith(":box") or r["slug"] == "box")

    def test_pinned_custom_provider_row_says_configured(self, monkeypatch):
        rows = self._rows(monkeypatch, [self._custom(discover_models=False)], discovered=None)
        row = self._box_row(rows)
        assert row["provenance"]["label"] == "configured models: list (discover_models: false)"

    def test_discovered_custom_provider_row_names_the_endpoint(self, monkeypatch):
        rows = self._rows(monkeypatch, [self._custom()], discovered=["a", "b", "c"])
        row = self._box_row(rows)
        assert row["provenance"]["source"] == prov.DISCOVERED
        assert row["provenance"]["label"].startswith("/models on http://box:8000/v1 · 3 models")

    def test_bare_custom_endpoint_row_also_reports_its_source(self, monkeypatch):
        # Section 3b: `model.provider: custom` + base_url, no custom_providers row.
        rows = self._rows(monkeypatch, [], discovered=None)
        row = next(r for r in rows if r["slug"] == "custom")
        assert row["provenance"]["label"] == "configured models: list — live /models unavailable"

    def test_payload_keeps_provenance_for_the_gui(self, monkeypatch):
        from hermes_cli.inventory import ConfigContext, build_models_payload

        rows = self._rows(monkeypatch, [self._custom()], discovered=["a"])
        monkeypatch.setattr(
            "hermes_cli.model_switch.list_authenticated_providers", lambda **_kw: rows)

        ctx = ConfigContext(
            current_provider="custom", current_base_url="http://box:8000/v1",
            current_model="a", user_providers={}, custom_providers=[], excluded_providers=[])
        payload = build_models_payload(ctx, max_models=None)
        assert self._box_row(payload["providers"])["provenance"]["label"]


class TestCliFlowMessage:
    def test_copilot_flow_prints_the_fallback_reason(self, monkeypatch, capsys):
        from hermes_cli import model_setup_flows
        from hermes_cli.models import _PROVIDER_MODELS

        prov.record("copilot", prov.BUNDLED, reason="GitHub catalog token not found")
        monkeypatch.setattr(model_setup_flows, "_say", lambda *lines, **kw: print(*lines))

        model_setup_flows._copilot_model_list(None)

        out = capsys.readouterr().out
        assert "showing the bundled list" in out
        assert "GitHub catalog token not found" in out
        assert len(_PROVIDER_MODELS["copilot"]) > 0  # the list it warns about is non-empty
