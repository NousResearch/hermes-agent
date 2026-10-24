from __future__ import annotations

import importlib.util
import unittest
from datetime import datetime, timezone, UTC
from pathlib import Path
from types import SimpleNamespace

PLUGIN_API = Path(__file__).resolve().parents[1] / "dashboard" / "plugin_api.py"


def load_plugin_api():
    spec = importlib.util.spec_from_file_location("ai_usage_meter_plugin_api", PLUGIN_API)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def snapshot(provider, *, percent=25.0, detail=None, plan=None):
    return SimpleNamespace(
        provider=provider,
        source="usage_api",
        fetched_at=datetime(2026, 10, 13, 9, 0, tzinfo=UTC),
        title="Account limits",
        plan=plan,
        windows=(
            SimpleNamespace(
                label="Weekly",
                used_percent=percent,
                reset_at=datetime(2026, 10, 20, 9, 0, tzinfo=UTC),
                detail=detail,
                scope="account",
            ),
        ),
        details=(),
        unavailable_reason=None,
        available=True,
    )


def discovered(*provider_ids):
    names = {
        "anthropic": "Anthropic",
        "openai-codex": "ChatGPT or Codex Subscription",
        "openrouter": "OpenRouter",
        "ollama-local": "Ollama Local",
    }
    return [
        {
            "id": provider_id,
            "name": names.get(provider_id, provider_id),
            "models": [],
            "aliases": [provider_id],
            "available": True,
            "source": "LIVE" if provider_id == "ollama-local" else "HERMES",
            "is_ollama": provider_id == "ollama-local",
        }
        for provider_id in provider_ids
    ]


class UsageMeterTests(unittest.TestCase):
    def test_only_discovered_providers_are_returned(self):
        api = load_plugin_api()
        payload = api.build_usage_payload(
            usage_fetcher=lambda provider: snapshot(provider),
            analytics_loader=lambda days: {"models": [], "period_days": days},
            provider_discovery=lambda refresh: discovered("openrouter", "ollama-local"),
        )
        providers = {provider["id"]: provider for provider in payload["providers"]}
        self.assertEqual(set(providers), {"openrouter", "ollama-local"})
        self.assertTrue(providers["ollama-local"]["available"])
        self.assertEqual(providers["ollama-local"]["source"], "LIVE")

    def test_build_usage_payload_keeps_same_model_separate_by_provider(self):
        api = load_plugin_api()
        snapshots = {
            "anthropic": snapshot("anthropic", percent=45.0),
            "openai-codex": snapshot("openai-codex", percent=14.0, plan="Plus"),
            "openrouter": snapshot("openrouter", percent=20.0, detail="$40.00 of $50.00 remaining"),
        }
        analytics = {
            "models": [
                {"model": "claude-opus-5.5", "provider": "anthropic", "input_tokens": 100,
                 "output_tokens": 25, "cache_read_tokens": 5, "reasoning_tokens": 0,
                 "actual_cost": 0.0, "estimated_cost": 1.25, "api_calls": 2, "sessions": 1},
                {"model": "anthropic/claude-opus-5.5", "provider": "openrouter", "input_tokens": 200,
                 "output_tokens": 50, "cache_read_tokens": 0, "reasoning_tokens": 10,
                 "actual_cost": 2.5, "estimated_cost": 0.0, "api_calls": 3, "sessions": 1},
            ],
            "period_days": 7,
        }
        payload = api.build_usage_payload(
            days=30,
            usage_fetcher=lambda provider: snapshots[provider],
            analytics_loader=lambda days: analytics,
            provider_discovery=lambda refresh: discovered("anthropic", "openai-codex", "openrouter"),
        )
        providers = {provider["id"]: provider for provider in payload["providers"]}
        anthropic_opus = next(model for model in providers["anthropic"]["models"] if model["name"] == "claude-opus-5.5")
        openrouter_opus = next(model for model in providers["openrouter"]["models"] if model["name"] == "anthropic/claude-opus-5.5")
        self.assertEqual(anthropic_opus["total_tokens"], 125)
        self.assertEqual(openrouter_opus["total_tokens"], 250)
        self.assertEqual(providers["openrouter"]["windows"][0]["remaining_percent"], 80.0)
        self.assertEqual(providers["openrouter"]["windows"][0]["detail"], "$40.00 of $50.00 remaining")
        self.assertEqual(payload["model_period_days"], 7)

    def test_build_usage_payload_ignores_malformed_historical_numbers(self):
        api = load_plugin_api()
        malformed = {"models": [{"model": "gpt-5.6-sol", "provider": "openai-codex",
                                   "input_tokens": "not-a-number", "output_tokens": None,
                                   "actual_cost": "broken"}], "period_days": 7}
        payload = api.build_usage_payload(
            usage_fetcher=lambda provider: snapshot(provider),
            analytics_loader=lambda days: malformed,
            provider_discovery=lambda refresh: discovered("openai-codex"),
        )
        self.assertEqual(payload["providers"][0]["models"], [])

    def test_models_are_rolling_seven_day_nonzero_descending_top_three(self):
        api = load_plugin_api()
        requested_days = []
        analytics = {
            "models": [
                {"model": "fourth", "provider": "openrouter", "input_tokens": 25, "output_tokens": 0},
                {"model": "first", "provider": "openrouter", "input_tokens": 100, "output_tokens": 5},
                {"model": "zero", "provider": "openrouter", "input_tokens": 0, "output_tokens": 0},
                {"model": "third", "provider": "openrouter", "input_tokens": 40, "output_tokens": 5},
                {"model": "second", "provider": "openrouter", "input_tokens": 75, "output_tokens": 0},
            ],
            "period_days": 7,
        }

        def load(days):
            requested_days.append(days)
            return analytics

        payload = api.build_usage_payload(
            usage_fetcher=lambda provider: None,
            analytics_loader=load,
            provider_discovery=lambda refresh: discovered("openrouter"),
        )
        models = payload["providers"][0]["models"]
        self.assertEqual(requested_days, [7])
        self.assertEqual([model["name"] for model in models], ["first", "second", "third"])
        self.assertEqual([model["total_tokens"] for model in models], [105, 75, 45])

    def test_refresh_flag_is_forwarded_to_provider_scan(self):
        api = load_plugin_api()
        refresh_values = []
        api.build_usage_payload(
            refresh=True,
            usage_fetcher=lambda provider: None,
            analytics_loader=lambda days: {"models": [], "period_days": days},
            provider_discovery=lambda refresh: refresh_values.append(refresh) or discovered("openrouter"),
        )
        self.assertEqual(refresh_values, [True])

    def test_payload_exposes_backend_version_for_stale_runtime_detection(self):
        api = load_plugin_api()
        payload = api.build_usage_payload(
            usage_fetcher=lambda provider: None,
            analytics_loader=lambda days: {"models": [], "period_days": days},
            provider_discovery=lambda refresh: discovered("openrouter"),
        )
        self.assertEqual(payload["backend_version"], "0.2.0")

    def test_ollama_probe_replaces_configured_catalog_with_live_models(self):
        api = load_plugin_api()
        normalized = api._normalize_discovered_providers(
            [{"slug": "ollama-local", "name": "Local endpoint", "models": ["configured-only"],
              "api_url": "http://127.0.0.1:11434/v1", "is_user_defined": True}],
            ollama_probe=lambda base_url: ["qwen3.5:2b"],
        )
        self.assertEqual(normalized[0]["name"], "Ollama Local")
        self.assertEqual(normalized[0]["models"], ["qwen3.5:2b"])
        self.assertEqual(normalized[0]["source"], "LIVE")

    def test_failed_ollama_probe_does_not_mark_endpoint_live(self):
        api = load_plugin_api()
        normalized = api._normalize_discovered_providers(
            [{"slug": "saved-local", "name": "Saved local", "models": ["configured-only"],
              "api_url": "http://127.0.0.1:11434/v1", "is_user_defined": True}],
            ollama_probe=lambda base_url: None,
        )
        self.assertEqual(normalized[0]["name"], "Saved local")
        self.assertEqual(normalized[0]["source"], "HERMES")
        self.assertFalse(normalized[0]["is_ollama"])

    def test_one_provider_usage_failure_does_not_hide_other_providers(self):
        api = load_plugin_api()

        def fetch(provider):
            if provider == "anthropic":
                raise RuntimeError("temporary failure")
            return snapshot(provider)

        payload = api.build_usage_payload(
            usage_fetcher=fetch,
            analytics_loader=lambda days: {"models": [], "period_days": days},
            provider_discovery=lambda refresh: discovered("anthropic", "openrouter"),
        )
        providers = {provider["id"]: provider for provider in payload["providers"]}
        self.assertEqual(set(providers), {"anthropic", "openrouter"})
        self.assertTrue(providers["openrouter"]["available"])
        self.assertTrue(any("anthropic usage unavailable" in error for error in payload["errors"]))


if __name__ == "__main__":
    unittest.main()
