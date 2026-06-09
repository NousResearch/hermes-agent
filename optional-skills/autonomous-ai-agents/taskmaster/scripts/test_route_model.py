"""Tests for TaskMaster model routing script.

Run with: python -m pytest optional-skills/autonomous-ai-agents/taskmaster/scripts/test_route_model.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the module under test
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from route_model import ROUTING_TABLE, TIER_DESCRIPTIONS, resolve_model


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestRoutingTable:
    """Validate the routing table structure."""

    def test_all_tiers_exist(self):
        assert set(ROUTING_TABLE.keys()) == {"LOW", "MEDIUM", "HIGH", "VOTE"}

    def test_all_tiers_have_descriptions(self):
        for tier in ROUTING_TABLE:
            assert tier in TIER_DESCRIPTIONS
            assert len(TIER_DESCRIPTIONS[tier]) > 0

    def test_each_tier_has_at_least_one_provider(self):
        for tier, providers in ROUTING_TABLE.items():
            assert len(providers) >= 1, f"Tier {tier} has no providers"

    def test_each_provider_has_at_least_one_model(self):
        for tier, providers in ROUTING_TABLE.items():
            for provider, models in providers.items():
                assert len(models) >= 1, (
                    f"Tier {tier}, provider {provider} has no models"
                )

    def test_model_names_are_strings(self):
        for tier, providers in ROUTING_TABLE.items():
            for provider, models in providers.items():
                for model in models:
                    assert isinstance(model, str)
                    assert len(model) > 0


class TestResolveModel:
    """Test the resolve_model function."""

    def test_resolve_low_tier(self):
        result = resolve_model("LOW")
        assert result["tier"] == "LOW"
        assert "model" in result
        assert "provider" in result

    def test_resolve_medium_tier(self):
        result = resolve_model("MEDIUM")
        assert result["tier"] == "MEDIUM"
        assert "model" in result

    def test_resolve_high_tier(self):
        result = resolve_model("HIGH")
        assert result["tier"] == "HIGH"
        assert "model" in result

    def test_resolve_vote_tier(self):
        result = resolve_model("VOTE")
        assert result["tier"] == "VOTE"
        assert "model" in result

    def test_resolve_with_provider(self):
        result = resolve_model("LOW", "openrouter")
        assert result["provider"] == "openrouter"
        assert result["tier"] == "LOW"
        assert "openrouter" in result["model"] or "/" in result["model"]

    def test_resolve_with_google_provider(self):
        result = resolve_model("MEDIUM", "google")
        assert result["provider"] == "google"
        assert "gemini" in result["model"]

    def test_unknown_tier_returns_error(self):
        result = resolve_model("INVALID")
        assert "error" in result
        assert "INVALID" in result["error"]

    def test_unknown_provider_returns_error(self):
        result = resolve_model("LOW", "nonexistent_provider")
        assert "error" in result
        assert "nonexistent_provider" in result["error"]

    def test_case_insensitive_tier(self):
        result = resolve_model("low")
        assert result["tier"] == "LOW"
        assert "model" in result

    def test_result_has_description(self):
        result = resolve_model("HIGH")
        assert "description" in result
        assert len(result["description"]) > 0

    def test_result_has_alternatives(self):
        result = resolve_model("LOW", "openrouter")
        assert "alternatives" in result
        assert isinstance(result["alternatives"], list)

    def test_no_provider_returns_all_providers(self):
        result = resolve_model("MEDIUM")
        assert "all_providers" in result
        assert len(result["all_providers"]) >= 1


class TestFallbackChain:
    """Test the fallback/recommendation logic."""

    def test_first_model_is_primary(self):
        result = resolve_model("LOW", "openrouter")
        primary = result["model"]
        alternatives = result.get("alternatives", [])
        # Primary should differ from alternatives
        assert primary not in alternatives or len(alternatives) == 0

    def test_high_tier_has_premium_models(self):
        result = resolve_model("HIGH")
        model = result["model"].lower()
        # High tier should use premium models
        premium_keywords = ["claude", "sonnet", "gemini-2.5-pro", "o3", "opus"]
        assert any(kw in model for kw in premium_keywords), (
            f"High tier model '{model}' doesn't look premium"
        )

    def test_low_tier_has_cheap_models(self):
        result = resolve_model("LOW")
        model = result["model"].lower()
        cheap_keywords = ["flash", "mini", "glm-4"]
        assert any(kw in model for kw in cheap_keywords), (
            f"Low tier model '{model}' doesn't look cheap"
        )


class TestCLI:
    """Test the command-line interface."""

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        script = SCRIPT_DIR / "route_model.py"
        return subprocess.run(
            [sys.executable, str(script), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_list_tiers(self):
        proc = self._run(["--list-tiers"])
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert "LOW" in data
        assert "HIGH" in data

    def test_tier_flag(self):
        proc = self._run(["--tier", "HIGH"])
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert data["tier"] == "HIGH"

    def test_tier_with_provider(self):
        proc = self._run(["--tier", "LOW", "--provider", "google"])
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert data["provider"] == "google"

    def test_invalid_tier_exits_nonzero(self):
        proc = self._run(["--tier", "INVALID"])
        assert proc.returncode != 0

    def test_no_args_shows_error(self):
        proc = self._run([])
        assert proc.returncode != 0

    def test_output_is_valid_json(self):
        proc = self._run(["--tier", "MEDIUM"])
        data = json.loads(proc.stdout)
        assert isinstance(data, dict)
