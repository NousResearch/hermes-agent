"""Tests for the cli-shim provider's picker + runtime-resolution wiring.

cli-shim (local claude/codex/gemini CLIs via OAuth, no API key) shipped with
client construction wired up but not the picker UX or the runtime credential
path. These tests pin the two fixes:

* ``resolve_runtime_provider`` resolves cli-shim instead of falling through to
  the API-key fallback (which defaulted to openrouter and bailed with an empty
  API key).
* cli-shim is surfaced in the ``hermes model`` picker (CANONICAL_PROVIDERS)
  despite its ``external_process`` auth_type.
"""

from __future__ import annotations

import sys
import types

import pytest

# Match the other hermes_cli provider tests: stub python-dotenv if absent so
# importing hermes_cli modules never blows up on a missing optional dep.
if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_cli.models import CANONICAL_PROVIDERS
from providers import get_provider_profile


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    # Clear env so the runtime resolver can't accidentally satisfy itself with
    # an unrelated provider key — the cli-shim branch must stand on its own.
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HERMES_INFERENCE_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)


class TestCliShimRuntimeResolution:
    def test_resolve_runtime_provider_returns_cli_shim(self):
        runtime = resolve_runtime_provider(requested="cli-shim")
        assert runtime["provider"] == "cli-shim"
        assert runtime["base_url"] == "cli://shim"
        # Placeholder key — CliShimClient ignores it for routing, but it must be
        # non-empty so the downstream empty-key guard doesn't abort.
        assert runtime["api_key"] == "cli-shim"
        assert runtime["api_mode"] == "chat_completions"

    def test_does_not_fall_back_to_openrouter(self):
        # Regression: before the fix, cli-shim fell through every named branch to
        # the API-key fallback, defaulted resolved_provider to "openrouter", and
        # returned an empty api_key — surfacing "Set OPENROUTER_API_KEY".
        runtime = resolve_runtime_provider(requested="cli-shim")
        assert runtime["provider"] != "openrouter"
        assert runtime["api_key"], "cli-shim must resolve a non-empty (placeholder) api_key"
        assert runtime["base_url"], "cli-shim must resolve a non-empty base_url"


class TestCliShimPickerIntegration:
    def test_cli_shim_in_canonical_providers(self):
        slugs = {p.slug for p in CANONICAL_PROVIDERS}
        assert "cli-shim" in slugs, (
            "cli-shim should be auto-injected into the model picker despite its "
            "external_process auth_type"
        )

    def test_cli_shim_picker_row_has_label_and_description(self):
        row = next(p for p in CANONICAL_PROVIDERS if p.slug == "cli-shim")
        assert row.label, "picker label should not be empty"
        # Description should make clear it is the no-API-key local-CLI path.
        assert "no api key" in row.tui_desc.lower()


class TestCliShimProfile:
    def test_profile_is_external_process(self):
        profile = get_provider_profile("cli-shim")
        assert profile is not None
        assert profile.auth_type == "external_process"

    def test_profile_exposes_cli_model_aliases(self):
        profile = get_provider_profile("cli-shim")
        models = profile.fetch_models()
        assert "claude-sonnet-cli" in models
        assert "claude-opus-cli" in models
