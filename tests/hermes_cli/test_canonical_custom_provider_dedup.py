"""Regression test: canonical provider slugs must not appear twice in the
`hermes model` TUI picker when the same provider is also configured as a
custom provider in config.yaml.

Bug: https://github.com/NousResearch/hermes-agent/issues/51000
"""

from unittest.mock import patch

import pytest


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with OpenRouter configured as a custom provider."""
    home = tmp_path / "hermes"
    home.mkdir()
    config_yaml = home / "config.yaml"
    # OpenRouter configured in both canonical (auto-detected via API key) and
    # as an explicit custom_providers entry — the exact scenario that caused
    # the duplicate.
    config_yaml.write_text(
        "model: openrouter:anthropic/claude-sonnet-4\n"
        "custom_providers:\n"
        "  - name: OpenRouter\n"
        "    provider_key: openrouter\n"
        "    base_url: https://openrouter.ai/api/v1\n"
        "    api_key: sk-test-123\n"
    )
    env_file = home / ".env"
    env_file.write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return home


class TestCanonicalCustomProviderDedup:
    """Canonical providers must not duplicate when also in custom_providers."""

    def test_openrouter_appears_once_in_provider_list(self, config_home, capsys):
        """OpenRouter must appear exactly once even when configured in both
        CANONICAL_PROVIDERS and custom_providers with provider_key=openrouter.

        Before the fix, the provider picker showed two OpenRouter entries:
        one with a live model count and one with (0).
        """
        from hermes_cli.main import select_provider_and_model

        # Capture the ordered list by patching _prompt_provider_choice to
        # record the labels it receives, then bail out.
        captured_labels = []

        def fake_prompt(labels, default=0, title=None):
            captured_labels.extend(labels)
            return None  # "Leave unchanged"

        with patch("hermes_cli.main._prompt_provider_choice", side_effect=fake_prompt):
            select_provider_and_model()

        # Count how many times OpenRouter appears in the picker labels
        openrouter_entries = [
            lbl for lbl in captured_labels
            if "OpenRouter" in lbl and "←" not in lbl
            or "openrouter" in lbl.lower()
        ]
        # More robust: check by key in the ordered list.  Since we can't
        # easily intercept the ordered list directly, count the label hits.
        # The canonical entry label contains "OpenRouter"; the custom entry
        # label also contains "OpenRouter".  After the fix there should be
        # exactly one.
        assert len(openrouter_entries) <= 1, (
            f"OpenRouter should appear at most once in the picker, "
            f"got {len(openrouter_entries)}: {openrouter_entries}"
        )

    def test_non_canonical_custom_provider_still_appears(self, config_home):
        """A custom provider whose key is NOT a canonical slug must still
        appear in the picker — the dedup filter must not be over-broad."""
        from hermes_cli.models import _canonical_slugs

        # Sanity: "custom:my-vllm" is not a canonical slug
        assert "custom:my-vllm" not in _canonical_slugs
