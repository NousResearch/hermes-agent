"""Tests for gh-copilot CLI deprecation detection and GitHub Models Azure URL mapping."""

import pytest

from agent.copilot_acp_client import _is_gh_copilot_deprecation_message


class TestDeprecationPatternDetection:
    """Verify that stderr from the deprecated `gh copilot` extension is caught
    without false-positiving on the new `@github/copilot` CLI."""

    _REAL_DEPRECATION_STDERR = (
        "The gh-copilot extension has been deprecated in favor of the newer "
        "GitHub Copilot CLI.\nFor more information, visit:\n"
        "- Copilot CLI: https://github.com/github/copilot-cli\n"
        "- Deprecation announcement: https://github.blog/changelog/"
        "2025-09-25-upcoming-deprecation-of-gh-copilot-cli-extension\n"
        "No commands will be executed."
    )

    def test_real_deprecation_message_matches(self):
        assert _is_gh_copilot_deprecation_message(self._REAL_DEPRECATION_STDERR)

    @pytest.mark.parametrize(
        "stderr_text",
        [
            # The deprecation banner uses both halves of the fingerprint.
            "The gh-copilot extension has been deprecated.",
            "gh-copilot: no commands will be executed.",
            # Mixed casing — match is case-insensitive.
            "The GH-Copilot Extension HAS BEEN DEPRECATED.",
        ],
    )
    def test_genuine_deprecation_variants_match(self, stderr_text: str):
        assert _is_gh_copilot_deprecation_message(stderr_text)

    @pytest.mark.parametrize(
        "stderr_text",
        [
            # Generic errors — no fingerprint at all.
            "Error: connection refused",
            "",
            # The NEW @github/copilot CLI's repo is github.com/github/copilot-cli.
            # Its stderr can legitimately mention "copilot-cli" or "deprecation"
            # in unrelated contexts; neither alone should trip the detector.
            "copilot-cli: failed to authenticate with the API",
            "warning: the --foo flag is scheduled for deprecation in v3",
            "See https://github.com/github/copilot-cli/issues for support",
            # Half the fingerprint without the other half.
            "gh-copilot: command not found",
            "extension has been deprecated (some other extension)",
        ],
    )
    def test_does_not_false_positive(self, stderr_text: str):
        assert not _is_gh_copilot_deprecation_message(stderr_text)


class TestGitHubModelsAzureUrl:
    """Verify that the Azure GitHub Models URL is recognised."""

    def test_url_to_provider_contains_azure_models(self):
        from agent.model_metadata import _URL_TO_PROVIDER

        # Maps to the canonical "copilot" provider (same convention as the
        # other GitHub-family entries) — not the "github-models" alias.
        assert _URL_TO_PROVIDER.get("models.inference.ai.azure.com") == "copilot"

    def test_is_github_models_base_url_recognises_azure(self):
        from hermes_cli.models import _is_github_models_base_url

        assert _is_github_models_base_url("https://models.inference.ai.azure.com")
        assert _is_github_models_base_url("https://models.inference.ai.azure.com/v1/chat")

    def test_is_github_models_base_url_still_recognises_github_ai(self):
        from hermes_cli.models import _is_github_models_base_url

        assert _is_github_models_base_url("https://models.github.ai/inference")
