"""Tests for gh-copilot CLI deprecation detection and GitHub Models Azure URL mapping."""

import pytest

from agent.copilot_acp_client import _DEPRECATION_PATTERNS


class TestDeprecationPatternDetection:
    """Verify that stderr messages from a deprecated gh-copilot CLI are caught."""

    _REAL_DEPRECATION_STDERR = (
        "The gh-copilot extension has been deprecated in favor of the newer "
        "GitHub Copilot CLI.\nFor more information, visit:\n"
        "- Copilot CLI: https://github.com/github/copilot-cli\n"
        "- Deprecation announcement: https://github.blog/changelog/"
        "2025-09-25-upcoming-deprecation-of-gh-copilot-cli-extension\n"
        "No commands will be executed."
    )

    def test_real_deprecation_message_matches(self):
        lower = self._REAL_DEPRECATION_STDERR.lower()
        assert any(pat in lower for pat in _DEPRECATION_PATTERNS)

    @pytest.mark.parametrize(
        "stderr_line",
        [
            "The gh-copilot extension has been deprecated",
            "No commands will be executed.",
            "See deprecation notice at ...",
            "Install copilot-cli instead",
        ],
    )
    def test_individual_patterns_match(self, stderr_line: str):
        lower = stderr_line.lower()
        assert any(pat in lower for pat in _DEPRECATION_PATTERNS)

    def test_normal_stderr_does_not_match(self):
        normal = "Error: connection refused"
        assert not any(pat in normal.lower() for pat in _DEPRECATION_PATTERNS)


class TestGitHubModelsAzureUrl:
    """Verify that the Azure GitHub Models URL is recognised."""

    def test_url_to_provider_contains_azure_models(self):
        from agent.model_metadata import _URL_TO_PROVIDER

        assert _URL_TO_PROVIDER.get("models.inference.ai.azure.com") == "github-models"

    def test_is_github_models_base_url_recognises_azure(self):
        from hermes_cli.models import _is_github_models_base_url

        assert _is_github_models_base_url("https://models.inference.ai.azure.com")
        assert _is_github_models_base_url("https://models.inference.ai.azure.com/v1/chat")

    def test_is_github_models_base_url_still_recognises_github_ai(self):
        from hermes_cli.models import _is_github_models_base_url

        assert _is_github_models_base_url("https://models.github.ai/inference")
