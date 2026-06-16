"""Regression tests: overlay extra_env_vars must match PROVIDER_REGISTRY.

Issue #47361: 18 HermesOverlay entries with auth_type="api_key" had empty
extra_env_vars, causing credential detection drift in
list_authenticated_providers() (model_switch.py Part 2).
"""

import pytest

# Providers that had missing extra_env_vars — these get their env vars from
# PROVIDER_REGISTRY, not models.dev.  Providers like openrouter and openai-api
# get env vars from models.dev, so empty overlay extra_env_vars is correct.
_AFFECTED_PROVIDERS = {
    "ollama-cloud": ("OLLAMA_API_KEY",),
    "kimi-for-coding": ("KIMI_API_KEY", "KIMI_CODING_API_KEY"),
    "minimax": ("MINIMAX_API_KEY",),
    "minimax-cn": ("MINIMAX_CN_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "alibaba": ("DASHSCOPE_API_KEY",),
    "alibaba-coding-plan": ("ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY"),
    "opencode": ("OPENCODE_ZEN_API_KEY",),
    "opencode-go": ("OPENCODE_GO_API_KEY",),
    "kilo": ("KILOCODE_API_KEY",),
    "huggingface": ("HF_TOKEN",),
    "novita": ("NOVITA_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "nvidia": ("NVIDIA_API_KEY",),
    "xiaomi": ("XIAOMI_API_KEY",),
    "tencent-tokenhub": ("TOKENHUB_API_KEY",),
    "arcee": ("ARCEEAI_API_KEY",),
    "azure-foundry": ("AZURE_FOUNDRY_API_KEY",),
}


class TestOverlayExtraEnvVars:
    """Overlay extra_env_vars must be populated for providers with PROVIDER_REGISTRY entries."""

    @pytest.mark.parametrize("provider", sorted(_AFFECTED_PROVIDERS.keys()))
    def test_overlay_has_extra_env_vars(self, provider):
        """Regression: overlay must declare extra_env_vars for credential detection."""
        from hermes_cli.providers import HERMES_OVERLAYS

        overlay = HERMES_OVERLAYS[provider]
        assert overlay.extra_env_vars, (
            f"Overlay '{provider}' has empty extra_env_vars — "
            "list_authenticated_providers() won't detect its credentials."
        )

    @pytest.mark.parametrize(
        "provider,expected",
        sorted(_AFFECTED_PROVIDERS.items()),
    )
    def test_overlay_env_vars_match_registry(self, provider, expected):
        """Overlay extra_env_vars must match PROVIDER_REGISTRY api_key_env_vars."""
        from hermes_cli.providers import HERMES_OVERLAYS

        overlay = HERMES_OVERLAYS[provider]
        assert overlay.extra_env_vars == expected, (
            f"Overlay '{provider}' extra_env_vars={overlay.extra_env_vars} "
            f"doesn't match expected {expected}"
        )
