"""Test that opencode-go appears in /model list when credentials are set."""

import os
from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers


# Minimum set of models that must be present for opencode-go no matter
# whether the picker sourced its list from curated-only or curated+models.dev.
# The curated list in hermes_cli/models.py defines the floor; models.dev only
# ever adds names on top of it via _merge_with_models_dev.
_OPENCODE_GO_REQUIRED = {
    "kimi-k2.6",
    "kimi-k2.5",
    "glm-5.1",
    "glm-5",
    "mimo-v2-pro",
    "mimo-v2-omni",
    "minimax-m2.7",
    "minimax-m2.5",
}


@patch.dict(os.environ, {"OPENCODE_GO_API_KEY": "test-key"}, clear=False)
def test_opencode_go_appears_when_api_key_set():
    """opencode-go should appear in list_authenticated_providers when OPENCODE_GO_API_KEY is set."""
    providers = list_authenticated_providers(
        current_provider="openrouter", max_models=50
    )

    # Find opencode-go in results
    opencode_go = next((p for p in providers if p["slug"] == "opencode-go"), None)

    assert opencode_go is not None, (
        "opencode-go should appear when OPENCODE_GO_API_KEY is set"
    )
    # Behavior check: the curated floor must be present. The list may also
    # include extra models.dev entries (e.g. mimo-v2.5-pro) when the registry
    # is reachable — that's the whole point of the models.dev-preferred merge
    # introduced for opencode-go, so don't pin to an exact list here.
    present = set(opencode_go["models"])
    missing = _OPENCODE_GO_REQUIRED - present
    assert not missing, (
        f"opencode-go picker should include the curated floor; missing: {sorted(missing)}. "
        f"Got: {opencode_go['models']}"
    )
    # opencode-go can appear as "built-in" (from PROVIDER_TO_MODELS_DEV when
    # models.dev is reachable) or "hermes" (from HERMES_OVERLAYS fallback when
    # the API is unavailable, e.g. in CI).
    assert opencode_go["source"] in {"built-in", "hermes"}


def test_opencode_go_not_appears_when_no_creds():
    """opencode-go should NOT appear when no credentials are set."""
    # Ensure OPENCODE_GO_API_KEY is not set
    env_without_key = {
        k: v for k, v in os.environ.items() if k != "OPENCODE_GO_API_KEY"
    }

    with patch.dict(os.environ, env_without_key, clear=True):
        providers = list_authenticated_providers(current_provider="openrouter")

        # opencode-go should not be in results
        opencode_go = next((p for p in providers if p["slug"] == "opencode-go"), None)
        assert opencode_go is None, "opencode-go should not appear without credentials"


def test_opencode_go_resolver_prefers_canonical_overlay_url(monkeypatch):
    """resolve_provider_full must return the canonical overlay URL even when
    models.dev advertises a stripped API URL.

    Regression guard for the ``base_url_override`` on the opencode-go
    HermesOverlay — without it, ``get_provider`` falls through to
    ``mdev_info.api`` (the raw models.dev catalog URL), which may lack the
    ``/v1`` suffix and break every caller on the pre-runtime resolver path.
    """
    from agent.models_dev import ProviderInfo
    from hermes_cli.providers import resolve_provider_full

    mocked = ProviderInfo(
        id="opencode-go",
        name="OpenCode Go",
        env=("OPENCODE_GO_API_KEY",),
        api="https://opencode.ai/zen/go",  # deliberately stripped — no /v1
        doc="",
    )

    monkeypatch.setattr(
        "agent.models_dev.get_provider_info",
        lambda provider_id: mocked if provider_id == "opencode-go" else None,
    )

    resolved = resolve_provider_full("opencode-go")

    assert resolved is not None, "resolve_provider_full returned None for opencode-go"
    assert resolved.id == "opencode-go", f"expected opencode-go, got {resolved.id!r}"
    assert resolved.base_url == "https://opencode.ai/zen/go/v1", (
        f"expected canonical overlay URL, got {resolved.base_url!r}"
    )

    # The rest of the overlay contract must also be intact.
    assert resolved.transport == "openai_chat", f"transport was {resolved.transport!r}"
    assert resolved.is_aggregator is True, (
        f"is_aggregator was {resolved.is_aggregator!r}"
    )
    assert resolved.base_url_env_var == "OPENCODE_GO_BASE_URL", (
        f"base_url_env_var was {resolved.base_url_env_var!r}"
    )
