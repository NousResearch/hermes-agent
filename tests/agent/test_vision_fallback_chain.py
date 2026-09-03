"""Test that the vision path in _call_llm_impl consults the configured fallback_chain.

Two scenarios:
1. When resolve_vision_provider_client returns no client for the configured
   provider, the vision branch in _call_llm_impl should walk
   auxiliary.vision.fallback_chain (per-task entries) before raising.
   Previously the vision branch skipped the fallback_chain entirely.

2. When auxiliary.vision.fallback_chain is empty/absent but the top-level
   fallback_providers config list is populated, the vision path falls
   through to it — proving a previously-dead config key now has a live
   consumer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_client(model: str = "test-model") -> MagicMock:
    """Return a bare-minimum MagicMock that passes _effective_provider_for_client."""
    client = MagicMock()
    client.chat = MagicMock()
    client.is_vision_capable = True
    return client


# ── Step 1: fallback_chain fires when the configured provider fails ──────────


def test_vision_fallback_chain_fires_on_provider_failure() -> None:
    """When the configured vision provider returns no client, the vision branch
    in _call_llm_impl should walk auxiliary.vision.fallback_chain (step 2 of
    _resolve_vision_with_fallback) before raising."""
    from agent.auxiliary_client import _resolve_vision_with_fallback

    fb_client = _make_client("fallback-model")

    with patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        return_value=("omniroute", None, None),  # Step 1: fails
    ), patch(
        "agent.auxiliary_client._try_configured_fallback_chain",
        return_value=(fb_client, "fallback-model", "fallback_chain[0](gemini)"),
    ):
        effective, client, model = _resolve_vision_with_fallback(
            resolved_provider="omniroute",
            resolved_model="gemini-flash",
            resolved_base_url="http://localhost:20128/v1",
            resolved_api_key="bad-key",
            main_runtime=None,
        )

    assert client is fb_client
    assert effective == "fallback_chain[0](gemini)"
    assert model == "fallback-model"


def test_vision_fallback_chain_skips_exhausted_entries() -> None:
    """If every entry in auxiliary.vision.fallback_chain fails to build a
    client, the function falls through to the top-level fallback_providers
    (step 3) and then the auto-backend (step 4) — it does not raise."""
    from agent.auxiliary_client import _resolve_vision_with_fallback

    with patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        return_value=("omniroute", None, None),  # Step 1: fails
    ), patch(
        "agent.auxiliary_client._try_configured_fallback_chain",
        return_value=(None, None, ""),  # Step 2: chain empty/exhausted
    ), patch(
        "hermes_cli.fallback_config.get_fallback_chain",
        return_value=[
            {"provider": "gemini", "model": "gemini-flash-latest"},
            {"provider": "nvidia", "model": "z-ai/glm-5.2"},
        ],
    ), patch(
        "hermes_cli.config.load_config_readonly",
        return_value={},
    ), patch(
        "agent.auxiliary_client._resolve_fallback_entry",
        side_effect=[(None, None), (_make_client("topfall-model"), "topfall-model")],
    ):
        effective, client, model = _resolve_vision_with_fallback(
            resolved_provider="omniroute",
            resolved_model="gemini-flash",
            resolved_base_url=None,
            resolved_api_key=None,
            main_runtime=None,
        )

    # Top-level fallback_providers entry at index 1 (index 0 already failed).
    assert client is not None
    assert model == "topfall-model"
    assert effective.startswith("fallback_providers[1]")


def test_vision_fallback_chain_raises_when_everything_exhausted() -> None:
    """When every fallback tier is empty/failed, the function returns (label,
    None, None) and _call_llm_impl raises the expected RuntimeError."""
    from agent.auxiliary_client import _resolve_vision_with_fallback

    with patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        return_value=("omniroute", None, None),
    ), patch(
        "agent.auxiliary_client._try_configured_fallback_chain",
        return_value=(None, None, ""),
    ), patch(
        "hermes_cli.fallback_config.get_fallback_chain",
        return_value=[],  # top-level fallback_providers empty
    ), patch(
        "hermes_cli.config.load_config_readonly",
        return_value={},
    ), patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        side_effect=[("omniroute", None, None), ("auto", None, None)],
    ):
        effective, client, model = _resolve_vision_with_fallback(
            resolved_provider="omniroute",
            resolved_model="gemini-flash",
            resolved_base_url=None,
            resolved_api_key=None,
            main_runtime=None,
        )

    assert client is None
    assert model is None
    # effective_provider should still be set so the error message is readable.
    assert effective == "vision"


# ── Step 2: top-level fallback_providers is consumed (was dead config) ───────


def test_vision_top_level_fallback_providers_consumed() -> None:
    """When auxiliary.vision.fallback_chain is empty AND the top-level
    fallback_providers list has a working entry, the vision path uses it.
    This proves the previously-dead config key now has a live consumer."""
    from agent.auxiliary_client import _resolve_vision_with_fallback

    fb_client = _make_client("nvidia-model")

    with patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        return_value=("omniroute", None, None),  # Step 1: fails
    ), patch(
        "agent.auxiliary_client._try_configured_fallback_chain",
        return_value=(None, None, ""),  # Step 2: chain empty/exhausted
    ), patch(
        "hermes_cli.fallback_config.get_fallback_chain",
        return_value=[
            {"provider": "gemini", "model": "gemini-flash-latest"},
            {"provider": "nvidia", "model": "z-ai/glm-5.2"},
        ],
    ), patch(
        "hermes_cli.config.load_config_readonly",
        return_value={},
    ), patch(
        "agent.auxiliary_client._resolve_fallback_entry",
        side_effect=[(None, None), (fb_client, "nvidia-model")],
    ):
        effective, client, model = _resolve_vision_with_fallback(
            resolved_provider="omniroute",
            resolved_model="gemini-flash",
            resolved_base_url=None,
            resolved_api_key=None,
            main_runtime=None,
        )

    # Top-level fallback_providers entry at index 1 (index 0 already failed).
    assert client is fb_client
    assert model == "nvidia-model"
    assert effective.startswith("fallback_providers[1]")


# ── Step 3: auto-backend only fires when no earlier tier succeeded ───────────


def test_vision_auto_backend_is_step_4_not_step_2() -> None:
    """When the explicit provider fails and the chains are empty, the
    auto-backend (step 4) is tried LAST, not as step 2."""
    from agent.auxiliary_client import _resolve_vision_with_fallback

    auto_client = _make_client("or-model")

    with patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        side_effect=[
            ("omniroute", None, None),  # Step 1: explicit provider fails
            ("auto", auto_client, "auto/multimodal"),  # Step 4: auto succeeds
        ]
    ), patch(
        "agent.auxiliary_client._try_configured_fallback_chain",
        return_value=(None, None, ""),
    ), patch(
        "hermes_cli.fallback_config.get_fallback_chain",
        return_value=[],
    ), patch(
        "hermes_cli.config.load_config_readonly",
        return_value={},
    ):
        effective, client, model = _resolve_vision_with_fallback(
            resolved_provider="omniroute",
            resolved_model="gemini-flash",
            resolved_base_url=None,
            resolved_api_key=None,
            main_runtime=None,
        )

    assert client is auto_client
    assert effective == "auto"
    assert client is not None
