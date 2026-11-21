"""Picking a managed llama.cpp model must not pin the managed server's address into config.yaml.

The managed server's port is ephemeral (a busy stable port falls back to a free one; a restart can
move it) and its API key is known only to the supervisor. ``model.base_url`` pinned to it makes
``provider: llamacpp`` bypass the managed endpoint resolver, so later requests hit a dead port
(connection refused) or the live router without its key (401).
"""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.model_switch import ModelSwitchResult, model_selection_config_updates

_MANAGED = {"base_url": "http://127.0.0.1:53918/v1", "api_key": "managed-key"}


def _result(base_url, provider="llamacpp"):
    return ModelSwitchResult(
        success=True, new_model="Ornith-1.5-35B-Q4_K_M", target_provider=provider,
        base_url=base_url, api_mode="chat_completions", is_global=True,
    )


def _updates(result, current=None):
    with patch("hermes_cli.local_runtime.endpoint._state_endpoint", return_value=_MANAGED):
        return model_selection_config_updates(result, current or {})


def test_managed_server_address_is_not_persisted():
    assert _updates(_result(_MANAGED["base_url"]))["base_url"] is None


def test_stale_managed_pin_is_cleared_on_the_next_pick():
    current = {"default": "old", "provider": "llamacpp", "base_url": "http://127.0.0.1:49982/v1"}
    assert _updates(_result(_MANAGED["base_url"]), current)["base_url"] is None


def test_external_llama_server_url_is_kept():
    """A user's own llama-server (not the managed one) is a real route and is stored verbatim."""
    lan = "http://192.168.1.20:8080/v1"
    assert _updates(_result(lan))["base_url"] == lan


def test_repicking_the_same_managed_model_keeps_a_context_pin():
    """With the address unset in config, the same managed route must not read as "route changed"
    (which would drop the user's context_length override on every re-pick)."""
    current = {"default": "Ornith-1.5-35B-Q4_K_M", "provider": "llamacpp", "context_length": 65536}
    assert "context_length" not in _updates(_result(_MANAGED["base_url"]), current)
