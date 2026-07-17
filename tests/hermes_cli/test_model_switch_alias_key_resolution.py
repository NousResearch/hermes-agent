"""Regression tests: a main model switched to a ``model_aliases:`` entry
(``DirectAlias``) with a custom ``base_url`` must send the alias's own API key,
not the ``no-key-required`` placeholder.

Repro: switching the main model to an alias whose ``base_url`` points at an
auth-required proxy (e.g. a LiteLLM gateway) stamped ``api_key`` with the
hard-coded ``no-key-required`` placeholder because ``DirectAlias`` carried no
credential field — so the very first request 401'd. The fix gives
``DirectAlias`` optional ``api_key`` / ``key_env`` fields and resolves them in
``switch_model`` before falling back to the placeholder, mirroring the
named-custom-provider key resolution in
``auxiliary_client.resolve_provider_client``.

Hermetic: the resolution chain is fully mocked (no network), mirroring
``tests/hermes_cli/test_model_switch_configured_provider_routing.py``.
"""

import os
from unittest.mock import patch

from hermes_cli.model_switch import DirectAlias, switch_model

_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}

_PROXY_URL = "https://litellm.proxy.example/v1"


def _switch_to_alias(alias, *, env=None):
    """Drive ``switch_model`` for a raw input that resolves to *alias*
    (a ``custom``-provider ``DirectAlias`` registered under ``"proxyalias"``).

    The raw input resolves to the ``custom`` provider with a current base_url
    set and an empty current api_key, so credential resolution takes the
    ``custom`` + current-base_url branch (no network/runtime lookup) and leaves
    ``api_key`` empty — making the DirectAlias key-resolution branch the only
    thing that can supply a key.
    """
    aliases = {"proxyalias": alias}
    with patch("hermes_cli.model_switch.resolve_alias",
               return_value=(alias.provider, alias.model, "proxyalias")), \
         patch("hermes_cli.model_switch._ensure_direct_aliases"), \
         patch.dict("hermes_cli.model_switch.DIRECT_ALIASES", aliases, clear=True), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("hermes_cli.model_switch.normalize_model_for_provider",
               side_effect=lambda model, provider: model), \
         patch("hermes_cli.model_switch.determine_api_mode", return_value=""), \
         patch("hermes_cli.models.validate_requested_model", return_value=_ACCEPTED), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch.dict(os.environ, env or {}, clear=False):
        return switch_model(
            raw_input="proxyalias",
            current_provider="openai",
            current_model="gpt-4o",
            current_base_url="http://previous-endpoint.example/v1",
            current_api_key="",
            user_providers={},
            custom_providers=[],
        )


def test_alias_key_env_is_resolved_not_placeholder():
    """``key_env:`` on the alias → read the env var, not ``no-key-required``."""
    alias = DirectAlias(
        model="qwen3.5:397b", provider="custom",
        base_url=_PROXY_URL, key_env="LITELLM_KEY",
    )
    result = _switch_to_alias(alias, env={"LITELLM_KEY": "sk-secret-123"})
    assert result.success is True, result.error_message
    assert result.base_url == _PROXY_URL
    assert result.api_key == "sk-secret-123"


def test_alias_inline_api_key_is_used():
    """Inline ``api_key:`` on the alias is used verbatim."""
    alias = DirectAlias(
        model="qwen3.5:397b", provider="custom",
        base_url=_PROXY_URL, api_key="sk-inline-456",
    )
    result = _switch_to_alias(alias)
    assert result.success is True, result.error_message
    assert result.api_key == "sk-inline-456"


def test_alias_without_key_still_falls_back_to_placeholder():
    """No key anywhere → the no-auth placeholder is preserved (local servers)."""
    alias = DirectAlias(
        model="llama3.4", provider="custom",
        base_url="http://localhost:11434/v1",
    )
    result = _switch_to_alias(alias)
    assert result.success is True, result.error_message
    assert result.api_key == "no-key-required"
