"""Regression test for _resolve_task_provider_model consulting ProviderProfile.api_mode.

When an explicit provider is given without a task-level api_mode override, the
resolver must fall back to the provider profile's declared api_mode so plugin
providers whose upstream speaks the Anthropic Messages API are wrapped with the
correct transport regardless of base-URL shape. Task config still wins.
"""

from types import SimpleNamespace
from unittest.mock import patch

import agent.auxiliary_client as ac


def test_resolver_falls_back_to_profile_api_mode():
    prof = SimpleNamespace(api_mode="anthropic_messages")
    with patch("providers.get_provider_profile", return_value=prof):
        provider, model, base_url, api_key, api_mode = ac._resolve_task_provider_model(
            provider="myplugin"
        )
    assert provider == "myplugin"
    assert api_mode == "anthropic_messages"


def test_task_config_api_mode_still_wins_over_profile():
    prof = SimpleNamespace(api_mode="anthropic_messages")
    with patch("providers.get_provider_profile", return_value=prof), \
         patch.object(ac, "_get_auxiliary_task_config",
                      return_value={"provider": "myplugin", "api_mode": "chat_completions"}):
        # Explicit provider arg with explicit api_mode override would be cfg-driven;
        # here we pass provider via task config so cfg_api_mode is set.
        provider, model, base_url, api_key, api_mode = ac._resolve_task_provider_model(
            task="compression"
        )
    # cfg_api_mode set -> resolver must NOT overwrite it with the profile value.
    assert api_mode == "chat_completions"


def test_resolver_no_profile_api_mode_leaves_none():
    prof = SimpleNamespace(api_mode=None)
    with patch("providers.get_provider_profile", return_value=prof):
        provider, model, base_url, api_key, api_mode = ac._resolve_task_provider_model(
            provider="plainplugin"
        )
    assert api_mode is None
