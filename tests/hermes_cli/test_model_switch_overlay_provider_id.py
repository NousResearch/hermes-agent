"""Regression for #157658: ``--provider copilot`` saved ``model.provider: github-copilot``.

A plugin profile with a ``provider_overlay`` (the GitHub Copilot plugin: ``name="copilot"``,
``models_dev_id="github-copilot"``) is keyed by its models.dev id for catalog lookups. The switch
persisted that key, which no picker row, Desktop match or auth command knows the provider by.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import providers as _providers
from providers.base import ProviderProfile


@pytest.fixture()
def overlay_provider():
    _providers._discover_providers()
    snapshot = (dict(_providers._REGISTRY), dict(_providers._ALIASES), _providers._PROVIDER_LIST_CACHE)
    _providers.register_provider(ProviderProfile(
        name="zz-copilot", models_dev_id="zz-github-copilot", aliases=("zz-github-copilot",),
        base_url="http://127.0.0.1:9/v1", env_vars=("ZZ_COPILOT_TOKEN",),
        provider_overlay={"transport": "openai_chat", "auth_type": "api_key",
                          "extra_env_vars": ("ZZ_COPILOT_TOKEN",), "base_url_override": "http://127.0.0.1:9/v1"}))
    yield "zz-copilot"
    _providers._REGISTRY.clear()
    _providers._REGISTRY.update(snapshot[0])
    _providers._ALIASES.clear()
    _providers._ALIASES.update(snapshot[1])
    _providers._PROVIDER_LIST_CACHE = snapshot[2]


def _switch(provider):
    from hermes_cli.model_switch import switch_model

    return switch_model("gpt-x", current_provider=provider, current_model="gpt-x", explicit_provider=provider)


def test_switching_to_an_overlay_provider_saves_its_own_id(overlay_provider):
    (Path(os.environ["HERMES_HOME"]) / ".env").write_text("ZZ_COPILOT_TOKEN=test-token\n")
    from hermes_cli.model_switch import model_selection_config_updates

    for requested in (overlay_provider, "zz-github-copilot"):
        result = _switch(requested)
        assert result.success, result.error_message
        assert result.target_provider == "zz-copilot"
        assert model_selection_config_updates(result, {"provider": overlay_provider})["provider"] == "zz-copilot"


def test_the_not_connected_hint_names_the_provider_id_auth_knows(overlay_provider):
    result = _switch(overlay_provider)

    assert not result.success
    assert "hermes auth add zz-copilot`" in result.error_message
