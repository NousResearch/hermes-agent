"""The auto aux lane expands a ``model_aliases:`` key in ``model.default`` like the chat path does.

Regression for #127785: with ``model.default: ds`` naming an alias, chat resolved it but every
auxiliary task on the auto lane sent the literal ``ds`` as the wire model id and the provider
rejected it, so the goal judge failed every turn until the goal loop auto-paused.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as aux

MAIN_URL = "https://api.synthetic.new/openai/v1"
OTHER_URL = "https://alias-host.test/v1"


def _write_config(tmp_path, monkeypatch, alias_base_url):
    import hermes_yaml as yaml

    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "ds", "provider": "custom", "base_url": MAIN_URL,
                  "api_mode": "chat_completions"},
        "model_aliases": {"ds": {"model": "hf:deepseek-ai/DeepSeek-V4.1-Flash", "provider": "custom",
                                 "base_url": alias_base_url, "key_env": "ALIAS_TEST_KEY"}},
    }))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("ALIAS_TEST_KEY", "sk-alias")
    aux.clear_runtime_main()
    aux._aux_unhealthy_until.clear()


def _route_calls():
    calls = []

    def fake_resolve(provider, model, **kwargs):
        calls.append((provider, model, kwargs.get("explicit_base_url"), kwargs.get("explicit_api_key")))
        return MagicMock(name="client"), model

    return calls, patch.object(aux, "resolve_provider_client", side_effect=fake_resolve)


@pytest.mark.parametrize("alias_base_url", [MAIN_URL, OTHER_URL])
def test_config_default_alias_routes_aux_to_the_alias_target(tmp_path, monkeypatch, alias_base_url):
    _write_config(tmp_path, monkeypatch, alias_base_url)
    calls, patched = _route_calls()
    with patched:
        client, model, provider = aux._resolve_auto_route(task="goal_judge")

    assert client is not None
    assert model == "hf:deepseek-ai/DeepSeek-V4.1-Flash"
    assert calls == [("custom", "hf:deepseek-ai/DeepSeek-V4.1-Flash", alias_base_url, "sk-alias")]


def test_live_runtime_model_is_not_reinterpreted_as_an_alias(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, OTHER_URL)
    runtime = {"provider": "custom", "model": "ds", "base_url": MAIN_URL, "api_key": "sk-session"}
    calls, patched = _route_calls()
    with patched:
        aux._resolve_auto_route(main_runtime=runtime, task="goal_judge")

    assert calls == [("custom", "ds", MAIN_URL, "sk-session")]


def test_vision_auto_lane_expands_the_config_default_alias(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, OTHER_URL)
    calls, patched = _route_calls()
    with patched, patch.object(aux, "_main_model_supports_vision", return_value=True):
        provider, client, model = aux._vision_auto_route({}, None, None, False)

    assert client is not None
    assert calls[0] == ("custom", "hf:deepseek-ai/DeepSeek-V4.1-Flash", OTHER_URL, "sk-alias")


def test_main_agent_fallback_expands_the_config_default_alias(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, OTHER_URL)
    calls, patched = _route_calls()
    with patched:
        client, model, label = aux._try_main_agent_model_fallback("openrouter", task="goal_judge")

    assert (model, label) == ("hf:deepseek-ai/DeepSeek-V4.1-Flash", "main-agent(custom)")
    assert calls == [("custom", "hf:deepseek-ai/DeepSeek-V4.1-Flash", OTHER_URL, "sk-alias")]
