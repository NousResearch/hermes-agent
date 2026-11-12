"""Every writer that re-points an ``auxiliary.<task>`` block drops the old route's credentials (#141832).

The aux resolver (``_resolve_task_provider_model``) hands a block's ``api_key``/``key_env`` to
whatever provider/base_url the block names, so a key left behind by any writer (``hermes model``
reset, the setup-wizard vision picker, the dashboard assignment) is sent to the next endpoint.
Each test drives the real writer and asserts what the resolver would send.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import hermes_yaml as yaml

OLD_SECRET = "old-endpoint-secret"


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OLD_ENDPOINT_KEY", OLD_SECRET)
    return hermes_home


def _seed(home, vision: dict) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump({"auxiliary": {"vision": vision}}))


def _resolved_vision():
    from agent.auxiliary_client import _resolve_task_provider_model
    return _resolve_task_provider_model("vision")


OPENROUTER_ROUTE = {"provider": "openrouter", "model": "m", "key_env": "OLD_ENDPOINT_KEY", "api_mode": "codex_responses"}


def _wizard_auto():
    import hermes_cli.tools_config as tc
    with patch.object(tc, "_prompt_choice", return_value=0):
        tc._configure_vision_backend()


def _hermes_model_reset():
    from hermes_cli.main_provider_setup import _reset_aux_to_auto
    _reset_aux_to_auto()


@pytest.mark.parametrize("reset_to_auto", [_wizard_auto, _hermes_model_reset], ids=["wizard_auto", "hermes_model_reset"])
def test_auto_reset_then_provider_set_sends_no_old_key(home, reset_to_auto):
    from hermes_cli.config import set_config_value
    # No api_mode: nothing route-shaped is left for the set-time guard to pop after the reset.
    _seed(home, {"provider": "openrouter", "model": "m", "key_env": "OLD_ENDPOINT_KEY"})

    reset_to_auto()
    set_config_value("auxiliary.vision.provider", "anthropic")

    provider, _model, base_url, api_key, api_mode = _resolved_vision()
    assert (provider, base_url, api_key, api_mode) == ("anthropic", None, None, None)


def test_wizard_vision_picker_switch_drops_old_credentials(home):
    from hermes_cli import inventory
    import hermes_cli.tools_config as tc
    _seed(home, OPENROUTER_ROUTE)
    rows = [{"slug": "anthropic", "name": "Anthropic", "models": ["claude-x"]}]
    choices = iter([1, 0, 0])  # "Pick a provider and model" → Anthropic → claude-x
    with patch.object(tc, "_prompt_choice", side_effect=lambda *a, **k: next(choices)), \
         patch.object(inventory, "build_aux_picker_rows", return_value=rows), \
         patch.object(inventory, "format_aux_picker_entries", return_value=[("anthropic", "Anthropic", [])]):
        tc._configure_vision_backend()

    assert _resolved_vision() == ("anthropic", "claude-x", None, None, None)


@pytest.mark.parametrize("seed", [
    OPENROUTER_ROUTE,
    {"provider": "custom", "model": "m", "base_url": "https://a.example/v1", "api_key": OLD_SECRET,
     "api_mode": "codex_responses"},
], ids=["openrouter_key_env", "custom_inline_key"])
def test_dashboard_new_base_url_sends_no_old_key(home, seed):
    from hermes_cli.config import load_config
    from hermes_cli.web_server_config import _apply_aux_assignment_sync
    _seed(home, seed)

    _apply_aux_assignment_sync(load_config(), "custom", "m", "vision", "https://b.example/v1", "")

    _provider, _model, base_url, api_key, api_mode = _resolved_vision()
    assert base_url == "https://b.example/v1"
    assert api_key != OLD_SECRET and api_mode is None


def test_dashboard_same_route_keeps_its_key(home):
    from hermes_cli.config import load_config
    from hermes_cli.web_server_config import _apply_aux_assignment_sync
    _seed(home, {"provider": "custom", "model": "m", "base_url": "https://a.example/v1", "api_key": "a-key"})

    _apply_aux_assignment_sync(load_config(), "custom", "m2", "vision", "https://a.example/v1", "")

    assert _resolved_vision()[2:4] == ("https://a.example/v1", "a-key")


@pytest.mark.parametrize("seed", [
    OPENROUTER_ROUTE,
    {"provider": "custom", "model": "m", "base_url": "https://a.example/v1", "api_key": OLD_SECRET,
     "api_mode": "codex_responses"},
], ids=["openrouter_key_env", "custom_inline_key"])
def test_wizard_custom_endpoint_sends_no_old_key(home, seed):
    import hermes_cli.tools_config as tc
    import hermes_cli.tools_config_providers as tcp
    _seed(home, seed)
    answers = iter(["https://b.example/v1", "new-key", "m"])
    with patch.object(tc, "_prompt_choice", return_value=2), \
         patch.object(tcp, "_prompt", side_effect=lambda *a, **k: next(answers)):
        tc._configure_vision_backend()

    assert _resolved_vision() == ("custom", "m", "https://b.example/v1", None, None)


def _put_config(config: dict) -> None:
    import asyncio
    from hermes_cli.web_routers.config_env import ConfigUpdate, update_config
    asyncio.run(update_config(ConfigUpdate(config=config)))


@pytest.mark.parametrize("seed", [
    {"provider": "openrouter", "key_env": "OLD_ENDPOINT_KEY"},
    {"provider": "openrouter", "key_env": "OLD_ENDPOINT_KEY", "base_url": "https://openrouter.ai/api/v1"},
], ids=["key_env", "key_env_and_base_url"])
def test_dashboard_config_form_provider_change_sends_no_old_key(home, seed):
    _seed(home, seed)

    # The schema form re-sends what GET returned (base_url) and never sends key_env.
    _put_config({"auxiliary": {"vision": {**{k: v for k, v in seed.items() if k != "key_env"},
                                          "provider": "anthropic"}}})

    assert _resolved_vision() == ("anthropic", None, None, None, None)


def test_dashboard_config_form_main_model_provider_change_drops_old_route(home):
    from hermes_cli.config import read_raw_config
    (home / "config.yaml").write_text(yaml.safe_dump({"model": {
        "provider": "custom", "default": "m", "base_url": "https://a.example/v1", "key_env": "OLD_ENDPOINT_KEY"}}))

    _put_config({"model": {"provider": "anthropic"}})

    assert read_raw_config()["model"] == {"provider": "anthropic", "default": "m"}


def test_dashboard_config_form_keeps_same_route_and_new_keys(home):
    _seed(home, {"provider": "custom", "model": "m", "base_url": "https://a.example/v1", "api_key": "a-key"})
    _put_config({"auxiliary": {"vision": {"model": "m2", "base_url": "https://a.example/v1", "api_key": "a-key"}}})
    assert _resolved_vision()[2:4] == ("https://a.example/v1", "a-key")

    _put_config({"auxiliary": {"vision": {"base_url": "https://b.example/v1", "api_key": "b-key"}}})
    assert _resolved_vision()[2:4] == ("https://b.example/v1", "b-key")
