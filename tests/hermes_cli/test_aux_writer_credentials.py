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


def test_dashboard_equivalent_https_url_keeps_its_key(home):
    """A different spelling of the same HTTPS URL must not lose endpoint credentials."""
    from hermes_cli.config import load_config
    from hermes_cli.web_server_config import _apply_aux_assignment_sync
    _seed(home, {"provider": "custom", "model": "m",
                 "base_url": "https://a.example/v1", "api_key": "a-key"})

    # URL hostnames are case-insensitive and :443 is the HTTPS default port.
    _apply_aux_assignment_sync(load_config(), "custom", "m2", "vision",
                               "https://A.EXAMPLE:443/v1/", "")

    assert _resolved_vision() == ("custom", "m2", "https://A.EXAMPLE:443/v1/", "a-key", None)


def test_dashboard_bare_custom_on_same_url_keeps_its_key(home):
    """Picking bare custom keeps the slot's URL (the keeps-old-URL branch), so the key configured
    for that URL stays with it, as `hermes config set ….provider custom` keeps it."""
    from hermes_cli.config import load_config
    from hermes_cli.web_server_config import _apply_aux_assignment_sync
    _seed(home, {"provider": "openrouter", "model": "m", "base_url": "https://openrouter.ai/api/v1", "api_key": "a-key"})

    _apply_aux_assignment_sync(load_config(), "custom", "m2", "vision", "", "")

    assert _resolved_vision() == ("custom", "m2", "https://openrouter.ai/api/v1", "a-key", None)


def test_dashboard_same_provider_model_change_keeps_its_key(home):
    """A model-only change (no base_url sent) keeps the slot's URL, so it is not a route change."""
    from hermes_cli.config import load_config
    from hermes_cli.web_server_config import _apply_aux_assignment_sync
    _seed(home, {"provider": "openrouter", "model": "m", "base_url": "https://openrouter.ai/api/v1", "api_key": "a-key"})

    _apply_aux_assignment_sync(load_config(), "openrouter", "m2", "vision", "", "")

    assert _resolved_vision()[2:4] == ("https://openrouter.ai/api/v1", "a-key")


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


@pytest.mark.parametrize("new_url, kept", [("https://a.example/v2", True), ("http://a.example/v1", False),
                                           ("https://a.example:8443/v1", False)],
                         ids=["same_origin_path", "http_downgrade", "other_port"])
def test_aux_picker_custom_url_binds_key_env_to_origin(home, new_url, kept):
    """``hermes model``'s custom aux endpoint keeps a key only across a path change on one origin (#130410)."""
    from hermes_cli.main_provider_setup import _save_aux_choice
    _seed(home, {"provider": "custom", "model": "m", "base_url": "https://a.example/v1", "key_env": "OLD_ENDPOINT_KEY"})

    _save_aux_choice("vision", provider="custom", model="m", base_url=new_url, api_key="")

    assert _resolved_vision()[2:4] == (new_url, OLD_SECRET if kept else None)


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


def test_dashboard_config_form_carried_mixed_case_url_still_dropped(home):
    """The carried-URL check must use the same URL identity as the route compare: a re-sent
    mixed-case base_url under a new provider is still the old route's and must go with its key."""
    _seed(home, {"provider": "custom", "model": "m", "base_url": "https://A.example/v1", "key_env": "OLD_ENDPOINT_KEY"})

    _put_config({"auxiliary": {"vision": {"provider": "anthropic", "model": "m", "base_url": "https://A.example/v1"}}})

    assert _resolved_vision() == ("anthropic", "m", None, None, None)


def test_dashboard_config_form_owner_of_carried_url_keeps_its_key(home):
    """A provider change to the carried URL's own provider keeps the URL, and its key with it."""
    _seed(home, {"provider": "custom", "model": "m", "base_url": "https://openrouter.ai/api/v1", "api_key": "a-key"})

    _put_config({"auxiliary": {"vision": {"provider": "openrouter", "model": "m", "base_url": "https://openrouter.ai/api/v1"}}})

    assert _resolved_vision()[2:4] == ("https://openrouter.ai/api/v1", "a-key")


def test_dashboard_config_form_round_trip_drops_referenced_key(home):
    """The settings form PUTs back what GET returned, and GET expands a ``${VAR}`` key (aux keys
    are stored as ``${AUXILIARY_<TASK>_API_KEY}``): the re-sent expansion is the old route's key."""
    import asyncio
    from hermes_cli.web_routers.config_env import get_config
    (home / ".env").write_text(f"AUXILIARY_VISION_API_KEY={OLD_SECRET}\n")
    _seed(home, {"provider": "custom", "model": "m", "base_url": "https://a.example/v1",
                 "api_key": "${AUXILIARY_VISION_API_KEY}"})
    form = asyncio.run(get_config())
    form["auxiliary"]["vision"]["provider"] = "anthropic"

    _put_config(form)

    assert _resolved_vision() == ("anthropic", "m", None, None, None)


def test_dashboard_config_form_round_trip_keeps_typed_key(home):
    import asyncio
    from hermes_cli.web_routers.config_env import get_config
    (home / ".env").write_text(f"AUXILIARY_VISION_API_KEY={OLD_SECRET}\n")
    _seed(home, {"provider": "custom", "model": "m", "base_url": "https://a.example/v1",
                 "api_key": "${AUXILIARY_VISION_API_KEY}"})
    form = asyncio.run(get_config())
    form["auxiliary"]["vision"].update(provider="anthropic", api_key="typed-key")

    _put_config(form)

    assert _resolved_vision()[3] == "typed-key"


def _delegation_form_change(home, change: dict) -> None:
    import asyncio
    from hermes_cli.web_routers.config_env import get_config
    (home / ".env").write_text(f"HERMES_DELEGATION_API_KEY={OLD_SECRET}\n")
    (home / "config.yaml").write_text(yaml.safe_dump({"delegation": {
        "provider": "custom", "model": "m", "base_url": "https://a.example/v1",
        "api_key": "${HERMES_DELEGATION_API_KEY}"}}))
    form = asyncio.run(get_config())
    form["delegation"].update(change)
    _put_config(form)


def test_dashboard_config_form_delegation_provider_change_drops_old_key(home):
    """``delegation`` sits in the same form; its resolver posts ``api_key`` to ``base_url``."""
    from hermes_cli.config import read_raw_config
    _delegation_form_change(home, {"provider": "anthropic"})

    assert not read_raw_config()["delegation"].get("api_key")


def test_dashboard_config_form_delegation_url_change_never_runs_on_parent_key(home):
    """A keyless ``delegation.base_url`` inherits the parent's key: a URL change must not leave one."""
    from types import SimpleNamespace
    from hermes_cli.env_loader import load_hermes_dotenv
    from tools.delegate_tool_config import _load_config, _resolve_child_runtime, _resolve_delegation_credentials
    _delegation_form_change(home, {"base_url": "https://b.example/v1"})
    load_hermes_dotenv(hermes_home=home)
    parent = SimpleNamespace(model="p", provider="anthropic", base_url="https://api.anthropic.com",
                             api_key="parent-key", api_mode="anthropic_messages", acp_command=None, acp_args=[])
    creds = _resolve_delegation_credentials(_load_config(), parent)

    runtime = _resolve_child_runtime(
        parent, _load_config(), parent.api_key, model=creds["model"], override_provider=creds["provider"],
        override_base_url=creds["base_url"], override_api_key=creds["api_key"],
        override_api_mode=creds["api_mode"], override_acp_command=None, override_acp_args=None)

    assert (runtime["base_url"], runtime["api_key"]) != ("https://b.example/v1", "parent-key")


@pytest.mark.parametrize("new_url, kept", [("https://a.example/v2", True), ("http://a.example/v1", False),
                                           ("https://a.example:8443/v1", False), ("https://b.example/v1", False)],
                         ids=["same_origin_path", "http_downgrade", "other_port", "other_host"])
def test_dashboard_config_form_url_edit_binds_key_to_origin(home, new_url, kept):
    """The settings form follows the origin rule the aux assignment and the ``hermes model`` picker use."""
    import asyncio
    from hermes_cli.web_routers.config_env import get_config
    _seed(home, {"provider": "custom", "model": "m", "base_url": "https://a.example/v1", "api_key": OLD_SECRET})
    form = asyncio.run(get_config())
    form["auxiliary"]["vision"]["base_url"] = new_url

    _put_config(form)

    assert _resolved_vision()[2:4] == (new_url, OLD_SECRET if kept else None)


def test_dashboard_config_form_same_origin_provider_switch_drops_key(home):
    """The origin exception is bare custom's: a switch to a named provider on one origin still drops the key."""
    import asyncio
    from hermes_cli.web_routers.config_env import get_config
    _seed(home, {"provider": "custom", "model": "m", "base_url": "https://a.example/v1", "api_key": OLD_SECRET})
    form = asyncio.run(get_config())
    form["auxiliary"]["vision"].update(provider="openrouter", base_url="https://a.example/v2")

    _put_config(form)

    assert _resolved_vision()[3] is None
