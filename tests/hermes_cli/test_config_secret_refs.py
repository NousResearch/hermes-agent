"""Config writers store credentials in ``.env`` and only a reference in ``config.yaml`` (#77471).

Each test runs a real writer against a temp ``HERMES_HOME``, then checks the contract: the secret
is not in ``config.yaml``, it is in the profile's ``.env``, and the runtime reader still resolves it.
"""

from __future__ import annotations

import pytest

SECRET = "sk-77471-synthetic-secret-value"


@pytest.fixture()
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").touch()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import os
    for name in list(os.environ):
        if name in {"HERMES_MODEL_API_KEY", "HERMES_DELEGATION_API_KEY", "AUXILIARY_VISION_API_KEY"} or (
                name.startswith(("HERMES_CUSTOM_", "HERMES_PROVIDER_"))):
            monkeypatch.delenv(name, raising=False)
    return home


def _config_text(home) -> str:
    return (home / "config.yaml").read_text(encoding="utf-8")


def _assert_secret_only_in_env(home) -> None:
    assert SECRET not in _config_text(home)
    assert SECRET in (home / ".env").read_text(encoding="utf-8")


def test_config_set_model_api_key_stores_a_reference(home, capsys):
    from hermes_cli.config import load_config, set_config_value
    from hermes_cli.runtime_provider import resolve_runtime_provider

    set_config_value("model.provider", "custom")
    set_config_value("model.base_url", "https://llm.example.com/v1")
    set_config_value("model.default", "m")
    set_config_value("model.api_key", SECRET)

    _assert_secret_only_in_env(home)
    assert "${HERMES_MODEL_API_KEY}" in _config_text(home)
    assert SECRET not in capsys.readouterr().out
    assert load_config()["model"]["api_key"] == SECRET
    assert resolve_runtime_provider(requested="custom")["api_key"] == SECRET


@pytest.mark.parametrize("key,env_name", [
    ("auxiliary.vision.api_key", "AUXILIARY_VISION_API_KEY"),
    ("delegation.api_key", "HERMES_DELEGATION_API_KEY"),
])
def test_config_set_aux_and_delegation_keys_store_a_reference(home, key, env_name):
    from hermes_cli.config import load_config, set_config_value

    set_config_value(key, SECRET)

    _assert_secret_only_in_env(home)
    assert f"${{{env_name}}}" in _config_text(home)
    section, *rest = key.split(".")
    node = load_config()[section]
    for part in rest[:-1]:
        node = node[part]
    assert node["api_key"] == SECRET


def test_config_set_keeps_an_explicit_reference_and_inline_values_still_resolve(home):
    """A ``${VAR}`` the user types is stored verbatim; an inline value written by hand (no migration)
    keeps resolving."""
    from hermes_cli.config import load_config, set_config_value

    set_config_value("model.api_key", "${MY_OWN_KEY}")
    assert "${MY_OWN_KEY}" in _config_text(home)
    assert "MY_OWN_KEY" not in (home / ".env").read_text(encoding="utf-8")

    (home / "config.yaml").write_text(f"model:\n  provider: custom\n  api_key: {SECRET}\n", encoding="utf-8")
    assert load_config()["model"]["api_key"] == SECRET


def test_dashboard_main_custom_endpoint_stores_a_reference(home, monkeypatch):
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.web_server_config import _apply_model_assignment_sync

    monkeypatch.setattr(
        "hermes_cli.models_validate.validate_requested_model",
        lambda *a, **k: {"accepted": True, "persist": True, "recognized": True, "message": None})
    (home / "config.yaml").write_text("model:\n  provider: openrouter\n  default: x\n", encoding="utf-8")

    _apply_model_assignment_sync("main", "custom", "m", "", "https://llm.example.com:8443/v1", SECRET)

    _assert_secret_only_in_env(home)
    cfg = load_config()
    assert cfg["model"]["api_key"] == SECRET
    row = next(e for e in cfg["custom_providers"] if e["base_url"] == "https://llm.example.com:8443/v1")
    assert row["key_env"].startswith("HERMES_CUSTOM_LLM_EXAMPLE_COM_8443_") and "api_key" not in row
    assert cfg["model"]["api_key"] == SECRET and f"${{{row['key_env']}}}" in _config_text(home)
    assert resolve_runtime_provider(requested="custom")["api_key"] == SECRET


def test_dashboard_aux_slot_stores_a_reference(home):
    from agent.auxiliary_client import _resolve_task_provider_model
    from hermes_cli.config import load_config
    from hermes_cli.web_server_config import _apply_model_assignment_sync

    (home / "config.yaml").write_text("model:\n  provider: openrouter\n  default: x\n", encoding="utf-8")

    _apply_model_assignment_sync("auxiliary", "custom", "m", "vision", "https://llm.example.com/v1", SECRET)

    _assert_secret_only_in_env(home)
    assert load_config()["auxiliary"]["vision"]["api_key"] == SECRET
    assert _resolve_task_provider_model("vision")[3] == SECRET


def test_hermes_model_aux_flow_stores_a_reference(home):
    from agent.auxiliary_client import _resolve_task_provider_model
    from hermes_cli.main_provider_setup import _save_aux_choice

    _save_aux_choice("vision", provider="custom", model="m", base_url="https://llm.example.com/v1", api_key=SECRET)

    _assert_secret_only_in_env(home)
    assert _resolve_task_provider_model("vision")[3] == SECRET


def test_new_custom_provider_entry_stores_key_env(home):
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.main_provider_setup import _save_custom_provider

    _save_custom_provider("https://llm.example.com/v1", SECRET, "m", name="Example")

    _assert_secret_only_in_env(home)
    entry = load_config()["custom_providers"][0]
    assert entry["key_env"].startswith("HERMES_CUSTOM_LLM_EXAMPLE_COM_") and "api_key" not in entry
    assert resolve_runtime_provider(requested="custom:example")["api_key"] == SECRET


def test_plugin_aux_task_keys_with_distinct_underscores_keep_distinct_credentials(home):
    """Two valid plugin task keys must not share the env variable after either is saved."""
    from hermes_cli.config import load_config, read_raw_config
    from hermes_cli.main_provider_setup import _save_aux_choice

    tasks = (("my_task", "KEY-ONE"), ("my__task", "KEY-TWO"), ("my_task_", "KEY-THREE"), ("_my_task", "KEY-FOUR"),
             ("My_Task", "KEY-FIVE"))
    for n, (task, key) in enumerate(tasks):
        _save_aux_choice(task, provider="custom", model="m", base_url=f"https://h{n}.example/v1", api_key=key)

    raw = read_raw_config()["auxiliary"]
    resolved = load_config()["auxiliary"]
    assert len({raw[task]["api_key"] for task, _ in tasks}) == len(tasks)
    assert {task: resolved[task]["api_key"] for task, _ in tasks} == dict(tasks)
    assert raw["my_task"]["api_key"] == "${AUXILIARY_MY_TASK_API_KEY}"  # lowercase keys keep the legacy name


def test_endpoint_names_separate_gateways_behind_one_host():
    from hermes_cli.config_secret_refs import endpoint_api_key_env

    assert endpoint_api_key_env("http://localhost:11434/v1").startswith("HERMES_CUSTOM_LOCALHOST_11434_")
    assert endpoint_api_key_env("http://h:9/old/v1") != endpoint_api_key_env("http://h:9/new/v1")


@pytest.mark.parametrize("first,second", [
    ("https://my-host.com/v1", "https://my.host.com/v1"),
    ("https://a.com/x_y/v1", "https://a.com/x/y/v1"),
    ("https://a.com:8443/v1", "https://a.com/8443"),
])
def test_endpoints_whose_slugs_coincide_each_resolve_their_own_key(home, first, second):
    """Two endpoints that differ only in punctuation (or port vs path) must not share a variable,
    or the second save overwrites the first and the first host is sent the second host's key."""
    from hermes_cli.config import load_config
    from hermes_cli.main_provider_setup import _save_custom_provider
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _save_custom_provider(first, "KEY-FIRST", "m1", name="First")
    _save_custom_provider(second, "KEY-SECOND", "m2", name="Second")

    rows = {r["base_url"]: r for r in load_config()["custom_providers"]}
    assert rows[first]["key_env"] != rows[second]["key_env"]
    assert resolve_runtime_provider(requested="custom:first")["api_key"] == "KEY-FIRST"
    assert resolve_runtime_provider(requested="custom:second")["api_key"] == "KEY-SECOND"


def test_provider_name_never_shares_an_endpoint_variable(home):
    """``providers.<name>`` names and endpoint hosts live in separate namespaces: setting
    ``providers.my.host.com``-style names must not overwrite the key of the ``my.host.com`` row."""
    from hermes_cli.config import load_config, set_config_value
    from hermes_cli.main_provider_setup import _save_custom_provider
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _save_custom_provider("https://my.host.com/v1", "KEY-HOST", "m", name="Host")
    (home / "config.yaml").write_text(
        (home / "config.yaml").read_text(encoding="utf-8")
        + "providers:\n  my_host_com:\n    base_url: https://other.example/v1\n", encoding="utf-8")
    set_config_value("providers.my_host_com.api_key", "KEY-NAMED")

    assert load_config()["providers"]["my_host_com"]["api_key"] == "KEY-NAMED"
    assert resolve_runtime_provider(requested="custom:host")["api_key"] == "KEY-HOST"


@pytest.mark.parametrize("config_text,second_key", [
    ("custom_providers:\n- name: work\n  base_url: https://one.example/v1\n"
     "- name: work\n  base_url: https://two.example/v1\n", "custom_providers.1.api_key"),
    ("custom_providers:\n- name: work\n  base_url: https://one.example/v1\n"
     "providers:\n  work:\n    base_url: https://two.example/v1\n", "providers.work.api_key"),
])
def test_config_set_rows_that_share_a_name_keep_their_own_keys(home, config_text, second_key):
    """Row names are not unique: two blocks named ``work`` for different hosts must not share a
    variable, or the second ``config set`` sends its key to the first host."""
    from hermes_cli.config import load_config, set_config_value

    (home / "config.yaml").write_text(config_text, encoding="utf-8")
    set_config_value("custom_providers.0.api_key", "KEY-ONE")
    set_config_value(second_key, "KEY-TWO")

    cfg = load_config()
    second = cfg["custom_providers"][1] if second_key.startswith("custom_") else cfg["providers"]["work"]
    assert cfg["custom_providers"][0]["api_key"] == "KEY-ONE"
    assert second["api_key"] == "KEY-TWO"


def test_config_set_refuses_a_provider_key_another_endpoint_reads(home, monkeypatch):
    """model.api_key shares a known row's variable; once ``config set`` moves that row to another
    URL, setting its key must not re-key the main model's endpoint."""
    from hermes_cli.config import load_config, set_config_value
    from hermes_cli.web_models import CustomEndpointUpdate
    from hermes_cli.web_routers.config_env import upsert_custom_endpoint
    from hermes_cli.web_server_config import _apply_model_assignment_sync

    monkeypatch.setattr(
        "hermes_cli.models_validate.validate_requested_model",
        lambda *a, **k: {"accepted": True, "persist": True, "recognized": True, "message": None})
    (home / "config.yaml").write_text("model:\n  provider: openrouter\n  default: x\n", encoding="utf-8")
    upsert_custom_endpoint(CustomEndpointUpdate(
        id="lab", name="lab", base_url="https://a.example/v1", model="m", api_key="KEY-A"))
    _apply_model_assignment_sync("main", "custom", "m", "", "https://a.example/v1", "KEY-A")
    set_config_value("providers.lab.base_url", "https://b.example/v1")

    with pytest.raises(SystemExit):
        set_config_value("providers.lab.api_key", "KEY-B")
    assert load_config()["model"]["api_key"] == "KEY-A"


def test_store_refuses_to_overwrite_a_variable_another_endpoint_reads(home):
    """Defence in depth: a key for endpoint B is never written over a variable an endpoint-A row
    already reads with a different value."""
    from hermes_cli.config_secret_refs import store_secret

    (home / ".env").write_text("SHARED_KEY=key-a\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "custom_providers:\n- name: A\n  base_url: https://a.example/v1\n  key_env: SHARED_KEY\n",
        encoding="utf-8")
    with pytest.raises(ValueError):
        store_secret("SHARED_KEY", "key-b", "https://b.example/v1")
    assert "key-a" in (home / ".env").read_text(encoding="utf-8")
    store_secret("SHARED_KEY", "key-a2", "https://a.example/v1/")  # the same endpoint rotates freely


def test_dashboard_main_save_for_a_known_endpoint_reuses_the_row_variable(home, monkeypatch):
    """model.api_key and the endpoint's existing row read one variable, so rotating the key from the
    dashboard updates what both resolve."""
    from hermes_cli.config import load_config
    from hermes_cli.main_provider_setup import _save_custom_provider
    from hermes_cli.web_server_config import _apply_model_assignment_sync

    monkeypatch.setattr(
        "hermes_cli.models_validate.validate_requested_model",
        lambda *a, **k: {"accepted": True, "persist": True, "recognized": True, "message": None})
    _save_custom_provider("https://llm.example.com/v1", "old-key", "m", name="Example")
    row_env = load_config()["custom_providers"][0]["key_env"]

    _apply_model_assignment_sync("main", "custom", "m", "", "https://llm.example.com/v1", "new-key")

    from hermes_cli.config import get_env_value, read_raw_config
    assert read_raw_config()["model"]["api_key"] == f"${{{row_env}}}"
    assert load_config()["model"]["api_key"] == "new-key" == get_env_value(row_env)


def test_webhook_env_secret_is_never_persisted_to_config_yaml(home, monkeypatch):
    """The issue's webhook claim: ``WEBHOOK_SECRET`` from the environment is folded into the
    in-memory gateway config only; config saves (home channel, ``config set``) never write it."""
    from gateway.config import HomeChannel, Platform, load_gateway_config, persist_home_channel
    from hermes_cli.config import set_config_value

    (home / "config.yaml").write_text("platforms:\n  webhook:\n    enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("WEBHOOK_SECRET", SECRET)

    assert load_gateway_config().platforms[Platform.WEBHOOK].extra["secret"] == SECRET
    persist_home_channel(HomeChannel(platform=Platform.WEBHOOK, chat_id="c1", name="Home"))
    set_config_value("display.skin", "default")

    assert SECRET not in _config_text(home)


def test_dashboard_endpoint_ids_that_slug_alike_each_keep_their_own_key(home):
    """Dashboard ids ``my-host`` and ``my_host`` are two endpoints; saving the second must neither
    overwrite the first's variable nor leave either row resolving the other's key."""
    from hermes_cli.config import get_env_value, load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.web_models import CustomEndpointUpdate
    from hermes_cli.web_routers.config_env import upsert_custom_endpoint

    for endpoint_id, host, key in (("my-host", "a.example", "KEY-DASH"), ("my_host", "b.example", "KEY-UNDER")):
        upsert_custom_endpoint(CustomEndpointUpdate(
            id=endpoint_id, name=endpoint_id, base_url=f"https://{host}/v1", model="m", api_key=key))

    providers = load_config()["providers"]
    assert providers["my-host"]["key_env"] != providers["my_host"]["key_env"]
    assert get_env_value(providers["my-host"]["key_env"]) == "KEY-DASH"
    assert resolve_runtime_provider(requested="my-host")["api_key"] == "KEY-DASH"
    assert resolve_runtime_provider(requested="my_host")["api_key"] == "KEY-UNDER"


def test_dashboard_save_keeps_a_legacy_variable_and_never_overwrites_it(home):
    """A row an older release wrote keeps its ``HERMES_CUSTOM_<ID>_API_KEY``; a new ``my_host`` save
    gets its own variable, and a save into a name another endpoint still reads is refused."""
    from fastapi import HTTPException
    from hermes_cli.config import get_env_value, load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.web_models import CustomEndpointUpdate
    from hermes_cli.web_routers.config_env import upsert_custom_endpoint

    legacy = "HERMES_CUSTOM_MY_HOST_API_KEY"
    (home / ".env").write_text(f"{legacy}=KEY-OLD\n", encoding="utf-8")
    row = "    base_url: https://{}/v1\n    model: m\n    key_env: " + legacy + "\n"
    (home / "config.yaml").write_text("providers:\n  my-host:\n    name: my-host\n" + row.format("a.example"),
                                      encoding="utf-8")
    upsert_custom_endpoint(CustomEndpointUpdate(
        id="my_host", name="my_host", base_url="https://b.example/v1", model="m", api_key="KEY-NEW"))
    upsert_custom_endpoint(CustomEndpointUpdate(
        id="my-host", name="my-host", base_url="https://a.example/v1", model="m", api_key="KEY-ROT"))
    assert load_config()["providers"]["my-host"]["key_env"] == legacy
    assert resolve_runtime_provider(requested="my-host")["api_key"] == "KEY-ROT"
    assert resolve_runtime_provider(requested="my_host")["api_key"] == "KEY-NEW"

    # Two legacy rows already sharing the name: rotating either would re-key the other host.
    (home / "config.yaml").write_text(
        "providers:\n  my-host:\n    name: my-host\n" + row.format("a.example")
        + "  my_host:\n    name: my_host\n" + row.format("b.example"), encoding="utf-8")
    with pytest.raises(HTTPException):
        upsert_custom_endpoint(CustomEndpointUpdate(
            id="my_host", name="my_host", base_url="https://b.example/v1", model="m", api_key="KEY-B"))
    assert get_env_value(legacy) == "KEY-ROT"


@pytest.mark.parametrize("block", [
    "fallback_model:\n  provider: custom\n  base_url: https://a.example/v1\n  api_key: ${SHARED_KEY}\n",
    "fallback_providers:\n- provider: custom\n  base_url: https://a.example/v1\n  key_env: SHARED_KEY\n",
])
def test_store_refuses_a_variable_a_fallback_block_reads(home, block):
    from hermes_cli.config_secret_refs import store_secret

    (home / ".env").write_text("SHARED_KEY=key-a\n", encoding="utf-8")
    (home / "config.yaml").write_text(block, encoding="utf-8")
    with pytest.raises(ValueError):
        store_secret("SHARED_KEY", "key-b", "https://b.example/v1")


def test_clearing_an_endpoint_key_keeps_the_variable_the_main_model_reads(home, monkeypatch):
    """The main picker points model.api_key at a known row's variable; a blank-key save of that row
    must not delete it, or model.api_key resolves to the literal ``${VAR}``."""
    from hermes_cli.config import load_config
    from hermes_cli.web_models import CustomEndpointUpdate
    from hermes_cli.web_routers.config_env import upsert_custom_endpoint
    from hermes_cli.web_server_config import _apply_model_assignment_sync

    monkeypatch.setattr(
        "hermes_cli.models_validate.validate_requested_model",
        lambda *a, **k: {"accepted": True, "persist": True, "recognized": True, "message": None})
    (home / "config.yaml").write_text("model:\n  provider: openrouter\n  default: x\n", encoding="utf-8")
    endpoint = {"id": "lab", "name": "lab", "base_url": "https://lab.example/v1", "model": "m"}
    upsert_custom_endpoint(CustomEndpointUpdate(**endpoint, api_key="KEY-LAB"))
    _apply_model_assignment_sync("main", "custom", "m", "", "https://lab.example/v1", "KEY-MAIN")

    upsert_custom_endpoint(CustomEndpointUpdate(**endpoint, api_key=""))

    assert "key_env" not in load_config()["providers"]["lab"]
    assert load_config()["model"]["api_key"] == "KEY-MAIN"
    # A variable only the cleared row reads is still deleted.
    solo = {"id": "solo", "name": "solo", "base_url": "https://solo.example/v1", "model": "m"}
    upsert_custom_endpoint(CustomEndpointUpdate(**solo, api_key="KEY-SOLO"))
    upsert_custom_endpoint(CustomEndpointUpdate(**solo, api_key=""))
    assert "KEY-SOLO" not in (home / ".env").read_text(encoding="utf-8")


def test_hermes_model_custom_flow_rotates_the_known_row_variable(home, monkeypatch):
    """``hermes model`` for an endpoint that already has a Hermes-written row writes the new key
    under that row's variable, so every block reading it (here an aux slot) gets the rotation."""
    from hermes_cli import model_setup_flows_custom as flow
    from hermes_cli.config import load_config, read_raw_config

    row_env = "HERMES_PROVIDER_LAB_0000ABCD_API_KEY"
    (home / ".env").write_text(f"{row_env}=KEY-OLD\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: x\n"
        f"providers:\n  lab:\n    base_url: https://lab.example/v1\n    key_env: {row_env}\n"
        "auxiliary:\n  vision:\n    provider: custom\n    base_url: https://lab.example/v1\n"
        f"    api_key: ${{{row_env}}}\n", encoding="utf-8")
    answers = iter(["https://lab.example/v1", "", "lab"])
    monkeypatch.setattr(flow, "line_input", lambda _prompt="": next(answers))
    monkeypatch.setattr("hermes_cli.secret_prompt.masked_secret_prompt", lambda _prompt="": "KEY-NEW")
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    monkeypatch.setattr("hermes_cli.models.probe_api_models", lambda *a, **k: {"models": ["m"]})
    monkeypatch.setattr("hermes_cli.main_provider_setup._prompt_custom_api_mode_selection",
                        lambda *a, **k: None)

    flow._model_flow_custom({})

    assert read_raw_config()["model"]["api_key"] == f"${{{row_env}}}"
    assert load_config()["auxiliary"]["vision"]["api_key"] == "KEY-NEW"


def test_config_set_on_a_dotted_provider_name_stores_a_reference(home):
    """``config set`` writes ``providers.lab.example.api_key`` into the existing
    ``providers["lab.example"]`` block, so the credential routing must classify it the same way."""
    from hermes_cli.config import load_config, set_config_value

    (home / "config.yaml").write_text(
        "providers:\n  lab.example:\n    base_url: https://lab.example/v1\n", encoding="utf-8")
    set_config_value("providers.lab.example.api_key", SECRET)

    _assert_secret_only_in_env(home)
    assert load_config()["providers"]["lab.example"]["api_key"] == SECRET


def test_config_set_on_hostless_rows_keeps_each_row_its_own_key(home):
    """Rows without a URL yet fall back to their name, so two named rows still get two variables."""
    from hermes_cli.config import load_config, set_config_value

    (home / "config.yaml").write_text("custom_providers:\n- name: one\n- name: two\n", encoding="utf-8")
    set_config_value("custom_providers.0.api_key", "KEY-ONE")
    set_config_value("custom_providers.1.api_key", "KEY-TWO")

    assert [row["api_key"] for row in load_config()["custom_providers"]] == ["KEY-ONE", "KEY-TWO"]
    assert "KEY-ONE" not in _config_text(home)
