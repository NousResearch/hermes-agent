"""``hermes providers list``: the scriptable provider inventory and its ``--offline`` guarantee."""

from __future__ import annotations

import argparse
import json
import socket

import pytest

from hermes_cli.inventory import ConfigContext
from hermes_cli.inventory_offline import build_offline_models_payload
from hermes_cli.subcommands.providers import build_providers_parser


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes")
    build_providers_parser(parser.add_subparsers(dest="command"))
    return parser


def _ctx(**overrides) -> ConfigContext:
    base = dict(current_provider="", current_model="", current_base_url="", user_providers={}, custom_providers=[])
    base.update(overrides)
    return ConfigContext(**base)


def _config_ctx(monkeypatch, config: dict) -> ConfigContext:
    """The context ``hermes providers list`` really reads: config.yaml through ``load_picker_context``."""
    from hermes_cli.inventory import load_picker_context

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    return load_picker_context()


def test_parser_accepts_scriptable_offline_flags():
    args = _parser().parse_args(["providers", "ls", "--all", "--no-live", "--json", "--provider", "openrouter"])
    assert (args.providers_action, args.offline, args.all, args.json, args.provider) == (
        "ls", True, True, True, "openrouter")


def test_offline_inventory_never_opens_a_socket(monkeypatch):
    """--offline must hold with every provider listed: a connect attempt anywhere is a failure."""
    def _refuse(*_a, **_kw):
        raise AssertionError("--offline opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    payload = build_offline_models_payload(_ctx(current_provider="openrouter"), include_unconfigured=True)
    assert any(row["slug"] == "anthropic" for row in payload["providers"])


def test_offline_inventory_uses_config_and_bundled_catalogs(monkeypatch):
    ctx = _config_ctx(monkeypatch, {
        "model": {"provider": "openrouter", "default": "example/current-model"},
        "providers": {"local": {"name": "Local gateway", "base_url": "http://localhost:9000/v1",
                                "models": {"qwen-local": {}}}},
        "custom_providers": [{"name": "Legacy endpoint", "base_url": "http://localhost:8000/v1",
                              "model": "legacy-model"}],
    })
    by_id = {row["slug"]: row for row in build_offline_models_payload(ctx)["providers"]}
    assert {"openrouter", "local", "custom:legacy-endpoint"} <= set(by_id)
    assert by_id["openrouter"]["is_current"] is True
    assert by_id["openrouter"]["models"][0] == "example/current-model"
    assert by_id["local"]["models"] == ["qwen-local"]
    assert by_id["custom:legacy-endpoint"]["models"] == ["legacy-model"]


@pytest.mark.parametrize("legacy_override", [
    {"key_env": "LEGACY_API_KEY"}, {"api_mode": "responses"}, {"extra_headers": {"X-Tenant": "legacy"}},
], ids=["credential", "api-mode", "headers"])
def test_offline_inventory_keeps_legacy_entries_with_distinct_endpoint_identity(monkeypatch, legacy_override):
    shared = {"base_url": "https://gateway.example.test/v1", "key_env": "KEYED_API_KEY",
              "api_mode": "chat_completions", "extra_headers": {"X-Tenant": "keyed"}}
    legacy = {**shared, "name": "Legacy endpoint", "model": "legacy-model", **legacy_override}
    ctx = _config_ctx(monkeypatch, {
        "providers": {"configured": {**shared, "name": "Configured", "models": ["configured-model"]}},
        "custom_providers": [legacy]})
    by_id = {row["slug"]: row for row in build_offline_models_payload(ctx)["providers"]}
    assert by_id["configured"]["models"] == ["configured-model"]
    assert by_id["custom:legacy-endpoint"]["models"] == ["legacy-model"]


def test_offline_inventory_lists_a_keyed_provider_once_with_its_normalized_fields(monkeypatch):
    """camelCase keys and API-mode aliases are read by the runtime normalizer, once: one row, full fields."""
    ctx = _config_ctx(monkeypatch, {"providers": {"gateway": {
        "name": "Gateway", "baseUrl": "https://gw.example.test/v1", "keyEnv": "GW_API_KEY",
        "defaultModel": "gw-default", "api_mode": "openai", "models": ["gw-other"]}}})
    rows = [row for row in build_offline_models_payload(ctx)["providers"] if row["is_user_defined"]]
    assert [(row["slug"], row["key_env"]) for row in rows] == [("gateway", "GW_API_KEY")]
    assert set(rows[0]["models"]) == {"gw-default", "gw-other"}


def test_offline_inventory_marks_a_saved_custom_key_current_on_its_own_row(monkeypatch):
    ctx = _config_ctx(monkeypatch, {
        "model": {"provider": "custom:local", "default": "picked"},
        "providers": {"local": {"name": "Local", "base_url": "http://localhost:9000/v1", "models": ["a", "b"]}}})
    rows = [row for row in build_offline_models_payload(ctx)["providers"] if row["is_user_defined"]]
    assert [(row["slug"], row["is_current"], row["models"]) for row in rows] == [("local", True, ["picked", "a", "b"])]


@pytest.mark.parametrize("config", [
    {"model": {"provider": "openrouter"}, "model_catalog": {"excluded_providers": ["openrouter"]},
     "providers": {"openrouter": {"base_url": "https://proxy.example.test/v1"}}},
    {"model": {"provider": "custom:local"},
     "providers": {"local": {"base_url": "http://localhost:9000/v1", "enabled": False}}},
    {"model": {"provider": "openrouter"}, "providers": {"openrouter": {"enabled": False}}},
], ids=["excluded", "disabled-endpoint", "disabled-canonical"])
def test_offline_inventory_never_lists_a_provider_the_user_hid(monkeypatch, config):
    monkeypatch.setenv("OPENROUTER_API_KEY", "set")
    slugs = {row["slug"] for row in build_offline_models_payload(_config_ctx(monkeypatch, config))["providers"]}
    assert not slugs & {"openrouter", "local", "custom:local"}


def _run(monkeypatch, capsys, argv):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
        "model": {"provider": "local", "default": "m1"},
        "providers": {"local": {"name": "Local", "base_url": "http://localhost:9000/v1",
                                "models": ["m1", "m2"], "key_env": "LOCAL_KEY"}}})
    args = _parser().parse_args(argv)
    code = args.func(args)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_json_document_shape(monkeypatch, capsys):
    code, out, err = _run(monkeypatch, capsys, ["providers", "list", "--offline", "--json", "--provider", "local"])
    doc = json.loads(out)
    assert code == 0 and err == ""
    assert next(iter(doc)) == "schema_version" and doc["schema_version"] == 1
    assert doc["offline"] is True and doc["current"] == {"provider": "local", "model": "m1"}
    assert doc["providers"] == [{
        "id": "local", "name": "Local", "auth_state": "configured", "auth_type": "api_key",
        "env_vars": ["LOCAL_KEY"], "is_current": True, "is_user_defined": True,
        "models": ["m1", "m2"], "model_count": 2, "source": "user-config",
    }]


def test_unknown_provider_errors_on_stderr_with_empty_stdout(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, capsys, ["providers", "list", "--offline", "--json", "--provider", "nope"])
    out = capsys.readouterr()
    assert exc.value.code == 2
    assert out.out == "" and "unknown or unavailable provider: nope" in out.err


def test_live_rows_report_registry_auth_type_and_endpoint_key_env(monkeypatch, capsys):
    """Live picker rows carry no auth fields; the JSON must still match what --offline reports."""
    monkeypatch.setattr("hermes_cli.inventory.build_models_payload", lambda *_a, **_kw: {"model": "", "provider": "", "providers": [
        {"slug": "openai-codex", "name": "OpenAI Codex", "is_current": False, "is_user_defined": False,
         "models": ["gpt-5"], "total_models": 1, "source": "built-in", "authenticated": True},
        {"slug": "gateway", "name": "Gateway", "is_current": False, "is_user_defined": True,
         "models": ["m"], "total_models": 1, "source": "user-config", "api_url": "https://gw.example.test/v1"}]})
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
        "providers": {"gateway": {"name": "Gateway", "base_url": "https://gw.example.test/v1", "key_env": "GW_KEY"}}})
    args = _parser().parse_args(["providers", "list", "--json"])
    assert args.func(args) == 0
    by_id = {row["id"]: row for row in json.loads(capsys.readouterr().out)["providers"]}
    assert by_id["openai-codex"]["auth_type"] == "oauth_external"
    assert by_id["gateway"]["env_vars"] == ["GW_KEY"]
