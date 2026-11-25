"""A ``providers:`` block whose config key carries uppercase (``Axonhub``, ``OpenAI``) is the
provider every picker row offers under that key, so selecting it must route to THAT endpoint on
each switch surface. ``resolve_provider_full`` lowercases the request before the user-config
lookup, and a case-sensitive dict read missed the key: ``OpenAI`` fell through to the legacy
``openai`` -> OpenRouter alias and ``hermes model`` / ``/model`` switched to the wrong host.

Real config on a temp ``HERMES_HOME``; loopback ``/v1/models``; no mocks on the switch chain.
"""

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest


class _Listing(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"data": [{"id": "m1"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def _write_config(key: str, provider: str = "openrouter") -> tuple[Path, str]:
    srv = HTTPServer(("127.0.0.1", 0), _Listing)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{srv.server_port}/v1"
    path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    path.write_text(f"model:\n  provider: {provider}\n  default: x\n"
                    f"providers:\n  {key}:\n    base_url: {base_url}\n    api_key_env: MIXED_KEY\n    models: [m1]\n")
    return path, base_url


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("MIXED_KEY", "mixed-key-123456")


@pytest.mark.parametrize("key", ["Axonhub", "OpenAI"])
def test_gateway_picker_slug_switches_to_the_configured_endpoint(key):
    """The Telegram picker forwards the configured key verbatim (``mp:<key>``) to the gateway switch."""
    from gateway.slash_commands_model import GatewayModelCommandsMixin, _ModelSwitchContext

    config_path, base_url = _write_config(key)
    ctx = _ModelSwitchContext(session_key="s", source=None, config_path=config_path, persist_global=False)
    ctx.read_config()
    result, error = asyncio.run(GatewayModelCommandsMixin._perform_model_switch(SimpleNamespace(), ctx, "m1", key, None))
    assert error is None, error
    assert (result.new_model, result.base_url, result.api_key) == ("m1", base_url, "mixed-key-123456")


@pytest.mark.parametrize("requested", ["OpenAI", "openai"])
def test_cli_model_command_switches_to_the_configured_endpoint(requested):
    """CLI ``/model m1 --provider <key>``: the context the CLI loads, then its switch seam."""
    from hermes_cli.cli_model_switch_mixin import _switch_model_from
    from hermes_cli.inventory import load_picker_context

    _, base_url = _write_config("OpenAI")
    ctx = load_picker_context()
    cli = SimpleNamespace(provider="openrouter", model="x", base_url="", api_key="")
    result = _switch_model_from(cli, "m1", is_global=False, explicit_provider=requested,
                                user_providers=ctx.user_providers, custom_providers=ctx.custom_providers)
    assert result.success, result.error_message
    assert (result.base_url, result.api_key) == (base_url, "mixed-key-123456")


def test_hermes_model_marks_the_configured_mixed_case_provider_active():
    """``hermes model`` opens with the configured ``providers.OpenAI`` row selected, not OpenRouter."""
    from hermes_cli.config import load_config
    from hermes_cli.main import _resolve_active_provider
    from hermes_cli.main_provider_setup import _named_custom_provider_map

    _write_config("OpenAI", provider="OpenAI")
    cfg = load_config()
    custom_map = _named_custom_provider_map(cfg)
    assert _resolve_active_provider(cfg, cfg["model"], "OpenAI", custom_map) in custom_map


@pytest.mark.parametrize("key", ["Anthropic", "OpenRouter", "anthropic", "openrouter", "deepseek", "DeepSeek"])
def test_builtin_name_with_an_owning_entry_switches_saves_and_runs_on_the_user_endpoint(key, monkeypatch):
    """Ruling on #151775 (option A): a bare built-in name, in any case, whose ``providers:`` entry
    declares its own base_url means THAT entry. ``--provider <name>`` validates the user endpoint,
    persists ``custom:<key>``, and the next turn runs there with the entry's credential, never the
    built-in's env key."""
    for env in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.setenv(env, f"builtin-{env.lower()}-123456")
    from hermes_cli.config import atomic_config_write, load_config
    from hermes_cli.model_switch import model_selection_config_updates, switch_model
    from hermes_cli.runtime_provider import resolve_runtime_provider

    config_path, base_url = _write_config(key, provider="bedrock")
    cfg = load_config()
    for requested in (key, key.lower()):
        result = switch_model("m1", "bedrock", "x", is_global=True, explicit_provider=requested,
                              user_providers=cfg["providers"], custom_providers=[])
        assert result.success, result.error_message
        assert (result.target_provider, result.base_url, result.api_key) == (
            f"custom:{key.lower()}", base_url, "mixed-key-123456")
        new_model = {k: v for k, v in {**cfg["model"], **model_selection_config_updates(result, cfg["model"])}.items()
                     if v is not None}
        atomic_config_write(config_path, {"model": new_model, "providers": cfg["providers"]})
        runtime = resolve_runtime_provider(target_model="m1")
        assert (runtime["base_url"].rstrip("/"), runtime["api_key"]) == (base_url, "mixed-key-123456")


@pytest.mark.parametrize("key", ["openrouter", "OpenRouter", "anthropic", "Anthropic", "deepseek", "DeepSeek"])
def test_config_saved_as_the_bare_builtin_name_runs_on_the_owning_entry(key, monkeypatch):
    """A config main already saved (``model.provider: <built-in>`` + the entry's URL) runs on the user
    endpoint with the entry's key after upgrade; before, it got the built-in env key (OpenRouter /
    DeepSeek, sent to the user's URL) or ran on the built-in host (Anthropic)."""
    for env in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.setenv(env, f"builtin-{env.lower()}-123456")
    from hermes_cli.runtime_provider import resolve_runtime_provider

    config_path, base_url = _write_config(key, provider=key)
    config_path.write_text(config_path.read_text().replace("  default: x\n", f"  default: m1\n  base_url: {base_url}\n", 1))
    runtime = resolve_runtime_provider(target_model="m1")
    assert (runtime["base_url"].rstrip("/"), runtime["api_key"]) == (base_url, "mixed-key-123456")


@pytest.mark.parametrize(("provider", "env"), [("openrouter", "OPENROUTER_API_KEY"), ("deepseek", "DEEPSEEK_API_KEY")])
def test_builtin_env_key_never_goes_to_a_url_another_entry_owns(provider, env, monkeypatch):
    """``model.provider: <built-in>`` + ``model.base_url`` owned by ``providers.myproxy`` that declares
    its own key: the request carries that key, never the built-in env key. Without a declared
    credential the #10622 mirror keeps the built-in key."""
    monkeypatch.setenv(env, "builtin-env-key-123456")
    from hermes_cli.runtime_provider import resolve_runtime_provider

    config_path, base_url = _write_config("myproxy")
    text = f"model:\n  provider: {provider}\n  default: m1\n  base_url: {base_url}\n" + config_path.read_text().split("\n", 3)[3]
    config_path.write_text(text)
    runtime = resolve_runtime_provider(target_model="m1")
    assert (runtime["base_url"].rstrip("/"), runtime["api_key"]) == (base_url, "mixed-key-123456")

    config_path.write_text(text.replace("    api_key_env: MIXED_KEY\n", ""))
    assert resolve_runtime_provider(target_model="m1")["api_key"] == "builtin-env-key-123456"


def test_tuning_only_block_keeps_the_builtin(monkeypatch):
    """A ``providers.<built-in>`` block with no base_url is tuning for the built-in, not an endpoint."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "builtin-deepseek-123456")
    from hermes_cli.runtime_provider import resolve_runtime_provider

    path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    path.write_text("model:\n  provider: DeepSeek\n  default: m1\nproviders:\n  deepseek:\n    request_timeout_seconds: 5\n")
    runtime = resolve_runtime_provider(target_model="m1")
    assert (runtime["provider"], runtime["api_key"]) == ("deepseek", "builtin-deepseek-123456")


@pytest.mark.parametrize("key", ["OpenRouter", "deepseek"])
def test_doctor_names_the_endpoint_a_builtin_named_provider_runs_on(key, capsys):
    from hermes_cli import doctor_config

    config_path, base_url = _write_config(key, provider=key.lower())
    issues: list = []
    doctor_config._validate_model_config(config_path, issues)
    found = [i for i in issues if f"providers.{key}" in i]
    assert len(found) == 1 and base_url in found[0] and f"custom:{key.lower()}" in found[0], issues

    config_path.write_text(config_path.read_text().replace(f"  provider: {key.lower()}", f"  provider: custom:{key.lower()}"))
    issues = []
    doctor_config._validate_model_config(config_path, issues)
    assert not [i for i in issues if f"providers.{key}" in i], issues


@pytest.mark.parametrize("key", ["Anthropic", "OpenRouter"])
def test_typed_model_declared_by_builtin_named_key_keeps_custom_runtime(key):
    """A typed model pick must use the same custom host/key on its next turn (#151775)."""
    from hermes_cli.config import atomic_config_write, load_config
    from hermes_cli.model_switch import model_selection_config_updates, switch_model
    from hermes_cli.runtime_provider import resolve_runtime_provider

    config_path, base_url = _write_config(key, provider="bedrock")
    cfg = load_config()
    result = switch_model("m1", "bedrock", "x", is_global=True,
                          user_providers=cfg["providers"], custom_providers=[])
    assert result.success, result.error_message
    assert (result.base_url, result.api_key) == (base_url, "mixed-key-123456")

    updates = model_selection_config_updates(result, cfg["model"])
    assert updates["provider"] == f"custom:{key.lower()}"
    new_model = {k: v for k, v in {**cfg["model"], **updates}.items() if v is not None}
    atomic_config_write(config_path, {"model": new_model, "providers": cfg["providers"]})

    runtime = resolve_runtime_provider(target_model="m1")
    assert runtime["provider"] == "custom"
    assert (runtime["base_url"].rstrip("/"), runtime["api_key"]) == (
        base_url, "mixed-key-123456")


@pytest.mark.parametrize("key", ["Anthropic", "OpenRouter"])
def test_dashboard_main_pick_of_builtin_named_key_runs_where_it_validated(key, monkeypatch):
    """The dashboard's main-model save goes through its own normalizer before ``switch_model``: the
    endpoint it validates and saves is the one the next turn runs on (#151775). Both built-ins hold
    a key, so a pick that validates one route and saves another shows up as a different key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-builtin-123456")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-builtin-123456")
    from hermes_cli.config import atomic_config_write, load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.model_assignment import apply_main_model_assignment
    from hermes_cli.web_server_config import _prepare_main_assignment

    config_path, _ = _write_config(key)
    cfg = load_config()
    _, result = _prepare_main_assignment(cfg, key, "m1", "", "")
    atomic_config_write(config_path, {"model": apply_main_model_assignment(cfg["model"], result),
                                      "providers": cfg["providers"]})

    runtime = resolve_runtime_provider(target_model="m1")
    assert (runtime["base_url"].rstrip("/"), runtime["api_key"]) == (result.base_url.rstrip("/"), result.api_key)


@pytest.mark.parametrize(("key", "env", "host"), [
    ("deepseek", "DEEPSEEK_API_KEY", "api.deepseek.com"),
    ("anthropic", "ANTHROPIC_API_KEY", "api.anthropic.com"),
    ("lmstudio", "LM_API_KEY", "127.0.0.1:1234"),
])
def test_typed_model_on_a_tuning_only_builtin_block_keeps_the_builtin(key, env, host, monkeypatch):
    """``/model <model>`` declared by a ``providers.<built-in>`` block with no base_url (tuning only)
    switches to the built-in, never a ``custom:<key>`` provider that does not exist."""
    monkeypatch.setenv(env, f"builtin-{key}-123456")
    from hermes_cli.model_switch import switch_model

    providers = {key: {"models": ["probe-model"], "request_timeout_seconds": 300}}
    result = switch_model("probe-model", "bedrock", "x", is_global=True,
                          user_providers=providers, custom_providers=[])
    assert result.success, result.error_message
    assert result.target_provider == key and host in result.base_url
