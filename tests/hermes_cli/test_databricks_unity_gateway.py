"""Databricks Unity Gateway CLI setup behavior tests."""

from __future__ import annotations

import json
import sys
from email.message import Message
from types import SimpleNamespace

import pytest


def _select_first_profile(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.model_setup_flows_common._curses_choice",
        lambda _title, _rows, _default: 0,
    )


def test_valid_u2m_profiles_preserve_literal_names_and_filter_unsafe_rows():
    from hermes_cli.model_setup_flows_databricks import valid_u2m_profiles

    payload = {
        "profiles": [
            {"name": "safe; $(literal)", "host": "https://workspace.example/", "valid": True,
             "auth_type": "databricks-cli"},
            {"name": "pat", "host": "https://workspace.example", "valid": True, "auth_type": "pat"},
            {"name": "expired", "host": "https://workspace.example", "valid": False,
             "auth_type": "databricks-cli"},
            {"name": "http", "host": "http://workspace.example", "valid": True,
             "auth_type": "databricks-cli"},
            {"name": "path", "host": "https://workspace.example/not-an-origin", "valid": True,
             "auth_type": "databricks-cli"},
            {"name": "query", "host": "https://workspace.example?redirect=elsewhere", "valid": True,
             "auth_type": "databricks-cli"},
            {"name": "fragment", "host": "https://workspace.example#fragment", "valid": True,
             "auth_type": "databricks-cli"},
        ]
    }

    assert valid_u2m_profiles(payload) == [
        {"name": "safe; $(literal)", "host": "https://workspace.example"}
    ]


def test_load_u2m_profiles_uses_literal_databricks_argv(monkeypatch):
    import hermes_cli.model_setup_flows_databricks as dbx

    payload = {"profiles": [{"name": "p", "host": "https://workspace.example", "valid": True,
                              "auth_type": "databricks-cli"}]}
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(dbx.subprocess, "run", fake_run)

    profiles, error = dbx.load_u2m_profiles()

    assert error is None
    assert profiles == [{"name": "p", "host": "https://workspace.example"}]
    assert seen["argv"] == ["databricks", "auth", "profiles", "--output", "json"]
    assert seen["kwargs"]["shell"] is False
    assert seen["kwargs"]["capture_output"] is True


def test_preflight_token_uses_selected_profile_as_one_literal_argument(monkeypatch):
    import hermes_cli.model_setup_flows_databricks as dbx

    seen = {}

    def fake_json(argv):
        seen["argv"] = argv
        return {"access_token": "SENTINEL", "expiry": "2099-01-01T00:00:00Z"}, None

    monkeypatch.setattr(dbx, "_run_databricks_json", fake_json)

    token, error = dbx.preflight_profile("profile; $(literal)")

    assert token == "SENTINEL"
    assert error is None
    assert seen["argv"] == [
        "databricks", "auth", "token", "--profile", "profile; $(literal)", "--output", "json"
    ]
    assert dbx.token_argv("profile; $(literal)") == seen["argv"]


def test_discover_models_keeps_chat_services_and_excludes_gpt_oss(monkeypatch):
    import hermes_cli.model_setup_flows_databricks as dbx

    payload = {
        "model_services": [
            {"name": "model-services/system.ai.kimi-code", "supported_api_types": ["mlflow/v1/chat/completions"]},
            {"name": "model-services/system.ai.future-chat", "supported_api_types": ["mlflow/v1/chat/completions"]},
            {"name": "model-services/system.ai.embed", "supported_api_types": ["mlflow/v1/embeddings"]},
            {"name": "model-services/system.ai.gpt-oss-120b", "supported_api_types": ["mlflow/v1/chat/completions"]},
            {"name": "not-model-services/system.ai.bad", "supported_api_types": ["mlflow/v1/chat/completions"]},
            {"name": "model-services/system.ai.bad-types", "supported_api_types": "mlflow/v1/chat/completions"},
        ]
    }
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(payload).encode()

    def fake_urlopen(request, **kwargs):
        seen["url"] = request.full_url
        seen["auth"] = request.headers.get("Authorization")
        seen["kwargs"] = kwargs
        return Response()

    monkeypatch.setattr(dbx, "open_credentialed_url", fake_urlopen, raising=False)
    monkeypatch.setattr(
        dbx.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("credentialed requests must not use raw urlopen")
        ),
    )

    models, error = dbx.discover_models("https://workspace.example", "SENTINEL")

    assert error is None
    assert models == ["system.ai.kimi-code", "system.ai.future-chat"]
    assert "parent=schemas%2Fsystem.ai" in seen["url"]
    assert seen["auth"] == "Bearer SENTINEL"


def test_discover_models_follows_page_tokens(monkeypatch):
    import hermes_cli.model_setup_flows_databricks as dbx

    payloads = [
        {"model_services": [{"name": "model-services/system.ai.first", "supported_api_types": ["mlflow/v1/chat/completions"]}],
         "next_page_token": "next-token"},
        {"model_services": [{"name": "model-services/system.ai.second", "supported_api_types": ["mlflow/v1/chat/completions"]}]},
    ]
    urls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(request, **_kwargs):
        urls.append(request.full_url)
        return Response(payloads.pop(0))

    monkeypatch.setattr(dbx, "open_credentialed_url", fake_urlopen)

    models, error = dbx.discover_models("https://workspace.example", "SENTINEL")

    assert error is None
    assert models == ["system.ai.first", "system.ai.second"]
    assert len(urls) == 2
    assert "page_token=next-token" in urls[1]


def test_discover_models_bounds_unique_page_tokens(monkeypatch):
    import hermes_cli.model_setup_flows_databricks as dbx

    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def fake_open(_request, **_kwargs):
        calls.append(len(calls) + 1)
        if len(calls) > 10:
            raise AssertionError("discovery exceeded its page bound")
        return Response({"model_services": [], "next_page_token": f"page-{len(calls)}"})

    monkeypatch.setattr(dbx, "open_credentialed_url", fake_open)

    models, error = dbx.discover_models("https://workspace.example", "SENTINEL")

    assert models == []
    assert error is not None and "page limit" in error
    assert len(calls) == 10


def test_discover_models_retries_transient_gateway_status(monkeypatch):
    import hermes_cli.model_setup_flows_databricks as dbx

    attempts = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "model_services": [{
                    "name": "model-services/system.ai.kimi-code",
                    "supported_api_types": ["mlflow/v1/chat/completions"],
                }]
            }).encode()

    def fake_urlopen(request, **_kwargs):
        attempts.append(request.full_url)
        if len(attempts) == 1:
            raise dbx.urllib.error.HTTPError(request.full_url, 499, "private-body", Message(), None)
        return Response()

    monkeypatch.setattr(dbx, "open_credentialed_url", fake_urlopen)
    monkeypatch.setattr(dbx, "time", SimpleNamespace(sleep=lambda _seconds: None), raising=False)

    models, error = dbx.discover_models("https://workspace.example", "SENTINEL")

    assert error is None
    assert models == ["system.ai.kimi-code"]
    assert len(attempts) == 2


def test_gateway_json_retries_unlisted_server_error(monkeypatch):
    import hermes_cli.model_setup_flows_databricks as dbx

    attempts = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(request, **_kwargs):
        attempts.append(request.full_url)
        if len(attempts) == 1:
            raise dbx.urllib.error.HTTPError(request.full_url, 501, "private-body", Message(), None)
        return Response()

    monkeypatch.setattr(dbx, "open_credentialed_url", fake_urlopen)
    monkeypatch.setattr(dbx, "time", SimpleNamespace(sleep=lambda _seconds: None), raising=False)
    request = dbx.urllib.request.Request("https://workspace.example/test")

    assert dbx._urlopen_json(request, retries=3) == {"ok": True}
    assert len(attempts) == 2


def test_gateway_json_retries_transient_network_error(monkeypatch):
    import hermes_cli.model_setup_flows_databricks as dbx

    attempts = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_open(_request, **_kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("temporary network failure")
        return Response()

    monkeypatch.setattr(dbx, "open_credentialed_url", fake_open)
    monkeypatch.setattr(dbx, "time", SimpleNamespace(sleep=lambda _seconds: None))
    request = dbx.urllib.request.Request("https://workspace.example/test")

    assert getattr(dbx, "_urlopen_json")(request, retries=3) == {"ok": True}
    assert len(attempts) == 2


def test_preflight_model_requires_forced_tool_call(monkeypatch):
    import hermes_cli.model_setup_flows_databricks as dbx

    seen = {}
    payload = {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {"name": "hermes_setup_probe", "arguments": '{"status":"ok"}'},
                }],
            },
        }],
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(payload).encode()

    def fake_urlopen(request, **kwargs):
        seen["url"] = request.full_url
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.data)
        seen["kwargs"] = kwargs
        return Response()

    monkeypatch.setattr(dbx, "open_credentialed_url", fake_urlopen)

    ok, error = getattr(dbx, "preflight_model")(
        "https://workspace.example", "SENTINEL", "system.ai.agent-model"
    )

    assert ok is True
    assert error is None
    assert seen["url"] == "https://workspace.example/ai-gateway/mlflow/v1/chat/completions"
    assert seen["auth"] == "Bearer SENTINEL"
    assert seen["body"]["tool_choice"]["function"]["name"] == "hermes_setup_probe"
    assert seen["body"]["max_tokens"] >= 512
    assert seen["body"]["stream"] is False
    assert "temperature" not in seen["body"]
    assert "stream_options" not in seen["body"]
    assert "reasoning_effort" not in seen["body"]


def test_preflight_uses_responses_for_gpt_5_6_tools(monkeypatch):
    import hermes_cli.model_setup_flows_databricks as dbx

    seen = {}

    def fake_urlopen_json(request, **_kwargs):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return {"output": [{
            "type": "function_call", "name": "hermes_setup_probe",
            "arguments": '{"status":"ok"}', "call_id": "call_setup",
        }]}

    monkeypatch.setattr(dbx, "_urlopen_json", fake_urlopen_json)

    ok, error = dbx.preflight_model(
        "https://workspace.example", "SENTINEL", "system.ai.gpt-5-6-sol"
    )

    assert ok is True
    assert error is None
    assert seen["url"].endswith("/ai-gateway/mlflow/v1/responses")
    assert seen["body"]["input"] == "Call the provided function."
    assert seen["body"]["tools"][0]["name"] == "hermes_setup_probe"
    assert seen["body"]["tool_choice"] == {"type": "function", "name": "hermes_setup_probe"}
    assert "reasoning_effort" not in seen["body"]


def test_preflight_uses_anthropic_messages_for_claude_tools(monkeypatch):
    import hermes_cli.model_setup_flows_databricks as dbx

    seen = {}

    def fake_urlopen_json(request, **_kwargs):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        seen["version"] = request.headers.get("Anthropic-version")
        return {"content": [{
            "type": "tool_use", "name": "hermes_setup_probe",
            "id": "tool_setup", "input": {"status": "ok"},
        }]}

    monkeypatch.setattr(dbx, "_urlopen_json", fake_urlopen_json)

    ok, error = dbx.preflight_model(
        "https://workspace.example", "SENTINEL", "system.ai.claude-sonnet-current",
    )

    assert ok is True
    assert error is None
    assert seen["url"] == "https://workspace.example/ai-gateway/anthropic/v1/messages"
    assert seen["version"] == "2023-06-01"
    assert seen["body"]["tools"][0]["name"] == "hermes_setup_probe"
    assert seen["body"]["tool_choice"] == {"type": "tool", "name": "hermes_setup_probe"}


def test_preflight_uses_gemini_native_for_gemini_tools(monkeypatch):
    import hermes_cli.model_setup_flows_databricks as dbx

    seen = {}

    def fake_urlopen_json(request, **_kwargs):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return {"candidates": [{"content": {"parts": [{
            "functionCall": {"name": "hermes_setup_probe", "args": {"status": "ok"}},
        }]}}]}

    monkeypatch.setattr(dbx, "_urlopen_json", fake_urlopen_json)

    ok, error = dbx.preflight_model(
        "https://workspace.example", "SENTINEL", "system.ai.gemini-current",
    )

    assert ok is True
    assert error is None
    assert seen["url"] == (
        "https://workspace.example/ai-gateway/gemini/v1beta/models/"
        "system.ai.gemini-current:generateContent"
    )
    declaration = seen["body"]["tools"][0]["functionDeclarations"][0]
    assert declaration["name"] == "hermes_setup_probe"
    assert seen["body"]["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY", "allowedFunctionNames": ["hermes_setup_probe"],
    }


def test_build_config_creates_one_secret_free_unified_provider():
    from hermes_cli.model_setup_flows_databricks import build_databricks_config

    original = {
        "display": {"skin": "mono"},
        "model": {
            "default": "old-model", "provider": "openrouter",
            "base_url": "https://stale.example/v1", "api_mode": "codex_responses",
            "api_key": "STALE-SENTINEL",
        },
    }

    result = build_databricks_config(
        original, profile="profile; $(literal)", host="https://workspace.example/",
        model="system.ai.kimi-code",
    )

    assert original["model"]["provider"] == "openrouter"
    assert result["display"] == {"skin": "mono"}
    assert result["model"] == {"default": "system.ai.kimi-code", "provider": "custom:databricks"}
    assert result["providers"] == {
        "databricks": {
            "provider": "custom:databricks",
            "name": "Databricks Unity Gateway",
            "base_url": "https://workspace.example/ai-gateway/mlflow/v1",
            "transport": "chat_completions",
            "default_model": "system.ai.kimi-code",
            "models": ["system.ai.kimi-code"],
            "discover_models": False,
            "key_cmd": [
                "databricks", "auth", "token", "--profile", "profile; $(literal)", "--output", "json"
            ],
        }
    }
    assert "SENTINEL" not in json.dumps(result)


def test_build_config_accumulates_only_selected_verified_models():
    from hermes_cli.model_setup_flows_databricks import build_databricks_config

    result = build_databricks_config(
        {},
        profile="profile",
        host="https://workspace.example",
        model="system.ai.gpt-5-6-sol",
    )
    result = build_databricks_config(
        result,
        profile="profile",
        host="https://workspace.example",
        model="system.ai.claude-opus-4-8",
    )

    provider = result["providers"]["databricks"]
    assert provider["models"] == [
        "system.ai.gpt-5-6-sol",
        "system.ai.claude-opus-4-8",
    ]
    assert provider["discover_models"] is False


def test_build_config_removes_excluded_models_from_existing_inventory():
    from hermes_cli.model_setup_flows_databricks import build_databricks_config

    config = build_databricks_config(
        {},
        profile="profile",
        host="https://workspace.example",
        model="system.ai.claude-opus-4-8",
    )
    config["providers"]["databricks"]["models"].append("system.ai.gpt-oss-120b")

    result = build_databricks_config(
        config,
        profile="profile",
        host="https://workspace.example",
        model="system.ai.gpt-5-6-sol",
    )

    assert result["providers"]["databricks"]["models"] == [
        "system.ai.claude-opus-4-8",
        "system.ai.gpt-5-6-sol",
    ]


def test_databricks_opus_explicit_reasoning_requests_summarized_thinking():
    from agent.transports.anthropic import AnthropicTransport
    from hermes_constants import resolve_reasoning_config
    from providers import get_provider_profile

    model = "system.ai.claude-opus-4-8"
    profile = get_provider_profile("databricks")
    assert profile is not None
    assert profile.resolve_api_mode(model, "chat_completions") == "anthropic_messages"

    kwargs = AnthropicTransport().build_kwargs(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        reasoning_config=resolve_reasoning_config(
            {"agent": {"reasoning_effort": "medium"}}, model,
        ),
        base_url=profile.resolve_base_url(
            model, "https://workspace.example/ai-gateway/mlflow/v1",
        ),
    )

    assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kwargs["output_config"] == {"effort": "medium"}


def test_configured_databricks_collapses_to_one_setup_picker_row():
    from hermes_cli.main_provider_setup import _build_provider_picker_rows

    custom_provider_map = {
        "custom:databricks": {
            "name": "Databricks Unity Gateway",
            "base_url": "https://workspace.example/ai-gateway/mlflow/v1",
            "model": "system.ai.gpt-5-6-sol",
            "provider_key": "databricks",
        }
    }

    rows, default_idx = _build_provider_picker_rows(
        {}, "custom:databricks", {}, custom_provider_map
    )

    matches = [row for row in rows if "databricks" in row[0]]
    assert len(matches) == 1
    assert matches[0][0] == "custom:databricks"
    assert "workspace.example/ai-gateway/mlflow/v1" in matches[0][1]
    assert "currently active" in matches[0][1]
    assert rows[default_idx] == matches[0]


def test_setup_backed_custom_picker_merge_uses_generic_provider_key():
    from hermes_cli.main_provider_setup import _build_provider_picker_rows

    custom_provider_map = {
        "custom:fireworks": {
            "name": "Fireworks Enterprise",
            "base_url": "https://fireworks.invalid/inference/v1",
            "model": "enterprise-model",
            "provider_key": "fireworks",
        },
        "custom:legacy-relay": {
            "name": "Legacy Relay",
            "base_url": "https://legacy.invalid/v1",
            "model": "legacy-model",
            "provider_key": "",
        },
        "custom:private-relay": {
            "name": "Private Relay",
            "base_url": "https://private.invalid/v1",
            "model": "private-model",
            "provider_key": "private-relay",
        },
    }

    rows, default_idx = _build_provider_picker_rows(
        {}, "custom:fireworks", {}, custom_provider_map
    )
    keys = [row[0] for row in rows]
    merged = rows[keys.index("custom:fireworks")]

    assert keys.count("custom:fireworks") == 1
    assert "fireworks" not in keys
    assert "Fireworks Enterprise" in merged[1]
    assert "fireworks.invalid/inference/v1" in merged[1]
    assert "enterprise-model" in merged[1]
    assert "currently active" in merged[1]
    assert rows[default_idx] == merged
    assert "novita" in keys
    assert "custom:legacy-relay" in keys
    assert "custom:private-relay" in keys


def test_in_chat_inventory_uses_named_databricks_identity_without_live_discovery(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_catalog._fetch_manifest_with_fallback",
        lambda *_args, **_kwargs: None,
    )
    live_discovery = []
    monkeypatch.setattr(
        "hermes_cli.model_switch_providers._fetch_picker_live_models",
        lambda *_args, **_kwargs: live_discovery.append(True) or [],
    )
    from hermes_cli.config import save_config
    from hermes_cli.inventory import build_models_payload, load_picker_context
    from hermes_cli.model_setup_flows_databricks import build_databricks_config

    models = [
        "system.ai.claude-sonnet-current",
        "system.ai.glm-5-3",
        "system.ai.gpt-5-6-sol",
    ]
    config = {}
    for model in models:
        config = build_databricks_config(
            config, profile="profile", host="https://workspace.example", model=model,
        )
    config["providers"]["databricks"]["key_cmd"] = [
        sys.executable, "-c", "print('roundtrip-token')",
    ]
    save_config(config)

    payload = build_models_payload(load_picker_context())
    rows = [row for row in payload["providers"] if "databricks" in row["slug"]]

    assert len(rows) == 1
    assert rows[0]["slug"] == "custom:databricks"
    assert rows[0]["name"] == "Databricks Unity Gateway"
    assert set(rows[0]["models"]) == set(models)
    assert live_discovery == []


def test_in_chat_inventory_includes_discovered_catalog_without_marking_it_verified(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_catalog._fetch_manifest_with_fallback",
        lambda *_args, **_kwargs: None,
    )
    live_discovery = []
    monkeypatch.setattr(
        "hermes_cli.model_switch_providers._fetch_picker_live_models",
        lambda *_args, **_kwargs: live_discovery.append(True) or [],
    )
    from hermes_cli.config import save_config
    from hermes_cli.inventory import build_models_payload, load_picker_context
    from hermes_cli.model_setup_flows_databricks import build_databricks_config

    verified_model = "system.ai.verified"
    discovered_models = [
        verified_model,
        "system.ai.discovered-claude",
        "system.ai.discovered-gemini",
    ]
    config = build_databricks_config(
        {},
        profile="profile",
        host="https://workspace.example",
        model=verified_model,
        catalog_models=discovered_models,
    )
    config["providers"]["databricks"]["key_cmd"] = [
        sys.executable, "-c", "print('roundtrip-token')",
    ]
    save_config(config)

    row = next(
        item for item in build_models_payload(load_picker_context())["providers"]
        if item["slug"] == "custom:databricks"
    )

    assert config["providers"]["databricks"]["models"] == [verified_model]
    assert row["models"] == discovered_models
    assert live_discovery == []


def test_databricks_catalog_survives_generic_custom_provider_projection():
    from hermes_cli.config import get_compatible_custom_providers
    from hermes_cli.model_setup_flows_databricks import build_databricks_config

    config = build_databricks_config(
        {},
        profile="profile",
        host="https://workspace.example",
        model="system.ai.verified",
        catalog_models=["system.ai.verified", "system.ai.catalog-only"],
    )

    projected = get_compatible_custom_providers(config)

    assert len(projected) == 1
    assert projected[0]["provider_key"] == "databricks"
    assert projected[0]["catalog_models"] == [
        "system.ai.verified",
        "system.ai.catalog-only",
    ]


def test_in_chat_picker_labels_active_databricks_route(monkeypatch):
    from hermes_cli.cli_model_switch_mixin import _show_model_picker
    from hermes_cli.inventory import ConfigContext

    row = {
        "slug": "custom:databricks",
        "name": "Databricks Unity Gateway",
        "is_current": True,
        "models": ["system.ai.glm-5-3"],
    }
    monkeypatch.setattr(
        "hermes_cli.inventory.build_models_payload",
        lambda *_args, **_kwargs: {"providers": [row]},
    )
    opened = []
    cli = SimpleNamespace(
        model="system.ai.glm-5-3",
        provider="custom",
        requested_provider="custom:databricks",
        _open_model_picker=lambda *args, **kwargs: opened.append((args, kwargs)),
    )
    context = ConfigContext(
        current_provider="custom:databricks",
        current_model=cli.model,
        current_base_url="https://workspace.example/ai-gateway/mlflow/v1",
        user_providers={},
        custom_providers=[],
    )

    _show_model_picker(cli, context, False)

    assert opened[0][0][2] == "Databricks Unity Gateway"


def test_in_chat_switch_preserves_databricks_route_and_callable_across_routes(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_catalog._fetch_manifest_with_fallback",
        lambda *_args, **_kwargs: None,
    )
    from agent.command_token_source import CommandTokenSource
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin, _switch_model_from
    from hermes_cli.config import save_config
    from hermes_cli.inventory import build_models_payload, load_picker_context
    from hermes_cli.model_setup_flows_databricks import build_databricks_config

    claude = "system.ai.claude-sonnet-current"
    glm = "system.ai.glm-5-3"
    config = build_databricks_config(
        {}, profile="profile", host="https://workspace.example", model=claude,
    )
    config = build_databricks_config(
        config, profile="profile", host="https://workspace.example", model=glm,
    )
    config["providers"]["databricks"]["key_cmd"] = [
        sys.executable, "-c", "print('roundtrip-token')",
    ]
    save_config(config)
    context = load_picker_context()
    row = next(
        item for item in build_models_payload(context)["providers"]
        if item["name"] == "Databricks Unity Gateway"
    )

    class _LiveAgent:
        def __init__(self):
            self.model = claude
            self.provider = "custom"
            self.requested_provider = "custom:databricks"
            self.api_key = None

        def switch_model(
            self, *, new_model, new_provider, new_requested_provider="", api_key,
            base_url, api_mode, capabilities,
        ):
            self.model = new_model
            self.provider = new_provider
            self.requested_provider = new_requested_provider or new_provider
            self.api_key = api_key

    class _CLI(CLIModelSwitchMixin, CLIAgentSetupMixin):
        def _normalize_model_for_provider(self, _provider):
            return False

    cli = _CLI()
    cli.model = claude
    cli.provider = "custom"
    cli.requested_provider = "custom:databricks"
    cli.api_key = None
    cli.base_url = "https://workspace.example/ai-gateway/anthropic"
    cli.api_mode = "anthropic_messages"
    cli.acp_command = None
    cli.acp_args = []
    cli._explicit_api_key = None
    cli._explicit_base_url = None
    cli._credential_pool = None
    cli._fallback_model = []
    cli._active_agent_route_signature = None
    cli.agent = _LiveAgent()

    switched = _switch_model_from(
        cli, glm, is_global=False, explicit_provider=row["slug"],
        user_providers=context.user_providers, custom_providers=context.custom_providers,
    )

    assert switched.success is True
    assert switched.provider_label == "Databricks Unity Gateway"
    assert switched.runtime_provider == "custom"
    assert switched.requested_provider == "custom:databricks"
    assert isinstance(switched.api_key, CommandTokenSource)
    cli._stage_and_swap_model(switched, claude)
    assert cli.provider == cli.agent.provider == "custom"
    assert cli.requested_provider == cli.agent.requested_provider == "custom:databricks"
    assert cli.api_key is cli.agent.api_key is switched.api_key
    assert cli._ensure_runtime_credentials() is True
    assert cli.api_key is switched.api_key

    switched_back = _switch_model_from(
        cli, claude, is_global=False, explicit_provider=row["slug"],
        user_providers=context.user_providers, custom_providers=context.custom_providers,
    )

    assert switched_back.success is True
    assert switched_back.api_mode == "anthropic_messages"
    assert switched_back.base_url == "https://workspace.example/ai-gateway/anthropic"
    cli._stage_and_swap_model(switched_back, glm)
    assert cli.provider == cli.agent.provider == "custom"
    assert cli.requested_provider == cli.agent.requested_provider == "custom:databricks"


def test_verified_inventory_allows_explicit_switch_without_models_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.config import save_config
    from hermes_cli.inventory import load_picker_context
    from hermes_cli.model_setup_flows_databricks import build_databricks_config
    from hermes_cli.model_switch import switch_model

    config = build_databricks_config(
        {}, profile="profile", host="https://workspace.example",
        model="system.ai.gpt-5-6-sol",
    )
    config = build_databricks_config(
        config, profile="profile", host="https://workspace.example",
        model="system.ai.claude-sonnet-current",
    )
    config["providers"]["databricks"]["key_cmd"] = [
        sys.executable, "-c", "print('roundtrip-token')",
    ]
    save_config(config)
    context = load_picker_context()

    result = switch_model(
        raw_input="system.ai.gpt-5-6-sol",
        current_provider="custom:databricks",
        current_model="system.ai.claude-sonnet-current",
        current_base_url="https://workspace.example/ai-gateway/anthropic",
        current_api_key="",
        is_global=False,
        explicit_provider="custom:databricks",
        user_providers=context.user_providers,
        custom_providers=context.custom_providers,
    )

    assert result.success is True
    assert result.target_provider == "custom:databricks"
    assert result.api_mode == "codex_responses"
    assert result.base_url == "https://workspace.example/ai-gateway/mlflow/v1"


def test_hot_switch_preflights_catalog_model_and_persists_verified_receipt(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_cli.model_setup_flows_databricks as dbx
    from hermes_cli.config import load_config, save_config

    verified_model = "system.ai.verified"
    catalog_model = "system.ai.catalog-only"
    config = dbx.build_databricks_config(
        {},
        profile="profile",
        host="https://workspace.example",
        model=verified_model,
        catalog_models=[verified_model, catalog_model],
    )
    save_config(config)
    token_calls = []
    preflight_calls = []
    credential = lambda: token_calls.append(True) or "SENTINEL_TOKEN"
    monkeypatch.setattr(
        dbx,
        "preflight_model",
        lambda host, token, model: (
            preflight_calls.append((host, token, model)) or True,
            None,
        ),
    )

    ok, error = dbx.ensure_databricks_model_verified(
        requested_provider="custom:databricks",
        model=catalog_model,
        credential=credential,
    )

    persisted = load_config()["providers"]["databricks"]
    assert ok is True
    assert error is None
    assert token_calls == [True]
    assert preflight_calls == [
        ("https://workspace.example", "SENTINEL_TOKEN", catalog_model),
    ]
    assert persisted["models"] == [verified_model, catalog_model]
    assert persisted["catalog_models"] == [verified_model, catalog_model]
    assert "SENTINEL_TOKEN" not in repr(persisted)


def test_hot_switch_does_not_swap_when_databricks_preflight_fails(monkeypatch):
    import hermes_cli.cli_model_switch_mixin as model_switch_mixin
    import hermes_cli.model_setup_flows_databricks as dbx

    monkeypatch.setattr(
        dbx,
        "ensure_databricks_model_verified",
        lambda **_kwargs: (False, "Selected Databricks model is not compatible with Hermes tools."),
    )
    monkeypatch.setattr(model_switch_mixin, "_print_switch_summary", lambda *_args, **_kwargs: None)
    swapped = []
    cli = SimpleNamespace(
        model="system.ai.verified",
        _snapshot_model_runtime=lambda: {"model": "system.ai.verified"},
        _stage_and_swap_model=lambda result, old_model: swapped.append(
            (result.new_model, old_model)
        ) or True,
    )
    result = SimpleNamespace(
        success=True,
        new_model="system.ai.catalog-only",
        target_provider="custom:databricks",
        requested_provider="custom:databricks",
        runtime_provider="custom",
        api_key=lambda: "SENTINEL_TOKEN",
        base_url="https://workspace.example/ai-gateway/mlflow/v1",
        api_mode="chat_completions",
        provider_label="Databricks Unity Gateway",
        warning_message="",
        model_info=None,
    )

    model_switch_mixin._commit_model_switch(
        cli,
        result,
        persist_global=False,
        one_turn=True,
    )

    assert swapped == []


def test_verified_inventory_resets_when_oauth_profile_changes():
    from hermes_cli.model_setup_flows_databricks import build_databricks_config

    config = build_databricks_config(
        {}, profile="profile-a", host="https://workspace.example",
        model="system.ai.gpt-5-6-sol",
    )
    config = build_databricks_config(
        config, profile="profile-b", host="https://workspace.example",
        model="system.ai.claude-sonnet-current",
    )

    assert config["providers"]["databricks"]["models"] == [
        "system.ai.claude-sonnet-current",
    ]


def test_build_config_disables_reasoning_only_for_gpt_5_6():
    from hermes_cli.model_setup_flows_databricks import build_databricks_config

    sol = build_databricks_config(
        {}, profile="profile", host="https://workspace.example",
        model="system.ai.gpt-5-6-sol",
    )
    glm = build_databricks_config(
        {}, profile="profile", host="https://workspace.example",
        model="system.ai.glm-5-3",
    )

    assert "extra_body" not in sol["providers"]["databricks"]
    assert "extra_body" not in glm["providers"]["databricks"]
    assert sol["providers"]["databricks"]["transport"] == "codex_responses"
    assert glm["providers"]["databricks"]["transport"] == "chat_completions"


def test_model_flow_saves_once_after_auth_discovery_and_selection(monkeypatch, capsys):
    import hermes_cli.model_setup_flows_databricks as dbx

    config = {"display": {"skin": "mono"}, "model": {"default": "old", "provider": "openrouter"}}
    saved = []
    model_preflights = []
    deactivated = []
    monkeypatch.setattr(
        dbx, "load_u2m_profiles",
        lambda: ([{"name": "profile; $(literal)", "host": "https://workspace.example"}], None),
    )
    _select_first_profile(monkeypatch)
    monkeypatch.setattr(dbx, "preflight_profile", lambda _profile: ("SENTINEL", None))
    monkeypatch.setattr(dbx, "discover_models", lambda _host, _token: (["system.ai.kimi-code"], None))
    monkeypatch.setattr(
        dbx, "preflight_model",
        lambda host, token, model: (
            model_preflights.append((host, token, model)) or True,
            None,
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        lambda models, **kwargs: models[0],
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda value: saved.append(value))
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: deactivated.append(True))

    dbx._model_flow_databricks(config, current_model="old")

    assert len(saved) == 1
    assert saved[0]["model"] == {"default": "system.ai.kimi-code", "provider": "custom:databricks"}
    assert saved[0]["providers"]["databricks"]["key_cmd"][-3:] == [
        "profile; $(literal)", "--output", "json"
    ]
    assert saved[0]["providers"]["databricks"]["models"] == [
        "system.ai.kimi-code"
    ]
    assert model_preflights == [
        ("https://workspace.example", "SENTINEL", "system.ai.kimi-code")
    ]
    assert deactivated == [True]
    assert "SENTINEL" not in capsys.readouterr().out


def test_model_flow_preserves_prior_receipts_and_adds_only_selected_model(monkeypatch, capsys):
    import hermes_cli.model_setup_flows_databricks as dbx

    discovered = [
        "system.ai.good-selected",
        "system.ai.incompatible",
        "system.ai.good-other",
    ]
    token = "SENTINEL_BEARER"
    saved = []
    preflighted = []
    deactivated = []
    monkeypatch.setattr(
        dbx, "load_u2m_profiles",
        lambda: ([{"name": "profile", "host": "https://workspace.example"}], None),
    )
    _select_first_profile(monkeypatch)
    monkeypatch.setattr(dbx, "preflight_profile", lambda _profile: (token, None))
    monkeypatch.setattr(dbx, "discover_models", lambda _host, _token: (discovered, None))
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection", lambda models, **_kwargs: models[0]
    )

    def preflight(_host, _token, model):
        preflighted.append(model)
        if model == "system.ai.incompatible":
            return False, "Selected model service failed the tool preflight."
        return True, None

    monkeypatch.setattr(dbx, "preflight_model", preflight)
    monkeypatch.setattr("hermes_cli.config.save_config", lambda value: saved.append(value))
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: deactivated.append(True))

    config = dbx.build_databricks_config(
        {},
        profile="profile",
        host="https://workspace.example",
        model="system.ai.already-verified",
    )
    dbx._model_flow_databricks(config, current_model="")

    assert len(saved) == 1
    assert saved[0]["providers"]["databricks"]["models"] == [
        "system.ai.already-verified",
        "system.ai.good-selected",
    ]
    assert preflighted == ["system.ai.good-selected"]
    assert saved[0]["model"]["default"] == "system.ai.good-selected"
    assert deactivated == [True]
    assert token not in repr(saved[0])
    assert token not in capsys.readouterr().out


def test_model_flow_preflights_only_selected_service_from_large_inventory(monkeypatch, capsys):
    import hermes_cli.model_setup_flows_databricks as dbx

    discovered = [f"system.ai.private-candidate-{index}" for index in range(52)]
    selected_model = discovered[17]
    preflighted = []
    saved = []
    monkeypatch.setattr(
        dbx, "load_u2m_profiles",
        lambda: ([{"name": "private-profile", "host": "https://private.invalid"}], None),
    )
    _select_first_profile(monkeypatch)
    monkeypatch.setattr(dbx, "preflight_profile", lambda _profile: ("PRIVATE_TOKEN", None))
    monkeypatch.setattr(dbx, "discover_models", lambda *_args: (discovered, None))
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection", lambda *_args, **_kwargs: selected_model
    )

    monkeypatch.setattr(
        dbx,
        "preflight_model",
        lambda _host, _token, model: (preflighted.append(model) or True, None),
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda value: saved.append(value))
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)

    dbx._model_flow_databricks({"model": {}}, current_model="")

    output = capsys.readouterr().out
    assert preflighted == [selected_model]
    assert saved[0]["providers"]["databricks"]["models"] == [selected_model]
    assert saved[0]["providers"]["databricks"]["catalog_models"] == discovered
    assert "Validating all" not in output
    assert "Validating service" not in output
    assert "private-profile" not in output
    assert "private.invalid" not in output
    assert "PRIVATE_TOKEN" not in output


def test_model_flow_uses_curses_to_select_from_multiple_profiles(monkeypatch):
    import hermes_cli.model_setup_flows_databricks as dbx

    profiles = [
        {"name": "first", "host": "https://first.example"},
        {"name": "second", "host": "https://second.example"},
    ]
    seen = {}
    monkeypatch.setattr(dbx, "load_u2m_profiles", lambda: (profiles, None))
    monkeypatch.setattr(
        "hermes_cli.model_setup_flows_common._curses_choice",
        lambda title, rows, default: seen.update(title=title, rows=rows, default=default) or 1,
    )
    monkeypatch.setattr(
        dbx, "preflight_profile", lambda profile: seen.update(profile=profile) or ("SENTINEL", None),
    )
    monkeypatch.setattr(
        dbx, "discover_models",
        lambda host, _token: seen.update(host=host) or (["system.ai.kimi-code"], None),
    )
    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", lambda models, **_kwargs: models[0])
    monkeypatch.setattr(dbx, "preflight_model", lambda _host, _token, _model: (True, None))
    monkeypatch.setattr("hermes_cli.config.save_config", lambda _value: None)
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)

    dbx._model_flow_databricks({"model": {}}, current_model="")

    assert seen["rows"] == [
        "first (https://first.example)",
        "second (https://second.example)",
    ]
    assert seen["profile"] == "second"
    assert seen["host"] == "https://second.example"


def test_model_flow_requires_explicit_selection_for_single_profile(monkeypatch, capsys):
    import hermes_cli.model_setup_flows_databricks as dbx

    seen = {}
    monkeypatch.setattr(
        dbx,
        "load_u2m_profiles",
        lambda: ([{"name": "only", "host": "https://only.example"}], None),
    )
    monkeypatch.setattr(
        "hermes_cli.model_setup_flows_common._curses_choice",
        lambda title, rows, default: seen.update(
            title=title, rows=rows, default=default,
        ) or None,
    )
    monkeypatch.setattr(
        dbx,
        "preflight_profile",
        lambda _profile: (_ for _ in ()).throw(
            AssertionError("profile must not be used without explicit selection")
        ),
    )

    dbx._model_flow_databricks({"model": {}}, current_model="")

    assert seen["rows"] == ["only (https://only.example)"]
    assert seen["default"] == 0
    assert "No change." in capsys.readouterr().out


def test_model_flow_reprompts_after_incompatible_model(monkeypatch):
    import hermes_cli.model_setup_flows_databricks as dbx

    saved = []
    choices = iter(["system.ai.bad", "system.ai.good"])
    offered = []
    preflighted = []
    monkeypatch.setattr(
        dbx, "load_u2m_profiles",
        lambda: ([{"name": "profile", "host": "https://workspace.example"}], None),
    )
    _select_first_profile(monkeypatch)
    monkeypatch.setattr(dbx, "preflight_profile", lambda _profile: ("SENTINEL", None))
    monkeypatch.setattr(
        dbx, "discover_models",
        lambda _host, _token: (
            ["system.ai.bad", "system.ai.good", "system.ai.good-other"], None
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        lambda models, **_kwargs: offered.append(list(models)) or next(choices),
    )
    monkeypatch.setattr(
        dbx,
        "preflight_model",
        lambda _host, _token, model: (
            preflighted.append(model)
            or (
                (False, "Selected model service failed the tool preflight.")
                if model == "system.ai.bad" else (True, None)
            )
        ),
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda value: saved.append(value))
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)

    dbx._model_flow_databricks({"model": {}}, current_model="")

    assert offered == [
        ["system.ai.bad", "system.ai.good", "system.ai.good-other"],
        ["system.ai.good", "system.ai.good-other"],
    ]
    assert preflighted == ["system.ai.bad", "system.ai.good"]
    assert saved[0]["model"]["default"] == "system.ai.good"
    assert saved[0]["providers"]["databricks"]["models"] == ["system.ai.good"]
    assert saved[0]["providers"]["databricks"]["catalog_models"] == [
        "system.ai.good",
        "system.ai.good-other",
    ]


def test_model_flow_rejects_manually_entered_gpt_oss(monkeypatch):
    import hermes_cli.model_setup_flows_databricks as dbx

    saved = []
    preflighted = []
    choices = iter(["system.ai.gpt-oss-120b", "system.ai.good"])
    monkeypatch.setattr(
        dbx, "load_u2m_profiles",
        lambda: ([{"name": "profile", "host": "https://workspace.example"}], None),
    )
    _select_first_profile(monkeypatch)
    monkeypatch.setattr(dbx, "preflight_profile", lambda _profile: ("SENTINEL", None))
    monkeypatch.setattr(dbx, "discover_models", lambda *_args: (["system.ai.good"], None))
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection", lambda *_args, **_kwargs: next(choices)
    )
    monkeypatch.setattr(
        dbx,
        "preflight_model",
        lambda _host, _token, model: (preflighted.append(model) or True, None),
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda value: saved.append(value))
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)

    dbx._model_flow_databricks({"model": {}}, current_model="")

    assert preflighted == ["system.ai.good"]
    assert saved[0]["model"]["default"] == "system.ai.good"


def test_model_flow_aborts_on_transient_preflight_failure(monkeypatch, capsys):
    import hermes_cli.model_setup_flows_databricks as dbx

    saved = []
    transient_error = getattr(dbx, "_GatewayRequestError")
    monkeypatch.setattr(
        dbx, "load_u2m_profiles",
        lambda: ([{"name": "profile", "host": "https://workspace.example"}], None),
    )
    _select_first_profile(monkeypatch)
    monkeypatch.setattr(dbx, "preflight_profile", lambda _profile: ("SENTINEL", None))
    monkeypatch.setattr(
        dbx, "discover_models", lambda _host, _token: (["system.ai.model"], None)
    )
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection", lambda models, **_kwargs: models[0]
    )
    monkeypatch.setattr(
        dbx, "preflight_model", lambda *_args: (_ for _ in ()).throw(transient_error(429))
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda value: saved.append(value))
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda: None)

    dbx._model_flow_databricks({"model": {}}, current_model="")

    assert saved == []
    assert "temporarily unavailable" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("status", "is_transient"),
    [
        (None, True),
        (408, True),
        (429, True),
        (501, True),
        (400, False),
    ],
)
def test_preflight_model_classifies_transient_failures(monkeypatch, status, is_transient):
    import hermes_cli.model_setup_flows_databricks as dbx

    monkeypatch.setattr(
        dbx,
        "_urlopen_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(dbx._GatewayRequestError(status)),
    )
    if is_transient:
        with pytest.raises(dbx._GatewayRequestError):
            dbx.preflight_model(
                "https://workspace.example", "SENTINEL", "system.ai.private-candidate"
            )
    else:
        ok, error = dbx.preflight_model(
            "https://workspace.example", "SENTINEL", "system.ai.private-candidate"
        )
        assert ok is False
        assert error == "Selected model service failed the tool preflight."


def test_databricks_appears_in_provider_picker_catalog():
    from hermes_cli.models_catalog_static import CANONICAL_PROVIDERS

    matches = [entry for entry in CANONICAL_PROVIDERS if entry.slug == "databricks"]

    assert len(matches) == 1
    assert matches[0].label == "Databricks Unity Gateway"


def test_databricks_has_metadata_only_provider_profile():
    from providers import get_provider_profile

    profile = get_provider_profile("databricks")

    assert profile is not None
    assert profile.auth_type == "oauth_external"
    assert profile.base_url == ""
    assert profile.supports_stream_options is False


def test_databricks_provider_dispatches_to_bespoke_setup_flow(monkeypatch):
    import hermes_cli.main as main

    seen = []
    monkeypatch.setattr(main, "_model_flow_databricks", lambda config, model: seen.append((config, model)))
    config = {"model": {}}

    main._PROVIDER_MODEL_FLOWS["databricks"](config, "current", object())

    assert seen == [(config, "current")]


def test_cli_failure_does_not_expose_child_output(monkeypatch):
    import hermes_cli.model_setup_flows_databricks as dbx

    monkeypatch.setattr(
        dbx.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="SECRET_STDOUT", stderr="SECRET_STDERR"
        ),
    )

    profiles, error = dbx.load_u2m_profiles()

    assert profiles == []
    assert error is not None
    assert "SECRET" not in error


def test_model_flow_auth_failure_is_atomic_and_redacted(monkeypatch, capsys):
    from copy import deepcopy
    import hermes_cli.model_setup_flows_databricks as dbx

    config = {"model": {"provider": "openrouter", "default": "old"}, "display": {"skin": "mono"}}
    before = deepcopy(config)
    def fail_write(*_args, **_kwargs):
        raise AssertionError("must not write state")

    monkeypatch.setattr(
        dbx,
        "load_u2m_profiles",
        lambda: ([{"name": "literal; profile", "host": "https://workspace.example"}], None),
    )
    _select_first_profile(monkeypatch)
    monkeypatch.setattr(dbx, "preflight_profile", lambda _name: (None, "OAuth preflight failed."))
    monkeypatch.setattr(
        "hermes_cli.config.save_config", fail_write
    )
    monkeypatch.setattr(
        "hermes_cli.auth.deactivate_provider", fail_write
    )

    getattr(dbx, "_model_flow_databricks")(config, "old")

    assert config == before
    assert "literal; profile" not in capsys.readouterr().out


def test_saved_config_round_trips_to_callable_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.config import load_config, save_config
    from hermes_cli.model_setup_flows_databricks import build_databricks_config
    from hermes_cli.runtime_provider import resolve_runtime_provider

    script = "import json; print(json.dumps({'access_token':'roundtrip-token'}))"
    config = build_databricks_config(
        {}, profile="literal; profile", host="https://workspace.example",
        model="system.ai.kimi-code",
    )
    config["providers"]["databricks"]["key_cmd"] = [
        sys.executable, "-c", script, "literal; profile"
    ]
    save_config(config)

    load_config()
    runtime = resolve_runtime_provider(requested=None)

    assert runtime["requested_provider"] == "custom:databricks"
    assert runtime["api_mode"] == "chat_completions"
    assert callable(runtime["api_key"])
    assert runtime["api_key"]() == "roundtrip-token"


def test_databricks_profile_selects_transport_by_effective_model():
    from providers import get_provider_profile

    profile = get_provider_profile("databricks")
    assert profile is not None

    assert profile.resolve_api_mode("system.ai.gpt-5-6-sol", "chat_completions") == "codex_responses"
    assert profile.resolve_api_mode("system.ai.gpt-5-6-terra", "chat_completions") == "codex_responses"
    assert profile.resolve_api_mode("system.ai.gpt-5-6-luna", "chat_completions") == "codex_responses"
    assert profile.resolve_api_mode("system.ai.claude-sonnet-current", "chat_completions") == "anthropic_messages"
    assert profile.resolve_api_mode("system.ai.gemini-current", "codex_responses") == "chat_completions"
    assert profile.resolve_api_mode("system.ai.glm-5-3", "codex_responses") == "chat_completions"
    assert profile.resolve_api_mode("system.ai.gpt-5-6-solar", "codex_responses") == "chat_completions"
    assert profile.resolve_api_mode("system.ai.my-gpt-5-6-sol-copy", "codex_responses") == "chat_completions"
    assert profile.resolve_api_mode("system.ai.my-claude-copy", "codex_responses") == "chat_completions"
    assert profile.resolve_api_mode("system.ai.my-gemini-copy", "codex_responses") == "chat_completions"

    configured = "https://workspace.example/ai-gateway/mlflow/v1"
    assert profile.resolve_base_url("system.ai.gpt-5-6-sol", configured) == configured
    assert profile.resolve_base_url(
        "system.ai.claude-sonnet-current", configured,
    ) == "https://workspace.example/ai-gateway/anthropic"
    assert profile.resolve_base_url(
        "system.ai.gemini-current", configured,
    ) == "https://workspace.example/ai-gateway/gemini/v1beta"
    assert profile.resolve_base_url("system.ai.glm-5-3", configured) == configured
    assert profile.resolve_base_url("system.ai.my-gemini-copy", configured) == configured


def test_saved_provider_target_model_does_not_inherit_sol_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.config import save_config
    from hermes_cli.model_setup_flows_databricks import build_databricks_config
    from hermes_cli.runtime_provider import resolve_runtime_provider

    config = build_databricks_config(
        {}, profile="profile", host="https://workspace.example",
        model="system.ai.gpt-5-6-sol",
    )
    config["providers"]["databricks"]["key_cmd"] = [
        sys.executable, "-c", "print('roundtrip-token')",
    ]
    save_config(config)

    runtime = resolve_runtime_provider(
        requested="custom:databricks", target_model="system.ai.glm-5-3",
    )
    sol_runtime = resolve_runtime_provider(
        requested="custom:databricks", target_model="system.ai.gpt-5-6-sol",
    )

    assert runtime["model"] == "system.ai.glm-5-3"
    assert runtime["api_mode"] == "chat_completions"
    assert not runtime.get("request_overrides")
    assert sol_runtime["api_mode"] == "codex_responses"


def test_saved_provider_target_models_recompute_databricks_native_routes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.config import save_config
    from hermes_cli.model_setup_flows_databricks import build_databricks_config
    from hermes_cli.runtime_provider import resolve_runtime_provider

    config = build_databricks_config(
        {}, profile="profile", host="https://workspace.example",
        model="system.ai.gpt-5-6-sol",
    )
    config["providers"]["databricks"]["key_cmd"] = [
        sys.executable, "-c", "print('roundtrip-token')",
    ]
    save_config(config)

    expected = {
        "system.ai.claude-sonnet-current": (
            "anthropic_messages", "https://workspace.example/ai-gateway/anthropic",
        ),
        "system.ai.gemini-current": (
            "chat_completions", "https://workspace.example/ai-gateway/gemini/v1beta",
        ),
        "system.ai.glm-5-3": (
            "chat_completions", "https://workspace.example/ai-gateway/mlflow/v1",
        ),
        "system.ai.gpt-5-6-sol": (
            "codex_responses", "https://workspace.example/ai-gateway/mlflow/v1",
        ),
    }

    for model, (api_mode, base_url) in expected.items():
        runtime = resolve_runtime_provider(
            requested="custom:databricks", target_model=model,
        )
        assert runtime["model"] == model
        assert runtime["api_mode"] == api_mode
        assert runtime["base_url"] == base_url


def test_named_databricks_gemini_runtime_supplies_existing_native_client():
    from agent.agent_runtime_helpers import _provider_supplied_client
    from agent.gemini_native_adapter import GeminiNativeClient

    token_provider = lambda: "refreshable-token"
    agent = SimpleNamespace(
        provider="custom",
        requested_provider="custom:databricks",
    )

    client = _provider_supplied_client(agent, {
        "api_key": token_provider,
        "base_url": "https://workspace.example/ai-gateway/gemini/v1beta",
    })

    assert isinstance(client, GeminiNativeClient)
    assert client.base_url == "https://workspace.example/ai-gateway/gemini/v1beta"
    assert client._bearer_token_provider is token_provider


def test_named_databricks_auxiliary_recomputes_native_routes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.auxiliary_client import AnthropicAuxiliaryClient, resolve_provider_client
    from agent.gemini_native_adapter import GeminiNativeClient
    from hermes_cli.config import save_config
    from hermes_cli.model_setup_flows_databricks import build_databricks_config

    config = build_databricks_config(
        {}, profile="profile", host="https://workspace.example",
        model="system.ai.gpt-5-6-sol",
    )
    config["providers"]["databricks"]["key_cmd"] = [
        sys.executable, "-c", "print('roundtrip-token')",
    ]
    save_config(config)

    anthropic_client, _ = resolve_provider_client(
        "custom:databricks", "system.ai.claude-sonnet-current",
        api_mode="anthropic_messages",
    )
    gemini_client, _ = resolve_provider_client(
        "custom:databricks", "system.ai.gemini-current",
        api_mode="chat_completions",
    )

    assert isinstance(anthropic_client, AnthropicAuxiliaryClient)
    assert anthropic_client.base_url == "https://workspace.example/ai-gateway/anthropic"
    assert isinstance(gemini_client, GeminiNativeClient)
    assert gemini_client.base_url == "https://workspace.example/ai-gateway/gemini/v1beta"
    assert callable(gemini_client._bearer_token_provider)
