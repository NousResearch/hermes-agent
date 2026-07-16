from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "sillytavern"


def load_plugin():
    package_name = "sillytavern_test_plugin"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


class _Context:
    def __init__(self):
        self.tools = []
        self.commands = []
        self.cli_commands = {}

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_command(self, *args, **kwargs):
        self.commands.append((args, kwargs))

    def register_cli_command(self, name, **kwargs):
        self.cli_commands[name] = kwargs


def test_register_exposes_sillytavern_to_agents():
    module = load_plugin()
    context = _Context()
    module.register(context)

    assert {tool["name"] for tool in context.tools} == {
        "sillytavern_capabilities",
        "sillytavern_status",
        "sillytavern_start",
        "sillytavern_stop",
        "sillytavern_generate",
    }
    assert {tool["toolset"] for tool in context.tools} == {"sillytavern"}
    assert "sillytavern" in context.cli_commands


def test_start_requires_configuration_and_acknowledgement(monkeypatch):
    module = load_plugin()
    monkeypatch.setattr(
        module.core,
        "_load_entry",
        lambda: {"allow_process_control": False},
    )
    payload = module.core.start_server({"acknowledge_side_effects": True})
    assert payload["success"] is False
    assert "allow_process_control" in payload["error"]

    monkeypatch.setattr(
        module.core,
        "_load_entry",
        lambda: {"allow_process_control": True},
    )
    payload = module.core.start_server({})
    assert payload["success"] is False
    assert "acknowledge_side_effects" in payload["error"]


def test_generate_normalizes_prompt_and_provider_response(monkeypatch):
    module = load_plugin()
    monkeypatch.setattr(
        module.core,
        "_load_entry",
        lambda: {
            "allow_network": True,
            "model": "test-model",
            "chat_completion_source": "openai",
        },
    )
    monkeypatch.setattr(
        module.core,
        "status_payload",
        lambda _values: {"healthy": True, "base_url": "http://127.0.0.1:8000"},
    )
    responses = iter(
        [
            {"ok": True, "status_code": 200, "data": {"token": "csrf-test"}},
            {
                "ok": True,
                "status_code": 200,
                "data": {
                    "model": "test-model",
                    "choices": [
                        {
                            "message": {"content": "hello from SillyTavern"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"total_tokens": 3},
                },
            },
        ]
    )
    monkeypatch.setattr(module.core, "_http_json", lambda *args, **kwargs: next(responses))

    payload = module.core.generate(
        {"prompt": "hello", "acknowledge_side_effects": True}
    )

    assert payload == {
        "success": True,
        "status": "completed",
        "reply": "hello from SillyTavern",
        "model": "test-model",
        "finish_reason": "stop",
        "usage": {"total_tokens": 3},
        "request_source": "openai",
    }


def test_generate_rejects_invalid_message_role():
    module = load_plugin()
    payload = module.core.generate(
        {
            "messages": [{"role": "tool", "content": "not allowed"}],
            "acknowledge_side_effects": True,
        }
    )
    assert payload["success"] is False
    assert "role" in payload["error"]


def test_stop_refuses_unmanaged_process(monkeypatch, tmp_path):
    module = load_plugin()
    state = tmp_path / "server.json"
    state.write_text(
        json.dumps({"pid": 1234, "data_root": str(tmp_path / "data")}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module.core, "state_file", lambda: state)
    monkeypatch.setattr(module.core, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(module.core, "_process_matches", lambda *_args: False)

    payload = module.core.stop_server({"acknowledge_side_effects": True})

    assert payload["success"] is False
    assert "unmanaged" in payload["error"]
