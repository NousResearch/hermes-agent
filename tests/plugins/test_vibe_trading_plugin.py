"""Tests for the Vibe-Trading Hermes plugin."""

import importlib.util
import json
import sys
from pathlib import Path
from urllib.error import URLError


def _load_plugin():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_dir = repo_root / "plugins" / "vibe-trading"
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.vibe_trading",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hermes_plugins.vibe_trading"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeResponse:
    def __init__(self, body, status=200):
        self.body = body.encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.body


class FakeContext:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, toolset, schema, handler, **kwargs):
        self.tools[name] = {
            "toolset": toolset,
            "schema": schema,
            "handler": handler,
            "kwargs": kwargs,
        }


def test_request_json_uses_base_url_and_get(monkeypatch):
    plugin = _load_plugin()
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["timeout"] = timeout
        return FakeResponse('{"ok": true}')

    monkeypatch.setenv("VIBE_TRADING_BASE_URL", "http://vibe.local:8899/")
    monkeypatch.setattr(plugin.urllib.request, "urlopen", fake_urlopen)

    result = json.loads(plugin._request_json("GET", "/health"))

    assert result == {"ok": True}
    assert seen == {
        "url": "http://vibe.local:8899/health",
        "method": "GET",
        "timeout": 20.0,
    }


def test_request_json_posts_json_body(monkeypatch):
    plugin = _load_plugin()
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["body"] = request.data.decode("utf-8")
        seen["content_type"] = request.headers["Content-type"]
        return FakeResponse('{"run_id": "r1"}')

    monkeypatch.setenv("VIBE_TRADING_BASE_URL", "http://vibe.local:8899")
    monkeypatch.setattr(plugin.urllib.request, "urlopen", fake_urlopen)

    result = json.loads(plugin._request_json("POST", "/swarm/runs", {"preset_name": "risk_committee"}))

    assert result == {"run_id": "r1"}
    assert seen == {
        "url": "http://vibe.local:8899/swarm/runs",
        "method": "POST",
        "body": '{"preset_name": "risk_committee"}',
        "content_type": "application/json",
    }


def test_request_json_returns_error_payload_on_network_error(monkeypatch):
    plugin = _load_plugin()

    def fake_urlopen(request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr(plugin.urllib.request, "urlopen", fake_urlopen)

    result = json.loads(plugin._request_json("GET", "/health"))

    assert result["success"] is False
    assert result["error_type"] == "URLError"
    assert "connection refused" in result["error"]


def test_register_exposes_first_version_tools():
    plugin = _load_plugin()
    ctx = FakeContext()

    plugin.register(ctx)

    assert set(ctx.tools) == {
        "vibe_health",
        "vibe_list_skills",
        "vibe_list_swarm_presets",
        "vibe_run_swarm",
        "vibe_get_swarm_run",
        "vibe_create_session",
        "vibe_send_message",
        "vibe_get_run_result",
        "vibe_list_runs",
    }
    assert ctx.tools["vibe_health"]["toolset"] == "vibe-trading"
    assert callable(ctx.tools["vibe_run_swarm"]["handler"])

