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


def test_register_exposes_phase_one_tools():
    plugin = _load_plugin()
    ctx = FakeContext()

    plugin.register(ctx)

    assert set(ctx.tools) == {
        "vibe_ask",
        "vibe_ask_ashare",
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


def test_vibe_ask_creates_session_sends_question_and_returns_assistant(monkeypatch):
    plugin = _load_plugin()
    calls = []

    def fake_request_json(method, path, payload=None, query=None):
        calls.append({"method": method, "path": path, "payload": payload, "query": query})
        if method == "POST" and path == "/sessions":
            return json.dumps({"session_id": "s1"})
        if method == "POST" and path == "/sessions/s1/messages":
            return json.dumps({"message_id": "m1", "attempt_id": "a1"})
        if method == "GET" and path == "/sessions/s1/messages":
            return json.dumps([
                {"role": "user", "content": "分析 600219.SH"},
                {"role": "assistant", "content": "南山铝业分析报告"},
            ], ensure_ascii=False)
        raise AssertionError(f"Unexpected call: {method} {path}")

    monkeypatch.setattr(plugin, "_request_json", fake_request_json)

    result = json.loads(plugin._vibe_ask({"question": "分析 600219.SH"}))

    assert result == {
        "success": True,
        "session_id": "s1",
        "attempt_id": "a1",
        "answer": "南山铝业分析报告",
    }
    assert calls[0]["payload"] == {"title": "Vibe-Trading Ask"}
    assert calls[1]["payload"] == {"content": "分析 600219.SH"}
    assert calls[2]["query"] == {"limit": 50}


def test_vibe_ask_ashare_wraps_question_with_akshare_first_instruction(monkeypatch):
    plugin = _load_plugin()
    sent_payloads = []

    def fake_request_json(method, path, payload=None, query=None):
        if method == "POST" and path == "/sessions":
            return json.dumps({"session_id": "s2"})
        if method == "POST" and path == "/sessions/s2/messages":
            sent_payloads.append(payload)
            return json.dumps({"message_id": "m2", "attempt_id": "a2"})
        if method == "GET" and path == "/sessions/s2/messages":
            return json.dumps([{"role": "assistant", "content": "A股 Agent 报告"}], ensure_ascii=False)
        raise AssertionError(f"Unexpected call: {method} {path}")

    monkeypatch.setattr(plugin, "_request_json", fake_request_json)

    result = json.loads(plugin._vibe_ask_ashare({"question": "南山铝业600219什么位置买卖比较好"}))

    assert result["success"] is True
    assert result["answer"] == "A股 Agent 报告"
    content = sent_payloads[0]["content"]
    assert "南山铝业600219什么位置买卖比较好" in content
    assert "A股" in content
    assert "AKShare" in content
    assert "不要默认调用 Tushare 或 QVeris" in content
    assert "最终回答不要输出" in content


def test_vibe_ask_strips_empty_think_tags(monkeypatch):
    plugin = _load_plugin()

    def fake_request_json(method, path, payload=None, query=None):
        if method == "POST" and path == "/sessions":
            return json.dumps({"session_id": "s3"})
        if method == "POST" and path == "/sessions/s3/messages":
            return json.dumps({"message_id": "m3", "attempt_id": "a3"})
        if method == "GET" and path == "/sessions/s3/messages":
            return json.dumps([
                {"role": "assistant", "content": "<think>\n</think>\n\n最终报告"},
            ], ensure_ascii=False)
        raise AssertionError(f"Unexpected call: {method} {path}")

    monkeypatch.setattr(plugin, "_request_json", fake_request_json)

    result = json.loads(plugin._vibe_ask({"question": "分析 600219.SH"}))

    assert result["answer"] == "最终报告"
