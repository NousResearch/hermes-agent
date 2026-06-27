"""Tests for the local OpenClaw bridge plugin."""

from __future__ import annotations

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace


def _decode(result: str) -> dict:
    return json.loads(result)


def test_openclaw_delegate_rejects_missing_env(monkeypatch):
    from plugins.openclaw_bridge.tools import openclaw_delegate

    monkeypatch.delenv("OPENCLAW_GATEWAY_URL", raising=False)
    monkeypatch.delenv("OPENCLAW_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("OPENCLAW_HERMES_BRIDGE_TOKEN", raising=False)

    result = _decode(openclaw_delegate({"taskId": "tasks.organize_today"}))

    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert result["error"] == "missing_environment"


def test_openclaw_delegate_rejects_non_dry_run(monkeypatch):
    from plugins.openclaw_bridge.tools import openclaw_delegate

    monkeypatch.setenv("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("OPENCLAW_GATEWAY_TOKEN", "gateway")
    monkeypatch.setenv("OPENCLAW_HERMES_BRIDGE_TOKEN", "bridge")

    result = _decode(
        openclaw_delegate(
            {
                "taskId": "tasks.organize_today",
                "intent": "run it",
                "dryRun": False,
                "allowedTools": [],
                "input": {"request": "run it"},
            }
        )
    )

    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert result["error"] == "invalid_request"
    assert "dryRun=true" in result["message"]


def test_openclaw_delegate_rejects_unknown_task(monkeypatch):
    from plugins.openclaw_bridge.tools import openclaw_delegate

    monkeypatch.setenv("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("OPENCLAW_GATEWAY_TOKEN", "gateway")
    monkeypatch.setenv("OPENCLAW_HERMES_BRIDGE_TOKEN", "bridge")

    result = _decode(
        openclaw_delegate(
            {
                "taskId": "message.send",
                "intent": "send",
                "dryRun": True,
                "allowedTools": [],
                "input": {"request": "send"},
            }
        )
    )

    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert result["error"] == "invalid_request"
    assert "tasks.organize_today" in result["message"]


def test_openclaw_delegate_allows_agent_team_dry_run(monkeypatch):
    from plugins.openclaw_bridge.tools import openclaw_delegate

    captured: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers["content-length"])
            captured["body"] = json.loads(self.rfile.read(length))
            payload = {
                "ok": True,
                "taskId": "agents.ask_team",
                "mode": "mock",
                "status": "succeeded",
                "summary": "Dry-run completed. No OpenClaw agents were started.",
                "auditLog": [{"step": "dry-run", "message": "no agents", "at": "1970-01-01T00:00:00.000Z"}],
                "artifacts": [],
                "output": {"team": "openclaw", "dryRun": True, "agentsStarted": False},
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        monkeypatch.setenv("OPENCLAW_GATEWAY_URL", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setenv("OPENCLAW_GATEWAY_TOKEN", "gateway-token")
        monkeypatch.setenv("OPENCLAW_HERMES_BRIDGE_TOKEN", "bridge-token")

        result = _decode(
            openclaw_delegate(
                {
                    "taskId": "agents.ask_team",
                    "intent": "請 OpenClaw agent 團隊協助分析，但只做 dry-run。",
                    "dryRun": True,
                    "allowedTools": [],
                    "input": {
                        "team": "openclaw",
                        "question": "為何 Hermes 還無法呼叫 OpenClaw agent 團隊？",
                    },
                    "requestId": "team-req-1",
                }
            )
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert captured["body"]["taskId"] == "agents.ask_team"
    assert captured["body"]["dryRun"] is True
    assert captured["body"]["allowedTools"] == []
    assert captured["body"]["input"]["team"] == "openclaw"
    assert captured["body"]["idempotencyKey"] == "team-req-1"
    assert result["status"] == "succeeded"
    assert result["summary"] == "Dry-run completed. No OpenClaw agents were started."


def test_openclaw_delegate_rejects_allowed_tools(monkeypatch):
    from plugins.openclaw_bridge.tools import openclaw_delegate

    monkeypatch.setenv("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("OPENCLAW_GATEWAY_TOKEN", "gateway")
    monkeypatch.setenv("OPENCLAW_HERMES_BRIDGE_TOKEN", "bridge")

    result = _decode(
        openclaw_delegate(
            {
                "taskId": "tasks.organize_today",
                "intent": "run",
                "dryRun": True,
                "allowedTools": ["telegram.send"],
                "input": {"request": "run"},
            }
        )
    )

    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert result["error"] == "invalid_request"
    assert "allowedTools=[]" in result["message"]


def test_openclaw_delegate_sends_headers_and_preserves_result(monkeypatch):
    from plugins.openclaw_bridge.tools import openclaw_delegate

    captured: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers["content-length"])
            captured["path"] = self.path
            captured["authorization"] = self.headers.get("authorization")
            captured["bridge_token"] = self.headers.get("x-openclaw-hermes-token")
            captured["body"] = json.loads(self.rfile.read(length))
            payload = {
                "ok": True,
                "taskId": "tasks.organize_today",
                "mode": "mock",
                "status": "succeeded",
                "summary": "Dry-run completed. No external side effects were performed.",
                "auditLog": [{"step": "dry-run", "message": "no effects", "at": "1970-01-01T00:00:00.000Z"}],
                "artifacts": [],
                "output": {"dryRun": True},
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        monkeypatch.setenv("OPENCLAW_GATEWAY_URL", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setenv("OPENCLAW_GATEWAY_TOKEN", "gateway-token")
        monkeypatch.setenv("OPENCLAW_HERMES_BRIDGE_TOKEN", "bridge-token")

        result = _decode(
            openclaw_delegate(
                {
                    "taskId": "tasks.organize_today",
                    "intent": "請 OpenClaw 幫我整理今天的任務，但只做 dry-run。",
                    "dryRun": True,
                    "allowedTools": [],
                    "input": {"request": "請 OpenClaw 幫我整理今天的任務，但只做 dry-run。"},
                    "requestId": "req-1",
                }
            )
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert captured["path"] == "/api/plugins/hermes-bridge/tasks"
    assert captured["authorization"] == "Bearer gateway-token"
    assert captured["bridge_token"] == "bridge-token"
    assert captured["body"]["taskId"] == "tasks.organize_today"
    assert captured["body"]["dryRun"] is True
    assert captured["body"]["allowedTools"] == []
    assert captured["body"]["idempotencyKey"] == "req-1"
    assert result["status"] == "succeeded"
    assert result["summary"] == "Dry-run completed. No external side effects were performed."
    assert result["auditLog"][0]["step"] == "dry-run"


def test_openclaw_delegate_blocks_unsafe_bridge_result(monkeypatch):
    from plugins.openclaw_bridge.tools import openclaw_delegate

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            payload = {
                "ok": True,
                "taskId": "tasks.organize_today",
                "mode": "live",
                "status": "succeeded",
                "summary": "Live execution completed.",
                "auditLog": [],
                "artifacts": [],
                "output": {"dryRun": False, "sideEffectsPerformed": True},
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        monkeypatch.setenv("OPENCLAW_GATEWAY_URL", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setenv("OPENCLAW_GATEWAY_TOKEN", "gateway-token")
        monkeypatch.setenv("OPENCLAW_HERMES_BRIDGE_TOKEN", "bridge-token")

        result = _decode(
            openclaw_delegate(
                {
                    "taskId": "tasks.organize_today",
                    "intent": "請 OpenClaw 幫我整理今天的任務，但只做 dry-run。",
                    "dryRun": True,
                    "allowedTools": [],
                    "input": {"request": "請 OpenClaw 幫我整理今天的任務，但只做 dry-run。"},
                }
            )
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert result["error"] == "unsafe_bridge_result"
    assert "dry-run" in result["message"]


def test_plugin_registers_tool():
    from plugins.openclaw_bridge import register

    calls = []

    class Ctx:
        def register_tool(self, **kwargs):
            calls.append(kwargs)

        def register_command(self, *args, **kwargs):
            return None

        def register_hook(self, *args, **kwargs):
            return None

    register(Ctx())

    assert calls[0]["name"] == "openclaw_delegate"
    assert calls[0]["toolset"] == "openclaw_bridge"
    assert calls[0]["schema"]["name"] == "openclaw_delegate"
    assert callable(calls[0]["handler"])
    assert callable(calls[0]["check_fn"])



def test_plugin_registers_command_and_gateway_hook():
    from plugins.openclaw_bridge import register

    calls = {"tools": [], "commands": [], "hooks": []}

    class Ctx:
        def register_tool(self, **kwargs):
            calls["tools"].append(kwargs)

        def register_command(self, *args, **kwargs):
            calls["commands"].append((args, kwargs))

        def register_hook(self, *args, **kwargs):
            calls["hooks"].append((args, kwargs))

    register(Ctx())

    assert calls["tools"][0]["name"] == "openclaw_delegate"
    assert calls["commands"][0][0][0] == "openclaw-dry-run"
    assert callable(calls["commands"][0][0][1])
    assert calls["commands"][1][0][0] == "clawops"
    assert callable(calls["commands"][1][0][1])
    assert calls["commands"][2][0][0] == "clawops-run"
    assert callable(calls["commands"][2][0][1])
    assert calls["commands"][3][0][0] == "clawops-approve"
    assert callable(calls["commands"][3][0][1])
    assert calls["hooks"][0][0][0] == "pre_gateway_dispatch"
    assert callable(calls["hooks"][0][0][1])


def test_plugin_registers_host_llm_for_clawops_execution():
    from plugins.openclaw_bridge import register
    from plugins.openclaw_bridge import tools

    fake_llm = object()

    class Ctx:
        @property
        def llm(self):
            return fake_llm

        def register_tool(self, **kwargs):
            return None

        def register_command(self, *args, **kwargs):
            return None

        def register_hook(self, *args, **kwargs):
            return None

    tools.set_clawops_host_llm(None)
    register(Ctx())

    assert tools.get_clawops_host_llm() is fake_llm


def test_clawops_route_hahow_course_design_uses_course_designer():
    from plugins.openclaw_bridge.tools import route_clawops_task

    result = route_clawops_task(
        {
            "project": "hahow_course",
            "taskType": "course_design",
            "riskLevel": "medium",
            "request": "請 ClawOps 規劃 Hahow 課程大綱",
        }
    )

    assert result["ok"] is True
    assert result["status"] == "routed"
    assert result["project"] == "hahow_course"
    assert result["taskType"] == "course_design"
    assert result["assignedAgent"] == "course_designer"
    assert result["primaryModel"] == "codex"
    assert result["fallbackModel"] == "gemini-3.1-pro"
    assert result["approvalRequired"] is False
    assert result["dryRun"] is True
    assert result["externalSideEffects"] is False


def test_clawops_command_executes_agent_team_by_default(monkeypatch):
    from plugins.openclaw_bridge import tools

    def fake_execute(args, **kwargs):
        assert args["request"] == "請規劃 Hahow 課程大綱"
        return {
            "ok": True,
            "status": "generated",
            "project": "hahow_course",
            "taskType": "course_design",
            "assignedAgent": "course_designer",
            "primaryModel": "codex",
            "fallbackModel": "gemini-3.1-pro",
            "modelUsed": "codex_app_server",
            "approvalRequired": False,
            "dryRun": True,
            "externalModelCall": True,
            "externalSideEffects": False,
            "output": "課程草稿：風險觀念、錢包安全、交易實作。",
        }

    monkeypatch.setattr(tools, "execute_clawops_task", fake_execute)

    result = tools.handle_clawops_command("請規劃 Hahow 課程大綱")

    assert "ClawOps agent execution" in result
    assert "Status: generated" in result
    assert "Project: hahow_course" in result
    assert "Task type: course_design" in result
    assert "Assigned agent: course_designer" in result
    assert "Model used: codex_app_server" in result
    assert "External model call: yes" in result
    assert "External actions: none" in result
    assert "課程草稿" in result
    assert "routing dry-run" not in result


def test_clawops_agent_model_registry_question_injects_registry_context():
    from plugins.openclaw_bridge import tools

    routing = tools.route_clawops_task({"request": "請幫我整理我們ClawOps的責任分工和使用的語言模型"})
    prompt = tools._codex_app_server_prompt(
        {"request": "請幫我整理我們ClawOps的責任分工和使用的語言模型"},
        routing,
    )

    assert "ClawOps registry context" in prompt
    assert "Hermes-Grace (hermes_grace)" in prompt
    assert "Operations Orchestrator (orchestrator)" in prompt
    assert "Course Designer Agent (course_designer)" in prompt
    assert "Content Creator Agent (content_creator)" in prompt
    assert "primary_model=codex" in prompt
    assert "primary_model=gemini-2.5-flash" in prompt
    assert "gemini-3.1-flash-lite" not in prompt


def test_clawops_execute_codex_primary_uses_host_llm():
    from plugins.openclaw_bridge import tools

    captured: dict = {}

    class FakeLlm:
        def complete(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                text="課程設計草稿：先建立風險觀念，再進入錢包與交易所安全實作。",
                provider="openai-codex",
                model="gpt-5.5",
            )

    tools.set_clawops_codex_app_server_enabled(False)
    tools.set_clawops_host_llm(FakeLlm())
    try:
        result = tools.execute_clawops_task({"request": "請規劃 Hahow 課程大綱"})
    finally:
        tools.set_clawops_host_llm(None)
        tools.set_clawops_codex_app_server_enabled(None)

    assert result["ok"] is True
    assert result["status"] == "generated"
    assert result["assignedAgent"] == "course_designer"
    assert result["primaryModel"] == "codex"
    assert result["fallbackModel"] == "gemini-3.1-pro"
    assert result["modelUsed"] == "gpt-5.5"
    assert result["modelProvider"] == "openai-codex"
    assert result["externalModelCall"] is True
    assert result["externalSideEffects"] is False
    assert "課程設計草稿" in result["output"]
    assert captured["kwargs"]["purpose"] == "clawops:course_designer"
    assert captured["kwargs"]["temperature"] == 0.3
    assert captured["messages"][0]["role"] == "system"
    assert "course_designer" in captured["messages"][0]["content"]
    assert "請規劃 Hahow 課程大綱" in captured["messages"][1]["content"]


def test_clawops_execute_codex_primary_uses_codex_app_server():
    from plugins.openclaw_bridge import tools

    captured: dict = {}

    class FakeTurn:
        final_text = "Codex app-server 課程草稿：先建立風險觀念，再設計安全實作。"
        error = None

    class FakeSession:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.closed = False

        def ensure_started(self):
            captured["started"] = True
            return "thread-1"

        def run_turn(self, user_input):
            captured["user_input"] = user_input
            return FakeTurn()

        def close(self):
            captured["closed"] = True

    tools.set_clawops_codex_app_server_enabled(True)
    tools.set_clawops_codex_app_server_session_factory(FakeSession)
    try:
        result = tools.execute_clawops_task({"request": "請規劃 Hahow 課程大綱"})
    finally:
        tools.set_clawops_codex_app_server_session_factory(None)
        tools.set_clawops_codex_app_server_enabled(None)

    assert result["ok"] is True
    assert result["status"] == "generated"
    assert result["assignedAgent"] == "course_designer"
    assert result["primaryModel"] == "codex"
    assert result["modelProvider"] == "codex_app_server"
    assert result["modelUsed"] == "codex_app_server"
    assert result["externalModelCall"] is True
    assert result["externalSideEffects"] is False
    assert captured["started"] is True
    assert captured["closed"] is True
    assert "course_designer" in captured["user_input"]
    assert "請規劃 Hahow 課程大綱" in captured["user_input"]
    assert "Codex app-server 課程草稿" in result["output"]


def test_clawops_execute_codex_primary_falls_back_to_gemini(monkeypatch):
    from plugins.openclaw_bridge import tools

    captured: dict = {}

    class FailingLlm:
        def complete(self, messages, **kwargs):
            raise RuntimeError("Codex host unavailable")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers["content-length"])
            captured["body"] = json.loads(self.rfile.read(length))
            payload = {
                "choices": [
                    {
                        "message": {
                            "content": "Fallback 課程草稿：先講風險，再講錢包安全。"
                        }
                    }
                ]
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        tools.set_clawops_codex_app_server_enabled(False)
        tools.set_clawops_host_llm(FailingLlm())
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        monkeypatch.setenv("CLAWOPS_GEMINI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1beta/openai")

        result = tools.execute_clawops_task({"request": "請規劃 Hahow 課程大綱"})
    finally:
        tools.set_clawops_host_llm(None)
        tools.set_clawops_codex_app_server_enabled(None)
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert result["ok"] is True
    assert result["status"] == "generated"
    assert result["assignedAgent"] == "course_designer"
    assert result["primaryModel"] == "codex"
    assert result["fallbackModel"] == "gemini-3.1-pro"
    assert result["fallbackUsed"] is True
    assert result["fallbackReason"] == "host_llm_request_failed"
    assert result["modelUsed"] == "gemini-3.1-pro-preview"
    assert captured["body"]["model"] == "gemini-3.1-pro-preview"
    assert "Fallback 課程草稿" in result["output"]


def test_clawops_execute_blocks_without_gemini_key(monkeypatch):
    from plugins.openclaw_bridge.tools import execute_clawops_task

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = execute_clawops_task(
        {
            "project": "course_marketing",
            "taskType": "campaign",
            "request": "請規劃課程招生行銷活動",
        }
    )

    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert result["error"] == "missing_model_api_key"
    assert result["assignedAgent"] == "marketing_operator"
    assert result["primaryModel"] == "gemini-2.5-flash"
    assert result["fallbackModel"] == "codex"
    assert result["dryRun"] is True
    assert result["externalSideEffects"] is False


def test_clawops_execute_calls_gemini_openai_compatible_endpoint(monkeypatch):
    from plugins.openclaw_bridge.tools import execute_clawops_task

    captured: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers["content-length"])
            captured["path"] = self.path
            captured["authorization"] = self.headers.get("authorization")
            captured["content_type"] = self.headers.get("content-type")
            captured["body"] = json.loads(self.rfile.read(length))
            payload = {
                "choices": [
                    {
                        "message": {
                            "content": "課程草稿：第一週風險觀念，第二週錢包安全，第三週交易所實作。"
                        }
                    }
                ]
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        monkeypatch.setenv("CLAWOPS_GEMINI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1beta/openai")

        result = execute_clawops_task(
            {
                "project": "course_marketing",
                "taskType": "campaign",
                "request": "請規劃課程招生行銷活動",
            }
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert captured["path"] == "/v1beta/openai/chat/completions"
    assert captured["authorization"] == "Bearer test-google-key"
    assert captured["content_type"] == "application/json"
    assert captured["body"]["model"] == "gemini-2.5-flash"
    assert captured["body"]["temperature"] == 0.4
    assert captured["body"]["messages"][0]["role"] == "system"
    assert "marketing_operator" in captured["body"]["messages"][0]["content"]
    assert captured["body"]["messages"][1]["role"] == "user"
    assert "請規劃課程招生行銷活動" in captured["body"]["messages"][1]["content"]
    assert result["ok"] is True
    assert result["status"] == "generated"
    assert result["assignedAgent"] == "marketing_operator"
    assert result["primaryModel"] == "gemini-2.5-flash"
    assert result["modelUsed"] == "gemini-2.5-flash"
    assert result["dryRun"] is True
    assert result["externalModelCall"] is True
    assert result["externalSideEffects"] is False
    assert "課程草稿" in result["output"]


def test_clawops_gemini_request_uses_https_context(monkeypatch):
    from plugins.openclaw_bridge import tools

    captured: dict = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "草稿"}}]}).encode("utf-8")

    def fake_urlopen(request, timeout, context=None):
        captured["timeout"] = timeout
        captured["context"] = context
        return FakeResponse()

    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setattr(tools.urllib.request, "urlopen", fake_urlopen)

    result = tools.execute_clawops_task(
        {
            "project": "course_marketing",
            "taskType": "campaign",
            "request": "請規劃課程招生行銷活動",
        }
    )

    assert result["ok"] is True
    assert captured["timeout"] == 60
    assert captured["context"] is not None


def test_clawops_execute_redacts_secret_bearing_http_errors(monkeypatch):
    from plugins.openclaw_bridge import tools

    def fake_urlopen(request, timeout, context=None):
        body = b'{"error":{"message":"bad key test-google-key and Authorization: Bearer secret-token"}}'
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    class FakeHttpError(urllib.error.HTTPError):
        def read(self):
            return b'{"error":{"message":"bad key test-google-key and Authorization: Bearer secret-token"}}'

    def raise_error(request, timeout, context=None):
        raise FakeHttpError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setattr(tools.urllib.request, "urlopen", raise_error)

    result = tools.execute_clawops_task(
        {
            "project": "course_marketing",
            "taskType": "campaign",
            "request": "請規劃課程招生行銷活動",
        }
    )

    assert result["ok"] is False
    assert result["error"] == "model_http_error"
    assert "test-google-key" not in result["message"]
    assert "secret-token" not in result["message"]
    assert "[REDACTED]" in result["message"]


def test_clawops_run_command_formats_generated_output(monkeypatch):
    from plugins.openclaw_bridge import tools

    def fake_execute(args, **kwargs):
        return {
            "ok": True,
            "status": "generated",
            "project": "hahow_course",
            "taskType": "course_design",
            "assignedAgent": "course_designer",
            "primaryModel": "gemini-3.1-pro",
            "fallbackModel": "codex",
            "modelUsed": "gemini-3.1-pro",
            "approvalRequired": False,
            "dryRun": True,
            "externalModelCall": True,
            "externalSideEffects": False,
            "output": "課程草稿：風險觀念、錢包安全、交易實作。",
        }

    monkeypatch.setattr(tools, "execute_clawops_task", fake_execute)

    result = tools.handle_clawops_run_command("請規劃 Hahow 課程大綱")

    assert "ClawOps agent execution" in result
    assert "Status: generated" in result
    assert "Assigned agent: course_designer" in result
    assert "Model used: gemini-3.1-pro" in result
    assert "External model call: yes" in result
    assert "External actions: none" in result
    assert "課程草稿" in result


def test_clawops_execution_formatter_shows_fallback_usage():
    from plugins.openclaw_bridge.tools import format_clawops_execution_result

    result = format_clawops_execution_result(
        {
            "ok": True,
            "status": "generated",
            "project": "hahow_course",
            "taskType": "course_design",
            "assignedAgent": "course_designer",
            "primaryModel": "codex",
            "fallbackModel": "gemini-3.1-pro",
            "modelUsed": "gemini-3.1-pro-preview",
            "fallbackUsed": True,
            "fallbackReason": "host_llm_request_failed",
            "approvalRequired": False,
            "dryRun": True,
            "externalModelCall": True,
            "externalSideEffects": False,
            "output": "課程草稿",
        }
    )

    assert "Fallback used: yes" in result
    assert "Fallback reason: host_llm_request_failed" in result
    assert "Model used: gemini-3.1-pro-preview" in result


def test_clawops_execution_formatter_marks_external_actions_pending_approval():
    from plugins.openclaw_bridge.tools import format_clawops_execution_result

    result = format_clawops_execution_result(
        {
            "ok": True,
            "status": "generated",
            "project": "course_marketing",
            "taskType": "campaign_planning",
            "assignedAgent": "marketing_operator",
            "primaryModel": "gemini-2.5-flash",
            "fallbackModel": "codex",
            "modelUsed": "gemini-2.5-flash",
            "approvalRequired": True,
            "dryRun": True,
            "externalModelCall": True,
            "externalSideEffects": False,
            "output": "招募文案草稿",
        }
    )

    assert "Approval required: yes" in result
    assert "External actions: pending your approval" in result


def test_clawops_approval_required_execution_creates_pending_approval(monkeypatch, tmp_path):
    from plugins.openclaw_bridge import tools

    monkeypatch.setenv("CLAWOPS_APPROVALS_DIR", str(tmp_path))

    def fake_gemini_task(**kwargs):
        fields = kwargs["fields"]
        return {
            "ok": True,
            "status": "generated",
            **fields,
            "modelUsed": "gemini-2.5-flash",
            "externalModelCall": True,
            "externalSideEffects": False,
            "output": "招生貼文草稿",
        }

    monkeypatch.setattr(tools, "_execute_clawops_gemini_task", fake_gemini_task)

    result = tools.execute_clawops_task(
        {
            "request": "請 ClawOps 規劃招生行銷貼文，待我確認後再發文",
            "project": "course_marketing",
            "taskType": "campaign",
            "actions": [
                {
                    "type": "audit.record",
                    "description": "Record approved publication intent.",
                    "payload": {"channel": "telegram", "summary": "招生貼文草稿"},
                }
            ],
        }
    )

    assert result["ok"] is True
    assert result["approvalRequired"] is True
    assert result["approvalStatus"] == "pending"
    assert result["approvalId"].startswith("clawops-")
    assert result["executableActions"] == 1

    approval_file = tmp_path / f"{result['approvalId']}.json"
    assert approval_file.is_file()
    saved = json.loads(approval_file.read_text(encoding="utf-8"))
    assert saved["status"] == "pending"
    assert saved["actions"][0]["type"] == "audit.record"


def test_clawops_approve_command_executes_allowlisted_action(monkeypatch, tmp_path):
    from plugins.openclaw_bridge import tools

    monkeypatch.setenv("CLAWOPS_APPROVALS_DIR", str(tmp_path))
    approval_id = "clawops-test-approve"
    (tmp_path / f"{approval_id}.json").write_text(
        json.dumps(
            {
                "id": approval_id,
                "status": "pending",
                "actions": [
                    {
                        "type": "audit.record",
                        "description": "Record approved action.",
                        "payload": {"summary": "approved"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = tools.handle_clawops_approve_command(approval_id)

    assert "ClawOps approved actions executed" in result
    assert "Approval id: clawops-test-approve" in result
    assert "Executed actions: 1" in result
    saved = json.loads((tmp_path / f"{approval_id}.json").read_text(encoding="utf-8"))
    assert saved["status"] == "executed"
    assert (tmp_path / "audit.log").is_file()


def test_clawops_approve_command_blocks_unsupported_action(monkeypatch, tmp_path):
    from plugins.openclaw_bridge import tools

    monkeypatch.setenv("CLAWOPS_APPROVALS_DIR", str(tmp_path))
    approval_id = "clawops-test-blocked"
    (tmp_path / f"{approval_id}.json").write_text(
        json.dumps(
            {
                "id": approval_id,
                "status": "pending",
                "actions": [
                    {
                        "type": "telegram.send",
                        "description": "Send a Telegram message.",
                        "payload": {"text": "hello"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = tools.handle_clawops_approve_command(approval_id)

    assert "ClawOps approved actions did not execute" in result
    assert "unsupported_action" in result
    saved = json.loads((tmp_path / f"{approval_id}.json").read_text(encoding="utf-8"))
    assert saved["status"] == "blocked"


def test_pre_gateway_dispatch_rewrites_openclaw_dry_run_message():
    from types import SimpleNamespace

    from plugins.openclaw_bridge.tools import pre_gateway_dispatch

    result = pre_gateway_dispatch(
        event=SimpleNamespace(text="請 OpenClaw 幫我整理今天的任務，但只做 dry-run。")
    )

    assert result == {
        "action": "rewrite",
        "text": "/openclaw-dry-run 請 OpenClaw 幫我整理今天的任務，但只做 dry-run。",
    }


def test_pre_gateway_dispatch_rewrites_clawops_message():
    from types import SimpleNamespace

    from plugins.openclaw_bridge.tools import pre_gateway_dispatch

    result = pre_gateway_dispatch(event=SimpleNamespace(text="Grace，請找 ClawOps 規劃 Hahow 課程"))

    assert result == {
        "action": "rewrite",
        "text": "/clawops Grace，請找 ClawOps 規劃 Hahow 課程",
    }


def test_openclaw_dry_run_command_delegates_to_bridge(monkeypatch):
    from plugins.openclaw_bridge.tools import handle_openclaw_dry_run_command

    captured: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers["content-length"])
            captured["body"] = json.loads(self.rfile.read(length))
            payload = {
                "ok": True,
                "taskId": "tasks.organize_today",
                "mode": "mock",
                "status": "succeeded",
                "summary": "Dry-run completed. No external side effects were performed.",
                "auditLog": [],
                "artifacts": [],
                "output": {"dryRun": True, "sideEffectsPerformed": False},
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        monkeypatch.setenv("OPENCLAW_GATEWAY_URL", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setenv("OPENCLAW_GATEWAY_TOKEN", "gateway-token")
        monkeypatch.setenv("OPENCLAW_HERMES_BRIDGE_TOKEN", "bridge-token")

        result = handle_openclaw_dry_run_command(
            "請 OpenClaw 幫我整理今天的任務，但只做 dry-run。"
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert captured["body"]["taskId"] == "tasks.organize_today"
    assert captured["body"]["dryRun"] is True
    assert captured["body"]["allowedTools"] == []
    assert "OpenClaw dry-run completed" in result
    assert "Status: succeeded" in result
    assert "Task: tasks.organize_today" in result
    assert "Dry-run completed. No external side effects were performed." in result
    assert "External side effects: no" in result
