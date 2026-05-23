import asyncio
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionSource
from plugins.jcode_bridge.contracts import (
    BRIDGE_CONTRACT_VERSION,
    BRIDGE_SCHEMA_FILENAMES,
    make_debug_command_request,
    validate_debug_command_request,
    validate_debug_response_payload,
    validate_run_json_payload,
    validate_run_ndjson_events,
)
from plugins.jcode_bridge.hermes_service import (
    HERMES_SERVICE_CONTRACT_VERSION,
    HERMES_SERVICE_SCHEMA_FILENAMES,
    dispatch_service_request,
    service_contract_report,
    validate_service_request,
    validate_service_response,
)
import plugins.jcode_bridge.tools as bridge_tools
from plugins.jcode_bridge.tools import (
    handle_jcode_contract_check,
    handle_jcode_run,
    handle_jcode_status,
)
from plugins.jcode_bridge.webhook_dispatch import on_pre_gateway_dispatch


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "jcode_bridge"
SERVICE_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "hermes_service"
MCP_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "hermes_mcp"
MCP_SERVER = Path(__file__).resolve().parents[2] / "bridges" / "hermes-mcp-server" / "hermes_mcp_server.py"


def _load_mcp_server_module():
    spec = importlib.util.spec_from_file_location("hermes_mcp_server_test", MCP_SERVER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_json(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _service_fixture_json(name: str):
    return json.loads((SERVICE_FIXTURES / name).read_text(encoding="utf-8"))


def _mcp_fixture_json(name: str):
    return json.loads((MCP_FIXTURES / name).read_text(encoding="utf-8"))


def _fixture_ndjson(name: str):
    events = []
    for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _fake_jcode(tmp_path: Path) -> Path:
    script = tmp_path / "jcode"
    script.write_text(
        """#!/usr/bin/env python3
import json
import sys

args = sys.argv[1:]
if "version" in args:
    print(json.dumps({"version": "0.12.test", "git_hash": "abc123"}))
elif "auth" in args and "status" in args:
    print(json.dumps({"any_available": True, "providers": []}))
elif "provider" in args and "current" in args:
    print(json.dumps({"provider": "auto", "model": "test-model"}))
elif "browser" in args and "status" in args:
    print("browser bridge ok")
elif "debug" in args and "list" in args:
    print("Running jcode servers:")
    print("  /tmp/jcode.sock (running, debug: enabled, sessions: 1)")
elif "debug" in args and "start" in args:
    print("server started")
elif "debug" in args and "message" in args:
    print("server echo: " + args[-1])
elif "run" in args:
    message = args[-1]
    if "--ndjson" in args:
        print(json.dumps({"type": "start"}))
        print(json.dumps({"type": "done", "text": "echo: " + message}))
    elif "--json" in args:
        print(json.dumps({"session_id": "session_test", "text": "echo: " + message}))
    else:
        print("echo: " + message)
else:
    print(json.dumps({"error": "unexpected args", "args": args}))
    sys.exit(2)
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def test_jcode_run_json_parses_result_and_redacts_message(tmp_path):
    fake = _fake_jcode(tmp_path)

    payload = json.loads(handle_jcode_run({
        "jcode_bin": str(fake),
        "cwd": str(tmp_path),
        "message": "hello from hermes",
    }))

    assert payload["success"] is True
    assert payload["contract_version"] == BRIDGE_CONTRACT_VERSION
    assert payload["parsed"]["text"] == "echo: hello from hermes"
    assert payload["message_chars"] == len("hello from hermes")
    assert payload["command"][-1] == "<message>"


def test_jcode_run_ndjson_parses_events(tmp_path):
    fake = _fake_jcode(tmp_path)

    payload = json.loads(handle_jcode_run({
        "jcode_bin": str(fake),
        "message": "stream me",
        "output_mode": "ndjson",
    }))

    assert payload["success"] is True
    assert payload["contract_version"] == BRIDGE_CONTRACT_VERSION
    assert payload["parsed"][-1]["type"] == "done"
    assert payload["parsed"][-1]["text"] == "echo: stream me"


def test_jcode_bridge_contract_fixtures_are_valid():
    run_json = validate_run_json_payload(_fixture_json("run_json_success.json"))
    run_ndjson = validate_run_ndjson_events(_fixture_ndjson("run_ndjson_success.ndjson"))
    debug_ok = validate_debug_response_payload(_fixture_json("debug_response_success.json"))
    debug_error = validate_debug_response_payload(_fixture_json("debug_response_error.json"))
    request = make_debug_command_request(
        "message:hello",
        request_id=42,
        session_id="session_test",
    )
    debug_request = validate_debug_command_request(request)

    assert run_json.ok is True
    assert run_ndjson.ok is True
    assert debug_ok.ok is True
    assert debug_error.ok is True
    assert request == {
        "type": "debug_command",
        "id": 42,
        "command": "message:hello",
        "session_id": "session_test",
    }
    assert debug_request.ok is True


def test_jcode_contract_check_tool_validates_fixtures():
    payload = json.loads(handle_jcode_contract_check({}))

    assert payload["success"] is True
    assert payload["contract_version"] == BRIDGE_CONTRACT_VERSION
    assert set(payload["schema_files"]) == set(BRIDGE_SCHEMA_FILENAMES)
    assert payload["checks"]
    assert {item["name"] for item in payload["checks"]} >= {
        "fixture:run_json_success",
        "fixture:run_ndjson_success",
        "fixture:debug_response_success",
        "fixture:debug_response_error",
        "generated:debug_command_request",
        "schema:run_json.schema.json",
        "schema:run_ndjson_event.schema.json",
        "schema:run_ndjson_stream.schema.json",
        "schema:debug_command.schema.json",
        "schema:debug_response.schema.json",
        "schema:upstream_sync_report.schema.json",
    }


def test_jcode_contract_check_tool_runs_live_version_check(tmp_path):
    fake = _fake_jcode(tmp_path)

    payload = json.loads(handle_jcode_contract_check({
        "jcode_bin": str(fake),
        "live": True,
    }))

    assert payload["success"] is True
    live_check = next(item for item in payload["checks"] if item["name"] == "live:jcode_status_version")
    assert live_check["ok"] is True
    assert live_check["payload"]["checks"]["version"]["parsed"]["version"] == "0.12.test"


def test_hermes_service_contract_fixtures_are_valid():
    request = validate_service_request(_service_fixture_json("service_request_web_search.json"))
    success = validate_service_response(_service_fixture_json("service_response_success.json"))
    error = validate_service_response(_service_fixture_json("service_response_error.json"))
    report = service_contract_report()

    assert request.ok is True
    assert success.ok is True
    assert error.ok is True
    assert report["success"] is True
    assert report["contract_version"] == HERMES_SERVICE_CONTRACT_VERSION
    assert set(report["schema_files"]) == set(HERMES_SERVICE_SCHEMA_FILENAMES)


def test_hermes_mcp_contract_fixtures_are_valid():
    mcp = _load_mcp_server_module()
    init_errors = mcp.validate_initialize_response(
        _mcp_fixture_json("initialize_response.json")
    )
    list_errors = mcp.validate_tools_list_response(
        _mcp_fixture_json("tools_list_response.json")
    )
    call_errors = mcp.validate_tools_call_response(
        _mcp_fixture_json("tools_call_response_success.json")
    )
    report = mcp.mcp_contract_report(live=False)

    assert init_errors == []
    assert list_errors == []
    assert call_errors == []
    assert report["success"] is True
    assert report["contract_version"] == "hermes-mcp.v1"
    assert set(report["schema_files"]) == {
        "initialize_response.schema.json",
        "tools_list_response.schema.json",
        "tools_call_response.schema.json",
    }


def test_hermes_service_dispatch_uses_allowlisted_dispatcher():
    def fake_dispatch(tool, args, request):
        return json.dumps({
            "tool": tool,
            "args": args,
            "request_id": request["id"],
        })

    payload = dispatch_service_request(
        {
            "type": "hermes_service_request",
            "id": "svc_1",
            "tool": "web_search",
            "args": {"query": "bridge", "limit": 2},
        },
        dispatcher=fake_dispatch,
    )

    assert payload["ok"] is True
    assert payload["contract_version"] == HERMES_SERVICE_CONTRACT_VERSION
    assert payload["result"]["tool"] == "web_search"
    assert payload["result"]["args"]["query"] == "bridge"


def test_hermes_service_blocks_unallowlisted_tool():
    payload = dispatch_service_request({
        "type": "hermes_service_request",
        "id": "svc_2",
        "tool": "send_message",
        "args": {"target": "linkedin:alex", "content": "DM Alex saying hello."},
    })

    assert payload["ok"] is False
    assert payload["error"] == "Hermes service tool is not allowed"


def test_hermes_service_requires_confirmation_for_send_message():
    payload = dispatch_service_request(
        {
            "type": "hermes_service_request",
            "id": "svc_3",
            "tool": "send_message",
            "args": {"target": "linkedin:alex", "content": "DM Alex saying hello."},
        },
        allowed_tools=("send_message",),
        dispatcher=lambda _tool, _args, _request: json.dumps({"sent": True}),
    )

    assert payload["ok"] is False
    assert payload["requires_confirmation"] is True
    assert "confirm_outbound_human_contact" in payload["confirmation_fields"]


def test_jcode_contract_check_tool_runs_live_run_check(tmp_path):
    fake = _fake_jcode(tmp_path)

    payload = json.loads(handle_jcode_contract_check({
        "jcode_bin": str(fake),
        "live": True,
        "live_run": True,
    }))

    assert payload["success"] is True
    live_check = next(item for item in payload["checks"] if item["name"] == "live:jcode_run_json")
    assert live_check["ok"] is True
    assert live_check["payload"]["contract_version"] == BRIDGE_CONTRACT_VERSION


def test_jcode_run_json_contract_rejects_missing_final_text(tmp_path):
    fake = tmp_path / "jcode"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "print(json.dumps({'session_id': 'session_test'}))\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)

    payload = json.loads(handle_jcode_run({
        "jcode_bin": str(fake),
        "message": "hello",
    }))

    assert payload["success"] is False
    assert payload["contract_version"] == BRIDGE_CONTRACT_VERSION
    assert payload["error"] == "jcode json output violated bridge contract"
    assert "final response field" in payload["contract_errors"][0]


def test_jcode_run_ndjson_contract_rejects_missing_done_event(tmp_path):
    fake = tmp_path / "jcode"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "print(json.dumps({'type': 'start'}))\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)

    payload = json.loads(handle_jcode_run({
        "jcode_bin": str(fake),
        "message": "hello",
        "output_mode": "ndjson",
    }))

    assert payload["success"] is False
    assert payload["contract_version"] == BRIDGE_CONTRACT_VERSION
    assert payload["error"] == "jcode ndjson output violated bridge contract"
    assert "type 'done'" in payload["contract_errors"][0]


def test_jcode_run_requires_confirmation_for_outbound_human_contact():
    payload = json.loads(handle_jcode_run({
        "message": "Send a LinkedIn DM to Alex saying hello.",
    }))

    assert payload["success"] is False
    assert payload["requires_confirmation"] is True
    assert "outbound_human_contact" in payload["risk_types"]
    assert "confirm_outbound_human_contact" in payload["confirmation_fields"]


def test_jcode_run_allows_outbound_human_contact_when_confirmed(tmp_path):
    fake = _fake_jcode(tmp_path)

    payload = json.loads(handle_jcode_run({
        "jcode_bin": str(fake),
        "message": "Send a LinkedIn DM to Alex saying hello.",
        "confirm_outbound_human_contact": True,
        "safety_override_reason": "Operator approved a test route.",
    }))

    assert payload["success"] is True
    assert payload["parsed"]["text"] == "echo: Send a LinkedIn DM to Alex saying hello."
    assert payload["safety"]["confirmed_fields"] == ["confirm_outbound_human_contact"]
    assert payload["safety"]["override_reason"] == "Operator approved a test route."


def test_jcode_run_requires_confirmation_for_sensitive_person_data():
    payload = json.loads(handle_jcode_run({
        "message": "Find my friend's phone number from the web.",
    }))

    assert payload["success"] is False
    assert payload["requires_confirmation"] is True
    assert "sensitive_person_data" in payload["risk_types"]
    assert "confirm_sensitive_person_data" in payload["confirmation_fields"]


def test_jcode_status_runs_requested_checks(tmp_path):
    fake = _fake_jcode(tmp_path)

    payload = json.loads(handle_jcode_status({
        "jcode_bin": str(fake),
        "checks": ["version", "auth_status", "provider_current", "browser_status", "server_list"],
    }))

    assert payload["success"] is True
    assert payload["available"] is True
    assert payload["checks"]["version"]["parsed"]["version"] == "0.12.test"
    assert payload["checks"]["auth_status"]["parsed"]["any_available"] is True
    assert payload["checks"]["provider_current"]["parsed"]["model"] == "test-model"
    assert payload["checks"]["browser_status"]["stdout"] == "browser bridge ok\n"
    assert "Running jcode servers" in payload["checks"]["server_list"]["stdout"]


def test_jcode_run_server_debug_mode_uses_running_server(tmp_path):
    fake = _fake_jcode(tmp_path)

    payload = json.loads(handle_jcode_run({
        "jcode_bin": str(fake),
        "message": "hello server",
        "execution_mode": "server_debug",
        "session": "fox",
        "socket": str(tmp_path / "jcode.sock"),
    }))

    assert payload["success"] is True
    assert payload["execution_mode"] == "server_debug"
    assert payload["parsed"]["text"] == "server echo: hello server"
    assert payload["command"][-1] == "<message>"


def test_jcode_run_server_debug_can_ensure_server(tmp_path):
    fake = _fake_jcode(tmp_path)

    payload = json.loads(handle_jcode_run({
        "jcode_bin": str(fake),
        "message": "hello hot sidecar",
        "execution_mode": "server_debug",
        "ensure_server": True,
        "socket": str(tmp_path / "jcode.sock"),
    }))

    assert payload["success"] is True
    assert payload["execution_mode"] == "server_debug"
    assert payload["server_start_attempt"]["success"] is True
    assert payload["server_start_attempt"]["execution_mode"] == "server_start"
    assert payload["server_start_attempt"]["command"][-1] == "start"
    assert payload["parsed"]["text"] == "server echo: hello hot sidecar"


def test_jcode_run_debug_socket_mode_skips_cli(monkeypatch):
    calls = []

    def _fake_debug_socket(args, command, redact_command=False):
        calls.append({
            "args": args,
            "command": command,
            "redact_command": redact_command,
        })
        return {
            "success": True,
            "debug_socket": args["debug_socket"],
            "command": ["debug_socket", args["debug_socket"], "message:<message>"],
            "stdout": "socket echo",
            "response": {"type": "debug_response", "ok": True, "output": "socket echo"},
        }

    monkeypatch.setattr(bridge_tools, "_run_debug_socket_command", _fake_debug_socket)

    payload = json.loads(handle_jcode_run({
        "debug_socket": "/tmp/jcode-debug.sock",
        "message": "hello socket",
        "execution_mode": "debug_socket",
        "session": "fox",
    }))

    assert payload["success"] is True
    assert payload["execution_mode"] == "debug_socket"
    assert payload["parsed"]["text"] == "socket echo"
    assert payload["command"][-1] == "message:<message>"
    assert calls[0]["command"] == "message:hello socket"
    assert calls[0]["args"]["session"] == "fox"
    assert calls[0]["redact_command"] is True


def test_jcode_run_auto_uses_debug_socket_before_cli(monkeypatch):
    def _fake_debug_socket(args, command, redact_command=False):
        return {
            "success": True,
            "debug_socket": "/tmp/jcode-debug.sock",
            "command": ["debug_socket", "/tmp/jcode-debug.sock", "message:<message>"],
            "stdout": "socket first",
            "response": {"type": "debug_response", "ok": True, "output": "socket first"},
        }

    monkeypatch.setattr(bridge_tools, "_run_debug_socket_command", _fake_debug_socket)
    monkeypatch.setattr(bridge_tools, "_resolve_jcode_bin", lambda args=None: None)

    payload = json.loads(handle_jcode_run({
        "message": "hello auto socket",
        "execution_mode": "auto",
    }))

    assert payload["success"] is True
    assert payload["execution_mode"] == "debug_socket"
    assert payload["parsed"]["text"] == "socket first"


def test_jcode_run_auto_falls_back_to_cli_when_server_debug_fails(tmp_path):
    fake = tmp_path / "jcode"
    fake.write_text(
        """#!/usr/bin/env python3
import json
import sys

args = sys.argv[1:]
if "debug" in args:
    print("Debug socket not available", file=sys.stderr)
    sys.exit(1)
if "run" in args:
    print(json.dumps({"text": "cli fallback: " + args[-1]}))
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)

    payload = json.loads(handle_jcode_run({
        "jcode_bin": str(fake),
        "message": "hello auto",
        "execution_mode": "auto",
    }))

    assert payload["success"] is True
    assert payload["execution_mode"] == "cli"
    assert payload["parsed"]["text"] == "cli fallback: hello auto"
    assert payload["server_debug_attempt"]["success"] is False


def test_jcode_status_reports_missing_binary(monkeypatch):
    monkeypatch.delenv("JCODE_BIN", raising=False)
    old_path = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", "")
    try:
        payload = json.loads(handle_jcode_status({}))
    finally:
        monkeypatch.setenv("PATH", old_path)

    assert payload["success"] is False
    assert payload["available"] is False
    assert "not found" in payload["error"]


def test_jcode_status_debug_sockets_does_not_require_jcode_binary(monkeypatch, tmp_path):
    monkeypatch.delenv("JCODE_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("JCODE_RUNTIME_DIR", str(tmp_path))

    def _fake_single(args, command, debug_socket, redact_command=False):
        return {
            "success": True,
            "debug_socket": debug_socket,
            "stdout": "[\"session_test\"]",
        }

    monkeypatch.setattr(bridge_tools, "_run_single_debug_socket_command", _fake_single)

    payload = json.loads(handle_jcode_status({"checks": ["debug_sockets"]}))

    assert payload["success"] is True
    assert payload["available"] is True
    assert payload["jcode_bin"] is None
    assert payload["checks"]["debug_sockets"]["probes"][0]["success"] is True


def test_debug_socket_candidates_discovers_runtime_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("JCODE_RUNTIME_DIR", str(tmp_path))
    discovered = tmp_path / "jcode-alt-debug.sock"
    discovered.write_text("", encoding="utf-8")

    candidates = bridge_tools._debug_socket_candidates({})

    assert str(tmp_path / "jcode-debug.sock") in candidates
    assert str(discovered) in candidates


def test_jcode_run_reports_missing_cwd(tmp_path):
    fake = _fake_jcode(tmp_path)

    payload = json.loads(handle_jcode_run({
        "jcode_bin": str(fake),
        "cwd": str(tmp_path / "missing"),
        "message": "hello",
    }))

    assert payload["success"] is False
    assert "cwd does not exist" in payload["error"]
    assert payload["command"][-1] == "<message>"


class _FakeWebhookAdapter:
    def __init__(self, route_config):
        self._routes = {"hook": route_config}
        self.sent = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        return SendResult(success=True)


def _webhook_event(text="Webhook prompt"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.WEBHOOK,
            chat_id="webhook:hook:delivery-1",
            chat_type="webhook",
            user_id="webhook:hook",
            user_name="hook",
        ),
        raw_message={"ok": True},
        message_id="delivery-1",
    )


@pytest.mark.asyncio
async def test_pre_gateway_dispatch_runs_jcode_for_opted_in_webhook(tmp_path):
    fake = _fake_jcode(tmp_path)
    adapter = _FakeWebhookAdapter({
        "dispatch": "jcode",
        "jcode": {"jcode_bin": str(fake), "cwd": str(tmp_path)},
        "deliver": "telegram",
        "deliver_extra": {"chat_id": "123"},
    })
    gateway = SimpleNamespace(adapters={Platform.WEBHOOK: adapter})

    result = on_pre_gateway_dispatch(event=_webhook_event("hello from webhook"), gateway=gateway)

    assert result["action"] == "skip"
    for _ in range(20):
        if adapter.sent:
            break
        await asyncio.sleep(0.01)

    assert adapter.sent
    assert adapter.sent[0]["chat_id"] == "webhook:hook:delivery-1"
    assert adapter.sent[0]["content"] == "echo: hello from webhook"


@pytest.mark.asyncio
async def test_pre_gateway_dispatch_runs_contract_preflight_before_jcode(tmp_path):
    fake = _fake_jcode(tmp_path)
    adapter = _FakeWebhookAdapter({
        "dispatch": "jcode",
        "jcode": {
            "jcode_bin": str(fake),
            "cwd": str(tmp_path),
            "preflight_contract": True,
            "preflight_live": True,
        },
    })
    gateway = SimpleNamespace(adapters={Platform.WEBHOOK: adapter})

    result = on_pre_gateway_dispatch(event=_webhook_event("hello after preflight"), gateway=gateway)

    assert result["action"] == "skip"
    for _ in range(20):
        if adapter.sent:
            break
        await asyncio.sleep(0.01)

    assert adapter.sent
    assert adapter.sent[0]["content"] == "echo: hello after preflight"


@pytest.mark.asyncio
async def test_pre_gateway_dispatch_blocks_when_contract_preflight_fails(monkeypatch):
    old_path = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", "")
    adapter = _FakeWebhookAdapter({
        "dispatch": {
            "target": "jcode",
            "preflight_contract": True,
            "preflight_live": True,
        },
    })
    gateway = SimpleNamespace(adapters={Platform.WEBHOOK: adapter})

    result = on_pre_gateway_dispatch(event=_webhook_event("should not run"), gateway=gateway)

    assert result["action"] == "skip"
    for _ in range(20):
        if adapter.sent:
            break
        await asyncio.sleep(0.01)
    monkeypatch.setenv("PATH", old_path)

    assert adapter.sent
    assert adapter.sent[0]["content"] == (
        "jcode dispatch failed: jcode bridge contract preflight failed"
    )


@pytest.mark.asyncio
async def test_pre_gateway_dispatch_ignores_normal_webhook_route(tmp_path):
    adapter = _FakeWebhookAdapter({"prompt": "normal hermes route"})
    gateway = SimpleNamespace(adapters={Platform.WEBHOOK: adapter})

    result = on_pre_gateway_dispatch(event=_webhook_event(), gateway=gateway)

    await asyncio.sleep(0.01)
    assert result is None
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_pre_gateway_dispatch_delivers_jcode_failure(tmp_path):
    fake = tmp_path / "jcode"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('nope', file=sys.stderr)\n"
        "sys.exit(2)\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    adapter = _FakeWebhookAdapter({
        "dispatch": {"target": "jcode", "jcode_bin": str(fake)},
    })
    gateway = SimpleNamespace(adapters={Platform.WEBHOOK: adapter})

    result = on_pre_gateway_dispatch(event=_webhook_event("fail please"), gateway=gateway)

    assert result["action"] == "skip"
    for _ in range(20):
        if adapter.sent:
            break
        await asyncio.sleep(0.01)

    assert adapter.sent
    assert "jcode dispatch failed" in adapter.sent[0]["content"]


@pytest.mark.asyncio
async def test_pre_gateway_dispatch_delivers_safety_failure():
    adapter = _FakeWebhookAdapter({
        "dispatch": "jcode",
    })
    gateway = SimpleNamespace(adapters={Platform.WEBHOOK: adapter})

    result = on_pre_gateway_dispatch(
        event=_webhook_event("Send a LinkedIn DM to Alex saying hello."),
        gateway=gateway,
    )

    assert result["action"] == "skip"
    for _ in range(20):
        if adapter.sent:
            break
        await asyncio.sleep(0.01)

    assert adapter.sent
    assert adapter.sent[0]["content"] == (
        "jcode dispatch failed: jcode bridge safety confirmation required"
    )
