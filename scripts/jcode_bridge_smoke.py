#!/usr/bin/env python3
"""Stdlib-only behavioral smoke tests for the Hermes <-> jcode bridge."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.config import Platform  # noqa: E402
from gateway.platforms.base import MessageEvent, SendResult  # noqa: E402
from gateway.session import SessionSource  # noqa: E402
from plugins.jcode_bridge.tools import (  # noqa: E402
    handle_jcode_contract_check,
    handle_jcode_run,
)
from plugins.jcode_bridge.hermes_service import (  # noqa: E402
    HERMES_SERVICE_CONTRACT_VERSION,
    dispatch_service_request,
    service_contract_report,
)
from plugins.jcode_bridge.webhook_dispatch import on_pre_gateway_dispatch  # noqa: E402
from scripts.hermes_jcode_mother_repo import build_manifest, scaffold  # noqa: E402
from scripts.jcode_native_registration_check import check_registration_patch  # noqa: E402
from scripts.jcode_supertool_registry_smoke import run_supertool_registry_smoke  # noqa: E402


class FakeWebhookAdapter:
    def __init__(self, route_config: dict[str, Any]):
        self._routes = {"hook": route_config}
        self.sent: list[dict[str, Any]] = []

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        return SendResult(success=True)


def _webhook_event(text: str) -> MessageEvent:
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


def _fake_jcode(path: Path, *, invalid_run_contract: bool = False) -> Path:
    script = path / "jcode"
    run_payload = (
        '{"session_id": "session_test"}'
        if invalid_run_contract
        else '{"session_id": "session_test", "text": "echo: " + message}'
    )
    script.write_text(
        f"""#!/usr/bin/env python3
import json
import sys

args = sys.argv[1:]
if "version" in args:
    print(json.dumps({{"version": "0.12.test", "git_hash": "abc123"}}))
elif "debug" in args and "start" in args:
    print("server started")
elif "debug" in args and "message" in args:
    print("server echo: " + args[-1])
elif "run" in args:
    message = args[-1]
    print(json.dumps({run_payload}))
else:
    print(json.dumps({{"error": "unexpected args", "args": args}}))
    sys.exit(2)
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _check(name: str, ok: bool, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "ok": bool(ok),
    }
    result.update(details)
    return result


def _run_check(name: str, fn) -> dict[str, Any]:
    try:
        details = fn()
    except Exception as exc:
        return _check(name, False, error=str(exc))
    if isinstance(details, dict):
        ok = bool(details.pop("ok", True))
        return _check(name, ok, **details)
    return _check(name, bool(details))


async def _wait_for_delivery(adapter: FakeWebhookAdapter) -> None:
    for _ in range(50):
        if adapter.sent:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out waiting for webhook delivery")


def check_contract_tool() -> dict[str, Any]:
    payload = json.loads(handle_jcode_contract_check({}))
    return {
        "ok": payload.get("success") is True,
        "contract_version": payload.get("contract_version"),
        "check_count": len(payload.get("checks", [])),
    }


def check_safety_blocks_outbound() -> dict[str, Any]:
    payload = json.loads(handle_jcode_run({
        "message": "Send a LinkedIn DM to Alex saying hello.",
    }))
    return {
        "ok": (
            payload.get("success") is False
            and payload.get("requires_confirmation") is True
            and "outbound_human_contact" in payload.get("risk_types", [])
        ),
        "payload": payload,
    }


def check_contract_rejects_bad_json() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temp:
        fake = _fake_jcode(Path(temp), invalid_run_contract=True)
        payload = json.loads(handle_jcode_run({
            "jcode_bin": str(fake),
            "message": "hello",
        }))
    return {
        "ok": (
            payload.get("success") is False
            and payload.get("error") == "jcode json output violated bridge contract"
        ),
        "payload": payload,
    }


def check_ensure_server_path() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temp:
        fake = _fake_jcode(Path(temp))
        payload = json.loads(handle_jcode_run({
            "jcode_bin": str(fake),
            "message": "hello hot sidecar",
            "execution_mode": "server_debug",
            "ensure_server": True,
        }))
    return {
        "ok": (
            payload.get("success") is True
            and payload.get("server_start_attempt", {}).get("success") is True
            and payload.get("parsed", {}).get("text") == "server echo: hello hot sidecar"
        ),
        "payload": payload,
    }


def check_hermes_service_contract() -> dict[str, Any]:
    payload = service_contract_report()
    return {
        "ok": payload.get("success") is True,
        "contract_version": payload.get("contract_version"),
        "check_count": len(payload.get("checks", [])),
    }


def check_hermes_service_dispatch() -> dict[str, Any]:
    def fake_dispatch(tool: str, args: dict[str, Any], request: dict[str, Any]) -> str:
        return json.dumps({
            "tool": tool,
            "args": args,
            "request_id": request.get("id"),
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
    return {
        "ok": (
            payload.get("ok") is True
            and payload.get("contract_version") == HERMES_SERVICE_CONTRACT_VERSION
            and payload.get("result", {}).get("args", {}).get("query") == "bridge"
        ),
        "payload": payload,
    }


def check_hermes_service_blocks_send_message() -> dict[str, Any]:
    payload = dispatch_service_request(
        {
            "type": "hermes_service_request",
            "id": "svc_2",
            "tool": "send_message",
            "args": {
                "target": "linkedin:alex",
                "content": "DM Alex saying hello.",
            },
        },
        allowed_tools=("send_message",),
        dispatcher=lambda _tool, _args, _request: json.dumps({"sent": True}),
    )
    return {
        "ok": (
            payload.get("ok") is False
            and payload.get("requires_confirmation") is True
            and "confirm_outbound_human_contact" in payload.get("confirmation_fields", [])
        ),
        "payload": payload,
    }


def check_jcode_tool_hermes_client() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        fake_service = temp_path / "fake_hermes_service.py"
        fake_service.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "import sys\n"
            "line = sys.stdin.readline().strip()\n"
            "request = json.loads(line)\n"
            "print(json.dumps({\n"
            "    'type': 'hermes_service_response',\n"
            "    'contract_version': 'hermes-service.v1',\n"
            "    'id': request.get('id'),\n"
            "    'ok': True,\n"
            "    'tool': request.get('tool'),\n"
            "    'result': {'args': request.get('args', {})},\n"
            "    'duration_ms': 1,\n"
            "}))\n",
            encoding="utf-8",
        )
        fake_service.chmod(0o755)
        env = {
            **os.environ,
            "CARGO_TARGET_DIR": str(temp_path / "target"),
        }
        completed = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--manifest-path",
                str(ROOT / "bridges" / "jcode-tool-hermes" / "Cargo.toml"),
                "--",
                "--service-command",
                f"{sys.executable} {fake_service}",
                "--tool",
                "web_search",
                "--args-json",
                '{"query":"bridge","limit":2}',
                "--id",
                "rust_svc_1",
            ],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = {"success": False, "stdout": completed.stdout}
    return {
        "ok": (
            completed.returncode == 0
            and payload.get("ok") is True
            and payload.get("contract_version") == "hermes-service.v1"
            and payload.get("result", {}).get("args", {}).get("query") == "bridge"
        ),
        "payload": payload,
        "stderr": completed.stderr,
    }


def check_hermes_mcp_server() -> dict[str, Any]:
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "smoke", "version": "1"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "hermes_web_search",
                "arguments": {"query": "bridge", "limit": 2},
            },
        },
    ]
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bridges" / "hermes-mcp-server" / "hermes_mcp_server.py"),
            "--mock",
        ],
        input="\n".join(json.dumps(item) for item in requests) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        responses = [json.loads(line) for line in lines]
        tool_names = {
            tool.get("name")
            for tool in responses[1].get("result", {}).get("tools", [])
            if isinstance(tool, dict)
        }
        call_text = responses[2]["result"]["content"][0]["text"]
        call_payload = json.loads(call_text)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return {
        "ok": (
            completed.returncode == 0
            and responses[0].get("result", {}).get("serverInfo", {}).get("name") == "hermes"
            and "hermes_tool" in tool_names
            and "hermes_web_search" in tool_names
            and call_payload.get("ok") is True
            and call_payload.get("tool") == "web_search"
            and call_payload.get("result", {}).get("args", {}).get("query") == "bridge"
        ),
        "tool_names": sorted(tool_names),
        "call_payload": call_payload,
        "stderr": completed.stderr,
    }


def check_hermes_mcp_contract() -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bridges" / "hermes-mcp-server" / "hermes_mcp_server.py"),
            "--check",
            "--live",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {"success": False, "stdout": completed.stdout}
    return {
        "ok": completed.returncode == 0 and payload.get("success") is True,
        "payload": payload,
        "stderr": completed.stderr,
    }


def check_jcode_native_registration_patch() -> dict[str, Any]:
    payload = check_registration_patch(
        ROOT / ".codex-research" / "jcode",
        ROOT / "patches" / "jcode" / "register-external-toolset.patch",
    )
    return {
        "ok": payload.get("success") is True,
        "payload": payload,
    }


def check_jcode_supertool_registry_smoke() -> dict[str, Any]:
    payload = run_supertool_registry_smoke(
        ROOT / ".codex-research" / "jcode",
        ROOT / "patches" / "jcode" / "register-external-toolset.patch",
        cargo=False,
        target_dir=Path(tempfile.gettempdir()) / "jcode-supertool-registry-smoke-target",
        keep_worktree=False,
    )
    return {
        "ok": payload.get("success") is True,
        "payload": payload,
    }


def check_mother_repo_scaffold() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "mother"
        result = scaffold(
            output,
            build_manifest(ROOT, ROOT / ".codex-research" / "jcode"),
            force=False,
        )
        completed = subprocess.run(
            [sys.executable, str(output / "scripts" / "check_bridge_contract.py")],
            text=True,
            capture_output=True,
            check=False,
        )
        service_completed = subprocess.run(
            [sys.executable, str(output / "scripts" / "hermes_service_bridge.py"), "check"],
            text=True,
            capture_output=True,
            check=False,
        )
        mcp_completed = subprocess.run(
            [
                sys.executable,
                str(output / "bridges" / "hermes-mcp-server" / "hermes_mcp_server.py"),
                "--mock",
            ],
            input=json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {},
            }) + "\n",
            text=True,
            capture_output=True,
            check=False,
        )
        mcp_contract_completed = subprocess.run(
            [
                sys.executable,
                str(output / "bridges" / "hermes-mcp-server" / "hermes_mcp_server.py"),
                "--check",
                "--live",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        latency_completed = subprocess.run(
            [
                sys.executable,
                str(output / "scripts" / "jcode_bridge_latency_probe.py"),
                "--iterations",
                "5",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        native_check_completed = subprocess.run(
            [
                sys.executable,
                str(output / "scripts" / "jcode_native_tool_check.py"),
                "--jcode",
                str(ROOT / ".codex-research" / "jcode"),
                "--skip-cargo",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        native_registration_completed = subprocess.run(
            [
                sys.executable,
                str(output / "scripts" / "jcode_native_registration_check.py"),
                "--jcode",
                str(ROOT / ".codex-research" / "jcode"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        supertool_registry_completed = subprocess.run(
            [
                sys.executable,
                str(output / "scripts" / "jcode_supertool_registry_smoke.py"),
                "--jcode",
                str(ROOT / ".codex-research" / "jcode"),
                "--skip-cargo",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = {"success": False, "stdout": completed.stdout}
        try:
            service_payload = json.loads(service_completed.stdout)
        except json.JSONDecodeError:
            service_payload = {"success": False, "stdout": service_completed.stdout}
        try:
            mcp_payload = json.loads(mcp_completed.stdout)
        except json.JSONDecodeError:
            mcp_payload = {"success": False, "stdout": mcp_completed.stdout}
        try:
            mcp_contract_payload = json.loads(mcp_contract_completed.stdout)
        except json.JSONDecodeError:
            mcp_contract_payload = {
                "success": False,
                "stdout": mcp_contract_completed.stdout,
            }
        try:
            latency_payload = json.loads(latency_completed.stdout)
        except json.JSONDecodeError:
            latency_payload = {
                "success": False,
                "stdout": latency_completed.stdout,
            }
        try:
            native_check_payload = json.loads(native_check_completed.stdout)
        except json.JSONDecodeError:
            native_check_payload = {
                "success": False,
                "stdout": native_check_completed.stdout,
            }
        try:
            native_registration_payload = json.loads(native_registration_completed.stdout)
        except json.JSONDecodeError:
            native_registration_payload = {
                "success": False,
                "stdout": native_registration_completed.stdout,
            }
        try:
            supertool_registry_payload = json.loads(supertool_registry_completed.stdout)
        except json.JSONDecodeError:
            supertool_registry_payload = {
                "success": False,
                "stdout": supertool_registry_completed.stdout,
            }
        config_exists = (output / "configs" / "jcode-mcp.hermes.json").exists()
        patch_exists = (
            output / "patches" / "jcode" / "register-external-toolset.patch"
        ).exists()
        native_tool = (
            output
            / "bridges"
            / "jcode-native-hermes-tool"
            / "src"
            / "lib.rs"
        )
        native_tool_text = (
            native_tool.read_text(encoding="utf-8")
            if native_tool.exists()
            else ""
        )
    return {
        "ok": (
            completed.returncode == 0
            and payload.get("success") is True
            and payload.get("jcode_bridge", {}).get("success") is True
            and payload.get("hermes_service", {}).get("success") is True
            and service_completed.returncode == 0
            and service_payload.get("success") is True
            and mcp_completed.returncode == 0
            and mcp_contract_completed.returncode == 0
            and mcp_contract_payload.get("success") is True
            and latency_completed.returncode == 0
            and latency_payload.get("success") is True
            and native_check_completed.returncode == 0
            and native_check_payload.get("success") is True
            and native_registration_completed.returncode == 0
            and native_registration_payload.get("success") is True
            and supertool_registry_completed.returncode == 0
            and supertool_registry_payload.get("success") is True
            and any(
                item.get("name") == "hermes_tool"
                for item in mcp_payload.get("result", {}).get("tools", [])
                if isinstance(item, dict)
            )
            and config_exists
            and patch_exists
            and "impl Tool for HermesNativeTool" in native_tool_text
            and "jcode_tool_core" in native_tool_text
            and str(output) in str(payload.get("jcode_bridge", {}).get("schema_dir", ""))
        ),
        "payload": payload,
        "service_payload": service_payload,
        "mcp_payload": mcp_payload,
        "mcp_contract_payload": mcp_contract_payload,
        "latency_payload": latency_payload,
        "native_check_payload": native_check_payload,
        "native_registration_payload": native_registration_payload,
        "supertool_registry_payload": supertool_registry_payload,
        "copied_count": len(result.get("copied", [])),
        "native_tool_scaffold": str(native_tool),
    }


async def check_webhook_preflight_pass() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temp:
        fake = _fake_jcode(Path(temp))
        adapter = FakeWebhookAdapter({
            "dispatch": "jcode",
            "jcode": {
                "jcode_bin": str(fake),
                "preflight_contract": True,
                "preflight_live": True,
            },
        })
        gateway = SimpleNamespace(adapters={Platform.WEBHOOK: adapter})
        result = on_pre_gateway_dispatch(
            event=_webhook_event("hello after preflight"),
            gateway=gateway,
        )
        await _wait_for_delivery(adapter)
    content = adapter.sent[0]["content"] if adapter.sent else ""
    return {
        "ok": result == {
            "action": "skip",
            "reason": "webhook route 'hook' dispatched to jcode",
        } and content == "echo: hello after preflight",
        "content": content,
    }


async def check_webhook_preflight_blocks() -> dict[str, Any]:
    adapter = FakeWebhookAdapter({
        "dispatch": {
            "target": "jcode",
            "jcode_bin": "/tmp/definitely-missing-jcode",
            "preflight_contract": True,
            "preflight_live": True,
        },
    })
    gateway = SimpleNamespace(adapters={Platform.WEBHOOK: adapter})
    result = on_pre_gateway_dispatch(event=_webhook_event("do not run"), gateway=gateway)
    await _wait_for_delivery(adapter)
    content = adapter.sent[0]["content"] if adapter.sent else ""
    return {
        "ok": (
            result is not None
            and result.get("action") == "skip"
            and content == "jcode dispatch failed: jcode bridge contract preflight failed"
        ),
        "content": content,
    }


async def run_smokes() -> dict[str, Any]:
    checks = [
        _run_check("contract_tool", check_contract_tool),
        _run_check("safety_blocks_outbound_human_contact", check_safety_blocks_outbound),
        _run_check("contract_rejects_bad_json", check_contract_rejects_bad_json),
        _run_check("ensure_server_path", check_ensure_server_path),
        _run_check("hermes_service_contract", check_hermes_service_contract),
        _run_check("hermes_service_dispatch", check_hermes_service_dispatch),
        _run_check("hermes_service_blocks_send_message", check_hermes_service_blocks_send_message),
        _run_check("jcode_tool_hermes_client", check_jcode_tool_hermes_client),
        _run_check("hermes_mcp_server", check_hermes_mcp_server),
        _run_check("hermes_mcp_contract", check_hermes_mcp_contract),
        _run_check("jcode_native_registration_patch", check_jcode_native_registration_patch),
        _run_check("jcode_supertool_registry_smoke", check_jcode_supertool_registry_smoke),
        _run_check("mother_repo_scaffold", check_mother_repo_scaffold),
    ]

    for name, fn in (
        ("webhook_preflight_pass", check_webhook_preflight_pass),
        ("webhook_preflight_blocks", check_webhook_preflight_blocks),
    ):
        try:
            details = await fn()
        except Exception as exc:
            checks.append(_check(name, False, error=str(exc)))
            continue
        ok = bool(details.pop("ok", True))
        checks.append(_check(name, ok, **details))

    return {
        "success": all(item["ok"] for item in checks),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    report = asyncio.run(run_smokes())
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
