from __future__ import annotations

import pytest


def test_delegated_task_and_result_schema_validation_accepts_required_payloads():
    from plugins.openclaw_bridge.schemas import (
        validate_delegated_result,
        validate_delegated_task,
    )

    task = {
        "task_id": "task-1",
        "requested_by": "hermes",
        "objective": "Read project status",
        "context_refs": ["obsidian:System/Tasks.md"],
        "allowed_tools": ["status_check"],
        "denied_tools": ["external_message"],
        "risk_level": "low",
        "requires_confirmation": False,
        "max_runtime_seconds": 60,
        "output_format": "markdown",
        "audit_required": True,
    }
    result = {
        "task_id": "task-1",
        "status": "succeeded",
        "summary": "Status checked",
        "artifacts": [],
        "tool_calls": [],
        "audit_log": [],
        "errors": [],
        "requires_human_review": False,
        "recommended_next_action": "none",
    }

    assert validate_delegated_task(task) == task
    assert validate_delegated_result(result) == result


def test_delegated_task_schema_rejects_missing_required_field():
    from plugins.openclaw_bridge.schemas import validate_delegated_task

    with pytest.raises(ValueError, match="objective"):
        validate_delegated_task({"task_id": "task-1"})


def test_high_risk_task_stops_at_approval_gate():
    from plugins.openclaw_bridge.tools import delegate_to_openclaw

    result = delegate_to_openclaw(
        {
            "objective": "Deploy production",
            "risk_level": "high",
            "allowed_tools": ["deploy"],
            "requested_by": "hermes",
        },
        transport=lambda _task: {"status": "succeeded"},
    )

    assert result["status"] == "blocked"
    assert result["requires_human_review"] is True
    assert "approval" in result["recommended_next_action"].lower()


def test_critical_risk_task_stops_at_approval_gate():
    from plugins.openclaw_bridge.tools import delegate_to_openclaw

    result = delegate_to_openclaw(
        {
            "objective": "Rotate production credentials",
            "risk_level": "critical",
            "allowed_tools": ["credentials"],
            "requested_by": "hermes",
        },
        transport=lambda _task: {"status": "succeeded"},
    )

    assert result["status"] == "blocked"
    assert result["requires_human_review"] is True
    assert "risk_level=critical" in result["summary"]


def test_requires_confirmation_stops_before_transport():
    from plugins.openclaw_bridge.tools import delegate_to_openclaw

    def fail_transport(_task):
        raise AssertionError("requires_confirmation tasks must not reach OpenClaw")

    result = delegate_to_openclaw(
        {
            "objective": "Send a Telegram message",
            "risk_level": "low",
            "requires_confirmation": True,
            "allowed_tools": ["telegram.send"],
            "requested_by": "hermes",
        },
        transport=fail_transport,
    )

    assert result["status"] == "blocked"
    assert result["requires_human_review"] is True


def test_low_risk_task_builds_valid_delegated_task_and_uses_transport():
    from plugins.openclaw_bridge.tools import delegate_to_openclaw

    seen = []

    def transport(task):
        seen.append(task)
        return {
            "task_id": task["task_id"],
            "status": "succeeded",
            "summary": "ok",
            "artifacts": [],
            "tool_calls": [],
            "audit_log": [],
            "errors": [],
            "requires_human_review": False,
            "recommended_next_action": "none",
        }

    result = delegate_to_openclaw(
        {
            "objective": "Read status",
            "risk_level": "low",
            "allowed_tools": ["status_check"],
            "requested_by": "hermes",
            "context_refs": ["obsidian:System/Tasks.md"],
        },
        transport=transport,
    )

    assert result["status"] == "succeeded"
    assert len(seen) == 1
    assert seen[0]["objective"] == "Read status"
    assert seen[0]["requires_confirmation"] is False


def test_external_facebook_work_is_not_sent_to_dry_run_bridge():
    from plugins.openclaw_bridge.tools import delegate_to_openclaw

    def fail_transport(_task):
        raise AssertionError("dry-run bridge must not receive real Facebook work")

    result = delegate_to_openclaw(
        {
            "objective": "檢查 Facebook 20 個社團，可刊登就實際刊登貼文",
            "risk_level": "low",
            "allowed_tools": ["status_check"],
            "requested_by": "hermes",
            "context_refs": ["kanban:t_16d6dfe3"],
        },
        transport=fail_transport,
    )

    assert result["status"] == "blocked"
    assert result["requires_human_review"] is True
    assert "Facebook" in result["summary"]
    assert "browser-capable executor" in result["recommended_next_action"]


def test_openclaw_delegate_handler_accepts_registry_keyword_args(monkeypatch):
    import json

    from plugins.openclaw_bridge import tools

    def fake_delegate(args):
        return {
            "task_id": args["task_id"],
            "status": "succeeded",
            "summary": args["objective"],
            "artifacts": [],
            "tool_calls": [],
            "audit_log": [],
            "errors": [],
            "requires_human_review": False,
            "recommended_next_action": "none",
        }

    monkeypatch.setattr(tools, "delegate_to_openclaw", fake_delegate)

    output = tools.handle_openclaw_delegate(
        {"objective": "Check Facebook status"},
        task_id="kanban-t-1",
    )

    result = json.loads(output)
    assert result["task_id"] == "kanban-t-1"
    assert result["summary"] == "Check Facebook status"


def test_plugin_registers_tool_command_and_gateway_hook():
    from plugins.openclaw_bridge import register

    calls = {"tools": [], "commands": [], "hooks": []}

    class Ctx:
        def register_tool(self, **kwargs):
            calls["tools"].append(kwargs["name"])

        def register_command(self, name, **_kwargs):
            calls["commands"].append(name)

        def register_hook(self, name, _handler):
            calls["hooks"].append(name)

    register(Ctx())

    assert calls == {
        "tools": ["openclaw_delegate"],
        "commands": ["openclaw-dry-run"],
        "hooks": ["pre_gateway_dispatch"],
    }


def test_pre_gateway_dispatch_routes_clawops_requests_to_runtime_queue():
    from types import SimpleNamespace

    from plugins.openclaw_bridge.tools import pre_gateway_dispatch

    event = SimpleNamespace(text="ClawOps: check bridge wiring")
    assert pre_gateway_dispatch(event=event) == {
        "action": "rewrite",
        "text": "/clawops check bridge wiring",
    }

    spaced = SimpleNamespace(text="ClawOps check queue callback")
    assert pre_gateway_dispatch(event=spaced) == {
        "action": "rewrite",
        "text": "/clawops check queue callback",
    }


def test_pre_gateway_dispatch_routes_facebook_publish_work_to_clawops():
    from types import SimpleNamespace

    from plugins.openclaw_bridge.tools import pre_gateway_dispatch

    event = SimpleNamespace(text="請繼續 #7 咖啡器材新舊交流團的刊登流程，只允許點 Next，不要送出")
    assert pre_gateway_dispatch(event=event) == {
        "action": "rewrite",
        "text": "/clawops 請繼續 #7 咖啡器材新舊交流團的刊登流程，只允許點 Next，不要送出",
    }

    upload = SimpleNamespace(text="Facebook 社團商品表單照片上傳後檢查 Next 是否解除鎖定")
    assert pre_gateway_dispatch(event=upload) == {
        "action": "rewrite",
        "text": "/clawops Facebook 社團商品表單照片上傳後檢查 Next 是否解除鎖定",
    }


def test_pre_gateway_dispatch_does_not_route_facebook_explanation_to_clawops():
    from types import SimpleNamespace

    from plugins.openclaw_bridge.tools import pre_gateway_dispatch

    event = SimpleNamespace(text="請說明 Facebook 社團刊登有哪些風險")
    assert pre_gateway_dispatch(event=event) is None


def test_pre_gateway_dispatch_keeps_openclaw_requests_as_bridge_preview():
    from types import SimpleNamespace

    from plugins.openclaw_bridge.tools import pre_gateway_dispatch

    event = SimpleNamespace(text="OpenClaw: check bridge wiring")
    assert pre_gateway_dispatch(event=event) == {
        "action": "rewrite",
        "text": "/openclaw-dry-run check bridge wiring",
    }

    normal = SimpleNamespace(text="請說明 OpenClaw 是什麼")
    assert pre_gateway_dispatch(event=normal) is None


def test_openclaw_dry_run_slash_does_not_enqueue_kanban_by_default(monkeypatch):
    from plugins.openclaw_bridge.tools import handle_openclaw_dry_run

    def fail_run_slash(_command):
        raise AssertionError("kanban should not be called unless explicitly requested")

    monkeypatch.setattr("hermes_cli.kanban.run_slash", fail_run_slash)
    monkeypatch.delenv("OPENCLAW_GATEWAY_URL", raising=False)
    monkeypatch.delenv("OPENCLAW_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("OPENCLAW_HERMES_BRIDGE_TOKEN", raising=False)

    output = handle_openclaw_dry_run("check bridge wiring")

    assert "OpenClaw bridge result" in output
    assert "status: blocked" in output
    assert "Kanban enqueue result" not in output


def test_openclaw_dry_run_can_explicitly_enqueue_kanban(monkeypatch):
    from plugins.openclaw_bridge.tools import handle_openclaw_dry_run

    seen = []

    def fake_run_slash(command):
        seen.append(command)
        return "Created t_abc123  (ready, assignee=-)"

    monkeypatch.setattr("hermes_cli.kanban.run_slash", fake_run_slash)
    monkeypatch.delenv("OPENCLAW_GATEWAY_URL", raising=False)
    monkeypatch.delenv("OPENCLAW_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("OPENCLAW_HERMES_BRIDGE_TOKEN", raising=False)

    output = handle_openclaw_dry_run("kanban check runtime queue")

    assert "Kanban enqueue result" in output
    assert "Created t_abc123" in output
    assert len(seen) == 1
    assert seen[0].startswith("create ")
    assert "--created-by hermes-openclaw-bridge" in seen[0]


def test_openclaw_dry_run_local_mode_keeps_old_validation_only_path(monkeypatch):
    from plugins.openclaw_bridge.tools import handle_openclaw_dry_run

    monkeypatch.delenv("OPENCLAW_GATEWAY_URL", raising=False)
    monkeypatch.delenv("OPENCLAW_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("OPENCLAW_HERMES_BRIDGE_TOKEN", raising=False)

    output = handle_openclaw_dry_run("local check bridge wiring")

    assert "status: queued" in output
    assert "no OpenClaw transport configured in this process" in output


def test_low_risk_task_posts_to_openclaw_bridge_when_configured(monkeypatch):
    import json

    from plugins.openclaw_bridge import tools

    monkeypatch.setenv("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:18789")
    monkeypatch.setenv("OPENCLAW_GATEWAY_TOKEN", "gateway-token")
    monkeypatch.setenv("OPENCLAW_HERMES_BRIDGE_TOKEN", "bridge-token")

    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return json.dumps(
                {
                    "ok": True,
                    "taskId": "agents.ask_team",
                    "status": "succeeded",
                    "summary": "Dry-run completed. No OpenClaw agents were started.",
                    "auditLog": [{"step": "accepted"}],
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["headers"] = dict(request.header_items())
        seen["timeout"] = timeout
        seen["payload"] = json.loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setattr(tools, "urlopen", fake_urlopen)

    result = tools.delegate_to_openclaw(
        {
            "objective": "Ask team for status",
            "risk_level": "low",
            "allowed_tools": ["status_check"],
            "requested_by": "hermes",
            "context_refs": ["telegram:test"],
        }
    )

    assert result["status"] == "succeeded"
    assert result["summary"] == "Dry-run completed. No OpenClaw agents were started."
    assert seen["url"] == "http://127.0.0.1:18789/api/plugins/hermes-bridge/tasks"
    assert seen["headers"]["Authorization"] == "Bearer gateway-token"
    assert seen["headers"]["X-openclaw-hermes-token"] == "bridge-token"
    assert seen["payload"]["taskId"] == "agents.ask_team"
    assert seen["payload"]["dryRun"] is True
    assert seen["payload"]["requestedBy"] == "hermes"
    assert seen["payload"]["input"]["objective"] == "Ask team for status"


def test_openclaw_payload_contract_forces_dry_run_and_fixed_route():
    from urllib.parse import urljoin

    from plugins.openclaw_bridge import tools

    task = tools.build_delegated_task(
        {
            "task_id": "contract-1",
            "objective": "Check bridge status",
            "risk_level": "low",
            "allowed_tools": ["status_check"],
            "requested_by": "hermes",
            "context_refs": ["telegram:test"],
            "output_format": "markdown",
        }
    )
    config = tools.OpenClawBridgeConfig(
        base_url="http://127.0.0.1:18789",
        gateway_token="gateway-token",
        bridge_token="bridge-token",
    )

    payload = tools._openclaw_payload(task, config)

    assert urljoin(config.base_url + "/", tools.DEFAULT_OPENCLAW_BRIDGE_PATH.lstrip("/")) == (
        "http://127.0.0.1:18789/api/plugins/hermes-bridge/tasks"
    )
    assert payload["taskId"] == "agents.ask_team"
    assert payload["dryRun"] is True
    assert payload["allowedTools"] == ["status_check"]
    assert payload["requiresConfirmation"] is False
    assert payload["idempotencyKey"] == "contract-1"


def test_openclaw_bridge_config_can_read_tokens_from_env_file(monkeypatch, tmp_path):
    from plugins.openclaw_bridge import tools

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "OPENCLAW_GATEWAY_URL=http://127.0.0.1:18789",
                "OPENCLAW_GATEWAY_TOKEN=gateway-token",
                "OPENCLAW_HERMES_BRIDGE_TOKEN=bridge-token",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENCLAW_GATEWAY_URL", raising=False)
    monkeypatch.delenv("OPENCLAW_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("OPENCLAW_HERMES_BRIDGE_TOKEN", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"openclaw_bridge": {"env_file": str(env_file)}},
    )

    config = tools.load_openclaw_bridge_config()

    assert config is not None
    assert config.base_url == "http://127.0.0.1:18789"
    assert config.gateway_token == "gateway-token"
    assert config.bridge_token == "bridge-token"
