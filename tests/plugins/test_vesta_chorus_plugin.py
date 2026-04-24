import importlib.util
import json
from pathlib import Path

_PLUGIN = Path(__file__).resolve().parents[2] / "plugins" / "vesta-chorus" / "__init__.py"
_spec = importlib.util.spec_from_file_location("vesta_chorus_plugin_under_test", _PLUGIN)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


class FakeCtx:
    def __init__(self):
        self.hooks = []
        self.tools = []

    def register_hook(self, name, fn):
        self.hooks.append(name)

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)


def test_vesta_registers_hooks_and_tools():
    ctx = FakeCtx()
    mod.register(ctx)

    assert {"on_session_start", "on_session_end", "on_session_finalize", "pre_gateway_dispatch", "pre_llm_call", "post_tool_call"}.issubset(ctx.hooks)
    tool_names = {tool["name"] for tool in ctx.tools}
    assert tool_names == {
        "vesta_wake_briefing",
        "vesta_closeout",
        "vesta_worker_audit",
        "vesta_gate_check",
        "vesta_workstream_sweep",
    }
    assert {tool["toolset"] for tool in ctx.tools} == {"vesta_chorus"}


def test_vesta_gate_check_gates_high_risk_actions():
    for action in ["spend.api_subscription", "rotate.secret", "change.dns", "open.source", "open.source_release"]:
        result = json.loads(mod._gate_check({"action": action, "description": "operator request"}))
        assert result["requires_approval"] is True
        assert "Create a Chorus approval gate" in result["next_step"]


def test_vesta_gate_check_allows_scoped_worker_launch_by_inu_doctrine():
    result = json.loads(mod._gate_check({"action": "launch.agent_worker", "description": "spawn scoped tmux worker"}))
    assert result["requires_approval"] is False
    assert "audit-only" in result["reason"]


def test_vesta_unknown_tool_returns_error():
    result = json.loads(mod._tool_handler({}, name="missing"))
    assert "error" in result


def test_tool_audit_redacts_secret_like_args(monkeypatch):
    calls = []

    class FakeClient:
        def safe_rpc(self, method, params):
            calls.append((method, params))
            return {"ok": True, "result": {}}

    monkeypatch.setattr(mod, "_enabled", lambda: True)
    monkeypatch.setattr(mod, "_is_vesta_context", lambda: True)
    monkeypatch.setattr(mod, "_client", lambda: FakeClient())

    mod._post_tool_call(tool_name="terminal", args={"command": "run --api_key=abc123", "CHORUS_API_KEY": "abc123"})

    assert calls
    content = calls[0][1]["content"]
    assert "abc123" not in content
    assert "[REDACTED]" in content


def test_wake_briefing_degrades_when_resume_unavailable(monkeypatch):
    class FakeClient:
        def safe_rpc(self, method, params):
            if method == "workflow/resume":
                return {"ok": False, "error": "offline"}
            return {"ok": True, "result": {"ok": method}}

    monkeypatch.setattr(mod, "_client", lambda: FakeClient())
    result = json.loads(mod._wake_briefing({"cwd": "/tmp"}))

    assert result["briefing"] == ""
    assert result["resume_status"] == {"error": "offline"}
    assert result["identity"] == {"ok": "identity/whoami"}
