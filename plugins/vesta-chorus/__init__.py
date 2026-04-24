"""Vesta ↔ Chorus standalone plugin.

This plugin complements the Chorus memory provider:
- lifecycle hooks emit lightweight session/audit signals
- pre_llm_call can inject a compact wake/watch context
- high-level Vesta tools encode governance doctrine over raw Chorus RPC
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from plugins.chorus_common import ChorusClient, compact_resume, emit_signal
from tools.registry import tool_error

_APPROVAL_ACTIONS = {
    "deploy.production",
    "spend.api_subscription",
    "contact.customer",
    "publish.public",
    "modify.secret",
    "rotate.secret",
    "change.dns",
    "open.source",
    "open.source_release",
    "create.profit_cell",
    "archive.memory",
    "close.workstream",
}
_ALLOWED_WORKER_ACTION = "launch.agent_worker"
# Worker launch is Vesta-native, but it is still audited. This encodes Inu's
# explicit exception to the older broad approval-boundary wording.
_AUDIT_ONLY_ACTIONS = {_ALLOWED_WORKER_ACTION}
_RISKY_TOOL_NAMES = {"terminal", "patch", "write_file", "cronjob", "delegate_task", "mcp_chorus_chorus_create_approval_gate", "mcp_chorus_chorus_approve_approval_gate"}
_SECRET_KEYS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH", "CREDENTIAL")


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _client() -> ChorusClient:
    return ChorusClient.from_env(timeout=float(os.getenv("CHORUS_TIMEOUT", "20")))


def _enabled() -> bool:
    # If the plugin is enabled in Hermes config, hooks should run by default.
    # Operators can set VESTA_CHORUS_HOOKS=0 to make it tools-only.
    return os.getenv("VESTA_CHORUS_HOOKS", "1").strip().lower() not in {"0", "false", "no", "off"}


def _is_vesta_context() -> bool:
    profile = os.getenv("HERMES_PROFILE", "") or os.getenv("HERMES_AGENT_IDENTITY", "")
    if profile.lower() == "vesta":
        return True
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home().name == "vesta" or str(get_hermes_home()).endswith("/profiles/vesta")
    except Exception:
        return False


def _short(text: str, n: int = 1000) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n - 15] + "…[truncated]"


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if any(marker in str(key).upper() for marker in _SECRET_KEYS):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"(?i)(api[_-]?key|token|secret|password|authorization)\s*[:=]\s*\S+", r"\1=[REDACTED]", value)
    return value


def _wake_briefing(args: Dict[str, Any]) -> str:
    c = _client()
    cwd = args.get("cwd") or os.getcwd()
    resume_result = c.safe_rpc("workflow/resume", {"cwd": cwd, "compact": True, "limit": int(args.get("limit", 8)), "max_tokens": int(args.get("max_tokens", 12000))})
    who = c.safe_rpc("identity/whoami", {})
    webhooks = c.safe_rpc("webhooks/list", {})
    resume = resume_result.get("result") if resume_result.get("ok") else {}
    text = compact_resume(resume) if resume else ""
    return _json({
        "briefing": text,
        "resume_status": "ok" if resume_result.get("ok") else {"error": resume_result.get("error")},
        "identity": who.get("result") if who.get("ok") else {"error": who.get("error")},
        "webhooks": webhooks.get("result") if webhooks.get("ok") else {"error": webhooks.get("error")},
        "note": "Approval gates/Circles/workstreams beyond workflow resume require Chorus MCP tools when not exposed through JSON-RPC.",
    })


def _closeout(args: Dict[str, Any]) -> str:
    c = _client()
    summary = args.get("summary") or "Vesta closeout"
    ring = args.get("to_ring") or "agents-of-proto"
    tags = args.get("tags") or ["vesta", "closeout", "hermes"]
    allowed_signal_types = {"pulse", "sense", "task", "query", "alert", "artifact", "proposal", "shift"}
    signal_type = args.get("signal_type", "sense")
    if signal_type not in allowed_signal_types:
        signal_type = "sense"
    signal = emit_signal(c, content=summary, signal_type=signal_type, to_ring=ring, urgency=max(0.0, min(1.0, float(args.get("urgency", 0.35)))), tags=tags if isinstance(tags, list) else [str(tags)], from_role=args.get("from_role", "ops"))
    memory = None
    if args.get("store_memory", True):
        safe_tags = tags if isinstance(tags, list) else [str(tags)]
        memory = c.safe_rpc("memory/note", {"content": summary, "namespace": args.get("namespace", "ring:agents-of-proto"), "tags": safe_tags})
    return _json({"signal": signal, "memory": memory})


def _worker_audit(args: Dict[str, Any]) -> str:
    c = _client()
    content = (
        "Vesta worker audit\n"
        f"bead: {args.get('bead_id','')}\n"
        f"worker_session: {args.get('worker_session','')}\n"
        f"result: {_short(args.get('result',''), 1500)}\n"
        f"files: {', '.join(args.get('files') or [])}"
    )
    signal = emit_signal(c, content=content, signal_type="sense", to_ring=args.get("to_ring", "agents-of-proto"), urgency=float(args.get("urgency", 0.3)), tags=["vesta", "worker-audit", "hermes"], from_role="ops")
    return _json({"signal": signal})


def _gate_check(args: Dict[str, Any]) -> str:
    action = (args.get("action") or "").strip()
    description = args.get("description") or ""
    requires = action in _APPROVAL_ACTIONS
    if action in _AUDIT_ONLY_ACTIONS:
        requires = False
    lower = f"{action} {description}".lower()
    heuristic_hits = [w for w in ["production", "spend", "subscription", "secret", "publish", "customer", "delete", "destroy", "legal"] if w in lower]
    if heuristic_hits and action not in _AUDIT_ONLY_ACTIONS:
        requires = True
    return _json({
        "action": action,
        "requires_approval": requires,
        "heuristic_hits": heuristic_hits,
        "reason": "Scoped Vesta worker spawning is audit-only by Inu's explicit doctrine; spend/secrets/deploy/customer/public/destructive actions stay gated unless an active Circle authorizes them.",
        "next_step": "Create a Chorus approval gate via MCP before execution." if requires else "Proceed inside scoped operational doctrine; emit audit signal for nontrivial work.",
    })


def _workstream_sweep(args: Dict[str, Any]) -> str:
    # JSON-RPC currently exposes workflow/resume/next but not full workstream list.
    # This tool still gives Vesta a one-call sweep from available RPC surfaces.
    c = _client()
    cwd = args.get("cwd") or os.getcwd()
    resume = c.safe_rpc("workflow/resume", {"cwd": cwd, "compact": True, "limit": int(args.get("limit", 12)), "max_tokens": int(args.get("max_tokens", 14000))})
    nxt = c.safe_rpc("workflow/next", {"cwd": cwd})
    return _json({"resume": resume, "next": nxt, "note": "For full workstream list/update, use Chorus MCP workstream tools."})


def _tool_handler(args: Dict[str, Any], *, name: str) -> str:
    try:
        if name == "vesta_wake_briefing":
            return _wake_briefing(args)
        if name == "vesta_closeout":
            return _closeout(args)
        if name == "vesta_worker_audit":
            return _worker_audit(args)
        if name == "vesta_gate_check":
            return _gate_check(args)
        if name == "vesta_workstream_sweep":
            return _workstream_sweep(args)
        return tool_error(f"unknown tool {name}")
    except Exception as exc:
        return tool_error(str(exc))


def _on_session_start(**kwargs) -> None:
    if not _enabled() or not _is_vesta_context():
        return
    c = _client()
    content = f"Vesta session started: session={kwargs.get('session_id','')}, model={kwargs.get('model','')}."
    c.safe_rpc("signals/batch_emit", {"signals": [{"signal_type": "pulse", "content": content, "from_role": "ops", "to_ring": "fleet", "urgency": 0.2, "tags": ["vesta", "session-start", "hermes"], "resources": [], "attachments": []}]})


def _on_session_end(**kwargs) -> None:
    if not _enabled() or not _is_vesta_context():
        return
    c = _client()
    content = f"Vesta session finalized: session={kwargs.get('session_id','')}."
    c.safe_rpc("signals/batch_emit", {"signals": [{"signal_type": "sense", "content": content, "from_role": "ops", "to_ring": "fleet", "urgency": 0.2, "tags": ["vesta", "session-end", "hermes"], "resources": [], "attachments": []}]})


def _pre_llm_call(**kwargs):
    if not _enabled() or not _is_vesta_context():
        return None
    text = kwargs.get("user_message") or ""
    if not re.search(r"\b(wake|brief|status|chorus|gate|worker|deploy|spend|secret|publish|customer)\b", text, re.I):
        return None
    c = _client()
    res = c.safe_rpc("workflow/resume", {"cwd": os.getcwd(), "compact": True, "limit": 5, "max_tokens": 6000})
    if not res.get("ok"):
        return None
    return {"context": compact_resume(res.get("result"), max_items=3)}


def _pre_gateway_dispatch(**kwargs):
    if not _enabled() or not _is_vesta_context():
        return None
    event = kwargs.get("event")
    text = getattr(event, "text", "") or ""
    if re.search(r"(?i)\b(api[_-]?key|token|password|secret)\b", text):
        c = _client()
        c.safe_rpc("memory/note", {"content": "Vesta gateway observed credential-like inbound text; content intentionally not stored.", "namespace": "ring:agents-of-proto", "tags": ["vesta", "gateway", "redaction"]})
    return {"action": "allow"}


def _post_tool_call(**kwargs) -> None:
    if not _enabled() or not _is_vesta_context():
        return
    tool_name = kwargs.get("tool_name") or ""
    if tool_name not in _RISKY_TOOL_NAMES:
        return
    args = kwargs.get("args") or {}
    summary = f"Vesta observed tool call: {tool_name} args={_short(json.dumps(_redact(args), ensure_ascii=False, default=str), 700)}"
    c = _client()
    c.safe_rpc("memory/note", {"content": summary, "namespace": "ring:agents-of-proto", "tags": ["vesta", "tool-audit", tool_name]})


def register(ctx) -> None:
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("on_session_finalize", _on_session_end)
    ctx.register_hook("pre_gateway_dispatch", _pre_gateway_dispatch)
    ctx.register_hook("pre_llm_call", _pre_llm_call)
    ctx.register_hook("post_tool_call", _post_tool_call)

    ctx.register_tool(
        name="vesta_wake_briefing",
        toolset="vesta_chorus",
        schema={"name": "vesta_wake_briefing", "description": "Run Vesta's one-call Chorus wake briefing: identity, resume context, inbox/tasks/recent memory, and webhook registry.", "parameters": {"type": "object", "properties": {"cwd": {"type": "string"}, "limit": {"type": "integer"}, "max_tokens": {"type": "integer"}}, "required": []}},
        handler=lambda args, **kw: _tool_handler(args, name="vesta_wake_briefing"),
        check_fn=lambda: _client().is_configured(),
    )
    ctx.register_tool(
        name="vesta_closeout",
        toolset="vesta_chorus",
        schema={"name": "vesta_closeout", "description": "Emit a Vesta closeout signal and optionally mirror the summary into Chorus memory.", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}, "to_ring": {"type": "string"}, "namespace": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}, "store_memory": {"type": "boolean"}, "signal_type": {"type": "string", "enum": ["pulse", "sense", "task", "query", "alert", "artifact", "proposal", "shift"]}, "urgency": {"type": "number", "minimum": 0, "maximum": 1}}, "required": ["summary"]}},
        handler=lambda args, **kw: _tool_handler(args, name="vesta_closeout"),
        check_fn=lambda: _client().is_configured(),
    )
    ctx.register_tool(
        name="vesta_worker_audit",
        toolset="vesta_chorus",
        schema={"name": "vesta_worker_audit", "description": "Record a worker/bead audit into Chorus as a signal.", "parameters": {"type": "object", "properties": {"bead_id": {"type": "string"}, "worker_session": {"type": "string"}, "result": {"type": "string"}, "files": {"type": "array", "items": {"type": "string"}}, "to_ring": {"type": "string"}, "urgency": {"type": "number"}}, "required": ["result"]}},
        handler=lambda args, **kw: _tool_handler(args, name="vesta_worker_audit"),
        check_fn=lambda: _client().is_configured(),
    )
    ctx.register_tool(
        name="vesta_gate_check",
        toolset="vesta_chorus",
        schema={"name": "vesta_gate_check", "description": "Check whether a proposed Vesta action crosses an approval boundary.", "parameters": {"type": "object", "properties": {"action": {"type": "string"}, "description": {"type": "string"}}, "required": ["action"]}},
        handler=lambda args, **kw: _tool_handler(args, name="vesta_gate_check"),
    )
    ctx.register_tool(
        name="vesta_workstream_sweep",
        toolset="vesta_chorus",
        schema={"name": "vesta_workstream_sweep", "description": "Sweep available Chorus workflow surfaces and return resume + next-action posture.", "parameters": {"type": "object", "properties": {"cwd": {"type": "string"}, "limit": {"type": "integer"}, "max_tokens": {"type": "integer"}}, "required": []}},
        handler=lambda args, **kw: _tool_handler(args, name="vesta_workstream_sweep"),
        check_fn=lambda: _client().is_configured(),
    )
