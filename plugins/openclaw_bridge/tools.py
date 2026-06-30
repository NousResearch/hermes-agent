from __future__ import annotations

import json
import os
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from plugins.openclaw_bridge.schemas import validate_delegated_result, validate_delegated_task
from proactive.tool_policy import PolicyLevel, decide_action, load_tool_policy


DEFAULT_OPENCLAW_BRIDGE_PATH = "/api/plugins/hermes-bridge/tasks"
DEFAULT_OPENCLAW_TEMPLATE = "agents.ask_team"


@dataclass(frozen=True)
class OpenClawBridgeConfig:
    base_url: str
    gateway_token: str
    bridge_token: str
    task_template: str = DEFAULT_OPENCLAW_TEMPLATE
    timeout_seconds: int = 30


OPENCLAW_DELEGATE_SCHEMA = {
    "type": "object",
    "properties": {
        "objective": {"type": "string"},
        "context_refs": {"type": "array", "items": {"type": "string"}},
        "allowed_tools": {"type": "array", "items": {"type": "string"}},
        "denied_tools": {"type": "array", "items": {"type": "string"}},
        "risk_level": {"type": "string"},
        "requires_confirmation": {"type": "boolean"},
        "max_runtime_seconds": {"type": "integer"},
        "output_format": {"type": "string"},
        "audit_required": {"type": "boolean"},
        "requested_by": {"type": "string"},
        "openclaw_task_id": {"type": "string"},
    },
    "required": ["objective"],
}


def build_delegated_task(args: dict[str, Any]) -> dict[str, Any]:
    risk = str(args.get("risk_level") or "medium").lower()
    requires_confirmation = bool(args.get("requires_confirmation", risk in {"high", "critical"}))
    task = {
        "task_id": str(args.get("task_id") or f"openclaw-{uuid.uuid4().hex[:12]}"),
        "requested_by": str(args.get("requested_by") or "hermes"),
        "objective": str(args.get("objective") or "").strip(),
        "context_refs": list(args.get("context_refs") or []),
        "allowed_tools": list(args.get("allowed_tools") or []),
        "denied_tools": list(args.get("denied_tools") or []),
        "risk_level": risk,
        "requires_confirmation": requires_confirmation,
        "max_runtime_seconds": int(args.get("max_runtime_seconds") or 300),
        "output_format": str(args.get("output_format") or "markdown"),
        "audit_required": bool(args.get("audit_required", True)),
    }
    return validate_delegated_task(task)


def _blocked_result(task: dict[str, Any] | None, reason: str) -> dict[str, Any]:
    return {
        "task_id": (task or {}).get("task_id", ""),
        "status": "blocked",
        "summary": reason,
        "artifacts": [],
        "tool_calls": [],
        "audit_log": [reason],
        "errors": [reason],
        "requires_human_review": True,
        "recommended_next_action": "Ask KJ for approval before delegating to OpenClaw.",
    }


def _default_transport(task: dict[str, Any]) -> dict[str, Any]:
    config = load_openclaw_bridge_config()
    if config is None:
        return {
            "task_id": task["task_id"],
            "status": "blocked",
            "summary": (
                "OpenClaw bridge transport is not configured. Set "
                "OPENCLAW_GATEWAY_URL, OPENCLAW_GATEWAY_TOKEN, and "
                "OPENCLAW_HERMES_BRIDGE_TOKEN, or configure openclaw_bridge "
                "in Hermes config."
            ),
            "artifacts": [],
            "tool_calls": [],
            "audit_log": ["Hermes blocked delegation before OpenClaw because bridge config is incomplete."],
            "errors": ["openclaw_bridge_not_configured"],
            "requires_human_review": True,
            "recommended_next_action": "Configure OpenClaw bridge URL and tokens, then retry the dry-run.",
        }
    return post_to_openclaw_bridge(task, config)


def _placeholder_transport(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "status": "queued",
        "summary": "DelegatedTask validated; no OpenClaw transport configured in this process.",
        "artifacts": [],
        "tool_calls": [],
        "audit_log": ["Hermes retained conversation authority; OpenClaw direct reply disabled."],
        "errors": [],
        "requires_human_review": False,
        "recommended_next_action": "Configure the OpenClaw bridge transport to execute this task.",
    }


def _config_get(mapping: dict[str, Any], *names: str) -> str:
    current: Any = mapping
    for name in names:
        if not isinstance(current, dict):
            return ""
        current = current.get(name)
    return str(current or "").strip()


def _read_env_file_value(path: str, key: str) -> str:
    if not path:
        return ""
    env_path = Path(path).expanduser()
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    prefix = f"{key}="
    export_prefix = f"export {key}="
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        value = ""
        if line.startswith(export_prefix):
            value = line[len(export_prefix) :]
        elif line.startswith(prefix):
            value = line[len(prefix) :]
        else:
            continue
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        return value.strip()
    return ""


def load_openclaw_bridge_config() -> OpenClawBridgeConfig | None:
    base_url = os.getenv("OPENCLAW_HERMES_BRIDGE_URL", "").strip()
    if not base_url:
        base_url = os.getenv("OPENCLAW_GATEWAY_URL", "").strip()
    gateway_token = os.getenv("OPENCLAW_GATEWAY_TOKEN", "").strip()
    bridge_token = os.getenv("OPENCLAW_HERMES_BRIDGE_TOKEN", "").strip()
    task_template = os.getenv("OPENCLAW_HERMES_TASK_TEMPLATE", "").strip() or DEFAULT_OPENCLAW_TEMPLATE
    timeout_seconds = int(os.getenv("OPENCLAW_HERMES_TIMEOUT_SECONDS", "30") or "30")

    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        bridge_cfg = cfg.get("openclaw_bridge") if isinstance(cfg, dict) else {}
        openclaw_cfg = cfg.get("openclaw") if isinstance(cfg, dict) else {}
        if isinstance(bridge_cfg, dict):
            base_url = base_url or _config_get(bridge_cfg, "url") or _config_get(bridge_cfg, "base_url")
            gateway_token = gateway_token or _config_get(bridge_cfg, "gateway_token")
            bridge_token = bridge_token or _config_get(bridge_cfg, "bridge_token")
            task_template = _config_get(bridge_cfg, "task_template") or task_template
            env_file = _config_get(bridge_cfg, "env_file")
            if env_file:
                base_url = base_url or _read_env_file_value(env_file, "OPENCLAW_GATEWAY_URL")
                gateway_token = gateway_token or _read_env_file_value(env_file, "OPENCLAW_GATEWAY_TOKEN")
                bridge_token = bridge_token or _read_env_file_value(env_file, "OPENCLAW_HERMES_BRIDGE_TOKEN")
            if bridge_cfg.get("timeout_seconds"):
                timeout_seconds = int(bridge_cfg["timeout_seconds"])
        if isinstance(openclaw_cfg, dict):
            base_url = base_url or _config_get(openclaw_cfg, "gateway_url") or _config_get(openclaw_cfg, "url")
    except Exception:
        pass

    if not (base_url and gateway_token and bridge_token):
        return None
    if base_url.startswith("ws://"):
        base_url = "http://" + base_url[len("ws://") :]
    elif base_url.startswith("wss://"):
        base_url = "https://" + base_url[len("wss://") :]
    return OpenClawBridgeConfig(
        base_url=base_url.rstrip("/"),
        gateway_token=gateway_token,
        bridge_token=bridge_token,
        task_template=task_template,
        timeout_seconds=timeout_seconds,
    )


def _openclaw_payload(task: dict[str, Any], config: OpenClawBridgeConfig) -> dict[str, Any]:
    template = str(task.get("openclaw_task_id") or config.task_template or DEFAULT_OPENCLAW_TEMPLATE)
    return {
        "taskId": template,
        "requestedBy": task["requested_by"],
        "intent": task["objective"],
        "priority": "normal",
        "requiresConfirmation": bool(task["requires_confirmation"]),
        "allowedTools": list(task.get("allowed_tools") or []),
        "input": {
            "objective": task["objective"],
            "contextRefs": list(task.get("context_refs") or []),
            "delegatedTaskId": task["task_id"],
            "outputFormat": task["output_format"],
        },
        "dryRun": True,
        "idempotencyKey": task["task_id"],
    }


def post_to_openclaw_bridge(task: dict[str, Any], config: OpenClawBridgeConfig) -> dict[str, Any]:
    payload = _openclaw_payload(task, config)
    url = urljoin(config.base_url + "/", DEFAULT_OPENCLAW_BRIDGE_PATH.lstrip("/"))
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {config.gateway_token}",
            "x-openclaw-hermes-token": config.bridge_token,
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        return {
            "task_id": task["task_id"],
            "status": "failed",
            "summary": f"OpenClaw bridge HTTP error: {exc.code}",
            "artifacts": [],
            "tool_calls": [{"name": "openclaw_bridge_http", "url": url, "status": exc.code}],
            "audit_log": ["Hermes sent DelegatedTask to OpenClaw bridge; OpenClaw returned HTTP error."],
            "errors": [f"http_{exc.code}"],
            "requires_human_review": True,
            "recommended_next_action": "Check OpenClaw gateway/plugin status and bridge token configuration.",
        }
    except URLError as exc:
        return {
            "task_id": task["task_id"],
            "status": "failed",
            "summary": f"OpenClaw bridge connection failed: {exc.reason}",
            "artifacts": [],
            "tool_calls": [{"name": "openclaw_bridge_http", "url": url, "status": "connection_failed"}],
            "audit_log": ["Hermes attempted OpenClaw bridge delegation but could not connect."],
            "errors": ["connection_failed"],
            "requires_human_review": True,
            "recommended_next_action": "Start OpenClaw gateway and confirm OPENCLAW_GATEWAY_URL.",
        }
    except TimeoutError:
        return {
            "task_id": task["task_id"],
            "status": "failed",
            "summary": "OpenClaw bridge request timed out.",
            "artifacts": [],
            "tool_calls": [{"name": "openclaw_bridge_http", "url": url, "status": "timeout"}],
            "audit_log": ["Hermes attempted OpenClaw bridge delegation but timed out."],
            "errors": ["timeout"],
            "requires_human_review": True,
            "recommended_next_action": "Check OpenClaw gateway health and task runtime.",
        }

    try:
        openclaw_result = json.loads(raw)
    except json.JSONDecodeError:
        openclaw_result = {"ok": False, "status": "failed", "summary": raw}

    ok = bool(openclaw_result.get("ok", openclaw_result.get("status") == "succeeded"))
    status = str(openclaw_result.get("status") or ("succeeded" if ok else "failed"))
    if status not in {"queued", "running", "succeeded", "failed", "blocked"}:
        status = "succeeded" if ok else "failed"
    audit = openclaw_result.get("auditLog") or openclaw_result.get("audit_log") or []
    return {
        "task_id": task["task_id"],
        "status": status,
        "summary": str(openclaw_result.get("summary") or "OpenClaw bridge returned a result."),
        "artifacts": list(openclaw_result.get("artifacts") or []),
        "tool_calls": [{"name": "openclaw_bridge_http", "url": url, "template": payload["taskId"]}],
        "audit_log": audit if isinstance(audit, list) else [audit],
        "errors": [] if ok else [openclaw_result.get("error") or "openclaw_bridge_failed"],
        "requires_human_review": bool(openclaw_result.get("requiresHumanReview", not ok)),
        "recommended_next_action": str(openclaw_result.get("recommendedNextAction") or ("Review OpenClaw result." if not ok else "Return summarized result to KJ.")),
    }


def delegate_to_openclaw(
    args: dict[str, Any],
    *,
    transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    policy_path: str | None = None,
) -> dict[str, Any]:
    task = build_delegated_task(args)
    risk = task["risk_level"]
    if task["requires_confirmation"] or risk in {"high", "critical"}:
        return _blocked_result(task, f"Delegated task risk_level={risk} requires approval.")

    policy = load_tool_policy(policy_path)
    for action in task["allowed_tools"]:
        decision = decide_action(action, policy)
        if decision.level is PolicyLevel.DENY:
            return _blocked_result(task, f"Tool policy denied delegated action: {action}")
        if decision.level is PolicyLevel.CONFIRM_FIRST and risk != "low":
            return _blocked_result(task, f"Delegated action requires confirmation: {action}")

    result = (transport or _default_transport)(task)
    if "task_id" not in result:
        result = {"task_id": task["task_id"], **result}
    return validate_delegated_result(result)


def handle_openclaw_delegate(args: dict[str, Any]) -> str:
    return json.dumps(delegate_to_openclaw(args), ensure_ascii=False)


def _objective_from_raw_args(raw_args: str) -> str:
    raw = (raw_args or "").strip()
    if not raw:
        return "OpenClaw bridge dry-run"
    return raw


def _notification_summary(result: dict[str, Any]) -> str:
    status = result.get("status", "unknown")
    task_id = result.get("task_id", "")
    summary = result.get("summary", "")
    review = "yes" if result.get("requires_human_review") else "no"
    next_action = result.get("recommended_next_action", "")
    return (
        "OpenClaw bridge result\n"
        f"- task_id: {task_id}\n"
        f"- status: {status}\n"
        f"- human_review: {review}\n"
        f"- summary: {summary}\n"
        f"- next_action: {next_action}"
    )


def _create_kanban_task_for_delegation(objective: str, result: dict[str, Any]) -> str:
    from hermes_cli.kanban import run_slash

    title = f"OpenClaw delegated task: {objective[:80]}"
    body = (
        "Hermes-created ClawOps/OpenClaw bridge task.\n\n"
        "OpenClaw remains execution-only; all user-facing decisions stay with Hermes.\n\n"
        "Delegated result:\n"
        f"```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```"
    )
    command = (
        "create "
        f"{shlex.quote(title)} "
        f"--body {shlex.quote(body)} "
        "--created-by hermes-openclaw-bridge "
        "--workspace scratch"
    )
    return run_slash(command)


def handle_openclaw_dry_run(raw_args: str) -> str:
    """Slash command handler for `/openclaw-dry-run`.

    Default mode validates the Hermes-side bridge only and does not enqueue a
    runtime task. Prefix with `kanban` to explicitly create a kanban task after
    validation, which gives KJ a visible runtime queue record without letting
    OpenClaw own the conversation.
    """
    raw = (raw_args or "").strip()
    create_kanban = False
    if raw.lower().startswith("kanban "):
        create_kanban = True
        raw = raw[7:].strip()
    use_placeholder = False
    if raw.lower().startswith("local "):
        use_placeholder = True
        raw = raw[6:].strip()

    objective = _objective_from_raw_args(raw)
    result = delegate_to_openclaw(
        {
            "objective": objective,
            "risk_level": "low",
            "allowed_tools": ["status_check"],
            "requested_by": "hermes",
            "context_refs": ["telegram:openclaw-dry-run"],
            "max_runtime_seconds": 60,
            "output_format": "markdown",
            "audit_required": True,
        },
        transport=_placeholder_transport if use_placeholder else None,
    )

    response = _notification_summary(result)
    if create_kanban:
        kanban_output = _create_kanban_task_for_delegation(objective, result)
        response = f"{response}\n\nKanban enqueue result:\n{kanban_output}"
    return response


def pre_gateway_dispatch(*, event: Any, **_kwargs: Any) -> dict[str, str] | None:
    """Conservatively rewrite explicit ClawOps/OpenClaw bridge requests.

    This intentionally avoids broad intent detection. Hermes should not send
    ordinary chat to OpenClaw just because the text mentions it.
    """
    text = str(getattr(event, "text", "") or "").strip()
    if not text or text.startswith("/"):
        return None

    lowered = text.lower()
    openclaw_prefixes = ("openclaw:", "openclaw ")
    if lowered.startswith(openclaw_prefixes):
        _, _, rest = text.partition(" ")
        if ":" in text.split(" ", 1)[0]:
            rest = text.split(":", 1)[1]
        rest = rest.strip() or "OpenClaw bridge dry-run"
        return {"action": "rewrite", "text": f"/openclaw-dry-run {rest}"}

    clawops_prefixes = ("clawops:", "clawops ")
    if lowered.startswith(clawops_prefixes):
        _, _, rest = text.partition(" ")
        if ":" in text.split(" ", 1)[0]:
            rest = text.split(":", 1)[1]
        rest = rest.strip() or "ClawOps runtime task"
        return {"action": "rewrite", "text": f"/clawops {rest}"}

    return None
