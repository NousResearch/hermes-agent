"""Tool handler for the local OpenClaw bridge."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import ssl
import urllib.error
import urllib.request
from typing import Any
from datetime import datetime, timezone
import uuid

import yaml


ALLOWED_TASKS = {"tasks.organize_today", "agents.ask_team"}
OPENCLAW_GATEWAY_URL_ENV = "OPENCLAW_GATEWAY_URL"
OPENCLAW_GATEWAY_TOKEN_ENV = "OPENCLAW_GATEWAY_TOKEN"
OPENCLAW_HERMES_BRIDGE_TOKEN_ENV = "OPENCLAW_HERMES_BRIDGE_TOKEN"
CLAWOPS_HOME_ENV = "CLAWOPS_HOME"
CLAWOPS_GEMINI_BASE_URL_ENV = "CLAWOPS_GEMINI_BASE_URL"
CLAWOPS_GEMINI_31_PRO_MODEL_ENV = "CLAWOPS_GEMINI_31_PRO_MODEL"
CLAWOPS_APPROVALS_DIR_ENV = "CLAWOPS_APPROVALS_DIR"
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
_CLAWOPS_HOST_LLM: Any = None
_CLAWOPS_CODEX_APP_SERVER_ENABLED: bool | None = None
_CLAWOPS_CODEX_APP_SERVER_SESSION_FACTORY: Any = None


def _json_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def set_clawops_host_llm(llm: Any) -> None:
    global _CLAWOPS_HOST_LLM
    _CLAWOPS_HOST_LLM = llm


def get_clawops_host_llm() -> Any:
    return _CLAWOPS_HOST_LLM


def set_clawops_codex_app_server_enabled(enabled: bool | None) -> None:
    global _CLAWOPS_CODEX_APP_SERVER_ENABLED
    _CLAWOPS_CODEX_APP_SERVER_ENABLED = enabled


def set_clawops_codex_app_server_session_factory(factory: Any) -> None:
    global _CLAWOPS_CODEX_APP_SERVER_SESSION_FACTORY
    _CLAWOPS_CODEX_APP_SERVER_SESSION_FACTORY = factory


def _env_ready() -> bool:
    return all(
        os.getenv(name)
        for name in (
            OPENCLAW_GATEWAY_URL_ENV,
            OPENCLAW_GATEWAY_TOKEN_ENV,
            OPENCLAW_HERMES_BRIDGE_TOKEN_ENV,
        )
    )


def _read_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _read_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _read_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = _read_string(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            items.append(normalized)
    return items


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _clawops_approvals_dir() -> Path:
    configured = _read_string(os.getenv(CLAWOPS_APPROVALS_DIR_ENV))
    if configured:
        return Path(configured)
    return Path.home() / ".hermes" / "clawops_approvals"


def _approval_id() -> str:
    return f"clawops-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"


def _approval_path(approval_id: str) -> Path:
    normalized = _read_string(approval_id) or ""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized):
        raise ValueError("Invalid ClawOps approval id.")
    return _clawops_approvals_dir() / f"{normalized}.json"


def _load_approval(approval_id: str) -> dict[str, Any]:
    path = _approval_path(approval_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"ClawOps approval not found: {approval_id}") from exc
    if not isinstance(data, dict):
        raise ValueError("ClawOps approval file is not an object.")
    return data


def _save_approval(record: dict[str, Any]) -> None:
    approval_id = _read_string(record.get("id"))
    if not approval_id:
        raise ValueError("ClawOps approval record is missing id.")
    directory = _clawops_approvals_dir()
    directory.mkdir(parents=True, exist_ok=True)
    _approval_path(approval_id).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_clawops_actions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    actions: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        action_type = _read_string(item.get("type"))
        if not action_type:
            continue
        actions.append(
            {
                "type": action_type,
                "description": _read_string(item.get("description")) or action_type,
                "payload": item.get("payload") if isinstance(item.get("payload"), dict) else {},
            }
        )
    return actions


def _clawops_actions_from_args(args: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    actions = _normalize_clawops_actions(args.get("actions"))
    if actions:
        return actions
    return _normalize_clawops_actions(result.get("actions") or result.get("proposedActions"))


def _attach_approval_if_needed(result: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    if result.get("ok") is not True or result.get("approvalRequired") is not True:
        return result

    actions = _clawops_actions_from_args(args, result)
    approval_id = _approval_id()
    record = {
        "id": approval_id,
        "status": "pending",
        "createdAt": _utc_now_iso(),
        "request": _normalize_task_value(args.get("request")) or _normalize_task_value(args.get("intent")),
        "project": result.get("project"),
        "taskType": result.get("taskType"),
        "assignedAgent": result.get("assignedAgent"),
        "modelUsed": result.get("modelUsed"),
        "output": result.get("output"),
        "actions": actions,
    }
    _save_approval(record)
    enriched = dict(result)
    enriched["approvalId"] = approval_id
    enriched["approvalStatus"] = "pending"
    enriched["executableActions"] = len(actions)
    return enriched


def _execute_audit_record_action(action: dict[str, Any], approval: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "at": _utc_now_iso(),
        "approvalId": approval.get("id"),
        "actionType": action.get("type"),
        "description": action.get("description"),
        "payload": action.get("payload") if isinstance(action.get("payload"), dict) else {},
    }
    audit_path = _clawops_approvals_dir() / "audit.log"
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return {"ok": True, "status": "executed", "type": "audit.record"}


def _execute_clawops_action(action: dict[str, Any], approval: dict[str, Any]) -> dict[str, Any]:
    action_type = _read_string(action.get("type")) or "unknown"
    if action_type == "audit.record":
        return _execute_audit_record_action(action, approval)
    return {
        "ok": False,
        "status": "blocked",
        "error": "unsupported_action",
        "message": f"ClawOps action type '{action_type}' is not allowlisted yet.",
        "type": action_type,
    }


def execute_clawops_approved_actions(approval_id: str) -> dict[str, Any]:
    try:
        approval = _load_approval(approval_id)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "ok": False,
            "status": "blocked",
            "error": "approval_not_found",
            "message": str(exc),
            "approvalId": _read_string(approval_id) or "unknown",
            "externalSideEffects": False,
        }

    current_status = _read_string(approval.get("status")) or "unknown"
    if current_status != "pending":
        return {
            "ok": False,
            "status": "blocked",
            "error": "approval_not_pending",
            "message": f"ClawOps approval is '{current_status}', not pending.",
            "approvalId": approval.get("id"),
            "externalSideEffects": False,
        }

    actions = _normalize_clawops_actions(approval.get("actions"))
    if not actions:
        approval["status"] = "blocked"
        approval["blockedAt"] = _utc_now_iso()
        approval["error"] = "no_actions"
        _save_approval(approval)
        return {
            "ok": False,
            "status": "blocked",
            "error": "no_actions",
            "message": "ClawOps approval has no executable actions.",
            "approvalId": approval.get("id"),
            "externalSideEffects": False,
        }

    executed: list[dict[str, Any]] = []
    for action in actions:
        action_result = _execute_clawops_action(action, approval)
        executed.append(action_result)
        if action_result.get("ok") is not True:
            approval["status"] = "blocked"
            approval["blockedAt"] = _utc_now_iso()
            approval["results"] = executed
            approval["error"] = action_result.get("error")
            _save_approval(approval)
            return {
                "ok": False,
                "status": "blocked",
                "error": _read_string(action_result.get("error")) or "action_failed",
                "message": _read_string(action_result.get("message")) or "ClawOps action did not execute.",
                "approvalId": approval.get("id"),
                "executedActions": sum(1 for item in executed if item.get("ok") is True),
                "externalSideEffects": False,
                "results": executed,
            }

    approval["status"] = "executed"
    approval["executedAt"] = _utc_now_iso()
    approval["results"] = executed
    _save_approval(approval)
    return {
        "ok": True,
        "status": "executed",
        "approvalId": approval.get("id"),
        "executedActions": len(executed),
        "externalSideEffects": True,
        "results": executed,
    }


def _clawops_docs_root() -> Path:
    configured = _read_string(os.getenv(CLAWOPS_HOME_ENV))
    if configured:
        return Path(configured)

    candidates = [
        Path("/Users/kj/my_agent_team/docs/projects/hub-ops"),
        Path.cwd() / "docs" / "projects" / "hub-ops",
        Path.cwd().parent / "docs" / "projects" / "hub-ops",
    ]
    for candidate in candidates:
        if (candidate / "agent-registry.yaml").is_file() and (candidate / "routing-rules.yaml").is_file():
            return candidate
    return candidates[0]


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"ClawOps config file not found: {path}") from exc
    if isinstance(data, dict):
        return data
    return {}


def _normalize_task_value(value: Any) -> str:
    text = _read_string(value)
    return text or ""


def _infer_clawops_project(text: str) -> str:
    normalized = text.lower()
    if "hahow" in normalized or "課程" in text:
        return "hahow_course"
    if "二手" in text or "secondhand" in normalized:
        return "secondhand_commerce"
    if "ingrids" in normalized or "網格" in text:
        return "ingrids_marketing"
    if "行銷" in text or "招生" in text or "marketing" in normalized:
        return "course_marketing"
    return "hub_ops"


def _infer_clawops_task_type(project: str, text: str) -> str:
    normalized = text.lower()
    if project == "hahow_course":
        return "course_design"
    if project == "secondhand_commerce":
        return "commerce_listing"
    if project == "ingrids_marketing":
        return "product_marketing"
    if "crm" in normalized or "名單" in text or "跟進" in text:
        return "crm"
    if "報告" in text or "週報" in text or "analytics" in normalized:
        return "analytics"
    if "部署" in text or "config" in normalized or "設定" in text:
        return "devops"
    if "策略" in text or "strategy" in normalized:
        return "strategy"
    if project == "course_marketing":
        return "campaign"
    return "strategy"


def _clawops_registry_context() -> str:
    docs_root = _clawops_docs_root()
    try:
        registry = _load_yaml_file(docs_root / "agent-registry.yaml")
    except FileNotFoundError as exc:
        return f"ClawOps registry context unavailable: {exc}"

    agents = registry.get("agents") if isinstance(registry.get("agents"), dict) else {}
    models = registry.get("models") if isinstance(registry.get("models"), dict) else {}
    lines = [
        "ClawOps registry context:",
        "Use this as grounding. Do not invent generic agent groups when answering responsibility/model questions.",
        "Agents:",
    ]
    for agent_id, agent in agents.items():
        if not isinstance(agent, dict):
            continue
        display_name = _normalize_task_value(agent.get("display_name")) or str(agent_id)
        role = _normalize_task_value(agent.get("role")) or "unknown"
        primary_model = _normalize_task_value(agent.get("primary_model")) or "unknown"
        fallback_model = _normalize_task_value(agent.get("fallback_model")) or "unknown"
        approval = "yes" if agent.get("approval_required") else "no"
        lines.append(
            f"- {display_name} ({agent_id}): role={role}; primary_model={primary_model}; "
            f"fallback_model={fallback_model}; approval_required={approval}"
        )

    if models:
        lines.append("Model policies:")
        for model_id, model in models.items():
            if not isinstance(model, dict):
                continue
            purpose = _normalize_task_value(model.get("purpose")) or "not specified"
            provider = _normalize_task_value(model.get("provider")) or "unknown"
            lines.append(f"- {model_id}: provider={provider}; purpose={purpose}")
    return "\n".join(lines)


def _match_route(route_match: dict[str, Any], task: dict[str, Any]) -> bool:
    for key, expected in route_match.items():
        actual = task.get(key)
        if actual != expected:
            return False
    return True


def _select_clawops_route(routing_rules: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    for route in routing_rules.get("routes", []):
        if not isinstance(route, dict):
            continue
        route_match = route.get("match")
        assignment = route.get("assign")
        if isinstance(route_match, dict) and isinstance(assignment, dict) and _match_route(route_match, task):
            return assignment
    default_route = routing_rules.get("default_route")
    if isinstance(default_route, dict):
        return default_route
    return {"agent": "orchestrator", "approval_required": True, "model_policy": "use_agent_primary"}


def _model_for_policy(policy: str, agent: dict[str, Any]) -> str:
    if policy == "force_codex":
        return "codex"
    return _normalize_task_value(agent.get("primary_model")) or "codex"


def route_clawops_task(args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Route a ClawOps task locally without external side effects."""
    del kwargs

    request_text = _normalize_task_value(args.get("request")) or _normalize_task_value(args.get("intent"))
    project = _normalize_task_value(args.get("project")) or _infer_clawops_project(request_text)
    task_type = _normalize_task_value(args.get("taskType")) or _normalize_task_value(args.get("task_type"))
    if not task_type:
        task_type = _infer_clawops_task_type(project, request_text)
    risk_level = _normalize_task_value(args.get("riskLevel")) or _normalize_task_value(args.get("risk_level")) or "low"

    docs_root = _clawops_docs_root()
    try:
        registry = _load_yaml_file(docs_root / "agent-registry.yaml")
        routing_rules = _load_yaml_file(docs_root / "routing-rules.yaml")
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "status": "blocked",
            "error": "missing_clawops_config",
            "message": str(exc),
            "dryRun": True,
            "externalSideEffects": False,
        }

    task = {"project": project, "task_type": task_type, "risk_level": risk_level}
    assignment = _select_clawops_route(routing_rules, task)
    assigned_agent = _normalize_task_value(assignment.get("agent")) or "orchestrator"
    agents = registry.get("agents") if isinstance(registry.get("agents"), dict) else {}
    agent = agents.get(assigned_agent)
    if not isinstance(agent, dict):
        return {
            "ok": False,
            "status": "blocked",
            "error": "unknown_clawops_agent",
            "message": f"ClawOps agent '{assigned_agent}' is not defined in agent-registry.yaml.",
            "dryRun": True,
            "externalSideEffects": False,
        }

    model_policy = _normalize_task_value(assignment.get("model_policy")) or "use_agent_primary"
    primary_model = _model_for_policy(model_policy, agent)
    fallback_model = _normalize_task_value(agent.get("fallback_model")) or "codex"
    approval_required = assignment.get("approval_required")
    if not isinstance(approval_required, bool):
        approval_required = bool(agent.get("approval_required", True))

    return {
        "ok": True,
        "status": "routed",
        "project": project,
        "taskType": task_type,
        "riskLevel": risk_level,
        "assignedAgent": assigned_agent,
        "agentDisplayName": _normalize_task_value(agent.get("display_name")) or assigned_agent,
        "primaryModel": primary_model,
        "fallbackModel": fallback_model,
        "approvalRequired": approval_required,
        "dryRun": True,
        "externalSideEffects": False,
        "message": "ClawOps route preview completed. No external side effects were performed.",
    }


def _clawops_routing_fields(routing: dict[str, Any]) -> dict[str, Any]:
    return {
        "project": routing.get("project"),
        "taskType": routing.get("taskType"),
        "riskLevel": routing.get("riskLevel"),
        "assignedAgent": routing.get("assignedAgent"),
        "agentDisplayName": routing.get("agentDisplayName"),
        "primaryModel": routing.get("primaryModel"),
        "fallbackModel": routing.get("fallbackModel"),
        "approvalRequired": routing.get("approvalRequired"),
        "dryRun": True,
        "externalModelCall": False,
        "externalSideEffects": False,
    }


def _redact_sensitive_text(text: str) -> str:
    redacted = text
    for name in (
        GOOGLE_API_KEY_ENV,
        GEMINI_API_KEY_ENV,
        OPENCLAW_GATEWAY_TOKEN_ENV,
        OPENCLAW_HERMES_BRIDGE_TOKEN_ENV,
    ):
        value = _read_string(os.getenv(name))
        if value:
            redacted = redacted.replace(value, "[REDACTED]")
    redacted = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", redacted, flags=re.IGNORECASE)
    return redacted[:2000]


def _enforce_bridge_dry_run_result(result: dict[str, Any]) -> dict[str, Any]:
    output = result.get("output")
    if not isinstance(output, dict):
        return result

    dry_run = output.get("dryRun")
    side_effects = output.get("sideEffectsPerformed")
    if dry_run is False or side_effects is True:
        return {
            "ok": False,
            "status": "blocked",
            "error": "unsafe_bridge_result",
            "message": "OpenClaw bridge returned a non-dry-run or side-effecting result; Hermes blocked the response.",
            "taskId": result.get("taskId"),
            "mode": result.get("mode"),
            "output": {
                "dryRun": dry_run,
                "sideEffectsPerformed": side_effects,
            },
        }
    return result


def _clawops_model_api_key() -> str | None:
    return _read_string(os.getenv(GOOGLE_API_KEY_ENV)) or _read_string(os.getenv(GEMINI_API_KEY_ENV))


def _clawops_api_model(model: str) -> str:
    if model == "gemini-3.1-pro":
        return _read_string(os.getenv(CLAWOPS_GEMINI_31_PRO_MODEL_ENV)) or "gemini-3.1-pro-preview"
    return model


def _clawops_gemini_chat_url() -> str:
    base_url = (
        _read_string(os.getenv(CLAWOPS_GEMINI_BASE_URL_ENV))
        or "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    return base_url.rstrip("/") + "/chat/completions"


def _clawops_https_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _clawops_system_prompt(routing: dict[str, Any]) -> str:
    approval = "required" if routing.get("approvalRequired") else "not required for this dry-run"
    return "\n".join(
        [
            "You are executing inside the Grace + ClawOps collaboration mechanism.",
            "Grace is the chief-of-staff interface: frame the user's request, preserve context, and return the final response.",
            "ClawOps is the multi-agent operations hub: route the task, select the agent/model policy, execute agent work, and enforce approval boundaries.",
            f"Assigned agent id: {_read_string(routing.get('assignedAgent')) or 'unknown'}",
            f"Project: {_read_string(routing.get('project')) or 'unknown'}",
            f"Task type: {_read_string(routing.get('taskType')) or 'unknown'}",
            "Boundary: dry-run only; do not perform external side effects, tool calls, purchases, writes, sends, deploys, or irreversible actions.",
            f"Approval boundary: approval is {approval}; any real-world execution must stop and request approval.",
            "For responsibility, routing, or model-assignment questions, answer from the ClawOps registry context and current collaboration mechanism, not from generic Strategy/Execution/Review/Ops categories.",
            "Safety: avoid guaranteed returns, individualized financial advice, or claims that reduce crypto investment risk to zero.",
            "Return the proposed output only.",
        ]
    )


def _extract_host_llm_text(response: Any) -> str | None:
    text = _read_string(getattr(response, "text", None))
    if text:
        return text
    if isinstance(response, dict):
        return _read_string(response.get("text")) or _read_string(response.get("output"))
    return _read_string(response)


def _codex_app_server_enabled() -> bool:
    if _CLAWOPS_CODEX_APP_SERVER_ENABLED is not None:
        return _CLAWOPS_CODEX_APP_SERVER_ENABLED
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        return model_cfg.get("openai_runtime") == "codex_app_server"
    except Exception:
        return False


def _codex_app_server_session_factory() -> Any:
    if _CLAWOPS_CODEX_APP_SERVER_SESSION_FACTORY is not None:
        return _CLAWOPS_CODEX_APP_SERVER_SESSION_FACTORY
    from agent.transports.codex_app_server_session import CodexAppServerSession

    return CodexAppServerSession


def _codex_app_server_prompt(args: dict[str, Any], routing: dict[str, Any]) -> str:
    request_text = _normalize_task_value(args.get("request")) or _normalize_task_value(args.get("intent"))
    return "\n\n".join(
        [
            _clawops_system_prompt(routing),
            _clawops_registry_context(),
            f"Original request:\n{request_text}",
        ]
    )


def _execute_clawops_codex_app_server_task(
    *,
    args: dict[str, Any],
    routing: dict[str, Any],
    fields: dict[str, Any],
) -> dict[str, Any]:
    session = None
    try:
        session_factory = _codex_app_server_session_factory()
        session = session_factory(cwd=str(Path.cwd()))
        session.ensure_started()
        turn = session.run_turn(user_input=_codex_app_server_prompt(args, routing))
        error = _read_string(getattr(turn, "error", None))
        if error:
            raise RuntimeError(error)
        output = _read_string(getattr(turn, "final_text", None))
        if not output:
            raise ValueError("Codex app-server response did not include final_text.")
        return {
            "ok": True,
            "status": "generated",
            **fields,
            "modelProvider": "codex_app_server",
            "modelUsed": "codex_app_server",
            "externalModelCall": True,
            "output": output,
            "message": "ClawOps Codex agent execution generated through codex app-server. No external side effects were performed.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "failed",
            "error": "codex_app_server_request_failed",
            "message": _redact_sensitive_text(str(exc)),
            **fields,
            "externalModelCall": True,
        }
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def _execute_clawops_codex_task(
    *,
    args: dict[str, Any],
    routing: dict[str, Any],
    fields: dict[str, Any],
) -> dict[str, Any]:
    if _codex_app_server_enabled():
        return _execute_clawops_codex_app_server_task(args=args, routing=routing, fields=fields)

    llm = get_clawops_host_llm()
    if llm is None:
        return {
            "ok": False,
            "status": "blocked",
            "error": "missing_host_llm",
            "message": "ClawOps Codex execution requires Hermes host LLM access from the live plugin context.",
            **fields,
        }

    request_text = _normalize_task_value(args.get("request")) or _normalize_task_value(args.get("intent"))
    messages = [
        {"role": "system", "content": "\n\n".join([_clawops_system_prompt(routing), _clawops_registry_context()])},
        {"role": "user", "content": f"Original request:\n{request_text}"},
    ]
    try:
        response = llm.complete(
            messages,
            temperature=0.3,
            max_tokens=1800,
            timeout=120,
            purpose=f"clawops:{_read_string(routing.get('assignedAgent')) or 'unknown'}",
        )
        output = _extract_host_llm_text(response)
        if not output:
            raise ValueError("Host LLM response did not include text.")
        return {
            "ok": True,
            "status": "generated",
            **fields,
            "modelProvider": _read_string(getattr(response, "provider", None)) or "host",
            "modelUsed": _read_string(getattr(response, "model", None)) or _read_string(routing.get("primaryModel")) or "codex",
            "externalModelCall": True,
            "output": output,
            "message": "ClawOps Codex agent execution generated through Hermes host LLM. No external side effects were performed.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "failed",
            "error": "host_llm_request_failed",
            "message": _redact_sensitive_text(str(exc)),
            **fields,
            "externalModelCall": True,
        }


def _execute_clawops_gemini_task(
    *,
    args: dict[str, Any],
    routing: dict[str, Any],
    fields: dict[str, Any],
    model: str,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    api_key = _clawops_model_api_key()
    if not api_key:
        result = {
            "ok": False,
            "status": "blocked",
            "error": "missing_model_api_key",
            "message": "Set GOOGLE_API_KEY or GEMINI_API_KEY before using ClawOps Gemini agent execution.",
            **fields,
        }
        if fallback_used:
            result["fallbackUsed"] = True
            result["fallbackReason"] = fallback_reason or "fallback"
        return result

    request_text = _normalize_task_value(args.get("request")) or _normalize_task_value(args.get("intent"))
    api_model = _clawops_api_model(model)
    payload = {
        "model": api_model,
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": "\n\n".join([_clawops_system_prompt(routing), _clawops_registry_context()])},
            {"role": "user", "content": f"Original request:\n{request_text}"},
        ],
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        _clawops_gemini_chat_url(),
        data=body,
        headers={
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60, context=_clawops_https_context()) as response:
            raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
        choices = parsed.get("choices") if isinstance(parsed, dict) else None
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        message = first_choice.get("message") if isinstance(first_choice, dict) else {}
        output = _read_string(message.get("content") if isinstance(message, dict) else None)
        if not output:
            raise ValueError("Model response did not include choices[0].message.content.")
        result = {
            "ok": True,
            "status": "generated",
            **fields,
            "modelUsed": api_model,
            "externalModelCall": True,
            "output": output,
            "message": "ClawOps agent execution generated. No external side effects were performed.",
        }
        if fallback_used:
            result["fallbackUsed"] = True
            result["fallbackReason"] = fallback_reason or "fallback"
        return result
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        result = {
            "ok": False,
            "status": "failed",
            "error": "model_http_error",
            "message": _redact_sensitive_text(raw or str(exc)),
            **fields,
            "externalModelCall": True,
        }
        if fallback_used:
            result["fallbackUsed"] = True
            result["fallbackReason"] = fallback_reason or "fallback"
        return result
    except (OSError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        result = {
            "ok": False,
            "status": "failed",
            "error": "model_request_failed",
            "message": _redact_sensitive_text(str(exc)),
            **fields,
            "externalModelCall": True,
        }
        if fallback_used:
            result["fallbackUsed"] = True
            result["fallbackReason"] = fallback_reason or "fallback"
        return result


def execute_clawops_task(args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Route and generate a ClawOps agent result without external side effects."""
    routing = route_clawops_task(args, **kwargs)
    if routing.get("ok") is not True:
        return routing

    fields = _clawops_routing_fields(routing)
    primary_model = _read_string(routing.get("primaryModel")) or ""
    if primary_model == "codex":
        codex_result = _execute_clawops_codex_task(args=args, routing=routing, fields=fields)
        if codex_result.get("ok") is True:
            return _attach_approval_if_needed(codex_result, args)
        fallback_model = _read_string(routing.get("fallbackModel")) or ""
        if fallback_model.lower().startswith("gemini"):
            return _attach_approval_if_needed(
                _execute_clawops_gemini_task(
                    args=args,
                    routing=routing,
                    fields=fields,
                    model=fallback_model,
                    fallback_used=True,
                    fallback_reason=_read_string(codex_result.get("error")) or "codex_unavailable",
                ),
                args=args,
            )
        return codex_result

    if not primary_model.lower().startswith("gemini"):
        return {
            "ok": False,
            "status": "blocked",
            "error": "unsupported_primary_model",
            "message": f"ClawOps agent execution only supports Gemini primary models; got '{primary_model or 'unknown'}'.",
            **fields,
        }

    return _attach_approval_if_needed(
        _execute_clawops_gemini_task(args=args, routing=routing, fields=fields, model=primary_model),
        args,
    )


def _validate_args(args: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    task_id = _read_string(args.get("taskId"))
    if task_id not in ALLOWED_TASKS:
        return None, "openclaw_delegate only allows taskId='tasks.organize_today' or taskId='agents.ask_team' in v1."

    if args.get("dryRun") is not True:
        return None, "openclaw_delegate requires dryRun=true."

    allowed_tools = _read_string_list(args.get("allowedTools"))
    if allowed_tools:
        return None, "openclaw_delegate requires allowedTools=[] in v1."

    intent = _read_string(args.get("intent")) or task_id
    input_obj = _read_object(args.get("input"))
    request_id = _read_string(args.get("requestId"))
    payload: dict[str, Any] = {
        "taskId": task_id,
        "requestedBy": "hermes",
        "intent": intent,
        "priority": "normal",
        "requiresConfirmation": False,
        "allowedTools": [],
        "dryRun": True,
        "input": input_obj,
    }
    if request_id:
        payload["requestId"] = request_id
        payload["idempotencyKey"] = request_id
    return payload, None


def _bridge_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/api/plugins/hermes-bridge/tasks"



def _looks_like_openclaw_dry_run(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized or normalized.startswith("/"):
        return False
    has_openclaw = "openclaw" in normalized
    has_dry_run = "dry-run" in normalized or "dry run" in normalized or "只做 dry" in normalized
    return has_openclaw and has_dry_run


def _task_id_for_text(text: str) -> str:
    normalized = text.lower()
    if "agent" in normalized or "團隊" in normalized or "team" in normalized:
        return "agents.ask_team"
    return "tasks.organize_today"


def build_openclaw_delegate_args(text: str) -> dict[str, Any]:
    task_id = _task_id_for_text(text)
    input_obj: dict[str, Any]
    if task_id == "agents.ask_team":
        input_obj = {"team": "openclaw", "question": text}
    else:
        input_obj = {"request": text}
    return {
        "taskId": task_id,
        "intent": text,
        "dryRun": True,
        "allowedTools": [],
        "input": input_obj,
    }



def format_openclaw_delegate_result(raw_result: str) -> str:
    try:
        result = json.loads(raw_result)
    except json.JSONDecodeError:
        return raw_result
    if not isinstance(result, dict):
        return raw_result

    status = _read_string(result.get("status")) or "unknown"
    summary = _read_string(result.get("summary")) or _read_string(result.get("message")) or "No summary returned."
    task_id = _read_string(result.get("taskId")) or "unknown"
    mode = _read_string(result.get("mode")) or "unknown"
    ok = result.get("ok") is True

    title = "OpenClaw dry-run completed" if ok else "OpenClaw dry-run did not complete"
    lines = [title, f"Status: {status}", f"Task: {task_id}", f"Mode: {mode}", f"Summary: {summary}"]

    output = result.get("output")
    if isinstance(output, dict):
        dry_run = output.get("dryRun")
        if isinstance(dry_run, bool):
            lines.append(f"Dry-run: {str(dry_run).lower()}")
        side_effects = output.get("sideEffectsPerformed")
        if isinstance(side_effects, bool):
            side_effect_text = "yes" if side_effects else "no"
            lines.append(f"External side effects: {side_effect_text}")
        organized_tasks = output.get("organizedTasks")
        if isinstance(organized_tasks, list):
            lines.append(f"Organized tasks: {len(organized_tasks)}")

    error = result.get("error")
    if isinstance(error, dict):
        error_type = _read_string(error.get("type")) or _read_string(result.get("error"))
        error_message = _read_string(error.get("message")) or _read_string(result.get("message"))
        if error_type:
            lines.append(f"Error: {error_type}")
        if error_message:
            lines.append(f"Error detail: {error_message}")
    elif isinstance(error, str) and error.strip():
        lines.append(f"Error: {error.strip()}")

    audit_log = result.get("auditLog")
    if isinstance(audit_log, list) and audit_log:
        lines.append("Audit log:")
        for item in audit_log[:5]:
            if not isinstance(item, dict):
                continue
            step = _read_string(item.get("step")) or "step"
            message = _read_string(item.get("message")) or ""
            lines.append(f"- {step}: {message}" if message else f"- {step}")

    return "\n".join(lines)

def handle_openclaw_dry_run_command(raw_args: str) -> str:
    text = _read_string(raw_args) or "請 OpenClaw 幫我整理今天的任務，但只做 dry-run。"
    return format_openclaw_delegate_result(openclaw_delegate(build_openclaw_delegate_args(text)))


def format_clawops_route_result(result: dict[str, Any]) -> str:
    if result.get("ok") is not True:
        return "\n".join(
            [
                "ClawOps route preview did not complete",
                f"Status: {_read_string(result.get('status')) or 'unknown'}",
                f"Error: {_read_string(result.get('error')) or 'unknown'}",
                f"Message: {_read_string(result.get('message')) or 'No detail returned.'}",
                "External side effects: no",
            ]
        )

    approval = "yes" if result.get("approvalRequired") else "no"
    return "\n".join(
        [
            "ClawOps route preview",
            f"Status: {_read_string(result.get('status')) or 'unknown'}",
            f"Project: {_read_string(result.get('project')) or 'unknown'}",
            f"Task type: {_read_string(result.get('taskType')) or 'unknown'}",
            f"Assigned agent: {_read_string(result.get('assignedAgent')) or 'unknown'}",
            f"Primary model: {_read_string(result.get('primaryModel')) or 'unknown'}",
            f"Fallback model: {_read_string(result.get('fallbackModel')) or 'unknown'}",
            f"Approval required: {approval}",
            "Dry-run: true",
            "External side effects: no",
        ]
    )


def handle_clawops_command(raw_args: str) -> str:
    text = _read_string(raw_args) or "請 ClawOps 執行任務。"
    return format_clawops_execution_result(execute_clawops_task({"request": text}))


def format_clawops_execution_result(result: dict[str, Any]) -> str:
    approval = "yes" if result.get("approvalRequired") else "no"
    external_actions = "pending your approval" if result.get("approvalRequired") else "none"
    lines = [
        "ClawOps agent execution",
        f"Status: {_read_string(result.get('status')) or 'unknown'}",
        f"Project: {_read_string(result.get('project')) or 'unknown'}",
        f"Task type: {_read_string(result.get('taskType')) or 'unknown'}",
        f"Assigned agent: {_read_string(result.get('assignedAgent')) or 'unknown'}",
        f"Model used: {_read_string(result.get('modelUsed')) or _read_string(result.get('primaryModel')) or 'unknown'}",
        f"Approval required: {approval}",
        f"External model call: {'yes' if result.get('externalModelCall') else 'no'}",
        f"External actions: {external_actions}",
    ]
    if result.get("fallbackUsed"):
        lines.append("Fallback used: yes")
        fallback_reason = _read_string(result.get("fallbackReason"))
        if fallback_reason:
            lines.append(f"Fallback reason: {fallback_reason}")
    approval_id = _read_string(result.get("approvalId"))
    if approval_id:
        lines.append(f"Approval id: {approval_id}")
        lines.append(f"Executable actions: {int(result.get('executableActions') or 0)}")
        lines.append(f"Approve command: /clawops-approve {approval_id}")

    if result.get("ok") is not True:
        lines.extend(
            [
                f"Error: {_read_string(result.get('error')) or 'unknown'}",
                f"Message: {_read_string(result.get('message')) or 'No detail returned.'}",
            ]
        )

    output = _read_string(result.get("output"))
    if output:
        lines.extend(["Output:", output])
    return "\n".join(lines)


def handle_clawops_run_command(raw_args: str) -> str:
    text = _read_string(raw_args) or "請 ClawOps 執行任務 dry-run。"
    return format_clawops_execution_result(execute_clawops_task({"request": text}))


def format_clawops_approval_result(result: dict[str, Any]) -> str:
    ok = result.get("ok") is True
    lines = [
        "ClawOps approved actions executed" if ok else "ClawOps approved actions did not execute",
        f"Status: {_read_string(result.get('status')) or 'unknown'}",
        f"Approval id: {_read_string(result.get('approvalId')) or 'unknown'}",
        f"Executed actions: {int(result.get('executedActions') or 0)}",
        f"External side effects: {'yes' if result.get('externalSideEffects') else 'no'}",
    ]
    if not ok:
        lines.append(f"Error: {_read_string(result.get('error')) or 'unknown'}")
        lines.append(f"Message: {_read_string(result.get('message')) or 'No detail returned.'}")
    return "\n".join(lines)


def handle_clawops_approve_command(raw_args: str) -> str:
    approval_id = _read_string(raw_args)
    if not approval_id:
        return format_clawops_approval_result(
            {
                "ok": False,
                "status": "blocked",
                "error": "missing_approval_id",
                "message": "Provide an approval id, for example: /clawops-approve clawops-20260627-1234abcd",
                "approvalId": "unknown",
                "externalSideEffects": False,
            }
        )
    return format_clawops_approval_result(execute_clawops_approved_actions(approval_id))


def pre_gateway_dispatch(**kwargs: Any) -> dict[str, str] | None:
    event = kwargs.get("event")
    text = getattr(event, "text", "")
    if not isinstance(text, str) or not _looks_like_openclaw_dry_run(text):
        if isinstance(text, str) and _looks_like_clawops_request(text):
            return {"action": "rewrite", "text": f"/clawops {text}"}
        return None
    return {"action": "rewrite", "text": f"/openclaw-dry-run {text}"}


def _looks_like_clawops_request(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized or normalized.startswith("/"):
        return False
    return "clawops" in normalized or "爪控中心" in text or "openclaw 營運中樞" in normalized

def openclaw_delegate(args: dict[str, Any], **kwargs: Any) -> str:
    """Delegate one approved dry-run task to OpenClaw."""
    del kwargs

    if not _env_ready():
        return _json_result(
            {
                "ok": False,
                "status": "blocked",
                "error": "missing_environment",
                "message": (
                    "Set OPENCLAW_GATEWAY_URL, OPENCLAW_GATEWAY_TOKEN, and "
                    "OPENCLAW_HERMES_BRIDGE_TOKEN before using openclaw_delegate."
                ),
            }
        )

    payload, error = _validate_args(args)
    if error:
        return _json_result(
            {
                "ok": False,
                "status": "blocked",
                "error": "invalid_request",
                "message": error,
            }
        )

    gateway_url = os.environ[OPENCLAW_GATEWAY_URL_ENV]
    gateway_token = os.environ[OPENCLAW_GATEWAY_TOKEN_ENV]
    bridge_token = os.environ[OPENCLAW_HERMES_BRIDGE_TOKEN_ENV]
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        _bridge_url(gateway_url),
        data=body,
        headers={
            "authorization": f"Bearer {gateway_token}",
            "content-type": "application/json",
            "x-openclaw-hermes-token": bridge_token,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            return _json_result(_enforce_bridge_dry_run_result(parsed))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"ok": False, "status": "failed", "error": "http_error", "message": raw}
        if isinstance(parsed.get("message"), str):
            parsed["message"] = _redact_sensitive_text(parsed["message"])
        return _json_result(parsed)
    except (OSError, TimeoutError, json.JSONDecodeError) as exc:
        return _json_result(
            {
                "ok": False,
                "status": "failed",
                "error": "bridge_request_failed",
                "message": str(exc),
            }
        )
