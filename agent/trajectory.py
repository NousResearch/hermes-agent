"""Trajectory saving utilities and static helpers.

_convert_to_trajectory_format stays as an AIAgent method (batch_runner.py
calls agent._convert_to_trajectory_format). Only the static helpers and
the file-write logic live here.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def convert_scratchpad_to_think(content: str) -> str:
    """Convert <REASONING_SCRATCHPAD> tags to <think> tags."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace("</REASONING_SCRATCHPAD>", "</think>")


def has_incomplete_scratchpad(content: str) -> bool:
    """Check if content has an opening <REASONING_SCRATCHPAD> without a closing tag."""
    if not content:
        return False
    return "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


def save_trajectory(trajectory: List[Dict[str, Any]], model: str,
                    completed: bool, filename: str = None):
    """Append a trajectory entry to a JSONL file.

    Args:
        trajectory: The ShareGPT-format conversation list.
        model: Model name for metadata.
        completed: Whether the conversation completed successfully.
        filename: Override output filename. Defaults to trajectory_samples.jsonl
                  or failed_trajectories.jsonl based on ``completed``.
    """
    if filename is None:
        filename = "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"

    entry = {
        "conversations": trajectory,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "completed": completed,
    }

    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Trajectory saved to %s", filename)
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)


_SIDE_EFFECT_TOOLS = frozenset({
    "terminal", "execute_code", "write_file", "patch", "process", "cronjob",
    "send_message", "skill_manage", "browser_click", "browser_type",
    "browser_press", "browser_scroll", "browser_back", "image_generate",
})
_EXTERNAL_ACTION_TOOLS = frozenset({"send_message", "cronjob", "image_generate"})
_FILE_ARG_KEYS = ("path", "file_path", "output_path", "workdir")
_VERIFIER_MARKERS = (
    "pytest", "unittest", "node --test", "npm test", "pnpm test", "yarn test",
    "py_compile", "mypy", "ruff", "eslint", "health", "curl ", "curl -",
    "browser_", "doctor", "status",
)
_SECRET_MARKERS = ("[redacted]", "redacted", "api_key", "token", "secret", "password")
_PII_MARKERS = ("email", "phone", "ssn", "address", "dob", "passport")


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _as_text(value: Any, limit: int = 2000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)[:limit]
    except Exception:
        return str(value)[:limit]


def _extract_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        return []
    out: List[Dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        if not isinstance(fn, dict):
            fn = {}
        name = fn.get("name") or call.get("name")
        if not isinstance(name, str) or not name:
            continue
        args = _safe_json_loads(fn.get("arguments", {}))
        out.append({
            "id": call.get("id") or call.get("call_id"),
            "name": name,
            "args": args if isinstance(args, dict) else {},
        })
    return out


def _tool_results_by_call_id(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    results: Dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        call_id = msg.get("tool_call_id")
        if isinstance(call_id, str) and call_id:
            results[call_id] = _as_text(msg.get("content"), 4000)
    return results


def _collect_file_paths(tool_name: str, args: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    for key in _FILE_ARG_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value.strip())
    if tool_name == "terminal":
        command = args.get("command")
        if isinstance(command, str):
            # Conservative extraction: obvious repo/local file paths in common shell commands.
            for match in re.findall(r"(?:^|\s)((?:[./~]|[\w.-]+/)[\w./-]+\.[A-Za-z0-9_+-]{1,12})(?=\s|$)", command):
                paths.append(match.strip())
    return paths


def _detect_external_actions(tool_name: str, args: Dict[str, Any]) -> List[str]:
    actions: List[str] = []
    if tool_name in _EXTERNAL_ACTION_TOOLS:
        actions.append(tool_name)
    command = args.get("command") if isinstance(args, dict) else None
    if tool_name == "terminal" and isinstance(command, str):
        lowered = command.lower()
        if re.search(r"\bgit\s+push\b", lowered):
            actions.append("git_push")
        if re.search(r"\b(curl|wget)\b", lowered):
            actions.append("http_probe")
        if re.search(r"\b(systemctl|service)\s+.*\b(restart|start|stop)\b", lowered):
            actions.append("service_control")
    return actions


def _detect_verifier(tool_name: str, args: Dict[str, Any], result_text: str) -> Dict[str, Any] | None:
    haystack = f"{tool_name} {_as_text(args)} {result_text}".lower()
    if not any(marker in haystack for marker in _VERIFIER_MARKERS):
        return None
    verifier_type = "healthcheck" if any(m in haystack for m in ("curl ", "curl -", "health", "browser_", "status", "doctor")) else "test"
    command = ""
    if isinstance(args, dict):
        command = args.get("command") or args.get("url") or args.get("path") or ""
    if re.search(r"exit_code['\"]?\s*[:=]\s*0", result_text.lower()) or re.search(r"\b(passed|pass|ok)\b", result_text.lower()):
        result = "pass"
    elif re.search(r"exit_code['\"]?\s*[:=]\s*[1-9]", result_text.lower()) or re.search(r"\b(failed|fail|error|traceback)\b", result_text.lower()):
        result = "fail"
    else:
        result = "unknown"
    return {"type": verifier_type, "command_or_endpoint": _as_text(command, 300), "result": result}


def _classify_intent(task_summary: str, tool_names: List[str]) -> str:
    text = f"{task_summary} {' '.join(tool_names)}".lower()
    if any(word in text for word in ("test", "patch", "fix", "implement", "code", "pytest", "write_file")):
        return "code_patch"
    if any(word in text for word in ("research", "scan", "source", "web", "browser", "search")):
        return "research"
    if any(word in text for word in ("cron", "monitor", "watch", "heartbeat", "status")):
        return "monitoring"
    return "ops"


def _privacy_gate(messages: List[Dict[str, Any]], tool_names: List[str]) -> Dict[str, Any]:
    combined = "\n".join(_as_text(m.get("content"), 500) for m in messages if isinstance(m, dict)).lower()
    durable_memory_write = any(name in {"memory", "skill_manage", "fact_store"} for name in tool_names)
    pii_risk = "possible" if any(marker in combined for marker in _PII_MARKERS) else "none"
    if pii_risk == "possible" and len(re.findall(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", combined)) >= 3:
        pii_risk = "high"
    return {
        "secret_redaction_applied": any(marker in combined for marker in _SECRET_MARKERS),
        "memory_context_fenced": "<memory" in combined or "memory" in tool_names,
        "durable_memory_write_detected": durable_memory_write,
        "pii_risk": pii_risk,
    }


def build_agent_run_trace(
    *,
    messages: List[Dict[str, Any]],
    run_id: str,
    started_at: str,
    ended_at: str,
    origin: str = "unknown",
    task_summary: str = "",
    completed: bool = False,
    interrupted: bool = False,
    turn_exit_reason: str | None = None,
) -> Dict[str, Any]:
    """Build a compact, transcript-derived summary for one agent run.

    This intentionally labels what happened; it does not enforce policy or persist
    raw sensitive data. The full transcript remains the source of truth.
    """
    messages = messages or []
    task_summary = _as_text(task_summary, 240).replace("\n", " ").strip()
    results_by_id = _tool_results_by_call_id(messages)
    tool_counts: Dict[str, int] = {}
    files_touched: List[str] = []
    external_actions: List[str] = []
    verifiers: List[Dict[str, Any]] = []
    tool_error_seen = False

    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for call in _extract_tool_calls(msg):
            name = call["name"]
            args = call.get("args") or {}
            tool_counts[name] = tool_counts.get(name, 0) + 1
            files_touched.extend(_collect_file_paths(name, args))
            external_actions.extend(_detect_external_actions(name, args))
            result_text = results_by_id.get(call.get("id") or "", "")
            if re.search(r"\b(error|traceback|exception|failed)\b", result_text.lower()):
                tool_error_seen = True
            verifier = _detect_verifier(name, args, result_text)
            if verifier:
                verifiers.append(verifier)

    tools_used = [
        {"name": name, "count": count, "side_effect": name in _SIDE_EFFECT_TOOLS}
        for name, count in sorted(tool_counts.items())
    ]
    tool_names = sorted(tool_counts)
    side_effect_tools = sorted({name for name in tool_names if name in _SIDE_EFFECT_TOOLS})
    files_touched = sorted(dict.fromkeys(files_touched))[:50]
    external_actions = sorted(dict.fromkeys(external_actions))

    if external_actions:
        risk_level = "high"
    elif side_effect_tools:
        risk_level = "medium"
    else:
        risk_level = "low"

    verifier = verifiers[-1] if verifiers else {"type": "none", "command_or_endpoint": "", "result": "none"}
    failure_mode = None
    if interrupted:
        failure_mode = "interrupted"
    elif turn_exit_reason and "max_iterations" in str(turn_exit_reason):
        failure_mode = "max_iterations"
    elif tool_error_seen and completed:
        failure_mode = "tool_error_recovered"
    elif not completed:
        failure_mode = "incomplete"

    outcome = "success" if completed and not interrupted else ("partial" if messages else "failed")
    if verifier.get("result") == "fail":
        outcome = "partial" if completed else "failed"

    promotion_skill = bool(tool_counts and (tool_error_seen or len(tool_counts) >= 5 or risk_level == "high"))
    promotion_eval = bool(tool_error_seen or verifier.get("result") == "fail" or failure_mode in {"max_iterations", "incomplete"})
    reason_bits = []
    if tool_error_seen:
        reason_bits.append("tool error observed")
    if risk_level != "low":
        reason_bits.append(f"{risk_level} risk tools/actions")
    if verifier.get("type") != "none":
        reason_bits.append(f"verifier {verifier.get('result')}")

    return {
        "schema_version": "agent_run_trace.v1",
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "origin": origin or "unknown",
        "task_summary": task_summary,
        "intent_type": _classify_intent(task_summary, tool_names),
        "risk_level": risk_level,
        "privacy_gate": _privacy_gate(messages, tool_names),
        "skills_loaded": sorted({
            (call.get("args") or {}).get("name")
            for msg in messages if isinstance(msg, dict) and msg.get("role") == "assistant"
            for call in _extract_tool_calls(msg)
            if call.get("name") == "skill_view" and isinstance((call.get("args") or {}).get("name"), str)
        }),
        "tools_used": tools_used,
        "side_effect_tools": side_effect_tools,
        "verifier_detected": verifier.get("type") != "none",
        "files_touched": files_touched,
        "external_actions": external_actions,
        "failure_mode": failure_mode,
        "verifier": verifier,
        "outcome": outcome,
        "promotion_hint": {
            "memory": False,
            "skill": promotion_skill,
            "eval_case": promotion_eval,
            "reason": "; ".join(reason_bits) or "low-complexity run",
        },
    }
