"""APEX RuntimeOS opt-in runtime rewrite and durable-write helpers.

The helpers in this module are deliberately conservative:
- disabled by default;
- deterministic and side-effect scoped;
- no raw prompts, messages, final responses, paths, or credentials are written;
- runtime rewrites currently only call Hermes' existing context compressor;
- candidate promotion writes only sanitized, generated entries and keeps rollback data.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hermes_constants import get_hermes_home
from tools.memory_tool import MemoryStore

_TRUE = {"1", "true", "yes", "on"}
_CODE_RE = re.compile(r"^[a-zA-Z0-9_\-]{3,80}$")


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE


def _mode() -> str:
    value = os.environ.get("APEX_RUNTIMEOS_GATE_MODE", "warn").strip().lower() or "warn"
    return value if value in {"dry_run", "warn", "enforce"} else "warn"


def _out_dir() -> Path:
    configured = os.environ.get("APEX_RUNTIMEOS_AUTOWRITE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".hermes" / "apex_runtimeos_autowrites"


def _promotion_audit_path() -> Path:
    configured = os.environ.get("APEX_RUNTIMEOS_PROMOTION_AUDIT_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return _out_dir() / "promotions.jsonl"


def _candidate_path() -> Path:
    return _out_dir() / "candidates.jsonl"


def _cron_ledger_path() -> Path:
    configured = os.environ.get("APEX_RUNTIMEOS_CRON_LEDGER_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return _out_dir() / "cron_dryrun_ledger.jsonl"


def _safe_scalar(value: Any, *, limit: int = 160) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    text = str(value)
    lower = text.lower()
    if any(hint in lower for hint in ("key=", "token=", "authorization", "password", "secret")):
        return "[REDACTED]"
    if "/Users/" in text or text.startswith("/") or "\\" in text:
        return "[REDACTED_PATH]"
    return text[:limit]


def _safe_code(value: Any) -> str:
    text = str(value or "unknown").strip()
    return text if _CODE_RE.match(text) else "unknown"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _redact_for_ledger(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        allowed = {
            "task", "ts", "enabled", "mode", "dry_run", "target", "ready_count",
            "promoted", "rolled_back", "skipped", "events_count", "reason",
            "status", "exit_code", "schema", "ledger_key", "plan_hash", "count",
            "first_seen_at", "last_seen_at", "redaction_version",
        }
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in {"secret", "token", "api_key", "authorization", "password", "old_text", "old_content", "path"}:
                out[key_text] = "[REDACTED]"
            elif key_text in allowed:
                out[key_text] = _redact_for_ledger(item)
        return out
    if isinstance(value, list):
        return [_redact_for_ledger(item) for item in value[:20]]
    return _safe_scalar(value, limit=240)


def _promotion_id(record: Mapping[str, Any]) -> str:
    return str(record.get("promotion_id") or record.get("content_hash") or _hash_text(json.dumps(record, sort_keys=True, ensure_ascii=False)))


def _rollback_done_ids(records: list[Dict[str, Any]]) -> set[str]:
    done = set()
    for item in records:
        if item.get("schema") == "ApexRuntimeOSRollbackEvent/v1" and item.get("rollback_status") in {"done", "skipped"}:
            pid = str(item.get("promotion_id") or "")
            if pid:
                done.add(pid)
    return done


def _read_jsonl(path: Path, *, limit: int = 1000) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    out: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if idx >= limit:
                break
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                out.append(item)
    return out


def _recommendation_items(checkpoint: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    recs = checkpoint.get("recommendations") if isinstance(checkpoint, Mapping) else None
    if isinstance(recs, Mapping) and isinstance(recs.get("items"), list):
        return [item for item in recs["items"] if isinstance(item, Mapping)]
    return []


def build_runtime_rewrite_plan(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a deterministic rewrite plan from RuntimeOS recommendations."""
    enabled = _flag("APEX_RUNTIMEOS_AUTO_REWRITE_ENABLED", "0")
    mode = _mode()
    actions: list[Dict[str, Any]] = []
    for item in _recommendation_items(checkpoint):
        code = str(item.get("code") or "")
        rec_actions_raw = item.get("actions")
        rec_actions = rec_actions_raw if isinstance(rec_actions_raw, list) else []
        if code == "planner_context_heavy" or "compress_context" in rec_actions:
            actions.append({
                "type": "compress_context",
                "reason_code": code or "planner_context_heavy",
                "severity": str(item.get("severity") or "info"),
            })
    if not enabled or mode != "enforce":
        actions = []
    return {
        "enabled": enabled,
        "mode": mode,
        "applied": False,
        "mutates_runtime": bool(actions),
        "actions": actions[:1],
        "reason": "ready" if actions else ("disabled" if not enabled else "no_supported_action"),
    }


def apply_runtime_rewrite(
    *,
    agent: Any,
    messages: list,
    system_message: Optional[str],
    active_system_prompt: Optional[str],
    approx_tokens: int,
    task_id: Optional[str],
    checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply the safe subset of runtime rewrites and return new messages."""
    plan = build_runtime_rewrite_plan(checkpoint)
    result: Dict[str, Any] = {
        **plan,
        "messages": messages,
        "active_system_prompt": active_system_prompt,
        "before_message_count": len(messages or []),
        "after_message_count": len(messages or []),
    }
    if not plan.get("actions"):
        return result
    action = plan["actions"][0]
    if action.get("type") != "compress_context":
        result["reason"] = "unsupported_action"
        result["actions"] = []
        result["mutates_runtime"] = False
        return result
    if not getattr(agent, "compression_enabled", False) or not getattr(agent, "context_compressor", None):
        result["reason"] = "compressor_unavailable"
        result["actions"] = []
        result["mutates_runtime"] = False
        return result
    try:
        new_messages, new_prompt = agent._compress_context(
            messages,
            system_message,
            approx_tokens=approx_tokens,
            task_id=task_id,
        )
    except Exception as exc:  # pragma: no cover - defensive path
        result.update({"reason": "rewrite_error", "error_code": type(exc).__name__})
        return result
    result.update({
        "messages": new_messages,
        "active_system_prompt": new_prompt,
        "after_message_count": len(new_messages or []),
        "applied": len(new_messages or []) < len(messages or []),
        "reason": "applied" if len(new_messages or []) < len(messages or []) else "no_effect",
    })
    return result


def persist_autowrite_candidate(
    *,
    stage: str,
    session_id: str = "",
    recommendations: Optional[Mapping[str, Any]] = None,
    source: str = "apex_runtimeos",
) -> Dict[str, Any]:
    """Persist a sanitized memory/skill write candidate."""
    enabled = _flag("APEX_RUNTIMEOS_AUTO_WRITE_ENABLED", "0")
    mode = _mode()
    if not enabled or mode != "enforce":
        return {"enabled": enabled, "mode": mode, "written": False, "reason": "disabled_or_not_enforce"}
    recs = recommendations if isinstance(recommendations, Mapping) else {}
    rec_items = recs.get("items") if isinstance(recs.get("items"), list) else []
    items = rec_items if isinstance(rec_items, list) else []
    safe_items = []
    for item in items[:8]:
        if not isinstance(item, Mapping):
            continue
        item_actions_raw = item.get("actions")
        item_actions = item_actions_raw if isinstance(item_actions_raw, list) else []
        safe_items.append({
            "organ": _safe_scalar(item.get("organ")),
            "code": _safe_code(item.get("code")),
            "severity": _safe_scalar(item.get("severity")),
            "actions": [_safe_scalar(action) for action in item_actions[:6]],
            "applied": bool(item.get("applied")),
        })
    record = {
        "schema": "ApexRuntimeOSAutoWriteCandidate/v1",
        "ts": time.time(),
        "stage": _safe_scalar(stage),
        "session_id_hash": _hash_text(session_id) if session_id else "",
        "source": _safe_scalar(source),
        "candidate_type": "memory_or_skill_review",
        "promotion_required": True,
        "applied_to_core_memory_or_skill": False,
        "recommendation_status": _safe_scalar(recs.get("status")),
        "items": safe_items,
    }
    out_dir = _out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "candidates.jsonl"
    _append_jsonl(out, record)
    return {
        "enabled": True,
        "mode": mode,
        "written": True,
        "candidate_path": str(out),
        "candidate_type": record["candidate_type"],
        "promotion_required": True,
        "applied_to_core_memory_or_skill": False,
    }


def _candidate_to_memory_entry(candidate: Mapping[str, Any]) -> str:
    codes = []
    severities = []
    for item in candidate.get("items", []) if isinstance(candidate.get("items"), list) else []:
        if not isinstance(item, Mapping):
            continue
        code = _safe_code(item.get("code"))
        if code != "unknown":
            codes.append(code)
        if item.get("severity"):
            severities.append(str(_safe_scalar(item.get("severity"))))
    codes = sorted(set(codes))[:6]
    severities = sorted(set(severities))[:3]
    return (
        "APEX RuntimeOS observed repeatable recommendation signal; "
        f"codes={','.join(codes) or 'none'}; "
        f"severity={','.join(severities) or 'unknown'}; "
        "treat as candidate for future review, not as factual task completion."
    )


def _candidate_to_skill_content(candidate: Mapping[str, Any]) -> str:
    codes = []
    actions = []
    for item in candidate.get("items", []) if isinstance(candidate.get("items"), list) else []:
        if not isinstance(item, Mapping):
            continue
        code = _safe_code(item.get("code"))
        if code != "unknown":
            codes.append(code)
        item_actions_raw = item.get("actions")
        item_actions = item_actions_raw if isinstance(item_actions_raw, list) else []
        actions.extend(str(_safe_scalar(action)) for action in item_actions)
    codes = sorted(set(codes))[:8]
    actions = sorted({a for a in actions if a and not a.startswith("[REDACTED")})[:8]
    return "\n".join([
        "# APEX RuntimeOS 自动候选技能",
        "",
        "## 适用场景",
        "",
        "- RuntimeOS 连续观察到可复用的建议信号。",
        "- 本技能由晋升门禁生成，来源为脱敏候选，不包含原始对话、路径或凭据。",
        "",
        "## 信号代码",
        "",
        *(f"- `{code}`" for code in (codes or ["none"])),
        "",
        "## 建议动作",
        "",
        *(f"- {action}" for action in (actions or ["人工复核后再固化具体流程"])),
        "",
        "## 边界",
        "",
        "- 该技能只作为复核入口，不证明任务已经完成。",
        "- 涉及法律依据、外部事实、文件状态时仍需真实查证。",
        "",
    ])


def score_autowrite_candidates(*, candidate_path: Optional[Path] = None, limit: int = 1000, min_occurrences: int = 2) -> Dict[str, Any]:
    """Score candidates by repeatable sanitized recommendation codes."""
    path = candidate_path or (_out_dir() / "candidates.jsonl")
    candidates = _read_jsonl(path, limit=limit)
    buckets: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        if candidate.get("schema") != "ApexRuntimeOSAutoWriteCandidate/v1":
            continue
        codes = []
        for item in candidate.get("items", []) if isinstance(candidate.get("items"), list) else []:
            if isinstance(item, Mapping):
                code = _safe_code(item.get("code"))
                if code != "unknown":
                    codes.append(code)
        if not codes:
            continue
        key = ",".join(sorted(set(codes)))
        bucket = buckets.setdefault(key, {"key": key, "count": 0, "candidate": candidate, "ready": False})
        bucket["count"] += 1
    ready = []
    for bucket in buckets.values():
        bucket["ready"] = int(bucket["count"] or 0) >= max(1, int(min_occurrences))
        if bucket["ready"]:
            ready.append(bucket)
    return {
        "schema": "ApexRuntimeOSStabilityScore/v1",
        "candidate_path_exists": path.exists(),
        "min_occurrences": max(1, int(min_occurrences)),
        "groups": list(buckets.values()),
        "ready": ready,
        "ready_count": len(ready),
    }


def run_autopromotion_scheduler(*, candidate_path: Optional[Path] = None, target: str = "memory", limit: int = 1000, min_occurrences: int = 2) -> Dict[str, Any]:
    """Score candidates and promote only stable groups when explicitly enabled."""
    enabled = _flag("APEX_RUNTIMEOS_AUTOPROMOTE_ENABLED", "0")
    mode = _mode()
    score = score_autowrite_candidates(candidate_path=candidate_path, limit=limit, min_occurrences=min_occurrences)
    if not enabled or mode != "enforce":
        return {"enabled": enabled, "mode": mode, "promoted": 0, "skipped": 0, "score": score, "reason": "disabled_or_not_enforce"}
    if not score["ready"]:
        return {"enabled": True, "mode": mode, "promoted": 0, "skipped": 0, "score": score, "reason": "no_stable_candidate"}
    staging = _out_dir() / "stable_candidates.jsonl"
    staging.parent.mkdir(parents=True, exist_ok=True)
    with staging.open("w", encoding="utf-8") as fh:
        for bucket in score["ready"]:
            fh.write(json.dumps(bucket["candidate"], ensure_ascii=False, separators=(",", ":")) + "\n")
    previous = os.environ.get("APEX_RUNTIMEOS_PROMOTION_ENABLED")
    os.environ["APEX_RUNTIMEOS_PROMOTION_ENABLED"] = "1"
    try:
        promotion = promote_autowrite_candidates(candidate_path=staging, target=target, limit=len(score["ready"]))
    finally:
        if previous is None:
            os.environ.pop("APEX_RUNTIMEOS_PROMOTION_ENABLED", None)
        else:
            os.environ["APEX_RUNTIMEOS_PROMOTION_ENABLED"] = previous
    return {"enabled": True, "mode": mode, "target": target, "score": score, **promotion}


def promote_autowrite_candidates(*, candidate_path: Optional[Path] = None, target: str = "memory", limit: int = 20) -> Dict[str, Any]:
    """Promote sanitized candidates into formal memory or skill artifacts.

    Promotion is disabled unless both APEX_RUNTIMEOS_PROMOTION_ENABLED=1 and
    APEX_RUNTIMEOS_GATE_MODE=enforce are set. All promotions are deduplicated by
    generated content hash and recorded in promotions.jsonl with rollback data.
    """
    enabled = _flag("APEX_RUNTIMEOS_PROMOTION_ENABLED", "0")
    mode = _mode()
    if target not in {"memory", "skill"}:
        return {"enabled": enabled, "mode": mode, "promoted": 0, "skipped": 0, "error": "invalid_target"}
    if not enabled or mode != "enforce":
        return {"enabled": enabled, "mode": mode, "promoted": 0, "skipped": 0, "reason": "disabled_or_not_enforce"}

    path = candidate_path or (_out_dir() / "candidates.jsonl")
    candidates = _read_jsonl(path, limit=limit)
    audit_path = _promotion_audit_path()
    existing_hashes = {str(item.get("content_hash")) for item in _read_jsonl(audit_path, limit=5000) if item.get("content_hash")}
    promoted = 0
    skipped = 0
    details = []

    for candidate in candidates:
        if candidate.get("schema") != "ApexRuntimeOSAutoWriteCandidate/v1":
            skipped += 1
            continue
        if not candidate.get("promotion_required"):
            skipped += 1
            continue
        if target == "memory":
            content = _candidate_to_memory_entry(candidate)
            content_hash = _hash_text("memory:" + content)
            if content_hash in existing_hashes:
                skipped += 1
                continue
            store = MemoryStore(memory_char_limit=10000, user_char_limit=10000)
            store.load_from_disk()
            before = list(store.memory_entries)
            result = store.add("memory", content)
            success = bool(result.get("success"))
            if success:
                promoted += 1
                existing_hashes.add(content_hash)
            else:
                skipped += 1
            audit = {
                "schema": "ApexRuntimeOSPromotion/v1",
                "ts": time.time(),
                "target": "memory",
                "content_hash": content_hash,
                "success": success,
                "rollback": {"action": "remove", "old_text": content[:80]} if success else {},
                "rollback_status": "pending" if success else "skipped",
                "before_count": len(before),
                "after_count": len(store.memory_entries),
            }
            _append_jsonl(audit_path, audit)
            details.append({"target": "memory", "success": success, "content_hash": content_hash})
        else:
            content = _candidate_to_skill_content(candidate)
            content_hash = _hash_text("skill:" + content)
            if content_hash in existing_hashes:
                skipped += 1
                continue
            skill_dir = get_hermes_home() / "skills" / "apex-runtimeos-autogen"
            skill_path = skill_dir / "SKILL.md"
            old_content = skill_path.read_text(encoding="utf-8") if skill_path.exists() else None
            skill_dir.mkdir(parents=True, exist_ok=True)
            skill_path.write_text(content, encoding="utf-8")
            promoted += 1
            existing_hashes.add(content_hash)
            current_content_hash = _hash_text(content)
            audit = {
                "schema": "ApexRuntimeOSPromotion/v1",
                "ts": time.time(),
                "target": "skill",
                "content_hash": content_hash,
                "success": True,
                "rollback": {
                    "action": "restore" if old_content is not None else "delete",
                    "path_hash": _hash_text(str(skill_path)),
                    "path": str(skill_path),
                    "current_content_hash": current_content_hash,
                    "old_content": old_content,
                    "old_content_hash": _hash_text(old_content) if old_content is not None else "",
                },
                "skill_name": "apex-runtimeos-autogen",
                "rollback_status": "pending",
            }
            _append_jsonl(audit_path, _sanitize_promotion_audit(audit))
            details.append({"target": "skill", "success": True, "content_hash": content_hash, "skill_name": "apex-runtimeos-autogen"})
    return {
        "enabled": True,
        "mode": mode,
        "target": target,
        "promoted": promoted,
        "skipped": skipped,
        "audit_path": str(audit_path),
        "details": details,
    }


def _audit_rollback(record: Mapping[str, Any]) -> Dict[str, Any]:
    raw = record.get("rollback")
    rollback: Dict[str, Any] = dict(raw) if isinstance(raw, Mapping) else {}
    safe: Dict[str, Any] = {str(k): v for k, v in rollback.items() if k not in {"path", "old_content"}}
    if rollback.get("path"):
        safe["path_basename"] = Path(str(rollback.get("path"))).name
    return safe


def _sanitize_promotion_audit(record: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    if "rollback" in out:
        out["rollback"] = _audit_rollback(record)
    return out


def _append_rollback_event(path: Path, *, promotion_id: str, target: str, status: str, reason: str = "") -> None:
    _append_jsonl(path, {
        "schema": "ApexRuntimeOSRollbackEvent/v1",
        "ts": time.time(),
        "promotion_id": promotion_id,
        "target": target,
        "rollback_status": status,
        "reason": reason,
    })


def execute_promotion_rollbacks(*, audit_path: Optional[Path] = None, target: str = "all", dry_run: bool = True, limit: int = 1000) -> Dict[str, Any]:
    """Execute pending promotion rollbacks with explicit opt-in and idempotence."""
    enabled = _flag("APEX_RUNTIMEOS_ROLLBACK_ENABLED", "0")
    mode = _mode()
    if target not in {"all", "memory", "skill"}:
        return {"enabled": enabled, "mode": mode, "rolled_back": 0, "skipped": 0, "error": "invalid_target"}
    if not enabled or mode != "enforce":
        dry_run = True
    path = audit_path or _promotion_audit_path()
    records = _read_jsonl(path, limit=limit)
    done_ids = _rollback_done_ids(records)
    rolled_back = 0
    skipped = 0
    events = []
    for record in records:
        if record.get("schema") != "ApexRuntimeOSPromotion/v1" or not record.get("success"):
            skipped += 1
            continue
        promotion_id = _promotion_id(record)
        if promotion_id in done_ids:
            skipped += 1
            events.append({"target": record.get("target"), "status": "skipped", "reason": "already_done", "promotion_id": promotion_id})
            continue
        rec_target = str(record.get("target") or "")
        if target != "all" and rec_target != target:
            skipped += 1
            continue
        if record.get("rollback_status") == "done":
            skipped += 1
            events.append({"target": rec_target, "status": "skipped", "reason": "already_done", "promotion_id": promotion_id})
            continue
        raw_rb = record.get("rollback")
        rollback: Dict[str, Any] = dict(raw_rb) if isinstance(raw_rb, Mapping) else {}
        action = str(rollback.get("action") or "")
        if rec_target == "memory" and action == "remove":
            old_text = str(rollback.get("old_text") or "")
            if not old_text:
                skipped += 1
                if not dry_run:
                    _append_rollback_event(path, promotion_id=promotion_id, target="memory", status="skipped", reason="missing_old_text")
                events.append({"target": "memory", "status": "skipped", "reason": "missing_old_text", "promotion_id": promotion_id})
                continue
            if dry_run:
                rolled_back += 1
                events.append({"target": "memory", "status": "dry_run", "action": "remove", "promotion_id": promotion_id})
                continue
            store = MemoryStore(memory_char_limit=10000, user_char_limit=10000)
            store.load_from_disk()
            result = store.remove("memory", old_text)
            if result.get("success"):
                rolled_back += 1
                _append_rollback_event(path, promotion_id=promotion_id, target="memory", status="done")
                events.append({"target": "memory", "status": "done", "action": "remove", "promotion_id": promotion_id})
            else:
                skipped += 1
                _append_rollback_event(path, promotion_id=promotion_id, target="memory", status="skipped", reason="remove_failed")
                events.append({"target": "memory", "status": "skipped", "reason": "remove_failed", "promotion_id": promotion_id})
        elif rec_target == "skill" and action in {"restore", "delete"}:
            raw_path = rollback.get("path")
            if not raw_path:
                skipped += 1
                if not dry_run:
                    _append_rollback_event(path, promotion_id=promotion_id, target="skill", status="skipped", reason="missing_path")
                events.append({"target": "skill", "status": "skipped", "reason": "missing_path", "promotion_id": promotion_id})
                continue
            skill_path = Path(str(raw_path)).expanduser()
            expected_hash = str(rollback.get("current_content_hash") or "")
            current = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""
            if expected_hash and _hash_text(current) != expected_hash:
                skipped += 1
                if not dry_run:
                    _append_rollback_event(path, promotion_id=promotion_id, target="skill", status="skipped", reason="content_hash_mismatch")
                events.append({"target": "skill", "status": "skipped", "reason": "content_hash_mismatch", "path_basename": skill_path.name, "promotion_id": promotion_id})
                continue
            if dry_run:
                rolled_back += 1
                events.append({"target": "skill", "status": "dry_run", "action": action, "path_basename": skill_path.name, "promotion_id": promotion_id})
                continue
            if action == "delete":
                if skill_path.exists():
                    skill_path.unlink()
            else:
                old_content = rollback.get("old_content")
                if old_content is None:
                    skipped += 1
                    _append_rollback_event(path, promotion_id=promotion_id, target="skill", status="skipped", reason="missing_old_content")
                    events.append({"target": "skill", "status": "skipped", "reason": "missing_old_content", "path_basename": skill_path.name, "promotion_id": promotion_id})
                    continue
                skill_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = skill_path.with_suffix(skill_path.suffix + ".tmp")
                tmp.write_text(str(old_content), encoding="utf-8")
                tmp.replace(skill_path)
            rolled_back += 1
            _append_rollback_event(path, promotion_id=promotion_id, target="skill", status="done")
            events.append({"target": "skill", "status": "done", "action": action, "path_basename": skill_path.name, "promotion_id": promotion_id})
        else:
            skipped += 1
            if not dry_run:
                _append_rollback_event(path, promotion_id=promotion_id, target=rec_target, status="skipped", reason="unsupported_rollback")
            events.append({"target": rec_target, "status": "skipped", "reason": "unsupported_rollback", "promotion_id": promotion_id})
    return {"enabled": enabled, "mode": mode, "dry_run": dry_run, "target": target, "rolled_back": rolled_back, "skipped": skipped, "events": events[:20]}


def record_cron_dryrun_result(*, task: str, result: Mapping[str, Any], ledger_path: Optional[Path] = None) -> Dict[str, Any]:
    """Append an aggregate-safe cron dry-run ledger snapshot.

    Storage stays append-only. Idempotence is represented by a stable ledger_key;
    readers keep the latest snapshot per key and use count/last_seen_at.
    """
    path = ledger_path or _cron_ledger_path()
    safe_result = _redact_for_ledger(result)
    now = time.time()
    task_code = _safe_code(task)
    plan_hash = _stable_hash({"task": task_code, "result": safe_result})
    ledger_key = _hash_text(f"{task_code}:{plan_hash}")
    existing = {}
    for item in _read_jsonl(path, limit=5000):
        if item.get("schema") == "ApexRuntimeOSCronDryRunLedger/v1" and item.get("ledger_key") == ledger_key:
            existing = item
    record = {
        "schema": "ApexRuntimeOSCronDryRunLedger/v1",
        "redaction_version": 1,
        "task": task_code,
        "ledger_key": ledger_key,
        "plan_hash": plan_hash,
        "first_seen_at": float(existing.get("first_seen_at") or now),
        "last_seen_at": now,
        "count": int(existing.get("count") or 0) + 1,
        "result": safe_result,
    }
    _append_jsonl(path, record)
    return {"written": True, "ledger_key": ledger_key, "plan_hash": plan_hash, "count": record["count"]}


def summarize_cron_dryrun_ledger(*, ledger_path: Optional[Path] = None, limit: int = 1000) -> Dict[str, Any]:
    path = ledger_path or _cron_ledger_path()
    latest: Dict[str, Dict[str, Any]] = {}
    bad_lines = 0
    total_lines = 0
    quarantine_path = path.with_suffix(path.suffix + ".bad")
    if path.exists():
        good_lines = []
        bad_raw_lines = []
        with path.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if idx >= limit:
                    break
                total_lines += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    bad_raw_lines.append(line.rstrip("\n")[:500])
                    continue
                if not isinstance(item, dict) or item.get("schema") != "ApexRuntimeOSCronDryRunLedger/v1":
                    continue
                good_lines.append(line.rstrip("\n"))
                key = str(item.get("ledger_key") or "")
                if key:
                    latest[key] = item
        if bad_raw_lines and _flag("APEX_RUNTIMEOS_CRON_LEDGER_REPAIR_ENABLED", "0") and _mode() == "enforce":
            quarantine_path.parent.mkdir(parents=True, exist_ok=True)
            with quarantine_path.open("a", encoding="utf-8") as qh:
                for raw in bad_raw_lines:
                    qh.write(json.dumps({"schema": "ApexRuntimeOSCronDryRunBadLine/v1", "ts": time.time(), "line_hash": _hash_text(raw)}, ensure_ascii=False, separators=(",", ":")) + "\n")
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text("\n".join(good_lines) + ("\n" if good_lines else ""), encoding="utf-8")
            tmp.replace(path)
    tasks: Dict[str, Dict[str, Any]] = {}
    last_seen_at = 0.0
    for item in latest.values():
        task_name = str(item.get("task") or "unknown")
        entry = tasks.setdefault(task_name, {"keys": 0, "total_count": 0, "last_seen_at": 0.0})
        entry["keys"] += 1
        entry["total_count"] += int(item.get("count") or 0)
        entry["last_seen_at"] = max(float(entry.get("last_seen_at") or 0.0), float(item.get("last_seen_at") or 0.0))
        last_seen_at = max(last_seen_at, float(item.get("last_seen_at") or 0.0))
    recent = sorted(latest.values(), key=lambda item: float(item.get("last_seen_at") or 0.0), reverse=True)[:10]
    return {
        "schema": "ApexRuntimeOSCronDryRunLedgerSummary/v1",
        "ledger_exists": path.exists(),
        "bad_lines": bad_lines,
        "total_lines": total_lines,
        "repair_enabled": _flag("APEX_RUNTIMEOS_CRON_LEDGER_REPAIR_ENABLED", "0") and _mode() == "enforce",
        "quarantine_exists": quarantine_path.exists(),
        "unique_keys": len(latest),
        "tasks": tasks,
        "last_seen_at": last_seen_at,
        "recent": [
            {
                "task": item.get("task"),
                "ledger_key": item.get("ledger_key"),
                "plan_hash": item.get("plan_hash"),
                "count": item.get("count"),
                "last_seen_at": item.get("last_seen_at"),
                "result": item.get("result") if isinstance(item.get("result"), dict) else {},
            }
            for item in recent
        ],
    }


def build_runtimeos_health_report(autonomy: Mapping[str, Any]) -> Dict[str, Any]:
    """Build aggregate-only health status and threshold alerts."""
    cron_raw = autonomy.get("cron_dryrun") if isinstance(autonomy.get("cron_dryrun"), Mapping) else {}
    cron: Mapping[str, Any] = cron_raw if isinstance(cron_raw, Mapping) else {}
    alerts = []
    bad_lines = int(cron.get("bad_lines") or 0)
    pending_rollbacks = int(autonomy.get("pending_rollbacks") or 0)
    stable_ready = int(autonomy.get("stable_ready_count") or 0)
    if bad_lines > 0:
        alerts.append({"code": "cron_ledger_bad_lines", "severity": "warn", "count": bad_lines, "message": "cron dry-run 账本存在坏行，建议启用 repair 审计隔离"})
    if pending_rollbacks > 0:
        alerts.append({"code": "pending_rollbacks", "severity": "warn", "count": pending_rollbacks, "message": "存在待回滚晋升记录，需要人工复核后再执行"})
    if stable_ready > 0 and not autonomy.get("autopromote_enabled"):
        alerts.append({"code": "stable_candidates_waiting", "severity": "info", "count": stable_ready, "message": "存在稳定候选，但自动晋升未开启"})
    if any(item.get("severity") == "warn" for item in alerts):
        status = "WATCH"
    elif alerts:
        status = "INFO"
    else:
        status = "OK"
    return {
        "schema": "ApexRuntimeOSHealthReport/v1",
        "status": status,
        "alerts": alerts,
        "alert_count": len(alerts),
        "metrics": {
            "cron_bad_lines": bad_lines,
            "pending_rollbacks": pending_rollbacks,
            "stable_ready_count": stable_ready,
            "cron_unique_keys": int(cron.get("unique_keys") or 0),
        },
        "side_effects": "read_only_report",
    }


def _watchdog_state_path() -> Path:
    configured = os.environ.get("APEX_RUNTIMEOS_WATCHDOG_STATE_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return _out_dir() / "watchdog_state.json"


def _watchdog_config() -> Dict[str, Any]:
    state_path = os.environ.get("APEX_RUNTIMEOS_WATCHDOG_STATE_PATH", "").strip()
    cooldown_seconds = int(os.environ.get("APEX_RUNTIMEOS_WATCHDOG_COOLDOWN_SECONDS", "2700") or 2700)
    return {
        "schema": "ApexRuntimeOSWatchdogConfig/v1",
        "state_configured": bool(state_path),
        "cooldown_seconds": max(1, cooldown_seconds),
    }


def build_runtimeos_health_watchdog_notice(
    payload: Mapping[str, Any],
    *,
    state_path: Optional[Path] = None,
    cooldown_seconds: int = 2700,
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Return a suppression-aware reminder decision for the watchdog script.

    The helper is intentionally stateful: it stores only a small signature and
    timestamp so repeated WATCH alerts can be throttled without persisting raw
    alert text, paths, or credentials.
    """
    payload_status = payload.get("status") if isinstance(payload.get("status"), Mapping) else {}
    status = payload_status if isinstance(payload_status, Mapping) else {}
    health = status.get("health_report") if isinstance(status.get("health_report"), Mapping) else {}
    health_status = str(health.get("status") or "OK")
    alerts = [item for item in (health.get("alerts") or []) if isinstance(item, Mapping)]
    alert_count = int(health.get("alert_count") or len(alerts))
    signature = _stable_hash({
        "health_status": health_status,
        "alert_count": alert_count,
        "alerts": [
            {
                "code": str(item.get("code") or ""),
                "severity": str(item.get("severity") or ""),
                "count": int(item.get("count") or 0),
            }
            for item in alerts[:20]
        ],
    })
    resolved_state_path = Path(state_path) if state_path is not None else _watchdog_state_path()
    now_value = float(now_ts if now_ts is not None else time.time())
    previous: Dict[str, Any] = {}
    if resolved_state_path.exists():
        try:
            loaded = json.loads(resolved_state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                previous = dict(loaded)
        except Exception:
            previous = {}

    if health_status == "OK":
        try:
            resolved_state_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass
        return {
            "should_emit": False,
            "reason": "ok",
            "health_status": health_status,
            "alert_count": alert_count,
            "alerts": alerts,
        }

    previous_signature = str(previous.get("signature") or "")
    previous_ts = float(previous.get("last_alert_at") or 0.0)
    if previous_signature == signature and previous_ts and (now_value - previous_ts) < float(cooldown_seconds):
        return {
            "should_emit": False,
            "reason": "cooldown",
            "health_status": health_status,
            "alert_count": alert_count,
            "alerts": alerts,
            "remaining_seconds": max(0, int(float(cooldown_seconds) - (now_value - previous_ts))),
        }

    resolved_state_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_state_path.write_text(
        json.dumps(
            {
                "schema": "ApexRuntimeOSHealthWatchdogState/v1",
                "signature": signature,
                "last_alert_at": now_value,
                "health_status": health_status,
                "alert_count": alert_count,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return {
        "should_emit": True,
        "reason": "alert",
        "health_status": health_status,
        "alert_count": alert_count,
        "alerts": alerts,
    }


def summarize_autonomy_status(*, limit: int = 1000, min_occurrences: int = 2) -> Dict[str, Any]:
    """Return aggregate-only RuntimeOS autonomy status for dashboards/API.

    The summary intentionally omits local paths, raw candidate items, rollback
    old_text/old_content, and any source prompts/messages. It is safe to expose
    to API clients and HTML dashboards.
    """
    candidate_path = _candidate_path()
    audit_path = _promotion_audit_path()
    score = score_autowrite_candidates(candidate_path=candidate_path, limit=limit, min_occurrences=min_occurrences)
    promotion_records = _read_jsonl(audit_path, limit=limit)
    promotions = [item for item in promotion_records if item.get("schema") == "ApexRuntimeOSPromotion/v1"]
    rollback_events = [item for item in promotion_records if item.get("schema") == "ApexRuntimeOSRollbackEvent/v1"]
    pending_rollbacks = 0
    done_ids = _rollback_done_ids(promotion_records)
    target_counts: Dict[str, int] = {}
    rollback_status: Dict[str, int] = {}
    for item in promotions:
        target_name = str(item.get("target") or "unknown")
        target_counts[target_name] = target_counts.get(target_name, 0) + 1
        pid = _promotion_id(item)
        status = str(item.get("rollback_status") or "unknown")
        if pid in done_ids:
            status = "done"
        rollback_status[status] = rollback_status.get(status, 0) + 1
        if item.get("success") and status == "pending":
            pending_rollbacks += 1
    event_status: Dict[str, int] = {}
    for item in rollback_events:
        status = str(item.get("rollback_status") or "unknown")
        event_status[status] = event_status.get(status, 0) + 1
    report = {
        "schema": "ApexRuntimeOSAutonomyStatus/v1",
        "mode": _mode(),
        "autopromote_enabled": _flag("APEX_RUNTIMEOS_AUTOPROMOTE_ENABLED", "0"),
        "rollback_enabled": _flag("APEX_RUNTIMEOS_ROLLBACK_ENABLED", "0"),
        "candidate_path_exists": candidate_path.exists(),
        "promotion_audit_exists": audit_path.exists(),
        "stable_ready_count": int(score.get("ready_count") or 0),
        "candidate_groups": len(score.get("groups") or []),
        "promotion_count": len(promotions),
        "promotion_targets": target_counts,
        "pending_rollbacks": pending_rollbacks,
        "rollback_status": rollback_status,
        "rollback_events": {"count": len(rollback_events), "status": event_status},
        "cron_dryrun": summarize_cron_dryrun_ledger(limit=limit),
        "default_side_effects": "disabled_unless_explicit_enforce",
    }
    report["health_report"] = build_runtimeos_health_report(report)
    return report


__all__ = [
    "build_runtime_rewrite_plan",
    "apply_runtime_rewrite",
    "persist_autowrite_candidate",
    "score_autowrite_candidates",
    "run_autopromotion_scheduler",
    "execute_promotion_rollbacks",
    "promote_autowrite_candidates",
    "summarize_autonomy_status",
    "record_cron_dryrun_result",
    "summarize_cron_dryrun_ledger",
    "build_runtimeos_health_report",
    "build_runtimeos_health_watchdog_notice",
    "_watchdog_config",
]
