"""plan-secretary core: precise capture + registry + session-scoped notifier.

Pure-Python, stdlib only. All state lives under ``get_hermes_home()/state/
plan_secretary/`` so it is profile-safe and portable. No local paths, no
session ids are hardcoded — every session id is an argument.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

VERSION = "plan-secretary-0.1.0"
TERMINAL_STATUSES = {"completed", "cancelled"}
ACTIVE_STATUSES = {"active", "deferred", "blocked"}

CAPTURE_INTENT_KEYWORDS = [
    "我计划", "下一步", "我会", "我准备", "下次", "明天", "稍后",
    "等会", "后续", "接下来", "我打算", "我需要",
    "i will", "i'll", "i\u2019ll", "next i will", "i plan to", "i need to",
]

CAPTURE_ACTOR_PATTERNS = [
    r"\bI\s+(?:will|can|am going to|need to|plan to)\b",
    r"\bI['\u2019]ll\b",
    r"\b(?:assistant|agent)\s+(?:will|can|is going to)\b",
    r"(?:小墨|我|助手)(?:接下来|稍后|等会|后续|会|将|准备|计划|需要|可以)?",
]

CAPTURE_ACTION_KEYWORDS = [
    "检查", "启动", "修复", "生成", "回收", "验证", "跑", "写入", "打开",
    "接入", "读取", "扫描", "清理", "更新", "补", "改", "实现", "定位",
    "测试", "验收",
    "register", "schedule", "start", "run", "check", "verify", "fix",
    "write", "update", "scan",
]

CAPTURE_OBJECT_PATTERNS = [
    r"\b[\w./\\-]+\.(?:py|json|jsonl|log|md|txt|db|yaml|yml)\b",
    r"\b(?:[\w./\\-]+/)?[\w-]+\.(?:sh|bat|ps1)\b",
    r"\b(?:script|file|process|watcher|log|pending capture|plan|registry|cursor)\b",
    r"(?:脚本|文件|进程|日志|计划|状态文件|过滤器|规则|监听|链路|数据库|会话|消息|小秘书|捕捉|误抓|短测|真实计划)",
]

NON_COMMITMENT_PATTERNS = [
    r"^\s*[/`].*\b(?:可以|后续|下一步)",
    r"\b(?:可以|可用于|用于|建议|一般可以|应该|最好|需要用户|让新会话|会更|更好恢复|更清晰|设计|说明|文档)\b",
    r"(?:这是|这类|这个设计|目标是|用于|不是|管|显示|显示：|支持参数|测试命令|日志内容|短测暴露|当前结论|推荐测试句|反例|正例[，：:].*)",
    r"(?:下一步一般可以|可以考虑|应该进入|后续好恢复|后续会更清晰)",
]

CONTEXT_NOISE_PATTERNS = [
    r"\b(?:alpha|seed|seed bank|种子池|二阶增强|工厂|Factory Autopilot|runner|429)\b",
]

NOISE_PATTERNS = [
    r"capture-\d{8}-\d{6}",
    r"关键词[:：]",
    r"原文[:：]",
    r"python\s+plan_secretary\.py",
    r"--source\s+(?:auto|file|stdin|hermes-db|hermes-session-dumps)",
    r"pending_captures",
    r"NO_CAPTURES",
    r"CAPTURED\s+\d+",
    r"DRY_RUN\s+scan-text",
]

MIN_CAPTURE_CHARS = 8
MAX_CAPTURE_CHARS = 240


def state_dir() -> Path:
    return get_hermes_home() / "state" / "plan_secretary"


def registry_path() -> Path:
    return state_dir() / "plan_registry.json"


def captures_path() -> Path:
    return state_dir() / "pending_captures.json"


def now_local() -> datetime:
    return datetime.now().astimezone()


def iso(dt: datetime) -> str:
    return dt.astimezone().replace(microsecond=0).isoformat()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_dt(value: str) -> datetime:
    text = value.strip()
    base = now_local()
    rel = re.fullmatch(r"(\d+)\s*([mhd])", text, re.IGNORECASE)
    if rel:
        n = int(rel.group(1))
        unit = rel.group(2).lower()
        if unit == "m":
            return base + timedelta(minutes=n)
        if unit == "h":
            return base + timedelta(hours=n)
        return base + timedelta(days=n)
    today = re.fullmatch(r"today\s+(\d{1,2}):(\d{2})", text, re.IGNORECASE)
    if today:
        hour, minute = int(today.group(1)), int(today.group(2))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"invalid today time: {value}")
        return base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=base.tzinfo)
    return parsed.astimezone()


def parse_plan_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone()
    except Exception:
        return None


def is_prereq_satisfied(plan: dict[str, Any]) -> bool:
    prereq = str(plan.get("prereq") or "").strip()
    if not prereq:
        return True
    lowered = prereq.lower()
    return lowered in {"none", "n/a", "na", "无"} or lowered.startswith(("ok:", "done:", "satisfied:"))


# ---------- precise capture ----------

def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])\s*|\r?\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def matched_keywords(sentence: str) -> list[str]:
    normalized = sentence.replace("\u2019", "'").replace("\u2018", "'").lower()
    hits = [kw for kw in CAPTURE_INTENT_KEYWORDS if kw in normalized]
    return [kw for kw in CAPTURE_INTENT_KEYWORDS if kw in sentence or kw in normalized] or hits


def has_capture_actor(sentence: str) -> bool:
    return any(re.search(p, sentence, re.IGNORECASE) for p in CAPTURE_ACTOR_PATTERNS)


def has_capture_action(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(kw.lower() in lowered for kw in CAPTURE_ACTION_KEYWORDS)


def has_capture_object(sentence: str) -> bool:
    return any(re.search(p, sentence, re.IGNORECASE) for p in CAPTURE_OBJECT_PATTERNS)


def is_non_commitment_context(sentence: str) -> bool:
    return any(re.search(p, sentence, re.IGNORECASE) for p in NON_COMMITMENT_PATTERNS)


def is_context_noise(sentence: str) -> bool:
    return any(re.search(p, sentence, re.IGNORECASE) for p in CONTEXT_NOISE_PATTERNS)


def is_capture_noise(sentence: str) -> bool:
    s = sentence.strip()
    if len(s) < MIN_CAPTURE_CHARS or len(s) > MAX_CAPTURE_CHARS:
        return True
    if s.startswith(("- `capture-", "capture-", "关键词", "原文", "|", "```", "~~~")):
        return True
    if sum(1 for kw in CAPTURE_INTENT_KEYWORDS if kw in s) >= 5:
        return True
    return any(re.search(p, s, re.IGNORECASE) for p in NOISE_PATTERNS)


def is_precise_capture(sentence: str) -> bool:
    s = sentence.strip()
    if is_capture_noise(s):
        return False
    if is_non_commitment_context(s):
        return False
    if is_context_noise(s) and not re.search(r"(?:小墨|我|助手).*(?:检查|验证|写入|修复|启动|跑)", s):
        return False
    return has_capture_actor(s) and has_capture_action(s) and has_capture_object(s)


def suggested_title(sentence: str) -> str:
    clean = re.sub(r"\s+", " ", sentence).strip(" ：:，,。！？!?；;")
    return clean[:40] or "待确认计划"


def capture_id(text: str) -> str:
    stamp = now_local().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "-", text).strip("-").lower()[:24]
    return f"capture-{stamp}-{slug or 'item'}"


def unique_capture_id(text: str, existing_ids: set[str]) -> str:
    base = capture_id(text)
    if base not in existing_ids:
        existing_ids.add(base)
        return base
    idx = 2
    while f"{base}-{idx}" in existing_ids:
        idx += 1
    uid = f"{base}-{idx}"
    existing_ids.add(uid)
    return uid


def plan_id(title: str) -> str:
    stamp = now_local().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower()[:24]
    return f"plan-{stamp}-{slug or 'item'}"


# ---------- state ----------

def load_registry() -> dict[str, Any]:
    data = read_json(registry_path(), {})
    plans = data.get("plans") if isinstance(data, dict) else None
    return {
        "version": VERSION,
        "created_at": (data or {}).get("created_at") or iso(now_local()),
        "updated_at": (data or {}).get("updated_at") or iso(now_local()),
        "plans": plans if isinstance(plans, list) else [],
    }


def load_captures() -> dict[str, Any]:
    data = read_json(captures_path(), {})
    captures = data.get("captures") if isinstance(data, dict) else None
    return {
        "version": VERSION,
        "created_at": (data or {}).get("created_at") or iso(now_local()),
        "updated_at": (data or {}).get("updated_at") or iso(now_local()),
        "captures": captures if isinstance(captures, list) else [],
    }


def save_captures(data: dict[str, Any]) -> None:
    data["updated_at"] = iso(now_local())
    write_json(captures_path(), data)


def save_registry(data: dict[str, Any]) -> None:
    data["updated_at"] = iso(now_local())
    write_json(registry_path(), data)


# ---------- scan / confirm / ignore ----------

def scan_text(text: str, source: str = "", source_id: str = "",
              source_role: str = "", source_session_id: str = "",
              source_message_id: str = "") -> list[dict[str, Any]]:
    captures_data = load_captures()
    existing = {(str(c.get("text")), str(c.get("source") or ""), str(c.get("source_id") or ""))
                for c in captures_data["captures"]}
    existing_ids = {str(c.get("id")) for c in captures_data["captures"] if c.get("id")}
    new_captures: list[dict[str, Any]] = []
    for sentence in split_sentences(text):
        if not matched_keywords(sentence) or not is_precise_capture(sentence):
            continue
        key = (sentence, source, source_id)
        legacy = (sentence, "", "")
        if key in existing or (not source_id and legacy in existing):
            continue
        capture = {
            "id": unique_capture_id(sentence, existing_ids),
            "text": sentence,
            "matched_keywords": matched_keywords(sentence),
            "suggested_title": suggested_title(sentence),
            "source": source,
            "source_id": source_id,
            "source_role": source_role,
            "source_session_id": source_session_id,
            "source_message_id": source_message_id,
            "created_at": iso(now_local()),
            "updated_at": iso(now_local()),
            "status": "pending",
        }
        new_captures.append(capture)
        existing.add(key)
    captures_data["captures"].extend(new_captures)
    save_captures(captures_data)
    return new_captures


def find_capture(captures_data: dict[str, Any], cid: str) -> dict[str, Any]:
    matches = [c for c in captures_data["captures"]
               if str(c.get("id")) == cid or str(c.get("id", "")).startswith(cid)]
    if not matches:
        raise SystemExit(f"CAPTURE_NOT_FOUND {cid}")
    if len(matches) > 1:
        raise SystemExit(f"CAPTURE_ID_AMBIGUOUS {cid}")
    return matches[0]


def confirm_capture(cid: str, due: str, mode: str = "parallel",
                    title: str = "", owner: str = "", worker: str = "",
                    priority: str = "normal", prereq: str = "",
                    next_action: str = "") -> dict[str, Any]:
    captures_data = load_captures()
    capture = find_capture(captures_data, cid)
    if capture.get("status") != "pending":
        raise SystemExit(f"CAPTURE_NOT_PENDING {capture.get('id')} status={capture.get('status')}")
    due_dt = parse_dt(due)
    registry = load_registry()
    plan = {
        "id": plan_id(title or capture.get("suggested_title") or "待确认计划"),
        "title": title or capture.get("suggested_title") or "待确认计划",
        "status": "active",
        "due": iso(due_dt),
        "owner": owner,
        "worker": worker,
        "priority": priority,
        "prereq": prereq or "",
        "next_action": next_action or capture.get("text") or "",
        "mode": mode,
        "conflict_with": [],
        "source_capture_id": capture.get("id"),
        "source_session_id": capture.get("source_session_id") or "",
        "source_message_id": capture.get("source_message_id") or "",
        "source": capture.get("source") or "",
        "source_id": capture.get("source_id") or "",
        "created_at": iso(now_local()),
        "updated_at": iso(now_local()),
        "history": [{"at": iso(now_local()), "action": "add_from_capture", "detail": f"capture={capture.get('id')}; mode={mode}"}],
    }
    registry["plans"].append(plan)
    capture["status"] = "confirmed"
    capture["updated_at"] = iso(now_local())
    capture["confirmed_plan_id"] = plan["id"]
    capture["confirmed_mode"] = mode
    save_registry(registry)
    save_captures(captures_data)
    return plan


def ignore_capture(cid: str, reason: str = "") -> dict[str, Any]:
    captures_data = load_captures()
    capture = find_capture(captures_data, cid)
    capture["status"] = "ignored"
    capture["updated_at"] = iso(now_local())
    capture["ignore_reason"] = reason or ""
    save_captures(captures_data)
    return capture


def set_plan_status(pid: str, status: str, reason: str = "") -> dict[str, Any]:
    registry = load_registry()
    matches = [p for p in registry["plans"] if str(p.get("id")) == pid or str(p.get("id", "")).startswith(pid)]
    if not matches:
        raise SystemExit(f"PLAN_NOT_FOUND {pid}")
    if len(matches) > 1:
        raise SystemExit(f"PLAN_ID_AMBIGUOUS {pid}")
    plan = matches[0]
    plan["status"] = status
    if status == "blocked":
        plan["block_reason"] = reason or ""
    if status == "cancelled":
        plan["cancel_reason"] = reason or ""
    plan["history"] = plan.get("history", []) + [{"at": iso(now_local()), "action": status, "detail": reason or ""}]
    plan["updated_at"] = iso(now_local())
    save_registry(registry)
    return plan


def defer_plan(pid: str, by: str, reason: str = "") -> dict[str, Any]:
    registry = load_registry()
    matches = [p for p in registry["plans"] if str(p.get("id")) == pid or str(p.get("id", "")).startswith(pid)]
    if not matches:
        raise SystemExit(f"PLAN_NOT_FOUND {pid}")
    plan = matches[0]
    current_due = parse_plan_time(plan.get("due"))
    base = max(current_due or now_local(), now_local())
    new_due = base + (parse_dt(by) - now_local())
    plan["status"] = "deferred"
    plan["due"] = iso(new_due)
    plan["defer_reason"] = reason or ""
    plan["updated_at"] = iso(now_local())
    save_registry(registry)
    return plan


# ---------- session-scoped notifications ----------

def capture_session_id(capture: dict[str, Any]) -> str:
    sid = str(capture.get("source_session_id") or "").strip()
    if sid:
        return sid
    source_id = str(capture.get("source_id") or "")
    parts = source_id.split(":")
    return parts[1] if len(parts) >= 3 and parts[0].endswith(".db") else ""


def plan_session_id(plan: dict[str, Any]) -> str:
    sid = str(plan.get("source_session_id") or "").strip()
    if sid:
        return sid
    source_id = str(plan.get("source_id") or "")
    parts = source_id.split(":")
    return parts[1] if len(parts) >= 3 and parts[0].endswith(".db") else ""


def in_scope(item_session_id: str, wanted_session_id: str) -> bool:
    return not wanted_session_id or not item_session_id or item_session_id == wanted_session_id


def active_plans_for_session(registry: dict[str, Any], session_id: str) -> list[dict[str, Any]]:
    scoped = []
    for plan in registry["plans"]:
        if plan.get("status") not in ACTIVE_STATUSES:
            continue
        if in_scope(plan_session_id(plan), session_id):
            scoped.append(plan)
    return sorted(scoped, key=lambda p: p.get("due") or "")


def compact_text(text: str, limit: int = 180) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def build_capture_prompt(capture: dict[str, Any], session_active: list[dict[str, Any]],
                         default_due: str) -> str:
    cid = str(capture.get("id") or "")
    title = compact_text(str(capture.get("suggested_title") or capture.get("text") or "待确认计划"), 70)
    text = compact_text(str(capture.get("text") or ""))
    lines = [
        "📝 小秘书捕捉到一条小墨承诺，需要先确认是否登记计划任务：",
        f"- capture: {cid}",
        f"- session: {capture_session_id(capture) or '-'}",
        f"- title: {title}",
        f"- text: {text}",
        "- 请选择：登记 / 忽略",
    ]
    if session_active:
        lines.extend([
            "- 当前这个 session 还有未完成计划；若登记新任务，请确认：替换前任务 / 新增并行 / 顺延到前任务后。",
            "- 未完成计划：",
        ])
        for plan in session_active[:5]:
            lines.append(f"  - {plan.get('id')} | due={plan.get('due')} | {compact_text(plan.get('title') or '', 80)}")
    lines.extend([
        "- 登记示例（先定时间点）：",
        f"  python -m plugins.plan_secretary confirm-capture {shell_quote(cid)} --due {shell_quote(default_due)} --mode parallel --owner 小墨 --worker agent --priority normal",
        "- 如果要顺延，可把 --due 改成前任务后面的具体时间；如果要替换，先 cancel 旧 plan 再 confirm。",
        "- 忽略示例：",
        f"  python -m plugins.plan_secretary ignore-capture {shell_quote(cid)} --reason 'not an actionable plan'",
    ])
    return "\n".join(lines)


def build_due_prompt(plan: dict[str, Any], unfinished_others: list[dict[str, Any]]) -> str:
    pid = str(plan.get("id") or "")
    lines = [
        "⏰ 小秘书提醒：约定时间点到了，请确认这条计划怎么处理：",
        f"- plan: {pid}",
        f"- session: {plan_session_id(plan) or '-'}",
        f"- due: {plan.get('due')}",
        f"- title: {compact_text(plan.get('title') or '待办计划', 80)}",
        f"- next_action: {compact_text(plan.get('next_action') or '') or '-'}",
    ]
    prereq = str(plan.get("prereq") or "").strip()
    if prereq and not is_prereq_satisfied(plan):
        lines.append(f"- prereq 未满足：{compact_text(prereq)}")
    if unfinished_others:
        lines.extend([
            "- 同 session 还有其他未完成计划；请确认：并行处理 / 顺延本任务 / 顺延其他任务。",
            "- 其他未完成计划：",
        ])
        for other in unfinished_others[:5]:
            lines.append(f"  - {other.get('id')} | due={other.get('due')} | {compact_text(other.get('title') or '', 80)}")
    lines.extend([
        "- 完成后关闭：",
        f"  python -m plugins.plan_secretary complete {shell_quote(pid)}",
        "- 需要顺延：",
        f"  python -m plugins.plan_secretary defer {shell_quote(pid)} --by 10m --reason 'waiting for current task decision'",
        "- 不做则取消：",
        f"  python -m plugins.plan_secretary cancel {shell_quote(pid)} --reason 'superseded or no longer needed'",
    ])
    return "\n".join(lines)


def notify(session_id: str = "", state_path: Path | None = None,
           default_due: str = "10m", due_repeat_minutes: int = 10,
           repeat_pending: bool = False) -> list[str]:
    captures_data = load_captures()
    registry = load_registry()
    if state_path is None:
        state_path = state_dir() / f"notification_state_{session_id or 'default'}.json"
    state = read_json(state_path, {})
    if not isinstance(state, dict):
        state = {}
    state.setdefault("notified_captures", {})
    state.setdefault("due_reminders", {})
    messages: list[str] = []
    now = now_local()
    active_scoped = active_plans_for_session(registry, session_id)

    notified = state.setdefault("notified_captures", {})
    for capture in captures_data["captures"]:
        cid = str(capture.get("id") or "")
        csid = capture_session_id(capture)
        if not cid or capture.get("status") != "pending":
            continue
        if session_id and csid and csid != session_id:
            continue
        if cid in notified and not repeat_pending:
            continue
        messages.append(build_capture_prompt(capture, active_scoped, default_due))
        notified[cid] = iso(now)

    reminders = state.setdefault("due_reminders", {})
    for plan in active_scoped:
        pid = str(plan.get("id") or "")
        due_at = parse_plan_time(plan.get("due"))
        if not pid or not due_at or due_at > now:
            continue
        last = reminders.get(pid)
        if last:
            last_dt = parse_plan_time(str(last))
            if last_dt and (now - last_dt) < timedelta(minutes=due_repeat_minutes):
                continue
        others = [p for p in active_scoped if p.get("id") != pid]
        messages.append(build_due_prompt(plan, others))
        reminders[pid] = iso(now)

    state["updated_at"] = iso(now)
    state["session_id"] = session_id
    write_json(state_path, state)
    return messages
