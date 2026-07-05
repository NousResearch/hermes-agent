#!/usr/bin/env python3
"""relay-suggest — แนะนำ AI coder + reviewer จากทะเบียนและสถานะสด"""
import argparse
import importlib.util
import json
import os
import pathlib
import shutil
from datetime import datetime


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def _load_sibling(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


relay_call = _load_sibling("relay_call", "relay-call.py")
relay_status = _load_sibling("relay_status", "relay-status.py")


TASK_TAGS = {
    "backend": ["backend", "logic", "security"],
    "logic": ["backend", "logic", "security"],
    "security": ["backend", "logic", "security"],
    "ui": ["ui", "large-context"],
    "fast": ["fast", "bulk", "repetitive"],
    "general": [],
}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    return [str(value)]


def _status_by_name(status_map):
    if isinstance(status_map, list):
        return {str(row.get("name")): row for row in status_map if isinstance(row, dict)}
    return status_map or {}


def _is_ready(status_map, name):
    row = _status_by_name(status_map).get(name)
    if isinstance(row, bool):
        return row
    if isinstance(row, dict):
        return row.get("ready") is True
    return False


def _cost_tier(meta):
    try:
        return int(meta.get("cost_tier", 99))
    except Exception:
        return 99


def _task_tags(task_type):
    key = (task_type or "general").strip().lower() or "general"
    if key in TASK_TAGS:
        return TASK_TAGS[key]
    # ชนิดงานใหม่ให้ลองเทียบกับ good_for ตรงตัว แต่ไม่เดาความหมายเอง
    return [key]


def _enabled_role_ready(reg, status_map, role):
    names = []
    for name, meta in (reg or {}).items():
        if not isinstance(meta, dict):
            continue
        if meta.get("enabled") is not True:
            continue
        if role not in _as_list(meta.get("roles")):
            continue
        if not _is_ready(status_map, name):
            continue
        names.append(name)
    return names


def _coder_matches(task_type, meta):
    tags = set(_task_tags(task_type))
    good_for = set(_as_list(meta.get("good_for")))
    return bool(tags and (tags & good_for))


def _sorted_coders(task_type, names, reg):
    # ลำดับชั้น (GPT-5 fix): ตรงงานก่อนเสมอ → ถูกกว่า → ชื่อ (ตัดเสมอแบบแน่นอน)
    # กันเคสตัว "ถูกแต่ไม่ตรงงาน" แซงตัว "ตรงงานแต่แพง" (สูตรบวกลบคะแนนเดิมมีช่องนี้)
    return sorted(
        names,
        key=lambda name: (
            0 if _coder_matches(task_type, reg[name]) else 1,
            _cost_tier(reg[name]),
            name,
        ),
    )


def _sorted_reviewers(names, reg):
    return sorted(names, key=lambda name: (_cost_tier(reg[name]), name))


def _human_tags(meta):
    tags = _as_list(meta.get("good_for"))
    return "/".join(tags) if tags else "งานทั่วไป"


def _reason_for_coder(name, task_type, meta):
    tags = set(_task_tags(task_type))
    good_for = set(_as_list(meta.get("good_for")))
    match = bool(tags and (tags & good_for))
    if (task_type or "general").strip().lower() == "general":
        fit = "เลือกจากระดับราคาต่ำและพร้อมใช้"
    elif match:
        fit = f"ถนัด {_human_tags(meta)} ตรงกับงาน"
    else:
        fit = f"พร้อมใช้ แต่ความถนัดหลักคือ {_human_tags(meta)}"
    return f"{name}: {fit} · ระดับราคา {_cost_tier(meta)}"


def _reason_for_reviewer(name, coder, meta):
    return f"{name}: คนละค่ายกับ {coder} · พร้อมใช้ · ระดับราคา {_cost_tier(meta)}"


def _login_hints(reg, names):
    hints = []
    for name in names:
        meta = (reg or {}).get(name, {})
        hint = meta.get("login_hint") or f"ติดตั้งหรือล็อกอิน {name}"
        hints.append(f"{name}: {hint}")
    return "; ".join(hints)


def _enabled_role_names(reg, role):
    return [
        name
        for name, meta in (reg or {}).items()
        if isinstance(meta, dict)
        and meta.get("enabled") is True
        and role in _as_list(meta.get("roles"))
    ]


def suggest(task_type, reg, status_map):
    """ฟังก์ชันบริสุทธิ์: รับข้อมูลที่ฉีดมา แล้วคืนคำแนะนำโดยไม่อ่านไฟล์/ไม่ยิงเน็ต"""
    status_map = _status_by_name(status_map)
    warnings = []
    reasons = []

    coders = _sorted_coders(task_type, _enabled_role_ready(reg, status_map, "coder"), reg)
    coder = coders[0] if coders else None
    fallbacks = coders[1:] if coder else []

    if not coder:
        possible = _enabled_role_names(reg, "coder")
        hint_text = _login_hints(reg, possible) or "ยังไม่มี AI ที่เปิดบทบาท coder ในทะเบียน"
        warnings.append(f"ไม่มี coder ที่พร้อม — ต้องติดตั้ง/ล็อกอิน: {hint_text}")
        return {
            "coder": None,
            "reviewer": None,
            "fallbacks": [],
            "reasons": reasons,
            "warnings": warnings,
        }

    reasons.append(_reason_for_coder(coder, task_type, reg[coder]))

    coder_vendor = reg[coder].get("vendor")
    reviewer_pool = [
        name
        for name in _enabled_role_ready(reg, status_map, "reviewer")
        if reg.get(name, {}).get("vendor") != coder_vendor
    ]
    reviewers = _sorted_reviewers(reviewer_pool, reg)
    reviewer = reviewers[0] if reviewers else None

    if reviewer:
        reasons.append(_reason_for_reviewer(reviewer, coder, reg[reviewer]))
    else:
        possible_reviewers = [
            name
            for name in _enabled_role_names(reg, "reviewer")
            if reg.get(name, {}).get("vendor") != coder_vendor
        ]
        hint_text = _login_hints(reg, possible_reviewers)
        if hint_text:
            warnings.append(
                "ไม่มีคนตรวจคนละค่ายที่พร้อม — "
                f"ควรล็อกอิน {hint_text} หรือยอมรับความเสี่ยง"
            )
        else:
            warnings.append(
                "ไม่มีคนตรวจคนละค่ายที่พร้อม — "
                "ควรเพิ่ม reviewer คนละค่ายในทะเบียนหรือยอมรับความเสี่ยง"
            )

    return {
        "coder": coder,
        "reviewer": reviewer,
        "fallbacks": fallbacks,
        "reasons": reasons,
        "warnings": warnings,
    }


def _collect_status_map(cwd, reg, probe_names=None):
    probe_names = set(probe_names or [])
    enabled = relay_call.registry_enabled(reg)
    cooldown_map = relay_status.read_cooldown_map(cwd)
    rows = {}
    for name, meta in enabled.items():
        probe_result = relay_status.run_probe(cwd, name) if name in probe_names else None
        rows[name] = relay_status.tool_status(
            name,
            meta,
            shutil.which,
            cooldown_map,
            probe_result=probe_result,
        )
    return rows


def _probe_targets(result):
    names = []
    for key in ("coder", "reviewer"):
        name = result.get(key)
        if name and name not in names:
            names.append(name)
    return names


def _role_label(meta):
    roles = _as_list(meta.get("roles"))
    if roles == ["brain"]:
        return "เป็นสมองไม่ใช่ coder"
    if roles:
        return "บทบาทคือ " + "/".join(roles) + " ไม่ใช่ coder"
    return "ไม่มีบทบาทในทะเบียน"


def _skip_lines(reg, status_map, result):
    selected = set(result.get("fallbacks") or [])
    for key in ("coder", "reviewer"):
        if result.get(key):
            selected.add(result[key])
    rows = _status_by_name(status_map)
    skipped = []
    for name, meta in sorted((reg or {}).items()):
        if not isinstance(meta, dict) or meta.get("enabled") is not True:
            continue
        if name in selected:
            continue
        if "coder" not in _as_list(meta.get("roles")):
            skipped.append(f"{name} ({_role_label(meta)})")
            continue
        row = rows.get(name, {})
        if not _is_ready(rows, name):
            hint = row.get("hint") if isinstance(row, dict) else ""
            skipped.append(f"{name} ({hint or meta.get('login_hint') or 'ยังไม่พร้อม'})")
    return skipped


def _short_pick_reason(name, reg, status_map):
    if not name:
        return ""
    meta = reg.get(name, {})
    row = _status_by_name(status_map).get(name, {})
    ready = "พร้อมใช้" if isinstance(row, dict) and row.get("ready") else "ยังไม่พร้อม"
    tags = _human_tags(meta)
    return f"ถนัด {tags} · {ready}"


def format_human(task_type, result, reg, status_map):
    task = (task_type or "general").strip().lower() or "general"
    lines = []
    coder = result.get("coder")
    if coder:
        fallback_text = ", ".join(result.get("fallbacks") or []) or "ไม่มี"
        lines.append(
            f"งาน {task} · แนะนำ **{coder}** ({_short_pick_reason(coder, reg, status_map)}) "
            f"· สำรอง: {fallback_text}"
        )
    else:
        lines.append(f"งาน {task} · ยังไม่มี coder ที่พร้อมใช้ตอนนี้ · สำรอง: ไม่มี")

    reviewer = result.get("reviewer")
    if reviewer and coder:
        lines.append(
            f"คนตรวจ: **{reviewer}** "
            f"(คนละค่ายกับ {coder} · {_short_pick_reason(reviewer, reg, status_map)})"
        )
    else:
        lines.append("คนตรวจ: ยังไม่มีคนตรวจคนละค่ายที่พร้อม")

    skipped = _skip_lines(reg, status_map, result)
    if skipped:
        lines.append("ข้าม: " + ", ".join(skipped))
    for warning in result.get("warnings") or []:
        lines.append("คำเตือน: " + warning)
    if result.get("reasons"):
        lines.append("เหตุผล: " + " | ".join(result["reasons"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-type", default="general", help="ชนิดงาน เช่น backend, ui, fast, security, logic")
    ap.add_argument("--cwd", default=os.getcwd(), help="โฟลเดอร์โปรเจกต์ ค่าเริ่มต้นคือโฟลเดอร์ปัจจุบัน")
    ap.add_argument("--probe", action="store_true", help="เช็ค login จริงเฉพาะตัวที่กำลังจะแนะนำ")
    ap.add_argument("--json", action="store_true", help="คืน JSON แทนภาษาคน")
    args = ap.parse_args()

    cwd = pathlib.Path(args.cwd).expanduser().resolve()
    reg = relay_call.load_registry(cwd)
    status_map = _collect_status_map(cwd, reg)
    result = suggest(args.task_type, reg, status_map)

    probed = []
    if args.probe:
        probed = _probe_targets(result)[:2]
        if probed:
            status_map = _collect_status_map(cwd, reg, probe_names=probed)
            result = suggest(args.task_type, reg, status_map)
            new_targets = [name for name in _probe_targets(result) if name not in probed]
            if new_targets:
                result["warnings"].append(
                    "หลัง probe ตัวเลือกแรกไม่ผ่าน จึงเลือกตัวสำรองจากสถานะติดตั้ง+สถานะพักเท่านั้น "
                    "เพื่อไม่เรียก AI เกิน 1-2 ตัว"
                )

    if args.json:
        out = {
            "root": str(cwd),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "task_type": args.task_type,
            "probe": bool(args.probe),
            "probed": probed,
            "suggestion": result,
            "status": status_map,
        }
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return

    print(format_human(args.task_type, result, reg, status_map))


if __name__ == "__main__":
    main()
