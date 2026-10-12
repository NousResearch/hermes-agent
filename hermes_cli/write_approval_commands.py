#!/usr/bin/env python3
"""Shared handlers for the /memory and /skills write-approval subcommands."""

from __future__ import annotations

import json
from typing import List, Optional

from tools import write_approval as wa


def _fmt_state(subsystem: str) -> str:
    on = wa.write_approval_enabled(subsystem)
    ar_name = "الذاكرة" if subsystem == wa.MEMORY else "المهارات"
    ar_status = "✅ مفعلة (on)" if on else "❌ معطلة (off)"
    lines = [f"حالة الموافقة على كتابة {ar_name}: {ar_status} | {subsystem}.write_approval = {'on' if on else 'off'}"]
    if subsystem == wa.MEMORY:
        from tools.memory_tool import get_builtin_memory_config
        from utils import is_truthy_value
        auto_cons = is_truthy_value(get_builtin_memory_config().get("auto_consolidate"), default=False)
        auto_status = "✅ مفعل (on)" if auto_cons else "❌ معطل (off)"
        lines.append(f"التوحيد الذاتي للذاكرة (Auto-consolidate): {auto_status} | memory.auto_consolidate = {'on' if auto_cons else 'off'}")
    return "\n".join(lines)


def _fmt_pending_list(subsystem: str) -> str:
    records = wa.list_pending(subsystem)
    ar_name = "الذاكرة" if subsystem == wa.MEMORY else "المهارات"
    if not records:
        return f"لا توجد عمليات معلقة في {ar_name} (No pending {subsystem} writes)."
    lines = [f"طلبات كتابة {ar_name} المعلقة (Pending {subsystem} writes: {len(records)}):"]
    for r in records:
        origin = r.get("origin", "foreground")
        tag = " [auto]" if origin == "background_review" else ""
        if subsystem == wa.MEMORY:
            payload = r.get("payload", {})
            target = payload.get("target", "memory")
            target_badge = "USER.md" if target == "user" else "MEMORY.md"
            raw_summary = wa.clean_compression_markers(r.get("summary", ""))
            if raw_summary.startswith("background review consolidation"):
                summary = f"مراجعة دورية وتوحيد للذاكرة ({raw_summary})"
            else:
                summary = raw_summary
            summary_part = f": {summary}" if summary else ""
            lines.append(f"\n{wa.RLM}  📌 [{r['id']}]{tag} [{target_badge}]{summary_part}:")
            ops = payload.get("operations") if payload.get("action") == "batch" else [payload]
            for op in ops:
                if not isinstance(op, dict):
                    continue
                act = op.get("action", "?")
                content = wa.clean_compression_markers(op.get("content") or op.get("new_text") or "")
                snippet = (content[:90] + "...") if len(content) > 90 else content
                old = wa.clean_compression_markers(op.get("old_text") or "")
                matched = wa.clean_compression_markers(op.get("matched_entry") or "")
                is_rtl = wa.has_rtl(old) or wa.has_rtl(matched) or wa.has_rtl(content)

                if is_rtl:
                    if act == "replace":
                        lines.append(f"{wa.RLM}    • 🔄 استبدال (المطابق لـ '{old[:40]}'):")
                        if matched:
                            lines.append(f"{wa.RLM}        المدخل الحالي: {matched}")
                        else:
                            lines.append(f"{wa.RLM}        استبدال: هدف قديم غير مثبت — ارفضه وأعد إنشاءه قبل الموافقة (unpinned legacy target)")
                        if snippet:
                            lines.append(f"{wa.RLM}        ← النص الجديد: {snippet}")
                    elif act == "remove":
                        lines.append(f"{wa.RLM}    • 🗑️ حذف (المطابق لـ '{old[:40]}'):")
                        if matched:
                            lines.append(f"{wa.RLM}        المدخل الحالي: {matched}")
                        else:
                            lines.append(f"{wa.RLM}        حذف: هدف قديم غير مثبت — ارفضه وأعد إنشاءه قبل الموافقة (unpinned legacy target)")
                    elif act == "add":
                        lines.append(f"{wa.RLM}    • ➕ إضافة: {snippet}")
                    else:
                        lines.append(f"{wa.RLM}    • {act}: {snippet}")
                else:
                    if act == "replace":
                        lines.append(f"    • 🔄 replace (matching '{old[:40]}'):")
                        if matched:
                            lines.append(f"        replaces entry: {matched}")
                        else:
                            lines.append("        replace: unpinned legacy target — reject and recreate before approving")
                        if snippet:
                            lines.append(f"        -> new text: {snippet}")
                    elif act == "remove":
                        lines.append(f"    • 🗑️ remove (matching '{old[:40]}'):")
                        if matched:
                            lines.append(f"        removes entry: {matched}")
                        else:
                            lines.append("        remove: unpinned legacy target — reject and recreate before approving")
                    elif act == "add":
                        lines.append(f"    • ➕ add: {snippet}")
                    else:
                        lines.append(f"    • {act}: {snippet}")
        else:
            lines.append(f"  {r['id']}{tag}  {r.get('summary', '')}")
    lines.append("")
    lines.append(f"للتطبيق: /{subsystem} approve <id>   للرفض: /{subsystem} reject <id>   للتفاصيل: /{subsystem} diff <id> (Diff: /{subsystem} diff <id>)")
    return "\n".join(lines)


def handle_pending_subcommand(
    subsystem: str, args: list[str], *, memory_store=None, set_mode_fn=None) -> Optional[str]:
    """Dispatch a /memory or /skills write-approval subcommand.

    ``memory_store`` applies approved memory writes (CLI passes its live store; gateway a freshly
    loaded one); ``set_mode_fn`` persists the write_approval boolean. Returns text for the user,
    or None when the args are not a write-approval subcommand so the caller falls through to its
    other handling (e.g. /skills search).
    """
    if not args:
        return f"{_fmt_state(subsystem)}\n\n" + _fmt_pending_list(subsystem)
    sub, rest = args[0].lower(), args[1:]
    if sub == "pending":
        return _fmt_pending_list(subsystem)
    if sub in {"approve", "apply"}:
        return _approve(subsystem, rest, memory_store)
    if sub in {"reject", "deny", "drop"}:
        return _reject(subsystem, rest)
    if sub == "diff":
        return _diff(subsystem, rest)
    if sub in {"approval", "mode"}:  # 'mode' kept as a back-compat alias
        return _set_approval(subsystem, rest, set_mode_fn)
    if sub in {"auto_consolidate", "auto-consolidate", "consolidate"}:
        return _set_auto_consolidate(subsystem, rest)
    return None  # not ours — caller handles


def _usage(subsystem: str) -> str:
    return f"Usage: /{subsystem} approve|reject <id>  (or 'all')"


def _approve(subsystem: str, rest: list[str], memory_store) -> str:
    if not rest:
        return _usage(subsystem)
    target = rest[0]
    records = wa.list_pending(subsystem)
    if not records:
        return f"No pending {subsystem} writes."
    if target.lower() == "all":
        targets = list(records)
    else:
        rec = wa.get_pending(subsystem, target)
        if not rec:
            return f"No pending {subsystem} write with id '{target}'."
        targets = [rec]

    applied, failed, overwritten, removed = 0, [], [], []
    for rec in targets:
        ok, msg, result = _apply_one(subsystem, rec, memory_store)
        if ok:
            wa.discard_pending(subsystem, rec["id"])
            applied += 1
            overwritten.extend(wa.bidi_line(text, prefix=f"  {rec['id']}: ") for text in _changed_entries(result, "replaced"))
            removed.extend(wa.bidi_line(text, prefix=f"  {rec['id']}: ") for text in _changed_entries(result, "removed"))
        else:
            failed.append(f"{rec['id']}: {msg}")

    out = [f"Approved {applied} {subsystem} write(s)."]
    if overwritten:
        # A memory 'replace' overwrites the WHOLE matched entry (#117952); the approver
        # is the last person who can notice a clause went missing, so show what was lost.
        out.append("Overwrote entire entry (re-add anything you still need):")
        out.extend(overwritten)
    if removed:
        out.append("Removed entry (re-add anything you still need):")
        out.extend(removed)
    if failed:
        out.append("Failed:")
        out.extend(f"  {f}" for f in failed)
    return "\n".join(out)


def _changed_entries(result: dict, kind: str) -> list[str]:
    """Full text of every entry a memory replace overwrote (``kind="replaced"``) or remove
    deleted (``"removed"``), single-op or batch shape."""
    single = result.get(f"{kind}_entry")
    batch = result.get(f"{kind}_entries") or {}
    return ([single] if single else []) + [batch[k] for k in sorted(batch, key=int)]


def _matched_entries(payload) -> list[str]:
    """The full entry each staged memory replace/remove is pinned to: the summary shows only
    the old_text search string, and approval applies to this entry, not to that search."""
    from tools.memory_tool import destructive_ops
    return [f"{op['action']}s entry: {op['matched_entry']}" if op.get("matched_entry")
            else f"{op['action']}: unpinned legacy target \u2014 reject and recreate before approving"
            for op in destructive_ops(payload)]


def _apply_one(subsystem: str, rec, memory_store):
    """``(ok, error, result)`` — *result* is the applier's full payload (empty on exceptions)."""
    payload = rec.get("payload", {})
    try:
        if subsystem == wa.MEMORY:
            if memory_store is None:
                return False, "memory store unavailable", {}
            from tools.memory_tool import apply_memory_pending
            result = apply_memory_pending(payload, memory_store)
        else:
            from tools.skill_manager_tool import apply_skill_pending
            result = json.loads(apply_skill_pending(payload))
        return bool(result.get("success")), result.get("error", ""), result
    except Exception as e:
        return False, str(e), {}


def _reject(subsystem: str, rest: list[str]) -> str:
    if not rest:
        return _usage(subsystem)
    target = rest[0]
    if target.lower() == "all":
        n = sum(1 for rec in wa.list_pending(subsystem) if wa.discard_pending(subsystem, rec["id"]))
        return f"Rejected {n} pending {subsystem} write(s)."
    if wa.discard_pending(subsystem, target):
        return f"Rejected pending {subsystem} write '{target}'."
    return f"No pending {subsystem} write with id '{target}'."


def _diff(subsystem: str, rest: list[str]) -> str:
    if not rest:
        return f"Usage: /{subsystem} diff <id>"
    rec = wa.get_pending(subsystem, rest[0])
    if not rec:
        return f"No pending {subsystem} write with id '{rest[0]}'."
    if subsystem == wa.SKILLS:
        return f"# Pending skill write {rec['id']}: {rec.get('summary', '')}\n\n" + wa.skill_pending_diff(rec)
    summary = wa.clean_compression_markers(rec.get("summary", ""))
    header = f"# Pending memory write {rec['id']}: {summary}" if summary else f"# Pending memory write {rec['id']}"
    if wa.has_rtl(header):
        header = f"{wa.RLM}{header}"
    return f"{header}\n\n" + wa.memory_pending_diff(rec)



_APPROVAL_VALUES = {
    **dict.fromkeys(("on", "true", "yes", "1", "enable", "enabled"), True),
    **dict.fromkeys(("off", "false", "no", "0", "disable", "disabled"), False)}


def _set_approval(subsystem: str, rest: list[str], set_mode_fn) -> str:
    """Turn the approval gate on/off for a subsystem."""
    if not rest:
        return (f"{_fmt_state(subsystem)}\n"
                f"Set with: /{subsystem} approval <on|off>")
    arg = rest[0].strip().lower()
    enabled = _APPROVAL_VALUES.get(arg)
    if enabled is None:
        return f"Invalid value '{arg}'. Use: on or off."
    if set_mode_fn is None:
        val = "true" if enabled else "false"
        return (f"To change the {subsystem} approval gate, run:\n"
                f"  hermes config set {subsystem}.write_approval {val}")
    try:
        set_mode_fn(enabled)
    except Exception as e:
        return f"Failed to set {subsystem}.write_approval: {e}"
    return f"{subsystem}.write_approval set to '{'on' if enabled else 'off'}'."


def _set_auto_consolidate(subsystem: str, rest: List[str]) -> str:
    """Turn autonomous background review consolidation on/off for memory."""
    if subsystem != wa.MEMORY:
        return f"Auto-consolidate is only applicable to {wa.MEMORY}."
    from tools.memory_tool import get_builtin_memory_config
    from utils import is_truthy_value
    if not rest:
        auto_cons = is_truthy_value(get_builtin_memory_config().get("auto_consolidate"), default=False)
        ar_status = "✅ مفعل (on)" if auto_cons else "❌ معطل (off)"
        return (f"التوحيد الذاتي للذاكرة (Auto-consolidate): {ar_status} | memory.auto_consolidate = {'on' if auto_cons else 'off'}\n"
                f"Set with: /memory auto_consolidate <on|off>")
    arg = rest[0].strip().lower()
    enabled = _APPROVAL_VALUES.get(arg)
    if enabled is None:
        return f"Invalid value '{arg}'. Use: on or off."
    try:
        from hermes_cli.config import read_user_config_raw, atomic_config_write
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
        user_config = read_user_config_raw(config_path)
        user_config.setdefault("memory", {})["auto_consolidate"] = bool(enabled)
        atomic_config_write(config_path, user_config)
    except Exception as e:
        return f"Failed to set memory.auto_consolidate: {e}"
    return f"memory.auto_consolidate set to '{'on' if enabled else 'off'}'."

