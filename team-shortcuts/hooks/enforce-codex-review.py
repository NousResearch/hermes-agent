#!/usr/bin/env python3
"""
enforce-codex-review.py  ·  Stop hook  ·  ด่านก่อนปิดงานโค้ด
------------------------------------------------------------------------
ปฐมเหตุ: AI รู้กติกา "Claude ทำ · Codex ตรวจ" แต่พอเร่งงานก็ข้ามขั้นส่ง Codex
        แล้วมาร์คว่า "เสร็จ" เอง · Codex ภายหลังเจอบั๊กจริง (เลขอ้างอิงชน · โหมดมืดเสีย)

หลักการ: ถ้ารอบนี้ AI "แก้ไฟล์โค้ดจริง" + คำตอบบอก "เสร็จ/ปิดงาน" + "ยังไม่ได้เรียก Codex"
        → BLOCK (exit 2) · บังคับให้ส่ง Codex ตรวจก่อนปิดงาน · ไม่พึ่งความจำ AI

ปิดด่าน: export CODEX_REVIEW_GATE_DISABLED=1
"""
import json
import os
import re
import sys
import time

LOG = os.path.expanduser("~/.claude/codex-review-gate.log")

CODE_EXT = (".ts", ".tsx", ".js", ".jsx", ".prisma", ".py", ".go", ".rs", ".java", ".css", ".scss")
EDIT_TOOLS = {"edit", "write", "multiedit", "notebookedit", "apply_patch", "functions.apply_patch"}
# คำที่บ่งว่า "อ้างว่าเสร็จ/ปิดงาน"
DONE_RE = re.compile(
    r"(เสร็จแล้ว|เสร็จครบ|เสร็จสมบูรณ์|ปิดงาน|ปิดครบ|ปิดเฟส|ปิด P\d|มาร์ค[^\n]{0,12}เสร็จ|พร้อมส่งมอบ|อัปเดตให้แล้ว|จัดการแล้ว|เรียบร้อย(?:แล้ว)?|แก้ให้แล้ว|ทำให้แล้ว|\bdone\b)",
    re.IGNORECASE,
)
# ถ้า AI พูดเองว่ากำลังจะส่ง/ยังไม่ส่ง Codex → ถือว่ารู้ตัว ไม่บล็อก (กัน false positive)
CODEX_TOOL_RE = re.compile(r"cross.?check|ask_gpt5|relay-call", re.IGNORECASE)
REVIEW_COMMAND_RE = re.compile(r"relay-call[^\n]*(?:--role\s+review|--tool\s+(?:codex|opus|grok))", re.IGNORECASE)


def read_payload():
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def is_code_file(path):
    p = (path or "").lower()
    if not p.endswith(CODE_EXT):
        return False
    # นับเฉพาะไฟล์ในซอร์สจริง · ไม่นับ hook ตัวเอง / สคริปต์ตั้งค่า
    if "/.claude/" in p or "/hooks/" in p:
        return False
    return True


def edited_paths(name, tool_input):
    """รองรับทั้งเครื่องมือเก่าและ apply_patch ที่ใช้จริงใน Codex."""
    low_name = (name or "").lower()
    if low_name not in EDIT_TOOLS and "apply_patch" not in low_name:
        return []
    tool_input = tool_input if isinstance(tool_input, dict) else {}
    paths = []
    for key in ("file_path", "path"):
        value = tool_input.get(key)
        if isinstance(value, str):
            paths.append(value)
    patch = tool_input.get("patch") or tool_input.get("input") or ""
    if isinstance(patch, str):
        paths.extend(re.findall(r"^\*{3} (?:Update|Add|Delete) File: (.+)$", patch, re.M))
    return paths


def main():
    if os.environ.get("CODEX_REVIEW_GATE_DISABLED") == "1":
        sys.exit(0)
    p = read_payload()
    if p.get("stop_hook_active"):
        sys.exit(0)
    tpath = p.get("transcript_path") or p.get("transcript")
    if not tpath or not os.path.exists(tpath):
        sys.exit(0)

    entries = []
    try:
        with open(tpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        sys.exit(0)

    # หา index ข้อความ "คนพิมพ์จริง" ล่าสุด (ไม่ใช่ tool_result)
    last_human = -1
    for i, e in enumerate(entries):
        if e.get("type") == "user" or e.get("role") == "user":
            msg = e.get("message", e)
            content = msg.get("content") if isinstance(msg, dict) else None
            is_tool_result = isinstance(content, list) and any(
                isinstance(c, dict) and c.get("type") == "tool_result" for c in content
            )
            if not is_tool_result:
                last_human = i
    turn = entries[last_human + 1:] if last_human >= 0 else entries

    edited_code = False
    review_requests = set()
    review_completed = False
    last_assistant_text = ""
    for e in turn:
        msg = e.get("message", e)
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "tool_use":
                name = c.get("name", "")
                tool_input = c.get("input") or {}
                if any(is_code_file(fp) for fp in edited_paths(name, tool_input)):
                        edited_code = True
                if CODEX_TOOL_RE.search(name):
                    command = tool_input.get("cmd") or tool_input.get("command") or ""
                    if "relay-call" not in name.lower() or REVIEW_COMMAND_RE.search(str(command)):
                        review_requests.add(c.get("id") or c.get("tool_use_id") or "__next__")
                command = tool_input.get("cmd") or tool_input.get("command") or ""
                if REVIEW_COMMAND_RE.search(str(command)):
                    review_requests.add(c.get("id") or c.get("tool_use_id") or "__next__")
            elif c.get("type") == "tool_result" and review_requests:
                result_id = c.get("tool_use_id") or "__next__"
                content_text = str(c.get("content", ""))
                has_review_content = len(content_text.strip()) >= 80 and re.search(
                    r"PASS|FAIL|finding|risk|diff|ปัญหา|ความเสี่ยง|ตรวจ",
                    content_text,
                    re.I,
                )
                if (result_id in review_requests or "__next__" in review_requests) and has_review_content and not c.get("is_error"):
                    review_completed = True
            elif c.get("type") == "text" and (e.get("type") == "assistant" or e.get("role") == "assistant"):
                last_assistant_text = c.get("text", "")

    if not edited_code or review_completed:
        sys.exit(0)
    if not DONE_RE.search(last_assistant_text):
        sys.exit(0)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | BLOCK (แก้โค้ด+อ้างเสร็จ+ไม่ส่ง Codex)\n")
    except Exception:
        pass
    sys.stderr.write(
        "🛑 ด่านก่อนปิดงานโค้ด · รอบนี้แก้ไฟล์โค้ดจริง + อ้างว่าเสร็จ "
        "แต่ยังไม่ได้ส่ง Codex ตรวจ\n"
        "กติกา: Claude ทำ · Codex ตรวจ · ห้ามมาร์คเสร็จก่อน Codex ตรวจ\n"
        "ให้เรียก mcp__cross-check__ask_gpt5 ตรวจงานโค้ดที่เพิ่งแก้ก่อน แล้วค่อยสรุปปิดงาน\n"
        "ปิดด่าน: export CODEX_REVIEW_GATE_DISABLED=1\n"
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
