#!/usr/bin/env python3
"""Block unsupported completion claims against the latest user prompt.

This gate does not trust a percentage or the word "evidence" by itself.  For a
turn that used tools and claims completion, it requires a prompt checklist and,
when files changed, a real verification command after the last edit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys


DONE_RE = re.compile(
    r"(เสร็จ(?:แล้ว|ครบ|สมบูรณ์)?|ครบ(?:แล้ว|ทุกข้อ|ทั้งหมด|\s*100\s*%)|"
    r"ทำได้แล้ว|พร้อม(?:ใช้|ส่งมอบ|ปิดงาน)|อัปเดตให้แล้ว|จัดการแล้ว|เรียบร้อย(?:แล้ว)?|แก้ให้แล้ว|ทำให้แล้ว|\b(?:done|complete|completed|finished)\b)",
    re.I,
)
STATUS_RE = re.compile(r"✅|⚠️|❌|ผ่าน|ไม่ผ่าน|ค้าง|ยังไม่ครบ|N/A|claimed|verified", re.I)
EVIDENCE_RE = re.compile(
    r"หลักฐาน|คำสั่งตรวจ|ผลตรวจ|exit\s*0|exit\s*code|passed|HTTP\s*[23]\d\d|"
    r"ภาพ|screenshot|ไฟล์|บรรทัด|SHA|commit|ทดสอบ",
    re.I,
)
VERIFY_CMD_RE = re.compile(
    r"(?:^|\s)(?:pytest|python\s+-m\s+pytest|npm\s+test|pnpm\s+(?:test|lint|build|type-check)|"
    r"npm\s+run\s+(?:test|lint|build)|cargo\s+test|go\s+test|curl\b|git\s+diff\s+--check|"
    r"bash\s+-n|py_compile|tsc\b|gate-run\b)",
    re.I,
)
EDIT_NAME_RE = re.compile(r"(?:^|\.)(?:edit|write|multiedit|notebookedit|apply_patch)$", re.I)


def read_payload() -> dict:
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def load_entries(payload: dict) -> list[dict]:
    transcript = payload.get("transcript_path") or payload.get("transcript")
    if isinstance(transcript, list):
        return transcript
    if not isinstance(transcript, str) or not os.path.exists(transcript):
        return payload.get("messages", []) if isinstance(payload.get("messages"), list) else []
    entries = []
    with open(transcript, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries


def blocks(entry: dict) -> list[dict]:
    msg = entry.get("message", entry)
    content = msg.get("content", []) if isinstance(msg, dict) else []
    return content if isinstance(content, list) else []


def is_real_user(entry: dict) -> bool:
    if entry.get("type") != "user" and entry.get("role") != "user":
        return False
    return not any(block.get("type") == "tool_result" for block in blocks(entry) if isinstance(block, dict))


def user_text(entry: dict) -> str:
    msg = entry.get("message", entry)
    content = msg.get("content", "") if isinstance(msg, dict) else ""
    if isinstance(content, str):
        return content
    return "\n".join(str(b.get("text", "")) for b in content if isinstance(b, dict))


def requirement_count(prompt: str) -> int:
    rows = re.findall(r"^\s*(?:[-*]|\d+[.)])\s+\S.+$", prompt, re.M)
    if rows:
        return min(len(rows), 12)
    questions = prompt.count("?") + prompt.count("？")
    return min(max(1, questions), 12)


def table_rows(response: str) -> list[str]:
    rows = [line for line in response.splitlines() if line.strip().startswith("|")]
    return [row for row in rows if not re.match(r"^\s*\|?\s*:?-{3,}", row)]


def tool_state(turn: list[dict]) -> tuple[bool, bool, int, bool, int]:
    used_tool = False
    edited = False
    last_edit = -1
    pending_verifications = {}
    successful_proofs = 0
    successful_results = 0
    strong_gate = False
    verified_after_edit = False
    for index, entry in enumerate(turn):
        for block in blocks(entry):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result" and pending_verifications:
                result_id = block.get("tool_use_id") or "__next__"
                result_text = str(block.get("content", ""))
                failed = block.get("is_error") or re.search(r"(?:exit\s*[1-9]|failed|error|traceback)", result_text, re.I)
                pending_key = result_id if result_id in pending_verifications else "__next__"
                command = pending_verifications.get(pending_key, "")
                if command and result_text.strip() and not failed:
                    verified_after_edit = True
                    successful_proofs += 1
                    successful_results += 1
                    if re.search(r"\bgate-run\b", command, re.I):
                        strong_gate = True
                continue
            if block.get("type") == "tool_result":
                result_text = str(block.get("content", ""))
                failed = block.get("is_error") or re.search(r"(?:exit\s*[1-9]|failed|error|traceback)", result_text, re.I)
                if result_text.strip() and not failed:
                    successful_results += 1
                continue
            if block.get("type") != "tool_use":
                continue
            used_tool = True
            name = str(block.get("name", ""))
            tool_input = block.get("input", {}) if isinstance(block.get("input"), dict) else {}
            command = str(tool_input.get("cmd") or tool_input.get("command") or "")
            if EDIT_NAME_RE.search(name) or "apply_patch" in name.lower():
                edited = True
                last_edit = index
                verified_after_edit = False
            elif edited and index >= last_edit and VERIFY_CMD_RE.search(command):
                pending_verifications[block.get("id") or block.get("tool_use_id") or "__next__"] = command
    return used_tool, edited, successful_proofs, strong_gate, successful_results


def main() -> int:
    if os.environ.get("PROMPT_EVIDENCE_GATE_DISABLED") == "1":
        return 0
    payload = read_payload()
    if not payload or payload.get("stop_hook_active"):
        return 0
    response = payload.get("last_assistant_message") or payload.get("response") or payload.get("message") or ""
    if not isinstance(response, str) or not DONE_RE.search(response):
        return 0
    entries = load_entries(payload)
    last_user = max((i for i, entry in enumerate(entries) if is_real_user(entry)), default=-1)
    if last_user < 0:
        return 0
    prompt = user_text(entries[last_user])
    turn = entries[last_user + 1 :]
    used_tool, edited, successful_proofs, strong_gate, successful_results = tool_state(turn)
    if not used_tool:
        return 0

    required = requirement_count(prompt)
    rows = table_rows(response)
    data_rows = [row for row in rows if STATUS_RE.search(row) and EVIDENCE_RE.search(row)]
    problems = []
    if "ตรวจครบตามคำสั่ง" not in response:
        problems.append("ไม่มีหัวข้อ 'ตรวจครบตามคำสั่ง'")
    if len(data_rows) < required:
        problems.append(f"ตารางหลักฐานมี {len(data_rows)}/{required} ข้อ")
    if successful_results == 0:
        problems.append("ไม่พบผลจากเครื่องมือจริงแม้แต่ 1 รายการ")
    if edited and successful_proofs == 0:
        problems.append("แก้ไฟล์แล้วแต่ไม่พบคำสั่งตรวจจริงหลังการแก้ล่าสุด")
    if edited and not strong_gate and successful_proofs < required:
        problems.append(f"ผลตรวจจริงมี {successful_proofs}/{required} ข้อ")
    if not problems:
        return 0

    print(
        "BLOCK · อ้างว่างานครบ แต่หลักฐานยังไม่ครบตาม Prompt จริง\n"
        + "\n".join(f"  - {item}" for item in problems)
        + "\nแก้: รันการตรวจจริง แล้วใส่ตาราง 'ตรวจครบตามคำสั่ง' หนึ่งแถวต่อหนึ่งข้อ "
        "พร้อมสถานะ หลักฐาน และสิ่งที่ยังค้าง · ถ้ายังไม่ครบให้รายงานเปอร์เซ็นต์จริง ห้ามใช้คำว่าเสร็จ/ครบ",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
