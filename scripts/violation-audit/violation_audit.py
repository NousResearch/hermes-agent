#!/usr/bin/env python3
"""
violation-audit — ตัวตรวจการข้ามด่านย้อนหลัง (F1)

ปัญหาที่แก้: มีด่าน/กฎเยอะ แต่การละเมิดยังเกิดซ้ำหลักร้อยต่อสัปดาห์
เพราะไม่มีใครไล่อ่าน log ย้อนหลัง — สคริปต์นี้อ่าน log จริงทุกไฟล์
แล้วสรุปเป็นรายงานภาษาคน เทียบสัปดาห์นี้กับสัปดาห์ก่อน ว่าอะไรแย่ลง/ดีขึ้น

แหล่งข้อมูล (อ่านอย่างเดียว ไม่แก้ ไม่ลบ):
  ~/.claude/hooks-violations.log   บรรทัด JSON มี ts + hook (นับเป๊ะรายช่วงเวลา)
  ~/.claude/protected-gate.log     ด่านกันแก้ไฟล์หวงห้าม
  ~/.claude/danger-bash-gate.log   ด่านกันคำสั่งอันตราย (commit/push/ลบ)
  ~/.claude/codex-review-gate.log  ด่านกันปิดงานโค้ดโดยไม่ส่งตรวจ
  ~/.claude/project-scope-gate.log ด่านกันทำนอกขอบเขต project

ใช้:
  violation-audit                 # รายงาน 7 วันล่าสุด เทียบ 7 วันก่อนหน้า
  violation-audit --days 14       # เปลี่ยนช่วงเวลา
  violation-audit --notify        # เด้งแจ้งเตือน macOS พร้อมสรุป 1 บรรทัด
  violation-audit --out DIR       # ที่เก็บไฟล์รายงาน (ค่าเริ่มต้น ~/.claude/ai-fail-stats)
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

VIOL_LOG = Path.home() / ".claude" / "hooks-violations.log"
GATE_LOGS = [
    ("ด่าน 1 · ไฟล์หวงห้าม (สี/CI/ความลับ)", Path.home() / ".claude" / "protected-gate.log"),
    ("ด่าน 2 · คำสั่งอันตราย (commit/push/ลบ)", Path.home() / ".claude" / "danger-bash-gate.log"),
    ("ด่าน 3 · ปิดงานโค้ดโดยไม่ส่ง Codex ตรวจ", Path.home() / ".claude" / "codex-review-gate.log"),
    ("ด่าน 4 · ทำนอกขอบเขต project", Path.home() / ".claude" / "project-scope-gate.log"),
]

# ชื่อ hook → คำอธิบายภาษาคน (ตรงกับหมวดใน repeat-violation-alert.py)
HOOK_LABELS = {
    "validate-response-contract": "ผิดสัญญารูปแบบคำตอบ (ให้เลือกหลายทาง/ไม่บอกคุณค่า/UI ไม่มีภาพ)",
    "validate-thai-language": "ไม่ใช้ภาษาไทย/ภาษาคนง่าย ๆ",
    "validate-preflight-card": "ไม่ทำการ์ดทบทวน prompt ก่อนตอบ",
    "enforce-prompt-compile": "ไม่ทวนคำสั่งเป็นตาราง checklist",
    "enforce-numerical-gate": "บอกว่าเสร็จโดยไม่มีตัวเลข %",
    "enforce-tech-glossary": "ใช้ศัพท์ช่างโดยไม่แปลไทย",
    "enforce-no-guess": "เดา/มั่วค่าที่ไม่ได้ตรวจจริง",
    "validate-ui-task-visual-proof": "อ้างว่างาน UI เสร็จโดยไม่มีภาพยืนยัน",
    "enforce-spec-evidence": "ปิดงานโค้ดโดยไม่มีตารางหลักฐานตรงสเปค (P1)",
    "enforce-design-score": "เริ่มเขียนโค้ดหน้าจอโดยไม่ให้คะแนนดีไซน์ก่อน (P2)",
    "enforce-canary": "ขึ้น production แล้วไม่เฝ้าระบบตามกติกา (P3)",
}


def parse_ts(value):
    try:
        return datetime.fromisoformat(value.strip()[:19])
    except (ValueError, AttributeError):
        return None


def load_hook_events():
    """คืนรายการ (ts, hook, ข้อความย่อ) จากบรรทัด JSON ที่มีวันที่เท่านั้น"""
    events = []
    if not VIOL_LOG.exists():
        return events
    with open(VIOL_LOG, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = parse_ts(row.get("ts", ""))
            if ts is None:
                continue
            detail = row.get("reason") or " · ".join(row.get("violations") or [])
            events.append((ts, row.get("hook", "unknown"), detail[:160]))
    return events


def load_gate_events(path):
    """คืนรายการ (ts, BLOCK/ALLOW, รายละเอียด) จาก log ด่านรูปแบบ 'YYYY-MM-DD HH:MM:SS | ...'"""
    events = []
    if not path.exists():
        return events
    pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| (\w+)[^|]*(?:\| (.*))?$")
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            ts = parse_ts(m.group(1).replace(" ", "T"))
            if ts:
                events.append((ts, m.group(2), (m.group(3) or "")[:120]))
    return events


def in_window(events, start, end):
    return [e for e in events if start <= e[0] < end]


def trend_mark(now_n, before_n):
    if now_n > before_n:
        return f"แย่ลง (+{now_n - before_n})"
    if now_n < before_n:
        return f"ดีขึ้น (-{before_n - now_n})"
    return "เท่าเดิม"


def build_report(days):
    now = datetime.now()
    start = now - timedelta(days=days)
    prev_start = now - timedelta(days=days * 2)

    hook_events = load_hook_events()
    cur = in_window(hook_events, start, now)
    prev = in_window(hook_events, prev_start, start)
    cur_by_hook = Counter(h for _, h, _ in cur)
    prev_by_hook = Counter(h for _, h, _ in prev)

    lines = []
    lines.append(f"# รายงานตรวจการละเมิดกฎย้อนหลัง {days} วัน")
    lines.append(f"ออกรายงาน: {now:%Y-%m-%d %H:%M} · ช่วงที่ตรวจ: {start:%Y-%m-%d} → {now:%Y-%m-%d} เทียบกับ {days} วันก่อนหน้า")
    lines.append("")
    verdict = trend_mark(len(cur), len(prev))
    lines.append(f"**สรุปรวม: ละเมิด {len(cur)} ครั้ง (ช่วงก่อนหน้า {len(prev)} ครั้ง) → {verdict}**")
    lines.append("")

    lines.append("## ละเมิดแยกตามกฎ (นับเป๊ะจากบรรทัดที่มีวันที่)")
    lines.append("")
    lines.append(f"| กฎที่ถูกละเมิด | {days} วันนี้ | ช่วงก่อน | แนวโน้ม |")
    lines.append("|---|---:|---:|---|")
    for hook, n in cur_by_hook.most_common():
        label = HOOK_LABELS.get(hook, hook)
        lines.append(f"| {label} | {n} | {prev_by_hook.get(hook, 0)} | {trend_mark(n, prev_by_hook.get(hook, 0))} |")
    for hook, n in prev_by_hook.most_common():
        if hook not in cur_by_hook:
            label = HOOK_LABELS.get(hook, hook)
            lines.append(f"| {label} | 0 | {n} | ดีขึ้น (-{n}) |")
    lines.append("")

    lines.append("## ด่านขวางก่อนลงมือ (BLOCK = ด่านช่วยหยุดไว้ได้จริง)")
    lines.append("")
    lines.append(f"| ด่าน | BLOCK {days} วันนี้ | BLOCK ช่วงก่อน | ALLOW {days} วันนี้ |")
    lines.append("|---|---:|---:|---:|")
    gate_block_total = 0
    for label, path in GATE_LOGS:
        ev = load_gate_events(path)
        c = in_window(ev, start, now)
        p = in_window(ev, prev_start, start)
        blocks = sum(1 for _, kind, _ in c if kind == "BLOCK")
        gate_block_total += blocks
        lines.append(
            f"| {label} | {blocks} | {sum(1 for _, k, _ in p if k == 'BLOCK')} | "
            f"{sum(1 for _, k, _ in c if k == 'ALLOW')} |"
        )
    lines.append("")

    if cur:
        lines.append("## ตัวอย่างล่าสุด 3 รายการ (ไว้ตามรอยว่าพลาดแบบไหน)")
        lines.append("")
        for ts, hook, detail in cur[-3:]:
            lines.append(f"- {ts:%m-%d %H:%M} · {HOOK_LABELS.get(hook, hook)} · {detail}")
        lines.append("")

    lines.append("## ข้อเสนอจากข้อมูลรอบนี้")
    if cur_by_hook:
        top_hook, top_n = cur_by_hook.most_common(1)[0]
        lines.append(
            f"- กฎที่โดนละเมิดมากสุดคือ \"{HOOK_LABELS.get(top_hook, top_hook)}\" ({top_n} ครั้ง) — "
            "ถ้าซ้ำติดกันหลายรอบ ควรยกระดับจาก \"เตือนตอนจบ\" เป็น \"ด่านบล็อกก่อนลงมือ\""
        )
    lines.append(f"- ด่านบล็อกช่วยหยุดได้ {gate_block_total} ครั้งในช่วงนี้ — ด่านที่ BLOCK สูงคือจุดที่กฎเตือนอย่างเดียวเอาไม่อยู่")
    summary = f"ละเมิด {len(cur)} ครั้งใน {days} วัน ({verdict}) · ด่านบล็อกช่วยไว้ {gate_block_total} ครั้ง"
    return "\n".join(lines) + "\n", summary


def main():
    ap = argparse.ArgumentParser(description="รายงานตรวจการละเมิดกฎย้อนหลัง")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", default=str(Path.home() / ".claude" / "ai-fail-stats"))
    ap.add_argument("--notify", action="store_true", help="เด้งแจ้งเตือน macOS")
    a = ap.parse_args()

    report, summary = build_report(a.days)
    out_dir = Path(a.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"violation-report-{datetime.now():%Y%m%d}.md"
    out_file.write_text(report, encoding="utf-8")

    print(report)
    print(f"[บันทึกรายงานแล้วที่] {out_file}")

    if a.notify and sys.platform == "darwin":
        try:
            subprocess.run(
                ["osascript", "-e",
                 f'display notification "{summary}" with title "Hermes · ตรวจการละเมิดกฎ"'],
                check=False, timeout=10,
            )
        except OSError:
            pass


if __name__ == "__main__":
    main()
