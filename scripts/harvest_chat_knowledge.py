#!/usr/bin/env python3
"""ตัวเก็บเกี่ยวความรู้จากแชท AI (เฟส 2 · ระบบ 360)

อ่านประวัติแชทอย่างเดียว ไม่แก้ของเดิม
สร้าง digest (วัตถุดิบย่อ) ต่อ session เพื่อให้ AI สรุปเป็นการ์ดความรู้ต่อไป

ใช้งาน:
    python scripts/harvest_chat_knowledge.py --source claude-code --limit 3
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
SOURCES = {
    "claude-code": HOME / ".claude" / "projects",
    "codex": HOME / ".codex" / "sessions",
}
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "360-content-system" / "knowledge-inbox"


def iter_text_blocks(content) -> list[str]:
    """ดึงเฉพาะข้อความคน อ่านง่าย ตัด noise ของเครื่องมือออก"""
    out: list[str] = []
    if isinstance(content, str):
        if content.strip():
            out.append(content.strip())
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text", "").strip():
                out.append(block["text"].strip())
    return out


def read_session(path: Path, max_msgs: int = 60) -> dict:
    """อ่าน 1 transcript คืนสรุปข้อความคน + meta"""
    msgs: list[tuple[str, str]] = []
    cwd = ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        cwd = obj.get("cwd", cwd)
        role = obj.get("type")
        if role not in ("user", "assistant"):
            continue
        message = obj.get("message", {})
        for text in iter_text_blocks(message.get("content")):
            # ตัด system-reminder / hook noise ที่ยาวและไม่ใช่สาระ
            if text.startswith("<") or "system-reminder" in text[:40]:
                continue
            msgs.append((role, text))
    return {
        "file": str(path),
        "cwd": cwd,
        "project": Path(cwd).name if cwd else path.parent.name,
        "msg_count": len(msgs),
        "messages": msgs[:max_msgs],
    }


def collect(source: str, limit: int) -> list[Path]:
    root = SOURCES[source]
    files = sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    # เก็บเฉพาะไฟล์ที่มีสาระ (ขนาด > 8 KB กันแชทสั้น)
    return [p for p in files if p.stat().st_size > 8192][:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), default="claude-code")
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()

    new_dir = OUT_DIR / "_new"
    new_dir.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "_approved").mkdir(exist_ok=True)
    (OUT_DIR / "_flagged").mkdir(exist_ok=True)
    (OUT_DIR / "_state").mkdir(exist_ok=True)

    picked = collect(args.source, args.limit)
    print(f"พบ transcript ที่มีสาระ {len(picked)} ไฟล์ (source={args.source})")
    for i, path in enumerate(picked, 1):
        s = read_session(path)
        digest = new_dir / f"digest-{args.source}-{i:02d}.md"
        lines = [
            f"# Digest {i} · {args.source}",
            f"- file: {s['file']}",
            f"- project: {s['project']}",
            f"- messages (human text): {s['msg_count']}",
            "",
            "## บทสนทนาย่อ (ข้อความคนเท่านั้น)",
            "",
        ]
        for role, text in s["messages"]:
            snippet = text if len(text) <= 600 else text[:600] + " …"
            lines.append(f"**{role}:** {snippet}\n")
        digest.write_text("\n".join(lines), encoding="utf-8")
        print(f"  [{i}] {s['project']} · {s['msg_count']} msgs → {digest.name}")


if __name__ == "__main__":
    main()
