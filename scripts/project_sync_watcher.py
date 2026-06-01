#!/usr/bin/env python3
"""ตัวเฝ้าดูการอัปเดตข้ามโปรเจกต์ (เฟส 7 · ระบบ 360)

ถ้าโปรเจกต์ไหนมี commit ใหม่ จะตรวจเจอและชี้ว่าไฟล์บริบทไหนควร sync ตาม
(ขั้นสรุป diff ด้วย AI = ต่อ LLM ภายหลัง · ตัวเฝ้าดูนี้ตรวจ + ชี้เป้าได้เลย)

ใช้งาน:  python scripts/project_sync_watcher.py
exit 0 = ไม่มีอะไรใหม่ · exit 10 = มีโปรเจกต์อัปเดต (ควร sync)
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

BASE = Path("/Users/rattanasak/Documents/Viber Project")
STATE_FILE = Path(__file__).resolve().parent.parent / "docs/360-content-system/_sync-state.json"

# โปรเจกต์ในระบบ 360 + ไฟล์บริบทที่ควร sync เมื่อโปรเจกต์นั้นเปลี่ยน
PROJECTS = {
    "hermes-agent": {
        "path": BASE / "Tech Tools/Hermes Agent",
        "sync_targets": ["docs/360-content-system/project-registry.md", "memory/MEMORY.md"],
    },
    "idea2logic": {
        "path": BASE / "Private Project/Idea2Logic",
        "sync_targets": ["docs/360-content-system/project-registry.md"],
    },
    "content-factory": {
        "path": BASE / "SaaS Project/Master Content Factory",
        "sync_targets": ["docs/360-content-system/project-registry.md"],
    },
}


def last_commit(path: Path) -> dict | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "log", "-1", "--format=%H|%ci|%s"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        h, when, subject = out.stdout.strip().split("|", 2)
        return {"hash": h, "when": when, "subject": subject}
    except Exception:
        return None


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    state = load_state()
    new_state = {}
    changed = []

    for name, cfg in PROJECTS.items():
        commit = last_commit(cfg["path"])
        if not commit:
            print(f"[{name}] ข้ามไม่ได้ (ไม่ใช่ git repo หรือเข้าไม่ถึง)")
            continue
        new_state[name] = commit["hash"]
        prev = state.get(name)
        if prev != commit["hash"]:
            changed.append((name, commit, cfg["sync_targets"], prev is None))

    if not changed:
        print("ไม่มีโปรเจกต์อัปเดตใหม่ — ทุกอย่าง sync แล้ว")
        STATE_FILE.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
        return 0

    print(f"พบ {len(changed)} โปรเจกต์มีการเปลี่ยนแปลง:")
    for name, commit, targets, first in changed:
        tag = "(ครั้งแรก)" if first else "(commit ใหม่)"
        print(f"\n  ▶ {name} {tag}")
        print(f"    commit: {commit['subject']}  [{commit['when'][:10]}]")
        print(f"    ควร sync ไฟล์บริบท: {', '.join(targets)}")
    STATE_FILE.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    return 10


if __name__ == "__main__":
    raise SystemExit(main())
