#!/usr/bin/env python3
"""
hermes_std — ตัวแจกจ่ายมาตรฐานกลาง Hermes (คลื่น 1)

คำสั่ง:
  python hermes_std.py sync [โฟลเดอร์ project]   # เขียนทับเฉพาะโซนกลาง ไม่แตะโซน project
  python hermes_std.py init [โฟลเดอร์ project]   # เปิด project ใหม่เข้ามาตรฐาน (รันซ้ำได้ ไม่ทับงานคน)

หลักการ:
  - ไฟล์ผสม มี 2 โซน คั่นด้วย HERMES-CENTRAL ... /HERMES-CENTRAL
  - sync แทนที่เฉพาะโซนกลาง · โซน project ใต้เส้นไม่แตะ
  - init สร้างไฟล์ที่ยังไม่มีจากแม่แบบ · ไฟล์ที่มีแล้วแค่ sync โซนกลาง · ไม่ทับไฟล์เฉพาะ project ที่มีอยู่
"""
import sys
import os
import argparse
from datetime import date

STD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # hermes-standard/
CENTRAL_FILE = os.path.join(STD_ROOT, "rules", "central-block.md")
TEMPLATES = os.path.join(STD_ROOT, "templates")

START = "HERMES-CENTRAL"
END = "/HERMES-CENTRAL"

# ไฟล์ผสม (มี 2 โซน · sync โซนกลางได้)
MIXED = [
    "CLAUDE.md", "AGENTS.md", "GEMINI.md", "QWEN.md",
    ".cursorrules", "cross-code.md",
    os.path.join(".cursor", "rules", "hermes-central.mdc"),
]
# ไฟล์เฉพาะ project (สร้างถ้ายังไม่มี · ไม่ sync ไม่ทับ)
PROJECT_ONLY = [
    "OverviewProgress.md", "BusinessPlan.md", "FeatureSpec.md",
    "DesignSystem.md", "hermes.project.yaml",
]


PROJECT_MARKER = "<!-- ▼▼▼ PROJECT · โซนของคุณ · แก้ได้อิสระ · sync ไม่แตะ ▼▼▼ -->"


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write(path, content):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def central_block():
    return read(CENTRAL_FILE).strip("\n")


def replace_zone(content, new_central):
    """แทนที่ช่วง HERMES-CENTRAL../HERMES-CENTRAL ด้วยโซนกลางใหม่ · คืน None ถ้าไม่เจอเส้นคั่น"""
    lines = content.splitlines()
    s = e = None
    for i, l in enumerate(lines):
        if START in l and s is None:
            s = i
        elif END in l and s is not None:
            e = i
            break
    if s is None or e is None or e < s:
        return None
    new_lines = lines[:s] + new_central.splitlines() + lines[e + 1:]
    out = "\n".join(new_lines)
    if content.endswith("\n"):
        out += "\n"
    return out


def cmd_sync(project_dir, dry_run=False):
    cb = central_block()
    changed, skipped, missing = [], [], []
    for rel in MIXED:
        path = os.path.join(project_dir, rel)
        if not os.path.exists(path):
            missing.append(rel)
            continue
        old = read(path)
        new = replace_zone(old, cb)
        if new is None:
            skipped.append(rel + " (ไม่เจอเส้นโซนกลาง · ไม่แตะ เพื่อกันทับงานคน)")
            continue
        if new != old:
            if not dry_run:
                write(path, new)
            changed.append(rel)
    return changed, skipped, missing


def cmd_init(project_dir):
    created, synced, kept = [], [], []
    cb = central_block()
    # ไฟล์ผสม
    for rel in MIXED:
        path = os.path.join(project_dir, rel)
        tpl = os.path.join(TEMPLATES, rel)
        if not os.path.exists(path):
            if os.path.exists(tpl):
                write(path, read(tpl))
                created.append(rel)
        else:
            old = read(path)
            new = replace_zone(old, cb)
            if new is not None:
                if new != old:
                    write(path, new)
                    synced.append(rel)
                else:
                    kept.append(rel)
            else:
                # ไฟล์เก่าที่ยังไม่มีโซนกลาง → onboard: ใส่โซนกลางไว้บน เก็บของเดิมเป็นโซน project
                onboarded = cb + "\n\n\n" + PROJECT_MARKER + "\n" + old
                write(path, onboarded)
                synced.append(rel + " (onboard ของเก่า)")
    # ไฟล์เฉพาะ project — สร้างถ้ายังไม่มี ไม่ทับ
    for rel in PROJECT_ONLY:
        path = os.path.join(project_dir, rel)
        tpl = os.path.join(TEMPLATES, rel)
        if not os.path.exists(path):
            if os.path.exists(tpl):
                content = read(tpl)
                if rel == "hermes.project.yaml":
                    content = content.replace('last_sync: ""', 'last_sync: "%s"' % date.today().isoformat())
                write(path, content)
                created.append(rel)
        else:
            kept.append(rel + " (มีอยู่แล้ว · ไม่ทับ)")
    return created, synced, kept


def main():
    ap = argparse.ArgumentParser(description="ตัวแจกจ่ายมาตรฐานกลาง Hermes")
    ap.add_argument("command", choices=["sync", "init"])
    ap.add_argument("project_dir", nargs="?", default=".")
    ap.add_argument("--dry-run", action="store_true", help="ลองดูว่าจะเปลี่ยนอะไร ไม่เขียนจริง")
    args = ap.parse_args()

    pdir = os.path.abspath(args.project_dir)
    if args.command == "init":
        os.makedirs(pdir, exist_ok=True)
    elif not os.path.isdir(pdir):
        print("ไม่พบโฟลเดอร์: %s" % pdir)
        sys.exit(1)

    if args.command == "sync":
        changed, skipped, missing = cmd_sync(pdir, args.dry_run)
        tag = "(ลองดู)" if args.dry_run else ""
        print("== hermes sync %s ==" % tag)
        print("อัปเดตโซนกลาง: %d ไฟล์ %s" % (len(changed), changed or ""))
        if skipped:
            print("ข้าม: %s" % skipped)
        if missing:
            print("ยังไม่มีในโปรเจกต์ (ใช้ init สร้าง): %s" % missing)
    else:
        created, synced, kept = cmd_init(pdir)
        print("== hermes init ==")
        print("สร้างใหม่: %d ไฟล์ %s" % (len(created), created or ""))
        print("อัปเดตโซนกลาง: %d ไฟล์ %s" % (len(synced), synced or ""))
        if kept:
            print("คงไว้ (ไม่ทับ): %s" % kept)


if __name__ == "__main__":
    main()
