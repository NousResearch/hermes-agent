#!/usr/bin/env python3
"""
wire_agents — ผูก agent ทุกตัวให้อ่านมาตรฐานกลาง (req-1 · ปลายทาง)
เติมบรรทัดอ้างอิงมาตรฐานกลางใน agent .md ทุกตัว · รันซ้ำได้ ไม่เติมซ้ำ (idempotent)

  python wire_agents.py <agents_dir> [--dry-run]
"""
import os
import sys

MARK = "<!-- hermes-standard:agent-bound -->"
BLOCK = (MARK + "\n"
         "> ก่อนทำงาน: อ่านมาตรฐานกลาง `hermes-standard/rules/central-block.md` "
         "(ด่านตรวจความจริง/ขอบเขต/ห้ามเดา/ภาษา/กันลืม) แล้วทำตามทุกข้อ\n")


def main():
    if len(sys.argv) < 2:
        print("ใช้: wire_agents.py <agents_dir> [--dry-run]")
        sys.exit(1)
    d = os.path.abspath(sys.argv[1])
    dry = "--dry-run" in sys.argv
    if not os.path.isdir(d):
        print("ไม่พบโฟลเดอร์: %s" % d)
        sys.exit(1)

    files = []
    for root, _, fns in os.walk(d):
        for fn in fns:
            if fn.endswith(".md"):
                files.append(os.path.join(root, fn))

    wired, already = 0, 0
    for fp in sorted(files):
        s = open(fp, encoding="utf-8").read()
        if MARK in s:
            already += 1
            continue
        if not dry:
            open(fp, "w", encoding="utf-8").write(BLOCK + "\n" + s)
        wired += 1

    print("== wire_agents%s ==" % (" (dry-run)" if dry else ""))
    print("ผูกใหม่: %d · มีอยู่แล้ว: %d · รวม agent: %d" % (wired, already, len(files)))


if __name__ == "__main__":
    main()
