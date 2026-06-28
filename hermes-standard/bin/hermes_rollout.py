#!/usr/bin/env python3
"""
hermes_rollout — แจกมาตรฐานลงหลาย project ทีเดียว (รองรับ 30-40 project)

อ่านรายชื่อ project จากไฟล์ทะเบียน (ค่าเริ่มต้น hermes-standard/projects.txt · 1 path ต่อบรรทัด)
แล้วรัน safe_apply ให้ทุกตัว (init = รับเข้าระบบ · sync = อัปเดตโซนกลาง) · กันพังด้วย safe_apply ในตัว

  python hermes_rollout.py            # sync ทุก project ในทะเบียน
  python hermes_rollout.py --init     # init/onboard ทุก project
  python hermes_rollout.py --list path1 path2   # ระบุเองไม่ใช้ทะเบียน
"""
import os
import sys
import subprocess

BIN = os.path.dirname(os.path.abspath(__file__))
STD_ROOT = os.path.dirname(BIN)
SAFE = os.path.join(BIN, "safe_apply.py")
REGISTRY = os.path.join(STD_ROOT, "projects.txt")


def load_registry():
    if not os.path.isfile(REGISTRY):
        return []
    out = []
    for line in open(REGISTRY, encoding="utf-8"):
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(os.path.expanduser(s))
    return out


def main():
    args = sys.argv[1:]
    init = "--init" in args
    projects = []
    if "--list" in args:
        i = args.index("--list")
        projects = []
        for a in args[i + 1:]:
            if a.startswith("-"):
                break
            projects.append(os.path.abspath(a))
    else:
        projects = load_registry()

    if not projects:
        print("ไม่มี project ให้แจก · ใส่ path ใน %s หรือใช้ --list" % REGISTRY)
        sys.exit(1)

    results = []
    for p in projects:
        if not os.path.isdir(p) and not init:
            results.append((p, "NOT_FOUND"))
            continue
        cmd = [sys.executable, SAFE, p] + (["--init"] if init else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        res = [l for l in r.stdout.splitlines() if l.startswith("RESULT=")]
        status = res[-1].split("=", 1)[1].strip() if res else ("rc=%d" % r.returncode)
        results.append((p, status))

    print("== hermes rollout (%s) · %d project ==" % ("init" if init else "sync", len(projects)))
    ok = 0
    for p, s in results:
        mark = "✅" if s.startswith("APPLIED_OK") else ("⚠️" if "UNVERIFIED" in s or "NO_BASELINE" in s else "❌")
        if mark == "✅":
            ok += 1
        print("  %s %s — %s" % (mark, os.path.basename(p.rstrip("/")), s))
    print("สรุป: ผ่านเขียว %d / %d" % (ok, len(results)))

    if "--json" in args:
        import json
        jp = args[args.index("--json") + 1]
        json.dump([{"project": p, "status": s} for p, s in results],
                  open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("เขียน JSON: %s" % jp)


if __name__ == "__main__":
    main()
