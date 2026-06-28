#!/usr/bin/env python3
"""
safe_apply — แจก/รับมาตรฐานแบบปลอดภัย กัน project พัง (req-66)

ลำดับ: รันเกณฑ์ของ project (ก่อน) → สำรองไฟล์ → ทำงาน(sync หรือ init) → รันเกณฑ์ (หลัง)
       → ถ้าก่อนผ่านแต่หลังพัง = ถอยกลับทันที (คืนไฟล์เดิม + ลบไฟล์ที่เพิ่งสร้าง)

  python safe_apply.py <project_dir>            # โหมด sync (อัปเดตโซนกลาง)
  python safe_apply.py <project_dir> --init     # โหมด init/onboard (รับ project เก่าเข้าระบบ)
  python safe_apply.py <project_dir> --dry-run

เกณฑ์ของ project (ตามลำดับ): hermes.gate (คำสั่ง 1 บรรทัด) > Makefile(test:) > package.json("test") > pytest
"""
import os
import sys
import subprocess

BIN = os.path.dirname(os.path.abspath(__file__))
SYNC = os.path.join(BIN, "hermes_std.py")
MIXED = ["CLAUDE.md", "AGENTS.md", "GEMINI.md", "QWEN.md", ".cursorrules", "cross-code.md",
         os.path.join(".cursor", "rules", "hermes-central.mdc")]
PROJECT_ONLY = ["OverviewProgress.md", "BusinessPlan.md", "FeatureSpec.md",
                "DesignSystem.md", "hermes.project.yaml"]


def detect_gate(p):
    g = os.path.join(p, "hermes.gate")
    if os.path.isfile(g):
        cmd = open(g, encoding="utf-8").read().strip()
        if cmd:
            return cmd
    mk = os.path.join(p, "Makefile")
    if os.path.isfile(mk) and "test:" in open(mk, encoding="utf-8", errors="replace").read():
        return "make test"
    pj = os.path.join(p, "package.json")
    if os.path.isfile(pj) and '"test"' in open(pj, encoding="utf-8", errors="replace").read():
        return "npm test"
    if os.path.isfile(os.path.join(p, "pyproject.toml")) or os.path.isfile(os.path.join(p, "pytest.ini")):
        return "pytest -q"
    return None


def run_gate(cmd, cwd):
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=300)
        return r.returncode == 0, (r.stdout + r.stderr)[-500:]
    except Exception as e:
        return False, "รันเกณฑ์ไม่ได้: %s" % e


def snapshot(p, files):
    """เก็บสภาพก่อน: path -> (existed, content)"""
    snap = {}
    for rel in files:
        fp = os.path.join(p, rel)
        if os.path.isfile(fp):
            snap[rel] = (True, open(fp, encoding="utf-8").read())
        else:
            snap[rel] = (False, None)
    return snap


def restore(p, snap):
    for rel, (existed, content) in snap.items():
        fp = os.path.join(p, rel)
        if existed:
            os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
            open(fp, "w", encoding="utf-8").write(content)
        elif os.path.isfile(fp):
            os.remove(fp)


def main():
    args = sys.argv[1:]
    init_mode = "--init" in args
    dry = "--dry-run" in args
    pos = [a for a in args if not a.startswith("--")]
    p = os.path.abspath(pos[0]) if pos else "."
    if not os.path.isdir(p) and not init_mode:
        print("ไม่พบโฟลเดอร์: %s" % p)
        sys.exit(1)

    cmd = "init" if init_mode else "sync"
    gate = detect_gate(p)
    print("== safe_apply (%s): %s ==" % (cmd, p))
    print("เกณฑ์ project: %s" % (gate or "ไม่พบ (วัดไม่ได้)"))

    base_ok = None
    if gate:
        base_ok, _ = run_gate(gate, p)
        print("เกณฑ์ก่อน: %s" % ("ผ่าน" if base_ok else "ไม่ผ่าน"))

    if dry:
        if not init_mode:
            subprocess.run([sys.executable, SYNC, "sync", p, "--dry-run"])
        print("(dry-run · ไม่เขียนจริง)")
        return

    files = MIXED + PROJECT_ONLY if init_mode else MIXED
    snap = snapshot(p, files)
    subprocess.run([sys.executable, SYNC, cmd, p])

    if gate and base_ok:
        after_ok, after_out = run_gate(gate, p)
        print("เกณฑ์หลัง: %s" % ("ผ่าน" if after_ok else "ไม่ผ่าน"))
        if not after_ok:
            restore(p, snap)
            print(">> sync/init ทำผลแย่ลง → ถอยกลับแล้ว · สถานะ: BROKE_ROLLED_BACK")
            print(">> output: %s" % after_out.strip()[:200])
            print("RESULT=BROKE_ROLLED_BACK")
            sys.exit(2)
        print(">> สถานะ: APPLIED_OK")
        print("RESULT=APPLIED_OK")
    elif gate and not base_ok:
        print(">> สถานะ: APPLIED_NO_BASELINE (project แดงอยู่ก่อนแล้ว ไม่ใช่ความผิด sync · ควรแก้ project ก่อน)")
        print("RESULT=APPLIED_NO_BASELINE")
    else:
        print(">> สถานะ: APPLIED_UNVERIFIED (ไม่มีเกณฑ์ให้วัด · ยังยืนยันไม่ได้ว่าไม่พัง)")
        print("RESULT=APPLIED_UNVERIFIED")


if __name__ == "__main__":
    main()
