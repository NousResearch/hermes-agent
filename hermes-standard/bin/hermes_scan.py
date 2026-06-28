#!/usr/bin/env python3
"""
hermes_scan — ตัวเฝ้าศูนย์กลาง (คลื่น 3)
สแกนทุก project หา: ไฟล์เกินขนาด, ไฟล์มาตรฐานหาย, โซนกลางถูกแก้/รุ่นไม่ตรง

  python hermes_scan.py <root>            # สแกนทุกโฟลเดอร์ย่อยที่มี hermes.project.yaml
  python hermes_scan.py <project> --one   # สแกน project เดียว
  --max-mb N   เกณฑ์ขนาดไฟล์ (ค่าเริ่มต้น 100)
  --html PATH  เขียนรายงาน HTML
"""
import os
import sys
import json
import argparse

STD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CENTRAL_FILE = os.path.join(STD_ROOT, "rules", "central-block.md")
START, END = "HERMES-CENTRAL", "/HERMES-CENTRAL"
MIXED = ["CLAUDE.md", "AGENTS.md", "GEMINI.md", "QWEN.md", ".cursorrules", "cross-code.md",
         os.path.join(".cursor", "rules", "hermes-central.mdc")]
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


def read(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_zone(content):
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
    return "\n".join(lines[s:e + 1]).strip()


def is_project(d):
    return os.path.isfile(os.path.join(d, "hermes.project.yaml"))


def required_files(project):
    """อ่าน required_files จาก hermes.project.yaml แบบง่าย (ไม่พึ่ง lib ภายนอก)"""
    out = []
    path = os.path.join(project, "hermes.project.yaml")
    if not os.path.isfile(path):
        return out
    grab = False
    for line in read(path).splitlines():
        if line.startswith("required_files:"):
            grab = True
            continue
        if grab:
            st = line.strip()
            if st.startswith("- "):
                out.append(st[2:].strip())
            elif st and not line.startswith(" "):
                break
    return out


def scan_project(project, max_mb, canonical_zone):
    issues = []
    # 1) ไฟล์มาตรฐานหาย
    for rel in required_files(project):
        if not os.path.exists(os.path.join(project, rel)):
            issues.append(("HIGH", "ไฟล์มาตรฐานหาย: %s" % rel))
    # 2) ไฟล์เกินขนาด
    limit = max_mb * 1024 * 1024
    for root, dirs, files in os.walk(project):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            if sz > limit:
                rel = os.path.relpath(fp, project)
                issues.append(("MEDIUM", "ไฟล์เกิน %sMB: %s (%.1f MB)" % (max_mb, rel, sz / 1024 / 1024)))
    # 3) โซนกลางถูกแก้/รุ่นไม่ตรง
    for rel in MIXED:
        fp = os.path.join(project, rel)
        if not os.path.exists(fp):
            continue
        z = extract_zone(read(fp))
        if z is None:
            issues.append(("HIGH", "โซนกลางหาย/เส้นคั่นเสีย: %s" % rel))
        elif z != canonical_zone:
            issues.append(("HIGH", "โซนกลางถูกแก้/รุ่นไม่ตรงกับส่วนกลาง: %s" % rel))
    if any(s == "HIGH" for s, _ in issues):
        status = "BROKEN"
    elif issues:
        status = "DEGRADED"
    else:
        status = "HEALTHY"
    return status, issues


def html_report(results, max_mb):
    rows = []
    for proj, status, issues in results:
        color = {"HEALTHY": "#1d9e75", "DEGRADED": "#ba7517", "BROKEN": "#a32d2d"}[status]
        det = "<br>".join("[%s] %s" % (s, m) for s, m in issues) or "ไม่พบปัญหา"
        rows.append("<tr><td>%s</td><td style='color:%s;font-weight:600'>%s</td><td>%s</td></tr>"
                    % (proj, color, status, det))
    return ("<!doctype html><meta charset='utf-8'><title>Hermes Scan</title>"
            "<style>body{font-family:sans-serif;padding:20px}table{border-collapse:collapse;width:100%%}"
            "td,th{border:1px solid #ccc;padding:8px;text-align:left;vertical-align:top}</style>"
            "<h2>รายงานสแกนสุขภาพ project (เกณฑ์ %sMB)</h2><table>"
            "<tr><th>project</th><th>สถานะ</th><th>รายละเอียด</th></tr>%s</table>"
            % (max_mb, "".join(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--one", action="store_true")
    ap.add_argument("--max-mb", type=float, default=100)
    ap.add_argument("--html")
    ap.add_argument("--json")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    canonical_zone = extract_zone(read(CENTRAL_FILE))

    if args.one:
        projects = [root]
    else:
        projects = sorted(os.path.join(root, d) for d in os.listdir(root)
                          if os.path.isdir(os.path.join(root, d)) and is_project(os.path.join(root, d)))

    results = []
    for p in projects:
        status, issues = scan_project(p, args.max_mb, canonical_zone)
        results.append((os.path.basename(p.rstrip("/")), status, issues))

    print("== hermes scan (เกณฑ์ %sMB) ==" % args.max_mb)
    if not results:
        print("ไม่พบ project (ต้องมี hermes.project.yaml)")
    for proj, status, issues in results:
        print("- %s : %s" % (proj, status))
        for s, m in issues:
            print("    [%s] %s" % (s, m))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([{"project": p, "status": s, "issues": i} for p, s, i in results],
                      f, ensure_ascii=False, indent=2)
    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(html_report(results, args.max_mb))
        print("เขียน HTML: %s" % args.html)


if __name__ == "__main__":
    main()
