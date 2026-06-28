#!/usr/bin/env python3
"""
hermes_analyze — สมองวิเคราะห์คำด่าเป็นรอบ (req-43 · "ทุก 3 ชม.")
อ่านทะเบียนปัญหา → จัดลำดับความร้อน → ชี้ตัวที่ "แก้แล้วแต่ด่าซ้ำ" (root ผิด) → บอกตัวที่ต้องเอา cross-check

  python hermes_analyze.py --data <dir> [--json out.json]
"""
import os
import sys
import json
import argparse


def load(data):
    p = os.path.join(data, "issues.json")
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--threshold", type=int, default=3)
    ap.add_argument("--json")
    args = ap.parse_args()

    issues = load(args.data)
    report = []
    for i in issues:
        open_now = i["status"] not in ("fixed", "verified")
        hot = open_now and i["count"] >= args.threshold
        root_wrong = i.get("repeat_after_fix", 0) > 0  # เคยแก้แล้วยังด่าซ้ำ = แก้ไม่ตรงราก
        need_xc = root_wrong or (hot and not i.get("root_target"))
        prio = "HIGH" if (root_wrong or hot) else ("MED" if open_now else "LOW")
        report.append({
            "id": i["id"], "target": i["target_blamed"], "count": i["count"],
            "status": i["status"], "priority": prio,
            "root_wrong": root_wrong, "need_crosscheck": need_xc,
        })

    report.sort(key=lambda x: {"HIGH": 0, "MED": 1, "LOW": 2}[x["priority"]])
    print("== วิเคราะห์คำด่า (เกณฑ์ซ้ำ %d) ==" % args.threshold)
    for r in report:
        flags = []
        if r["root_wrong"]:
            flags.append("แก้แล้วยังด่าซ้ำ→root ผิด")
        if r["need_crosscheck"]:
            flags.append("ควรเอา cross-check หา root")
        print("  [%s] %s เป้า=%s ด่า=%d %s"
              % (r["priority"], r["id"], r["target"], r["count"],
                 ("· " + " · ".join(flags)) if flags else ""))
    hi = [r for r in report if r["priority"] == "HIGH"]
    xc = [r["id"] for r in report if r["need_crosscheck"]]
    print("สรุป: ร้อน(HIGH) %d เรื่อง · ต้อง cross-check: %s" % (len(hi), xc or "ไม่มี"))

    if args.json:
        json.dump(report, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
