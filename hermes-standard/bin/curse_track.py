#!/usr/bin/env python3
"""
curse_track — วงจรเรียนรู้จากคำด่า (คลื่น 4 · แกนหลัก)
ถือคำด่าเป็น shortcut สั่งให้ระบบจับ → แยกเป้า → ทะเบียนปัญหา → เตือนเมื่อซ้ำเกินเกณฑ์ → ตาราง HTML

  python curse_track.py log "<ข้อความ>"   [--data DIR]
  python curse_track.py report            [--data DIR] [--html PATH]

ข้อมูลเก็บที่ DIR (ค่าเริ่มต้น learning/data ข้าง repo):
  events.jsonl  = ทุกเหตุการณ์ด่า   ·   issues.json = ทะเบียนปัญหา (วงจรปิด)
"""
import os
import sys
import json
import argparse
from datetime import datetime

STD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KW_FILE = os.path.join(STD_ROOT, "learning", "curse-keywords.json")


def load_kw():
    with open(KW_FILE, encoding="utf-8") as f:
        return json.load(f)


def detect(msg, kw):
    low = msg.lower()
    hit = [k for k in kw["keywords"] if k.lower() in low]
    if not hit:
        return None
    target = next((t for t in kw["targets"] if t in low), "ai")
    return {"keywords": hit, "target": target}


def load_issues(data):
    p = os.path.join(data, "issues.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def cmd_log(msg, data, kw):
    d = detect(msg, kw)
    if not d:
        print("ไม่พบคำด่า · ไม่บันทึก")
        return
    ts = datetime.now().isoformat(timespec="seconds")
    # 1) เก็บเหตุการณ์
    with open(os.path.join(data, "events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": ts, "msg": msg, "target": d["target"],
                            "keywords": d["keywords"]}, ensure_ascii=False) + "\n")
    # 2) อัปเดตทะเบียนปัญหา (fingerprint = เป้าที่ด่า)
    issues = load_issues(data)
    fp = d["target"]
    cur = next((i for i in issues if i["fingerprint"] == fp), None)
    if cur is None:
        cur = {"id": "ISS-%s" % (len(issues) + 1), "fingerprint": fp,
               "target_blamed": d["target"], "root_target": "",
               "category": "ยังไม่จัด", "count": 0, "status": "open",
               "fix_count": 0, "repeat_after_fix": 0, "last_seen": ts}
        issues.append(cur)
    cur["count"] += 1
    cur["last_seen"] = ts
    if cur["status"] in ("fixed", "verified"):
        cur["status"] = "open"
        cur["repeat_after_fix"] += 1
        print("เตือน: เรื่องเดิม (%s) ที่เคยแก้แล้ว ถูกด่าซ้ำ → root อาจยังไม่ถูก" % cur["id"])
    save_json(os.path.join(data, "issues.json"), issues)
    thr = kw.get("repeat_warn_threshold", 3)
    flag = "  ** เกินเกณฑ์ %d ครั้ง ต้องหา root **" % thr if cur["count"] >= thr else ""
    print("บันทึก: เป้า=%s ครั้งที่=%d สถานะ=%s%s" % (cur["target_blamed"], cur["count"], cur["status"], flag))


def cmd_report(data, html, kw):
    issues = load_issues(data)
    thr = kw.get("repeat_warn_threshold", 3)
    print("== ทะเบียนปัญหาจากคำด่า ==")
    for i in sorted(issues, key=lambda x: -x["count"]):
        warn = " [เกินเกณฑ์]" if i["count"] >= thr and i["status"] not in ("fixed", "verified") else ""
        print("- %s เป้า=%s ด่า=%d ครั้ง สถานะ=%s แก้ไป=%d ด่าซ้ำหลังแก้=%d%s"
              % (i["id"], i["target_blamed"], i["count"], i["status"],
                 i["fix_count"], i["repeat_after_fix"], warn))
    if html:
        rows = []
        for i in sorted(issues, key=lambda x: -x["count"]):
            over = i["count"] >= thr and i["status"] not in ("fixed", "verified")
            rows.append("<tr%s><td>%s</td><td>%s</td><td>%d</td><td>%s</td><td>%s</td><td>%d</td><td>%d</td></tr>"
                        % (" style='background:#fce8e8'" if over else "",
                           i["id"], i["target_blamed"], i["count"], i["category"],
                           i["status"], i["fix_count"], i["repeat_after_fix"]))
        out = ("<!doctype html><meta charset='utf-8'><title>Curse Tracker</title>"
               "<style>body{font-family:sans-serif;padding:20px}table{border-collapse:collapse;width:100%%}"
               "td,th{border:1px solid #ccc;padding:8px;text-align:left}</style>"
               "<h2>ทะเบียนปัญหาจากคำด่า (วงจรปิด)</h2><table>"
               "<tr><th>id</th><th>เป้าที่ด่า</th><th>ด่ากี่ครั้ง</th><th>หมวด</th>"
               "<th>สถานะ</th><th>แก้กี่ครั้ง</th><th>ด่าซ้ำหลังแก้</th></tr>%s</table>"
               % "".join(rows))
        save_json_text(html, out)
        print("เขียน HTML: %s" % html)


def save_json_text(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def cmd_promote(issue_id, root_text, data):
    """ยืนยัน root ที่แก้จริงแล้ว → status=verified + เขียนโน้ตถาวร (req-42 · ผ่านการรีวิวคือการรันคำสั่งนี้)"""
    issues = load_issues(data)
    cur = next((i for i in issues if i["id"] == issue_id), None)
    if cur is None:
        print("ไม่พบ %s" % issue_id)
        return
    if not root_text:
        print("ต้องใส่ --root \"<สาเหตุจริงที่ยืนยันแล้ว>\"")
        return
    from datetime import datetime
    cur["status"] = "verified"
    cur["root_cause"] = root_text
    cur["fix_count"] = cur.get("fix_count", 0) + 1
    save_json(os.path.join(data, "issues.json"), issues)
    rdir = os.path.join(data, "roots")
    os.makedirs(rdir, exist_ok=True)
    note = os.path.join(rdir, "%s.md" % issue_id)
    save_json_text(note, "# %s · root ที่ยืนยันแล้ว (%s)\nเป้า: %s\nสาเหตุจริง: %s\n"
                   % (issue_id, datetime.now().isoformat(timespec="seconds"), cur["target_blamed"], root_text))
    print("ยืนยัน %s = verified · เขียนโน้ตถาวร: %s" % (issue_id, note))
    print("เอาเข้าความจำกลาง (~/.claude/memory/user-facts/) = งานเจ้าของ (ติดด่าน relay)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["log", "report", "promote"])
    ap.add_argument("message", nargs="?", default="")
    ap.add_argument("--data", default=os.path.join(STD_ROOT, "learning", "data"))
    ap.add_argument("--html")
    ap.add_argument("--root", default="")
    args = ap.parse_args()
    os.makedirs(args.data, exist_ok=True)
    kw = load_kw()
    if args.command == "log":
        cmd_log(args.message, args.data, kw)
    elif args.command == "promote":
        cmd_promote(args.message, args.root, args.data)
    else:
        cmd_report(args.data, args.html, kw)


if __name__ == "__main__":
    main()
