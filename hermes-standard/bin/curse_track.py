#!/usr/bin/env python3
"""
curse_track — วงจรเรียนรู้จากคำด่า (คลื่น 4 · แกนหลัก)
ถือคำด่าเป็น shortcut สั่งให้ระบบจับ → แยกเป้า → ทะเบียนปัญหา → เตือนเมื่อซ้ำเกินเกณฑ์ → ตาราง HTML

  python curse_track.py log "<ข้อความ>"   [--data DIR]
  python curse_track.py ingest --from PATH [--data DIR]
  python curse_track.py report            [--data DIR] [--html PATH]

ข้อมูลเก็บที่ DIR (ค่าเริ่มต้น learning/data ข้าง repo):
  events.jsonl  = ทุกเหตุการณ์ด่า   ·   issues.json = ทะเบียนปัญหา (วงจรปิด)
"""
import os
import sys
import json
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from html import escape

STD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KW_FILE = os.path.join(STD_ROOT, "learning", "curse-keywords.json")


def load_kw():
    with open(KW_FILE, encoding="utf-8") as f:
        return json.load(f)


def _as_clean_list(value):
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = str(item).strip().lower()
        if text:
            cleaned.append(text)
    return cleaned


def _merge_target_phrase(targets, name, phrase):
    name = str(name).strip().lower()
    phrase = str(phrase).strip().lower()
    if not name or not phrase:
        return
    targets.setdefault(name, [])
    if phrase not in targets[name]:
        targets[name].append(phrase)


def normalize_keywords(data):
    """ตรรกะเดียวกับ learning/hooks/ai-fail-stats-v2.py (CT-I1)."""
    if not isinstance(data, dict):
        data = {}

    disabled = set(_as_clean_list(data.get("disabled")))
    targets = {}
    raw_targets = data.get("targets")

    if isinstance(raw_targets, dict):
        for name, phrases in raw_targets.items():
            target_name = str(name).strip().lower()
            for phrase in _as_clean_list(phrases):
                if phrase not in disabled:
                    _merge_target_phrase(targets, target_name, phrase)
    elif isinstance(raw_targets, list):
        for name in raw_targets:
            target_name = str(name).strip().lower()
            if target_name == "ai":
                target_name = "claude"
            for prefix in ("fuck you", "fuck u", "f u"):
                phrase = "%s %s" % (prefix, str(name).strip().lower())
                if phrase not in disabled:
                    _merge_target_phrase(targets, target_name, phrase)

    generic_source = data.get("generic_curse")
    if generic_source is None:
        generic_source = data.get("keywords")
    generic_curse = [
        phrase for phrase in _as_clean_list(generic_source) if phrase not in disabled
    ]
    jargon_markers = [
        phrase for phrase in _as_clean_list(data.get("jargon_markers")) if phrase not in disabled
    ]

    try:
        threshold = int(data.get("repeat_warn_threshold", 3))
    except Exception:
        threshold = 3

    return {
        "version": 2,
        "repeat_warn_threshold": threshold,
        "targets": targets,
        "generic_curse": generic_curse,
        "jargon_markers": jargon_markers,
        "disabled": sorted(disabled),
    }


def _target_category(target):
    if target == "hermes":
        return "hermes-fail"
    if target == "claude":
        return "ai-fail"
    return "target:%s" % target


def detect_hits(msg, kw):
    low = (msg or "").lower()
    book = normalize_keywords(kw)
    hits = []

    curse_hit = None
    for target, phrases in book.get("targets", {}).items():
        for phrase in phrases:
            if phrase and phrase in low:
                curse_hit = {
                    "category": _target_category(target),
                    "phrase": phrase,
                    "target": target,
                    "keywords": [phrase],
                }
                break
        if curse_hit:
            break

    if curse_hit is None:
        for phrase in book.get("generic_curse", []):
            if phrase and phrase in low:
                curse_hit = {
                    "category": "curse-generic",
                    "phrase": phrase,
                    "target": "-",
                    "keywords": [phrase],
                }
                break

    if curse_hit:
        hits.append(curse_hit)
        return hits

    for phrase in book.get("jargon_markers", []):
        if phrase and phrase in low:
            hits.append({
                "category": "jargon",
                "phrase": phrase,
                "target": "-",
                "keywords": [phrase],
            })
            break

    return hits


def detect(msg, kw):
    hits = detect_hits(msg, kw)
    if not hits:
        return None
    return hits[0]


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


def save_json_text(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _fingerprint_for_event(event):
    fp = str(event.get("fingerprint") or "").strip()
    if fp:
        return fp
    target = str(event.get("target") or "").strip()
    if target and target != "-":
        return target
    category = str(event.get("category") or "").strip()
    return category or "unknown"


def _target_for_issue(event):
    target = str(event.get("target") or "").strip()
    if target and target != "-":
        return target
    category = str(event.get("category") or "").strip()
    return category or "unknown"


def _category_for_issue(event):
    return str(event.get("category") or "ยังไม่จัด")


def _append_event(data, event):
    os.makedirs(data, exist_ok=True)
    with open(os.path.join(data, "events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _update_issue(data, event):
    issues = load_issues(data)
    fp = _fingerprint_for_event(event)
    cur = next((i for i in issues if i["fingerprint"] == fp), None)
    reopened = False
    if cur is None:
        cur = {"id": "ISS-%s" % (len(issues) + 1), "fingerprint": fp,
               "target_blamed": _target_for_issue(event), "root_target": "",
               "category": _category_for_issue(event), "count": 0, "status": "open",
               "fix_count": 0, "repeat_after_fix": 0, "last_seen": event["ts"]}
        issues.append(cur)
    cur["count"] += 1
    cur["last_seen"] = event["ts"]
    if cur.get("category") in ("", "ยังไม่จัด"):
        cur["category"] = _category_for_issue(event)
    if cur["status"] in ("fixed", "verified"):
        cur["status"] = "open"
        cur["repeat_after_fix"] += 1
        reopened = True
    save_json(os.path.join(data, "issues.json"), issues)
    return cur, reopened


def _record_event(data, event):
    event["fingerprint"] = _fingerprint_for_event(event)
    _append_event(data, event)
    return _update_issue(data, event)


def _event_from_detection(msg, detection, ts=None, cwd=None, source="log"):
    phrase = detection.get("phrase") or (detection.get("keywords") or [""])[0]
    return {
        "ts": ts or datetime.now().isoformat(timespec="seconds"),
        "msg": msg,
        "target": detection.get("target") or "-",
        "keywords": detection.get("keywords") or ([phrase] if phrase else []),
        "category": detection.get("category") or "ยังไม่จัด",
        "phrase": phrase,
        "cwd": cwd if cwd is not None else os.getcwd(),
        "source": source,
    }


def cmd_log(msg, data, kw):
    d = detect(msg, kw)
    if not d:
        print("ไม่พบคำด่า · ไม่บันทึก")
        return
    cur, reopened = _record_event(data, _event_from_detection(msg, d))
    if reopened:
        print("เตือน: เรื่องเดิม (%s) ที่เคยแก้แล้ว ถูกด่าซ้ำ → root อาจยังไม่ถูก" % cur["id"])
    thr = kw.get("repeat_warn_threshold", 3)
    flag = "  ** เกินเกณฑ์ %d ครั้ง ต้องหา root **" % thr if cur["count"] >= thr else ""
    print("บันทึก: เป้า=%s ครั้งที่=%d สถานะ=%s%s" % (cur["target_blamed"], cur["count"], cur["status"], flag))


def _read_jsonl(path):
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _ingest_key(row):
    return (
        str(row.get("ts") or ""),
        str(row.get("phrase") or ""),
        str(row.get("host") or ""),
    )


def _load_ingested_keys(data):
    keys = set()
    for row in _read_jsonl(os.path.join(data, "ingested.jsonl")):
        keys.add(_ingest_key(row))
    return keys


def _event_from_ingested_row(row):
    phrase = str(row.get("phrase") or "")
    target = str(row.get("target") or "").strip() or "-"
    category = str(row.get("category") or "").strip() or "unknown"
    return {
        "ts": str(row.get("ts") or datetime.now().isoformat(timespec="seconds")),
        "msg": str(row.get("msg") or phrase),
        "target": target,
        "keywords": [phrase] if phrase else [],
        "category": category,
        "phrase": phrase,
        "cwd": str(row.get("cwd") or ""),
        "host": str(row.get("host") or ""),
        "source": "ingest",
    }


def cmd_ingest(from_path, data, kw):
    if not from_path:
        print("ต้องใส่ --from <path ไป log.jsonl>")
        sys.exit(1)
    if not os.path.isfile(from_path):
        print("ไม่พบไฟล์นำเข้า: %s" % from_path)
        sys.exit(1)

    seen = _load_ingested_keys(data)
    imported = 0
    skipped = 0
    ingested_path = os.path.join(data, "ingested.jsonl")
    os.makedirs(data, exist_ok=True)

    with open(from_path, encoding="utf-8") as src, open(ingested_path, "a", encoding="utf-8") as ledger:
        for line in src:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception:
                skipped += 1
                continue
            if not isinstance(row, dict):
                skipped += 1
                continue
            key = _ingest_key(row)
            if key in seen:
                skipped += 1
                continue
            cur, reopened = _record_event(data, _event_from_ingested_row(row))
            if reopened:
                print("เตือน: เรื่องเดิม (%s) ที่เคยแก้แล้ว ถูกด่าซ้ำ → root อาจยังไม่ถูก" % cur["id"])
            ledger.write(json.dumps({
                "ts": key[0],
                "phrase": key[1],
                "host": key[2],
                "source": os.path.abspath(from_path),
            }, ensure_ascii=False) + "\n")
            seen.add(key)
            imported += 1

    print("นำเข้าใหม่=%d / ข้ามซ้ำ=%d" % (imported, skipped))


def _project_name(cwd):
    cwd = str(cwd or "").strip()
    if not cwd:
        return "ไม่ทราบ"
    name = os.path.basename(os.path.normpath(cwd))
    return name or "ไม่ทราบ"


def _project_counts_by_fingerprint(data):
    counts = defaultdict(Counter)
    for event in _read_jsonl(os.path.join(data, "events.jsonl")):
        counts[_fingerprint_for_event(event)][_project_name(event.get("cwd"))] += 1
    return counts


def _top_projects_text(counter):
    if not counter:
        return "ไม่ทราบ"
    return ", ".join("%s (%d)" % (name, count) for name, count in counter.most_common(3))


def cmd_report(data, html, kw):
    issues = load_issues(data)
    thr = kw.get("repeat_warn_threshold", 3)
    project_counts = _project_counts_by_fingerprint(data)
    total = sum(int(i.get("count", 0)) for i in issues)
    open_count = sum(1 for i in issues if i.get("status") == "open")
    over_count = sum(1 for i in issues if i.get("count", 0) >= thr and i.get("status") not in ("fixed", "verified"))
    print("== ทะเบียนปัญหาจากคำด่า ==")
    print("สรุปรวม: ด่าทั้งหมด %d ครั้ง · เรื่องเปิดอยู่ %d เรื่อง · เรื่องเกินเกณฑ์ %d เรื่อง" % (total, open_count, over_count))
    for i in sorted(issues, key=lambda x: -x["count"]):
        warn = " [เกินเกณฑ์]" if i["count"] >= thr and i["status"] not in ("fixed", "verified") else ""
        top_projects = _top_projects_text(project_counts.get(i["fingerprint"]))
        print("- %s เป้า=%s ด่า=%d ครั้ง โปรเจกต์=%s สถานะ=%s แก้ไป=%d ด่าซ้ำหลังแก้=%d%s"
              % (i["id"], i["target_blamed"], i["count"], top_projects,
                 i["status"], i["fix_count"], i["repeat_after_fix"], warn))
    if html:
        rows = []
        for i in sorted(issues, key=lambda x: -x["count"]):
            over = i["count"] >= thr and i["status"] not in ("fixed", "verified")
            rows.append("<tr%s><td>%s</td><td>%s</td><td>%s</td><td>%d</td><td>%s</td><td>%s</td><td>%d</td><td>%d</td></tr>"
                        % (" style='background:#fce8e8'" if over else "",
                           escape(str(i["id"])), escape(str(i["target_blamed"])),
                           escape(_top_projects_text(project_counts.get(i["fingerprint"]))),
                           i["count"], escape(str(i["category"])),
                           escape(str(i["status"])), i["fix_count"], i["repeat_after_fix"]))
        summary = (
            "<tr style='background:#eef3ff'><td colspan='8'><strong>สรุปรวม</strong> · "
            "ด่าทั้งหมด %d ครั้ง · เรื่องเปิดอยู่ %d เรื่อง · เรื่องเกินเกณฑ์ %d เรื่อง</td></tr>"
            % (total, open_count, over_count)
        )
        out = ("<!doctype html><meta charset='utf-8'><title>Curse Tracker</title>"
               "<style>body{font-family:sans-serif;padding:20px}table{border-collapse:collapse;width:100%%}"
               "td,th{border:1px solid #ccc;padding:8px;text-align:left}</style>"
               "<h2>ทะเบียนปัญหาจากคำด่า (วงจรปิด)</h2><table>"
               "<tr><th>id</th><th>เป้าที่ด่า</th><th>โปรเจกต์ที่โดนด่าบ่อยสุด</th><th>ด่ากี่ครั้ง</th><th>หมวด</th>"
               "<th>สถานะ</th><th>แก้กี่ครั้ง</th><th>ด่าซ้ำหลังแก้</th></tr>%s%s</table>"
               % (summary, "".join(rows)))
        save_json_text(html, out)
        print("เขียน HTML: %s" % html)


def cmd_promote(issue_id, root_text, data):
    """ยืนยัน root ที่แก้จริงแล้ว → status=verified + เขียนโน้ตถาวร (req-42 · ผ่านการรีวิวคือการรันคำสั่งนี้)"""
    issues = load_issues(data)
    cur = next((i for i in issues if i["id"] == issue_id), None)
    if cur is None:
        print("ไม่พบ %s" % issue_id)
        sys.exit(1)
    if not root_text:
        print("ต้องใส่ --root \"<สาเหตุจริงที่ยืนยันแล้ว>\"")
        sys.exit(1)
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
    ap.add_argument("command", choices=["log", "ingest", "report", "promote"])
    ap.add_argument("message", nargs="?", default="")
    ap.add_argument("--data", default=os.path.join(STD_ROOT, "learning", "data"))
    ap.add_argument("--from", dest="from_path", default="")
    ap.add_argument("--html")
    ap.add_argument("--root", default="")
    args = ap.parse_args()
    os.makedirs(args.data, exist_ok=True)
    kw = load_kw()
    if args.command == "log":
        cmd_log(args.message, args.data, kw)
    elif args.command == "ingest":
        cmd_ingest(args.from_path, args.data, kw)
    elif args.command == "promote":
        cmd_promote(args.message, args.root, args.data)
    else:
        cmd_report(args.data, args.html, kw)


if __name__ == "__main__":
    main()
