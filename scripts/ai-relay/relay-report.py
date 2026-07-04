#!/usr/bin/env python3
"""relay-report — สรุปรายจ่าย/การใช้งาน AI Relay จาก ledger จริง (อ่านอย่างเดียว)

ให้เจ้าของเห็นในตารางเดียวว่า: วันไหนเรียก AI ตัวไหนกี่ครั้ง · สำเร็จ/พังเท่าไร ·
สมองพิเศษ (fable) ถูกเรียกกี่ครั้ง · สลับตัวสำรองกี่ครั้ง · gate ผ่าน/ตกเท่าไร
ใช้:  python relay-report.py --cwd <โฟลเดอร์โปรเจกต์> [--days 30] [--json]
อ่านจาก: .hermes/ai-relay/calls-*.md (ฝั่งเรียก AI) + .hermes/ledger/*.md (ฝั่งรัน gate)
"""
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path


def parse_md_table(path: Path):
    """อ่านตาราง markdown เป็น list ของ dict ตามหัวคอลัมน์ · แถวเสียข้ามไม่พัง"""
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return rows
    header = None
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if all(set(c) <= {"-", ":", " "} for c in cells):  # แถวขีดคั่นหัวตาราง
            continue
        if len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwd", default=os.environ.get("AI_RELAY_ROOT") or ".",
                    help="โฟลเดอร์โปรเจกต์ที่มี .hermes")
    ap.add_argument("--days", type=int, default=30, help="ย้อนหลังกี่วัน (ตามวันที่ใน ledger)")
    ap.add_argument("--json", action="store_true", help="พิมพ์เป็น JSON แทนตารางภาษาคน")
    a = ap.parse_args()
    cwd = Path(a.cwd).expanduser().resolve()

    call_rows, gate_rows = [], []
    for f in sorted((cwd / ".hermes" / "ai-relay").glob("calls-*.md")):
        call_rows += parse_md_table(f)
    for f in sorted((cwd / ".hermes" / "ledger").glob("*.md")):
        gate_rows += parse_md_table(f)

    if not call_rows and not gate_rows:
        print(json.dumps({"status": "empty",
                          "reason_human": f"ไม่พบ ledger ใต้ {cwd}/.hermes — ยังไม่เคยใช้ Relay ที่นี่"},
                         ensure_ascii=False))
        sys.exit(2)

    # จัดกลุ่ม วัน × tool
    days = defaultdict(lambda: defaultdict(lambda: {"calls": 0, "ok": 0, "fail": 0, "rotated": 0, "skipped": 0}))
    fable_total = 0          # เฉพาะ fable ที่ "เรียกจริง" (ไม่นับที่ถูกข้ามเพราะเกินเพดาน)
    fable_skipped_total = 0   # fable ที่ถูกข้ามไป opus เพราะชนเพดาน
    for r in call_rows:
        day = (r.get("timestamp") or "")[:10] or "ไม่ทราบวัน"
        tool = r.get("tool") or "?"
        status = (r.get("status") or "").strip()
        e = days[day][tool]
        if status in ("skipped_by_cap", "skipped_not_owner_machine"):
            # ไม่ใช่การเรียกจริง · fable ถูกข้ามไปสมองสำรอง (เกินเพดาน หรือเครื่องไม่อยู่ในรายชื่ออนุญาต)
            e["skipped"] += 1
            if tool == "fable":
                fable_skipped_total += 1
            continue
        e["calls"] += 1
        if status == "ok":
            e["ok"] += 1
        else:
            e["fail"] += 1
        if (r.get("rotated_from") or "").strip():
            e["rotated"] += 1
        if tool == "fable":
            fable_total += 1

    gates = defaultdict(lambda: {"pass": 0, "fail": 0, "no_gate": 0, "error": 0})
    for r in gate_rows:
        day = (r.get("timestamp") or "")[:10] or "ไม่ทราบวัน"
        st = (r.get("result") or r.get("status") or "").strip()
        if st in gates[day]:
            gates[day][st] += 1

    all_days = sorted(set(days) | set(gates), reverse=True)[: a.days]

    if a.json:
        out = {"root": str(cwd), "fable_total": fable_total,
               "days": {d: {"calls": {t: v for t, v in days[d].items()}, "gate": gates.get(d, {})}
                        for d in all_days}}
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return

    print(f"═══ AI Relay · สรุปการใช้งานจาก ledger จริง ═══")
    print(f"โปรเจกต์: {cwd}")
    print(f"สมองพิเศษ (fable) ถูกเรียกจริงทั้งหมด: {fable_total} ครั้ง"
          + (f" · ถูกข้ามไป opus เพราะชนเพดาน: {fable_skipped_total} ครั้ง" if fable_skipped_total else ""))
    no_gate_total = sum(g.get("no_gate", 0) for g in gates.values())
    gate_err_total = sum(g.get("error", 0) for g in gates.values())
    if no_gate_total or gate_err_total:
        print(f"⚠️ งานที่ยังไม่มีหลักฐานตัวตรวจ: no_gate {no_gate_total} ครั้ง · gate รันไม่สำเร็จ {gate_err_total} ครั้ง — ห้ามนับว่า verified")
    print()
    print("| วัน | AI | เรียก | สำเร็จ | พัง | สลับมา | gate ผ่าน | gate ตก | no_gate | gate พัง |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in all_days:
        g = gates.get(d, {})
        tools = days.get(d, {})
        if not tools:
            print(f"| {d} | (มีแต่ gate) | 0 | 0 | 0 | 0 | {g.get('pass',0)} | {g.get('fail',0)} | {g.get('no_gate',0)} | {g.get('error',0)} |")
        for i, (tool, e) in enumerate(sorted(tools.items())):
            gp = g.get("pass", 0) if i == 0 else ""
            gf = g.get("fail", 0) if i == 0 else ""
            gn = g.get("no_gate", 0) if i == 0 else ""
            ge = g.get("error", 0) if i == 0 else ""
            print(f"| {d} | {tool} | {e['calls']} | {e['ok']} | {e['fail']} | {e['rotated']} | {gp} | {gf} | {gn} | {ge} |")
    print()
    print("อ่านยังไง: 'สลับมา' = จำนวนครั้งที่งานถูกโยนมาจากตัวที่พัง · gate ผ่าน/ตก/no_gate/พัง มาจาก gate-run จริง")


if __name__ == "__main__":
    main()
