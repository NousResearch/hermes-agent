#!/usr/bin/env python3
"""relay-report — สรุปรายจ่าย/การใช้งาน AI Relay จาก ledger จริง (อ่านอย่างเดียว)

ให้เจ้าของเห็นในตารางเดียวว่า: วันไหนเรียก AI ตัวไหนกี่ครั้ง · สำเร็จ/พังเท่าไร ·
สลับตัวสำรองกี่ครั้ง · gate ผ่าน/ตกเท่าไร · ใช้งบประมาณเท่าไร
ใช้:  python relay-report.py --cwd <โฟลเดอร์โปรเจกต์> [--days 30] [--json]
อ่านจาก: .hermes/ai-relay/calls-*.md (ฝั่งเรียก AI) + .hermes/ledger/*.md (ฝั่งรัน gate)
"""
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path


REMOVED_TOOLS = {"fable"}


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


def load_prices(cwd):
    # ราคาต่อการเรียก 1 ครั้ง (บาท) อ่านจากทะเบียน (registry) ช่อง price_per_call_thb
    # ค่าเป็น [สมมติ] เจ้าของแก้ได้ในทะเบียน · อ่านไม่ได้/ไม่มี = {} (รายงานจะบอกว่าไม่มีราคา)
    try:
        import importlib.util
        p = Path(__file__).resolve().parent / "relay-call.py"
        spec = importlib.util.spec_from_file_location("relay_call_for_price", p)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        reg = m.load_registry(cwd)
    except Exception:
        return {}
    prices = {}
    for name, meta in (reg or {}).items():
        if not isinstance(meta, dict):
            continue
        v = meta.get("price_per_call_thb")
        try:
            if v is not None:
                prices[str(name)] = float(v)
        except (TypeError, ValueError):
            pass
    return prices


def compute_cost(calls_by_tool, prices):
    # ฟังก์ชันบริสุทธิ์ (เทสต์ง่าย): เรียกกี่ครั้ง × ราคาต่อครั้ง = บาท ต่อ tool + รวม
    # ตัวที่ไม่มีราคาในทะเบียน = ไม่นับเงิน (ไม่เดา) แต่รายงานแยกว่า "ไม่มีราคา"
    cost_by_tool, unknown, legacy_removed = {}, [], []
    raw_total = 0.0
    for tool, calls in (calls_by_tool or {}).items():
        if tool in REMOVED_TOOLS:
            if calls:
                legacy_removed.append(tool)
            continue
        if tool in prices:
            raw = calls * prices[tool]
            raw_total += raw                       # รวมยอดดิบก่อน แล้วค่อยปัด (กันเศษเพี้ยนสะสม)
            cost_by_tool[tool] = round(raw, 2)     # ค่ารายตัวปัดไว้แค่ตอนแสดง
        elif calls:
            unknown.append(tool)
    total = round(raw_total, 2)
    return {"cost_by_tool": cost_by_tool, "total_thb": total,
            "no_price_tools": sorted(set(unknown)),
            "legacy_removed_tools": sorted(set(legacy_removed)),
            # total_thb = ยอดเฉพาะ tool ที่มีราคา · ถ้ามี no_price_tools แปลว่ายังไม่ครบทั้งหมด
            "total_is_partial": bool(unknown)}


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
    for r in call_rows:
        day = (r.get("timestamp") or "")[:10] or "ไม่ทราบวัน"
        tool = r.get("tool") or "?"
        status = (r.get("status") or "").strip()
        e = days[day][tool]
        if status in ("skipped_by_cap", "skipped_not_owner_machine"):
            # ไม่ใช่การเรียกจริง · tool ถูกข้ามตามกฎเพดานหรือกฎเครื่อง
            e["skipped"] += 1
            continue
        e["calls"] += 1
        if status == "ok":
            e["ok"] += 1
        else:
            e["fail"] += 1
        if (r.get("rotated_from") or "").strip():
            e["rotated"] += 1

    gates = defaultdict(lambda: {"pass": 0, "fail": 0, "no_gate": 0, "error": 0})
    for r in gate_rows:
        day = (r.get("timestamp") or "")[:10] or "ไม่ทราบวัน"
        st = (r.get("result") or r.get("status") or "").strip()
        if st in gates[day]:
            gates[day][st] += 1

    all_days = sorted(set(days) | set(gates), reverse=True)[: a.days]

    # คิดเงิน (บาท) จากจำนวนครั้งจริง × ราคาต่อครั้งในทะเบียน (ราคา [สมมติ] เจ้าของแก้ได้)
    calls_by_tool = defaultdict(int)
    for d in all_days:
        for tool, e in days.get(d, {}).items():
            calls_by_tool[tool] += e["calls"]
    cost = compute_cost(calls_by_tool, load_prices(cwd))

    if a.json:
        out = {"root": str(cwd), "cost": cost,
               "days": {d: {"calls": {t: v for t, v in days[d].items()}, "gate": gates.get(d, {})}
                        for d in all_days}}
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return

    print(f"═══ AI Relay · สรุปการใช้งานจาก ledger จริง ═══")
    print(f"โปรเจกต์: {cwd}")
    print("สมองหลัก: Opus 4.8 ตัวเดียว · ไม่มี Fable/Faber/Fiber ในเส้นทางเรียกงาน")
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
            tool_label = f"{tool} (ถอดแล้ว)" if tool in REMOVED_TOOLS else tool
            print(f"| {d} | {tool_label} | {e['calls']} | {e['ok']} | {e['fail']} | {e['rotated']} | {gp} | {gf} | {gn} | {ge} |")
    print()
    # ── ค่าใช้จ่าย (บาท) ── ตอบคำถามเจ้าของว่าเดือนนี้ใช้ AI ตัวไหนเท่าไร
    if cost["cost_by_tool"]:
        print("═══ ค่าใช้จ่ายประมาณ (บาท · ราคาต่อครั้งเป็นค่า [สมมติ] แก้ได้ในทะเบียน registry) ═══")
        for tool, thb in sorted(cost["cost_by_tool"].items(), key=lambda x: -x[1]):
            print(f"  {tool}: {thb:,.2f} บาท ({calls_by_tool[tool]} ครั้ง)")
        total_label = "รวมเฉพาะที่มีราคา" if cost.get("total_is_partial") else "รวมทั้งหมด"
        print(f"  {total_label}: {cost['total_thb']:,.2f} บาท")
    else:
        print("ค่าใช้จ่าย: ยังไม่มีราคาในทะเบียน (ใส่ price_per_call_thb ต่อ AI ใน registry เพื่อให้คิดเงินได้)")
    if cost["no_price_tools"]:
        print(f"  (ยังไม่ได้ตั้งราคา จึงไม่นับเงิน: {', '.join(cost['no_price_tools'])})")
    if cost.get("legacy_removed_tools"):
        print(f"  (ประวัติจากเครื่องมือที่ถอดแล้ว ไม่นับเป็นเส้นทางใช้งานปัจจุบัน: {', '.join(cost['legacy_removed_tools'])})")
    print()
    print("อ่านยังไง: 'สลับมา' = จำนวนครั้งที่งานถูกโยนมาจากตัวที่พัง · gate ผ่าน/ตก/no_gate/พัง มาจาก gate-run จริง")


if __name__ == "__main__":
    main()
