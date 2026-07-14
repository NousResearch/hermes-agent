#!/usr/bin/env python3
"""mw-spec-check — เครื่องตรวจโครงสร้าง SPEC ของแผน MW (Use Migrate Web)

ใช้แทนผู้ตรวจ AI รอบที่ 3 ตามกติกา relay v2.16 (ผู้ตรวจเดิมครบ 2 รอบ → เปลี่ยนเป็นเครื่องตรวจ)
ตรวจ 5 ด่าน:
  1. ตารางแม่ต้องนับได้ 55 แถว (I1-25 + I2-4 + I3-12 + I3-R8 + I4-4 + I5-2 = 55)
  2. ทุกแถว [G] ต้องอ้างเครื่องมือ §10-x อย่างน้อย 1 ตัว (หรือ gate-run/Use QA QC ที่นิยามใน §0/§9)
  3. ทุก § ที่ถูกอ้างในตารางแม่ ต้องมีหัวข้อจริงในไฟล์
  4. จุดเคาะต้องครบ 13 จุดในตารางฝัง
  5. baseline: .project/mw-flow-baseline.md ต้องมี และ sha256 ในตารางตรงกับไฟล์จริง ณ ตอนรัน

exit 0 = ผ่านทุกด่าน · exit 1 = ตกอย่างน้อย 1 ด่าน (พิมพ์รายการตก)
"""
import ast
import hashlib
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / ".project" / "mw-spec-draft.md"
BASELINE = ROOT / ".project" / "mw-flow-baseline.md"
EXPECTED_ROWS = 55
EXPECTED_DECISIONS = 13
# เครื่องมือที่ [G] อ้างได้: §10-1..§10-9 หรือชื่อกลไกที่นิยามไว้ใน spec เอง
G_TOOL_RE = re.compile(r"§10-\d+|gate-run|Use QA QC|\[Q\]")

def fail(msgs):
    print("MW-SPEC-CHECK: FAIL")
    for m in msgs:
        print(f"  ✗ {m}")
    return 1

MAP_FILE = ROOT / ".project" / "mw-g-testid-map.md"
TESTS_DIR = ROOT / "tests" / "scripts" / "mw"
ALLOWED_EXTERNAL = {"ds-check", "hermes-write-permit", "gitleaks", "Use QA QC"}
_MAP_ROW_RE = re.compile(
    r"^\|\s*(I\d+-R?\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(mapped|external|pending-i2e)\s*\|"
)


def _collect_test_defs():
    """คืน (counts, skipped) ผ่าน AST — กัน `def test_x(` ในสตริง/คอมเมนต์นับเป็นของจริง.

    counts: Counter ชื่อฟังก์ชันเทสต์ (>1 = กำกวมชื่อซ้ำ) · skipped: ชื่อที่ติด skip/xfail.
    """
    counts = Counter()
    skipped = set()
    if not TESTS_DIR.exists():
        return counts, skipped
    for tf in TESTS_DIR.glob("*.py"):
        try:
            tree = ast.parse(tf.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                counts[node.name] += 1
                for dec in node.decorator_list:
                    if "skip" in ast.dump(dec).lower() or "xfail" in ast.dump(dec).lower():
                        skipped.add(node.name)
    return counts, skipped


def check_g_testid_map(rows, errors) -> str:
    """ด่าน 6 (§13.1): ทุกแถว [G] ต้องมีใน mw-g-testid-map + test id ที่มีจริง (fail closed).

    เข้ม (หลัง GPT-5 review): กัน map แถวซ้ำ · test ชื่อกำกวม/skip/ในคอมเมนต์ · [G] ผิดคอลัมน์ ·
    pending ได้เฉพาะ §10-8 · external = ประกาศ (ยังไม่พิสูจน์ในเรโปนี้) · pending>0 = INCOMPLETE.
    """
    # เก็บ [G] row codes + เครื่องมือ §10 ต่อแถวจากตารางแม่ + กัน [G] นอกช่องคำตัดสิน (คอลัมน์ผิด)
    g_rows = []
    g_tools = {}  # code -> set ของ "10-x" ที่ตารางแม่อ้างในช่องคำตัดสิน
    for l in rows:
        cells = l.split("|")
        m = re.match(r"^\| (I\d+-R?\d+)", l)
        if not m:
            continue
        code = m.group(1)
        verdict_cell = cells[4] if len(cells) > 4 else ""
        # [G] ต้องอยู่ช่องคำตัดสินเท่านั้น — ก่อนหรือหลังช่องนั้น = โครงผิด
        non_verdict = "".join(cells[1:4]) + "".join(cells[5:])
        if "[G]" in non_verdict:
            errors.append(f"§13.1: {code} มี [G] นอกช่องคำตัดสิน (โครงตารางผิด)")
        if "[G]" in verdict_cell:
            g_rows.append(code)
            g_tools[code] = set(re.findall(r"§?(10-\d+)", verdict_cell))

    if not MAP_FILE.exists():
        errors.append(f"§13.1: ไม่พบ {MAP_FILE} (ตาราง [G]→test id)")
        return "§13.1: ไม่มีไฟล์ map"

    entries = {}
    for ml in MAP_FILE.read_text(encoding="utf-8").splitlines():
        mm = _MAP_ROW_RE.match(ml)
        if mm:
            code = mm.group(1)
            if code in entries:
                errors.append(f"§13.1: map มีแถวซ้ำ {code} (ห้ามซ้ำ — กันปิดบังข้อ mapped ที่หาย)")
            entries[code] = (mm.group(2).strip(), mm.group(3).strip(), mm.group(4).strip())

    counts, skipped = _collect_test_defs()
    n_mapped = n_external = n_pending = 0
    for code in g_rows:
        if code not in entries:
            errors.append(f"§13.1: แถว [G] {code} ไม่มีใน mw-g-testid-map.md")
            continue
        tool, tid, status = entries[code]
        # เครื่องมือใน map ต้องตรงกับ §10 ที่ตารางแม่อ้างจริง (กันแถว §10-2 ถูกยัดเป็น §10-8 pending)
        tool_norm = tool.replace("§", "").strip()
        if g_tools.get(code) and tool_norm not in g_tools[code]:
            want = sorted("§" + t for t in g_tools[code])
            errors.append(f"§13.1: {code} map tool {tool!r} ไม่ตรงเครื่องมือในตารางแม่ {want}")
        if status == "mapped":
            c = counts.get(tid, 0)
            if c == 0:
                errors.append(f"§13.1: {code} mapped แต่ไม่พบ test '{tid}' (AST) ใน tests/scripts/mw/")
            elif c > 1:
                errors.append(f"§13.1: {code} mapped test '{tid}' กำกวม — มีชื่อซ้ำ {c} ตัว")
            elif tid in skipped:
                errors.append(f"§13.1: {code} mapped test '{tid}' ติด skip/xfail — พิสูจน์ไม่ได้")
            else:
                n_mapped += 1
        elif status == "external":
            if tid not in ALLOWED_EXTERNAL:
                errors.append(f"§13.1: {code} external tag '{tid}' ไม่อยู่ในชุดที่อนุญาต {sorted(ALLOWED_EXTERNAL)}")
            else:
                n_external += 1
        elif status == "pending-i2e":
            if tool != "§10-8":
                errors.append(f"§13.1: {code} pending-i2e แต่ tool={tool!r} — pending ได้เฉพาะ §10-8 (mw-backend-check)")
            else:
                n_pending += 1

    gset = set(g_rows)
    for code in entries:
        if code not in gset:
            errors.append(f"§13.1: mw-g-testid-map มีแถว {code} ที่ไม่ใช่ [G] ในตารางแม่ (stale)")

    # strict mode (opt-in): ผู้เรียกที่ต้องการ §13.1 ครบจริงตั้ง env นี้ → pending>0 = ตก
    if n_pending and os.environ.get("MW_SPEC_REQUIRE_G13_COMPLETE") == "1":
        errors.append(
            f"§13.1: strict — ยัง INCOMPLETE ({n_pending} pending §10-8 · ต้องจบ I2e/P4 ก่อน)"
        )

    word = "COMPLETE" if n_pending == 0 else f"INCOMPLETE ({n_pending} pending §10-8 → P4)"
    return (
        f"§13.1 {word}: {len(g_rows)} [G] · mapped(verified) {n_mapped} · "
        f"external(unverified-here) {n_external} · pending-i2e {n_pending}"
    )


def main() -> int:
    errors = []
    if not SPEC.exists():
        return fail([f"ไม่พบ {SPEC}"])
    text = SPEC.read_text(encoding="utf-8")
    lines = text.splitlines()

    # ด่าน 1: นับตารางแม่
    rows = [l for l in lines if re.match(r"^\| I[1-5]", l)]
    if len(rows) != EXPECTED_ROWS:
        errors.append(f"ตารางแม่นับได้ {len(rows)} แถว (ต้อง {EXPECTED_ROWS})")
    codes = [re.match(r"^\| (I[0-9R-]+-\d+)", l) for l in rows]
    ids = [m.group(1) for m in codes if m]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        errors.append(f"รหัสแถวซ้ำ: {sorted(dup)}")

    # ด่าน 2: ทุกแถว [G] ต้องอ้างเครื่องมือ
    for l in rows:
        code = l.split("|")[1].strip()
        cells = l.split("|")
        verdict_cell = cells[4] if len(cells) > 4 else ""
        if "[G]" in verdict_cell and not G_TOOL_RE.search(verdict_cell):
            errors.append(f"{code}: [G] ไม่อ้างเครื่องมือ §10-x/gate-run/QA QC → ระบุเครื่องมือ หรือเปลี่ยนเป็น [F]/[M]")

    # ด่าน 3: § ที่ถูกอ้างต้องมีจริง
    headers = set(re.findall(r"^## (\d+)\.", text, flags=re.M))
    for l in rows:
        for sec in re.findall(r"§(\d+)", l):
            if sec not in headers and sec != "10":  # §10-x จับใน pattern เดียวกัน
                errors.append(f"อ้าง §{sec} แต่ไม่มีหัวข้อ '## {sec}.' ในไฟล์ ({l.split('|')[1].strip()})")
    # §10-x ต้องมีแถวเครื่องมือจริง
    tools = set(re.findall(r"^\| (10-\d+)", text, flags=re.M))
    for l in rows:
        for t in re.findall(r"§(10-\d+)", l):
            if t not in tools:
                errors.append(f"อ้าง §{t} แต่ไม่มีแถวเครื่องมือ '| {t} |' ใน §10 ({l.split('|')[1].strip()})")

    # ด่าน 4: จุดเคาะ 13
    dec = re.findall(r"^\| (\d+) [^|]*\| ([^|]+) \|", text, flags=re.M)
    dec_nums = {int(a) for a, b in dec if int(a) <= 13}
    missing = set(range(1, EXPECTED_DECISIONS + 1)) - dec_nums
    if missing:
        errors.append(f"จุดเคาะหาย: {sorted(missing)} (ต้องครบ 1-13)")

    # ด่าน 5: baseline sha256 ตรงจริง
    if not BASELINE.exists():
        errors.append(f"ไม่พบ {BASELINE}")
    else:
        btext = BASELINE.read_text(encoding="utf-8")
        entries = re.findall(r"^\| (/[^|]+?) \| ([0-9a-f]{64}) \|", btext, flags=re.M)
        if len(entries) < 3:
            errors.append(f"baseline มี {len(entries)} รายการ (ต้อง ≥3)")
        for path, sha in entries:
            fp = Path(path.strip())
            if not fp.exists():
                errors.append(f"baseline: ไฟล์ต้นทางหาย {fp}")
                continue
            actual = hashlib.sha256(fp.read_bytes()).hexdigest()
            if actual != sha:
                errors.append(f"baseline: sha256 ไม่ตรง {fp.name} (ต้นทางถูกแก้หลังจดทะเบียน — หยุด แจ้งเจ้าของ)")

    # ด่าน 6: §13.1 — ทุกแถว [G] ต้องผูก test id จริงใน .project/mw-g-testid-map.md
    g13_tally = check_g_testid_map(rows, errors)

    if errors:
        return fail(errors)
    print(
        f"MW-SPEC-CHECK: PASS — ตารางแม่ {len(rows)}/{EXPECTED_ROWS} · "
        f"[G] อ้างเครื่องมือครบ · § อ้างอิงครบ · จุดเคาะ 13/13 · "
        f"baseline sha256 ตรง {len(re.findall(chr(92)+'| /', BASELINE.read_text())) if BASELINE.exists() else 0} ไฟล์ · "
        f"{g13_tally}"
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
