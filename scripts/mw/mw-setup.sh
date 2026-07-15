#!/usr/bin/env bash
# mw-setup — ติดตั้งเครื่องมือ Use Migrate Web (MW) ให้เรียกได้จากทุกที่
#
# ใช้ได้ 2 ทาง (notebook + VPS): ทีม pull repo แล้วรันสคริปต์นี้ครั้งเดียวต่อเครื่อง
#   bash scripts/mw/mw-setup.sh
# ติดตั้งเครื่องมือทีม 7 ตัว (runtime) เข้า ~/.local/bin (หรือ $MW_BIN_DIR)
# หมายเหตุ: mw-spec-check เป็นเครื่องมือฝั่งนักพัฒนา shortcut (ต้องมี .project/mw-spec-draft.md ใน repo)
#   ไม่ติดตั้งให้เครื่องทีม — ใช้จาก repo ตรง ๆ: python3 scripts/mw-spec-check.py
# แล้ว "ยิงทดสอบจริง" ทุกตัว (--help ต้อง exit 0) — ไม่ใช่แค่เจอไฟล์
# (บทเรียน relay: รุ่นติดตั้ง ≠ repo · ต้องรันจริงถึงจะนับว่าใช้ได้)
#
# portable: ไม่ใช้ bash associative array (รองรับ bash 3.2 ของ macOS)
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MW_DIR="$SCRIPT_DIR"                      # scripts/mw
BIN_DIR="${MW_BIN_DIR:-$HOME/.local/bin}"
# ที่เก็บถาวรของตัวเครื่องมือ — ต้อง "คัดลอก" มาไว้ที่นี่ก่อนค่อย link
# (บทเรียน 2026-07-15: เครื่องทีมติดตั้งผ่าน curl → ไฟล์ต้นทางอยู่ /tmp ที่ถูกลบหลังติดตั้ง
#  ถ้า symlink ชี้ต้นทางตรง ๆ เครื่องมือจะตายทั้งชุดทันทีที่ /tmp หาย)
# path นี้ต้องตรงกับ fallback ของ hook enforce-flow-gate.py (~/.hermes/mw/flow_gate.py)
LIB_DIR="${MW_LIB_DIR:-$HOME/.hermes/mw}"
PYTHON="${MW_PYTHON:-python3}"

# name:filename (ไฟล์ใน scripts/mw)
TOOLS="
mw-work-locks:work_locks.py
mw-menu-gate:menu_gate.py
mw-page-check:page_check.py
mw-doctor:mw_doctor.py
mw-rtm-report:rtm_report.py
mw-wow-report:wow_report.py
mw-backend-check:backend_check.py
"

# ไฟล์ที่ hook/เครื่องมือใช้ร่วม — ต้องไปอยู่ $LIB_DIR ด้วยแม้ไม่มีชื่อใน BIN
# (flow_eval = โมดูลที่ menu_gate/flow_gate import · flow-rules.yaml = กติกากลาง 13 ขั้น
#  flow_gate = ตัวที่ hook enforce-flow-gate ค้นหาที่ ~/.hermes/mw บนเครื่องทีม)
SHARED_FILES="flow_eval.py flow_gate.py flow-rules.yaml"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ผิดพลาด: ไม่พบ $PYTHON — ต้องมี Python 3.9+ ก่อนติดตั้งเครื่องมือ MW" >&2
  exit 1
fi

mkdir -p "$BIN_DIR"
mkdir -p "$LIB_DIR"

echo "== mw-setup: คัดลอกเครื่องมือ MW เข้า $LIB_DIR แล้ว link เข้า $BIN_DIR =="

# ── จังหวะ 1 · คัดลอกให้ครบทั้งชุดก่อน — ขาดตัวไหน = หยุดก่อนแตะ $BIN_DIR
# (Codex review 2026-07-15: กันติดตั้งค้างครึ่งชุด เครื่องมือใหม่-เก่าปนกัน)
failed=0
for f in $SHARED_FILES; do
  if [ -f "$MW_DIR/$f" ]; then
    cp -f "$MW_DIR/$f" "$LIB_DIR/$f"
  else
    echo "  ❌ ไม่พบไฟล์ร่วม: $MW_DIR/$f"
    failed=$((failed + 1))
  fi
done
for pair in $TOOLS; do
  name="${pair%%:*}"
  rel="${pair##*:}"
  src="$MW_DIR/$rel"
  if [ ! -f "$src" ]; then
    echo "  ❌ ไม่พบไฟล์: $src"
    failed=$((failed + 1))
    continue
  fi
  lib_file="$LIB_DIR/$rel"
  cp -f "$src" "$lib_file"
  chmod +x "$lib_file" 2>/dev/null || true
done
if [ "$failed" -ne 0 ]; then
  echo "ผิดพลาด: คัดลอกไม่ครบ $failed รายการ — หยุดก่อนสร้างคำสั่ง (กันชุดครึ่ง ๆ กลาง ๆ)" >&2
  exit 1
fi

# ── จังหวะ 2 · สร้างคำสั่งใน $BIN_DIR ชี้สำเนาถาวร
installed=0
clobber=0
for pair in $TOOLS; do
  name="${pair%%:*}"
  rel="${pair##*:}"
  lib_file="$LIB_DIR/$rel"
  target="$BIN_DIR/$name"
  if [ -L "$target" ]; then
    # symlink เดิม (รวม symlink-ชี้-โฟลเดอร์) → ลบทิ้งก่อน แล้วค่อยสร้างใหม่
    # (กัน `ln -sf` เดินตาม symlink-ชี้-โฟลเดอร์ ไปสร้างลิงก์ทับไฟล์ข้างใน — GPT-5 รอบ 2)
    rm -f "$target"
  elif [ -e "$target" ]; then
    # มีอยู่จริงและไม่ใช่ symlink = ไฟล์/โฟลเดอร์จริงของผู้ใช้ → ไม่แตะ
    echo "  ❌ $name: มีไฟล์จริงอยู่แล้วที่ $target — ไม่เขียนทับ (ลบเองก่อนถ้าต้องการ)"
    clobber=$((clobber + 1))
    continue
  fi
  ln -s "$lib_file" "$target"
  installed=$((installed + 1))
done

# ล้างลิงก์ mw-spec-check ของรุ่นเก่า (เคยติดตั้งให้ทีมแล้วเลิก — Codex review 2026-07-15)
# ลบเฉพาะเมื่อเป็น symlink เท่านั้น ไฟล์จริงของผู้ใช้ไม่แตะ
if [ -L "$BIN_DIR/mw-spec-check" ]; then
  rm -f "$BIN_DIR/mw-spec-check"
  echo "  ลบลิงก์เก่า mw-spec-check (เครื่องมือฝั่งนักพัฒนา ไม่ติดตั้งให้ทีมแล้ว)"
fi

echo "== ยิงทดสอบจริง (เรียกตรงผ่าน symlink · ทดสอบ shebang+exec ตามที่โฆษณา) =="
smoke_fail=0
for pair in $TOOLS; do
  name="${pair%%:*}"
  bin="$BIN_DIR/$name"
  [ -L "$bin" ] || continue
  # เรียกตรง (ไม่ผ่าน $PYTHON) เพื่อพิสูจน์ว่า `$name --help` ใช้ได้จริง (shebang+exec bit)
  if "$bin" --help >/dev/null 2>&1; then
    echo "  ✅ $name"
  else
    echo "  ❌ $name (เรียกตรง --help ไม่ผ่าน — เช็ก shebang/exec bit)"
    smoke_fail=$((smoke_fail + 1))
  fi
done

echo "== สรุป: symlink $installed ตัว · ทดสอบไม่ผ่าน $smoke_fail · ทับไม่ได้ $clobber =="
if [ "$smoke_fail" -ne 0 ] || [ "$clobber" -ne 0 ]; then
  echo "ยังไม่พร้อม 100% — แก้รายการ ❌ ก่อน (ตรวจ Python/ไฟล์/ไฟล์ชนชื่อ)" >&2
  exit 1
fi

case ":$PATH:" in
  *":$BIN_DIR:"*) : ;;
  *) echo "หมายเหตุ: $BIN_DIR ยังไม่อยู่ใน PATH — เพิ่มบรรทัดนี้ใน ~/.zshrc หรือ ~/.bashrc:"
     echo "  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac
if [ -f "$LIB_DIR/flow_gate.py" ] && [ -f "$LIB_DIR/flow_eval.py" ] && [ -f "$LIB_DIR/flow-rules.yaml" ]; then
  echo "flow-gate สำหรับ hook: ครบ 3 ไฟล์ที่ $LIB_DIR (enforce-flow-gate หาเจอ)"
else
  echo "ผิดพลาด: ไฟล์ flow-gate ไม่ครบที่ $LIB_DIR — hook ด่านก่อนเขียนจะ block งาน MW" >&2
  exit 1
fi
echo "mw-setup เสร็จ · เครื่องมือทีมครบ 7 ตัวเรียกได้แล้ว · สำเนาถาวรที่ $LIB_DIR"
