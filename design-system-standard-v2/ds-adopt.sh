#!/usr/bin/env bash
# ds-adopt.sh — ตัวช่วย adopt Design System เข้าโปรเจกต์ · 1 คำสั่ง · ใช้ได้ทั้ง VPS + Notebook
#
# แนวคิด: รวมงาน "กลไก" (ติดตั้ง/build/วัด/รันด่านตรวจ) ให้เหลือ 2 คำสั่ง
# ส่วนงาน "สมอง" (วิเคราะห์ + สร้าง DS + เลือกสี) ให้พิมพ์ `Use Design System` ในแชตของโปรเจกต์
#
# ใช้:
#   bash ds-adopt.sh prep  <target-project>                  # ก่อนสั่ง AI: ติดตั้ง playwright + build token + วัด baseline
#   bash ds-adopt.sh check <target-project> [หน้าโชว์.html]  # ก่อนปิดงาน: รันด่านครบ + เทียบ baseline (exit 1 ถ้าไม่ผ่าน)
#
# คืน exit 0 = ผ่าน · 1 = มีด่านไม่ผ่าน (ห้ามปิดงาน) · 2 = ใช้ผิด
set -euo pipefail

DS_DIR="$(cd "$(dirname "$0")" && pwd)"   # โฟลเดอร์ design-system-standard-v2
CMD="${1:-}"
TARGET="${2:-}"
PAGE="${3:-}"

if [ -z "$CMD" ] || [ -z "$TARGET" ]; then
  echo "ใช้:"
  echo "  bash ds-adopt.sh prep  <target-project>"
  echo "  bash ds-adopt.sh check <target-project> [หน้าโชว์.html]"
  exit 2
fi
[ -d "$TARGET" ] || { echo "⚠ ไม่พบโฟลเดอร์เป้าหมาย: $TARGET"; exit 2; }

need_node() { command -v node >/dev/null 2>&1 || { echo "✗ ไม่มี node — ติดตั้ง Node.js ก่อน"; exit 2; }; }

case "$CMD" in
  prep)
    echo "== เตรียม adopt DS เข้า: $TARGET =="
    need_node
    # 1) ติดตั้ง playwright ครั้งแรก (ตัวตรวจ contrast หน้าจริง)
    if [ ! -d "$DS_DIR/tools/node_modules/playwright" ]; then
      echo "→ ติดตั้ง playwright (ครั้งแรกครั้งเดียว)"
      ( cd "$DS_DIR/tools" && npm i playwright >/dev/null 2>&1 ) && echo "  ✓ playwright ติดตั้งแล้ว" || echo "  ⚠ ติดตั้ง playwright ไม่สำเร็จ — ลอง: cd $DS_DIR/tools && npm i playwright"
    else
      echo "  ✓ playwright มีอยู่แล้ว"
    fi
    # 2) build token กลาง
    ( cd "$DS_DIR/tokens" && node build-tokens.mjs >/dev/null ) && echo "  ✓ build token กลาง (dist/)"
    # 3) วัดคะแนนเดิมของโปรเจกต์ (baseline กันถอยหลัง)
    base="$TARGET/.ds-baseline.txt"
    python3 "$DS_DIR/tools/ds-check.py" "$TARGET" 2>/dev/null | tee "$base" | tail -1
    echo "  ✓ baseline เก็บที่ $base"
    echo ""
    echo "พร้อมแล้ว → พิมพ์ \`Use Design System\` ในแชตของโปรเจกต์นี้ (AI จะสร้าง DS ให้)"
    echo "เว็บ production (DRA/Content Thailand): บอก AI ว่า MIGRATION + showcase ก่อน · ห้าม greenfield ทับ"
    ;;

  check)
    echo "== ตรวจก่อนปิดงาน DS: $TARGET =="
    need_node
    fail=0
    # build token
    ( cd "$DS_DIR/tokens" && node build-tokens.mjs >/dev/null ) && echo "✓ build-tokens (exit 0)" || { echo "✗ build-tokens"; fail=1; }
    # ds-check token adoption
    if python3 "$DS_DIR/tools/ds-check.py" "$TARGET" >/tmp/dscheck.out 2>&1; then
      tail -1 /tmp/dscheck.out; echo "✓ ds-check รันผ่าน"
    else
      tail -3 /tmp/dscheck.out; echo "✗ ds-check"; fail=1
    fi
    # กันสี OneManFleet หลุด
    if bash "$DS_DIR/tools/brand-leak-check.sh" "$TARGET" >/tmp/leak.out 2>&1; then
      echo "✓ brand-leak-check (ไม่เหลือสี OneManFleet)"
    else
      tail -3 /tmp/leak.out; echo "✗ brand-leak-check — เหลือสี OneManFleet ต้องเปลี่ยนเป็นสีแบรนด์โปรเจกต์"; fail=1
    fi
    # contrast หน้าจริง (ถ้าระบุหน้า)
    if [ -n "$PAGE" ]; then
      if node "$DS_DIR/tools/contrast-audit-run.mjs" "$PAGE" >/tmp/contrast.out 2>&1; then
        echo "✓ contrast-audit (0 fail) · ภาพ: ${PAGE%.html}-contrast-audit.png"
      else
        tail -4 /tmp/contrast.out; echo "✗ contrast-audit — มีข้อความตก WCAG AA"; fail=1
      fi
    else
      echo "… ข้าม contrast (ไม่ได้ระบุหน้าโชว์) — ควรรัน: bash ds-adopt.sh check $TARGET <หน้า.html>"
    fi
    echo ""
    if [ "$fail" = 0 ]; then
      echo "✅ ผ่านทุกด่าน — พร้อม commit (branch แยก · ห้าม merge/deploy เอง)"
    else
      echo "❌ มีด่านไม่ผ่าน — ห้ามปิดงาน วนแก้จนผ่าน"
      exit 1
    fi
    ;;

  *)
    echo "คำสั่งไม่รู้จัก: $CMD (ใช้ prep หรือ check)"; exit 2 ;;
esac
