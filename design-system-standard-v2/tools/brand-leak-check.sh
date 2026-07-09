#!/usr/bin/env bash
# brand-leak-check — กันสีตัวอย่าง OneManFleet หลุดไปหน้าโชว์ของโปรเจกต์อื่น
#
# ทำไมต้องมี: onemanfleet-ds.html เป็น "โครงตัวอย่าง" ที่มีสีแบรนด์ OneManFleet
#   (#E94560 แดง · #1A1A2E กรมท่า) ฝังอยู่ในตัว · เวลาเอาไปตั้งต้นให้โปรเจกต์อื่น
#   ต้องเปลี่ยนเป็นสีแบรนด์ของโปรเจกต์นั้น · ตัวนี้กันไม่ให้เผลอปล่อยสีตัวอย่างค้าง
#
# ใช้:  bash brand-leak-check.sh <path ที่ต้องตรวจ เช่น myproject/design-system/>
# ผล:  exit 0 = สะอาด (ไม่เหลือสีตัวอย่าง) · exit 1 = เจอสีค้าง ต้องแก้ก่อนปิดงาน
set -euo pipefail

TARGET="${1:-.}"
# สีตัวอย่าง OneManFleet ที่ห้ามหลุดไปโปรเจกต์อื่น
LEAK_COLORS="E94560|1A1A2E"

if [ ! -e "$TARGET" ]; then
  echo "⚠️ ไม่พบ path: $TARGET" >&2
  exit 2
fi

# ตรวจเฉพาะไฟล์หน้าโชว์/สไตล์ · ข้ามตัวเอกสารเตือน (ADOPT-RECIPE) และสคริปต์ตัวนี้เอง
hits=$(grep -rniE "$LEAK_COLORS" "$TARGET" 2>/dev/null \
  | grep -viE "brand-leak-check|ADOPT-RECIPE" || true)

if [ -n "$hits" ]; then
  echo "❌ เจอสีตัวอย่าง OneManFleet ค้างใน: $TARGET"
  echo "   ต้องเปลี่ยนเป็นสีแบรนด์ของโปรเจกต์นี้ก่อนปิดงาน (ผ่านด่านสี Phase 3)"
  echo "$hits" | head -20
  echo "จำนวนจุดที่ต้องแก้: $(printf '%s\n' "$hits" | grep -c . || echo 0)"
  exit 1
fi

echo "✅ สะอาด — ไม่เหลือสีตัวอย่าง OneManFleet (#E94560/#1A1A2E) ใน $TARGET"
exit 0
