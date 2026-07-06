#!/usr/bin/env bash
# link-brand-ds — ต่อ Design System กลางเข้าโปรเจกต์นี้ (ฉลาด: local=symlink · ที่อื่น=clone)
# ใช้: รันที่รากโปรเจกต์  →  bash link-brand-ds.sh
set -euo pipefail

MASTER="/Users/rattanasak/Documents/Viber Project/Tech Tools/Hermes Agent/design-system-standard-v2"
HUB="https://gitlab.dev.jigsawgroups.work/Nat-Rattanasak/designsystem_onemanfleet.git"
DEST="design/brand-ds"

mkdir -p design
if [ -e "$DEST" ] || [ -L "$DEST" ]; then rm -rf "$DEST"; fi

if [ -d "$MASTER" ]; then
  # เครื่องเจ้าของ: symlink ไป master → แก้ที่เดียวเห็นทุกโปรเจกต์ทันที ไม่ต้อง pull
  ln -s "$MASTER" "$DEST"
  echo "✓ ต่อแบบ symlink → $MASTER  (แก้ master = ทุกโปรเจกต์เปลี่ยนทันที)"
else
  # เครื่องอื่น/VPS: clone จาก GitLab (ต้องมีสิทธิ์ล็อกอิน)
  echo "ไม่พบ master ในเครื่อง → ดึงจาก GitLab"
  if ! git clone --depth 1 "$HUB" "$DEST" 2>/dev/null; then
    echo "✗ clone ไม่ได้ (repo private · เครื่องนี้ยังไม่มีสิทธิ์ GitLab)"
    echo "  → แจ้งเจ้าของให้เพิ่มสิทธิ์ หรือ copy โฟลเดอร์ design-system-standard-v2 มาไว้ที่ $MASTER"
    exit 1
  fi
  echo "✓ clone แล้ว"
fi

# build token ให้พร้อมใช้
if command -v node >/dev/null 2>&1; then
  ( cd "$DEST/tokens" && node build-tokens.mjs >/dev/null ) && echo "✓ build token: $DEST/tokens/dist/{front,admin}.css"
fi
# กัน design/brand-ds ปนเข้า git โปรเจกต์
if [ -d .git ]; then touch .gitignore; grep -qxF "design/brand-ds" .gitignore || printf "\ndesign/brand-ds\n" >> .gitignore; fi
echo "✓ พร้อมใช้ · หน้าโชว์: $DEST/preview/onemanfleet-ds.html"
