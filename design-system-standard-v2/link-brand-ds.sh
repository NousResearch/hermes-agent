#!/usr/bin/env bash
# link-brand-ds — ต่อ Design System กลางเข้าโปรเจกต์นี้ (ฉลาด: local=symlink · ที่อื่น=clone)
# ใช้: รันที่รากโปรเจกต์  →  bash link-brand-ds.sh
set -euo pipefail

# หา master จากหลายที่ (ไม่ฝัง path เครื่องเดียว) · ตั้ง DS_MASTER=<path> เองได้
MASTER="${DS_MASTER:-}"
for c in \
  "/Users/rattanasak/Documents/Viber Project/Tech Tools/Hermes Agent/design-system-standard-v2" \
  "$HOME/projects/hermes-agent/main/design-system-standard-v2" \
  "$HOME/SynerryTools/hermes-agent/main/design-system-standard-v2"; do
  [ -z "$MASTER" ] && [ -d "$c" ] && MASTER="$c"
done
# เครื่องที่ไม่มี master → clone ชุดจริงจาก GitHub (ชุดกลางอยู่ที่นี่ · GitLab เดิมเป็น mirror เก่า)
HUB="https://github.com/rattanasak-ops/hermes-agent.git"
HUB_SUBDIR="design-system-standard-v2"
DEST="design/brand-ds"

mkdir -p design
if [ -e "$DEST" ] || [ -L "$DEST" ]; then rm -rf "$DEST"; fi

if [ -d "$MASTER" ]; then
  # เครื่องเจ้าของ: symlink ไป master → แก้ที่เดียวเห็นทุกโปรเจกต์ทันที ไม่ต้อง pull
  ln -s "$MASTER" "$DEST"
  echo "✓ ต่อแบบ symlink → $MASTER  (แก้ master = ทุกโปรเจกต์เปลี่ยนทันที)"
else
  # เครื่องอื่น/VPS: ดึงชุดจริงจาก GitHub (ชุดอยู่ subfolder ของ repo hermes-agent)
  echo "ไม่พบ master ในเครื่อง → ดึงชุดจาก GitHub (hermes-agent)"
  TMP="$(mktemp -d)"
  if ! git clone --depth 1 "$HUB" "$TMP" 2>/dev/null; then
    echo "✗ clone ไม่ได้ (เครื่องนี้เข้า GitHub ไม่ได้ หรือ repo private)"
    echo "  → แจ้งเจ้าของ หรือ copy โฟลเดอร์ design-system-standard-v2 มาไว้ที่ candidate path ใด path หนึ่ง"
    rm -rf "$TMP"; exit 1
  fi
  cp -r "$TMP/$HUB_SUBDIR" "$DEST"
  rm -rf "$TMP"
  echo "✓ ดึงชุดจาก GitHub แล้ว"
fi

# build token ให้พร้อมใช้
if command -v node >/dev/null 2>&1; then
  ( cd "$DEST/tokens" && node build-tokens.mjs >/dev/null ) && echo "✓ build token: $DEST/tokens/dist/{front,admin}.css"
fi
# กัน design/brand-ds ปนเข้า git โปรเจกต์
if [ -d .git ]; then touch .gitignore; grep -qxF "design/brand-ds" .gitignore || printf "\ndesign/brand-ds\n" >> .gitignore; fi
echo "✓ พร้อมใช้ · หน้าโชว์: $DEST/preview/onemanfleet-ds.html"
