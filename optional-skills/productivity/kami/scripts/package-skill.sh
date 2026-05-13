#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-"$ROOT/dist/kami.zip"}"

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

cd "$ROOT"

MANIFEST="$(mktemp)"
FILTERED_MANIFEST="$(mktemp)"
trap 'rm -f "$MANIFEST" "$FILTERED_MANIFEST"' EXIT

git ls-files > "$MANIFEST"
awk '
  /^assets\/fonts\/TsangerJinKai02-W0[45]\.ttf$/ { next }
  /^assets\/examples\// { next }
  /^assets\/illustrations\// { next }
  /^dist\// { next }
  /^\.vercel\// { next }
  /(^|\/)__pycache__\// { next }
  /\.pyc$/ { next }
  /(^|\/)\.DS_Store$/ { next }
  { print }
' "$MANIFEST" > "$FILTERED_MANIFEST"

zip -q "$OUT" -@ < "$FILTERED_MANIFEST"

if zipinfo -1 "$OUT" | grep -qE 'assets/fonts/TsangerJinKai02-W0[45]\.ttf$'; then
  echo "ERROR: bundled TsangerJinKai02 TTF found in $OUT" >&2
  exit 1
fi

echo "OK: wrote $OUT"
