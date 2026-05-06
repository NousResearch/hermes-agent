#!/usr/bin/env bash
# update_translations.sh
#
# One-command refresh after modifying setup.py:
#   1. Extract new msgids from the source
#   2. Merge into each language .po file
#   3. Compile .mo binaries
#
# Usage:
#   cd hermes_cli/i18n && bash update_translations.sh
#
# To add a new language:
#   mkdir -p <locale>/LC_MESSAGES
#   cp setup.pot <locale>/LC_MESSAGES/setup.po
#   # edit <locale>/LC_MESSAGES/setup.po → set Language: and translate msgstr
#   # add the locale to languages.py
#   bash update_translations.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE="$SCRIPT_DIR/../setup.py"

echo "=== (1) Extracting msgids from setup.py ==="
xgettext -k_ -o "$SCRIPT_DIR/setup.pot" \
  --from-code=UTF-8 --language=Python "$SOURCE"

echo "=== (2) Merging into language .po files ==="
for PO in "$SCRIPT_DIR"/*/LC_MESSAGES/setup.po; do
  LANG_DIR="$(dirname "$(dirname "$PO")")"
  LANG_CODE="$(basename "$LANG_DIR")"
  echo "  $LANG_CODE..."
  msgmerge -U "$PO" "$SCRIPT_DIR/setup.pot" 2>&1 | tail -1
done

echo "=== (3) Compiling .mo files ==="
for PO in "$SCRIPT_DIR"/*/LC_MESSAGES/setup.po; do
  MO="${PO%.po}.mo"
  msgfmt -o "$MO" "$PO"
  echo "  $MO"
done

echo "=== Done ==="
