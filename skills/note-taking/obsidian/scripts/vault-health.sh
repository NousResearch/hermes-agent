#!/usr/bin/env bash
# Hermes Obsidian Vault diagnostics and utilities
# Resolves OBSIDIAN_VAULT_PATH from env, then reports vault health.
set -euo pipefail

VAULT="${OBSIDIAN_VAULT_PATH:-${HOME}/Documents/Obsidian Vault}"
if [ ! -d "$VAULT" ]; then
  # Fallback for Linux servers
  [ -d /opt/data/vault ] && VAULT=/opt/data/vault
fi

if [ ! -d "$VAULT" ]; then
  echo "Vault not found at $VAULT. Set OBSIDIAN_VAULT_PATH."
  exit 1
fi

CMD="${1:-health}"

case "$CMD" in
  health|status)
    echo "=== Obsidian Vault Health ==="
    echo "Path: $VAULT"
    NOTES=$(find "$VAULT" -name '*.md' -type f 2>/dev/null | wc -l)
    DIRS=$(find "$VAULT" -type d 2>/dev/null | wc -l)
    SIZE=$(du -sh "$VAULT" 2>/dev/null | cut -f1)
    echo "Notes: $NOTES"
    echo "Directories: $DIRS"
    echo "Size: $SIZE"
    if [ -f "$VAULT/.obsidian/app.json" ]; then
      echo "Config: OK"
    else
      echo "Config: MISSING (.obsidian/app.json)"
    fi
    echo ""
    echo "=== Projects ==="
    for d in "$VAULT"/*/; do
      name=$(basename "$d")
      if [ "$name" != ".obsidian" ] && [ "$name" != "assets" ] && [ -d "$d" ]; then
        count=$(find "$d" -name '*.md' -type f 2>/dev/null | wc -l)
        echo "  $name/ ($count notes)"
      fi
    done
    echo ""
    echo "=== Recent Notes ==="
    find "$VAULT" -name '*.md' -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -10 | while read -r ts path; do
      rel="${path#$VAULT/}"
      echo "  $rel"
    done
    ;;
  list)
    find "$VAULT" -name '*.md' -type f 2>/dev/null | while read -r f; do
      echo "${f#$VAULT/}"
    done
    ;;
  search)
    shift
    grep -rl "${1:-}" "$VAULT" --include='*.md' 2>/dev/null | while read -r f; do
      echo "${f#$VAULT/}"
    done
    ;;
  tags)
    grep -roh 'tags:\s*\[[^]]*\]' "$VAULT" --include='*.md' 2>/dev/null | sort | uniq -c | sort -rn | head -20
    ;;
  wikilinks)
    echo "=== Broken Wikilinks ==="
    for f in $(find "$VAULT" -name '*.md' -type f 2>/dev/null); do
      grep -oP '\[\[([^\]|]+)' "$f" 2>/dev/null | sed 's/\[\[//' | while read -r link; do
        # Check if the linked note exists
        link_note="$VAULT/${link}.md"
        if [ ! -f "$link_note" ]; then
          # Try as folder/note
          dir=$(dirname "$f")
          link_note2="$dir/${link}.md"
          if [ ! -f "$link_note2" ]; then
            echo "  MISSING: [[$link]] from ${f#$VAULT/}"
          fi
        fi
      done
    done
    ;;
  path)
    echo "$VAULT"
    ;;
  *)
    echo "Usage: $(basename "$0") {health|list|search <term>|tags|wikilinks|path}"
    exit 1
    ;;
esac
