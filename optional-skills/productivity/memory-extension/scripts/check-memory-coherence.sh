#!/usr/bin/env bash
# Verifies index <-> extended/ file coherence for the extended memory system.
# Usage: bash check-memory-coherence.sh [HERMES_HOME]
#   - orphan           : extended/*.md not referenced in MEMORY.md/USER.md
#   - dangling ref     : index points to extended/<file>.md that does not exist
# Exit 0 if coherent, 1 otherwise.
set -u
HERMES_HOME="${1:-${HERMES_HOME:-$HOME/.hermes}}"
MEM_DIR="$HERMES_HOME/memories"
EXT_DIR="$MEM_DIR/extended"

if [ ! -d "$EXT_DIR" ]; then
  echo "ERROR: $EXT_DIR not found (HERMES_HOME=$HERMES_HOME)"
  exit 1
fi

if [ ! -f "$MEM_DIR/MEMORY.md" ] && [ ! -f "$MEM_DIR/USER.md" ]; then
  echo "ERROR: no index file in $MEM_DIR (MEMORY.md / USER.md missing)"
  exit 1
fi

orphans=0
dangling=0

# 1. extended/ files not referenced in any index
while IFS= read -r f; do
  name=$(basename "$f")
  if ! grep -q "extended/$name" "$MEM_DIR/MEMORY.md" "$MEM_DIR/USER.md" 2>/dev/null; then
    echo "⚠️  Orphan: extended/$name is not referenced in any index"
    orphans=$((orphans + 1))
  fi
done < <(find "$EXT_DIR" -maxdepth 1 -name '*.md' ! -name 'README.md' | sort)

# 2. Index references to missing files
for idx in "$MEM_DIR/MEMORY.md" "$MEM_DIR/USER.md"; do
  [ -f "$idx" ] || continue
  while read -r target; do
    [ -z "$target" ] && continue
    if [ ! -f "$EXT_DIR/$target" ]; then
      echo "⚠️  Dangling ref: $(basename "$idx") -> extended/$target missing"
      dangling=$((dangling + 1))
    fi
  done < <(grep -oE 'extended/[^[:space:])";,]*\.md' "$idx" | sed 's|extended/||' | sort -u)
done

echo "---"
if [ "$orphans" -eq 0 ] && [ "$dangling" -eq 0 ]; then
  total=$(find "$EXT_DIR" -maxdepth 1 -name '*.md' ! -name 'README.md' | wc -l)
  echo "OK: $total extended/ file(s), all referenced"
  exit 0
else
  echo "FAIL: $orphans orphan(s), $dangling dangling reference(s)"
  exit 1
fi
