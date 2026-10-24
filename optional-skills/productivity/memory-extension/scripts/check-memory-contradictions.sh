#!/usr/bin/env bash
# Deterministic contradiction detection in the extended memory system.
# Usage: bash check-memory-contradictions.sh [HERMES_HOME]
# Detects (no LLM):
#   1. Duplicate keys: same "Key: value" subject with different values
#   2. Strong negations: "never/do not/always" lines (candidates for LLM pass)
#   3. Multiple versions: same tool with different vX.Y
#   4. Multiple dates: same line with several JJ/MM/AA dates
# Exit 0 if no candidate, 1 otherwise. Output = candidates (LLM pass confirms).
set -u
# Force C locale: the [A-Za-zÀ-ÿ0-9] ranges break grep on UTF-8 locales
# ("grep: Fin d'intervalle invalide" / "invalid range end") — bug report VM res-89.
export LC_ALL=C
HERMES_HOME="${1:-${HERMES_HOME:-$HOME/.hermes}}"
MEM_DIR="$HERMES_HOME/memories"
EXT_DIR="$MEM_DIR/extended"

if [ ! -d "$EXT_DIR" ]; then
  echo "ERROR: $EXT_DIR not found (HERMES_HOME=$HERMES_HOME)"
  exit 1
fi

# Concatenate index + detail files into a temp stream with source markers
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

for idx in "$MEM_DIR/MEMORY.md" "$MEM_DIR/USER.md"; do
  [ -f "$idx" ] || continue
  echo "### $(basename "$idx")" >> "$TMP"
  cat "$idx" >> "$TMP"
done
for f in "$EXT_DIR"/*.md; do
  [ -f "$f" ] || continue
  [ "$(basename "$f")" = "README.md" ] && continue
  echo "### extended/$(basename "$f")" >> "$TMP"
  cat "$f" >> "$TMP"
done

found=0

# ── 1. Duplicate keys: "Key: value" appearing with different values ──
declare -A seen_keys
while IFS= read -r line; do
  case "$line" in
    "### "*) continue ;;
  esac
  if [[ "$line" =~ ^[[:space:]]*([A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9 _-]{1,40}):[[:space:]]*(.*)$ ]]; then
    key="${BASH_REMATCH[1]}"
    val="${BASH_REMATCH[2]}"
    norm=$(echo "$key" | tr '[:upper:]' '[:lower:]' | tr -s ' _' '__')
    [ -z "$norm" ] && continue
    # Skip overly generic keys (false positives)
    case "$norm" in
      *"see"*|*"example"*|*"note"*|*"todo"*|*"remark"*) continue ;;
    esac
    if [ -n "${seen_keys[$norm]:-}" ] && [ "${seen_keys[$norm]}" != "$val" ]; then
      echo "WARN duplicate key: \"$key\" -> \"${seen_keys[$norm]}\" vs \"$val\""
      found=$((found + 1))
    else
      seen_keys[$norm]="$val"
    fi
  fi
done < "$TMP"

# ── 2. Strong negations: lines with never/do not/always markers ──
# Heuristic: flag lines with strong negative markers for manual/LLM review.
neg_markers='never|do not|don.t|forbidden|stop|without |no '
while IFS= read -r line; do
  case "$line" in
    "### "*) continue ;;
  esac
  if echo "$line" | grep -qiE "$neg_markers"; then
    echo "CANDIDATE strong negation: $line"
    found=$((found + 1))
  fi
done < "$TMP"

# ── 3. Multiple versions: same tool with different vX.Y ──
declare -A ver_map
while IFS= read -r line; do
  case "$line" in
    "### "*) continue ;;
  esac
  while read -r m; do
    [ -z "$m" ] && continue
    tool=$(echo "$m" | sed -E 's/^(.*[[:space:]])?v?[0-9]+\.[0-9]+.*$/\1/' | tr -s ' ' | sed 's/[[:space:]]*$//')
    ver=$(echo "$m" | grep -oE 'v?[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
    [ -z "$tool" ] && continue
    key=$(echo "$tool" | tr '[:upper:]' '[:lower:]')
    if [ -n "${ver_map[$key]:-}" ] && [ "${ver_map[$key]}" != "$ver" ]; then
      echo "WARN multiple versions: $tool -> ${ver_map[$key]} vs $ver"
      found=$((found + 1))
    else
      ver_map[$key]="$ver"
    fi
  done < <(echo "$line" | grep -oE '[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9 _-]{1,30}[[:space:]]+v?[0-9]+\.[0-9]+(\.[0-9]+)?')
done < "$TMP"

# ── 4. Multiple dates: same line with several JJ/MM/AA dates ──
while IFS= read -r line; do
  case "$line" in
    "### "*) continue ;;
  esac
  dates=$(echo "$line" | grep -oE '[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}' | sort -u | tr '\n' ' ')
  n=$(echo "$dates" | wc -w)
  if [ "$n" -ge 2 ]; then
    echo "CANDIDATE multiple dates: $line"
    found=$((found + 1))
  fi
done < "$TMP"

echo "---"
if [ "$found" -eq 0 ]; then
  echo "OK: no contradiction candidate detected (deterministic pass)"
  exit 0
else
  echo "FOUND: $found candidate(s) — run the LLM pass to confirm:"
  echo "   python check-memory-contradictions-llm.py \"$HERMES_HOME\""
  exit 1
fi
