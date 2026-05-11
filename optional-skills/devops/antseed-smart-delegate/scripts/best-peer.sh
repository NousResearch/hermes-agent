#!/usr/bin/env bash
# antseed-smart-delegate/best-peer.sh — Find optimal AntSeed peer+model for a task type
# Usage: bash best-peer.sh <task_type> [--json] [--peer <peer-id>]
#   task_type: code | research | vision | chat | cheap | any
# Output: JSON to stdout, human-readable summary to stderr
set -uo pipefail

TASK_TYPE="${1:-any}"
JSON_ONLY=false
TARGET_PEER=""
shift 2>/dev/null || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json) JSON_ONLY=true; shift ;;
    --peer) TARGET_PEER="$2"; shift 2 ;;
    *) shift ;;
  esac
done

ANTSEED_BIN="$(command -v antseed 2>/dev/null || echo /home/ubuntu/.hermes/node/bin/antseed)"
PROXY_URL="http://127.0.0.1:8377"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# === Scoring function ===
score_model() {
  local tags="$1" is_free="$2" score=0
  [[ "$is_free" == "true" ]] && { score=$((score+20)); [[ "$TASK_TYPE" == "cheap" ]] && score=$((score+30)); }
  case ",$tags," in
    *,coding,*)           [[ "$TASK_TYPE" == "code" ]]     && score=$((score+10)) ;;
    *,reasoning,*)        [[ "$TASK_TYPE" == "code" ]]     && score=$((score+7))  ;;
    *,reasoning,*)        [[ "$TASK_TYPE" == "research" ]] && score=$((score+10)) ;;
    *,vision,*)           [[ "$TASK_TYPE" == "vision" ]]   && score=$((score+10)) ;;
    *,multimodal,*)       [[ "$TASK_TYPE" == "vision" ]]   && score=$((score+9))  ;;
    *,fast,*)             [[ "$TASK_TYPE" == "chat" ]]     && score=$((score+10)) ;;
    *,chat,*)             [[ "$TASK_TYPE" == "chat" ]]     && score=$((score+9))  ;;
    *,cheap,*|*,free,*)   [[ "$TASK_TYPE" == "cheap" ]]    && score=$((score+10)) ;;
    *,chat,*|*,code,*|*,reasoning,*) [[ "$TASK_TYPE" == "any" ]] && score=$((score+5)) ;;
  esac
  echo "$score"
}

# === Collect peers and models ===
collect_peers() {
  local out_file="$TMPDIR/models.tsv"
  : > "$out_file"

  # Attempt network browse
  local raw_browse
  raw_browse=$("$ANTSEED_BIN" network browse --top 30 2>/dev/null) || true

  if [[ -n "$raw_browse" ]]; then
    # Use Python for robust peer extraction (avoid bash Unicode issues)
    python3 -c "
import sys, json
data = sys.stdin.read()
lines = data.split('\n')
peers = []
for line in lines:
    # Match lines with peer IDs
    import re
    m = re.search(r'([0-9a-fA-F]{40,})', line)
    if m:
        peers.append(m.group(1))
print('\n'.join(peers))
" <<< "$raw_browse" > "$TMPDIR/peers.txt" 2>/dev/null || true
  fi

  # If no peers from browse, try proxy /v1/models
  if [[ ! -s "$TMPDIR/peers.txt" ]]; then
    curl -sf --max-time 5 "$PROXY_URL/v1/models" \
      -H "Authorization: Bearer antseed-p2p" 2>/dev/null \
      | python3 -c "
import json, sys
try:
    for m in json.load(sys.stdin).get('data', []):
        mid = m['id']
        print(f'0\\tnone\\tunknown\\t{mid}\\t0.00\\t0.00\\tunknown\\t\\tfalse\\t0')
    if not json.load(sys.stdin).get('data'):
        print('{\"error\":\"no models from proxy\"}')
except: pass
" > "$out_file" 2>/dev/null || true

    if [[ -s "$out_file" ]]; then
      return 0
    fi
    return 1
  fi

  # Filter by target peer if specified
  if [[ -n "$TARGET_PEER" ]]; then
    grep -F "$TARGET_PEER" "$TMPDIR/peers.txt" > "$TMPDIR/peers_filtered.txt" 2>/dev/null || true
    if [[ -s "$TMPDIR/peers_filtered.txt" ]]; then
      mv "$TMPDIR/peers_filtered.txt" "$TMPDIR/peers.txt"
    fi
  fi

  # Fetch details per peer
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue

    local peer_detail
    peer_detail=$("$ANTSEED_BIN" network peer "$pid" 2>/dev/null) || continue
    [[ -z "$peer_detail" ]] && continue

    local pname pscore
    pname=$(echo "$peer_detail" | python3 -c "
import sys
for line in sys.stdin:
    if 'Display name' in line or 'display_name' in line.lower():
        parts = line.split('\u2502')
        if len(parts) >= 3:
            print(parts[2].strip())
            break
" 2>/dev/null || echo "unknown")

    pscore=$(python3 -c "
import sys, re
data = sys.stdin.read()
# Extract peer score from browse output
m = re.search(r'$pid.*?\|\s*([\d.]+)', data, re.DOTALL)
if m: print(m.group(1))
else: print('0')
" <<< "$raw_browse" 2>/dev/null || echo "0")

    # Parse each service line from peer detail
    while IFS= read -r svc; do
      [[ -z "$svc" ]] && continue
      # Must contain pricing (has "in" and "$")
      [[ "$svc" == *"in"* && "$svc" == *'$'* ]] || continue

      local model pin pout protocol tags is_free mscore

      # Extract model name (first token)
      model=$(echo "$svc" | awk '{print $1}' 2>/dev/null || true)
      [[ -z "$model" ]] && continue

      # Price extraction via awk (avoids $ escaping issues)
      pin=$(echo "$svc" | awk '{for(i=1;i<=NF;i++) if($i=="in"){gsub(/[^0-9.]/,"",$(i+1));print $(i+1);exit}}' || echo "999")
      pout=$(echo "$svc" | awk '{for(i=1;i<=NF;i++) if($i=="out"){gsub(/[^0-9.]/,"",$(i+1));print $(i+1);exit}}' || echo "999")

      # Protocol extraction
      protocol=$(echo "$svc" | python3 -c "
import sys
for line in sys.stdin:
    if 'protocol' in line.lower():
        parts = line.split()
        for i, p in enumerate(parts):
            if 'protocol' in p.lower():
                val = parts[i+1] if i+1 < len(parts) else 'unknown'
                print(val.strip('\\n\\r ,'))
                break
" 2>/dev/null || echo "unknown")

      # Tags extraction
      tags=$(echo "$svc" | python3 -c "
import sys
for line in sys.stdin:
    if 'tags:' in line.lower():
        parts = line.split('tags:')
        if len(parts) > 1:
            print(parts[1].strip().replace(' ',''))
            break
" 2>/dev/null || echo "")

      is_free="false"
      [[ "$pin" == "0" && "$pout" == "0" ]] && is_free="true"
      [[ "$svc" == *"[Ff]ree"* ]] && is_free="true"

      # Skip openai-responses (requires streaming — breaks auxiliaries)
      [[ "$protocol" == "openai-responses" && "$TASK_TYPE" != "any" ]] && continue

      mscore=$(score_model "$tags" "$is_free")

      # Price penalty
      local pnum="${pin%.*}"
      pnum="${pnum:-0}"
      if [[ "$pnum" -gt 0 ]] 2>/dev/null; then
        [[ "$pnum" -gt 10 ]] 2>/dev/null && pnum=10
        mscore=$((mscore - pnum))
      fi
      [[ "$mscore" -lt 0 ]] && mscore=0

      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$mscore" "$pid" "$pname" "$model" "$pin" "$pout" "$protocol" "$tags" "$is_free" "$pscore" >> "$out_file"
    done <<< "$peer_detail"
  done < "$TMPDIR/peers.txt"
}

# === Main ===
main() {
  collect_peers

  local out_file="$TMPDIR/models.tsv"

  if [[ ! -s "$out_file" ]]; then
    # Final fallback: proxy /v1/models
    local proxy_data
    proxy_data=$(curl -sf --max-time 5 "$PROXY_URL/v1/models" \
      -H "Authorization: Bearer antseed-p2p" 2>/dev/null | python3 -c "
import json, sys
try:
    for m in json.load(sys.stdin).get('data', []):
        print(f'0\\tnone\\tunknown\\t{m[\"id\"]}\\t0.00\\t0.00\\tunknown\\t\\tfalse\\t0')
except: pass
" 2>/dev/null || true)
    if [[ -n "$proxy_data" ]]; then
      printf '%s\n' "$proxy_data" > "$out_file"
    else
      echo '{"error":"No peers or models found","task_type":"'"$TASK_TYPE"'"}'
      return 2
    fi
  fi

  # Sort by score descending
  sort -t$'\t' -k1 -rn "$out_file" > "$TMPDIR/sorted.tsv"

  # Read into arrays
  local -a r_peer r_name r_model r_pin r_pout r_proto r_tags r_free seen_peers
  while IFS=$'\t' read -r score pid pname model pin pout protocol tags is_free pscore; do
    [[ -z "$model" ]] && continue
    r_peer+=("$pid"); r_name+=("$pname"); r_model+=("$model")
    r_pin+=("$pin"); r_pout+=("$pout"); r_proto+=("$protocol")
    r_tags+=("$tags"); r_free+=("$is_free")

    local found=false
    for sp in "${seen_peers[@]:-}"; do [[ "$sp" == "$pid" ]] && found=true && break; done
    [[ "$found" != true ]] && seen_peers+=("$pid")
  done < "$TMPDIR/sorted.tsv"

  local n=${#r_model[@]}
  [[ "$n" -eq 0 ]] && echo '{"error":"no models matched"}' && return 2

  # Build JSON
  local i alt fb
  alt=""; fb=""
  for i in $(seq 1 $(( n < 6 ? n : 6 )) ); do
    [[ "$i" -ge "$n" ]] && break
    [[ -n "$alt" ]] && alt="$alt,"
    alt="$alt{\"peer_id\":\"${r_peer[$i]}\",\"peer_name\":\"${r_name[$i]}\",\"model\":\"${r_model[$i]}\",\"price_in\":\"\$${r_pin[$i]}/1M\",\"free\":${r_free[$i]},\"protocol\":\"${r_proto[$i]}\"}"
  done
  for sp in "${seen_peers[@]:0:5}"; do
    [[ -n "$fb" ]] && fb="$fb,"
    fb="$fb\"$sp\""
  done

  local disp_tags
  disp_tags=$(echo "${r_tags[0]}" | sed 's/,/, /g')

  cat << JSONEOF
{
  "task_type": "$TASK_TYPE",
  "total_candidates": $n,
  "unique_peers": ${#seen_peers[@]},
  "recommended": {
    "peer_id": "${r_peer[0]}",
    "peer_name": "${r_name[0]}",
    "model": "${r_model[0]}",
    "price_in": "\$${r_pin[0]}/1M",
    "price_out": "\$${r_pout[0]}/1M",
    "protocol": "${r_proto[0]}",
    "tags": "$disp_tags",
    "free": ${r_free[0]}
  },
  "alternatives": [$alt],
  "fallback_chain": [$fb]
}
JSONEOF

  # Human-readable summary (stderr)
  if [[ "$JSON_ONLY" != true ]]; then
    echo ""
    echo "🐝 Best peer for '$TASK_TYPE': ${r_name[0]}"
    echo "   Model: ${r_model[0]} (\$${r_pin[0]}/\$${r_pout[0]} per 1M tokens)"
    [[ "${r_free[0]}" == "true" ]] && echo "   ✨ FREE model!"
    echo "   Protocol: ${r_proto[0]}"
    echo "   Tags: ${r_tags[0]:-none}"
    echo ""
    echo "   Alternatives: $((n-1)) more across ${#seen_peers[@]} peers"
    echo "   Fallback chain: ${#seen_peers[@]} unique peers ready"
  fi >&2
}

main "$@"