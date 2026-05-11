#!/usr/bin/env bash
# antseed-smart-delegate/best-peer.sh
# Find optimal AntSeed peer+model for a given task type.
# Usage: bash best-peer.sh <task_type>
#   task_type: code | research | vision | chat | cheap | any
# Output: JSON {recommended, alternatives[], fallback_chain[]}
set -uo pipefail

TASK_TYPE="${1:-any}"
ANTSEED_BIN="$(command -v antseed 2>/dev/null || echo /usr/local/bin/antseed)"
PROXY_URL="http://127.0.0.1:8377"
TMPDIR_TMP=$(mktemp -d)
trap 'rm -rf "$TMPDIR_TMP"' EXIT

# Score model tags for task type → numeric score
score_model() {
  local tags="$1" is_free="$2" score=0
  [ "$is_free" = "true" ] && { score=$((score+20)); [ "$TASK_TYPE" = "cheap" ] && score=$((score+30)); }
  case ",$tags," in
    *,coding*,*|*,code*,*)    [ "$TASK_TYPE" = "code" ]     && score=$((score+10)) ;;
    *,reasoning*,*)           [ "$TASK_TYPE" = "code" ]     && score=$((score+7)) ;;
    *,reasoning*,*)           [ "$TASK_TYPE" = "research" ] && score=$((score+10)) ;;
    *,research*,*)            [ "$TASK_TYPE" = "research" ] && score=$((score+9)) ;;
    *,web-search*,*)          [ "$TASK_TYPE" = "research" ] && score=$((score+8)) ;;
    *,vision*,*)              [ "$TASK_TYPE" = "vision" ]   && score=$((score+10)) ;;
    *,multimodal*,*)          [ "$TASK_TYPE" = "vision" ]   && score=$((score+9)) ;;
    *,fast*,*)                [ "$TASK_TYPE" = "chat" ]    && score=$((score+10)) ;;
    *,chat*,*)                [ "$TASK_TYPE" = "chat" ]    && score=$((score+9)) ;;
    *,cheap*,*|*,free*,*)     [ "$TASK_TYPE" = "cheap" ]   && score=$((score+10)) ;;
    *,chat*,*|*,code*,*|*,reasoning*,*) [ "$TASK_TYPE" = "any" ] && score=$((score+5)) ;;
  esac
  echo $score
}

# Collect all candidate models into $TMPDIR_TMP/models.tsv
# Format: score<TAB>pid<TAB>name<TAB>model<TAB>pin<TAB>pout<TAB>proto<TAB>tags<TAB>free<TAB>pscore
collect_peers() {
  local raw_browse pid pname pscore peer_detail
  raw_browse=$("$ANTSEED_BIN" network browse --top 15 2>/dev/null) || true

  # Extract peer IDs (hex between │ and ✓)
  local pids_file="$TMPDIR_TMP/peers.txt"
  : > "$pids_file"
  echo "$raw_browse" | grep -oP '[0-9a-f]{40,64}(?=\s*✓)' > "$pids_file" || true

  local out_file="$TMPDIR_TMP/models.tsv"
  : > "$out_file"

  while IFS= read -r pid; do
    [ -z "$pid" ] && continue
    peer_detail=$("$ANTSEED_BIN" network peer "$pid" 2>/dev/null) || true
    pname=$(echo "$peer_detail" | grep -i "Display name" | sed 's/.*:[[:space:]]*//' | sed 's/^ *//;s/ *$//' || echo "unknown")
    pscore=$(echo "$raw_browse" | grep "$pid" | grep -oE '\|[ ]*[0-9.]+ ' | tail -1 | grep -oE '[0-9.]+' || echo "0")

    # Parse each line of peer detail — look for pricing pattern inline
    while IFS= read -r svc; do
      # Glob match for pricing line — must contain "in" and literal "$"
      [[ "$svc" == *"in"* && "$svc" == *'$'* ]] || continue

      local model pin pout protocol tags is_free mscore pnum
      model=$(echo "$svc" | awk '{print $1}' || true)
      [ -z "$model" ] && continue

      # Extract price using awk (avoids grep $ escaping issues)
      pin=$(echo "$svc" | awk '{for(i=1;i<=NF;i++) if($i=="in"){gsub(/[^0-9.]/,"",$(i+1));print $(i+1);exit}}' || echo "999")
      pout=$(echo "$svc" | awk '{for(i=1;i<=NF;i++) if($i=="out"){gsub(/[^0-9.]/,"",$(i+1));print $(i+1);exit}}' || echo "999")
      protocol=$(echo "$svc" | grep -oP 'protocols:\s*\K\S+' || echo "unknown")
      tags=$(echo "$svc" | grep -oP 'tags:\s*\K.*' | tr -d ' \t\n\r' || echo "")

      is_free="false"
      [ "$pin" = "0" ] && [ "$pout" = "0" ] && is_free="true"
      [[ "$svc" == *"[Ff]ree"* ]] && is_free="true"

      # Skip openai-responses (requires streaming)
      [ "$protocol" = "openai-responses" ] && [ "$TASK_TYPE" != "any" ] && continue

      mscore=$(score_model "$tags" "$is_free")

      # Price penalty (lower = better)
      pnum=${pin%.*}; pnum=${pnum:-0}
      if [ "$pnum" -gt 0 ] 2>/dev/null; then
        [ "$pnum" -gt 10 ] 2>/dev/null && pnum=10
        mscore=$((mscore - pnum))
      fi
      [ "$mscore" -lt 0 ] && mscore=0

      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$mscore" "$pid" "$pname" "$model" "$pin" "$pout" "$protocol" "$tags" "$is_free" "$pscore" >> "$out_file"
    done <<< "$peer_detail"
  done < "$pids_file"
}

main() {
  collect_peers

  local out_file="$TMPDIR_TMP/models.tsv"

  if [ ! -s "$out_file" ]; then
    # Fallback: try proxy /v1/models endpoint
    local proxy_data
    proxy_data=$(curl -sf --max-time 5 "$PROXY_URL/v1/models" \
      -H "Authorization: Bearer antseed-p2p" 2>/dev/null | python3 -c "
import json,sys
try:
  for m in json.load(sys.stdin).get('data',[]):
    print('0\tnone\tunknown\t%s\t0.00\t0.00\tunknown\t\tfalse\t0' % m['id'])
except: pass
" 2>/dev/null || true)

    if [ -n "$proxy_data" ]; then
      printf '%s\n' "$proxy_data" > "$out_file"
    else
      printf '{"error":"No peers or models found","task_type":"%s"}\n' "$TASK_TYPE"
      return 2
    fi
  fi

  # Sort by score descending
  local sorted_file="$TMPDIR_TMP/sorted.tsv"
  sort -t$'\t' -k1 -rn "$out_file" > "$sorted_file"

  # Parse into arrays
  local -a r_peer r_name r_model r_pin r_pout r_proto r_tags r_free seen_peers

  while IFS=$'\t' read -r score pid pname model pin pout protocol tags is_free pscore; do
    [ -z "$model" ] && continue
    r_peer+=("$pid"); r_name+=("$pname"); r_model+=("$model")
    r_pin+=("$pin"); r_pout+=("$pout"); r_proto+=("$protocol")
    r_tags+=("$tags"); r_free+=("$is_free")

    local found=false
    for sp in "${seen_peers[@]:-}"; do [ "$sp" = "$pid" ] && found=true && break; done
    [ "$found" != true ] && seen_peers+=("$pid")
  done < "$sorted_file"

  local n=${#r_model[@]}
  [ "$n" -eq 0 ] && printf '{"error":"no models matched"}\n' && return 2

  # Build JSON
  local i alt fb
  alt="" ; fb=""
  for i in $(seq 1 $(( n < 6 ? n : 6 )) ); do
    [ "$i" -ge "$n" ] && break
    [ -n "$alt" ] && alt="$alt,"
    alt="$alt{\"peer_id\":\"${r_peer[$i]}\",\"peer_name\":\"${r_name[$i]}\",\"model\":\"${r_model[$i]}\",\"price_in\":\"\$${r_pin[$i]}/1M\",\"free\":${r_free[$i]}}"
  done
  for sp in "${seen_peers[@]:0:5}"; do
    [ -n "$fb" ] && fb="$fb,"
    fb="$fb\"$sp\""
  done

  local disp_tags
  disp_tags="$(echo "${r_tags[0]}" | sed 's/,/, /g')"

  cat <<JSONEOF
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

  {
    echo ""
    echo "🐝 Best peer for '$TASK_TYPE': ${r_name[0]}"
    echo "   Model: ${r_model[0]} (\$${r_pin[0]}/\$${r_pout[0]} per 1M tokens)"
    [ "${r_free[0]}" = "true" ] && echo "   ✨ FREE model!"
    echo "   Protocol: ${r_proto[0]}"
    echo "   Tags: ${r_tags[0]:-none}"
    echo ""
    echo "   Alternatives: $((n-1)) more across ${#seen_peers[@]} peers"
    echo "   Fallback chain: ${#seen_peers[@]} unique peers ready"
  } >&2
}

main "$@"
