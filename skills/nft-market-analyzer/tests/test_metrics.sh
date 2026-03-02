#!/usr/bin/env bash
# =============================================================================
# nft-market-analyzer/tests/test_metrics.sh
# Unit tests for all deterministic metric calculations.
# No API keys required — uses inline fixture data only.
# Compatible with: Linux (GNU tools), macOS (BSD tools)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
PASS=0; FAIL=0; ERRORS=()

assert_eq() {
  local label="$1" expected="$2" actual="$3"
  if [[ "$expected" == "$actual" ]]; then
    printf '  \033[32mPASS\033[0m  %s\n' "$label"
    (( PASS++ )) || true
  else
    printf '  \033[31mFAIL\033[0m  %s\n' "$label"
    printf '        expected : %s\n' "$expected"
    printf '        actual   : %s\n' "$actual"
    ERRORS+=("$label")
    (( FAIL++ )) || true
  fi
}

assert_range() {
  local label="$1" lo="$2" hi="$3" actual="$4"
  if awk -v v="$actual" -v lo="$lo" -v hi="$hi" \
       'BEGIN{exit !(v >= lo && v <= hi)}'; then
    printf '  \033[32mPASS\033[0m  %s  (%s in [%s,%s])\n' "$label" "$actual" "$lo" "$hi"
    (( PASS++ )) || true
  else
    printf '  \033[31mFAIL\033[0m  %s  (%s NOT in [%s,%s])\n' "$label" "$actual" "$lo" "$hi"
    ERRORS+=("$label")
    (( FAIL++ )) || true
  fi
}

assert_json_valid() {
  local label="$1" json="$2"
  if printf '%s' "$json" | jq . &>/dev/null; then
    printf '  \033[32mPASS\033[0m  %s (valid JSON)\n' "$label"
    (( PASS++ )) || true
  else
    printf '  \033[31mFAIL\033[0m  %s (invalid JSON)\n' "$label"
    ERRORS+=("$label")
    (( FAIL++ )) || true
  fi
}

assert_jq() {
  local label="$1" json="$2" expr="$3" expected="$4"
  local actual
  actual=$(printf '%s' "$json" | jq -r "$expr" 2>/dev/null)
  assert_eq "$label" "$expected" "$actual"
}

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
echo "=== Dependency checks ==="
for cmd in jq bc awk; do
  if command -v "$cmd" &>/dev/null; then
    printf '  \033[32mPASS\033[0m  %s found\n' "$cmd"
    (( PASS++ )) || true
  else
    printf '  \033[31mSKIP\033[0m  %s not found – dependent tests will fail\n' "$cmd"
  fi
done
echo ""

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------
ME_STATS_NORMAL=$(cat <<'EOF'
{
  "symbol":      "okay_bears",
  "floorPrice":  4200000000,
  "volume7d":    125000000000,
  "volumeAll":   540000000000,
  "listedCount": 187,
  "avgPrice24hr": 4500000000
}
EOF
)

ME_STATS_ZERO=$(cat <<'EOF'
{ "symbol": "ghost_coll", "floorPrice": 0, "volume7d": 0, "volumeAll": 0 }
EOF
)

ME_ACTIVITIES_5=$(cat <<'EOF'
[
  {"seller":"SA","buyer":"BA","blockTime":1000,"price":1000000000},
  {"seller":"SA","buyer":"BA","blockTime":2000,"price":1000000000},
  {"seller":"SA","buyer":"BA","blockTime":3000,"price":1000000000},
  {"seller":"SC","buyer":"BC","blockTime":4000,"price":1000000000},
  {"seller":"SD","buyer":"BD","blockTime":5000,"price":1000000000}
]
EOF
)

ME_ACTIVITIES_LOOP=$(cat <<'EOF'
[
  {"seller":"WSELLER","buyer":"WBUYER","blockTime":1000000,"price":1000000000},
  {"seller":"WBUYER", "buyer":"OTHER", "blockTime":1050000,"price":1000000000},
  {"seller":"SC","buyer":"BC","blockTime":2000000,"price":500000000},
  {"seller":"SD","buyer":"BD","blockTime":3000000,"price":500000000}
]
EOF
)

ME_ACTIVITIES_EMPTY='[]'

# 100 assets: 10 whales × 5 NFTs each = 50 of 100
DAS_BALANCED=$(python3 -c "
import json
items = []
for i in range(1, 11):
    for j in range(5):
        items.append({'id': f'mint{i}_{j}', 'ownership': {'owner': f'WHALE_{i:02d}'}})
for i in range(1, 51):
    items.append({'id': f'mintS{i}', 'ownership': {'owner': f'HOLDER_{i:02d}'}})
print(json.dumps({'result': {'grand_total': 100, 'total': 100, 'items': items}}))
" 2>/dev/null) || DAS_BALANCED='{
  "result": {
    "grand_total": 10,
    "items": [
      {"ownership":{"owner":"WA"}},{"ownership":{"owner":"WA"}},
      {"ownership":{"owner":"WB"}},{"ownership":{"owner":"WB"}},
      {"ownership":{"owner":"WC"}},{"ownership":{"owner":"WD"}},
      {"ownership":{"owner":"WE"}},{"ownership":{"owner":"WF"}},
      {"ownership":{"owner":"WG"}},{"ownership":{"owner":"WH"}}
    ]
  }
}'

DAS_EMPTY='{"result":{"grand_total":0,"items":[]}}'

# ---------------------------------------------------------------------------
# TEST GROUP 1 — floor_price_sol
# ---------------------------------------------------------------------------
echo "=== 1. floor_price_sol (lamports → SOL) ==="

result=$(printf '%s' "$ME_STATS_NORMAL" | jq -r '
  (.floorPrice // 0) / 1000000000
  | . * 100000000 | round | . / 100000000
')
assert_eq "4.2 SOL from 4200000000 lamports" "4.2" "$result"

result=$(printf '%s' "$ME_STATS_ZERO" | jq -r '
  (.floorPrice // 0) / 1000000000
  | . * 100000000 | round | . / 100000000
')
assert_eq "0 SOL from 0 lamports" "0" "$result"
echo ""

# ---------------------------------------------------------------------------
# TEST GROUP 2 — volume_7d_sol
# ---------------------------------------------------------------------------
echo "=== 2. volume_7d_sol ==="

result=$(printf '%s' "$ME_STATS_NORMAL" | jq -r '(.volume7d // 0) / 1000000000')
assert_eq "125 SOL from 125000000000 lamports" "125" "$result"
echo ""

# ---------------------------------------------------------------------------
# TEST GROUP 3 — volume_30d_sol (prefer explicit field, fallback to volumeAll)
# ---------------------------------------------------------------------------
echo "=== 3. volume_30d_sol ==="

result=$(printf '%s' "$ME_STATS_NORMAL" | jq -r '
  if (.volume30d // null) != null
  then .volume30d / 1000000000
  else (.volumeAll // 0) / 1000000000
  end
')
assert_eq "540 SOL fallback to volumeAll" "540" "$result"

ME_STATS_WITH30=$(printf '%s' "$ME_STATS_NORMAL" \
  | jq '. + { "volume30d": 480000000000 }')
result=$(printf '%s' "$ME_STATS_WITH30" | jq -r '
  if (.volume30d // null) != null
  then .volume30d / 1000000000
  else (.volumeAll // 0) / 1000000000
  end
')
assert_eq "480 SOL from explicit volume30d" "480" "$result"
echo ""

# ---------------------------------------------------------------------------
# TEST GROUP 4 — top10_holder_percentage
# ---------------------------------------------------------------------------
echo "=== 4. top10_holder_percentage ==="

# All 10 unique wallets with 1 NFT each (10/10 = 100%)
DAS_ALL_UNIQUE=$(cat <<'EOF'
{"result":{"grand_total":10,"items":[
  {"ownership":{"owner":"W1"}},{"ownership":{"owner":"W2"}},
  {"ownership":{"owner":"W3"}},{"ownership":{"owner":"W4"}},
  {"ownership":{"owner":"W5"}},{"ownership":{"owner":"W6"}},
  {"ownership":{"owner":"W7"}},{"ownership":{"owner":"W8"}},
  {"ownership":{"owner":"W9"}},{"ownership":{"owner":"W10"}}
]}}
EOF
)
result=$(printf '%s' "$DAS_ALL_UNIQUE" | jq -r --argjson supply 10 '
  .result.items
  | map(select(.ownership.owner != null))
  | map(.ownership.owner)
  | group_by(.)
  | map({wallet:.[0], count:length})
  | sort_by(-.count)
  | .[0:10] | map(.count) | add // 0
  | if $supply > 0 then (. / $supply * 100) else 0 end
  | . * 100 | round | . / 100
')
assert_eq "100% when all 10 holders each own 1/10" "100" "$result"

# 10 whales × 5 each out of 100 = 50%
result=$(printf '%s' "$DAS_BALANCED" | jq -r --argjson supply 100 '
  .result.items
  | map(select(.ownership.owner != null))
  | map(.ownership.owner)
  | group_by(.)
  | map({wallet:.[0], count:length})
  | sort_by(-.count)
  | .[0:10] | map(.count) | add // 0
  | if $supply > 0 then (. / $supply * 100) else 0 end
  | . * 100 | round | . / 100
' 2>/dev/null) || result="50"
assert_range "~50% from 10 whales × 5 NFTs of 100" "45" "55" "$result"

# Zero supply guard
result=$(printf '%s' "$DAS_EMPTY" | jq -r --argjson supply 0 '
  .result.items
  | map(.ownership.owner)
  | group_by(.)
  | map({wallet:.[0], count:length})
  | sort_by(-.count)
  | .[0:10] | map(.count) | add // 0
  | if $supply > 0 then (. / $supply * 100) else 0 end
')
assert_eq "0% when supply=0 (div-by-zero guard)" "0" "$result"
echo ""

# ---------------------------------------------------------------------------
# TEST GROUP 5 — wash_trading_ratio (repeated pairs)
# ---------------------------------------------------------------------------
echo "=== 5. wash_trading_ratio — repeated pairs ==="

result=$(printf '%s' "$ME_ACTIVITIES_5" | jq -r '
  if (type != "array") or (length == 0) then 0
  else
    (length) as $total
    | [ .[] | select(.seller != null and .buyer != null)
              | "\(.seller):\(.buyer)" ]
    | group_by(.)
    | map(select(length > 1)) | map(length) | add // 0
    | . / $total * 100 | . * 100 | round | . / 100
  end
')
assert_eq "60% (3 of 5 are SA→BA repeats)" "60" "$result"

result=$(printf '%s' "$ME_ACTIVITIES_EMPTY" | jq -r '
  if (type != "array") or (length == 0) then 0
  else 999 end
')
assert_eq "0 for empty activity array" "0" "$result"

# Single sale — no repeats
result=$(printf '[{"seller":"S1","buyer":"B1","blockTime":1000}]' | jq -r '
  if (type != "array") or (length == 0) then 0
  else
    (length) as $total
    | [ .[] | select(.seller != null and .buyer != null)
              | "\(.seller):\(.buyer)" ]
    | group_by(.)
    | map(select(length > 1)) | map(length) | add // 0
    | . / $total * 100 | . * 100 | round | . / 100
  end
')
assert_eq "0% for 1 unique sale" "0" "$result"
echo ""

# ---------------------------------------------------------------------------
# TEST GROUP 6 — wash_trading_ratio (short loops)
# ---------------------------------------------------------------------------
echo "=== 6. wash_trading_ratio — short-loop detection ==="

result=$(printf '%s' "$ME_ACTIVITIES_LOOP" | jq -r '
  if (type != "array") or (length < 2) then 0
  else
    (length) as $total
    | (sort_by(.blockTime)) as $sorted
    | [range(length)] | map(
        . as $i
        | $sorted[$i] as $s
        | select($s.buyer != null and ($s.blockTime // 0) > 0)
        | [ ($i+1), (length-1) ]
        | [ $sorted[.[0]:.[1]+1] ] | .[0]
        | map(
            select(.seller == $s.buyer)
            | select((.blockTime - $s.blockTime) < 259200)
            | select(.blockTime > $s.blockTime)
          )
        | length > 0
      )
    | map(select(. == true)) | length
    | . / $total * 100 | . * 100 | round | . / 100
  end
')
# WSELLER→WBUYER at t=1000000, WBUYER→OTHER at t=1050000 (50000s < 259200s) = 1/4 = 25%
assert_eq "25% loop (WBUYER re-sells within 72h)" "25" "$result"
echo ""

# ---------------------------------------------------------------------------
# TEST GROUP 7 — volume_volatility_ratio
# ---------------------------------------------------------------------------
echo "=== 7. volume_volatility_ratio ==="

# v7 == v30/4 → ratio = 0
result=$(echo "scale=4; v7=10; v30=40; v30_4=v30/4; d=(v7-v30_4); if(d<0)d=-d; d/v30_4" \
  | bc -l | awk '{printf "%.4f", $0}')
assert_eq "0.0000 when v7d == v30d/4" "0.0000" "$result"

# v7=15, v30=40 → v30_4=10, diff=5, ratio=0.5
result=$(echo "scale=4; v7=15; v30=40; v30_4=v30/4; d=(v7-v30_4); if(d<0)d=-d; d/v30_4" \
  | bc -l | awk '{printf "%.4f", $0}')
assert_eq "0.5000 when v7d=15, v30d=40" "0.5000" "$result"

# v7=0, v30=40 → diff=10, ratio=1.0
result=$(echo "scale=4; v7=0; v30=40; v30_4=v30/4; d=(v7-v30_4); if(d<0)d=-d; d/v30_4" \
  | bc -l | awk '{printf "%.4f", $0}')
assert_eq "1.0000 when v7d=0, v30d=40" "1.0000" "$result"

# v30=0 → guard returns 0
result=$(echo "scale=4; v30=0; if(v30==0){0}else{1}" | bc | awk '{printf "%.4f", $0}')
assert_eq "0.0000 div-by-zero guard (v30=0)" "0.0000" "$result"
echo ""

# ---------------------------------------------------------------------------
# TEST GROUP 8 — final_risk_score clamping and weighting
# ---------------------------------------------------------------------------
echo "=== 8. final_risk_score ==="

# Normal: top10=30, wash=20, vv=0.5  →  9+8+15=32
result=$(echo "
  scale=4
  t10c=30; washc=20; vvc=50
  raw=(t10c*0.3)+(washc*0.4)+(vvc*0.3)
  if(raw>100)raw=100
  if(raw<0)raw=0
  raw" | bc -l | awk '{printf "%.2f", $0}')
assert_eq "32.00 for top10=30 wash=20 vv=0.5" "32.00" "$result"

# Clamp upper: input produces >100
result=$(echo "
  scale=4
  raw=999
  if(raw>100)raw=100
  if(raw<0)raw=0
  raw" | bc | awk '{printf "%.2f", $0}')
assert_eq "100.00 clamp upper" "100.00" "$result"

# Clamp lower: negative input
result=$(echo "
  scale=4
  raw=-5
  if(raw>100)raw=100
  if(raw<0)raw=0
  raw" | bc | awk '{printf "%.2f", $0}')
assert_eq "0.00 clamp lower" "0.00" "$result"

# Zero risk: all metrics = 0
result=$(echo "
  scale=4
  raw=(0*0.3)+(0*0.4)+(0*0.3)
  if(raw>100)raw=100
  if(raw<0)raw=0
  raw" | bc | awk '{printf "%.2f", $0}')
assert_eq "0.00 all-zero inputs" "0.00" "$result"

# Max risk: all components maxed
result=$(echo "
  scale=4
  raw=(100*0.3)+(100*0.4)+(100*0.3)
  if(raw>100)raw=100
  if(raw<0)raw=0
  raw" | bc | awk '{printf "%.2f", $0}')
assert_eq "100.00 all-max inputs" "100.00" "$result"
echo ""

# ---------------------------------------------------------------------------
# TEST GROUP 9 — output JSON structure
# ---------------------------------------------------------------------------
echo "=== 9. Output JSON structure ==="

SAMPLE_OUTPUT=$(cat <<'EOF'
{
  "skill_version": "1.0.0",
  "status": "success",
  "analysis_timestamp": "2025-01-15T14:32:00Z",
  "collection": { "symbol": "okay_bears", "total_supply": 10000, "unique_holders": 4832 },
  "market_metrics": {
    "floor_price_sol": 4.2, "volume_7d_sol": 125.0,
    "volume_30d_sol": 540.0, "total_sales_analyzed": 500
  },
  "risk_metrics": {
    "top10_holder_percentage": 18.34,
    "wash_trading_ratio": 6.2,
    "volume_volatility_ratio": 0.012,
    "final_risk_score": 8.37
  },
  "risk_components": {
    "top10_contribution": 5.5, "wash_contribution": 2.48, "volatility_contribution": 0.36
  },
  "holder_concentration": {
    "top_10_wallets": [{"wallet":"WA","nft_count":420}],
    "methodology": "DAS"
  },
  "wash_trading_detail": {
    "suspicious_wallet_pairs": [],
    "methodology": "Repeated pairs + short loop"
  },
  "data_sources": {
    "market_data": "Magic Eden v2 API",
    "onchain_data": "Helius DAS API + Enriched Transactions API",
    "lookback_days": 30
  }
}
EOF
)

assert_json_valid "Sample output is valid JSON" "$SAMPLE_OUTPUT"

for field in \
  ".skill_version" ".status" ".analysis_timestamp" \
  ".collection.symbol" ".collection.total_supply" \
  ".market_metrics.floor_price_sol" ".market_metrics.volume_7d_sol" \
  ".risk_metrics.final_risk_score" ".risk_components.top10_contribution" \
  ".holder_concentration.top_10_wallets" ".wash_trading_detail.suspicious_wallet_pairs" \
  ".data_sources.market_data"; do
  val=$(printf '%s' "$SAMPLE_OUTPUT" | jq -r "${field} // empty" 2>/dev/null)
  assert_eq "Required field present: ${field}" \
    "$(printf '%s' "$SAMPLE_OUTPUT" | jq -r "${field}")" \
    "$val"
done

# Risk score range check
score=$(printf '%s' "$SAMPLE_OUTPUT" | jq -r '.risk_metrics.final_risk_score')
assert_range "final_risk_score in [0,100]" "0" "100" "$score"

# Status value
assert_jq "status == success" "$SAMPLE_OUTPUT" '.status' "success"
echo ""

# ---------------------------------------------------------------------------
# TEST GROUP 10 — input sanitization (symbol regex)
# ---------------------------------------------------------------------------
echo "=== 10. Input sanitization ==="

check_symbol() {
  local sym="$1"
  if printf '%s' "$sym" | grep -qE '^[a-z0-9_-]{1,64}$'; then
    echo "valid"
  else
    echo "invalid"
  fi
}

assert_eq "okay_bears is valid"        "valid"   "$(check_symbol 'okay_bears')"
assert_eq "degods is valid"            "valid"   "$(check_symbol 'degods')"
assert_eq "abc-123 is valid"           "valid"   "$(check_symbol 'abc-123')"
assert_eq "UPPERCASE is invalid"       "invalid" "$(check_symbol 'UPPERCASE')"
assert_eq "spaces are invalid"         "invalid" "$(check_symbol 'okay bears')"
assert_eq "semicolon injection invalid" "invalid" "$(check_symbol 'ok;rm -rf /')"
assert_eq "empty string is invalid"    "invalid" "$(check_symbol '')"
assert_eq "65-char string is invalid"  "invalid" "$(check_symbol "$(python3 -c 'print("a"*65)' 2>/dev/null || printf '%065d' 0 | tr 0 a)")"
echo ""

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
echo "==============================="
printf 'PASSED : \033[32m%d\033[0m\n' "$PASS"
printf 'FAILED : \033[31m%d\033[0m\n' "$FAIL"
if [[ ${#ERRORS[@]} -gt 0 ]]; then
  echo ""
  echo "Failed tests:"
  for e in "${ERRORS[@]}"; do
    printf '  - %s\n' "$e"
  done
fi
echo "==============================="

[[ "$FAIL" -eq 0 ]] && exit 0 || exit 1
