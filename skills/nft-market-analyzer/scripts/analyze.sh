#!/usr/bin/env bash
# =============================================================================
# nft-market-analyzer/scripts/analyze.sh
# Production Hermes Skill – Solana NFT Collection Risk Analysis
#
# Architecture contract:
#   stdout  → final JSON only (consumed by Hermes)
#   stderr  → structured log lines (consumed by Hermes logging pipeline)
#   exit 0  → success
#   exit 1  → error (structured error JSON emitted to stdout before exit)
#
# Security: read-only. No private keys. No transaction signing. No offers.
# =============================================================================
set -euo pipefail
IFS=$'\n\t'

# ---------------------------------------------------------------------------
# 0. CONSTANTS
# ---------------------------------------------------------------------------
readonly SKILL_VERSION="1.0.0"
readonly MAGICEDEN_API_BASE="https://api-mainnet.magiceden.dev/v2"
readonly HELIUS_API_BASE="https://api.helius.xyz/v0"
readonly HELIUS_RPC_BASE="https://mainnet.helius-rpc.com"

readonly TIMEOUT="${NFT_REQUEST_TIMEOUT:-15}"
readonly MAX_RETRIES="${NFT_MAX_RETRIES:-3}"
readonly HOLDER_SAMPLE="${NFT_HOLDER_SAMPLE_SIZE:-500}"
readonly TRANSFER_DAYS="${NFT_TRANSFER_LOOKBACK_DAYS:-30}"

# Work dir — all temp files live here and are purged on exit
WORK_DIR=$(mktemp -d)
readonly WORK_DIR
trap 'rm -rf "${WORK_DIR}"' EXIT

# Preserve original stdout (fd 3) so we can emit the final JSON after
# redirecting fd 1 to the log for the duration of the script.
exec 3>&1
exec 1>>"${WORK_DIR}/skill.log" 2>&1

# ---------------------------------------------------------------------------
# 1. LOGGING
# ---------------------------------------------------------------------------
_log() { local level="$1"; shift; echo "[${level}] $(date -u +%FT%TZ) $*" >&2; }
log_info()  { _log "INFO " "$@"; }
log_warn()  { _log "WARN " "$@"; }
log_error() { _log "ERROR" "$@"; }

# Emit structured error JSON to original stdout, then exit
die() {
  local message="$1"
  local code="${2:-INTERNAL_ERROR}"
  log_error "FATAL code=${code} msg=${message}"
  printf '%s\n' "$(jq -n \
    --arg ver "$SKILL_VERSION" \
    --arg msg "$message" \
    --arg cod "$code" \
    '{ skill_version:$ver, status:"error", error:{ message:$msg, code:$cod } }'
  )" >&3
  exit 1
}

# ---------------------------------------------------------------------------
# 2. DEPENDENCY CHECKS
# ---------------------------------------------------------------------------
check_deps() {
  local missing=()
  for cmd in curl jq bc awk; do
    command -v "$cmd" &>/dev/null || missing+=("$cmd")
  done
  [[ ${#missing[@]} -eq 0 ]] || die "Missing required tools: ${missing[*]}" "MISSING_DEPENDENCY"

  # jq version check
  local jq_ver
  jq_ver=$(jq --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
  awk -v v="$jq_ver" 'BEGIN{if(v+0 < 1.6){exit 1}}' \
    || die "jq >= 1.6 required, found ${jq_ver}" "DEPENDENCY_VERSION"
}

# ---------------------------------------------------------------------------
# 3. INPUT VALIDATION
# ---------------------------------------------------------------------------
validate_inputs() {
  [[ -n "${MAGICEDEN_API_KEY:-}" ]] || die "MAGICEDEN_API_KEY is not set" "MISSING_ENV_VAR"
  [[ -n "${HELIUS_API_KEY:-}"    ]] || die "HELIUS_API_KEY is not set"    "MISSING_ENV_VAR"

  local symbol="$1"
  [[ -n "$symbol" ]] || die "Usage: $0 <collection_symbol>" "MISSING_INPUT"

  # Guard against shell injection: only lowercase alphanumeric, dash, underscore
  if ! printf '%s' "$symbol" | grep -qE '^[a-z0-9_-]{1,64}$'; then
    die "Invalid collection_symbol '${symbol}': must match [a-z0-9_-]{1,64}" "INVALID_INPUT"
  fi
}

# ---------------------------------------------------------------------------
# 4. HTTP HELPER — GET with retry, rate-limit backoff, no credential leakage
# ---------------------------------------------------------------------------
http_get() {
  local url="$1"
  local out="$2"
  local attempt=0
  local http_code

  while (( attempt < MAX_RETRIES )); do
    (( attempt++ )) || true
    http_code=$(curl \
      --silent --show-error --fail-with-body \
      --max-time "${TIMEOUT}" \
      --retry 0 \
      --write-out "%{http_code}" \
      --output "${out}" \
      -H "Authorization: Bearer ${MAGICEDEN_API_KEY}" \
      "${url}" 2>>"${WORK_DIR}/curl.log") || http_code="000"

    case "$http_code" in
      200) log_info "GET ${url%%\?*} → 200 (attempt ${attempt})"; return 0 ;;
      404) log_warn "GET ${url%%\?*} → 404 (not found)"; echo "null" >"${out}"; return 0 ;;
      429) log_warn "Rate limited. Sleeping $(( attempt * 10 ))s"; sleep $(( attempt * 10 )) ;;
      5*)  log_warn "Server error ${http_code}. Sleeping $(( attempt * 5 ))s"; sleep $(( attempt * 5 )) ;;
      *)   log_warn "Unexpected HTTP ${http_code} for ${url%%\?*}" ;;
    esac
  done
  die "Exhausted ${MAX_RETRIES} retries for ${url%%\?*} (last HTTP ${http_code})" "HTTP_FAILURE"
}

# ---------------------------------------------------------------------------
# 5. HTTP HELPER — POST JSON with retry
# ---------------------------------------------------------------------------
http_post_json() {
  local url="$1"
  local payload="$2"
  local out="$3"
  local attempt=0
  local http_code

  while (( attempt < MAX_RETRIES )); do
    (( attempt++ )) || true
    http_code=$(curl \
      --silent --show-error --fail-with-body \
      --max-time "${TIMEOUT}" \
      --retry 0 \
      --write-out "%{http_code}" \
      --output "${out}" \
      -H "Content-Type: application/json" \
      -d "${payload}" \
      "${url}" 2>>"${WORK_DIR}/curl.log") || http_code="000"

    case "$http_code" in
      200) log_info "POST ${url%%\?*} → 200 (attempt ${attempt})"; return 0 ;;
      429) sleep $(( attempt * 10 )) ;;
      5*)  sleep $(( attempt * 5 )) ;;
      *)   log_warn "Unexpected HTTP ${http_code} for POST ${url%%\?*}" ;;
    esac
  done
  die "Exhausted ${MAX_RETRIES} retries for POST ${url%%\?*}" "HTTP_FAILURE"
}

# ---------------------------------------------------------------------------
# 6. CROSS-PLATFORM EPOCH HELPER
# ---------------------------------------------------------------------------
epoch_days_ago() {
  local days="$1"
  # GNU date (Linux)
  date -u -d "${days} days ago" +%s 2>/dev/null && return
  # BSD date (macOS)
  date -u -v "-${days}d"        +%s 2>/dev/null && return
  die "Cannot compute epoch: GNU or BSD date required" "DEPENDENCY_ERROR"
}

# ---------------------------------------------------------------------------
# 7. FETCH — MAGIC EDEN
# ---------------------------------------------------------------------------
fetch_magic_eden() {
  local symbol="$1"
  log_info "Fetching Magic Eden stats for ${symbol}..."
  http_get \
    "${MAGICEDEN_API_BASE}/collections/${symbol}/stats" \
    "${WORK_DIR}/me_stats.json"

  # Validate: must be an object with floorPrice
  if ! jq -e 'type == "object" and has("floorPrice")' \
       "${WORK_DIR}/me_stats.json" &>/dev/null; then
    die "Unexpected Magic Eden stats response for '${symbol}'. Check the collection symbol." \
        "COLLECTION_NOT_FOUND"
  fi

  log_info "Fetching Magic Eden metadata for ${symbol}..."
  http_get \
    "${MAGICEDEN_API_BASE}/collections/${symbol}" \
    "${WORK_DIR}/me_meta.json"

  log_info "Fetching Magic Eden activities for ${symbol}..."
  http_get \
    "${MAGICEDEN_API_BASE}/collections/${symbol}/activities?offset=0&limit=500&type=buyNow" \
    "${WORK_DIR}/me_activities.json"

  # Graceful degradation: if activities are null/not-array, write empty array
  if ! jq -e 'type == "array"' "${WORK_DIR}/me_activities.json" &>/dev/null; then
    log_warn "Activities response is not an array — defaulting to []"
    echo "[]" > "${WORK_DIR}/me_activities.json"
  fi
}

# ---------------------------------------------------------------------------
# 8. FETCH — HELIUS
# ---------------------------------------------------------------------------
fetch_helius() {
  local symbol="$1"
  log_info "Fetching Helius DAS holder list for ${symbol}..."

  local das_payload
  das_payload=$(jq -n \
    --arg sym   "$symbol" \
    --argjson lim "$HOLDER_SAMPLE" \
    '{
      jsonrpc: "2.0", id: "nft-skill-das",
      method: "getAssetsByGroup",
      params: {
        groupKey: "collection", groupValue: $sym,
        page: 1, limit: $lim,
        options: { showUnverifiedCollections: false, showGrandTotal: true }
      }
    }')

  http_post_json \
    "${HELIUS_RPC_BASE}/?api-key=${HELIUS_API_KEY}" \
    "$das_payload" \
    "${WORK_DIR}/helius_das.json"

  log_info "Fetching Helius enriched transactions for ${symbol}..."
  http_get \
    "${HELIUS_API_BASE}/addresses/${symbol}/transactions?api-key=${HELIUS_API_KEY}&limit=100&type=NFT_SALE" \
    "${WORK_DIR}/helius_txns.json"
}

# ---------------------------------------------------------------------------
# 9. DETERMINISTIC METRIC COMPUTATIONS
# ---------------------------------------------------------------------------
compute_metrics() {
  local symbol="$1"

  # -- 9a. floor_price_sol ------------------------------------------------
  FLOOR_SOL=$(jq -r '
    (.floorPrice // 0) / 1000000000
    | . * 100000000 | round | . / 100000000
  ' "${WORK_DIR}/me_stats.json")

  # -- 9b. volume_7d_sol --------------------------------------------------
  VOL_7D=$(jq -r '(.volume7d // 0) / 1000000000' "${WORK_DIR}/me_stats.json")

  # -- 9c. volume_30d_sol -------------------------------------------------
  # Prefer explicit volume30d field; fall back to volumeAll (lifetime proxy)
  VOL_30D=$(jq -r '
    if (.volume30d // null) != null
    then .volume30d / 1000000000
    else (.volumeAll // 0) / 1000000000
    end
  ' "${WORK_DIR}/me_stats.json")

  # -- 9d. total_supply ---------------------------------------------------
  TOTAL_SUPPLY=$(jq -r '.supply // .totalItems // 0' "${WORK_DIR}/me_meta.json")
  if [[ "$TOTAL_SUPPLY" == "0" || "$TOTAL_SUPPLY" == "null" ]]; then
    TOTAL_SUPPLY=$(jq -r '.result.grand_total // .result.total // 0' "${WORK_DIR}/helius_das.json")
    log_warn "supply not in ME metadata — using Helius DAS grand_total=${TOTAL_SUPPLY}"
  fi

  # -- 9e. top10_holder_percentage ----------------------------------------
  TOP10_PCT=$(jq -r --argjson supply "$TOTAL_SUPPLY" '
    .result.items
    | map(select(.ownership.owner != null))
    | map(.ownership.owner)
    | group_by(.)
    | map({ wallet: .[0], count: length })
    | sort_by(-.count)
    | .[0:10]
    | map(.count) | add // 0
    | if $supply > 0 then (. / $supply * 100) else 0 end
    | . * 100 | round | . / 100
  ' "${WORK_DIR}/helius_das.json")

  # -- 9f. wash_trading_ratio ---------------------------------------------
  # Heuristic 1: repeated (seller, buyer) pairs
  WASH_REPEATED=$(jq -r '
    if (type != "array") or (length == 0) then 0
    else
      (length) as $total
      | [ .[] | select(.seller != null and .buyer != null)
                | "\(.seller):\(.buyer)" ]
      | group_by(.)
      | map(select(length > 1))
      | map(length) | add // 0
      | . / $total * 100
      | . * 100 | round | . / 100
    end
  ' "${WORK_DIR}/me_activities.json")

  # Heuristic 2: short buy-sell loop (buyer → seller within 72 h)
  WASH_LOOPS=$(jq -r '
    if (type != "array") or (length < 2) then 0
    else
      (length) as $total
      | (sort_by(.blockTime)) as $sorted
      | [range(length)] | map(
          . as $i
          | $sorted[$i] as $s
          | select($s.buyer != null and ($s.blockTime // 0) > 0)
          | [ ($i+1) , (length-1) ]
          | [ $sorted[.[0]:.[1]+1] ]
          | .[0]
          | map(
              select(.seller == $s.buyer)
              | select((.blockTime - $s.blockTime) < 259200)
              | select(.blockTime > $s.blockTime)
            )
          | length > 0
        )
      | map(select(. == true)) | length
      | . / $total * 100
      | . * 100 | round | . / 100
    end
  ' "${WORK_DIR}/me_activities.json")

  # Take max of both heuristics (avoid double-counting)
  WASH_RATIO=$(awk -v r="$WASH_REPEATED" -v l="$WASH_LOOPS" \
    'BEGIN { print (r > l) ? r : l }')

  # -- 9g. volume_volatility_ratio ----------------------------------------
  VOL_VOLATILITY=$(echo "
    scale=8
    v7=${VOL_7D}
    v30=${VOL_30D}
    if (v30 == 0) {
      0
    } else {
      v30_4 = v30 / 4
      diff  = v7 - v30_4
      if (diff < 0) diff = -diff
      diff / v30_4
    }
  " | bc -l | awk '{printf "%.4f", $0 + 0}')

  # -- 9h. final_risk_score (clamp 0–100) ---------------------------------
  RISK_SCORE=$(echo "
    scale=8
    t10   = ${TOP10_PCT}
    wash  = ${WASH_RATIO}
    vv    = ${VOL_VOLATILITY}

    t10c  = if (t10 > 100) 100 else t10
    washc = if (wash > 100) 100 else wash
    vvc   = vv * 100
    if (vvc > 100) vvc = 100

    raw = (t10c * 0.3) + (washc * 0.4) + (vvc * 0.3)
    if (raw > 100) raw = 100
    if (raw < 0)   raw = 0
    raw
  " | bc -l | awk '{printf "%.2f", $0 + 0}')

  log_info "Metrics: floor=${FLOOR_SOL} v7=${VOL_7D} v30=${VOL_30D} supply=${TOTAL_SUPPLY} top10=${TOP10_PCT}% wash=${WASH_RATIO}% vv=${VOL_VOLATILITY} risk=${RISK_SCORE}"
}

# ---------------------------------------------------------------------------
# 10. BUILD DETAIL PAYLOADS
# ---------------------------------------------------------------------------
build_holder_detail() {
  jq -r '
    .result.items
    | map(select(.ownership.owner != null))
    | map(.ownership.owner)
    | group_by(.)
    | map({ wallet: .[0], nft_count: length })
    | sort_by(-.nft_count)
    | .[0:20]
  ' "${WORK_DIR}/helius_das.json" 2>/dev/null \
    || echo "[]"
}

build_suspicious_detail() {
  jq -r '
    if (type != "array") then []
    else
      [ .[] | select(.seller != null and .buyer != null)
              | { key: "\(.seller):\(.buyer)", seller: .seller, buyer: .buyer } ]
      | group_by(.key)
      | map(select(length > 1))
      | map({
          seller:      .[0].seller,
          buyer:       .[0].buyer,
          occurrences: length,
          flag:        "repeated_pair"
        })
      | sort_by(-.occurrences)
      | .[0:10]
    end
  ' "${WORK_DIR}/me_activities.json" 2>/dev/null \
    || echo "[]"
}

# ---------------------------------------------------------------------------
# 11. ASSEMBLE FINAL JSON
# ---------------------------------------------------------------------------
assemble_output() {
  local symbol="$1"
  local timestamp
  timestamp=$(date -u +%FT%TZ)

  local total_sales unique_holders
  total_sales=$(jq 'if type=="array" then length else 0 end' \
    "${WORK_DIR}/me_activities.json")
  unique_holders=$(jq -r '
    .result.items
    | map(.ownership.owner) | unique | length
  ' "${WORK_DIR}/helius_das.json" 2>/dev/null || echo "0")

  local holder_json suspicious_json
  holder_json=$(build_holder_detail)
  suspicious_json=$(build_suspicious_detail)

  # Write detail files for jq --slurpfile
  printf '%s' "$holder_json"    > "${WORK_DIR}/holder_detail.json"
  printf '%s' "$suspicious_json" > "${WORK_DIR}/suspicious_detail.json"

  jq -n \
    --arg  ver        "$SKILL_VERSION" \
    --arg  ts         "$timestamp" \
    --arg  sym        "$symbol" \
    --arg  floor      "$FLOOR_SOL" \
    --arg  v7         "$VOL_7D" \
    --arg  v30        "$VOL_30D" \
    --arg  supply     "$TOTAL_SUPPLY" \
    --arg  top10      "$TOP10_PCT" \
    --arg  wash       "$WASH_RATIO" \
    --arg  vv         "$VOL_VOLATILITY" \
    --arg  risk       "$RISK_SCORE" \
    --arg  nsales     "$total_sales" \
    --arg  nholders   "$unique_holders" \
    --arg  lbdays     "$TRANSFER_DAYS" \
    --slurpfile holders  "${WORK_DIR}/holder_detail.json" \
    --slurpfile suspects "${WORK_DIR}/suspicious_detail.json" \
    '
    # Helper: safely convert string to number
    def n: tonumber;

    # Risk components (un-clamped per-factor contribution)
    (($top10|n) * 0.3  | . * 100 | round | . / 100) as $c_top10 |
    (($wash|n)  * 0.4  | . * 100 | round | . / 100) as $c_wash  |
    (($vv|n) * 100 * 0.3 | . * 100 | round | . / 100) as $c_vv  |

    # Final score: hard clamp via jq (defense-in-depth)
    ($risk|n) | (if . > 100 then 100 elif . < 0 then 0 else . end) as $score |

    {
      skill_version:        $ver,
      status:               "success",
      analysis_timestamp:   $ts,

      collection: {
        symbol:         $sym,
        total_supply:   ($supply|n),
        unique_holders: ($nholders|n)
      },

      market_metrics: {
        floor_price_sol:       ($floor|n),
        volume_7d_sol:         ($v7|n),
        volume_30d_sol:        ($v30|n),
        total_sales_analyzed:  ($nsales|n)
      },

      risk_metrics: {
        top10_holder_percentage: ($top10|n),
        wash_trading_ratio:      ($wash|n),
        volume_volatility_ratio: ($vv|n),
        final_risk_score:        $score
      },

      risk_components: {
        top10_contribution:      $c_top10,
        wash_contribution:       $c_wash,
        volatility_contribution: $c_vv
      },

      holder_concentration: {
        top_10_wallets: ($holders | first),
        methodology:    "DAS getAssetsByGroup – sample capped at NFT_HOLDER_SAMPLE_SIZE"
      },

      wash_trading_detail: {
        suspicious_wallet_pairs: ($suspects | first),
        methodology: "Repeated (seller,buyer) pairs AND short-loop buy/sell within 72h; max of both heuristics"
      },

      data_sources: {
        market_data:  "Magic Eden v2 API",
        onchain_data: "Helius DAS API + Enriched Transactions API",
        lookback_days: ($lbdays|n)
      }
    }
    '
}

# ---------------------------------------------------------------------------
# 12. OUTPUT VALIDATION
# ---------------------------------------------------------------------------
validate_output() {
  local json="$1"
  local required_fields=(
    ".skill_version"
    ".status"
    ".collection.symbol"
    ".market_metrics.floor_price_sol"
    ".risk_metrics.final_risk_score"
    ".risk_metrics.top10_holder_percentage"
    ".risk_metrics.wash_trading_ratio"
    ".risk_metrics.volume_volatility_ratio"
  )
  for field in "${required_fields[@]}"; do
    local val
    val=$(printf '%s' "$json" | jq -r "${field} // empty" 2>/dev/null)
    [[ -n "$val" ]] || die "Output validation failed: ${field} is missing or null" "SCHEMA_VALIDATION"
  done

  local score
  score=$(printf '%s' "$json" | jq -r '.risk_metrics.final_risk_score')
  awk -v s="$score" 'BEGIN { if (s < 0 || s > 100) exit 1 }' \
    || die "risk_score ${score} out of range [0,100]" "SCHEMA_VALIDATION"
}

# ---------------------------------------------------------------------------
# 13. MAIN
# ---------------------------------------------------------------------------
main() {
  local symbol="${1:-}"

  check_deps
  validate_inputs "$symbol"

  log_info "=== nft-market-analyzer v${SKILL_VERSION} ==="
  log_info "Collection: ${symbol}"

  fetch_magic_eden "$symbol"
  fetch_helius     "$symbol"
  compute_metrics  "$symbol"

  local output_json
  output_json=$(assemble_output "$symbol")
  validate_output  "$output_json"

  log_info "Analysis complete. risk_score=${RISK_SCORE}"

  # Emit final JSON to original stdout (fd 3)
  printf '%s\n' "$output_json" >&3
}

main "$@"
