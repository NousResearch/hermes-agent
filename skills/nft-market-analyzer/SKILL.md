---
name: nft-market-analyzer
description: >
  Analyzes Solana NFT collections using Magic Eden (marketplace) and Helius
  (on-chain) APIs. Fetches floor price, 7d/30d volume, holder distribution, and
  recent sales, then deterministically computes seven risk metrics entirely in
  shell (jq + bc) before passing structured JSON to Hermes for narrative
  interpretation. Triggers on any user request to evaluate, score, audit, or
  analyze an NFT collection; inspect wash trading or holder concentration;
  assess NFT investment risk; or review on-chain activity for a Solana NFT
  project. Use this skill even when the user only mentions a collection name
  alongside words like risk, safe, rug, wash, holders, volume, or floor.
---

# NFT Market Analyzer

Fetches read-only Solana NFT data from Magic Eden and Helius, computes seven
deterministic risk metrics in shell, and produces a structured JSON report that
Hermes interprets into a plain-language risk narrative. No transactions.
No private keys. No bidding.

---

## When to Use

Use this skill whenever the user asks about:

- Risk rating, audit, or due-diligence for a Solana NFT collection
- Wash trading detection or suspicion in NFT sale history
- Holder concentration or whale dominance in a collection
- Floor price, volume trends, or volume volatility for an NFT collection
- A "rug pull" risk assessment or collection health check
- Any question involving Magic Eden collection slugs (e.g. "okay_bears")

**Do not use** for EVM chains (Ethereum, Polygon, Base) — Magic Eden and Helius
are Solana-only data sources. Do not execute any trades, bids, or offers.

---

## Quick Reference

| Goal | Command |
|------|---------|
| Full analysis | `bash scripts/analyze.sh <collection_symbol>` |
| Output JSON | Emitted to stdout; logs to stderr |
| Unit tests | `bash tests/test_metrics.sh` |
| Schema reference | `docs/output_schema.json` |
| LLM prompt | `prompts/interpret_analysis.md` |
| API examples | `docs/api_reference.md` |

**Required environment variables** (must be set before running):
```bash
export MAGICEDEN_API_KEY=<your_key>
export HELIUS_API_KEY=<your_key>
```

---

## Procedure

Hermes must follow these steps in order. Do not skip or reorder steps.

### Step 1 — Validate Inputs and Environment

Before touching any API:

1. Confirm `MAGICEDEN_API_KEY` and `HELIUS_API_KEY` are non-empty. If either
   is missing, emit a structured error JSON (see Pitfalls §1) and stop.
2. Confirm the collection symbol matches `^[a-z0-9_\-]{1,64}$`. Reject and
   explain to the user if it does not.
3. Confirm `curl`, `jq` (≥1.6), `bc`, and `awk` are present on PATH.

### Step 2 — Fetch Magic Eden Marketplace Data

Run the following fetches (full curl examples in `docs/api_reference.md`):

1. **Collection stats** — `GET /v2/collections/{symbol}/stats`
   Extracts: `floorPrice`, `volume7d`, `volumeAll` (proxy for 30d).
2. **Collection metadata** — `GET /v2/collections/{symbol}`
   Extracts: `supply` / `totalItems` (total NFT count).
3. **Recent buy activities** — `GET /v2/collections/{symbol}/activities?limit=500&type=buyNow`
   Used for wash-trade detection: seller, buyer, blockTime per sale.

All calls must use exponential-backoff retry (10s × attempt on 429,
5s × attempt on 5xx), capped at `NFT_MAX_RETRIES`.

### Step 3 — Fetch Helius On-Chain Data

1. **DAS getAssetsByGroup** (POST to Helius RPC) — resolve ownership per mint.
   Extracts: `ownership.owner` for each asset → build holder map.
2. **Enriched transactions** — `GET /v0/addresses/{symbol}/transactions?type=NFT_SALE`
   Supplements Magic Eden activity data for wash detection.

### Step 4 — Compute Deterministic Metrics (shell only)

All calculations must be performed with `jq` and `bc`. The LLM must not
compute any numbers.

#### 4a. floor_price_sol
```bash
jq -r '.floorPrice / 1000000000' me_stats.json
```

#### 4b. volume_7d_sol / volume_30d_sol
```bash
jq -r '.volume7d  / 1000000000' me_stats.json
jq -r '.volumeAll / 1000000000' me_stats.json
```

#### 4c. top10_holder_percentage
```bash
jq -r --argjson supply TOTAL '
  .result.items
  | map(.ownership.owner)
  | group_by(.)
  | map({wallet: .[0], count: length})
  | sort_by(-.count)
  | .[0:10] | map(.count) | add // 0
  | if $supply > 0 then (. / $supply * 100) else 0 end
  | . * 100 | round | . / 100
' helius_das.json
```

#### 4d. wash_trading_ratio
Two heuristics, take max to avoid double-counting:
- **Repeated pair**: same `(seller, buyer)` tuple appears more than once.
- **Short loop**: buyer becomes seller within 72 h (259,200 s).

```bash
# Repeated-pair count
jq -r '
  . as $sales | length as $total
  | if $total == 0 then "0" else
    [ .[] | "\(.seller):\(.buyer)" ]
    | group_by(.)
    | map(select(length > 1)) | map(length) | add // 0
    | . / $total * 100 | . * 100 | round | . / 100 | tostring
  end
' me_activities.json
```

#### 4e. volume_volatility_ratio
```bash
echo "scale=4; v30_4=$V30/4; d=($V7-v30_4); if(d<0)d=-d; d/v30_4" | bc -l
```

#### 4f. final_risk_score
```bash
echo "
  scale=4
  raw=($TOP10*0.3)+($WASH*0.4)+(($VV*100)*0.3)
  if(raw>100)raw=100
  if(raw<0)raw=0
  raw
" | bc -l | awk '{printf "%.2f", $0}'
```

Clamp `final_risk_score` to [0, 100] both in `bc` and again in `jq`
post-assembly as defense-in-depth.

### Step 5 — Assemble Output JSON

Assemble the final JSON with `jq -n` using all computed values.
See the schema in `docs/output_schema.json`. Required top-level keys:

```
skill_version, status, analysis_timestamp, collection,
market_metrics, risk_metrics, risk_components,
holder_concentration, wash_trading_detail, data_sources
```

Emit the final JSON to **stdout only**. All log lines go to **stderr**.

### Step 6 — Validate Output

Before emitting, verify with `jq`:
- `.status == "success"`
- `.risk_metrics.final_risk_score` is between 0 and 100
- All required top-level keys are present and non-null

If validation fails, emit a structured error JSON and exit non-zero.

### Step 7 — LLM Interpretation

Pass the validated JSON to the prompt in `prompts/interpret_analysis.md`.
The LLM must not alter any numbers — it only writes the narrative sections
using values already present in the JSON.

---

## Pitfalls

### 1. Missing API Keys
**Symptom**: `MAGICEDEN_API_KEY is not set`
**Cause**: Environment not configured.
**Fix**: Emit structured error JSON and stop immediately.
```json
{ "status": "error", "error": { "message": "MAGICEDEN_API_KEY is not set", "code": "MISSING_ENV_VAR" } }
```
Never proceed with partial credentials.

### 2. Unknown Collection Symbol
**Symptom**: Magic Eden `/stats` returns 404 or `null` body.
**Cause**: Slug does not exist or was entered in wrong case (must be lowercase).
**Fix**: Emit error JSON with code `COLLECTION_NOT_FOUND`. Suggest the user
verify the slug on `magiceden.io/marketplace/<slug>`.

### 3. Helius DAS Returns Zero Items
**Symptom**: `top10_holder_percentage` is 0 even for a live collection.
**Cause**: The `groupValue` passed to `getAssetsByGroup` may be the on-chain
collection mint address, not the Magic Eden slug. Or the collection uses
a non-standard Metaplex grouping.
**Fix**: Fall back to counting NFTs from the enriched transaction history.
Log a `WARN` to stderr noting the fallback.

### 4. volume_30d Not in Magic Eden Response
**Symptom**: `volumeAll` is used as the 30d proxy but may be lifetime volume.
**Cause**: Magic Eden v2 `/stats` does not always expose a discrete 30d field.
**Fix**: The script uses `volumeAll` as the denominator; note this in the
`data_sources` field of the output JSON and in the LLM narrative.

### 5. Rate Limiting (HTTP 429)
**Symptom**: Magic Eden returns 429 after several requests.
**Fix**: Sleep `attempt × 10` seconds and retry. After `NFT_MAX_RETRIES`
exhausted, emit error JSON with code `HTTP_FAILURE`. Never silently continue
with stale/partial data.

### 6. `date` Portability on macOS vs Linux
**Symptom**: `date -d "30 days ago"` fails on macOS (uses BSD `date`).
**Fix**: The script detects which `date` variant is available:
```bash
EPOCH=$(date -u -d "30 days ago" +%s 2>/dev/null \
     || date -u -v "-30d"        +%s 2>/dev/null) \
     || die "GNU or BSD date required"
```

### 7. jq Calculation Precision
**Symptom**: Risk score has floating-point artefacts (e.g. 23.999999).
**Fix**: All intermediate values are rounded via `| . * 100 | round | . / 100`
in jq. Final score is formatted with `awk '{printf "%.2f", $0}'`.

### 8. Empty Activity Feed
**Symptom**: `me_activities.json` is `[]` (new or illiquid collection).
**Fix**: `wash_trading_ratio` defaults to `0`. The LLM narrative should note
"Insufficient sales history for wash-trade analysis."

### 9. Shell Injection via Collection Symbol
**Symptom**: Malicious input such as `; rm -rf /`.
**Fix**: Symbol is validated against `^[a-z0-9_\-]{1,64}$` before use in
any shell variable or URL construction. Validation failure → error JSON, exit 1.

---

## Verification

Hermes confirms correct execution by checking:

1. **Exit code 0** from `scripts/analyze.sh`.
2. **stdout** is valid JSON: `<output> | jq .status` returns `"success"`.
3. **Risk score range**: `jq '.risk_metrics.final_risk_score | (. >= 0 and . <= 100)'` returns `true`.
4. **Required fields present**:
   ```bash
   jq 'has("skill_version") and has("collection") and has("market_metrics") and has("risk_metrics")' <output>
   ```
   Must return `true`.
5. **Unit tests pass** (no API keys required):
   ```bash
   bash tests/test_metrics.sh
   # Expected: PASSED: 13  FAILED: 0
   ```
6. **Log integrity**: stderr contains timestamped `[INFO]` lines; stdout
   contains exactly one JSON object — no mixed log/JSON content.

---

## Reference Files

For deeper implementation detail, read these files as needed:

- `docs/api_reference.md` — Full curl examples for every API endpoint
- `docs/output_schema.json` — JSON Schema Draft-07 for output validation
- `prompts/interpret_analysis.md` — LLM prompt template for narrative generation
- `scripts/analyze.sh` — Full bash implementation with inline comments
