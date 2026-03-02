# nft-market-analyzer — API & jq Reference

Complete reference for every API call the skill makes and every deterministic
calculation it performs. All calculations use `jq` ≥ 1.6 and POSIX `bc`.

---

## Magic Eden API Calls

**Base URL**: `https://api-mainnet.magiceden.dev/v2`
**Auth**: `Authorization: Bearer ${MAGICEDEN_API_KEY}` header on every request.

---

### 1. Collection Stats

```bash
curl -s \
  -H "Authorization: Bearer ${MAGICEDEN_API_KEY}" \
  "https://api-mainnet.magiceden.dev/v2/collections/okay_bears/stats"
```

**Response fields used:**

| Field        | Type   | Used for              |
|-------------|--------|-----------------------|
| `floorPrice` | number | floor_price_sol       |
| `volume7d`  | number | volume_7d_sol         |
| `volumeAll` | number | volume_30d_sol (proxy)|
| `volume30d` | number | volume_30d_sol (preferred, if present) |

**Example response:**
```json
{
  "symbol":      "okay_bears",
  "floorPrice":  4200000000,
  "volume7d":    125000000000,
  "volumeAll":   540000000000,
  "listedCount": 187,
  "avgPrice24hr": 4500000000
}
```

---

### 2. Collection Metadata

```bash
curl -s \
  -H "Authorization: Bearer ${MAGICEDEN_API_KEY}" \
  "https://api-mainnet.magiceden.dev/v2/collections/okay_bears"
```

**Response fields used:**

| Field        | Type   | Used for       |
|-------------|--------|----------------|
| `supply`    | number | total_supply   |
| `totalItems`| number | total_supply (fallback) |

**Example response:**
```json
{
  "symbol":      "okay_bears",
  "name":        "Okay Bears",
  "description": "10,000 bears on Solana",
  "supply":      10000,
  "totalItems":  10000
}
```

---

### 3. Recent Buy Activities

```bash
curl -s \
  -H "Authorization: Bearer ${MAGICEDEN_API_KEY}" \
  "https://api-mainnet.magiceden.dev/v2/collections/okay_bears/activities?offset=0&limit=500&type=buyNow"
```

**Response fields used per item:**

| Field       | Type   | Used for                            |
|------------|--------|-------------------------------------|
| `seller`   | string | wash-trade pair key                 |
| `buyer`    | string | wash-trade pair key                 |
| `blockTime`| number | short-loop window check (Unix epoch)|
| `price`    | number | (informational, not in risk score)  |

**Example response:**
```json
[
  {
    "signature":  "5HbZ...",
    "type":       "buyNow",
    "tokenMint":  "AbC1...",
    "collection": "okay_bears",
    "slot":       234567890,
    "blockTime":  1717000000,
    "buyer":      "WaLLeTa1111111111111111111111111111111111111",
    "seller":     "WaLLeTb2222222222222222222222222222222222222",
    "price":      4200000000
  }
]
```

---

## Helius API Calls

**RPC Base**: `https://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}`
**REST Base**: `https://api.helius.xyz/v0`

---

### 4. DAS getAssetsByGroup (Holder List)

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id":      "nft-skill-das",
    "method":  "getAssetsByGroup",
    "params": {
      "groupKey":   "collection",
      "groupValue": "okay_bears",
      "page":       1,
      "limit":      500,
      "options": {
        "showUnverifiedCollections": false,
        "showGrandTotal":            true
      }
    }
  }' \
  "https://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}"
```

**Response fields used:**

| Path                          | Used for                  |
|-------------------------------|---------------------------|
| `.result.grand_total`         | total_supply (fallback)   |
| `.result.items[].ownership.owner` | holder map for top10  |

**Example response (truncated):**
```json
{
  "jsonrpc": "2.0",
  "id":      "nft-skill-das",
  "result": {
    "grand_total": 10000,
    "total":       500,
    "page":        1,
    "items": [
      {
        "id": "AbC1...",
        "ownership": {
          "owner":     "WaLLeTa1111111111111111111111111111111111111",
          "frozen":    false,
          "delegated": false
        }
      }
    ]
  }
}
```

---

### 5. Helius Enriched NFT Transactions

```bash
curl -s \
  "https://api.helius.xyz/v0/addresses/okay_bears/transactions?api-key=${HELIUS_API_KEY}&limit=100&type=NFT_SALE"
```

**Note:** This endpoint supplements Magic Eden activity data. Used for
cross-referencing transfer history when ME activities are sparse.

---

## Deterministic jq Calculations

All computations run in shell. The LLM receives only the assembled output JSON.

---

### floor_price_sol

```bash
jq -r '
  (.floorPrice // 0) / 1000000000
  | . * 100000000 | round | . / 100000000
' me_stats.json
# Example: 4.2
```

---

### volume_7d_sol

```bash
jq -r '(.volume7d // 0) / 1000000000' me_stats.json
# Example: 125
```

---

### volume_30d_sol (prefer explicit field, fallback to volumeAll)

```bash
jq -r '
  if (.volume30d // null) != null
  then .volume30d / 1000000000
  else (.volumeAll // 0) / 1000000000
  end
' me_stats.json
# Example: 540
```

---

### top10_holder_percentage

```bash
TOTAL_SUPPLY=10000

jq -r --argjson supply "$TOTAL_SUPPLY" '
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
' helius_das.json
# Example: 18.34
```

---

### wash_trading_ratio — repeated pairs heuristic

```bash
jq -r '
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
' me_activities.json
# Example: 6.2
```

---

### wash_trading_ratio — short-loop heuristic (buyer re-sells within 72 h)

```bash
jq -r '
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
    | . / $total * 100
    | . * 100 | round | . / 100
  end
' me_activities.json
```

**Final wash ratio**: `max(repeated_pairs_pct, short_loop_pct)` via `awk`.

---

### volume_volatility_ratio

```bash
V7=125      # volume_7d_sol
V30=540     # volume_30d_sol

echo "
  scale=8
  v7=$V7; v30=$V30
  if (v30 == 0) {
    0
  } else {
    v30_4 = v30 / 4
    diff  = v7 - v30_4
    if (diff < 0) diff = -diff
    diff / v30_4
  }
" | bc -l | awk '{printf "%.4f", $0 + 0}'
# v30/4 = 135, v7=125, diff=10, ratio=0.0741
```

---

### final_risk_score

```bash
TOP10=18.34
WASH=6.2
VV=0.0741

echo "
  scale=8
  t10c=18.34; washc=6.2; vvc=7.41
  raw=(t10c * 0.3) + (washc * 0.4) + (vvc * 0.3)
  if (raw > 100) raw = 100
  if (raw < 0)   raw = 0
  raw
" | bc -l | awk '{printf "%.2f", $0 + 0}'
# (5.50) + (2.48) + (2.22) = 10.20
```

---

## Example Final Output

```json
{
  "skill_version": "1.0.0",
  "status": "success",
  "analysis_timestamp": "2025-01-15T14:32:00Z",
  "collection": {
    "symbol": "okay_bears",
    "total_supply": 10000,
    "unique_holders": 4832
  },
  "market_metrics": {
    "floor_price_sol": 4.2,
    "volume_7d_sol": 125.0,
    "volume_30d_sol": 540.0,
    "total_sales_analyzed": 500
  },
  "risk_metrics": {
    "top10_holder_percentage": 18.34,
    "wash_trading_ratio": 6.2,
    "volume_volatility_ratio": 0.0741,
    "final_risk_score": 10.2
  },
  "risk_components": {
    "top10_contribution": 5.5,
    "wash_contribution": 2.48,
    "volatility_contribution": 2.22
  },
  "holder_concentration": {
    "top_10_wallets": [
      { "wallet": "WaLLeTa111...", "nft_count": 420 },
      { "wallet": "WaLLeTb222...", "nft_count": 315 }
    ],
    "methodology": "DAS getAssetsByGroup – sample capped at NFT_HOLDER_SAMPLE_SIZE"
  },
  "wash_trading_detail": {
    "suspicious_wallet_pairs": [
      {
        "seller": "WaLLeTx999...",
        "buyer":  "WaLLeTy888...",
        "occurrences": 5,
        "flag": "repeated_pair"
      }
    ],
    "methodology": "Repeated (seller,buyer) pairs AND short-loop buy/sell within 72h; max of both heuristics"
  },
  "data_sources": {
    "market_data":  "Magic Eden v2 API",
    "onchain_data": "Helius DAS API + Enriched Transactions API",
    "lookback_days": 30
  }
}
```
