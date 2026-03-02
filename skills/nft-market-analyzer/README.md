# nft-market-analyzer

**Hermes Skill v1.0.0** — Solana NFT Collection Risk Analysis

Fetches read-only data from Magic Eden and Helius, computes seven deterministic
risk metrics in shell, and produces a structured JSON report that Hermes
interprets into a plain-language risk narrative.

**No trading. No transactions. No private keys.**

---

## Quick Start

```bash
# 1. Set credentials
export MAGICEDEN_API_KEY=your_key_here
export HELIUS_API_KEY=your_key_here

# 2. Run analysis
bash scripts/analyze.sh okay_bears

# 3. Pretty-print key results
bash scripts/analyze.sh okay_bears | jq '{
  status,
  floor_sol:   .market_metrics.floor_price_sol,
  risk_score:  .risk_metrics.final_risk_score,
  wash_pct:    .risk_metrics.wash_trading_ratio,
  top10_pct:   .risk_metrics.top10_holder_percentage
}'
```

---

## Hermes CLI Usage

```bash
# Basic analysis
hermes --toolsets skills \
  -q "Analyze the okay_bears NFT collection for wash trading and holder risk"

# Specific concern
hermes --toolsets skills \
  -q "Is there wash trading in the degods collection?"

# Risk score only
hermes --toolsets skills \
  -q "What is the risk score for claynosaurz on Magic Eden?"

# Holder concentration check
hermes --toolsets skills \
  -q "Check whale concentration for the tensorians NFT collection"
```

---

## Directory Structure

```
nft-market-analyzer/
├── skill.yaml                     # Hermes skill manifest
├── SKILL.md                       # LLM-facing documentation (procedure, pitfalls)
├── README.md                      # This file
├── scripts/
│   └── analyze.sh                 # Main script: fetch → compute → emit JSON
├── prompts/
│   └── interpret_analysis.md      # Hermes LLM prompt template
├── docs/
│   ├── output_schema.json         # JSON Schema Draft-07 for output
│   ├── api_reference.md           # API call examples + jq formulas
│   └── PR_DESCRIPTION.md          # PR template for Hermes repository
└── tests/
    └── test_metrics.sh            # Unit tests (no API keys required)
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `MAGICEDEN_API_KEY` | ✅ | — | Magic Eden v2 API key |
| `HELIUS_API_KEY` | ✅ | — | Helius API key |
| `NFT_REQUEST_TIMEOUT` | ❌ | `15` | HTTP timeout per request (seconds) |
| `NFT_MAX_RETRIES` | ❌ | `3` | Max retries on 429 / 5xx |
| `NFT_HOLDER_SAMPLE_SIZE` | ❌ | `500` | Holder records fetched for concentration |
| `NFT_TRANSFER_LOOKBACK_DAYS` | ❌ | `30` | Days of transfer history for wash detection |

---

## Metrics Reference

| Metric | Range | Formula |
|--------|-------|---------|
| `floor_price_sol` | ≥ 0 | `floorPrice / 1e9` |
| `volume_7d_sol` | ≥ 0 | `volume7d / 1e9` |
| `volume_30d_sol` | ≥ 0 | `volume30d ?? volumeAll / 1e9` |
| `top10_holder_percentage` | 0–100 | `(top10_nfts / supply) × 100` |
| `wash_trading_ratio` | 0–100 | `max(repeated_pairs%, short_loop%) × 100` |
| `volume_volatility_ratio` | ≥ 0 | `abs(v7d − v30d/4) / (v30d/4)` |
| `final_risk_score` | 0–100 | `top10×0.3 + wash×0.4 + vv×100×0.3` |

### Risk Score Labels

| Score | Label |
|-------|-------|
| 0–33 | Low Risk |
| 34–66 | Medium Risk |
| 67–100 | High Risk |

---

## Expected Output Format

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
      { "wallet": "WaLLeTa111...", "nft_count": 420 }
    ],
    "methodology": "DAS getAssetsByGroup – sample capped at NFT_HOLDER_SAMPLE_SIZE"
  },
  "wash_trading_detail": {
    "suspicious_wallet_pairs": [],
    "methodology": "Repeated (seller,buyer) pairs AND short-loop buy/sell within 72h"
  },
  "data_sources": {
    "market_data": "Magic Eden v2 API",
    "onchain_data": "Helius DAS API + Enriched Transactions API",
    "lookback_days": 30
  }
}
```

**Error output format:**
```json
{
  "skill_version": "1.0.0",
  "status": "error",
  "error": {
    "message": "MAGICEDEN_API_KEY is not set",
    "code": "MISSING_ENV_VAR"
  }
}
```

---

## Running Tests

```bash
# Unit tests — no API keys required
bash tests/test_metrics.sh

# Expected: PASSED: 30  FAILED: 0
```

---

## Dependencies

| Tool | Min Version | Notes |
|------|-------------|-------|
| `curl` | 7.58 | HTTP client |
| `jq` | 1.6 | JSON parsing + metric computation |
| `bc` | any | Arbitrary-precision arithmetic |
| `awk` | POSIX | Number formatting |
| `date` | GNU or BSD | Epoch calculation (auto-detected) |

Install on Ubuntu/Debian:
```bash
sudo apt-get install -y curl jq bc gawk
```

Install on macOS (Homebrew):
```bash
brew install curl jq gnu-bc gawk
```

---

## Security

- No private keys anywhere
- No transaction signing, bidding, or offer automation
- `collection_symbol` validated against `^[a-z0-9_-]{1,64}$` before use
- API keys read only from environment variables, never logged
- All API calls are read-only (GET / POST query)
- Temp directory purged on exit via `trap`
- Rate limits handled with exponential backoff (10s × attempt on 429)

---

## License

MIT — see `LICENSE` in the Hermes skills repository root.
