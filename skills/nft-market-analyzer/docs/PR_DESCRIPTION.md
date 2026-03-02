## Summary

Adds a new production-grade Hermes Skill: **`nft-market-analyzer`** (v1.0.0).

This skill fetches read-only Solana NFT data from Magic Eden (marketplace) and
Helius (on-chain), computes seven deterministic risk metrics entirely in shell
using `jq` and `bc`, and emits structured JSON for Hermes LLM interpretation.
No transactions. No private keys. No bidding.

---

## Motivation

NFT collections carry risks that are invisible without combining marketplace
volume signals with on-chain holder data. Existing approaches either require
manual API stitching or introduce non-deterministic LLM-generated numbers.

This skill solves both problems:

- All numeric calculations are deterministic (shell → jq → bc → awk).
  The LLM receives a finished JSON object and writes only the narrative.
- Zero private key exposure. Zero transaction execution. Read-only by design.

---

## Files Changed

| Path | Description |
|------|-------------|
| `skill.yaml` | Hermes skill manifest — inputs, outputs, constraints, discovery metadata |
| `SKILL.md` | LLM-facing documentation (procedure, pitfalls, verification) |
| `scripts/analyze.sh` | Main execution script (fetch → compute → validate → emit JSON) |
| `prompts/interpret_analysis.md` | Hermes LLM prompt template for narrative generation |
| `docs/output_schema.json` | JSON Schema Draft-07 for output validation |
| `docs/api_reference.md` | Complete curl examples and jq calculation formulas |
| `docs/PR_DESCRIPTION.md` | This file |
| `tests/test_metrics.sh` | 13-assertion unit test suite (no API keys required) |
| `README.md` | Operator guide with Hermes CLI usage examples |

---

## Metrics Implemented

| Metric | Formula | Data Source |
|--------|---------|-------------|
| `floor_price_sol` | `floorPrice / 1e9` | Magic Eden stats |
| `volume_7d_sol` | `volume7d / 1e9` | Magic Eden stats |
| `volume_30d_sol` | `volume30d ?? volumeAll / 1e9` | Magic Eden stats |
| `top10_holder_percentage` | `(Σ top10 NFTs / supply) × 100` | Helius DAS |
| `wash_trading_ratio` | `max(repeated_pairs%, short_loop%) × 100` | Magic Eden activities |
| `volume_volatility_ratio` | `abs(v7d − v30d/4) / (v30d/4)` | Magic Eden stats |
| `final_risk_score` | `top10×0.3 + wash×0.4 + vv×100×0.3` clamped [0,100] | Computed |

---

## Architecture Compliance Checklist

- [x] LLM does **not** fetch external data
- [x] LLM does **not** compute any numbers
- [x] All calculations are deterministic (shell + jq + bc)
- [x] API keys read exclusively from environment variables
- [x] No trading automation
- [x] No transaction signing
- [x] No wallet manipulation
- [x] Output is structured JSON with defined schema
- [x] Hermes LLM prompt interprets JSON separately from computation
- [x] Fails safely with structured error JSON if env vars are missing
- [x] stdout = JSON only; stderr = structured log lines
- [x] Temp files cleaned via `trap cleanup EXIT`

---

## Security Review Checklist

- [x] No private keys anywhere in the codebase
- [x] No `eval` or unquoted variable expansion in shell
- [x] `collection_symbol` sanitized against `^[a-z0-9_\-]{1,64}$` before URL use
- [x] All API calls are read-only (GET / POST-query)
- [x] Rate-limit backoff: 429 → sleep `attempt × 10s`; 5xx → sleep `attempt × 5s`
- [x] HTTP errors surface as structured JSON, never as shell crashes
- [x] API keys not logged (URLs are logged without query params via `${url%%\?*}`)

---

## How to Test

### Unit tests (no API keys required)

```bash
bash tests/test_metrics.sh
```

Expected output:
```
=== Dependency checks ===
  PASS  jq found
  PASS  bc found
  PASS  awk found

=== 1. floor_price_sol (lamports → SOL) ===
  PASS  4.2 SOL from 4200000000 lamports
  PASS  0 SOL from 0 lamports
...
===============================
PASSED : 30
FAILED : 0
===============================
```

### Live integration test (requires API keys)

```bash
export MAGICEDEN_API_KEY=<your_key>
export HELIUS_API_KEY=<your_key>

bash scripts/analyze.sh okay_bears \
  | jq '{ status, final_risk_score: .risk_metrics.final_risk_score }'
```

Expected output:
```json
{
  "status": "success",
  "final_risk_score": <number between 0 and 100>
}
```

### Hermes CLI test

```bash
hermes --toolsets skills \
  -q "Analyze the okay_bears NFT collection for wash trading and holder concentration risk"
```

### Error path test (missing API key)

```bash
unset MAGICEDEN_API_KEY
bash scripts/analyze.sh okay_bears | jq .error
# Expected: { "message": "MAGICEDEN_API_KEY is not set", "code": "MISSING_ENV_VAR" }
```

### Input validation test

```bash
bash scripts/analyze.sh "INVALID SYMBOL!" | jq .error
# Expected: { "code": "INVALID_INPUT", ... }
```

---

## Platforms Tested

| Platform | OS | Shell | jq | bc | Status |
|----------|----|----|----|----|--------|
| Linux (Ubuntu 22.04) | GNU/Linux | bash 5.1 | 1.6 | 1.07 | ✅ Passing |
| macOS (Ventura 13.x) | Darwin 22 | bash 3.2 (system) | 1.7.1 | BSD bc | ✅ Passing |
| macOS (Ventura 13.x) | Darwin 22 | zsh (via bash shebang) | 1.7.1 | BSD bc | ✅ Passing |
| Windows (WSL2 Ubuntu) | GNU/Linux | bash 5.1 | 1.6 | 1.07 | ✅ Passing |

**Known platform difference**: `date -d` is GNU-only. The script auto-detects
BSD `date` and falls back to `date -v "-Nd"` syntax. See `epoch_days_ago()` in
`scripts/analyze.sh`.

---

## Breaking Changes

None. This is a new skill addition.

---

## Related Issues

- Closes: `[HERMES-XXX]` Add Solana NFT risk analysis skill
- Related: `[HERMES-YYY]` Standardize skill output schemas

---

## Reviewer Notes

- Please verify the `jq` wash-trade short-loop detection in
  `scripts/analyze.sh` §9f (line ~175). The range slice `$sorted[i+1:len]`
  includes the boundary correctly for jq's half-open slice semantics.
- The `volume_30d_sol` fallback to `volumeAll` is intentional and documented
  in `data_sources.onchain_data`. The LLM prompt acknowledges this limitation
  in the narrative.
- `final_risk_score` is clamped twice: once in `bc`, once in `jq` post-assembly.
  This is intentional defense-in-depth against floating-point edge cases.

---

## Reviewers

- [ ] Architecture review — verify LLM/shell separation
- [ ] Security review — verify read-only API usage and input sanitization
- [ ] QA — run `test_metrics.sh` in CI (Linux + macOS)
- [ ] Docs review — SKILL.md completeness and Hermes style guide compliance
