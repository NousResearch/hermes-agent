# BTC Vol Desk Monitor v0 — Completion Readout

**Status:** v0 complete for internal evidence review  
**Evidence control:** SCREEN-ONLY · NOT EXECUTABLE  
**Latest run:** btcvol-20260515-121752  
**Generated:** 2026-05-15 CST

## Completion Standard

This v0 is complete when it can:

1. Fetch live public Deribit BTC option screen data.
2. Fetch official iShares IBIT holdings and compute BTC/share.
3. Fetch public Nasdaq IBIT option-chain screen data.
4. Normalize IBIT options into BTC-equivalent exposure.
5. Estimate screen-only IBIT implied vol from public bid/ask mids.
6. Select ATM term-structure rows for Deribit and IBIT.
7. Generate screen-only dislocation candidates.
8. Rank candidates into an internal review queue without executable language.
9. Score data quality and source freshness.
10. Render a static internal dashboard.
11. Persist run history.
12. Create an evidence manifest with byte counts and SHA-256 hashes.
13. Package evidence into a ZIP bundle.
14. Independently verify the ZIP bundle against the manifest.
15. Pass the local regression suite.
16. Render visually without obvious dashboard breakage.

All criteria above are met for v0.

## Latest Verification

- Test suite: `42 passed in 0.38s`
- Latest run ID: `btcvol-20260515-121752`
- Quality score: `100`
- Quality grade: `green`
- Freshness grade: `green`
- Freshness stale sources: `[]`
- Freshness missing sources: `[]`
- Deribit ATM rows: `12`
- IBIT ATM rows: `7`
- Dislocation candidates: `7`
- BTC reference: `$79,682.66`
- IBIT BTC/share: `0.0005679118586151414`
- Evidence bundle SHA-256: `b81186890ba7163c92b13407552d6ef83a83b32f98a0f8885187acd5c64c31ef`

## Latest Artifacts

- Dashboard: `artifacts/institutional/dashboard/index.html`
- Report: `artifacts/institutional/data/reports/btc-vol-desk-monitor-2026-05-15-121752.md`
- Evidence bundle: `artifacts/institutional/data/normalized/btcvol-20260515-121752/btcvol-20260515-121752-evidence-bundle.zip`
- Evidence manifest: `artifacts/institutional/data/normalized/btcvol-20260515-121752/evidence_manifest.json`
- Evidence index: `artifacts/institutional/data/normalized/btcvol-20260515-121752/evidence_index.md`
- Candidate triage ledger: `artifacts/institutional/data/normalized/btcvol-20260515-121752/candidate_triage.jsonl`
- Run history: `artifacts/institutional/data/run_manifest.jsonl`

## Reviewer Commands

Run monitor:

```bash
PYTHONPATH=. python -m institutional_btc_vol.cli run artifacts/institutional/data
```

Verify an evidence bundle:

```bash
PYTHONPATH=. python -m institutional_btc_vol.cli verify-bundle artifacts/institutional/data/normalized/btcvol-20260515-121752/btcvol-20260515-121752-evidence-bundle.zip
```

Run regression tests:

```bash
PYTHONPATH=. pytest -o addopts='' tests/test_btc_vol_*.py -q
```

## Explicit Non-Completion Items / External Gates

These are intentionally outside v0 completion:

- CME integration: blocked until licensed/vendor/broker source exists.
- Executable RFQ workflow: blocked until legal/counsel approval.
- Investor-facing claims: blocked until quote-verified/trade-verified evidence exists.
- SaaS/front-end productization: deferred until the static evidence workflow proves durable.
- Alerts/notifications: optional next layer, not required for v0 completion.

## Evidence Policy

This system does not prove executable economics. It proves a repeatable internal evidence workflow using public screen data, model-estimated IVs, explicit source/freshness controls, and artifact chain-of-custody. All candidate economics remain **SCREEN-ONLY · NOT EXECUTABLE** unless independently quote-verified or trade-verified under an approved legal wrapper.
