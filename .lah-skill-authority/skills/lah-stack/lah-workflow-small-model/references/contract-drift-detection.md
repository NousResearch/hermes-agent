# Contract Drift Detection Pattern

Session: 2026-08-18 — P9 contract drift detection for LAH campaign compiler.

## Pattern

Compute SHA-256 fingerprints of all canonical authority files at adapter load time. Produce an aggregate `CAMPAIGN_CONTRACT_FINGERPRINT`. If any fingerprint changes, the aggregate changes and Fast Path becomes ineligible.

## Authority Files to Fingerprint

1. Parameter catalog (local)
2. ExoClick maps (lah-brain)
3. ExoClick normalizer (local)
4. Tracking URL factory (lah-brain)
5. CrakRevenue tracking URL builder (local)
6. Redirect gateway / destination authority (local)
7. Offer map / proposition authority (local)
8. P6 certification contract (local)
9. Safety contract (compile-invariants.js)
10. Attribution contract (compile-invariants.js TRACKING_MACROS)

## Drift Status Values

- `CONTRACT_STABLE` — all fingerprints match certified state → Fast Path eligible
- `PROVIDER_CONTRACT_DRIFT` — provider-side authority changed (maps, normalizer, tracking URL builder) → BLOCK
- `CAMPAIGN_CONTRACT_DRIFT` — campaign-side authority changed (safety, attribution, offer map) → BLOCK

## Key Constraint

No repository archaeology (grep/find/search) during normal execution. Fingerprints are computed from explicit file paths only.

## Implementation Reference

See `src/services/campaign-compiler/contract-drift-detector.js` for the canonical implementation.