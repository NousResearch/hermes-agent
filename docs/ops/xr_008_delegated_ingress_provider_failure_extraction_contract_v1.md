# XR-008 Delegated Ingress / Provider Failure Extraction Contract (v1)

Date: 2026-03-28  
Status: active (XR-008 bounded extraction contract)  
Owner: Architect  
Slice: `XR-008 delegated_ingress_provider_failure_extraction_from_walletdb`

## Purpose
Define a bounded, fail-closed extraction contract for adapting the delegated-ingress + provider-failure stack from `src/walletdb/*` into neutral OpenClaw continuity surfaces **without** breaking out-of-core quarantine boundaries.

## Source lineage (read-only origin)
- `src/walletdb/autopilot_delegated_ingress.py`
- `src/walletdb/provider_failure.py`
- `tests/test_autopilot_delegated_ingress.py`

## In-scope extraction surface (allowed)
1. Delegated completion ingress normalization logic:
   - completion-packet candidate extraction
   - gate-decision summarization
   - bounded retry profile derivation
2. Provider failure classification + retry planning:
   - transient vs non-retryable classification
   - bounded backoff + one-shot retry semantics
   - redacted provider-failure summary payload shape

## Out-of-scope (explicitly prohibited in XR-008)
- Any direct promotion of `src/walletdb/*` runtime modules into canonical lane-map/runtime references.
- Any mutation that weakens XR-003 quarantine behavior.
- Any broad namespace migration outside the bounded extraction interface below.

## Neutral target namespace contract (canonical target)
Extraction target for follow-on runtime wiring is:
- `ops/openclaw/continuity/delegated_ingress.py`
- `ops/openclaw/continuity/provider_failure.py`

Required for safe adaptation before runtime activation:
1. Remove all `walletdb.*` imports from extracted target modules.
2. Depend only on neutral OpenClaw contracts/redaction helpers in `ops/openclaw/continuity/*`.
3. Preserve deterministic queue reasons/backoff semantics (parity with lineage behavior).
4. Add parity tests in `tests/test_continuity_delegated_ingress.py` and `tests/test_continuity_provider_failure.py`.

## Fail-closed adaptation gate (XR-008 decision gate)
`namespace_adaptation_safe=true` only if all conditions hold:
1. **Import safety:** no `walletdb.*` imports in extracted neutral modules.
2. **Contract safety:** delegated ingress references neutral contract paths, not walletdb contract paths.
3. **Parity safety:** parity test pack passes against mapped extraction cases.
4. **Boundary safety:** out-of-core quarantine rules remain unchanged for `src/walletdb/*`.

If any condition fails or cannot be verified, decision is:
- `namespace_adaptation_safe=false`
- `release_decision=FAIL_CLOSED_NO_RUNTIME_PROMOTION`

## XR-008 completion semantics
XR-008 is considered complete when this bounded contract + evidence packet is landed in canonical expanded-roadmap assets and the queue reflects fail-closed posture if adaptation safety is not yet proven.

This slice does **not** authorize unverified runtime extraction.
