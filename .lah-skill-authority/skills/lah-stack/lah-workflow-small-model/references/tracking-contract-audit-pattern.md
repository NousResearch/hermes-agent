# Tracking Contract Audit Pattern

Session: T05_AFFILIATE_TRACKING_CONTRACT_INTEGRITY_AUDIT_V1 (2026-08-18)

## Purpose

Structured 7-phase diagnostic for auditing affiliate tracking URL contracts
against canonical LAH/CrakRevenue sources. Used when campaign variation URLs
are suspected of missing tracking parameters.

## Phases

1. **Recover canonical contract** — Read `tracking-url.factory.js` (buildTrackingUrl)
   and `redirect-gateway.js` (offer URL templates) to establish the current
   canonical tracking URL shape including all LAH params and CrakRevenue aff_sub macros.

2. **Reconcile with global postback** — Read `conversion-importer.js` to map
   postback fields to outbound click dimensions. Trace: OUTBOUND CLICK ->
   CrakRevenue SubIDs -> Conversion/Transaction -> Global Postback -> LAH Attribution.

3. **Read actual provider state** — Read-only ExoClick API calls for campaign
   and variations. Confirm PAUSED status. Extract actual variation URLs.

4. **Expected vs actual** — Compare semantic tracking dimensions (not raw URL
   strings). Classify each dimension as PASS/MISSING for each arm.

5. **Find first loss boundary** — Trace the construction pipeline from canonical
   offer URL through destination resolution, experiment packet, tracking URL
   construction, variation creation input, FormData/API request, to provider
   readback. Classify root cause.

6. **Data contamination assessment** — Read provider stats. Classify existing
   observations as CLEAN / PARTIALLY_ATTRIBUTABLE / ATTRIBUTION_COMPROMISED /
   NO_MEANINGFUL_TRAFFIC_BEFORE_PAUSE.

7. **Design repair** — Specify smallest canonical fix. Prefer reusing
   buildTrackingUrl() over manual parameter injection. Define new hard gate
   AFFILIATE_TRACKING_CONTRACT_VERIFIED.

## Key Canonical Sources

- `lah-brain/src/campaign-factory/tracking-url.factory.js` — buildTrackingUrl()
- `lah-brain/src/redirect-gateway.js` — offer URL templates with aff_sub macros
- `lah-brain/src/money/conversion-importer.js` — postback field mapping (Mode B)
- `lah-brain/src/tracking-trust-blockers-contract.js` — required tracking keys

## Root Cause Classification

- `T05_PACKET_MISSING_TRACKING_CONTEXT` — T05 ARMS definition uses bare CrakRevenue
  URLs instead of buildTrackingUrl() output. The tracking URL builder is bypassed
  entirely at the variation creation step.

## Hard Gate

`AFFILIATE_TRACKING_CONTRACT_VERIFIED` — must operate on PROVIDER READBACK.
Future PLAY requires BOTH VARIATION_SEMANTIC_IDENTITY_VERIFIED AND this gate.

## Pitfall: TEMPLATE_SUBS false positive (2026-08-19)

A previous diagnostic incorrectly flagged `redirect-gateway.js` TEMPLATE_SUBS as
incomplete (claiming only `{click_id}` was handled). The actual code has 12 entries
covering all six canonical macros in both URL-encoded and raw forms. Always verify
TEMPLATE_SUBS coverage by reading the live source, not by trusting prior diagnostic
reports. The `exoclick-campaign-operations` skill's `references/template-subs-bug.md`
was corrected to reflect this.

## Pitfall: ExoClick clicks=0 for Popunder CPM is NORMAL

ExoClick's "clicks" metric for Popunder CPM campaigns counts user click-throughs on
the popunder ad itself — NOT tracking pixel fires. Zero clicks is EXPECTED behavior
for this format. CrakRevenue "clicks" are deduplicated tracking-link hits (landing
page loads), a different semantic. Never interpret ExoClick clicks=0 as a tracking
defect for Popunder CPM campaigns.

## Pitfall: Financial unit display (cents shown as dollars)

The ExoClick API returns spend/budget values in cents. The display/reporting layer
shows raw cent values as dollars without dividing by 100, creating a 100x inflation
illusion (e.g., $858.00 displayed for actual $8.58 spend). Always verify unit
interpretation against provider raw values and arithmetic proofs.