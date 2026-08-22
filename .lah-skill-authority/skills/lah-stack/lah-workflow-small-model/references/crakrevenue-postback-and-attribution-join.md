# CrakRevenue Postback Chain & Attribution Join

Session: 2026-08-20 — LAH_FAST_CAMPAIGN_PROVIDER_CANARY_V3

## Outbound Tracking (Confirmed)

The campaign creation draft passes through these fields to the ExoClick provider:

| Campaign Draft Field | Provider Field | Postback Field | Internal Field |
|---------------------|----------------|----------------|----------------|
| tracking_url | tracking_url | (not in postback) | tracking_url |
| subid1 | aff_sub1 | aff_sub1 | subid1 |
| geo | geo | (not in postback) | country |
| zone_id | zone_id | (not in postback) | zone_id |
| campaign_id | campaign_id | (not in postback) | campaign_id |
| creative_angle | angles | (not in postback) | creative_angle |
| device | device | (not in postback) | device_placement |

The SubID1 → aff_sub1 mapping is handled by `buildCrakRevenueDestinationUrl()` in `crakrevenue-tracking-url-builder.js`:
- Validates subid1 (max 64 chars, `[A-Za-z0-9_.-]` pattern)
- Injects aff_sub1 exactly once (never duplicates)
- Refuses aff_sub5 (PROHIBITED_RESERVED_BY_NETWORK)
- Redacts full URL in logs/receipts

## Inbound Postback Chain (NOT VERIFIED)

The following fields are NOT explicitly handled in the checked source code:
- transaction_id
- payout
- offer_id
- goal_id

The `zone-statistics-contract.js` reads `revenue`, `payout`, `earning` from raw rows but does not implement the full postback attribution chain.

## Event vs Paid Conversion (NOT VERIFIED)

No deterministic classification function between `event` and `paid_conversion` was found in the checked codebase. The executor handles `UNKNOWN_OUTCOME` states (blocks duplicate execution, requires reconciliation) but does not classify events as paid_conversion vs non-paid events.

## Attribution Join Gap

The outbound tracking and inbound postback share only `subid1/aff_sub1` as a join key. The following fields have no join path:
- transaction_id
- payout
- offer_id
- goal_id

This means deterministic attribution from postback to campaign is not proven for these fields.

## Implications for Canary

- The subid1/aff_sub1 chain is confirmed and usable for basic tracking attribution.
- Full postback/attribution verification (transaction_id, payout, offer_id, goal_id) is a separate gap that must be closed before a production canary.
- Event vs paid conversion classification is a separate gap that must be closed before a production canary.