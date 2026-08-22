# Campaign Snapshot Data Quality Authority Rule

Shared rule for the lah-stack-skills umbrella. When a campaign snapshot returns `data_quality.status`, Hermes MUST treat the payload as campaign fact authority and MUST NOT re-query DB, provider APIs, secrets, evidence directories, or source code merely to re-establish facts already present.

## Status Values

| Status | Meaning | Hermes Action |
|--------|---------|---------------|
| `PASS` | All checks green, data is authoritative | Treat as campaign fact authority. Do not re-query. |
| `PARTIAL` | Warnings present but no blockers | Usable for decision inputs. Surface warnings to operator. |
| `FAIL` | Blockers present, data is incomplete/missing | Return `BLOCKED_CANONICAL_DATA`. Do not escalate into forensic exploration. |

## Key Constraints

- **No provider network calls** — the snapshot is a read model over already-ingested canonical LAH data
- **No provider authentication** — no API keys or credentials needed
- **No .env reads for provider credentials** — the snapshot does not access secrets
- **No evidence-directory scraping** — the snapshot is self-contained
- **No SQLite exploration by the agent** — the snapshot query is encapsulated in `buildCampaignSnapshot`
- **No cents/USD ambiguity** — all monetary values are normalized to USD
- **provider_click and affiliate_arrival remain semantically distinct** — provider clicks are ExoClick click-throughs, affiliate arrivals are CrakRevenue tracking-link hits
- **missing != zero** — a missing field is not the same as a zero value; the data_quality status captures this distinction
- **downstream_event != paid_conversion** — downstream events are all affiliate activity, paid conversions are only payout > 0 rows

### T05 Pitfall: PARTIAL with SPEND_WITHOUT_REVENUE Is Negative Evidence, Not Missing Data

A PARTIAL status with SPEND_WITHOUT_REVENUE warning means spend > 0 but revenue = 0 and conversions = 0. This is observed negative evidence, not insufficient data. When information_readiness is READY and no positive signal exists, the correct decision is TERMINATE — do not suppress it by treating PARTIAL as "not enough data." Only FAIL status blocks the decision with BLOCKED_CANONICAL_DATA.

## Canonical Implementation

- File: `src/campaign-snapshot.js`
- Function: `buildCampaignSnapshot(campaignId, options)`
- Schema: `lah.campaign.snapshot.v1`
- Tests: `test/campaign-snapshot.test.js` (17/17 passing)