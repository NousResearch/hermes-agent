---
id: references/operational-rate-limits-and-compute-units.md
name: 'Rate Limits and Compute Units'
description: 'Alchemy meters usage using compute units (CU) and enforces per-key rate limits. Plan your request patterns accordingly.'
tags:
  - alchemy
  - operational
  - operations
related:
  - operational-pricing-and-plans.md
  - data-apis
updated: 2026-02-05
---
# Rate Limits and Compute Units

## Summary
Alchemy meters usage using compute units (CU) and enforces per-key rate limits. Plan your request patterns accordingly.

## Guidance
- Batch or consolidate requests when possible.
- Prefer Data APIs over repeated JSON-RPC calls for historical data.
- Implement exponential backoff for `429` responses.

## Practical Patterns
- Use pagination and resume tokens (`pageKey`) for large datasets.
- Cache token metadata and NFT metadata aggressively.

## Related Files
- `operational-pricing-and-plans.md`
- `data-overview.md`

## Official Docs
- [Compute Units](https://www.alchemy.com/docs/reference/compute-units)
- [Compute Unit Costs](https://www.alchemy.com/docs/reference/compute-unit-costs)
