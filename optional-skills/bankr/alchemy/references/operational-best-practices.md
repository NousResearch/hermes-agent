---
id: references/operational-best-practices.md
name: 'Production Best Practices'
description: 'Operational checklist for deploying Alchemy-backed services.'
tags:
  - alchemy
  - operational
  - operations
related:
  - operational-auth-and-keys.md
  - operational-rate-limits-and-compute-units.md
  - operational-request-logs.md
updated: 2026-02-05
---
# Production Best Practices

## Summary
Operational checklist for deploying Alchemy-backed services.

## Checklist
- Use separate API keys per environment.
- Store secrets in a secure secret manager.
- Implement retries with exponential backoff.
- Cache metadata and repeated reads.
- Validate all inbound webhook signatures.
- Monitor CU usage and error rates.

## Related Files
- `operational-auth-and-keys.md`
- `operational-rate-limits-and-compute-units.md`
- `operational-request-logs.md`

## Official Docs
- [Allowlists for Apps](https://www.alchemy.com/docs/how-to-add-allowlists-to-your-apps-for-enhanced-security)
- [Compute Units](https://www.alchemy.com/docs/reference/compute-units)
