# Synthetic sample input

> **SYNTHETIC FIXTURE — not real company data.** Fictional names, captures, prices, and dates are for offline comparison only and are not current facts.

Target: Northstar Notes (fictional). Category: pricing.

- Capture A, observed 2026-03-01, fixture URL `https://example.invalid/northstar/a`: Team plan shown as 12 fictional credits/user/month, minimum 5 users.
- Capture B, observed 2026-03-08, fixture URL `https://example.invalid/northstar/b`: Team plan shown as 14 fictional credits/user/month, minimum 5 users, annual billing label visible.

Expected task: report an observed fixture difference, preserve before/after terms, state that `.invalid` URLs were not retrieved, and avoid claiming a live change.
