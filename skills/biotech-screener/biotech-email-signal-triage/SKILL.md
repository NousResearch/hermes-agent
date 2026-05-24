---
name: biotech-email-signal-triage
description: "Use when scanning inbox alerts/digests to extract biotech investment signals into structured watchlists."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [biotech, email, triage, watchlist, signals]
    related_skills: [himalaya, biotech-mover-validation, biotech-screener-ops-ledger]
---

# Biotech Email Signal Triage

## Overview
Convert high-volume inbox traffic into a compact, actionable biotech signal feed. Focus on repeatable extraction from mover alerts, digest emails, and calendar updates.

## When to Use
- User asks to scan many emails for market/investment signals.
- Inbox contains intraday mover alerts vs benchmark (XBI), pre-market digests, and biotech calendar updates.
- You need one structured output instead of raw email dumps.

## Workflow
1. Pull the last N envelopes as JSON:
   - `set -a; source ~/.hermes/.env; set +a; ~/.local/bin/himalaya envelope list --output json --page-size 100`
2. Filter by subject classes:
   - mover alerts: `intraday`, `vs XBI`
   - research digests: `Biotech Digest`, `Newsletter Digest`, `Weekly Roll-Up`
   - catalysts: `Biotech Calendar Update`
3. Emit a normalized watchlist JSON with fields:
   - `email_id`, `timestamp`, `ticker`, `signal_type`, `headline`, `benchmark_delta`, `confidence`, `next_action`
4. Flag uncertain extraction as `confidence: "LOW"` and `ticker: "UNKNOWN"`.

## Output schema
```json
{
  "as_of": "ISO-8601",
  "signals": [
    {
      "email_id": "130836",
      "timestamp": "2026-05-18T10:00:00-07:00",
      "ticker": "CLYM",
      "signal_type": "intraday_mover",
      "headline": "[HIGH] CLYM -9.8% intraday ...",
      "benchmark_delta_pp": -7.9,
      "confidence": "HIGH",
      "next_action": "validate_official_source"
    }
  ]
}
```

## Common Pitfalls
- Overmatching technical alerts as investment content; keep classifier strict.
- Treating digest summaries as verified catalysts; they are leads, not confirmations.
- Forgetting to source `~/.hermes/.env` before Himalaya commands.

## Verification Checklist
- [ ] Envelope pull succeeds from Himalaya
- [ ] Output includes only investment-relevant subject classes
- [ ] UNKNOWN/LOW fields used where data is missing
- [ ] Final output is machine-parseable JSON
