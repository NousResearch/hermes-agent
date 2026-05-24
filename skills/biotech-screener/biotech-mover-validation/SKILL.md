---
name: biotech-mover-validation
description: "Use when biotech intraday mover alerts require source verification before escalation or model use."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [biotech, validation, movers, governance, verification]
    related_skills: [biotech-email-signal-triage, biotech-screener-ops-ledger]
---

# Biotech Mover Validation

## Overview
Validate intraday biotech mover alerts with a governance-first approach. Do not treat email alerts as confirmed facts until official sources are checked.

## When to Use
- Subject contains: `"[HIGH] <TICKER> ... intraday ... vs XBI"`.
- Alert says: `"no official same-day source found"`.
- User requests investment-signal confidence or escalation.

## Workflow
1. Parse ticker and claimed move from subject/body.
2. Check official same-day sources in order:
   - Company IR/news page
   - Press wire (GlobeNewswire/PR Newswire/Business Wire)
   - SEC filing feed (if relevant)
3. Classify:
   - `VERIFIED_CATALYST`
   - `PRICE_ONLY_NO_CATALYST`
   - `UNVERIFIED`
4. Produce a short memo:
   - Facts
   - Inference
   - Options
   - Action

## Decision rules
- IF official source found same day THEN `VERIFIED_CATALYST`.
- IF no source found and move is large THEN `PRICE_ONLY_NO_CATALYST` and monitor.
- IF ticker/source extraction ambiguous THEN `UNVERIFIED` and escalate.

## Common Pitfalls
- Assuming social/newsletter summaries are official confirmation.
- Mixing date/timezones and marking prior-day releases as same-day.
- Upgrading confidence without primary-source URL evidence.

## Verification Checklist
- [ ] Primary-source URLs captured
- [ ] Classification present and justified
- [ ] Missing data explicitly marked UNKNOWN/UNVERIFIED
- [ ] Recommendation is non-destructive and governance-safe
