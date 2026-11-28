# Pricing Change Detection

## Technique
The daemon uses `PricingExtractor` which:
1. Strips HTML noise (nav, footer, scripts, styles)
2. Extracts only pricing-relevant text: dollar amounts, plan names, per-seat pricing, contact-sales signals
3. Hashes the extracted text for comparison

## Why Full-Page SHA Failed
Full-page SHA hashing produced constant false positives because:
- Dynamic elements (session IDs, timestamps, analytics scripts) changed on every fetch
- Layout and CSS changes triggered diffs even when pricing was identical

## Edge Cases
- Pricing page redesigned but prices unchanged → should NOT trigger alert
- Plan renamed but price unchanged → SHOULD trigger alert
- New plan added → SHOULD trigger alert

## Reset Procedure
When false positives accumulate, clear the hash cache and let the daemon re-establish baselines.