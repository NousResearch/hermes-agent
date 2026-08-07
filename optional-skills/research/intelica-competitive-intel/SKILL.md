---
name: intelica-competitive-intel
description: Get competitive intelligence via the Intelica API.
version: "1.0.0"
author: teodorofodocrispin-cmyk
license: MIT
metadata:
  hermes:
    tags: [Research, Competitive-Intelligence, x402, Business]
    category: research
    related_skills: [osint-investigation, domain-intel, mpp-agent]
---

# Intelica Competitive Intel Skill

Calls the hosted Intelica API for structured competitive intelligence: a moat
score (IMI), competitor mapping, and a go/no-go recommendation for a company
or market description. It returns analysis only — it does not execute any
trade, investment, or business decision itself.

## When to Use

- The user asks for a competitive assessment of a company, product, or
  market ("is this a good market to enter", "who competes with X").
- The user wants a moat/IMI score or an enter/monitor/avoid recommendation
  backed by cited sources.

## Prerequisites

- `terminal` tool to run `curl`.
- No wallet needed for the free tier — a trial key covers 5 calls.
- For paid modes ($1.00 elite modes): any x402-capable wallet client (see
  the `mpp-agent` skill's wallet table, e.g. `agentcash` or `mppx`).

## How to Run

Get a free trial key (no wallet, no payment):

```
curl https://api.intelica.dev/api-keys/trial
```

## Quick Reference

| Mode | Price | Use case |
|------|-------|----------|
| `competitive` (default) | $0.05 | Quick competitor scan |
| `venture_screening`, `market_entry_execution`, `regulatory_compliance`, `risk_assessment`, `sales_enablement` | $1.00 | Deep, cited analysis with execution plan |

## Procedure

### 1. Get a trial key

```
curl https://api.intelica.dev/api-keys/trial
```

Returns `{"api_key": "trial_..."}` — 5 free calls, no wallet required.

### 2. Run an analysis

```
curl -X POST https://api.intelica.dev/intel \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: <trial_key>" \
  -d '{"text": "Describe the company or market to analyze", "mode": "competitive"}'
```

### 3. For paid elite modes, pay via x402

Once the trial quota is used, elite modes require payment. Use `mppx` (or
the equivalent for whichever wallet the user has funded):

```
mppx https://api.intelica.dev/intel --method POST \
  --data '{"text": "...", "mode": "venture_screening"}'
```

### 4. Read the result

```json
{
  "intelica_moat_index": 0.78,
  "decision_recommendation": {"action": "monitor", "confidence_score": 0.82},
  "detected_competitors": ["Competitor A", "Competitor B"],
  "confidence": "low",
  "quality_caveat": "..."
}
```

Check `source_verification_status` and `quality_caveat` (when present) before
treating budget or market-share figures as confirmed — Intelica flags its
own low-confidence outputs explicitly.

## Pitfalls

- **Trial key covers only 5 calls.** After that, every request needs
  payment — don't assume the trial key works indefinitely.
- **`confidence: "low"` with a `quality_caveat` means take the numbers as
  directional, not confirmed** — this happens automatically when Intelica
  can't fully verify its own cited sources.
- **Wallet keys never enter agent context** for the paid path — whichever
  client signs the x402 payment stores its keys under its own config.

## Verification

```
curl -s https://api.intelica.dev/health | python3 -c "import json,sys; d=json.load(sys.stdin); assert d.get('status')=='ok'; print('OK')"
```
