---
name: veradata-verified-data
description: Query verified Latin American data via VeraData.
version: "1.0.0"
author: teodorofodocrispin-cmyk
license: MIT
metadata:
  hermes:
    tags: [Research, LATAM, Compliance, Sanctions, x402]
    category: research
    related_skills: [osint-investigation, mpp-agent]
---

# VeraData Verified Data Skill

Calls the hosted VeraData API for verified Latin American financial and
compliance data: central-bank rates, sanctions/KYB screening, and company
enrichment from LATAM registries. It returns data only — it does not file,
submit, or act on any compliance decision itself.

## When to Use

- The user needs a LATAM central-bank rate (TRM, DTF, TIIE, Selic, UF, etc.).
- The user needs a sanctions or risk screen against OFAC/EU/UK lists plus
  LATAM-specific sources.
- The user needs company enrichment from RUES (Colombia), Receita (Brazil),
  or SAT (Mexico).

## Prerequisites

- `terminal` tool to run `curl`.
- No wallet needed for the free tier — the `X-TRIAL` header covers 5 calls
  per IP per 24h.
- For paid endpoints: any x402-capable wallet client (see the `mpp-agent`
  skill's wallet table, e.g. `agentcash` or `mppx`).

## How to Run

Free trial call (no wallet, no payment):

```
curl -X POST https://api.veradata.dev/rates \
  -H "Content-Type: application/json" \
  -H "X-TRIAL: true" \
  -d '{"country": "CO", "signals": ["usd_cop"]}'
```

## Quick Reference

| Endpoint | Price | Auth |
|----------|-------|------|
| `POST /rates` (trial) | free (5/IP/24h) | `X-TRIAL: true` header |
| `POST /rates` (paid) | $0.02-$0.10 USDC | x402 `X-PAYMENT` header |
| Sanctions/KYB bundles | $0.08-$0.15 USDC | x402 `X-PAYMENT` header |

## Procedure

### 1. Try the free trial

```
curl -X POST https://api.veradata.dev/rates \
  -H "Content-Type: application/json" \
  -H "X-TRIAL: true" \
  -d '{"country": "CO", "signals": ["usd_cop", "dtf"]}'
```

### 2. For paid endpoints, pay via x402

Once past the trial quota, use `mppx` (or the equivalent for whichever
wallet the user has funded):

```
mppx https://api.veradata.dev/rates --method POST \
  --data '{"country": "CO", "signals": ["usd_cop"]}'
```

### 3. Read the result

```json
{
  "country": "CO",
  "usd_cop": 3248.87,
  "trm_official": 3248.87,
  "source": "Banco de la República de Colombia"
}
```

## Pitfalls

- **Trial quota is per-IP, 5 calls per 24h** — don't assume it resets
  instantly or works from a different network without limit.
- **Sanctions data spans multiple lists** (OFAC SDN, EU Consolidated, UK HM
  Treasury, plus LATAM PEP screening) — a clean result on one list doesn't
  mean clean on all; check which lists a given endpoint actually covers.
- **Wallet keys never enter agent context** for the paid path.

## Verification

```
curl -s https://api.veradata.dev/health | python3 -c "import json,sys; d=json.load(sys.stdin); assert d.get('status')=='ok'; print('OK')"
```
