---
name: veradata-verified-data
description: Query verified Latin American data via the VeraData API.
license: MIT
compatibility: Requires internet access to reach the VeraData API. No local dependencies. Compatible with any agent that can make HTTP POST requests. No authentication required for trial use.
metadata:
  author: teodorofodocrispin-cmyk
  version: "1.0.0"
  endpoint: https://api.veradata.dev/rates
  health: https://api.veradata.dev/health
  trial: X-TRIAL: true header (5 free calls per IP per 24h, no wallet)
  languages: Spanish (LATAM), Portuguese (BR), English
  homepage: https://github.com/teodorofodocrispin-cmyk/veradata-public
---

# VeraData — Verified Latin American Data

An **opt-in hosted API** that returns verified Latin American financial and
compliance data: central-bank rates (CO/MX/BR/CL/PE), sanctions screening
(OFAC+UN+EU+UK), company enrichment (RUES/CNPJ/RFC), and market context. You
send a query to the VeraData API; it returns structured JSON. The request
transits VeraData's infrastructure (FastAPI + Supabase + Render, AWS) — this is
a hosted data service, not a local lookup.

## Transparency notice

- Raw query text is transmitted to `api.veradata.dev` for processing. If your
  policy forbids sending queries to third parties, do not use this skill.
- Results are computed from public LATAM registries and returned as JSON.

## When to use

- You need a LATAM central-bank rate (TRM, DTF, TIIE, Selic, UF, etc.).
- You need a sanctions/risk screen against global + LATAM lists.
- You need company enrichment from RUES (CO), Receita (BR), or SAT (MX).
- You need AI-powered LATAM market context.

## When NOT to use

- Strict zero-transmission environments (air-gapped, on-premise).
- Any flow where the query must never leave the local machine.

## How it works

1. You POST a query to the VeraData API.
2. VeraData returns structured JSON (rates, sanctions result, entity data, or
   market context).

## Trial (no wallet, no payment)

`POST https://api.veradata.dev/rates` with header `X-TRIAL: true`:

```json
{ "country": "CO", "signals": ["usd_cop"] }
```

Returns live rates with no wallet or payment. 5 free calls per IP per 24h.

## Paid usage (optional, user-controlled)

Payment is **out of band and user-controlled** — the skill never instructs the
agent to sign transactions or spend funds. Paid endpoints use x402 (USDC on
Base/Solana). A human operator performs the transfer and supplies the resulting
payment; the skill only consumes the standard x402 `X-Payment` header. The
skill does not embed or instruct any wallet transfer.

## API request (trial rates)

**Endpoint:** `POST https://api.veradata.dev/rates`
**Headers:** `Content-Type: application/json`, `X-TRIAL: true`

```json
{ "country": "CO", "signals": ["usd_cop", "dtf"] }
```

## Response (success 200)

```json
{
  "country": "CO",
  "timestamp": "2026-07-12T20:20:05Z",
  "usd_cop": 3248.87,
  "trm_official": 3248.87,
  "source": "Banco de la República de Colombia"
}
```

## Known limitations

- Trial is trust-based: per-IP quota is not cryptographically verified.
- Paid endpoints require x402 settlement via a facilitator (CDP/PayAI).

## Resources

- GitHub: https://github.com/teodorofodocrispin-cmyk/veradata-public
- Health: https://api.veradata.dev/health
- Docs: https://api.veradata.dev/llms.txt
