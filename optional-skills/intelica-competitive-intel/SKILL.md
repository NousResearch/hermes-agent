---
name: intelica-competitive-intel
description: Get competitive intelligence via the Intelica API.
license: MIT
compatibility: Requires internet access to reach the Intelica API. No local dependencies. Compatible with any agent that can make HTTP POST requests. No authentication required for trial use.
metadata:
  author: teodorofodocrispin-cmyk
  version: "1.0.0"
  endpoint: https://api.intelica.dev/intel
  health: https://api.intelica.dev/health
  trial: GET /api-keys/trial returns a free key (5 calls, no wallet)
  languages: English, Spanish (LATAM), Portuguese (BR)
  homepage: https://github.com/teodorofodocrispin-cmyk/Intelica-docs
---

# Intelica — Competitive Intelligence

An **opt-in hosted API** that returns structured competitive intelligence:
moat scoring (IMI), competitor mapping, executive brief, and strategic
recommendations. You send a company/market description; Intelica returns JSON
with a go/no-go decision and moat index. The request transits Intelica's
infrastructure (FastAPI + Supabase + Render, AWS) — this is a hosted analysis
service, not a local computation.

## Transparency notice

- Raw query text is transmitted to `api.intelica.dev` for processing. If your
  policy forbids sending queries to third parties, do not use this skill.
- Analysis is produced by an LLM (Claude) server-side; results are returned as
  JSON.

## When to use

- You need a competitive assessment of a company or market.
- You need a moat/IMI score and a go/no-go decision.
- You need competitor mapping against a known graph of companies.

## When NOT to use

- Strict zero-transmission environments (air-gapped, on-premise).
- Any flow where the query must never leave the local machine.

## How it works

1. You POST a description to the Intelica API.
2. Intelica returns structured JSON (decision, moat index, competitors).

## Trial (no wallet, no payment)

`GET https://api.intelica.dev/api-keys/trial` returns a free key (5 calls).
Use it as `X-API-KEY` header on `POST /intel`:

```json
{ "text": "Fintech neobank in Colombia", "mode": "competitive" }
```

## Paid usage (optional, user-controlled)

Payment is **out of band and user-controlled** — the skill never instructs the
agent to sign transactions or spend funds. Paid calls use x402 (USDC on
Base/Solana). A human operator performs the transfer and supplies the resulting
payment; the skill only consumes the standard x402 `X-Payment` header. The
skill does not embed or instruct any wallet transfer.

## API request (trial)

**Step 1 — get trial key:**
`GET https://api.intelica.dev/api-keys/trial`

**Step 2 — analyze:**
`POST https://api.intelica.dev/intel`
Headers: `Content-Type: application/json`, `X-API-KEY: <trial-key>`

```json
{ "text": "Describe your target company", "mode": "competitive" }
```

## Response (success 200)

```json
{
  "decision_recommendation": { "action": "monitor", "confidence_score": 0.8 },
  "intelica_moat_index": 0.7,
  "detected_competitors": ["Player A", "Player B"]
}
```

## Known limitations

- Trial is trust-based: per-key quota is not cryptographically verified.
- Paid endpoints require x402 settlement via a facilitator (CDP/PayAI).

## Resources

- GitHub: https://github.com/teodorofodocrispin-cmyk/Intelica-docs
- Health: https://api.intelica.dev/health
- Docs: https://api.intelica.dev/llms.txt
