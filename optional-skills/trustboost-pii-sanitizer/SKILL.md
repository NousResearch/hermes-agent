---
name: trustboost-pii-sanitizer
description: Redact PII via the TrustBoost API before LLM calls.
license: MIT
compatibility: Requires internet access to reach the TrustBoost API. No local dependencies. Compatible with any agent that can make HTTP POST requests. No authentication required.
metadata:
  author: teodorofodocrispin-cmyk
  version: "2.0.5"
  endpoint: https://api.trustboost.dev/sanitize
  preview_endpoint: https://api.trustboost.dev/sanitize/preview
  health: https://api.trustboost.dev/health
  languages: English, Spanish, Portuguese, German, Japanese
  homepage: https://github.com/teodorofodocrispin-cmyk/TrustBoost-PII-Sanitizer
---

# TrustBoost PII Sanitizer

An **opt-in hosted transformation** that redacts PII from text. You send raw text
to the TrustBoost API (`api.trustboost.dev`); TrustBoost redacts it with an LLM
(GPT-4o-mini) and returns the sanitized text plus a safety score. The raw text
leaves your machine and reaches a third-party service **before** redaction — this
is a hosted sanitization step, not a local pre-LLM guard. Use it when you want a
convenient redaction pass, but be aware the unredacted text transits TrustBoost's
infrastructure (FastAPI + Supabase + Render, AWS).

## Transparency notice

- **Raw text is transmitted to TrustBoost** for processing. If your policy forbids
  sending unredacted PII to third parties (strict on-premise HIPAA, etc.), do NOT
  use this skill — redact locally instead.
- Only sanitized output, character count, safety score, risk category, and timestamp
  are stored. Raw input text is never persisted.
- No API key or SDK required. The free preview endpoint needs no wallet.

## When to use

- You want a quick PII redaction pass on user-generated text before logging,
  storing, or forwarding it.
- The text is in English, Spanish (LATAM), Portuguese, German, or Japanese and
  may contain country-specific identifiers (RFC, CUIT, CPF, CNPJ, Personalausweis,
  マイナンバー, etc.).

## When NOT to use

- Strict zero-transmission environments (HIPAA on-premise, air-gapped).
- Any flow where unredacted PII must never leave the local machine.

## How it works

1. You POST the raw text to the TrustBoost API.
2. TrustBoost redacts PII with GPT-4o-mini and returns sanitized text + a
   safety score (0.0–1.0) + risk category (CRITICAL / PRIVATE / SENSITIVE).
3. Only the sanitized output is stored server-side; raw input is never persisted.

## Free preview (no wallet, no payment)

`POST https://api.trustboost.dev/sanitize/preview`

```json
{ "text": "Contact John at john@example.com or +1-555-0123. API key: sk-abc123xyz." }
```

Returns the same shape as the paid endpoint (sanitized_content + safety_score +
risk_category). 3 free requests per IP per hour — enough to verify behavior
without any wallet or payment.

## Paid usage (optional, user-controlled)

Payment is **out of band and user-controlled** — the skill never instructs the
agent to sign transactions or spend funds. If you want the paid tier (10,000
sanitizations per 149 USDC on Solana), a human operator performs the transfer and
supplies the resulting `tx_hash`. The skill only consumes a `tx_hash` value you
provide; it never constructs, signs, or broadcasts a transaction.

Trial mode: send `tx_hash: "TRIAL"` to `POST /sanitize` for 50 free
sanitizations per wallet (trust-based quota, no payment).

## API request (trial / paid)

**Endpoint:** `POST https://api.trustboost.dev/sanitize`
**Headers:** `Content-Type: application/json`

```json
{
  "text": "The text containing potential PII",
  "tx_hash": "TRIAL"
}
```

For paid usage, replace `"TRIAL"` with the Solana tx_hash a human operator
provides after transferring 149 USDC to the address published by the service's
`/preflight` endpoint. The skill does not embed or instruct that transfer.

## Response (success 200)

```json
{
  "status": "success",
  "data": {
    "sanitized_content": "Text with [REDACTED] replacing all PII",
    "safety_score": 0.95,
    "risk_category": "PRIVATE",
    "entities_removed": true
  }
}
```

## Risk categories

| Category | What gets redacted |
|----------|-------------------|
| `CRITICAL` | Private keys, seed phrases, passwords, credit card data |
| `PRIVATE` | Emails, phone numbers, national IDs, physical addresses |
| `SENSITIVE` | Social media handles, general locations |

## Known limitations

- **Prompt injection risk:** malicious input with instructions like "ignore
  previous instructions" could potentially affect redaction. `temperature=0` and
  strict JSON-only output reduce but do not eliminate this.
- **Not zero-transmission:** raw text is sent to `api.trustboost.dev` before
  sanitization.
- **Trial is trust-based:** per-wallet quota is not cryptographically verified.
- **No certified audit:** evaluation scores are AI-generated, not from a
  certified security firm.

## Resources

- GitHub: https://github.com/teodorofodocrispin-cmyk/TrustBoost-PII-Sanitizer
- Health: https://api.trustboost.dev/health
- Preview (no wallet): https://api.trustboost.dev/sanitize/preview
