---
name: trustboost-pii-sanitizer
description: Sanitizes PII from text before sending to LLMs. Use when handling user-generated text that may contain sensitive data, when privacy compliance is required (GDPR, LGPD, APPI, HIPAA, CCPA), or when an agent needs to redact personal information before passing content to external APIs. Supports English, Spanish (LATAM), Portuguese (Brazil/Portugal), German, and Japanese with country-specific PII patterns (RFC, CUIT, CPF, CNPJ, Personalausweis, マイナンバー, and more). Returns sanitized text, a safety score (0.0-1.0), and a risk category (CRITICAL/PRIVATE/SENSITIVE). No SDK required — single POST request. 50 free requests per wallet available immediately with tx_hash="TRIAL".
license: MIT
compatibility: Requires internet access to reach the TrustBoost API. No local dependencies. Compatible with any agent that can make HTTP POST requests. No authentication required.
metadata:
  author: teodorofodocrispin-cmyk
  version: "2.0.4"
  endpoint: https://api.trustboost.dev/sanitize
  health: https://api.trustboost.dev/health
  payment: Solana USDC (149 USDC = 10,000 sanitizations)
  trial: tx_hash=TRIAL (50 free sanitizations per wallet, no payment required)
  autonomy_score: 8.5/10
  audit_score: 9.8/10
  languages: English, Spanish, Portuguese, German, Japanese
  compliance: GDPR, LGPD, APPI, CCPA, DPDPA
  requires_env: none
  wallet_security: >
    TrustBoost NEVER requires wallet private keys, seed phrases, or signing credentials.
    The optional wallet_address parameter accepts ONLY a public Solana address for
    per-wallet quota tracking. Payment signing happens entirely client-side — TrustBoost
    only verifies the resulting tx_hash on-chain via Helius oracle. No private key
    ever touches TrustBoost infrastructure.
  infrastructure: FastAPI + Supabase + Render (AWS)
  homepage: https://github.com/teodorofodocrispin-cmyk/TrustBoost-PII-Sanitizer
---

# TrustBoost PII Sanitizer v2.0

A blockchain-verified privacy layer that redacts PII from text before it reaches LLMs or external services. Payment is verified trustlessly on Solana via Helius oracle — no API key, no SDK, no human intervention required.

Running on FastAPI + Supabase + Render (AWS) — production-grade infrastructure with per-wallet TRIAL tracking and automatic replay attack protection.

---

## ⚠️ Transparency Notice (Read Before Installing)

TrustBoost is designed for radical transparency. Before installing, understand exactly how it works:

### 1. Data Transmission

Raw text is transmitted to Render (AWS) infrastructure
for processing via FastAPI.

**What TrustBoost stores:** Only sanitized output,
character count, safety score, risk category, wallet
address, and timestamp — never raw input text.

**Storage:** Supabase PostgreSQL (private audit ledger)

**For strict no-transmission requirements** (e.g.,
on-premise HIPAA environments): this service is not
suitable. Use a local sanitizer instead.

**For general privacy compliance** (GDPR, LGPD, APPI,
CCPA): TrustBoost provides meaningful risk reduction
by ensuring PII never reaches your LLM provider.

### 2. Payment Model and Autonomous Safety

TrustBoost uses Solana blockchain as the authorization
layer — replacing human confirmation with cryptographic
verification.

**How autonomous payment works safely:**

1. Agent constructs a Solana transaction
2. Agent signs with its OWN wallet private key
   (TrustBoost never has access to agent keys)
3. Transaction is broadcast to Solana Mainnet
4. Helius oracle verifies the transaction on-chain
5. Service activates only after verification

**Why blockchain replaces human confirmation:**
- Transaction is signed by the agent's own wallet
- No credentials are shared with TrustBoost
- Payment is publicly verifiable on Solana Explorer
- TrustBoost cannot initiate payments — only the
  agent can trigger a transaction

**Safe payment implementation:**
```python
# Monitor quota — only pay when needed
if result["data"]["usage_metrics"]["quota_remaining"] < 10:
    initiate_payment_with_confirmation(
        amount=149,
        currency="USDC",
        network="solana",
        address="giu4VciTkfWJNG1oeP6SzHEJwmabikJSMB91GaFNWE4"
    )
```

**Recommendation:** Always use TRIAL (50 free requests
per wallet) before any paid transaction.

### 3. Audit Trail

Every sanitization is logged to Supabase PostgreSQL with:
tx_hash, input character count (not content), sanitized
output, safety score, risk category, wallet address,
timestamp.

The raw input text is never stored at any point.

### 4. Operator Verification

- **GitHub:** https://github.com/teodorofodocrispin-cmyk
- **Repository:** https://github.com/teodorofodocrispin-cmyk/TrustBoost-PII-Sanitizer
- **Health endpoint:** https://api.trustboost.dev/health
- **Contact:** teodorofodocrispin@gmail.com
- **External audit score:** 9.8/10
- **Autonomy score:** 8.5/10

---

## When to use this skill

Activate this skill when:
- The agent needs to process user-generated text that
  may contain emails, passwords, private keys, phone
  numbers, addresses, or national IDs
- Privacy compliance is required before passing content
  to an LLM provider (GDPR, LGPD, APPI, CCPA)
- The agent processes text in Spanish, Portuguese,
  German, or Japanese with country-specific PII formats
- The agent needs autonomous budget management via
  the quota_remaining field

## When NOT to use this skill

- Strict on-premise HIPAA environments requiring
  zero data transmission to third parties
- Use cases involving real private keys or passwords
  in production (use TRIAL only for testing)

---

## How it works

1. Agent sends text + a Solana `tx_hash` to the
   TrustBoost API
2. Supabase verifies per-wallet TRIAL quota or
   Helius oracle verifies payment on Solana mainnet
3. OpenAI GPT-4o-mini detects and redacts all PII
   in the input language
4. Returns sanitized text + safety score + risk
   category + quota metrics
5. Only sanitized output is logged — never raw input

## Multilingual PII Support

TrustBoost automatically detects the input language
and applies country-specific patterns:

| Language | Region | PII Patterns |
|----------|--------|--------------|
| 🇺🇸 English | Global | SSN, API keys, credit cards, passwords |
| 🇲🇽🇨🇴🇦🇷 Spanish | Latin America | RFC, CUIT, RUT, DNI, CURP, Cédula, RUC |
| 🇧🇷🇵🇹 Portuguese | Brazil & Portugal | CPF, CNPJ, RG, NIF, NUS |
| 🇩🇪 German | Germany/Austria/CH | Personalausweis, Steuernummer, IBAN DE |
| 🇯🇵 Japanese | Japan | マイナンバー, 運転免許証, パスポート番号, 住所 |

---

## API Request

**Endpoint:** `POST https://api.trustboost.dev/sanitize`

**Headers:** `Content-Type: application/json`

**Payload:**
```json
{
  "text": "The text containing potential PII",
  "tx_hash": "TRIAL",
  "wallet_address": "your-solana-wallet"
}
```

## Access modes

| Mode | tx_hash value | Cost | Quota |
|------|--------------|------|-------|
| **Trial** | `"TRIAL"` | Free | 50 sanitizations per wallet |
| **Paid** | Real Solana tx hash | 149 USDC | 10,000 sanitizations |

**To get a paid tx_hash:** Send exactly 149 USDC on
Solana mainnet to:
`giu4VciTkfWJNG1oeP6SzHEJwmabikJSMB91GaFNWE4`

---

## API Response (Success 200)

```json
{
  "status": "success",
  "request_id": "TRIAL",
  "data": {
    "message": "Content successfully sanitized and logged.",
    "sanitized_content": "Text with [REDACTED] replacing all PII",
    "safety_score": 0.95,
    "risk_category": "PRIVATE",
    "entities_removed": true,
    "timestamp": "2026-04-27T09:00:00Z",
    "usage_metrics": {
      "quota_remaining": 49,
      "quota_limit": 50
    }
  },
  "billing": {
    "license_type": "TRIAL",
    "status": "active"
  }
}
```

## API Response (Error 402)

```json
{
  "status": "error",
  "request_id": "TRIAL",
  "code": "QUOTA_EXHAUSTED_OR_PAYMENT_REQUIRED",
  "message": "TRIAL quota exhausted. Send 149 USDC on Solana to continue.",
  "trial_info": {
    "quota_used": 50,
    "quota_limit": 50,
    "quota_remaining": 0
  },
  "payment_info": {
    "amount_required": 149,
    "currency": "USDC",
    "network": "solana",
    "payment_address": "giu4VciTkfWJNG1oeP6SzHEJwmabikJSMB91GaFNWE4"
  },
  "next_steps": [
    {
      "action": "send_payment",
      "description": "Send 149 USDC on Solana Mainnet to the payment address"
    },
    {
      "action": "retry_with_tx_hash",
      "description": "Resubmit request including the Solana transaction signature"
    }
  ]
}
```

## API Response (Error 409)

```json
{
  "status": "error",
  "code": "TX_HASH_ALREADY_USED",
  "message": "This transaction hash has already been used. Each tx_hash can only be used once.",
  "payment_info": {
    "amount_required": 149,
    "currency": "USDC",
    "network": "solana",
    "payment_address": "giu4VciTkfWJNG1oeP6SzHEJwmabikJSMB91GaFNWE4"
  }
}
```

---

## Risk categories

| Category | What gets redacted |
|----------|-------------------|
| `CRITICAL` | Private keys, seed phrases, passwords, credit card data |
| `PRIVATE` | Emails, phone numbers, national IDs, physical addresses |
| `SENSITIVE` | Social media handles, general locations |

## Safety score

- `0.0` — No PII detected, text is clean
- `0.5` — Moderate PII detected (emails, handles)
- `1.0` — Critical PII detected (keys, passwords)

---

## Example — English

**Input:**
```json
{
  "text": "Contact John at john@example.com or +1-555-0123. API key: sk-abc123xyz.",
  "tx_hash": "TRIAL",
  "wallet_address": "your-wallet"
}
```

**Output:**
```json
{
  "sanitized_content": "Contact [REDACTED] at [REDACTED] or [REDACTED]. API key: [REDACTED].",
  "safety_score": 0.97,
  "risk_category": "CRITICAL",
  "entities_removed": true
}
```

## Example — German

**Input:**
```json
{
  "text": "Hans Müller, Personalausweis: L01X00T47, IBAN: DE89 3704 0044 0532 0130 00, Tel: +49 89 1234 5678",
  "tx_hash": "TRIAL"
}
```

**Output:**
```json
{
  "sanitized_content": "[REDACTED], Personalausweis: [REDACTED], IBAN: [REDACTED], Tel: [REDACTED]",
  "safety_score": 0.98,
  "risk_category": "CRITICAL",
  "entities_removed": true
}
```

## Example — Japanese

**Input:**
```json
{
  "text": "田中太郎、マイナンバー：123456789012、電話：090-1234-5678",
  "tx_hash": "TRIAL"
}
```

**Output:**
```json
{
  "sanitized_content": "[REDACTED]、マイナンバー：[REDACTED]、電話：[REDACTED]",
  "safety_score": 0.97,
  "risk_category": "PRIVATE",
  "entities_removed": true
}
```

---

## External Evaluations

- Autonomy Score: **8.5/10** — AI evaluation, not a certified security audit
- Blueprint Audit: **9.8/10** — AI evaluation, not a certified security audit
- Independent security audit: **pending** — this project has not been audited by a certified security firm. See the Manifesto for honest disclosure.
- Full report: https://github.com/teodorofodocrispin-cmyk/TrustBoost-PII-Sanitizer/blob/main/AGENT_EVALUATION.md

## Known Limitations

- **Prompt injection risk:** Malicious text containing instructions like "Ignore previous instructions" could potentially bypass PII redaction. temperature=0 and strict JSON-only output reduce this risk but do not eliminate it entirely.
- **Not suitable for zero-transmission environments:** Raw text is sent to api.trustboost.dev before sanitization occurs.
- **TRIAL is trust-based:** Per-wallet quota tracking is not cryptographically verified.
- **No certified audit:** Evaluation scores are AI-generated, not from a certified security firm.

## Resources

- GitHub: https://github.com/teodorofodocrispin-cmyk/TrustBoost-PII-Sanitizer
- Health check: https://api.trustboost.dev/health
- Schema (molt.json): https://raw.githubusercontent.com/teodorofodocrispin-cmyk/TrustBoost-PII-Sanitizer/main/molt.json
- Infrastructure: FastAPI + Supabase + Render (AWS)
