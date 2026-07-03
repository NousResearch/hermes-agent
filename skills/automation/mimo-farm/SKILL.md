---
name: mimo-farm
description: "Automated Xiaomi MiMo Platform account registration — reCAPTCHA Enterprise bypass + IMAP code verification in single-process Python/Playwright."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [Xiaomi, MiMo, Account-Registration, reCAPTCHA, Capsolver, IMAP, Playwright, Farming]
    related_skills: []
---

# MimoFarm — Xiaomi MiMo Account Registration Automation

Single-process Python script automating Xiaomi MiMo platform account creation.
Playwright (headless Chromium) + Capsolver (reCAPTCHA Enterprise) + IMAP (Gmail code extraction).
No temp files, no Node.js dependency.

## Architecture

```
Register Page → Fill Form → Click Next → CAPTCHA appears
     → Solve reCAPTCHA via Capsolver API
     → Inject token + trigger ___grecaptcha_cfg callback
     → MiVerify detects success → API call #2 (result:ok)
     → Navigate to /register/email/verify
     → IMAP poll for 6-digit code
     → Fill code → Submit → Redirect to console/balance
```

## Prerequisites

```bash
pip3 install playwright requests
playwright install chromium
```

## Environment Variables

| Var | Required | Description |
|---|---|---|
| `CAPSOLVER_API_KEY` | ✅ | Capsolver API key (starts with `CAP-`) |
| `IMAP_USER` | ✅ | Gmail address for receiving Xiaomi codes |
| `IMAP_PASS` | ✅ | Gmail app password (16-char, spaces OK) |
| `IMAP_HOST` | ❌ | Default: `imap.gmail.com` |
| `IMAP_PORT` | ❌ | Default: `993` |

## Usage — Single Account

```bash
CAPSOLVER_API_KEY=CAP-XXXX \
IMAP_USER="you@gmail.com" \
IMAP_PASS="xxxx xxxx xxxx xxxx" \
python3 mimo-farm/scripts/xiaomi_register.py \
  --email cf15@nounrich.works --password "Kontol22@" --ref QFQAKP
```

## Usage — Batch

```bash
for i in $(seq 15 30); do
  CAPSOLVER_API_KEY=CAP-XXXX \
  IMAP_USER="you@gmail.com" \
  IMAP_PASS="xxxx xxxx xxxx xxxx" \
  python3 mimo-farm/scripts/xiaomi_register.py \
    --email "cf${i}@nounrich.works" --password "Kontol22@" --ref QFQAKP
  sleep 5  # rate limit buffer
done
```

## CLI Arguments

| Arg | Required | Default | Description |
|---|---|---|---|
| `--email` | ✅ | — | Email address to register |
| `--password` | ✅ | — | Account password (8–16 chars, 2+ of: digits, letters, symbols) |
| `--ref` | ❌ | `QFQAKP` | Referral code |
| `--captcha-site-key` | ❌ | `6LeBM0ocAAAAAEwYcFUjtxpVbs-0rnbSVXBBXmh4` | reCAPTCHA Enterprise site key |
| `--max-imap-wait` | ❌ | `120` | Max seconds to wait for email code |

## Key Technical Details

### reCAPTCHA Enterprise Bypass

- **Site key**: `6LeBM0ocAAAAAEwYcFUjtxpVbs-0rnbSVXBBXmh4`
- **Type**: Enterprise v2 (not standard v2)
- **Capsolver task type**: `ReCaptchaV2EnterpriseTaskProxyLess`
- **Page URL for solving**: `https://global.account.xiaomi.com/fe/service/register`

### Token Injection Flow

Xiaomi wraps Google reCAPTCHA with its MiVerify panel. The first `sendEmailRegTicket`
API call always returns `CAPTCHA_VERIFY_ERROR` (code 87001). The callback mechanism
triggers a **second** API call with the token that succeeds.

Injection JS steps:
1. Set `#g-recaptcha-response` textarea value to solved token
2. Walk `window.___grecaptcha_cfg.clients` to find and call the `callback` function
3. Optionally try `window.miVerify.onCaptchaVerify(token)` and dispatch `recaptcha-success` event

### IMAP Code Extraction

- Polls every 5 seconds for up to 120s
- Searches `FROM "xiaomi"` in INBOX
- Parses email date with `email.utils.parsedate_to_datetime`
- Only considers emails **after** `reg_start_time` (UTC) to avoid stale codes
- Extracts first 6-digit match from email body (text/plain preferred, fallback text/html)

### Known Error Codes

| Code | Meaning | Action |
|---|---|---|
| `87001` | CAPTCHA_VERIFY_ERROR | Expected on 1st call. 2nd call should succeed |
| `20332` | Email send rate limit | Wait or use different email |
| `200010` | Account already exists | Skip this email |

### Success Indicator

After Submit, page redirects to `platform.xiaomimimo.com/console/balance?userId=XXXXXXXXXX`.

## Pitfalls

1. **Capsolver token expires fast** (~2 min). Solve CAPTCHA right before browser flow, not minutes ahead.
2. **Email rate limit** — same email blocked after ~5 attempts. Use fresh email per registration.
3. **reCAPTCHA anchor iframe** — sometimes available, sometimes not. Token injection via `___grecaptcha_cfg` callback works without it.
4. **Verification codes expire** — typically valid for a few minutes. IMAP polling starts immediately after verify page loads.
5. **Password requirements** — 8–16 chars, at least 2 of: digits, letters, special symbols.
