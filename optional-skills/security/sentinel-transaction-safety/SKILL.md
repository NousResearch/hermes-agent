---
name: sentinel-transaction-safety
description: Get a pre-execution safety verdict via SENTINEL.
version: "1.0.0"
author: teodorofodocrispin-cmyk
license: MIT
metadata:
  hermes:
    tags: [Security, Blockchain, x402, Base, Transaction-Safety]
    category: security
    related_skills: [mpp-agent, evm]
---

# Sentinel Transaction Safety Skill

Calls the hosted SENTINEL API to get a SAFE/UNSAFE/UNKNOWN verdict on a Base
transaction before it's signed. It does not sign or broadcast anything itself
— it only returns a risk assessment for the calling agent to act on.

## When to Use

- An agent is about to sign a transaction on Base and wants a pre-flight
  check for rug pulls, honeypots, or malicious contracts.
- The user asks to "check if this contract is safe" or "verify this
  transaction before I sign it".

## Prerequisites

- `terminal` tool to run `curl`.
- An x402-capable wallet on Base with a small USDC balance — every paid call
  costs $0.005. There is no free trial for the verdict endpoint itself.
- Any client that can sign an EIP-3009 `transferWithAuthorization` and attach
  it as an `X-PAYMENT` header. Options: the `agentcash` MCP tool if
  available, `npx agentcash onboard` (see the `mpp-agent` skill's wallet
  table), or `mppx <url> --method POST --data '<json>'`.

## How to Run

Free discovery calls (no payment, no wallet needed):

```
curl https://sentinel-agent.dev/health
curl https://sentinel-agent.dev/pricing
```

## Quick Reference

| Call | Cost | Auth |
|------|------|------|
| `GET /health` | free | none |
| `GET /pricing` | free | none |
| `POST /v1/guard` | $0.005 USDC (Base) | x402 `X-PAYMENT` header |

## Procedure

### 1. Probe the endpoint without payment

```
curl -X POST https://sentinel-agent.dev/v1/guard \
  -H "Content-Type: application/json" \
  -d '{"chain":"base","from":"0xYourWallet","tx":{"to":"0xTarget","data":"0x","value":"0x0"}}'
```

This returns `HTTP 402` with an `accepts` array — confirm the price and
`payTo` address before paying.

### 2. Pay and retry with the wallet CLI

Using `mppx` (or the equivalent step for whichever wallet client the user
has funded — see `mpp-agent`):

```
mppx https://sentinel-agent.dev/v1/guard --method POST \
  --data '{"chain":"base","from":"0xYourWallet","tx":{"to":"0xTarget","data":"0x","value":"0x0"}}'
```

The client handles the 402 challenge/payment dance and prints SENTINEL's
actual JSON verdict on success.

### 3. Read the verdict

```json
{
  "verdict": "SAFE",
  "sentinelScore": 94,
  "grade": "AAA",
  "signature": "ed25519:...",
  "signer": "sentinel-agent.dev"
}
```

`verdict` is one of `SAFE`, `UNSAFE`, `UNKNOWN`. `sentinelScore` is 0-100.

## Pitfalls

- **No free trial.** Every `/v1/guard` call requires payment — don't expect
  a 200 without an `X-PAYMENT` header.
- **Verdicts aren't a certified audit.** SENTINEL combines rule-based checks
  (GoPlus, Alchemy simulation, honeypot.is) with an LLM council — treat
  `UNKNOWN` as "insufficient signal", not "safe by default".
- **Wallet keys never enter agent context.** Whichever client signs the
  x402 authorization stores its keys under its own config — do not
  `read_file` them into the transcript.

## Verification

```
curl -s https://sentinel-agent.dev/health | python3 -c "import json,sys; d=json.load(sys.stdin); assert d.get('status')=='ok'; print('OK')"
```
