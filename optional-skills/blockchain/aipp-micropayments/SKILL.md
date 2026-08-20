---
name: aipp-micropayments
description: "Accept L402 Lightning payments with AIPP."
version: 1.0.0
author: Hermes Agent (aipp.dev)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [AIPP, L402, Lightning, X402, USDC, micropayment, monetization, phoenixd]
    category: blockchain
---

# AIPP Micro-Payment Tool (Hermes)

Enables Hermes agents to **monetize premium tools and reasoning** and **pay for external APIs** with Bitcoin Lightning (L402) or Base USDC (X402) micro-payments — zero custody, no signup friction, EU AI Act Article 26 compliant receipts.

## What This Solves

- **Sell** access to a premium tool/inference: issue a payment challenge before returning results (`issue_aipp_charge`).
- **Buy** from external paid APIs: settle a Lightning invoice with a local node (`pay_aipp_invoice`).
- **Prove** settlement with a cryptographic preimage + compliance receipt (`verify_aipp_settlement`).

## Prerequisites

- An AIPP merchant API key (`aipp_merch_...`), obtained by registering a Lightning address at `https://aipp.dev` (Wallet = Identity, no password).
- Python 3.8+ with `requests`.
- **To pay invoices (`pay_aipp_invoice`):** the default path requires **Docker** and a running `aipp-phoenixd` container (`docker exec aipp-phoenixd /phoenix/phoenix-cli payinvoice ...`). To use another wallet, set `AIPP_PAY_CMD` to any compatible payment command (e.g. a direct `phoenix-cli` binary or another Lightning CLI). Issuing charges and verifying settlement do not require Docker.

## Setup

```bash
pip install requests
export AIPP_API_KEY="aipp_merch_..."   # keep it secret — never commit it
```

## Usage

```python
import os
from aipp_micropayments import HermesAippTool

tool = HermesAippTool(api_key=os.environ["AIPP_API_KEY"])

# 1. Issue a micro-payment charge ($0.01 = ~16 sats L402)
charge = tool.issue_aipp_charge(amount_usd=0.01, memo="Premium inference", protocol="L402")
print(charge["payment_request"])  # BOLT11 invoice
print(charge["payment_hash"])

# 2. After the user pays, verify settlement
receipt = tool.verify_aipp_settlement(charge["payment_hash"])
# Always check `status` first: failure responses may contain only `error`.
if receipt.get("status") == "FAILED":
    print(receipt.get("error"))
else:
    print(receipt.get("preimage"))
# -> {"status": "SETTLED", "preimage": "...", "receipt_id": "rec_...", "compliance": "EU AI Act Art. 26"}
```

## API Reference

| Method | Description | Returns |
|--------|-------------|---------|
| `issue_aipp_charge(amount_usd, memo, protocol)` | Issue L402/X402/DUAL payment challenge ($0.01–$100) | `payment_request`, `payment_hash`, `amount_sats`, `checkout_url` |
| `verify_aipp_settlement(payment_hash)` | Check settlement + fetch EU AI Act Art. 26 receipt | `status`, `preimage`, `receipt_id`, `total_amount_usd` |
| `pay_aipp_invoice(payment_request)` | Settle an external BOLT11 invoice via local phoenixd node | `payment_preimage`, `payment_hash`, `routing_fee_sat` |

## Self-Verification Test

```bash
python scripts/selftest.py
```

Issues a real $0.01 L402 charge, settles it, and verifies the receipt — end-to-end proof the tool works.

## Security Notes

- **Never hardcode API keys.** Read from environment or a secrets manager. The bundled `selftest.py` reads `AIPP_API_KEY` from env.
- API keys are checked against the merchant DB on every request (`401 Invalid AIPP API key` otherwise) — an unknown key is rejected, not silently accepted.
- Charges are capped at $0.01–$100 per invoice to bound exposure.

## Related

- AIPP docs: `https://aipp.dev/docs`
