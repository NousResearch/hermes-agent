#!/usr/bin/env python3
"""
Hermes Agent — AIPP self-test: issue -> pay -> verify (L402 Lightning).

Requires:
  - AIPP_API_KEY env var (aipp_merch_...), or --key argument
  - A running phoenixd node accessible via `docker exec aipp-phoenixd` (default)
    or a PAY_CMD override.

Usage:
  AIPP_API_KEY=aipp_merch_... python3 selftest.py
"""
import os
import sys
import json
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aipp_micropayments import HermesAippTool

API_KEY = os.environ.get("AIPP_API_KEY", "")
PAY_CMD = os.environ.get(
    "AIPP_PAY_CMD",
    "docker exec aipp-phoenixd /phoenix/phoenix-cli payinvoice --invoice={invoice}",
)
BASE_URL = os.environ.get("AIPP_BASE_URL", "https://aipp.dev")

if not API_KEY:
    print("ERROR: set AIPP_API_KEY (aipp_merch_...) first", file=sys.stderr)
    sys.exit(1)

print("=" * 60)
print("[TEST 1/3] Issue L402 charge for $0.01 (~16 sats)...")
tool = HermesAippTool(api_key=API_KEY, base_url=BASE_URL)
charge = tool.issue_aipp_charge(amount_usd=0.01, memo="Hermes Autonomous Self-Upgrade")
if charge.get("status") != "PAYMENT_REQUIRED":
    print("CHARGE FAILED:", charge, file=sys.stderr)
    sys.exit(1)
print("  payment_request:", charge["payment_request"][:50] + "...")
print("  payment_hash   :", charge["payment_hash"])
print("  amount_sats    :", charge["amount_sats"])

print("[TEST 2/3] Settle invoice via local node...")
cmd = PAY_CMD.format(invoice=charge["payment_request"]).split()
res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode != 0:
    print("  PAY FAILED:", res.stderr[:300], file=sys.stderr)
    sys.exit(1)
try:
    pay = json.loads(res.stdout)
except json.JSONDecodeError:
    pay = {"raw": res.stdout}
print("  paymentPreimage :", pay.get("paymentPreimage"))
print("  paymentHash     :", pay.get("paymentHash"))
print("  recipientAmount :", pay.get("recipientAmountSat"), "sats")
print("  routingFee      :", pay.get("routingFeeSat"), "sats")

print("[TEST 3/3] Verify settlement & EU AI Act receipt...")
time.sleep(2)
verify = tool.verify_aipp_settlement(charge["payment_hash"])
print("  status      :", verify.get("status"))
if verify.get("status") == "SETTLED":
    print("  preimage    :", verify["preimage"])
    print("  receipt_id  :", verify["receipt_id"])
    print("  compliance  :", verify["compliance"])
    print("=" * 60)
    print(">>> SUCCESS: AIPP tool 100% operational & verified! <<<")
    print("=" * 60)
else:
    print("  message     :", verify.get("message"))
    sys.exit(1)
