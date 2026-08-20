"""
AIPP Micro-Payment Tool for Hermes Agent.

Enables Hermes agents to monetize premium tools and pay for external APIs
with Bitcoin Lightning (L402) or Base USDC (X402) micro-payments.

- issue_aipp_charge: Issue an L402/X402 payment challenge ($0.01 - $100)
- verify_aipp_settlement: Verify cryptographic settlement + EU AI Act Art. 26 receipt
- pay_aipp_invoice: Settle a Lightning invoice via a local phoenixd node
"""

import json
import os
import shlex
import subprocess
from typing import Any, Dict, Optional

import requests

__all__ = ["HermesAippTool"]


class HermesAippTool:
    """Official AIPP micro-payment client for Hermes Agent."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://aipp.dev",
    ):
        self.api_key = api_key or os.environ.get("AIPP_API_KEY", "")
        if not self.api_key:
            raise ValueError("AIPP API key required (pass api_key= or set AIPP_API_KEY)")
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------ #
    # JSON function definitions for agent tool-calling schemas            #
    # ------------------------------------------------------------------ #
    def get_hermes_function_definitions(self) -> list:
        return [
            {
                "name": "issue_aipp_charge",
                "description": "Issue an L402 Bitcoin Lightning or Base USDC payment challenge before executing a premium tool or delivering private knowledge.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount_usd": {
                            "type": "number",
                            "description": "Amount to charge in USD (0.01 to 100.0).",
                        },
                        "memo": {
                            "type": "string",
                            "description": "Description of the work or inference being monetized.",
                        },
                        "protocol": {
                            "type": "string",
                            "enum": ["L402", "X402", "DUAL"],
                            "description": "L402 = Bitcoin Lightning, X402 = Base USDC, DUAL = Both.",
                        },
                    },
                    "required": ["amount_usd", "memo"],
                },
            },
            {
                "name": "verify_aipp_settlement",
                "description": "Verify if an invoice has been cryptographically settled and retrieve the EU AI Act Article 26 receipt.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "payment_hash_or_preimage": {
                            "type": "string",
                            "description": "The payment hash or 32-byte cryptographic preimage string.",
                        }
                    },
                    "required": ["payment_hash_or_preimage"],
                },
            },
            {
                "name": "pay_aipp_invoice",
                "description": "Pay a Bitcoin Lightning (L402) invoice to consume an external paid API or data source.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "payment_request": {
                            "type": "string",
                            "description": "The BOLT11 Lightning invoice string starting with lnbc...",
                        }
                    },
                    "required": ["payment_request"],
                },
            },
        ]

    # ------------------------------------------------------------------ #
    # Core operations                                                     #
    # ------------------------------------------------------------------ #
    def issue_aipp_charge(self, amount_usd: float, memo: str, protocol: str = "L402") -> Dict[str, Any]:
        """Issue a micro-payment challenge."""
        try:
            res = requests.post(
                f"{self.base_url}/invoice/create",
                json={"amount_usd": amount_usd, "memo": memo, "protocol": protocol},
                headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
                timeout=30,
            )
            res.raise_for_status()
            data = res.json()
            return {
                "status": "PAYMENT_REQUIRED",
                "http_status": 402,
                "amount_sats": data.get("amount_sats"),
                "amount_usd": amount_usd,
                "payment_request": data.get("payment_request"),
                "payment_hash": data.get("payment_hash"),
                "checkout_url": f"{self.base_url}/pay/{data.get('payment_hash')}",
                "instructions": f"Pay via Lightning to unlock execution.",
            }
        except Exception as e:
            return {"error": str(e), "status": "FAILED"}

    def verify_aipp_settlement(self, payment_hash_or_preimage: str) -> Dict[str, Any]:
        """Verify cryptographic settlement and return compliance receipt."""
        try:
            res = requests.get(
                f"{self.base_url}/invoice/receipt/{payment_hash_or_preimage}",
                headers={"x-api-key": self.api_key},
                timeout=30,
            )
            if res.status_code == 404:
                # Not settled yet
                status = requests.get(
                    f"{self.base_url}/invoice/status/{payment_hash_or_preimage}",
                    headers={"x-api-key": self.api_key},
                    timeout=30,
                )
                st = status.json()
                return {
                    "status": "SETTLED" if st.get("paid") else "PENDING",
                    "paid": bool(st.get("paid")),
                    "message": "Payment not yet confirmed on network."
                    if not st.get("paid")
                    else "Settled.",
                }
            res.raise_for_status()
            receipt = res.json()
            return {
                "status": "SETTLED",
                "paid": True,
                "preimage": receipt.get("payment_details", {}).get("proof"),
                "receipt_id": receipt.get("receipt_id"),
                "compliance": "EU AI Act Article 26 Verifiable Receipt",
                "total_amount_usd": receipt.get("financials", {}).get("total_amount"),
            }
        except Exception as e:
            return {"error": str(e), "status": "FAILED"}

    def pay_aipp_invoice(self, payment_request: str) -> Dict[str, Any]:
        """Pay an external Lightning invoice via a local phoenixd node."""
        try:
            pay_cmd = os.environ.get(
                "AIPP_PAY_CMD",
                "docker exec aipp-phoenixd /phoenix/phoenix-cli payinvoice --invoice={invoice}",
            ).format(invoice=payment_request)
            res = subprocess.run(shlex.split(pay_cmd), capture_output=True, text=True, timeout=60)
            if res.returncode != 0:
                return {"status": "PAYMENT_FAILED", "error": res.stderr}
            data = json.loads(res.stdout)
            return {
                "status": "PAID",
                "payment_preimage": data.get("paymentPreimage"),
                "payment_hash": data.get("paymentHash"),
                "recipient_amount_sat": data.get("recipientAmountSat"),
                "routing_fee_sat": data.get("routingFeeSat"),
            }
        except Exception as e:
            return {"error": str(e), "status": "FAILED"}
