#!/usr/bin/env python3
"""spraay_gateway.py — Thin wrapper around the Spraay x402 gateway REST API.

Usage:
    python spraay_gateway.py health                         # Check gateway status
    python spraay_gateway.py scan                           # List categories and primitives
    python spraay_gateway.py quote <primitive> <chain> [json_params]
    python spraay_gateway.py execute <primitive> <chain> <sender> --payment <header> [json_params]

The execute command requires an X-402-Payment header. Obtain the payment amount
from the quote command, sign it with the agent's wallet, and pass the signed
header via --payment.

Designed to be invoked by Hermes Agent via the terminal tool.
"""

import json
import os
import sys
import urllib.request
import urllib.error

GATEWAY_URL = os.environ.get(
    "SPRAAY_GATEWAY_URL", "https://gateway.spraay.app"
)


def _request(
    method: str,
    path: str,
    data: dict | None = None,
    headers: dict | None = None,
) -> dict:
    """Make an HTTP request to the gateway and return parsed JSON."""
    url = f"{GATEWAY_URL}{path}"
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        url, data=body, headers=req_headers, method=method
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        return {"error": True, "status": e.code, "message": error_body}
    except urllib.error.URLError as e:
        return {"error": True, "message": str(e.reason)}


def cmd_health():
    """Check gateway health."""
    result = _request("GET", "/health")
    print(json.dumps(result, indent=2))


def cmd_scan():
    """List all available categories and primitives."""
    result = _request("GET", "/scan")
    print(json.dumps(result, indent=2))


def cmd_quote(primitive: str, chain: str, params_json: str = "{}"):
    """Get a price quote for a primitive."""
    params = json.loads(params_json)
    params["primitive"] = primitive
    params["chain"] = chain
    result = _request("POST", "/quote", data=params)
    print(json.dumps(result, indent=2))


def cmd_execute(
    primitive: str,
    chain: str,
    sender: str,
    payment_header: str,
    params_json: str = "{}",
):
    """Execute a primitive with an x402 payment header.

    The payment_header is the signed X-402-Payment value obtained after
    quoting and signing the required micro-payment with the agent's wallet.
    """
    if not payment_header:
        print(
            "Error: --payment <header> is required for execute.\n"
            "Run 'quote' first to get the payment amount, sign it with\n"
            "the agent's wallet, then pass the signed header here."
        )
        sys.exit(1)

    params = json.loads(params_json)
    params["primitive"] = primitive
    params["chain"] = chain
    params["sender"] = sender

    result = _request(
        "POST",
        "/execute",
        data=params,
        headers={"X-402-Payment": payment_header},
    )
    print(json.dumps(result, indent=2))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "health":
        cmd_health()

    elif command == "scan":
        cmd_scan()

    elif command == "quote":
        if len(sys.argv) < 4:
            print(
                "Usage: spraay_gateway.py quote <primitive> <chain>"
                " [json_params]"
            )
            sys.exit(1)
        params_json = sys.argv[4] if len(sys.argv) > 4 else "{}"
        cmd_quote(sys.argv[2], sys.argv[3], params_json)

    elif command == "execute":
        if len(sys.argv) < 5:
            print(
                "Usage: spraay_gateway.py execute <primitive> <chain>"
                " <sender> --payment <header> [json_params]"
            )
            sys.exit(1)

        # Parse --payment flag
        payment_header = ""
        remaining_args = sys.argv[5:]
        params_json = "{}"

        i = 0
        while i < len(remaining_args):
            if remaining_args[i] == "--payment" and i + 1 < len(remaining_args):
                payment_header = remaining_args[i + 1]
                i += 2
            else:
                # Treat as params JSON
                params_json = remaining_args[i]
                i += 1

        cmd_execute(
            sys.argv[2], sys.argv[3], sys.argv[4], payment_header, params_json
        )

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
