#!/usr/bin/env python3
"""spraay_gateway.py — Thin wrapper around the Spraay x402 gateway REST API.

Usage:
    python spraay_gateway.py scan                           # List categories and primitives
    python spraay_gateway.py quote <primitive> <chain> [json_params]
    python spraay_gateway.py execute <primitive> <chain> <sender> [json_params]
    python spraay_gateway.py health                         # Check gateway status

Designed to be invoked by Hermes Agent via the terminal tool.
"""

import json
import os
import sys
import urllib.request
import urllib.error

GATEWAY_URL = os.environ.get("SPRAAY_GATEWAY_URL", "https://gateway.spraay.app")


def _request(method: str, path: str, data: dict | None = None, headers: dict | None = None) -> dict:
    """Make an HTTP request to the gateway and return parsed JSON."""
    url = f"{GATEWAY_URL}{path}"
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)

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


def cmd_execute(primitive: str, chain: str, sender: str, params_json: str = "{}"):
    """Execute a primitive (returns unsigned transaction)."""
    params = json.loads(params_json)
    params["primitive"] = primitive
    params["chain"] = chain
    params["sender"] = sender
    result = _request("POST", "/execute", data=params)
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
            print("Usage: spraay_gateway.py quote <primitive> <chain> [json_params]")
            sys.exit(1)
        params_json = sys.argv[4] if len(sys.argv) > 4 else "{}"
        cmd_quote(sys.argv[2], sys.argv[3], params_json)
    elif command == "execute":
        if len(sys.argv) < 5:
            print("Usage: spraay_gateway.py execute <primitive> <chain> <sender> [json_params]")
            sys.exit(1)
        params_json = sys.argv[5] if len(sys.argv) > 5 else "{}"
        cmd_execute(sys.argv[2], sys.argv[3], sys.argv[4], params_json)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
