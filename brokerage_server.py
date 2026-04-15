#!/usr/bin/env python3
"""Standalone runner for the Hermes Brokerage FastAPI service.

Usage:
    python brokerage_server.py [--host HOST] [--port PORT] [--token TOKEN] [--reload]

Defaults:
    Host: 127.0.0.1
    Port: 8787
    Token: BROKERAGE_SERVICE_TOKEN env var, or "dev-token" in dev mode

This script starts the FastAPI app defined in brokerage/app.py using uvicorn.
It is intended for local development and as the entry point for `hermes brokerage start`.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on the path so `brokerage` package is importable
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Start the Hermes Brokerage FastAPI service")
    parser.add_argument("--host", default=os.environ.get("BROKERAGE_HOST", "127.0.0.1"),
                        help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("BROKERAGE_PORT", "8787")),
                        help="Bind port (default: 8787)")
    parser.add_argument("--token", default=os.environ.get("BROKERAGE_SERVICE_TOKEN", ""),
                        help="Bearer auth token (default: BROKERAGE_SERVICE_TOKEN env var)")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")
    parser.add_argument("--no-auth", action="store_true",
                        help="Disable bearer token authentication (dev only)")

    args = parser.parse_args()

    # Set env vars that the FastAPI app reads at startup
    if args.token:
        os.environ["BROKERAGE_SERVICE_TOKEN"] = args.token
    elif args.no_auth:
        os.environ["BROKERAGE_SERVICE_TOKEN"] = ""
    elif not os.environ.get("BROKERAGE_SERVICE_TOKEN"):
        # Dev mode: generate a random token and print it
        import secrets
        dev_token = f"dev-{secrets.token_hex(8)}"
        os.environ["BROKERAGE_SERVICE_TOKEN"] = dev_token
        print(f"⚠  No token set. Using generated dev token: {dev_token}")
        print(f"   Add to ~/.hermes/.env: BROKERAGE_SERVICE_TOKEN={dev_token}")

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is not installed. Install with: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    print(f"Starting Hermes Brokerage Service on {args.host}:{args.port}")
    print(f"  Auth: {'disabled' if args.no_auth else 'bearer token required'}")
    print(f"  Health: http://{args.host}:{args.port}/healthz")

    uvicorn.run(
        "brokerage.app:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )


if __name__ == "__main__":
    main()
