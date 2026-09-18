#!/usr/bin/env python3
"""Minimal production healthcheck for Hermes Agent.

- Liveness: process responds and Python runtime is available.
- Readiness: critical Hermes modules import successfully.

Usage:
  python scripts/healthcheck.py
  python scripts/healthcheck.py --serve --port 8787
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

READINESS_IMPORTS = [
    "hermes_cli.main",
    "run_agent",
    "model_tools",
]


def check_liveness() -> tuple[bool, dict[str, Any]]:
    payload = {
        "status": "ok",
        "pid": os.getpid(),
        "python": sys.version.split()[0],
    }
    return True, payload


def check_readiness() -> tuple[bool, dict[str, Any]]:
    errors: list[dict[str, str]] = []
    for module in READINESS_IMPORTS:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append({"module": module, "error": f"{type(exc).__name__}: {exc}"})

    if errors:
        return False, {"status": "fail", "errors": errors}
    return True, {"status": "ok", "checked_modules": READINESS_IMPORTS}


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/live":
            ok, payload = check_liveness()
            self._send_json(200 if ok else 503, payload)
            return

        if self.path == "/ready":
            ok, payload = check_readiness()
            self._send_json(200 if ok else 503, payload)
            return

        self._send_json(404, {"status": "fail", "error": "not found", "path": self.path})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Hermes production liveness/readiness checks")
    parser.add_argument("--serve", action="store_true", help="Start HTTP server with /live and /ready")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    if args.serve:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        print(f"healthcheck server listening on http://{args.host}:{args.port}")
        server.serve_forever()

    live_ok, live_payload = check_liveness()
    ready_ok, ready_payload = check_readiness()
    payload = {"live": live_payload, "ready": ready_payload}
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if live_ok and ready_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
