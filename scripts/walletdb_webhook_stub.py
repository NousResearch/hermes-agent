#!/usr/bin/env python3
"""Minimal webhook stub for WalletDB (stdlib only).

Usage:
  python scripts/walletdb_webhook_stub.py --port 8787

POST JSON to http://localhost:8787/webhook
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def _json(self, status, body):
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/webhook":
            return self._json(404, {"status": "error", "message": "not_found"})
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else None
        except Exception:
            return self._json(400, {"status": "error", "message": "invalid_json"})

        if not isinstance(payload, dict):
            return self._json(400, {"status": "error", "message": "payload_not_object"})

        missing = [k for k in ("event_id", "event_type", "idempotency_key", "occurred_at", "source", "payload") if k not in payload]
        if missing:
            return self._json(400, {"status": "error", "message": "missing_fields", "fields": missing})

        # TODO: insert idempotency check + durable enqueue here
        resp = {
            "status": "ok",
            "accepted": True,
            "idempotency_key": payload.get("idempotency_key"),
            "message": "queued",
        }
        return self._json(200, resp)

    def log_message(self, fmt, *args):
        # keep stdout clean in cron usage
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Webhook stub listening on http://127.0.0.1:{args.port}/webhook")
    server.serve_forever()


if __name__ == "__main__":
    main()
