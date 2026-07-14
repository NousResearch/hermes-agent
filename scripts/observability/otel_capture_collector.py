#!/usr/bin/env python3
"""Tiny local OTLP/HTTP capture collector for Hermes gateway health smoke tests.

This is not a production collector. It accepts OTLP protobuf POSTs on /v1/traces
and /v1/metrics, records request metadata as JSONL, and returns 200 so local
exporters can be exercised without Docker or a vendor backend.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class CaptureHandler(BaseHTTPRequestHandler):
    log_path: Path

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        record = {
            "ts": time.time(),
            "path": self.path,
            "content_type": self.headers.get("content-type"),
            "content_length": length,
            "body_prefix_hex": body[:24].hex(),
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format: str, *args) -> None:
        # Keep tmux panes clean; JSONL file is the assertion surface.
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4318)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    CaptureHandler.log_path = log_path
    server = ThreadingHTTPServer((args.host, args.port), CaptureHandler)
    print(f"OTLP capture collector listening on http://{args.host}:{args.port}; log={log_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
