#!/usr/bin/env python3
"""Inference relay for sandboxed cells: the ONLY way out of a cell's network namespace.

    relay.py <unix socket path> <log.jsonl>

Listens on a unix socket that the battery binds into each bubblewrap sandbox. Forwards
``/v1/...`` requests to the Nous inference host over TLS and adds the agent key itself (read
fresh from ``$HERMES_HOME/auth.json`` per request), so the key never enters a cell. Anything
else is refused. One JSON line per request goes to the log: that file is the record of every
outbound call a battery made.
"""
import http.client
import json
import os
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit

SOCK, LOG = sys.argv[1], sys.argv[2]
AUTH = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes") / "auth.json"
_HOP = {"host", "authorization", "connection", "proxy-connection", "keep-alive", "content-length", "transfer-encoding"}
_lock = threading.Lock()


def _upstream():
    nous = json.loads(AUTH.read_text(encoding="utf-8"))["providers"]["nous"]
    url = urlsplit(nous["inference_base_url"])
    return url.hostname, nous["agent_key"]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # unix-socket peers have no address to format
        pass

    def _relay(self):
        t0, status, sent, tail = time.time(), 0, 0, b""
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        host, key = _upstream()
        try:
            if not self.path.startswith("/v1/"):
                status = 403
                self.send_error(403, "relay forwards /v1/ only")
                return
            conn = http.client.HTTPSConnection(host, timeout=600)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
            headers.update({"Host": host, "Authorization": f"Bearer {key}", "Content-Length": str(len(body))})
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            status = resp.status
            self.send_response_only(status)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            while chunk := resp.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
                sent += len(chunk)
                tail = (tail + chunk)[-1500:]
            conn.close()
        except (OSError, http.client.HTTPException) as exc:
            status = status or 502
            tail = repr(exc).encode()
        finally:
            self.close_connection = True
            with _lock, open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": round(t0, 1), "method": self.command, "path": self.path, "upstream": host,
                                    "status": status, "req_bytes": len(body), "resp_bytes": sent,
                                    "wall_s": round(time.time() - t0, 1),
                                    "tail": tail.decode("utf-8", "replace") if status != 200 else ""}) + "\n")

    do_GET = do_POST = _relay


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


if __name__ == "__main__":
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    with Server(SOCK, Handler) as srv:
        srv.serve_forever()
