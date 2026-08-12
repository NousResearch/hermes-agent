"""Loopback shim server for the desktop update hand-off.

Two GET routes: / serves ui.html, /progress serves the status file the
orchestrator script writes ({"status": "running"|"done"|"error", ...}).
Exists because a file:// page cannot receive events from a detached
process. Prints the chosen ephemeral port on stdout, serves until killed.
"""

import http.server
import json
import socketserver
import sys

html_path, status_path = sys.argv[1], sys.argv[2]
with open(html_path, "rb") as f:
    HTML = f.read()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - base class signature
        pass

    def do_GET(self):
        if self.path.startswith("/progress"):
            try:
                with open(status_path, "rb") as f:
                    body = f.read()
                json.loads(body)
            except Exception:
                body = b'{"status":"running","message":""}'
            ctype = "application/json; charset=utf-8"
        elif self.path == "/":
            body, ctype = HTML, "text/html; charset=utf-8"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


with socketserver.TCPServer(("127.0.0.1", 0), Handler) as srv:
    print(srv.server_address[1], flush=True)
    srv.serve_forever()
