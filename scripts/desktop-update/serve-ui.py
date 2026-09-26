"""Loopback shim server for the desktop update hand-off.

Three GET routes: / serves ui.html, /progress serves the status file the
orchestrator script writes ({"status": "running"|"done"|"error", ...}),
/favicon.ico serves the app icon so the chromeless browser app window is
identifiable in the taskbar/window switcher instead of showing a generic
default. Exists because a file:// page cannot receive events from a
detached process. Prints the chosen ephemeral port on stdout, serves
until killed.

`elapsed_seconds` is stamped per request, not read from the status file:
stages are minutes apart, so a value frozen at the last publish would sit
still through the longest waits -- exactly the stall the page exists to
disprove. Windows' in-process listener computes it the same way.
"""

import http.server
import json
import socketserver
import sys
import time

html_path, status_path = sys.argv[1], sys.argv[2]
started_at = float(sys.argv[3]) if len(sys.argv) > 3 else time.time()
icon_path = sys.argv[4] if len(sys.argv) > 4 else ""
with open(html_path, "rb") as f:
    HTML = f.read()

ICON = b""
if icon_path:
    try:
        with open(icon_path, "rb") as f:
            ICON = f.read()
    except OSError:
        pass


def progress_body():
    try:
        with open(status_path, "rb") as f:
            state = json.loads(f.read())
        if not isinstance(state, dict):
            raise ValueError(state)
    except Exception:
        state = {"status": "running", "message": ""}
    state["elapsed_seconds"] = max(0, int(time.time() - started_at))

    return json.dumps(state).encode("utf-8")


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - base class signature
        pass

    def do_GET(self):
        if self.path.startswith("/progress"):
            body = progress_body()
            ctype = "application/json; charset=utf-8"
        elif self.path == "/":
            body, ctype = HTML, "text/html; charset=utf-8"
        elif self.path == "/favicon.ico" and ICON:
            body, ctype = ICON, "image/x-icon"
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
