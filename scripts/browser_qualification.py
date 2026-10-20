"""Deterministic 20-case live qualification for Hermes browser integration."""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

os.environ.setdefault("HERMES_HOME", r"C:\Users\chara\AppData\Local\hermes\profiles\haios-primary-operator")

from tools import browser_tool as b
from tools import browser_tool_lifecycle as lifecycle
from tools.browser_tool_install import _chromium_installed, _find_agent_browser


HTML = """<!doctype html><html><head><title>Hermes Browser Qualification</title></head>
<body><h1>Hermes Browser QA</h1><p id='status'>idle</p>
<button id='go' onclick="document.getElementById('status').textContent='clicked'">Click Me</button>
<form onsubmit="event.preventDefault();document.getElementById('result').textContent=document.getElementById('name').value;console.log('submitted')">
<label>Name <input id='name' aria-label='Name'></label><button id='submit'>Submit</button></form>
<p id='result'></p><div style='height:1800px'></div>
<script>console.warn('qa-warning');window.qaValue='ready';</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if urlparse(self.path).path != "/":
            self.send_response(404); self.end_headers(); return
        body = HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def log_message(self, *_args):
        return


def start_server() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, "http://127.0.0.1:%d/" % server.server_port


def load(value):
    try:
        return json.loads(value)
    except Exception:
        return {"raw": value}


def main() -> int:
    # Start from a clean browser registry so prior sessions cannot contaminate qualification.
    lifecycle.cleanup_all_browsers()
    server, url = start_server()
    results = []

    def case(number, name, passed, evidence=""):
        results.append((number, name, bool(passed), evidence))
        print("CASE %02d %-24s %s %s" %
              (number, name, "PASS" if passed else "FAIL", evidence), flush=True)

    try:
        case(1, "local_backend", b._cloud._is_local_backend())
        case(2, "agent_browser_resolve", bool(_find_agent_browser()))
        case(3, "chromium_installed", _chromium_installed())

        nav = load(b.browser_navigate(url, "qual-main"))
        case(4, "navigate", nav.get("success") is True, str(nav.get("title", "")))
        case(5, "auto_snapshot", "Hermes Browser QA" in json.dumps(nav))
        snap = load(b.browser_snapshot(False, "qual-main"))
        case(6, "snapshot", snap.get("success") is True)

        click = load(b.browser_click("@e2", "qual-main"))
        case(7, "click", click.get("success") is True)
        state = load(b.browser_console(False, "document.getElementById('status').textContent", "qual-main"))
        case(8, "click_effect", "clicked" in json.dumps(state))

        typed = load(b.browser_type("@e4", "Alice", "qual-main"))
        case(9, "type", typed.get("success") is True)
        pressed = load(b.browser_press("Enter", "qual-main"))
        case(10, "press", pressed.get("success") is True)
        evaluated = load(b.browser_console(False, "window.qaValue", "qual-main"))
        case(11, "javascript_eval", "ready" in json.dumps(evaluated))

        scrolled = load(b.browser_scroll("down", "qual-main"))
        case(12, "scroll", scrolled.get("success") is True)
        b.browser_navigate("https://example.com/", "qual-main")
        back = load(b.browser_back("qual-main"))
        case(13, "back_none_guard", back.get("success") is True)

        bad = load(b.browser_scroll("sideways", "qual-main"))
        case(14, "invalid_scroll_guard", "Invalid direction" in bad.get("error", ""))
        secret = load(b.browser_navigate("https://example.com/?token=sk-TESTSECRET1234567890", "qual-main"))
        case(15, "secret_url_guard", secret.get("success") is False)
        metadata = load(b.browser_navigate("http://169.254.169.254/", "qual-main"))
        case(16, "metadata_guard", metadata.get("success") is False)


        # Retire the long-lived session before the concurrency/isolation phase.
        lifecycle.cleanup_browser("qual-main")
        a = load(b.browser_navigate(url, "qual-A"))
        bb = load(b.browser_navigate(url, "qual-B"))
        case(17, "session_A", a.get("success") is True)
        case(18, "session_B", bb.get("success") is True)
        b.browser_type("@e4", "A-only", "qual-A")
        b_snap = load(b.browser_snapshot(False, "qual-B"))
        case(19, "session_isolation", "A-only" not in json.dumps(b_snap))

        lifecycle.cleanup_browser("qual-A")
        lifecycle.cleanup_browser("qual-B")
        lifecycle.cleanup_browser("qual-main")
        case(20, "cleanup", True)
    finally:
        server.shutdown()
        server.server_close()

    failed = [r for r in results if not r[2]]
    print("SUMMARY: %d/20 PASS, %d FAIL" % (20 - len(failed), len(failed)), flush=True)
    for n, name, _, evidence in failed:
        print("FAILED %02d %s :: %s" % (n, name, evidence), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
