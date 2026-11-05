#!/usr/bin/env python3
"""
Shared OAuth callback catcher for Hermes MCP connectors.

Serves on localhost:8743, exposed publicly as https://cb.mstylehome.ru via a
named Cloudflare Tunnel (mstyle-oauth-callback). Each connector gets its own
path, e.g. /avito for the Avito connector -- add more paths here as more
connectors need a real, owned-domain OAuth redirect_uri instead of one
subdomain per service.

Captures the `code` (and `state`) query params from an incoming OAuth
redirect and writes them to a per-path JSON file under
~/.hermes/mcp-tokens/oauth-callbacks/<path>.json, so the connector's own
login-finish tool can pick it up without the user manually copying anything.
Also shows a plain confirmation page in the browser.
"""
import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PORT = 8743
OUT_DIR = Path.home() / ".hermes" / "mcp-tokens" / "oauth-callbacks"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        path_key = parsed.path.strip("/").replace("/", "_") or "root"
        params = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}

        out_file = OUT_DIR / f"{path_key}.json"
        out_file.write_text(json.dumps({"path": parsed.path, "params": params}, indent=2))
        try:
            os.chmod(out_file, 0o600)
        except OSError:
            pass

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        code = params.get("code", "")
        body = f"""<!doctype html><html><head><meta charset="utf-8">
        <title>Авторизация получена</title></head>
        <body style="font-family: sans-serif; max-width: 480px; margin: 60px auto; text-align: center;">
        <h2>Готово</h2>
        <p>Код получен, можно закрыть эту страницу и вернуться в чат.</p>
        <p style="color:#888; font-size: 0.9em;">{'code: ' + code[:8] + '...' if code else ''}</p>
        </body></html>"""
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # keep stdout clean; nothing sensitive printed either way


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"oauth callback catcher listening on 127.0.0.1:{PORT}")
    server.serve_forever()
