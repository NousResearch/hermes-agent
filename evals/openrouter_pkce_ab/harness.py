#!/usr/bin/env python3
"""Local A/B harness for `hermes auth add openrouter --type oauth` against a FAKE OpenRouter.

Runs the REAL entry point (``hermes_cli.auth_commands.auth_add_command``) with a temp HERMES_HOME.
Only the network authority is replaced: ``webbrowser.open`` is swapped for a scripted "browser"
that follows the auth URL's ``callback_url`` the way openrouter.ai would (redirecting the loopback
listener with ``?code=``), and ``OPENROUTER_AUTH_KEYS_URL`` points at a local fake code-exchange
server that enforces OpenRouter's documented contract (S256 verifier check, single-use code, 403).

Scenarios (each prints PASS/FAIL, exit code = number of failures):
  legit          full flow → pool entry written, auth_type=api_key, source=manual:openrouter_pkce
  wrong_state    browser redirects to a different callback path (forged nonce) → 404, no exchange
  replayed_code  code already consumed at the fake server → 403 → AuthError, nothing persisted
  malformed      exchange returns JSON without "key" → AuthError, nothing persisted
  api_key_path   `hermes auth add openrouter --api-key` still works with no --type (regression)

Usage: HERMES_PYTHON=<venv python> python3 evals/openrouter_pkce_ab/harness.py [--json OUT]
Run against origin/main to see the BEFORE state (every oauth scenario fails with SystemExit
"not implemented"), then against the salvage branch for AFTER.
"""
from __future__ import annotations

import argparse
import hashlib
import base64
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


class FakeOpenRouter(ThreadingHTTPServer):
    """Fake ``POST /api/v1/auth/keys`` implementing the published contract."""

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.issued: dict[str, str] = {}   # code -> code_challenge
        self.consumed: set[str] = set()
        self.mode = "ok"                   # ok | malformed
        self.exchanges: list[dict] = []

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}/api/v1/auth/keys"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: A003
        return

    def do_POST(self):  # noqa: N802
        srv: FakeOpenRouter = self.server  # type: ignore[assignment]
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0")) or 0) or b"{}")
        srv.exchanges.append(body)
        code, verifier, method = body.get("code"), body.get("code_verifier", ""), body.get("code_challenge_method")
        if method not in ("S256", "plain", None):
            return self._json(400, {"error": {"code": 400, "message": "Invalid code_challenge_method"}})
        if code not in srv.issued or code in srv.consumed:
            return self._json(403, {"error": {"code": 403, "message": "Invalid code or code_verifier"}})
        expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        if expected != srv.issued[code]:
            return self._json(403, {"error": {"code": 403, "message": "Invalid code or code_verifier"}})
        srv.consumed.add(code)
        if srv.mode == "malformed":
            return self._json(200, {"user_id": "user_x"})
        return self._json(200, {"key": "sk-or-v1-" + hashlib.sha256(code.encode()).hexdigest(), "user_id": "user_x"})

    def _json(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def scripted_browser(fake: FakeOpenRouter, *, tamper_path=False, pre_consume=False):
    """Return a ``webbrowser.open`` stand-in that behaves like openrouter.ai/auth after user consent."""
    def _open(url):
        q = parse_qs(urlparse(url).query)
        callback = q["callback_url"][0]
        code = "auth_code_" + hashlib.sha1(callback.encode()).hexdigest()[:12]
        fake.issued[code] = q["code_challenge"][0]
        if pre_consume:
            fake.consumed.add(code)
        if tamper_path:  # attacker guesses the port but not the nonce path
            p = urlparse(callback)
            callback = f"{p.scheme}://{p.netloc}/callback/forged-nonce"
        def _redirect():
            try:
                with urllib.request.urlopen(f"{callback}?code={code}", timeout=5) as r:
                    _open.last_status = r.status
            except urllib.error.HTTPError as e:
                _open.last_status = e.code
            except Exception as e:  # listener already closed
                _open.last_status = repr(e)
        threading.Thread(target=_redirect, daemon=True).start()
        return True
    _open.last_status = None
    return _open


def run(scenario: str, fake: FakeOpenRouter, home: str) -> dict:
    os.environ["HERMES_HOME"] = home
    for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "SSH_CLIENT", "SSH_TTY"):
        os.environ.pop(k, None)
    for m in [m for m in sys.modules if m.startswith(("hermes_cli", "agent", "hermes_constants"))]:
        del sys.modules[m]
    fake.mode = "malformed" if scenario == "malformed" else "ok"
    browser = scripted_browser(fake, tamper_path=(scenario == "wrong_state"), pre_consume=(scenario == "replayed_code"))
    outcome = {"scenario": scenario, "exchanges_before": len(fake.exchanges)}
    try:
        from hermes_cli.auth_commands import auth_add_command
    except Exception as e:  # e.g. the original PR branch's auth.py fails at import time
        outcome.update(result=f"IMPORT FAILURE {type(e).__name__}: {e}", browser_redirect_status=None,
                       exchange_calls=0, pool_entries=[])
        outcome.pop("exchanges_before")
        return outcome
    try:  # BEFORE (origin/main) has no auth_openrouter sibling; the flow itself must then fail.
        import hermes_cli.auth_openrouter as orm
        import hermes_cli.auth_device_flow as dfl
        orm.OPENROUTER_AUTH_KEYS_URL = fake.url
        dfl._can_open_graphical_browser = lambda: True
        orm.webbrowser.open = browser
    except ImportError:
        pass
    args = SimpleNamespace(provider="openrouter", auth_type="oauth", label="pkce-test", api_key=None,
                           no_browser=False, timeout=6)
    if scenario == "api_key_path":
        args = SimpleNamespace(provider="openrouter", auth_type=None, label="plain", api_key="sk-or-v1-manual")
    try:
        auth_add_command(args)
        outcome["result"] = "ok"
    except SystemExit as e:
        outcome["result"] = f"SystemExit: {e}"
    except Exception as e:  # AuthError etc.
        outcome["result"] = f"{type(e).__name__}({getattr(e, 'code', '')}): {e}"
    outcome["browser_redirect_status"] = browser.last_status
    outcome["exchange_calls"] = len(fake.exchanges) - outcome.pop("exchanges_before")
    auth_json = os.path.join(home, "auth.json")
    entries = []
    if os.path.exists(auth_json):
        entries = json.load(open(auth_json, encoding="utf-8")).get("credential_pool", {}).get("openrouter", [])
    outcome["pool_entries"] = [{k: e.get(k) for k in ("auth_type", "source", "label", "base_url")}
                               | {"key_prefix": str(e.get("access_token", ""))[:9]} for e in entries]
    return outcome


EXPECT = {
    "legit": lambda o: o["result"] == "ok" and o["exchange_calls"] == 1 and any(
        e["auth_type"] == "api_key" and e["source"] == "manual:openrouter_pkce" and e["key_prefix"] == "sk-or-v1-"
        for e in o["pool_entries"]),
    "wrong_state": lambda o: o["browser_redirect_status"] == 404 and o["exchange_calls"] == 0
        and "openrouter_callback_timeout" in o["result"] and not o["pool_entries"],
    "replayed_code": lambda o: "openrouter_token_exchange_denied" in o["result"] and not o["pool_entries"],
    "malformed": lambda o: "openrouter_token_exchange_invalid" in o["result"] and not o["pool_entries"],
    "api_key_path": lambda o: o["result"] == "ok" and o["exchange_calls"] == 0 and any(
        e["auth_type"] == "api_key" and e["source"] == "manual" for e in o["pool_entries"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    fake = FakeOpenRouter()
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    results, fails = [], 0
    for scenario in a.only or EXPECT:
        home = tempfile.mkdtemp(prefix=f"or-pkce-{scenario}-")
        o = run(scenario, fake, home)
        o["pass"] = bool(EXPECT[scenario](o))
        fails += not o["pass"]
        print(("PASS" if o["pass"] else "FAIL"), json.dumps(o))
        results.append(o)
    fake.shutdown()
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
    sys.exit(fails)


if __name__ == "__main__":
    main()
