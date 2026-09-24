"""Live E2E: vault fill through the REAL browser_exec path (Browser Use CLI + Hermes' packaged Chromium).

Proves problem (1) of the #106480 re-review is fixed: on the default browser backend the login page lives in
a tab browser_exec opened, the supervisor is attached by browser_exec itself, browser_vault_fill focuses the
tab on the bound origin and injects the password over the supervisor's CDP WebSocket, and the secret is absent
from every model-facing result. Also exercises a payment fill (confirm gate, card fields, decline = no write).

Run: HERMES_E2E_BROWSER=1 <venv>/bin/python evals/vault_fill_live_e2e.py
"""
from __future__ import annotations

import http.server
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HOME = Path(tempfile.mkdtemp(prefix="hermes-vault-e2e-"))
os.environ["HERMES_HOME"] = str(HOME)

PAGES = {
    "/login": b"""<!doctype html><title>login</title>
<form><input name=email type=email autocomplete=username><input name=pw type=password autocomplete=current-password>
<input type=submit></form>""",
    "/checkout": b"""<!doctype html><title>checkout</title>
<form><input name=cardnum placeholder="Card number"><input name=exp placeholder="Expiry (MM/YY)">
<input name=cvv placeholder="CVC"><select name=country><option value=DE>Germany<option value=US>United States</select>
<input name=email type=email></form>""",
}
TASK = "vault-e2e"


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = PAGES.get(self.path, b"nope")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: ARG002 — quiet
        pass


def _exec(code: str) -> dict:
    from tools import browser_use_cli as bu

    out = json.loads(bu.browser_exec(code, task_id=TASK, timeout_s=90))
    assert out.get("success"), out
    return out


def main() -> int:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{srv.server_address[1]}"

    from tools.browser_supervisor import SUPERVISOR_REGISTRY
    try:
        # browser_exec opens its own tabs; the login page is deliberately NOT the first one.
        _exec("new_tab('about:blank')")
        _exec(f"new_tab({origin + '/login'!r}); wait_for_load()")
        _exec(f"new_tab({origin + '/checkout'!r}); wait_for_load(); print(page_info()['url'])")

        sup = SUPERVISOR_REGISTRY.get(TASK)
        assert sup is not None, "browser_exec did not attach a supervisor for its task (problem 1 regressed)"
        print("supervisor attached by browser_exec; its page before fill:", sup.evaluate_runtime("location.href")["result"])

        from tools import browser_vault_tool as bvt
        from tools.browser_cdp_tool import _redact_cdp_output
        from agent.vault_store import get_vault_store
        from agent import redact
        import tools.approval_prompt as ap

        store = get_vault_store()
        login = store.add_item("login", "site", {"identifier_type": "email", "identifier": "a@b.c", "password": "pw-E2E-8842"}, origin=origin)
        card = store.add_item("payment", "visa", {"card_number": "4111111111111111", "exp_month": "7", "exp_year": "2029", "cvc": "987"}, origin=origin)

        raw = bvt.browser_vault_fill(login.id, task_id=TASK)
        out = json.loads(raw)
        print("login fill:", out)
        assert out["success"] and out["filled_fields"] == 1, out
        assert "pw-E2E-8842" not in raw
        dom = sup.evaluate_runtime("location.pathname + ' ' + document.querySelector('input[name=pw]').value")
        assert dom["result"] == "/login pw-E2E-8842", dom  # raw supervisor read (not a model surface): the write landed in the login tab
        assert "pw-E2E-8842" not in json.dumps(_redact_cdp_output({"result": {"value": dom["result"]}}))
        print("login: password landed in the /login tab; model-facing read is scrubbed")

        ap.request_elicitation_consent = lambda *a, **k: "decline"
        out = json.loads(bvt.browser_vault_fill(card.id, task_id=TASK))
        assert out["error_type"] == "payment_declined", out
        assert sup.focus_page(origin, accept=bvt._TAB_PROBES["payment"])["ok"]
        r = sup.evaluate_runtime("['cardnum','exp','cvv'].map(n => document.querySelector('[name='+n+']').value).join('|')")
        assert r.get("result") == "||", r
        print("payment: declined confirmation wrote nothing")

        ap.request_elicitation_consent = lambda *a, **k: "accept"
        raw = bvt.browser_vault_fill(card.id, task_id=TASK)
        out = json.loads(raw)
        print("payment fill:", out)
        assert out["success"] and out["filled_fields"] == 3 and out["fields"] == ["cc-csc", "cc-exp", "cc-number"], out
        assert "4111" not in raw and "987" not in raw
        r = sup.evaluate_runtime("['cardnum','exp','cvv','country','email'].map(n => document.querySelector('[name='+n+']').value).join('|')")
        assert r.get("result") == "4111111111111111|07/29|987|DE|", r
        print("payment: card/expiry/cvc filled on the /checkout tab, email untouched, select untouched without a value")

        # save-on-page: the supervisor's default page is the blank first tab; the tool must find the login
        # tab itself (Browser Use daemon tabs are how a real session looks) and bind the item to ITS origin.
        from agent.vault_backends import unlock as vault_unlock
        store.remove_item(login.id)
        sup.evaluate_runtime("document.querySelector('input[name=pw]') && (document.querySelector('input[name=pw]').value = '')")
        assert sup.focus_page("about:blank")["ok"] is False  # about: pages are never candidates
        vault_unlock.set_save_login_prompt_callback(lambda o, site: {"identifier": "new@b.c", "password": "pw-SAVE-5150"})
        vault_unlock.set_unlock_prompt_callback(lambda *a: "")  # an interactive surface installs both; can_prompt_here keys off this one
        raw = bvt.browser_vault_save_login(task_id=TASK)
        out = json.loads(raw)
        vault_unlock.set_save_login_prompt_callback(None); vault_unlock.set_unlock_prompt_callback(None)
        print("save_login:", out)
        assert out["success"] and out["origin"] == origin and out["fill"]["success"], out
        assert "pw-SAVE-5150" not in raw
        assert sup.focus_page(origin, accept=bvt._TAB_PROBES["login"])["ok"]
        dom = sup.evaluate_runtime("location.pathname + ' ' + document.querySelector('input[name=pw]').value")
        assert dom["result"] == "/login pw-SAVE-5150", dom
        print("save_login: found the login tab from a blank default page, bound to its origin, filled")

        redact.clear_vault_redaction_values()
        print("E2E OK")
        return 0
    finally:
        SUPERVISOR_REGISTRY.stop_all()
        try:
            from tools.browser_tool_lifecycle import cleanup_all_browsers
            cleanup_all_browsers()
        except Exception:
            pass
        srv.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
