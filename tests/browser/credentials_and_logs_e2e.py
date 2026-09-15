"""Credentials and logs, driven in a browser and checked against the runtime's own files.

Credential entry is the one path in NOVA that sends a secret, so most of what this asserts
is about where the secret does *not* go: not into any read route, not into the audit log,
not back into the form after a reload. It types a distinctive value through the real input
and then greps the API surface and the audit log for it from inside the browser's own
session, which is the only way to be sure no route leaks it back to a caller who has one.

It also proves the allowlist, by trying LD_PRELOAD the way an attacker with an admin
session would: `.env` is loaded into the environment of the process that runs the agent, so
a write path accepting any name would be code execution rather than configuration.

Run it:

    python -m nova apply <bundle>
    python -m nova serve <bundle> --host 127.0.0.1 --port 19300 &
    python tests/browser/credentials_and_logs_e2e.py <runtime-home>
"""

import os
import json, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = os.environ.get("NOVA_E2E_BASE", "http://127.0.0.1:19300")
HOME = Path(sys.argv[1])
ENV = HOME / "profiles" / "operations" / ".env"
SECRET = "123:BROWSER-TYPED-SECRET"
ok, fail = [], []

def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {detail}" if detail and not cond else ""))

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=os.environ.get("NOVA_E2E_CHROMIUM", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"),
                          args=["--no-sandbox"])
    page = b.new_page(viewport={"width": 1500, "height": 1150})
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    def goto(frag):
        page.goto("about:blank")
        page.goto(f"{BASE}/#{frag}", wait_until="networkidle")
        page.wait_for_timeout(400)

    # ---- CREDENTIALS -------------------------------------------------------
    print("\n# Credentials")
    goto("/agents/operations")
    page.get_by_role("tab", name="Channels").click()
    page.wait_for_selector("text=Credentials", timeout=10000)
    check("only declared credentials are offered",
          page.locator("label[for='cred-TELEGRAM_BOT_TOKEN']").count() == 1
          and page.locator("label[for='cred-SLACK_BOT_TOKEN']").count() == 0)
    check("an unset credential says so", "Not set" in page.content())

    field = page.locator("#cred-TELEGRAM_BOT_TOKEN")
    check("the input is a password field", field.get_attribute("type") == "password")
    save = page.get_by_role("button", name="Save credentials")
    check("save disabled with nothing typed", save.is_disabled())

    field.fill(SECRET)
    save.click()
    page.wait_for_selector("text=Stored: TELEGRAM_BOT_TOKEN", timeout=20000)
    check("save confirms what was stored", True)

    check("the value reached the runtime's .env", SECRET in ENV.read_text())
    check("the file is owner-only", oct(ENV.stat().st_mode & 0o777) == "0o600")
    check("the field was cleared after saving", page.locator("#cred-TELEGRAM_BOT_TOKEN").input_value() == "")
    page.wait_for_selector("text=Set", timeout=10000)

    goto("/agents/operations")
    page.get_by_role("tab", name="Channels").click()
    page.wait_for_selector("#cred-TELEGRAM_BOT_TOKEN", timeout=10000)
    check("the value is never shown again after a reload",
          SECRET not in page.content()
          and page.locator("#cred-TELEGRAM_BOT_TOKEN").input_value() == "")

    # The whole API surface, from the browser's own session.
    leaked = page.evaluate("""async () => {
      const routes = ['/agents','/channels','/policy','/health',
                      '/agents/operations/config','/agents/operations/credentials',
                      '/agents/operations/activity'];
      const out = [];
      for (const r of routes) {
        const t = await (await fetch('/platform/v1' + r)).text();
        if (t.includes('BROWSER-TYPED-SECRET')) out.push(r);
      }
      return out;
    }""")
    check("no read route returns the value", leaked == [], str(leaked))
    check("no audit record holds the value",
          "BROWSER-TYPED-SECRET" not in (HOME/"nova"/"audit.jsonl").read_text())

    # Injection, through the same session a browser has.
    hostile = json.loads(page.evaluate("""
      fetch('/platform/v1/agents/operations/credentials', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({values:{LD_PRELOAD:'/tmp/evil.so'}})}).then(r=>r.text())"""))
    check("an undeclared variable is refused",
          hostile.get("error",{}).get("status") == 400 and "LD_PRELOAD" in hostile["error"]["message"])
    check("the refused write did not touch the file", "LD_PRELOAD" not in ENV.read_text())

    # Remove it again.
    page.get_by_role("button", name="Remove").first.click()
    page.get_by_role("button", name="Save credentials").click()
    page.wait_for_selector("text=Stored: TELEGRAM_BOT_TOKEN", timeout=20000)
    check("removing clears it from the runtime", SECRET not in ENV.read_text())

    # ---- LOGS --------------------------------------------------------------
    print("\n# Runtime logs and activity")
    goto("/agents/operations")
    page.get_by_role("tab", name="Activity").click()
    page.wait_for_selector("text=Recent work", timeout=10000)
    check("empty sections say the runtime recorded nothing",
          "The runtime holds no work for this agent" in page.content()
          and "No execution has been recorded" in page.content())

    page.wait_for_selector("text=agent.log", timeout=10000)
    page.wait_for_selector("text=agent turn 799", timeout=10000)
    check("the tail of the real log is shown", True)
    check("a long file is labelled as truncated", "Showing the end of a longer file" in page.content())

    page.get_by_role("button", name="errors.log").click()
    page.wait_for_selector("text=provider returned 429", timeout=10000)
    check("switching stream reads the other file", True)

    page.get_by_role("button", name="gateway.log").first
    check("a log the runtime never wrote is offered but disabled",
          page.get_by_role("button", name="gateway.log").first.is_disabled())

    page.screenshot(path=str(HOME.parent/"p3.png"))
    real = [e for e in errs if "favicon" not in e.lower() and "failed to load resource" not in e.lower()]
    check("no console errors", not real, str(real[:2]))
    b.close()

print(f"\n=== {len(ok)} passed, {len(fail)} failed ===")
if fail:
    print("failed:", fail); sys.exit(1)
