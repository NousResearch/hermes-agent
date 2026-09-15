"""MCP grants, plugin states and S3 corpora, driven in a real browser.

What this exists to catch is the class of bug a unit test cannot: a control that reports a
change the backend never made, or a screen that claims a capability is live when the
runtime cannot call it.

So every assertion here is paired with the runtime's own files. A grant is not "passed"
because the button flipped; it is passed because ``<profile>/config.yaml`` has the server
in it afterwards, and because the bundle has it too — the second is what makes it survive
the next apply.

Run it:

    python -m nova apply <bundle>
    python -m nova serve <bundle> --host 127.0.0.1 --port 19300 &
    python tests/browser/extensions_e2e.py <bundle> <runtime-home>
"""

import os
import sys
from pathlib import Path

import yaml
from playwright.sync_api import sync_playwright

BASE = os.environ.get("NOVA_E2E_BASE", "http://127.0.0.1:19300")
BUNDLE = Path(sys.argv[1])
HOME = Path(sys.argv[2])
AGENT = "operations"
ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {detail}" if detail and not cond else ""))


def agent_yaml():
    return yaml.safe_load((BUNDLE / "agents" / f"{AGENT}.yaml").read_text()) or {}


def profile_config():
    path = HOME / "profiles" / AGENT / "config.yaml"
    return (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}


with sync_playwright() as p:
    b = p.chromium.launch(
        executable_path=os.environ.get(
            "NOVA_E2E_CHROMIUM", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
        ),
        args=["--no-sandbox"],
    )
    page = b.new_page(viewport={"width": 1500, "height": 1150})
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)

    def goto(frag):
        page.goto("about:blank")
        page.goto(f"{BASE}/#{frag}", wait_until="networkidle")
        page.wait_for_timeout(400)

    # ---- MCP ---------------------------------------------------------------
    print("\n# MCP servers")
    goto(f"/agents/{AGENT}")
    page.get_by_role("tab", name="Extensions").click()
    page.wait_for_selector("text=MCP servers", timeout=10000)

    check("the catalogue is the runtime's, not a hand-written list",
          "in this runtime's catalogue" in page.content())
    check("nothing is granted to begin with",
          "granted no MCP servers" in page.content())
    check("the toolset a grant creates is stated",
          "mcp-" in page.content())

    # The catalogue is 65 long; the screen hides it behind a search rather than
    # rendering sixty checkboxes.
    page.get_by_label("Search MCP servers").fill("linear")
    page.wait_for_timeout(500)
    row = page.locator("li", has_text="linear").first
    check("searching finds a catalogue server", row.count() == 1)
    check("the destination of the data is shown before granting",
          "mcp.linear.app" in page.content())

    row.get_by_role("button", name="Grant").click()
    page.wait_for_selector("text=Saved and applied", timeout=20000)
    check("granting reports saved AND applied", True)

    check("the grant landed in the BUNDLE (so it survives the next apply)",
          "linear" in ((agent_yaml().get("extensions") or {}).get("mcp") or []))
    check("the grant was compiled into the runtime profile",
          "linear" in (profile_config().get("mcp_servers") or {}))
    check("the compiled entry is marked enabled",
          (profile_config().get("mcp_servers") or {}).get("linear", {}).get("enabled") is True)

    page.reload(wait_until="networkidle")
    page.get_by_role("tab", name="Extensions").click()
    page.wait_for_selector("text=MCP servers", timeout=10000)
    check("an OAuth grant is NOT reported as ready to use",
          "Needs authorization" in page.content())
    check("it says exactly what would make it work",
          "hermes mcp login linear" in page.content())

    # ---- plugins -----------------------------------------------------------
    print("\n# Plugins")
    check("channels are not offered a second time here",
          "platforms" not in page.content().lower() or "Channels tab" in page.content())

    block = page.locator("li", has_text="google_meet").first
    check("an opt-in plugin is listed", block.count() == 1)
    block.get_by_role("button", name="Enabled").click()
    page.wait_for_selector("text=Saved and applied", timeout=20000)
    enabled = (profile_config().get("plugins") or {}).get("enabled") or []
    check("enabling reaches the profile", "google_meet" in enabled)
    check("NOVA's own policy plugin was not dropped", "nova-policy" in enabled)

    page.reload(wait_until="networkidle")
    page.get_by_role("tab", name="Extensions").click()
    page.wait_for_selector("text=Plugins", timeout=10000)
    tavily = page.locator("li", has_text="web/tavily").first
    check("a bundled backend is shown as already loading",
          "Loads by default" in tavily.inner_text())
    tavily.get_by_role("button", name="Disabled").click()
    page.wait_for_selector("text=Saved and applied", timeout=20000)
    check("disabling a bundled backend reaches the profile",
          "web/tavily" in ((profile_config().get("plugins") or {}).get("disabled") or []))

    # ---- revoking ----------------------------------------------------------
    print("\n# Revoking")
    page.reload(wait_until="networkidle")
    page.get_by_role("tab", name="Extensions").click()
    page.wait_for_selector("text=MCP servers", timeout=10000)
    page.locator("li", has_text="linear").first.get_by_role("button", name="Revoke").click()
    page.wait_for_selector("text=Saved and applied", timeout=20000)
    check("revoking clears the bundle",
          "linear" not in ((agent_yaml().get("extensions") or {}).get("mcp") or []))
    check("revoking clears the profile",
          "linear" not in (profile_config().get("mcp_servers") or {}))

    # ---- no leakage --------------------------------------------------------
    print("\n# Nothing sensitive is exposed")
    body = page.evaluate(
        "async () => (await fetch('/platform/v1/extensions', "
        "{headers: {'accept': 'application/json'}})).text()"
    )
    check("the catalogue carries no credential VALUES",
          '"value"' not in body and "secret_value" not in body)
    check("the catalogue does carry credential NAMES", '"credentials"' in body)

    print("\n# Page errors")
    real = [e for e in errs if "Failed to load resource" not in e]
    check("no uncaught errors in the browser", not real, "; ".join(real[:3]))

    b.close()

print(f"\n{len(ok)} passed, {len(fail)} failed")
if fail:
    for name in fail:
        print("  FAILED: " + name)
    sys.exit(1)
