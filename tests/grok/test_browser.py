"""Test 1: Browser launch, login detection, session persistence.

Run: cd ~/Projects/hermes-agent && python3 tests/grok/test_browser.py

Tests:
  - Patchright launches Chromium with persistent profile
  - grok.com loads without Cloudflare block
  - Session persists (logged in from saved profile)
  - Cookie banner gets dismissed
  - ProseMirror editor is found and clickable
"""

from patchright.sync_api import sync_playwright
from pathlib import Path
import time
import sys

PASS = 0
FAIL = 0

def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}")

profile_dir = Path.home() / ".hermes" / "grok_profile"

print("=== Test 1: Browser & Session ===\n")

# Launch
print("[1] Launching browser...")
pw = sync_playwright().start()
context = pw.chromium.launch_persistent_context(
    user_data_dir=str(profile_dir),
    headless=False,
    viewport={"width": 1280, "height": 900},
)
page = context.pages[0] if context.pages else context.new_page()
check("Browser launched", page is not None)

# Navigate
print("[2] Loading grok.com...")
page.goto("https://grok.com/", wait_until="networkidle")
time.sleep(3)
check("Page loaded", "grok.com" in page.url)
check("Title is Grok", "Grok" in page.title())

# Dismiss cookie banner
print("[3] Cookie banner...")
try:
    btn = page.query_selector("#onetrust-accept-btn-handler")
    if btn:
        btn.click()
        time.sleep(1)
        print("  Cookie banner dismissed.")
    else:
        print("  No cookie banner (already dismissed).")
except Exception:
    print("  No cookie banner found.")

# Login check
print("[4] Login status...")
content = page.content()
logged_in = "Sign in" not in content
check("Logged in (session persisted)", logged_in)

if not logged_in:
    print("\n  >>> Not logged in. Run the adapter test first to login. <<<")
    context.close()
    pw.stop()
    sys.exit(1)

# Editor check
print("[5] Editor...")
editor = page.query_selector(".ProseMirror")
check("ProseMirror editor found", editor is not None)

# Profile persistence check
print("[6] Profile directory...")
check("Profile dir exists", profile_dir.exists())
check("Profile has data", any(profile_dir.iterdir()))

# Cleanup
context.close()
pw.stop()

print(f"\n=== Results: {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL > 0 else 0)
