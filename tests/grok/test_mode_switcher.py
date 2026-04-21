"""Integration test: _set_mode() actually clicks the grok.com dropdown.

Real-browser test that switches grok.com between modes via the new
_BrowserThread._set_mode() method and verifies the trigger button text
updates after each switch.

Run:
  cd ~/Projects/hermes-agent && python3 tests/grok/test_mode_switcher.py

Prerequisites:
  - Logged into grok.com (profile in ~/.hermes/grok_profile/)
  - patchright + chromium installed

What it tests:
  1. Each of the 5 modes can be selected via UI clicking
  2. After click, the trigger button shows the new label
  3. Calling _set_mode with the same mode twice is idempotent (no-op)
  4. Unknown mode is silently ignored (no crash)

Skips standalone PASS/FAIL counter — uses inline assertions per mode.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.patchright_client import _MODE_LABELS, _get_profile_dir, _GROK_URL


PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}")
        if detail:
            print(f"        {detail}")


def _read_current_mode_label(page) -> str:
    """Inspect grok.com to determine which mode label the dropdown trigger
    is currently showing. Returns empty string if not found."""
    try:
        triggers = page.query_selector_all('button[aria-haspopup="menu"]')
        for btn in triggers:
            try:
                txt = (btn.inner_text() or "").strip()
                if txt in _MODE_LABELS.values():
                    return txt
            except Exception:
                continue
    except Exception:
        pass
    return ""


def main():
    from patchright.sync_api import sync_playwright

    profile_dir = _get_profile_dir()
    if not profile_dir.exists():
        print(f"[ERROR] Profile dir not found at {profile_dir}")
        print("        Run a regular hermes session first to log in.")
        sys.exit(1)

    print("=== Test: _set_mode() integration ===\n")
    print(f"[1] Launching Patchright with profile {profile_dir}")

    pw = sync_playwright().start()
    ctx = pw.chromium.launch_persistent_context(
        user_data_dir=str(profile_dir),
        headless=False,
        viewport={"width": 1280, "height": 900},
        args=["--disable-blink-features=AutomationControlled"],
    )
    page = ctx.pages[0] if ctx.pages else ctx.new_page()

    print(f"[2] Navigating to {_GROK_URL}")
    page.goto(_GROK_URL, wait_until="networkidle")
    time.sleep(3)

    # Dismiss cookie banner if present
    try:
        btn = page.query_selector("#onetrust-accept-btn-handler")
        if btn:
            btn.click()
            time.sleep(1)
    except Exception:
        pass

    # Verify logged in
    if "Sign in" in page.content() and "Ask anything" not in page.content():
        print("[ERROR] Not logged in — please log in at grok.com first")
        ctx.close()
        pw.stop()
        sys.exit(1)

    # Build a thin _BrowserThread instance bound to this real page
    from agent.patchright_client import _BrowserThread
    bt = _BrowserThread.__new__(_BrowserThread)
    bt._page = page

    initial_label = _read_current_mode_label(page)
    print(f"[3] Initial mode label on grok.com: '{initial_label}'")

    # Test each mode in sequence
    print("\n[4] Switching through all 5 modes via _set_mode()...")
    for mode_key, expected_label in _MODE_LABELS.items():
        print(f"\n  -> _set_mode('{mode_key}') (expecting label '{expected_label}')")
        bt._set_mode(mode_key)
        time.sleep(1.5)  # Let UI settle after click
        actual = _read_current_mode_label(page)
        check(
            f"After _set_mode('{mode_key}'), trigger shows '{expected_label}'",
            actual == expected_label,
            detail=f"got '{actual}'",
        )

    # Idempotency: calling _set_mode for the current mode should be no-op
    print("\n[5] Idempotency — calling _set_mode for current mode again...")
    current_mode_key = next(
        (k for k, v in _MODE_LABELS.items() if v == _read_current_mode_label(page)),
        None,
    )
    if current_mode_key:
        before = _read_current_mode_label(page)
        bt._set_mode(current_mode_key)
        time.sleep(0.5)
        after = _read_current_mode_label(page)
        check(
            f"Idempotent: _set_mode('{current_mode_key}') unchanged label '{after}'",
            before == after,
        )

    # Unknown mode handling
    print("\n[6] Unknown mode should silently no-op (no crash)...")
    before = _read_current_mode_label(page)
    try:
        bt._set_mode("grok-99-future")
        bt._set_mode("grok-4")  # legacy / not in _MODE_LABELS
        bt._set_mode("")
        bt._set_mode(None)
        time.sleep(0.5)
        after = _read_current_mode_label(page)
        check("Unknown / empty / None modes don't crash", True)
        check("Unknown modes don't change current label", before == after)
    except Exception as e:
        check("Unknown modes don't crash", False, detail=str(e))

    # Restore initial mode for clean exit
    if initial_label:
        initial_key = next(
            (k for k, v in _MODE_LABELS.items() if v == initial_label), None
        )
        if initial_key:
            print(f"\n[7] Restoring initial mode '{initial_label}'...")
            bt._set_mode(initial_key)
            time.sleep(1.0)

    print(f"\n=== Results: {PASS} passed, {FAIL} failed ===")
    print("Closing browser in 5s (Ctrl+C to skip)...")
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        pass

    ctx.close()
    pw.stop()
    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
