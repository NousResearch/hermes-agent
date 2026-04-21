"""Interactive browser launcher for auth-flow screenshots + manual debugging.

Run: cd ~/Projects/hermes-agent && python3 tests/grok/debug_auth_flow.py

Opens Patchright Chromium with the persistent Hermes grok profile, navigates
to grok.com, and blocks until you press Enter. Unlike test_browser.py (which
runs checks + closes), this one stays open as long as you need — useful for:

  - Screenshotting the logged-in grok.com UI (modes dropdown, chat panel, etc.)
  - Logging out via the account menu, then capturing the X OAuth auth flow
    when you re-authenticate
  - Clicking through any state manually while the same profile dir Hermes
    uses is live
  - Reproducing / filming visual issues for PR reviewers

Profile dir: ~/.hermes/grok_profile  (same dir Hermes CLI uses)
"""

from pathlib import Path
import time

from patchright.sync_api import sync_playwright


PROFILE_DIR = Path.home() / ".hermes" / "grok_profile"


def main():
    print("=== Grok browser — interactive launcher ===\n")
    print(f"Profile dir: {PROFILE_DIR}")
    print("Opening Patchright Chromium at grok.com...\n")

    pw = sync_playwright().start()
    try:
        context = pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
            viewport={"width": 1280, "height": 900},
        )
        page = context.pages[0] if context.pages else context.new_page()

        page.goto("https://grok.com/", wait_until="domcontentloaded")
        time.sleep(2)

        # Best-effort cookie dismiss so it doesn't cover screenshots
        try:
            btn = page.query_selector("#onetrust-accept-btn-handler")
            if btn:
                btn.click()
                time.sleep(0.5)
        except Exception:
            pass

        content = page.content()
        logged_in = "Sign in" not in content
        print(f"Logged in: {logged_in}")
        if not logged_in:
            print("  You're on the sign-in page — click X OAuth to trigger the auth flow.")
        else:
            print("  Session active. Logout from the account menu to re-trigger auth.")

        import sys as _sys
        print()
        print("Browser is live. Screenshot whatever you need.")
        if _sys.stdin.isatty():
            print("Press Enter in THIS terminal to close the browser when done.")
            print()
            try:
                input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
        else:
            # Background invocation — no interactive stdin. Keep the browser
            # open until the parent process is killed (SIGTERM / task-stop).
            # Prints a signal every 30s so log tailing confirms liveness.
            print("(non-TTY stdin detected — browser will stay open until this")
            print(" process is killed. Send SIGTERM/SIGINT to close cleanly.)")
            print()
            import signal

            stop = {"flag": False}

            def _on_signal(signum, frame):
                stop["flag"] = True

            signal.signal(signal.SIGTERM, _on_signal)
            signal.signal(signal.SIGINT, _on_signal)

            ticks = 0
            while not stop["flag"]:
                time.sleep(1)
                ticks += 1
                if ticks % 30 == 0:
                    print(f"[alive] browser still open ({ticks}s elapsed)", flush=True)

        try:
            context.close()
        except Exception:
            # Browser context may already be gone (manual window close,
            # logout re-auth flow, SIGTERM during a page navigation).
            # Harmless on shutdown.
            pass
    finally:
        try:
            pw.stop()
        except Exception:
            pass
        print("Browser closed.")


if __name__ == "__main__":
    main()
