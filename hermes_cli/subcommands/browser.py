"""``hermes browser`` subcommand parser."""

from __future__ import annotations

import sys


def build_browser_parser(subparsers) -> None:
    """Attach the ``browser`` subcommand to ``subparsers``."""
    browser_parser = subparsers.add_parser(
        "browser", help="Real-profile browsing helpers (close a browser locking its profile)",
        description="Helpers for real-profile browsing (browser.use_real_profile). "
            "close-profile terminates the browser process tree holding your "
            "default profile so Hermes can copy it — DESTRUCTIVE (unsaved tabs "
            "in that browser are lost). The agent runs this only after you "
            "approve closing the browser.")
    browser_subparsers = browser_parser.add_subparsers(dest="browser_action")
    browser_close = browser_subparsers.add_parser(
        "close-profile",
        help="Close the browser locking your real profile (asks nothing — "
             "run only with the user's explicit OK; loses unsaved tabs)")
    browser_close.add_argument(
        "--browser",
        help="Override the resolved browser (chrome/edge/brave/brave-origin/chromium)")
    browser_subparsers.add_parser(
        "profiles", help="List the browsers and profiles real-profile browsing can use")

    def _print_profiles() -> int:
        from hermes_cli.browser_connect import list_real_profile_candidates

        info = list_real_profile_candidates()
        for row in info["browsers"]:
            tags = [t for t, on in (("system default", row["is_system_default"]),
                                    ("active", row["key"] == info["resolved_browser"]),
                                    ("not installed", not row["installed"]),
                                    ("never launched", row["installed"] and not row["has_profile"]))
                    if on]
            print(f"{row['key']}  ({row['label']}){' — ' + ', '.join(tags) if tags else ''}")
            for prof in row["profiles"]:
                marks = [t for t, on in (
                    ("last used", prof["last_used"]),
                    ("in use", row["key"] == info["resolved_browser"]
                     and prof["directory"] == info["resolved_profile"])) if on]
                print(f"    {prof['directory']:<12} {prof['name']}"
                      f"{'  [' + ', '.join(marks) + ']' if marks else ''}")
        print("\nSet browser.real_profile_browser to a key above and browser.real_profile_pin "
              "to a profile directory to choose which identity the agent browses as.")
        if info["error"]:
            print(f"\n! {info['error']}", file=sys.stderr)
        return 0

    def _dispatch_browser(_args):
        from hermes_cli.browser_connect import (
            UNSUPPORTED_CHANNEL, close_browser_holding_profile, real_profile_data_dir,
            resolve_real_profile_browser)

        action = getattr(_args, "browser_action", None)
        if action == "profiles":
            return _print_profiles()
        if action != "close-profile":
            browser_parser.print_help()
            return 2
        # Same resolver the launch path uses, so this closes the browser Hermes would
        # actually drive — an explicit --browser still wins for recovery.
        browser = getattr(_args, "browser", None)
        if not browser:
            browser, err = resolve_real_profile_browser()
            if err:
                print(f"✗ {err}", file=sys.stderr)
                return 1
        if not browser or browser == UNSUPPORTED_CHANNEL:
            print("✗ No supported Chromium browser resolved for real-profile browsing.",
                  file=sys.stderr)
            return 1
        src = real_profile_data_dir(browser)
        if not src:
            print(f"✗ Could not resolve the {browser} profile directory.", file=sys.stderr)
            return 1
        closed, msg = close_browser_holding_profile(src)
        if closed:
            print(f"✓ {msg}")
            return 0
        print(f"✗ {msg}", file=sys.stderr)
        return 1

    browser_parser.set_defaults(func=_dispatch_browser)
