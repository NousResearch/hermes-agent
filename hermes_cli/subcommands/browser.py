"""``hermes browser`` subcommand parser."""

from __future__ import annotations

import sys


def build_browser_parser(subparsers) -> None:
    """Attach the ``browser`` subcommand to ``subparsers``."""
    browser_parser = subparsers.add_parser(
        "browser", help="Pick which browser Hermes uses; close a browser locking its profile",
        description="Helpers for local Chromium browsing. ``select`` picks which installed "
            "browser real-profile browsing uses (saved as browser.preferred_browser; empty "
            "follows your OS default). ``close-profile`` terminates the browser process tree "
            "holding your default profile so Hermes can copy it — DESTRUCTIVE (unsaved tabs "
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
    browser_select = browser_subparsers.add_parser(
        "select",
        help="Choose which installed browser Hermes uses (writes browser.preferred_browser)")
    browser_select.add_argument(
        "--browser",
        help="Set directly without the interactive picker "
             "(chrome/edge/brave/brave-origin/chromium)")

    def _dispatch_browser(_args):
        from hermes_cli.browser_connect import (
            UNSUPPORTED_CHANNEL, close_browser_holding_profile,
            installed_chromium_browsers, real_profile_data_dir, resolve_real_profile_browser,
            _BROWSER_BY_KEY,
        )

        action = getattr(_args, "browser_action", None)
        if action == "select":
            return _select_browser(_args, installed_chromium_browsers, _BROWSER_BY_KEY)
        if action != "close-profile":
            browser_parser.print_help()
            return 2
        override = getattr(_args, "browser", None)
        if override:
            browser, error = override, None
        else:
            browser, error = resolve_real_profile_browser()
        if error:
            print(f"✗ {error}", file=sys.stderr)
            return 1
        if not browser or browser == UNSUPPORTED_CHANNEL:
            print("✗ No supported Chromium browser resolved (set one with "
                  "`hermes browser select`).", file=sys.stderr)
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


def _select_browser(_args, installed_chromium_browsers, _browser_by_key) -> int:
    """Implement ``hermes browser select``; helpers injected for tests."""
    from hermes_cli.config import set_config_value

    override = (getattr(_args, "browser", None) or "").strip().lower()
    if override:
        if override not in _browser_by_key:
            valid = ", ".join(b.key for b in _browser_by_key.values())
            print(f"✗ Unknown browser '{override}' (supported: {valid}).", file=sys.stderr)
            return 1
        set_config_value("browser.preferred_browser", override)
        print(f"✓ Hermes will use {override} (browser.preferred_browser).")
        return 0
    installed = installed_chromium_browsers()
    if not installed:
        print("✗ No supported Chromium browser found on this machine "
              "(Chrome, Edge, Brave, Brave Origin, Chromium).", file=sys.stderr)
        return 1
    if len(installed) == 1:
        key = installed[0][0]
        set_config_value("browser.preferred_browser", key)
        print(f"✓ Only {key} is installed — Hermes will use it (browser.preferred_browser).")
        return 0
    if not sys.stdin.isatty():
        print("Multiple browsers installed; pick one:", file=sys.stderr)
        for key, exe in installed:
            print(f"  {key}  ({exe})", file=sys.stderr)
        print("Re-run with `--browser <key>` (or `hermes browser select` in a terminal).",
              file=sys.stderr)
        return 2
    from hermes_cli.curses_ui import curses_radiolist

    labels: list = [f"{key}  ({exe})" for key, exe in installed]
    current = _current_preference()
    default = next((i for i, (key, _) in enumerate(installed) if key == current), 0)
    chosen = curses_radiolist("Which browser should Hermes use?", labels,
                              selected=default, cancel_returns=default)
    key = installed[chosen][0]
    set_config_value("browser.preferred_browser", key)
    print(f"✓ Hermes will use {key} (browser.preferred_browser).")
    return 0


def _current_preference() -> str:
    """Current ``browser.preferred_browser`` value ('' when unset/unreadable)."""
    try:
        from hermes_cli.config import read_raw_config

        browser_cfg = read_raw_config().get("browser", {})
        value = browser_cfg.get("preferred_browser") if isinstance(browser_cfg, dict) else None
        return value.strip().lower() if isinstance(value, str) else ""
    except Exception:
        return ""
