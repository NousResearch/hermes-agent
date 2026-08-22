"""``hermes browser`` subcommand — attach browser sessions to desktop apps.

``hermes browser attach`` scans running processes for Electron apps, probes
for exposed CDP endpoints, and registers one under a named browser_exec
session so the agent can drive the app with exact DOM control (no focus
steal). For a detected app that does not expose CDP, it offers the
relaunch-with-``--remote-debugging-port`` itself — the confirmation prompt
is the user's consent moment, which is why this lives behind a user-invoked
CLI command rather than an agent tool.

Related: the in-session ``/browser connect`` flow (live Chrome for the
built-in browser tools) — this command is its desktop-app sibling and
feeds ``browser_exec`` named sessions instead of the global override.
"""

from __future__ import annotations

import argparse


def cmd_browser(args: argparse.Namespace) -> int:
    action = getattr(args, "browser_action", None)
    if action == "attach":
        return _cmd_attach(args)
    if action == "list":
        return _cmd_list()
    if action == "detach":
        return _cmd_detach(args)
    print("Usage: hermes browser attach|list|detach  (see --help)")
    return 2


def _cmd_list() -> int:
    from hermes_cli.browser_attach import load_registry

    sessions = load_registry()
    if not sessions:
        print("No attached browser sessions. Run `hermes browser attach` to add one.")
        return 0
    print("Attached browser sessions (browser_exec session → app):")
    for name, entry in sorted(sessions.items()):
        print(f"  {name:20s} {entry.get('app', '?'):24s} {entry.get('cdp_url', '')}")
    return 0


def _cmd_detach(args: argparse.Namespace) -> int:
    from hermes_cli.browser_attach import remove_session_endpoint

    name = getattr(args, "session", None) or ""
    if not name:
        print("Usage: hermes browser detach <session-name>")
        return 2
    if remove_session_endpoint(name):
        print(f"Detached session '{name}'.")
        return 0
    print(f"No attached session named '{name}'. `hermes browser list` shows them.")
    return 1


def _cmd_attach(args: argparse.Namespace) -> int:
    from hermes_cli.browser_attach import (
        relaunch_with_debug_port,
        save_session_endpoint,
        scan_electron_apps,
        session_slug,
    )

    print("Scanning for running Electron apps…")
    apps = scan_electron_apps()
    if not apps:
        print(
            "No running Electron apps found.\n"
            "Start the app you want to drive, then re-run `hermes browser attach`.\n"
            "(For a regular web browser, use `/browser connect` inside a session.)"
        )
        return 1

    requested = (getattr(args, "app", None) or "").strip().lower()
    if requested:
        apps = [a for a in apps if requested in a["name"].lower()]
        if not apps:
            print(f"No running Electron app matches '{requested}'.")
            return 1

    target = apps[0]
    if len(apps) > 1:
        # Always show the picker on ambiguity — a filter can legitimately
        # match several apps ('code' → 'Code' and 'Code - Insiders'), and
        # silently taking the first would attach to the wrong one.
        print("Matching Electron apps:" if requested else "Running Electron apps:")
        for i, app in enumerate(apps, 1):
            state = f"CDP live on {app['cdp_url']}" if app["cdp_url"] else (
                f"debug port {app['debug_port']} (not answering)" if app["debug_port"]
                else "no debug port"
            )
            print(f"  {i}. {app['name']:24s} pid {app['pid']:<8d} {state}")
        try:
            choice = input(f"Attach to which app? [1-{len(apps)}] ").strip()
            target = apps[int(choice) - 1]
        except (ValueError, IndexError, EOFError, KeyboardInterrupt):
            print("Cancelled.")
            return 1

    session = session_slug(getattr(args, "session", None) or target["name"])

    cdp_url = target["cdp_url"]
    if not cdp_url:
        if getattr(args, "no_relaunch", False):
            print(
                f"{target['name']} is running without a CDP debug port.\n"
                f"Relaunch it manually with --remote-debugging-port=<port> and retry."
            )
            return 1
        print(
            f"{target['name']} is running but does not expose a CDP debug port.\n"
            f"Attaching requires QUITTING and RELAUNCHING it with "
            f"--remote-debugging-port.\n"
            f"Note: this exposes everything the app can access (messages, files,\n"
            f"vault contents) to local debugger connections while it runs."
        )
        try:
            answer = input(f"Quit and relaunch {target['name']} now? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer not in ("y", "yes"):
            print("Cancelled — app left untouched.")
            return 1
        print(f"Relaunching {target['name']} with a debug port…")
        cdp_url = relaunch_with_debug_port(target)
        if not cdp_url:
            print(
                f"{target['name']} did not come back up with a reachable CDP "
                "endpoint. Check that it fully quit (background/tray instances "
                "block the single-instance lock) and retry."
            )
            return 1

    save_session_endpoint(session, cdp_url, target["name"])
    print(
        f"✓ Attached: browser_exec session '{session}' → {target['name']} ({cdp_url})\n"
        f"  The agent drives it with: browser_exec(code=…, session='{session}')\n"
        f"  Detach with: hermes browser detach {session}"
    )
    return 0


def build_browser_parser(subparsers) -> None:
    """Attach the ``browser`` subcommand to ``subparsers``."""
    browser_parser = subparsers.add_parser(
        "browser",
        help="Attach browser sessions to desktop Electron apps over CDP",
        description=(
            "Attach a named browser_exec session to a running desktop\n"
            "Electron/Chromium app (Obsidian, Slack, VS Code, ...) over the\n"
            "Chrome DevTools Protocol, for exact DOM-level control without\n"
            "focus steal. For connecting a regular web browser to the\n"
            "built-in browser tools, use `/browser connect` inside a session."
        ),
    )
    browser_sub = browser_parser.add_subparsers(dest="browser_action")

    attach = browser_sub.add_parser(
        "attach",
        help="Scan for Electron apps and attach a session to one",
    )
    attach.add_argument(
        "app",
        nargs="?",
        help="App name filter (e.g. 'obsidian'). Omit to pick from a list.",
    )
    attach.add_argument(
        "--session",
        help="browser_exec session name to register (default: the app's name)",
    )
    attach.add_argument(
        "--no-relaunch",
        action="store_true",
        help="Never offer to quit/relaunch the app; only attach to an already-exposed port",
    )

    browser_sub.add_parser("list", help="List attached app sessions")

    detach = browser_sub.add_parser("detach", help="Remove an attached session")
    detach.add_argument("session", nargs="?", help="Session name to remove")

    browser_parser.set_defaults(func=cmd_browser)
