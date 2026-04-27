"""CLI commands for the google_meet plugin.

Wires ``hermes meet <subcommand>``:
  setup       — preflight playwright, chromium, auth file, print fixes
  auth        — open a browser to sign into Google, save storage state
  join <url>  — join a Meet URL synchronously (also callable from the agent)
  status      — print current bot state
  transcript  — print the transcript
  stop        — leave the current meeting
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

from plugins.google_meet import process_manager as pm
from plugins.google_meet.meet_bot import _is_safe_meet_url


def _auth_state_path() -> Path:
    return Path(get_hermes_home()) / "workspace" / "meetings" / "auth.json"


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Build the ``hermes meet`` argparse tree.

    Called by :func:`_register_cli_commands` at plugin load time.
    """
    subs = subparser.add_subparsers(dest="meet_command")

    subs.add_parser("setup", help="Preflight: playwright, chromium, auth")

    subs.add_parser("auth", help="Sign in to Google and save session state")

    join_p = subs.add_parser("join", help="Join a Meet URL")
    join_p.add_argument("url", help="https://meet.google.com/...")
    join_p.add_argument("--guest-name", default="Hermes Agent")
    join_p.add_argument("--duration", default=None, help="e.g. 30m, 2h, 90s")
    join_p.add_argument("--headed", action="store_true", help="show browser")
    join_p.add_argument(
        "--mode", choices=("transcribe", "realtime"), default="transcribe",
        help="transcribe (default, listen-only) or realtime (speak via OpenAI Realtime)"
    )
    join_p.add_argument(
        "--node", default=None,
        help="remote node name, or 'auto' to use the sole registered node"
    )

    subs.add_parser("status", help="Print current Meet bot state")

    tr_p = subs.add_parser("transcript", help="Print the scraped transcript")
    tr_p.add_argument("--last", type=int, default=None)

    say_p = subs.add_parser("say", help="Speak text in an active realtime meeting")
    say_p.add_argument("text", help="what to say")
    say_p.add_argument("--node", default=None)

    subs.add_parser("stop", help="Leave the current meeting")

    # v3: remote node host management.
    node_p = subs.add_parser(
        "node",
        help="Manage remote meet node hosts (run/list/approve/remove/status/ping)",
    )
    try:
        from plugins.google_meet.node.cli import register_cli as _register_node_cli
        _register_node_cli(node_p)
    except Exception as e:  # pragma: no cover — defensive
        # If the node module fails to import for any reason (optional dep
        # missing at import time etc.), leave the subparser present but
        # flag it. The argparse dispatch will surface a clear error.
        def _node_unavailable(args):
            print(f"hermes meet node: module unavailable ({e})")
            return 1
        node_p.set_defaults(func=_node_unavailable)

    subparser.set_defaults(func=meet_command)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def meet_command(args: argparse.Namespace) -> int:
    sub = getattr(args, "meet_command", None)
    if not sub:
        print("usage: hermes meet {setup,auth,join,status,transcript,say,stop,node}")
        return 2
    if sub == "setup":
        return _cmd_setup()
    if sub == "auth":
        return _cmd_auth()
    if sub == "join":
        return _cmd_join(
            url=args.url,
            guest_name=args.guest_name,
            duration=args.duration,
            headed=args.headed,
            mode=getattr(args, "mode", "transcribe"),
            node=getattr(args, "node", None),
        )
    if sub == "status":
        return _cmd_status()
    if sub == "transcript":
        return _cmd_transcript(last=args.last)
    if sub == "say":
        return _cmd_say(text=args.text, node=getattr(args, "node", None))
    if sub == "stop":
        return _cmd_stop()
    if sub == "node":
        # Dispatch was set by the node cli's register_cli; fall through to
        # whatever its subparsers wired.
        fn = getattr(args, "func", None)
        if fn is None or fn is meet_command:
            print("usage: hermes meet node {run,list,approve,remove,status,ping}")
            return 2
        return fn(args)
    print(f"unknown subcommand: {sub}")
    return 2


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_setup() -> int:
    import platform as _p

    print("google_meet preflight")
    print("---------------------")

    system = _p.system()
    system_ok = system in ("Linux", "Darwin")
    print(f"  platform       : {system}  [{'ok' if system_ok else 'unsupported'}]")

    try:
        import playwright  # noqa: F401
        pw_ok = True
        pw_msg = "installed"
    except ImportError:
        pw_ok = False
        pw_msg = "NOT installed — run: pip install playwright"
    print(f"  playwright     : {pw_msg}")

    chromium_ok = False
    chromium_msg = "unknown"
    if pw_ok:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                try:
                    exe = p.chromium.executable_path
                    if exe and Path(exe).exists():
                        chromium_ok = True
                        chromium_msg = f"ok ({exe})"
                    else:
                        chromium_msg = (
                            "not installed — run: "
                            "python -m playwright install chromium"
                        )
                except Exception as e:
                    chromium_msg = f"probe failed: {e}"
        except Exception as e:
            chromium_msg = f"probe failed: {e}"
    print(f"  chromium       : {chromium_msg}")

    auth_path = _auth_state_path()
    auth_ok = auth_path.is_file()
    print(
        "  google auth    : "
        + (f"ok ({auth_path})" if auth_ok else "not saved — run: hermes meet auth")
    )

    print()
    all_ok = system_ok and pw_ok and chromium_ok
    if all_ok:
        print(
            "ready. Join a meeting:  "
            "hermes meet join https://meet.google.com/abc-defg-hij"
        )
    else:
        print("not ready yet — fix the items above.")
    return 0 if all_ok else 1


def _cmd_auth() -> int:
    """Open a headed Chromium, let the user sign in, save storage_state."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is not installed. run:\n"
            "  pip install playwright && python -m playwright install chromium"
        )
        return 1

    path = _auth_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"opening Chromium — sign in to Google, then return here and press Enter.")
    print(f"saving storage state to: {path}")
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=False)
            context = browser.new_context()
            page = context.new_page()
            page.goto("https://accounts.google.com/", wait_until="domcontentloaded")
            try:
                input("press Enter after you've signed in ... ")
            except EOFError:
                pass
            context.storage_state(path=str(path))
            browser.close()
    except Exception as e:
        print(f"auth failed: {e}")
        return 1
    print("saved. you can now run: hermes meet join <url>")
    return 0


def _cmd_join(
    url: str,
    *,
    guest_name: str,
    duration: Optional[str],
    headed: bool,
    mode: str = "transcribe",
    node: Optional[str] = None,
) -> int:
    if not _is_safe_meet_url(url):
        print(f"refusing: not a meet.google.com URL: {url}")
        return 2
    if node:
        # Remote: go through NodeClient.
        try:
            from plugins.google_meet.node.registry import NodeRegistry
            from plugins.google_meet.node.client import NodeClient
        except ImportError as e:
            print(f"node module unavailable: {e}")
            return 1
        reg = NodeRegistry()
        entry = reg.resolve(node if node != "auto" else None)
        if entry is None:
            print(f"no registered node matches {node!r}")
            return 1
        client = NodeClient(url=entry["url"], token=entry["token"])
        try:
            res = client.start_bot(
                url=url, guest_name=guest_name, duration=duration,
                headed=headed, mode=mode,
            )
        except Exception as e:
            print(f"remote start_bot failed: {e}")
            return 1
        print(json.dumps({"node": entry.get("name"), **res}, indent=2))
        return 0 if res.get("ok") else 1

    auth = _auth_state_path()
    res = pm.start(
        url=url,
        headed=headed,
        guest_name=guest_name,
        duration=duration,
        auth_state=str(auth) if auth.is_file() else None,
        mode=mode,
    )
    print(json.dumps(res, indent=2))
    return 0 if res.get("ok") else 1


def _cmd_say(text: str, node: Optional[str] = None) -> int:
    if not (text or "").strip():
        print("refusing: empty text")
        return 2
    if node:
        try:
            from plugins.google_meet.node.registry import NodeRegistry
            from plugins.google_meet.node.client import NodeClient
        except ImportError as e:
            print(f"node module unavailable: {e}")
            return 1
        reg = NodeRegistry()
        entry = reg.resolve(node if node != "auto" else None)
        if entry is None:
            print(f"no registered node matches {node!r}")
            return 1
        client = NodeClient(url=entry["url"], token=entry["token"])
        try:
            res = client.say(text)
        except Exception as e:
            print(f"remote say failed: {e}")
            return 1
        print(json.dumps({"node": entry.get("name"), **res}, indent=2))
        return 0 if res.get("ok") else 1

    res = pm.enqueue_say(text)
    print(json.dumps(res, indent=2))
    return 0 if res.get("ok") else 1


def _cmd_status() -> int:
    res = pm.status()
    print(json.dumps(res, indent=2))
    return 0 if res.get("ok") else 1


def _cmd_transcript(last: Optional[int]) -> int:
    res = pm.transcript(last=last)
    if not res.get("ok"):
        print(json.dumps(res, indent=2))
        return 1
    for ln in res.get("lines", []):
        print(ln)
    return 0


def _cmd_stop() -> int:
    res = pm.stop(reason="hermes meet stop")
    print(json.dumps(res, indent=2))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(prog="hermes meet")
    register_cli(parser)
    ns = parser.parse_args()
    sys.exit(meet_command(ns))
