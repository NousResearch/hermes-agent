"""CLI commands for the google_meet plugin (``hermes meet <subcommand>``).

setup / install — preflight and install prerequisites
auth            — open a browser to sign into Google, save storage state
join <url>      — join a Meet URL (locally or on a remote node)
status / transcript / say / stop — drive the active bot
node            — remote node host management (see node/cli.py)
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

from plugins.google_meet import process_manager as pm
from plugins.google_meet.meet_bot import _is_safe_meet_url
from plugins.google_meet.node.cli import (
    node_command,
    register_cli as _register_node_cli,
)
from plugins.google_meet.tools import resolve_node


def _auth_state_path() -> Path:
    return Path(get_hermes_home()) / "workspace" / "meetings" / "auth.json"


# ``hermes meet <sub>`` in help order.
_SUBCOMMAND_HELP = (
    ("setup", "Preflight: playwright, chromium, auth"),
    ("install", "Install prerequisites (pip deps, Chromium, platform audio tools)"),
    ("auth", "Sign in to Google and save session state"),
    ("join", "Join a Meet URL"),
    ("status", "Print current Meet bot state"),
    ("transcript", "Print the scraped transcript"),
    ("say", "Speak text in an active realtime meeting"),
    ("stop", "Leave the current meeting"),
    ("node", "Manage remote meet node hosts (run/list/approve/remove/status/ping)"),
)


def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Build the ``hermes meet`` argparse tree (called at plugin load time)."""
    subs = subparser.add_subparsers(dest="meet_command")
    parsers = {
        name: subs.add_parser(name, help=help_text)
        for name, help_text in _SUBCOMMAND_HELP
    }
    parsers["install"].add_argument(
        "--realtime",
        action="store_true",
        help=(
            "Also install realtime audio tools (pulseaudio-utils on Linux, BlackHole+ffmpeg on "
            "macOS). Uses sudo/brew, prompts before invoking either."
        ),
    )
    parsers["install"].add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Answer yes to all prompts (use with care; will run sudo apt-get or brew without asking).",
    )
    parsers["join"].add_argument("url", help="https://meet.google.com/...")
    parsers["join"].add_argument("--guest-name", default="Hermes Agent")
    parsers["join"].add_argument("--duration", default=None, help="e.g. 30m, 2h, 90s")
    parsers["join"].add_argument(
        "--persist-after-session",
        action="store_true",
        help="Keep the bot running after the current Hermes session ends.",
    )
    parsers["join"].add_argument(
        "--use-auth-state",
        action="store_true",
        help="Explicitly reuse saved Google auth state from hermes meet auth instead of guest mode.",
    )
    parsers["join"].add_argument("--headed", action="store_true", help="show browser")
    parsers["join"].add_argument(
        "--mode",
        choices=("transcribe", "realtime"),
        default="transcribe",
        help="transcribe (default, listen-only) or realtime (speak via OpenAI Realtime)",
    )
    parsers["join"].add_argument(
        "--node",
        default=None,
        help="remote node name, or 'auto' to use the sole registered node",
    )
    parsers["transcript"].add_argument("--last", type=int, default=None)
    parsers["say"].add_argument("text", help="what to say")
    parsers["say"].add_argument("--node", default=None)
    _register_node_cli(parsers["node"])
    subparser.set_defaults(func=meet_command)


_DISPATCH = {
    "setup": lambda args: _cmd_setup(),
    "install": lambda args: _cmd_install(
        realtime=bool(args.realtime), assume_yes=bool(args.yes)
    ),
    "auth": lambda args: _cmd_auth(),
    "join": lambda args: _cmd_join(
        url=args.url,
        guest_name=args.guest_name,
        duration=args.duration,
        headed=args.headed,
        mode=args.mode,
        node=args.node,
        persist_after_session=bool(args.persist_after_session),
        use_auth_state=bool(args.use_auth_state),
    ),
    "status": lambda args: _print_result(pm.status()),
    "transcript": lambda args: _cmd_transcript(last=args.last),
    "say": lambda args: _cmd_say(text=args.text, node=args.node),
    "stop": lambda args: _print_result(pm.stop(reason="hermes meet stop")),
    "node": node_command,
}


def meet_command(args: argparse.Namespace) -> int:
    subcommand = args.meet_command
    if not subcommand:
        print("usage: hermes meet {setup,auth,join,status,transcript,say,stop,node}")
        return 2
    handler = _DISPATCH.get(subcommand)
    if handler is None:
        print(f"unknown subcommand: {subcommand}")
        return 2
    return handler(args)


def _cmd_setup() -> int:
    print("google_meet preflight\n---------------------")
    system = platform.system()
    system_ok = system in {"Linux", "Darwin"}
    print(f"  platform       : {system}  [{'ok' if system_ok else 'unsupported'}]")
    playwright_ok = importlib.util.find_spec("playwright") is not None
    print(
        "  playwright     : "
        + (
            "installed"
            if playwright_ok
            else "NOT installed — run: pip install playwright"
        )
    )
    chromium_ok, chromium_message = False, "unknown"
    if playwright_ok:
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as browser:
                executable = browser.chromium.executable_path
            chromium_ok = bool(executable and Path(executable).exists())
            chromium_message = (
                f"ok ({executable})"
                if chromium_ok
                else "not installed — run: python -m playwright install chromium"
            )
        except Exception as exc:
            chromium_message = f"probe failed: {exc}"
    print(f"  chromium       : {chromium_message}")
    auth_path = _auth_state_path()
    print(
        "  google auth    : "
        + (
            f"ok ({auth_path})"
            if auth_path.is_file()
            else "not saved — run: hermes meet auth"
        )
    )
    print()
    all_ok = system_ok and playwright_ok and chromium_ok
    print(
        "ready. Join a meeting:  hermes meet join https://meet.google.com/abc-defg-hij"
        if all_ok
        else "not ready yet — fix the items above."
    )
    return 0 if all_ok else 1


def _cmd_install(*, realtime: bool, assume_yes: bool) -> int:
    """Install browser dependencies and optional realtime audio dependencies."""
    system = platform.system()
    if system not in {"Linux", "Darwin"}:
        print(f"google_meet install: {system} is not supported (linux/macos only)")
        return 1

    def _install_packages(prompt: str, command: list[str], failure: str) -> None:
        try:
            confirmed = assume_yes or input(f"{prompt} [y/N] ").strip().lower() in {
                "y",
                "yes",
            }
        except EOFError:
            confirmed = False
        if not confirmed:
            print("  skipped (you can run it manually later)")
            return
        print(f"  $ {' '.join(command)}")
        if subprocess.run(command, check=False).returncode != 0:
            print(failure)

    print("google_meet install\n-------------------")
    packages = ["playwright", "websockets"]
    print(f"\n[1/3] pip install: {' '.join(packages)}")
    try:
        from hermes_cli.tools_config import _pip_install

        if _pip_install(["--upgrade", *packages], capture_output=False).returncode != 0:
            print("  pip install failed")
            return 1
    except Exception as exc:
        print(f"  pip install failed: {exc}")
        return 1

    print("\n[2/3] python -m playwright install chromium")
    try:
        if (
            subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                check=False,
                stdin=subprocess.DEVNULL,
            ).returncode
            != 0
        ):
            print("  playwright install failed (may already be installed)")
    except Exception as exc:
        print(f"  playwright install failed: {exc}")
        return 1

    if not realtime:
        print("\n[3/3] skipped (pass --realtime to install audio tooling too)")
    else:
        print("\n[3/3] realtime audio deps")
        if system == "Linux":
            if shutil.which("paplay") and shutil.which("pactl"):
                print("  pulseaudio-utils already installed.")
            else:
                _install_packages(
                    "  install pulseaudio-utils? this runs `sudo apt-get install -y pulseaudio-utils`",
                    ["sudo", "apt-get", "install", "-y", "pulseaudio-utils"],
                    "  apt install failed — install pulseaudio-utils manually",
                )
        elif system == "Darwin":
            try:
                have_blackhole = "BlackHole" in subprocess.check_output(
                    ["system_profiler", "SPAudioDataType"],
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stdin=subprocess.DEVNULL,
                )
            except Exception:
                have_blackhole = False
            needs = [
                package
                for package, available in (
                    ("blackhole-2ch", have_blackhole),
                    ("ffmpeg", shutil.which("ffmpeg")),
                )
                if not available
            ]
            if not needs:
                print("  BlackHole and ffmpeg already installed.")
            elif not shutil.which("brew"):
                print(
                    "  missing: "
                    + ", ".join(needs)
                    + "\n  install Homebrew first (https://brew.sh) or install the packages manually."
                )
            else:
                _install_packages(
                    f"  install via brew: {' '.join(needs)}?",
                    ["brew", "install", *needs],
                    "  brew install failed — install them manually",
                )
            print(
                "\n  NOTE: macOS does not auto-route audio. Open\n    System Settings → Sound → "
                "Input\n  and select 'BlackHole 2ch' before starting a realtime meeting.\n  "
                "hermes will not switch your default input for you."
            )
    print("\ndone. verify with: hermes meet setup")
    return 0


def _cmd_auth() -> int:
    """Open headed Chromium and save an explicitly reusable storage state."""
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
    print(
        "opening Chromium — sign in to Google, then return here and press Enter.\n"
        f"saving storage state to: {path}"
    )
    try:
        with sync_playwright() as browser:
            chromium = browser.chromium.launch(headless=False)
            context = chromium.new_context()
            context.new_page().goto(
                "https://accounts.google.com/", wait_until="domcontentloaded"
            )
            with contextlib.suppress(EOFError):
                input("press Enter after you've signed in ... ")
            context.storage_state(path=str(path))
            chromium.close()
    except Exception as exc:
        print(f"auth failed: {exc}")
        return 1
    print(
        "saved. use --use-auth-state with a local hermes meet join to reuse this session."
    )
    return 0


def _print_result(result: dict) -> int:
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def _remote(node: str, operation: str, call) -> int:
    """Run *call(client)* against the registered node and print its result."""
    try:
        client, name = resolve_node(node)
    except ImportError as exc:
        print(f"node module unavailable: {exc}")
        return 1
    if client is None:
        print(f"no registered node matches {node!r}")
        return 1
    try:
        result = call(client)
    except Exception as exc:
        print(f"remote {operation} failed: {exc}")
        return 1
    return _print_result({"node": name, **result})


def _cmd_join(
    url: str,
    *,
    guest_name: str,
    duration: Optional[str],
    headed: bool,
    mode: str = "transcribe",
    node: Optional[str] = None,
    persist_after_session: bool = False,
    use_auth_state: bool = False,
) -> int:
    if not _is_safe_meet_url(url):
        print(f"refusing: not a meet.google.com URL: {url}")
        return 2
    if node:
        if use_auth_state:
            print(
                "use_auth_state is local-only; authenticate on the node host or omit --node"
            )
            return 1
        return _remote(
            node,
            "start_bot",
            lambda client: client.start_bot(
                url=url,
                guest_name=guest_name,
                duration=duration,
                persist_after_session=persist_after_session,
                headed=headed,
                mode=mode,
            ),
        )

    auth_state = _auth_state_path()
    return _print_result(
        pm.start(
            url=url,
            headed=headed,
            guest_name=guest_name,
            duration=duration,
            persist_after_session=persist_after_session,
            auth_state=str(auth_state)
            if use_auth_state and auth_state.is_file()
            else None,
            mode=mode,
        )
    )


def _cmd_say(text: str, node: Optional[str] = None) -> int:
    if not (text or "").strip():
        print("refusing: empty text")
        return 2
    if node:
        return _remote(node, "say", lambda client: client.say(text))
    return _print_result(pm.enqueue_say(text))


def _cmd_transcript(last: Optional[int]) -> int:
    result = pm.transcript(last=last)
    if not result.get("ok"):
        return _print_result(result)
    for line in result.get("lines", []):
        print(line)
    return 0


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(prog="hermes meet")
    register_cli(parser)
    sys.exit(meet_command(parser.parse_args()))
