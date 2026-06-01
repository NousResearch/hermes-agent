"""CLI for the LM-twitterer Hermes plugin."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import core


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="lm_twitterer_command")

    subs.add_parser("status", help="Show dependency and configuration readiness")
    subs.add_parser("auth-check", help="Validate X cookies without posting")

    setup = subs.add_parser("setup", help="Save bot screen name and X cookies to ~/.hermes/.env")
    setup.add_argument("--screen-name", default=None, help="Bot X screen name, without @")
    setup.add_argument(
        "--skip-secrets",
        action="store_true",
        help="Only save the screen name; do not prompt for cookies",
    )

    auth = subs.add_parser("auth-browser", help="Log in to X in a temporary browser and save X cookies")
    auth.add_argument("--screen-name", default=None, help="Bot X screen name, without @")
    auth.add_argument(
        "--browser",
        choices=("chromium", "edge"),
        default="chromium",
        help="Browser to open for login. Use edge for Microsoft Edge.",
    )
    auth.add_argument(
        "--channel",
        default=None,
        help="Playwright browser channel override, e.g. msedge. Usually set by --browser edge.",
    )
    auth.add_argument(
        "--profile-dir",
        default=None,
        help="Playwright profile directory. Defaults to ~/.hermes/lm-twitterer/<browser>-profile",
    )
    auth.add_argument(
        "--wait-seconds",
        type=int,
        default=0,
        help="Poll the visible browser for X cookies instead of waiting for Enter. 0 keeps manual Enter mode.",
    )

    edge_direct = subs.add_parser(
        "auth-edge-direct",
        help="Advanced local-only: open Edge and read X cookies through local CDP after manual login",
    )
    edge_direct.add_argument("--screen-name", default=None, help="Bot X screen name, without @")
    edge_direct.add_argument(
        "--wait-seconds",
        type=int,
        default=900,
        help="Seconds to wait for X cookies after opening Edge.",
    )
    edge_direct.add_argument(
        "--port",
        type=int,
        default=9223,
        help="Preferred local CDP port.",
    )
    edge_direct.add_argument(
        "--profile-dir",
        default=None,
        help="Dedicated Edge profile directory. Defaults to ~/.hermes/lm-twitterer/edge-direct-profile",
    )
    edge_direct.add_argument(
        "--edge-path",
        default=None,
        help="Path to msedge.exe if it cannot be found automatically.",
    )

    edge = subs.add_parser(
        "import-edge-cookies",
        help="Advanced local-only: import X cookies from a Microsoft Edge profile without printing secrets",
    )
    edge.add_argument("--screen-name", default=None, help="Bot X screen name, without @")
    edge.add_argument(
        "--profile",
        default=None,
        help="Edge profile directory name, e.g. Default or Profile 1. Defaults to Edge's last-used profile.",
    )
    edge.add_argument(
        "--user-data-dir",
        default=None,
        help="Edge User Data directory. Defaults to %LOCALAPPDATA%\\Microsoft\\Edge\\User Data.",
    )

    install = subs.add_parser("install-deps", help="Install X client dependencies into this Python environment")
    install.add_argument("--yes", "-y", action="store_true", help="Do not prompt before installing")
    install.add_argument(
        "--browser",
        action="store_true",
        help="Also install Playwright and Chromium for auth-browser setup",
    )

    trust = subs.add_parser(
        "trust-llm-overrides",
        help="Allow this plugin to use explicit Hermes provider/model overrides",
    )
    trust.add_argument("--provider", default="opencode-zen", help="Provider to allowlist")
    trust.add_argument("--model", default="auto-free", help="Model to allowlist")
    trust.add_argument(
        "--allow-any",
        action="store_true",
        help="Allow any provider/model override for this plugin instead of allowlisting one pair",
    )

    post = subs.add_parser("post", help="Generate a post")
    post.add_argument("topic", nargs="*", help="Post topic or instruction")
    post.add_argument("--live", action="store_true", help="Publish instead of dry-run")
    post.add_argument("--provider", default=None, help="Hermes provider override for generation")
    post.add_argument("--model", default=None, help="Hermes model override for generation")

    replies = subs.add_parser("replies", help="Check mentions and generate replies")
    replies.add_argument("--live", action="store_true", help="Publish instead of dry-run")
    replies.add_argument("--count", type=int, default=20)
    replies.add_argument("--mark-seen-on-dry-run", action="store_true")
    replies.add_argument("--provider", default=None, help="Hermes provider override for generation")
    replies.add_argument("--model", default=None, help="Hermes model override for generation")

    mentions = subs.add_parser("mentions", help="List recent mention candidates without replying")
    mentions.add_argument("--count", type=int, default=20)
    mentions.add_argument("--max-text-chars", type=int, default=180)

    wl = subs.add_parser("whitelist", help="Manage the reply whitelist")
    wl_subs = wl.add_subparsers(dest="whitelist_command")
    wl_subs.add_parser("list", help="List whitelisted screen names")
    wl_add = wl_subs.add_parser("add", help="Add a screen name")
    wl_add.add_argument("screen_name")
    wl_rm = wl_subs.add_parser("remove", help="Remove a screen name")
    wl_rm.add_argument("screen_name")
    wl_import = wl_subs.add_parser(
        "import-mentioned-followers",
        help="Add recent mention authors who follow the bot account",
    )
    wl_import.add_argument("--count", type=int, default=100)

    cron = subs.add_parser("cron", help="Install recurring Hermes cron jobs")
    cron_subs = cron.add_subparsers(dest="cron_command")
    install_cron = cron_subs.add_parser("install", help="Create post/reply cron jobs")
    install_cron.add_argument("--post-schedule", default="every 6h")
    install_cron.add_argument("--reply-schedule", default="every 1h")
    install_cron.add_argument("--deliver", default="local")
    install_cron.add_argument("--profile", default=None)
    install_cron.add_argument("--post-topic", default="")
    install_cron.add_argument("--reply-count", type=int, default=20)
    install_cron.add_argument("--provider", default=None, help="Hermes provider override for cron generation")
    install_cron.add_argument("--model", default=None, help="Hermes model override for cron generation")
    install_cron.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the jobs that would be created without writing cron state",
    )
    install_cron.add_argument(
        "--force",
        action="store_true",
        help="Create jobs even if X cookies, bot screen name, or whitelist are missing",
    )
    install_cron.add_argument(
        "--paused",
        action="store_true",
        help="Create the cron jobs but immediately pause them so nothing posts until resumed.",
    )

    subparser.set_defaults(func=lm_twitterer_command)


def lm_twitterer_command(args: argparse.Namespace) -> int:
    command = getattr(args, "lm_twitterer_command", None)
    if not command:
        print("usage: hermes lm-twitterer {status,auth-check,setup,auth-browser,auth-edge-direct,import-edge-cookies,install-deps,trust-llm-overrides,post,mentions,replies,whitelist,cron}")
        return 2
    if command == "status":
        return _print(core.status())
    if command == "auth-check":
        return _print(core.auth_check())
    if command == "setup":
        return _setup_command(args)
    if command == "auth-browser":
        return _auth_browser_command(args)
    if command == "auth-edge-direct":
        return _auth_edge_direct_command(args)
    if command == "import-edge-cookies":
        return _import_edge_cookies_command(args)
    if command == "install-deps":
        return _install_deps(
            bool(getattr(args, "yes", False)),
            include_browser=bool(getattr(args, "browser", False)),
        )
    if command == "trust-llm-overrides":
        return _trust_llm_overrides_command(args)
    if command == "post":
        topic = " ".join(getattr(args, "topic", []) or [])
        return _print(
            core.post(
                topic,
                dry_run=not bool(getattr(args, "live", False)),
                provider=getattr(args, "provider", None),
                model=getattr(args, "model", None),
            )
        )
    if command == "replies":
        return _print(
            core.reply_mentions(
                dry_run=not bool(getattr(args, "live", False)),
                count=int(getattr(args, "count", 20) or 20),
                mark_seen_on_dry_run=bool(getattr(args, "mark_seen_on_dry_run", False)),
                provider=getattr(args, "provider", None),
                model=getattr(args, "model", None),
            )
        )
    if command == "mentions":
        return _print(
            core.mention_candidates(
                count=int(getattr(args, "count", 20) or 20),
                max_text_chars=int(getattr(args, "max_text_chars", 180) or 180),
            )
        )
    if command == "whitelist":
        return _whitelist_command(args)
    if command == "cron":
        return _cron_command(args)
    print(f"unknown command: {command}")
    return 2


def _print(data: dict) -> int:
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0 if data.get("ok") else 1


def _install_deps(assume_yes: bool, *, include_browser: bool = False) -> int:
    packages = [
        "twitter-openapi-python>=0.0.44,<0.1",
        "tweepy-authlib>=1.7.3,<2",
    ]
    if include_browser:
        packages.append("playwright>=1.44,<2")
    if not assume_yes:
        try:
            answer = input(
                "Install X client dependencies into this Hermes Python environment? [y/N] "
            ).strip().lower()
        except EOFError:
            answer = ""
        if answer not in {"y", "yes"}:
            print("skipped")
            return 1
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        return _install_browser_runtime(include_browser)
    # Some Hermes Windows venvs are uv-managed and intentionally omit pip.
    uv_cmd = ["uv", "pip", "install", "--python", sys.executable, *packages]
    result = subprocess.run(uv_cmd, check=False)
    if result.returncode != 0:
        return result.returncode
    return _install_browser_runtime(include_browser)


def _install_browser_runtime(include_browser: bool) -> int:
    if not include_browser:
        return 0
    result = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        check=False,
    )
    return result.returncode


def _setup_command(args: argparse.Namespace) -> int:
    try:
        from hermes_cli.config import get_env_value, save_env_value
        from hermes_cli.secret_prompt import masked_secret_prompt
        from hermes_constants import display_hermes_home
    except Exception as exc:
        print(f"setup unavailable: {exc}")
        return 1

    print("LM-twitterer setup")
    print("------------------")
    print("Use the logged-in X account browser session.")
    print("Get cookies from DevTools > Application > Cookies > https://x.com")
    print()

    screen_name = (getattr(args, "screen_name", None) or "").strip().lstrip("@")
    if not screen_name:
        existing = get_env_value("LM_TWITTERER_BOT_SCREEN_NAME") or ""
        prompt = "Bot screen name without @"
        if existing:
            prompt += f" [{existing}]"
        try:
            entered = input(f"{prompt}: ").strip().lstrip("@")
        except EOFError:
            entered = ""
        screen_name = entered or existing
    if screen_name:
        save_env_value("LM_TWITTERER_BOT_SCREEN_NAME", screen_name)

    if not bool(getattr(args, "skip_secrets", False)):
        for env_name, label in (
            ("LM_TWITTERER_AUTH_TOKEN", "X auth_token cookie"),
            ("LM_TWITTERER_CT0", "X ct0 cookie"),
        ):
            existing = get_env_value(env_name)
            suffix = " (Enter to keep current)" if existing else ""
            try:
                value = masked_secret_prompt(f"{label}{suffix}: ").strip()
            except EOFError:
                value = ""
            if value:
                save_env_value(env_name, value)

    print(f"Saved settings to {display_hermes_home()}/.env")
    return _print(core.status())


def _trust_llm_overrides_command(args: argparse.Namespace) -> int:
    try:
        import yaml
        from hermes_cli.config import ensure_hermes_home, get_config_path
        from utils import atomic_yaml_write
    except Exception as exc:
        print(f"config update unavailable: {exc}")
        return 1

    provider = (getattr(args, "provider", "") or "").strip()
    model = (getattr(args, "model", "") or "").strip()
    allow_any = bool(getattr(args, "allow_any", False))

    ensure_hermes_home()
    config_path = get_config_path()
    user_config = {}
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                user_config = yaml.safe_load(fh) or {}
        except Exception:
            user_config = {}

    plugins = user_config.setdefault("plugins", {})
    entries = plugins.setdefault("entries", {})
    entry = entries.setdefault("lm-twitterer", {})
    llm = entry.setdefault("llm", {})
    llm["allow_provider_override"] = True
    llm["allow_model_override"] = True

    if allow_any:
        llm["allowed_providers"] = ["*"]
        llm["allowed_models"] = ["*"]
    else:
        if provider:
            llm["allowed_providers"] = _append_allowlist(llm.get("allowed_providers"), provider)
        if model:
            llm["allowed_models"] = _append_allowlist(llm.get("allowed_models"), model)

    atomic_yaml_write(config_path, user_config, sort_keys=False)
    return _print(
        {
            "ok": True,
            "config_path": str(config_path),
            "llm": {
                "allow_provider_override": llm.get("allow_provider_override"),
                "allow_model_override": llm.get("allow_model_override"),
                "allowed_providers": llm.get("allowed_providers"),
                "allowed_models": llm.get("allowed_models"),
            },
        }
    )


def _auth_browser_command(args: argparse.Namespace) -> int:
    try:
        from hermes_cli.config import get_env_value, save_env_value
        from hermes_constants import get_hermes_home, display_hermes_home
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        print(
            "browser auth unavailable. Run:\n"
            "  hermes lm-twitterer install-deps --browser --yes\n\n"
            f"Details: {exc}"
        )
        return 1

    screen_name = (getattr(args, "screen_name", None) or "").strip().lstrip("@")
    if not screen_name:
        existing = get_env_value("LM_TWITTERER_BOT_SCREEN_NAME") or ""
        prompt = "Bot screen name without @"
        if existing:
            prompt += f" [{existing}]"
        try:
            entered = input(f"{prompt}: ").strip().lstrip("@")
        except EOFError:
            entered = ""
        screen_name = entered or existing

    browser = str(getattr(args, "browser", "chromium") or "chromium")
    channel = (getattr(args, "channel", None) or "").strip()
    if browser == "edge" and not channel:
        channel = "msedge"

    profile_dir_raw = getattr(args, "profile_dir", None)
    default_profile_name = "edge-profile" if browser == "edge" else "browser-profile"
    profile_dir = (
        Path(profile_dir_raw).expanduser()
        if profile_dir_raw
        else Path(get_hermes_home()) / "lm-twitterer" / default_profile_name
    )
    profile_dir.mkdir(parents=True, exist_ok=True)

    browser_label = "Microsoft Edge" if browser == "edge" else "Chromium"
    print(f"A temporary {browser_label} profile will open.")
    if int(getattr(args, "wait_seconds", 0) or 0) > 0:
        print("Sign in to the X account. Hermes will save cookies once X is logged in.")
    else:
        print("Sign in to the X account, then return here and press Enter.")
    print("Only auth_token and ct0 from x.com will be saved to Hermes .env.")
    print()

    try:
        with sync_playwright() as pw:
            launch_kwargs = {
                "user_data_dir": str(profile_dir),
                "headless": False,
            }
            if channel:
                launch_kwargs["channel"] = channel
            context = pw.chromium.launch_persistent_context(**launch_kwargs)
            page = context.new_page()
            page.goto("https://x.com/i/flow/login", wait_until="domcontentloaded")
            wait_seconds = max(0, int(getattr(args, "wait_seconds", 0) or 0))
            if wait_seconds:
                cookies = _wait_for_x_cookies(context, wait_seconds=wait_seconds)
            else:
                try:
                    input("Press Enter after X is logged in ... ")
                except EOFError:
                    pass
                cookies = _safe_context_cookies(context)
            try:
                context.close()
            except Exception:
                pass
    except Exception as exc:
        print(f"browser auth failed: {exc}")
        return 1

    by_name = {cookie.get("name"): cookie.get("value") for cookie in cookies}
    auth_token = by_name.get("auth_token")
    ct0 = by_name.get("ct0")
    if not auth_token or not ct0:
        print("Did not find both auth_token and ct0 cookies. Make sure X is fully logged in.")
        return 1

    save_env_value("LM_TWITTERER_AUTH_TOKEN", str(auth_token))
    save_env_value("LM_TWITTERER_CT0", str(ct0))
    if screen_name:
        save_env_value("LM_TWITTERER_BOT_SCREEN_NAME", screen_name)

    print(f"Saved X cookies to {display_hermes_home()}/.env")
    return _print(core.status())


def _wait_for_x_cookies(context, *, wait_seconds: int):
    deadline = time.monotonic() + wait_seconds
    last_cookies = []
    while time.monotonic() < deadline:
        cookies = _safe_context_cookies(context)
        if not cookies:
            return last_cookies
        last_cookies = cookies
        by_name = {cookie.get("name"): cookie.get("value") for cookie in last_cookies}
        if by_name.get("auth_token") and by_name.get("ct0"):
            return last_cookies
        time.sleep(2)
    return last_cookies


def _safe_context_cookies(context):
    try:
        return context.cookies("https://x.com")
    except Exception:
        return []


def _auth_edge_direct_command(args: argparse.Namespace) -> int:
    try:
        from hermes_cli.config import get_env_value, save_env_value
        from hermes_constants import get_hermes_home, display_hermes_home
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        print(
            "direct Edge auth unavailable. Run:\n"
            "  hermes lm-twitterer install-deps --browser --yes\n\n"
            f"Details: {exc}"
        )
        return 1

    screen_name = (getattr(args, "screen_name", None) or "").strip().lstrip("@")
    if not screen_name:
        existing = get_env_value("LM_TWITTERER_BOT_SCREEN_NAME") or ""
        screen_name = existing.strip().lstrip("@")

    profile_dir_raw = getattr(args, "profile_dir", None)
    profile_dir = (
        Path(profile_dir_raw).expanduser()
        if profile_dir_raw
        else Path(get_hermes_home()) / "lm-twitterer" / "edge-direct-profile"
    )
    profile_dir.mkdir(parents=True, exist_ok=True)

    try:
        edge_path = _find_edge_executable(getattr(args, "edge_path", None))
    except _EdgeCookieImportError as exc:
        return _print({"ok": False, "error": str(exc)})

    try:
        port = _choose_cdp_port(int(getattr(args, "port", 9223) or 9223))
    except _EdgeCookieImportError as exc:
        return _print({"ok": False, "error": str(exc)})
    wait_seconds = max(10, int(getattr(args, "wait_seconds", 900) or 900))
    cmd = [
        str(edge_path),
        f"--remote-debugging-port={port}",
        "--remote-allow-origins=*",
        "--no-first-run",
        "--new-window",
        f"--user-data-dir={profile_dir}",
        "https://x.com/i/flow/login",
    ]
    print("A normal Microsoft Edge window will open.")
    print("Sign in to the X account in that window.")
    print("Hermes will only save auth_token and ct0 from x.com; cookie values are not printed.")
    print()

    process = subprocess.Popen(cmd)
    endpoint = f"http://127.0.0.1:{port}"
    if not _wait_for_cdp_endpoint(endpoint, timeout_seconds=20):
        return _print(
            {
                "ok": False,
                "error": "Edge CDP endpoint did not become ready.",
                "process_id": process.pid,
            }
        )

    cookies = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.connect_over_cdp(endpoint)
            cookies = _wait_for_cdp_x_cookies(browser, wait_seconds=wait_seconds)
            try:
                browser.close()
            except Exception:
                pass
    except Exception as exc:
        return _print(
            {
                "ok": False,
                "error": f"Could not read cookies through Edge CDP: {exc}",
                "process_id": process.pid,
            }
        )

    by_name = {cookie.get("name"): cookie.get("value") for cookie in cookies}
    auth_token = by_name.get("auth_token")
    ct0 = by_name.get("ct0")
    if not auth_token or not ct0:
        return _print(
            {
                "ok": False,
                "error": "Timed out before both auth_token and ct0 appeared in Edge.",
                "process_id": process.pid,
                "found_cookie_names": sorted(name for name in by_name if name in {"auth_token", "ct0"}),
                "next_steps": [
                    "Finish the X login in the Edge window, then rerun this command.",
                    "If X requests email or 2FA verification, complete it in the browser window.",
                ],
            }
        )

    save_env_value("LM_TWITTERER_AUTH_TOKEN", str(auth_token))
    save_env_value("LM_TWITTERER_CT0", str(ct0))
    if screen_name:
        save_env_value("LM_TWITTERER_BOT_SCREEN_NAME", screen_name)

    print(f"Saved X cookies to {display_hermes_home()}/.env")
    return _print(
        {
            "ok": True,
            "saved_cookie_names": ["auth_token", "ct0"],
            "bot_screen_name_set": bool(screen_name),
            "process_id": process.pid,
            "status": core.status(),
        }
    )


def _find_edge_executable(raw: str | None) -> Path:
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw).expanduser())
    path_candidate = shutil.which("msedge.exe") or shutil.which("msedge")
    if path_candidate:
        candidates.append(Path(path_candidate))
    for base in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        root = os.environ.get(base)
        if root:
            candidates.append(Path(root) / "Microsoft" / "Edge" / "Application" / "msedge.exe")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise _EdgeCookieImportError("Could not find msedge.exe. Pass --edge-path.")


def _choose_cdp_port(preferred: int) -> int:
    for port in [preferred, *range(preferred + 1, preferred + 25)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise _EdgeCookieImportError("Could not find a free local CDP port.")


def _wait_for_cdp_endpoint(endpoint: str, *, timeout_seconds: int) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{endpoint}/json/version", timeout=2) as resp:
                return resp.status == 200
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    return False


def _wait_for_cdp_x_cookies(browser, *, wait_seconds: int):
    deadline = time.monotonic() + wait_seconds
    last_cookies = []
    while time.monotonic() < deadline:
        contexts = list(getattr(browser, "contexts", []) or [])
        for context in contexts:
            cookies = _safe_context_cookies(context)
            if cookies:
                last_cookies = cookies
                by_name = {cookie.get("name"): cookie.get("value") for cookie in cookies}
                if by_name.get("auth_token") and by_name.get("ct0"):
                    return cookies
        time.sleep(2)
    return last_cookies


class _EdgeCookieImportError(Exception):
    pass


def _import_edge_cookies_command(args: argparse.Namespace) -> int:
    try:
        from hermes_cli.config import get_env_value, save_env_value
        from hermes_constants import display_hermes_home
    except Exception as exc:
        print(f"edge cookie import unavailable: {exc}")
        return 1

    try:
        user_data_dir = _edge_user_data_dir(getattr(args, "user_data_dir", None))
        profile = (getattr(args, "profile", None) or "").strip()
        if not profile:
            profile = _edge_last_used_profile(user_data_dir)
        profile = _safe_edge_profile_name(profile)
        result = _read_edge_x_cookies(user_data_dir, profile)
    except _EdgeCookieImportError as exc:
        screen_name = (getattr(args, "screen_name", None) or "").strip().lstrip("@")
        return _print(
            {
                "ok": False,
                "error": str(exc),
                "next_steps": _edge_import_next_steps(str(exc), screen_name=screen_name),
            }
        )

    cookies = result["cookies"]
    auth_token = cookies.get("auth_token")
    ct0 = cookies.get("ct0")
    if not auth_token or not ct0:
        return _print(
            {
                "ok": False,
                "profile": profile,
                "found_cookie_names": sorted(cookies),
                "source_mode": result["mode"],
                "error": "Did not find both auth_token and ct0 for x.com in the selected Edge profile.",
                "next_steps": [
                    "Confirm Edge is logged in at https://x.com with the intended X account.",
                    "Close Edge and retry this command.",
                ],
            }
        )

    screen_name = (getattr(args, "screen_name", None) or "").strip().lstrip("@")
    if not screen_name:
        existing = get_env_value("LM_TWITTERER_BOT_SCREEN_NAME") or ""
        screen_name = existing.strip().lstrip("@")

    save_env_value("LM_TWITTERER_AUTH_TOKEN", auth_token)
    save_env_value("LM_TWITTERER_CT0", ct0)
    if screen_name:
        save_env_value("LM_TWITTERER_BOT_SCREEN_NAME", screen_name)

    print(f"Saved X auth cookies from Edge profile '{profile}' to {display_hermes_home()}/.env")
    return _print(
        {
            "ok": True,
            "profile": profile,
            "source_mode": result["mode"],
            "saved_cookie_names": ["auth_token", "ct0"],
            "bot_screen_name_set": bool(screen_name),
            "status": core.status(),
        }
    )


def _edge_user_data_dir(raw: str | None) -> Path:
    if raw:
        path = Path(os.path.expandvars(raw)).expanduser()
    else:
        localappdata = os.environ.get("LOCALAPPDATA")
        if not localappdata:
            raise _EdgeCookieImportError("LOCALAPPDATA is not set, so the Edge profile directory cannot be located.")
        path = Path(localappdata) / "Microsoft" / "Edge" / "User Data"
    if not path.exists():
        raise _EdgeCookieImportError(f"Edge User Data directory was not found: {path}")
    return path


def _edge_import_next_steps(message: str, *, screen_name: str = "") -> list[str]:
    screen_arg = screen_name or "<name>"
    lower = message.lower()
    if "app-bound" in lower:
        return [
            f"Run: hermes lm-twitterer auth-browser --browser edge --screen-name {screen_arg} --wait-seconds 600",
            f"Or run: hermes lm-twitterer setup --screen-name {screen_arg}",
            "Manual setup reads cookies from DevTools and does not need Edge profile DB decryption.",
        ]
    return [
        "Finish logging in to X in Edge, then close Edge completely and retry this command.",
        f"Or run: hermes lm-twitterer setup --screen-name {screen_arg}",
        f"Or run: hermes lm-twitterer auth-browser --browser edge --screen-name {screen_arg}",
    ]


def _edge_last_used_profile(user_data_dir: Path) -> str:
    local_state = user_data_dir / "Local State"
    if not local_state.exists():
        return "Default"
    try:
        with local_state.open("r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
    except Exception:
        return "Default"
    profile = data.get("profile", {}).get("last_used") or "Default"
    return str(profile)


def _safe_edge_profile_name(profile: str) -> str:
    profile = profile.strip() or "Default"
    path = Path(profile)
    if path.is_absolute() or ".." in path.parts:
        raise _EdgeCookieImportError("Edge profile must be a profile directory name, not a path.")
    return profile


def _read_edge_x_cookies(user_data_dir: Path, profile: str) -> dict[str, object]:
    cookie_db = user_data_dir / profile / "Network" / "Cookies"
    if not cookie_db.exists():
        raise _EdgeCookieImportError(f"Edge cookie database was not found for profile '{profile}'.")

    master_key = _edge_master_key(user_data_dir)
    temp_dir = Path(tempfile.mkdtemp(prefix="lm-twitterer-edge-cookies-"))
    temp_db = temp_dir / "Cookies"
    try:
        try:
            _copy_cookie_db(cookie_db, temp_db)
            rows = _query_edge_cookie_rows(temp_db, immutable=False)
            mode = "copied-profile-db"
        except OSError:
            rows = _query_edge_cookie_rows(cookie_db, immutable=True)
            mode = "live-profile-db-readonly"

        cookies = _decrypt_edge_cookie_rows(rows, master_key)
        return {"cookies": cookies, "mode": mode}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _copy_cookie_db(cookie_db: Path, temp_db: Path) -> None:
    shutil.copy2(cookie_db, temp_db)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(cookie_db) + suffix)
        if sidecar.exists():
            try:
                shutil.copy2(sidecar, Path(str(temp_db) + suffix))
            except OSError:
                pass


def _query_edge_cookie_rows(db_path: Path, *, immutable: bool) -> list[tuple[str, str, str, bytes]]:
    try:
        if immutable:
            uri = db_path.resolve().as_uri() + "?mode=ro&immutable=1"
            conn = sqlite3.connect(uri, uri=True)
        else:
            conn = sqlite3.connect(str(db_path))
        try:
            return list(
                conn.execute(
                    """
                    SELECT host_key, name, value, encrypted_value
                    FROM cookies
                    WHERE name IN ('auth_token', 'ct0')
                      AND (host_key = 'x.com' OR host_key LIKE '%.x.com')
                    ORDER BY host_key, name
                    """
                )
            )
        finally:
            conn.close()
    except sqlite3.Error as exc:
        if immutable:
            raise _EdgeCookieImportError(
                "Could not read the Edge cookie database. Close Edge completely and retry."
            ) from exc
        raise OSError(str(exc)) from exc


def _edge_master_key(user_data_dir: Path) -> bytes:
    local_state = user_data_dir / "Local State"
    if not local_state.exists():
        raise _EdgeCookieImportError("Edge Local State file was not found.")
    try:
        with local_state.open("r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        encrypted_key = data.get("os_crypt", {}).get("encrypted_key")
        if not encrypted_key:
            raise ValueError("missing encrypted_key")
        raw = base64.b64decode(encrypted_key)
        if raw.startswith(b"DPAPI"):
            raw = raw[5:]
    except Exception as exc:
        raise _EdgeCookieImportError("Could not read Edge's encrypted cookie key metadata.") from exc

    try:
        import win32crypt
    except Exception as exc:
        raise _EdgeCookieImportError("pywin32 is required to decrypt Edge cookies on Windows.") from exc

    try:
        return bytes(win32crypt.CryptUnprotectData(raw, None, None, None, 0)[1])
    except Exception as exc:
        raise _EdgeCookieImportError("Windows refused to decrypt Edge's cookie key for this user.") from exc


def _decrypt_edge_cookie_rows(
    rows: list[tuple[str, str, str, bytes]],
    master_key: bytes,
) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for _, name, value, encrypted_value in rows:
        if name not in {"auth_token", "ct0"}:
            continue
        decoded = _decrypt_edge_cookie_value(value, encrypted_value, master_key)
        if decoded:
            cookies[name] = decoded
    return cookies


def _decrypt_edge_cookie_value(value: str, encrypted_value: bytes, master_key: bytes) -> str:
    if value:
        return str(value)
    blob = bytes(encrypted_value or b"")
    if not blob:
        return ""
    if blob.startswith((b"v10", b"v11")):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except Exception as exc:
            raise _EdgeCookieImportError("cryptography is required to decrypt Edge cookies.") from exc
        try:
            nonce = blob[3:15]
            ciphertext = blob[15:]
            return AESGCM(master_key).decrypt(nonce, ciphertext, None).decode("utf-8")
        except Exception as exc:
            raise _EdgeCookieImportError("Could not decrypt an Edge cookie value for this Windows user.") from exc
    if blob.startswith(b"v20"):
        raise _EdgeCookieImportError(
            "This Edge profile uses app-bound cookie encryption, which this importer cannot decrypt. "
            "Use auth-browser or manual setup instead."
        )

    try:
        import win32crypt

        plaintext = win32crypt.CryptUnprotectData(blob, None, None, None, 0)[1]
        return bytes(plaintext).decode("utf-8")
    except Exception as exc:
        raise _EdgeCookieImportError("Could not decrypt a legacy Edge cookie value.") from exc


def _whitelist_command(args: argparse.Namespace) -> int:
    sub = getattr(args, "whitelist_command", None) or "list"
    cfg = core.settings()
    core._ensure_state(cfg)
    names = core.load_whitelist(cfg)
    if sub == "list":
        return _print({"ok": True, "whitelist": sorted(names), "path": str(cfg.whitelist_file)})
    if sub == "add":
        names.add(args.screen_name.lstrip("@").lower())
        core.save_whitelist(names, cfg)
        return _print({"ok": True, "whitelist": sorted(names), "path": str(cfg.whitelist_file)})
    if sub == "remove":
        names.discard(args.screen_name.lstrip("@").lower())
        core.save_whitelist(names, cfg)
        return _print({"ok": True, "whitelist": sorted(names), "path": str(cfg.whitelist_file)})
    if sub == "import-mentioned-followers":
        candidates = core.mention_candidates(
            count=int(getattr(args, "count", 100) or 100),
            max_text_chars=80,
            cfg=cfg,
        )
        if not candidates.get("ok"):
            return _print(candidates)
        before = set(names)
        added = sorted(
            {
                str(item.get("username") or "").strip().lstrip("@").lower()
                for item in candidates.get("candidates", [])
                if item.get("followed_by") is True and item.get("username")
            }
        )
        names.update(added)
        core.save_whitelist(names, cfg)
        return _print(
            {
                "ok": True,
                "added": sorted(set(names) - before),
                "added_count": len(set(names) - before),
                "whitelist": sorted(names),
                "path": str(cfg.whitelist_file),
                "source": "recent_mentions_followed_by_true",
            }
        )
    print("usage: hermes lm-twitterer whitelist {list,add,remove,import-mentioned-followers}")
    return 2


def _cron_command(args: argparse.Namespace) -> int:
    sub = getattr(args, "cron_command", None)
    if sub != "install":
        print("usage: hermes lm-twitterer cron install [--post-schedule every 6h] [--reply-schedule every 1h]")
        return 2
    try:
        from cron.jobs import create_job, pause_job
    except Exception as exc:
        print(f"cron module unavailable: {exc}")
        return 1

    readiness = core.status()
    missing = []
    if not readiness.get("auth_token_set"):
        missing.append("LM_TWITTERER_AUTH_TOKEN")
    if not readiness.get("ct0_set"):
        missing.append("LM_TWITTERER_CT0")
    if not readiness.get("bot_screen_name_set"):
        missing.append("LM_TWITTERER_BOT_SCREEN_NAME")
    if not readiness.get("whitelist_count"):
        missing.append("reply whitelist")
    if missing and not bool(getattr(args, "force", False)):
        next_steps = []
        if "LM_TWITTERER_AUTH_TOKEN" in missing or "LM_TWITTERER_CT0" in missing:
            next_steps.append("hermes lm-twitterer auth-edge-direct --screen-name <name>")
        if "reply whitelist" in missing:
            next_steps.append("hermes lm-twitterer whitelist add <screen-name>")
            next_steps.append("hermes lm-twitterer replies --count 20")
        return _print(
            {
                "ok": False,
                "error": "Refusing to create live cron jobs before setup is complete.",
                "missing": missing,
                "next_steps": next_steps,
            }
        )

    scripts = _ensure_cron_scripts(
        post_topic=str(getattr(args, "post_topic", "") or ""),
        reply_count=int(getattr(args, "reply_count", 20) or 20),
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
    )
    preview = [
        {
            "name": "LM-twitterer post",
            "schedule": args.post_schedule,
            "script": scripts["post"],
            "script_path": scripts["post_path"],
            "no_agent": True,
            "deliver": args.deliver,
            "profile": args.profile,
            "provider": args.provider,
            "model": args.model,
        },
        {
            "name": "LM-twitterer reply mentions",
            "schedule": args.reply_schedule,
            "script": scripts["replies"],
            "script_path": scripts["replies_path"],
            "no_agent": True,
            "deliver": args.deliver,
            "profile": args.profile,
            "provider": args.provider,
            "model": args.model,
        },
    ]
    if bool(getattr(args, "dry_run", False)):
        return _print({"ok": True, "dry_run": True, "jobs": preview})

    jobs = []
    for spec in preview:
        job = create_job(
            prompt=spec["name"],
            schedule=spec["schedule"],
            name=spec["name"],
            deliver=spec["deliver"],
            script=spec["script"],
            no_agent=True,
            profile=spec["profile"],
        )
        if bool(getattr(args, "paused", False)):
            paused_job = pause_job(job["id"], reason="LM-twitterer installed paused pending explicit live approval")
            if paused_job:
                job = paused_job
        jobs.append(job)
    return _print(
        {
            "ok": True,
            "paused": bool(getattr(args, "paused", False)),
            "jobs": [
                {
                    "id": job["id"],
                    "name": job["name"],
                    "state": job.get("state"),
                    "enabled": job.get("enabled"),
                    "schedule": job["schedule_display"],
                    "next_run_at": job["next_run_at"],
                    "deliver": job["deliver"],
                    "script": job.get("script"),
                    "no_agent": job.get("no_agent"),
                    "paused_reason": job.get("paused_reason"),
                }
                for job in jobs
            ],
            "resume_commands": [
                f"hermes cron resume {job['id']}" for job in jobs if bool(getattr(args, "paused", False))
            ],
        }
    )


def _ensure_cron_scripts(
    *,
    post_topic: str = "",
    reply_count: int = 20,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, str]:
    from hermes_constants import get_hermes_home

    import hermes_cli

    hermes_home = Path(get_hermes_home())
    scripts_dir = hermes_home / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(hermes_cli.__file__).resolve().parent.parent
    python_exe = Path(sys.executable).resolve()
    post_script = scripts_dir / "lm-twitterer-post.py"
    replies_script = scripts_dir / "lm-twitterer-replies.py"

    _write_cron_wrapper(
        post_script,
        python_exe=python_exe,
        repo_root=repo_root,
        args=[
            "lm-twitterer",
            "post",
            "--live",
            *_model_args(provider, model),
            *([post_topic] if post_topic else []),
        ],
    )
    _write_cron_wrapper(
        replies_script,
        python_exe=python_exe,
        repo_root=repo_root,
        args=[
            "lm-twitterer",
            "replies",
            "--live",
            "--count",
            str(reply_count),
            *_model_args(provider, model),
        ],
    )
    return {
        "post": post_script.name,
        "replies": replies_script.name,
        "post_path": str(post_script),
        "replies_path": str(replies_script),
    }


def _write_cron_wrapper(
    path: Path,
    *,
    python_exe: Path,
    repo_root: Path,
    args: list[str],
) -> None:
    payload = {
        "python": str(python_exe),
        "repo_root": str(repo_root),
        "args": args,
        "preflight_args": ["lm-twitterer", "auth-check"] if "--live" in args else [],
    }
    content = f"""# Auto-generated by hermes lm-twitterer cron install.
import json
import os
import subprocess
import sys

PAYLOAD = {json.dumps(payload, ensure_ascii=True, indent=2)}

env = os.environ.copy()
env.setdefault("PYTHONIOENCODING", "utf-8")

def run_hermes(args, *, timeout):
    return subprocess.run(
        [PAYLOAD["python"], "-m", "hermes_cli.main", *args],
        cwd=PAYLOAD["repo_root"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )

if PAYLOAD.get("preflight_args"):
    preflight = run_hermes(PAYLOAD["preflight_args"], timeout=120)
    if preflight.returncode != 0:
        stderr = (preflight.stderr or "").strip()
        stdout = (preflight.stdout or "").strip()
        print("LM-twitterer preflight failed; live action was not run.")
        if stderr:
            print(stderr)
        if stdout:
            print(stdout)
        sys.exit(preflight.returncode)
    if os.environ.get("LM_TWITTERER_CRON_PREFLIGHT_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}:
        print("LM-twitterer preflight ok; live action skipped by LM_TWITTERER_CRON_PREFLIGHT_ONLY.")
        sys.exit(0)

cmd = [PAYLOAD["python"], "-m", "hermes_cli.main", *PAYLOAD["args"]]
result = run_hermes(PAYLOAD["args"], timeout=900)
stdout = (result.stdout or "").strip()
stderr = (result.stderr or "").strip()
if result.returncode != 0:
    if stderr:
        print(stderr)
    if stdout:
        print(stdout)
    sys.exit(result.returncode)
if stdout:
    print(stdout)
"""
    path.write_text(content, encoding="utf-8")


def _model_args(provider: str | None, model: str | None) -> list[str]:
    args: list[str] = []
    if provider:
        args.extend(["--provider", str(provider).strip()])
    if model:
        args.extend(["--model", str(model).strip()])
    return args


def _append_allowlist(raw: object, value: str) -> list[str]:
    values = [str(item).strip() for item in raw] if isinstance(raw, list) else []
    if "*" in values:
        return ["*"]
    normalized = {item.lower() for item in values}
    if value.lower() not in normalized:
        values.append(value)
    return values
