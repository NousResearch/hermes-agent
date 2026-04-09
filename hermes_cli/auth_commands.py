"""Credential-pool auth subcommands."""

from __future__ import annotations

import asyncio
from getpass import getpass
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from types import SimpleNamespace
import urllib.request
import uuid

from agent.credential_pool import (
    AUTH_TYPE_API_KEY,
    AUTH_TYPE_OAUTH,
    CUSTOM_POOL_PREFIX,
    SOURCE_MANUAL,
    STATUS_EXHAUSTED,
    STRATEGY_FILL_FIRST,
    STRATEGY_ROUND_ROBIN,
    STRATEGY_RANDOM,
    STRATEGY_LEAST_USED,
    PooledCredential,
    _exhausted_until,
    _normalize_custom_pool_name,
    get_pool_strategy,
    label_from_token,
    list_custom_pool_providers,
    load_pool,
)
import hermes_cli.auth as auth_mod
from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_constants import OPENROUTER_BASE_URL, get_hermes_home

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment]


# Providers that support OAuth login in addition to API keys.
_OAUTH_CAPABLE_PROVIDERS = {"anthropic", "nous", "openai-codex", "qwen-oauth"}


def _get_custom_provider_names() -> list:
    """Return list of (display_name, pool_key) tuples for custom_providers in config."""
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        return []
    custom_providers = config.get("custom_providers")
    if not isinstance(custom_providers, list):
        return []
    result = []
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        pool_key = f"{CUSTOM_POOL_PREFIX}{_normalize_custom_pool_name(name)}"
        result.append((name.strip(), pool_key))
    return result


def _resolve_custom_provider_input(raw: str) -> str | None:
    """If raw input matches a custom_providers entry name (case-insensitive), return its pool key."""
    normalized = (raw or "").strip().lower().replace(" ", "-")
    if not normalized:
        return None
    # Direct match on 'custom:name' format
    if normalized.startswith(CUSTOM_POOL_PREFIX):
        return normalized
    for display_name, pool_key in _get_custom_provider_names():
        if _normalize_custom_pool_name(display_name) == normalized:
            return pool_key
    return None


def _normalize_provider(provider: str) -> str:
    normalized = (provider or "").strip().lower()
    if normalized in {"or", "open-router"}:
        return "openrouter"
    # Check if it matches a custom provider name
    custom_key = _resolve_custom_provider_input(normalized)
    if custom_key:
        return custom_key
    return normalized


def _provider_base_url(provider: str) -> str:
    if provider == "openrouter":
        return OPENROUTER_BASE_URL
    if provider.startswith(CUSTOM_POOL_PREFIX):
        from agent.credential_pool import _get_custom_provider_config

        cp_config = _get_custom_provider_config(provider)
        if cp_config:
            return str(cp_config.get("base_url") or "").strip()
        return ""
    pconfig = PROVIDER_REGISTRY.get(provider)
    return pconfig.inference_base_url if pconfig else ""


def _oauth_default_label(provider: str, count: int) -> str:
    return f"{provider}-oauth-{count}"


def _api_key_default_label(count: int) -> str:
    return f"api-key-{count}"


def _looks_like_jwt(token: str) -> bool:
    return isinstance(token, str) and token.count(".") == 2


def _display_source(source: str) -> str:
    return source.split(":", 1)[1] if source.startswith("manual:") else source


def _format_exhausted_status(entry) -> str:
    if entry.last_status != STATUS_EXHAUSTED:
        return ""
    reason = getattr(entry, "last_error_reason", None)
    reason_text = f" {reason}" if isinstance(reason, str) and reason.strip() else ""
    code = f" ({entry.last_error_code})" if entry.last_error_code else ""
    exhausted_until = _exhausted_until(entry)
    if exhausted_until is None:
        return f" exhausted{reason_text}{code}"
    remaining = max(0, int(math.ceil(exhausted_until - time.time())))
    if remaining <= 0:
        return f" exhausted{reason_text}{code} (ready to retry)"
    minutes, seconds = divmod(remaining, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        wait = f"{days}d {hours}h"
    elif hours:
        wait = f"{hours}h {minutes}m"
    elif minutes:
        wait = f"{minutes}m {seconds}s"
    else:
        wait = f"{seconds}s"
    return f" exhausted{reason_text}{code} ({wait} left)"


def auth_add_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", ""))
    if provider not in PROVIDER_REGISTRY and provider != "openrouter" and not provider.startswith(CUSTOM_POOL_PREFIX):
        raise SystemExit(f"Unknown provider: {provider}")

    requested_type = str(getattr(args, "auth_type", "") or "").strip().lower()
    if requested_type in {AUTH_TYPE_API_KEY, "api-key"}:
        requested_type = AUTH_TYPE_API_KEY
    if not requested_type:
        if provider.startswith(CUSTOM_POOL_PREFIX):
            requested_type = AUTH_TYPE_API_KEY
        else:
            requested_type = AUTH_TYPE_OAUTH if provider in {"anthropic", "nous", "openai-codex", "chatgpt-web", "qwen-oauth"} else AUTH_TYPE_API_KEY

    pool = load_pool(provider)

    if requested_type == AUTH_TYPE_API_KEY:
        token = (getattr(args, "api_key", None) or "").strip()
        if provider == "chatgpt-web":
            token_mode = str(getattr(args, "token_mode", "") or "").strip().lower()
            if not token:
                if token_mode == "session_token":
                    token = getpass("Paste your ChatGPT Web session token: ").strip()
                elif token_mode == "access_token":
                    token = getpass("Paste your ChatGPT Web API key or access token: ").strip()
                else:
                    token = getpass("Paste your ChatGPT Web access token or session token: ").strip()
            if not token:
                raise SystemExit("No ChatGPT Web token provided.")
            default_label = _api_key_default_label(len(pool.entries()) + 1)
            label = (getattr(args, "label", None) or "").strip()
            if not label:
                label = input(f"Label (optional, default: {default_label}): ").strip() or default_label

            source = SOURCE_MANUAL
            access_token = token
            extra = {}
            should_exchange_session = (
                token_mode == "session_token"
                or (token_mode not in {"access_token", "session_token"} and not _looks_like_jwt(token))
            )
            if should_exchange_session:
                from hermes_cli.chatgpt_web import _fetch_chatgpt_web_access_token_from_session

                try:
                    access_token = _fetch_chatgpt_web_access_token_from_session(token)
                except Exception as exc:
                    raise SystemExit(f"Could not exchange ChatGPT Web session token: {exc}") from exc
                source = f"{SOURCE_MANUAL}:session_token"
                extra["session_token"] = token

            entry = PooledCredential(
                provider=provider,
                id=uuid.uuid4().hex[:6],
                label=label,
                auth_type=AUTH_TYPE_API_KEY,
                priority=0,
                source=source,
                access_token=access_token,
                base_url=_provider_base_url(provider),
                extra=extra,
            )
            pool.add_entry(entry)
            print(f'Added {provider} credential #{len(pool.entries())}: "{label}"')
            return

        if not token:
            token = getpass("Paste your API key: ").strip()
        if not token:
            raise SystemExit("No API key provided.")
        default_label = _api_key_default_label(len(pool.entries()) + 1)
        label = (getattr(args, "label", None) or "").strip()
        if not label:
            label = input(f"Label (optional, default: {default_label}): ").strip() or default_label
        entry = PooledCredential(
            provider=provider,
            id=uuid.uuid4().hex[:6],
            label=label,
            auth_type=AUTH_TYPE_API_KEY,
            priority=0,
            source=SOURCE_MANUAL,
            access_token=token,
            base_url=_provider_base_url(provider),
        )
        pool.add_entry(entry)
        print(f'Added {provider} credential #{len(pool.entries())}: "{label}"')
        return

    if provider == "anthropic":
        from agent import anthropic_adapter as anthropic_mod

        creds = anthropic_mod.run_hermes_oauth_login_pure()
        if not creds:
            raise SystemExit("Anthropic OAuth login did not return credentials.")
        label = (getattr(args, "label", None) or "").strip() or label_from_token(
            creds["access_token"],
            _oauth_default_label(provider, len(pool.entries()) + 1),
        )
        entry = PooledCredential(
            provider=provider,
            id=uuid.uuid4().hex[:6],
            label=label,
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=f"{SOURCE_MANUAL}:hermes_pkce",
            access_token=creds["access_token"],
            refresh_token=creds.get("refresh_token"),
            expires_at_ms=creds.get("expires_at_ms"),
            base_url=_provider_base_url(provider),
        )
        pool.add_entry(entry)
        print(f'Added {provider} OAuth credential #{len(pool.entries())}: "{entry.label}"')
        return

    if provider == "nous":
        creds = auth_mod._nous_device_code_login(
            portal_base_url=getattr(args, "portal_url", None),
            inference_base_url=getattr(args, "inference_url", None),
            client_id=getattr(args, "client_id", None),
            scope=getattr(args, "scope", None),
            open_browser=not getattr(args, "no_browser", False),
            timeout_seconds=getattr(args, "timeout", None) or 15.0,
            insecure=bool(getattr(args, "insecure", False)),
            ca_bundle=getattr(args, "ca_bundle", None),
            min_key_ttl_seconds=max(60, int(getattr(args, "min_key_ttl_seconds", 5 * 60))),
        )
        label = (getattr(args, "label", None) or "").strip() or label_from_token(
            creds.get("access_token", ""),
            _oauth_default_label(provider, len(pool.entries()) + 1),
        )
        entry = PooledCredential.from_dict(provider, {
            **creds,
            "label": label,
            "auth_type": AUTH_TYPE_OAUTH,
            "source": f"{SOURCE_MANUAL}:device_code",
            "base_url": creds.get("inference_base_url"),
        })
        pool.add_entry(entry)
        print(f'Added {provider} OAuth credential #{len(pool.entries())}: "{entry.label}"')
        return

    if provider == "openai-codex":
        creds = auth_mod._codex_device_code_login()
        label = (getattr(args, "label", None) or "").strip() or label_from_token(
            creds["tokens"]["access_token"],
            _oauth_default_label(provider, len(pool.entries()) + 1),
        )
        entry = PooledCredential(
            provider=provider,
            id=uuid.uuid4().hex[:6],
            label=label,
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=f"{SOURCE_MANUAL}:device_code",
            access_token=creds["tokens"]["access_token"],
            refresh_token=creds["tokens"].get("refresh_token"),
            base_url=creds.get("base_url"),
            last_refresh=creds.get("last_refresh"),
        )
        pool.add_entry(entry)
        print(f'Added {provider} OAuth credential #{len(pool.entries())}: "{entry.label}"')
        return

    if provider == "chatgpt-web":
        creds = auth_mod._codex_device_code_login()
        label = (getattr(args, "label", None) or "").strip() or label_from_token(
            creds["tokens"]["access_token"],
            _oauth_default_label(provider, len(pool.entries()) + 1),
        )
        entry = PooledCredential(
            provider=provider,
            id=uuid.uuid4().hex[:6],
            label=label,
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=f"{SOURCE_MANUAL}:device_code",
            access_token=creds["tokens"]["access_token"],
            refresh_token=creds["tokens"].get("refresh_token"),
            base_url=_provider_base_url(provider),
            last_refresh=creds.get("last_refresh"),
        )
        pool.add_entry(entry)
        print(f'Added {provider} OAuth credential #{len(pool.entries())}: "{entry.label}"')
        return

    if provider == "qwen-oauth":
        creds = auth_mod.resolve_qwen_runtime_credentials(refresh_if_expiring=False)
        label = (getattr(args, "label", None) or "").strip() or label_from_token(
            creds["api_key"],
            _oauth_default_label(provider, len(pool.entries()) + 1),
        )
        entry = PooledCredential(
            provider=provider,
            id=uuid.uuid4().hex[:6],
            label=label,
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=f"{SOURCE_MANUAL}:qwen_cli",
            access_token=creds["api_key"],
            base_url=creds.get("base_url"),
        )
        pool.add_entry(entry)
        print(f'Added {provider} OAuth credential #{len(pool.entries())}: "{entry.label}"')
        return

    raise SystemExit(f"`hermes auth add {provider}` is not implemented for auth type {requested_type} yet.")


def auth_list_command(args) -> None:
    provider_filter = _normalize_provider(getattr(args, "provider", "") or "")
    if provider_filter:
        providers = [provider_filter]
    else:
        providers = sorted({
            *PROVIDER_REGISTRY.keys(),
            "openrouter",
            *list_custom_pool_providers(),
        })
    for provider in providers:
        pool = load_pool(provider)
        entries = pool.entries()
        if not entries:
            continue
        current = pool.peek()
        print(f"{provider} ({len(entries)} credentials):")
        for idx, entry in enumerate(entries, start=1):
            marker = "  "
            if current is not None and entry.id == current.id:
                marker = "← "
            status = _format_exhausted_status(entry)
            source = _display_source(entry.source)
            print(f"  #{idx}  {entry.label:<20} {entry.auth_type:<7} {source}{status} {marker}".rstrip())
        print()


def auth_remove_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", ""))
    target = getattr(args, "target", None)
    if target is None:
        target = getattr(args, "index", None)
    pool = load_pool(provider)
    index, matched, error = pool.resolve_target(target)
    if matched is None or index is None:
        raise SystemExit(f"{error} Provider: {provider}.")
    removed = pool.remove_index(index)
    if removed is None:
        raise SystemExit(f'No credential matching "{target}" for provider {provider}.')
    print(f"Removed {provider} credential #{index} ({removed.label})")

    # If this was an env-seeded credential, also clear the env var from .env
    # so it doesn't get re-seeded on the next load_pool() call.
    if removed.source.startswith("env:"):
        env_var = removed.source[len("env:"):]
        if env_var:
            from hermes_cli.config import remove_env_value
            cleared = remove_env_value(env_var)
            if cleared:
                print(f"Cleared {env_var} from .env")

    # If this was a singleton-seeded credential (OAuth device_code, hermes_pkce),
    # clear the underlying auth store / credential file so it doesn't get
    # re-seeded on the next load_pool() call.
    elif removed.source == "device_code" and provider in ("openai-codex", "nous"):
        from hermes_cli.auth import (
            _load_auth_store, _save_auth_store, _auth_store_lock,
        )
        with _auth_store_lock():
            auth_store = _load_auth_store()
            providers_dict = auth_store.get("providers")
            if isinstance(providers_dict, dict) and provider in providers_dict:
                del providers_dict[provider]
                _save_auth_store(auth_store)
                print(f"Cleared {provider} OAuth tokens from auth store")

    elif removed.source == "hermes_pkce" and provider == "anthropic":
        from hermes_constants import get_hermes_home
        oauth_file = get_hermes_home() / ".anthropic_oauth.json"
        if oauth_file.exists():
            oauth_file.unlink()
            print("Cleared Hermes Anthropic OAuth credentials")

    elif removed.source == "claude_code" and provider == "anthropic":
        print("Note: Claude Code credentials live in ~/.claude/.credentials.json")
        print("      Remove them manually if you want to deauthorize Claude Code.")


def auth_reset_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", ""))
    pool = load_pool(provider)
    count = pool.reset_statuses()
    print(f"Reset status on {count} {provider} credentials")


def _is_termux() -> bool:
    prefix = os.getenv("PREFIX", "")
    return bool(os.getenv("TERMUX_VERSION") or "com.termux/files/usr" in prefix)


def _termux_x11_android_app_installed() -> bool:
    pm_command = "/system/bin/pm"
    if not Path(pm_command).exists():
        return False
    result = subprocess.run(
        [pm_command, "list", "packages", "com.termux.x11"],
        capture_output=True,
        text=True,
        check=False,
        env={k: v for k, v in os.environ.items() if k != "LD_PRELOAD"},
    )
    return "package:com.termux.x11" in (result.stdout or "")


def _find_termux_x11_command() -> str | None:
    return shutil.which("termux-x11")


def _find_chromium_browser_command() -> str | None:
    return shutil.which("chromium-browser") or shutil.which("chromium")


def _write_chatgpt_web_browser_launch_scripts(
    base_dir: Path,
    termux_x11_command: str,
    browser_command: str,
    debug_port: int,
) -> tuple[Path, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    startup_script = base_dir / "startup.sh"
    launcher_script = base_dir / "launch.sh"

    startup_script.write_text(
        "#!/data/data/com.termux/files/usr/bin/bash\n"
        "set -euo pipefail\n\n"
        f"BASE_DIR={shlex.quote(str(base_dir))}\n"
        "PROFILE_DIR=\"$BASE_DIR/profile\"\n"
        "LOG_DIR=\"$BASE_DIR/logs\"\n"
        "mkdir -p \"$PROFILE_DIR\" \"$LOG_DIR\"\n\n"
        "export DISPLAY=\"${DISPLAY:-:0}\"\n"
        "export XDG_RUNTIME_DIR=\"${TMPDIR:-$PREFIX/tmp}\"\n"
        f"exec {shlex.quote(browser_command)} \\\n"
        "  --no-sandbox \\\n"
        "  --password-store=basic \\\n"
        "  --user-data-dir=\"$PROFILE_DIR\" \\\n"
        "  --remote-debugging-address=127.0.0.1 \\\n"
        f"  --remote-debugging-port={int(debug_port)} \\\n"
        "  --no-first-run \\\n"
        "  --no-default-browser-check \\\n"
        "  --disable-fre \\\n"
        "  --disable-crash-reporter \\\n"
        "  --disable-session-crashed-bubble \\\n"
        "  --window-size=1280,900 \\\n"
        "  https://chatgpt.com \\\n"
        "  >>\"$LOG_DIR/chromium.log\" 2>&1\n",
        encoding="utf-8",
    )

    launcher_script.write_text(
        "#!/data/data/com.termux/files/usr/bin/bash\n"
        "set -euo pipefail\n\n"
        f"BASE_DIR={shlex.quote(str(base_dir))}\n"
        "DISPLAY_FILE=\"$BASE_DIR/display\"\n"
        "mkdir -p \"$BASE_DIR\"\n"
        "rm -f \"$DISPLAY_FILE\"\n\n"
        "exec 3<>\"$DISPLAY_FILE\"\n"
        f"exec {shlex.quote(termux_x11_command)} -displayfd 3 -noreset -xstartup {shlex.quote(str(startup_script))}\n",
        encoding="utf-8",
    )

    startup_script.chmod(0o755)
    launcher_script.chmod(0o755)
    return launcher_script, startup_script


def _launch_chatgpt_web_browser(launcher_script: Path, base_dir: Path):
    log_path = base_dir / "termux-x11.log"
    with log_path.open("ab") as handle:
        return subprocess.Popen(
            [str(launcher_script)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(base_dir),
            start_new_session=True,
        )


def _wait_for_debugger(debug_base: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{debug_base}/json/version", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise SystemExit(f"Timed out waiting for Chromium DevTools at {debug_base}: {last_error}")


async def _get_chatgpt_web_session_token(debug_base: str) -> str | None:
    if websockets is None:
        raise SystemExit("Python package 'websockets' is required for browser auth.")

    with urllib.request.urlopen(f"{debug_base}/json/list", timeout=5) as response:
        pages = json.load(response)

    page = None
    for item in pages:
        if item.get("type") == "page" and "chatgpt.com" in str(item.get("url") or ""):
            page = item
            break
    if page is None:
        return None

    ws_url = str(page.get("webSocketDebuggerUrl") or "").strip()
    if not ws_url:
        return None

    async with websockets.connect(ws_url, max_size=20_000_000) as ws:
        next_id = 1

        async def send(method: str, params: dict | None = None):
            nonlocal next_id
            payload = {"id": next_id, "method": method}
            if params is not None:
                payload["params"] = params
            await ws.send(json.dumps(payload))
            my_id = next_id
            next_id += 1
            while True:
                message = json.loads(await ws.recv())
                if message.get("id") == my_id:
                    return message

        await send("Network.enable")
        result = await send("Network.getCookies", {"urls": ["https://chatgpt.com/"]})
        cookies = result.get("result", {}).get("cookies", [])
        for cookie in cookies:
            if cookie.get("name") == "__Secure-next-auth.session-token":
                value = str(cookie.get("value") or "").strip()
                if value:
                    return value
    return None


def _wait_for_chatgpt_web_session_token(
    debug_base: str,
    *,
    timeout_seconds: int = 15 * 60,
    poll_seconds: int = 5,
) -> str | None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            token = asyncio.run(_get_chatgpt_web_session_token(debug_base))
        except Exception:
            token = None
        if token:
            return token
        print("waiting for ChatGPT login in Termux browser...")
        time.sleep(poll_seconds)
    return None


def _terminate_process(proc, timeout: float = 5.0) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def auth_browser_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", "") or "chatgpt-web")
    if provider != "chatgpt-web":
        raise SystemExit("Browser auth currently supports only chatgpt-web.")
    if not _is_termux():
        raise SystemExit("Browser auth currently supports only Termux/Android.")
    if websockets is None:
        raise SystemExit("Python package 'websockets' is required for browser auth.")
    if not _termux_x11_android_app_installed():
        raise SystemExit("Termux:X11 Android app (com.termux.x11) is not installed.")

    termux_x11_command = _find_termux_x11_command()
    if not termux_x11_command:
        raise SystemExit("termux-x11 command not found. Install `termux-x11-nightly`.")

    browser_command = _find_chromium_browser_command()
    if not browser_command:
        raise SystemExit("Chromium command not found. Install `chromium`.")

    label = (getattr(args, "label", None) or "termux-x11-browser").strip() or "termux-x11-browser"
    timeout_seconds = max(30, int(getattr(args, "timeout", None) or 15 * 60))
    debug_port = max(1024, int(getattr(args, "debug_port", None) or 9222))
    keep_open = bool(getattr(args, "keep_open", False))
    debug_base = f"http://127.0.0.1:{debug_port}"
    base_dir = get_hermes_home() / "chatgpt-web-browser"

    shutil.rmtree(base_dir, ignore_errors=True)
    launcher_script, _startup_script = _write_chatgpt_web_browser_launch_scripts(
        base_dir,
        termux_x11_command,
        browser_command,
        debug_port,
    )
    proc = _launch_chatgpt_web_browser(launcher_script, base_dir)
    print("Started local Termux browser for ChatGPT Web auth.")
    print("Open the Termux:X11 Android app manually, then finish logging into ChatGPT in Chromium.")

    try:
        _wait_for_debugger(debug_base, timeout=min(60.0, float(timeout_seconds)))
        session_token = _wait_for_chatgpt_web_session_token(
            debug_base,
            timeout_seconds=timeout_seconds,
        )
        if not session_token:
            raise SystemExit("Timed out waiting for __Secure-next-auth.session-token from Chromium.")

        auth_add_command(SimpleNamespace(
            provider="chatgpt-web",
            auth_type="api-key",
            api_key=session_token,
            label=label,
            token_mode="session_token",
            portal_url=None,
            inference_url=None,
            client_id=None,
            scope=None,
            no_browser=False,
            timeout=None,
            insecure=False,
            ca_bundle=None,
        ))
        print("Stored chatgpt-web credential from Termux browser.")
        print(f'Added it to the credential pool as "{label}".')
        print("Verify with: hermes auth list")
    finally:
        if not keep_open:
            _terminate_process(proc)
            shutil.rmtree(base_dir, ignore_errors=True)


def _interactive_auth() -> None:
    """Interactive credential pool management when `hermes auth` is called bare."""
    # Show current pool status first
    print("Credential Pool Status")
    print("=" * 50)

    auth_list_command(SimpleNamespace(provider=None))
    print()

    # Main menu
    choices = [
        "Add a credential",
        "Remove a credential",
        "Reset cooldowns for a provider",
        "Set rotation strategy for a provider",
        "Exit",
    ]
    print("What would you like to do?")
    for i, choice in enumerate(choices, 1):
        print(f"  {i}. {choice}")

    try:
        raw = input("\nChoice: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not raw or raw == str(len(choices)):
        return

    if raw == "1":
        _interactive_add()
    elif raw == "2":
        _interactive_remove()
    elif raw == "3":
        _interactive_reset()
    elif raw == "4":
        _interactive_strategy()


def _pick_provider(prompt: str = "Provider") -> str:
    """Prompt for a provider name with auto-complete hints."""
    known = sorted(set(list(PROVIDER_REGISTRY.keys()) + ["openrouter"]))
    custom_names = _get_custom_provider_names()
    if custom_names:
        custom_display = [name for name, _key in custom_names]
        print(f"\nKnown providers: {', '.join(known)}")
        print(f"Custom endpoints: {', '.join(custom_display)}")
    else:
        print(f"\nKnown providers: {', '.join(known)}")
    try:
        raw = input(f"{prompt}: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit()
    return _normalize_provider(raw)


def _interactive_add() -> None:
    provider = _pick_provider("Provider to add credential for")
    if provider not in PROVIDER_REGISTRY and provider != "openrouter" and not provider.startswith(CUSTOM_POOL_PREFIX):
        raise SystemExit(f"Unknown provider: {provider}")

    # For OAuth-capable providers, ask which type
    token_mode = None
    if provider == "chatgpt-web":
        print(f"\n{provider} supports API keys/access tokens, OAuth login, session tokens, and local browser bootstrap.")
        print("  1. API key / access token")
        print("  2. OAuth login (authenticate via browser/device code)")
        print("  3. Session token (paste __Secure-next-auth.session-token)")
        print("  4. Termux browser bootstrap (launch Chromium in Termux:X11 and capture the session automatically)")
        try:
            type_choice = input("Type [1/2/3/4]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if type_choice == "2":
            auth_type = "oauth"
        elif type_choice == "3":
            auth_type = "api_key"
            token_mode = "session_token"
        elif type_choice == "4":
            auth_type = "browser"
        else:
            auth_type = "api_key"
            token_mode = "access_token"
    elif provider in _OAUTH_CAPABLE_PROVIDERS:
        print(f"\n{provider} supports both API keys and OAuth login.")
        print("  1. API key (paste a key from the provider dashboard)")
        print("  2. OAuth login (authenticate via browser)")
        try:
            type_choice = input("Type [1/2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if type_choice == "2":
            auth_type = "oauth"
        else:
            auth_type = "api_key"
    else:
        auth_type = "api_key"

    label = None
    try:
        typed_label = input("Label / account name (optional): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if typed_label:
        label = typed_label

    if auth_type == "browser":
        auth_browser_command(SimpleNamespace(
            provider=provider,
            label=label,
            timeout=None,
            debug_port=None,
            keep_open=False,
        ))
        return

    auth_add_command(SimpleNamespace(
        provider=provider, auth_type=auth_type, label=label, api_key=None,
        portal_url=None, inference_url=None, client_id=None, scope=None,
        no_browser=False, timeout=None, insecure=False, ca_bundle=None,
        token_mode=token_mode,
    ))


def _interactive_remove() -> None:
    provider = _pick_provider("Provider to remove credential from")
    pool = load_pool(provider)
    if not pool.has_credentials():
        print(f"No credentials for {provider}.")
        return

    # Show entries with indices
    for i, e in enumerate(pool.entries(), 1):
        exhausted = _format_exhausted_status(e)
        print(f"  #{i}  {e.label:25s} {e.auth_type:10s} {e.source}{exhausted} [id:{e.id}]")

    try:
        raw = input("Remove #, id, or label (blank to cancel): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not raw:
        return

    auth_remove_command(SimpleNamespace(provider=provider, target=raw))


def _interactive_reset() -> None:
    provider = _pick_provider("Provider to reset cooldowns for")

    auth_reset_command(SimpleNamespace(provider=provider))


def _interactive_strategy() -> None:
    provider = _pick_provider("Provider to set strategy for")
    current = get_pool_strategy(provider)
    strategies = [STRATEGY_FILL_FIRST, STRATEGY_ROUND_ROBIN, STRATEGY_LEAST_USED, STRATEGY_RANDOM]

    print(f"\nCurrent strategy for {provider}: {current}")
    print()
    descriptions = {
        STRATEGY_FILL_FIRST: "Use first key until exhausted, then next",
        STRATEGY_ROUND_ROBIN: "Cycle through keys evenly",
        STRATEGY_LEAST_USED: "Always pick the least-used key",
        STRATEGY_RANDOM: "Random selection",
    }
    for i, s in enumerate(strategies, 1):
        marker = " ←" if s == current else ""
        print(f"  {i}. {s:15s} — {descriptions.get(s, '')}{marker}")

    try:
        raw = input("\nStrategy [1-4]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not raw:
        return

    try:
        idx = int(raw) - 1
        strategy = strategies[idx]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return

    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    pool_strategies = cfg.get("credential_pool_strategies") or {}
    if not isinstance(pool_strategies, dict):
        pool_strategies = {}
    pool_strategies[provider] = strategy
    cfg["credential_pool_strategies"] = pool_strategies
    save_config(cfg)
    print(f"Set {provider} strategy to: {strategy}")


def auth_command(args) -> None:
    action = getattr(args, "auth_action", "")
    if action == "add":
        auth_add_command(args)
        return
    if action == "list":
        auth_list_command(args)
        return
    if action == "remove":
        auth_remove_command(args)
        return
    if action == "reset":
        auth_reset_command(args)
        return
    if action == "browser":
        auth_browser_command(args)
        return
    # No subcommand — launch interactive mode
    _interactive_auth()
