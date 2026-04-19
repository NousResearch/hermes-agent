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
import sys
import time
from types import SimpleNamespace
from typing import Any
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
except Exception:
    websockets = None  # type: ignore[assignment]


# Providers that support OAuth login in addition to API keys.
_OAUTH_CAPABLE_PROVIDERS = {
    "anthropic",
    "nous",
    "openai-codex",
    "chatgpt-web",
    "qwen-oauth",
    "google-gemini-cli",
    "minimax-oauth",
}


def _get_custom_provider_names() -> list:
    """Return list of (display_name, pool_key, provider_key) tuples."""
    try:
        from hermes_cli.config import get_compatible_custom_providers, load_config

        config = load_config()
    except Exception:
        return []
    result = []
    for entry in get_compatible_custom_providers(config):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        pool_key = f"{CUSTOM_POOL_PREFIX}{_normalize_custom_pool_name(name)}"
        provider_key = str(entry.get("provider_key", "") or "").strip()
        result.append((name.strip(), pool_key, provider_key))
    return result


def _resolve_custom_provider_input(raw: str) -> str | None:
    """If raw input matches a custom_providers entry name (case-insensitive), return its pool key."""
    normalized = (raw or "").strip().lower().replace(" ", "-")
    if not normalized:
        return None
    # Direct match on 'custom:name' format
    if normalized.startswith(CUSTOM_POOL_PREFIX):
        return normalized
    for display_name, pool_key, provider_key in _get_custom_provider_names():
        if _normalize_custom_pool_name(display_name) == normalized:
            return pool_key
        if provider_key and provider_key.strip().lower() == normalized:
            return pool_key
    return None


def _normalize_provider(provider: str) -> str:
    normalized = (provider or "").strip().lower()
    if normalized in {"or", "open-router"}:
        return "openrouter"
    if normalized in {"grok-oauth", "xai-oauth", "x-ai-oauth", "xai-grok-oauth"}:
        return "xai-oauth"
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


def _classify_exhausted_status(entry) -> tuple[str, bool]:
    code = getattr(entry, "last_error_code", None)
    reason = str(getattr(entry, "last_error_reason", "") or "").strip().lower()
    message = str(getattr(entry, "last_error_message", "") or "").strip().lower()

    if code == 429 or any(token in reason for token in ("rate_limit", "usage_limit", "quota", "exhausted")) or any(
        token in message for token in ("rate limit", "usage limit", "quota", "too many requests")
    ):
        return "rate-limited", True

    if code in {401, 403} or any(token in reason for token in ("invalid_token", "invalid_grant", "unauthorized", "forbidden", "auth")) or any(
        token in message for token in ("unauthorized", "forbidden", "expired", "revoked", "invalid token", "authentication")
    ):
        return "auth failed", False

    return "exhausted", True



def _format_exhausted_status(entry) -> str:
    if entry.last_status != STATUS_EXHAUSTED:
        return ""
    label, show_retry_window = _classify_exhausted_status(entry)
    reason = getattr(entry, "last_error_reason", None)
    reason_text = f" {reason}" if isinstance(reason, str) and reason.strip() else ""
    code = f" ({entry.last_error_code})" if entry.last_error_code else ""
    if not show_retry_window:
        return f" {label}{reason_text}{code} (re-auth may be required)"
    exhausted_until = _exhausted_until(entry)
    if exhausted_until is None:
        return f" {label}{reason_text}{code}"
    remaining = max(0, int(math.ceil(exhausted_until - time.time())))
    if remaining <= 0:
        return f" {label}{reason_text}{code} (ready to retry)"
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
    return f" {label}{reason_text}{code} ({wait} left)"


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
            requested_type = AUTH_TYPE_OAUTH if provider in _OAUTH_CAPABLE_PROVIDERS else AUTH_TYPE_API_KEY

    pool = load_pool(provider)

    # Clear ALL suppressions for this provider — re-adding a credential is
    # a strong signal the user wants auth re-enabled.  This covers env:*
    # (shell-exported vars), gh_cli (copilot), claude_code, qwen-cli,
    # device_code (codex), etc.  One consistent re-engagement pattern.
    # Matches the Codex device_code re-link pattern that predates this.
    if not provider.startswith(CUSTOM_POOL_PREFIX):
        try:
            from hermes_cli.auth import (
                _load_auth_store,
                unsuppress_credential_source,
            )
            suppressed = _load_auth_store().get("suppressed_sources", {})
            for src in list(suppressed.get(provider, []) or []):
                unsuppress_credential_source(provider, src)
        except Exception:
            pass

    if requested_type == AUTH_TYPE_API_KEY:
        token = (getattr(args, "api_key", None) or "").strip()
        if provider == "chatgpt-web":
            token_mode = str(getattr(args, "token_mode", "") or "").strip().lower()
            cookie_header = str(getattr(args, "cookie_header", "") or "").strip()
            browser_cookies = getattr(args, "browser_cookies", None)
            device_id = str(getattr(args, "device_id", "") or "").strip()
            user_agent = str(getattr(args, "user_agent", "") or "").strip()
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
                    access_token = _fetch_chatgpt_web_access_token_from_session(
                        token,
                        cookie_header=cookie_header,
                        browser_cookies=browser_cookies,
                        device_id=device_id,
                        user_agent=user_agent,
                    )
                except Exception as exc:
                    raise SystemExit(f"Could not exchange ChatGPT Web session token: {exc}") from exc
                source = f"{SOURCE_MANUAL}:session_token"
                extra["session_token"] = token
            if cookie_header:
                extra["cookie_header"] = cookie_header
            if browser_cookies:
                extra["browser_cookies"] = browser_cookies
            if device_id:
                extra["device_id"] = device_id
            if user_agent:
                extra["user_agent"] = user_agent

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
            if sys.stdin.isatty():
                label = input(f"Label (optional, default: {default_label}): ").strip() or default_label
            else:
                label = default_label
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
        # Codex-style auto-import: if a shared Nous credential lives at
        # <hermes-root>/shared/nous_auth.json (written by any previous
        # successful login), offer to import it instead of running the
        # full device-code flow. This makes `hermes --profile <name>
        # auth add nous --type oauth` a one-tap operation for users who
        # run multiple profiles.
        shared = auth_mod._read_shared_nous_state()
        if shared:
            try:
                path = auth_mod._nous_shared_store_path()
            except RuntimeError:
                path = None
            print()
            if path:
                print(f"Found existing Nous OAuth credentials at {path}")
            else:
                print("Found existing shared Nous OAuth credentials")
            try:
                do_import = input("Import these credentials? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                do_import = "y"
            if do_import in {"", "y", "yes"}:
                print("Rehydrating Nous session from shared credentials...")
                rehydrated = auth_mod._try_import_shared_nous_state(
                    timeout_seconds=getattr(args, "timeout", None) or 15.0,
                    min_key_ttl_seconds=max(
                        60, int(getattr(args, "min_key_ttl_seconds", 5 * 60))
                    ),
                )
                if rehydrated is not None:
                    custom_label = (getattr(args, "label", None) or "").strip() or None
                    entry = auth_mod.persist_nous_credentials(rehydrated, label=custom_label)
                    shown_label = entry.label if entry is not None else label_from_token(
                        rehydrated.get("access_token", ""), _oauth_default_label(provider, 1),
                    )
                    print(f'Imported {provider} OAuth credentials: "{shown_label}"')
                    return
                # Rehydrate failed (expired refresh_token, portal down, etc.)
                # — fall through to device-code flow.
                print("Could not refresh shared credentials — falling back to device-code login.")

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
        # Honor `--label <name>` so nous matches other providers' UX.  The
        # helper embeds this into providers.nous so that label_from_token
        # doesn't overwrite it on every subsequent load_pool("nous").
        custom_label = (getattr(args, "label", None) or "").strip() or None
        entry = auth_mod.persist_nous_credentials(creds, label=custom_label)
        shown_label = entry.label if entry is not None else label_from_token(
            creds.get("access_token", ""), _oauth_default_label(provider, 1),
        )
        print(f'Saved {provider} OAuth device-code credentials: "{shown_label}"')
        return

    if provider == "openai-codex":
        # Clear any existing suppression marker so a re-link after `hermes auth
        # remove openai-codex` works without the new tokens being skipped.
        auth_mod.unsuppress_credential_source(provider, "device_code")
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

    if provider == "google-gemini-cli":
        from agent.google_oauth import run_gemini_oauth_login_pure

        creds = run_gemini_oauth_login_pure()
        label = (getattr(args, "label", None) or "").strip() or (
            creds.get("email") or _oauth_default_label(provider, len(pool.entries()) + 1)
        )
        entry = PooledCredential(
            provider=provider,
            id=uuid.uuid4().hex[:6],
            label=label,
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=f"{SOURCE_MANUAL}:google_pkce",
            access_token=creds["access_token"],
            refresh_token=creds.get("refresh_token"),
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

    if provider == "minimax-oauth":
        creds = auth_mod._minimax_oauth_login(
            open_browser=not getattr(args, "no_browser", False),
            timeout_seconds=getattr(args, "timeout", None) or 15.0,
        )
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
            source=f"{SOURCE_MANUAL}:minimax_oauth",
            access_token=creds["access_token"],
            refresh_token=creds.get("refresh_token"),
            base_url=creds.get("inference_base_url"),
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

    # Unified removal dispatch.  Every credential source Hermes reads from
    # (env vars, external OAuth files, auth.json blocks, custom config)
    # has a RemovalStep registered in agent.credential_sources.  The step
    # handles its source-specific cleanup and we centralise suppression +
    # user-facing output here so every source behaves identically from
    # the user's perspective.
    from agent.credential_sources import find_removal_step
    from hermes_cli.auth import suppress_credential_source

    step = find_removal_step(provider, removed.source)
    if step is None:
        # Unregistered source — e.g. "manual", which has nothing external
        # to clean up.  The pool entry is already gone; we're done.
        return

    result = step.remove_fn(provider, removed)
    for line in result.cleaned:
        print(line)
    if result.suppress:
        suppress_credential_source(provider, removed.source)
    for line in result.hints:
        print(line)


def auth_reset_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", ""))
    pool = load_pool(provider)
    count = pool.reset_statuses()
    print(f"Reset status on {count} {provider} credentials")


def auth_status_command(args) -> None:
    provider = _normalize_provider(getattr(args, "provider", "") or "")
    if not provider:
        raise SystemExit("Provider is required. Example: `hermes auth status spotify`.")
    status = auth_mod.get_auth_status(provider)
    if not status.get("logged_in"):
        reason = status.get("error")
        if reason:
            print(f"{provider}: logged out ({reason})")
        else:
            print(f"{provider}: logged out")
        return

    print(f"{provider}: logged in")
    for key in ("auth_type", "client_id", "redirect_uri", "scope", "expires_at", "api_base_url"):
        value = status.get(key)
        if value:
            print(f"  {key}: {value}")


def auth_logout_command(args) -> None:
    auth_mod.logout_command(SimpleNamespace(provider=getattr(args, "provider", None)))


def auth_spotify_command(args) -> None:
    action = str(getattr(args, "spotify_action", "") or "login").strip().lower()
    if action in {"", "login"}:
        auth_mod.login_spotify_command(args)
        return
    if action == "status":
        auth_status_command(SimpleNamespace(provider="spotify"))
        return
    if action == "logout":
        auth_logout_command(SimpleNamespace(provider="spotify"))
        return
    raise SystemExit(f"Unknown Spotify auth action: {action}")


def _is_termux() -> bool:
    prefix = os.getenv("PREFIX", "")
    return bool(os.getenv("TERMUX_VERSION") or "com.termux/files/usr" in prefix)


def _is_windows() -> bool:
    return os.name == "nt"


def _is_wsl() -> bool:
    if _is_windows() or _is_termux():
        return False
    if os.getenv("WSL_INTEROP") or os.getenv("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except Exception:
        return False


def _find_windows_browser_command() -> str | None:
    for candidate in (
        shutil.which("msedge.exe"),
        shutil.which("chrome.exe"),
        shutil.which("chromium.exe"),
    ):
        if candidate:
            return candidate
    common_paths = [
        Path(os.getenv("ProgramFiles", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
        Path(os.getenv("ProgramFiles(x86)", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
        Path(os.getenv("ProgramFiles", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(os.getenv("ProgramFiles(x86)", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
    ]
    for path in common_paths:
        if str(path) and path.exists():
            return str(path)
    return None


def _find_desktop_browser_command() -> str | None:
    if _is_windows():
        return _find_windows_browser_command()
    return (
        shutil.which("chromium-browser")
        or shutil.which("chromium")
        or shutil.which("google-chrome")
        or shutil.which("microsoft-edge")
        or shutil.which("microsoft-edge-stable")
    )


def _chatgpt_web_browser_base_dir(browser_command: str | None = None) -> Path:
    override = os.getenv("HERMES_CHATGPT_WEB_BROWSER_BASE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    command = str(browser_command or "").strip()
    if command.startswith("/snap/bin/"):
        return Path.home() / "hermes-chatgpt-web-browser"
    return get_hermes_home() / "chatgpt-web-browser"


def _wsl_host_candidates() -> list[str]:
    candidates: list[str] = []
    try:
        resolv = Path("/etc/resolv.conf")
        if resolv.exists():
            for line in resolv.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("nameserver "):
                    value = line.split(None, 1)[1].strip()
                    if value and value not in candidates:
                        candidates.append(value)
    except Exception:
        pass
    return candidates


def _debug_base_candidates(debug_port: int, *, expose_wsl_host: bool = False) -> list[str]:
    candidates = [f"http://127.0.0.1:{debug_port}", f"http://localhost:{debug_port}"]
    if expose_wsl_host:
        for host in _wsl_host_candidates():
            candidates.append(f"http://{host}:{debug_port}")
    seen: list[str] = []
    for item in candidates:
        if item not in seen:
            seen.append(item)
    return seen


def _launch_chatgpt_web_desktop_browser(
    browser_command: str,
    base_dir: Path,
    debug_port: int,
    *,
    expose_wsl_host: bool = False,
):
    base_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = base_dir / "profile"
    logs_dir = base_dir / "logs"
    profile_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (logs_dir / "browser.log").open("ab")
    debug_address = "0.0.0.0" if expose_wsl_host else "127.0.0.1"
    command = [
        browser_command,
        f"--user-data-dir={profile_dir}",
        f"--remote-debugging-address={debug_address}",
        f"--remote-debugging-port={int(debug_port)}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-fre",
        "--disable-session-crashed-bubble",
        "https://chatgpt.com",
    ]
    popen_kwargs = {
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "cwd": str(base_dir),
        "start_new_session": True,
    }
    if _is_windows():
        popen_kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    proc = subprocess.Popen(command, **popen_kwargs)
    return proc, _debug_base_candidates(debug_port, expose_wsl_host=expose_wsl_host)


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


async def _get_chatgpt_web_browser_auth_state(debug_base: str) -> dict[str, Any] | None:
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
        await send("Runtime.enable")
        result = await send("Network.getCookies", {"urls": ["https://chatgpt.com/", "https://auth.openai.com/"]})
        cookies = result.get("result", {}).get("cookies", [])
        from hermes_cli import chatgpt_web as chatgpt_web_mod

        normalized_cookies = chatgpt_web_mod._normalize_browser_cookies(cookies)
        cookie_header = chatgpt_web_mod._build_cookie_header(
            browser_cookies=normalized_cookies,
        )
        session_token = ""
        device_id = ""
        for cookie in normalized_cookies:
            name = str(cookie.get("name") or "").strip()
            value = str(cookie.get("value") or "").strip()
            if name == "__Secure-next-auth.session-token" and value:
                session_token = value
            elif name == "oai-did" and value:
                device_id = value
        if not session_token:
            return None
        user_agent = ""
        try:
            result = await send(
                "Runtime.evaluate",
                {"expression": "navigator.userAgent", "returnByValue": True},
            )
            user_agent = str(
                result.get("result", {})
                .get("result", {})
                .get("value")
                or ""
            ).strip()
        except Exception:
            user_agent = ""
        return {
            "session_token": session_token,
            "cookie_header": cookie_header,
            "browser_cookies": normalized_cookies,
            "device_id": device_id,
            "user_agent": user_agent,
        }
    return None


def _wait_for_chatgpt_web_browser_auth_state(
    debug_base: str,
    *,
    timeout_seconds: int = 15 * 60,
    poll_seconds: int = 5,
) -> dict[str, Any] | None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            state = asyncio.run(_get_chatgpt_web_browser_auth_state(debug_base))
        except Exception:
            state = None
        if isinstance(state, dict) and str(state.get("session_token") or "").strip():
            return state
        print("waiting for ChatGPT login in browser...")
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
    if websockets is None:
        raise SystemExit("Python package 'websockets' is required for browser auth.")
    timeout_seconds = max(30, int(getattr(args, "timeout", None) or 15 * 60))
    debug_port = max(1024, int(getattr(args, "debug_port", None) or 9222))
    keep_open = bool(getattr(args, "keep_open", False))
    if _is_termux():
        if not _termux_x11_android_app_installed():
            raise SystemExit("Termux:X11 Android app (com.termux.x11) is not installed.")
        termux_x11_command = _find_termux_x11_command()
        if not termux_x11_command:
            raise SystemExit("termux-x11 command not found. Install `termux-x11-nightly`.")
        browser_command = _find_chromium_browser_command()
        if not browser_command:
            raise SystemExit("Chromium command not found. Install `chromium`.")
        base_dir = _chatgpt_web_browser_base_dir(browser_command)
        label = (getattr(args, "label", None) or "termux-x11-browser").strip() or "termux-x11-browser"
        shutil.rmtree(base_dir, ignore_errors=True)
        launcher_script, _startup_script = _write_chatgpt_web_browser_launch_scripts(
            base_dir,
            termux_x11_command,
            browser_command,
            debug_port,
        )
        proc = _launch_chatgpt_web_browser(launcher_script, base_dir)
        debug_base = _debug_base_candidates(debug_port)[0]
        print("Started local Termux browser for ChatGPT Web auth.")
        print("Open the Termux:X11 Android app manually, then finish logging into ChatGPT in Chromium.")
        success_message = "Stored chatgpt-web credential from Termux browser."
    else:
        browser_command = _find_desktop_browser_command()
        if not browser_command:
            if _is_windows():
                raise SystemExit("No supported browser found. Install Microsoft Edge, Google Chrome, or Chromium.")
            if _is_wsl():
                raise SystemExit("No supported browser found in WSL. Install Chromium/Chrome in WSLg or run this command from native Windows.")
            raise SystemExit("No supported browser found. Install Chromium, Chrome, or Edge.")
        base_dir = _chatgpt_web_browser_base_dir(browser_command)
        label_default = "windows-browser" if _is_windows() else ("wsl-browser" if _is_wsl() else "desktop-browser")
        label = (getattr(args, "label", None) or label_default).strip() or label_default
        shutil.rmtree(base_dir, ignore_errors=True)
        proc, debug_bases = _launch_chatgpt_web_desktop_browser(
            browser_command,
            base_dir,
            debug_port,
            expose_wsl_host=_is_wsl(),
        )
        if _is_windows():
            print("Started local Windows browser for ChatGPT Web auth.")
            print("Finish logging into ChatGPT in the launched browser window.")
            success_message = "Stored chatgpt-web credential from Windows browser."
        elif _is_wsl():
            print("Started local WSL browser for ChatGPT Web auth.")
            print("Finish logging into ChatGPT in the launched browser window (or WSLg session).")
            success_message = "Stored chatgpt-web credential from WSL browser."
        else:
            print("Started local browser for ChatGPT Web auth.")
            print("Finish logging into ChatGPT in the launched browser window.")
            success_message = "Stored chatgpt-web credential from desktop browser."

    try:
        if _is_termux():
            _wait_for_debugger(debug_base, timeout=min(60.0, float(timeout_seconds)))
        else:
            debug_base = _wait_for_any_debugger(debug_bases, timeout=min(60.0, float(timeout_seconds)))
        browser_auth_state = _wait_for_chatgpt_web_browser_auth_state(
            debug_base,
            timeout_seconds=timeout_seconds,
        )
        if not browser_auth_state:
            raise SystemExit("Timed out waiting for __Secure-next-auth.session-token from Chromium.")
        session_token = str(browser_auth_state.get("session_token") or "").strip()

        auth_add_command(SimpleNamespace(
            provider="chatgpt-web",
            auth_type="api-key",
            api_key=session_token,
            label=label,
            token_mode="session_token",
            cookie_header=str(browser_auth_state.get("cookie_header") or "").strip(),
            browser_cookies=browser_auth_state.get("browser_cookies"),
            device_id=str(browser_auth_state.get("device_id") or "").strip(),
            user_agent=str(browser_auth_state.get("user_agent") or "").strip(),
            portal_url=None,
            inference_url=None,
            client_id=None,
            scope=None,
            no_browser=False,
            timeout=None,
            insecure=False,
            ca_bundle=None,
        ))
        print(success_message)
        print(f'Added it to the credential pool as "{label}".')
        print("Verify with: hermes auth list")
    finally:
        if not keep_open:
            _terminate_process(proc)
            shutil.rmtree(base_dir, ignore_errors=True)


def _wait_for_any_debugger(debug_bases: list[str], timeout: float = 30.0) -> str:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        for debug_base in debug_bases:
            try:
                with urllib.request.urlopen(f"{debug_base}/json/version", timeout=5) as response:
                    if response.status == 200:
                        return debug_base
            except Exception as exc:
                last_error = exc
        time.sleep(1)
    joined = ", ".join(debug_bases)
    raise SystemExit(f"Timed out waiting for Chromium DevTools at any of [{joined}]: {last_error}")


def _interactive_auth() -> None:
    """Interactive credential pool management when `hermes auth` is called bare."""
    # Show current pool status first
    print("Credential Pool Status")
    print("=" * 50)

    auth_list_command(SimpleNamespace(provider=None))

    # Show AWS Bedrock credential status (not in the pool — uses boto3 chain)
    try:
        from agent.bedrock_adapter import has_aws_credentials, resolve_aws_auth_env_var, resolve_bedrock_region
        if has_aws_credentials():
            auth_source = resolve_aws_auth_env_var() or "unknown"
            region = resolve_bedrock_region()
            print(f"bedrock (AWS SDK credential chain):")
            print(f"  Auth: {auth_source}")
            print(f"  Region: {region}")
            try:
                import boto3
                sts = boto3.client("sts", region_name=region)
                identity = sts.get_caller_identity()
                arn = identity.get("Arn", "unknown")
                print(f"  Identity: {arn}")
            except Exception:
                print(f"  Identity: (could not resolve — boto3 STS call failed)")
            print()
    except ImportError:
        pass  # boto3 or bedrock_adapter not available

    # Show Azure Foundry Entra ID status
    try:
        from hermes_cli.config import load_config
        _cfg = load_config()
        _model_cfg = _cfg.get("model") if isinstance(_cfg, dict) else None
        if isinstance(_model_cfg, dict):
            _cfg_provider = str(_model_cfg.get("provider") or "").strip().lower()
            _cfg_auth_mode = str(_model_cfg.get("auth_mode") or "").strip().lower()
            if _cfg_provider == "azure-foundry" and _cfg_auth_mode == "entra_id":
                from agent.azure_identity_adapter import (
                    EntraIdentityConfig,
                    SCOPE_AI_AZURE_DEFAULT,
                    describe_active_credential,
                    has_azure_identity_installed,
                )
                _base_url = str(_model_cfg.get("base_url") or "").strip()
                _entra = _model_cfg.get("entra") or {}
                if not isinstance(_entra, dict):
                    _entra = {}
                _scope = (
                    str(_entra.get("scope") or "").strip()
                    or SCOPE_AI_AZURE_DEFAULT
                )
                print(f"azure-foundry (Microsoft Entra ID):")
                print(f"  Endpoint: {_base_url or '(not configured)'}")
                print(f"  Scope: {_scope}")
                if not has_azure_identity_installed():
                    print("  Status: ⚠ azure-identity not installed "
                          "(pip install azure-identity)")
                else:
                    _entra_cfg = EntraIdentityConfig(
                        scope=_scope,
                    )
                    _info = describe_active_credential(config=_entra_cfg, timeout_seconds=10.0)
                    _env_sources = _info.get("env_sources") or []
                    if _info.get("ok"):
                        _tag = ", ".join(_env_sources) if _env_sources else "default chain"
                        print(f"  Status: ✓ token acquired ({_tag})")
                    else:
                        _err = _info.get("error") or "credential chain exhausted"
                        print(f"  Status: ⚠ {_err}")
                        _hint = _info.get("hint")
                        if _hint:
                            print(f"  Hint: {_hint}")
                print()
    except Exception:
        pass
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
        custom_display = [name for name, _key, _provider_key in custom_names]
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
        print("  4. Local browser bootstrap (Termux, Windows, or WSL)")
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
    if action == "status":
        auth_status_command(args)
        return
    if action == "logout":
        auth_logout_command(args)
        return
    if action == "spotify":
        auth_spotify_command(args)
        return
    # No subcommand — launch interactive mode
    _interactive_auth()
