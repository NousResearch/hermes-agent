"""hermes gmail-push — configure and inspect native Gmail push ingestion."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from gateway.config import PlatformConfig
from gateway.platforms.gmail_push import (
    DEFAULT_HOST,
    DEFAULT_INCLUDE_HEADERS,
    DEFAULT_MAX_BODY_CHARS,
    DEFAULT_PATH,
    DEFAULT_PORT,
    DEFAULT_RENEW_EVERY_HOURS,
    GMAIL_PUSH_OAUTH_SCOPES,
    GmailPushAdapter,
    gmail_push_account_paths,
)
from hermes_constants import display_hermes_home, get_hermes_home
from hermes_cli.config import load_config, save_config

try:
    from google_auth_oauthlib.flow import Flow

    GOOGLE_OAUTHLIB_AVAILABLE = True
except ImportError:
    Flow = None  # type: ignore[assignment]
    GOOGLE_OAUTHLIB_AVAILABLE = False


def gmail_push_command(args: Namespace) -> None:
    """Entry point for ``hermes gmail-push``."""
    action = getattr(args, "gmail_push_action", None)
    if not action:
        print("Usage: hermes gmail-push {setup|status|renew|resync|test}")
        return

    if action == "setup":
        _cmd_setup(args)
        return
    if action == "status":
        _cmd_status(args)
        return
    if action == "renew":
        _cmd_renew(args)
        return
    if action == "resync":
        _cmd_resync(args)
        return
    if action == "test":
        _cmd_test(args)


def _load_gmail_push_block() -> dict[str, Any]:
    return (load_config().get("platforms", {}) or {}).get("gmail_push", {}) or {}


def _save_gmail_push_block(block: dict[str, Any]) -> None:
    config = load_config()
    platforms = config.setdefault("platforms", {})
    if not isinstance(platforms, dict):
        platforms = {}
        config["platforms"] = platforms
    platforms["gmail_push"] = block
    save_config(config)


def _display_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    hermes_home = get_hermes_home().resolve()
    try:
        rel = resolved.relative_to(hermes_home)
    except ValueError:
        return str(resolved)
    suffix = rel.as_posix()
    if not suffix:
        return display_hermes_home()
    return f"{display_hermes_home()}/{suffix}"


def _prompt_value(label: str, current: str = "", *, required: bool = False) -> str:
    if not sys.stdin.isatty():
        return current
    prompt = f"{label}"
    if current:
        prompt += f" [{current}]"
    prompt += ": "
    value = input(prompt).strip()
    if value:
        return value
    if current:
        return current
    if required:
        print(f"Error: {label} is required.")
    return ""


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _extract_code_and_state(code_or_url: str) -> tuple[str, str | None]:
    value = str(code_or_url or "").strip()
    if not value.startswith("http"):
        return value, None
    params = parse_qs(urlparse(value).query)
    codes = params.get("code")
    if not codes:
        raise ValueError("No OAuth code found in redirect URL")
    return codes[0], params.get("state", [None])[0]


def _pick_redirect_uri(client_secret_path: Path) -> str:
    try:
        data = json.loads(client_secret_path.read_text(encoding="utf-8"))
    except Exception:
        return "http://localhost"
    client = data.get("installed") or data.get("web") or {}
    redirect_uris = list(client.get("redirect_uris") or [])
    for uri in redirect_uris:
        if isinstance(uri, str) and uri.startswith("http://localhost"):
            return uri
    for uri in redirect_uris:
        if isinstance(uri, str) and uri:
            return uri
    return "http://localhost"


def _store_client_secret(source: str, destination: Path) -> None:
    src_path = Path(source).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, destination)


def _build_platform_block(args: Namespace) -> dict[str, Any]:
    existing = _load_gmail_push_block()
    extra = dict(existing.get("extra") or {})
    endpoint = dict(extra.get("endpoint") or {})
    oauth = dict(extra.get("oauth") or {})
    watch = dict(extra.get("watch") or {})
    processing = dict(extra.get("processing") or {})
    push_auth = dict(extra.get("push_auth") or {})
    state = dict(extra.get("state") or {})

    account = (
        getattr(args, "account", "") or extra.get("account") or ""
    ).strip()
    account = _prompt_value("Gmail account", account, required=True)
    if not account:
        raise ValueError("Gmail account is required")

    paths = gmail_push_account_paths(account)
    client_secret_path = Path(
        oauth.get("client_secret_path") or paths["client_secret_path"]
    ).expanduser()
    token_path = Path(oauth.get("token_path") or paths["token_path"]).expanduser()
    state_path = Path(state.get("path") or paths["state_path"]).expanduser()

    client_secret_arg = getattr(args, "client_secret", "") or ""
    if client_secret_arg:
        _store_client_secret(client_secret_arg, client_secret_path)
    elif not client_secret_path.exists():
        prompt_path = _prompt_value("OAuth client_secret.json path", "", required=True)
        if not prompt_path:
            raise ValueError("OAuth client_secret.json is required")
        _store_client_secret(prompt_path, client_secret_path)

    topic = (
        getattr(args, "topic", "") or extra.get("topic") or ""
    ).strip()
    subscription = (
        getattr(args, "subscription", "") or extra.get("subscription") or ""
    ).strip()
    public_url = (
        getattr(args, "public_url", "") or endpoint.get("public_url") or ""
    ).strip()
    service_account_email = (
        getattr(args, "service_account_email", "")
        or push_auth.get("service_account_email")
        or ""
    ).strip()

    topic = _prompt_value("Pub/Sub topic", topic, required=True)
    subscription = _prompt_value("Pub/Sub subscription", subscription, required=True)
    public_url = _prompt_value("Public HTTPS callback URL", public_url, required=True)
    service_account_email = _prompt_value(
        "Pub/Sub push service account email",
        service_account_email,
        required=True,
    )
    if not all((topic, subscription, public_url, service_account_email)):
        raise ValueError("Topic, subscription, public URL, and service account email are required")

    host = str(getattr(args, "host", "") or endpoint.get("host") or DEFAULT_HOST)
    path = str(getattr(args, "path", "") or endpoint.get("path") or DEFAULT_PATH).strip() or DEFAULT_PATH
    if not path.startswith("/"):
        path = f"/{path}"
    port = int(getattr(args, "port", 0) or endpoint.get("port") or DEFAULT_PORT)
    audience = (
        getattr(args, "audience", "")
        or push_auth.get("audience")
        or public_url
    ).strip()
    renew_every_hours = int(
        getattr(args, "renew_every_hours", 0)
        or watch.get("renew_every_hours")
        or DEFAULT_RENEW_EVERY_HOURS
    )
    label_ids = _parse_csv(getattr(args, "label_ids", "")) or list(watch.get("label_ids") or ["INBOX"])
    include_html = bool(
        getattr(args, "include_html", False)
        or processing.get("include_html", False)
    )
    max_body_chars = int(
        getattr(args, "max_body_chars", 0)
        or processing.get("max_body_chars")
        or DEFAULT_MAX_BODY_CHARS
    )

    oauth["client_secret_path"] = str(client_secret_path)
    oauth["token_path"] = str(token_path)
    state["path"] = str(state_path)
    endpoint.update(
        {
            "host": host,
            "port": port,
            "path": path,
            "public_url": public_url,
        }
    )
    watch.update(
        {
            "label_ids": label_ids,
            "label_filter_behavior": watch.get("label_filter_behavior", "INCLUDE"),
            "renew_every_hours": renew_every_hours,
        }
    )
    processing.update(
        {
            "history_types": list(processing.get("history_types") or ["messageAdded"]),
            "fetch_format": processing.get("fetch_format", "full"),
            "include_headers": list(processing.get("include_headers") or DEFAULT_INCLUDE_HEADERS),
            "include_html": include_html,
            "max_body_chars": max_body_chars,
        }
    )
    push_auth.update(
        {
            "service_account_email": service_account_email,
            "audience": audience,
        }
    )
    extra.update(
        {
            "account": account,
            "topic": topic,
            "subscription": subscription,
            "endpoint": endpoint,
            "oauth": oauth,
            "watch": watch,
            "push_auth": push_auth,
            "processing": processing,
            "state": state,
        }
    )
    return {"enabled": True, "extra": extra}


def _exchange_auth_code(
    client_secret_path: Path,
    token_path: Path,
    auth_code_or_url: str,
    *,
    redirect_uri: str | None = None,
) -> None:
    if not GOOGLE_OAUTHLIB_AVAILABLE:
        raise RuntimeError(
            "google-auth-oauthlib is not installed. Run: pip install 'hermes-agent[gmail-push]'"
        )

    chosen_redirect_uri = redirect_uri or _pick_redirect_uri(client_secret_path)
    flow = Flow.from_client_secrets_file(
        str(client_secret_path),
        scopes=GMAIL_PUSH_OAUTH_SCOPES,
        redirect_uri=chosen_redirect_uri,
        autogenerate_code_verifier=True,
    )
    auth_url, expected_state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )

    code_input = auth_code_or_url.strip()
    if not code_input:
        print("\nOpen this URL in your browser and approve Gmail access:\n")
        print(auth_url)
        print("\nPaste the full redirect URL or the raw authorization code below.")
        if not sys.stdin.isatty():
            raise RuntimeError("No auth code provided. Re-run with --auth-code '<code-or-url>'.")
        code_input = input("Auth code or redirect URL: ").strip()
    if not code_input:
        raise RuntimeError("OAuth code is required")

    code, returned_state = _extract_code_and_state(code_input)
    if returned_state and returned_state != expected_state:
        raise RuntimeError("OAuth state mismatch. Start setup again with a fresh authorization URL.")

    os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
    flow.fetch_token(code=code)
    creds = flow.credentials
    payload = json.loads(creds.to_json())
    granted_scopes = list(getattr(creds, "granted_scopes", []) or [])
    if granted_scopes:
        payload["scopes"] = granted_scopes
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_setup_summary(block: dict[str, Any]) -> None:
    extra = block["extra"]
    endpoint = extra["endpoint"]
    oauth = extra["oauth"]
    push_auth = extra["push_auth"]
    account = extra["account"]
    print("\nGmail push configured.\n")
    print(f"  Account:        {account}")
    print(f"  Topic:          {extra['topic']}")
    print(f"  Subscription:   {extra['subscription']}")
    print(f"  Callback URL:   {endpoint['public_url']}")
    print(f"  Audience:       {push_auth['audience']}")
    print(f"  Push auth SA:   {push_auth['service_account_email']}")
    print(f"  Token path:     {_display_path(oauth['token_path'])}")
    print(f"  State path:     {_display_path(extra['state']['path'])}")
    print("\nRequired Google IAM / Pub/Sub wiring:")
    print(
        "  1. Grant publish permission on the topic to "
        "gmail-api-push@system.gserviceaccount.com."
    )
    print(
        f"  2. Configure the push subscription to POST to {endpoint['public_url']} "
        f"with audience {push_auth['audience']}."
    )
    print(
        f"  3. Use {push_auth['service_account_email']} as the authenticated push service account."
    )
    print("\nStart ingestion with: hermes gateway run")


def _make_cli_adapter(*, require_enabled: bool = True) -> GmailPushAdapter:
    block = _load_gmail_push_block()
    if not block:
        raise RuntimeError("Gmail push is not configured. Run: hermes gmail-push setup")
    if require_enabled and not block.get("enabled"):
        raise RuntimeError("Gmail push is configured but disabled in config.yaml")
    return GmailPushAdapter(PlatformConfig.from_dict(block))


def _cmd_setup(args: Namespace) -> None:
    block = _build_platform_block(args)
    oauth = block["extra"]["oauth"]
    _exchange_auth_code(
        Path(oauth["client_secret_path"]),
        Path(oauth["token_path"]),
        getattr(args, "auth_code", "") or "",
        redirect_uri=getattr(args, "redirect_uri", "") or None,
    )
    _save_gmail_push_block(block)
    _print_setup_summary(block)


def _cmd_status(args: Namespace) -> None:
    block = _load_gmail_push_block()
    if not block:
        print("Gmail push is not configured.")
        return

    adapter = GmailPushAdapter(PlatformConfig.from_dict(block))
    state = dict(adapter._state)
    extra = block.get("extra") or {}
    endpoint = extra.get("endpoint") or {}
    push_auth = extra.get("push_auth") or {}
    oauth = extra.get("oauth") or {}
    print("Gmail push status\n")
    print(f"  Enabled:                 {bool(block.get('enabled'))}")
    print(f"  Account:                 {extra.get('account', '(unset)')}")
    print(f"  Callback URL:            {endpoint.get('public_url', '(unset)')}")
    print(f"  Audience:                {push_auth.get('audience', '(unset)')}")
    print(f"  Topic:                   {extra.get('topic', '(unset)')}")
    print(f"  Subscription:            {extra.get('subscription', '(unset)')}")
    print(f"  Token path:              {_display_path(oauth.get('token_path', '')) if oauth.get('token_path') else '(unset)'}")
    print(f"  Last history id:         {state.get('last_history_id') or '(none)'}")
    print(f"  Watch expiration (ms):   {state.get('watch_expiration_ms') or '(none)'}")
    print(f"  Last watch renewed at:   {state.get('last_watch_renewed_at') or '(never)'}")
    print(f"  Last notification at:    {state.get('last_notification_at') or '(never)'}")
    print(f"  Last successful push id: {state.get('last_successful_pubsub_message_id') or '(none)'}")
    print(f"  Degraded:                {bool(state.get('degraded'))}")
    print(f"  Last error:              {state.get('last_error') or '(none)'}")


def _cmd_renew(args: Namespace) -> None:
    adapter = _make_cli_adapter()
    watch_result = asyncio.run(adapter.refresh_watch_now())
    print("Watch renewed.")
    print(f"  History ID: {watch_result.get('historyId')}")
    print(f"  Expiration: {watch_result.get('expiration')}")


def _cmd_resync(args: Namespace) -> None:
    adapter = _make_cli_adapter()
    watch_result = asyncio.run(adapter.rebaseline())
    print("Baseline history cursor reset.")
    print(f"  History ID: {watch_result.get('historyId')}")
    print(f"  Expiration: {watch_result.get('expiration')}")


def _cmd_test(args: Namespace) -> None:
    adapter = _make_cli_adapter(require_enabled=False)
    result = asyncio.run(adapter.run_health_check())
    print("Gmail push health check\n")
    print(f"  OK:      {result['ok']}")
    print(f"  Account: {result['account'] or '(unset)'}")
    endpoint = result.get("endpoint") or {}
    print(
        f"  Endpoint: {endpoint.get('host', DEFAULT_HOST)}:"
        f"{endpoint.get('port', DEFAULT_PORT)}{endpoint.get('path', DEFAULT_PATH)}"
    )
    issues = result.get("issues") or []
    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo issues detected.")
