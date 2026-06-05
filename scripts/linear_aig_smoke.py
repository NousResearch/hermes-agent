"""Smoke harness for the Linear Agent Interaction gateway adapter.

This helper intentionally never prints Linear credential values. It reads
only the specific environment variables required for the smoke test, starts
the real ``LinearAIGAdapter``, and returns a small deterministic Hermes-style
response for any received Agent Session.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gateway.config import PlatformConfig
from gateway.platforms.linear_aig import LinearAIGAdapter


DEFAULT_TOKEN_URL = "https://api.linear.app/oauth/token"
TOKEN_ENV_NAMES = (
    "LINEAR_OAUTH_TOKEN",
    "LINEAR_ACCESS_TOKEN",
    "HERMES_LINEAR_AIG_ACCESS_TOKEN",
    "LINEAR_API_KEY",
)
SECRET_ENV_NAMES = (
    "LINEAR_AIG_WEBHOOK_SECRET",
    "LINEAR_WEBHOOK_SECRET",
    "HERMES_LINEAR_AIG_WEBHOOK_SECRET",
    "LINEAR_HERMES_SIGNING_SECRET",
    "LINEAR_HERMES__SIGNING_SECRET",
)
CLIENT_ID_ENV_NAMES = (
    "HERMES_LINEAR_AIG_CLIENT_ID",
    "LINEAR_HERMES_CLIENT_ID",
    "LINEAR_OAUTH_CLIENT_ID",
    "LINEAR_CLIENT_ID",
)
CLIENT_SECRET_ENV_NAMES = (
    "HERMES_LINEAR_AIG_CLIENT_SECRET",
    "LINEAR_HERMES_CLIENT_SECRET",
    "LINEAR-HERMES_CLIENT_SECRET",
    "LINEAR_HERMES__SECRET",
    "LINEAR_OAUTH_CLIENT_SECRET",
    "LINEAR_CLIENT_SECRET",
)
CLIENT_SCOPE_ENV_NAMES = (
    "HERMES_LINEAR_AIG_CLIENT_CREDENTIALS_SCOPE",
    "LINEAR_HERMES_CLIENT_CREDENTIALS_SCOPE",
    "LINEAR_CLIENT_CREDENTIALS_SCOPE",
)
TOKEN_SHAPE_RE = re.compile(r"\b(?:lin_api|lin_oauth)_[A-Za-z0-9_-]+\b")


def _windows_env(name: str) -> str:
    if platform.system() != "Windows":
        return ""
    try:
        import winreg
    except ImportError:
        return ""
    for root, subkey in (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    ):
        try:
            with winreg.OpenKey(root, subkey) as key:
                value, _ = winreg.QueryValueEx(key, name)
                return str(value or "")
        except OSError:
            continue
    return ""


def _get_secret_env(names: tuple[str, ...]) -> tuple[str, str]:
    for name in names:
        value = (os.environ.get(name) or _windows_env(name)).strip()
        if value:
            return name, value
    return "", ""


def _mask_state(name: str, value: str) -> str:
    if value:
        return f"{name}=present length={len(value)}"
    return f"{name}=missing"


def _redact_secret_shapes(text: str) -> str:
    return TOKEN_SHAPE_RE.sub("<redacted-token>", str(text or ""))


def _scope_looks_like_secret(value: str) -> bool:
    stripped = str(value or "").strip()
    return bool(TOKEN_SHAPE_RE.search(stripped))


def _oauth_error_summary(status: int, text: str) -> str:
    redacted = _redact_secret_shapes(text)
    try:
        payload = json.loads(redacted)
    except json.JSONDecodeError:
        return f"HTTP {status} {redacted[:200]}"
    error = str(payload.get("error") or "error")
    description = str(payload.get("error_description") or "").strip()
    if description:
        return f"HTTP {status} {error}: {description[:160]}"
    return f"HTTP {status} {error}"


def _client_credentials_config(args: argparse.Namespace) -> tuple[str, str, str, str]:
    client_id_name, client_id = _get_secret_env(CLIENT_ID_ENV_NAMES)
    client_secret_name, client_secret = _get_secret_env(CLIENT_SECRET_ENV_NAMES)
    scope_name, scope = _get_secret_env(CLIENT_SCOPE_ENV_NAMES)
    scope = (args.client_credentials_scope or scope).strip()
    if args.client_credentials_scope:
        scope_name = "--client-credentials-scope"
    source = (
        f"client_id={client_id_name or 'missing'} "
        f"client_secret={client_secret_name or 'missing'} "
        f"scope={scope_name or 'missing'}"
    )
    return client_id, client_secret, scope, source


async def _fetch_client_credentials_token(args: argparse.Namespace) -> tuple[str, str]:
    client_id, client_secret, scope, source = _client_credentials_config(args)
    print(f"client credentials source: {source}")
    if not client_id or not client_secret or not scope:
        print(
            "client credentials skipped: set client id, client secret, and scope. "
            "Scope is required because Linear may replace existing app tokens when "
            "a different scope set is requested."
        )
        return "", ""
    if _scope_looks_like_secret(scope):
        print(
            "client credentials skipped: scope value looks like a token or secret. "
            "Set it to a comma-separated Linear scope list instead."
        )
        return "", ""
    try:
        import aiohttp
    except ImportError:
        print("client credentials failed: aiohttp is not installed")
        return "", ""

    data = {
        "grant_type": "client_credentials",
        "scope": scope,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            args.client_credentials_token_url,
            data=data,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            text = await response.text()
            if response.status >= 400:
                print(
                    "client credentials failed: "
                    f"{_oauth_error_summary(response.status, text)}"
                )
                return "", ""
            try:
                payload = await response.json()
            except Exception:
                print("client credentials failed: token endpoint returned non-JSON")
                return "", ""
    token = str(payload.get("access_token") or "").strip()
    if not token:
        print("client credentials failed: access_token missing from response")
        return "", ""
    print(_mask_state("client credentials token", token))
    return "client_credentials", token


def _build_adapter(args: argparse.Namespace) -> tuple[LinearAIGAdapter, str, str]:
    token_name, token = _get_secret_env(TOKEN_ENV_NAMES)
    secret_name, secret = _get_secret_env(SECRET_ENV_NAMES)
    extra = {
        "host": args.host,
        "port": args.port,
        "webhook_path": args.path,
        "webhook_secret": secret,
        "initial_ack": args.initial_ack,
        "progress_action": "Smoke test",
    }
    if token_name == "LINEAR_API_KEY":
        config = PlatformConfig(enabled=True, api_key=token, extra=extra)
    else:
        config = PlatformConfig(enabled=True, token=token, extra=extra)
    return LinearAIGAdapter(config), token_name, secret_name


async def _doctor(adapter: LinearAIGAdapter, token_name: str, secret_name: str) -> int:
    print(_mask_state("linear token", adapter._resolved_access_token()))
    print(_mask_state("webhook secret", adapter._resolved_webhook_secret()))
    print(f"token source={token_name or 'missing'}")
    print(f"webhook secret source={secret_name or 'missing'}")
    if not adapter._resolved_access_token() or not adapter._resolved_webhook_secret():
        return 2
    try:
        data = await adapter._graphql(
            "query { viewer { id name } organization { name urlKey } }",
            {},
        )
    except Exception as exc:
        print(f"linear api failed: {exc}")
        if token_name in {"LINEAR_OAUTH_TOKEN", "LINEAR_ACCESS_TOKEN", "HERMES_LINEAR_AIG_ACCESS_TOKEN"}:
            print(
                "hint: OAuth tokens can expire or be revoked. Refresh the Hermes "
                "Linear Agent app OAuth token and rerun this doctor check."
            )
        return 1
    viewer = data.get("viewer") or {}
    organization = data.get("organization") or {}
    print(
        "linear api ok: "
        f"viewer={viewer.get('name') or '<unknown>'} "
        f"organization={organization.get('name') or '<unknown>'} "
        f"urlKey={organization.get('urlKey') or '<unknown>'}"
    )
    if token_name == "LINEAR_API_KEY":
        print(
            "warning: LINEAR_API_KEY authenticates as a user. "
            "For a true Linear agent app user smoke test, use the OAuth token "
            "for the Hermes Linear Agent app."
        )
    return 0


async def _serve(args: argparse.Namespace) -> int:
    if args.client_credentials_token:
        token_name, token = await _fetch_client_credentials_token(args)
        if not token:
            return 2
        os.environ["HERMES_LINEAR_AIG_ACCESS_TOKEN"] = token
        print("client credentials token source=process-only")

    adapter, token_name, secret_name = _build_adapter(args)
    doctor_status = await _doctor(adapter, token_name, secret_name)
    if doctor_status != 0:
        return doctor_status
    if args.doctor_only:
        return 0

    async def _handler(event):
        now = datetime.now(timezone.utc).isoformat()
        print(
            "received AgentSessionEvent: "
            f"chat_id={event.source.chat_id} message_id={event.message_id} "
            f"prompt_len={len(event.text or '')}"
        )
        return (
            "Hermes Linear AIG smoke test response.\n\n"
            f"Received by Hermes at {now}."
        )

    adapter.set_message_handler(_handler)
    await adapter.connect()
    endpoint = f"http://{args.host}:{args.port}{args.path}"
    public_url = args.public_url.rstrip("/") if args.public_url else ""
    print(f"local endpoint: {endpoint}")
    if public_url:
        print(f"configure Linear Agent Session Events URL: {public_url}{args.path}")
    else:
        print(
            "public URL not supplied. Start a HTTPS tunnel and configure "
            f"Linear Agent Session Events to <public-url>{args.path}."
        )
    print("waiting for Linear AgentSessionEvent webhooks; press Ctrl+C to stop")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass
    try:
        await stop_event.wait()
    finally:
        await adapter.disconnect()
    return 0


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8647)
    parser.add_argument("--path", default="/linear/aig")
    parser.add_argument("--public-url", default="")
    parser.add_argument("--doctor-only", action="store_true")
    parser.add_argument(
        "--client-credentials-token",
        action="store_true",
        help=(
            "Fetch a process-only Linear client_credentials app token before "
            "running doctor/serve. Requires client id, client secret, and scope."
        ),
    )
    parser.add_argument(
        "--client-credentials-scope",
        default="",
        help=(
            "Comma-separated Linear scopes for client_credentials token minting. "
            "Required unless HERMES_LINEAR_AIG_CLIENT_CREDENTIALS_SCOPE or "
            "LINEAR_CLIENT_CREDENTIALS_SCOPE is set."
        ),
    )
    parser.add_argument(
        "--client-credentials-token-url",
        default=DEFAULT_TOKEN_URL,
        help="Linear OAuth token endpoint.",
    )
    parser.add_argument(
        "--initial-ack",
        default="Hermes received the Linear smoke test and is starting work.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(_serve(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
