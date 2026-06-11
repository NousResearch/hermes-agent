"""Telegram Managed Bot onboarding client.

Uses Telegram's Managed Bots feature to create a user-owned child bot without
manual BotFather token copy-paste. Hermes talks only to the Nous onboarding
service; the raw Telegram token is saved locally after one-time retrieval.
"""

from __future__ import annotations

import os
import re
import secrets
import sys
import time
import urllib.parse
from dataclasses import dataclass
from typing import Optional

import httpx

# Default pairing API base URL (Nous-hosted Cloudflare Worker).
# Override for PoC/staging with TELEGRAM_ONBOARDING_URL.
DEFAULT_API_URL = "https://setup.hermes-agent.nousresearch.com"
TELEGRAM_ONBOARDING_URL_ENV = "TELEGRAM_ONBOARDING_URL"

# The Nous-hosted manager bot username (without @). The backend returns the
# actual deep link, so this is only used by local helpers/tests.
DEFAULT_MANAGER_BOT = "HermesSetupBot"

DEFAULT_BOT_NAME = "Hermes Agent"
DEFAULT_POLL_TIMEOUT = 180
POLL_INTERVAL = 2

_USERNAME_SLUG_ALPHABET = "abcdefghijklmnopqrstuvwxyz234567"
_TELEGRAM_BOT_TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]{30,}$")


@dataclass(frozen=True)
class TelegramPairing:
    """Pairing record returned by the Telegram onboarding service."""

    pairing_id: str
    poll_token: str
    suggested_username: str
    deep_link: str
    qr_payload: str
    expires_at: str | None = None


@dataclass(frozen=True)
class TelegramBotSetupResult:
    """Successful Telegram onboarding result returned by the setup service."""

    token: str
    bot_username: str | None = None
    owner_user_id: int | None = None


def _api_url(api_url: str | None = None) -> str:
    """Resolve the onboarding API URL, honoring the PoC env override."""
    return (
        api_url or os.environ.get(TELEGRAM_ONBOARDING_URL_ENV) or DEFAULT_API_URL
    ).rstrip("/")


def is_valid_telegram_bot_token(token: object) -> bool:
    """Return True when *token* has Telegram's bot-token shape."""
    return isinstance(token, str) and bool(_TELEGRAM_BOT_TOKEN_RE.match(token))


def _parse_owner_user_id(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.isdecimal():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def render_qr_terminal(url: str) -> str:
    """Render a URL as a QR code string suitable for terminal output."""
    try:
        import io

        import qrcode  # type: ignore[import-untyped]

        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=1,
            border=1,
        )
        qr.add_data(url)
        qr.make(fit=True)

        buf = io.StringIO()
        qr.print_ascii(out=buf, invert=True)
        return buf.getvalue()
    except ImportError:
        return ""


def print_qr_code(url: str, *, include_link: bool = True) -> None:
    """Print a QR code to stdout, with URL fallback if qrcode is missing."""
    qr_text = render_qr_terminal(url)
    if qr_text:
        print(qr_text)
    else:
        print("  (Install 'qrcode' for a scannable QR code: pip install qrcode)")
    if include_link:
        print(f"  Link: {url}")


def generate_username_slug(length: int = 16) -> str:
    """Generate a base32-ish slug for Telegram username correlation.

    Sixteen characters from a 32-symbol alphabet gives 80 bits of entropy while
    keeping ``hermes_<slug>_bot`` under Telegram's 32-character username limit.
    """
    return "".join(secrets.choice(_USERNAME_SLUG_ALPHABET) for _ in range(length))


def generate_bot_username(profile_name: Optional[str] = None) -> str:
    """Generate a secure suggested bot username like ``hermes_<slug>_bot``.

    ``profile_name`` is accepted for backward compatibility with the original
    PoC, but is intentionally not embedded in the username. The username has to
    carry enough entropy for backend correlation.
    """
    _ = profile_name
    return f"hermes_{generate_username_slug()}_bot"


def generate_deep_link(
    manager_bot: str = DEFAULT_MANAGER_BOT,
    suggested_username: Optional[str] = None,
    suggested_name: Optional[str] = None,
) -> str:
    """Build a ``t.me/newbot`` deep link for managed bot creation."""
    manager = manager_bot.lstrip("@")
    username = suggested_username or generate_bot_username()
    base_url = (
        "https://t.me/newbot/"
        f"{urllib.parse.quote(manager)}/"
        f"{urllib.parse.quote(username)}"
    )

    if suggested_name:
        params = urllib.parse.urlencode({"name": suggested_name})
        return f"{base_url}?{params}"
    return base_url


def generate_pairing_nonce() -> str:
    """Generate a legacy-compatible random nonce string.

    The new protocol uses service-created ``pairing_id`` + bearer
    ``poll_token`` instead of a path nonce, but this helper is harmless and
    still useful for callers/tests that need a generic random id.
    """
    return secrets.token_hex(16)


def create_pairing(
    api_url: str | None = None,
    bot_name: str = DEFAULT_BOT_NAME,
    timeout: float = 10.0,
) -> TelegramPairing | None:
    """Create a Telegram onboarding pairing.

    ``POST /v1/telegram/pairings`` returns the deep link, QR payload, public
    pairing id, and secret poll token. The token is only used as a bearer
    credential while polling.
    """
    try:
        resp = httpx.post(
            f"{_api_url(api_url)}/v1/telegram/pairings",
            json={"bot_name": bot_name},
            timeout=timeout,
        )
        if resp.status_code not in (200, 201):
            return None
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return None

    required = ("pairing_id", "poll_token", "suggested_username", "deep_link")
    if not all(isinstance(data.get(key), str) and data.get(key) for key in required):
        return None

    qr_payload = data.get("qr_payload") or data["deep_link"]
    if not isinstance(qr_payload, str):
        return None

    expires_at = data.get("expires_at")
    return TelegramPairing(
        pairing_id=data["pairing_id"],
        poll_token=data["poll_token"],
        suggested_username=data["suggested_username"],
        deep_link=data["deep_link"],
        qr_payload=qr_payload,
        expires_at=expires_at if isinstance(expires_at, str) else None,
    )


def poll_pairing_result_once(
    api_url: str | None,
    pairing: TelegramPairing,
    timeout: float = 10.0,
) -> TelegramBotSetupResult | None:
    """Poll the onboarding service once. Returns setup metadata when ready."""
    resp = httpx.get(
        f"{_api_url(api_url)}/v1/telegram/pairings/{pairing.pairing_id}",
        headers={"Authorization": f"Bearer {pairing.poll_token}"},
        timeout=timeout,
    )
    if resp.status_code != 200:
        return None

    data = resp.json()
    if data.get("status") != "ready":
        return None
    token = data.get("token")
    if not is_valid_telegram_bot_token(token):
        return None

    bot_username = data.get("bot_username")
    return TelegramBotSetupResult(
        token=token,
        bot_username=bot_username
        if isinstance(bot_username, str) and bot_username
        else None,
        owner_user_id=_parse_owner_user_id(data.get("owner_user_id")),
    )


def poll_pairing_once(
    api_url: str | None,
    pairing: TelegramPairing,
    timeout: float = 10.0,
) -> str | None:
    """Poll the onboarding service once. Returns the token when ready."""
    result = poll_pairing_result_once(api_url, pairing, timeout=timeout)
    return result.token if result else None


def poll_for_setup_result(
    api_url: str | None,
    pairing: TelegramPairing,
    timeout: float = DEFAULT_POLL_TIMEOUT,
    interval: float = POLL_INTERVAL,
) -> Optional[TelegramBotSetupResult]:
    """Poll the pairing API until setup metadata is available or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = poll_pairing_result_once(api_url, pairing)
            if result:
                return result
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(interval)
    return None


def poll_for_token(
    api_url: str | None,
    pairing: TelegramPairing,
    timeout: float = DEFAULT_POLL_TIMEOUT,
    interval: float = POLL_INTERVAL,
) -> Optional[str]:
    """Poll the pairing API until the bot token is available or timeout."""
    result = poll_for_setup_result(api_url, pairing, timeout=timeout, interval=interval)
    return result.token if result else None


def auto_setup_telegram_bot_result(
    api_url: str | None = None,
    manager_bot: str = DEFAULT_MANAGER_BOT,
    profile_name: Optional[str] = None,
    poll_timeout: float = DEFAULT_POLL_TIMEOUT,
) -> Optional[TelegramBotSetupResult]:
    """Run the full automatic Telegram bot creation flow."""
    _ = manager_bot, profile_name
    resolved_api_url = _api_url(api_url)
    print()
    print(f"  Contacting Hermes Telegram onboarding service: {resolved_api_url}")
    sys.stdout.flush()
    pairing = create_pairing(resolved_api_url)
    if not pairing:
        print("  ✗ Could not reach the Hermes Telegram onboarding service.")
        print("    Try the manual setup instead, or check your network.")
        return None

    print("  ✓ Pairing created")
    print("  Rendering QR code...")
    sys.stdout.flush()
    print()
    print("  Scan this QR code with your phone, or open the link below:")
    print()
    print_qr_code(pairing.qr_payload, include_link=False)
    print()
    print(f"  Link: {pairing.deep_link}")
    print()
    print("  When Telegram opens, tap 'Create Bot' to confirm.")
    print("  (You can edit the bot display name before confirming)")
    print()

    spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    start = time.monotonic()
    deadline = start + poll_timeout
    idx = 0

    while time.monotonic() < deadline:
        char = spinner_chars[idx % len(spinner_chars)]
        elapsed = int(time.monotonic() - start)
        remaining = max(0, int(poll_timeout - elapsed))
        sys.stdout.write(
            f"\r  {char} Waiting for bot creation... ({remaining}s remaining) "
        )
        sys.stdout.flush()
        idx += 1

        try:
            result = poll_pairing_result_once(resolved_api_url, pairing)
            if result:
                sys.stdout.write(
                    "\r  ✓ Bot created successfully!                              \n"
                )
                sys.stdout.flush()
                return result
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(POLL_INTERVAL)

    sys.stdout.write("\r  ✗ Timed out waiting for bot creation.                    \n")
    sys.stdout.flush()
    print("    The bot may still be created — check Telegram.")
    print("    You can paste the token manually below, or re-run setup.")
    return None


def auto_setup_telegram_bot(
    api_url: str | None = None,
    manager_bot: str = DEFAULT_MANAGER_BOT,
    profile_name: Optional[str] = None,
    poll_timeout: float = DEFAULT_POLL_TIMEOUT,
) -> Optional[str]:
    """Run automatic Telegram bot creation and return only the bot token."""
    result = auto_setup_telegram_bot_result(
        api_url=api_url,
        manager_bot=manager_bot,
        profile_name=profile_name,
        poll_timeout=poll_timeout,
    )
    return result.token if result else None
