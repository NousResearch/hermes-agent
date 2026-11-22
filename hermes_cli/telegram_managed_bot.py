"""Telegram Managed Bot onboarding client: creates a user-owned child bot via the Nous onboarding
service (no BotFather copy-paste); the raw Telegram token is saved locally after one retrieval."""

from __future__ import annotations

from pm import install_hint
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

import httpx

# Nous-hosted pairing API; override for PoC/staging with TELEGRAM_ONBOARDING_URL.
DEFAULT_API_URL = "https://setup.hermes-agent.nousresearch.com"
TELEGRAM_ONBOARDING_URL_ENV = "TELEGRAM_ONBOARDING_URL"
DEFAULT_BOT_NAME = "Hermes Agent"
DEFAULT_POLL_TIMEOUT = 180
POLL_INTERVAL = 2

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
    return (api_url or os.environ.get(TELEGRAM_ONBOARDING_URL_ENV) or DEFAULT_API_URL).rstrip("/")


def is_valid_telegram_bot_token(token: object) -> bool:
    """Return True when *token* has Telegram's bot-token shape."""
    return isinstance(token, str) and bool(_TELEGRAM_BOT_TOKEN_RE.match(token))


def _parse_owner_user_id(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.isdecimal():
        value = int(value)
    return value if isinstance(value, int) and value > 0 else None


def render_qr_terminal(url: str) -> str:
    """Render a URL as a QR code string suitable for terminal output."""
    try:
        import io
        import qrcode  # type: ignore[import-untyped]
    except ImportError:
        return ""
    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=1, border=1)
    qr.add_data(url)
    qr.make(fit=True)
    buf = io.StringIO()
    qr.print_ascii(out=buf, invert=True)
    return buf.getvalue()


def print_qr_code(url: str, *, include_link: bool = True) -> None:
    """Print a QR code to stdout, with URL fallback if qrcode is missing."""
    print(render_qr_terminal(url) or (
        "  (QR code unavailable. From the Hermes environment, run: "
        f"{install_hint('messaging')}. "
        "Then restart Hermes.)"))
    if include_link:
        print(f"  Link: {url}")


def create_pairing(
    api_url: str | None = None, bot_name: str = DEFAULT_BOT_NAME, timeout: float = 10.0
) -> TelegramPairing | None:
    """POST a pairing; the returned poll token is only used as a bearer credential while polling."""
    try:
        resp = httpx.post(f"{_api_url(api_url)}/v1/telegram/pairings", json={"bot_name": bot_name}, timeout=timeout)
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
    return TelegramPairing(*(data[key] for key in required), qr_payload=qr_payload,
                           expires_at=expires_at if isinstance(expires_at, str) else None)


def poll_pairing_result_once(
    api_url: str | None, pairing: TelegramPairing, timeout: float = 10.0
) -> TelegramBotSetupResult | None:
    """Poll the onboarding service once. Returns setup metadata when ready."""
    resp = httpx.get(f"{_api_url(api_url)}/v1/telegram/pairings/{pairing.pairing_id}",
                     headers={"Authorization": f"Bearer {pairing.poll_token}"}, timeout=timeout)
    if resp.status_code != 200:
        return None
    data = resp.json()
    token = data.get("token")
    if data.get("status") != "ready" or not is_valid_telegram_bot_token(token):
        return None
    bot_username = data.get("bot_username")
    return TelegramBotSetupResult(
        token, bot_username if isinstance(bot_username, str) and bot_username else None,
        _parse_owner_user_id(data.get("owner_user_id")))


def poll_for_setup_result(
    api_url: str | None, pairing: TelegramPairing, timeout: float = DEFAULT_POLL_TIMEOUT,
    interval: float = POLL_INTERVAL, on_tick=None) -> Optional[TelegramBotSetupResult]:
    """Poll the pairing API until setup metadata is available or timeout. ``on_tick(elapsed_s)``
    runs before each attempt (progress display)."""
    start = time.monotonic()
    deadline = start + timeout
    while time.monotonic() < deadline:
        if on_tick:
            on_tick(time.monotonic() - start)
        try:  # transport/JSON errors count as 'not ready yet'
            if result := poll_pairing_result_once(api_url, pairing):
                return result
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(interval)
    return None


def auto_setup_telegram_bot_result(
    api_url: str | None = None, manager_bot: str = "HermesSetupBot",
    profile_name: Optional[str] = None, poll_timeout: float = DEFAULT_POLL_TIMEOUT,
) -> Optional[TelegramBotSetupResult]:
    """Run the full automatic Telegram bot creation flow."""
    _ = manager_bot, profile_name  # accepted for callers; the service decides both
    resolved_api_url = _api_url(api_url)
    print(f"\n  Contacting Hermes Telegram onboarding service: {resolved_api_url}")
    sys.stdout.flush()
    pairing = create_pairing(resolved_api_url)
    if not pairing:
        print("  ✗ Could not reach the Hermes Telegram onboarding service.\n"
              "    Try the manual setup instead, or check your network.")
        return None

    print("  ✓ Pairing created\n  Rendering QR code...")
    sys.stdout.flush()
    print("\n  Scan this QR code with your phone, or open the link below:\n")
    print_qr_code(pairing.qr_payload, include_link=False)
    print(f"\n  Link: {pairing.deep_link}\n"
          "  When Telegram opens, tap 'Create Bot' to confirm.\n"
          "  (You can edit the bot display name before confirming)\n")

    spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    ticks = iter(range(1 << 30))

    def spin(elapsed: float) -> None:
        char = spinner_chars[next(ticks) % len(spinner_chars)]
        remaining = max(0, int(poll_timeout - int(elapsed)))
        sys.stdout.write(f"\r  {char} Waiting for bot creation... ({remaining}s remaining) ")
        sys.stdout.flush()

    result = poll_for_setup_result(resolved_api_url, pairing, poll_timeout, POLL_INTERVAL, on_tick=spin)
    if result:
        sys.stdout.write("\r  ✓ Bot created successfully!                              \n")
        sys.stdout.flush()
        return result
    sys.stdout.write("\r  ✗ Timed out waiting for bot creation.                    \n")
    sys.stdout.flush()
    print("    The bot may still be created — check Telegram.\n"
          "    You can paste the token manually below, or re-run setup.")
    return None
