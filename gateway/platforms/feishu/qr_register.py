"""QR-based Feishu app registration / bot onboarding.

Hermes-side onboarding helper; not part of the lark-oapi SDK surface.

Public API (re-exported via ``gateway.platforms.feishu.__init__`` so
``hermes_cli/gateway.py`` can import them as
``from gateway.platforms.feishu import qr_register, probe_bot``):

    - probe_bot(app_id, app_secret, domain) -> Optional[dict]
    - qr_register(*, initial_domain="feishu", timeout_seconds=600) -> Optional[dict]
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import lark_oapi as lark
    from lark_oapi.core import AccessTokenType, HttpMethod
    from lark_oapi.core.const import FEISHU_DOMAIN, LARK_DOMAIN
    from lark_oapi.core.model import BaseRequest

    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    lark = None  # type: ignore[assignment]
    FEISHU_DOMAIN = None  # type: ignore[assignment]
    LARK_DOMAIN = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


# QR onboarding constants.
_ONBOARD_ACCOUNTS_URLS = {
    "feishu": "https://accounts.feishu.cn",
    "lark": "https://accounts.larksuite.com",
}
_ONBOARD_OPEN_URLS = {
    "feishu": "https://open.feishu.cn",
    "lark": "https://open.larksuite.com",
}
_REGISTRATION_PATH = "/oauth/v1/app/registration"
_ONBOARD_REQUEST_TIMEOUT_S = 10


# =============================================================================
# QR scan-to-create onboarding
#
# Device-code flow: user scans a QR code with Feishu/Lark mobile app and the
# platform creates a fully configured bot application automatically.
# Called by `hermes gateway setup` via _setup_feishu() in hermes_cli/gateway.py.
# =============================================================================


def _accounts_base_url(domain: str) -> str:
    return _ONBOARD_ACCOUNTS_URLS.get(domain, _ONBOARD_ACCOUNTS_URLS["feishu"])


def _onboard_open_base_url(domain: str) -> str:
    return _ONBOARD_OPEN_URLS.get(domain, _ONBOARD_OPEN_URLS["feishu"])


def _post_registration(base_url: str, body: Dict[str, str]) -> dict:
    """POST form-encoded data to the registration endpoint, return parsed JSON.

    The registration endpoint returns JSON even on 4xx (e.g. poll returns
    authorization_pending as a 400). We always parse the body regardless of
    HTTP status.
    """
    url = f"{base_url}{_REGISTRATION_PATH}"
    data = urlencode(body).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urlopen(req, timeout=_ONBOARD_REQUEST_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body_bytes = exc.read()
        if body_bytes:
            try:
                return json.loads(body_bytes.decode("utf-8"))
            except (ValueError, json.JSONDecodeError):
                raise exc from None
        raise


def _init_registration(domain: str = "feishu") -> None:
    """Verify the environment supports client_secret auth.

    Raises RuntimeError if not supported.
    """
    base_url = _accounts_base_url(domain)
    res = _post_registration(base_url, {"action": "init"})
    methods = res.get("supported_auth_methods") or []
    if "client_secret" not in methods:
        raise RuntimeError(
            f"Feishu / Lark registration environment does not support client_secret auth. "
            f"Supported: {methods}"
        )


def _begin_registration(domain: str = "feishu") -> dict:
    """Start the device-code flow. Returns device_code, qr_url, user_code, interval, expire_in."""
    base_url = _accounts_base_url(domain)
    res = _post_registration(base_url, {
        "action": "begin",
        "archetype": "PersonalAgent",
        "auth_method": "client_secret",
        "request_user_info": "open_id",
    })
    device_code = res.get("device_code")
    if not device_code:
        raise RuntimeError("Feishu / Lark registration did not return a device_code")
    qr_url = res.get("verification_uri_complete", "")
    if "?" in qr_url:
        qr_url += "&from=hermes&tp=hermes"
    else:
        qr_url += "?from=hermes&tp=hermes"
    return {
        "device_code": device_code,
        "qr_url": qr_url,
        "user_code": res.get("user_code", ""),
        "interval": res.get("interval") or 5,
        "expire_in": res.get("expire_in") or 600,
    }


def _poll_registration(
    *,
    device_code: str,
    interval: int,
    expire_in: int,
    domain: str = "feishu",
) -> Optional[dict]:
    """Poll until the user scans the QR code, or timeout/denial.

    Returns dict with app_id, app_secret, domain, open_id on success.
    Returns None on failure.
    """
    deadline = time.monotonic() + expire_in
    current_domain = domain
    domain_switched = False
    poll_count = 0

    while time.monotonic() < deadline:
        base_url = _accounts_base_url(current_domain)
        try:
            res = _post_registration(base_url, {
                "action": "poll",
                "device_code": device_code,
                "tp": "ob_app",
            })
        except (URLError, OSError, json.JSONDecodeError):
            time.sleep(interval)
            continue

        poll_count += 1
        if poll_count == 1:
            print("  Fetching configuration results...", end="", flush=True)
        elif poll_count % 6 == 0:
            print(".", end="", flush=True)

        # Domain auto-detection
        user_info = res.get("user_info") or {}
        tenant_brand = user_info.get("tenant_brand")
        if tenant_brand == "lark" and not domain_switched:
            current_domain = "lark"
            domain_switched = True
            # Fall through — server may return credentials in this same response.

        # Success
        if res.get("client_id") and res.get("client_secret"):
            if poll_count > 0:
                print()  # newline after "Fetching configuration results..." dots
            return {
                "app_id": res["client_id"],
                "app_secret": res["client_secret"],
                "domain": current_domain,
                "open_id": user_info.get("open_id"),
            }

        # Terminal errors
        error = res.get("error", "")
        if error in ("access_denied", "expired_token"):
            if poll_count > 0:
                print()
            logger.warning("[Feishu onboard] Registration %s", error)
            return None

        # authorization_pending or unknown — keep polling
        time.sleep(interval)

    if poll_count > 0:
        print()
    logger.warning("[Feishu onboard] Poll timed out after %ds", expire_in)
    return None


try:
    import qrcode as _qrcode_mod
except (ImportError, TypeError):
    _qrcode_mod = None  # type: ignore[assignment]


def _render_qr(url: str) -> bool:
    """Try to render a QR code in the terminal. Returns True if successful."""
    if _qrcode_mod is None:
        return False
    try:
        qr = _qrcode_mod.QRCode()
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        return True
    except Exception:
        return False


def probe_bot(app_id: str, app_secret: str, domain: str) -> Optional[dict]:
    """Verify bot connectivity via /open-apis/bot/v3/info.

    Uses lark_oapi SDK when available, falls back to raw HTTP otherwise.
    Returns {"bot_name": ..., "bot_open_id": ...} on success, None on failure.

    Note: ``bot_open_id`` here is the bot's app-scoped open_id — the same ID
    that Feishu puts in @mention payloads.  It is NOT the app_id.
    """
    if FEISHU_AVAILABLE:
        return _probe_bot_sdk(app_id, app_secret, domain)
    return _probe_bot_http(app_id, app_secret, domain)


def _build_onboard_client(app_id: str, app_secret: str, domain: str) -> Any:
    """Build a lark Client for the given credentials and domain."""
    sdk_domain = LARK_DOMAIN if domain == "lark" else FEISHU_DOMAIN
    return (
        lark.Client.builder()
        .app_id(app_id)
        .app_secret(app_secret)
        .domain(sdk_domain)
        .log_level(lark.LogLevel.WARNING)
        .build()
    )


def _parse_bot_response(data: dict) -> Optional[dict]:
    # /bot/v3/info returns bot.app_name; legacy paths used bot_name — accept both.
    if data.get("code") != 0:
        return None
    bot = data.get("bot") or data.get("data", {}).get("bot") or {}
    return {
        "bot_name": bot.get("app_name") or bot.get("bot_name"),
        "bot_open_id": bot.get("open_id"),
    }


def _probe_bot_sdk(app_id: str, app_secret: str, domain: str) -> Optional[dict]:
    """Probe bot info using lark_oapi SDK."""
    try:
        client = _build_onboard_client(app_id, app_secret, domain)
        req = (
            BaseRequest.builder()
            .http_method(HttpMethod.GET)
            .uri("/open-apis/bot/v3/info")
            .token_types({AccessTokenType.TENANT})
            .build()
        )
        resp = client.request(req)
        content = getattr(getattr(resp, "raw", None), "content", None)
        if content is None:
            return None
        return _parse_bot_response(json.loads(content))
    except Exception as exc:
        logger.debug("[Feishu onboard] SDK probe failed: %s", exc)
        return None


def _probe_bot_http(app_id: str, app_secret: str, domain: str) -> Optional[dict]:
    """Fallback probe using raw HTTP (when lark_oapi is not installed)."""
    base_url = _onboard_open_base_url(domain)
    try:
        token_data = json.dumps({"app_id": app_id, "app_secret": app_secret}).encode("utf-8")
        token_req = Request(
            f"{base_url}/open-apis/auth/v3/tenant_access_token/internal",
            data=token_data,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(token_req, timeout=_ONBOARD_REQUEST_TIMEOUT_S) as resp:
            token_res = json.loads(resp.read().decode("utf-8"))

        access_token = token_res.get("tenant_access_token")
        if not access_token:
            return None

        bot_req = Request(
            f"{base_url}/open-apis/bot/v3/info",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        with urlopen(bot_req, timeout=_ONBOARD_REQUEST_TIMEOUT_S) as resp:
            bot_res = json.loads(resp.read().decode("utf-8"))

        return _parse_bot_response(bot_res)
    except (URLError, OSError, KeyError, json.JSONDecodeError) as exc:
        logger.debug("[Feishu onboard] HTTP probe failed: %s", exc)
        return None


def qr_register(
    *,
    initial_domain: str = "feishu",
    timeout_seconds: int = 600,
) -> Optional[dict]:
    """Run the Feishu / Lark scan-to-create QR registration flow.

    Returns on success::

        {
            "app_id": str,
            "app_secret": str,
            "domain": "feishu" | "lark",
            "open_id": str | None,
            "bot_name": str | None,
            "bot_open_id": str | None,
        }

    Returns None on expected failures (network, auth denied, timeout).
    Unexpected errors (bugs, protocol regressions) propagate to the caller.
    """
    try:
        return _qr_register_inner(initial_domain=initial_domain, timeout_seconds=timeout_seconds)
    except (RuntimeError, URLError, OSError, json.JSONDecodeError) as exc:
        logger.warning("[Feishu onboard] Registration failed: %s", exc)
        return None


def _qr_register_inner(
    *,
    initial_domain: str,
    timeout_seconds: int,
) -> Optional[dict]:
    """Run init → begin → poll → probe. Raises on network/protocol errors."""
    print("  Connecting to Feishu / Lark...", end="", flush=True)
    _init_registration(initial_domain)
    begin = _begin_registration(initial_domain)
    print(" done.")

    print()
    qr_url = begin["qr_url"]
    if _render_qr(qr_url):
        print(f"\n  Scan the QR code above, or open this URL directly:\n  {qr_url}")
    else:
        print(f"  Open this URL in Feishu / Lark on your phone:\n\n  {qr_url}\n")
        print("  Tip: pip install qrcode  to display a scannable QR code here next time")
    print()

    result = _poll_registration(
        device_code=begin["device_code"],
        interval=begin["interval"],
        expire_in=min(begin["expire_in"], timeout_seconds),
        domain=initial_domain,
    )
    if not result:
        return None

    # Probe bot — best-effort, don't fail the registration
    bot_info = probe_bot(result["app_id"], result["app_secret"], result["domain"])
    if bot_info:
        result["bot_name"] = bot_info.get("bot_name")
        result["bot_open_id"] = bot_info.get("bot_open_id")
    else:
        result["bot_name"] = None
        result["bot_open_id"] = None

    return result
