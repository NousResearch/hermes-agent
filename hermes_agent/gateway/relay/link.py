"""Account linking to the shared Team Gateway bot, done from the instance itself.

A relay-fronted deployment (Hermes Cloud with Team Gateway) reaches Telegram, Discord and Slack
only through the connector's SHARED bot: the gateway turns every direct adapter off
(``config_env._relay``), and the connector routes a chat account's messages to this instance once
that account is linked (``/link <code>`` DMed to the shared bot). The Portal can mint those codes;
this module lets the instance's own dashboard and Desktop do it too, so a user who opens Channels
on the instance gets a working setup instead of a bot of their own the relay will never run.

Codes are minted with the gateway's per-gateway secret, which lives only in the gateway process
(``self_provision_relay``), so the dashboard reaches these calls through the ``relay-link``
control-socket verb. Wire: ``POST /manage/link``, ``/manage/link/state``, ``/manage/link-status``
and ``/manage/unlink`` on the connector, all authenticated with the WS-upgrade bearer.

A Portal-managed agent (the connector's ``/manage/access/list`` answers ``managedBy: portal``) is
the exception: there the Portal access setting is the only input, so a code minted with the
gateway bearer carries no owner and its redeem binds nothing. For those agents linking happens in
the Portal, which mints owner-attested codes; the instance only lists the linked accounts.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

_HTTP_TIMEOUT_S = 10.0
# Longer than one connector call so the dashboard sees the connector's answer, not a socket timeout.
CONTROL_TIMEOUT_S = _HTTP_TIMEOUT_S + 5.0


class RelayLinkError(RuntimeError):
    """A link call that cannot be served; ``code`` is a stable slug the surfaces translate."""

    def __init__(self, code: str, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def _bot_entry(platform: str) -> dict:
    from hermes_agent.gateway.relay import _relay_bot_ids_map

    entry = _relay_bot_ids_map().get(platform)
    return entry if isinstance(entry, dict) else {}


def shared_bot_url(platform: str, code: Optional[str] = None) -> Optional[str]:
    """Where to DM the shared bot, from the ``GATEWAY_RELAY_BOT_IDS`` entry the Portal stamps
    (same rule as the Portal's ``buildBotDmUrl``). None when the entry names no reachable handle."""
    entry = _bot_entry(platform)
    if platform == "telegram":
        username = str(entry.get("username") or "").strip().lstrip("@")
        return f"https://t.me/{username}" if username else None
    if platform == "discord":
        bot_id = str(entry.get("botId") or "").strip()
        return f"https://discord.com/users/{bot_id}" if bot_id else None
    if platform == "whatsapp":
        number = "".join(ch for ch in str(entry.get("waNumber") or entry.get("botId") or "") if ch.isdigit())
        if not number:
            return None
        from urllib.parse import quote

        return f"https://wa.me/{number}?text={quote(f'/link {code}' if code else '/link')}"
    return None


def relay_channel_view() -> tuple[frozenset[str], bool]:
    """``(platforms the shared bots carry here, direct adapters are switched off)``.

    The second half mirrors ``config_env._relay``: only the ``GATEWAY_RELAY_URL`` env stamp without
    ``GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS`` makes the relay exclusive. Reads this process's env, which
    on a hosted instance is the container stamp every process shares."""
    from hermes_agent.gateway.relay import relay_fronted_platforms, relay_url
    from utils import is_truthy_value

    if not relay_url():
        return frozenset(), False
    exclusive = bool(os.environ.get("GATEWAY_RELAY_URL", "").strip()) and not is_truthy_value(
        os.environ.get("GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS"))
    return frozenset(relay_fronted_platforms()), exclusive


def _bearer_and_base() -> tuple[str, str]:
    from hermes_agent.gateway.relay import _connector_url, relay_connection_auth, relay_url
    from hermes_agent.gateway.relay.auth import make_upgrade_token

    dial_url = relay_url()
    if not dial_url:
        raise RelayLinkError("relay_not_configured", "Team Gateway is not configured on this instance.")
    gateway_id, secret = relay_connection_auth()
    if not (gateway_id and secret):
        raise RelayLinkError(
            "relay_not_provisioned",
            "The gateway has not connected to Team Gateway yet. Wait a minute and try again.")
    return make_upgrade_token(gateway_id, secret), _connector_url(dial_url, "")


# Connector HTTP status -> (slug, sentence the surfaces show). Anything else 4xx is a refusal; 5xx/408/429
# mean "try again" (``connector_unavailable``).
_HTTP_REFUSALS: dict[int, tuple[str, str]] = {
    401: ("relay_not_authorized",
          "Team Gateway no longer accepts this instance's credentials. Restart the gateway to reconnect it."),
    403: ("platform_not_provisioned", "Team Gateway is not set up for this platform on this instance."),
    409: ("instance_not_bound", "This instance is not bound to Team Gateway yet. Restart the gateway and try again."),
}


def _http_error(code: int, reason: str) -> RelayLinkError:
    if code in _HTTP_REFUSALS:
        slug, message = _HTTP_REFUSALS[code]
        return RelayLinkError(slug, message, code)
    if code >= 500 or code in (408, 429):
        return RelayLinkError("connector_unavailable", "Team Gateway is not answering right now. Try again shortly.", code)
    return RelayLinkError("connector_refused", reason or f"Team Gateway answered HTTP {code}.", code)


def _post(path: str, body: dict) -> dict:
    from hermes_agent.gateway.relay import _json_post
    from hermes_agent.gateway.relay.http_diagnostics import connector_reason, read_error_body, redact_excerpt

    token, base = _bearer_and_base()
    try:
        with _json_post(f"{base}{path}", token, body, _HTTP_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        reason = redact_excerpt(connector_reason(read_error_body(exc)), (token,))
        raise _http_error(exc.code, reason) from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RelayLinkError("connector_unreachable", f"Could not reach Team Gateway: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def _get(path: str) -> dict:
    from hermes_agent.gateway.relay.http_diagnostics import connector_reason, read_error_body, redact_excerpt

    token, base = _bearer_and_base()
    req = urllib.request.Request(f"{base}{path}", method="GET", headers={
        "Authorization": f"Bearer {token}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        reason = redact_excerpt(connector_reason(read_error_body(exc)), (token,))
        raise _http_error(exc.code, reason) from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RelayLinkError("connector_unreachable", f"Could not reach Team Gateway: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def portal_agents_url() -> str:
    """The Portal this instance signs in to (operator override, stored login, then default)."""
    from hermes_cli.auth import get_provider_auth_state
    from hermes_cli.auth_nous import _nous_portal_base_url

    return f"{_nous_portal_base_url(get_provider_auth_state('nous') or {})}/agents"


def managed_by() -> str:
    """``portal`` when the Portal access setting governs this agent, else ``connector``.

    A connector that does not serve access grants (no route: 404) has no Portal setting either."""
    try:
        payload = _get("/manage/access/list")
    except RelayLinkError as exc:
        if exc.status == 404:
            return "connector"
        raise
    return "portal" if payload.get("managedBy") == "portal" else "connector"


def _managed_by_for_listing() -> str:
    # The list must not disappear when only this question fails: minting re-asks and refuses.
    try:
        return managed_by()
    except RelayLinkError:
        return "unknown"


def request_link(platform: str) -> dict[str, Any]:
    """Mint a one-time code that binds the chat account which DMs it to this instance."""
    from hermes_agent.gateway.relay import relay_fronted_platforms

    if platform not in relay_fronted_platforms():
        raise RelayLinkError("platform_not_fronted", f"Team Gateway does not carry {platform} on this instance.")
    if managed_by() == "portal":
        raise RelayLinkError(
            "managed_by_portal",
            "Accounts for this agent are linked from Nous Portal: open the agent there and use Message platforms.",
            409)
    payload = _post("/manage/link", {"platform": platform})
    code = str(payload.get("code") or "").strip()
    if not code:
        raise RelayLinkError("connector_refused", "Team Gateway did not issue a link code.")
    return {"platform": platform, "code": code, "command": f"/link {code}",
            "expires_at": payload.get("expiresAt"), "bot_url": shared_bot_url(platform, code)}


def link_code_state(code: str) -> dict[str, Any]:
    """``pending`` until redeemed; ``inactive`` once used or expired; ``refused`` with a reason."""
    payload = _post("/manage/link/state", {"code": code})
    return {"state": payload.get("state") or "inactive", "reason": payload.get("reason")}


def link_status() -> dict[str, Any]:
    """Accounts linked to this instance, per fronted platform, plus each shared bot's DM link."""
    from hermes_agent.gateway.relay import relay_fronted_platforms

    platforms = sorted(relay_fronted_platforms())
    payload = _post("/manage/link-status", {"platforms": platforms}) if platforms else {}
    accounts = {row.get("platform"): row.get("accounts") or []
                for row in payload.get("statuses") or [] if isinstance(row, dict)}
    return {"managed_by": _managed_by_for_listing() if platforms else "connector", "portal_url": portal_agents_url(),
            "platforms": [{"platform": p, "bot_url": shared_bot_url(p), "accounts": accounts.get(p, [])}
                          for p in platforms]}


def unlink(platform: str, platform_user_id: str) -> dict[str, Any]:
    _post("/manage/unlink", {"platform": platform, "platformUserId": platform_user_id})
    return {"unlinked": True}


_ACTIONS: dict[str, Callable[[dict], dict[str, Any]]] = {
    "request": lambda p: request_link(str(p.get("platform") or "")),
    "state": lambda p: link_code_state(str(p.get("code") or "")),
    "status": lambda p: link_status(),
    "unlink": lambda p: unlink(str(p.get("platform") or ""), str(p.get("platform_user_id") or "")),
}


def relay_link_verb(params: dict) -> dict[str, Any]:
    """Control-socket ``relay-link`` verb: ``{"action": request|state|status|unlink, ...}``.

    Answers ``{"ok": True, ...}`` or ``{"ok": False, "error": slug, "message", "status"}`` so the
    dashboard can tell "not ready yet" from a connector refusal; never raises."""
    action = _ACTIONS.get(str(params.get("action") or ""))
    if action is None:
        return {"ok": False, "error": "unknown_action", "message": "Unknown relay-link action."}
    try:
        return {"ok": True, **action(params)}
    except RelayLinkError as exc:
        return {"ok": False, "error": exc.code, "message": str(exc), "status": exc.status}
