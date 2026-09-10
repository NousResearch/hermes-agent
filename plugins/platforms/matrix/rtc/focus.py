"""MatrixRTC (MSC4143) focus discovery and the LiveKit JWT exchange.

Three calls, verified end to end against a live Synapse + LiveKit deployment:

1. ``GET  {homeserver}/.well-known/matrix/client`` -> ``org.matrix.msc4143.rtc_foci``
   -> the ``livekit`` focus's ``livekit_service_url``
2. ``POST {homeserver}/_matrix/client/v3/user/{user_id}/openid/request_token``
   -> a short-lived OpenID token proving we own the Matrix account
3. ``POST {service_url}/sfu/get`` with that token -> the SFU websocket URL + a
   LiveKit JWT whose participant identity is ``{user_id}:{device_id}``

Two behaviours of the JWT service that callers depend on:

* The OpenID token is **single use** — mint a fresh one for every ``/sfu/get``.
* ``/sfu/get`` validates neither room membership nor room existence. Joining the
  media plane therefore needs no RTC membership state event — which is precisely why
  the bot could be audible on the SFU and absent from Element's widget at the same
  time. ``join.publish_call_membership`` writes that event, and needs the *service*
  URL discovered in step 1 (the one clients dial), not the SFU websocket URL,
  so the chain hands both back.

No function here logs or returns a credential in a message. Tokens are passed by
value and never formatted into an exception.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

WELL_KNOWN_PATH = "/.well-known/matrix/client"
RTC_FOCI_KEY = "org.matrix.msc4143.rtc_foci"


class MatrixRTCError(RuntimeError):
    """Focus discovery or the JWT exchange failed. Never carries a credential."""


async def discover_livekit_focus(session, homeserver: str) -> str:
    """Return the LiveKit JWT service URL advertised by *homeserver* (no trailing slash)."""
    url = homeserver.rstrip("/") + WELL_KNOWN_PATH
    async with session.get(url) as resp:
        if resp.status != 200:
            raise MatrixRTCError(f"{WELL_KNOWN_PATH} returned HTTP {resp.status}")
        body = await resp.json(content_type=None)
    foci = body.get(RTC_FOCI_KEY) or []
    for focus in foci:
        if focus.get("type") == "livekit" and focus.get("livekit_service_url"):
            return str(focus["livekit_service_url"]).rstrip("/")
    raise MatrixRTCError(
        f"no livekit focus in {RTC_FOCI_KEY} (found types: {[f.get('type') for f in foci]})")


async def request_openid_token(
        session, homeserver: str, user_id: str, access_token: str) -> dict[str, Any]:
    """Mint a fresh OpenID token for *user_id*. Single use — one per ``/sfu/get``."""
    url = f"{homeserver.rstrip('/')}/_matrix/client/v3/user/{user_id}/openid/request_token"
    async with session.post(url, headers={"Authorization": f"Bearer {access_token}"},
                            json={}) as resp:
        if resp.status != 200:
            raise MatrixRTCError(f"openid/request_token returned HTTP {resp.status}")
        body = await resp.json(content_type=None)
    if not body.get("access_token"):
        raise MatrixRTCError("openid/request_token returned no access_token")
    return body


async def request_sfu_credentials(
        session, service_url: str, room_id: str, openid: dict[str, Any],
        device_id: str) -> tuple[str, str]:
    """Exchange an OpenID token for ``(sfu_websocket_url, livekit_jwt)``."""
    payload = {"room": room_id, "openid_token": openid, "device_id": device_id}
    async with session.post(f"{service_url.rstrip('/')}/sfu/get", json=payload) as resp:
        if resp.status != 200:
            raise MatrixRTCError(f"/sfu/get returned HTTP {resp.status}")
        body = await resp.json(content_type=None)
    if not body.get("url") or not body.get("jwt"):
        raise MatrixRTCError(f"/sfu/get returned no url/jwt (keys: {sorted(body)})")
    return str(body["url"]), str(body["jwt"])


async def fetch_livekit_credentials(
        homeserver: str, user_id: str, access_token: str, room_id: str, device_id: str,
        session=None, ssl: Optional[Any] = None) -> tuple[str, str, str]:
    """Run the whole chain: ``(sfu_websocket_url, livekit_jwt, focus_service_url)``.

    Pass *session* to reuse the adapter's HTTP client. Otherwise one is created for the
    call; *ssl* is forwarded to its connector so a deployment behind a private CA can
    supply its own ``ssl.SSLContext`` instead of the module inventing a verify-off knob.
    """
    if session is not None:
        return await _fetch(session, homeserver, user_id, access_token, room_id, device_id)
    import aiohttp
    connector = aiohttp.TCPConnector(ssl=ssl) if ssl is not None else None
    async with aiohttp.ClientSession(connector=connector) as owned:
        return await _fetch(owned, homeserver, user_id, access_token, room_id, device_id)


async def _fetch(session, homeserver, user_id, access_token, room_id, device_id):
    service_url = await discover_livekit_focus(session, homeserver)
    logger.debug("MatrixRTC focus discovered: %s", service_url)
    openid = await request_openid_token(session, homeserver, user_id, access_token)
    sfu_url, jwt = await request_sfu_credentials(
        session, service_url, room_id, openid, device_id)
    logger.debug("MatrixRTC /sfu/get OK: url=%s jwt=<%d chars>", sfu_url, len(jwt))
    return sfu_url, jwt, service_url
