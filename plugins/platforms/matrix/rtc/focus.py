"""MatrixRTC focus discovery and the LiveKit token exchange.

Hermes uses the legacy ``/sfu/get`` endpoint of the MatrixRTC authorisation service
(``lk-jwt-service``), which pairs with the session form of call membership that
``membership`` writes. The exchange takes three requests:

1. ``GET {homeserver}/.well-known/matrix/client`` returns the
   ``org.matrix.msc4143.rtc_foci`` list, and the ``livekit`` entry gives the
   authorisation service URL.
2. ``POST {homeserver}/_matrix/client/v3/user/{user_id}/openid/request_token`` returns
   an OpenID token for the bot's account.
3. ``POST {service_url}/sfu/get`` with that token, the room id and the device id
   returns the SFU websocket URL and a LiveKit JWT. The service verifies the token
   through the homeserver's federation API, and the JWT's participant identity is
   ``{user_id}:{device_id}``. The user id comes from that verification; the device id
   is only claimed by the client.

``/sfu/get`` checks neither room membership nor room existence, so a token does not
put the bot in anyone's participant list. ``join`` publishes the call membership for
that, and the membership advertises the service URL from step 1, not the SFU URL.

No function here logs or returns a credential in a message. Tokens are passed by
value and never formatted into an exception.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

WELL_KNOWN_PATH = "/.well-known/matrix/client"
RTC_FOCI_KEY = "org.matrix.msc4143.rtc_foci"


class MatrixRTCError(RuntimeError):
    """Focus discovery, the token exchange or the call setup failed. Never carries a credential."""


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
    """Mint an OpenID token for *user_id*."""
    url = f"{homeserver.rstrip('/')}/_matrix/client/v3/user/{quote(user_id, safe='')}/openid/request_token"
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
