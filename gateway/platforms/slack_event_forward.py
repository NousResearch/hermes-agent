"""
Config-driven Slack Events API / interactivity HTTP forwarding.

When Socket Mode is enabled on a Slack app, channel message events and Block
Kit actions are delivered over the WebSocket and never reach the Events API /
interactivity Request URLs. External local services that implement Slack's
standard HTTP callbacks (e.g. a profile's order/approval server) would
therefore never see them.

A profile can declare forwards in config.yaml::

    slack:
      event_forwards:
        C0123456789: http://127.0.0.1:8787/slack/events
      action_forwards:
        wpc_: http://127.0.0.1:8787/slack/actions

``event_forwards`` maps a channel ID to a URL; ``message`` events from that
channel are wrapped in a standard ``event_callback`` envelope and POSTed as
JSON. ``action_forwards`` maps an ``action_id`` prefix to a URL; the full
``block_actions`` payload is POSTed form-encoded as ``payload=<json>``, the
same shape Slack uses for HTTP interactivity. Both are signed with the
standard Slack v0 scheme using ``SLACK_SIGNING_SECRET`` from the gateway's
environment, so the receiving service's existing signature verification
keeps working unchanged.
"""

import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

FORWARD_TIMEOUT_SECONDS = 15.0


def _parse_str_mapping(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    mapping: Dict[str, str] = {}
    for key, url in raw.items():
        clean_key = str(key).strip()
        clean_url = str(url).strip() if url else ""
        if clean_key and clean_url:
            mapping[clean_key] = clean_url
    return mapping


def parse_event_forwards(raw: Any) -> Dict[str, str]:
    """Normalize ``event_forwards`` config to ``{channel_id: url}``."""
    return _parse_str_mapping(raw)


def parse_action_forwards(raw: Any) -> Dict[str, str]:
    """Normalize ``action_forwards`` config to ``{action_id_prefix: url}``."""
    return _parse_str_mapping(raw)


def _sign(body: bytes, signing_secret: str, timestamp: Optional[str]) -> Tuple[str, str]:
    ts = timestamp if timestamp is not None else str(int(time.time()))
    base = b"v0:" + ts.encode("utf-8") + b":" + body
    signature = "v0=" + hmac.new(signing_secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    return ts, signature


def build_signed_event_request(
    event: Dict[str, Any],
    signing_secret: str,
    *,
    timestamp: Optional[str] = None,
) -> Tuple[bytes, Dict[str, str]]:
    """Wrap a Socket Mode message event in an Events API envelope and sign it."""
    body = json.dumps({"type": "event_callback", "event": event}).encode("utf-8")
    ts, signature = _sign(body, signing_secret, timestamp)
    headers = {
        "Content-Type": "application/json",
        "X-Slack-Request-Timestamp": ts,
        "X-Slack-Signature": signature,
    }
    return body, headers


def build_signed_action_request(
    payload: Dict[str, Any],
    signing_secret: str,
    *,
    timestamp: Optional[str] = None,
) -> Tuple[bytes, Dict[str, str]]:
    """Encode a block_actions payload the way Slack's HTTP interactivity does."""
    body = urllib.parse.urlencode({"payload": json.dumps(payload)}).encode("utf-8")
    ts, signature = _sign(body, signing_secret, timestamp)
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "X-Slack-Request-Timestamp": ts,
        "X-Slack-Signature": signature,
    }
    return body, headers


def _default_session_factory():
    import aiohttp

    return aiohttp.ClientSession(trust_env=True)


async def _post_signed(
    body: bytes,
    headers: Dict[str, str],
    url: str,
    *,
    timeout_seconds: float,
    session_factory: Optional[Callable[[], Any]],
) -> Dict[str, Any]:
    factory = session_factory or _default_session_factory
    try:
        async with factory() as session:
            kwargs: Dict[str, Any] = {"data": body, "headers": headers}
            if session_factory is None:
                import aiohttp

                kwargs["timeout"] = aiohttp.ClientTimeout(total=timeout_seconds)
            async with session.post(url, **kwargs) as resp:
                status = resp.status
                text = await resp.text()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    if status != 200:
        return {"ok": False, "status": status, "error": text[:500] or f"HTTP {status}"}
    return {"ok": True, "status": status, "body": text[:500]}


async def forward_event(
    event: Dict[str, Any],
    url: str,
    signing_secret: str,
    *,
    timeout_seconds: float = FORWARD_TIMEOUT_SECONDS,
    session_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Relay a message event to ``url`` as a signed Events API callback."""
    if not signing_secret:
        return {"ok": False, "error": "SLACK_SIGNING_SECRET is not configured in the gateway environment"}
    body, headers = build_signed_event_request(event, signing_secret)
    return await _post_signed(
        body, headers, url, timeout_seconds=timeout_seconds, session_factory=session_factory
    )


async def forward_block_actions(
    payload: Dict[str, Any],
    url: str,
    signing_secret: str,
    *,
    timeout_seconds: float = FORWARD_TIMEOUT_SECONDS,
    session_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Relay a block_actions payload to ``url`` as signed HTTP interactivity."""
    if not signing_secret:
        return {"ok": False, "error": "SLACK_SIGNING_SECRET is not configured in the gateway environment"}
    body, headers = build_signed_action_request(payload, signing_secret)
    return await _post_signed(
        body, headers, url, timeout_seconds=timeout_seconds, session_factory=session_factory
    )
