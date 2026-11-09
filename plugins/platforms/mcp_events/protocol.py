"""MCP Events subscriber protocol: JSON-RPC framing for the *proposed* MCP Events
extension (``events/list``, ``events/subscribe``, ``events/unsubscribe``) against a
remote event *emitter*, plus Standard Webhooks verification for inbound deliveries
and the local subscription store.

Hermes is the subscriber here, not the emitter: the agent calls ``events/subscribe``
on an emitter's MCP endpoint, hands it our webhook callback URL and signing secret,
and the emitter POSTs signed event deliveries to us. Stdlib only.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
import uuid

logger = logging.getLogger(__name__)

# The protocol revision the proposed MCP Events extension is defined against.
PROTOCOL_VERSION = "2026-07-28"

# Per the documented event model: one event per delivery, hard ceiling.
MAX_EVENT_BYTES = 262_144

# Subscriptions live outside the context-compaction pipeline so restarts and
# compactions never silently drop them (same rationale as a2a_conversations/).
_SUBSCRIPTIONS_FILE = "mcp_events_subscriptions.json"

# Emitter callback verification: how long a signed delivery stays valid.
DEFAULT_TIMESTAMP_SKEW = 300


def _mcp_headers(method: str, extra: dict | None = None) -> dict[str, str]:
    """2026-07-28 transport: the ``Mcp-Method`` header must match the JSON-RPC body,
    otherwise the emitter rejects the call with -32020. ``extra`` adds emitter
    auth headers (e.g. a bearer token from config) — never logged."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Mcp-Method": method,
        "MCP-Protocol-Version": PROTOCOL_VERSION,
    }
    if extra:
        headers.update({str(k): str(v) for k, v in extra.items()})
    return headers


def _rpc_request(method: str, params: dict) -> dict:
    # 2026-07-28: per-request metadata goes in params._meta; protocolVersion and
    # clientCapabilities are required, clientInfo SHOULD be sent.
    return {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": method,
        "params": {
            **params,
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
                "io.modelcontextprotocol/clientCapabilities": {},
                "io.modelcontextprotocol/clientInfo": {"name": "hermes-agent", "version": "mcp-events-plugin"},
            },
        },
    }


def _post_json(url: str, payload: dict, timeout: float = 15.0, headers: dict | None = None) -> dict:
    """POST a JSON-RPC call to an emitter's MCP endpoint; returns the ``result`` dict.

    Raises on transport errors and on JSON-RPC ``error`` responses — the caller
    decides what the agent sees.
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=_mcp_headers(payload["method"], headers), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read(MAX_EVENT_BYTES + 1024)
    except Exception as e:
        raise RuntimeError(f"emitter request failed: {e}") from e
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"emitter returned non-JSON: {e}") from e
    if isinstance(envelope, dict) and envelope.get("error"):
        err = envelope["error"]
        raise RuntimeError(f"emitter error {err.get('code')}: {err.get('message')}")
    result = (envelope or {}).get("result")
    if not isinstance(result, dict):
        raise RuntimeError("emitter returned no result object")
    return result


def server_supports_events(mcp_url: str, timeout: float = 15.0, headers: dict | None = None) -> bool:
    """Best-effort ``server/discover`` probe: True when the emitter advertises the
    ``events`` capability. A False here is advisory — subscribe() still tries."""
    try:
        result = _post_json(mcp_url, _rpc_request("server/discover", {}), timeout=timeout, headers=headers)
    except Exception as e:
        logger.debug("MCP Events: server/discover probe failed: %s", e)
        return False
    caps = result.get("capabilities") or {}
    return isinstance(caps, dict) and "events" in caps


def list_events(mcp_url: str, timeout: float = 15.0, headers: dict | None = None) -> list[dict]:
    """``events/list`` against the emitter; returns [{name, description, ...}]."""
    result = _post_json(mcp_url, _rpc_request("events/list", {}), timeout=timeout, headers=headers)
    events = result.get("events") or []
    return [e for e in events if isinstance(e, dict) and e.get("name")]


def _iso_to_epoch(value) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    from datetime import datetime
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def subscribe(mcp_url: str, event: str, callback_url: str, secret: str,
              filter_args: dict | None = None, timeout: float = 20.0,
              headers: dict | None = None) -> dict:
    """``events/subscribe``: registers our webhook callback for ``event``.

    ``secret`` is the ``whsec_``-prefixed signing secret the emitter will use to
    sign deliveries — the documented ChatGPT model: the subscriber supplies the
    secret, the emitter signs with it, we verify on receipt.
    Returns the subscription record (id, expires_at, ...).
    """
    params: dict = {
        "name": event,
        "arguments": filter_args or {},
        "delivery": {"mode": "webhook", "url": callback_url, "secret": secret},
    }
    result = _post_json(mcp_url, _rpc_request("events/subscribe", params), timeout=timeout, headers=headers)
    sub_id = str(result.get("id") or "")
    if not sub_id:
        raise RuntimeError("emitter accepted the subscription but returned no id")
    return {
        "id": sub_id,
        "emitter_url": mcp_url,
        "event": event,
        "callback_url": callback_url,
        "filter": filter_args or {},
        "created_at": time.time(),
        # refreshBefore is an ISO 8601 grant (null = no expiry); stored as epoch seconds for the refresh check.
        "expires_at": _iso_to_epoch(result.get("refreshBefore")),
        "challenge": result.get("challenge"),  # answered by the adapter when present
    }


def unsubscribe(mcp_url: str, record: dict, timeout: float = 15.0, headers: dict | None = None) -> bool:
    """``events/unsubscribe`` by the subscription key (name, arguments, delivery URL);
    the derived id is not accepted as input. True when the emitter acknowledged."""
    params = {"name": record["event"], "arguments": record.get("filter") or {},
              "delivery": {"url": record["callback_url"]}}
    try:
        _post_json(mcp_url, _rpc_request("events/unsubscribe", params), timeout=timeout, headers=headers)
        return True
    except Exception as e:
        logger.debug("MCP Events: unsubscribe failed: %s", e)
        return False


def _decode_secret(secret: str) -> bytes:
    """``whsec_<base64>`` -> raw bytes. Rejects malformed secrets fail-closed."""
    raw = (secret or "").strip()
    if raw.startswith("whsec_"):
        raw = raw[len("whsec_"):]
    try:
        key = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("webhook secret is not valid base64") from e
    if not 24 <= len(key) <= 64:
        raise ValueError("webhook secret must decode to 24-64 bytes")
    return key


def sign_delivery(secret: str, webhook_id: str, timestamp: str, body: bytes) -> str:
    """Standard Webhooks signature: ``v1,<base64(HMAC-SHA256(secret, id.ts.body))>``."""
    key = _decode_secret(secret)
    signed = f"{webhook_id}.{timestamp}.".encode("utf-8") + body
    digest = hmac.new(key, signed, hashlib.sha256).digest()
    return "v1," + base64.b64encode(digest).decode("ascii")


def verify_delivery(secret: str, webhook_id: str, timestamp: str, signature_header: str,
                    body: bytes, skew_seconds: int = DEFAULT_TIMESTAMP_SKEW) -> bool:
    """Verify an inbound event delivery. All failure modes return False (fail closed);
    the adapter logs the reason and answers the HTTP status, never the why."""
    if not (secret and webhook_id and timestamp and signature_header and body):
        return False
    try:
        ts = int(timestamp)
    except (ValueError, TypeError):
        return False
    if abs(time.time() - ts) > max(1, skew_seconds):
        return False  # replayed or clock-skewed delivery
    if len(body) > MAX_EVENT_BYTES:
        return False
    try:
        expected = sign_delivery(secret, webhook_id, timestamp, body)
    except ValueError:
        return False
    expected_sig = expected.split(",", 1)[1]
    # The header may carry several space-separated signatures (secret rotation).
    for candidate in signature_header.split():
        presented = candidate.split(",", 1)[1] if "," in candidate else candidate
        if hmac.compare_digest(presented, expected_sig):
            return True
    return False


class SubscriptionStore:
    """Thread-safe JSON subscription store under the profile's Hermes home.
    Keyed by subscription id; survives restarts and compaction."""

    def __init__(self, home_dir: str | None = None):
        self._lock = threading.Lock()
        self._home = home_dir  # resolved lazily so tests can inject a temp dir
        self._subs: dict[str, dict] = {}
        self._loaded = False

    def _path(self) -> str:
        home = self._home
        if not home:
            from hermes_constants import get_hermes_home
            home = str(get_hermes_home())
            self._home = home
        os.makedirs(home, exist_ok=True)
        return os.path.join(home, _SUBSCRIPTIONS_FILE)

    def _ensure_loaded(self) -> None:
        # Reload when the file changed: the tools write through their own instances.
        try:
            mtime = os.path.getmtime(self._path())
        except OSError:
            mtime = None
        if self._loaded and mtime == getattr(self, "_mtime", None):
            return
        self._mtime = mtime
        try:
            with open(self._path(), encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._subs = {k: v for k, v in data.items() if isinstance(v, dict)}
        except (OSError, ValueError):
            self._subs = {}
        self._loaded = True

    def _save(self) -> None:
        path = self._path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._subs, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)

    def add(self, record: dict) -> None:
        with self._lock:
            self._ensure_loaded()
            self._subs[str(record["id"])] = record
            self._save()

    def remove(self, subscription_id: str) -> bool:
        with self._lock:
            self._ensure_loaded()
            if subscription_id in self._subs:
                del self._subs[subscription_id]
                self._save()
                return True
            return False

    def get(self, subscription_id: str) -> dict | None:
        with self._lock:
            self._ensure_loaded()
            rec = self._subs.get(subscription_id)
            return dict(rec) if rec else None

    def list(self) -> list[dict]:
        with self._lock:
            self._ensure_loaded()
            return [dict(v) for v in self._subs.values()]

    def find_by_callback(self, callback_url: str) -> dict | None:
        """Match an inbound delivery's path to its subscription (callback URLs embed the id)."""
        with self._lock:
            self._ensure_loaded()
            for rec in self._subs.values():
                if rec.get("callback_url") == callback_url:
                    return dict(rec)
            return None

    def resolve(self, ref: str) -> dict | None:
        """Find a subscription by emitter id, local id, or event name (first match)."""
        with self._lock:
            self._ensure_loaded()
            if ref in self._subs:
                return dict(self._subs[ref])
            for rec in self._subs.values():
                if rec.get("local_id") == ref:
                    return dict(rec)
            for rec in self._subs.values():
                if rec.get("event") == ref:
                    return dict(rec)
            return None

    def expired(self, now: float | None = None) -> list[dict]:
        """Subscriptions past ``expires_at`` — the adapter re-subscribes or drops them."""
        now = time.time() if now is None else now
        with self._lock:
            self._ensure_loaded()
            return [dict(v) for v in self._subs.values()
                    if isinstance(v.get("expires_at"), (int, float)) and v["expires_at"] <= now]


def new_webhook_secret() -> str:
    """Generate a fresh ``whsec_``-prefixed signing secret (32 random bytes)."""
    import secrets
    return "whsec_" + base64.b64encode(secrets.token_bytes(32)).decode("ascii")
