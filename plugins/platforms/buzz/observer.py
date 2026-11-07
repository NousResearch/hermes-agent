"""Buzz observer telemetry for the Hermes native adapter (opt-in).

Publishes what the Buzz Desktop activity view consumes from buzz-acp:

- kind:24200 observer frames (``turn_started`` / ``turn_liveness`` /
  ``turn_completed`` / ``harness_started``), NIP-44 v2 encrypted to the agent
  owner taken from the NIP-OA auth tag;
- kind:20002 typing indicators;
- kind:20001 presence (``online`` heartbeat, ``offline`` on close).

Design: telemetry is **state, not a queue**. The adapter records facts (a turn
started, a turn ended, the agent is typing); a single loop owns the relay
socket and, once per second, publishes whatever that state says is due. Frames
are built at publish time, so they are always fresh and nothing accumulates.

Everything is best-effort. The consumer tolerates loss by design: Desktop
prunes a turn ~25 s after its last liveness frame and recreates one from a
liveness frame it never saw start; presence expires on the relay TTL. Nothing
here retries past the next tick, and ``close()`` is bounded by its timeout.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import itertools
import json
import logging
import math
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from . import nostr_auth as _na
except ImportError:  # loaded as a bare module (tests, file-path loader)
    import importlib.util
    import sys
    from pathlib import Path

    _path = Path(__file__).with_name("nostr_auth.py")
    _spec = importlib.util.spec_from_file_location("hermes_buzz_observer_nostr_auth", _path)
    _na = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _na
    _spec.loader.exec_module(_na)

public_key_hex = _na.public_key_hex

OBSERVER_KIND = 24200
TYPING_KIND = 20002
PRESENCE_KIND = 20001

TICK_S = 1.0  # at most one 24200 frame per tick (buzz-acp OBSERVER_PUBLISH_TICK)
LIVENESS_EVERY_S = 10.0  # Desktop LIVENESS_INTERVAL_MS
TYPING_EVERY_S = 3.0
PRESENCE_EVERY_S = 60.0  # relay presence TTL is 180 s
OK_TIMEOUT_S = 5.0
RECONNECT_MAX_S = 30.0
MAX_OPEN_TURNS = 256  # only a core bug (start without complete) can reach this
# Turns per 24200 frame. Each turn adds at most two events of bounded shape (~0.5 KB
# serialized), so 32 keeps a batch far below NIP-44's 65535-byte plaintext limit;
# the rest of the channel goes out on the next ticks.
MAX_TURNS_PER_FRAME = 32

# ── NIP-44 v2 (validated against the paulmillr/nip44 official vectors) ───────

_FIELD_P = 2**256 - 2**32 - 977


def _lift_x(pub_hex: str) -> tuple[int, int]:
    """x-only pubkey -> (x, even y). Raises ValueError when x is not on the curve."""
    x = int(pub_hex[:64], 16)
    y_sq = (pow(x, 3, _FIELD_P) + 7) % _FIELD_P
    y = pow(y_sq, (_FIELD_P + 1) // 4, _FIELD_P)
    if (y * y - y_sq) % _FIELD_P != 0:
        raise ValueError("pubkey x not on curve")
    return x, (_FIELD_P - y if y % 2 else y)


def _valid_pubkey_hex(pub_hex: str) -> bool:
    """64 hex chars, below the secp256k1 field prime, and on the curve.

    NIP-44 silently derives a wrong key from a malformed pubkey, so a bad owner
    in BUZZ_AUTH_TAG must fail here, loudly, rather than encrypt to nobody."""
    if len(pub_hex) != 64:
        return False
    try:
        if int(pub_hex, 16) >= _FIELD_P:
            return False
        _lift_x(pub_hex)
    except ValueError:
        return False
    return True


def get_conversation_key(priv: str, other_pub_hex: str) -> bytes:
    if not _valid_pubkey_hex(other_pub_hex):
        raise ValueError(f"invalid pubkey for NIP-44: {other_pub_hex[:16]}…")
    sx, _sy = _na._point_multiply(_na.decode_private_key(priv), _lift_x(other_pub_hex))
    return hmac.new(b"nip44-v2", sx.to_bytes(32, "big"), hashlib.sha256).digest()


def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    okm = block = b""
    counter = 1
    while len(okm) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        okm += block
        counter += 1
    return okm[:length]


def _get_message_keys(conversation_key: bytes, nonce: bytes) -> tuple[bytes, bytes, bytes]:
    okm = _hkdf_expand(conversation_key, nonce, 76)
    return okm[:32], okm[32:44], okm[44:76]


def _calc_padded_len(n: int) -> int:
    if n <= 32:
        return 32
    next_power = 1 << (n - 1).bit_length()
    chunk = 32 if next_power <= 256 else next_power // 8
    return chunk * (((n - 1) // chunk) + 1)


def _chacha(key: bytes, nonce12: bytes, data: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    enc = Cipher(algorithms.ChaCha20(key, b"\x00" * 4 + nonce12), mode=None).encryptor()
    return enc.update(data) + enc.finalize()


def nip44_encrypt(plaintext: str, conversation_key: bytes, nonce: Optional[bytes] = None) -> str:
    nonce = nonce or secrets.token_bytes(32)
    raw = plaintext.encode()
    if not 1 <= len(raw) <= 65535:
        raise ValueError(f"nip44: plaintext length {len(raw)} outside 1..65535")
    ck, cn, mk = _get_message_keys(conversation_key, nonce)
    padded = len(raw).to_bytes(2, "big") + raw + b"\x00" * (_calc_padded_len(len(raw)) - len(raw))
    ct = _chacha(ck, cn, padded)
    mac = hmac.new(mk, nonce + ct, hashlib.sha256).digest()
    return base64.b64encode(b"\x02" + nonce + ct + mac).decode()


def nip44_decrypt(payload: str, conversation_key: bytes) -> str:
    """Inverse of :func:`nip44_encrypt`; checks version, MAC and padding."""
    raw = base64.b64decode(payload, validate=True)
    if not raw or raw[0] != 2:
        raise ValueError(f"unknown encryption version {raw[0] if raw else 'empty'}")
    nonce, ct, mac = raw[1:33], raw[33:-32], raw[-32:]
    ck, cn, mk = _get_message_keys(conversation_key, nonce)
    if not hmac.compare_digest(mac, hmac.new(mk, nonce + ct, hashlib.sha256).digest()):
        raise ValueError("invalid MAC")
    padded = _chacha(ck, cn, ct)
    n = int.from_bytes(padded[:2], "big")
    out = padded[2:2 + n]
    if not 1 <= n <= 65535 or len(out) != n or len(padded) != 2 + _calc_padded_len(n):
        raise ValueError("invalid padding")
    return out.decode()


# ── State ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def _ws_url(relay_url: str) -> str:
    url = (relay_url or "").strip()
    for http, ws in (("http://", "ws://"), ("https://", "wss://")):
        if url.startswith(http):
            return ws + url[len(http):]
    return url if url.startswith(("ws://", "wss://")) else "wss://" + url


@dataclass
class _Turn:
    token: object  # held so its id() cannot be reused while the turn is open
    channel_id: str
    turn_id: str
    started_at: str
    start_payload: dict
    thread_root: Optional[str]
    start_sent: bool = False
    last_sent: float = 0.0  # monotonic time of the last frame for this turn
    end_payload: Optional[dict] = None  # set once the turn is over


@dataclass
class _Typing:
    requested: float  # monotonic time of the latest send_typing()
    sent: float = -math.inf


def _connection_closed():
    from websockets.exceptions import ConnectionClosed
    return ConnectionClosed


class _AuthRejected(Exception):
    """The relay answered our NIP-42 AUTH with OK=false: a configuration problem."""


@dataclass
class _Due:
    """What one tick publishes; ``commit()`` marks it sent once the relay answered."""
    channel_id: Optional[str] = None
    events: list = field(default_factory=list)
    turns: list = field(default_factory=list)  # (key, turn, ends_it) covered by ``events``
    typing: list = field(default_factory=list)  # (key, _Typing) pairs


class ObserverPublisher:
    """Owns the telemetry state and the one socket that publishes it.

    The adapter calls the synchronous recorders (never raise, never block);
    only ``_run`` touches the socket."""

    def __init__(self, *, private_key: str, auth_tag_json: str, relay_url: str,
                 session_id: Optional[str] = None):
        tag = _na.parse_auth_tag(auth_tag_json, "BUZZ_AUTH_TAG")
        self.owner = str(tag[1]).lower()
        self._priv = private_key
        self._auth_tag_json = auth_tag_json
        self.agent_pub = public_key_hex(private_key)
        self._conversation_key = get_conversation_key(private_key, self.owner)  # ValueError on a bad owner
        self.ws_url = _ws_url(relay_url)
        self.session_id = session_id or f"hermes-{uuid.uuid4().hex[:12]}"
        self._seq = itertools.count(1)
        self._turns: dict[int, _Turn] = {}
        self._typing: dict[tuple[str, Optional[str]], _Typing] = {}
        self._closing = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._announced = False
        self.published = 0

    # ── recorders (adapter-facing) ──────────────────────────────────────────

    def turn_started(self, token: object, *, channel_id: str, message_id: Optional[str],
                     thread_root: Optional[str], is_dm: bool) -> None:
        """Record a turn identified by ``token`` — the MessageEvent object, which the
        core hands unchanged to both the start and the complete hook."""
        key = id(token)
        if key in self._turns or self._closing.is_set():
            return
        if len(self._turns) >= MAX_OPEN_TURNS:
            dropped = self._turns.pop(next(iter(self._turns)))
            logger.warning("Buzz observer: %d open turns, forgetting the oldest (%s)",
                           MAX_OPEN_TURNS, dropped.turn_id)
        self._turns[key] = _Turn(
            token=token, channel_id=channel_id, turn_id=str(uuid.uuid4()), started_at=_now_iso(),
            thread_root=thread_root,
            start_payload={"source": "dm" if is_dm else "channel",
                           "triggeringEventIds": [message_id] if message_id else [],
                           "threadRootEventId": thread_root})

    def turn_completed(self, token: object, outcome: str, reason: Optional[str] = None) -> None:
        turn = self._turns.get(id(token))
        if turn is None or turn.end_payload is not None:
            return
        turn.end_payload = {"outcome": outcome, **({"reason": reason} if reason else {})}

    def typing(self, channel_id: str, thread_root: Optional[str] = None) -> None:
        if self._closing.is_set():
            return
        entry = self._typing.setdefault((channel_id, thread_root), _Typing(requested=0.0))
        entry.requested = time.monotonic()

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="buzz-observer")

    async def close(self, timeout: float = 2.5) -> None:
        """Best-effort goodbye, bounded by ``timeout``. Only the caller's own
        cancellation propagates.

        Open turns become ``cancelled``; the loop gets one pass to publish
        them and ``offline`` on the socket it already has, then is cancelled.
        Whatever does not make it decays on the consumer side."""
        task, self._task = self._task, None
        if task is None:
            return
        for turn in self._turns.values():
            if turn.end_payload is None:
                turn.end_payload = {"outcome": "cancelled", "reason": "gateway_shutdown"}
        self._closing.set()
        try:
            await asyncio.wait({task}, timeout=max(timeout - 0.5, 0.1))
        finally:  # also when the caller is cancelled: never leave the socket owner running
            if not task.done():
                task.cancel()
        await asyncio.wait({task}, timeout=0.5)
        logger.info("Buzz observer: closed after %d published events", self.published)

    # ── what is due ─────────────────────────────────────────────────────────

    def _due(self, now: float) -> _Due:
        """Pure read of the state: the 24200 events for ONE channel (buzz-acp
        batches per channel) plus every typing indicator that is due."""
        due = _Due()
        for key, turn in self._turns.items():
            if len(due.turns) >= MAX_TURNS_PER_FRAME:
                break
            if due.channel_id is not None and turn.channel_id != due.channel_id:
                continue
            events = []
            if not turn.start_sent:
                events.append(("turn_started", turn.start_payload))
            ends = turn.end_payload is not None
            if ends:
                events.append(("turn_completed", turn.end_payload))
            elif turn.start_sent and now - turn.last_sent >= LIVENESS_EVERY_S:
                events.append(("turn_liveness",
                               {"threadRootEventId": turn.thread_root} if turn.thread_root else {}))
            if events:
                due.channel_id = turn.channel_id
                due.events += [(kind, payload, turn) for kind, payload in events]
                due.turns.append((key, turn, ends))
        if not self._closing.is_set():
            due.typing = [(k, t) for k, t in self._typing.items()
                          if t.requested > t.sent and now - t.sent >= TYPING_EVERY_S]
        return due

    def _commit(self, due: _Due, now: float) -> None:
        for key, turn, ends in due.turns:
            turn.start_sent, turn.last_sent = True, now
            if ends:  # not ``turn.end_payload``: it may have been set while publishing
                self._turns.pop(key, None)
        for _key, entry in due.typing:
            entry.sent = now
        # Forget channels nobody has typed in for a while; the map stays small.
        for key in [k for k, t in self._typing.items() if now - t.requested > 60.0]:
            del self._typing[key]

    # ── wire ────────────────────────────────────────────────────────────────

    def _sign(self, kind: int, tags: list, content: str) -> dict:
        created_at = int(time.time())
        serialized = json.dumps([0, self.agent_pub, created_at, kind, tags, content],
                                separators=(",", ":"), ensure_ascii=False).encode()
        event_id = hashlib.sha256(serialized).digest()
        return {"id": event_id.hex(), "pubkey": self.agent_pub, "created_at": created_at,
                "kind": kind, "tags": tags, "content": content,
                "sig": _na.schnorr_sign(event_id, self._priv).hex()}

    def _observer_event(self, kind: str, payload: dict, *, channel_id: Optional[str] = None,
                        turn_id: Optional[str] = None, started_at: Optional[str] = None) -> dict:
        now = _now_iso()
        return {"seq": next(self._seq), "timestamp": now, "kind": kind, "agentIndex": 0,
                "channelId": channel_id or "", "sessionId": self.session_id,
                "turnId": turn_id or str(uuid.uuid4()), "startedAt": started_at or now,
                "payload": payload}

    def _observer_frame(self, events: list[dict]) -> dict:
        """One event ships bare; several get buzz-acp's batch envelope, whose
        metadata mirrors the LAST inner event."""
        plain = events[0] if len(events) == 1 else {**events[-1], "kind": "batch",
                                                    "payload": {"events": events}}
        content = nip44_encrypt(json.dumps(plain, separators=(",", ":"), ensure_ascii=False),
                                self._conversation_key)
        return self._sign(OBSERVER_KIND, [["p", self.owner], ["agent", self.agent_pub],
                                          ["frame", "telemetry"]], content)

    def _typing_event(self, channel_id: str, thread_root: Optional[str]) -> dict:
        tags = [["h", channel_id]] + ([["e", thread_root, "", "root"]] if thread_root else [])
        return self._sign(TYPING_KIND, tags, "")

    def _presence_event(self, status: str) -> dict:
        return self._sign(PRESENCE_KIND, [["status", status]], status)

    @staticmethod
    async def _await_ok(ws, event_id: str) -> tuple[bool, str]:
        deadline = time.monotonic() + OK_TIMEOUT_S
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=max(deadline - time.monotonic(), 0.01)))
            if isinstance(msg, list) and len(msg) >= 3 and msg[0] == "OK" and msg[1] == event_id:
                return msg[2] is True, str(msg[3]) if len(msg) > 3 else ""

    async def _send(self, ws, event: dict) -> bool:
        await ws.send(json.dumps(["EVENT", event], separators=(",", ":")))
        ok, reason = await self._await_ok(ws, event["id"])
        if not ok:
            logger.warning("Buzz observer: relay rejected kind %d: %s", event["kind"], reason)
        return ok

    async def _authenticate(self, ws) -> None:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=OK_TIMEOUT_S))
        if not (isinstance(msg, list) and len(msg) >= 2 and msg[0] == "AUTH"):
            raise ConnectionError(f"expected a NIP-42 challenge, got {str(msg)[:80]}")
        auth = await asyncio.to_thread(
            _na.build_auth_event, private_key=self._priv, challenge=str(msg[1]),
            relay_url=self.ws_url, auth_tag_json=self._auth_tag_json)
        await ws.send(json.dumps(["AUTH", auth], separators=(",", ":")))
        # The relay denies with OK=false; NIP-FI paths deny with a NOTICE, and a ban
        # closes the socket. All three must surface as a rejection, not a timeout.
        deadline = time.monotonic() + OK_TIMEOUT_S
        try:
            while True:
                msg = json.loads(await asyncio.wait_for(
                    ws.recv(), timeout=max(deadline - time.monotonic(), 0.01)))
                if not isinstance(msg, list) or not msg:
                    continue
                if msg[0] == "OK" and len(msg) >= 3 and msg[1] == auth["id"]:
                    if msg[2] is True:
                        return
                    raise _AuthRejected(str(msg[3]) if len(msg) > 3 else "OK=false")
                if msg[0] in ("NOTICE", "CLOSED"):
                    raise _AuthRejected(str(msg[-1]))
        except TimeoutError:
            raise ConnectionError(f"no answer to AUTH within {OK_TIMEOUT_S:.0f}s") from None
        except _connection_closed() as exc:
            raise _AuthRejected(f"socket closed during AUTH ({exc})") from None

    async def _publish_due(self, ws, due: _Due, now: float) -> None:
        if due.events:
            events = [self._observer_event(kind, payload, channel_id=turn.channel_id,
                                           turn_id=turn.turn_id, started_at=turn.started_at)
                      for kind, payload, turn in due.events]
            try:
                frame = await asyncio.to_thread(self._observer_frame, events)
            except ValueError as exc:  # unbuildable (e.g. over the NIP-44 limit): retrying cannot help
                logger.warning("Buzz observer: dropped an unbuildable frame: %s", exc)
                frame = None
            if frame is not None and await self._send(ws, frame):
                self.published += len(events)
        for (channel_id, thread_root), _entry in due.typing:
            await self._send(ws, await asyncio.to_thread(self._typing_event, channel_id, thread_root))
        # Rejected frames are not retried: a rejection would repeat forever.
        self._commit(due, now)

    async def _session(self, ws) -> None:
        if not self._announced:
            frame = await asyncio.to_thread(self._observer_frame, [self._observer_event(
                "harness_started", {"adapter": "hermes-native-observer", "telemetry": "kind:24200"})])
            self._announced = await self._send(ws, frame)
        last_presence = -math.inf
        while not self._closing.is_set():
            now = time.monotonic()
            if now - last_presence >= PRESENCE_EVERY_S:
                await self._send(ws, await asyncio.to_thread(self._presence_event, "online"))
                last_presence = now
            await self._publish_due(ws, self._due(now), now)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._closing.wait(), timeout=TICK_S)
        # Goodbye: every remaining turn is closed by now; one frame per channel, then offline.
        while self._turns:
            await self._publish_due(ws, self._due(time.monotonic()), time.monotonic())
        await self._send(ws, await asyncio.to_thread(self._presence_event, "offline"))

    async def _run(self) -> None:
        import websockets
        backoff, failures, last_rejection = 1.0, 0, None
        while not self._closing.is_set():
            try:
                async with websockets.connect(self.ws_url, max_size=2_000_000,
                                              open_timeout=10, close_timeout=0.5) as ws:
                    await self._authenticate(ws)
                    # Reset only once authenticated: a relay that keeps rejecting
                    # AUTH must keep backing off, not be redialled every second.
                    failures, backoff, last_rejection = 0, 1.0, None
                    await self._session(ws)
                    return
            except asyncio.CancelledError:
                raise
            except _AuthRejected as exc:
                # Not fatal: the relay also answers OK=false for transient causes
                # ("error: internal") and for "auth-required: …; reconnect".
                reason = str(exc)
                log = logger.error if reason != last_rejection else logger.debug
                last_rejection = reason
                log("Buzz observer: relay rejected AUTH (%s) — check BUZZ_AUTH_TAG; "
                    "retrying with backoff", reason)
            except Exception as exc:
                failures += 1
                log = logger.warning if failures == 1 else logger.debug
                log("Buzz observer: connection lost (%s: %s); reconnecting", type(exc).__name__, exc)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._closing.wait(), timeout=backoff)
            backoff = min(backoff * 2, RECONNECT_MAX_S)
