"""Interactive SAS (emoji) verification support for the Matrix adapter.

Implements the responder side of the Matrix ``m.sas.v1`` verification
flow (https://spec.matrix.org/latest/client-server-api/#sas-method) so the
bot can participate in interactive device verification initiated from any
Matrix client (Element Web/X, FluffyChat, ...).

The bot acts as the *responder*: a human user starts "Verify by emoji"
against the bot's device, and this handler answers the
``m.key.verification.*`` to-device events.  Because the bot has no screen,
the emoji short-auth-string is posted into the DM so the user can compare
it with what their client displays, then confirm on their side.

Flow handled here (responder role, ``we_started_it = False``):

    user -> bot   m.key.verification.request
    bot -> user   m.key.verification.ready
    user -> bot   m.key.verification.start   (method: m.sas.v1)
    bot -> user   m.key.verification.accept  (with commitment)
    user -> bot   m.key.verification.key     (their ephemeral pubkey)
    bot -> user   m.key.verification.key     (our ephemeral pubkey)
    bot -> dm     emoji short-auth-string (for the user to compare)
    user -> bot   m.key.verification.mac     (user confirmed the emojis)
    bot -> user   m.key.verification.mac
    user -> bot   m.key.verification.done
    bot -> user   m.key.verification.done

Cryptographic details follow the spec (HKDF-SHA256, ``curve25519-hkdf-sha256``
key agreement, ``hkdf-hmac-sha256.v2`` MAC method, unpadded-base64
commitment).  The commitment is computed exactly like matrix-rust-sdk does:
``unpadded_base64(SHA256(our_pubkey_b64 + canonical_json(start_content)))``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4

logger = logging.getLogger("hermes.matrix.verification")

# Session lifetime: the spec suggests clients time out after 10 minutes.
_SESSION_TTL_SECONDS = 10 * 60

# 64 emojis from the spec table (https://spec.matrix.org/latest/...sas-method-emoji).
# Index 0..63, exactly as published in matrix-spec data-definitions/sas-emoji.json.
_EMOJIS: list[tuple[str, str]] = [
    ("🐶", "Dog"), ("🐱", "Cat"), ("🦁", "Lion"), ("🐎", "Horse"),
    ("🦄", "Unicorn"), ("🐷", "Pig"), ("🐘", "Elephant"), ("🐰", "Rabbit"),
    ("🐼", "Panda"), ("🐓", "Rooster"), ("🐧", "Penguin"), ("🐢", "Turtle"),
    ("🐟", "Fish"), ("🐙", "Octopus"), ("🦋", "Butterfly"), ("🌷", "Flower"),
    ("🌳", "Tree"), ("🌵", "Cactus"), ("🍄", "Mushroom"), ("🌏", "Globe"),
    ("🌙", "Moon"), ("☁️", "Cloud"), ("🔥", "Fire"), ("🍌", "Banana"),
    ("🍎", "Apple"), ("🍓", "Strawberry"), ("🌽", "Corn"), ("🍕", "Pizza"),
    ("🎂", "Cake"), ("❤️", "Heart"), ("😀", "Smiley"), ("🤖", "Robot"),
    ("🎩", "Hat"), ("👓", "Glasses"), ("🔧", "Wrench"), ("🎅", "Santa"),
    ("👍", "Thumbs up"), ("☂️", "Umbrella"), ("⌛", "Hourglass"), ("⏰", "Clock"),
    ("🎁", "Gift"), ("💡", "Light Bulb"), ("📕", "Book"), ("✏️", "Pencil"),
    ("📎", "Paperclip"), ("✂️", "Scissors"), ("🔒", "Lock"), ("🔑", "Key"),
    ("🔨", "Hammer"), ("☎️", "Telephone"), ("🏁", "Flag"), ("🚂", "Train"),
    ("🚲", "Bicycle"), ("✈️", "Airplane"), ("🚀", "Rocket"), ("🏆", "Trophy"),
    ("⚽", "Ball"), ("🎸", "Guitar"), ("🎺", "Trumpet"), ("🔔", "Bell"),
    ("⚓", "Anchor"), ("🎧", "Headphones"), ("📁", "Folder"), ("📌", "Pin"),
]

# Key agreement / hash / MAC / SAS method identifiers (spec).
_METHOD_SAS_V1 = "m.sas.v1"
_KEY_AGREEMENT = "curve25519-hkdf-sha256"
_HASH = "sha256"
_MAC_V2 = "hkdf-hmac-sha256.v2"
_MAC_V1 = "hkdf-hmac-sha256"
_SHORT_AUTH_STRINGS = ["emoji", "decimal"]


def _canonical_json(data: Any) -> str:
    """Matrix canonical JSON: sorted keys, compact separators, UTF-8.

    Matches the canonical JSON used by matrix-sdk's ``CanonicalJsonValue``
    for the SAS commitment: objects are serialized with lexicographically
    sorted keys, no whitespace, ``:`` and ``,`` separators.
    """
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _unpadded_base64(raw: bytes) -> str:
    import base64
    return base64.b64encode(raw).decode("ascii").rstrip("=")


def _plain(obj: Any) -> Any:
    """Recursively convert mautrix Obj/Lst (and nested structures) to plain
    Python dicts/lists/strings so standard dict APIs (.items(), .keys())
    work.  mautrix's Obj is a dict-like that lacks those methods."""
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj
    if hasattr(obj, "serialize"):
        try:
            return _plain(obj.serialize())
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    # Fallback: duck-typed attribute containers (Obj without serialize,
    # test fakes, ...) -> expose their __dict__ as a plain dict.
    if hasattr(obj, "__dict__"):
        return {k: _plain(v) for k, v in vars(obj).items()}
    return obj


def _txid_from(content: Any, event: Any = None) -> str:
    """Resolve the SAS transaction id from a verification event.

    Legacy to-device events carry ``transaction_id`` in the content.  In-room
    verification (MSC 2241, Element X) has *no* transaction_id — the whole
    exchange is chained via ``m.relates_to.event_id`` pointing at the anchor
    (the original ``m.key.verification.request``).  That anchor event id IS
    the transaction id for in-room flows, so fall back to it.
    """
    if isinstance(content, dict) and content.get("transaction_id"):
        return str(content["transaction_id"])
    if isinstance(content, dict):
        relates = content.get("m.relates_to") or {}
        if isinstance(relates, dict) and relates.get("event_id"):
            return str(relates["event_id"])
    # Last resort: the event id of the event itself (e.g. the request).
    evt_id = getattr(event, "event_id", None)
    if evt_id:
        return str(evt_id)
    return ""


def _compute_commitment(our_pubkey_b64: str, start_content: Dict[str, Any]) -> str:
    """Compute the SAS commitment exactly like matrix-rust-sdk.

    commitment = unpadded_base64(SHA256(our_pubkey_b64 + canonical_json(start_content)))
    """
    digest = hashlib.sha256(
        our_pubkey_b64.encode("ascii") + _canonical_json(start_content).encode("utf-8")
    ).digest()
    return _unpadded_base64(digest)


def _bytes_to_emoji_indices(raw: bytes) -> list[int]:
    """Split the first 42 bits of the 6 HKDF bytes into 7 groups of 6 bits.

    Mirrors vodozemac's ``SasBytes::bytes_to_emoji_index``.
    """
    num = 0
    for b in raw[:6]:
        num = (num << 8) | b
    return [
        (num >> 42) & 63,
        (num >> 36) & 63,
        (num >> 30) & 63,
        (num >> 24) & 63,
        (num >> 18) & 63,
        (num >> 12) & 63,
        (num >> 6) & 63,
    ]


@dataclass
class _SasSession:
    """State for one in-flight SAS verification."""

    transaction_id: str
    other_user: str
    other_device: str
    room_id: Optional[str] = None
    # In-room (MSC 2241): EVERY follow-up event references the ANCHOR
    # (the request event id) via m.relates_to — matrix-rust-sdk builds
    # each event with Reference::new(anchor), not the previous event.
    anchor_event_id: Optional[str] = None
    last_event_id: Optional[str] = None  # in-room: event we must reply to
    sas: Any = None  # python-olm Sas instance
    our_pubkey: str = ""
    their_pubkey: str = ""
    commitment: str = ""
    chosen_mac: str = _MAC_V2
    mac_sent: bool = False
    done_sent: bool = False
    created_at: float = field(default_factory=time.time)
    we_initiated: bool = False  # True when *we* sent the request (initiator)
    their_commitment: str = ""  # initiator: commitment from their accept
    start_content: Optional[dict] = None  # initiator: our sent start content


class SasVerificationHandler:
    """Responder for inbound Matrix SAS (emoji) verification flows."""

    def __init__(self, adapter: Any, client: Any, olm: Any) -> None:
        self._adapter = adapter
        self._client = client
        self._olm = olm
        self._sessions: Dict[str, _SasSession] = {}
        self._registered = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self) -> None:
        """Register to-device AND in-room event handlers on the mautrix client.

        Element X (and newer Element clients) send verification requests as
        *room* events chained via ``m.relates_to`` references (MSC 2241,
        "in-room verification"), while older clients use to-device messages.
        We register both so either transport works.
        """
        if self._registered or not self._client:
            return
        from mautrix.types import EventType

        handlers = {
            "m.key.verification.request": self._on_request,
            "m.key.verification.ready": self._on_ready,
            "m.key.verification.start": self._on_start,
            "m.key.verification.accept": self._on_accept,
            "m.key.verification.key": self._on_key,
            "m.key.verification.mac": self._on_mac,
            "m.key.verification.done": self._on_done,
            "m.key.verification.cancel": self._on_cancel,
        }
        for evt_name, handler in handlers.items():
            # to-device transport (legacy clients)
            evt_type = EventType.find(evt_name, EventType.Class.TO_DEVICE)
            self._client.add_event_handler(evt_type, handler, wait_sync=True)
            # in-room transport (Element X / MSC 2241): timeline events land
            # with Class.MESSAGE, so register the same handler for that class.
            room_type = EventType.find(evt_name, EventType.Class.MESSAGE)
            self._client.add_event_handler(room_type, handler, wait_sync=True)
        # MSC 2241 in-room REQUESTS are m.room.message with
        # msgtype=m.key.verification.request (NOT a bare
        # m.key.verification.request event type — Element X matches
        # MessageType::VerificationRequest in event_enums.rs).  Route those
        # into _on_request via a msgtype filter.
        room_msg_type = EventType.find("m.room.message", EventType.Class.MESSAGE)
        self._client.add_event_handler(room_msg_type, self._on_room_message_request, wait_sync=True)
        self._registered = True
        logger.info("Matrix: SAS verification handler registered (emoji compare, to_device + in-room)")

    async def _on_room_message_request(self, event: Any) -> None:
        """Route m.room.message with msgtype=m.key.verification.request (MSC 2241)."""
        content = _plain(getattr(event, "content", None))
        if not isinstance(content, dict):
            return
        if content.get("msgtype") != "m.key.verification.request":
            return
        await self._on_request(event)

    # ------------------------------------------------------------------
    # Bot-initiated verification (we are the initiator)
    # ------------------------------------------------------------------

    async def start_verification(self, user_id: str, room_id: Optional[str] = None) -> bool:
        """Start an in-room SAS verification with *user_id* (we initiate).

        Sends the request as an ``m.room.message`` with
        ``msgtype: m.key.verification.request`` (MSC 2241 — NOT as a bare
        ``m.key.verification.request`` event type).  Element X only
        recognises in-room requests in this form (event_enums.rs matches
        ``MessageType::VerificationRequest``), so the wrong type makes it
        show the fallback body instead of the verification prompt.  The
        rest of the flow then runs over in-room events (MSC 2241).
        Returns True when the request was sent.
        """
        try:
            room_id = room_id or await self._find_dm_room(user_id)
            if not room_id:
                logger.warning("Matrix: no DM room found for %s — cannot start verification", user_id)
                return False
            txid = str(uuid4())
            session = _SasSession(
                transaction_id=txid,
                other_user=user_id,
                other_device="",
                room_id=room_id,
                we_initiated=True,
            )
            self._sessions[txid] = session
            self._expire_old_sessions()
            # MSC 2241 request shape: m.room.message with msgtype + body
            # fallback + to + from_device + methods.  The anchor for the
            # whole flow is the event_id of this message.
            event_id = await self._send_room_event(
                "m.room.message",
                session,
                {
                    "body": (
                        f"{self._user_id} is requesting to verify your key, but "
                        "your client does not support in-chat key verification, so "
                        "you may need to use a different verification method."
                    ),
                    "msgtype": "m.key.verification.request",
                    "to": user_id,
                    "from_device": self._device_id,
                    "methods": [_METHOD_SAS_V1],
                },
            )
            if not event_id:
                self._sessions.pop(txid, None)
                logger.warning("Matrix: verification request send failed for %s", user_id)
                return False
            # In-room flows key every follow-up event by the ANCHOR event id
            # (the request we just sent) via m.relates_to — re-key the session
            # so _on_ready/_on_accept/... find it via _txid_from().
            self._sessions.pop(txid, None)
            session.transaction_id = str(event_id)
            session.anchor_event_id = str(event_id)
            session.last_event_id = str(event_id)
            self._sessions[str(event_id)] = session
            logger.info(
                "Matrix: SAS verification request sent to %s (anchor %s, room %s)",
                user_id, event_id, room_id,
            )
            return True
        except Exception as exc:
            logger.exception("Matrix: start_verification failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Event handlers (all run as asyncio tasks via mautrix dispatcher)
    # ------------------------------------------------------------------

    async def _on_request(self, event: Any) -> None:
        """User initiated verification -> answer with ready.

        Works for both transports: to-device (legacy clients) and in-room
        (Element X / MSC 2241).  For in-room the request carries no
        transaction_id; we generate one and echo it back in ``ready``.
        """
        try:
            sender = getattr(event, "sender", None)
            content = _plain(getattr(event, "content", None))
            if not sender or not content:
                return
            # Echo-Schutz: nie auf Events des eigenen Accounts antworten
            # (der Bot-Account kann weitere Clients haben, z.B. ein offenes
            # Element Web, dessen Verification der Bot nicht beantworten darf).
            if sender == self._user_id:
                logger.debug("Matrix: ignoring own-account verification request from %s", sender)
                return
            room_id = getattr(event, "room_id", None) or None
            event_id = getattr(event, "event_id", None) or None
            # txid: to-device requests carry transaction_id; in-room requests
            # are the chain ANCHOR — their own event_id is the txid (MSC 2241).
            # Using the anchor here is what makes _on_start/_on_key/... find
            # this session, because Element X references the anchor in every
            # subsequent m.relates_to.
            txid = _txid_from(content, event) or str(uuid4())
            from_device = str(content.get("from_device") or "")
            if not from_device:
                logger.warning("Matrix: verification request without from_device from %s", sender)
                return
            # MSC 2241: only answer requests addressed to us.  (to-device
            # requests carry no "to"; in-room m.room.message requests do.)
            target = str(content.get("to") or "")
            if target and target != self._user_id:
                logger.debug(
                    "Matrix: verification request for %s not addressed to us — ignoring", target
                )
                return

            session = _SasSession(
                transaction_id=txid,
                other_user=sender,
                other_device=from_device,
                room_id=room_id,
                last_event_id=event_id,
            )
            self._sessions[txid] = session
            self._expire_old_sessions()

            await self._send(
                "m.key.verification.ready",
                session,
                {
                    "from_device": self._device_id,
                    "methods": [_METHOD_SAS_V1],
                    "transaction_id": txid,
                },
            )
            logger.info(
                "Matrix: SAS verification requested by %s (%s), sent ready (transport=%s)",
                sender, from_device, "room" if room_id else "to_device",
            )
        except Exception as exc:
            logger.exception("Matrix: SAS request handler failed: %s", exc)

    async def _on_ready(self, event: Any) -> None:
        """User accepted our request (we are the initiator) -> send start."""
        try:
            content = _plain(getattr(event, "content", None))
            sender = getattr(event, "sender", None)
            if not content or not sender:
                return
            if sender == self._user_id:
                return
            room_id = getattr(event, "room_id", None) or None
            event_id = getattr(event, "event_id", None) or None
            txid = _txid_from(content, event)
            session = self._sessions.get(txid)
            if not session or not session.we_initiated or session.other_user != sender:
                logger.warning(
                    "Matrix: SAS ready with unknown txid %s from %s", txid, sender
                )
                return
            session.room_id = room_id or session.room_id
            session.last_event_id = event_id or session.last_event_id
            session.other_device = str(content.get("from_device") or "")
            if not session.other_device:
                logger.warning("Matrix: SAS ready without from_device from %s", sender)
                return

            # Create our ephemeral SAS key pair and compute the commitment
            # over our start content (MSC 2241: includes m.relates_to).
            import olm as olm_lib

            sas = olm_lib.Sas()
            session.sas = sas
            session.our_pubkey = str(sas.pubkey)

            start_content = {
                "from_device": self._device_id,
                "method": _METHOD_SAS_V1,
                "key_agreement_protocols": [_KEY_AGREEMENT],
                "hashes": [_HASH],
                "message_authentication_codes": [_MAC_V2, _MAC_V1],
                "short_authentication_string": ["emoji"],
            }
            # In-room (MSC 2241): no transaction_id in the content — the
            # chain anchor event id IS the transaction id.  The commitment
            # is over the content AS SENT, i.e. including the m.relates_to
            # reference Element X expects in in-room flows.
            commit_content = dict(start_content)
            if session.room_id and session.last_event_id:
                commit_content["m.relates_to"] = {
                    "rel_type": "m.reference",
                    "event_id": session.last_event_id,
                }
            session.commitment = _compute_commitment(session.our_pubkey, commit_content)
            session.start_content = commit_content
            session.chosen_mac = _MAC_V2

            await self._send("m.key.verification.start", session, start_content)
            logger.info("Matrix: SAS verification started with %s (tx %s)", sender, txid)
        except Exception as exc:
            logger.exception("Matrix: SAS ready handler failed: %s", exc)

    async def _on_accept(self, event: Any) -> None:
        """User accepted our start (we are the initiator) -> send our key."""
        try:
            content = _plain(getattr(event, "content", None))
            sender = getattr(event, "sender", None)
            if not content or not sender:
                return
            if sender == self._user_id:
                return
            room_id = getattr(event, "room_id", None) or None
            event_id = getattr(event, "event_id", None) or None
            txid = _txid_from(content, event)
            session = self._sessions.get(txid)
            if not session or not session.we_initiated or session.other_user != sender:
                logger.warning(
                    "Matrix: SAS accept with unknown txid %s from %s", txid, sender
                )
                return
            session.room_id = room_id or session.room_id
            session.last_event_id = event_id or session.last_event_id
            session.their_commitment = str(content.get("commitment") or "")
            chosen = str(content.get("message_authentication_code") or "")
            if chosen in (_MAC_V2, _MAC_V1):
                session.chosen_mac = chosen

            # Share our ephemeral key right after accept.
            await self._send(
                "m.key.verification.key",
                session,
                {"transaction_id": txid, "key": session.our_pubkey},
            )
            logger.info("Matrix: SAS accept received from %s (tx %s)", sender, txid)
        except Exception as exc:
            logger.exception("Matrix: SAS accept handler failed: %s", exc)

    async def _on_start(self, event: Any) -> None:
        """User picked emoji verification -> create SAS, send accept + key."""
        try:
            content = _plain(getattr(event, "content", None))
            sender = getattr(event, "sender", None)
            if not content or not sender:
                return
            if sender == self._user_id:
                logger.debug("Matrix: ignoring own-account start from %s", sender)
                return
            room_id = getattr(event, "room_id", None) or None
            event_id = getattr(event, "event_id", None) or None
            txid = _txid_from(content, event)
            session = self._sessions.get(txid)
            if not session or session.other_user != sender:
                # Element sometimes sends `start` directly without a prior
                # request/ready handshake (e.g. verifying from the session
                # list).  Create the session on the fly in that case.
                if session and session.other_user != sender:
                    logger.warning(
                        "Matrix: SAS start txid %s reused by different user %s",
                        txid, sender,
                    )
                    return
                from_device = str(content.get("from_device") or "")
                if not from_device:
                    logger.warning("Matrix: SAS start without from_device from %s", sender)
                    return
                session = _SasSession(
                    transaction_id=txid,
                    other_user=sender,
                    other_device=from_device,
                    room_id=room_id,
                    last_event_id=event_id,
                )
                self._sessions[txid] = session
            else:
                # Transport may differ from request (rare); keep in sync.
                session.room_id = room_id or session.room_id
                session.last_event_id = event_id or session.last_event_id

            method = str(content.get("method") or "")
            if method != _METHOD_SAS_V1:
                await self._cancel(session, "m.unknown_method", "Unsupported method")
                return
            if "emoji" not in (content.get("short_authentication_string") or []):
                await self._cancel(session, "m.unknown_method", "Emoji not offered")
                return

            # Create the ephemeral SAS key pair.
            import olm as olm_lib

            sas = olm_lib.Sas()
            session.sas = sas
            session.our_pubkey = str(sas.pubkey)
            # MSC 2241: the commitment is over the decrypted start content
            # AND must include m.relates_to even if the decrypted content
            # lacks it (Element X's rust-sdk stores start_content with the
            # Reference it added on send, then hashes that).
            commit_content = dict(content)
            if "m.relates_to" not in commit_content and session.transaction_id:
                commit_content["m.relates_to"] = {
                    "rel_type": "m.reference",
                    "event_id": session.transaction_id,
                }
            session.commitment = _compute_commitment(
                session.our_pubkey, commit_content
            )

            # Choose MAC method: prefer v2 (correct base64), fall back to v1.
            offered_macs = content.get("message_authentication_codes") or []
            if _MAC_V2 in offered_macs:
                session.chosen_mac = _MAC_V2
            elif _MAC_V1 in offered_macs:
                session.chosen_mac = _MAC_V1
            else:
                await self._cancel(session, "m.unknown_method", "No supported MAC method")
                return

            await self._send(
                "m.key.verification.accept",
                session,
                {
                    "transaction_id": txid,
                    "key_agreement_protocol": _KEY_AGREEMENT,
                    "hash": _HASH,
                    "message_authentication_code": session.chosen_mac,
                    "short_authentication_string": ["emoji"],
                    "commitment": session.commitment,
                },
            )
            # Share our ephemeral key right after accept (spec step 9/10 order:
            # responder's key may be sent once the accept is out).
            await self._send(
                "m.key.verification.key",
                session,
                {"transaction_id": txid, "key": session.our_pubkey},
            )
            logger.info("Matrix: SAS start accepted for %s (tx %s)", sender, txid)
        except Exception as exc:
            logger.exception("Matrix: SAS start handler failed: %s", exc)

    async def _on_key(self, event: Any) -> None:
        """User shared their ephemeral key -> derive SAS, show emojis."""
        try:
            content = _plain(getattr(event, "content", None))
            sender = getattr(event, "sender", None)
            if not content or not sender:
                return
            if sender == self._user_id:
                logger.debug("Matrix: ignoring own-account key from %s", sender)
                return
            room_id = getattr(event, "room_id", None) or None
            event_id = getattr(event, "event_id", None) or None
            txid = _txid_from(content, event)
            session = self._sessions.get(txid)
            if not session or session.other_user != sender:
                logger.warning("Matrix: SAS key with unknown txid %s from %s", txid, sender)
                return
            session.room_id = room_id or session.room_id
            session.last_event_id = event_id or session.last_event_id
            their_key = str(content.get("key") or "")
            if not their_key or not session.sas:
                return

            session.their_pubkey = their_key
            try:
                session.sas.set_their_pubkey(their_key)
            except Exception as exc:
                logger.warning("Matrix: SAS set_their_pubkey failed: %s", exc)
                await self._cancel(session, "m.invalid_message", "Bad public key")
                return

            # Initiator path: verify their accept-commitment (their pubkey +
            # our start content) before trusting the key.
            if session.we_initiated and session.their_commitment and session.start_content:
                expected = _compute_commitment(their_key, session.start_content)
                if expected != session.their_commitment:
                    logger.warning("Matrix: SAS commitment mismatch from %s", sender)
                    await self._cancel(session, "m.mismatched_commitment", "Commitment mismatch")
                    return

            # Compute the emoji short-auth-string and post it into the DM.
            emojis = self._compute_emojis(session)
            if emojis:
                await self._post_emojis(session, emojis)

            # Initiator path: we send our MACs first (responder verifies).
            if session.we_initiated and not session.mac_sent:
                await self._send_our_mac(session)
        except Exception as exc:
            logger.exception("Matrix: SAS key handler failed: %s", exc)

    async def _on_mac(self, event: Any) -> None:
        """User confirmed the emojis -> verify their MACs, send ours."""
        try:
            content = _plain(getattr(event, "content", None))
            sender = getattr(event, "sender", None)
            if not content or not sender:
                return
            if sender == self._user_id:
                logger.debug("Matrix: ignoring own-account mac from %s", sender)
                return
            room_id = getattr(event, "room_id", None) or None
            event_id = getattr(event, "event_id", None) or None
            txid = _txid_from(content, event)
            session = self._sessions.get(txid)
            if not session or session.other_user != sender:
                logger.warning("Matrix: SAS mac with unknown txid %s from %s", txid, sender)
                return
            session.room_id = room_id or session.room_id
            session.last_event_id = event_id or session.last_event_id

            # Verify the user's MACs if we can resolve their keys.
            mac_ok = await self._verify_user_macs(session, content)
            if not mac_ok:
                await self._cancel(session, "m.key_mismatch", "MAC verification failed")
                return

            # Responder path (default): we send our MACs only after the user's
            # arrive.  Initiator path: we already sent ours in _on_key, so
            # here we just verify and finish the exchange.
            if not session.mac_sent:
                await self._send_our_mac(session)
            if not session.done_sent:
                await self._send(
                    "m.key.verification.done",
                    session,
                    {"transaction_id": txid},
                )
                session.done_sent = True
            await self._finalize(session)
        except Exception as exc:
            logger.exception("Matrix: SAS mac handler failed: %s", exc)

    async def _on_done(self, event: Any) -> None:
        """User finished -> send our done, mark verified, clean up."""
        try:
            content = _plain(getattr(event, "content", None))
            sender = getattr(event, "sender", None)
            if not content or not sender:
                return
            if sender == self._user_id:
                return
            room_id = getattr(event, "room_id", None) or None
            event_id = getattr(event, "event_id", None) or None
            txid = _txid_from(content, event)
            session = self._sessions.get(txid)
            if not session or session.other_user != sender:
                return
            session.room_id = room_id or session.room_id
            session.last_event_id = event_id or session.last_event_id
            if not session.done_sent:
                await self._send(
                    "m.key.verification.done",
                    session,
                    {"transaction_id": txid},
                )
                session.done_sent = True
            await self._finalize(session)
        except Exception as exc:
            logger.exception("Matrix: SAS done handler failed: %s", exc)

    async def _on_cancel(self, event: Any) -> None:
        """User cancelled or errored -> drop the session."""
        try:
            content = _plain(getattr(event, "content", None))
            sender = getattr(event, "sender", None)
            if not content or not sender:
                return
            if sender == self._user_id:
                return
            txid = _txid_from(content, event)
            session = self._sessions.pop(txid, None)
            if session:
                code = content.get("code") or "m.user"
                logger.info(
                    "Matrix: SAS verification with %s cancelled (%s)", sender, code
                )
                await self._notify_cancelled(session, code)
        except Exception as exc:
            logger.exception("Matrix: SAS cancel handler failed: %s", exc)

    # ------------------------------------------------------------------
    # Crypto helpers
    # ------------------------------------------------------------------

    def _compute_emojis(self, session: _SasSession) -> list[tuple[str, str]]:
        """Derive the 7 emojis from the shared SAS secret."""
        try:
            info = self._sas_info(session)
            raw = session.sas.generate_bytes(info, 6)
            indices = _bytes_to_emoji_indices(bytes(raw))
            return [_EMOJIS[i] for i in indices]
        except Exception as exc:
            logger.warning("Matrix: SAS emoji generation failed: %s", exc)
            return []

    def _sas_info(self, session: _SasSession) -> str:
        """HKDF info string for the SAS short-auth-string (v2 key agreement).

        Spec: ``MATRIX_KEY_VERIFICATION_SAS|{start_user}|{start_device}|{start_key}|
        {accept_user}|{accept_device}|{accept_key}|{transaction_id}``

        The side that sent ``start`` is the "start" side, the other one the
        "accept" side.  Normally the user sends ``start`` (we are the
        responder); when *we* initiated, the roles are reversed.
        """
        if session.we_initiated:
            return (
                "MATRIX_KEY_VERIFICATION_SAS|"
                f"{self._user_id}|{self._device_id}|{session.our_pubkey}|"
                f"{session.other_user}|{session.other_device}|{session.their_pubkey}|"
                f"{session.transaction_id}"
            )
        return (
            "MATRIX_KEY_VERIFICATION_SAS|"
            f"{session.other_user}|{session.other_device}|{session.their_pubkey}|"
            f"{self._user_id}|{self._device_id}|{session.our_pubkey}|"
            f"{session.transaction_id}"
        )

    def _mac_info_for(self, session: _SasSession, our_side: bool) -> str:
        """MAC HKDF info.

        For the MACs *we* send (our_side=True): own user/device first.
        For verifying MACs the *user* sent (our_side=False): their ids first.
        """
        if our_side:
            return (
                "MATRIX_KEY_VERIFICATION_MAC"
                f"{self._user_id}{self._device_id}"
                f"{session.other_user}{session.other_device}"
                f"{session.transaction_id}"
            )
        return (
            "MATRIX_KEY_VERIFICATION_MAC"
            f"{session.other_user}{session.other_device}"
            f"{self._user_id}{self._device_id}"
            f"{session.transaction_id}"
        )

    def _mac_func(self, session: _SasSession):
        """Return the python-olm MAC function for the chosen MAC method."""
        import olm as olm_lib

        sas = session.sas
        if session.chosen_mac == _MAC_V2:
            # hkdf-hmac-sha256.v2 uses the corrected (unpadded) base64 encoding.
            return sas.calculate_mac_fixed_base64
        # Legacy hkdf-hmac-sha256 (libolm's original base64 encoding).
        return sas.calculate_mac

    async def _send_our_mac(self, session: _SasSession) -> None:
        """Send our MACs for: our device key, our cross-signing master and
        self-signing keys (so the user can mark the whole identity verified)."""
        mac_func = self._mac_func(session)
        info = self._mac_info_for(session, our_side=True)

        keys_to_mac: Dict[str, str] = {}
        # Our device ed25519 key.
        try:
            device_key = self._olm.account.identity_keys.get("ed25519")
            if device_key:
                keys_to_mac[f"ed25519:{self._device_id}"] = str(device_key)
        except Exception as exc:
            logger.debug("Matrix: device ed25519 key lookup failed: %s", exc)

        # Our cross-signing keys (master + self-signing), if published.
        try:
            xsign = await self._olm.get_own_cross_signing_public_keys()
            if xsign:
                if getattr(xsign, "master_key", None):
                    mk = str(xsign.master_key)
                    keys_to_mac[f"ed25519:{mk}"] = mk
                if getattr(xsign, "self_signing_key", None):
                    ssk = str(xsign.self_signing_key)
                    keys_to_mac[f"ed25519:{ssk}"] = ssk
        except Exception as exc:
            logger.debug("Matrix: cross-signing key lookup failed: %s", exc)

        if not keys_to_mac:
            logger.warning("Matrix: no keys available to MAC — cannot verify")
            await self._cancel(session, "m.unexpected_message", "No keys to MAC")
            return

        macs = {}
        for key_id, key_value in keys_to_mac.items():
            try:
                macs[key_id] = str(mac_func(key_value, info + key_id))
            except Exception as exc:
                logger.warning("Matrix: MAC calc failed for %s: %s", key_id, exc)
        if not macs:
            await self._cancel(session, "m.unexpected_message", "MAC calculation failed")
            return

        sorted_key_ids = ",".join(sorted(macs.keys()))
        try:
            keys_mac = str(mac_func(sorted_key_ids, info + "KEY_IDS"))
        except Exception as exc:
            logger.warning("Matrix: KEY_IDS MAC calc failed: %s", exc)
            await self._cancel(session, "m.unexpected_message", "KEY_IDS MAC failed")
            return

        await self._send(
            "m.key.verification.mac",
            session,
            {
                "mac": macs,
                "keys": keys_mac,
                "transaction_id": session.transaction_id,
            },
        )
        session.mac_sent = True
        logger.info(
            "Matrix: SAS MACs sent to %s (keys: %s)", session.other_user, sorted_key_ids
        )

    async def _verify_user_macs(self, session: _SasSession, content: Any) -> bool:
        """Verify the MACs the user sent, against their known keys.

        Returns True when everything we could check matched.  If we don't
        have the user's keys cached yet we try a fresh key query first.
        """
        try:
            macs = content.get("mac") or {}
            keys_mac_expected = str(content.get("keys") or "")
            if not macs:
                return False

            # Make sure we have the user's device + cross-signing keys.
            await self._ensure_user_keys(session)

            # Build key_id -> public key map from what we know about the user.
            known: Dict[str, str] = {}
            try:
                dev = await self._olm.crypto_store.get_device(
                    session.other_user, session.other_device
                )
                if dev and getattr(dev, "signing_key", None):
                    known[f"ed25519:{session.other_device}"] = str(dev.signing_key)
            except Exception as exc:
                logger.debug("Matrix: user device lookup failed: %s", exc)
            try:
                xsign = await self._olm.get_cross_signing_public_keys(session.other_user)
                if xsign:
                    mk = str(xsign.master_key)
                    known[f"ed25519:{mk}"] = mk
            except Exception as exc:
                logger.debug("Matrix: user cross-signing lookup failed: %s", exc)

            if not known:
                logger.warning(
                    "Matrix: no known keys for %s to verify MACs against", session.other_user
                )
                return False

            mac_func = self._mac_func(session)
            info = self._mac_info_for(session, our_side=False)

            # Verify the KEY_IDS MAC over the sorted key-id list.
            sorted_key_ids = ",".join(sorted(macs.keys()))
            if keys_mac_expected:
                try:
                    calc = str(mac_func(sorted_key_ids, info + "KEY_IDS"))
                    if calc != keys_mac_expected:
                        logger.warning("Matrix: user KEY_IDS MAC mismatch")
                        return False
                except Exception as exc:
                    logger.debug("Matrix: KEY_IDS MAC verify failed: %s", exc)
                    return False

            # Verify every MAC we have a key for.
            for key_id, mac_value in macs.items():
                public_key = known.get(key_id)
                if public_key is None:
                    continue  # unknown key (e.g. another device) — skip
                try:
                    calc = str(mac_func(public_key, info + key_id))
                except Exception as exc:
                    logger.warning("Matrix: MAC verify failed for %s: %s", key_id, exc)
                    return False
                if calc != str(mac_value):
                    logger.warning("Matrix: MAC mismatch for %s", key_id)
                    return False

            logger.info("Matrix: user MACs verified for %s", session.other_user)
            return True
        except Exception as exc:
            logger.exception("Matrix: MAC verification error: %s", exc)
            return False

    async def _ensure_user_keys(self, session: _SasSession) -> None:
        """Trigger a key query for the other user so their device/cross-signing
        keys are fresh in the store."""
        try:
            if hasattr(self._olm, "_fetch_keys"):
                await self._olm._fetch_keys([session.other_user], include_untracked=True)
            elif hasattr(self._olm, "query_keys"):
                await self._olm.query_keys([session.other_user])
            else:
                # Fallback: get_cross_signing_public_keys fetches on demand.
                await self._olm.get_cross_signing_public_keys(session.other_user)
        except Exception as exc:
            logger.debug("Matrix: key query for %s failed: %s", session.other_user, exc)

    # ------------------------------------------------------------------
    # Messaging helpers
    # ------------------------------------------------------------------

    async def _send(self, event_name: str, session: _SasSession, content: Dict[str, Any]) -> None:
        """Send a verification event via the session's transport.

        In-room sessions (MSC 2241) send timeline events chained via
        ``m.relates_to: {rel_type: m.reference, event_id: <last received>}``;
        everything else uses classic to-device messages.
        """
        if session.room_id:
            await self._send_room_event(event_name, session, content)
        else:
            await self._send_to_device(event_name, session, content)

    async def _send_room_event(
        self, event_name: str, session: _SasSession, content: Dict[str, Any]
    ) -> Optional[str]:
        """Send an in-room verification event (MSC 2241).

        In-room events use ``m.relates_to`` instead of ``transaction_id``
        (spec: "instead of the transaction_id property, an m.relates_to
        property is used instead") — strip any transaction_id from the
        content so Element X's strict deserializer never sees a mismatch.

        Returns the new event id (or None on failure) so callers can
        re-key sessions by anchor id.
        """
        from mautrix.types import EventType

        evt_type = EventType.find(event_name, EventType.Class.MESSAGE)
        payload = dict(content)
        payload.pop("transaction_id", None)
        # Every in-room verification event references the ANCHOR (the
        # original request event id), exactly like matrix-rust-sdk's
        # Reference::new(anchor).  The request itself IS the anchor and has
        # no relation yet — a fresh uuid4() transaction_id (initiator path,
        # before the request event exists) is NOT an event id, so only
        # reference real event ids (Matrix event ids start with '$').
        anchor = session.anchor_event_id
        if not anchor:
            txid = str(session.transaction_id or "")
            if txid.startswith("$"):
                anchor = txid
        if not anchor:
            anchor = session.last_event_id
        if anchor:
            payload["m.relates_to"] = {"rel_type": "m.reference", "event_id": anchor}
        try:
            send = getattr(self._client, "send_message_event", None)
            if send is None:
                send = self._client.api.send_message_event
            event_id = await send(session.room_id, evt_type, payload)
            if event_id:
                session.last_event_id = str(event_id)
            return str(event_id) if event_id else None
        except Exception as exc:
            logger.warning("Matrix: send room event %s failed: %s", event_name, exc)
            return None

    async def _send_to_device(self, event_name: str, session: _SasSession, content: Dict[str, Any]) -> None:
        from mautrix.types import EventType

        evt_type = EventType.find(event_name, EventType.Class.TO_DEVICE)
        try:
            # send_to_device lives on the client (CryptoMethods) in mautrix 0.21.
            send = getattr(self._client, "send_to_device", None)
            if send is None:
                send = self._client.api.send_to_device
            await send(
                evt_type,
                {session.other_user: {session.other_device: content}},
            )
        except Exception as exc:
            logger.warning("Matrix: send_to_device %s failed: %s", event_name, exc)

    async def _post_emojis(self, session: _SasSession, emojis: list[tuple[str, str]]) -> None:
        """Post the emoji short-auth-string into the DM for comparison."""
        room_id = session.room_id or await self._find_dm_room(session.other_user)
        if not room_id:
            logger.warning(
                "Matrix: no DM room found for %s — cannot show emojis", session.other_user
            )
            return
        session.room_id = room_id
        emoji_chars = " ".join(e for e, _ in emojis)
        descriptions = ", ".join(f"{e} = {d}" for e, d in emojis)
        text = (
            "🔐 **Device verification (SAS)**\n\n"
            "Compare these emojis with the ones on your screen. "
            "If they **match in this order**, confirm the "
            "verification in your client.\n\n"
            f"{emoji_chars}\n\n"
            f"({descriptions})"
        )
        try:
            await self._adapter.send(room_id, text)
        except Exception as exc:
            logger.warning("Matrix: posting SAS emojis failed: %s", exc)

    async def _notify_cancelled(self, session: _SasSession, code: str) -> None:
        room_id = session.room_id or await self._find_dm_room(session.other_user)
        if not room_id:
            return
        try:
            await self._adapter.send(
                room_id,
                f"Verifikation abgebrochen ({code}). Du kannst es jederzeit erneut versuchen.",
            )
        except Exception as exc:
            logger.debug("Matrix: cancel notification failed: %s", exc)

    async def _find_dm_room(self, user_id: str) -> Optional[str]:
        """Find the DM room with *user_id* via m.direct account data."""
        try:
            resp = await self._client.get_account_data("m.direct")
            dm_data = _plain(getattr(resp, "content", None) or (resp if isinstance(resp, dict) else None))
            if not dm_data:
                return None
            rooms = dm_data.get(user_id) if hasattr(dm_data, "get") else None
            if not rooms:
                return None
            joined = set(self._adapter._joined_rooms) if hasattr(self._adapter, "_joined_rooms") else set()
            for room_id in rooms:
                if room_id in joined:
                    return room_id
        except Exception as exc:
            logger.debug("Matrix: DM room lookup failed: %s", exc)
        return None

    async def _cancel(self, session: _SasSession, code: str, reason: str) -> None:
        """Send a cancel event and drop the session."""
        try:
            await self._send(
                "m.key.verification.cancel",
                session,
                {
                    "code": code,
                    "reason": reason,
                    "transaction_id": session.transaction_id,
                },
            )
        except Exception:
            pass
        self._sessions.pop(session.transaction_id, None)
        logger.info("Matrix: SAS verification cancelled (%s: %s)", code, reason)

    async def _finalize(self, session: _SasSession) -> None:
        """Mark the verification as complete."""
        self._sessions.pop(session.transaction_id, None)
        logger.info(
            "Matrix: SAS verification with %s completed successfully 🎉",
            session.other_user,
        )
        room_id = session.room_id or await self._find_dm_room(session.other_user)
        if room_id:
            try:
                await self._adapter.send(
                    room_id,
                    "✅ Verification successful! The device is now marked as "
                    "verified.",
                )
            except Exception as exc:
                logger.debug("Matrix: success notification failed: %s", exc)

    def _expire_old_sessions(self) -> None:
        now = time.time()
        stale = [
            txid
            for txid, s in self._sessions.items()
            if now - s.created_at > _SESSION_TTL_SECONDS
        ]
        for txid in stale:
            self._sessions.pop(txid, None)

    # ------------------------------------------------------------------
    # Identity helpers
    # ------------------------------------------------------------------

    @property
    def _user_id(self) -> str:
        return str(getattr(self._client, "mxid", "") or "")

    @property
    def _device_id(self) -> str:
        return str(getattr(self._client, "device_id", "") or "")
