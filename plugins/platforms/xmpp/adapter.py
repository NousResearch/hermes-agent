"""XMPP (Jabber) platform plugin.

Built on slixmpp. Connects to any XMPP server, supports 1:1 chats and MUC
groupchat, and uses XEP-0363 (HTTP File Upload) for attachments.

Encryption posture: TLS-to-server is always on and plaintext is refused.
OMEMO (XEP-0384) end-to-end encryption ships with the platform and is on by
default: slixmpp-omemo is part of the ``platform.xmpp`` dependency group, so
``pip install 'hermes-agent[xmpp]'`` includes it and the lazy-install path
pulls it on first use — the same "encryption ships with the platform" model
Matrix uses for mautrix[encryption]. When it's present and ``omemo_enabled``
is true, outbound 1:1 messages are OMEMO-encrypted where the recipient has
published device keys, and inbound OMEMO messages are decrypted
automatically. Trust policy is BTBV (blind trust before verification) — the
pragmatic choice for a headless bot that cannot verify fingerprints
interactively. Note that XEP-0363 file uploads are NOT end-to-end encrypted
(no XEP-0454 yet).

Ships as a bundled platform plugin (zero core changes) — registered via
``register(ctx)`` per gateway/platforms/ADDING_A_PLATFORM.md. Dependency
specs live in ``tools/lazy_deps.py`` ("platform.xmpp") and the ``xmpp``
pyproject extra, matching the other bundled plugins with pip deps
(matrix, teams, dingtalk, feishu).

The OMEMO layer (storage, BTBV trust, session-manager recovery, legacy
oldmemo fallback) is ported from fastfinge/hermes-xmpp-plugin (MIT), which
itself derives from this adapter — thanks to @fastfinge for battle-testing
it in daily production use.
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import slixmpp

    SLIXMPP_AVAILABLE = True
except ImportError:
    SLIXMPP_AVAILABLE = False
    slixmpp = None  # type: ignore[assignment]

# OMEMO (XEP-0384) support is optional and ships with the platform, mirroring
# how Matrix bundles E2EE: slixmpp-omemo is part of the ``platform.xmpp``
# lazy-install group and the ``xmpp`` extra, so it auto-installs on first use.
# Everything degrades to TLS-only operation when it isn't installed.
try:
    from slixmpp.plugins import register_plugin as _register_slixmpp_plugin
    from slixmpp_omemo import TrustLevel, XEP_0384
    from omemo.storage import Just, Nothing, Storage

    SLIXMPP_OMEMO_AVAILABLE = True
except ImportError:
    SLIXMPP_OMEMO_AVAILABLE = False
    _register_slixmpp_plugin = None  # type: ignore[assignment]
    TrustLevel = None  # type: ignore[assignment,misc]
    XEP_0384 = None  # type: ignore[assignment,misc]
    Storage = object  # type: ignore[assignment,misc]
    Just = None  # type: ignore[assignment,misc]
    Nothing = None  # type: ignore[assignment,misc]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)


def check_xmpp_requirements() -> bool:
    """Confirm the ``platform.xmpp`` deps are installed.

    Lazy-installs the whole ``platform.xmpp`` group (slixmpp + slixmpp-omemo)
    via ``tools.lazy_deps.ensure("platform.xmpp")`` on first call if missing —
    the same auto-install pattern the other bundled platforms use. Then loads
    OMEMO (best-effort: a headless/offline box that can't pull the crypto
    stack still runs TLS-only). Returns True as long as core slixmpp is usable.
    """
    global SLIXMPP_AVAILABLE, slixmpp
    if not SLIXMPP_AVAILABLE:
        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("platform.xmpp", prompt=False)
        except Exception:
            return False
        try:
            import slixmpp as _slixmpp
        except ImportError:
            return False
        slixmpp = _slixmpp
        SLIXMPP_AVAILABLE = True
    # OMEMO is optional; installing/rebinding it must never fail the platform.
    _ensure_omemo_loaded()
    return True


def _ensure_omemo_loaded() -> bool:
    """Ensure slixmpp-omemo is installed and its symbols/classes are bound.

    Idempotent. On a fresh box the ``platform.xmpp`` ensure in
    ``check_xmpp_requirements`` already installs slixmpp-omemo; this rebinds
    the module-level OMEMO symbols (which were ``None`` at import time because
    the package wasn't present yet) and rebuilds the OMEMO plugin classes
    against the freshly imported base types — the ``ensure_and_bind`` idiom
    used by the Matrix adapter for its E2EE deps. Returns True when OMEMO is
    ready, False when it stays unavailable (adapter then runs TLS-only).
    """
    if SLIXMPP_OMEMO_AVAILABLE and _XEP_0384Impl is not None:
        return True

    def _import() -> Dict[str, Any]:
        from slixmpp.plugins import register_plugin
        from slixmpp_omemo import TrustLevel as _TrustLevel, XEP_0384 as _XEP_0384
        from omemo.storage import Just as _Just, Nothing as _Nothing, Storage as _Storage
        return {
            "_register_slixmpp_plugin": register_plugin,
            "TrustLevel": _TrustLevel,
            "XEP_0384": _XEP_0384,
            "Storage": _Storage,
            "Just": _Just,
            "Nothing": _Nothing,
            "SLIXMPP_OMEMO_AVAILABLE": True,
        }

    try:
        from tools.lazy_deps import ensure_and_bind
    except Exception:
        return False
    if not ensure_and_bind("platform.xmpp", _import, globals(), prompt=False):
        return False
    # Base types are now bound in globals(); (re)build the OMEMO subclasses.
    _build_omemo_classes()
    return _XEP_0384Impl is not None


# ---------------------------------------------------------------------
# OMEMO (XEP-0384) — JSON-file storage + BTBV trust policy
# Ported from fastfinge/hermes-xmpp-plugin (MIT), including its two
# production fixes: recovery from a failed session-manager init, and a
# legacy-OMEMO (oldmemo) fallback for servers where OMEMO:2 init fails.
#
# The classes are built by ``_build_omemo_classes()`` rather than at module
# top so they can be (re)constructed after a lazy install rebinds the base
# types (``Storage`` / ``XEP_0384`` are ``None``/``object`` until then).
# ---------------------------------------------------------------------

_StorageImpl = None  # type: ignore[misc,assignment]  # built by _build_omemo_classes()
_XEP_0384Impl = None  # type: ignore[misc,assignment]


def _build_omemo_classes() -> None:
    """Define the OMEMO storage + plugin classes against the imported base
    types and bind them into module globals. Called at import when the extra
    is already present, and again after a lazy install rebinds the base types.
    """
    global _StorageImpl, _XEP_0384Impl

    class _StorageImpl(Storage):  # type: ignore[misc]
        """Simple JSON-file backed OMEMO storage.

        Holds the bot's identity key, device list, and per-contact
        sessions/trust. Losing this file changes the bot's OMEMO identity
        and every contact's client will warn about a new device.
        """

        def __init__(self, json_file_path: Path) -> None:
            super().__init__()  # type: ignore[misc]
            self._path = json_file_path
            self._data: Dict[str, Any] = {}
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except FileNotFoundError:
                pass
            except Exception:
                # A corrupt store means a NEW OMEMO identity: every contact's
                # client will warn about an unknown device. Be loud about it.
                logger.warning(
                    "OMEMO: key store %s is unreadable/corrupt — starting with "
                    "a fresh identity. Contacts will see a new-device warning.",
                    self._path, exc_info=True,
                )

        async def _load(self, key: str) -> Any:  # type: ignore[override]
            if key in self._data:
                return Just(self._data[key])
            return Nothing()

        async def _store(self, key: str, value: Any) -> None:  # type: ignore[override]
            self._data[key] = value
            self._save()

        async def _delete(self, key: str) -> None:
            self._data.pop(key, None)
            self._save()

        def _save(self) -> None:
            # Atomic replace so a crash mid-write can't corrupt the key store
            # (a corrupt store = a new OMEMO identity). The temp file is
            # created 0600 from the start — it holds private keys.
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2)
                os.replace(tmp, self._path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise

    class _XEP_0384Impl(XEP_0384):  # type: ignore[misc,valid-type]
        """Concrete OMEMO plugin with BTBV and JSON-file storage."""

        default_config = {
            "fallback_message": "This message is OMEMO encrypted.",
            "json_file_path": None,
        }

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[misc]
            self.__storage = None  # type: ignore[var-annotated]

        def plugin_init(self) -> None:
            if not self.json_file_path:  # type: ignore[attr-defined]
                raise Exception("OMEMO JSON file path not specified.")
            self.__storage = _StorageImpl(Path(self.json_file_path))  # type: ignore[attr-defined]
            super().plugin_init()  # type: ignore[misc]

        @property
        def storage(self):
            return self.__storage

        @property
        def _btbv_enabled(self) -> bool:
            # Blind trust before verification: a headless bot cannot verify
            # fingerprints interactively, so new devices are trusted on first
            # sight until the operator manually distrusts one.
            return True

        async def _devices_blindly_trusted(  # type: ignore[override]
            self,
            blindly_trusted,
            identifier,
        ) -> None:
            logger.info(
                "OMEMO: blindly trusted %d device(s) [%s]",
                len(blindly_trusted), identifier,
            )

        async def _prompt_manual_trust(  # type: ignore[override]
            self,
            manually_trusted,
            identifier,
        ) -> None:
            # BTBV is enabled so this is rare. Log and auto-distrust to avoid
            # blocking the send path on a prompt nobody can answer.
            session_manager = await self.get_session_manager()
            for device in manually_trusted:
                logger.warning(
                    "OMEMO: manual trust required for %s %s — distrusting to avoid block",
                    device.bare_jid,
                    device.device_id,
                )
                await session_manager.set_trust(
                    device.bare_jid,
                    device.identity_key,
                    TrustLevel.DISTRUSTED.value,
                )

        async def get_session_manager(self):  # type: ignore[override]
            """Return a usable OMEMO session manager, recovering from failed init.

            slixmpp-omemo caches the in-flight initialization task. If that task
            fails once (for example, a server times out while fetching a twomemo
            device list), later decrypt/encrypt attempts await the same failed
            task forever. That makes one transient PubSub hiccup poison OMEMO
            until the whole gateway restarts. Reset the cached task on failure.

            Some servers/clients still behave much better with legacy OMEMO
            (oldmemo) than OMEMO:2 (twomemo). If initialization fails while
            touching the twomemo namespace, fall back to an oldmemo-only session
            manager so existing Conversations/Gajim-style devices can still
            decrypt instead of getting a useless "message from myself" blob.
            """
            try:
                return await super().get_session_manager()  # type: ignore[misc]
            except Exception as exc:
                self._reset_failed_session_manager()
                if "urn:xmpp:omemo:2" in str(exc):
                    logger.warning(
                        "OMEMO: twomemo initialization failed (%s); falling back to legacy OMEMO only",
                        exc,
                    )
                    try:
                        manager = await self._create_oldmemo_only_session_manager()
                        setattr(self, "_XEP_0384__session_manager", manager)
                        self.xmpp.event("omemo_initialized")  # type: ignore[attr-defined]
                        return manager
                    except Exception:
                        self._reset_failed_session_manager()
                        logger.exception("OMEMO: legacy fallback initialization failed")
                raise

        def _reset_failed_session_manager(self) -> None:
            task = getattr(self, "_XEP_0384__session_manager_task", None)
            if task is not None and not getattr(task, "done", lambda: True)():
                task.cancel()
            setattr(self, "_XEP_0384__session_manager_task", None)
            setattr(self, "_XEP_0384__session_manager", None)

        async def _create_oldmemo_only_session_manager(self):
            from slixmpp_omemo.xep_0384 import _make_session_manager
            from oldmemo.oldmemo import Oldmemo

            session_manager_cls = _make_session_manager(self.xmpp, self)  # type: ignore[attr-defined]
            manager = session_manager_cls.__new__(session_manager_cls)
            storage = self.storage
            if storage is None:
                raise RuntimeError("OMEMO storage is not initialized")
            backend = Oldmemo(storage)
            name_map = {
                "__backends": [backend],
                "__storage": storage,
                "__own_bare_jid": self.xmpp.boundjid.bare,  # type: ignore[attr-defined]
                "__undecided_trust_level_name": TrustLevel.UNDECIDED.value,
                "__synchronizing": False,
            }
            for name, value in name_map.items():
                setattr(manager, f"_SessionManager{name}", value)
            own_device_id = (await storage.load_primitive("/own_device_id", int)).from_just()
            setattr(manager, "_SessionManager__own_device_id", own_device_id)
            return manager

    # The ``global`` declaration above makes the two ``class`` statements bind
    # directly to the module-level names, so nothing else is needed here.


if SLIXMPP_OMEMO_AVAILABLE:
    _build_omemo_classes()


@dataclass
class _MucRoom:
    """A configured MUC room the adapter joins on connect."""
    room: str           # bare room JID, e.g. "team@conference.example.org"
    nick: Optional[str] = None  # optional override; falls back to muc_nick


def _parse_muc_rooms(value: str, default_nick: Optional[str]) -> List[_MucRoom]:
    rooms: List[_MucRoom] = []
    for entry in (value or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "/" in entry:
            room, _, nick = entry.partition("/")
            rooms.append(_MucRoom(room=room.strip(), nick=nick.strip() or default_nick))
        else:
            rooms.append(_MucRoom(room=entry, nick=default_nick))
    return rooms


class XmppAdapter(BasePlatformAdapter):
    """slixmpp-backed adapter satisfying BasePlatformAdapter.

    The slixmpp client is constructed lazily in ``connect()``; ``__init__``
    only parses config so unit tests can introspect adapter state without
    paying the slixmpp import / event-loop cost.
    """

    # How long connect() waits for the session to establish before declaring a
    # retryable failure. Kept under the gateway's per-platform connect timeout
    # (_PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT = 30s) so our own diagnosis wins.
    _CONNECT_TIMEOUT_SECS = 20.0

    # Per-stanza body length cap (Unicode code-points). XMPP itself defines no
    # protocol-level body limit and slixmpp exposes no negotiated stanza-size
    # value, so this is a conservative default that every common server
    # (Prosody, ejabberd, etc.) accepts with room to spare for OMEMO/base64
    # inflation and XML escaping. Overridable via config/env. The gateway's
    # streaming consumer reads this attribute (it clips at 4096 for adapters
    # that don't define it) and the adapter's own send paths chunk against it.
    MAX_MESSAGE_LENGTH = 10000

    # send() chunks via _chunk_body(MAX_MESSAGE_LENGTH); without this flag
    # gateway/delivery.py truncates cron output at 4,000 chars before send()
    # ever sees it.
    splits_long_messages = True

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("xmpp"))
        extra = config.extra or {}

        # Operators can tune the per-message cap (a server with a tighter
        # max_stanza_size can lower it). Falls back to the class default on
        # any bad value.
        _max_len_raw = extra.get("max_message_length") or os.getenv("XMPP_MAX_MESSAGE_LENGTH")
        if _max_len_raw:
            try:
                _max_len = int(_max_len_raw)
                if _max_len > 0:
                    self.MAX_MESSAGE_LENGTH = _max_len
            except (TypeError, ValueError):
                logger.warning("xmpp: invalid max_message_length %r, using default", _max_len_raw)

        self.jid: str = str(extra.get("jid", ""))
        self._password: str = str(extra.get("password", ""))
        self.host: Optional[str] = extra.get("host")
        self.port: int = int(extra.get("port", 5222))

        # Direct TLS ("TLS from byte one", XEP-0368 — the classic port-5223
        # posture). Auto-enabled when the operator points us at port 5223,
        # overridable either way via XMPP_DIRECT_TLS / direct_tls. slixmpp
        # falls back to STARTTLS if the direct-TLS attempt fails, and
        # plaintext stays refused in every configuration.
        _direct_raw = str(
            extra.get("direct_tls", os.getenv("XMPP_DIRECT_TLS", ""))
        ).strip().lower()
        if _direct_raw in ("1", "true", "yes", "on"):
            self.direct_tls: bool = True
        elif _direct_raw in ("0", "false", "no", "off"):
            self.direct_tls = False
        else:
            self.direct_tls = self.port == 5223
        self.muc_nick: str = extra.get("muc_nick") or self._default_nick()
        self.muc_rooms: List[_MucRoom] = _parse_muc_rooms(
            extra.get("muc_rooms", ""), self.muc_nick
        )

        # Allow-list: bare JIDs the bot will accept inbound from.
        self.allow_all_users: bool = (
            os.getenv("XMPP_ALLOW_ALL_USERS", "").strip().lower() in ("1", "true", "yes")
        )
        allowed_env = os.getenv("XMPP_ALLOWED_USERS", "").strip()
        self.allowed_users = {
            j.strip() for j in allowed_env.split(",") if j.strip()
        }

        # True only for the one-shot send_xmpp_message() path: connect as an
        # ephemeral second resource without taking (or releasing) the
        # per-account scoped lock held by the running gateway.
        self._ephemeral_sender = False

        # Lazily created in connect()
        self.client: Optional[Any] = None
        self._process_task: Optional[asyncio.Task] = None
        self._session_ready: Optional[asyncio.Event] = None
        self._self_bare = self._bare(self.jid)

        # True between session_start and the next disconnect. connect() waits
        # on _session_ready (the asyncio.Event below) to know the session came
        # up; this bool is the durable record of it for observability/tests.
        # The escalate-vs-stay-quiet decision in _on_disconnected keys off
        # _closing / has_fatal_error, not this flag.
        self._session_established = False
        # Set while disconnect() is tearing the client down on purpose, so the
        # "disconnected" handler doesn't mistake a deliberate close for an
        # outage and escalate it to a reconnect (#28919).
        self._closing = False

        # Set of bare room JIDs we've configured for groupchat send routing.
        self._known_mucs = {r.room for r in self.muc_rooms}
        # Bare JIDs we have observed sending us 1:1 ("chat") messages. Keeps
        # _is_muc() from misclassifying a real user as a group when the
        # user's domain happens to match a MUC-style prefix (e.g. Snikket
        # installs to "chat.example.com", so users are alice@chat.example.com).
        self._known_dms: set[str] = set()

        # Track which optional plugins registered successfully so feature
        # methods (chat states, HTTP upload) can no-op gracefully if a plugin
        # is missing instead of raising TypeError on unknown kwargs.
        self._registered_plugins: set[str] = set()

        # OMEMO: enabled by default when slixmpp-omemo is installed; the
        # operator can force it off with XMPP_OMEMO_ENABLED=false (or the
        # omemo_enabled config key). The JSON key store defaults to
        # ~/.hermes/xmpp_omemo.json and must persist across restarts.
        omemo_raw = str(
            extra.get("omemo_enabled", os.getenv("XMPP_OMEMO_ENABLED", "true"))
        ).strip().lower()
        self._omemo_enabled: bool = omemo_raw not in ("false", "0", "no", "off")
        self._omemo_storage_path: str = str(
            extra.get("omemo_storage_path")
            or os.getenv("XMPP_OMEMO_STORAGE_PATH", "")
        ) or str(
            Path(os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes")))
            / "xmpp_omemo.json"
        )

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        # ``is_reconnect`` is part of the BasePlatformAdapter.connect contract.
        # XMPP keeps no client-side offline queue (the server stores offline
        # messages), so there is nothing extra to preserve or drop on a
        # watcher-driven reconnect — we accept the flag and connect normally.
        self._closing = False
        self._session_established = False
        # One gateway per account: two adapters logged into the same JID both
        # receive inbound stanzas (the server fans out by resource), so every
        # message would be handled twice. Lock on the normalized bare JID —
        # localpart/domain are case-insensitive. A same-process reacquire
        # succeeds, so watcher-driven reconnects pass. Released from
        # disconnect(), which every failed-connect path below goes through.
        # The one-shot cron sender (send_xmpp_message) skips this: it attaches
        # as a short-lived second resource, which XMPP permits by design.
        if not self._ephemeral_sender:
            lock_jid = self._bare(self.jid).lower()
            if not self._acquire_platform_lock(
                "xmpp", lock_jid, f"XMPP account {lock_jid}"
            ):
                # The base helper leaves the identity set on failure; clear it
                # so a later disconnect() from this never-connected adapter
                # can't release a lock owned by a same-PID profile.
                self._platform_lock_identity = None
                return False
        client = slixmpp.ClientXMPP(self.jid, self._password)
        # Plugins we need: chat states, MUC, HTTP File Upload, OOB, disco,
        # ping, delayed delivery (MUC history detection), EME hints.
        for plugin in ("xep_0030", "xep_0045", "xep_0066", "xep_0085", "xep_0199",
                       "xep_0203", "xep_0363", "xep_0380"):
            try:
                client.register_plugin(plugin)
                self._registered_plugins.add(plugin)
            except Exception:
                logger.warning("xmpp: failed to register slixmpp plugin %s", plugin)

        # OMEMO (XEP-0384): optional, requires slixmpp-omemo.
        omemo_active = False
        if self._omemo_enabled and SLIXMPP_OMEMO_AVAILABLE:
            try:
                _register_slixmpp_plugin(_XEP_0384Impl)
                client.register_plugin(
                    "xep_0384",
                    {"json_file_path": self._omemo_storage_path},
                )
                self._registered_plugins.add("xep_0384")
                omemo_active = True
            except Exception:
                logger.exception("xmpp: failed to register OMEMO plugin")
        elif self._omemo_enabled and not SLIXMPP_OMEMO_AVAILABLE:
            logger.warning(
                "xmpp: OMEMO enabled but slixmpp-omemo is not installed — "
                "running TLS-only. It normally auto-installs with the platform; "
                "install manually with: pip install 'hermes-agent[xmpp]'"
            )

        # Enforce TLS, refuse plaintext. The load-bearing knob in current
        # slixmpp is ``enable_plaintext = False``. ``enable_direct_tls``
        # follows the direct-TLS config (auto-on for port 5223) with STARTTLS
        # kept available as a fallback; the legacy ``use_starttls`` /
        # ``force_starttls`` names are set too so the posture holds across the
        # supported slixmpp range (older releases honored those instead).
        client.enable_starttls = True
        client.enable_direct_tls = self.direct_tls
        client.enable_plaintext = False
        client.use_starttls = True       # legacy slixmpp name
        client.force_starttls = True     # legacy slixmpp name

        client.add_event_handler("session_start", self._on_session_start)
        # slixmpp dispatches MUC stanzas to BOTH "message" and
        # "groupchat_message", so registering _on_message for both would run
        # it twice for every group-chat message (double agent dispatch).
        # Register "message" only — it already covers chat/normal/groupchat.
        client.add_event_handler("message", self._on_message)
        client.add_event_handler("disconnected", self._on_disconnected)
        # Handle "failed_all_auth" (fired once, after every SASL mechanism the
        # server offered has been exhausted) rather than "failed_auth" (fired
        # once per *rejected* mechanism). slixmpp moves on to the next
        # mechanism after each failed_auth, so reacting to failed_auth marks
        # the adapter dead even when a later mechanism (e.g. SCRAM after a
        # rejected PLAIN) succeeds — silently poisoning a working connection
        # (#28919).
        client.add_event_handler("failed_all_auth", self._on_failed_all_auth)

        # Presence subscriptions. slixmpp's blanket auto_authorize would
        # approve *anyone*; gate on the allow-list instead so an unauthorized
        # JID can't force a roster entry. auto_subscribe stays off — we send
        # our own subscribe-back only to approved peers so the subscription is
        # mutual (required for OMEMO device-list PEP pushes to reach the peer's
        # client and clear its stale "no OMEMO" cache).
        client.roster.auto_authorize = False
        client.roster.auto_subscribe = False
        client.add_event_handler("presence_subscribe", self._on_subscribe)

        self.client = client
        self._session_ready = asyncio.Event()

        # slixmpp >=1.x uses ``connect(host, port)``; very old releases used
        # ``connect(address=(host, port))``. Try the modern signature first and
        # fall back WITHOUT dropping the configured host/port — an earlier
        # version of this code silently discarded XMPP_HOST/XMPP_PORT on the
        # fallback, sending the bot to SRV/JID-domain instead of the operator's
        # server.
        if self.host:
            try:
                client.connect(host=self.host, port=self.port)
            except TypeError:
                client.connect(address=(self.host, self.port))
        else:
            client.connect()

        loop = asyncio.get_event_loop()
        self._process_task = loop.create_task(self._run_process())

        # ``connect()`` only kicks off the TCP/TLS/SASL handshake (slixmpp
        # returns a Future, never a bool — so there is no synchronous failure to
        # check here). Wait, bounded, for the session to actually establish
        # before reporting success. Otherwise an unreachable server, a wrong
        # host, or a rejected login would leave the gateway believing XMPP is
        # connected over a dead socket with nothing driving recovery (#28919).
        try:
            await asyncio.wait_for(
                self._session_ready.wait(), timeout=self._CONNECT_TIMEOUT_SECS
            )
        except asyncio.TimeoutError:
            if not self.has_fatal_error:
                self._set_fatal_error(
                    "xmpp_connect_timeout",
                    f"XMPP session did not establish within "
                    f"{self._CONNECT_TIMEOUT_SECS:g}s — check XMPP_HOST/XMPP_PORT "
                    "and server reachability",
                    retryable=True,
                )
            await self.disconnect()
            return False
        if self.has_fatal_error:
            # e.g. failed_all_auth fired (bad credentials) while we waited.
            await self.disconnect()
            return False

        if omemo_active:
            logger.info(
                "xmpp: OMEMO enabled (key store: %s)", self._omemo_storage_path
            )
        else:
            logger.warning(
                "XMPP adapter is running without OMEMO. Messages are encrypted in "
                "transit (TLS) but visible to your XMPP server operator."
            )
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._closing = True
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception:
                logger.exception("xmpp: error during disconnect()")
        if self._process_task is not None:
            self._process_task.cancel()
            self._process_task = None
        self.client = None
        self._release_platform_lock()
        self._mark_disconnected()

    async def _run_process(self) -> None:
        """slixmpp's process loop runs in the existing asyncio loop."""
        try:
            await self.client.disconnected
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("xmpp: process loop crashed")

    async def _on_session_start(self, _event: Any) -> None:
        self._session_established = True
        self.client.send_presence()
        try:
            await self.client.get_roster()
        except Exception:
            logger.exception("xmpp: get_roster failed")
        for room in self.muc_rooms:
            try:
                self.client.plugin["xep_0045"].join_muc(room.room, room.nick or self.muc_nick)
            except Exception:
                logger.exception("xmpp: failed to join MUC %s", room.room)
        # Proactively request a presence subscription to each allow-listed
        # peer. A mutual subscription is what makes the peer's client receive
        # our OMEMO device-list PEP updates (and clears a stale "no OMEMO for
        # this contact" cache); it also lets us see their presence. Only bare
        # JIDs we already trust get an unsolicited request.
        for jid in self.allowed_users:
            try:
                self.client.send_presence(pto=jid, ptype="subscribe")
            except Exception:
                logger.debug("xmpp: subscribe request to %s failed", jid, exc_info=True)
        if self._session_ready is not None:
            self._session_ready.set()

    async def _on_subscribe(self, presence: Any) -> None:
        """Approve an inbound presence-subscription request from an allowed peer.

        Runs the same allow-list gate as inbound messages, so only trusted
        JIDs get into the bot's roster. On approval we also subscribe back
        (if not already) so the subscription is mutual and our OMEMO
        device-list updates push to the peer's client.
        """
        if self.client is None:
            return
        try:
            requester = self._bare(str(presence.get_from()))
        except Exception:
            return
        if not (self.allow_all_users or requester in self.allowed_users):
            logger.debug("xmpp: ignoring subscription request from %s (not allowed)", requester)
            return
        try:
            self.client.send_presence(pto=requester, ptype="subscribed")
            self.client.send_presence(pto=requester, ptype="subscribe")
            logger.info("xmpp: approved presence subscription for %s", requester)
        except Exception:
            logger.debug("xmpp: failed to approve subscription for %s", requester, exc_info=True)

    async def _on_disconnected(self, _event: Any) -> None:
        self._session_established = False
        # A deliberate disconnect(), or a failure we have already reported
        # (e.g. failed_all_auth set a fatal error), needs no escalation —
        # just record the disconnect and let the existing status stand.
        if self._closing or self.has_fatal_error:
            self._mark_disconnected()
            return
        # An unexpected drop of a live connection (server restart, network
        # blip, ...). Escalate to a *retryable* fatal error AND notify the
        # gateway so its reconnect watcher actually retries. Marking the
        # adapter disconnected without notifying would leave the platform
        # down with nothing driving a reconnect — a silently dead bridge in
        # an otherwise-healthy gateway (#28919).
        self._set_fatal_error(
            "xmpp_connection_lost", "XMPP connection lost", retryable=True
        )
        await self._notify_fatal_error()

    async def _on_failed_all_auth(self, _event: Any) -> None:
        # Every SASL mechanism the server offered has been rejected —
        # credentials are wrong or the account is disabled. Non-retryable,
        # and we notify the gateway so the failure actually surfaces instead
        # of leaving a connected-looking but dead adapter (#28919).
        self._set_fatal_error(
            "xmpp_auth_failed",
            "XMPP authentication failed — check XMPP_JID/XMPP_PASSWORD",
            retryable=False,
        )
        await self._notify_fatal_error()

    # -----------------------------------------------------------------
    # Inbound
    # -----------------------------------------------------------------

    async def _on_message(self, stanza: Any) -> None:
        """Convert a slixmpp Message stanza into a MessageEvent and dispatch."""
        try:
            stanza_type = stanza["type"]
            if stanza_type in ("error", "headline"):
                return
            if stanza_type not in ("chat", "groupchat", "normal"):
                return

            from_jid = stanza.get_from()
            from_full = str(from_jid)
            from_bare = getattr(from_jid, "bare", None) or self._bare(from_full)
            from_resource = getattr(from_jid, "resource", "") or ""

            # Filter our own echoes (especially common in MUC).
            if from_bare == self._self_bare:
                return

            # In a MUC the stanza's `from` is room@host/nick, so the bare-JID
            # check above never matches our own JID. Reflected messages (and
            # MUC history replay on join) carry our own nick as the resource —
            # skip them, otherwise the bot replies to itself in an endless
            # loop.
            if stanza_type == "groupchat":
                room_nick = self._muc_room_nick(from_bare)
                if from_resource and from_resource == room_nick:
                    return
                # MUC history replay carries a delay stamp; ignore old messages.
                try:
                    if stanza.get_plugin("delay", check=True) is not None:
                        return
                except (AttributeError, TypeError):
                    pass

            # Authorization runs on the stanza ENVELOPE, before any OMEMO
            # work. Decrypting first would let unauthorized (federated)
            # senders force X3DH session builds, blind-trust key-store writes,
            # and the library's automatic empty key-exchange replies — CPU,
            # disk, and a liveness oracle, all pre-allow-list.
            if stanza_type == "groupchat":
                chat_type = "group"
                chat_id = from_bare  # room JID
                user_name = from_resource or None  # MUC nick lives in the resource
                user_id = self._muc_real_jid(stanza) or chat_id
            else:
                chat_type = "dm"
                chat_id = from_bare
                user_name = None
                user_id = from_bare

            if not self._is_authorized(chat_type=chat_type, chat_id=chat_id, user_jid=user_id):
                logger.debug(
                    "xmpp: dropping unauthorized %s from %s in %s",
                    chat_type, user_id, chat_id,
                )
                return

            if chat_type == "dm":
                # Remember this is a real 1:1 peer so we never reply to it as
                # a group, regardless of the domain's prefix (see _known_dms).
                # Only authorized peers land here, so the set stays bounded.
                self._known_dms.add(from_bare)

            body = stanza["body"] or ""
            stanza_to_dispatch = stanza

            # OMEMO decryption (XEP-0384). A decrypt failure is dropped, not
            # dispatched: the ciphertext body ("This message is OMEMO
            # encrypted.") would otherwise reach the agent as user text.
            if (
                self.client is not None
                and "xep_0384" in self._registered_plugins
                and SLIXMPP_OMEMO_AVAILABLE
            ):
                xep_0384 = self.client["xep_0384"]
                if xep_0384.is_encrypted(stanza):
                    try:
                        decrypted_stanza, device_info = await xep_0384.decrypt_message(stanza)
                        body = decrypted_stanza.get("body", "") or ""
                        stanza_to_dispatch = decrypted_stanza
                        logger.debug(
                            "OMEMO: decrypted message from %s (device %s)",
                            from_bare, device_info.device_id,
                        )
                    except Exception as exc:
                        logger.warning(
                            "OMEMO: failed to decrypt message from %s: %s",
                            from_bare, exc,
                        )
                        return

            if not body:
                return

            source = self.build_source(
                chat_id=chat_id,
                chat_type=chat_type,
                user_id=user_id,
                user_name=user_name,
            )
            event = MessageEvent(
                text=body,
                message_type=MessageType.TEXT,
                source=source,
                raw_message=stanza_to_dispatch,
                message_id=stanza.get("id") or None,
            )
            await self.handle_message(event)
        except Exception:
            logger.exception("xmpp: error handling inbound stanza")

    def _muc_real_jid(self, stanza: Any) -> Optional[str]:
        """Best-effort extraction of the MUC sender's real bare JID.

        Only returned when the room exposes occupants' real JIDs (semi-anon
        or non-anon rooms). For fully anonymous rooms this is unavailable
        and we fall back to the room JID for authorization.
        """
        try:
            muc = stanza.get("muc")
            if muc and getattr(muc, "jid", None):
                return self._bare(str(muc["jid"]))
        except Exception:
            pass
        return None

    def _is_authorized(self, *, chat_type: str, chat_id: str, user_jid: str) -> bool:
        if self.allow_all_users:
            return True
        if chat_type == "group":
            # MUC presence is gated by room membership: the bot operator opts
            # into a room by configuring it in XMPP_MUC_ROOMS. Anonymous and
            # semi-anonymous rooms hide participants' real JIDs from the
            # bot, so per-user filtering is structurally impossible — the
            # room itself is the access boundary.
            return chat_id in self._known_mucs
        if not self.allowed_users:
            # Empty allow-list AND no allow-all flag → reject by default.
            # This matches the security posture of the Signal / WhatsApp
            # adapters: misconfiguration must not silently accept the world.
            return False
        return self._bare(user_jid) in self.allowed_users

    # -----------------------------------------------------------------
    # Outbound
    # -----------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if self.client is None:
            return SendResult(success=False, error="xmpp not connected", retryable=True)
        mtype = "groupchat" if self._is_muc(chat_id) else "chat"

        # OMEMO-encrypt 1:1 messages when the plugin is active. Group chats
        # stay plaintext: reliable MUC OMEMO needs non-anonymous rooms and
        # per-occupant device discovery, deferred for now. A failed
        # encryption falls back to plaintext with a warning rather than
        # dropping the reply — the recipient opted into this bot; a lost
        # answer is worse than a TLS-only one.
        if (
            "xep_0384" in self._registered_plugins
            and SLIXMPP_OMEMO_AVAILABLE
            and mtype == "chat"
        ):
            try:
                return await self._send_encrypted(chat_id, content)
            except Exception as exc:
                logger.warning(
                    "OMEMO: encryption failed for %s (%s), sending plaintext",
                    chat_id, exc,
                )
                # fall through to plain send

        try:
            last_stanza = None
            last_msg_id = None
            for chunk in self._chunk_body(content):
                last_stanza = self.client.send_message(
                    mto=chat_id,
                    mbody=chunk,
                    mtype=mtype,
                )
                try:
                    last_msg_id = last_stanza["id"]
                except Exception:
                    pass
            return SendResult(success=True, message_id=last_msg_id, raw_response=last_stanza)
        except Exception as exc:
            logger.exception("xmpp: send failed")
            return SendResult(success=False, error=str(exc), retryable=True)

    # -----------------------------------------------------------------
    # Message chunking
    # -----------------------------------------------------------------

    def _chunk_body(self, content: str) -> List[str]:
        """Split a message body into stanza-sized chunks.

        Prefers the inherited ``truncate_message`` (preserves code-fence
        boundaries, adds "(1/N)" indicators) against ``MAX_MESSAGE_LENGTH``.
        Falls back to a simple boundary-aware splitter if that helper fails.
        Always returns at least one chunk so callers can loop unconditionally;
        an empty body yields a single empty chunk.
        """
        if not content:
            return [content]
        limit = self.MAX_MESSAGE_LENGTH
        if len(content) <= limit:
            return [content]
        try:
            chunks = list(self.truncate_message(content, limit))
            if chunks:
                return chunks
        except Exception:
            logger.debug(
                "xmpp: truncate_message failed, using fallback splitter",
                exc_info=True,
            )
        return self._fallback_split(content, limit)

    @staticmethod
    def _fallback_split(content: str, limit: int) -> List[str]:
        """Simple length-bounded splitter, preferring newline/space boundaries."""
        chunks: List[str] = []
        remaining = content
        while len(remaining) > limit:
            window = remaining[:limit]
            split_at = window.rfind("\n")
            if split_at < limit // 2:
                split_at = window.rfind(" ")
            if split_at < 1:
                split_at = limit
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip("\n ")
        if remaining:
            chunks.append(remaining)
        return chunks or [content]

    # -----------------------------------------------------------------
    # OMEMO outbound
    # -----------------------------------------------------------------

    async def _send_encrypted(self, chat_id: str, content: str) -> SendResult:
        """Send an OMEMO-encrypted 1:1 chat message, splitting long bodies.

        Each chunk is encrypted and sent as its own stanza. The result of the
        last chunk is returned; the first failing chunk short-circuits.

        Failure semantics matter here: if NO chunk went out yet, the
        exception propagates so send() can fall back to plaintext. Once any
        encrypted chunk has been delivered, a later failure returns an error
        result instead — falling back would resend already-delivered content
        as plaintext (duplication + a silent, retroactive E2E downgrade).
        """
        if self.client is None:
            return SendResult(success=False, error="xmpp not connected", retryable=True)
        last_result = SendResult(success=True)
        for idx, chunk in enumerate(self._chunk_body(content)):
            try:
                last_result = await self._send_encrypted_one(chat_id, chunk)
            except Exception:
                if idx == 0:
                    raise  # nothing delivered yet — plaintext fallback is safe
                logger.warning(
                    "OMEMO: chunk %d failed after %d encrypted chunk(s) were "
                    "delivered — not falling back to plaintext", idx + 1, idx,
                )
                return SendResult(
                    success=False,
                    error="OMEMO encryption failed mid-message",
                    retryable=True,
                )
            if not last_result.success:
                return last_result
        return last_result

    async def _send_encrypted_one(self, chat_id: str, content: str) -> SendResult:
        """Encrypt and send a single OMEMO stanza (one chunk)."""
        if self.client is None:
            return SendResult(success=False, error="xmpp not connected", retryable=True)
        client = self.client
        xep_0384 = client["xep_0384"]
        stanza = client.make_message(mto=chat_id, mtype="chat")
        stanza["body"] = content
        stanza.set_from(client.boundjid)

        recipient = slixmpp.JID(chat_id)
        message, encryption_errors = await xep_0384.encrypt_message(stanza, {recipient})

        if encryption_errors:
            logger.info("OMEMO: encryption non-critical errors: %s", encryption_errors)

        if message is None:
            # Recipient has no usable OMEMO devices — plaintext is the only
            # way to reach them.
            logger.warning("OMEMO: nothing to encrypt for %s, sending plaintext", chat_id)
            plain = client.send_message(mto=chat_id, mbody=content, mtype="chat")
            msg_id = None
            try:
                msg_id = plain["id"]
            except Exception:
                pass
            return SendResult(success=True, message_id=msg_id, raw_response=plain)

        # Explicit Message Encryption (XEP-0380) hint so non-OMEMO clients
        # show "this message is encrypted" instead of garbage.
        if "xep_0380" in self._registered_plugins:
            try:
                import oldmemo
                ns = oldmemo.oldmemo.NAMESPACE
                message["eme"]["namespace"] = ns
                message["eme"]["name"] = client["xep_0380"].mechanisms[ns]
            except Exception:
                pass

        # Attach XEP-0085 chat state after encryption (encrypt_message clears
        # non-OMEMO elements, so it must be set on the encrypted stanza).
        if "xep_0085" in self._registered_plugins:
            try:
                message["chat_state"] = "active"
            except Exception:
                pass
        message.send()
        msg_id = None
        try:
            msg_id = message["id"]
        except Exception:
            pass
        return SendResult(success=True, message_id=msg_id, raw_response=message)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        if self.client is None or "xep_0085" not in self._registered_plugins:
            return
        mtype = "groupchat" if self._is_muc(chat_id) else "chat"
        try:
            self.client.send_message(mto=chat_id, mtype=mtype, mchat_state="composing")
        except Exception:
            logger.debug("xmpp: send_typing failed", exc_info=True)

    async def stop_typing(self, chat_id: str) -> None:
        if self.client is None or "xep_0085" not in self._registered_plugins:
            return
        mtype = "groupchat" if self._is_muc(chat_id) else "chat"
        try:
            self.client.send_message(mto=chat_id, mtype=mtype, mchat_state="active")
        except Exception:
            logger.debug("xmpp: stop_typing failed", exc_info=True)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        return await self._upload_and_send(chat_id, image_path, caption)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        # Param name must match BasePlatformAdapter.send_document(file_path=...)
        # — the gateway calls these by keyword, so a renamed positional arg
        # raises TypeError and the attachment silently never sends.
        return await self._upload_and_send(chat_id, file_path, caption)

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        **kwargs,
    ) -> SendResult:
        # Param name must match BasePlatformAdapter.send_voice(audio_path=...).
        return await self._upload_and_send(chat_id, audio_path, caption=None)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        # Param name must match BasePlatformAdapter.send_video(video_path=...).
        return await self._upload_and_send(chat_id, video_path, caption)

    async def _upload_and_send(
        self, chat_id: str, path: str, caption: Optional[str]
    ) -> SendResult:
        """XEP-0363 HTTP File Upload + body containing the GET URL.

        Receiving clients (Conversations, Dino, Gajim) render the URL inline
        as a media bubble. We also include a XEP-0066 OOB hint via metadata
        when available.
        """
        if self.client is None:
            return SendResult(success=False, error="xmpp not connected", retryable=True)
        if "xep_0363" not in self._registered_plugins:
            return SendResult(
                success=False,
                error="xmpp HTTP File Upload (XEP-0363) not available",
                retryable=False,
            )
        # Pass content-type so the receiving server's slot grant carries an
        # accurate Content-Type. Some clients render media inline based on it
        # rather than the URL extension.
        content_type, _ = mimetypes.guess_type(path)
        upload_kwargs: Dict[str, Any] = {
            "filename": Path(path).name,
            "input_file": path,
        }
        if content_type:
            upload_kwargs["content_type"] = content_type
        try:
            upload = self.client["xep_0363"].upload_file
            try:
                url = await upload(**upload_kwargs)
            except TypeError:
                # Older slixmpp signatures don't accept content_type kwarg.
                upload_kwargs.pop("content_type", None)
                url = await upload(**upload_kwargs)
        except Exception as exc:
            logger.exception("xmpp: HTTP upload (XEP-0363) failed")
            return SendResult(success=False, error=str(exc), retryable=True)

        body = url if not caption else f"{caption}\n{url}"
        mtype = "groupchat" if self._is_muc(chat_id) else "chat"
        try:
            stanza = self.client.send_message(mto=chat_id, mbody=body, mtype=mtype)
            msg_id = None
            try:
                msg_id = stanza["id"]
            except Exception:
                pass
            return SendResult(success=True, message_id=msg_id, raw_response=stanza)
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)

    # -----------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        chat_type = "group" if self._is_muc(chat_id) else "dm"
        return {"chat_id": chat_id, "type": chat_type, "name": chat_id}

    # Common MUC subdomain conventions across major XMPP servers. Operators
    # using a non-standard prefix should set XMPP_MUC_ROOMS, which takes
    # precedence over the heuristic.
    _MUC_DOMAIN_PREFIXES = ("conference.", "muc.", "rooms.", "chat.", "groups.")

    def _is_muc(self, chat_id: str) -> bool:
        # Explicit knowledge wins over any heuristic.
        if chat_id in self._known_mucs:
            return True
        if chat_id in self._known_dms:
            return False
        domain = chat_id.split("@", 1)[-1]
        return any(domain.startswith(p) for p in self._MUC_DOMAIN_PREFIXES)

    def _muc_room_nick(self, room_bare: str) -> str:
        """Return the nick we joined a given MUC room under.

        Rooms may be configured with a per-room nick (room/nick); fall back
        to the global muc_nick when no specific match is found.
        """
        for room in self.muc_rooms:
            if room.room == room_bare:
                return room.nick or self.muc_nick
        return self.muc_nick

    @staticmethod
    def _bare(jid: str) -> str:
        return jid.split("/", 1)[0] if "/" in jid else jid

    @staticmethod
    def _default_nick() -> str:
        return "hermes"


# ---------------------------------------------------------------------
# Standalone helper for cron / send_message_tool — sends a single message
# without spinning up the full adapter inside the gateway process.
# ---------------------------------------------------------------------

async def send_xmpp_message(
    pconfig: PlatformConfig, chat_id: str, message: str
) -> Dict[str, Any]:
    """One-shot send used by cron jobs and the send_message tool.

    Connects, sends, disconnects. Heavy for chat use, but the right shape
    for "send this single notification" callers that don't have access to
    a running gateway adapter instance.
    """
    adapter = XmppAdapter(pconfig)
    # Ephemeral resource on the account — must not consult or release the
    # gateway adapter's per-JID scoped lock (compare IRC's "-cron" nick).
    adapter._ephemeral_sender = True
    if not check_xmpp_requirements():
        return {"success": False, "error": "slixmpp not installed"}
    try:
        ok = await adapter.connect()
        if not ok:
            return {"success": False, "error": adapter.fatal_error_message or "connect failed"}
        # Wait for session_start (and any MUC joins) to complete, with a
        # bounded timeout so a slow/unresponsive server can't hang cron jobs.
        if adapter._session_ready is not None:
            try:
                await asyncio.wait_for(adapter._session_ready.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("xmpp: session_start did not fire within 10s; sending anyway")
        result = await adapter.send(chat_id=chat_id, content=message)
        return {
            "success": result.success,
            "message_id": result.message_id,
            "error": result.error,
        }
    finally:
        await adapter.disconnect()


# ---------------------------------------------------------------------
# Plugin registration hooks
# ---------------------------------------------------------------------

def _validate_config(cfg: Any) -> bool:
    """Minimally configured = JID + password present in extra."""
    extra = getattr(cfg, "extra", None) or {}
    return bool(extra.get("jid") and extra.get("password"))


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env vars during gateway config load.

    Called by the platform registry's env-enablement hook BEFORE adapter
    construction, so ``gateway status`` and ``get_connected_platforms()``
    reflect env-only configuration without instantiating the slixmpp client.
    Returns ``None`` when XMPP isn't minimally configured; the caller skips
    auto-enabling.

    The special ``home_channel`` key in the returned dict is handled by the
    core hook — it becomes a proper ``HomeChannel`` dataclass on the
    ``PlatformConfig`` rather than being merged into ``extra``.
    """
    jid = os.getenv("XMPP_JID", "").strip()
    password = os.getenv("XMPP_PASSWORD", "").strip()
    if not (jid and password):
        return None
    seed: dict = {"jid": jid, "password": password}
    host = os.getenv("XMPP_HOST", "").strip()
    if host:
        seed["host"] = host
    port = os.getenv("XMPP_PORT", "").strip()
    if port:
        try:
            seed["port"] = int(port)
        except ValueError:
            pass
    muc_rooms = os.getenv("XMPP_MUC_ROOMS", "").strip()
    if muc_rooms:
        seed["muc_rooms"] = muc_rooms
    muc_nick = os.getenv("XMPP_MUC_NICK", "").strip()
    if muc_nick:
        seed["muc_nick"] = muc_nick
    home = os.getenv("XMPP_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("XMPP_HOME_CHANNEL_NAME", "Home"),
        }
    return seed


async def _standalone_send(
    pconfig: Any,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> dict:
    """Out-of-process delivery for cron ``deliver=xmpp`` jobs.

    Opens a one-shot connection, sends, disconnects. XMPP has no threads;
    ``thread_id`` is accepted and ignored. Media attachments are not sent by
    the one-shot path (it exists for text notifications) — media-bearing
    sends go through the live gateway adapter.
    """
    try:
        result = await send_xmpp_message(pconfig, chat_id, message)
    except Exception as exc:
        return {"error": f"XMPP send failed: {exc}"}
    if result.get("success"):
        return {"success": True, "message_id": result.get("message_id")}
    return {"error": result.get("error") or "XMPP send failed"}


def interactive_setup() -> None:
    """Interactive `hermes gateway setup` flow for the XMPP platform.

    Lazy-imports ``hermes_cli.setup`` helpers so the plugin stays importable
    in non-CLI contexts (gateway runtime, tests).
    """
    from hermes_cli.setup import (
        prompt,
        prompt_yes_no,
        save_env_value,
        get_env_value,
        print_header,
        print_info,
        print_warning,
        print_success,
    )

    print_header("XMPP (Jabber)")
    existing_jid = get_env_value("XMPP_JID")
    if existing_jid:
        print_info(f"XMPP: already configured (JID: {existing_jid})")
        if not prompt_yes_no("Reconfigure XMPP?", False):
            return

    print_info("Connect Hermes to any XMPP server (Prosody, ejabberd, public")
    print_info("   servers like disroot.org, or your own). Create a dedicated")
    print_info("   bot account and note its JID and password. TLS is required.")
    print()

    jid = prompt("Bot JID (e.g. hermes@example.org)", default=existing_jid or "")
    if not jid or "@" not in jid:
        print_warning("A JID like user@server is required — skipping XMPP setup")
        return
    save_env_value("XMPP_JID", jid.strip())

    password = prompt("XMPP password", password=True)
    if not password:
        print_warning("Password is required — skipping XMPP setup")
        return
    save_env_value("XMPP_PASSWORD", password)

    host = prompt(
        "Server host (optional, leave blank for SRV/JID-domain lookup)",
        default=get_env_value("XMPP_HOST") or "",
    )
    if host:
        save_env_value("XMPP_HOST", host.strip())
    port = prompt(
        "Port (default 5222 STARTTLS; 5223 switches to direct TLS)",
        default=get_env_value("XMPP_PORT") or "",
    )
    if port:
        try:
            save_env_value("XMPP_PORT", str(int(port)))
        except ValueError:
            print_warning("Invalid port — using default 5222")

    muc_rooms = prompt(
        "MUC rooms to join (comma-separated, optional — e.g. team@conference.example.org/hermes)",
        default=get_env_value("XMPP_MUC_ROOMS") or "",
    )
    if muc_rooms:
        save_env_value("XMPP_MUC_ROOMS", muc_rooms.strip())

    allowed = prompt(
        "Allowed sender bare JIDs (comma-separated — empty denies all DMs)",
        default=get_env_value("XMPP_ALLOWED_USERS") or "",
    )
    if allowed:
        save_env_value("XMPP_ALLOWED_USERS", allowed.strip())
    else:
        print_warning(
            "No allowed users configured — the bot will reject all DMs until "
            "XMPP_ALLOWED_USERS is set (or XMPP_ALLOW_ALL_USERS=true)."
        )

    home = prompt(
        "Home channel JID for cron/notification delivery (optional)",
        default=get_env_value("XMPP_HOME_CHANNEL") or "",
    )
    if home:
        save_env_value("XMPP_HOME_CHANNEL", home.strip())

    print_success("XMPP configured")
    print_info("OMEMO end-to-end encryption for 1:1 chats is on by default and "
               "ships with the platform (auto-installs on first use). Set "
               "XMPP_OMEMO_ENABLED=false to run TLS-only, where messages are "
               "encrypted to the server but visible to the server operator.")


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="xmpp",
        label="XMPP (Jabber)",
        adapter_factory=lambda cfg: XmppAdapter(cfg),
        check_fn=check_xmpp_requirements,
        validate_config=_validate_config,
        is_connected=_validate_config,
        required_env=["XMPP_JID", "XMPP_PASSWORD"],
        install_hint="pip install 'hermes-agent[xmpp]' (slixmpp; lazy-installed on first use)",
        setup_fn=interactive_setup,
        # Env-driven auto-configuration: seeds PlatformConfig.extra with
        # jid/password/host/port/muc_rooms + home_channel so env-only setups
        # show up in gateway status without instantiating the adapter.
        env_enablement_fn=_env_enablement,
        # Cron home-channel delivery: deliver=xmpp routes to XMPP_HOME_CHANNEL.
        cron_deliver_env_var="XMPP_HOME_CHANNEL",
        # Out-of-process cron delivery. Without this hook, deliver=xmpp cron
        # jobs fail with "No live adapter" when cron runs separately from the
        # gateway.
        standalone_sender_fn=_standalone_send,
        # Auth env vars for _is_user_authorized() integration
        allowed_users_env="XMPP_ALLOWED_USERS",
        allow_all_env="XMPP_ALLOW_ALL_USERS",
        # Matches XmppAdapter.MAX_MESSAGE_LENGTH: XMPP has no protocol-level
        # body limit, but common server stanza-size policies make ~10k a safe
        # per-stanza cap; longer messages are split on code-fence/word
        # boundaries instead of being clipped at the gateway's 4096 default.
        max_message_length=10000,
        pii_safe=False,
        emoji="💬",
        allow_update_command=True,
        # LLM guidance
        platform_hint=(
            "You are on an XMPP (Jabber) chat. XMPP clients vary widely; assume "
            "plain text with light Markdown only. Avoid tables and rich formatting. "
            "You can send media files natively: include MEDIA:/absolute/path/to/file "
            "in your response — the adapter uploads via XEP-0363 HTTP File Upload "
            "and the URL renders inline as a media bubble in modern clients "
            "(Conversations, Dino, Gajim, Movim). Group chats are MUCs; addressing "
            "by mentioning the user's nick is conventional but not required."
        ),
    )
