# Copyright (c) 2026 NousResearch / Hermes Agent
#
# Matrix platform adapter for the Hermes gateway.
# Uses mautrix-python for full E2EE + cross-signing support.
#
# Dependencies (install via `hermes gateway setup matrix`, or manually):
#   pip install "mautrix[e2be]" asyncpg aiosqlite base58
#   System: libolm  (Arch: pacman -S libolm | Debian: apt install libolm-dev)
#
# Why asyncpg + aiosqlite?
#   PgCryptoStore (from mautrix) uses asyncpg's SQL dialect but the aiosqlite
#   backend lets it run on SQLite without a PostgreSQL server. This stores all
#   Olm sessions, inbound Megolm sessions, and device keys in a local file so
#   they survive gateway restarts — the same approach used by maubot and all
#   production mautrix bridges.
#
# Environment variables (configure with `hermes gateway setup matrix`):
#   MATRIX_HOMESERVER_URL   Required. e.g. https://matrix.example.org
#   MATRIX_ACCESS_TOKEN     Required. syt_... bot account access token
#   MATRIX_USER_ID          Required. @bot:example.org
#   MATRIX_DEVICE_ID        Recommended. Locks the bot to a specific device so
#                           E2EE sessions survive restarts without re-verification.
#   MATRIX_ALLOWED_USERS    Comma-separated user IDs that may message the bot.
#   MATRIX_HOME_CHANNEL     Optional room ID for cron job delivery.
#   MATRIX_VERIFY_SSL       true/false (default true; false for self-signed certs)
#   MATRIX_E2EE             true/false (default false; requires libolm + mautrix[e2be])
#   MATRIX_PASSWORD         Optional. Bot account password for cross-signing bootstrap.
#
# Setup flow:
#   1. hermes gateway setup matrix   — configure credentials and E2EE
#   2. hermes gateway run            — start the gateway (uploads device keys)
#   3. hermes gateway verify-matrix  — establish trust with allowed users (one-time)

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent as GatewayMessageEvent,
    SendResult,
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_bytes,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Python 3.14 compatibility shims for mautrix 0.21.0
# ---------------------------------------------------------------------------

def _patch_mautrix_py314() -> None:
    """Monkey-patch mautrix 0.21.0 for Python 3.14 compatibility.

    Python 3.14 tightened several asyncio and stdlib behaviors. This function
    applies targeted fixes at import time. Most patches are no-ops on Python < 3.14.

    Patches applied:
    1. MegolmEncryptionMachine._megolm_locks — replace defaultdict(asyncio.Lock)
       with a plain dict; create locks lazily inside async methods.
       (Python 3.14: Lock() in defaultdict lambda is created outside a running loop)
    2. MemoryCryptoStore.transaction() — replace async-def-returning-None with a
       proper @asynccontextmanager no-op.
       (Python 3.14: async with None raises TypeError; was silently tolerated before)
    3. MemoryStateStore.find_shared_rooms() — add missing method that
       DeviceListMachine calls unconditionally but only asyncpg store implements.
       (Runtime bug on all Python versions when using MemoryStateStore with device lists)
    4. MemoryCryptoStore.drop_signatures_by_key() — fix len(None) crash when
       dict.pop() returns None for a missing key.
       (mautrix bug present on all Python versions, not version-specific)
    5. MemoryCryptoStore.put_cross_signing_key() — handle read-only NamedTuple.
       (Python 3.14: NamedTuple fields became strictly immutable)
    """
    import sys
    import asyncio
    from contextlib import asynccontextmanager

    # Patch 4 fixes a real mautrix bug present on all Python versions.
    # Apply it unconditionally so it protects users on Python 3.12/3.13 too.
    try:
        from mautrix.crypto.store.memory import MemoryCryptoStore as _MCS_all

        async def _drop_sigs_all(self, signer):
            deleted = self._signatures.pop(signer, None)
            return len(deleted) if deleted is not None else 0

        _MCS_all.drop_signatures_by_key = _drop_sigs_all
    except Exception as e:
        logger.debug("mautrix patch 4 (drop_signatures_by_key) skipped: %s", e)

    # Patch 3 (find_shared_rooms) is also needed on all Python versions.
    try:
        from mautrix.client.state_store import MemoryStateStore as _MSS_all

        if not hasattr(_MSS_all, "find_shared_rooms"):
            async def _find_shared_rooms_all(self, user_id):
                return list(getattr(self, "_members", {}).keys())

            _MSS_all.find_shared_rooms = _find_shared_rooms_all
    except Exception as e:
        logger.debug("mautrix patch 3 (find_shared_rooms) skipped: %s", e)

    if sys.version_info < (3, 14):
        return

    # 1. encrypt_megolm: lazy lock creation
    try:
        from mautrix.crypto import encrypt_megolm as _emeg

        _orig_emeg_init = _emeg.MegolmEncryptionMachine.__init__

        def _emeg_init(self, *a, **kw):
            _orig_emeg_init(self, *a, **kw)
            self._megolm_locks = {}

        _emeg.MegolmEncryptionMachine.__init__ = _emeg_init

        async def _patched_encrypt_meg(self, room_id, event_type, content):
            if room_id not in self._megolm_locks:
                self._megolm_locks[room_id] = asyncio.Lock()
            async with self._megolm_locks[room_id]:
                return await self._encrypt_megolm_event(room_id, event_type, content)

        _emeg.MegolmEncryptionMachine.encrypt_megolm_event = _patched_encrypt_meg
    except Exception as e:
        logger.debug("mautrix py314 patch 1 (encrypt_megolm) skipped: %s", e)

    # 2. MemoryCryptoStore.transaction: proper async context manager
    try:
        from mautrix.crypto.store.memory import MemoryCryptoStore as _MCS

        @asynccontextmanager
        async def _transaction_noop(self):
            yield

        _MCS.transaction = _transaction_noop
    except Exception as e:
        logger.debug("mautrix py314 patch 2 (transaction) skipped: %s", e)

    # 5. MemoryCryptoStore.put_cross_signing_key: handle read-only NamedTuple
    try:
        from mautrix.crypto.store.memory import MemoryCryptoStore as _MCS3

        _orig_put_csk = _MCS3.put_cross_signing_key

        async def _put_csk(self, user_id, usage, key):
            try:
                await _orig_put_csk(self, user_id, usage, key)
            except AttributeError:
                from mautrix.crypto.store.memory import TOFUSigningKey
                try:
                    existing = self._cross_signing_keys[user_id][usage]
                    self._cross_signing_keys[user_id][usage] = TOFUSigningKey(
                        key=key, first=existing.first
                    )
                except (KeyError, TypeError):
                    self._cross_signing_keys.setdefault(user_id, {})[usage] = (
                        TOFUSigningKey(key=key, first=key)
                    )

        _MCS3.put_cross_signing_key = _put_csk
    except Exception as e:
        logger.debug("mautrix py314 patch 5 (put_cross_signing_key) skipped: %s", e)

    logger.debug("mautrix Python 3.14 compatibility patches applied")


# ---------------------------------------------------------------------------
# Availability check — import all mautrix symbols used by the adapter
# ---------------------------------------------------------------------------

try:
    import aiohttp
    from mautrix.client import Client
    from mautrix.client.state_store import MemoryStateStore
    from mautrix.crypto import OlmMachine
    from mautrix.crypto.store.asyncpg import PgCryptoStore
    from mautrix.errors import SessionNotFound
    from mautrix.types import (
        EventType,
        Format,
        MediaMessageEventContent,
        MemberStateEventContent,
        Membership,
        MessageEvent,
        MessageType,
        RoomID,
        StrippedStateEvent,
        TextMessageEventContent,
        TrustState,
        UserID,
    )
    from mautrix.util.async_db import Database
    _MAUTRIX_AVAILABLE = True
    _patch_mautrix_py314()
except ImportError:
    _MAUTRIX_AVAILABLE = False


def check_matrix_requirements() -> bool:
    """Return True if mautrix, asyncpg/aiosqlite, and aiohttp are importable."""
    return _MAUTRIX_AVAILABLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MATRIX_ID_RE = re.compile(r"@([^:]{2})[^:]*(:.*)")


def _redact_matrix_id(user_id: str) -> str:
    """Partially redact a Matrix user ID for safe log output.

    e.g. ``@alice:example.org`` → ``@al**:example.org``
    """
    return _MATRIX_ID_RE.sub(r"@\1**\2", user_id)


def _bool_env(value: Any, default: bool = False) -> bool:
    """Parse a boolean from an env-var string or Python bool."""
    if isinstance(value, bool):
        return value
    return str(value).lower() not in ("false", "0", "no", "")


# ---------------------------------------------------------------------------
# MatrixAdapter
# ---------------------------------------------------------------------------

class MatrixAdapter(BasePlatformAdapter):
    """Matrix platform adapter backed by mautrix-python.

    Supports both plain-text and fully end-to-end encrypted rooms.

    E2EE design (when MATRIX_E2EE=true):
    - Uses PgCryptoStore backed by SQLite so all Olm/Megolm sessions survive
      gateway restarts. This is the same approach used by maubot and all
      production mautrix bridges.
    - Cross-signing bootstrap uploads master/self-signing/user-signing keys
      so Element can display the bot as a verified device.
    - ``hermes gateway verify-matrix`` signs the bot's master key with each
      allowed user's identity key, completing the trust chain.

    Non-E2EE design:
    - Uses the standard mautrix Client without OlmMachine.
    - Identical message handling, invite acceptance, and media support.
    """

    PLATFORM = Platform.MATRIX

    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config, Platform.MATRIX)

        extra = config.extra or {}

        self.homeserver_url: str = (
            os.getenv("MATRIX_HOMESERVER_URL") or extra.get("homeserver_url", "")
        ).rstrip("/")

        self.access_token: str = (
            os.getenv("MATRIX_ACCESS_TOKEN") or config.token or ""
        )

        self.user_id: str = (
            os.getenv("MATRIX_USER_ID") or extra.get("user_id", "")
        )

        # Device ID — pin to a fixed device so E2EE sessions survive restarts.
        self.device_id: str = (
            os.getenv("MATRIX_DEVICE_ID") or extra.get("device_id", "")
        )

        self.allowed_users: set[str] = set(
            u.strip()
            for u in (
                os.getenv("MATRIX_ALLOWED_USERS") or extra.get("allowed_users", "")
            ).split(",")
            if u.strip()
        )

        self.verify_ssl: bool = _bool_env(
            os.getenv("MATRIX_VERIFY_SSL") or extra.get("verify_ssl", "true"),
            default=True,
        )

        self.e2ee: bool = _bool_env(
            os.getenv("MATRIX_E2EE") or extra.get("e2ee", "false"),
            default=False,
        )

        self.password: str = os.getenv("MATRIX_PASSWORD") or ""

        # Recovery key for loading cross-signing private keys from SSSS on startup.
        # This allows the bot to self-sign its device on every restart without
        # re-running generate_recovery_key. Set from the key printed on first bootstrap.
        self.recovery_key: str = os.getenv("MATRIX_RECOVERY_KEY") or ""

        self.home_channel: str = (
            os.getenv("MATRIX_HOME_CHANNEL") or extra.get("home_channel", "")
        )

        # Allow all users (gateway-level override for internal testing)
        self._allow_all: bool = (
            _bool_env(os.getenv("MATRIX_ALLOW_ALL_USERS", "false"))
            or _bool_env(os.getenv("GATEWAY_ALLOW_ALL_USERS", "false"))
        )

        # Resolved at connect time
        self._client: Optional[Client] = None
        self._crypto: Optional[OlmMachine] = None
        self._crypto_db: Optional[Database] = None

        # Message deduplication (event_id → timestamp)
        self._seen_event_ids: dict[str, float] = {}
        self._seen_cleanup_interval: float = 60.0
        self._last_seen_cleanup: float = 0.0

        # Active SAS verification sessions keyed by transaction_id
        self._sas_sessions: dict[str, Any] = {}

        # Data directory
        self._data_dir = Path(os.path.expanduser("~/.hermes/matrix"))

        logger.info(
            "Matrix adapter initialized: homeserver=%s user=%s e2ee=%s ssl=%s",
            self.homeserver_url,
            _redact_matrix_id(self.user_id),
            self.e2ee,
            self.verify_ssl,
        )

    # ------------------------------------------------------------------
    # BasePlatformAdapter interface
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        if not self.homeserver_url or not self.access_token or not self.user_id:
            logger.error(
                "Matrix: missing required config "
                "(MATRIX_HOMESERVER_URL, MATRIX_ACCESS_TOKEN, MATRIX_USER_ID)"
            )
            return False
        if not _MAUTRIX_AVAILABLE:
            logger.error(
                "Matrix: mautrix not installed. "
                "Run: pip install 'mautrix[e2be]' asyncpg aiosqlite base58"
            )
            return False
        try:
            return await self._do_connect()
        except Exception:
            logger.exception("Matrix: connect() failed")
            return False

    async def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.stop()
                if self._client.syncing_task is not None:
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(self._client.syncing_task),
                            timeout=10.0,
                        )
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
            except Exception as exc:
                logger.debug("Matrix: stop error: %s", exc)
            try:
                await self._client.api.session.close()
            except Exception:
                pass
            self._client = None
            self._crypto = None
        if self._crypto_db is not None:
            try:
                await self._crypto_db.stop()
            except Exception:
                pass
            self._crypto_db = None
        logger.info("Matrix: disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> SendResult:
        return await self.send_message(chat_id, content, reply_to_message_id=reply_to)

    async def get_chat_info(self, chat_id: str) -> dict:
        if self._client is None:
            return {"id": chat_id, "name": chat_id, "platform": "matrix"}
        try:
            state = await self._client.get_state(RoomID(chat_id))
            name = chat_id
            for evt in (state or []):
                if hasattr(evt, "type") and str(evt.type) == "m.room.name":
                    name = getattr(evt.content, "name", chat_id) or chat_id
                    break
            return {"id": chat_id, "name": name, "platform": "matrix"}
        except Exception:
            return {"id": chat_id, "name": chat_id, "platform": "matrix"}

    async def send_message(
        self,
        channel_id: str,
        text: str,
        reply_to_message_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        if self._client is None:
            return SendResult(success=False, error="Not connected")
        try:
            # Render Markdown → HTML for rich display in Element.
            # The plain-text body is kept as fallback for clients that
            # don't render formatted_body.
            formatted_body: Optional[str] = None
            try:
                import markdown as _md
                formatted_body = _md.markdown(
                    text, extensions=["fenced_code", "tables", "nl2br"]
                )
            except ImportError:
                if "<" in text:
                    formatted_body = text

            content = TextMessageEventContent(
                msgtype=MessageType.TEXT,
                body=text,
                format=Format.HTML if formatted_body else None,
                formatted_body=formatted_body,
            )
            event_id = await self._client.send_message(RoomID(channel_id), content)
            logger.info("Matrix: sent message (%d chars) to %s", len(text), channel_id)
            return SendResult(success=True, message_id=str(event_id))
        except Exception as exc:
            logger.error("Matrix: send_message failed: %s", exc, exc_info=True)
            return SendResult(success=False, error=str(exc))

    async def send_image(
        self, channel_id: str, image_path: str,
        caption: Optional[str] = None, **kwargs: Any,
    ) -> SendResult:
        return await self._send_media(channel_id, image_path, caption, MessageType.IMAGE)

    async def send_audio(
        self, channel_id: str, audio_path: str,
        caption: Optional[str] = None, **kwargs: Any,
    ) -> SendResult:
        return await self._send_media(channel_id, audio_path, caption, MessageType.AUDIO)

    async def send_document(
        self, channel_id: str, document_path: str,
        caption: Optional[str] = None, **kwargs: Any,
    ) -> SendResult:
        return await self._send_media(channel_id, document_path, caption, MessageType.FILE)

    async def _send_media(
        self,
        channel_id: str,
        file_path: str,
        caption: Optional[str],
        msgtype: MessageType,
    ) -> SendResult:
        if self._client is None:
            return SendResult(success=False, error="Not connected")
        try:
            import mimetypes
            path = Path(file_path)
            data = path.read_bytes()
            mime, _ = mimetypes.guess_type(str(path))
            mime = mime or "application/octet-stream"
            mxc_url = await self._client.upload_media(data, mime_type=mime, filename=path.name)
            content = MediaMessageEventContent(
                msgtype=msgtype, body=caption or path.name, url=mxc_url
            )
            event_id = await self._client.send_message(RoomID(channel_id), content)
            return SendResult(success=True, message_id=str(event_id))
        except Exception as exc:
            logger.error("Matrix: send_media failed: %s", exc)
            return SendResult(success=False, error=str(exc))

    def get_home_channel(self) -> Optional[str]:
        return self.home_channel or None

    # ------------------------------------------------------------------
    # Internal: connection setup
    # ------------------------------------------------------------------

    async def _do_connect(self) -> bool:
        ssl_ctx: Any = False if not self.verify_ssl else None
        if not self.verify_ssl:
            logger.warning(
                "Matrix: SSL verification disabled — only suitable for private homeservers"
            )

        # Suppress mautrix internal debug messages that leak as WARNING
        # (e.g. "Didn't find cross-signing key master of ...") — these are
        # expected during bootstrap and are not actionable by the user.
        import logging as _logging
        for _noisy_logger in ("mautrix.crypto", "mau.crypto", "mau.client.crypto"):
            _logging.getLogger(_noisy_logger).setLevel(_logging.ERROR)

        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        session = aiohttp.ClientSession(connector=connector)
        state_store = MemoryStateStore()

        # E2EE: open SQLite crypto store (PgCryptoStore on SQLite backend).
        # This is identical to the approach used by maubot and all production
        # mautrix bridges — proper session persistence across restarts.
        if self.e2ee:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            db_path = self._data_dir / "crypto.db"
            self._crypto_db = Database.create(
                f"sqlite:{db_path}",
                upgrade_table=PgCryptoStore.upgrade_table,
            )
            await self._crypto_db.start()
            crypto_store = PgCryptoStore(
                account_id=self.user_id,
                pickle_key="hermes-matrix-e2ee",
                db=self._crypto_db,
            )
            await crypto_store.open()
        else:
            crypto_store = None

        # Build mautrix Client
        self._client = Client(
            mxid=UserID(self.user_id),
            device_id=self.device_id or "",
            base_url=self.homeserver_url,
            token=self.access_token,
            client_session=session,
            state_store=state_store,
            sync_store=crypto_store,
        )

        # Verify connectivity and pin device_id
        try:
            whoami = await self._client.whoami()
            if whoami.device_id:
                self._client.device_id = whoami.device_id
            logger.info(
                "Matrix: authenticated as %s device=%s",
                _redact_matrix_id(whoami.user_id),
                whoami.device_id or "(none)",
            )
        except Exception as exc:
            logger.error("Matrix: whoami failed: %s", exc)
            await session.close()
            return False

        # E2EE: load OlmMachine, upload keys if needed
        if self.e2ee and crypto_store is not None:
            self._crypto = OlmMachine(
                client=self._client,
                crypto_store=crypto_store,
                state_store=state_store,
            )
            self._client.crypto = self._crypto
            await self._crypto.load()
            logger.info("Matrix: E2EE crypto loaded (SQLite store)")

            # Upload device keys and one-time keys if the account is fresh
            # or if keys were wiped from the server. Mirrors maubot's approach:
            # always check and upload if account.shared is False, otherwise
            # verify the server still has our keys.
            stored_device_id = await crypto_store.get_device_id()
            if not stored_device_id:
                await crypto_store.put_device_id(self._client.device_id)
            if not self._crypto.account.shared:
                await self._crypto.share_keys()
                logger.info("Matrix: E2EE device keys uploaded")
            else:
                # Verify our keys are still on the server (guards against DB wipes)
                await self._verify_keys_on_server()

            # Remove ghost devices created by previous cross-signing key uploads.
            # Synapse stores cross-signing public keys as device rows; Element then
            # tries to encrypt Olm messages to them, causing decryption failures.
            await self._purge_ghost_devices()

        # Register event handlers
        self._client.add_event_handler(EventType.ROOM_MESSAGE, self._on_message)
        self._client.add_event_handler(EventType.ROOM_MEMBER, self._on_member)

        # Register key verification handlers (E2EE only)
        if self.e2ee:
            self._register_verification_handlers()

        # Bootstrap cross-signing on first successful sync
        if self.e2ee and self._crypto is not None:
            from mautrix.client.syncer import InternalEventType

            async def _on_first_sync(data: Any) -> None:
                self._client.remove_event_handler(
                    InternalEventType.SYNC_SUCCESSFUL, _on_first_sync
                )
                await self._bootstrap_cross_signing()

            self._client.add_event_handler(
                InternalEventType.SYNC_SUCCESSFUL, _on_first_sync
            )

        # Start sync loop
        self._client.start(None)
        logger.info("Matrix: connected to %s", self.homeserver_url)
        return True

    async def _verify_keys_on_server(self) -> None:
        """Verify our device keys are still present on the homeserver.

        If they've been wiped (e.g. after a Synapse DB reset), re-upload.
        Mirrors maubot's ``_verify_keys_are_on_server`` pattern.
        """
        if self._crypto is None or self._client is None:
            return
        try:
            resp = await self._client.query_keys([self.user_id])
            device_keys = (
                resp.device_keys.get(self.user_id, {})
                .get(self._client.device_id, {})
            )
            # device_keys may be a mautrix DeviceKeys object (has .keys attribute)
            # or a plain dict. getattr(dict, "keys", {}) would return the dict's
            # .keys() method (truthy but not a key map), so check explicitly.
            key_map = device_keys.keys if hasattr(device_keys, "keys") and not callable(device_keys.keys) else {}
            if device_keys and len(key_map) > 0:
                return  # keys are on server
            logger.warning(
                "Matrix: device keys missing from server — re-uploading"
            )
            await self._crypto.share_keys()
        except Exception as exc:
            logger.warning("Matrix: key verification check failed: %s", exc)

    async def _purge_ghost_devices(self) -> None:
        """Remove non-real devices that Synapse creates from cross-signing key uploads.

        When `keys/device_signing/upload` is called, some Synapse versions store the
        cross-signing public keys as rows in the `devices` table alongside real device
        sessions. These "ghost devices" have base64-encoded IDs (44 chars) and no
        associated OTKs or key JSON. Element encrypts Olm messages to *all* known
        devices — including ghosts — which causes decryption failures because no Olm
        account exists for them.

        This method calls the CS API to delete any devices that:
        1. Are not the bot's current device_id
        2. Were never used (no last_seen_ts or last_seen_ip from a real login)

        This is the correct fix: don't create ghosts in the first place, and clean up
        any that exist on startup.
        """
        if self._client is None:
            return
        try:
            # Fetch all devices for this account
            devices_resp = await self._client.api.request(
                "GET", f"/client/v3/devices"
            )
            all_devices = devices_resp.get("devices", [])
            current_device = self._client.device_id or ""

            # Identify ghost devices: base64-looking IDs (44 chars) that aren't ours
            ghosts = [
                d["device_id"]
                for d in all_devices
                if d["device_id"] != current_device
                and len(d["device_id"]) > 20  # real device IDs are ~8-10 uppercase chars
                and not d.get("last_seen_ts")  # never had a real login
            ]

            if not ghosts:
                return

            logger.info(
                "Matrix: removing %d ghost device(s) created by cross-signing uploads",
                len(ghosts),
            )
            for device_id in ghosts:
                try:
                    # Try without auth first — works on homeservers that don't
                    # require UIA for device deletion (e.g. self-hosted Synapse
                    # with relaxed settings). If it fails with 401, retry with
                    # password UIA using the bot's stored password.
                    try:
                        await self._client.delete_device(device_id, auth=None)
                    except Exception as first_exc:
                        if "401" in str(first_exc) and self.password:
                            import json as _j
                            _uia_data = _j.loads(str(first_exc).split("401: ", 1)[-1])
                            _auth = {
                                "type": "m.login.password",
                                "identifier": {"type": "m.id.user", "user": self.user_id},
                                "password": self.password,
                                "session": _uia_data.get("session", ""),
                            }
                            await self._client.delete_device(device_id, auth=_auth)
                        else:
                            raise
                    logger.debug("Matrix: deleted ghost device %s", device_id[:16])
                except Exception as exc:
                    logger.debug(
                        "Matrix: could not delete ghost device %s: %s",
                        device_id[:16], exc,
                    )
        except Exception as exc:
            logger.debug("Matrix: ghost device purge failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal: event handlers
    # ------------------------------------------------------------------

    async def _on_message(self, event: Any) -> None:
        """Handle an incoming room message event."""
        if self._client is None:
            return

        sender = str(event.sender)
        room_id = str(event.room_id)
        event_id = str(event.event_id)

        # Deduplicate — Matrix can re-deliver events on reconnect
        now = time.time()
        if event_id in self._seen_event_ids:
            return
        self._seen_event_ids[event_id] = now

        # Periodic cleanup of stale seen IDs and expired SAS sessions.
        if now - self._last_seen_cleanup > self._seen_cleanup_interval:
            cutoff = now - self._seen_cleanup_interval
            self._seen_event_ids = {
                k: v for k, v in self._seen_event_ids.items() if v > cutoff
            }
            # Purge SAS verification sessions that were started but never completed.
            # Without this, abandoned verification attempts accumulate indefinitely.
            self._sas_sessions = {
                k: v for k, v in self._sas_sessions.items()
                if now - v.get("_started_at", now) < 300  # 5 min max per SAS flow
            }
            self._last_seen_cleanup = now

        # Ignore our own messages
        if sender == self.user_id:
            return

        # Authorization
        if not self._allow_all and self.allowed_users and sender not in self.allowed_users:
            logger.debug(
                "Matrix: ignoring message from unauthorized sender %s",
                _redact_matrix_id(sender),
            )
            return

        content = event.content

        # Drop key verification messages — handled by dedicated handlers
        msgtype_str = str(getattr(content, "msgtype", ""))
        if "verification" in msgtype_str:
            return

        # Extract text and media URL
        text: Optional[str] = getattr(content, "body", None) or ""
        media_url: Optional[str] = None
        msg_type = MessageType.TEXT

        if hasattr(content, "msgtype"):
            mt = content.msgtype
            if mt == MessageType.IMAGE:
                msg_type = MessageType.IMAGE
                media_url = str(content.url) if getattr(content, "url", None) else None
            elif mt == MessageType.AUDIO:
                msg_type = MessageType.AUDIO
                media_url = str(content.url) if getattr(content, "url", None) else None
            elif mt in (MessageType.VIDEO, MessageType.FILE):
                msg_type = MessageType.FILE
                media_url = str(content.url) if getattr(content, "url", None) else None

        if not text and not media_url:
            return

        logger.debug(
            "Matrix: message from %s in %s: %s",
            _redact_matrix_id(sender), room_id, text[:60],
        )

        # Build gateway event
        from gateway.session import SessionSource
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id=room_id,
            chat_name=room_id,
            chat_type="dm",
            user_id=sender,
            user_name=sender.split(":")[0].lstrip("@"),
        )
        gateway_event = GatewayMessageEvent(
            text=text or "",
            source=source,
            raw_message=event,
            message_id=event_id,
        )

        # Download media
        if media_url and msg_type == MessageType.IMAGE:
            try:
                data = await self._client.download_media(media_url)
                gateway_event.image_path = cache_image_from_bytes(data)
                gateway_event.message_type = "image"
            except Exception as exc:
                logger.warning("Matrix: image download failed: %s", exc)
        elif media_url and msg_type == MessageType.AUDIO:
            try:
                data = await self._client.download_media(media_url)
                gateway_event.audio_path = cache_audio_from_bytes(data)
                gateway_event.message_type = "audio"
            except Exception as exc:
                logger.warning("Matrix: audio download failed: %s", exc)
        elif media_url:
            try:
                data = await self._client.download_media(media_url)
                gateway_event.document_path = cache_document_from_bytes(data)
                gateway_event.message_type = "document"
            except Exception as exc:
                logger.warning("Matrix: media download failed: %s", exc)

        await self.handle_message(gateway_event)

    async def _on_member(self, event: Any) -> None:
        """Auto-join rooms on invite from an authorized user."""
        if self._client is None:
            return
        content = event.content
        if not isinstance(content, MemberStateEventContent):
            return
        if content.membership != Membership.INVITE:
            return
        if str(event.state_key) != self.user_id:
            return

        inviter = str(event.sender)
        room_id = str(event.room_id)

        if not self._allow_all and self.allowed_users and inviter not in self.allowed_users:
            logger.info(
                "Matrix: rejecting invite to %s from unauthorized inviter %s",
                room_id, _redact_matrix_id(inviter),
            )
            try:
                await self._client.leave_room(RoomID(room_id))
            except Exception:
                pass
            return

        logger.info(
            "Matrix: accepting invite to %s from %s",
            room_id, _redact_matrix_id(inviter),
        )
        try:
            await self._client.join_room(RoomID(room_id))
        except Exception as exc:
            logger.error("Matrix: failed to join room %s: %s", room_id, exc)

    # ------------------------------------------------------------------
    # Internal: E2EE cross-signing bootstrap
    # ------------------------------------------------------------------

    async def _bootstrap_cross_signing(self) -> None:
        """Upload cross-signing keys for the bot account (one-time setup).

        Publishes master, self-signing, and user-signing keys so that
        Element can verify the bot's identity. Mirrors the approach in
        mautrix/bridge/e2ee.py: check the current trust level first,
        and only bootstrap if needed.

        After this runs, users should run ``hermes gateway verify-matrix``
        to sign the bot's master key with their own user-signing key,
        completing the trust chain.
        """
        if self._crypto is None:
            return

        # Canonical mautrix bridge pattern (mautrix/bridge/e2ee.py lines ~268-275):
        # Check local trust state first. The SQLite crypto store persists the
        # cross-signing signatures from the initial bootstrap, so resolve_trust
        # returns VERIFIED on all subsequent restarts without any server queries.
        # Only run generate_recovery_key when the device is genuinely not yet signed.
        try:
            trust_state = await asyncio.wait_for(
                self._crypto.resolve_trust(self._crypto.own_identity),
                timeout=15.0,
            )
            if trust_state >= TrustState.CROSS_SIGNED_UNTRUSTED:
                logger.debug(
                    "Matrix: cross-signing already established (trust=%s) — skipping bootstrap",
                    trust_state,
                )
                return
        except Exception as exc:
            logger.debug("Matrix: trust check failed (%s) — proceeding with bootstrap", exc)

        try:
            logger.info("Matrix: bootstrapping cross-signing keys...")
            recovery_key = await asyncio.wait_for(
                self._crypto.generate_recovery_key(passphrase=self.password or None),
                timeout=30.0,
            )
            # Auto-save the recovery key to .env so it's available if the crypto DB
            # is ever lost (matching your intuition — we know the password at setup
            # and we already print the key, so we should persist it automatically).
            try:
                import re as _re
                env_path = os.path.expanduser("~/.hermes/.env")
                env = open(env_path).read()
                if "MATRIX_RECOVERY_KEY=" in env:
                    env = _re.sub(
                        r"^MATRIX_RECOVERY_KEY=.*$",
                        f"MATRIX_RECOVERY_KEY={recovery_key}",
                        env, flags=_re.MULTILINE,
                    )
                else:
                    env += f"\nMATRIX_RECOVERY_KEY={recovery_key}\n"
                open(env_path, "w").write(env)
                self.recovery_key = recovery_key
                logger.info(
                    "Matrix: cross-signing bootstrap complete — "
                    "recovery key saved to .env: %s",
                    recovery_key,
                )
            except Exception as save_exc:
                # If we can't save to .env, at least log it prominently
                logger.info(
                    "Matrix: cross-signing bootstrap complete. "
                    "SAVE THIS RECOVERY KEY: %s",
                    recovery_key,
                )
                logger.debug("Matrix: could not auto-save recovery key: %s", save_exc)

        except asyncio.TimeoutError:
            logger.warning(
                "Matrix: cross-signing bootstrap timed out — "
                "run 'hermes gateway verify-matrix' to complete setup"
            )
        except Exception as exc:
            # generate_recovery_key fails with 401 when the homeserver requires
            # password UIA to upload cross-signing keys (e.g. Synapse when keys
            # already exist). Retry by calling _publish_cross_signing_keys directly
            # with the UIA auth dict, then run the rest of the bootstrap manually.
            exc_str = str(exc)
            if "401" in exc_str and self.password:
                import json as _json
                try:
                    # Extract UIA session from the error message
                    uia_data = _json.loads(exc_str.split("401: ", 1)[-1])
                    uia_session = uia_data.get("session", "")
                    auth = {
                        "type": "m.login.password",
                        "identifier": {"type": "m.id.user", "user": self.user_id},
                        "password": self.password,
                        "session": uia_session,
                    }
                    from mautrix.crypto.cross_signing_key import CrossSigningSeeds
                    seeds = CrossSigningSeeds.generate()
                    ssss_key = await self._crypto.ssss.generate_and_upload_key(
                        self.password or None
                    )
                    await self._crypto._upload_cross_signing_keys_to_ssss(ssss_key, seeds)
                    await self._crypto._publish_cross_signing_keys(seeds.to_keys(), auth=auth)
                    await self._crypto.ssss.set_default_key_id(ssss_key.id)
                    await self._crypto.sign_own_device(self._crypto.own_identity)
                    recovery_key = ssss_key.recovery_key
                    # Auto-save recovery key
                    try:
                        import re as _re
                        env_path = os.path.expanduser("~/.hermes/.env")
                        env = open(env_path).read()
                        if "MATRIX_RECOVERY_KEY=" in env:
                            env = _re.sub(
                                r"^MATRIX_RECOVERY_KEY=.*$",
                                f"MATRIX_RECOVERY_KEY={recovery_key}",
                                env, flags=_re.MULTILINE,
                            )
                        else:
                            env += f"\nMATRIX_RECOVERY_KEY={recovery_key}\n"
                        open(env_path, "w").write(env)
                        self.recovery_key = recovery_key
                    except Exception:
                        pass
                    logger.info(
                        "Matrix: cross-signing bootstrap complete (UIA). "
                        "Recovery key saved to .env: %s",
                        recovery_key,
                    )
                except Exception as uia_exc:
                    logger.warning(
                        "Matrix: cross-signing bootstrap failed even with UIA: %s — "
                        "run 'hermes gateway verify-matrix' to set up manually",
                        uia_exc,
                    )
            else:
                logger.warning(
                    "Matrix: cross-signing bootstrap failed: %s — "
                    "set MATRIX_PASSWORD and run 'hermes gateway verify-matrix'",
                    exc,
                )

    # ------------------------------------------------------------------
    # Internal: SAS key verification (auto-accept)
    # ------------------------------------------------------------------

    def _register_verification_handlers(self) -> None:
        """Register handlers for both to-device and in-room verification events.

        Modern Element (post-2022) sends verification events as room messages
        in DM rooms. Older clients and some flows use to-device messages.
        We register handlers for both paths.
        """
        if self._client is None:
            return

        # to-device path
        for event_type_str, handler in [
            ("m.key.verification.request", self._on_verification_request),
            ("m.key.verification.start", self._on_verification_start),
            ("m.key.verification.key", self._on_verification_key),
            ("m.key.verification.mac", self._on_verification_mac),
            ("m.key.verification.cancel", self._on_verification_cancel),
        ]:
            et = EventType.find(event_type_str, t_class=EventType.Class.TO_DEVICE)
            self._client.add_event_handler(et, handler)

        # in-room path (room message events)
        for event_type_str in [
            "m.key.verification.request",
            "m.key.verification.start",
            "m.key.verification.key",
            "m.key.verification.mac",
            "m.key.verification.cancel",
            "m.key.verification.done",
        ]:
            et = EventType.find(event_type_str, t_class=EventType.Class.MESSAGE)
            self._client.add_event_handler(et, self._on_room_verification_event)

    def _extract_verif_content(self, event: Any) -> dict:
        """Extract verification event content as a plain dict."""
        content = getattr(event, "content", {})
        if isinstance(content, dict):
            return content
        # mautrix event content object — access raw JSON
        raw = getattr(content, "_json", None)
        if isinstance(raw, dict):
            return raw
        # fallback: walk known fields
        return {
            k: getattr(content, k, None)
            for k in ("transaction_id", "from_device", "methods", "method",
                       "key", "keys", "mac", "reason", "m.relates_to")
            if getattr(content, k, None) is not None
        }

    async def _on_room_verification_event(self, event: Any) -> None:
        """Route in-room verification events to the to-device handlers."""
        try:
            event_type = str(getattr(event, "type", ""))
            raw = self._extract_verif_content(event)

            # Room-based verification uses m.relates_to.event_id as txid
            if "transaction_id" not in raw:
                relates = raw.get("m.relates_to") or {}
                if isinstance(relates, dict):
                    raw = dict(raw)
                    raw["transaction_id"] = relates.get("event_id") or str(
                        getattr(event, "event_id", "")
                    )

            # Synthesize a simple event-like object the handlers can consume
            class _Evt:
                sender = str(event.sender)
                content = raw

            fake = _Evt()

            if "verification.request" in event_type:
                await self._on_verification_request(fake)
            elif "verification.start" in event_type:
                await self._on_verification_start(fake)
            elif "verification.key" in event_type:
                await self._on_verification_key(fake)
            elif "verification.mac" in event_type:
                await self._on_verification_mac(fake)
            elif "verification.cancel" in event_type:
                await self._on_verification_cancel(fake)
        except Exception as exc:
            logger.debug("Matrix: room verification routing error: %s", exc)

    async def _on_verification_request(self, event: Any) -> None:
        """Respond to m.key.verification.request with m.key.verification.ready."""
        if self._client is None:
            return
        try:
            sender = str(event.sender)
            raw = self._extract_verif_content(event)
            txid = raw.get("transaction_id", "")
            from_device = raw.get("from_device", "")
            methods = raw.get("methods", [])

            if not txid or "m.sas.v1" not in methods:
                return
            if not self._allow_all and self.allowed_users and sender not in self.allowed_users:
                return

            logger.info(
                "Matrix verification: request from %s txid=%s",
                _redact_matrix_id(sender), txid,
            )
            await self._client.send_to_one_device(
                EventType.find("m.key.verification.ready", t_class=EventType.Class.TO_DEVICE),
                UserID(sender), from_device,
                {
                    "transaction_id": txid,
                    "from_device": self._client.device_id or "",
                    "methods": ["m.sas.v1"],
                },
            )
        except Exception as exc:
            logger.warning("Matrix verification: request handler error: %s", exc)

    async def _on_verification_start(self, event: Any) -> None:
        """Accept SAS start — create olm.Sas, send our public key."""
        if self._client is None:
            return
        try:
            import olm as _olm
            sender = str(event.sender)
            raw = self._extract_verif_content(event)
            txid = raw.get("transaction_id", "")
            if raw.get("method") != "m.sas.v1" or not txid:
                return

            sas = _olm.Sas()
            self._sas_sessions[txid] = {
                "sas": sas,
                "sender": sender,
                "from_device": raw.get("from_device", ""),
                "_started_at": time.time(),  # for periodic expiry cleanup
            }
            await self._client.send_to_one_device(
                EventType.find("m.key.verification.key", t_class=EventType.Class.TO_DEVICE),
                UserID(sender), raw.get("from_device", ""),
                {"transaction_id": txid, "key": sas.pubkey},
            )
            logger.info(
                "Matrix verification: sent SAS key for txid=%s sender=%s",
                txid, _redact_matrix_id(sender),
            )
        except Exception as exc:
            logger.warning("Matrix verification: start handler error: %s", exc)

    async def _on_verification_key(self, event: Any) -> None:
        """Receive counterpart's key; compute and send MAC (auto-accept)."""
        if self._client is None:
            return
        try:
            raw = self._extract_verif_content(event)
            txid = raw.get("transaction_id", "")
            their_key = raw.get("key", "")
            session = self._sas_sessions.get(txid)
            if not session or not their_key:
                return

            sas = session["sas"]
            sas.set_their_pubkey(their_key)

            sender = session["sender"]
            sender_device = session["from_device"]
            our_device = self._client.device_id or ""

            # Matrix SAS MAC info string (MSC1758 spec)
            mac_base = (
                f"MATRIX_KEY_VERIFICATION_MAC"
                f"{sender}{sender_device}"
                f"{self.user_id}{our_device}{txid}"
            )

            our_ed25519 = ""
            if self._crypto and self._crypto.account:
                our_ed25519 = self._crypto.account.identity_keys.get("ed25519", "")

            key_id = f"ed25519:{our_device}"
            key_mac = sas.calculate_mac_fixed_base64(our_ed25519, f"{mac_base}{key_id}")
            keys_mac = sas.calculate_mac_fixed_base64(key_id, f"{mac_base}KEY_IDS")

            await self._client.send_to_one_device(
                EventType.find("m.key.verification.mac", t_class=EventType.Class.TO_DEVICE),
                UserID(sender), sender_device,
                {"transaction_id": txid, "keys": keys_mac, "mac": {key_id: key_mac}},
            )
            logger.info("Matrix verification: sent MAC for txid=%s (auto-accepted)", txid)
        except Exception as exc:
            logger.warning("Matrix verification: key handler error: %s", exc)

    async def _on_verification_mac(self, event: Any) -> None:
        """Counterpart confirmed — send done, clean up session."""
        if self._client is None:
            return
        try:
            raw = self._extract_verif_content(event)
            txid = raw.get("transaction_id", "")
            session = self._sas_sessions.pop(txid, None)
            if not session:
                return
            sender = session["sender"]
            sender_device = session["from_device"]
            await self._client.send_to_one_device(
                EventType.find("m.key.verification.done", t_class=EventType.Class.TO_DEVICE),
                UserID(sender), sender_device,
                {"transaction_id": txid},
            )
            logger.info(
                "Matrix verification: complete txid=%s sender=%s",
                txid, _redact_matrix_id(sender),
            )
        except Exception as exc:
            logger.warning("Matrix verification: mac handler error: %s", exc)

    async def _on_verification_cancel(self, event: Any) -> None:
        """Clean up canceled verification session."""
        try:
            raw = self._extract_verif_content(event)
            txid = raw.get("transaction_id", "")
            self._sas_sessions.pop(txid, None)
            logger.info(
                "Matrix verification: canceled txid=%s reason=%s",
                txid, raw.get("reason", ""),
            )
        except Exception as exc:
            logger.debug("Matrix verification: cancel handler error: %s", exc)
