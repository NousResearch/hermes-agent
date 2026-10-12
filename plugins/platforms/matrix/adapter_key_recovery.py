"""Matrix room-key recovery and durable Olm one-time-key uploads."""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class MatrixKeyRecoveryMixin:
    def _acquire_matrix_store_lock(self) -> bool:
        from gateway.status import _try_acquire_file_lock

        handle = (self._store_dir / "adapter.lock").open("a+")
        if not _try_acquire_file_lock(handle):
            handle.close()
            self._set_fatal_error("matrix_store_busy", "Matrix crypto store is already in use", retryable=True)
            return False
        if not self._acquire_platform_lock(
                "matrix-crypto-store", str(self._crypto_db_path.resolve()), "Matrix crypto store"):
            handle.close()
            return False
        self._matrix_store_lock_handle = handle
        return True

    def _release_matrix_store_lock(self) -> None:
        handle = getattr(self, "_matrix_store_lock_handle", None)
        if handle is not None:
            from gateway.status import _release_file_lock
            self._release_platform_lock()
            _release_file_lock(handle)
            handle.close()
            self._matrix_store_lock_handle = None

    async def _abort_connect(self, api: Any, crypto_db: Any = None) -> bool:
        """Close what connect() opened so far; always False so callers can ``return await``."""
        try:
            if crypto_db is not None:
                await crypto_db.stop()
        finally:
            try:
                await api.session.close()
            finally:
                self._release_matrix_store_lock()
        return False


    async def _allow_matrix_key_share(self, device: Any, request: Any) -> bool:
        """Authorize recovery of a missed Megolm room key.

        Mautrix only accepts room-key requests from another device of the
        same Matrix account by default. Permit recovery for explicitly
        allowed users, but only while the bot and requester are both joined
        to the requested room and that room passes the configured policy.
        """
        from plugins.platforms.matrix.adapter import RoomID

        client = self._client
        olm = getattr(client, "crypto", None) if client else None
        if olm is None:
            return False

        requester = str(getattr(device, "user_id", ""))
        own_user_id = str(getattr(client, "mxid", "") or self._user_id)
        if requester == own_user_id:
            return await olm.default_allow_key_share(device, request)

        # Cross-user recovery is deliberately narrower than normal message
        # authorization: open/allow-all mode must never grant room history.
        if not requester or requester not in self._allowed_user_ids:
            return False
        if bool(getattr(device, "deleted", False)):
            return False

        room_id = str(getattr(request, "room_id", ""))
        if not room_id or room_id not in self._joined_rooms:
            return False
        if not await self._is_allowed_matrix_room_event(room_id):
            return False

        try:
            joined_members = await client.get_joined_members(RoomID(room_id))
        except Exception as exc:
            logger.warning(
                "Matrix: refusing room-key request from %s for %s: "
                "could not verify current membership: %s",
                requester,
                room_id,
                exc,
                exc_info=True,
            )
            return False
        if not {requester, own_user_id}.issubset({str(user_id) for user_id in joined_members}):
            return False

        try:
            trust = await olm.resolve_trust(device)
        except Exception as exc:
            logger.warning(
                "Matrix: refusing room-key request from %s device %s: "
                "could not resolve trust: %s",
                requester,
                getattr(device, "device_id", "?"),
                exc,
                exc_info=True,
            )
            return False
        if trust < olm.share_keys_min_trust:
            return False

        logger.info(
            "Matrix: accepting recovery key request from allowed member %s "
            "device %s for room %s",
            requester,
            getattr(device, "device_id", "?"),
            room_id,
        )
        return True


    async def _share_matrix_keys_crash_safely(
        self, olm: Any, current_otk_count: int | None
    ) -> None:
        """Upload Olm one-time keys without losing their private half."""
        if current_otk_count is None or (
            olm._last_key_share + 60 > time.monotonic()
            and current_otk_count < 10
        ):
            olm.log.debug("Checking OTK count on server")
            counts = await olm.client.upload_keys()
            current_otk_count = next(
                (
                    int(count)
                    for algorithm, count in counts.items()
                    if str(algorithm) == "signed_curve25519"
                ),
                0,
            )

        current_otk_count = max(0, int(current_otk_count))
        max_otk_count = int(olm.account.max_one_time_keys)
        if current_otk_count > max_otk_count:
            logger.error(
                "Matrix: homeserver reports %d OTKs for this device, above "
                "the local Olm maximum of %d; stale server keys must be reset",
                current_otk_count,
                max_otk_count,
            )

        device_keys = (
            olm.account.get_device_keys(olm.client.mxid, olm.client.device_id)
            if not olm.account.shared
            else None
        )
        unpublished_count = sum(
            len(keys) for keys in olm.account.one_time_keys.values()
        )
        generation_count = current_otk_count
        if unpublished_count:
            # These keys may be from a committed upload whose response was
            # lost. Re-upload them verbatim instead of generating a new batch.
            generation_count = max(generation_count, max_otk_count // 2)
        one_time_keys = olm.account.get_one_time_keys(
            olm.client.mxid, olm.client.device_id, generation_count
        )
        if not device_keys and not one_time_keys:
            olm.log.warning(
                "No one-time keys nor device keys got when trying to share keys"
            )
            return

        if device_keys:
            olm.log.debug("Going to upload initial account keys")
        olm.log.debug("Uploading %d one-time keys", len(one_time_keys))

        if one_time_keys:
            # Once Synapse can see a public OTK, its private counterpart is
            # already durable locally. This is the crash-safety boundary.
            await olm.crypto_store.put_account(olm.account)

        response = await olm.client.upload_keys(
            one_time_keys=one_time_keys,
            device_keys=device_keys,
        )
        olm.account.shared = True
        olm.account.mark_keys_as_published()
        olm._last_key_share = time.monotonic()
        await olm.crypto_store.put_account(olm.account)
        olm.log.debug("Shared keys and saved account, new keys: %s", response)


    async def _e2ee_setup_failed(self, what: str, exc: Exception, api: Any) -> bool:
        """Optional mode: log + disable E2EE and return True; required mode: close + return False."""
        from plugins.platforms.matrix.adapter import _E2EE_INSTALL_HINT

        crypto_db = getattr(self, "_crypto_db", None)
        if crypto_db is not None:
            await crypto_db.stop()
            self._crypto_db = None
        self._release_matrix_store_lock()
        if self._e2ee_mode == "optional":
            logger.warning(
                "Matrix: failed to %s optional E2EE client; continuing without encrypted-room "
                "support: %s. %s", what, exc, _E2EE_INSTALL_HINT)
            self._encryption = False
            return True
        logger.error("Matrix: failed to %s E2EE client: %s. %s", what, exc, _E2EE_INSTALL_HINT)
        return await self._abort_connect(api)
