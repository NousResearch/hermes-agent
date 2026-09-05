"""Matrix lifecycle methods; runtime dependencies remain on the adapter facade."""

from __future__ import annotations

from typing import Any, Dict, Optional


class MatrixLifecycleMixin:
    @staticmethod
    def _extract_server_ed25519(device_keys_obj: Any) -> Optional[str]:
        from . import adapter as _adapter

        for kid, kval in (getattr(device_keys_obj, "keys", {}) or {}).items():
            if str(kid).startswith("ed25519:"):
                return str(kval)
        return None

    @staticmethod
    async def _query_own_device_keys(client: Any):
        """query_keys for our own device; the DeviceKeys entry or None."""
        from . import adapter as _adapter

        resp = await client.query_keys({client.mxid: [client.device_id]})
        our_user_devices = (getattr(resp, "device_keys", {}) or {}).get(str(client.mxid)) or {}
        return our_user_devices.get(str(client.device_id))

    async def _reverify_keys_after_upload(self, client: Any, local_ed25519: str) -> bool:
        """Re-query the server after share_keys() and verify our ed25519 key matches."""
        from . import adapter as _adapter

        if not client.device_id or self._device_id_unverified:
            _adapter.logger.warning("Matrix: skipping post-upload key verification — device_id not yet established")
            return True
        try:
            dev = await self._query_own_device_keys(client)
            if dev and self._extract_server_ed25519(dev) != local_ed25519:
                _adapter.logger.error(
                    "Matrix: device %s has immutable identity keys that don't match this "
                    "installation. Generate a new access token with a fresh device.", client.device_id)
                return False
        except Exception as exc:
            _adapter.logger.error("Matrix: post-upload key verification failed: %s", exc, exc_info=True)
            return False
        return True

    async def _reset_crypto_store_if_device_changed(self, crypto_store: Any, device_id: str) -> bool:
        """Reset the Olm account when the token's device changed; True if reset. The store is keyed
        by user ID, so a new device would inherit the old Olm account whose identity keys can never
        be published under the new device ID."""
        from . import adapter as _adapter

        if not device_id:
            return False
        try:
            stored_device_id = await crypto_store.get_device_id()
        except Exception as exc:
            _adapter.logger.warning("Matrix: could not read stored device ID: %s", exc)
            return False
        if not stored_device_id or stored_device_id == device_id:
            return False
        _adapter.logger.warning(
            "Matrix: access token belongs to a new device (%s -> %s) — resetting local Olm account "
            "so fresh identity keys are generated for this device", stored_device_id, device_id)
        await crypto_store.delete()
        return True

    async def _migrate_legacy_crypto_pickle(
            self, crypto_store: Any, crypto_db: Any, acct_id: str, pickle_key: str) -> bool:
        """Re-pickle the Olm account under the current pickle key when it changed. The key embeds the
        device ID; an account created before MATRIX_DEVICE_ID was set lives under ``<acct>:default``
        and later fails with BAD_ACCOUNT_KEY (silently disabling optional E2EE). False only when an
        account exists but no key opens it."""
        from . import adapter as _adapter

        with _adapter.suppress(Exception):
            await crypto_store.get_account()
            return True
        from mautrix.crypto.store.asyncpg import PgCryptoStore
        for legacy_key in (f"{acct_id}:default", acct_id):
            if legacy_key == pickle_key:
                continue
            try:
                account = await PgCryptoStore(account_id=acct_id, pickle_key=legacy_key, db=crypto_db).get_account()
            except Exception:
                account = None
            if account is None:
                continue
            # Sessions first, account last: the account is the commit marker (the fast path
            # above short-circuits once it reads), so an interrupted sweep is retried.
            try:
                await self._repickle_crypto_sessions(crypto_db, acct_id, legacy_key, pickle_key)
            except Exception as exc:
                _adapter.logger.error(
                    "Matrix: pickle key migration failed while re-pickling sessions (%s) — leaving "
                    "the account under the legacy key so the migration is retried on the next start.", exc)
                return False
            await crypto_store.put_account(account)
            _adapter.logger.info(
                "Matrix: re-pickled crypto store account and sessions under the current pickle key "
                "(device ID was configured after the account was created)")
            return True
        _adapter.logger.error(
            "Matrix: crypto store account exists but cannot be unpickled with the current or any "
            "legacy pickle key. If MATRIX_DEVICE_ID was changed manually, restore its previous value.")
        return False

    async def _repickle_crypto_sessions(self, crypto_db: Any, acct_id: str, legacy_key: str, pickle_key: str) -> None:
        """Re-pickle olm/megolm sessions too — they share the key; account-only breaks key sharing."""
        from . import adapter as _adapter

        import olm as olm_lib
        tables = {
            "crypto_olm_session": olm_lib.Session, "crypto_megolm_inbound_session": olm_lib.InboundGroupSession,
            "crypto_megolm_outbound_session": olm_lib.OutboundGroupSession}
        for table, session_cls in tables.items():
            rows = await crypto_db.fetch(f"SELECT session_id, session FROM {table} WHERE account_id=$1", acct_id)
            for row in rows:
                if row["session"] is None:
                    continue
                pickled = bytes(row["session"])
                with _adapter.suppress(Exception):
                    session_cls.from_pickle(pickled, pickle_key)
                    continue  # already readable with the current key
                try:
                    session = session_cls.from_pickle(pickled, legacy_key)
                except Exception as exc:
                    # Readable under neither key: leave it inert rather than delete crypto material.
                    _adapter.logger.warning(
                        "Matrix: %s row %s cannot be unpickled with the current or legacy key; leaving "
                        "it in place, its sessions are unrecoverable: %s", table, row["session_id"], exc)
                    continue
                await crypto_db.execute(
                    f"UPDATE {table} SET session=$1 WHERE account_id=$2 AND session_id=$3",
                    session.pickle(pickle_key), acct_id, row["session_id"])

    async def _verify_device_keys_on_server(self, client: Any, olm: Any) -> bool:
        """True if our device keys are on the server (or were re-uploaded); False ⇒ refuse E2EE."""
        from . import adapter as _adapter

        if not client.device_id or self._device_id_unverified:
            _adapter.logger.warning("Matrix: skipping device key verification — device_id not yet established")
            return True
        try:
            our_keys = await self._query_own_device_keys(client)
        except Exception as exc:
            _adapter.logger.error("Matrix: cannot verify device keys on server: %s — refusing E2EE", exc, exc_info=True)
            return False
        local_ed25519 = olm.account.identity_keys.get("ed25519")

        async def _reupload(error_fmt: str, *error_args) -> bool:
            try:
                await olm.share_keys()
            except Exception as exc:
                _adapter.logger.error(error_fmt, *error_args, exc, exc_info=True)
                return False
            return await self._reverify_keys_after_upload(client, local_ed25519)
        if not our_keys:
            _adapter.logger.warning("Matrix: device keys missing from server — re-uploading")
            olm.account.shared = False
            return await _reupload("Matrix: failed to re-upload device keys: %s")
        if self._extract_server_ed25519(our_keys) == local_ed25519:
            return True
        if olm.account.shared:
            _adapter.logger.error(
                "Matrix: server has different identity keys for device %s — local crypto state is "
                "stale. Delete %s and restart.", client.device_id, str(self._crypto_db_path))
            return False
        _adapter.logger.warning("Matrix: server has stale keys for device %s — attempting re-upload", client.device_id)
        with _adapter.suppress(Exception):
            await client.api.request(
                client.api.Method.DELETE if hasattr(client.api, "Method") else "DELETE",
                f"/_matrix/client/v3/devices/{client.device_id}")
            _adapter.logger.info("Matrix: deleted stale device %s from server", client.device_id)
        return await _reupload(
            "Matrix: cannot upload device keys for %s: %s. Try generating a new access token to get a fresh device.",
            client.device_id)

    @staticmethod
    async def _abort_connect(api: Any, crypto_db: Any = None) -> bool:
        """Close what connect() opened so far; always False so callers can ``return await``."""
        if crypto_db is not None:
            await crypto_db.stop()
        await api.session.close()
        return False

    async def _connect_authenticate(self, client: Any, api: Any) -> bool:
        """Authenticate via access token (whoami) or password login; resolve user/device IDs."""
        from . import adapter as _adapter

        if self._access_token:
            api.token = self._access_token
            try:
                resp = await client.whoami()
                resolved_user_id = getattr(resp, "user_id", "") or self._user_id
                resolved_device_id = str(getattr(resp, "device_id", "") or "")
                if resolved_user_id:
                    self._user_id = str(resolved_user_id)
                    client.mxid = _adapter.UserID(self._user_id)
                # The configured device_id wins when whoami() reports none, but a token can
                # only upload keys for its own device — on conflict whoami() wins, loudly.
                if resolved_device_id and self._device_id and resolved_device_id != self._device_id:
                    _adapter.logger.error(
                        "Matrix: MATRIX_DEVICE_ID=%s does not match the device this access token "
                        "belongs to (%s). A token can only upload keys for its own device, so the "
                        "configured value is being ignored. Unset MATRIX_DEVICE_ID, or use a token "
                        "issued for %s.", self._device_id, resolved_device_id, self._device_id)
                    effective_device_id = resolved_device_id
                else:
                    effective_device_id = self._device_id or resolved_device_id
                if effective_device_id:
                    client.device_id = effective_device_id
                if not client.device_id:
                    try:
                        dev_resp = await client.query_keys({client.mxid: []})
                        all_devices = (getattr(dev_resp, "device_keys", {}) or {}).get(str(client.mxid)) or {}
                        if len(all_devices) == 1:
                            client.device_id = next(iter(all_devices))
                        elif not all_devices:
                            _adapter.logger.warning(
                                "Matrix: no devices found for %s — key verification will be skipped", client.mxid)
                    except Exception as exc:
                        _adapter.logger.warning("Matrix: device list query failed: %s", exc)
                if not client.device_id:
                    _adapter.logger.warning(
                        "Matrix: device_id could not be resolved for %s. Set MATRIX_DEVICE_ID for full "
                        "key verification. E2EE will proceed without server-side device key confirmation.",
                        client.mxid)
                    self._device_id_unverified = True
                _adapter.logger.info(
                    "Matrix: using access token for %s%s", self._user_id or "(unknown user)",
                    f" (device {effective_device_id})" if effective_device_id else "")
            except Exception as exc:
                _adapter.logger.error(
                    "Matrix: whoami failed — check MATRIX_ACCESS_TOKEN and MATRIX_HOMESERVER: %s", exc, exc_info=True)
                return await self._abort_connect(api)
        elif self._password and self._user_id:
            try:
                resp = await client.login(
                    identifier=self._user_id, password=self._password, device_name="Hermes Agent",
                    device_id=self._device_id or None)
                if resp and hasattr(resp, "device_id"):
                    client.device_id = resp.device_id
                _adapter.logger.info("Matrix: logged in as %s", self._user_id)
            except Exception as exc:
                _adapter.logger.error("Matrix: login failed — %s", exc)
                return await self._abort_connect(api)
        else:
            _adapter.logger.error("Matrix: need MATRIX_ACCESS_TOKEN or MATRIX_USER_ID + MATRIX_PASSWORD")
            return await self._abort_connect(api)
        return True

    async def _connect_setup_e2ee(self, client: Any, api: Any, state_store: Any) -> bool:
        """Set up the Olm machine + crypto store. Returns False when connect must abort."""
        from . import adapter as _adapter

        if not _adapter._check_e2ee_deps():
            if self._e2ee_mode == "optional":
                _adapter.logger.warning(
                    "Matrix: E2EE optional but dependencies are missing. Continuing without "
                    "encrypted-room support. %s", _adapter._E2EE_INSTALL_HINT)
                self._encryption = False
            else:
                _adapter.logger.error(
                    "Matrix: E2EE is required but dependencies are missing. %s. Refusing to connect — "
                    "encrypted rooms would silently fail.", _adapter._E2EE_INSTALL_HINT)
                return await self._abort_connect(api)
        if not self._encryption:
            return True
        phase = "import"
        try:
            from mautrix.crypto import OlmMachine
            from mautrix.crypto.store.asyncpg import PgCryptoStore
            from mautrix.util.async_db import Database
            self._store_dir.mkdir(parents=True, exist_ok=True)
            phase = "create"
            if (self._store_dir / "crypto_store.pickle").exists():  # pre-SQLite era
                _adapter.logger.info("Matrix: removing legacy crypto_store.pickle (migrated to SQLite)")
                (self._store_dir / "crypto_store.pickle").unlink()
            crypto_db = Database.create(
                f"sqlite:///{self._crypto_db_path}", upgrade_table=PgCryptoStore.upgrade_table)
            await crypto_db.start()
            self._crypto_db = crypto_db
            _acct_id = self._user_id or "hermes"
            # Key on the RESOLVED client.device_id (token's real device), not the configured
            # one, or the Olm account is stored under a key that can never be looked up.
            _pickle_key = f"{_acct_id}:{client.device_id or self._device_id or 'default'}"
            crypto_store = PgCryptoStore(account_id=_acct_id, pickle_key=_pickle_key, db=crypto_db)
            await crypto_store.open()
            _store_was_reset = False
            if client.device_id:
                _store_was_reset = await self._reset_crypto_store_if_device_changed(crypto_store, client.device_id)
                await crypto_store.put_device_id(client.device_id)
            # A just-deleted store has no account to migrate.
            if not _store_was_reset and not await self._migrate_legacy_crypto_pickle(
                    crypto_store, crypto_db, _acct_id, _pickle_key):
                _adapter.logger.warning("Matrix: crypto pickle migration failed — E2EE may not work correctly")
            crypto_state = _adapter._CryptoStateStore(state_store, self._joined_rooms, client)
            olm = OlmMachine(client, crypto_store, crypto_state)
            olm.share_keys_min_trust = _adapter.TrustState.UNVERIFIED
            olm.send_keys_min_trust = _adapter.TrustState.UNVERIFIED
            await olm.load()
            if not await self._verify_device_keys_on_server(client, olm):
                return await self._abort_connect(api, crypto_db)
            try:
                await olm.share_keys()
            except Exception as exc:
                if "already exists" in str(exc):
                    _adapter.logger.error(
                        "Matrix: device %s has stale one-time keys on the server signed with a "
                        "previous identity key. Delete the device from the homeserver and restart, "
                        "or generate a new access token to get a fresh device ID.", client.device_id)
                    return await self._abort_connect(api, crypto_db)
                _adapter.logger.warning("Matrix: share_keys() warning during startup: %s", exc)
            await self._verify_or_bootstrap_cross_signing(olm, client)
            client.crypto = olm
            _adapter.logger.info(
                "Matrix: E2EE enabled (store: %s%s)", str(self._crypto_db_path),
                f", device_id={client.device_id}" if client.device_id else "")
        except Exception as exc:
            return await self._e2ee_setup_failed(phase, exc, api)
        return True

    async def _e2ee_setup_failed(self, what: str, exc: Exception, api: Any) -> bool:
        """Optional mode: log + disable E2EE and return True; required mode: close + return False."""
        from . import adapter as _adapter

        if self._e2ee_mode == "optional":
            _adapter.logger.warning(
                "Matrix: failed to %s optional E2EE client; continuing without encrypted-room "
                "support: %s. %s", what, exc, _adapter._E2EE_INSTALL_HINT)
            self._encryption = False
            return True
        _adapter.logger.error("Matrix: failed to %s E2EE client: %s. %s", what, exc, _adapter._E2EE_INSTALL_HINT)
        return await self._abort_connect(api)

    async def _verify_or_bootstrap_cross_signing(self, olm: Any, client: Any) -> None:
        """Verify cross-signing via MATRIX_RECOVERY_KEY, or bootstrap a new key (non-fatal)."""
        # Honor the active profile's secret scope so a secondary profile under gateway.multiplex_profiles
        # resolves its own recovery key instead of the default profile's (which fails E2EE verification with
        # "Key MAC does not match", #69090).
        from . import adapter as _adapter

        recovery_key = _adapter._scoped_recovery_key()
        if recovery_key:
            try:
                await olm.verify_with_recovery_key(recovery_key)
                _adapter.logger.info("Matrix: cross-signing verified via recovery key")
            except Exception as exc:
                _adapter.logger.warning("Matrix: recovery key verification failed: %s", exc)
        else:
            try:
                own_xsign = await olm.get_own_cross_signing_public_keys()
            except Exception as exc:
                own_xsign = None
                _adapter.logger.warning("Matrix: cross-signing key lookup failed: %s", exc)
            if own_xsign is None:
                _, output_error = _adapter._get_matrix_recovery_key_output_target()
                if output_error:
                    reason = {
                        "not_configured": "is not configured. Configure MATRIX_RECOVERY_KEY from your Matrix client "
                                          "or set MATRIX_RECOVERY_KEY_OUTPUT_FILE to write a new recovery key once "
                                          "with mode 0600.",
                        "exists": "already exists and will not be overwritten.",
                    }.get(output_error, "is not usable: %s")
                    _adapter.logger.warning(
                        "Matrix: cross-signing keys are missing, but automatic bootstrap is skipped because "
                        "MATRIX_RECOVERY_KEY_OUTPUT_FILE " + reason,
                        *([output_error] if output_error not in ("not_configured", "exists") else []))
                else:
                    try:
                        new_recovery_key = await olm.generate_recovery_key()
                        _adapter._handle_generated_matrix_recovery_key(str(client.mxid), new_recovery_key)
                    except Exception as exc:
                        _adapter.logger.warning(
                            "Matrix: cross-signing bootstrap failed (non-fatal — Element will show "
                            "'not verified by its owner'): %s", exc)

    async def _connect_initial_sync(self, client: Any) -> None:
        """Full initial sync: seed joined rooms, DM cache, and dispatch queued to-device events."""
        from . import adapter as _adapter

        try:
            sync_data = await client.sync(timeout=10000, full_state=True)
            if isinstance(sync_data, dict):
                self._joined_rooms.clear()
                await self._absorb_sync(client, sync_data, initial=True)
            else:
                _adapter.logger.warning("Matrix: initial sync returned unexpected type %s", type(sync_data).__name__)
        except Exception as exc:
            _adapter.logger.warning("Matrix: initial sync error: %s", exc)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        from . import adapter as _adapter

        self._device_id_unverified = False
        if self._client is not None:
            try:
                await self.disconnect()
            except Exception as exc:
                _adapter.logger.warning("Matrix: error disconnecting before reconnect: %s", exc)
        from mautrix.api import HTTPAPI
        from mautrix.client import Client
        from mautrix.client.state_store import MemoryStateStore, MemorySyncStore
        if not self._homeserver:
            _adapter.logger.error("Matrix: homeserver URL not configured")
            return False
        # Resolved here, inside the profile scope, so multiplexed profiles never share it.
        self._resolve_store_dir().mkdir(parents=True, exist_ok=True)
        client_session = _adapter._create_matrix_session(self._proxy_url)
        api = HTTPAPI(base_url=self._homeserver, token=self._access_token or "", client_session=client_session)
        state_store = MemoryStateStore()
        sync_store = MemorySyncStore()
        client = Client(
            mxid=_adapter.UserID(self._user_id) if self._user_id else _adapter.UserID(""), device_id=self._device_id or None,
            api=api, state_store=state_store, sync_store=sync_store)
        self._client = client
        if not await self._connect_authenticate(client, api):
            return False
        if self._encryption and not await self._connect_setup_e2ee(client, api, state_store):
            return False
        from mautrix.client import InternalEventType as IntEvt
        from mautrix.client.dispatcher import MembershipEventDispatcher
        client.add_dispatcher(MembershipEventDispatcher)  # without this INVITE never fires
        client.add_event_handler(_adapter.EventType.ROOM_MESSAGE, self._on_room_message, wait_sync=True)
        client.add_event_handler(_adapter.EventType.REACTION, self._on_reaction, wait_sync=True)
        client.add_event_handler(IntEvt.INVITE, self._on_invite, wait_sync=True)
        self._startup_ts = _adapter.time.time()
        self._reset_clock_skew_detector()  # a reconnect after an NTP fix starts clean
        self._closing = False
        await self._connect_initial_sync(client)
        if self._encryption and getattr(client, "crypto", None):
            try:
                await client.crypto.share_keys()
            except Exception as exc:
                _adapter.logger.warning("Matrix: initial key share failed: %s", exc)
        self._sync_task = _adapter.asyncio.create_task(self._sync_loop())
        self._mark_connected()
        self._wire_plugin_handlers(self._client)  # plugin-registered native handlers
        return True

    async def disconnect(self) -> None:
        from . import adapter as _adapter
        from .choice_picker import cancel_choice_pages

        cancel_choice_pages(self)
        self._closing = True
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except (_adapter.asyncio.CancelledError, Exception):
                pass
        for tasks in (self._invite_join_tasks.values(), self._reaction_redaction_tasks):
            pending = list(tasks)
            for task in pending:
                if not task.done():
                    task.cancel()
            if pending:
                await _adapter.asyncio.gather(*pending, return_exceptions=True)
        self._invite_join_tasks.clear()
        self._reaction_redaction_tasks.clear()
        if getattr(self, "_crypto_db", None):
            try:
                await self._crypto_db.stop()
            except Exception as exc:
                _adapter.logger.debug("Matrix: could not close crypto DB on disconnect: %s", exc)
        if self._client:
            with _adapter.suppress(Exception):
                await self._client.api.session.close()
            self._client = None
        _adapter.logger.info("Matrix: disconnected")

    async def _sync_loop(self) -> None:
        from . import adapter as _adapter

        client = self._client
        next_batch = await client.sync_store.get_next_batch()  # resume from the initial sync
        while not self._closing:
            try:
                # 45s outer cap guards TCP-level hangs the 30s long-poll timeout can't catch.
                sync_data = await _adapter.asyncio.wait_for(client.sync(since=next_batch, timeout=30000), timeout=45.0)
                # Auth failures (M_UNKNOWN_TOKEN) arrive as SyncError objects, not exceptions.
                _sync_msg = getattr(sync_data, "message", None)
                if isinstance(_sync_msg, str) and "unknown_token" in _sync_msg.lower():
                    _adapter.logger.error("Matrix: permanent auth error from sync: %s — stopping", _sync_msg)
                    return
                if isinstance(sync_data, dict):
                    next_batch = await self._absorb_sync(client, sync_data) or next_batch
                    await _adapter.asyncio.sleep(0)  # let fresh invite joins start before the next sync
            except _adapter.asyncio.CancelledError:
                return
            except Exception as exc:
                if self._closing:
                    return
                if any(k in str(exc).lower() for k in ("401", "403", "unauthorized", "forbidden")):
                    _adapter.logger.error("Matrix: permanent auth error: %s — stopping sync", exc)
                    return
                _adapter.logger.warning("Matrix: sync error: %s — retrying in 5s", exc)
                await _adapter.asyncio.sleep(5)

    async def _absorb_sync(self, client: Any, sync_data: Dict[str, Any], *, initial: bool = False) -> Optional[str]:
        """Apply one sync response: joined rooms, next_batch, event dispatch, pending invites. Returns next_batch.
        The initial (full-state) sync also seeds the DM cache and dispatches so the OlmMachine sees
        to-device key shares queued while offline."""
        from . import adapter as _adapter

        self._last_sync_ts = _adapter.time.time()
        rooms_join = sync_data.get("rooms", {}).get("join", {})
        if rooms_join or initial:
            self._joined_rooms.update(rooms_join.keys())
            self._invalidate_room_identities()
        nb = sync_data.get("next_batch")  # incremental syncs resume from here
        if nb:
            await client.sync_store.put_next_batch(nb)
        if initial:
            _adapter.logger.info("Matrix: initial sync complete, joined %d rooms", len(self._joined_rooms))
            await self._refresh_dm_cache()
        try:
            await self._dispatch_sync(sync_data)
        except Exception as exc:
            _adapter.logger.warning("Matrix: %s: %s", "initial sync event dispatch error" if initial else "sync event dispatch error", exc)
        self._schedule_pending_invite_joins(sync_data)
        return nb

    async def _dispatch_sync(self, sync_data: Dict[str, Any]) -> None:
        """Dispatch a sync response through the mautrix event machinery."""
        from . import adapter as _adapter

        client = self._client
        if not client or not hasattr(client, "handle_sync"):
            return
        tasks = client.handle_sync(sync_data)
        if _adapter.inspect.isawaitable(tasks):
            tasks = await tasks
        if tasks:
            # return_exceptions=True: one failing handler must not drop its SIBLING events.
            results = await _adapter.asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    _adapter.logger.warning("Matrix: event handler failed during sync dispatch: %s", result)
