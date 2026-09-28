"""Durable Matrix sync positions and completion of required event dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mautrix.crypto.store.asyncpg import PgCryptoStore
    from mautrix.types import ToDeviceEvent

logger = logging.getLogger(__name__)

MAX_PENDING_INTAKES = 4096


class DurableSyncStore:
    def __init__(
        self, directory: Path, homeserver: str, user_id: str, device_id: str, token: str
    ):
        identity = json.dumps(
            [homeserver, user_id, device_id, token if not device_id else ""],
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()
        self.path = directory / f"sync-{digest}.json"
        self._next_batch: str | None = None
        self._accepted_events: set[str] = set()
        self._reserved_events: set[str] = set()
        self._write_lock = asyncio.Lock()

    async def load(self) -> None:
        self._next_batch, self._accepted_events = await asyncio.to_thread(self._read)

    def _read(self) -> tuple[str | None, set[str]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return None, set()
        except (ValueError, UnicodeError) as exc:
            logger.warning("Matrix: invalid sync cursor file %s: %s", self.path, exc)
            return None, set()
        token = data.get("next_batch") if isinstance(data, dict) else None
        accepted = data.get("accepted_events", []) if isinstance(data, dict) else []
        if (
            not isinstance(accepted, list)
            or len(accepted) > MAX_PENDING_INTAKES
            or any(
                not isinstance(event_id, str)
                or not event_id
                or len(event_id.encode("utf-8")) > 255
                for event_id in accepted
            )
        ):
            raise OSError("invalid pending Matrix intake checkpoint")
        return token if isinstance(token, str) and token else None, set(accepted)

    async def get_next_batch(self) -> str | None:
        return self._next_batch

    async def put_next_batch(self, next_batch: str) -> None:
        await self._commit(next_batch=next_batch)

    def reserve_intake(self, event_id: str) -> bool:
        if event_id in self._accepted_events:
            return False
        if len(event_id.encode("utf-8")) > 255:
            raise ValueError("Matrix event ID exceeds the transport checkpoint limit")
        if (
            len(self._accepted_events | self._reserved_events | {event_id})
            > MAX_PENDING_INTAKES
        ):
            raise RuntimeError(
                "Matrix pending intake checkpoint is full; batch remains unacknowledged"
            )
        self._reserved_events.add(event_id)
        return True

    def release_intake(self, event_id: str) -> None:
        self._reserved_events.discard(event_id)

    async def accept_intake(self, event_id: str) -> None:
        if event_id in self._accepted_events:
            return
        await self._commit(event_id=event_id)

    async def _commit(
        self, *, next_batch: str | None = None, event_id: str | None = None
    ) -> None:
        async def write() -> None:
            async with self._write_lock:
                accepted = self._accepted_events | {event_id} if event_id else set()
                cursor = next_batch if next_batch is not None else self._next_batch
                await asyncio.to_thread(self._write, cursor, accepted)
                self._next_batch, self._accepted_events = cursor, accepted

        task = asyncio.create_task(write())
        cancelled = False
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                # Parent dispatch and disconnect can both cancel an admitted handler.
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    def _write(self, next_batch: str | None, accepted: set[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                payload: dict[str, Any] = {"next_batch": next_batch}
                if accepted:
                    payload["accepted_events"] = sorted(accepted)
                json.dump(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
            if os.name == "posix":
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def is_invalid_sync_cursor(exc: Exception) -> bool:
    errcode = getattr(exc, "errcode", None)
    if errcode == "M_UNKNOWN_POS":
        return True
    status = getattr(exc, "http_status", None)
    message = str(exc).lower()
    return (
        status == 400
        and errcode in {"M_INVALID_PARAM", "M_UNKNOWN"}
        and (
            "invalid stream token" in message
            or ("since" in message and "token" in message)
        )
    )


class SyncDispatch:
    def __init__(self, client: Any):
        self.client = client
        self._sync_dispatch_tasks: list[asyncio.Task] = []
        self._owned_tasks: set[asyncio.Task] = set()
        self.failed_sync_handlers: set[tuple[Any, str]] = set()
        self._completed_key_handlers: set[tuple[Any, bytes]] = set()
        self._rooms_only = False
        self.intake_handlers: set[Any] = set()

    async def _catch_errors(self, handler, data):
        completed = None
        if getattr(getattr(data, "type", None), "is_to_device", False) is True:
            payload = json.dumps(
                data.serialize(), sort_keys=True, separators=(",", ":")
            )
            completed = (handler, hashlib.sha256(payload.encode()).digest())
            if completed in self._completed_key_handlers:
                return
        event_id = str(getattr(data, "event_id", "") or "")
        store = self.client.sync_store
        intake = (
            handler in self.intake_handlers
            and bool(event_id)
            and isinstance(store, DurableSyncStore)
        )
        if intake and not store.reserve_intake(event_id):
            return
        try:
            result = await handler(data)
            if intake:
                if result is False:
                    self.failed_sync_handlers.add((handler, event_id))
                elif result is True:
                    await store.accept_intake(event_id)
            if completed is not None:
                self._completed_key_handlers.add(completed)
        except BaseException as exc:
            event_id = getattr(data, "event_id", None)
            if event_id:
                self.failed_sync_handlers.add((handler, str(event_id)))
            if isinstance(exc, Exception):
                logger.exception(
                    "Matrix: sync handler failed for event %s", event_id or "<internal>"
                )
            raise
        finally:
            if intake:
                store.release_intake(event_id)

    def own_tasks(self, tasks: list[asyncio.Task]) -> None:
        for task in tasks:
            if task not in self._owned_tasks:
                self._owned_tasks.add(task)
                self._sync_dispatch_tasks.append(task)

    async def cancel(self) -> None:
        while self._owned_tasks:
            tasks = tuple(self._owned_tasks)
            for task in tasks:
                if not task.done() and not task.cancelling():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._owned_tasks.difference_update(tasks)
        self._sync_dispatch_tasks.clear()

    async def _drain_sync_tasks(self) -> None:
        errors = []
        while self._sync_dispatch_tasks:
            tasks, self._sync_dispatch_tasks = self._sync_dispatch_tasks, []
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                self._owned_tasks.difference_update(tasks)
            errors.extend(
                result for result in results if isinstance(result, BaseException)
            )
        if errors:
            raise errors[0]

    def acknowledge(self) -> None:
        self._completed_key_handlers.clear()

    async def dispatch_sync(self, response: dict[str, Any]) -> None:
        self.failed_sync_handlers.clear()
        try:
            self.client.handle_sync({
                key: value for key, value in response.items() if key != "rooms"
            })
            await self._drain_sync_tasks()
            self._rooms_only = True
            rooms = response.get("rooms", {})
            joined = rooms.get("join", {})
            state = {
                room_id: {"state": room.get("state", {})}
                for room_id, room in joined.items()
            }
            self.client.handle_sync({"rooms": {"join": state}})
            await self._drain_sync_tasks()
            timelines = {
                room_id: {key: value for key, value in room.items() if key != "state"}
                for room_id, room in joined.items()
            }
            self.client.handle_sync({"rooms": {**rooms, "join": timelines}})
            await self._drain_sync_tasks()
        except BaseException:
            await self.cancel()
            raise
        finally:
            self._rooms_only = False

    async def decrypt_sync_event(self, event) -> None:
        from mautrix.errors import DecryptionError

        try:
            decrypted = await self.client.crypto.decrypt_megolm_event(event)
        except DecryptionError as exc:
            self.client.crypto_log.warning(
                "Failed to decrypt %s: %s", event.event_id, exc
            )
            return
        self.client.dispatch_event(decrypted, event.source)


def create_sync_client(**kwargs):
    from mautrix.client import Client, InternalEventType

    class MatrixSyncClient(Client):
        def __init__(self, **kwargs):
            self.hermes_sync = SyncDispatch(self)
            super().__init__(**kwargs)

        async def _catch_errors(self, handler, data):
            await self.hermes_sync._catch_errors(handler, data)

        def dispatch_manual_event(self, *args, **kwargs):
            tasks = super().dispatch_manual_event(*args, **kwargs)
            self.hermes_sync.own_tasks(tasks)
            return tasks

        def _dispatch_manual_event(
            self, event_type, data, include_global_handlers, force_synchronous, source
        ):
            # handle_sync emits device counts even when the response contains only rooms.
            if self.hermes_sync._rooms_only and event_type in {
                InternalEventType.DEVICE_OTK_COUNT,
                InternalEventType.DEVICE_LISTS,
            }:
                return []
            tasks = super()._dispatch_manual_event(
                event_type, data, include_global_handlers, True, source
            )
            self.hermes_sync.own_tasks(tasks)
            return tasks

    return MatrixSyncClient(**kwargs)


class _CancelledOlmDispatch(Exception):
    pass


def create_sync_olm_machine(client: Any, crypto_store: PgCryptoStore, state_store: Any):
    from mautrix.crypto import OlmAccount, OlmMachine
    from mautrix.errors import DecryptionError
    from mautrix.types import DecryptedOlmEvent, EventType, SerializerError

    class MatrixOlmMachine(OlmMachine):
        crypto_store: PgCryptoStore

        def __init__(self, client: Any, crypto_store: PgCryptoStore, state_store: Any):
            self._to_device_lock = asyncio.Lock()
            super().__init__(client, crypto_store, state_store)

        async def _handle_sync_to_device(self, event: ToDeviceEvent) -> None:
            try:
                decrypted = await self._decrypt_olm_event(event)
            except (DecryptionError, json.JSONDecodeError, SerializerError):
                self.log.warning("Matrix: skipped unreadable Olm to-device ciphertext")
                return
            if decrypted.type == EventType.ROOM_KEY:
                await self._receive_room_key(decrypted)
                return
            if decrypted.type == EventType.FORWARDED_ROOM_KEY:
                await self._receive_forwarded_room_key(decrypted)
                return
            decrypted.type = decrypted.type.with_class(EventType.Class.TO_DEVICE)
            source = getattr(event, "source")
            setattr(decrypted, "source", source)
            await asyncio.gather(
                *self.client.dispatch_manual_event(
                    decrypted.type,
                    decrypted,
                    include_global_handlers=True,
                    source=source,
                )
            )

        async def handle_to_device_event(self, evt: ToDeviceEvent) -> None:
            if isinstance(evt, DecryptedOlmEvent):
                self.log.warning("Matrix: skipped nested encrypted to-device event")
                return
            async with self._to_device_lock:
                store = self.crypto_store
                current_account = self.account
                if current_account is None:
                    raise RuntimeError(
                        "Matrix crypto account must be loaded before to-device dispatch"
                    )
                account = OlmAccount.from_pickle(
                    current_account.pickle(store.pickle_key),
                    passphrase=store.pickle_key,
                    shared=current_account.shared,
                )
                try:
                    async with store.transaction():
                        try:
                            await self._handle_sync_to_device(evt)
                        except asyncio.CancelledError as exc:
                            # mautrix 0.21.1's SQLite transaction rolls back only Exception.
                            raise _CancelledOlmDispatch() from exc
                except BaseException as exc:
                    # SQL rollback does not restore the SDK's mutable ratchet and account caches.
                    store._olm_cache.clear()
                    store._account = self.account = account
                    if isinstance(exc, _CancelledOlmDispatch):
                        raise asyncio.CancelledError from exc
                    raise

    return MatrixOlmMachine(client, crypto_store, state_store)
