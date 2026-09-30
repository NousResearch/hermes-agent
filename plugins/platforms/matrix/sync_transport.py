"""Durable Matrix sync positions and completion of required event dispatch."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Callable

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

    async def put_next_batch(
        self, next_batch: str, *, keep: Callable[[], frozenset[str]] = frozenset
    ) -> None:
        """Advance the cursor and drop the admitted event IDs that *keep* does not return.

        *keep* names the IDs of responses whose cursors are not written yet. It is called under
        the write lock, so an ID admitted while this write waits for the lock is kept too.
        """
        await self._commit(next_batch=next_batch, keep=keep)

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
        await self.accept_intakes((event_id,))

    async def accept_intakes(self, event_ids: tuple[str, ...]) -> None:
        await self._commit(event_ids=event_ids)

    async def _commit(
        self,
        *,
        next_batch: str | None = None,
        event_ids: tuple[str, ...] = (),
        keep: Callable[[], frozenset[str]] = frozenset,
    ) -> None:
        async def write() -> None:
            async with self._write_lock:
                accepted = (
                    self._accepted_events | set(event_ids)
                    if event_ids
                    else self._accepted_events & keep()
                )
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


def _consumed(receipt: asyncio.Future) -> bool:
    return (
        receipt.done()
        and not receipt.cancelled()
        and receipt.exception() is None
        and receipt.result() is True
    )


@dataclass(frozen=True)
class _PendingCheckpoint:
    next_batch: str
    seen: frozenset[str]
    receipts: tuple[tuple[str, asyncio.Future], ...]


class SyncCheckpoints:
    """Persist sync cursors in response order once their buffered intake is consumed.

    A text batch waits for a quiet period before it reaches the gateway, and the sync loop
    keeps requesting later responses meanwhile so that later messages can join the batch. The
    cursor of a response that contributed to an open batch is persisted only after the batch
    reports that the gateway consumed it, and only after every earlier response. A restart
    before then resumes from the older cursor and delivers the buffered events again.
    """

    def __init__(
        self,
        store: DurableSyncStore,
        dispatching: Callable[[], frozenset[str]] = frozenset,
    ) -> None:
        self.store = store
        self._dispatching = dispatching
        self._pending: deque[_PendingCheckpoint] = deque()
        self._task: asyncio.Task | None = None
        self._failed: tuple[str, ...] | None = None

    async def commit(
        self,
        next_batch: str,
        seen: frozenset[str],
        receipts: tuple[tuple[str, asyncio.Future], ...],
    ) -> None:
        if self._failed is not None:
            return
        receipts = tuple(
            (event_id, receipt)
            for event_id, receipt in receipts
            if not _consumed(receipt)
        )
        if not receipts and not self._pending:
            await self.store.put_next_batch(next_batch, keep=self._unwritten_intakes)
            return
        self._pending.append(_PendingCheckpoint(next_batch, seen, receipts))
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        while self._pending:
            checkpoint = self._pending[0]
            results = await asyncio.gather(
                *(receipt for _event_id, receipt in checkpoint.receipts),
                return_exceptions=True,
            )
            failed = tuple(
                event_id
                for (event_id, _receipt), result in zip(checkpoint.receipts, results)
                if result is not True
            )
            if failed:
                logger.warning(
                    "Matrix: buffered intake of %s was not consumed; the sync position "
                    "will be retried",
                    ", ".join(failed),
                )
                self._failed = failed
                self._pending.clear()
                return
            await self.store.put_next_batch(
                checkpoint.next_batch, keep=self._unwritten_intakes
            )
            self._pending.popleft()

    def _unwritten_intakes(self) -> frozenset[str]:
        """Intake IDs of the responses after the cursor being written: the queued ones and
        the one that is still being dispatched."""
        return frozenset().union(
            *(pending.seen for pending in list(self._pending)[1:]), self._dispatching()
        )

    def take_failure(self) -> tuple[str, ...] | None:
        """Event IDs whose buffered intake failed since the last call, or None."""
        failed, self._failed = self._failed, None
        return failed

    async def settled(self) -> None:
        while self._task is not None and not self._task.done():
            await asyncio.wait({self._task})

    async def cancel(self) -> None:
        self._pending.clear()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)


class SyncDispatch:
    def __init__(self, client: Any):
        self.client = client
        self._sync_dispatch_tasks: list[asyncio.Task] = []
        self._owned_tasks: set[asyncio.Task] = set()
        self.failed_sync_handlers: set[tuple[Any, str]] = set()
        self._completed_key_handlers: set[tuple[Any, bytes]] = set()
        self._rooms_only = False
        self.intake_handlers: set[Any] = set()
        self._seen_intakes: set[str] = set()
        self._deferred_intakes: dict[str, asyncio.Future] = {}

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
        if intake:
            self._seen_intakes.add(event_id)
            if not store.reserve_intake(event_id):
                return
        deferred = False
        try:
            result = await handler(data)
            if intake:
                if isinstance(result, asyncio.Future):
                    # The receipt resolves when the text batch reaches the gateway.
                    self._deferred_intakes[event_id] = result
                    result.add_done_callback(lambda _: store.release_intake(event_id))
                    deferred = True
                elif result is False:
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
            if intake and not deferred:
                store.release_intake(event_id)

    def own_tasks(self, tasks: list[asyncio.Task]) -> None:
        for task in tasks:
            if task not in self._owned_tasks:
                self._owned_tasks.add(task)
                self._sync_dispatch_tasks.append(task)

    def own_background_task(self, task: asyncio.Task) -> None:
        """Cancel *task* on disconnect without making sync dispatch wait for it."""
        self._owned_tasks.add(task)
        task.add_done_callback(self._owned_tasks.discard)

    def dispatching_intakes(self) -> frozenset[str]:
        """The intake event IDs of the response that is being dispatched."""
        return frozenset(self._seen_intakes)

    def take_intakes(self) -> tuple[frozenset[str], tuple[tuple[str, asyncio.Future], ...]]:
        """The intake event IDs of the last response, and the receipts still buffered."""
        seen, deferred = frozenset(self._seen_intakes), tuple(self._deferred_intakes.items())
        self._seen_intakes.clear()
        self._deferred_intakes.clear()
        return seen, deferred

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
        if self.failed_sync_handlers:
            raise RuntimeError("Matrix gateway refused sync intake")

    def acknowledge(self) -> None:
        self._completed_key_handlers.clear()

    async def dispatch_sync(self, response: dict[str, Any]) -> None:
        self.failed_sync_handlers.clear()
        self._seen_intakes.clear()
        self._deferred_intakes.clear()
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

        def dispatch_manual_event(
            self,
            event_type,
            data,
            include_global_handlers=False,
            force_synchronous=False,
            source=None,
        ):
            params = (
                event_type,
                data,
                include_global_handlers,
                force_synchronous,
                source,
            )
            middlewares = self.event_middlewares.get(event_type, [])
            if not middlewares:
                return self._dispatch_manual_event(*params)

            async def run_middlewares():
                for middleware in middlewares:
                    if not await middleware(data):
                        return
                await asyncio.gather(*self._dispatch_manual_event(*params))

            tasks = [asyncio.create_task(run_middlewares())]
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
