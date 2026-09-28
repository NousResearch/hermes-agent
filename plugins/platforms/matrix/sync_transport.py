"""Durable Matrix sync positions and completion of required event dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


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

    async def load(self) -> None:
        self._next_batch = await asyncio.to_thread(self._read)

    def _read(self) -> str | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (ValueError, UnicodeError) as exc:
            logger.warning("Matrix: invalid sync cursor file %s: %s", self.path, exc)
            return None
        token = data.get("next_batch") if isinstance(data, dict) else None
        return token if isinstance(token, str) and token else None

    async def get_next_batch(self) -> str | None:
        return self._next_batch

    async def put_next_batch(self, next_batch: str) -> None:
        await asyncio.to_thread(self._write, next_batch)
        self._next_batch = next_batch

    def _write(self, next_batch: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump({"next_batch": next_batch}, stream)
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
        self.failed_sync_event_ids: set[str] = set()
        self._completed_key_handlers: set[tuple[Any, bytes]] = set()
        self._rooms_only = False

    async def _catch_errors(self, handler, data):
        completed = None
        if getattr(getattr(data, "type", None), "is_to_device", False) is True:
            payload = json.dumps(
                data.serialize(), sort_keys=True, separators=(",", ":")
            )
            completed = (handler, hashlib.sha256(payload.encode()).digest())
            if completed in self._completed_key_handlers:
                return
        try:
            await handler(data)
            if completed is not None:
                self._completed_key_handlers.add(completed)
        except BaseException as exc:
            event_id = getattr(data, "event_id", None)
            if event_id:
                self.failed_sync_event_ids.add(str(event_id))
            if isinstance(exc, Exception):
                logger.exception(
                    "Matrix: sync handler failed for event %s", event_id or "<internal>"
                )
            raise

    async def _drain_sync_tasks(self) -> None:
        errors = []
        while self._sync_dispatch_tasks:
            tasks, self._sync_dispatch_tasks = self._sync_dispatch_tasks, []
            results = await asyncio.gather(*tasks, return_exceptions=True)
            errors.extend(
                result for result in results if isinstance(result, BaseException)
            )
        if errors:
            raise errors[0]

    def acknowledge(self) -> None:
        self._completed_key_handlers.clear()

    async def dispatch_sync(self, response: dict[str, Any]) -> None:
        self.failed_sync_event_ids.clear()
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
            for task in self._sync_dispatch_tasks:
                task.cancel()
            await asyncio.gather(*self._sync_dispatch_tasks, return_exceptions=True)
            self._sync_dispatch_tasks.clear()
            raise
        finally:
            self._rooms_only = False

    async def decrypt_sync_event(self, event) -> None:
        decrypted = await self.client.crypto.decrypt_megolm_event(event)
        self.client.dispatch_event(decrypted, event.source)


def create_sync_client(**kwargs):
    from mautrix.client import Client, InternalEventType

    class MatrixSyncClient(Client):
        def __init__(self, **kwargs):
            self.hermes_sync = SyncDispatch(self)
            super().__init__(**kwargs)

        async def _catch_errors(self, handler, data):
            await self.hermes_sync._catch_errors(handler, data)

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
            self.hermes_sync._sync_dispatch_tasks.extend(tasks)
            return tasks

    return MatrixSyncClient(**kwargs)
