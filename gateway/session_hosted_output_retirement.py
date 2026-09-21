"""Internal verification of retained Home publication while viewers are fenced.

Borrow the existing store; never construct it or expose this adapter to Files RPC.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from gateway.hosted_room_artifacts import RoomArtifactError, RoomArtifactScope
from gateway.hosted_room_attachments import (
    AttachmentData, HostedRoomAttachmentStore, read_viewer_from_store,
)
from gateway.hosted_room_output_fence import require_output_task


class RetainedPublication:
    def __init__(self, service: Any, scope: RoomArtifactScope) -> None:
        self.service, self.scope = service, scope
        self.store: HostedRoomAttachmentStore = service.output_attachments
        self.db_path = self.store.db_path
        self.clock = self.store.clock

    def _require_viewer_room(
        self, conn: sqlite3.Connection, *, room_id: str,
        authority_gateway_id: str, authority_epoch: int,
    ) -> object:
        scope = self.scope
        if (room_id, authority_gateway_id, authority_epoch) != (scope.room_id, scope.authority_gateway_id, scope.authority_epoch):
            raise RoomArtifactError('Group Chat cleanup publication changed')
        self.service._output_owner(conn)
        row = conn.execute('SELECT cancel_generation FROM hosted_room_driver_tasks WHERE room_id=? AND task_id=?',
                           (scope.room_id, scope.task_id)).fetchone()
        if row is None:
            raise RoomArtifactError('Group Chat cleanup task unavailable')
        return require_output_task(conn, scope, row[0], cleanup=True)

    @contextmanager
    def _viewer_snapshot(self) -> Iterator[sqlite3.Connection]:
        with self.service._output_policy_read() as conn:
            yield conn

    def read_viewer(
        self, *, room_id: Any, attachment_id: Any, event_id: Any | None = None,
        recipient_member_id: Any = None, authority_gateway_id: Any,
        authority_epoch: Any,
    ) -> AttachmentData:
        return read_viewer_from_store(
            self, room_id=room_id, attachment_id=attachment_id, event_id=event_id,
            recipient_member_id=recipient_member_id,
            authority_gateway_id=authority_gateway_id,
            authority_epoch=authority_epoch,
        )

    _read_committed_row = staticmethod(HostedRoomAttachmentStore._read_committed_row)

    def _read_blob(self, **kwargs):
        return self.store._read_blob(**kwargs)

    def _metadata(self, row):
        return self.store._metadata(row)
