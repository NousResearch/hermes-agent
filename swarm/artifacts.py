"""Swarm artifact bus — shared artifact storage with provenance tracking."""

from __future__ import annotations

import hashlib
import mimetypes
import shutil
import threading
from pathlib import Path

from .types import ArtifactRef


class ArtifactStore:
    def __init__(self, base_dir: Path) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # artifact_id -> ArtifactRef
        self._index: dict[str, ArtifactRef] = {}

    # -- storage --

    def store(self, task_id: str, path: Path, mime_type: str | None = None) -> ArtifactRef:
        path = Path(path)
        data = path.read_bytes()
        if mime_type is None:
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return self.store_bytes(task_id, data, path.name, mime_type)

    def store_bytes(self, task_id: str, data: bytes, name: str, mime_type: str | None = None) -> ArtifactRef:
        checksum = hashlib.sha256(data).hexdigest()
        if mime_type is None:
            mime_type = mimetypes.guess_type(name)[0] or "application/octet-stream"

        ref = ArtifactRef(
            task_id=task_id,
            path="",  # filled below
            mime_type=mime_type,
            size_bytes=len(data),
            checksum=checksum,
        )

        task_dir = self._base / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        dest = task_dir / f"{ref.id}_{name}"
        ref.path = str(dest)

        dest.write_bytes(data)

        with self._lock:
            self._index[ref.id] = ref

        return ref

    # -- retrieval --

    def get(self, artifact_id: str) -> Path | None:
        with self._lock:
            ref = self._index.get(artifact_id)
        if ref is None:
            return None
        p = Path(ref.path)
        return p if p.exists() else None

    def get_by_task(self, task_id: str) -> list[ArtifactRef]:
        with self._lock:
            return [r for r in self._index.values() if r.task_id == task_id]

    def list_all(self) -> list[ArtifactRef]:
        with self._lock:
            return list(self._index.values())

    # -- deletion --

    def delete(self, artifact_id: str) -> None:
        with self._lock:
            ref = self._index.pop(artifact_id, None)
        if ref is None:
            return
        p = Path(ref.path)
        if p.exists():
            p.unlink()

    def cleanup_task(self, task_id: str) -> None:
        with self._lock:
            ids = [r.id for r in self._index.values() if r.task_id == task_id]
            for aid in ids:
                self._index.pop(aid, None)
        task_dir = self._base / task_id
        if task_dir.exists():
            shutil.rmtree(task_dir)

    # -- manifest --

    def export_manifest(self) -> dict:
        with self._lock:
            refs = list(self._index.values())
        return {
            "artifacts": [
                {
                    "id": r.id,
                    "task_id": r.task_id,
                    "path": r.path,
                    "mime_type": r.mime_type,
                    "size_bytes": r.size_bytes,
                    "checksum": r.checksum,
                    "created_at": r.created_at.isoformat(),
                }
                for r in refs
            ]
        }
