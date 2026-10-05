"""Durable preservation of EXTERNAL completion-contract artifacts (Kanban P3b).

An artifact declared in a card's ``completion_contract``
(``{"required_artifacts": [...]}``) that lives OUTSIDE the managed scratch
workspace is validated by ``_gate_contract_completion`` (existence on disk) and
re-checked in-txn by ``_recheck_contract_in_txn`` — but a pathname is not proof
of content, and an SQLite lock does not lock the filesystem: a delete/replace
between the in-txn recheck and the commit (or after it) left a ``done`` card
pointing at content that no longer existed or had changed. That is the
``delete-after-recheck`` race this module closes.

The guarantee is a Hermes-MANAGED durable copy:

  1. **Capture** (before the write lock) — each real external requirement is
     read whole, hashed (sha256) and re-statted; a file that changes/disappears
     while being read fails the completion (never close onto unstable bytes).
     A file above ``KANBAN_ATTACHMENT_MAX_BYTES`` is a typed refusal.
  2. **Atomic publish** (before the write lock) — the captured bytes are written
     to a per-task staging file in the attachments dir, fsync'd, then published
     with ``os.replace``; the directory is fsync'd too. Large copies never hold
     the write lock.
  3. **In-txn binding** — inside ``complete_task``'s write transaction (after the
     existing in-txn contract recheck) the preserved copy is recorded as an
     attachment, bound on the closing run's metadata, promoted into the
     ``completed`` payload, and audited with an ``external_artifact_preserved``
     event (origin + managed dest + sha256 + size).

Only the managed copy is guaranteed durable: the continued existence of the
external original is NOT — the event records where it came from, not a promise
it stays. A crash before commit leaves at most a recoverable orphan copy (a
``.p3b-staging-*`` blob, or a published blob with no attachment row); it can
never leave a ``done`` card referencing content that is gone, because the binding
and the status flip share one transaction. A rollback discards the published
copy (see :func:`discard_published_artifacts`).

This module deliberately does NOT reuse scratch staging
(``_persist_scratch_completion_artifacts``): scratch artifacts are ours and
copied in-txn, while external ones need a pre-lock capture to survive exactly the
window the lock cannot cover. Cards without external requirements return an
empty capture — byte-for-byte pre-P3b behaviour.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Test seam: ``callable(origin_path: str)`` invoked between reading an origin's
# bytes and the post-read stability check, so tests can simulate a file that
# mutates mid-capture without sleeps. Production leaves this ``None``.
_CAPTURE_MID_READ_HOOK = None

_READ_CHUNK = 1024 * 1024
_STAGING_PREFIX = ".p3b-staging-"


@dataclass
class CapturedExternalArtifact:
    """One external requirement captured in memory (bytes + digest + origin)."""
    origin: str
    data: bytes
    sha256: str
    size: int
    board: Optional[str] = None
    published_path: Optional[Path] = None


def _required_artifacts(conn: sqlite3.Connection, task_id: str) -> list[str]:
    """The card's declared external requirements, read WITHOUT side effects.

    Mirrors the gate's tolerant parse (inert/legacy/unparseable/non-dict shapes
    declare nothing) but deliberately does NOT call the event-emitting helpers:
    capture must never add audit noise for a card whose contract enforces
    nothing.
    """
    try:
        row = conn.execute(
            "SELECT completion_contract FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
    except sqlite3.Error:
        return []
    if row is None:
        return []
    raw = row["completion_contract"]
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() in ("", "local-only"):
        return []
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, dict):
        return []
    artifacts = parsed.get("required_artifacts")
    if not isinstance(artifacts, (list, tuple)):
        return []
    return [item for item in artifacts if isinstance(item, str) and item.strip()]


def _board_for_task(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Board whose attachments dir should hold the copy: the managed scratch
    workspace's board when it has one, else ``None`` (the current board)."""
    workspace = _kb._scratch_workspace(conn, task_id)
    if workspace is None:
        return None
    managed, board = _kb._managed_scratch_path_info(workspace)
    return board if managed else None


def capture_external_artifacts(
    conn: sqlite3.Connection, task_id: str,
) -> list[CapturedExternalArtifact]:
    """Read every real EXTERNAL requirement whole, with a digest and a
    stability check. Returns ``[]`` for a card with no external requirements
    (nothing to preserve, pre-P3b behaviour). Raises
    :data:`ExternalArtifactPreservationError` when a requirement is too large or
    changes/disappears while being read — never returns unstable bytes."""
    captured: list[CapturedExternalArtifact] = []
    seen: set[str] = set()
    board: Optional[str] = None
    board_resolved = False
    for item in _required_artifacts(conn, task_id):
        if item in seen:
            continue
        seen.add(item)
        path = Path(item).expanduser()
        try:
            if not path.is_file():
                continue  # the gate/recheck own the missing-file refusal
            if _kb._is_managed_scratch_path(path):
                continue  # inside managed scratch: staged in-txn, not ours here
        except OSError:
            continue
        if not board_resolved:
            board = _board_for_task(conn, task_id)
            board_resolved = True
        data, digest, size = _read_stable(path, item)
        captured.append(CapturedExternalArtifact(
            origin=item, data=data, sha256=digest, size=size, board=board,
        ))
    return captured


def _read_stable(path: Path, artifact: str) -> tuple[bytes, str, int]:
    """Read *path* whole; abort when it exceeds the cap or changes mid-read.

    Stability is checked three ways around the read: the fd's size/mtime/ino
    fingerprint before vs after, and the path itself re-stat'd (a replaced path
    has a different inode). A mismatch is a typed refusal — a card must never be
    closed against content that was not the content validated.
    """
    cap = _kb.KANBAN_ATTACHMENT_MAX_BYTES
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = handle.read(_READ_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > cap:
                raise ExternalArtifactPreservationError(
                    f"declared external artifact exceeds the {cap}-byte limit: {artifact}"
                )
            chunks.append(chunk)
        data = b"".join(chunks)
        hook = _CAPTURE_MID_READ_HOOK
        if callable(hook):
            hook(str(path))
        after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise ExternalArtifactPreservationError(
            f"declared external artifact changed while being captured: {artifact}"
        )
    try:
        after_path = os.stat(path)
    except OSError as exc:
        raise ExternalArtifactPreservationError(
            f"declared external artifact disappeared while being captured: {artifact}"
        ) from exc
    if after_path.st_ino != before.st_ino:
        raise ExternalArtifactPreservationError(
            f"declared external artifact was replaced while being captured: {artifact}"
        )
    return data, hashlib.sha256(data).hexdigest(), len(data)


def publish_external_artifacts(
    captured: list[CapturedExternalArtifact], task_id: str,
) -> list[CapturedExternalArtifact]:
    """Atomically publish each captured copy into the task's attachments dir.

    A staging file is written (fsync), then ``os.replace``d onto its final
    collision-free name, then the directory is fsync'd. Returns the captured
    items with ``published_path`` set; any failure discards the already
    published copies before propagating.
    """
    published: list[CapturedExternalArtifact] = []
    used: set[Path] = set()
    try:
        for cap in captured:
            dest_dir = _kb.task_attachments_dir(task_id, board=cap.board)
            dest_dir.mkdir(parents=True, exist_ok=True)
            try:
                safe_name = _kb._safe_attachment_name(Path(cap.origin).name)
            except ValueError:
                safe_name = "artifact"
            final = _kb._unique_attachment_path(dest_dir, safe_name, used)
            staging = dest_dir / (_STAGING_PREFIX + secrets.token_hex(12))
            fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(cap.data)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                with contextlib.suppress(OSError):
                    staging.unlink(missing_ok=True)
                raise
            os.replace(staging, final)
            _fsync_dir(dest_dir)
            used.add(final)
            cap.published_path = final
            published.append(cap)
    except BaseException:
        discard_published_artifacts(published)
        raise
    return published


def _fsync_dir(directory: Path) -> None:
    """Best-effort directory fsync so a published rename survives a crash."""
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def discard_published_artifacts(published: list[CapturedExternalArtifact]) -> None:
    """Remove published copies whose binding never committed (txn rollback).

    Best-effort, like the scratch equivalents: a leaked copy is recoverable
    orphan debris, never a broken reference (no attachment row points at it).
    Empty per-task dirs are removed too.
    """
    for cap in published:
        if cap.published_path is None:
            continue
        with contextlib.suppress(OSError):
            Path(cap.published_path).unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            Path(cap.published_path).parent.rmdir()


def bind_external_artifacts(
    conn: sqlite3.Connection, task_id: str,
    published: list[CapturedExternalArtifact], metadata: dict, now: int,
) -> list[dict]:
    """Bind the preserved copies inside the completion write transaction.

    For each published copy: record an attachment row, append the auditable
    ``external_artifact_preserved`` event, and add the binding to the closing
    run's ``metadata['external_artifacts_preserved']`` plus the managed path to
    ``metadata['artifacts']`` (so the ``completed`` payload names the durable
    copy, not only the origin). Returns the binding dicts.
    """
    bindings: list[dict] = []
    for cap in published:
        if cap.published_path is None:
            continue
        stored = str(Path(cap.published_path).resolve())
        _kb._insert_completion_attachment(
            conn, task_id, filename=Path(cap.published_path).name, stored_path=stored,
            size=cap.size, created_at=now, uploaded_by="kanban_complete",
        )
        _kb._append_event(
            conn, task_id, "external_artifact_preserved",
            {
                "origin": cap.origin, "stored_path": stored,
                "sha256": cap.sha256, "size": cap.size, "source": "completion_contract",
            },
        )
        bindings.append({
            "origin": cap.origin, "stored_path": stored,
            "sha256": cap.sha256, "size": cap.size,
        })
    if bindings and isinstance(metadata, dict):
        metadata["external_artifacts_preserved"] = bindings
        artifacts = metadata.get("artifacts")
        merged = list(artifacts) if isinstance(artifacts, (list, tuple)) else []
        for binding in bindings:
            if binding["stored_path"] not in merged:
                merged.append(binding["stored_path"])
        metadata["artifacts"] = merged
    return bindings


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it. The typed
# error is the origin's canonical ``ExternalArtifactPreservationError`` (a
# subclass of ``ArtifactPreservationError``), so the existing tool error
# handlers (``tools/kanban_tools.py``) already treat it as a recoverable
# preservation failure; re-exported here for callers that only import this
# module.
from hermes_cli import kanban_db as _kb  # noqa: E402

ExternalArtifactPreservationError = _kb.ExternalArtifactPreservationError
