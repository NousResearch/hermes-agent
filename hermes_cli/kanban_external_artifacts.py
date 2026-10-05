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
     A file above ``KANBAN_ATTACHMENT_MAX_BYTES`` is a typed refusal. The set
     of requirements is taken from the GATE's contract snapshot (round 2):
     capture never re-reads ``tasks.completion_contract`` autonomously, so a
     concurrent contract swap can no longer make it preserve a version the
     closing transaction will not validate. A declared external requirement
     that is absent at capture time is a TYPED REFUSAL, never a silent skip.
  2. **Atomic publish** (before the write lock) — the captured bytes are written
     to a per-attempt staging file in the attachments dir, fsync'd, then
     LINKED (``os.link``) onto a collision-free final name; the link is atomic
     and NEVER overwrites an existing file, so two concurrent completions of the
     same task cannot replace or delete each other's proof. Staging files carry
     a per-attempt random token. The directory (and any parent created by
     ``mkdir(parents=True)``) is fsync'd; a directory fsync that cannot be
     guaranteed is a typed refusal, not a silent promise.
  3. **In-txn binding** — inside ``complete_task``'s write transaction (after the
     existing in-txn contract recheck) the preserved copy is recorded as an
     attachment, bound on the closing run's metadata, promoted into the
     ``completed`` payload, and audited with an ``external_artifact_preserved``
     event (origin + managed dest + sha256 + size). The txn additionally
     verifies that the captured set COVERS every external requirement of the
     in-force contract — an uncovered requirement rolls the txn back with the
     existing divergence mechanism.

Only the managed copy is guaranteed durable: the continued existence of the
external original is NOT — the event records where it came from, not a promise
it stays. A crash before commit leaves at most a recoverable orphan copy (a
``.p3b-staging-*`` blob, or a published blob with no attachment row); it can
never leave a ``done`` card referencing content that is gone, because the binding
and the status flip share one transaction. Copies are discarded ONLY when their
binding did not commit (see :func:`discard_published_artifacts`): a transaction
that committed and then failed a post-commit invariant keeps its bound copies.

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

# Test seam: ``callable(candidate_path: str)`` invoked immediately BEFORE the
# atomic ``os.link`` that reserves a final name — the exact window where a
# concurrent attempt can win the name. Tests use it to make the collision
# deterministic (create the colliding file, then assert we do not overwrite it).
# Production leaves this ``None``.
_PUBLISH_PRE_LINK_HOOK = None

# Sentinel for "no gate snapshot supplied": capture falls back to reading the
# stored contract directly. Passed explicitly by ``complete_task`` so capture
# rides the SAME contract version the gate enforced (round-2 HIGH #2).
_NO_SNAPSHOT = object()

_READ_CHUNK = 1024 * 1024
_STAGING_PREFIX = ".p3b-staging-"
_MAX_NAME_ATTEMPTS = 100_000


@dataclass
class CapturedExternalArtifact:
    """One external requirement captured in memory (bytes + digest + origin)."""
    origin: str
    data: bytes
    sha256: str
    size: int
    board: Optional[str] = None
    published_path: Optional[Path] = None


def _requirements_from_text(text: Optional[str]) -> list[str]:
    """Tolerant parse of a stored ``completion_contract`` value into its declared
    ``required_artifacts`` (non-string entries and blanks ignored). Mirrors the
    gate's parse exactly so capture/coverage never see a different list.
    """
    if text is None:
        return []
    stripped = str(text).strip()
    if not stripped or stripped.lower() in _kb._INERT_COMPLETION_CONTRACTS:
        return []
    try:
        parsed = json.loads(stripped)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, dict):
        return []
    artifacts = parsed.get("required_artifacts")
    if not isinstance(artifacts, (list, tuple)):
        return []
    return [item for item in artifacts if isinstance(item, str) and item.strip()]


def _required_artifacts(conn: sqlite3.Connection, task_id: str) -> list[str]:
    """The card's declared external requirements, read WITHOUT side effects.

    Fallback used only when no gate snapshot was supplied; ``complete_task``
    passes the gate's snapshot so the normal path never re-reads autonomously.
    """
    try:
        row = conn.execute(
            "SELECT completion_contract FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
    except sqlite3.Error:
        return []
    if row is None:
        return []
    return _requirements_from_text(row["completion_contract"])


def _board_for_task(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Board whose attachments dir should hold the copy: the managed scratch
    workspace's board when it has one, else ``None`` (the current board)."""
    workspace = _kb._scratch_workspace(conn, task_id)
    if workspace is None:
        return None
    managed, board = _kb._managed_scratch_path_info(workspace)
    return board if managed else None


def _is_external_requirement(item: str) -> bool:
    """True when *item* names a file OUTSIDE managed scratch storage (i.e. not a
    scratch artifact the in-txn staging owns). Unreadable paths are treated as
    external so a missing declaration is still refused rather than ignored."""
    try:
        return not _kb._is_managed_scratch_path(Path(item).expanduser())
    except OSError:
        return True


def required_external_artifacts(
    conn: sqlite3.Connection, task_id: str, contract_snapshot: object = _NO_SNAPSHOT,
) -> list[str]:
    """External requirements of the contract IN FORCE (the gate snapshot when
    given, else the stored value), deduped and in declaration order."""
    if contract_snapshot is _NO_SNAPSHOT:
        items = _required_artifacts(conn, task_id)
    else:
        items = _requirements_from_text(contract_snapshot)  # type: ignore[arg-type]
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        if _is_external_requirement(item):
            out.append(item)
    return out


def uncaptured_external_requirements(
    conn: sqlite3.Connection, task_id: str,
    published: list[CapturedExternalArtifact],
    contract_snapshot: object = _NO_SNAPSHOT,
) -> list[str]:
    """External requirements of the in-force contract NOT covered by *published*.

    An empty result means the captured/published set covers every external
    requirement the closing transaction will validate; otherwise the caller must
    refuse (the captured set is incomplete relative to the validated contract).
    """
    covered = {cap.origin for cap in published}
    return [
        req for req in required_external_artifacts(conn, task_id, contract_snapshot)
        if req not in covered
    ]


def capture_external_artifacts(
    conn: sqlite3.Connection, task_id: str, *,
    contract_snapshot: object = _NO_SNAPSHOT,
) -> list[CapturedExternalArtifact]:
    """Read every real EXTERNAL requirement whole, with a digest and a
    stability check. Returns ``[]`` for a card with no external requirements
    (nothing to preserve, pre-P3b behaviour). Raises
    :data:`ExternalArtifactPreservationError` when a requirement is too large,
    changes/disappears while being read, or is DECLARED BUT ABSENT at capture
    time — never silently skips a required artifact.

    The requirement list is the GATE's contract snapshot when supplied (round 2):
    capture never re-reads the stored contract when the caller says which
    version the completion rides on, so a concurrent A→B→A swap cannot make it
    preserve a version the transaction will not validate.
    """
    if contract_snapshot is _NO_SNAPSHOT:
        requirements = _required_artifacts(conn, task_id)
    else:
        requirements = _requirements_from_text(contract_snapshot)  # type: ignore[arg-type]
    captured: list[CapturedExternalArtifact] = []
    seen: set[str] = set()
    board: Optional[str] = None
    board_resolved = False
    for item in requirements:
        if item in seen:
            continue
        seen.add(item)
        if not _is_external_requirement(item):
            continue  # inside managed scratch: staged in-txn, not ours here
        path = Path(item).expanduser()
        # HIGH #2 (round 2): a DECLARED external requirement that is absent when
        # we try to capture it is a typed refusal — NOT a skip. Silently
        # dropping it let a file vanish before capture and reappear before the
        # recheck, closing the card with no preserved copy at all.
        try:
            present = path.is_file()
        except OSError as exc:
            raise ExternalArtifactPreservationError(
                f"declared external artifact could not be examined: {item}: {exc}"
            ) from exc
        if not present:
            raise ExternalArtifactPreservationError(
                f"declared external artifact is missing when captured: {item}"
            )
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

    Stability is checked four ways around the read: the fd's
    size/mtime/ctime/ino fingerprint before vs after, and the path itself
    re-stat'd (a replaced path has a different ``st_dev``/inode, and an
    in-place change bumps ``ctime``). ``ctime`` closes the same-length +
    restored-mtime rewrite that size/mtime alone cannot see. Any failure to
    open/read/stat is a typed refusal — a card must never be closed against
    content that was not the content validated.
    """
    cap = _kb.KANBAN_ATTACHMENT_MAX_BYTES
    try:
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
    except ExternalArtifactPreservationError:
        raise
    except OSError as exc:
        # LOW (round 2): every raw OS failure (open/read/fstat — a delete
        # between is_file() and open(), EACCES, EIO) is a typed, recoverable
        # refusal, not a raw OSError escaping the preservation contract.
        raise ExternalArtifactPreservationError(
            f"could not read declared external artifact {artifact}: {exc}"
        ) from exc
    if (
        before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino
    ) != (
        after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino
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
    if (after_path.st_dev, after_path.st_ino, after_path.st_ctime_ns) != (
        after.st_dev, after.st_ino, after.st_ctime_ns
    ):
        raise ExternalArtifactPreservationError(
            f"declared external artifact was replaced while being captured: {artifact}"
        )
    return data, hashlib.sha256(data).hexdigest(), len(data)


def publish_external_artifacts(
    captured: list[CapturedExternalArtifact], task_id: str,
) -> list[CapturedExternalArtifact]:
    """Atomically publish each captured copy into the task's attachments dir.

    For each copy: a per-attempt staging file is written (O_EXCL, fsync), then
    LINKED (``os.link``) onto a collision-free final name. The link is atomic
    and fails if the name exists, so a concurrent completion of the same task
    can never overwrite or delete another attempt's proof — the loser simply
    advances to the next name. The destination directory (and any parent
    created by ``mkdir(parents=True)``) is fsync'd; a directory fsync that
    cannot be guaranteed raises the typed refusal. Returns the captured items
    with ``published_path`` set; any failure discards the already published
    copies OF THIS ATTEMPT before propagating.
    """
    published: list[CapturedExternalArtifact] = []
    used: set[Path] = set()
    try:
        for cap in captured:
            dest_dir = _kb.task_attachments_dir(task_id, board=cap.board)
            _ensure_dir_durable(dest_dir)
            try:
                safe_name = _kb._safe_attachment_name(Path(cap.origin).name)
            except ValueError:
                safe_name = "artifact"
            final = _link_publish(cap.data, dest_dir, safe_name, used)
            # Set ownership BEFORE the directory fsync: if the fsync raises the
            # enclosing handler must discard THIS copy too (never an orphan).
            cap.published_path = final
            published.append(cap)
            _fsync_dir(dest_dir)
            used.add(final)
    except BaseException:
        discard_published_artifacts(published)
        raise
    return published


def _link_publish(
    data: bytes, dest_dir: Path, safe_name: str, used: set[Path],
) -> Path:
    """Write *data* to a staging file then atomically link it onto a free final
    name under *dest_dir*. Never overwrites; raises the typed refusal on an
    unlinkable/O_EXCL-unsupported destination."""
    staging = dest_dir / (_STAGING_PREFIX + secrets.token_hex(12))
    fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        _unlink_quietly(staging)
        raise
    stem = Path(safe_name).stem or "artifact"
    suffix = Path(safe_name).suffix
    idx = 0
    try:
        while True:
            candidate = dest_dir / (safe_name if idx == 0 else f"{stem}_{idx}{suffix}")
            if candidate in used:
                idx += 1
                continue
            hook = _PUBLISH_PRE_LINK_HOOK
            if callable(hook):
                hook(str(candidate))
            try:
                os.link(staging, candidate)
            except FileExistsError:
                idx += 1
                if idx > _MAX_NAME_ATTEMPTS:  # pragma: no cover - defensive
                    raise ExternalArtifactPreservationError(
                        f"could not find a free attachment name for {safe_name}"
                    )
                continue
            except OSError as exc:
                raise ExternalArtifactPreservationError(
                    f"could not publish external artifact atomically "
                    f"({safe_name}): {exc}"
                ) from exc
            return candidate
    finally:
        _unlink_quietly(staging)


def _unlink_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _ensure_dir_durable(directory: Path) -> None:
    """Create *directory* (and missing parents) then fsync the new entries so a
    crash cannot lose the directory itself or the rename that follows."""
    to_create: list[Path] = []
    probe = directory
    while not probe.exists() and probe != probe.parent:
        to_create.append(probe)
        probe = probe.parent
    directory.mkdir(parents=True, exist_ok=True)
    # Persist each newly created directory's entry in its own parent, then the
    # directory itself. Raises the typed refusal if the OS cannot guarantee it.
    for created in reversed(to_create):
        _fsync_dir(created.parent)
    _fsync_dir(directory)


def _fsync_dir(directory: Path) -> None:
    """Durably persist *directory*'s entries (so a published rename survives a
    crash). A failed open/fsync is a TYPED refusal — never a silent promise
    (round-2 MEDIUM #5): a completion must not be confirmed if the rename's
    directory entry may not persist."""
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError as exc:
        raise ExternalArtifactPreservationError(
            f"cannot open attachments directory for fsync: {directory}: {exc}"
        ) from exc
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        raise ExternalArtifactPreservationError(
            f"cannot fsync attachments directory (durability not guaranteed): "
            f"{directory}: {exc}"
        ) from exc
    finally:
        os.close(dir_fd)


def discard_published_artifacts(
    published: list[CapturedExternalArtifact],
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Remove published copies whose binding never committed (txn rollback).

    Best-effort, like the scratch equivalents: a leaked copy is recoverable
    orphan debris, never a broken reference (no attachment row points at it).
    Empty per-task dirs are removed too.

    When *conn* is given, a copy whose ``stored_path`` already has a committed
    attachment row is KEPT (the transaction confirmed it — round-2 HIGH #3): a
    post-commit failure must never delete a bound proof. Without *conn* every
    listed copy is removed (used only where the rollback is certain); an
    unreadable DB with *conn* keeps everything rather than risk a bound copy.
    """
    committed: set[str] = set()
    if conn is not None:
        paths = [
            str(Path(cap.published_path).resolve())
            for cap in published if cap.published_path is not None
        ]
        if paths:
            try:
                placeholders = ",".join("?" * len(paths))
                committed = {
                    row[0] for row in conn.execute(
                        f"SELECT stored_path FROM task_attachments "
                        f"WHERE stored_path IN ({placeholders})",
                        paths,
                    ).fetchall()
                }
            except sqlite3.Error:
                return  # cannot prove which copies are unbound: keep them all
    for cap in published:
        if cap.published_path is None:
            continue
        path = Path(cap.published_path)
        if str(path.resolve()) in committed:
            continue
        _unlink_quietly(path)
        with contextlib.suppress(OSError):
            path.parent.rmdir()


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
