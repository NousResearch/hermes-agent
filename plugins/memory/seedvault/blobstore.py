"""Content-addressed blob store for SeedVault artifacts.

Stores verbatim artifact bytes (code blocks, shell commands, diffs) as
content-addressed files on disk, with a SQLite index for metadata lookups.

Design (per Phase 8 redline):
- Raw bytes: <vault_dir>/blobs/<sha256[:2]>/<sha256>.blob (sharded by first
  2 hex chars to keep directory entries manageable).
- Index: <vault_dir>/blobs/blob_index.db (SQLite, WAL mode).  Holds
  metadata only — blob_hash, seed_id, content_type, byte_length, created_at.
  The schema is designed to be extensible for Phase 9 lineage fields
  (lineage_id, tag columns can be added without migration).
- vault_manifest.json is NOT touched — the blob index is separate.

Concurrency:
- SQLite WAL mode handles concurrent index reads/writes with its own
  transaction/locking model — no fcntl.flock needed for the index.
- Blob file writes use the same atomic pattern as SeedVault seeds: unique
  tmp filename (pid+tid suffix) + os.replace().
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _make_artifact_seed_id(domain: str, topic: str, index: int) -> str:
    """Generate an artifact seed ID in domain-topic-NNN format.

    index is a global serial allocated by SeedVault.next_artifact_seed_id()
    (not a per-batch counter — per-batch counters collided across batches
    and let version N+1 overwrite version N's seed file).
    """
    domain = re.sub(r"[^a-z0-9]", "", domain.lower())[:20] or "artifact"
    topic = re.sub(r"[^a-z0-9]", "", topic.lower())[:20] or "general"
    return f"{domain}-{topic}-{index:06d}"


DEFAULT_MAX_BLOB_BYTES = 1 * 1024 * 1024  # 1 MB hard limit


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BlobStore:
    """Content-addressed blob store with SQLite WAL index.

    Instantiated by SeedVault via composition.  The blob index lives at
    ``<vault_dir>/blobs/blob_index.db`` (SQLite, WAL mode).  Raw bytes live
    as ``<vault_dir>/blobs/<sha256[:2]>/<sha256>.blob``.
    """

    def __init__(
        self,
        vault_dir: Path,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
    ) -> None:
        self.vault_dir = Path(vault_dir)
        self.blobs_dir = self.vault_dir / "blobs"
        self.db_path = self.blobs_dir / "blob_index.db"
        self.max_blob_bytes = max_blob_bytes
        self._local = threading.local()  # per-thread SQLite connections
        # Back-reference to the owning SeedVault, set by SeedVault.__init__.
        # The extractor reaches vault-level services (global artifact seed-ID
        # allocator, lineage normalizer) through this bridge — the provider
        # only passes blob_store into extract_seeds().
        self.owner: Optional[Any] = None
        self.ensure_dirs()

    # -- Directory + DB setup -----------------------------------------------

    def ensure_dirs(self) -> None:
        """Create blobs/ and all 256 shard subdirectories (lazy via mkdir)."""
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

    # -- Vault-service bridge (Phase 9) --------------------------------------

    def allocate_artifact_seed_id(self, domain: str, topic: str) -> str:
        """Delegate to the owning vault's global artifact serial allocator.

        Raises RuntimeError when BlobStore is used standalone (no owner) —
        the caller should fall back to batch-local numbering in that case.
        """
        if self.owner is None:
            raise RuntimeError(
                "BlobStore: allocate_artifact_seed_id requires an owning "
                "SeedVault (owner is None)"
            )
        return self.owner.next_artifact_seed_id(domain, topic)

    def normalize_lineage_id(self, raw: str) -> str:
        """Delegate to the owning vault's lineage normalizer.

        Falls back to the static normalizer when standalone — normalization
        is stateless, so the static path is always safe.
        """
        if self.owner is not None:
            return self.owner.normalize_lineage_id(raw)
        from .vault import SeedVault as _SV
        return _SV.normalize_lineage_id(raw)

    def _get_conn(self) -> sqlite3.Connection:
        """Get a per-thread SQLite connection in WAL mode."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.execute("SELECT 1")
                return conn
            except sqlite3.Error:
                # Stale connection — close and reopen
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
                del self._local.conn

        conn = sqlite3.connect(
            str(self.db_path),
            isolation_level=None,  # autocommit mode; we manage txns explicitly
            timeout=30.0,  # wait up to 30s for locks
        )
        conn.row_factory = sqlite3.Row
        # WAL mode for concurrent readers + single writer
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")  # safe with WAL
        conn.execute("PRAGMA busy_timeout=30000")  # 30s busy timeout

        # Create table if not exists — extensible schema for Phase 9
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blob_index (
                blob_hash    TEXT    PRIMARY KEY,
                seed_id      TEXT    NOT NULL,
                content_type TEXT    NOT NULL,
                byte_length  INTEGER NOT NULL,
                created_at   TEXT    NOT NULL,
                language     TEXT    DEFAULT NULL,
                lineage_id   TEXT    DEFAULT NULL,
                tag          TEXT    DEFAULT NULL
            )
            """
        )
        # Index for seed_id lookups (used by retrieval/dedup)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_blob_seed_id ON blob_index(seed_id)"
        )
        # Index for content_type lookups (future filtering)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_blob_content_type "
            "ON blob_index(content_type)"
        )

        self._local.conn = conn
        return conn

    # -- Core CRUD -----------------------------------------------------------

    def write_blob(
        self,
        raw_bytes: bytes,
        seed_id: str,
        content_type: str,
        language: Optional[str] = None,
        lineage_id: Optional[str] = None,
        lineage_tag: Optional[str] = None,
    ) -> Optional[str]:
        """Write artifact bytes to disk and index in SQLite.

        Returns the SHA-256 blob_hash on success, or None if:
        - The blob exceeds max_blob_bytes (oversized artifacts are dropped).
        - A disk/DB error occurs.

        Dedup: if the exact hash already exists in the index, the blob file
        is not re-written (it's already on disk) and the existing hash is
        returned unchanged (first writer owns the index entry; the gate's
        orphan-handoff logic decides ownership when the old owner is dead).
        lineage_id / lineage_tag are recorded for Phase 9 version chaining.
        """
        if not raw_bytes:
            return None

        if len(raw_bytes) > self.max_blob_bytes:
            logger.warning(
                "BlobStore: blob rejected — %d bytes exceeds limit %d "
                "(seed_id=%s, content_type=%s)",
                len(raw_bytes),
                self.max_blob_bytes,
                seed_id,
                content_type,
            )
            return None

        blob_hash = hashlib.sha256(raw_bytes).hexdigest()

        # Check if this hash already exists (dedup)
        if self.blob_exists(blob_hash):
            # Phase 9 orphan hand-off: if the indexed seed_id no longer
            # exists as a seed, re-point the index entry at the new caller.
            # Without this, a gate-rejected/deleted seed would hold the hash
            # forever and identical content could never re-enter the vault.
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT seed_id FROM blob_index WHERE blob_hash = ? LIMIT 1",
                    (blob_hash,),
                ).fetchone()
                if row is not None:
                    old_owner = row["seed_id"] or ""
                    if old_owner and old_owner != seed_id:
                        # Ownership check is delegated to the caller via
                        # claim_blob_if_orphan() — write_blob never deletes.
                        logger.debug(
                            "BlobStore: dedup hit for hash %s (old owner=%s, new=%s)",
                            blob_hash, old_owner, seed_id,
                        )
            except sqlite3.Error as e:
                logger.warning("BlobStore: dedup lookup failed: %s", e)
            return blob_hash

        # Write the blob file atomically
        shard_dir = self.blobs_dir / blob_hash[:2]
        shard_dir.mkdir(parents=True, exist_ok=True)
        blob_path = shard_dir / f"{blob_hash}.blob"
        tmp_path = blob_path.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.tmp"
        )

        try:
            with open(tmp_path, "wb") as f:
                f.write(raw_bytes)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, blob_path)
        except OSError as e:
            logger.error("BlobStore: failed to write blob %s: %s", blob_hash, e)
            # Clean up tmp if it's still around
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

        # Index the blob in SQLite
        now = _utc_now()
        conn = self._get_conn()
        try:
            conn.execute(
                "BEGIN IMMEDIATE"
            )  # get write lock immediately for the insert
            conn.execute(
                """
                INSERT OR IGNORE INTO blob_index
                    (blob_hash, seed_id, content_type, byte_length, created_at,
                     language, lineage_id, tag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (blob_hash, seed_id, content_type, len(raw_bytes), now, language,
                 lineage_id, lineage_tag),
            )
            conn.execute("COMMIT")
        except sqlite3.Error as e:
            logger.error("BlobStore: failed to index blob %s: %s", blob_hash, e)
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            # The blob file is on disk but not indexed — it's orphaned.
            # We leave it (it will be found by future writes with the same hash
            # and indexed then).  Return None so the caller knows the index
            # write failed.
            return None

        logger.debug(
            "BlobStore: wrote blob %s (%d bytes, type=%s, seed=%s, lineage=%s)",
            blob_hash,
            len(raw_bytes),
            content_type,
            seed_id,
            lineage_id or lineage_tag or "-",
        )
        return blob_hash

    def claim_blob(self, blob_hash: str, new_seed_id: str,
                   new_lineage_id: Optional[str] = None,
                   new_lineage_tag: Optional[str] = None) -> bool:
        """Re-point an orphaned blob index entry at a new seed (Phase 9).

        An orphan is an index entry whose seed_id no longer exists as a seed
        file in the vault (deleted, superseded-and-pruned, or gate-rejected).
        Orphans would otherwise permanently block identical content from
        being committed again (exact-hash dedup matches the dead owner).

        Only re-points the owner — never overwrites a live owner's entry.

        Returns True if ownership was claimed (or already owned by
        new_seed_id), False when a live different owner holds the blob.
        """
        if not blob_hash:
            return False
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT seed_id FROM blob_index WHERE blob_hash = ? LIMIT 1",
                (blob_hash,),
            ).fetchone()
            if row is None:
                # Blob not indexed (orphan file from a failed index write).
                # Nothing to claim — caller will insert a fresh row.
                return True
            owner = row["seed_id"] or ""
            if not owner or owner == new_seed_id:
                # Unowned or already ours — claim/update annotation
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    UPDATE blob_index
                    SET seed_id = ?, lineage_id = COALESCE(?, lineage_id),
                        tag = COALESCE(?, tag)
                    WHERE blob_hash = ?
                    """,
                    (new_seed_id, new_lineage_id, new_lineage_tag, blob_hash),
                )
                conn.execute("COMMIT")
                return True
            # Owner exists in the index; the caller must verify the owner
            # seed still exists. BlobStore has no vault access — return the
            # decision to the caller via a False-with-info contract:
            # claim is only safe when the caller KNOWS the owner is dead.
            return False
        except sqlite3.Error as e:
            logger.error("BlobStore: claim_blob failed for %s: %s", blob_hash, e)
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            return False

    def force_claim_blob(self, blob_hash: str, new_seed_id: str,
                         new_lineage_id: Optional[str] = None,
                         new_lineage_tag: Optional[str] = None) -> bool:
        """Unconditionally re-point a blob index entry at a new seed.

        Only call this when the CURRENT owner seed is confirmed dead (seed
        file no longer in the vault). The gate does that check via
        vault.get_seed() before calling.
        """
        if not blob_hash:
            return False
        conn = self._get_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE blob_index
                SET seed_id = ?, lineage_id = COALESCE(?, lineage_id),
                    tag = COALESCE(?, tag)
                WHERE blob_hash = ?
                """,
                (new_seed_id, new_lineage_id, new_lineage_tag, blob_hash),
            )
            conn.execute("COMMIT")
            return True
        except sqlite3.Error as e:
            logger.error("BlobStore: force_claim_blob failed for %s: %s", blob_hash, e)
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            return False

    def find_by_lineage(self, lineage_id: str) -> List[Dict[str, Any]]:
        """Find all blob index rows in a lineage (Phase 9).

        Matches either the path-based lineage_id or the explicit tag column.
        """
        if not lineage_id:
            return []
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM blob_index
            WHERE lineage_id = ? OR tag = ?
            ORDER BY created_at ASC
            """,
            (lineage_id, lineage_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def read_blob(self, blob_hash: str) -> Optional[bytes]:
        """Read raw blob bytes from disk by SHA-256 hash.

        Returns None if the blob file doesn't exist on disk.
        """
        if not blob_hash:
            return None
        shard_dir = self.blobs_dir / blob_hash[:2]
        blob_path = shard_dir / f"{blob_hash}.blob"
        if not blob_path.exists():
            logger.warning("BlobStore: blob not found on disk: %s", blob_hash)
            return None
        try:
            with open(blob_path, "rb") as f:
                return f.read()
        except OSError as e:
            logger.error("BlobStore: failed to read blob %s: %s", blob_hash, e)
            return None

    def blob_exists(self, blob_hash: str) -> bool:
        """Check if a blob hash exists in the SQLite index.

        This is an exact-match lookup — no Jaccard, no threshold.
        """
        if not blob_hash:
            return False
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM blob_index WHERE blob_hash = ? LIMIT 1",
            (blob_hash,),
        ).fetchone()
        return row is not None

    def get_blob_metadata(self, blob_hash: str) -> Optional[Dict[str, Any]]:
        """Get blob metadata from the SQLite index.

        Returns a dict with keys: blob_hash, seed_id, content_type,
        byte_length, created_at, language, lineage_id, tag.
        """
        if not blob_hash:
            return None
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM blob_index WHERE blob_hash = ? LIMIT 1",
            (blob_hash,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def find_by_seed_id(self, seed_id: str) -> list[Dict[str, Any]]:
        """Find all blobs associated with a seed ID."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM blob_index WHERE seed_id = ?",
            (seed_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        """Total number of blobs in the index."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM blob_index").fetchone()
        return row[0] if row else 0

    def all_hashes(self) -> list[str]:
        """Return all blob hashes in the index (for integrity checks)."""
        conn = self._get_conn()
        rows = conn.execute("SELECT blob_hash FROM blob_index").fetchall()
        return [r["blob_hash"] for r in rows]

    def close(self) -> None:
        """Close the per-thread SQLite connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
            del self._local.conn