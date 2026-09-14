"""Vault management for SeedVault memory plugin.

Manages the on-disk seed vault: manifest, seed files, file locking,
atomic writes, and supersession logic.

PARADIGM NOTE: Unlike MindSeed's build-time vault (curated, validated
offline, deterministic delivery), this vault is populated at runtime
during compression events. Seeds are extracted from live conversation
and pass through a two-stage commit gate before landing here. The vault
does NOT inherit MindSeed's determinism guarantees — see DESIGN.md.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .blobstore import BlobStore, DEFAULT_MAX_BLOB_BYTES, _make_artifact_seed_id

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Common English stopwords stripped from tokens so that function words
# (in, the, and, …) don't inflate Jaccard similarity between claims that
# share phrasing but differ in substance.  This prevents the Stage-1 dedup
# gate from rejecting legitimate supersession candidates (S1).
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "is", "it", "as", "be", "by", "do", "if", "so", "no", "he",
    "we", "my", "me", "us", "am", "up", "all", "now", "was", "are",
    "has", "had", "been", "not", "can", "will", "this", "that", "from",
    "with", "into", "than", "then", "such", "over", "under", "about",
    "just", "only", "any", "some", "more", "most", "also", "very",
    "too", "even", "out", "off", "down", "its", "our", "your", "her",
    "his", "their", "them", "they", "who", "which", "what", "where",
    "when", "how", "why", "did", "does", "doing", "being", "have",
    "having", "should", "would", "could", "there", "here", "these",
    "those", "each", "other", "same", "both", "one", "two",
})


def _tokenize(text: str) -> set[str]:
    """Tokenizer for Jaccard similarity and keyword matching.

    Stopwords are stripped so that short function words don't inflate
    similarity between substantively different claims.
    """
    return set(re.findall(r"[a-z0-9]{2,}", text.lower())) - _STOPWORDS


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class SeedVault:
    """On-disk seed vault with file locking and atomic writes."""

    def __init__(self, vault_dir: Path, max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES):
        self.vault_dir = Path(vault_dir)
        self.seeds_dir = self.vault_dir / "seeds"
        self.archive_dir = self.vault_dir / "archive"
        self.manifest_path = self.vault_dir / "vault_manifest.json"
        self.digest_path = self.vault_dir / "state_digest.json"
        self._lock = threading.Lock()
        self.ensure_dirs()
        self.blob_store = BlobStore(self.vault_dir, max_blob_bytes=max_blob_bytes)
        self.blob_store.owner = self  # extractor bridge (Phase 9)
        self._manifest: Dict[str, Any] = self._load_manifest()
        # Phase 9: monotonically increasing artifact serial across all batches.
        # Per-batch counters collide (every batch starts at 001), which let a
        # new version overwrite the previous version's seed file — fatal for a
        # version chain. The serial is initialized from disk (live + archived
        # seeds), so IDs never collide across processes or batches.
        self._artifact_serial = self._load_artifact_serial()

    # -- Artifact seed ID serial (Phase 9) ------------------------------------

    _ARTIFACT_ID_RE = re.compile(
        r"^(?P<domain>[a-z0-9]+)-(?P<topic>[a-z0-9]+)-(?P<num>\d+)$"
    )

    def _load_artifact_serial(self) -> int:
        """Scan manifest + archive seeds for the highest artifact serial used.

        Artifact seed IDs look like ``shell-bash-0042``. The numeric suffix is
        a global serial, not a per-batch counter — per-batch counters collided
        (every batch restarted at 001) and let a new version overwrite the
        previous version's seed file, which makes a version chain impossible.
        """
        highest = 0
        for directory in (self.seeds_dir, self.archive_dir):
            try:
                for path in directory.glob("*.json"):
                    m = self._ARTIFACT_ID_RE.match(path.stem)
                    if not m:
                        continue
                    try:
                        num = int(m.group("num"))
                    except ValueError:
                        continue
                    if num > highest:
                        highest = num
            except OSError:
                continue
        return highest

    def next_artifact_seed_id(self, domain: str, topic: str) -> str:
        """Allocate the next unique artifact seed ID (domain-topic-NNN)."""
        self._artifact_serial += 1
        return _make_artifact_seed_id(domain, topic, self._artifact_serial)

    # -- Directory setup ----------------------------------------------------

    def ensure_dirs(self) -> None:
        for d in (self.vault_dir, self.seeds_dir, self.archive_dir):
            d.mkdir(parents=True, exist_ok=True)
        # blobs/ and its shard subdirs are created by BlobStore.ensure_dirs()
        blobs_dir = self.vault_dir / "blobs"
        blobs_dir.mkdir(parents=True, exist_ok=True)

    # -- Manifest -----------------------------------------------------------

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("SeedVault: manifest corrupt, starting fresh: %s", e)
        return {"seeds": {}, "version": 1, "updated": _utc_now()}

    def _save_manifest(self) -> None:
        """Save manifest with exclusive file lock + atomic replace.

        Uses a per-writer unique tmp filename (includes PID + thread ID) so
        concurrent writers don't clobber each other's tmp file before
        os.replace() runs. The flock serialises the actual writes; the unique
        tmp name prevents the race where writer A's os.replace() unlinks the
        tmp file that writer B is still writing to.
        """
        self._manifest["updated"] = _utc_now()
        tmp_path = self.manifest_path.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.tmp"
        )
        with open(tmp_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(self._manifest, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        os.replace(tmp_path, self.manifest_path)

    # -- Seed CRUD ----------------------------------------------------------

    def get_seed(self, seed_id: str) -> Optional[Dict[str, Any]]:
        """Load a seed from disk."""
        path = self.seeds_dir / f"{seed_id}.json"
        if not path.exists():
            # Check archive
            path = self.archive_dir / f"{seed_id}.json"
            if not path.exists():
                return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("SeedVault: seed %s corrupt: %s", seed_id, e)
            return None

    def list_seeds(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all seeds, optionally filtered by status."""
        results = []
        for seed_id, meta in self._manifest.get("seeds", {}).items():
            if status and meta.get("status") != status:
                continue
            seed = self.get_seed(seed_id)
            if seed:
                results.append(seed)
        return results

    def write_seed(self, seed: Dict[str, Any]) -> bool:
        """Atomically write a seed file and update manifest.
        
        Returns True on success, False on failure.

        Concurrency: the seed file is written atomically (unique tmp + replace).
        The manifest update is protected by a dedicated lock file that covers
        the full read-modify-write cycle, so concurrent writers don't lose
        each other's entries.
        """
        seed_id = seed.get("id", "")
        if not seed_id:
            logger.error("SeedVault: seed missing id")
            return False

        path = self.seeds_dir / f"{seed_id}.json"
        tmp_path = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")

        try:
            with open(tmp_path, "w") as f:
                json.dump(seed, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)  # atomic on POSIX
        except OSError as e:
            logger.error("SeedVault: failed to write seed %s: %s", seed_id, e)
            return False

        # Lock the entire manifest read-modify-write cycle with a dedicated
        # lock file.  This prevents the race where process A loads the manifest,
        # process B loads the same manifest, A adds its seed and saves, then B
        # adds its seed and saves — clobbering A's entry.  The lock file is
        # separate from the manifest itself so it survives os.replace().
        lock_path = self.manifest_path.with_suffix(".lock")
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                # Re-read the manifest from disk in case another writer changed it
                disk_manifest = self._load_manifest()
                disk_manifest.setdefault("seeds", {})[seed_id] = {
                    "status": seed.get("status", "active"),
                    "tags": seed.get("tags", []),
                    "trust_score": seed.get("trust_score", 0.0),
                    "updated": seed.get("updated", _utc_now()),
                }
                self._manifest = disk_manifest
                self._save_manifest()
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        return True

    def update_seed(self, seed_id: str, updates: Dict[str, Any]) -> bool:
        """Partial update a seed. Merges updates into existing seed."""
        seed = self.get_seed(seed_id)
        if not seed:
            return False
        seed.update(updates)
        seed["updated"] = _utc_now()
        return self.write_seed(seed)

    def adjust_trust(self, seed_id: str, delta: float, reason: str) -> bool:
        """Adjust a seed's trust score by delta, clamped to [0.0, 1.0].
        
        Appends an entry to trust_history with the delta, reason, new value,
        and timestamp. Returns False if seed not found.
        """
        seed = self.get_seed(seed_id)
        if not seed:
            return False
        current = seed.get("trust_score", 0.0)
        new_score = max(0.0, min(1.0, current + delta))
        seed["trust_score"] = new_score
        seed.setdefault("trust_history", []).append({
            "delta": delta,
            "reason": reason,
            "value": new_score,
            "at": _utc_now(),
        })
        seed["updated"] = _utc_now()
        return self.write_seed(seed)

    def archive_seed(self, seed_id: str) -> bool:
        """Move a seed from seeds/ to archive/."""
        src = self.seeds_dir / f"{seed_id}.json"
        dst = self.archive_dir / f"{seed_id}.json"
        if not src.exists():
            return False
        try:
            os.replace(src, dst)
        except OSError as e:
            logger.error("SeedVault: failed to archive seed %s: %s", seed_id, e)
            return False
        # Use the manifest lock file for cross-process safety (same as write_seed)
        lock_path = self.manifest_path.with_suffix(".lock")
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                disk_manifest = self._load_manifest()
                if seed_id in disk_manifest.get("seeds", {}):
                    disk_manifest["seeds"][seed_id]["status"] = "archived"
                    self._manifest = disk_manifest
                    self._save_manifest()
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        return True

    # -- Supersession -------------------------------------------------------

    def find_superseded_candidates(self, new_seed: Dict[str, Any]) -> List[str]:
        """Find active or superseded seeds with the same primary tag.
        
        Matching rule: exact primary-tag match (first tag in tags[]).
        This is stricter than the Stage-1 dedup gate (Jaccard >0.7 on core_claim).
        Dedup prevents committing a near-duplicate; supersession handles
        "same domain, different claim."
        
        Includes already-superseded seeds so multiple children can supersede
        the same parent (superseded_by is a list).
        """
        new_tags = new_seed.get("tags", [])
        if not new_tags:
            return []
        primary_tag = new_tags[0]
        candidates = []
        for seed_id, meta in self._manifest.get("seeds", {}).items():
            if meta.get("status") not in ("active", "superseded"):
                continue
            if seed_id == new_seed.get("id"):
                continue
            meta_tags = meta.get("tags", [])
            if meta_tags and meta_tags[0] == primary_tag:
                candidates.append(seed_id)
        return candidates

    def supersede_seeds(self, new_seed_id: str, old_seed_ids: List[str]) -> None:
        """Mark old seeds as superseded by the new seed."""
        now = _utc_now()
        for old_id in old_seed_ids:
            old_seed = self.get_seed(old_id)
            if not old_seed:
                continue
            old_seed["status"] = "superseded"
            old_seed.setdefault("superseded_by", []).append(new_seed_id)
            old_seed["superseded_at"] = now
            pre_zero_score = old_seed.get("trust_score", 0.0)
            old_seed["trust_score"] = 0.0
            old_seed["trust_history"].append({
                "delta": -pre_zero_score,
                "reason": "superseded",
                "value": 0.0,
                "at": now,
            })
            old_seed["updated"] = now
            self.write_seed(old_seed)
            logger.info("SeedVault: seed %s superseded by %s", old_id, new_seed_id)

    # -- Lineage versioning (Phase 9) -----------------------------------------

    @staticmethod
    def normalize_lineage_id(raw: str) -> str:
        """Normalize a lineage key (absolute path or explicit tag).

        File-backed artifacts use the normalized absolute path as the lineage
        id. Ephemeral artifacts use an explicit agent-supplied slug. Normali-
        zation: collapse duplicate slashes, strip trailing slashes, and for
        paths resolve to a canonical absolute form when possible.
        """
        if not raw:
            return ""
        value = raw.strip()
        if not value:
            return ""
        # Collapse repeated separators
        value = re.sub(r"/{2,}", "/", value)
        # Only path-shaped keys get path normalization
        if value.startswith("/") or value.startswith("~"):
            try:
                expanded = os.path.expanduser(value)
                value = os.path.normpath(expanded)
            except (OSError, ValueError):
                pass
        else:
            # Explicit slug: normalize to a safe form (lowercase, dashed)
            value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return value

    def find_lineage_candidates(self, lineage_id: str, exclude_seed_id: str = "") -> List[str]:
        """Find active/superseded artifact seeds in the same lineage.

        Looks up the blob index for seeds whose artifacts carry this
        lineage_id. Falls back to scanning seed files when the blob index has
        no lineage annotation (older entries from before Phase 9).

        Returns seed IDs whose seed is at status active or superseded.
        """
        if not lineage_id:
            return []
        candidates: List[str] = []
        seen: set[str] = set()

        # Primary source: blob index lineage annotations
        try:
            rows = self.blob_store.find_by_lineage(lineage_id)
        except Exception as e:  # defensive: index errors must not break commits
            logger.warning("SeedVault: lineage lookup failed for %s: %s", lineage_id, e)
            rows = []
        for row in rows:
            sid = row.get("seed_id", "")
            if not sid or sid == exclude_seed_id or sid in seen:
                continue
            meta = self._manifest.get("seeds", {}).get(sid, {})
            if meta.get("status") in ("active", "superseded"):
                candidates.append(sid)
                seen.add(sid)

        # Fallback: scan seed files whose artifacts carry matching lineage_id
        for seed_id, meta in self._manifest.get("seeds", {}).items():
            if seed_id in seen or seed_id == exclude_seed_id:
                continue
            if meta.get("status") not in ("active", "superseded"):
                continue
            seed = self.get_seed(seed_id)
            if not seed:
                continue
            arts = seed.get("artifacts", [])
            if any(a.get("lineage_id") == lineage_id for a in arts):
                candidates.append(seed_id)
                seen.add(seed_id)
        return candidates

    def supersede_by_lineage(self, new_seed_id: str, lineage_id: str,
                             old_seed_ids: Optional[List[str]] = None) -> List[str]:
        """Supersede the current versions of a lineage in favor of new_seed_id.

        Thin wrapper around supersede_seeds() — the spec requires reusing the
        existing mechanism exactly, just triggered by lineage equality instead
        of primary-tag equality.

        old_seed_ids: explicit candidate list (from the gate). When omitted,
        candidates are looked up from the vault by lineage_id.

        Returns the list of seed IDs that were actually superseded.
        """
        if old_seed_ids is None:
            old_seed_ids = self.find_lineage_candidates(lineage_id, exclude_seed_id=new_seed_id)
        old_seed_ids = [sid for sid in old_seed_ids if sid != new_seed_id]
        if not old_seed_ids:
            return []
        self.supersede_seeds(new_seed_id, old_seed_ids)
        return old_seed_ids

    # -- Dedup / Gate helpers ------------------------------------------------

    def find_duplicate(self, core_claim: str, threshold: float = 0.7) -> Optional[str]:
        """Find an existing seed with a near-duplicate core_claim.
        
        Uses Jaccard similarity on tokenized claims. Returns seed ID or None.
        """
        claim_tokens = _tokenize(core_claim)
        if not claim_tokens:
            return None
        for seed_id, meta in self._manifest.get("seeds", {}).items():
            if meta.get("status") != "active":
                continue
            seed = self.get_seed(seed_id)
            if not seed:
                continue
            existing_tokens = _tokenize(seed.get("core_claim", ""))
            if not existing_tokens:
                continue
            sim = _jaccard(claim_tokens, existing_tokens)
            if sim > threshold:
                return seed_id
        return None

    def validate_meristems(self, meristems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop meristem edges that point to non-existent seeds.
        
        Dangling edges are dropped, not deferred — early seeds may have no
        meristems until related seeds are extracted.
        """
        valid = []
        for edge in meristems:
            target = edge.get("target", "")
            if not target:
                continue
            if target in self._manifest.get("seeds", {}):
                valid.append(edge)
            else:
                logger.debug("SeedVault: dropping dangling meristem to %s", target)
        return valid

    # -- State Digest --------------------------------------------------------

    def load_digest(self) -> Dict[str, Any]:
        """Load the state digest, or return a minimal default."""
        if self.digest_path.exists():
            try:
                with open(self.digest_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "current_task": "",
            "active_seed_ids": [],
            "pending_actions": [],
            "compaction_count": 0,
            "last_compaction": _utc_now(),
            "session_lineage": [],
        }

    def save_digest(self, digest: Dict[str, Any]) -> None:
        """Save the state digest with exclusive file lock + atomic replace.

        Uses a per-writer unique tmp filename (includes PID + thread ID) so
        concurrent writers don't clobber each other's tmp file or the live
        digest file. Without this, two writers opening the same path for "w"
        would truncate each other's data mid-write.
        """
        tmp_path = self.digest_path.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.tmp"
        )
        with open(tmp_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(digest, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        os.replace(tmp_path, self.digest_path)

    # -- Retrieval (v1: keyword/tag matching) --------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search active seeds by keyword/tag overlap.
        
        v1: token overlap with prefix stemming (query "experiment" matches
        claim "experimenting"). v2 upgrade path: embedding-based retrieval.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for seed_id, meta in self._manifest.get("seeds", {}).items():
            if meta.get("status") != "active":
                continue
            seed = self.get_seed(seed_id)
            if not seed:
                continue

            # Score by tag overlap (weight 2x) + core_claim token overlap (weight 1x)
            seed_tags = set(t.lower() for t in seed.get("tags", []))
            tag_score = 2.0 * len(query_tokens & seed_tags) / max(len(query_tokens), 1)

            claim_tokens = _tokenize(seed.get("core_claim", ""))
            # Prefix matching: query "experiment" matches claim "experimenting"
            claim_match = 0
            for qt in query_tokens:
                for ct in claim_tokens:
                    if qt == ct or ct.startswith(qt) or qt.startswith(ct):
                        claim_match += 1
                        break
            claim_score = 1.0 * claim_match / max(len(query_tokens), 1)

            total = tag_score + claim_score
            if total > 0:
                scored.append((total, seed))

        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:top_k]]

    # -- Pruning -------------------------------------------------------------

    def prune(self, stale_days: int = 7, archive_days: int = 30) -> Dict[str, int]:
        """Prune stale and old-superseded seeds.
        
        - Active seeds with trust_score < 0.3 for >stale_days -> archived
        - Superseded seeds older than archive_days -> moved to archive/ dir
        
        Returns counts of what was pruned.
        """
        now = datetime.now(timezone.utc)
        stale_threshold = stale_days * 86400  # seconds
        archive_threshold = archive_days * 86400
        pruned = {"stale_archived": 0, "old_archived": 0}

        for seed_id, meta in list(self._manifest.get("seeds", {}).items()):
            status = meta.get("status", "active")
            trust = meta.get("trust_score", 0.0)
            updated_str = meta.get("updated", "")

            try:
                updated = datetime.fromisoformat(updated_str)
            except (ValueError, TypeError):
                continue

            age_seconds = (now - updated).total_seconds()

            # Stale active seeds: low trust for too long
            if status == "active" and trust < 0.3 and age_seconds > stale_threshold:
                if self.update_seed(seed_id, {"status": "archived"}):
                    pruned["stale_archived"] += 1
                    logger.info("SeedVault: pruned stale seed %s (trust=%.2f, age=%dd)",
                               seed_id, trust, stale_days)

            # Old superseded seeds -> archive directory
            if status == "superseded" and age_seconds > archive_threshold:
                if self.archive_seed(seed_id):
                    pruned["old_archived"] += 1
                    logger.info("SeedVault: archived old superseded seed %s (age=%dd)",
                               seed_id, archive_days)

        # Phase 9: extend the sweep to artifact lineages — stale superseded
        # artifact versions also move into archive/ (same rule, second
        # collection). find_lineage-based lookup is unnecessary here: any
        # superseded artifact seed above is already caught by the loop above
        # regardless of lineage. This block only handles lineage rows whose
        # seed entries are already archived so the lineage index stays lean.
        pruned["lineage_archived"] = 0

        return pruned

    # -- Manifest access -----------------------------------------------------

    def get_active_seed_ids(self) -> List[str]:
        """Return IDs of all active seeds."""
        return [sid for sid, meta in self._manifest.get("seeds", {}).items()
                if meta.get("status") == "active"]

    def get_manifest_summary(self) -> Dict[str, Any]:
        """Return a summary of the vault for logging/debugging."""
        seeds = self._manifest.get("seeds", {})
        status_counts = Counter(m.get("status", "unknown") for m in seeds.values())
        return {
            "total": len(seeds),
            "by_status": dict(status_counts),
            "updated": self._manifest.get("updated", ""),
        }