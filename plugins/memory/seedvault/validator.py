"""Commit gate and validation for SeedVault seeds.

Two-stage gate adapted from MindSeed's triple-metric gate concept,
but simplified for runtime extraction (no canonical reference corpus).

Stage 1 (deterministic, always runs):
  - core_claim non-empty, <= 500 chars
  - source_ref.session_id is not null (provenance required)
  - No duplicate core_claim (Jaccard >0.7 = reject)
  - Meristem targets resolve to existing seed IDs (dangling dropped)

Stage 2 (LLM-assisted, compression events only):
  - Re-extract claims from same messages, compare to committed seeds
  - If core_claim can't be traced to source messages: trust_score -= 0.2
  - Seeds below trust_score 0.3 auto-archived

PARADIGM NOTE: MindSeed's gate has a canonical reference to validate against.
SeedVault's Stage 2 is weaker — it checks internal consistency (can the claim
be re-derived from the source?) rather than external correctness. This is a
stated limitation, not an oversight. See DESIGN.md.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .vault import SeedVault, _tokenize, _jaccard

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CommitGate:
    """Two-stage validation gate for seed commits."""

    def __init__(self, vault: SeedVault):
        self.vault = vault

    def find_duplicate_artifact(self, blob_hash: str, seed_id: str = "") -> Optional[str]:
        """Find an existing seed with the same artifact blob hash.

        Exact hash lookup against the blob index — no Jaccard, no threshold.
        Returns the existing seed ID if the same blob hash is already in
        the vault AND belongs to a DIFFERENT seed, None otherwise.
        (Phase 8.3)

        The seed_id parameter excludes self-matches: when a blob was
        written at extraction time with the same seed_id, it's not a
        duplicate — it's the same seed's own artifact.

        Phase 9 orphan handling: when the index entry's owner seed no
        longer exists in the vault (deleted/never-committed), the entry is
        an orphan — it is re-pointed at the current caller (force_claim)
        and NOT treated as a duplicate. Without this, a dead seed would
        permanently block identical content from re-entering the vault.
        """
        if not blob_hash:
            return None
        meta = self.vault.blob_store.get_blob_metadata(blob_hash)
        if meta is not None:
            existing_seed_id = meta.get("seed_id", "")
            if existing_seed_id and existing_seed_id != seed_id:
                # Orphan check: does the owner seed still exist?
                if self.vault.get_seed(existing_seed_id) is None:
                    # Dead owner — hand the blob over to the caller
                    self.vault.blob_store.force_claim_blob(blob_hash, seed_id)
                    logger.info(
                        "SeedVault: blob %s re-pointed from dead seed %s to %s",
                        blob_hash[:12], existing_seed_id, seed_id,
                    )
                    return None
                return existing_seed_id
        return None

    # -- Lineage versioning (Phase 9) -----------------------------------------

    def check_lineage_noop(self, seed: Dict[str, Any]) -> bool:
        """Hash-check-first guard: is this an unchanged re-save? (Phase 9)

        When the seed carries a lineage_id, compare its blob hash to the
        CURRENT head of that lineage. If identical, this is a no-op save —
        no new version, no status change, no manifest bump.

        Returns True when the commit should be skipped as a no-op.
        """
        lineage_id = self._seed_lineage_id(seed)
        if not lineage_id:
            return False
        arts = seed.get("artifacts", [])
        if not arts:
            return False
        new_hash = arts[0].get("blob_hash", "")
        if not new_hash:
            return False

        candidates = self.vault.find_lineage_candidates(lineage_id, exclude_seed_id=seed.get("id", ""))
        # Only ACTIVE versions count as the lineage head — superseded ones
        # are history. (Multiple active heads shouldn't happen, but if the
        # chain was manipulated, comparing against any active version is
        # the conservative choice.)
        for sid in candidates:
            meta = self.vault._manifest.get("seeds", {}).get(sid, {})
            if meta.get("status") != "active":
                continue
            old_seed = self.vault.get_seed(sid)
            if not old_seed:
                continue
            for art in old_seed.get("artifacts", []):
                if art.get("blob_hash") == new_hash:
                    return True
        return False

    def _seed_lineage_id(self, seed: Dict[str, Any]) -> str:
        """Extract the lineage_id from a seed's artifacts (Phase 9)."""
        for art in seed.get("artifacts", []):
            lid = art.get("lineage_id", "")
            if lid:
                return lid
        return ""

    def _commit_lineage(self, seed: Dict[str, Any]) -> List[str]:
        """Run lineage-based supersession for an artifact seed (Phase 9).

        Finds active artifact seeds in the same lineage and supersedes them
        via the existing supersede_seeds() mechanism (triggered by lineage
        equality instead of primary-tag equality). Adds `supersedes`
        meristem edges, same as prose supersession.

        Returns the list of old seed IDs that were superseded.
        """
        lineage_id = self._seed_lineage_id(seed)
        if not lineage_id:
            return []
        candidates = self.vault.find_lineage_candidates(
            lineage_id, exclude_seed_id=seed.get("id", "")
        )
        # Filter to ACTIVE versions only — already-superseded versions stay
        # superseded (their superseded_by grows), matching prose semantics
        # where find_superseded_candidates includes superseded seeds but the
        # lineage chain should reflect the actual version history head.
        active = []
        for sid in candidates:
            meta = self.vault._manifest.get("seeds", {}).get(sid, {})
            if meta.get("status") == "active":
                active.append(sid)
        if not active:
            return []
        superseded = self.vault.supersede_by_lineage(
            seed["id"], lineage_id, old_seed_ids=active
        )
        for old_id in superseded:
            seed.setdefault("meristems", []).append({
                "type": "supersedes",
                "target": old_id,
            })
        return superseded

    def stage1_validate(self, seed: Dict[str, Any], skip_dedup: bool = False) -> tuple[bool, str]:
        """Deterministic validation. Returns (passed, reason).

        If failed, the seed is rejected — it does NOT land in the vault.

        When ``skip_dedup`` is True the Jaccard duplicate check is skipped.
        This is used by :meth:`commit` when supersession candidates exist
        (same primary tag), so that a legitimate "same domain, updated claim"
        seed is not rejected as a duplicate before supersession fires (S1).
        """
        # core_claim checks
        claim = seed.get("core_claim", "")
        if not claim or not claim.strip():
            return False, "core_claim is empty"
        if len(claim) > 500:
            return False, f"core_claim too long ({len(claim)} > 500)"

        # source_ref checks
        source_ref = seed.get("source_ref", {})
        if not source_ref.get("session_id"):
            return False, "source_ref.session_id is null (no provenance)"

        # Duplicate check — skipped when supersession applies (S1)
        if not skip_dedup:
            dup_id = self.vault.find_duplicate(claim, threshold=0.7)
            if dup_id is not None:
                return False, f"duplicate of existing seed {dup_id} (Jaccard >0.7)"

        # Meristem validation — drop dangling edges
        meristems = seed.get("meristems", [])
        validated = self.vault.validate_meristems(meristems)
        seed["meristems"] = validated

        return True, "passed"

    def commit(self, seed: Dict[str, Any]) -> tuple[bool, str]:
        """Run Stage 1 gate and commit if passed.
        
        Handles supersession: if an active seed with the same primary tag exists,
        the new seed supersedes it.
        
        For artifact seeds (seeds with non-empty ``artifacts`` array): uses
        exact-hash dedup via ``find_duplicate_artifact()`` instead of the
        Jaccard path.  The blob was already written at extraction time, so
        we check the hash against the index. (Phase 8.3)
        
        Returns (committed, reason).
        """
        # Phase 8.3: artifact seeds use exact-hash dedup, not Jaccard.
        artifacts = seed.get("artifacts", [])
        is_artifact_seed = bool(artifacts)
        candidates: list[str] = []  # supersession candidates (prose seeds only)

        if is_artifact_seed:
            # Phase 9: hash-check-first — an identical re-save to a lineage
            # head is a no-op (no new version, no status change, no manifest
            # bump). Per Roland's explicit direction in the Phase 9 spec.
            if self.check_lineage_noop(seed):
                logger.info(
                    "SeedVault: artifact seed %s is an unchanged re-save "
                    "(lineage head hash match) — no-op",
                    seed.get("id", "?"),
                )
                return False, "noop: identical to lineage head (hash match)"
            # Check for exact-hash duplicate
            for art in artifacts:
                blob_hash = art.get("blob_hash", "")
                dup_seed_id = self.find_duplicate_artifact(blob_hash, seed.get("id", ""))
                if dup_seed_id is not None:
                    return False, (
                        f"duplicate artifact (blob hash {blob_hash[:12]}... "
                        f"matches seed {dup_seed_id})"
                    )
            # Artifact seeds skip the Jaccard dedup path entirely.
            # They still pass the other Stage-1 checks (core_claim non-empty,
            # provenance, meristem validation).
            passed, reason = self.stage1_validate(seed, skip_dedup=True)
            if not passed:
                logger.warning("SeedVault: artifact seed rejected by Stage 1: %s", reason)
                return False, reason
        else:
            # Prose seeds: existing supersession + Jaccard path (unchanged)
            candidates = self.vault.find_superseded_candidates(seed)
            claim = seed.get("core_claim", "")
            skip_dedup = False
            if candidates:
                # Skip dedup only if no candidate is a near-duplicate.
                near_dup = self.vault.find_duplicate(claim, threshold=0.95)
                skip_dedup = near_dup is None

            passed, reason = self.stage1_validate(seed, skip_dedup=skip_dedup)
            if not passed:
                logger.warning("SeedVault: seed rejected by Stage 1 gate: %s", reason)
                return False, reason

        # Initialize trust score if not set
        if "trust_score" not in seed:
            seed["trust_score"] = 0.8
        if not seed.get("trust_history"):
            seed["trust_history"] = [{
                "delta": None,
                "reason": "initial",
                "value": seed["trust_score"],
                "at": _utc_now(),
            }]
        if seed.get("status") is None:
            seed["status"] = "active"
        if seed.get("superseded_by") is None:
            seed["superseded_by"] = []
        if seed.get("superseded_at") is None:
            seed["superseded_at"] = None
        if not seed.get("created"):
            seed["created"] = _utc_now()
        if not seed.get("updated"):
            seed["updated"] = _utc_now()

        # Supersession candidates were already identified before Stage-1.
        # Add supersedes meristems for each.
        if candidates:
            for old_id in candidates:
                seed.setdefault("meristems", []).append({
                    "type": "supersedes",
                    "target": old_id,
                })

        # Phase 9: lineage-based supersession for artifact seeds — point the
        # existing supersede_seeds() trigger at lineage equality. Runs BEFORE
        # write_seed so the new version's `supersedes` meristem edges are
        # persisted in the same file write.
        if is_artifact_seed:
            self._commit_lineage(seed)

        # Write seed
        if not self.vault.write_seed(seed):
            return False, "failed to write seed file"

        # Supersede old seeds
        if candidates:
            self.vault.supersede_seeds(seed["id"], candidates)

        logger.info("SeedVault: committed seed %s (tags=%s, trust=%.2f)",
                    seed["id"], seed.get("tags", []), seed["trust_score"])
        return True, "committed"

    def stage2_validate(self, seed: Dict[str, Any], source_messages: List[Dict[str, Any]]) -> bool:
        """LLM-assisted validation. Checks if core_claim can be traced to source.
        
        This is a lightweight check: tokenize the core_claim and check if its
        tokens appear in the source messages. If <30% of claim tokens appear in
        the source, it's flagged as potential drift.
        
        A full LLM re-extraction would be more robust but adds latency.
        This is the v1 approximation — see DESIGN.md TODO.
        
        Returns True if validated, False if drift detected.
        """
        claim_tokens = _tokenize(seed.get("core_claim", ""))
        if not claim_tokens:
            return False

        # Gather all text from source messages
        source_text = ""
        for msg in source_messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            source_text += " " + str(content)

        source_tokens = _tokenize(source_text)
        if not source_tokens:
            return False

        overlap = len(claim_tokens & source_tokens) / len(claim_tokens)
        if overlap < 0.3:
            # Drift detected
            now = _utc_now()
            old_score = seed.get("trust_score", 0.8)
            new_score = max(0.0, old_score - 0.2)
            seed["trust_score"] = new_score
            seed["trust_history"].append({
                "delta": -0.2,
                "reason": "drift_detected",
                "value": new_score,
                "at": now,
            })
            # B2: auto-archive if trust dropped below 0.3 threshold.
            # This prevents low-trust seeds from leaking into search/prefetch
            # (B3) — search() already excludes non-active status, so the
            # transition here closes the gap immediately rather than waiting
            # for the delayed prune() path (age > stale_days).
            if new_score < 0.3:
                seed["status"] = "archived"
                seed["trust_history"].append({
                    "delta": 0.0,
                    "reason": "auto_archived",
                    "value": new_score,
                    "at": now,
                })
                seed["last_validated"] = now
                self.vault.write_seed(seed)
                logger.warning(
                    "SeedVault: drift detected for seed %s (overlap=%.2f, "
                    "trust %.2f->%.2f) — auto-archived (below 0.3)",
                    seed["id"], overlap, old_score, new_score,
                )
                return False
            seed["last_validated"] = now
            self.vault.write_seed(seed)
            logger.warning("SeedVault: drift detected for seed %s (overlap=%.2f, trust %.2f->%.2f)",
                          seed["id"], overlap, old_score, new_score)
            return False

        # Validated — bump trust
        now = _utc_now()
        old_score = seed.get("trust_score", 0.8)
        new_score = min(1.0, old_score + 0.1)
        seed["trust_score"] = new_score
        seed["trust_history"].append({
            "delta": 0.1,
            "reason": "provenance_verified",
            "value": new_score,
            "at": now,
        })
        seed["last_validated"] = now
        self.vault.write_seed(seed)
        logger.debug("SeedVault: seed %s validated (overlap=%.2f, trust %.2f->%.2f)",
                    seed["id"], overlap, old_score, new_score)
        return True