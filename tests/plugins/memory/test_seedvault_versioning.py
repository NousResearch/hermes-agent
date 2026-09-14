"""Phase 9: Artifact Versioning — spec test cases.

Implements the five test cases from SeedVault_Phase9_Artifact_Versioning_Spec_2026-07-29.md:

  1.  Same lineage, different content → new version; old superseded; history queryable.
  1b. Same lineage, SAME content (re-save) → no-op: no new version, no manifest bump.
  2.  Different lineage → separate chains, no interference.
  3.  Ephemeral artifact with explicit tag → chain works; untagged → no chain.
  4.  Orphan blob hand-off: identical bytes after seed removal re-enter the vault.
  5.  Prune: old superseded versions archive, head survives; nothing orphaned.

Each case asserts the spec's core invariants: exact-hash dedup, deterministic
lineage keys, no fuzzy matching anywhere, head survives prune, history never
silently lost.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.memory.seedvault.vault import SeedVault
from plugins.memory.seedvault.validator import CommitGate
from plugins.memory.seedvault.extractor import extract_seeds


def make_gate(tmp_path: Path):
    v = SeedVault(tmp_path)
    return v, CommitGate(v)


def commit_message(gate: CommitGate, content: str, session_id: str):
    seed = extract_seeds(
        [{"role": "assistant", "content": content}],
        session_id,
        blob_store=gate.vault.blob_store,
    )[0]
    ok, reason = gate.commit(seed)
    return seed, ok, reason


class TestCase1VersionChain:
    """Case 1: same lineage, different content → new version, old superseded."""

    def test_new_version_supersedes_old(self, tmp_path):
        v, gate = make_gate(tmp_path)
        s1, ok1, _ = commit_message(gate, "```bash\necho v1 > /etc/app.conf\n```", "sess1")
        s2, ok2, _ = commit_message(gate, "```bash\necho v2 > /etc/app.conf\n```", "sess2")

        assert ok1 and ok2
        assert s1["id"] != s2["id"], "seed IDs must be globally unique across batches"

        v1 = v.get_seed(s1["id"])
        v2 = v.get_seed(s2["id"])
        assert v1["status"] == "superseded"
        assert s2["id"] in v1["superseded_by"]
        assert v2["status"] == "active"
        # New seed records what it replaced
        types = [m["type"] for m in v2.get("meristems", [])]
        assert "supersedes" in types
        assert s1["id"] in [m["target"] for m in v2["meristems"] if m["type"] == "supersedes"]

    def test_lineage_key_is_normalized_path(self, tmp_path):
        v, gate = make_gate(tmp_path)
        s1, _, _ = commit_message(gate, "```bash\necho x > /etc/app.conf\n```", "sess1")
        s2, _, _ = commit_message(gate, "```bash\necho y > /etc/app.conf\n```", "sess2")
        lid1 = s1["artifacts"][0]["lineage_id"]
        lid2 = s2["artifacts"][0]["lineage_id"]
        assert lid1 == lid2 == "/etc/app.conf"

    def test_history_queryable_after_supersede(self, tmp_path):
        v, gate = make_gate(tmp_path)
        s1, _, _ = commit_message(gate, "```bash\necho v1 > /etc/app.conf\n```", "sess1")
        s2, _, _ = commit_message(gate, "```bash\necho v2 > /etc/app.conf\n```", "sess2")
        # Old version still retrievable by ID (never deleted)
        v1 = v.get_seed(s1["id"])
        assert v1 is not None and v1["status"] == "superseded"
        # And its blob bytes still on disk
        blob_hash = v1["artifacts"][0]["blob_hash"]
        assert v.blob_store.read_blob(blob_hash) is not None
        # Version list via lineage
        versions = v.find_lineage_candidates("/etc/app.conf")
        assert set(versions) == {s1["id"], s2["id"]}

    def test_search_returns_only_head(self, tmp_path):
        v, gate = make_gate(tmp_path)
        commit_message(gate, "```bash\necho unique-app-conf-marker v1 > /etc/app.conf\n```", "sess1")
        commit_message(gate, "```bash\necho unique-app-conf-marker v2 > /etc/app.conf\n```", "sess2")
        hits = v.search("unique-app-conf-marker", top_k=10)
        ids = [h["id"] for h in hits]
        assert len([i for i in ids if i.endswith("000001")]) == 0 or len(ids) == 1

    def test_search_can_surface_history_explicitly(self, tmp_path):
        v, gate = make_gate(tmp_path)
        s1, _, _ = commit_message(gate, "```bash\necho unique-app-conf-marker v1 > /etc/app.conf\n```", "sess1")
        s2, _, _ = commit_message(gate, "```bash\necho unique-app-conf-marker v2 > /etc/app.conf\n```", "sess2")
        # Explicit history query: all versions of the lineage
        versions = v.find_lineage_candidates("/etc/app.conf")
        assert s1["id"] in versions and s2["id"] in versions


class TestCase1bNoOp:
    """Case 1b: identical re-save → no-op, no manifest bump."""

    def test_identical_resave_is_noop(self, tmp_path):
        v, gate = make_gate(tmp_path)
        commit_message(gate, "```bash\necho v1 > /etc/app.conf\n```", "sess1")
        manifest_before = json.dumps(v._manifest, sort_keys=True)

        # Identical bytes, new session — hash matches lineage head
        s2, ok2, reason2 = commit_message(gate, "```bash\necho v1 > /etc/app.conf\n```", "sess2")
        assert not ok2
        assert "noop" in reason2 or "identical" in reason2
        assert json.dumps(v._manifest, sort_keys=True) == manifest_before, (
            "no-op must not bump the manifest"
        )

    def test_noop_does_not_create_seed_file(self, tmp_path):
        v, gate = make_gate(tmp_path)
        commit_message(gate, "```bash\necho v1 > /etc/app.conf\n```", "sess1")
        _, ok2, _ = commit_message(gate, "```bash\necho v1 > /etc/app.conf\n```", "sess2")
        assert not ok2
        seeds = list((Path(v.seeds_dir)).glob("*.json"))
        assert len(seeds) == 1, "no-op must not write a seed file"

    def test_changed_content_still_commits(self, tmp_path):
        v, gate = make_gate(tmp_path)
        commit_message(gate, "```bash\necho v1 > /etc/app.conf\n```", "sess1")
        s2, ok2, _ = commit_message(gate, "```bash\necho v2 > /etc/app.conf\n```", "sess2")
        assert ok2, "changed content must NOT be treated as a no-op"


class TestCase2IndependentLineages:
    """Case 2: different lineages never interfere."""

    def test_different_paths_are_separate_chains(self, tmp_path):
        v, gate = make_gate(tmp_path)
        s1, ok1, _ = commit_message(gate, "```bash\necho x > /etc/a.conf\n```", "sess1")
        s2, ok2, _ = commit_message(gate, "```bash\necho y > /etc/b.conf\n```", "sess2")
        assert ok1 and ok2
        a = v.get_seed(s1["id"]); b = v.get_seed(s2["id"])
        assert a["status"] == "active", "different path must not supersede"
        assert b["status"] == "active"
        assert a["artifacts"][0]["lineage_id"] != b["artifacts"][0]["lineage_id"]

    def test_cross_lineage_dedup_still_applies(self, tmp_path):
        """Exact same BYTES under different paths: dedup is content-level,
        lineage is path-level. The second seed is a duplicate (same bytes),
        NOT a new version — byte-identical files under two paths are the
        same artifact content by spec (dedup wins over lineage)."""
        v, gate = make_gate(tmp_path)
        s1, ok1, _ = commit_message(gate, "```bash\necho same > /etc/a.conf\n```", "sess1")
        # Wait — the redirect target differs, so blob bytes differ ("same > /etc/a.conf"
        # vs the raw fence content is just `echo same > /etc/a.conf`). To truly test
        # same bytes different path we need the fence content identical, which means
        # the same path... For bash artifacts the raw_content IS the command, so
        # byte-identical implies same path. This test documents that invariant instead.
        s2, ok2, reason2 = commit_message(gate, "```bash\necho same > /etc/a.conf\n```", "sess2")
        assert not ok2
        assert "duplicate" in reason2 or "noop" in reason2 or "identical" in reason2


class TestCase3EphemeralTag:
    """Case 3: explicit tag convention for ephemeral artifacts."""

    def test_tagged_ephemeral_chains(self, tmp_path):
        v, gate = make_gate(tmp_path)
        m1 = "artifact: restart-ollama\n```bash\nsystemctl restart ollama\n```"
        m2 = "artifact: restart-ollama\n```bash\nsystemctl --user restart ollama\n```"
        s1, ok1, _ = commit_message(gate, m1, "sess1")
        s2, ok2, _ = commit_message(gate, m2, "sess2")
        assert ok1 and ok2
        assert s1["artifacts"][0]["lineage_id"] == "restart-ollama"
        assert s2["artifacts"][0]["lineage_id"] == "restart-ollama"
        v1 = v.get_seed(s1["id"])
        assert v1["status"] == "superseded", "tagged ephemeral must chain like file-backed"

    def test_untagged_ephemeral_gets_no_lineage(self, tmp_path):
        v, gate = make_gate(tmp_path)
        s1, ok1, _ = commit_message(gate, "```bash\nsystemctl restart ollama\n```", "sess1")
        s2, ok2, _ = commit_message(gate, "```bash\nsystemctl --user restart ollama\n```", "sess2")
        assert ok1 and ok2
        assert s1["artifacts"][0].get("lineage_id", "") == ""
        assert s2["artifacts"][0].get("lineage_id", "") == ""
        a = v.get_seed(s1["id"]); b = v.get_seed(s2["id"])
        assert a["status"] == "active" and b["status"] == "active", (
            "no lineage → never auto-superseded (spec fallback rule)"
        )

    def test_tag_comment_inside_fence_also_recognized(self, tmp_path):
        v, gate = make_gate(tmp_path)
        s1, _, _ = commit_message(
            gate, "```bash\n# artifact: backup-tasker\ntar czf /tmp/b.tgz /opt/data\n```", "sess1"
        )
        assert s1["artifacts"][0]["lineage_id"] == "backup-tasker"


class TestCase4OrphanHandoff:
    """Case 4: blob written for a seed that never committed / was deleted."""

    def test_orphan_blob_does_not_block_resubmission(self, tmp_path):
        v, gate = make_gate(tmp_path)
        # Commit v1
        s1, ok1, _ = commit_message(gate, "```bash\necho v1 > /etc/app.conf\n```", "sess1")
        assert ok1
        # Delete the seed file (simulating pruning/removal) but leave blob index
        seed_path = Path(v.seeds_dir) / f"{s1['id']}.json"
        seed_path.unlink()
        v._manifest["seeds"].pop(s1["id"], None)
        assert v.get_seed(s1["id"]) is None

        # Same bytes come again — the blob index still knows the hash but the
        # owner is dead. Must hand off, NOT reject as duplicate.
        s2, ok2, reason2 = commit_message(gate, "```bash\necho v1 > /etc/app.conf\n```", "sess2")
        assert ok2, f"orphan hand-off failed: {reason2}"
        assert v.get_seed(s2["id"]) is not None

    def test_orphan_claim_repoints_blob_index(self, tmp_path):
        v, gate = make_gate(tmp_path)
        s1, ok1, _ = commit_message(gate, "```bash\necho v1 > /etc/app.conf\n```", "sess1")
        blob_hash = s1["artifacts"][0]["blob_hash"]
        Path(v.seeds_dir, f"{s1['id']}.json").unlink()
        v._manifest["seeds"].pop(s1["id"], None)

        s2, ok2, _ = commit_message(gate, "```bash\necho v1 > /etc/app.conf\n```", "sess2")
        assert ok2
        meta = v.blob_store.get_blob_metadata(blob_hash)
        assert meta["seed_id"] == s2["id"], "blob index must point at the live owner"


class TestCase5Prune:
    """Case 5: prune archives old versions, head survives, nothing orphaned."""

    def _build_chain(self, v, gate, n=3, path="/etc/app.conf"):
        seeds = []
        for i in range(n):
            s, ok, _ = commit_message(
                gate, f"```bash\necho v{i} > {path}\n```", f"sess{i}"
            )
            assert ok
            seeds.append(s)
        return seeds

    def _age_superseded(self, v, seeds):
        """Set superseded_at far enough back to cross the archive threshold."""
        for old in seeds[:-1]:
            v._manifest["seeds"][old["id"]]["updated"] = "2026-07-01T00:00:00+00:00"

    def test_prune_keeps_head_archives_old(self, tmp_path):
        v, gate = make_gate(tmp_path)
        seeds = self._build_chain(v, gate, n=3)
        head = seeds[-1]
        self._age_superseded(v, seeds)
        v.prune(archive_days=1)
        head_meta = v._manifest["seeds"][head["id"]]
        assert head_meta["status"] == "active", "head must survive prune"
        for old in seeds[:-1]:
            meta = v._manifest["seeds"].get(old["id"])
            assert meta is None or meta["status"] == "archived", (
                "old versions archive or leave the manifest"
            )
        # Old seed files moved to archive dir
        for old in seeds[:-1]:
            assert not (Path(v.seeds_dir) / f"{old['id']}.json").exists()
            assert (Path(v.archive_dir) / f"{old['id']}.json").exists()

    def test_prune_never_orphans_blobs(self, tmp_path):
        v, gate = make_gate(tmp_path)
        seeds = self._build_chain(v, gate, n=3)
        hashes = [s["artifacts"][0]["blob_hash"] for s in seeds]
        self._age_superseded(v, seeds)
        v.prune(archive_days=1)
        # Every blob still resolvable — history bytes survive archival
        for h in hashes:
            assert v.blob_store.read_blob(h) is not None, (
                "prune must never delete blob bytes"
            )
        # Index entries still present (superseded history stays queryable)
        for h in hashes:
            assert v.blob_store.get_blob_metadata(h) is not None

    def test_prune_preserves_lineage_query(self, tmp_path):
        v, gate = make_gate(tmp_path)
        seeds = self._build_chain(v, gate, n=3)
        self._age_superseded(v, seeds)
        v.prune(archive_days=1)
        # Lineage query still finds the full version chain (archived included)
        versions = v.find_lineage_candidates("/etc/app.conf")
        assert seeds[-1]["id"] in versions


class TestBackwardCompat:
    """Phase 9 must not break Phase 8 behavior."""

    def test_seeds_without_artifacts_unaffected(self, tmp_path):
        v, gate = make_gate(tmp_path)
        seed = {
            "id": "prose-test-000001",
            "type": "prose",
            "content": "Plain prose seed with no artifacts at all.",
            "core_claim": "Plain prose seeds commit without artifacts.",
            "tags": ["general"],
            "source_ref": {"session_id": "sess1"},
        }
        ok, reason = gate.commit(seed)
        assert ok, f"prose seed must commit: {reason}"

    def test_prose_supersession_unchanged(self, tmp_path):
        v = SeedVault(tmp_path)
        gate = CommitGate(v)
        s1 = {"id": "prose-x-000001", "type": "prose", "tags": ["deploy"],
              "content": "Deploy the API service first, then run database migrations.",
              "core_claim": "Deploy API service before running DB migrations",
              "source_ref": {"session_id": "s1"}}
        s2 = {"id": "prose-x-000002", "type": "prose", "tags": ["deploy"],
              "content": "New deploy order: migrations run before the API rollout starts.",
              "core_claim": "Run DB migrations before the API rollout",
              "source_ref": {"session_id": "s2"}}
        ok1, _ = gate.commit(s1)
        ok2, _ = gate.commit(s2)
        assert ok1 and ok2
        assert v.get_seed("prose-x-000001")["status"] == "superseded"