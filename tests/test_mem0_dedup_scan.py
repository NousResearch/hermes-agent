"""Tests for scripts/mem0_dedup_scan.py — mem0 near-duplicate scanner.

Covers:
- Profile config resolution (uses get_hermes_home(), not hardcoded)
- Deduplication of overlapping pairs (no ID queued twice)
- Dry-run mode (no deletions without --yes)

Qdrant client is mocked throughout to avoid needing a real Qdrant instance.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to import the script under test as a module
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "mem0_dedup_scan.py"


def _load_module() -> types.ModuleType:
    """Import mem0_dedup_scan as a fresh module.

    The conftest autouse fixture already sets HERMES_HOME to a per-test
    tempdir via monkeypatch.setenv, so hermes_constants.get_hermes_home()
    will return the correct isolated path during the test.
    """
    sys.modules.pop("mem0_dedup_scan", None)
    spec = importlib.util.spec_from_file_location("mem0_dedup_scan", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def hermes_home(tmp_path: Path) -> Path:
    """Return the HERMES_HOME tempdir that conftest already set up.

    The conftest autouse fixture sets HERMES_HOME=tmp_path/hermes_test.
    We retrieve it here so tests can write files into it.
    """
    home = Path(os.environ["HERMES_HOME"])
    return home


@pytest.fixture()
def mem0_config(hermes_home: Path) -> Path:
    """Write a minimal mem0.json into the isolated HERMES_HOME."""
    cfg = {
        "user_id": "test-user-42",
        "oss": {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "my_memories",
                    "path": str(hermes_home / "mem0_qdrant"),
                },
            }
        },
    }
    config_path = hermes_home / "mem0.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# 1. Profile config resolution
# ---------------------------------------------------------------------------

class TestProfileConfigResolution:
    """get_hermes_home() must be used — not a hardcoded ~/.hermes path."""

    def test_mem0_config_path_not_hardcoded(self, hermes_home: Path) -> None:
        """MEM0_CONFIG must point under the active HERMES_HOME, not ~/.hermes."""
        mod = _load_module()
        # The module-level MEM0_CONFIG should be under the per-test HERMES_HOME,
        # not the real ~/.hermes.
        assert str(mod.MEM0_CONFIG).startswith(str(hermes_home)), (
            f"MEM0_CONFIG={mod.MEM0_CONFIG!r} should be under {hermes_home}"
        )
        assert str(Path.home() / ".hermes") not in str(mod.MEM0_CONFIG) or \
            str(hermes_home) == str(Path.home() / ".hermes"), (
            "MEM0_CONFIG must not hardcode ~/.hermes when HERMES_HOME is overridden"
        )

    def test_get_hermes_home_returns_active_home(self, hermes_home: Path) -> None:
        """get_hermes_home() should return the currently active HERMES_HOME."""
        mod = _load_module()
        result = mod.get_hermes_home()
        assert result == hermes_home

    def test_fallback_when_hermes_constants_unavailable(self) -> None:
        """Without hermes_constants, the script falls back to Path.home() / '.hermes'."""
        sys.modules.pop("mem0_dedup_scan", None)
        # Temporarily hide hermes_constants
        original = sys.modules.get("hermes_constants")
        sys.modules["hermes_constants"] = None  # type: ignore[assignment]
        try:
            spec = importlib.util.spec_from_file_location("mem0_dedup_scan", _SCRIPT_PATH)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            result = mod.get_hermes_home()
            assert result == Path.home() / ".hermes"
        finally:
            if original is not None:
                sys.modules["hermes_constants"] = original
            else:
                sys.modules.pop("hermes_constants", None)
            sys.modules.pop("mem0_dedup_scan", None)

    def test_load_mem0_config_reads_from_active_home(
        self, mem0_config: Path, hermes_home: Path
    ) -> None:
        """_load_mem0_config() should read mem0.json from the active hermes home."""
        mod = _load_module()
        cfg = mod._load_mem0_config()
        assert cfg.get("user_id") == "test-user-42"
        assert cfg["oss"]["vector_store"]["config"]["collection_name"] == "my_memories"

    def test_config_qdrant_path_from_mem0_json(
        self, mem0_config: Path, hermes_home: Path
    ) -> None:
        """_config_qdrant_path() should return the embedded path from mem0.json."""
        mod = _load_module()
        cfg = mod._load_mem0_config()
        path = mod._config_qdrant_path(cfg)
        assert path == str(hermes_home / "mem0_qdrant")

    def test_config_qdrant_url_returns_none_for_embedded(
        self, mem0_config: Path
    ) -> None:
        """When mem0.json has only path=, _config_qdrant_url() should return None."""
        mod = _load_module()
        cfg = mod._load_mem0_config()
        url = mod._config_qdrant_url(cfg)
        assert url is None

    def test_config_qdrant_url_from_mem0_json(self, hermes_home: Path) -> None:
        """_config_qdrant_url() returns the HTTP URL when mem0.json specifies one."""
        cfg = {
            "oss": {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {"url": "http://myserver:6333"},
                }
            }
        }
        (hermes_home / "mem0.json").write_text(json.dumps(cfg), encoding="utf-8")
        mod = _load_module()
        parsed_cfg = mod._load_mem0_config()
        url = mod._config_qdrant_url(parsed_cfg)
        assert url == "http://myserver:6333"

    def test_load_mem0_config_returns_empty_dict_when_missing(
        self, hermes_home: Path
    ) -> None:
        """_load_mem0_config() returns {} when mem0.json doesn't exist."""
        mod = _load_module()
        # No mem0.json written in this test
        cfg = mod._load_mem0_config()
        assert cfg == {}


# ---------------------------------------------------------------------------
# 2. Deduplication of overlapping pairs (no ID queued twice)
# ---------------------------------------------------------------------------

def _make_point(pid: str, user_id: str, text: str, vec: list[float]) -> dict:
    return {
        "id": pid,
        "vector": vec,
        "payload": {"user_id": user_id, "data": text},
    }


class TestDeduplicationCorrectness:
    """Overlapping duplicate pairs must not cause the same ID to be deleted twice."""

    def test_group_pairs_single_pair(self) -> None:
        """One pair produces one group with two members."""
        mod = _load_module()
        a = _make_point("a1", "u", "hello world", [1.0, 0.0])
        b = _make_point("b1", "u", "hello world copy", [0.99, 0.14])
        score = mod.cosine_similarity(a["vector"], b["vector"])
        pairs = [(score, a, b)]
        groups = mod.group_pairs(pairs)
        assert len(groups) == 1
        assert set(groups[0]["members"]) == {"a1", "b1"}

    def test_overlapping_pairs_do_not_double_queue(self) -> None:
        """When the same ID appears in two pairs, consolidate_groups deletes it once."""
        mod = _load_module()

        # Build points: B is near-duplicate of both A and C
        a = _make_point("a1", "u", "short text A", [1.0, 0.0, 0.0])
        b = _make_point("b1", "u", "short text B very long so it is kept", [0.999, 0.045, 0.0])
        c = _make_point("c1", "u", "short text C", [0.0, 0.0, 1.0])

        # Manually build two overlapping groups (A,B) and (B,C)
        groups = [
            {
                "members": ["a1", "b1"],
                "points": {"a1": a, "b1": b},
                "edge_scores": {("a1", "b1"): 0.95},
            },
            {
                "members": ["b1", "c1"],
                "points": {"b1": b, "c1": c},
                "edge_scores": {("b1", "c1"): 0.93},
            },
        ]

        deleted_calls: list[list] = []

        def fake_delete_points(ids: list) -> bool:
            deleted_calls.append(list(ids))
            return True

        with patch.object(mod, "delete_points", side_effect=fake_delete_points):
            kept, deleted = mod.consolidate_groups(groups, dry_run=False)

        # Should have resolved both groups
        assert kept == 2

        # Collect all IDs passed to delete_points across all calls
        all_deleted_ids: list[str] = [
            pid for call in deleted_calls for pid in call
        ]
        # No ID should appear more than once
        assert len(all_deleted_ids) == len(set(all_deleted_ids)), (
            f"Duplicate IDs in delete calls: {all_deleted_ids}"
        )
        # deleted count should match unique IDs
        assert deleted == len(set(all_deleted_ids))

    def test_overlapping_pairs_transitive_safety(self) -> None:
        """A point must not be deleted when its keeper is also slated for deletion.

        Setup: groups are sorted descending by similarity.
          Group 0 (higher similarity): B~C, C is longest so C is kept, B deleted.
          Group 1 (lower similarity):  A~B, B is already in ids_to_delete.

        With the transitive-safety guard, Group 1 is skipped entirely, so A
        survives even though it was near-duplicate of B (which is now gone).
        A and C are not similar so A must not be collateral damage.
        """
        mod = _load_module()
        a = _make_point("a1", "u", "short A", [1.0, 0.0, 0.0])
        b = _make_point("b1", "u", "medium text B", [0.999, 0.045, 0.0])
        c = _make_point("c1", "u", "very long text C is the longest here", [0.0, 0.0, 1.0])

        # Present higher-similarity pair first (B~C) so B gets marked for deletion
        # before the lower-similarity pair (A~B) is processed.
        groups = [
            {
                "members": ["b1", "c1"],
                "points": {"b1": b, "c1": c},
                "edge_scores": {("b1", "c1"): 0.96},
            },
            {
                "members": ["a1", "b1"],
                "points": {"a1": a, "b1": b},
                "edge_scores": {("a1", "b1"): 0.95},
            },
        ]

        deleted_calls: list[list] = []

        def fake_delete_points(ids: list) -> bool:
            deleted_calls.append(list(ids))
            return True

        with patch.object(mod, "delete_points", side_effect=fake_delete_points):
            mod.consolidate_groups(groups, dry_run=False)

        all_deleted = {pid for call in deleted_calls for pid in call}
        # 'a1' must survive: its keeper 'b1' was deleted, and 'a1' is not
        # similar to 'c1', so deleting it would be permanent information loss.
        assert "a1" not in all_deleted, (
            f"'a1' was unsafely deleted (transitive deletion bug). Deleted: {all_deleted}"
        )
        assert "b1" in all_deleted

    def test_pick_keeper_prefers_longer_text(self) -> None:
        """pick_keeper should choose the member with the longer text."""
        mod = _load_module()
        short = _make_point("s1", "u", "short", [1.0, 0.0])
        long_ = _make_point("l1", "u", "this is a much longer memory text", [0.99, 0.14])
        group = {
            "members": ["s1", "l1"],
            "points": {"s1": short, "l1": long_},
            "edge_scores": {("s1", "l1"): 0.95},
        }
        keeper = mod.pick_keeper(group)
        assert keeper == "l1"

    def test_pick_keeper_breaks_ties_by_updated_at(self) -> None:
        """When text lengths are equal, pick_keeper keeps the most recently updated."""
        mod = _load_module()
        older = _make_point("old", "u", "same text", [1.0, 0.0])
        older["payload"]["updated_at"] = "2024-01-01T00:00:00"
        newer = _make_point("new", "u", "same text", [0.99, 0.14])
        newer["payload"]["updated_at"] = "2025-06-01T00:00:00"
        group = {
            "members": ["old", "new"],
            "points": {"old": older, "new": newer},
            "edge_scores": {("old", "new"): 0.98},
        }
        keeper = mod.pick_keeper(group)
        assert keeper == "new"

    def test_find_duplicates_filters_by_user_id(self) -> None:
        """find_duplicates should only compare points for the target user."""
        mod = _load_module()
        a = _make_point("a1", "alice", "hello", [1.0, 0.0])
        b = _make_point("b1", "alice", "hello copy", [0.999, 0.045])
        c = _make_point("c1", "bob", "hello", [1.0, 0.0])
        points = [a, b, c]

        pairs, count = mod.find_duplicates(points, "alice")
        member_ids = {str(p["id"]) for _, p1, p2 in pairs for p in [p1, p2]}
        assert "c1" not in member_ids
        assert count == 2  # only alice's two points

    def test_cosine_similarity_identical_vectors(self) -> None:
        """Identical vectors have similarity 1.0."""
        mod = _load_module()
        v = [0.5, 0.5, 0.5, 0.5]
        assert abs(mod.cosine_similarity(v, v) - 1.0) < 1e-9

    def test_cosine_similarity_orthogonal_vectors(self) -> None:
        """Orthogonal vectors have similarity 0.0."""
        mod = _load_module()
        assert mod.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_cosine_similarity_zero_vector(self) -> None:
        """Zero vector returns 0.0 (not a division error)."""
        mod = _load_module()
        assert mod.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# 3. Dry-run mode — no deletions without --yes
# ---------------------------------------------------------------------------

class TestDryRunSafety:
    """consolidate_groups(dry_run=True) must never call delete_points."""

    def _one_group(self, mod: types.ModuleType) -> list[dict]:
        a = _make_point("a1", "u", "memory A", [1.0, 0.0])
        b = _make_point("b1", "u", "memory B — much longer so it is kept as the keeper", [0.999, 0.045])
        return [
            {
                "members": ["a1", "b1"],
                "points": {"a1": a, "b1": b},
                "edge_scores": {("a1", "b1"): 0.999},
            }
        ]

    def test_dry_run_does_not_call_delete_points(self) -> None:
        """dry_run=True must never invoke delete_points."""
        mod = _load_module()
        groups = self._one_group(mod)

        with patch.object(mod, "delete_points") as mock_delete:
            mod.consolidate_groups(groups, dry_run=True)
            mock_delete.assert_not_called()

    def test_dry_run_reports_correct_counts(self) -> None:
        """dry_run should return counts as if the deletion happened (for reporting)."""
        mod = _load_module()
        groups = self._one_group(mod)

        with patch.object(mod, "delete_points"):
            kept, deleted = mod.consolidate_groups(groups, dry_run=True)

        assert kept == 1     # one group resolved
        assert deleted == 1  # one duplicate would be removed

    def test_wet_run_calls_delete_points(self) -> None:
        """dry_run=False must call delete_points with the loser IDs."""
        mod = _load_module()
        groups = self._one_group(mod)

        with patch.object(mod, "delete_points", return_value=True) as mock_delete:
            kept, deleted = mod.consolidate_groups(groups, dry_run=False)

        mock_delete.assert_called_once()
        # The call should contain the loser ID 'a1' (shorter text)
        call_args = mock_delete.call_args[0][0]
        assert "a1" in call_args
        assert kept == 1
        assert deleted == 1

    def test_multiple_groups_single_delete_batch(self) -> None:
        """Multiple groups with non-overlapping IDs should be batched into one delete call."""
        mod = _load_module()

        a = _make_point("a1", "u", "group1 A", [1.0, 0.0])
        b = _make_point("b1", "u", "group1 B very long text so it wins", [0.999, 0.045])
        c = _make_point("c1", "u", "group2 C", [0.0, 1.0])
        d = _make_point("d1", "u", "group2 D much longer text here so it wins", [0.045, 0.999])

        groups = [
            {
                "members": ["a1", "b1"],
                "points": {"a1": a, "b1": b},
                "edge_scores": {("a1", "b1"): 0.95},
            },
            {
                "members": ["c1", "d1"],
                "points": {"c1": c, "d1": d},
                "edge_scores": {("c1", "d1"): 0.93},
            },
        ]

        with patch.object(mod, "delete_points", return_value=True) as mock_delete:
            kept, deleted = mod.consolidate_groups(groups, dry_run=False)

        # Should be a single batch call containing both loser IDs
        assert mock_delete.call_count == 1
        batch = mock_delete.call_args[0][0]
        assert set(batch) == {"a1", "c1"}
        assert kept == 2
        assert deleted == 2

    def test_verify_qdrant_connection_embedded_no_http_call(
        self, tmp_path: Path
    ) -> None:
        """verify_qdrant_connection in embedded mode must not attempt any HTTP call."""
        mod = _load_module()
        qdrant_path = tmp_path / "mem0_qdrant"
        qdrant_path.mkdir()  # path must exist; verify_qdrant_connection exits(1) if missing

        with patch.object(mod, "api_get") as mock_api_get:
            mod.verify_qdrant_connection(
                qdrant_url=None,
                qdrant_path=str(qdrant_path),
            )
            mock_api_get.assert_not_called()

    def test_verify_qdrant_connection_embedded_exits_on_missing_path(
        self, tmp_path: Path
    ) -> None:
        """verify_qdrant_connection must exit(1) when the embedded path is absent."""
        mod = _load_module()
        missing_path = str(tmp_path / "does_not_exist")

        with pytest.raises(SystemExit) as exc_info:
            mod.verify_qdrant_connection(qdrant_url=None, qdrant_path=missing_path)
        assert exc_info.value.code == 1
