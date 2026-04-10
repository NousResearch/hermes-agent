"""Tests for memory housekeeping — noise deletion, stale archival, stats recomputation."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from plugins.memory.tiered.housekeeping import HousekeepingResult, run_housekeeping


@pytest.fixture()
def mock_db():
    """Create a mock TieredMemoryDB."""
    db = MagicMock()
    db.delete_noise.return_value = 0
    db.archive_stale.return_value = 0
    db.deduplicate_similar.return_value = 0
    db.update_stats.return_value = None
    db._conn = MagicMock()
    return db


class TestRunHousekeeping:
    def test_run_housekeeping_all_phases(self, mock_db):
        """All phases called in order."""
        mock_db.delete_noise.return_value = 2
        mock_db.archive_stale.return_value = 5
        mock_db.deduplicate_similar.return_value = 0

        result = run_housekeeping(mock_db)

        mock_db.delete_noise.assert_called_once()
        mock_db.archive_stale.assert_called_once()
        mock_db.deduplicate_similar.assert_called_once()
        mock_db.update_stats.assert_called_once()
        mock_db._conn.execute.assert_called_once_with("PRAGMA optimize")
        assert result.deleted == 2
        assert result.archived == 5
        assert result.deduped == 0
        assert result.errors == []

    def test_run_housekeeping_noise_failure_continues(self, mock_db):
        """If delete_noise raises, archive_stale still called."""
        mock_db.delete_noise.side_effect = RuntimeError("db locked")
        mock_db.archive_stale.return_value = 3

        result = run_housekeeping(mock_db)

        mock_db.archive_stale.assert_called_once()
        assert result.deleted == 0
        assert result.archived == 3
        assert len(result.errors) == 1
        assert "noise_deletion" in result.errors[0]

    def test_run_housekeeping_partial_results(self, mock_db):
        """Result dataclass has correct counts."""
        mock_db.delete_noise.return_value = 10
        mock_db.archive_stale.return_value = 0

        result = run_housekeeping(mock_db)

        assert result.deleted == 10
        assert result.archived == 0
        assert result.errors == []

    def test_run_housekeeping_errors_collected(self, mock_db):
        """All failures added to errors list."""
        mock_db.delete_noise.side_effect = RuntimeError("noise error")
        mock_db.archive_stale.side_effect = RuntimeError("archive error")
        mock_db.deduplicate_similar.side_effect = RuntimeError("dedup error")
        mock_db.update_stats.side_effect = RuntimeError("stats error")
        mock_db._conn.execute.side_effect = RuntimeError("pragma error")

        result = run_housekeeping(mock_db)

        assert len(result.errors) == 5
        assert "noise_deletion" in result.errors[0]
        assert "stale_archival" in result.errors[1]
        assert "semantic_dedup" in result.errors[2]
        assert "stats_recomputation" in result.errors[3]
        assert "pragma_optimize" in result.errors[4]
        assert result.deleted == 0
        assert result.archived == 0
        assert result.deduped == 0


    def test_housekeeping_calls_deduplicate(self, mock_db):
        """deduplicate_similar() is called during housekeeping."""
        mock_db.deduplicate_similar.return_value = 0

        run_housekeeping(mock_db)

        mock_db.deduplicate_similar.assert_called_once()

    def test_housekeeping_dedup_count_in_result(self, mock_db):
        """HousekeepingResult.deduped contains the count from deduplicate_similar."""
        mock_db.deduplicate_similar.return_value = 7

        result = run_housekeeping(mock_db)

        assert result.deduped == 7

    def test_housekeeping_dedup_error_handled(self, mock_db):
        """Dedup failure is logged and added to errors list, doesn't crash pipeline."""
        mock_db.deduplicate_similar.side_effect = RuntimeError("vec index corrupt")

        result = run_housekeeping(mock_db)

        # Pipeline continues — stats and PRAGMA still called
        mock_db.update_stats.assert_called_once()
        mock_db._conn.execute.assert_called_once_with("PRAGMA optimize")
        assert result.deduped == 0
        assert any("semantic_dedup" in e for e in result.errors)

    def test_housekeeping_dedup_runs_after_stale_archival(self, mock_db):
        """Ordering: noise -> stale -> dedup -> stats -> optimize."""
        call_order = []
        mock_db.delete_noise.side_effect = lambda: call_order.append("noise") or 0
        mock_db.archive_stale.side_effect = lambda: call_order.append("stale") or 0
        mock_db.deduplicate_similar.side_effect = lambda: call_order.append("dedup") or 0
        mock_db.update_stats.side_effect = lambda: call_order.append("stats")
        mock_db._conn.execute.side_effect = lambda _: call_order.append("optimize")

        run_housekeeping(mock_db)

        assert call_order == ["noise", "stale", "dedup", "stats", "optimize"]


class TestHousekeepingResult:
    def test_defaults(self):
        """Default values are correct."""
        result = HousekeepingResult()
        assert result.deleted == 0
        assert result.archived == 0
        assert result.deduped == 0
        assert result.errors == []

    def test_errors_list_not_shared(self):
        """Each instance gets its own errors list."""
        r1 = HousekeepingResult()
        r2 = HousekeepingResult()
        r1.errors.append("test")
        assert r2.errors == []
