"""Tests for sync_jobs_provider_snapshots (#61404)."""

from unittest.mock import patch

import pytest

from cron.jobs import sync_jobs_provider_snapshots


class TestSyncJobsProviderSnapshots:
    """Batch-refresh provider/model snapshots on unpinned cron jobs."""

    # ── helpers ──────────────────────────────────────────────────────

    _SNAPSHOT_PAIR = ("openrouter", "openai/gpt-5")

    @staticmethod
    def _mock_jobs_store(jobs_list, monkeypatch):
        """Patch load_jobs/save_jobs to an in-memory list."""
        store: list[dict] = list(jobs_list)

        def _load():
            return list(store)

        def _save(jobs):
            store.clear()
            store.extend(jobs)

        monkeypatch.setattr("cron.jobs.load_jobs", _load)
        monkeypatch.setattr("cron.jobs.save_jobs", _save)
        monkeypatch.setattr("cron.jobs._save_jobs_unlocked", _save)
        return store

    @staticmethod
    def _mock_snapshots(monkeypatch, provider="openrouter",
                        model="openai/gpt-5"):
        monkeypatch.setattr(
            "cron.jobs._compute_provider_model_snapshots",
            lambda **kw: (provider, model),
        )

    # ── tests ────────────────────────────────────────────────────────

    def test_all_unpinned_synced(self, monkeypatch):
        """Two unpinned jobs → both snapshots refreshed to current global."""
        jobs = [
            {"id": "j1", "provider_snapshot": "old", "model_snapshot": "old"},
            {"id": "j2", "provider_snapshot": "old", "model_snapshot": "x"},
        ]
        store = self._mock_jobs_store(jobs, monkeypatch)
        self._mock_snapshots(monkeypatch)

        result = sync_jobs_provider_snapshots()

        assert result["updated"] == ["j1", "j2"]
        assert result["skipped"] == 0
        assert result["no_agent"] == 0
        assert result["left"] == 0

        # Verify store was mutated
        assert store[0]["provider_snapshot"] == "openrouter"
        assert store[0]["model_snapshot"] == "openai/gpt-5"
        assert store[1]["provider_snapshot"] == "openrouter"
        assert store[1]["model_snapshot"] == "openai/gpt-5"

    def test_idempotent_noop(self, monkeypatch):
        """Already up-to-date jobs → left unchanged, no writes."""
        jobs = [
            {"id": "j1", "provider_snapshot": "openrouter",
             "model_snapshot": "openai/gpt-5"},
        ]
        store = self._mock_jobs_store(jobs, monkeypatch)
        self._mock_snapshots(monkeypatch)

        result = sync_jobs_provider_snapshots()

        assert result["left"] == 1
        assert result["updated"] == []
        assert store[0]["provider_snapshot"] == "openrouter"

    def test_pinned_provider_skipped(self, monkeypatch):
        """Pinned provider → skipped, no mutation."""
        jobs = [
            {"id": "j1", "provider": "nous", "provider_snapshot": "old"},
            {"id": "j2", "provider_snapshot": "old"},
        ]
        store = self._mock_jobs_store(jobs, monkeypatch)
        self._mock_snapshots(monkeypatch)

        result = sync_jobs_provider_snapshots()

        assert result["skipped"] == 1
        assert result["updated"] == ["j2"]
        assert store[0]["provider_snapshot"] == "old"  # untouched
        assert store[1]["provider_snapshot"] == "openrouter"  # refreshed

    def test_pinned_model_skipped(self, monkeypatch):
        """Pinned model → skipped, no mutation."""
        jobs = [
            {"id": "j1", "model": "claude-sonnet-4",
             "model_snapshot": "old-m"},
            {"id": "j2", "model_snapshot": "old-m"},
        ]
        store = self._mock_jobs_store(jobs, monkeypatch)
        self._mock_snapshots(monkeypatch)

        result = sync_jobs_provider_snapshots()

        assert result["skipped"] == 1
        assert result["updated"] == ["j2"]
        assert store[0]["model_snapshot"] == "old-m"  # untouched

    def test_no_agent_jobs_skipped(self, monkeypatch):
        """no_agent jobs → skipped categorically."""
        jobs = [
            {"id": "na1", "no_agent": True},
            {"id": "j1", "provider_snapshot": "old"},
        ]
        store = self._mock_jobs_store(jobs, monkeypatch)
        self._mock_snapshots(monkeypatch)

        result = sync_jobs_provider_snapshots()

        assert result["no_agent"] == 1
        assert result["updated"] == ["j1"]
        assert "provider_snapshot" not in store[0]  # never written

    def test_mixed_fleet(self, monkeypatch):
        """5 jobs: 2 unpinned (sync), 1 pinned provider, 1 no_agent, 1 already current."""
        jobs = [
            {"id": "j1", "provider_snapshot": "stale"},
            {"id": "j2", "provider_snapshot": "stale"},
            {"id": "j3", "provider": "nous"},
            {"id": "j4", "no_agent": True},
            {"id": "j5", "provider_snapshot": "openrouter",
             "model_snapshot": "openai/gpt-5"},
        ]
        store = self._mock_jobs_store(jobs, monkeypatch)
        self._mock_snapshots(monkeypatch)

        result = sync_jobs_provider_snapshots()

        assert result["updated"] == ["j1", "j2"]
        assert result["skipped"] == 1
        assert result["no_agent"] == 1
        assert result["left"] == 1

    def test_empty_jobs_list(self, monkeypatch):
        """No jobs → clean return, no crash."""
        self._mock_jobs_store([], monkeypatch)
        self._mock_snapshots(monkeypatch)

        result = sync_jobs_provider_snapshots()

        assert result == {
            "updated": [], "left": 0, "skipped": 0,
            "no_agent": 0, "errors": [],
        }

    def test_target_none_not_overwritten(self, monkeypatch):
        """When global default resolution returns None for both axes,
        existing snapshot fields are left alone (fail-safe)."""
        jobs = [
            {"id": "j1", "provider_snapshot": "existing-p"},
        ]
        store = self._mock_jobs_store(jobs, monkeypatch)
        monkeypatch.setattr(
            "cron.jobs._compute_provider_model_snapshots",
            lambda **kw: (None, None),
        )

        result = sync_jobs_provider_snapshots()

        assert result["updated"] == []
        assert store[0]["provider_snapshot"] == "existing-p"

    def test_lock_exclusion(self, monkeypatch):
        """Verifies _jobs_lock context manager is entered."""
        jobs = [{"id": "j1", "provider_snapshot": "old"}]
        self._mock_jobs_store(jobs, monkeypatch)
        self._mock_snapshots(monkeypatch)

        lock_entered = []

        class _FakeLock:
            def __enter__(self):
                lock_entered.append(True)
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr("cron.jobs._jobs_lock", lambda: _FakeLock())

        sync_jobs_provider_snapshots()

        assert len(lock_entered) == 1, "sync must hold the jobs lock"