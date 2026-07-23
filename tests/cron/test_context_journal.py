"""Tests for cron/context_journal.py — append, prune, atomic write, clear.

Verifies that the context journal correctly records cron deliveries, bounds
file growth via count-based pruning, handles empty content gracefully, and
survives concurrent appends without data loss.
"""

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def journal_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty cron context journal."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.context_journal as cj
    monkeypatch.setattr(cj, "_get_home", lambda: hermes_home)
    return hermes_home, cj


class TestAppendAndRead:
    """Basic append and read-roundtrip."""

    def test_append_creates_entry(self, journal_env):
        hermes_home, cj = journal_env
        cj.append_entry("job_1", "Test Job", "This is the output content")

        entries = cj.read_journal()
        assert len(entries) == 1
        assert entries[0]["job_id"] == "job_1"
        assert entries[0]["job_name"] == "Test Job"
        assert "output content" in entries[0]["summary"]

    def test_append_empty_content_skipped(self, journal_env):
        _, cj = journal_env
        cj.append_entry("job_1", "Empty", "")
        assert len(cj.read_journal()) == 0

    def test_append_whitespace_only_skipped(self, journal_env):
        _, cj = journal_env
        cj.append_entry("job_1", "Whitespace", "   \n  \n  ")
        assert len(cj.read_journal()) == 0

    def test_read_journal_no_file(self, journal_env):
        _, cj = journal_env
        assert cj.read_journal() == []

    def test_read_journal_corrupted_file(self, journal_env):
        hermes_home, cj = journal_env
        path = hermes_home / "cron" / "context_journal.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json", encoding="utf-8")
        assert cj.read_journal() == []

    def test_multiple_appends_preserve_order(self, journal_env):
        _, cj = journal_env
        cj.append_entry("job_1", "First", "Alpha")
        cj.append_entry("job_2", "Second", "Beta")
        entries = cj.read_journal()
        assert len(entries) == 2
        assert entries[0]["job_id"] == "job_1"
        assert entries[1]["job_id"] == "job_2"


class TestPruning:
    """Count-based and size-based pruning."""

    def test_prune_to_max_entries(self, journal_env):
        _, cj = journal_env
        for i in range(25):
            cj.append_entry(f"job_{i}", f"Job {i}", f"Output {i}")

        entries = cj.read_journal()
        assert len(entries) == 20
        assert entries[0]["job_id"] == "job_5"
        assert entries[-1]["job_id"] == "job_24"

    def test_custom_max_entries_via_config(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config_dir = tmp_path / ".config"
        config_dir.mkdir()
        cfg_file = config_dir / "config.yaml"
        cfg_file.write_text("cron:\n  context_journal:\n    max_entries: 5\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_CONFIG", str(config_dir))

        import cron.context_journal as cj
        monkeypatch.setattr(cj, "_get_home", lambda: hermes_home)

        assert cj._max_entries() == 5
        for i in range(10):
            cj.append_entry(f"job_{i}", f"Job {i}", f"Output {i}")
        assert len(cj.read_journal()) == 5

    def test_zero_max_entries_disables(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        import cron.context_journal as cj
        monkeypatch.setattr(cj, "_get_home", lambda: hermes_home)

        assert cj._max_entries() == 20

    def test_entry_has_delivered_at_timestamp(self, journal_env):
        _, cj = journal_env
        cj.append_entry("job_1", "Test", "content")
        entry = cj.read_journal()[0]
        assert "delivered_at" in entry
        assert "T" in entry["delivered_at"]
        assert entry["delivered_at"].endswith("Z")


class TestClear:
    """Journal clear operation."""

    def test_clear_removes_entries(self, journal_env):
        _, cj = journal_env
        cj.append_entry("job_1", "Test", "content")
        assert len(cj.read_journal()) == 1

        count = cj.clear_journal()
        assert count == 1
        assert cj.read_journal() == []

    def test_clear_empty_journal(self, journal_env):
        _, cj = journal_env
        count = cj.clear_journal()
        assert count == 0
        assert cj.read_journal() == []


class TestSummarize:
    """Output summarization."""

    def test_summarize_truncates_long_output(self, journal_env):
        _, cj = journal_env
        long_text = "\n".join(f"line {i}" for i in range(100))
        summary = cj._summarize(long_text, max_chars=50)
        assert len(summary) <= 55
        assert "line 0" in summary

    def test_summarize_empty(self, journal_env):
        _, cj = journal_env
        assert cj._summarize("") == ""
        assert cj._summarize("  ") == ""

    def test_summarize_short_passthrough(self, journal_env):
        _, cj = journal_env
        text = "Short output"
        assert cj._summarize(text) == text

    def test_summarize_skips_blank_lines(self, journal_env):
        _, cj = journal_env
        text = "\n\n\nfirst\n\n\nsecond\n\n\n"
        summary = cj._summarize(text)
        assert summary == "first\nsecond"

    def test_summarize_appends_truncation_marker(self, journal_env):
        _, cj = journal_env
        text = "\n".join(f"line {i}" for i in range(50))
        summary = cj._summarize(text, max_chars=30)
        assert "[...]" in summary


class TestConcurrency:
    """Thread safety under concurrent appends."""

    def test_concurrent_appends_no_data_loss(self, journal_env):
        _, cj = journal_env
        n_threads = 5
        entries_per_thread = 10

        def _append(start):
            for i in range(start, start + entries_per_thread):
                cj.append_entry(f"job_{i}", f"Job {i}", f"output {i}")

        threads = [
            threading.Thread(target=_append, args=(i * entries_per_thread,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = cj.read_journal()
        assert len(entries) == 20
        ids = {e["job_id"] for e in entries}
        assert all(f"job_{i}" in ids for i in range(40, 50))


class TestAtomWrite:
    """Atomic write guarantees."""

    def test_write_is_atomic(self, journal_env):
        hermes_home, cj = journal_env
        cj.append_entry("job_1", "Test", "content")
        path = hermes_home / "cron" / "context_journal.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data) == 1

        cj.clear_journal()
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == []
