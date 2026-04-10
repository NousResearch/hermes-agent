"""Tests for enrichment.py — context enrichment for prefetch system prompt."""

import json
import logging
import os
import sqlite3
import subprocess
import time
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from plugins.memory.tiered.enrichment import (
    MAX_ENRICHMENT_CHARS,
    _get_jira_token,
    _query_calendar,
    _query_jira,
    compose_enrichment_block,
    get_active_principles,
    read_enrichment_cache,
    refresh_enrichment_async,
    refresh_enrichment_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jira_stdout(tasks: list[tuple[str, str, str]]) -> str:
    """Build tab-separated jira CLI output from (key, summary, status) tuples."""
    return "\n".join(f"{k}\t{s}\t{st}" for k, s, st in tasks)


def _make_cache(jira: list | None = None, calendar: list | None = None) -> dict:
    """Build a valid enrichment cache dict."""
    return {
        "refreshed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "jira": jira or [],
        "calendar": calendar or [],
    }


def _mock_db_with_principles(titles: list[str]) -> MagicMock:
    """Create a mock TieredMemoryDB whose query_active_principles returns the given titles."""
    db = MagicMock()
    db.query_active_principles.return_value = titles
    return db


# ---------------------------------------------------------------------------
# _get_jira_token
# ---------------------------------------------------------------------------

class TestGetJiraToken:
    def test_token_from_env(self, monkeypatch):
        """Token present in env -> returns it directly."""
        monkeypatch.setenv("JIRA_API_TOKEN", "env-token-123")
        assert _get_jira_token() == "env-token-123"

    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_token_from_envchain(self, mock_run, monkeypatch):
        """Token not in env, envchain returns it -> returns envchain result."""
        monkeypatch.delenv("JIRA_API_TOKEN", raising=False)
        mock_run.return_value = MagicMock(
            returncode=0, stdout="envchain-token-456\n",
        )

        result = _get_jira_token()

        assert result == "envchain-token-456"
        mock_run.assert_called_once_with(
            ["envchain", "hermes-jira", "printenv", "ATLASSIAN_API_TOKEN"],
            capture_output=True, text=True, timeout=5,
        )

    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_token_envchain_fails(self, mock_run, monkeypatch):
        """Token not in env, envchain fails -> returns None."""
        monkeypatch.delenv("JIRA_API_TOKEN", raising=False)
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error msg")

        assert _get_jira_token() is None

    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_token_envchain_fails_logs_message(self, mock_run, monkeypatch, caplog):
        """Non-zero envchain exit code -> logs info message with exit code and stderr."""
        monkeypatch.delenv("JIRA_API_TOKEN", raising=False)
        mock_run.return_value = MagicMock(returncode=2, stdout="", stderr="namespace not found")

        with caplog.at_level(logging.INFO, logger="plugins.memory.tiered.enrichment"):
            result = _get_jira_token()

        assert result is None
        assert "exit 2" in caplog.text
        assert "namespace not found" in caplog.text

    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_token_envchain_not_found(self, mock_run, monkeypatch):
        """Token not in env, envchain binary missing -> returns None."""
        monkeypatch.delenv("JIRA_API_TOKEN", raising=False)
        mock_run.side_effect = FileNotFoundError("envchain not found")

        assert _get_jira_token() is None

    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_token_envchain_timeout(self, mock_run, monkeypatch):
        """Token not in env, envchain times out -> returns None."""
        monkeypatch.delenv("JIRA_API_TOKEN", raising=False)
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="envchain", timeout=5)

        assert _get_jira_token() is None


# ---------------------------------------------------------------------------
# _query_jira
# ---------------------------------------------------------------------------

class TestQueryJira:
    @patch("plugins.memory.tiered.enrichment._get_jira_token", return_value="tok")
    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_successful_query(self, mock_run, mock_token):
        """Successful jira query with tab-separated output -> parses tasks using last field as status."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=_make_jira_stdout([
                ("MOL-101", "Fix auth timeout", "In Progress"),
                ("MOL-102", "Deploy canary", "To Do"),
            ]),
            stderr="",
        )

        tasks = _query_jira(timeout=5)

        assert len(tasks) == 2
        assert tasks[0]["key"] == "MOL-101"
        assert tasks[0]["summary"] == "Fix auth timeout"
        assert tasks[0]["status"] == "In Progress"
        assert tasks[1]["key"] == "MOL-102"
        assert tasks[1]["status"] == "To Do"

    @patch("plugins.memory.tiered.enrichment._get_jira_token", return_value="tok")
    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_malformed_single_column_line_skipped(self, mock_run, mock_token):
        """Line with a single column (no tabs) -> skipped as malformed."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="MOL-101\tFix auth\tIn Progress\nJUNK_LINE_NO_TABS\nMOL-102\tDeploy\tTo Do",
            stderr="",
        )

        tasks = _query_jira(timeout=5)

        assert len(tasks) == 2
        assert tasks[0]["key"] == "MOL-101"
        assert tasks[1]["key"] == "MOL-102"

    @patch("plugins.memory.tiered.enrichment._get_jira_token", return_value="tok")
    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_unrecognized_status_preserved(self, mock_run, mock_token):
        """Unrecognized status string (e.g. 'Awaiting QA') -> preserved as-is via parts[-1]."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="MOL-201\tReview compliance\tAwaiting QA",
            stderr="",
        )

        tasks = _query_jira(timeout=5)

        assert len(tasks) == 1
        assert tasks[0]["status"] == "Awaiting QA"

    @patch("plugins.memory.tiered.enrichment._get_jira_token", return_value="tok")
    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_no_results(self, mock_run, mock_token):
        """Exit code 1 with 'No result found' in stderr -> returns []."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="No result found for the given query",
        )

        assert _query_jira(timeout=5) == []

    @patch("plugins.memory.tiered.enrichment._get_jira_token", return_value="tok")
    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_timeout(self, mock_run, mock_token):
        """Subprocess timeout -> returns []."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="jira", timeout=5)

        assert _query_jira(timeout=5) == []

    @patch("plugins.memory.tiered.enrichment._get_jira_token", return_value="tok")
    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_jira_not_found(self, mock_run, mock_token):
        """jira binary not in PATH -> returns []."""
        mock_run.side_effect = FileNotFoundError("jira not found")

        assert _query_jira(timeout=5) == []

    @patch("plugins.memory.tiered.enrichment._get_jira_token", return_value="tok")
    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_auth_failure(self, mock_run, mock_token):
        """Exit code 1 with auth error -> returns []."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="401 Unauthorized: invalid credentials",
        )

        assert _query_jira(timeout=5) == []

    @patch("plugins.memory.tiered.enrichment._get_jira_token", return_value=None)
    def test_no_token(self, mock_token):
        """No Jira token available -> returns [] without calling subprocess."""
        assert _query_jira(timeout=5) == []


# ---------------------------------------------------------------------------
# _query_calendar
# ---------------------------------------------------------------------------

class TestQueryCalendar:
    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_successful_query(self, mock_run):
        """Successful gws JSON output -> parses events."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "count": 2,
                "events": [
                    {"calendar": "user@example.com", "start": "2026-04-07T09:00:00-04:00",
                     "end": "2026-04-07T10:00:00-04:00", "summary": "Standup", "location": ""},
                    {"calendar": "user@example.com", "start": "2026-04-07T14:00:00-04:00",
                     "end": "2026-04-07T15:00:00-04:00", "summary": "Sprint Review", "location": ""},
                ],
            }),
            stderr="",
        )

        events = _query_calendar(timeout=5)

        assert len(events) == 2
        assert events[0]["title"] == "Standup"
        assert events[0]["start"] == "09:00"
        assert events[1]["title"] == "Sprint Review"

    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_no_title_event(self, mock_run):
        """Event with empty summary -> shows (No title)."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"count": 1, "events": [
                {"calendar": "x", "start": "2026-04-07T10:00:00-04:00", "end": "", "summary": ""},
            ]}),
            stderr="",
        )

        events = _query_calendar(timeout=5)
        assert events[0]["title"] == "(No title)"

    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_all_day_event_preserves_date(self, mock_run):
        """All-day event (no 'T' in start) -> date string passed through as-is."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"count": 1, "events": [
                {"calendar": "x", "start": "2026-04-07", "end": "2026-04-08", "summary": "Company Holiday"},
            ]}),
            stderr="",
        )

        events = _query_calendar(timeout=5)

        assert len(events) == 1
        assert events[0]["title"] == "Company Holiday"
        assert events[0]["start"] == "2026-04-07"

    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_timeout(self, mock_run):
        """Subprocess timeout -> returns []."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="gws", timeout=5)
        assert _query_calendar(timeout=5) == []

    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_gws_not_found(self, mock_run):
        """gws binary not in PATH -> returns []."""
        mock_run.side_effect = FileNotFoundError("gws not found")
        assert _query_calendar(timeout=5) == []

    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_auth_failure(self, mock_run):
        """gws auth error -> returns []."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="401 auth failed")
        assert _query_calendar(timeout=5) == []

    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_invalid_json(self, mock_run):
        """Corrupt JSON output -> returns [] (JSONDecodeError path)."""
        mock_run.return_value = MagicMock(returncode=0, stdout="not json", stderr="")
        assert _query_calendar(timeout=5) == []

    @patch("plugins.memory.tiered.enrichment.subprocess.run")
    def test_type_error_from_json_loads(self, mock_run):
        """json.loads receives non-string (e.g., None stdout) raising TypeError -> returns []."""
        # json.loads(None) raises TypeError — this can happen if stdout is unexpectedly None
        mock_run.return_value = MagicMock(returncode=0, stdout=None, stderr="")
        assert _query_calendar(timeout=5) == []


# ---------------------------------------------------------------------------
# refresh_enrichment_cache
# ---------------------------------------------------------------------------

class TestRefreshEnrichmentCache:
    @patch("plugins.memory.tiered.enrichment._query_calendar", return_value=[])
    @patch("plugins.memory.tiered.enrichment._query_jira")
    def test_writes_valid_json(self, mock_jira, mock_cal, tmp_path):
        """Cache file contains valid JSON with expected keys."""
        mock_jira.return_value = [
            {"key": "MOL-1", "summary": "Task 1", "status": "To Do"},
        ]
        cache_path = tmp_path / "cache" / "enrichment.json"

        refresh_enrichment_cache(cache_path, timeout=5)

        assert cache_path.exists()
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert "refreshed_at" in data
        assert data["jira"] == [{"key": "MOL-1", "summary": "Task 1", "status": "To Do"}]
        assert data["calendar"] == []

    @patch("plugins.memory.tiered.enrichment._query_calendar")
    @patch("plugins.memory.tiered.enrichment._query_jira")
    def test_creates_parent_directories(self, mock_jira, mock_cal, tmp_path):
        """Parent directories created automatically."""
        mock_jira.return_value = [{"key": "MOL-1", "summary": "Placeholder", "status": "To Do"}]
        mock_cal.return_value = []
        cache_path = tmp_path / "deep" / "nested" / "enrichment.json"

        refresh_enrichment_cache(cache_path, timeout=5)

        assert cache_path.exists()
        assert cache_path.parent.exists()

    @patch("plugins.memory.tiered.enrichment._query_calendar", return_value=[])
    @patch("plugins.memory.tiered.enrichment._query_jira", return_value=[])
    def test_both_empty_does_not_write_cache(self, mock_jira, mock_cal, tmp_path):
        """If both sources return [], cache file is NOT written — allows retry next prefetch."""
        cache_path = tmp_path / "enrichment.json"

        refresh_enrichment_cache(cache_path, timeout=5)

        assert not cache_path.exists()

    @patch("plugins.memory.tiered.enrichment._query_calendar")
    @patch("plugins.memory.tiered.enrichment._query_jira", return_value=[])
    def test_calendar_only_writes_cache(self, mock_jira, mock_cal, tmp_path):
        """If only calendar returns data, cache is still written."""
        mock_cal.return_value = [{"title": "Standup", "start": "09:00", "end": ""}]
        cache_path = tmp_path / "enrichment.json"

        refresh_enrichment_cache(cache_path, timeout=5)

        assert cache_path.exists()
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert data["jira"] == []
        assert len(data["calendar"]) == 1

    @patch("plugins.memory.tiered.enrichment._query_calendar", return_value=[])
    @patch("plugins.memory.tiered.enrichment._query_jira")
    def test_non_empty_results_cached(self, mock_jira, mock_cal, tmp_path):
        """Non-empty Jira results are written to cache file."""
        mock_jira.return_value = [
            {"key": "MOL-1", "summary": "Fix auth", "status": "In Progress"},
        ]
        cache_path = tmp_path / "enrichment.json"

        refresh_enrichment_cache(cache_path, timeout=5)

        assert cache_path.exists()
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert len(data["jira"]) == 1
        assert "refreshed_at" in data

    @patch("plugins.memory.tiered.enrichment.as_completed")
    @patch("plugins.memory.tiered.enrichment.ThreadPoolExecutor")
    def test_timeout_error_from_as_completed(self, mock_pool_cls, mock_as_completed, tmp_path):
        """TimeoutError from as_completed -> writes partial results (empty = no cache)."""
        mock_as_completed.side_effect = TimeoutError("pool timed out")

        # Mock the pool context manager
        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.submit.return_value = MagicMock()
        mock_pool_cls.return_value = mock_pool

        cache_path = tmp_path / "enrichment.json"

        # Should not raise
        refresh_enrichment_cache(cache_path, timeout=5)

        # Both sources default to [] when pool times out, so no cache written
        assert not cache_path.exists()


# ---------------------------------------------------------------------------
# read_enrichment_cache
# ---------------------------------------------------------------------------

class TestReadEnrichmentCache:
    def test_valid_fresh_cache(self, tmp_path):
        """Fresh cache with valid JSON -> returns data dict."""
        cache_path = tmp_path / "enrichment.json"
        cache_data = _make_cache(
            jira=[{"key": "MOL-1", "summary": "Task", "status": "To Do"}],
        )
        cache_path.write_text(json.dumps(cache_data), encoding="utf-8")

        result = read_enrichment_cache(cache_path, max_age_minutes=30)

        assert result is not None
        assert result["jira"][0]["key"] == "MOL-1"
        assert "refreshed_at" in result

    def test_stale_cache(self, tmp_path):
        """Cache older than max_age_minutes -> returns None."""
        cache_path = tmp_path / "enrichment.json"
        cache_data = _make_cache()
        cache_path.write_text(json.dumps(cache_data), encoding="utf-8")
        # Backdate mtime by 2 hours
        old_time = time.time() - 7200
        os.utime(str(cache_path), (old_time, old_time))

        assert read_enrichment_cache(cache_path, max_age_minutes=30) is None

    def test_missing_file(self, tmp_path):
        """Non-existent cache file -> returns None."""
        cache_path = tmp_path / "does_not_exist.json"

        assert read_enrichment_cache(cache_path) is None

    def test_corrupt_json(self, tmp_path):
        """Invalid JSON in cache -> returns None."""
        cache_path = tmp_path / "enrichment.json"
        cache_path.write_text("{not valid json!!!", encoding="utf-8")

        assert read_enrichment_cache(cache_path) is None

    def test_missing_refreshed_at_key(self, tmp_path):
        """Valid JSON but missing refreshed_at -> returns None."""
        cache_path = tmp_path / "enrichment.json"
        cache_path.write_text(json.dumps({"jira": [], "calendar": []}), encoding="utf-8")

        assert read_enrichment_cache(cache_path) is None


# ---------------------------------------------------------------------------
# get_active_principles
# ---------------------------------------------------------------------------

class TestGetActivePrinciples:
    def test_returns_principle_titles(self):
        """Principles in DB -> returns list of title strings."""
        db = _mock_db_with_principles(["Ship fast", "Security first"])

        result = get_active_principles(db)

        assert result == ["Ship fast", "Security first"]
        db.query_active_principles.assert_called_once()

    def test_empty_result(self):
        """No principles in DB -> returns []."""
        db = _mock_db_with_principles([])

        assert get_active_principles(db) == []

    def test_db_operational_error(self):
        """sqlite3.OperationalError from DB -> returns [] gracefully."""
        db = MagicMock()
        db.query_active_principles.side_effect = sqlite3.OperationalError("database is locked")

        assert get_active_principles(db) == []

    def test_generic_exception_fallback(self):
        """Generic Exception from DB -> returns [] via fallback handler."""
        db = MagicMock()
        db.query_active_principles.side_effect = ValueError("unexpected error")

        assert get_active_principles(db) == []


# ---------------------------------------------------------------------------
# refresh_enrichment_async
# ---------------------------------------------------------------------------

class TestRefreshEnrichmentAsync:
    @patch("plugins.memory.tiered.enrichment.refresh_enrichment_cache")
    def test_spawns_daemon_thread(self, mock_refresh, tmp_path):
        """refresh_enrichment_async spawns a daemon thread targeting refresh_enrichment_cache."""
        cache_path = tmp_path / "enrichment.json"

        with patch("plugins.memory.tiered.enrichment.threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            refresh_enrichment_async(cache_path, timeout=5)

            mock_thread_cls.assert_called_once_with(
                target=mock_refresh,
                args=(cache_path, 5),
                daemon=True,
            )
            mock_thread.start.assert_called_once()


# ---------------------------------------------------------------------------
# compose_enrichment_block
# ---------------------------------------------------------------------------

class TestComposeEnrichmentBlock:
    def test_full_data(self):
        """Jira + calendar + principles -> block includes all sections."""
        cache = {
            "jira": [
                {"key": "MOL-1", "summary": "Fix auth", "status": "In Progress"},
                {"key": "MOL-2", "summary": "Deploy canary", "status": "To Do"},
            ],
            "calendar": [
                {"title": "Standup", "start": "09:00"},
                {"title": "Sprint review", "start": "15:00"},
            ],
        }
        principles = ["Ship fast", "Security first"]

        result = compose_enrichment_block(cache, principles)

        assert "## Situational context" in result
        assert "**Board (critical/high):**" in result
        assert "MOL-1: Fix auth [In Progress]" in result
        assert "MOL-2: Deploy canary [To Do]" in result
        assert "**Today's agenda:**" in result
        assert "Standup (09:00)" in result
        assert "Sprint review (15:00)" in result
        assert "**Active principles:**" in result
        assert "- Ship fast" in result
        assert "- Security first" in result

    def test_jira_only(self):
        """Only Jira tasks -> includes board section, no agenda or principles."""
        cache = {
            "jira": [{"key": "MOL-1", "summary": "Fix bug", "status": "To Do"}],
            "calendar": [],
        }

        result = compose_enrichment_block(cache, [])

        assert "**Board (critical/high):**" in result
        assert "MOL-1: Fix bug [To Do]" in result
        assert "**Today's agenda:**" not in result
        assert "**Active principles:**" not in result

    def test_principles_only(self):
        """Only principles -> includes principles section, no board or agenda."""
        cache = {"jira": [], "calendar": []}

        result = compose_enrichment_block(cache, ["Always test first"])

        assert "**Active principles:**" in result
        assert "- Always test first" in result
        assert "**Board (critical/high):**" not in result
        assert "**Today's agenda:**" not in result

    def test_empty_data_returns_empty_string(self):
        """No jira, no calendar, no principles -> returns ''."""
        cache = {"jira": [], "calendar": []}

        assert compose_enrichment_block(cache, []) == ""

    def test_truncation_at_max_chars(self):
        """Block exceeding MAX_ENRICHMENT_CHARS -> truncated at last newline with '\\n...'."""
        # Create enough tasks to exceed 1500 chars
        tasks = [
            {"key": f"MOL-{i}", "summary": f"Very long task description number {i} " * 5, "status": "In Progress"}
            for i in range(50)
        ]
        cache = {"jira": tasks, "calendar": []}

        result = compose_enrichment_block(cache, ["Principle " * 20])

        assert len(result) <= MAX_ENRICHMENT_CHARS
        assert result.endswith("\n...")

    def test_none_cache_handled(self):
        """cache=None with no principles -> returns ''."""
        assert compose_enrichment_block(None, []) == ""

    def test_none_cache_with_principles(self):
        """cache=None but principles present -> returns principles section."""
        result = compose_enrichment_block(None, ["Be kind"])

        assert "**Active principles:**" in result
        assert "- Be kind" in result

    def test_jira_task_without_status(self):
        """Task with empty status -> no status suffix in output."""
        cache = {
            "jira": [{"key": "MOL-1", "summary": "No status task", "status": ""}],
            "calendar": [],
        }

        result = compose_enrichment_block(cache, [])

        assert "MOL-1: No status task" in result
        # No trailing bracket for empty status
        assert "[" not in result.split("No status task")[1].split("\n")[0]

    def test_jira_tasks_capped_at_ten(self):
        """More than 10 Jira tasks -> only first 10 shown."""
        tasks = [
            {"key": f"MOL-{i}", "summary": f"Task {i}", "status": "To Do"}
            for i in range(15)
        ]
        cache = {"jira": tasks, "calendar": []}

        result = compose_enrichment_block(cache, [])

        assert "MOL-9" in result
        assert "MOL-10" not in result

    def test_principles_capped_at_five(self):
        """More than 5 principles -> only first 5 shown."""
        principles = [f"Principle {i}" for i in range(8)]

        result = compose_enrichment_block({"jira": [], "calendar": []}, principles)

        assert "Principle 4" in result
        assert "Principle 5" not in result

    def test_calendar_events_capped_at_five(self):
        """More than 5 calendar events -> only first 5 shown."""
        events = [
            {"title": f"Meeting {i}", "start": f"{9 + i}:00"}
            for i in range(8)
        ]
        cache = {"jira": [], "calendar": events}

        result = compose_enrichment_block(cache, [])

        assert "Meeting 4" in result
        assert "Meeting 5" not in result


# ---------------------------------------------------------------------------
# TestPrefetchEnrichment — integration tests for prefetch()
# ---------------------------------------------------------------------------

class TestPrefetchEnrichment:
    """Integration tests for TieredMemoryProvider.prefetch() enrichment flow."""

    def _make_provider(self, tmp_path):
        """Create a TieredMemoryProvider with mocked internals and a real hermes_home."""
        from plugins.memory.tiered import TieredMemoryProvider

        provider = TieredMemoryProvider.__new__(TieredMemoryProvider)
        provider._db = MagicMock()
        provider._session_id = "test"
        provider._sync_thread = None
        provider._hermes_home = str(tmp_path)
        provider._memory_dir = tmp_path / "memories"
        provider._memory_dir.mkdir(parents=True, exist_ok=True)
        return provider

    @patch("plugins.memory.tiered.maybe_update_hot_cache")
    @patch("plugins.memory.tiered.hybrid_search")
    @patch("plugins.memory.tiered.compose_enrichment_block")
    @patch("plugins.memory.tiered.get_active_principles")
    @patch("plugins.memory.tiered.refresh_enrichment_async")
    @patch("plugins.memory.tiered.read_enrichment_cache")
    def test_enrichment_block_prepended_to_recall(
        self, mock_read_cache, mock_refresh_async, mock_principles,
        mock_compose, mock_search, mock_hot_cache, tmp_path,
    ):
        """Enrichment block appears before recalled memory results."""
        provider = self._make_provider(tmp_path)

        mock_read_cache.return_value = {"jira": [{"key": "MOL-1", "summary": "T", "status": "To Do"}], "calendar": []}
        mock_principles.return_value = ["Ship fast"]
        mock_compose.return_value = "## Situational context\n**Board (critical/high):**\n- MOL-1: T [To Do]"
        mock_search.return_value = [
            {"id": "1", "title": "Meeting notes", "content": "Discussed Q4", "score": 0.85, "search_mode": "hybrid"},
        ]

        result = provider.prefetch("test query")

        assert result.startswith("## Situational context")
        assert "[Recalled from memory]" in result
        assert result.index("Situational context") < result.index("[Recalled from memory]")
        mock_refresh_async.assert_not_called()  # Cache was fresh

    @patch("plugins.memory.tiered.maybe_update_hot_cache")
    @patch("plugins.memory.tiered.hybrid_search")
    @patch("plugins.memory.tiered.compose_enrichment_block")
    @patch("plugins.memory.tiered.get_active_principles")
    @patch("plugins.memory.tiered.refresh_enrichment_async")
    @patch("plugins.memory.tiered.read_enrichment_cache")
    def test_stale_cache_triggers_async_refresh(
        self, mock_read_cache, mock_refresh_async, mock_principles,
        mock_compose, mock_search, mock_hot_cache, tmp_path,
    ):
        """When cache is stale (read returns None), refresh_enrichment_async is called."""
        provider = self._make_provider(tmp_path)

        mock_read_cache.return_value = None  # Stale or missing
        mock_principles.return_value = []
        mock_compose.return_value = ""
        mock_search.return_value = []

        provider.prefetch("test query")

        mock_refresh_async.assert_called_once()
        called_path = mock_refresh_async.call_args[0][0]
        assert str(called_path).endswith("cache/enrichment.json")

    @patch("plugins.memory.tiered.maybe_update_hot_cache")
    @patch("plugins.memory.tiered.hybrid_search")
    @patch("plugins.memory.tiered.compose_enrichment_block")
    @patch("plugins.memory.tiered.get_active_principles")
    @patch("plugins.memory.tiered.refresh_enrichment_async")
    @patch("plugins.memory.tiered.read_enrichment_cache")
    def test_enrichment_failure_does_not_break_recall(
        self, mock_read_cache, mock_refresh_async, mock_principles,
        mock_compose, mock_search, mock_hot_cache, tmp_path,
    ):
        """If enrichment compose raises, recall still works."""
        provider = self._make_provider(tmp_path)

        mock_read_cache.return_value = {"jira": [], "calendar": []}
        mock_principles.side_effect = Exception("DB locked")
        mock_compose.return_value = ""  # compose called with empty principles
        mock_search.return_value = [
            {"id": "1", "title": "Important note", "content": "Data here", "score": 0.9, "search_mode": "fts_only"},
        ]

        result = provider.prefetch("test query")

        assert "[Recalled from memory]" in result
        assert "Important note" in result

    @patch("plugins.memory.tiered.maybe_update_hot_cache")
    @patch("plugins.memory.tiered.hybrid_search")
    @patch("plugins.memory.tiered.compose_enrichment_block")
    @patch("plugins.memory.tiered.get_active_principles")
    @patch("plugins.memory.tiered.refresh_enrichment_async")
    @patch("plugins.memory.tiered.read_enrichment_cache")
    def test_empty_enrichment_with_results_still_works(
        self, mock_read_cache, mock_refresh_async, mock_principles,
        mock_compose, mock_search, mock_hot_cache, tmp_path,
    ):
        """Empty enrichment block + search results -> only recall section returned."""
        provider = self._make_provider(tmp_path)

        mock_read_cache.return_value = {"jira": [], "calendar": []}
        mock_principles.return_value = []
        mock_compose.return_value = ""  # Empty enrichment
        mock_search.return_value = [
            {"id": "1", "title": "Note", "content": "Content", "score": 0.7, "search_mode": "fts_only"},
        ]

        result = provider.prefetch("test query")

        assert "[Recalled from memory]" in result
        assert "## Situational context" not in result
