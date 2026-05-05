"""Tests for SessionStore.reconcile_orphaned_json_files().

Regression tests for https://github.com/NousResearch/hermes-agent/issues/20098

Gateway session storage creates a new session JSON/JSONL file on every
restart without cleaning up old ones, causing the ``sessions.json`` index to
desync and cross-session tool-call-id contamination.

The fix adds ``reconcile_orphaned_json_files()`` which removes files whose
basename (session_id) is no longer tracked in the in-memory ``_entries``
index.  It is called automatically during ``start_gateway()`` startup.
"""

import json
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.session import SessionEntry, SessionSource, SessionStore, Platform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow():
    return datetime.now(timezone.utc)


def _make_store(sessions_dir: Path) -> SessionStore:
    """Build a minimal SessionStore pointed at a temp directory."""
    config = MagicMock()
    config.group_sessions_per_user = True
    config.thread_sessions_per_user = False

    def _no_reset(entry, source):
        return None

    config.get_reset_policy.return_value = MagicMock(
        type="never", inactivity_timeout_seconds=None
    )

    store = SessionStore.__new__(SessionStore)
    store.sessions_dir = sessions_dir
    store.config = config
    store._entries = {}
    store._loaded = False
    store._lock = threading.Lock()
    store._has_active_processes_fn = None
    store._db = None
    return store


def _write_sessions_json(sessions_dir: Path, entries: dict) -> None:
    """Write a sessions.json index with the given {key: entry_dict} mapping."""
    (sessions_dir / "sessions.json").write_text(
        json.dumps(entries, indent=2), encoding="utf-8"
    )


def _make_entry_dict(session_id: str) -> dict:
    """Return a minimal SessionEntry dict for use in sessions.json."""
    now_iso = _utcnow().isoformat()
    return {
        "session_key": f"key_{session_id}",
        "session_id": session_id,
        "created_at": now_iso,
        "updated_at": now_iso,
        "platform": "cli",
        "chat_type": "private",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReconcileOrphanedJsonFiles:
    """SessionStore.reconcile_orphaned_json_files() removes untracked files."""

    def test_removes_orphaned_json_file(self, tmp_path):
        """A .json file whose session_id is not in the index is deleted."""
        orphan_id = "20260505_020304_deadbeef"
        orphan_file = tmp_path / f"{orphan_id}.json"
        orphan_file.write_text("{}", encoding="utf-8")

        # sessions.json tracks a *different* session
        live_id = "20260505_010101_cafebabe"
        _write_sessions_json(tmp_path, {f"key_{live_id}": _make_entry_dict(live_id)})
        # Also create the live file so the directory is realistic
        (tmp_path / f"{live_id}.json").write_text("{}", encoding="utf-8")

        store = _make_store(tmp_path)
        removed = store.reconcile_orphaned_json_files()

        assert removed == 1
        assert not orphan_file.exists(), "orphaned file should have been deleted"
        assert (tmp_path / f"{live_id}.json").exists(), "live file should be kept"

    def test_removes_orphaned_jsonl_file(self, tmp_path):
        """A .jsonl transcript whose session_id is not in the index is deleted."""
        orphan_id = "20260505_020304_aabbccdd"
        orphan_file = tmp_path / f"{orphan_id}.jsonl"
        orphan_file.write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")

        _write_sessions_json(tmp_path, {})  # empty index — no live sessions

        store = _make_store(tmp_path)
        removed = store.reconcile_orphaned_json_files()

        assert removed == 1
        assert not orphan_file.exists()

    def test_noop_when_no_orphans(self, tmp_path):
        """Returns 0 and deletes nothing when every file is tracked."""
        live_id = "20260505_010101_aabbccdd"
        live_file = tmp_path / f"{live_id}.json"
        live_file.write_text("{}", encoding="utf-8")

        _write_sessions_json(tmp_path, {f"key_{live_id}": _make_entry_dict(live_id)})

        store = _make_store(tmp_path)
        removed = store.reconcile_orphaned_json_files()

        assert removed == 0
        assert live_file.exists()

    def test_never_deletes_sessions_json_index(self, tmp_path):
        """The sessions.json index itself must never be removed."""
        _write_sessions_json(tmp_path, {})

        store = _make_store(tmp_path)
        store.reconcile_orphaned_json_files()

        assert (tmp_path / "sessions.json").exists(), \
            "sessions.json index must not be deleted by orphan cleanup"

    def test_never_deletes_temp_files(self, tmp_path):
        """Temp files written by _save() (prefixed with '.') are left alone."""
        tmp_file = tmp_path / ".sessions_abc123.tmp"
        tmp_file.write_text("{}", encoding="utf-8")

        _write_sessions_json(tmp_path, {})

        store = _make_store(tmp_path)
        removed = store.reconcile_orphaned_json_files()

        assert removed == 0
        assert tmp_file.exists(), "dot-prefixed temp file should not be removed"

    def test_multiple_orphans_all_removed(self, tmp_path):
        """All orphaned files from repeated restarts are deleted at once."""
        orphan_ids = [
            "20260505_053804_914eb0",
            "20260505_054636_3956ed",
            "20260505_055014_52e486",
            "20260505_055040_43da0617",
            "20260505_061146_1acbd7",
            "20260505_062453_2804db",
            "20260505_063027_d6ec06",
        ]
        for oid in orphan_ids:
            (tmp_path / f"{oid}.json").write_text("{}", encoding="utf-8")

        # Only one live session (the current one)
        live_id = "20260505_063204_f24f96d5"
        (tmp_path / f"{live_id}.json").write_text("{}", encoding="utf-8")
        _write_sessions_json(tmp_path, {f"key_{live_id}": _make_entry_dict(live_id)})

        store = _make_store(tmp_path)
        removed = store.reconcile_orphaned_json_files()

        assert removed == 7
        for oid in orphan_ids:
            assert not (tmp_path / f"{oid}.json").exists(), \
                f"orphan {oid} should have been deleted"
        assert (tmp_path / f"{live_id}.json").exists(), "live session kept"

    def test_empty_sessions_dir_is_safe(self, tmp_path):
        """Calling on an empty directory returns 0 without errors."""
        store = _make_store(tmp_path)
        removed = store.reconcile_orphaned_json_files()
        assert removed == 0

    def test_skips_unknown_file_extensions(self, tmp_path):
        """Non-.json/.jsonl files (e.g. request_dump_*) are not touched."""
        dump_file = tmp_path / "request_dump_20260505_abc.bin"
        dump_file.write_bytes(b"\x00\x01")

        _write_sessions_json(tmp_path, {})

        store = _make_store(tmp_path)
        removed = store.reconcile_orphaned_json_files()

        assert removed == 0
        assert dump_file.exists()

    def test_returns_zero_when_no_sessions_json(self, tmp_path):
        """If sessions.json does not exist, no tracked entries → all files are orphans."""
        orphan_id = "20260505_aabbccdd_11223344"
        orphan_file = tmp_path / f"{orphan_id}.json"
        orphan_file.write_text("{}", encoding="utf-8")

        # No sessions.json written → _entries stays empty → file is an orphan
        store = _make_store(tmp_path)
        removed = store.reconcile_orphaned_json_files()

        assert removed == 1
        assert not orphan_file.exists()
