"""Tests for F4: atomic board.json writes, per-board lock, fail-closed toggle reads.

Covers write_board_metadata()/read_board_metadata() hardening:

* write_board_metadata() writes via the shared utils.atomic_write_text
  (temp file + fsync + os.replace), not a plain path.write_text().
* Concurrent write_board_metadata() calls on different fields, run under
  real threading, must not lose either update (per-board flock serializes
  the whole read-modify-write cycle).
* A present-but-corrupt/unparsable board.json (including invalid UTF-8
  bytes) fails CLOSED on the safety-relevant toggle fields
  (dispatch_enabled / auto_decompose_enabled / review_dispatch_enabled)
  and logs a WARNING naming the board slug.
* A genuinely missing board.json (new board, never written) still
  defaults those same toggles to True — regression guard against
  overcorrecting the fail-closed fix.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path

import pytest

# Ensure the worktree (not the stale global clone) is first on sys.path.
_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb

_SAFETY_TOGGLES = ("dispatch_enabled", "auto_decompose_enabled", "review_dispatch_enabled")


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with no prior kanban state (mirrors test_kanban_boards.py)."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    return home


class TestAtomicWrite:
    def test_write_board_metadata_never_leaves_a_tmp_file_behind(self, fresh_home):
        kb.write_board_metadata("atomic-board", name="Atomic Board")
        path = kb.board_metadata_path("atomic-board")
        assert path.exists()
        # No leftover temp artifacts from the atomic_write_text temp+rename dance.
        leftovers = [p for p in path.parent.iterdir() if p.name.startswith(".board_")]
        assert leftovers == []

    def test_write_board_metadata_content_round_trips(self, fresh_home):
        kb.write_board_metadata("atomic-board", name="Atomic Board", description="d")
        raw = json.loads(kb.board_metadata_path("atomic-board").read_text(encoding="utf-8"))
        assert raw["name"] == "Atomic Board"
        assert raw["description"] == "d"


class TestConcurrentWrites:
    def test_concurrent_writes_on_different_fields_both_survive(self, fresh_home):
        """Two threads each set a different field; under a real flock-serialized
        read-modify-write cycle, neither update should be lost to the other."""
        slug = "concurrent-board"
        kb.write_board_metadata(slug, name="Initial")

        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def set_description():
            try:
                barrier.wait(timeout=5)
                for _ in range(25):
                    kb.write_board_metadata(slug, description="from-thread-A")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def set_icon():
            try:
                barrier.wait(timeout=5)
                for _ in range(25):
                    kb.write_board_metadata(slug, icon="from-thread-B")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=set_description)
        t2 = threading.Thread(target=set_icon)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not errors, f"threads raised: {errors}"
        final = kb.read_board_metadata(slug)
        # Both fields must reflect the last writer's value — neither thread's
        # entire write was lost to a torn/unlocked read-modify-write race.
        assert final["description"] == "from-thread-A"
        assert final["icon"] == "from-thread-B"
        # name from the initial write must not have been clobbered to blank.
        assert final["name"] == "Initial"

    def test_many_concurrent_writers_no_corruption(self, fresh_home):
        """Heavier fan-out: N threads hammering write_board_metadata concurrently
        must always leave a parseable board.json — no torn/interleaved writes."""
        slug = "hammer-board"
        kb.write_board_metadata(slug, name="Hammer")

        def worker(n):
            for i in range(10):
                kb.write_board_metadata(slug, description=f"w{n}-{i}")

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # Must still parse cleanly after the storm.
        raw = json.loads(kb.board_metadata_path(slug).read_text(encoding="utf-8"))
        assert isinstance(raw, dict)
        assert raw["name"] == "Hammer"


class TestFailClosedOnCorruption:
    def test_missing_board_json_defaults_toggles_true(self, fresh_home):
        """Regression guard: a genuinely new/never-written board must still
        default safety toggles to True (inherit-global), not fail closed."""
        meta = kb.read_board_metadata("brand-new-board")
        assert not kb.board_metadata_path("brand-new-board").exists()
        for field in _SAFETY_TOGGLES:
            assert meta[field] is True, field

    def test_corrupt_json_fails_closed_and_warns(self, fresh_home, caplog):
        slug = "corrupt-board"
        path = kb.board_metadata_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json!!", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            meta = kb.read_board_metadata(slug)

        for field in _SAFETY_TOGGLES:
            assert meta[field] is False, field
        assert any(
            slug in rec.message or slug in str(rec.args)
            for rec in caplog.records
            if rec.levelno == logging.WARNING
        )

    def test_invalid_utf8_bytes_fails_closed_and_warns(self, fresh_home, caplog):
        """Torn write leaving invalid UTF-8 must be treated as corruption too,
        not raise out of read_board_metadata()."""
        slug = "torn-write-board"
        path = kb.board_metadata_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Invalid UTF-8 continuation byte with no valid lead byte.
        path.write_bytes(b'{"name": "x", "broken": "\xff\xfe"')

        with caplog.at_level(logging.WARNING):
            meta = kb.read_board_metadata(slug)

        for field in _SAFETY_TOGGLES:
            assert meta[field] is False, field
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)
        # Must never raise / must still synthesize a usable dict.
        assert meta["slug"] == slug

    def test_empty_file_fails_closed(self, fresh_home, caplog):
        """An empty board.json (classic torn-write artifact) is a parse
        failure, not treated the same as 'file does not exist'."""
        slug = "empty-board"
        path = kb.board_metadata_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            meta = kb.read_board_metadata(slug)

        for field in _SAFETY_TOGGLES:
            assert meta[field] is False, field
