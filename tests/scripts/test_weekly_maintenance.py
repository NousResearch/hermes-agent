"""Unit tests for ``scripts/weekly_maintenance.py``.

Pin the contract of every helper in isolation:

  * :func:`resolve_paths` — every Hermes path MUST come from
    ``hermes_constants.get_hermes_home()``; never from ``Path.home() /
    ".hermes"``. This is the heart of the #24035 fix.
  * :func:`vacuum_state_db` — graceful skip on missing DB, dry-run is
    inert, real VACUUM shrinks the file.
  * :func:`prune_snapshots` — mtime-based eviction, retention boundary,
    dry-run inertness.
  * :func:`rotate_logs` — gzip threshold, archive-pruning horizon.
  * :func:`_select_phases` / ``main()`` — argparse + exit codes.

The end-to-end integration anchor that re-runs the whole driver under
a synthetic ``HERMES_HOME`` lives in
``tests/scripts/test_weekly_maintenance_integration.py``.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "weekly_maintenance.py"


@pytest.fixture(scope="module")
def weekly_module() -> Any:
    """Import ``scripts/weekly_maintenance.py`` as a module."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "weekly_maintenance_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader, "spec_from_file_location returned no loader"
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# resolve_paths — the #24035 fix
# ---------------------------------------------------------------------------


class TestResolvePaths:
    def test_resolves_from_get_hermes_home_not_path_home(
        self, weekly_module, tmp_path
    ):
        """The script MUST consult get_hermes_home, never Path.home()."""
        target = tmp_path / "hermes-work"
        target.mkdir()

        with patch.object(
            weekly_module, "get_hermes_home", return_value=target
        ) as mock_home:
            paths = weekly_module.resolve_paths()

        mock_home.assert_called_once()
        assert paths.home == target.resolve()
        assert paths.state_db == target.resolve() / "state.db"
        assert paths.snapshots_dir == target.resolve() / "state-snapshots"
        assert paths.logs_dir == target.resolve() / "logs"

    def test_explicit_home_override_wins_for_tests(self, weekly_module, tmp_path):
        """``home=`` parameter exists for tests — pin that contract."""
        with patch.object(weekly_module, "get_hermes_home") as mock_home:
            paths = weekly_module.resolve_paths(home=tmp_path)
        mock_home.assert_not_called()
        assert paths.home == tmp_path.resolve()
        assert paths.state_db == tmp_path.resolve() / "state.db"

    def test_paths_object_is_immutable(self, weekly_module, tmp_path):
        """``MaintenancePaths`` is a frozen dataclass — verify by attempting
        a mutation. If this ever stops raising, the path-resolution contract
        is no longer locked.
        """
        paths = weekly_module.resolve_paths(home=tmp_path)
        with pytest.raises((AttributeError, TypeError, Exception)):
            paths.state_db = tmp_path / "wrong.db"  # type: ignore[misc]

    def test_to_json_roundtrip_contains_every_path(
        self, weekly_module, tmp_path
    ):
        paths = weekly_module.resolve_paths(home=tmp_path)
        payload = json.loads(paths.to_json())
        assert set(payload.keys()) == {
            "home", "state_db", "snapshots_dir", "logs_dir",
        }
        for key, value in payload.items():
            assert value.startswith(str(tmp_path)), (
                f"{key} = {value!r} does not start with the override home "
                "— resolve_paths is leaking a non-profile path"
            )

    @pytest.mark.parametrize("subpath", ["state.db", "state-snapshots", "logs"])
    def test_paths_use_get_hermes_home_subdirectories(
        self, weekly_module, tmp_path, subpath
    ):
        paths = weekly_module.resolve_paths(home=tmp_path)
        attr = {
            "state.db":         "state_db",
            "state-snapshots":  "snapshots_dir",
            "logs":             "logs_dir",
        }[subpath]
        assert getattr(paths, attr) == tmp_path.resolve() / subpath


# ---------------------------------------------------------------------------
# vacuum_state_db
# ---------------------------------------------------------------------------


class TestVacuumStateDb:
    def test_missing_db_skips_cleanly(self, weekly_module, tmp_path):
        paths = weekly_module.resolve_paths(home=tmp_path)
        result = weekly_module.vacuum_state_db(paths, dry_run=False)
        assert result.skipped is True
        assert result.error is None
        assert any("not found" in line for line in result.details)

    def test_dry_run_does_not_modify_db(self, weekly_module, tmp_path):
        paths = weekly_module.resolve_paths(home=tmp_path)
        db = paths.state_db
        self._make_db_with_garbage(db, rows=200)
        size_before = db.stat().st_size

        result = weekly_module.vacuum_state_db(paths, dry_run=True)
        assert result.error is None
        assert result.skipped is True
        assert db.stat().st_size == size_before, (
            "dry-run modified the DB — that's a safety regression"
        )

    def test_real_vacuum_shrinks_db(self, weekly_module, tmp_path):
        paths = weekly_module.resolve_paths(home=tmp_path)
        db = paths.state_db
        self._make_db_with_garbage(db, rows=2000)
        # Bloat the file by deleting all rows (SQLite keeps the pages).
        with sqlite3.connect(str(db)) as c:
            c.execute("DELETE FROM junk")
            c.commit()
        size_before = db.stat().st_size

        result = weekly_module.vacuum_state_db(paths, dry_run=False)
        assert result.error is None, result.error
        assert result.skipped is False
        size_after = db.stat().st_size
        assert size_after < size_before, (
            "VACUUM did not reclaim disk — fix path is broken"
        )

    @staticmethod
    def _make_db_with_garbage(path: Path, *, rows: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(path)) as c:
            c.execute("CREATE TABLE junk (k INTEGER PRIMARY KEY, v TEXT)")
            c.executemany(
                "INSERT INTO junk(v) VALUES (?)",
                [("x" * 256,) for _ in range(rows)],
            )
            c.commit()


# ---------------------------------------------------------------------------
# prune_snapshots
# ---------------------------------------------------------------------------


class TestPruneSnapshots:
    def test_missing_dir_skips_cleanly(self, weekly_module, tmp_path):
        paths = weekly_module.resolve_paths(home=tmp_path)
        result = weekly_module.prune_snapshots(
            paths, retention_days=30, dry_run=False
        )
        assert result.skipped is True

    def test_old_snapshots_are_removed(self, weekly_module, tmp_path):
        paths = weekly_module.resolve_paths(home=tmp_path)
        snapshots = paths.snapshots_dir
        snapshots.mkdir(parents=True)
        old = snapshots / "2024-01-01-old"
        old.mkdir()
        (old / "manifest.json").write_text("{}")
        fresh = snapshots / "2026-05-08-fresh"
        fresh.mkdir()
        (fresh / "manifest.json").write_text("{}")

        now = time.time()
        # Force ``old`` to look 100 days stale.
        for entry in old.rglob("*"):
            entry.touch()
        import os
        os.utime(old, (now - 100 * 86400, now - 100 * 86400))

        result = weekly_module.prune_snapshots(
            paths, retention_days=30, dry_run=False, now=now
        )
        assert result.error is None
        assert not old.exists(), "old snapshot was not removed"
        assert fresh.exists(), "fresh snapshot must be kept"

    def test_dry_run_keeps_old_snapshots(self, weekly_module, tmp_path):
        paths = weekly_module.resolve_paths(home=tmp_path)
        paths.snapshots_dir.mkdir(parents=True)
        old = paths.snapshots_dir / "old"
        old.mkdir()
        import os
        now = time.time()
        os.utime(old, (now - 100 * 86400, now - 100 * 86400))

        result = weekly_module.prune_snapshots(
            paths, retention_days=30, dry_run=True, now=now
        )
        assert result.error is None
        assert old.exists(), "dry-run removed a snapshot — that's a regression"
        assert any("would remove" in line for line in result.details)

    def test_retention_boundary_keeps_snapshots_younger_than_cutoff(
        self, weekly_module, tmp_path
    ):
        """A snapshot exactly *retention_days - 1 day* old must be kept."""
        paths = weekly_module.resolve_paths(home=tmp_path)
        paths.snapshots_dir.mkdir(parents=True)
        recent = paths.snapshots_dir / "recent"
        recent.mkdir()
        import os
        now = time.time()
        os.utime(recent, (now - 29 * 86400, now - 29 * 86400))

        result = weekly_module.prune_snapshots(
            paths, retention_days=30, dry_run=False, now=now
        )
        assert result.error is None
        assert recent.exists()


# ---------------------------------------------------------------------------
# rotate_logs
# ---------------------------------------------------------------------------


class TestRotateLogs:
    def test_missing_dir_skips_cleanly(self, weekly_module, tmp_path):
        paths = weekly_module.resolve_paths(home=tmp_path)
        result = weekly_module.rotate_logs(
            paths, retention_days=30, dry_run=False
        )
        assert result.skipped is True

    def test_small_log_is_not_rotated(self, weekly_module, tmp_path):
        paths = weekly_module.resolve_paths(home=tmp_path)
        paths.logs_dir.mkdir(parents=True)
        small = paths.logs_dir / "agent.log"
        small.write_text("only a few bytes")

        result = weekly_module.rotate_logs(
            paths, retention_days=30, dry_run=False
        )
        assert result.error is None
        assert small.exists()
        assert not list(paths.logs_dir.glob("agent.log.*.gz"))

    def test_large_log_is_gzipped_and_removed(self, weekly_module, tmp_path):
        paths = weekly_module.resolve_paths(home=tmp_path)
        paths.logs_dir.mkdir(parents=True)
        big = paths.logs_dir / "gateway.log"
        big.write_bytes(b"0" * (11 * 1024 * 1024))

        result = weekly_module.rotate_logs(
            paths, retention_days=30, dry_run=False
        )
        assert result.error is None, result.error
        assert not big.exists(), "rotated log was not removed"
        archives = list(paths.logs_dir.glob("gateway.log.*.gz"))
        assert len(archives) == 1
        with gzip.open(archives[0], "rb") as f:
            content = f.read()
        assert content == b"0" * (11 * 1024 * 1024)

    def test_dry_run_does_not_rotate(self, weekly_module, tmp_path):
        paths = weekly_module.resolve_paths(home=tmp_path)
        paths.logs_dir.mkdir(parents=True)
        big = paths.logs_dir / "gateway.log"
        big.write_bytes(b"0" * (11 * 1024 * 1024))

        result = weekly_module.rotate_logs(
            paths, retention_days=30, dry_run=True
        )
        assert result.error is None
        assert big.exists()
        assert not list(paths.logs_dir.glob("gateway.log.*.gz"))
        assert any("would rotate" in line for line in result.details)

    def test_old_archives_are_pruned(self, weekly_module, tmp_path):
        paths = weekly_module.resolve_paths(home=tmp_path)
        paths.logs_dir.mkdir(parents=True)
        archive = paths.logs_dir / "old.log.123.gz"
        archive.write_bytes(b"\x1f\x8b\x08\x00")  # gzip header bytes
        import os
        now = time.time()
        os.utime(archive, (now - 100 * 86400, now - 100 * 86400))

        result = weekly_module.rotate_logs(
            paths, retention_days=30, dry_run=False, now=now
        )
        assert result.error is None
        assert not archive.exists()


# ---------------------------------------------------------------------------
# Driver — _select_phases + main()
# ---------------------------------------------------------------------------


class TestDriver:
    def test_select_all_phases_by_default(self, weekly_module):
        assert list(weekly_module._select_phases(None)) == [
            "vacuum", "snapshots", "logs",
        ]

    @pytest.mark.parametrize("phase", ["vacuum", "snapshots", "logs"])
    def test_select_single_phase(self, weekly_module, phase):
        assert list(weekly_module._select_phases(phase)) == [phase]

    def test_select_unknown_phase_exits(self, weekly_module):
        with pytest.raises(SystemExit):
            list(weekly_module._select_phases("nonexistent"))

    def test_human_bytes_is_compact(self, weekly_module):
        h = weekly_module._human_bytes
        assert h(0) == "0 B"
        assert h(1023) == "1023 B"
        assert h(2048) == "2.0 KB"
        assert h(5 * 1024 * 1024) == "5.0 MB"

    def test_main_returns_zero_on_dry_run(self, weekly_module, tmp_path, capsys):
        with patch.object(
            weekly_module, "get_hermes_home", return_value=tmp_path
        ):
            rc = weekly_module.main(["--dry-run"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "DRY-RUN" in captured.out
        assert str(tmp_path) in captured.out

    def test_main_rejects_zero_retention(self, weekly_module):
        with pytest.raises(SystemExit):
            weekly_module.main(["--retention-days", "0"])

    def test_main_rejects_negative_retention(self, weekly_module):
        with pytest.raises(SystemExit):
            weekly_module.main(["--retention-days", "-1"])

    def test_main_returns_one_on_phase_error(self, weekly_module, tmp_path):
        paths = weekly_module.resolve_paths(home=tmp_path)
        paths.state_db.parent.mkdir(parents=True, exist_ok=True)
        paths.state_db.write_bytes(b"not-a-sqlite-file")

        with patch.object(
            weekly_module, "get_hermes_home", return_value=tmp_path
        ):
            rc = weekly_module.main(["--only", "vacuum"])
        assert rc == 1
