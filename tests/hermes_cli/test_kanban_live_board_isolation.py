"""Pins for the kanban live-board guard in ``hermes_cli.kanban_db``.

A guard in ``tests/conftest.py`` alone is bypassable, because a shim that
fakes ``pytest`` in ``sys.modules`` never loads conftest. These pins drive
the real guard at the ``connect()`` and ``init_db()`` chokepoint, including
the shim case and a remapped ``HOME``. Background in PR #101997.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# The live-board predicate: shape recognition
# ---------------------------------------------------------------------------

class TestLiveBoardPredicate:
    """``_is_production_kanban_db`` must match real boards, not scratch ones."""

    def test_matches_the_default_board(self):
        root = Path("/home/someone/.hermes")
        assert kb._is_production_kanban_db(root / "kanban.db", root)

    def test_matches_a_named_board(self):
        root = Path("/home/someone/.hermes")
        target = root / "kanban" / "boards" / "atm10" / "kanban.db"
        assert kb._is_production_kanban_db(target, root)

    def test_does_not_match_a_task_workspace_board(self):
        """Workers create throwaway boards there, so they must stay writable."""
        root = Path("/home/someone/.hermes")
        target = root / "kanban" / "workspaces" / "t_abc123" / "kanban.db"
        assert not kb._is_production_kanban_db(target, root)

    def test_does_not_match_an_unrelated_path(self):
        root = Path("/home/someone/.hermes")
        assert not kb._is_production_kanban_db(Path("/tmp/x/kanban.db"), root)


# ---------------------------------------------------------------------------
# Real-root resolution must survive a remapped HOME
# ---------------------------------------------------------------------------

class TestRealRootResolution:
    """The deny-list must find the real root even when HOME is remapped."""

    def test_resolves_root_from_profile_mode_home(self, monkeypatch, tmp_path):
        """``<root>/profiles/<name>/home`` must resolve to ``<root>``.

        The layout a dispatched worker runs under.
        """
        root = tmp_path / ".hermes"
        worker_home = root / "profiles" / "scotty" / "home"
        worker_home.mkdir(parents=True)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.setattr(os.path, "expanduser", lambda p: str(worker_home))

        roots = kb._real_kanban_roots()

        assert root.resolve() in roots, (
            f"profile-mode HOME {worker_home} did not resolve to root {root}; "
            f"got {roots}. The deny-list would protect nothing."
        )

    def test_honors_hermes_real_home(self, monkeypatch, tmp_path):
        real = tmp_path / "operator"
        (real / ".hermes").mkdir(parents=True)
        monkeypatch.setenv("HERMES_REAL_HOME", str(real))

        assert (real / ".hermes").resolve() in kb._real_kanban_roots()


# ---------------------------------------------------------------------------
# Shim detection
# ---------------------------------------------------------------------------

class TestShimDetection:
    def test_a_fake_pytest_module_counts_as_test_context(self, monkeypatch):
        """A shim must install a fake ``pytest`` for decorators to import."""
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("PYTEST_VERSION", raising=False)
        monkeypatch.delenv("HERMES_TEST_ISOLATION", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_DB_GUARD_BYPASS", raising=False)

        # 'pytest' is genuinely in sys.modules here (we are pytest), which is
        # the same observable state a shim creates.
        assert kb._guard_in_test_context()

    def test_bypass_env_disarms_the_guard(self, monkeypatch):
        """An explicit escape hatch must exist, mirroring hermes_state."""
        monkeypatch.setenv("HERMES_KANBAN_DB_GUARD_BYPASS", "1")
        assert not kb._guard_in_test_context()


# ---------------------------------------------------------------------------
# The load-bearing pin: connect() refuses the live board
# ---------------------------------------------------------------------------

class TestConnectRefusesLiveBoard:
    def test_connect_refuses_a_board_under_the_real_root(
        self, monkeypatch, tmp_path
    ):
        """Fake operator root, remapped HOME, board path at ``<root>/kanban.db``."""
        root = tmp_path / ".hermes"
        worker_home = root / "profiles" / "scotty" / "home"
        worker_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_REAL_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_KANBAN_DB_GUARD_BYPASS", raising=False)

        live_board = root / "kanban.db"

        with pytest.raises(RuntimeError, match="live-board guard"):
            kb.connect(db_path=live_board)

        assert not live_board.exists(), (
            "the guard must refuse BEFORE the board file is created"
        )

    def test_init_db_refuses_a_board_under_the_real_root(
        self, monkeypatch, tmp_path
    ):
        """``init_db`` is a public entry point and must not be a way around."""
        root = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_REAL_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_KANBAN_DB_GUARD_BYPASS", raising=False)

        with pytest.raises(RuntimeError, match="live-board guard"):
            kb.init_db(db_path=root / "kanban.db")

    def test_a_sandboxed_board_still_works(self, monkeypatch, tmp_path):
        """The guard must not break hermetic tests, so no false positives."""
        monkeypatch.setenv("HERMES_REAL_HOME", str(tmp_path / "operator"))
        sandbox = tmp_path / "sandbox" / "kanban.db"

        conn = kb.connect(db_path=sandbox)
        try:
            assert sandbox.exists()
            task_id = kb.create_task(
                conn, title="sandboxed card", assignee="nobody"
            )
            assert task_id
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# AC3: a fixture-created card lands in tmp, live board row count unchanged
# ---------------------------------------------------------------------------

def _live_board_row_count() -> tuple[Path, int | None]:
    """Read the real board's count over read-only sqlite.

    Bypasses ``kanban_db`` so the fixtures under test cannot redirect this
    measurement.
    """
    roots = kb._real_kanban_roots()
    for root in roots:
        candidate = root / "kanban.db"
        if candidate.exists():
            uri = f"file:{candidate}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            try:
                n = conn.execute("select count(*) from tasks").fetchone()[0]
                return candidate, int(n)
            finally:
                conn.close()
    return (roots[0] if roots else Path("/nonexistent")), None


class TestFixtureCardsLandInTmp:
    def test_card_lands_in_tmp_and_live_board_is_untouched(self, tmp_path):
        """Measure the live board rather than assert isolation.

        A tmp board that is created and then ignored passes either way, so
        only a real-board measurement distinguishes the two.
        """
        live_path, before = _live_board_row_count()
        if before is None:
            pytest.skip("no live board on this host, nothing to measure")

        sandbox = tmp_path / "board" / "kanban.db"
        conn = kb.connect(db_path=sandbox)
        try:
            task_id = kb.create_task(
                conn,
                title="isolation pin card",
                assignee="nobody",
                body="created by test_kanban_live_board_isolation",
            )
            rows = conn.execute("select count(*) from tasks").fetchone()[0]
        finally:
            conn.close()

        assert task_id
        assert rows == 1, f"card did not land in the tmp board (rows={rows})"

        _, after = _live_board_row_count()
        assert after == before, (
            f"live board mutated: {live_path} went from {before} to {after} "
            f"rows. Test isolation is broken and cards minted there are "
            f"dispatchable."
        )

    def test_resolved_board_is_never_the_live_board_under_conftest(self):
        """The autouse hermetic conftest must not resolve to production."""
        resolved = kb.kanban_db_path().resolve()
        for root in kb._real_kanban_roots():
            assert not kb._is_production_kanban_db(resolved, root), (
                f"conftest left the board resolving to production: {resolved}"
            )
