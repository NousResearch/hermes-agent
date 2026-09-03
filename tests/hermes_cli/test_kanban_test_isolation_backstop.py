"""Hard test-isolation backstop for the t_83bfe788 incident (BUI-942 item 4).

When ``PYTEST_CURRENT_TEST`` is set, opening/creating/migrating a Kanban
board whose resolved target is the *real* Hermes root / live board must be
refused loudly, even if ``HERMES_KANBAN_DB`` / ``HERMES_KANBAN_HOME`` were
inherited to point there. Temp isolated test roots must still open.

These tests must NEVER touch the real board: the injected-root cases point
the "real root" resolver at a tmp dir so a broken guard can only ever
create files under tmp, and the pure-path assertions never call connect().
"""

from __future__ import annotations

import os
import sys as _sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def test_real_board_detection_targets_real_home_kanban_area():
    """The real-root resolver anchors on the real OS home, independent of any
    ``Path.home`` monkeypatch or ``HERMES_*`` override, and flags the default
    board db plus anything under ``<root>/kanban/`` as the live board."""
    real_home = Path(os.path.expanduser("~"))
    roots = kb._real_hermes_kanban_roots()
    assert any(r == (real_home / ".hermes") for r in roots), (
        f"expected {real_home / '.hermes'} in real roots {roots}"
    )
    real_root = real_home / ".hermes"
    # Default board db and named boards under kanban/ are the live board.
    assert kb._is_real_live_board_path(real_root / "kanban.db")
    assert kb._is_real_live_board_path(real_root / "kanban" / "boards" / "x" / "kanban.db")
    # A sibling worktree path under the real root that is NOT the kanban data
    # area must not be misclassified (avoids false positives for repos checked
    # out under ~/.hermes/worktrees/...).
    assert not kb._is_real_live_board_path(real_root / "worktrees" / "wt" / "kanban.db")


def test_connect_refuses_real_board_under_pytest(tmp_path, monkeypatch):
    """With PYTEST_CURRENT_TEST set, connecting to a path inside the real
    kanban area raises and creates nothing. The 'real' root is injected as a
    tmp dir so this test can never touch the actual board."""
    fake_real_root = tmp_path / "pretend_real_hermes"
    monkeypatch.setattr(kb, "_real_hermes_kanban_roots", lambda: [fake_real_root])
    # PYTEST_CURRENT_TEST is already set by pytest during this call.
    target = fake_real_root / "kanban.db"
    with pytest.raises(kb.KanbanRealBoardInTestError):
        kb.connect(target)
    assert not target.exists(), "guard must refuse BEFORE creating the DB file"
    assert not fake_real_root.exists(), "guard must not mkdir the real root"


def test_connect_refuses_named_board_under_real_kanban_dir(tmp_path, monkeypatch):
    fake_real_root = tmp_path / "pretend_real_hermes"
    monkeypatch.setattr(kb, "_real_hermes_kanban_roots", lambda: [fake_real_root])
    target = fake_real_root / "kanban" / "boards" / "prod" / "kanban.db"
    with pytest.raises(kb.KanbanRealBoardInTestError):
        kb.connect(target)
    assert not target.exists()


def test_connect_allows_tmp_board_under_pytest(tmp_path, monkeypatch):
    """A genuinely isolated tmp_path board still opens normally."""
    fake_real_root = tmp_path / "pretend_real_hermes"
    monkeypatch.setattr(kb, "_real_hermes_kanban_roots", lambda: [fake_real_root])
    safe = tmp_path / "isolated" / "kanban.db"
    conn = kb.connect(safe)
    try:
        assert safe.exists()
        # Board is usable: the schema initialised (tasks table present).
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "tasks" in names
    finally:
        conn.close()


def test_guard_is_noop_in_production(tmp_path, monkeypatch):
    """Outside any pytest process the guard never fires, so production opening
    the real board is unaffected. We simulate production by forcing
    ``_is_pytest_process`` False (in-process we cannot remove pytest from
    sys.modules)."""
    fake_real_root = tmp_path / "pretend_real_hermes"
    monkeypatch.setattr(kb, "_real_hermes_kanban_roots", lambda: [fake_real_root])
    monkeypatch.setattr(kb, "_is_pytest_process", lambda: False)
    target = fake_real_root / "kanban.db"
    conn = kb.connect(target)
    try:
        assert target.exists()
    finally:
        conn.close()


def test_is_pytest_process_true_during_test():
    """The exact requested PYTEST_CURRENT_TEST behavior is preserved: while a
    test is executing, detection is positive."""
    assert os.environ.get("PYTEST_CURRENT_TEST")  # pytest sets this per-test
    assert kb._is_pytest_process() is True


def test_is_pytest_process_detects_env_signal_alone(monkeypatch):
    """PYTEST_CURRENT_TEST alone (the exact requested signal) triggers
    detection even if the other signals were absent."""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "some_test (call)")
    assert kb._is_pytest_process() is True


def test_is_pytest_process_detects_collection_window_without_current_test(
    monkeypatch,
):
    """During collection/import PYTEST_CURRENT_TEST is unset, yet detection must
    still be positive (pytest is imported in sys.modules). This is the window
    the original env-only guard missed."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    # pytest / _pytest remain in sys.modules for the whole session.
    assert "pytest" in _sys.modules or "_pytest" in _sys.modules
    assert kb._is_pytest_process() is True


def test_connect_refuses_real_board_at_collection_time_subprocess(tmp_path):
    """A hermetic, collection-time regression: a test module that opens the
    'real' board at *import* time (before PYTEST_CURRENT_TEST is set) must be
    refused during ``pytest --collect-only``, and must NOT create the DB file
    or even its parent dir. The 'real' root is redirected to a tmp HOME so a
    broken guard can only ever touch tmp — never the actual board."""
    import subprocess

    repo_root = Path(__file__).resolve().parents[2]
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    real_board = fake_home / ".hermes" / "kanban.db"

    mod = tmp_path / "collect_time_board_open.py"
    mod.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "from hermes_cli import kanban_db as kb\n"
        "# Runs at COLLECTION/import time — PYTEST_CURRENT_TEST is unset here.\n"
        "kb.connect(Path(os.environ['HOME']) / '.hermes' / 'kanban.db')\n"
        "def test_placeholder():\n"
        "    pass\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["HOME"] = str(fake_home)
    env.pop("PYTEST_CURRENT_TEST", None)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [
            _sys.executable, "-m", "pytest", "--collect-only",
            "-p", "no:cacheprovider", str(mod),
        ],
        cwd=str(tmp_path), env=env,
        capture_output=True, text=True, timeout=180,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"collection should have errored:\n{combined}"
    assert "KanbanRealBoardInTestError" in combined, combined
    # Refused BEFORE mkdir / SQLite: neither the DB nor its parent dir exist.
    assert not real_board.exists()
    assert not (fake_home / ".hermes").exists()
