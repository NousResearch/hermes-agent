"""Shared kanban DB isolation helper for tests.

Why this exists
---------------
``kanban_db.py`` resolves the database path via:
  HERMES_KANBAN_DB  >  HERMES_KANBAN_HOME  >  HERMES_HOME

The dispatcher injects ``HERMES_KANBAN_DB`` (and several companion vars)
into every worker's environment.  When a worker runs a pytest session
*outside* the ``tests/`` tree (e.g. copies the file to ``/tmp/`` and
executes it directly), the ``conftest.py`` autouse fixtures never load —
so those dispatcher-injected vars survive into the test process and point
at the real ``~/.hermes/kanban.db``.  Tests that write tasks then leak
fake data into the developer's live board.

``pin_kanban_db()`` is the one-stop fix: strip every ``HERMES_KANBAN_*``
env var and hard-pin ``HERMES_KANBAN_DB`` to a fresh temp path.  The
resulting DB path is guaranteed to sit inside ``tmp_path`` regardless of
whether conftest.py is loaded.

Usage
-----
Call ``pin_kanban_db(tmp_path, monkeypatch)`` at the top of a fixture
*before* any call to ``kb.kanban_db_path()`` or ``kb.connect()``.

    from tests.hermes_cli._kanban_isolation import pin_kanban_db

    @pytest.fixture
    def kanban_home(tmp_path, monkeypatch):
        db_path = pin_kanban_db(tmp_path, monkeypatch)
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        kb.init_db()
        return home
"""
from __future__ import annotations

from pathlib import Path

# Every HERMES_KANBAN_* env var that kanban_db.py reads.
# Stripping all of them before pinning HERMES_KANBAN_DB ensures that no
# dispatcher-injected override can win over our temp isolation.
_KANBAN_ENV_VARS: tuple[str, ...] = (
    # Path resolution (highest-priority first in kanban_db.py)
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_HOME",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_ATTACHMENTS_ROOT",
    "HERMES_KANBAN_LOGS_ROOT",
    # Worker identity pins injected by the dispatcher
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_CLAIM_LOCK",
    "HERMES_KANBAN_DISPATCH_IN_GATEWAY",
    # Git / goal-loop pins injected by the dispatcher (kanban_db.py:9499-9510)
    "HERMES_KANBAN_BRANCH",
    "HERMES_KANBAN_GOAL_MODE",
    "HERMES_KANBAN_GOAL_MAX_TURNS",
    # Timing overrides that can affect dispatch behavior in tests
    "HERMES_KANBAN_CLAIM_TTL_SECONDS",
    "HERMES_KANBAN_CRASH_GRACE_SECONDS",
    "HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS",
    "HERMES_KANBAN_BUSY_TIMEOUT_MS",
)


def pin_kanban_db(tmp_path: Path, monkeypatch) -> Path:
    """Strip all HERMES_KANBAN_* env vars then pin HERMES_KANBAN_DB to a
    fresh temp path.

    This makes the fixture safe even when run outside the ``tests/``
    directory (no conftest.py autouse guard) or when a dispatcher has
    injected HERMES_KANBAN_DB into the worker environment.

    Args:
        tmp_path: pytest ``tmp_path`` fixture — the database is placed
            directly inside this directory as ``kanban.db``.
        monkeypatch: pytest ``monkeypatch`` fixture used for env
            manipulation (automatically restored after the test).

    Returns:
        Path to the pinned ``kanban.db`` (the file is NOT created; call
        ``kb.init_db()`` or open a connection after this returns).
    """
    for var in _KANBAN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    return db_path
