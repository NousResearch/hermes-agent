"""Shared hermetic-DB fixture for OPS-PROBE tests.

Each test gets a fresh ``mkdtemp`` SQLite file under a unique slug so
the dispatcher never watches it. Re-uses the same shape as
``tests/hermes_cli/test_kanban_decompose_priority.py`` so any kanban
DB helper can be exercised directly with no LLM, no gateway.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hermes_cli import kanban_db as kb  # noqa: E402


def _seed_board(conn, slug: str) -> None:
    try:
        conn.execute(
            "INSERT OR IGNORE INTO kanban_boards (slug, name, tenant) "
            "VALUES (?, ?, NULL)",
            (slug, slug),
        )
    except Exception:
        pass


@pytest.fixture
def tmp_kanban(tmp_path_factory):
    """Hermetic kanban DB: a fresh file under a unique slug.

    Each test gets a fresh ``mkdtemp`` SQLite file; the temp dir is
    cleaned up by ``tmp_path_factory`` finalizer at session end.
    The test MUST pass ``db_path=path`` to ``kb.connect_closing()``
    rather than relying on env vars — pytest may sanitise env between
    tests, breaking in-process connects that fall back to the
    workspace default DB.
    """
    tmpdir = tmp_path_factory.mktemp(f"ops-probe-{uuid.uuid4().hex[:8]}")
    db_path = tmpdir / "kanban.db"
    slug = f"ops-probe-{uuid.uuid4().hex[:8]}"
    os.environ["HERMES_KANBAN_DB"] = str(db_path)
    os.environ["HERMES_KANBAN_BOARD"] = slug
    kb.init_db(db_path)
    with kb.connect_closing(db_path=db_path) as conn:
        _seed_board(conn, slug)
    return {"path": str(db_path), "slug": slug, "tmpdir": str(tmpdir)}