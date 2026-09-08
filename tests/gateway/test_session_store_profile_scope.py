"""A DEFAULT session must persist to default's state.db, labelled default.

Fifth "profile seam" fix. The store path and the profile label used to be
resolved by two independent code paths at different instants: the store was
opened LAZILY against the context-local ``HERMES_HOME`` at first DB open, while
``profile_name`` was stamped separately from ``get_active_profile_name()``. On
the reused compute-host executor a residual foreign ``HERMES_HOME`` override
active at the store-open instant bound a DEFAULT session's rows to another
profile's ``state.db`` while the label stayed ``default`` -- a cross-profile
data-isolation break.

These tests assert the PHYSICAL db file that holds the row (design finding R4)
under a residual foreign home, not merely ``store._own_profile_name() == label``
at a build site (which passes on HEAD). ``HERMES_HOME`` is set via env
(monkeypatch.setenv), NOT only the ``_HERMES_HOME_OVERRIDE`` ContextVar, so
``get_default_hermes_root()`` (which reads ``os.environ``) resolves the temp
default-root INSIDE the tree and ``_own_profile_name`` yields ``default``, not
None (design finding R5).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_constants as hc
import hermes_state
from hermes_state import SessionDB
from hermes_state_registry import close_all
from run_agent import AIAgent


def _profile_of(db_path, session_id):
    if not Path(db_path).exists():
        return "__MISSING_FILE__"
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT profile_name FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return ("__NO_ROW__" if row is None else row[0])
    finally:
        conn.close()


def _has_row(db_path, session_id) -> bool:
    if not Path(db_path).exists():
        return False
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """Temp default-root with a tommy profile subtree, wired via env.

    HERMES_HOME is set via env so ``get_default_hermes_root()`` (reads
    os.environ) resolves the temp default-root in-tree. DEFAULT_DB_PATH is
    restored to its import-time snapshot so the lazy no-path
    ``get_shared_session_db()`` open resolves through ``get_hermes_home()``
    (the context-local override) the way production does -- otherwise the
    escape hatch would pin every lookup to one fixed path and mask the
    divergence.
    """
    root = tmp_path / "hermes"
    root.mkdir(parents=True)
    (root / "profiles" / "tommy").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    yield SimpleNamespace(
        root=root,
        default_db=(root / "state.db"),
        tommy_home=(root / "profiles" / "tommy"),
        tommy_db=(root / "profiles" / "tommy" / "state.db"),
    )
    close_all()


def _agent_shell(session_id: str):
    """Minimal AIAgent shell that owns the real recall + lazy-create seam."""
    agent = SimpleNamespace(
        _persist_disabled=False,
        _session_db=None,
        _owns_session_db=False,
        _session_db_created=False,
        platform="cli",
        session_id=session_id,
        model="test-model",
        _session_init_model_config=None,
        _cached_system_prompt=None,
        _parent_session_id=None,
    )
    agent._get_session_db_for_recall = (
        AIAgent._get_session_db_for_recall.__get__(agent, AIAgent)
    )
    agent._ensure_db_session = (
        AIAgent._ensure_db_session.__get__(agent, AIAgent)
    )
    agent._session_row_model_config = (
        AIAgent._session_row_model_config.__get__(agent, AIAgent)
    )
    return agent


def test_default_under_residual_home_writes_to_default_store(roots):
    """Integration: a DEFAULT build under a residual tommy home lands the row
    in ``<root>/state.db`` labelled ``default`` -- never in tommy's store."""
    agent = _agent_shell("seam-default-1")

    # Residual tommy home active ONLY at the store-open instant (mirror the
    # reused compute-host executor): the earliest lazy open happens here.
    tommy_token = hc.set_hermes_home_override(str(roots.tommy_home))
    try:
        agent._get_session_db_for_recall()
    finally:
        # Residue unwinds before the label is computed -- the label side used
        # to run at a clean instant, which is exactly how store and label
        # diverged.
        hc.reset_hermes_home_override(tommy_token)

    agent._ensure_db_session()

    # PHYSICAL location assertion (design finding R4).
    assert _has_row(roots.default_db, "seam-default-1"), (
        "default session row must physically live in <root>/state.db"
    )
    assert not _has_row(roots.tommy_db, "seam-default-1"), (
        "default session row leaked into tommy's state.db -- isolation break"
    )
    assert _profile_of(roots.default_db, "seam-default-1") == "default"


def test_named_profile_session_unchanged(roots):
    """Regression guard: a named-profile (tommy) build still lands in
    ``<root>/profiles/tommy/state.db`` labelled ``tommy``."""
    from hermes_state_registry import acquire, release_or_close

    db = acquire(roots.tommy_db)
    try:
        db.create_session("seam-tommy-1", source="desktop", model="test-model")
    finally:
        release_or_close(db)

    assert _has_row(roots.tommy_db, "seam-tommy-1")
    assert not _has_row(roots.default_db, "seam-tommy-1")
    assert _profile_of(roots.tommy_db, "seam-tommy-1") == "tommy"


def test_own_profile_name_derivation(roots, tmp_path):
    """Unit: ``_own_profile_name`` yields default / <name> / None by path."""
    default_db = SessionDB(db_path=roots.default_db)
    try:
        assert default_db._own_profile_name() == "default"
    finally:
        default_db.close()

    named_db = SessionDB(db_path=roots.tommy_db)
    try:
        assert named_db._own_profile_name() == "tommy"
    finally:
        named_db.close()

    outside = tmp_path / "elsewhere" / "state.db"
    outside.parent.mkdir()
    outside_db = SessionDB(db_path=outside)
    try:
        assert outside_db._own_profile_name() is None
    finally:
        outside_db.close()
