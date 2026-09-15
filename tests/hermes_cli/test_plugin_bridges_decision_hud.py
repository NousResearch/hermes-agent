"""Tests for hermes_cli.plugin_bridges.decision_hud — the F1 minimal bridge
into the standalone decision-hud plugin's db.py, used by the kanban
dispatcher's batch-approval gate. See decision-hub-first-work/plans/02-....
"""

from __future__ import annotations

import textwrap

import pytest


@pytest.fixture(autouse=True)
def _point_hermes_home_at_tmp(tmp_path, monkeypatch):
    """Never let a bridge test touch the operator's real ~/.hermes install."""
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)


def _plugin_dir(tmp_path):
    d = tmp_path / "plugins" / "decision-hud"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _reset_bridge_module_cache():
    """The bridge caches loaded plugin modules in sys.modules by resolved
    path; each test writes a fresh stub at the same path, so the cache must
    be cleared or a later test would silently reuse an earlier stub."""
    import sys

    for key in list(sys.modules):
        if key.startswith("_hermes_decision_hud_db_bridge:"):
            del sys.modules[key]


@pytest.fixture(autouse=True)
def _clear_bridge_cache():
    _reset_bridge_module_cache()
    yield
    _reset_bridge_module_cache()


def test_check_batch_approval_missing_plugin_returns_false(tmp_path):
    from hermes_cli.plugin_bridges.decision_hud import check_batch_approval

    approved, reason = check_batch_approval(project="proj", batch_id="b1")
    assert approved is False
    assert "decision-hud unavailable" in reason


def test_check_batch_approval_schema_too_old_returns_false(tmp_path):
    import sqlite3

    plugin_dir = _plugin_dir(tmp_path)
    db_path = plugin_dir / "decision_hud.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    conn.close()

    (plugin_dir / "db.py").write_text(
        textwrap.dedent(
            f"""
            def db_path():
                return {str(db_path)!r}

            def connect():
                raise AssertionError("must not connect on an unsupported schema")

            class BatchNotApproved(Exception):
                pass

            def require_batch_approval(conn, *, project_id, batch_id):
                raise AssertionError("must not be called on an unsupported schema")
            """
        ),
        encoding="utf-8",
    )

    from hermes_cli.plugin_bridges.decision_hud import check_batch_approval

    approved, reason = check_batch_approval(project="proj", batch_id="b1")
    assert approved is False
    assert "schema too old" in reason


def _write_stub(tmp_path, *, require_batch_approval_body: str) -> None:
    import sqlite3

    plugin_dir = _plugin_dir(tmp_path)
    db_path = plugin_dir / "decision_hud.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 6")
    conn.commit()
    conn.close()

    (plugin_dir / "db.py").write_text(
        textwrap.dedent(
            f"""
            def db_path():
                return {str(db_path)!r}

            def connect():
                import sqlite3
                return sqlite3.connect({str(db_path)!r})

            class BatchNotApproved(Exception):
                pass

            def require_batch_approval(conn, *, project_id, batch_id):
{textwrap.indent(require_batch_approval_body, "                ")}
            """
        ),
        encoding="utf-8",
    )


def test_check_batch_approval_approved(tmp_path):
    _write_stub(tmp_path, require_batch_approval_body="return None\n")

    from hermes_cli.plugin_bridges.decision_hud import check_batch_approval

    approved, reason = check_batch_approval(project="proj", batch_id="b1")
    assert (approved, reason) == (True, "")


def test_check_batch_approval_not_approved(tmp_path):
    _write_stub(
        tmp_path,
        require_batch_approval_body=(
            'raise BatchNotApproved("batch b1 is pending")\n'
        ),
    )

    from hermes_cli.plugin_bridges.decision_hud import check_batch_approval

    approved, reason = check_batch_approval(project="proj", batch_id="b1")
    assert approved is False
    assert "pending" in reason


def test_check_batch_approval_unexpected_exception_folds_to_false(tmp_path):
    _write_stub(
        tmp_path,
        require_batch_approval_body='raise RuntimeError("boom")\n',
    )

    from hermes_cli.plugin_bridges.decision_hud import check_batch_approval

    approved, reason = check_batch_approval(project="proj", batch_id="b1")
    assert approved is False
    assert "batch approval check errored" in reason


def test_load_decision_hud_db_reloads_on_mtime_change(tmp_path, monkeypatch):
    """A live edit to db.py while the process is running (e.g. an operator
    patches decision-hud's policy) must be picked up on the next call, not
    silently masked by a stale sys.modules cache — regression for a real
    bug found in adversarial review: the old code cached by identity only,
    with no actual mtime check despite its docstring's claim."""
    import time
    from hermes_cli.plugin_bridges import decision_hud as bridge

    plugin_dir = tmp_path / "plugins" / "decision-hud"
    plugin_dir.mkdir(parents=True)
    db_file = plugin_dir / "db.py"
    db_file.write_text(
        "import sqlite3\n"
        "def db_path():\n"
        "    return '%s'\n"
        "def connect():\n"
        "    return sqlite3.connect(':memory:')\n"
        "class BatchNotApproved(Exception): pass\n"
        "def require_batch_approval(conn, *, project_id, batch_id):\n"
        "    return None  # v1: approves everything\n"
        % str(tmp_path / "empty.db")
    )
    monkeypatch.setattr(
        "hermes_cli.plugin_bridges.decision_hud._decision_hud_db_path",
        lambda: db_file,
    )

    approved, _ = bridge.check_batch_approval(project="p", batch_id="b")
    assert approved is True  # v1 approves

    time.sleep(0.01)  # ensure a distinct mtime on filesystems with coarse resolution
    db_file.write_text(
        "import sqlite3\n"
        "def db_path():\n"
        "    return '%s'\n"
        "def connect():\n"
        "    return sqlite3.connect(':memory:')\n"
        "class BatchNotApproved(Exception): pass\n"
        "def require_batch_approval(conn, *, project_id, batch_id):\n"
        "    raise BatchNotApproved('v2: rejects everything')\n"
        % str(tmp_path / "empty.db")
    )

    approved2, reason2 = bridge.check_batch_approval(project="p", batch_id="b")
    assert approved2 is False  # v2 picked up, not a stale cached v1
    assert "v2" in reason2


def test_check_batch_approval_never_raises_when_batch_not_approved_class_missing(
    tmp_path, monkeypatch,
):
    """A malformed/partial decision-hud install whose db.py doesn't define
    BatchNotApproved at all must still fold into (False, reason), never let
    an AttributeError escape from the except-clause attribute lookup —
    regression for a real bug found in adversarial review."""
    from hermes_cli.plugin_bridges import decision_hud as bridge

    plugin_dir = tmp_path / "plugins" / "decision-hud"
    plugin_dir.mkdir(parents=True)
    db_file = plugin_dir / "db.py"
    db_file.write_text(
        "import sqlite3\n"
        "def db_path():\n"
        "    return '%s'\n"
        "def connect():\n"
        "    return sqlite3.connect(':memory:')\n"
        "def require_batch_approval(conn, *, project_id, batch_id):\n"
        "    raise RuntimeError('no BatchNotApproved class defined at all')\n"
        % str(tmp_path / "empty.db")
    )
    monkeypatch.setattr(
        "hermes_cli.plugin_bridges.decision_hud._decision_hud_db_path",
        lambda: db_file,
    )

    approved, reason = bridge.check_batch_approval(project="p", batch_id="b")
    assert approved is False
    assert "errored" in reason
