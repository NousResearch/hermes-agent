"""Tests for the configurable ``kanban.block_recurrence_limit`` (issue #59333).

Covers the user-facing knob that lets pipelines raise the unblock-loop
breaker above the hard-coded default of 2 without editing ``kanban_db.py``.
The constant ``BLOCK_RECURRENCE_LIMIT`` stays as the legacy fallback so
existing imports keep working; the new resolver
:func:`hermes_cli.kanban_db.get_block_recurrence_limit` is the call-site
read path that ``block_task`` uses.

Each test writes its own minimal ``~/.hermes/config.yaml`` under the
hermetic ``HERMES_HOME`` provided by ``tests/conftest.py`` and asserts the
resolver behaviour for one configuration shape. ``load_config`` is
mtime-cached so a fresh write between tests is observed without any extra
cache-busting.
"""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(home: Path, body: str) -> Path:
    """Write a raw YAML config under the test's HERMES_HOME.

    We deliberately bypass ``save_config`` here: we want to test the
    resolver's tolerance of hand-edited YAML, and ``save_config`` strips
    keys whose values match ``DEFAULT_CONFIG`` (which would defeat the
    whole point of these tests).
    """
    config_path = home / "config.yaml"
    config_path.write_text(dedent(body).lstrip("\n"), encoding="utf-8")
    return config_path


def _running_task(conn):
    """Create a task and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title="t", assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


def _make_running_again(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Hermetic HERMES_HOME wired to ``init_db`` for end-to-end block_task tests."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Default behaviour — config key absent
# ---------------------------------------------------------------------------


def test_default_when_config_missing(kanban_home: Path) -> None:
    """No config.yaml at all → resolver falls back to the module default (2)."""
    assert not (kanban_home / "config.yaml").exists()
    assert kb.get_block_recurrence_limit() == kb.BLOCK_RECURRENCE_LIMIT == 2


def test_default_when_kanban_section_present_but_key_missing(
    kanban_home: Path,
) -> None:
    """``kanban:`` exists but ``block_recurrence_limit`` is unset → default."""
    _write_config(
        kanban_home,
        """
        kanban:
          failure_limit: 5
          dispatch_interval_seconds: 30
        """,
    )
    assert kb.get_block_recurrence_limit() == 2


# ---------------------------------------------------------------------------
# Custom value applied when config key present
# ---------------------------------------------------------------------------


def test_custom_int_value_applied(kanban_home: Path) -> None:
    """A positive int override is honoured."""
    _write_config(
        kanban_home,
        """
        kanban:
          block_recurrence_limit: 5
        """,
    )
    assert kb.get_block_recurrence_limit() == 5


def test_custom_numeric_string_coerced(kanban_home: Path) -> None:
    """Hand-edited YAML often yields strings — coerce when possible."""
    _write_config(
        kanban_home,
        """
        kanban:
          block_recurrence_limit: "7"
        """,
    )
    assert kb.get_block_recurrence_limit() == 7


def test_other_kanban_keys_ignored(kanban_home: Path) -> None:
    """Unrelated kanban settings don't accidentally trip the override."""
    _write_config(
        kanban_home,
        """
        kanban:
          failure_limit: 99
          dispatch_interval_seconds: 1
        """,
    )
    # Still 2 — the user did not set the recurrence knob.
    assert kb.get_block_recurrence_limit() == 2


# ---------------------------------------------------------------------------
# Invalid values fall back to default (never raise)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_yaml, description",
    [
        (
            "kanban:\n  block_recurrence_limit: 0\n",
            "zero is not a valid recurrence count",
        ),
        (
            "kanban:\n  block_recurrence_limit: -3\n",
            "negative values are rejected",
        ),
        (
            "kanban:\n  block_recurrence_limit: 'oops'\n",
            "non-numeric strings fall through to the default",
        ),
        (
            "kanban:\n  block_recurrence_limit: true\n",
            "bools are rejected (not silently coerced to 1/0)",
        ),
        (
            "kanban:\n  block_recurrence_limit: [1, 2]\n",
            "lists are rejected",
        ),
        (
            "kanban:\n  block_recurrence_limit: 1.5\n",
            "floats are rejected (limit is a count, must be an int)",
        ),
    ],
)
def test_invalid_values_fall_back_to_default(
    kanban_home: Path, bad_yaml: str, description: str
) -> None:
    """Any garbage value resolves to the default 2 — never raises."""
    _write_config(kanban_home, bad_yaml)
    assert kb.get_block_recurrence_limit() == 2, description


# ---------------------------------------------------------------------------
# Config reload picks up runtime changes
# ---------------------------------------------------------------------------


def test_runtime_config_change_picked_up(kanban_home: Path) -> None:
    """Edits to config.yaml between calls are honoured without a process restart.

    ``load_config`` is mtime-cached, so writing the file with a new mtime is
    sufficient to invalidate the cache on the next read. This is the
    contract users will rely on when tweaking the value at runtime.
    """
    _write_config(
        kanban_home,
        """
        kanban:
          block_recurrence_limit: 3
        """,
    )
    assert kb.get_block_recurrence_limit() == 3

    # Rewrite with a different value. Use a fresh write (not append) so
    # the mtime definitely advances even on coarse-grained filesystems.
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  block_recurrence_limit: 8\n", encoding="utf-8"
    )
    assert kb.get_block_recurrence_limit() == 8

    # And removing the key drops back to the default.
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  failure_limit: 9\n", encoding="utf-8"
    )
    assert kb.get_block_recurrence_limit() == 2


# ---------------------------------------------------------------------------
# End-to-end: the override actually changes block_task's routing behaviour
# ---------------------------------------------------------------------------


def test_high_override_defers_triage_routing(kanban_home: Path) -> None:
    """With ``block_recurrence_limit: 5``, a 2-cycle loop stays in ``blocked``.

    Mirrors the original ``test_same_cause_reblock_routes_to_triage`` in
    ``tests/hermes_cli/test_kanban_block_kinds.py`` but raises the limit so
    the same two cycles do NOT trip the breaker. The task lands in
    ``blocked`` (still inside the human bucket) until the user-configured
    threshold is reached.
    """
    _write_config(
        kanban_home,
        """
        kanban:
          block_recurrence_limit: 5
        """,
    )

    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="need creds", kind="needs_input")
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        assert kb.block_task(conn, tid, reason="still need creds", kind="needs_input")
        t = kb.get_task(conn, tid)
        # Default limit was 2 → would have routed to triage. With 5, still blocked.
        assert t.status == "blocked"
        assert t.block_recurrences == 2


def test_override_at_threshold_routes_to_triage(kanban_home: Path) -> None:
    """Hitting the configured limit still triggers the loop-breaker."""
    _write_config(
        kanban_home,
        """
        kanban:
          block_recurrence_limit: 1
        """,
    )

    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        # First block itself reaches the (lowered) limit → triage immediately.
        assert kb.block_task(conn, tid, reason="x", kind="capability")
        t = kb.get_task(conn, tid)
        assert t.status == "triage"
        # The block_loop_detected event records the effective limit, not
        # the hard-coded constant.
        events = [
            e for e in kb.list_events(conn, tid)
            if e.kind == "block_loop_detected"
        ]
        assert events, "expected a block_loop_detected event"
        payload = events[-1].payload or {}
        assert payload.get("limit") == 1


# ---------------------------------------------------------------------------
# Legacy constant still importable (back-compat)
# ---------------------------------------------------------------------------


def test_legacy_constant_still_exported() -> None:
    """External callers that import ``BLOCK_RECURRENCE_LIMIT`` keep working."""
    assert hasattr(kb, "BLOCK_RECURRENCE_LIMIT")
    assert kb.BLOCK_RECURRENCE_LIMIT == 2
