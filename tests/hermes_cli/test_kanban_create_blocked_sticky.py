"""A card created with ``initial_status="blocked"`` must actually stay blocked.

``create_task(initial_status="blocked")`` is the documented way to park work
that needs a human ("tasks that require immediate human ops"). It used to write
``status='blocked'`` straight into the row and emit only a ``created`` event.
``_has_sticky_block`` decides stickiness by looking for a ``blocked`` /
``unblocked`` event row and returns False when there is none — by design, so
the crash circuit-breaker can auto-recover. A create-time block was therefore
indistinguishable from a circuit-breaker block, and ``recompute_ready``
promoted the card on the next dispatcher tick, spawning a worker on a brief
that said "do not start until the human answers".

The fix records the block the way ``block_task`` does (a ``blocked`` event
plus ``block_kind``), so the handoff the caller asked for is the handoff the
dispatcher sees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _events(conn, task_id: str) -> list[str]:
    return [
        r["kind"]
        for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall()
    ]


def test_created_blocked_emits_sticky_block_event(kanban_home: Path) -> None:
    """The event row is the whole mechanism — assert it exists, not just the
    status, because the status alone was already correct before the fix."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="waiting on a human decision",
            assignee="alice",
            initial_status="blocked",
        )
        assert kb.get_task(conn, tid).status == "blocked"
        assert "blocked" in _events(conn, tid)
        assert kb._has_sticky_block(conn, tid) is True


def test_created_blocked_survives_recompute_ready(kanban_home: Path) -> None:
    """The actual failure: the dispatcher's promote pass ran and the card
    started running."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="do not start until the human answers",
            assignee="alice",
            initial_status="blocked",
        )
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "blocked"


def test_created_blocked_is_typed_needs_input(kanban_home: Path) -> None:
    """Untyped blocks work as generic human blockers but leave diagnostics
    unable to say *why*. The only reason to create a card already blocked is
    that it waits on a human, so ``needs_input`` is the default."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="waiting on a human decision",
            assignee="alice",
            initial_status="blocked",
        )
        assert kb.get_task(conn, tid).block_kind == "needs_input"


def test_capability_kind_is_accepted(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="needs a human to hand over a paywalled PDF",
            assignee="alice",
            initial_status="blocked",
            initial_block_kind="capability",
            initial_block_reason="paywall — no agent can fetch it",
        )
        assert kb.get_task(conn, tid).block_kind == "capability"
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'blocked'",
            (tid,),
        ).fetchone()
        assert "paywall" in (row["payload"] or "")


def test_dependency_kind_is_refused_at_create(kanban_home: Path) -> None:
    """``dependency`` routes to ``todo`` in ``block_task``. Accepting it here
    would turn "create this blocked" into "create this runnable"."""
    with kb.connect() as conn:
        with pytest.raises(ValueError, match="initial_block_kind"):
            kb.create_task(
                conn,
                title="x",
                assignee="alice",
                initial_status="blocked",
                initial_block_kind="dependency",
            )


def test_unblock_clears_the_sticky_row(kanban_home: Path) -> None:
    """A human answering must still be able to release the card — otherwise the
    fix trades a runaway worker for a card nobody can start."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="waiting on a human decision",
            assignee="alice",
            initial_status="blocked",
        )
        assert kb.unblock_task(conn, tid)
        assert kb._has_sticky_block(conn, tid) is False
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status in ("ready", "todo")


def test_normal_create_is_untouched(kanban_home: Path) -> None:
    """The fix must not leak into the default path: an ordinary card carries
    no block event and no block kind."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ordinary work", assignee="alice")
        assert kb.get_task(conn, tid).block_kind is None
        assert "blocked" not in _events(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"
