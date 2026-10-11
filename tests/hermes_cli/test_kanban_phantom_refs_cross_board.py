"""A completion summary that names a card on a SIBLING board is not a phantom.

The measured defect (2026-09-29, card t_89626ce6): ``_flag_phantom_prose_refs``
resolved every ``t_<hex>`` prose reference against ONE board — the completing
connection — so a real cross-board card (an ops-board child named in a defcon card's
completion handoff) was durably recorded as ``suspected_hallucinated_references``.
A verified reference must not be recorded as an invention.

``test_cross_board_reference_is_not_a_phantom`` FAILS against pre-fix code; see the
card for the captured before/after output. ``test_invented_reference_still_flags``
pins the other direction: a genuinely nonexistent id must still be flagged, and the
payload must name the boards searched so an operator can tell the two apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb  # noqa: E402
from hermes_cli import kanban_db_connect as kbc  # noqa: E402


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with no prior kanban state and no ambient pins."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_ATTACHMENTS_ROOT",
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


def _boards(*slugs: str) -> None:
    for slug in slugs:
        kb.create_board(slug)


def _phantom_events(conn, task_id):
    return [
        e for e in kb.list_events(conn, task_id)
        if e.kind == "suspected_hallucinated_references"
    ]


def test_cross_board_reference_is_not_a_phantom(fresh_home):
    """A card that exists on a SIBLING board is a cross-board reference, not a phantom."""
    _boards("alpha", "beta")
    with kbc.connect(board="beta") as beta:
        sibling = kb.create_task(beta, title="landing child", assignee="platform-worker")
    assert sibling.startswith("t_")

    with kbc.connect(board="alpha") as conn:
        card = kb.create_task(conn, title="closing card", assignee="platform-coder")
        kb._flag_phantom_prose_refs(
            conn, card, None,
            f"the landing brief lives on the beta board, as card {sibling}", None, [],
        )
        events = _phantom_events(conn, card)

    assert events == [], (
        "a card that exists on the sibling board 'beta' was recorded as a phantom: "
        + repr([e.payload for e in events])
    )


def test_invented_reference_still_flags_and_names_the_boards(fresh_home):
    """A genuinely nonexistent id still flags, and the payload names the boards searched."""
    _boards("alpha", "beta")
    with kbc.connect(board="alpha") as conn:
        card = kb.create_task(conn, title="closing card", assignee="platform-coder")
        kb._flag_phantom_prose_refs(
            conn, card, None, "see t_deadbeefdeadbeef for the follow-up", None, [],
        )
        events = _phantom_events(conn, card)

    assert len(events) == 1, f"expected exactly one phantom event, got {events!r}"
    payload = events[0].payload or {}
    assert payload.get("phantom_refs") == ["t_deadbeefdeadbeef"]
    searched = payload.get("boards_searched") or []
    assert {"alpha", "beta"} <= set(searched), (
        f"payload does not name the boards searched: {payload!r}"
    )


def test_verified_created_card_is_never_flagged(fresh_home):
    """The existing ``verified_cards`` exemption is preserved."""
    _boards("alpha")
    with kbc.connect(board="alpha") as conn:
        card = kb.create_task(conn, title="closing card", assignee="platform-coder")
        kb._flag_phantom_prose_refs(
            conn, card, None, "created t_aabbccdd11223344", None, ["t_aabbccdd11223344"],
        )
        events = _phantom_events(conn, card)
    assert events == []
