"""kanban_db_path: an explicit board must never be silently overridden.

The dispatcher pins a worker's kanban context via ``HERMES_KANBAN_DB`` /
``HERMES_KANBAN_BOARD`` so "workers cannot accidentally see other boards".
The pin is correct — but it used to beat an *explicit* ``board`` argument
silently, and the CLI's ``--board`` flag resolves the DB through this
function before scoping. The combination meant a pinned process running

    hermes kanban --board life create "[attention] ..."

wrote to its own board while the output header printed ``Board: life`` —
the board that was *requested*, not the one that was read. Nothing in the
output could reveal the substitution: a dispatched worker filed its
escalations onto its own board for a full night and truthfully reported,
run after run, that it had filed them elsewhere.

The fix: an explicit board that contradicts the pin raises
``BoardPinConflict``. ``board=None`` still honours the pin, so worker
containment is unchanged.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture()
def boards(monkeypatch, tmp_path):
    """Two real board dirs under an isolated HERMES_HOME, pin unset."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    for slug in ("alpha", "beta"):
        (tmp_path / "kanban" / "boards" / slug).mkdir(parents=True)
    return tmp_path


def _pin_to(monkeypatch, home, slug):
    pinned = home / "kanban" / "boards" / slug / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", slug)
    return pinned


def test_unpinned_explicit_board_resolves_normally(boards):
    path = kb.kanban_db_path("alpha")
    assert path == boards / "kanban" / "boards" / "alpha" / "kanban.db"


def test_pinned_with_no_board_keeps_the_pin(boards, monkeypatch):
    """The dispatcher→worker handoff: containment must be unchanged."""
    pinned = _pin_to(monkeypatch, boards, "alpha")
    assert kb.kanban_db_path(None) == pinned


def test_pinned_with_matching_board_is_allowed(boards, monkeypatch):
    """Naming your own board is not a contradiction."""
    pinned = _pin_to(monkeypatch, boards, "alpha")
    assert kb.kanban_db_path("alpha") == pinned


def test_pinned_with_conflicting_board_raises(boards, monkeypatch):
    """The silent-mislabel case: must fail, never return the pinned path."""
    _pin_to(monkeypatch, boards, "alpha")
    with pytest.raises(kb.BoardPinConflict) as exc:
        kb.kanban_db_path("beta")
    # The message must carry what an agent (or human) needs to act on it:
    # the requested board, the actual pin, and the sanctioned way through.
    msg = str(exc.value)
    assert "beta" in msg
    assert "HERMES_KANBAN_DB" in msg
    assert "env -u" in msg


def test_pinned_default_board_conflict_also_raises(boards, monkeypatch):
    """`default` resolves to the home-root DB; a pin elsewhere still conflicts."""
    _pin_to(monkeypatch, boards, "alpha")
    with pytest.raises(kb.BoardPinConflict):
        kb.kanban_db_path("default")


def test_unresolvable_pin_path_compares_literally(boards, monkeypatch):
    """A pin pointing at a not-yet-created path must not crash the check."""
    ghost = boards / "kanban" / "boards" / "alpha" / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(ghost))
    # same board: fine even though the file does not exist yet
    assert kb.kanban_db_path("alpha") == ghost
    with pytest.raises(kb.BoardPinConflict):
        kb.kanban_db_path("beta")
