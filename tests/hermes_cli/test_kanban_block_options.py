"""Structured block options — the blocking worker's choices travel as DATA.

Contract (identical to the one documented on ``kanban_block`` and on
``kanban_db.block_task``):

* ``block_task(options=[{label, value}])`` validates the list and persists it on
  the block event payload, so the resolver behind a press — whose callback only
  carries task id + index — can re-read the label/value pair from the card
  (``block_options_for_task``).
* The canonical prose fallback line ``OPTIONS: a=<text> | b=<text>`` is
  recognised so a card blocked by hand (or by an older worker) keeps its
  choices. Exact-line-only: malformed, duplicated or free-form prose yields NO
  options — a wrong button is worse than none.
* A block with no options leaves the payload exactly as it was before the field
  existed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

OPTIONS = [{"label": "Relax the daily cap to 3", "value": "a"},
           {"label": "Keep it at 1 and accept the gap", "value": "b"}]


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    """Create a task and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


def _payload(conn, tid, kind="blocked"):
    return [e for e in kb.list_events(conn, tid) if e.kind == kind][-1].payload or {}


# ---------------------------------------------------------------------------
# Round trip: declared options land on the payload and are re-readable
# ---------------------------------------------------------------------------


def test_block_with_options_persists_them_on_the_event_payload(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="Which daily cap?", kind="needs_input",
                             options=OPTIONS) is True
        payload = _payload(conn, tid)
        assert payload["options"] == OPTIONS
        assert payload["kind"] == "needs_input"
        got = kb.block_options_for_task(conn, tid)
        assert got["origin"] == "declared"
        assert got["options"] == OPTIONS
        assert got["event_kind"] == "blocked"


def test_declared_options_are_mirrored_into_the_stored_reason(kanban_home: Path) -> None:
    """Text-only readers (board UI, `kanban show`, older watchers) keep the
    choices: the canonical line is appended to the reason."""
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="Which daily cap?", kind="needs_input",
                      options=OPTIONS)
        reason = _payload(conn, tid)["reason"]
        assert reason.startswith("Which daily cap?")
        assert ("OPTIONS: a=Relax the daily cap to 3 | "
                "b=Keep it at 1 and accept the gap") in reason
        # the two forms agree by construction
        assert kb.parse_block_options(reason) == OPTIONS


def test_existing_options_line_is_left_verbatim(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        reason = "Which daily cap?\nOPTIONS: a=Relax it | b=Keep it"
        kb.block_task(conn, tid, reason=reason, kind="needs_input", options=OPTIONS)
        payload = _payload(conn, tid)
        assert payload["reason"] == reason          # untouched, not duplicated
        assert payload["options"] == OPTIONS        # data still wins on read
        assert kb.block_options_for_task(conn, tid)["origin"] == "declared"


def test_options_ride_every_block_route(kanban_home: Path) -> None:
    """A block that leaves the `blocked` lane still carries its choices.

    ``dependency`` with no incomplete parent re-kinds to ``needs_input``
    (there is nothing to wait for, so ``todo`` would only have it re-promoted
    and respawned); the declared ``options`` ride whatever route it lands on,
    because ``_route_block`` copies them onto every block payload.
    """
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="waiting on the parent", kind="dependency",
                             options=OPTIONS) is True
        assert kb.get_task(conn, tid).status == "blocked"
        payload = _payload(conn, tid, kind="blocked")
        assert payload["options"] == OPTIONS
        assert payload["kind"] == "needs_input"
        assert payload["requested_kind"] == "dependency"
        assert payload["rekind_reason"] == "no_open_parent"
        got = kb.block_options_for_task(conn, tid)
        assert got["origin"] == "declared" and got["event_kind"] == "blocked"


def test_block_without_options_is_byte_identical_to_today(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="Which daily cap?", kind="needs_input") is True
        payload = _payload(conn, tid)
        assert "options" not in payload
        assert payload["reason"] == "Which daily cap?"
        assert set(payload) == {"reason", "kind", "recurrences", "source_status"}
        assert payload["recurrences"] == 1
        got = kb.block_options_for_task(conn, tid)
        assert got["options"] == [] and got["origin"] == "none"
        assert got["reason"] == "Which daily cap?"


def test_empty_options_list_declares_nothing(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="nothing to pick", options=[])
        assert "options" not in _payload(conn, tid)


# ---------------------------------------------------------------------------
# Prose fallback: the exact-line-only rule
# ---------------------------------------------------------------------------


def test_prose_fallback_line_is_recognised_for_a_hand_blocked_card(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="Which daily cap?\nOPTIONS: a=Relax it | b=Keep it",
                      kind="needs_input")
        assert "options" not in _payload(conn, tid)
        got = kb.block_options_for_task(conn, tid)
        assert got["origin"] == "prose"
        assert got["options"] == [{"label": "Relax it", "value": "a"},
                                  {"label": "Keep it", "value": "b"}]


@pytest.mark.parametrize("reason", [
    "reply (a) or (b)",                                  # free-form prose
    "Pick one:\n1. relax it\n2. keep it",                # numbered prose
    "OPTIONS: no pairs here",                            # no id=label pair
    "OPTIONS: a=One | broken",                           # malformed part
    "OPTIONS: a=One\nOPTIONS: b=Two",                    # two such lines
    " OPTIONS: a=One",                                   # not at column 0
    "see OPTIONS: a=One mid-line",                       # mid-line, not a line
    "OPTIONS: ",                                         # empty body
    "OPTIONS: a=One | b=Two | c=Three | d=Four | e=Five",  # over the cap
    "OPTIONS: a=One | a=Two",                            # duplicate ids
    "OPTIONS: a b=One",                                  # id with a space
    "OPTIONS: =One",                                     # empty id
    "OPTIONS: a=",                                       # empty label
])
def test_prose_is_never_guessed_into_buttons(reason: str) -> None:
    assert kb.parse_block_options(reason) == []


def test_parse_handles_no_reason() -> None:
    assert kb.parse_block_options(None) == []
    assert kb.parse_block_options("") == []


# ---------------------------------------------------------------------------
# Validation: a bad list is an error, never half-persisted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    "a",                                                     # not a list
    {"label": "x", "value": "y"},                            # a dict, not a list
    [{"label": "x", "value": "y"}] * 5,                      # over the cap
    ["label only"],                                          # not an object
    [{"value": "a"}],                                        # no label
    [{"label": "x"}],                                        # no value
    [{"label": "  ", "value": "a"}],                         # blank label
    [{"label": "x", "value": "a b"}],                        # id with a space
    [{"label": "x", "value": "a=b"}],                        # id with '='
    [{"label": "x", "value": "a|b"}],                        # id with '|'
    [{"label": "one | two", "value": "a"}],                  # label with '|'
    [{"label": "one\ntwo", "value": "a"}],                   # label over two lines
    [{"label": "x" * 49, "value": "a"}],                     # label over the cap
    [{"label": "x", "value": "a" * 33}],                     # value over the cap
    [{"label": "one", "value": "a"}, {"label": "two", "value": "a"}],  # dup ids
])
def test_normalize_rejects_malformed_lists(bad) -> None:
    with pytest.raises(ValueError):
        kb.normalize_block_options(bad)


def test_normalize_accepts_and_normalises() -> None:
    assert kb.normalize_block_options(None) is None
    assert kb.normalize_block_options([]) is None
    assert kb.normalize_block_options([{"label": "  Yes  ", "value": " a ",
                                        "extra": "ignored"}]) == [{"label": "Yes", "value": "a"}]
    four = kb.normalize_block_options([{"label": f"L{i}", "value": str(i)} for i in range(4)])
    assert len(four) == 4


def test_block_task_rejects_bad_options_without_touching_the_card(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        with pytest.raises(ValueError):
            kb.block_task(conn, tid, reason="pick one", options=[{"label": "only a label"}])
        assert kb.get_task(conn, tid).status == "running"
        assert not [e for e in kb.list_events(conn, tid) if e.kind == "blocked"]
