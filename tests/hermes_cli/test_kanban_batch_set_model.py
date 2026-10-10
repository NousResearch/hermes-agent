"""``kanban set-model`` over many cards: ids, ``--where``, ``--all-active``,
``--reclaim``, and the one-transaction guarantee behind them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _routes(conn, *ids):
    return {t: (kb.get_task(conn, t).provider_override, kb.get_task(conn, t).model_override) for t in ids}


def _cards(conn):
    a1 = kb.create_task(conn, title="a1", assignee="alpha")
    a2 = kb.create_task(conn, title="a2", assignee="alpha")
    b1 = kb.create_task(conn, title="b1", assignee="beta")
    done = kb.create_task(conn, title="done", assignee="alpha")
    kb.complete_task(conn, done, summary="x")
    return a1, a2, b1, done


# --- primitive -------------------------------------------------------------


def test_batch_writes_every_card_in_one_commit(kanban_home):
    with kbc.connect_closing() as conn:
        a1, a2, b1, _ = _cards(conn)
        written = kb.apply_batch_route_writes(conn, [
            kb.BatchRouteWrite(task_id=t, model="m", provider="p") for t in (a1, a2, b1)
        ])
        assert written == [a1, a2, b1]
        assert set(_routes(conn, a1, a2, b1).values()) == {("p", "m")}
        kinds = [e.kind for e in kb.list_events(conn, a2)]
    assert "model_override_set" in kinds


def test_archived_card_mid_batch_rolls_back_every_card(kanban_home):
    """The race the batch primitive exists for: a card archived AFTER
    selection must not leave the cards ahead of it committed."""
    with kbc.connect_closing() as conn:
        a1, a2, b1, _ = _cards(conn)
        kb.archive_task(conn, a2)
        with pytest.raises(RuntimeError, match="archived"):
            kb.apply_batch_route_writes(conn, [
                kb.BatchRouteWrite(task_id=t, model="m", provider="p") for t in (a1, a2, b1)
            ])
        assert _routes(conn, a1, b1) == {a1: (None, None), b1: (None, None)}


def test_bad_arguments_refuse_before_any_write(kanban_home):
    with kbc.connect_closing() as conn:
        a1, a2, _, _ = _cards(conn)
        with pytest.raises(ValueError):
            kb.apply_batch_route_writes(conn, [
                kb.BatchRouteWrite(task_id=a1, model="m"),
                kb.BatchRouteWrite(task_id=a2, model=None, provider="orphan-provider"),
            ])
        assert _routes(conn, a1) == {a1: (None, None)}


def test_selector_card_that_stopped_matching_is_skipped_and_reported(kanban_home):
    with kbc.connect_closing() as conn:
        a1, a2, _, _ = _cards(conn)
        kb.assign_task(conn, a2, "beta")  # changed after a --where assignee=alpha selection
        skipped: dict = {}
        written = kb.apply_batch_route_writes(conn, [
            kb.BatchRouteWrite(task_id=t, model="m", require_assignees=frozenset({"alpha"}),
                               skip_if_unmatched=True)
            for t in (a1, a2)
        ], skipped=skipped)
        assert written == [a1]
        assert skipped == {a2: "assignee is now beta"}
        assert _routes(conn, a2) == {a2: (None, None)}


def test_named_card_that_stopped_matching_aborts_the_batch(kanban_home):
    with kbc.connect_closing() as conn:
        a1, a2, _, _ = _cards(conn)
        kb.complete_task(conn, a2, summary="done meanwhile")
        with pytest.raises(RuntimeError, match="no cards were changed"):
            kb.apply_batch_route_writes(conn, [
                kb.BatchRouteWrite(task_id=t, model="m",
                                   require_statuses=frozenset({"ready", "todo"}))
                for t in (a1, a2)
            ])
        assert _routes(conn, a1) == {a1: (None, None)}


def test_observers_fire_only_after_commit(kanban_home, monkeypatch):
    calls: list = []
    monkeypatch.setattr(kb, "notify_task_updated", lambda conn, tid, fields: calls.append(tid))
    with kbc.connect_closing() as conn:
        a1, a2, _, _ = _cards(conn)
        kb.archive_task(conn, a2)
        with pytest.raises(RuntimeError):
            kb.apply_batch_route_writes(conn, [kb.BatchRouteWrite(task_id=t, model="m") for t in (a1, a2)])
    assert calls == []


# --- CLI -------------------------------------------------------------------


def test_cli_single_card_keeps_the_original_sentence(kanban_home):
    with kbc.connect_closing() as conn:
        a1, *_ = _cards(conn)
    out = kc.run_slash(f"set-model {a1} gpt-x --provider openrouter")
    assert out.strip() == f"Set model override on {a1}: openrouter:gpt-x (applies on next dispatch)"
    out = kc.run_slash(f"set-model {a1} none")
    assert out.startswith(f"Cleared model override on {a1}")
    with kbc.connect_closing() as conn:
        assert _routes(conn, a1) == {a1: (None, None)}


def test_cli_many_ids_print_one_receipt_line_each(kanban_home):
    with kbc.connect_closing() as conn:
        a1, a2, b1, _ = _cards(conn)
    out = kc.run_slash(f"set-model {a1} {b1} gpt-x --provider p")
    assert out.splitlines() == [
        f"{a1}: route=p/gpt-x applies=next-dispatch",
        f"{b1}: route=p/gpt-x applies=next-dispatch",
    ]
    with kbc.connect_closing() as conn:
        assert _routes(conn, a2) == {a2: (None, None)}


def test_cli_where_selects_by_status_and_assignee(kanban_home):
    with kbc.connect_closing() as conn:
        a1, a2, b1, done = _cards(conn)
    out = kc.run_slash("set-model gpt-x --where status=ready,todo assignee=alpha")
    assert {line.split(":")[0] for line in out.splitlines()} == {a1, a2}
    with kbc.connect_closing() as conn:
        routes = _routes(conn, a1, a2, b1, done)
    assert routes[a1] == routes[a2] == (None, "gpt-x")
    assert routes[b1] == routes[done] == (None, None)


def test_cli_all_active_excludes_terminal_cards(kanban_home):
    with kbc.connect_closing() as conn:
        a1, a2, b1, done = _cards(conn)
    kc.run_slash("set-model --model gpt-x --all-active")
    with kbc.connect_closing() as conn:
        routes = _routes(conn, a1, a2, b1, done)
    assert routes[done] == (None, None)
    assert {routes[t] for t in (a1, a2, b1)} == {(None, "gpt-x")}


@pytest.mark.parametrize("line,needle", [
    ("set-model gpt-x", "needs task ids"),
    ("set-model {a1} gpt-x --all-active", "not both"),
    ("set-model gpt-x --where color=red", "unsupported --where key"),
    ("set-model gpt-x --where status=done", "not an active status"),
    ("set-model {a1} gpt-x other-x", "at most one model"),
    ("set-model {a1} gpt-x --model gpt-y", "both positionally and with --model"),
    ("set-model {a1} t_nope gpt-x", "no such task: t_nope"),
    ("set-model {done} gpt-x", "cannot set model override on done task"),
])
def test_cli_refusals_write_nothing(kanban_home, line, needle):
    with kbc.connect_closing() as conn:
        a1, a2, b1, done = _cards(conn)
    out = kc.run_slash(line.format(a1=a1, done=done))
    assert needle in out
    with kbc.connect_closing() as conn:
        assert set(_routes(conn, a1, a2, b1, done).values()) == {(None, None)}


def test_cli_selector_that_matches_nothing_is_an_error(kanban_home):
    out = kc.run_slash("set-model gpt-x --where assignee=nobody")
    assert "matched no active tasks" in out


def test_cli_reclaim_releases_running_cards_after_commit(kanban_home):
    with kbc.connect_closing() as conn:
        a1, a2, *_ = _cards(conn)
        assert kb.claim_task(conn, a1) is not None
    out = kc.run_slash(f"set-model {a1} {a2} gpt-x --reclaim")
    assert f"{a1}: route=gpt-x applies=redispatch" in out
    # a2 was never claimed: reclaim is a designed no-op, not an error.
    assert f"{a2}: route=gpt-x applies=next-dispatch" in out
    assert "reclaim failed" not in out
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, a1).status == "ready"
        assert _routes(conn, a1, a2) == {a1: (None, "gpt-x"), a2: (None, "gpt-x")}


def test_cli_reclaim_failure_keeps_the_receipt(kanban_home, monkeypatch):
    """Reclaim runs after the routes commit; a failure there must not hide
    which cards moved."""
    with kbc.connect_closing() as conn:
        a1, a2, *_ = _cards(conn)

    def boom(conn, task_id, reason=None):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(kb, "reclaim_task", boom)
    out = kc.run_slash(f"set-model {a1} {a2} gpt-x --reclaim")
    assert f"{a1}: route=gpt-x applies=next-dispatch" in out
    assert f"{a2}: route=gpt-x applies=next-dispatch" in out
    assert f"{a1}: route applied but reclaim failed: OperationalError: database is locked" in out
    with kbc.connect_closing() as conn:
        assert _routes(conn, a1, a2) == {a1: (None, "gpt-x"), a2: (None, "gpt-x")}
