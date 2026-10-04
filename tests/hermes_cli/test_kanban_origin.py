"""Origin index: producer (``kanban_create`` / tui subscribe) -> ``state_meta`` -> reader.

The index is a navigation aid; every assertion that matters is about what the reader reports after
re-opening the board itself: lineage, boards, profiles, and honest per-ref failures.
"""
import json
from pathlib import Path

import pytest

from gateway.session_context import scoped_current_session_id
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from hermes_cli.kanban_origin import origin_tasks
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB
from tools import kanban_tools as kt


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Default profile + named profile ``b`` sharing one board root (the kanban root is host-wide)."""
    default = tmp_path / ".hermes"
    named = default / "profiles" / "b"
    named.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(default))
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB", "HERMES_KANBAN_HOME"):
        monkeypatch.delenv(var, raising=False)
    kb.init_db()
    return default, named


def seed_sessions(home, *ids):
    db = SessionDB(db_path=home / "state.db")
    for sid in ids:
        db.create_session(sid, source="cli")
    db.close()


def seed_compression(home, root, tip):
    db = SessionDB(db_path=home / "state.db")
    db.create_session(root, source="cli")
    db.end_session(root, "compression")
    db.create_session(tip, source="cli", parent_session_id=root)
    db.close()


def in_profile(home, fn):
    token = set_hermes_home_override(home)
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def create(**args):
    result = json.loads(kt._handle_create({"title": "t", "assignee": "default", **args}))
    assert result["ok"], result
    return result["task_id"]


def linked(result):
    return {(r["board"], r["task_id"]): r["evidence"] for r in result["refs"]}


def test_created_tasks_are_found_across_boards_and_lineage_then_failures_stay_visible(homes, monkeypatch):
    default, _ = homes
    kb.create_board("alpha")
    seed_compression(default, "root-1", "tip-1")
    # Strict no-scan: a reader that enumerated boards would trip this.
    monkeypatch.setattr(kb, "list_boards", lambda **_: pytest.fail("reader enumerated boards"))

    with scoped_current_session_id("root-1"):
        on_alpha = create(board="alpha")
    with scoped_current_session_id("tip-1"):
        on_default = create()

    # Asked by the compression TIP only: the root-stamped task on the other board still resolves.
    result = origin_tasks(["tip-1"])
    assert linked(result) == {("alpha", on_alpha): "ok", ("default", on_default): "ok"}
    assert result["truncated"]["refs"] is False

    kb.remove_board("alpha", archive=False)
    with kbc.connect_closing() as conn:
        conn.execute("DELETE FROM tasks WHERE id = ?", (on_default,))
        conn.commit()
    after = origin_tasks(["tip-1"])
    assert linked(after) == {("alpha", on_alpha): "board_missing", ("default", on_default): "task_missing"}


def test_index_and_reader_stay_inside_the_owning_profile(homes):
    from agent.secret_scope import set_multiplex_active
    from hermes_cli.web_server_profiles import _config_profile_scope

    default, named = homes
    for home in (default, named):
        seed_sessions(home, "shared-id")  # the same literal id exists in both stores

    def create_as(home):
        def run():
            with scoped_current_session_id("shared-id"):
                return create()
        return in_profile(home, run)

    task_a, task_b = create_as(default), create_as(named)

    # Reads go through the dashboard's real per-request profile scope, with the process already a
    # multi-profile host (conftest resets the latch): an unbound read would fail closed, not borrow.
    set_multiplex_active(True)

    def seen(profile):
        with _config_profile_scope(profile):
            return {t for _b, t in linked(origin_tasks(["shared-id"]))}

    assert seen(None) == {task_a}
    assert seen("b") == {task_b}
    assert seen(None) == {task_a}  # A -> B -> A
    assert seen("b") == {task_b}


def test_a_read_never_creates_a_profiles_store(homes):
    default, named = homes  # profile b exists but has never held a session

    result = in_profile(named, lambda: origin_tasks(["anything"]))

    assert result["refs"] == [] and result["unknown_sessions"] == ["anything"]
    assert not (named / "state.db").exists()


def test_worker_inherited_subscription_indexes_the_owner_never_the_worker(homes, monkeypatch):
    default, named = homes
    seed_sessions(default, "origin-1")
    with kbc.connect_closing() as conn:
        owner = kb.create_task(conn, title="owner", session_id="origin-1")
        kbn.add_notify_sub(conn, task_id=owner, platform="tui", chat_id="origin-1", notifier_profile="default")
    monkeypatch.setenv("HERMES_KANBAN_TASK", owner)

    child = in_profile(named, create)  # created by a worker running as profile b

    assert linked(origin_tasks(["origin-1"])) == {("default", owner): "ok", ("default", child): "ok"}
    worker_store = SessionDB(db_path=named / "state.db")
    try:
        assert worker_store.list_meta_prefix("kanban_origin:") == []
    finally:
        worker_store.close()


def test_subscription_for_a_missing_profile_never_falls_back_to_default(homes):
    default, _ = homes
    seed_sessions(default, "s-9")
    with kbc.connect_closing() as conn:
        task = kb.create_task(conn, title="t", session_id="s-9")
        kbn.add_notify_sub(conn, task_id=task, platform="tui", chat_id="s-9", notifier_profile="ghost")
        # The task's own stamp still links by association, but nothing was INDEXED for it.
    assert origin_tasks(["s-9"])["refs"] == []

    with kbc.connect_closing() as conn:
        kbn.add_notify_sub(conn, task_id=task, platform="tui", chat_id="s-9", notifier_profile="default")
    assert linked(origin_tasks(["s-9"])) == {("default", task): "ok"}
