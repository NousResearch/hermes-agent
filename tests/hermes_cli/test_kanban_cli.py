"""Tests for the kanban CLI surface (hermes_cli.kanban)."""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Workspace flag parsing
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# run_slash smoke tests (end-to-end via the same entry both CLI and gateway use)
# ---------------------------------------------------------------------------



def test_kanban_list_json_includes_session_id(kanban_home):
    """JSON output exposes `session_id` so external clients (Scarf, web
    dashboards) don't need a side query to filter by chat session."""
    from hermes_cli import kanban_db as kb
    with kb.connect() as conn:
        kb.create_task(
            conn, title="acp task", assignee="alice", session_id="acp-x"
        )
    raw = kc.run_slash("list --json")
    payload = json.loads(raw)
    assert any(
        row.get("title") == "acp task"
        and row.get("session_id") == "acp-x"
        for row in payload
    )


def test_kanban_show_text_renders_graph_with_open_connection(kanban_home):
    with kb.connect_closing() as conn:
        parent_id = kb.create_task(conn, title="parent task")
        child_id = kb.create_task(conn, title="child task")
        kb.link_tasks(conn, parent_id=parent_id, child_id=child_id)

    output = kc.run_slash(f"show {child_id}")

    assert f"Task {child_id}: child task" in output
    assert f"parents:   {parent_id}" in output
    assert "Cannot operate on a closed database" not in output


def test_board_override_is_isolated_per_concurrent_call(kanban_home, monkeypatch):
    kb.create_board("alpha")
    kb.create_board("beta")

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)

    barrier = threading.Barrier(2)
    original_init_db = kb.init_db

    def slow_init_db(*args, **kwargs):
        try:
            barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            pass
        return original_init_db(*args, **kwargs)

    monkeypatch.setattr(kb, "init_db", slow_init_db)

    failures: list[str] = []

    def worker(board: str, title: str) -> None:
        args = parser.parse_args(["kanban", "--board", board, "create", title])
        rc = kc.kanban_command(args)
        if rc != 0:
            failures.append(f"{board}:{rc}")

    t1 = threading.Thread(target=worker, args=("alpha", "alpha-task"))
    t2 = threading.Thread(target=worker, args=("beta", "beta-task"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert failures == []

    with kb.connect_closing(board="alpha") as conn:
        alpha_titles = [row.title for row in kb.list_tasks(conn, limit=100)]
    with kb.connect_closing(board="beta") as conn:
        beta_titles = [row.title for row in kb.list_tasks(conn, limit=100)]

    assert alpha_titles == ["alpha-task"]
    assert beta_titles == ["beta-task"]


# ---------------------------------------------------------------------------
# Integration with the COMMAND_REGISTRY
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# reclaim + reassign CLI smoke tests
# ---------------------------------------------------------------------------

def test_run_slash_reclaim_running_task(kanban_home):
    import re
    import time
    import secrets
    from hermes_cli import kanban_db as kb

    out1 = kc.run_slash("create 'stuck worker task' --assignee broken-model")
    m = re.search(r"(t_[a-f0-9]+)", out1)
    assert m
    tid = m.group(1)

    # Simulate a running claim outside TTL.
    conn = kb.connect()
    try:
        lock = secrets.token_hex(4)
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=? WHERE id=?",
            (lock, int(time.time()) + 3600, 4242, tid),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (tid, lock, int(time.time()) + 3600, 4242, int(time.time())),
        )
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (rid, tid))
        conn.commit()
    finally:
        conn.close()

    out = kc.run_slash(f"reclaim {tid} --reason 'test'")
    assert "Reclaimed" in out, out
    # Status back to ready.
    out2 = kc.run_slash(f"show {tid}")
    assert "ready" in out2.lower()




# ---------------------------------------------------------------------------
# /kanban specify — slash surface (same entry point CLI + gateway use)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI create → configured-home auto-subscription fallback
# ---------------------------------------------------------------------------


@pytest.fixture
def home_channels_env(monkeypatch):
    """Simulate telegram + discord homes configured via env overlays."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc:fake")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "1234567")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL_THREAD_ID", "42")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "disc_fake")
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "9999999")


def _cli_sub_keys(task_id):
    with kb.connect_closing() as conn:
        subs = kb.list_notify_subs(conn, task_id)
    return sorted(
        (s["platform"], s["chat_id"], s.get("thread_id") or "") for s in subs
    )


def test_cli_create_auto_subscribes_configured_homes_when_unattached(
    kanban_home, home_channels_env,
):
    """`hermes kanban create ...` from a plain shell (no gateway/TUI origin
    session) must land notify subscriptions on every configured home channel
    so dispatcher-completed tasks reach the board owner."""
    raw = kc.run_slash(
        'create "cli-orchestrated task" --assignee alice --json'
    )
    task = json.loads(raw)
    assert task["id"], raw

    assert _cli_sub_keys(task["id"]) == [
        ("discord", "9999999", ""),
        ("telegram", "1234567", "42"),
    ]


def test_cli_create_skips_homes_when_gateway_origin_present(
    kanban_home, home_channels_env, monkeypatch,
):
    """A gateway-originated CLI create (session vars set, as the /kanban
    slash handler does before run_slash) must NOT double-subscribe homes —
    the origin subscription is owned by the slash handler."""
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "chat-99")

    raw = kc.run_slash(
        'create "gateway-originated task" --assignee alice --json'
    )
    task = json.loads(raw)
    assert task["id"], raw

    # No home fallback row: the origin subscription is owned by the slash
    # handler and the CLI must not add duplicate homes on top.
    assert _cli_sub_keys(task["id"]) == []


def test_cli_create_survives_broken_homes_config(
    kanban_home, home_channels_env, monkeypatch,
):
    """A homes-config failure must not fail the CLI create (resilience),
    and --json stdout must stay clean machine-parseable output."""
    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(kb, "subscribe_task_to_configured_homes", _boom)
    raw = kc.run_slash(
        'create "resilient create" --assignee alice --json'
    )
    task = json.loads(raw)
    assert task["id"], raw


def test_decompose_triage_task_inherits_root_notify_subscriptions(kanban_home):
    """Swarm decomposition must not lose delivery: every child produced by
    decompose_triage_task inherits the root task's notify subscriptions
    (idempotent INSERT OR IGNORE on the subscription primary key), so the
    originating channel still hears when a fan-out child BLOCKs or
    completes even though only the root was explicitly subscribed — and a
    background/CLI-created root gets that subscription from the home
    fallback in the first place."""
    conn = kb.connect()
    try:
        root = kb.create_task(
            conn, title="triage root", triage=True, assignee="orchestrator",
        )
        _add_full_parent_sub(conn, root)

        child_ids = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[
                {"title": "first child", "assignee": "worker1"},
                {
                    "title": "second child", "assignee": "worker2",
                    "parents": [0],
                },
            ],
            author="triager",
            auto_promote=False,
        )
        assert child_ids is not None

        subs = [kb.list_notify_subs(conn, cid) for cid in child_ids]

        # The cursor starts caught up: no pre-link history replays.
        _, old_events = kb.unseen_events_for_sub(
            conn,
            task_id=child_ids[0],
            platform="telegram",
            chat_id="chat1",
            thread_id="topic1",
            kinds=["blocked"],
        )
    finally:
        conn.close()

    assert len(subs) == 2
    for s in subs:
        _assert_full_inherited_sub(s)
    assert old_events == []


def _add_full_parent_sub(conn, parent):
    kb.add_notify_sub(
        conn, task_id=parent, platform="telegram", chat_id="chat1",
        thread_id="topic1", user_id="user1",
        chat_type="dm", notifier_profile="default",
        delivery_mode="notify+wake",
        delivery_metadata={"reply_fallback": "general"},
    )


def _assert_full_inherited_sub(subs):
    assert len(subs) == 1
    s = subs[0]
    assert s["platform"] == "telegram"
    assert s["chat_id"] == "chat1"
    assert s["thread_id"] == "topic1"
    assert s["user_id"] == "user1"
    assert s["chat_type"] == "dm"
    assert s["delivery_mode"] == "notify+wake"


