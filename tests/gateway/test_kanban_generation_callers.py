"""Production generated-board readers use the selected generation end to end."""

import json
import sqlite3
from types import SimpleNamespace

import pytest


def _published_board(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_notify import add_notify_sub
    from hermes_cli.kanban_generation import (
        acquire_fence,
        activate_first_write,
        publish_prepared,
        release_fence,
    )
    from hermes_cli.kanban_reconcile import reconcile
    from tests.hermes_cli.test_kanban_recovery_round_trip import _db

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    kb.create_board("default")
    control = kb.board_dir("default")
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    _db(source, "source")
    _db(target, "selected")
    with sqlite3.connect(target) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("UPDATE tasks SET status='ready' WHERE id='selected'")
        conn.commit()
        add_notify_sub(conn, task_id="selected", platform="tui", chat_id="session")
        conn.execute(
            "INSERT INTO task_events(id, task_id, kind, created_at) VALUES (11, 'selected', 'completed', 2)"
        )
        conn.commit()
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps({"task_ids": ["source"]}))
    generation = control / "generations" / "g1"
    report = reconcile(source, target, seeds, generation, apply=True)
    fence = acquire_fence(control, generation="g1", owner="test")
    publish_prepared(
        generation,
        expected_manifest=report["manifest_digest"],
        fence_token=fence["token"],
    )
    activate_first_write(generation, fence_epoch=fence["epoch"])
    release_fence(control, token=fence["token"])
    legacy = control / "kanban.db"
    if legacy.exists():
        legacy.unlink()
    return kb, control, generation


def test_dashboard_tui_and_idle_read_selected_generation_without_legacy(tmp_path, monkeypatch):
    kb, _control, generation = _published_board(tmp_path, monkeypatch)

    from gateway.run_idle_exit import _kanban_busy
    from plugins.kanban.dashboard.plugin_api import _EventTail, _board_counts
    from tui_gateway.session_notifications import _kb_board_key, _kb_poll_board

    assert _board_counts("default")["ready"] == 1
    tail = _EventTail("default")
    try:
        assert tail._latest() >= 11
        assert list((generation / "leases").glob("*.json"))
    finally:
        tail._close()
    assert not list((generation / "leases").glob("*.json"))

    slug, selected = _kb_board_key(kb, {"slug": "default"})
    assert slug == "default" and selected == str((generation / "kanban.db").resolve())
    assert _kb_poll_board(kb, "default", "session")
    assert _kanban_busy() == "kanban work on board default"


def test_notifier_and_dispatcher_open_selected_generation_and_fail_closed_on_drift(tmp_path, monkeypatch):
    kb, control, generation = _published_board(tmp_path, monkeypatch)
    from gateway import kanban_watchers_dispatcher as dispatcher
    from gateway.kanban_watchers_notifier import _Collector
    from hermes_cli.kanban_generation import GenerationError, write_pointer

    collector = _Collector.__new__(_Collector)
    collector.kb = kb
    collector.deliveries = []
    collector.gc_due = False
    collector.gc_retention_days = 30
    collector.notifier_profiles = set()
    collector.include_unowned = True
    seen = []
    collector._board_has_subs = lambda _slug: True
    collector._claim_for_sub = lambda conn, _slug, _sub: seen.append(
        getattr(conn, "_kanban_generation_id", None)
    )
    collector.collect_board("default")
    assert seen and set(seen) == {"g1"}
    assert not list((generation / "leases").glob("*.json"))

    runtime = dispatcher._KanbanDispatcher(kb, dispatcher._resolve_dispatcher_settings({}, kb))
    fingerprint = runtime.board_db_fingerprint("default")
    assert fingerprint[0] == str((generation / "kanban.db").resolve())
    monkeypatch.setattr(
        dispatcher,
        "_kbd",
        lambda: SimpleNamespace(
            dispatch_once=lambda conn, **_kwargs: getattr(conn, "_kanban_generation_id", None)
        ),
    )
    assert runtime.tick_once_for_board("default") == "g1"

    pointer = json.loads((control / "generation.current").read_text())
    write_pointer(
        control,
        {**pointer, "generation_id": "replacement", "generation_nonce": "replacement"},
    )
    from hermes_cli.kanban_db_connect import connect
    with pytest.raises(GenerationError):
        connect(board="default")
    assert collector.collect_board("default") is None
    with pytest.raises(GenerationError):
        runtime.board_db_fingerprint("default")
