"""Finite CLI parents persist terminal completions before teardown (#104511)."""

from __future__ import annotations

from types import SimpleNamespace

from hermes_state import SessionDB
from tools.process_registry import ProcessRegistry


def _completion(session_id: str, session_key: str, output: str) -> dict:
    return {
        "type": "completion",
        "session_id": session_id,
        "session_key": session_key,
        "task_id": session_key,
        "command": f"run {session_id}",
        "exit_code": 0,
        "completion_reason": "exited",
        "termination_source": "natural",
        "output": output,
    }


def test_persist_oneshot_completions_coalesces_owned_results_and_requeues_others(tmp_path):
    from hermes_cli.oneshot_completions import persist_oneshot_process_completions

    db = SessionDB(db_path=tmp_path / "state.db")
    registry = ProcessRegistry()
    owner = "finite-parent"
    db.create_session(owner, source="cli")
    db.append_message(owner, "user", content="review this")
    db.append_message(owner, "assistant", content="review started")

    registry.completion_queue.put(_completion("proc_first", owner, "first result"))
    registry.completion_queue.put({"type": "watch_match", "session_id": "proc_watch", "session_key": owner})
    registry.completion_queue.put(_completion("proc_foreign", "another-session", "secret"))
    registry.completion_queue.put(_completion("proc_second", owner, "second result"))

    persisted = persist_oneshot_process_completions(db, owner, registry=registry)

    assert persisted == ["proc_first", "proc_second"]
    rows = db.get_messages(owner)
    assert [row["role"] for row in rows] == ["user", "assistant", "user"]
    notice = rows[-1]
    assert "first result" in notice["content"]
    assert "second result" in notice["content"]
    assert notice["display_kind"] == "internal_notification"
    assert notice["display_metadata"] == {
        "delivery_kind": "oneshot_process_completion",
        "process_ids": ["proc_first", "proc_second"],
    }
    queued = []
    while not registry.completion_queue.empty():
        queued.append(registry.completion_queue.get_nowait())
    assert [event["type"] for event in queued] == ["watch_match", "completion"]
    assert queued[-1]["session_id"] == "proc_foreign"
    db.close()


def test_both_oneshot_teardown_paths_persist_after_linger_before_close(monkeypatch):
    import cli as cli_mod
    import hermes_cli.oneshot as z_mod
    import hermes_cli.oneshot_completions as completion_mod
    from tools.process_registry import process_registry

    calls: list[str] = []
    monkeypatch.setattr(
        process_registry,
        "wait_for_pending_completions",
        lambda *_a, **_k: calls.append("wait") or {"waited": [], "completed": [], "timed_out": []},
    )
    monkeypatch.setattr(
        completion_mod,
        "persist_oneshot_process_completions",
        lambda *_a, **_k: calls.append("persist") or [],
    )

    cli_agent = SimpleNamespace(
        session_id="chat-q",
        _persist_disabled=False,
        _session_messages=[],
        _session_db=SimpleNamespace(
            flush_token_counts=lambda: calls.append("flush"),
            end_session=lambda *_a: calls.append("end"),
        ),
    )
    cli = SimpleNamespace(
        agent=cli_agent,
        session_id="chat-q",
        _release_active_session=lambda: calls.append("release"),
    )
    monkeypatch.setattr(
        cli_mod,
        "_notify_single_query_session_finalize",
        lambda *_a, **_k: calls.append("notify"),
    )
    monkeypatch.setattr(cli_mod, "_run_cleanup", lambda **_k: calls.append("cleanup"))

    cli_mod._finalize_single_query(cli)
    assert calls == ["wait", "persist", "flush", "end", "notify", "cleanup", "release"]

    calls.clear()
    z_agent = SimpleNamespace(
        session_id="z-parent",
        _session_db=object(),
        shutdown_memory_provider=lambda *_a: calls.append("shutdown"),
        close=lambda: calls.append("close"),
    )
    z_mod._close_agent(z_agent, z_agent._session_db)
    assert calls == ["wait", "persist", "shutdown", "close"]
