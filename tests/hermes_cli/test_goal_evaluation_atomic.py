"""Goal evaluation may commit only against the exact durable snapshot it judged."""

import pytest

from hermes_cli import goals


@pytest.mark.parametrize("timing", ["before", "gate", "judge", "shared_judge", "commit"])
@pytest.mark.parametrize("mutation", ["pause", "clear", "replace", "pause_resume"])
@pytest.mark.parametrize("verdict", ["continue", "done", "blocked", "wait"])
def test_command_mutation_wins_over_stale_evaluation(tmp_path, monkeypatch, timing, mutation, verdict):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    manager = goals.GoalManager("session")
    manager.set("original")
    manager.add_gate("true")
    command = manager if timing == "shared_judge" else goals.GoalManager("session")
    expected = []

    def mutate():
        if mutation == "pause":
            command.pause()
        elif mutation == "clear":
            command.clear()
        elif mutation == "replace":
            command.set("replacement")
        else:
            command.pause()
            command.resume(reset_budget=False)
        expected.append(db.get_meta("goal:session"))

    def gate(_gate, *, cwd):
        if timing == "gate":
            mutate()
        return True, 0, "ok"

    def judge(*args, **kwargs):
        if timing in {"judge", "shared_judge"}:
            mutate()
        return verdict, "result", False, {"seconds": 60} if verdict == "wait" else None, False

    monkeypatch.setattr(goals, "run_gate", gate)
    monkeypatch.setattr(goals, "judge_goal", judge)
    if timing == "before":
        mutate()
    elif timing == "commit":
        real_write = db._execute_write

        def racing_write(callback, *args, **kwargs):
            monkeypatch.setattr(db, "_execute_write", real_write)
            mutate()
            return real_write(callback, *args, **kwargs)

        monkeypatch.setattr(db, "_execute_write", racing_write)
    try:
        decision = manager.evaluate_after_turn("old answer")
        assert expected
        assert decision["should_continue"] is False
        assert decision["verdict"] == "interrupted"
        assert db.get_meta("goal:session") == expected[-1]
        assert manager.state.to_json() == goals.load_goal("session").to_json()
    finally:
        db.close()


@pytest.mark.parametrize("failure", ["db_unavailable", "read", "write", "gate", "judge"])
def test_evaluation_failures_preserve_goal_state(tmp_path, monkeypatch, failure):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    manager = goals.GoalManager("session")
    manager.set("original")
    if failure == "gate":
        manager.add_gate("true")
    before = db.get_meta("goal:session")
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: ("continue", "more", False, None, False))

    def fail(*args, **kwargs):
        raise OSError("disk unavailable")

    read_meta = db.get_meta
    if failure == "db_unavailable":
        monkeypatch.setattr(goals, "_get_session_db", lambda: None)
    elif failure == "read":
        monkeypatch.setattr(db, "get_meta", fail)
    elif failure == "write":
        monkeypatch.setattr(db, "_execute_write", fail)
    else:
        monkeypatch.setattr(goals, "run_gate" if failure == "gate" else "judge_goal", fail)
    try:
        if failure in {"gate", "judge"}:
            with pytest.raises(OSError, match="disk unavailable"):
                manager.evaluate_after_turn("answer")
        else:
            decision = manager.evaluate_after_turn("answer")
            assert decision["should_continue"] is False
            assert decision["verdict"] == "persistence_failed"
        assert manager.state.to_json() == before
        assert read_meta("goal:session") == before
    finally:
        db.close()
