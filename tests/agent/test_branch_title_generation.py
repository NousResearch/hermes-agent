"""Branch titles use new intent without renaming the parent or a manual title."""

import copy
import threading
from contextvars import ContextVar
from types import SimpleNamespace

import pytest

from agent import title_generator
from hermes_state import SessionDB


@pytest.mark.parametrize("outcome", ["generated", "manual_before", "manual_during", "failed", "disabled"])
def test_branch_title_first_followup_is_contextual_and_stable(tmp_path, monkeypatch, outcome):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("parent", source="desktop")
    db.set_session_title("parent", "Fix sidebar labels")
    db.create_session("child", source="desktop", parent_session_id="parent",
                      model_config={"_branched_from": "parent"})
    inherited = db.get_next_title_in_lineage("Fix sidebar labels")
    db.set_auto_title("child", inherited, source="branch")
    # The provisional provenance must survive a restart before the first follow-up.
    db.close()
    db = SessionDB(tmp_path / "state.db")
    history: list[dict] = [
        {"role": "system", "content": "DO NOT SEND SYSTEM PROMPTS"},
        {"role": "user", "content": "OLD CONTEXT " * 300},
        {"role": "assistant", "content": "Earlier explanation"},
        {"role": "tool", "content": "DO NOT SEND TOOL OUTPUT"},
        {"role": "user", "content": "Fix the disconnected sidebar labels"},
        {"role": "assistant", "content": "We can hide disconnected labels or explain their state."},
    ]
    followup = "Hide them instead, but keep connected labels visible."
    history.append({"role": "user", "content": [{"type": "text", "text": followup}]})
    original = copy.deepcopy(history)
    requests, events, workers = [], [], []
    started, release = threading.Event(), threading.Event()
    owner = ContextVar("title_profile", default="wrong profile")
    owner.set("branch profile")
    worker_owners = []
    real_thread = threading.Thread

    def thread(**kwargs):
        worker = real_thread(**kwargs)
        workers.append(worker)
        return worker

    def call_llm(**kwargs):
        requests.append(kwargs)
        worker_owners.append(owner.get())
        started.set()
        assert release.wait(10), "test did not release the title request"
        if outcome == "failed":
            raise RuntimeError("fixture provider unavailable")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"title": "Hide disconnected sidebar labels"}'))])

    monkeypatch.setattr(title_generator.threading, "Thread", thread)
    monkeypatch.setattr(title_generator, "call_llm", call_llm)
    monkeypatch.setattr(title_generator, "_title_config", lambda: {"enabled": outcome != "disabled"})
    if outcome == "manual_before":
        db.set_session_title("child", "My chosen branch name")
    try:
        title_generator.maybe_auto_title(db, "child", followup, history,
                                         title_callback=lambda *event: events.append(event))
        if outcome in {"manual_before", "disabled"}:
            assert not workers
            assert db.get_session_title("child") == (
                "My chosen branch name" if outcome == "manual_before" else inherited)
            return
        assert started.wait(10), "branch did not start title generation"
        assert db.get_session_title("child") == inherited
        assert db.get_session_title_source("child") == "branch_fallback"
        # A second submit while generation is blocked must not race a new title request.
        later = history + [{"role": "assistant", "content": "Okay"},
                           {"role": "user", "content": "Now test keyboard navigation"}]
        title_generator.maybe_auto_title(db, "child", "Now test keyboard navigation", later)
        assert len(workers) == 1
        if outcome == "manual_during":
            db.set_session_title("child", "My chosen branch name")
        release.set()
        workers[0].join(10)
        assert not workers[0].is_alive()
        expected = {"generated": "Hide disconnected sidebar labels",
                    "manual_during": "My chosen branch name", "failed": inherited}[outcome]
        assert db.get_session_title("child") == expected
        assert db.get_session_title("parent") == "Fix sidebar labels"
        assert history == original
        prompt = requests[0]["messages"]
        assert followup in prompt[-1]["content"]
        assert "Fix sidebar labels" in prompt[-1]["content"]
        assert "We can hide disconnected labels" in prompt[-1]["content"]
        assert "DO NOT SEND" not in str(prompt)
        assert len(prompt[-1]["content"]) <= 2 * title_generator.MAX_TITLE_INPUT_CHARS
        assert events == ([(expected, "llm")] if outcome == "generated" else [])
        title_generator.maybe_auto_title(db, "child", "Now test keyboard navigation", later)
        assert len(workers) == 1
        # Resume/compaction can reduce model history; that must not re-arm this branch.
        title_generator.maybe_auto_title(db, "child", "Now test keyboard navigation", later[-1:])
        assert len(workers) == 1
        assert worker_owners == ["branch profile"]
    finally:
        release.set()
        for worker in workers:
            worker.join(10)
        db.close()
