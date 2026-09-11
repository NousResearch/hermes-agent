"""Real compression/SQLite regressions for independently reproduced notice defects."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent import conversation_compression as cc
from agent.context_notices import record_compression_outcome
from hermes_state import SessionDB
from tests.agent.test_compression_attempt_lifecycle import _build_agent
from tui_gateway import server


def _messages():
    messages = [{"role": "system", "content": "sys"}]
    for i in range(12):
        messages.extend([{"role": "user", "content": f"Question {i} " * 100},
                         {"role": "assistant", "content": f"Answer {i} " * 100}])
    return messages


def _agent(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(server, "_emit", lambda kind, sid, payload=None: events.append((kind, sid, payload)))
    db, agent = _build_agent(tmp_path, "conversation")
    callbacks = server._agent_cbs("runtime")
    agent.notice_callback = callbacks["notice_callback"]
    agent.notice_clear_callback = callbacks["notice_clear_callback"]
    agent.context_compressor.tail_token_budget = 10
    return db, agent, events


def _seed_failure(agent, attempt_id):
    record_compression_outcome(agent, {
        "session_id": "conversation", "attempt_id": attempt_id,
        "commit_status": "aborted", "failure_class": "summary_network_failure",
    })


def test_committed_deterministic_fallback_keeps_pre_lifecycle_verdict(tmp_path, monkeypatch):
    db, agent, events = _agent(tmp_path, monkeypatch)
    _seed_failure(agent, "prior-1")
    _seed_failure(agent, "prior-2")
    compressor = agent.context_compressor
    compressor.abort_on_summary_failure = False

    def failed_summary(*args, **kwargs):
        compressor._last_summary_error = "summary unavailable"
        return None

    monkeypatch.setattr(compressor, "_generate_summary", failed_summary)
    before_commit = []
    real_commit = agent.commit_memory_session

    def observe_commit(messages):
        before_commit.append(dict(compressor._last_compression_telemetry))
        return real_commit(messages)

    monkeypatch.setattr(agent, "commit_memory_session", observe_commit)
    messages = _messages()
    try:
        result, _ = cc.compress_context(agent, messages, "sys", force=True, approx_tokens=50_000)
        assert result != messages
        assert before_commit[0]["failure_class"] == "summary_generation_failed"
        assert before_commit[0]["fallback_used"] is True
        assert not [e for e in events if e[0] == "notification.clear"], "fallback is not primary-route recovery"
        state = db.get_context_notice_state("conversation")
        assert state["failures"] == 3
        assert state["failure_class"] == "summary_generation_failed"
        latest_notice = [e for e in events if e[0] == "notification.show"][-1][2]["text"]
        assert "compression has failed" not in latest_notice, "a committed fallback is route degradation, not failed compaction"
        assert not any(advice in latest_notice for advice in ("Check the", "/compress", "/new"))
        db.close()
        db = SessionDB(db_path=tmp_path / "state.db")
        assert db.get_context_notice_state("conversation") == state
    finally:
        db.close()


def test_detached_worker_cannot_consume_successor_terminal_identity(tmp_path, monkeypatch):
    from agent.auxiliary_client import AuxiliaryExplicitCancellation

    db, agent, events = _agent(tmp_path, monkeypatch)
    _seed_failure(agent, "prior")
    entered_a, release_a, done_a = threading.Event(), threading.Event(), threading.Event()
    entered_b, release_b = threading.Event(), threading.Event()
    calls = []

    def summary(*args, **kwargs):
        calls.append(agent._compression_attempt_id)
        if len(calls) == 1:
            entered_a.set()
            assert release_a.wait(10)
            raise AuxiliaryExplicitCancellation()
        entered_b.set()
        assert release_b.wait(10)
        return "A healthy summary of previous work."

    monkeypatch.setattr(agent.context_compressor, "_generate_summary", summary)
    messages = _messages()
    real_compress = cc.compress_context

    def primary(*args, **kwargs):
        try:
            return real_compress(*args, **kwargs)
        finally:
            done_a.set()

    monkeypatch.setattr(cc, "compress_context", primary)
    monkeypatch.setattr(cc, "resolve_context_compression_timeouts", lambda: (0.1, 10.0))
    monkeypatch.setattr(cc, "_retry_compression_on_fallback_chain", lambda **kwargs: None)
    try:
        unchanged, _ = agent._compress_context(messages, "sys", force=True, approx_tokens=50_000)
        assert entered_a.is_set() and unchanged == messages
        before = db.get_context_notice_state("conversation")
        assert before["failures"] == 2
        with ThreadPoolExecutor(max_workers=1) as pool:
            successor = pool.submit(real_compress, agent, messages, "sys", force=True, approx_tokens=50_000)
            try:
                assert entered_b.wait(5)
                release_a.set()
                assert done_a.wait(5)
                assert db.get_context_notice_state("conversation") == before, "late A must not publish using B's identity"
            finally:
                release_b.set()
            result, _ = successor.result(timeout=5)
        assert result != messages
        state = db.get_context_notice_state("conversation")
        assert state["failures"] == 0
        assert state["outcome_attempts"] == ["prior", *calls]
        assert len([e for e in events if e[0] == "notification.clear"]) == 1
    finally:
        release_a.set()
        release_b.set()
        assert done_a.wait(5)
        db.close()


@pytest.mark.parametrize("prior_failures", [0, 2])
def test_timeout_backup_commit_does_not_recover_primary_route(tmp_path, monkeypatch, prior_failures):
    from types import SimpleNamespace
    from agent import auxiliary_client, context_compressor

    db, agent, events = _agent(tmp_path, monkeypatch)
    compressor = agent.context_compressor
    messages = _messages()

    def failed_summary(*args, **kwargs):
        compressor._last_summary_error = "summary network unavailable"
        compressor._last_summary_network_failure = True
        return None

    with monkeypatch.context() as seed:
        seed.setattr(compressor, "_generate_summary", failed_summary)
        for _ in range(prior_failures):
            result, _ = cc.compress_context(agent, messages, "sys", force=True, approx_tokens=50_000)
            assert result == messages
    assert db.get_context_notice_state("conversation").get("failures", 0) == prior_failures
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def provider(**kwargs):
        calls.append((kwargs.get("provider"), kwargs.get("model")))
        if kwargs.get("provider") != "backup":
            entered.set()
            assert release.wait(10)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="A healthy summary from the selected route."), finish_reason="stop")])

    # Task config seam, not route selection: the real resolver/pin/call path runs.
    monkeypatch.setattr(auxiliary_client, "_get_auxiliary_task_config", lambda task: {
        "fallback_chain": [{"provider": "backup", "model": "backup-model", "timeout": 5}],
    })
    monkeypatch.setattr(context_compressor, "call_llm", provider)
    real_compress = cc.compress_context

    def observe(*args, **kwargs):
        try:
            return real_compress(*args, **kwargs)
        finally:
            if len(calls) == 2 and release.is_set():
                done.set()

    monkeypatch.setattr(cc, "compress_context", observe)
    monkeypatch.setattr(cc, "resolve_context_compression_timeouts", lambda: (0.1, 10.0))
    try:
        result, _ = agent._compress_context(messages, "sys", force=True, approx_tokens=50_000)
        assert entered.is_set() and not release.is_set()
        assert result != messages
        assert len(calls) == 2 and calls[1] == ("backup", "backup-model")
        if prior_failures:
            assert not [e for e in events if e[0] == "notification.clear"], "backup is not primary-route recovery"
        before = db.get_context_notice_state("conversation")
        assert before["failures"] == prior_failures + 1, "the host retry is not another summary-route failure"
        assert before["pending_usage"] == agent._compression_attempt_id
        if not prior_failures:
            assert not [e for e in events if e[0] == "notification.show"]
        release.set()
        assert done.wait(5)
        assert db.get_context_notice_state("conversation") == before
    finally:
        release.set()
        assert done.wait(5)
        db.close()
