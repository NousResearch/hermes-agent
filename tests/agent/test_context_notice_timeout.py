"""The host's timeout verdict cannot be lost or cleared by a detached summary."""
import threading

from agent import conversation_compression as cc
from hermes_state import SessionDB
from tests.agent.test_compression_attempt_lifecycle import _build_agent
from tui_gateway import server


def test_host_timeout_counts_once_and_late_summary_cannot_recover(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(server, "_emit", lambda kind, sid, payload=None: events.append((kind, sid, payload)))
    db, agent = _build_agent(tmp_path, "conversation")
    callbacks = server._agent_cbs("runtime")
    agent.notice_callback, agent.notice_clear_callback = callbacks["notice_callback"], callbacks["notice_clear_callback"]
    compressor = agent.context_compressor
    compressor.tail_token_budget = 10
    messages = [{"role": "system", "content": "sys"}]
    for i in range(12):
        messages.extend([{"role": "user", "content": f"Question {i} " * 100},
                         {"role": "assistant", "content": f"Answer {i} " * 100}])

    def failed_summary(*args, **kwargs):
        compressor._last_summary_error = "summary network unavailable"
        compressor._last_summary_network_failure = True
        return None

    monkeypatch.setattr(compressor, "_generate_summary", failed_summary)
    unchanged, _ = cc.compress_context(agent, messages, "sys", force=True, approx_tokens=50_000)
    assert unchanged == messages and not [e for e in events if e[0] == "notification.show"]
    assert db.get_context_notice_state("conversation")["failure_class"] == "summary_network_failure"
    db.close()
    db = SessionDB(db_path=tmp_path / "state.db")
    _, agent = _build_agent(tmp_path, "conversation", db=db)
    agent.notice_callback, agent.notice_clear_callback = callbacks["notice_callback"], callbacks["notice_clear_callback"]
    agent.context_compressor.tail_token_budget = 10
    entered, release, done = threading.Event(), threading.Event(), threading.Event()

    def slow_summary(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return "A healthy but late summary of previous work."

    real_compress = cc.compress_context
    def observed_compress(*args, **kwargs):
        try:
            return real_compress(*args, **kwargs)
        finally:
            done.set()

    monkeypatch.setattr(agent.context_compressor, "_generate_summary", slow_summary)
    monkeypatch.setattr(cc, "compress_context", observed_compress)
    monkeypatch.setattr(cc, "resolve_context_compression_timeouts", lambda: (0.1, 10.0))
    monkeypatch.setattr(cc, "_retry_compression_on_fallback_chain", lambda **kwargs: None)
    try:
        result, _ = agent._compress_context(messages, "sys", force=True, approx_tokens=50_000)
        assert entered.wait(2)
        assert result == messages
        shows = [e for e in events if e[0] == "notification.show"]
        assert len(shows) == 1, "host timeout is a real second failure even while its worker remains blocked"
        before = db.get_context_notice_state("conversation")
        release.set()
        assert done.wait(5)
        assert not [e for e in events if e[0] == "notification.clear"]
        assert db.get_context_notice_state("conversation") == before
    finally:
        release.set()
        assert done.wait(5)
        db.close()
