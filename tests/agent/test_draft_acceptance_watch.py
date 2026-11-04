"""Behaviour contract for the speculative-acceptance loop signal."""
import logging

from agent.draft_acceptance_watch import CUT_AFTER_S, DraftAcceptanceWatch, metrics_url, read_counters
from tests.agent import test_stream_runaway_repetition as runaway_stream
from tests.agent.test_stream_runaway_repetition import _CYCLE_125650, _run_turn

endpoint = runaway_stream.endpoint


def _metrics(draft, accepted):
    return (f'vllm:spec_decode_num_draft_tokens_total{{engine="0",model_name="m"}} {draft}\n'
            f'vllm:spec_decode_num_accepted_tokens_total{{engine="0",model_name="m"}} {accepted}\n')


class _Server:
    """Fake clock + /metrics: each poll advances 2 s and adds 100 draft tokens."""

    def __init__(self):
        self.t, self.draft, self.acc = 0.0, 0.0, 0.0

    def step(self, ratio):
        self.t += 2.0
        self.draft += 100
        self.acc += 100 * ratio


def _watch(server):
    return DraftAcceptanceWatch(lambda: _metrics(server.draft, server.acc), clock=lambda: server.t)


def test_cuts_only_after_sustained_full_acceptance():
    srv = _Server()
    w = _watch(srv)
    for _ in range(30):
        srv.step(0.45)
        assert not w.poll()
    fired_at = None
    for i in range(1, 60):
        srv.step(1.0)
        if w.poll():
            fired_at = i * 2.0
            break
    assert fired_at is not None and CUT_AFTER_S <= fired_at <= CUT_AFTER_S + 2.0


def test_near_full_acceptance_of_verbatim_copying_never_cuts():
    # copying a file back measured 4.6-6.0 accepted tokens/step at K=5, one 5 s stretch at 6.0
    srv = _Server()
    w = _watch(srv)
    for i in range(300):
        srv.step(1.0 if i % 4 == 0 else 0.9)
        assert not w.poll()


def test_server_without_spec_counters_disables_the_watch():
    calls = []
    w = DraftAcceptanceWatch(lambda: calls.append(1) or "vllm:num_requests_running 1\n", clock=lambda: 0.0)
    w.poll()
    w._clock = lambda: 1e9
    assert not w.poll() and len(calls) == 1


def test_only_local_endpoints_are_watched():
    assert DraftAcceptanceWatch.for_base_url("https://api.deepseek.com/v1") is None
    assert DraftAcceptanceWatch.for_base_url("http://127.0.0.1:8000/v1") is not None
    assert metrics_url("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000/metrics"
    assert read_counters(_metrics(10, 7)) == (10.0, 7.0)


def test_full_acceptance_cuts_the_stream_once(endpoint, monkeypatch, caplog):
    srv = _Server()

    def fake_for_base_url(cls, base_url):
        def fetch():
            srv.step(1.0)
            return _metrics(srv.draft, srv.acc)
        return cls(fetch, clock=lambda: srv.t)

    monkeypatch.setattr(DraftAcceptanceWatch, "for_base_url", classmethod(fake_for_base_url))
    server = endpoint("content", _CYCLE_125650, endless=True)

    with caplog.at_level(logging.WARNING):
        result = _run_turn(server, "chat_completions")

    assert (result["completed"], result["failure_reason"]) == (False, "truncated")
    assert server.requests == 1
    assert "every speculative draft accepted" in caplog.text
