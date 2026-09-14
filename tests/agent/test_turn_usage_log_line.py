"""The per-call log line carries what a cost or cache investigation needs: the cache write count, the
provider's response id, and the serving upstream when the route reports one. Existing parsers read a
prefix of the line, so the fields are appended and optional."""
import logging
from types import SimpleNamespace


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    return AIAgent(api_key="k", base_url="https://inference-api.nousresearch.com/v1", provider="nous",
                   api_mode="chat_completions", model="anthropic/claude-fable-5.1", session_id="t", platform="cli",
                   quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])


def _usage(read, write, prompt):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=7, total_tokens=prompt + 7,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=read, cache_write_tokens=write),
                           completion_tokens_details=None)


def _line(agent, caplog, resp):
    from agent import turn_usage
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="agent.turn_usage"):
        turn_usage.record_response_usage(agent, resp, messages=[{"role": "user", "content": "hi"}], api_call_count=1,
                                         api_duration=0.2, compression_attempts=0, max_compression_attempts=3)
    return next(r.getMessage() for r in caplog.records if r.getMessage().startswith("API call #"))


def test_write_id_and_upstream_are_on_the_line(tmp_path, monkeypatch, caplog):
    a = _agent(tmp_path, monkeypatch)
    try:
        line = _line(a, caplog, SimpleNamespace(usage=_usage(34_283, 28_604, 62_889), id="gen-1788636728-qMa1SbYZcwjrvUzJuF1g",
                                                provider="Anthropic", model="anthropic/claude-fable-5.1"))
    finally:
        a.close()
    assert " cache=34283/62889 (55%)" in line
    assert " write=28604" in line
    assert " id=gen-1788636728-qMa1SbYZcwjrvUzJuF1g" in line
    assert " upstream=Anthropic" in line
    # the pre-existing prefix is unchanged, so older parsers keep matching
    assert line.startswith("API call #1: model=anthropic/claude-fable-5.1 provider=nous in=62889 out=7 total=62896 latency=0.2s")


def test_fields_are_omitted_when_absent(tmp_path, monkeypatch, caplog):
    a = _agent(tmp_path, monkeypatch)
    try:
        line = _line(a, caplog, SimpleNamespace(usage=_usage(0, 0, 100), id=None, model="anthropic/claude-fable-5.1"))
    finally:
        a.close()
    assert "write=" not in line and "id=" not in line and "upstream=" not in line


def test_forensics_parser_reads_the_new_fields(tmp_path):
    from evals.postmortem.forensics.logcalls import parse_logs
    log = tmp_path / "agent.log"
    log.write_text(
        "2026-09-06 19:32:08,549 INFO [s1] agent.conversation_loop: API call #3: model=anthropic/claude-fable-5.1 provider=nous "
        "in=62889 out=7 total=62896 latency=0.2s cache=34283/62889 (55%) write=28604 id=gen-1788636728-qMa1 upstream=Claude Platform on AWS\n"
        "2026-09-06 19:32:09,549 INFO [s1] agent.conversation_loop: API call #4: model=anthropic/claude-fable-5.1 provider=nous "
        "in=91079 out=7 total=91086 latency=0.2s cache=62887/91079 (69%)\n", encoding="utf-8")
    calls = parse_logs([str(log)], {"s1"})
    assert [c["n"] for c in calls] == [3, 4]
    assert calls[0]["write"] == 28604 and calls[0]["id"] == "gen-1788636728-qMa1" and calls[0]["upstream"] == "Claude Platform on AWS"
    assert "write" not in calls[1] and "id" not in calls[1]


class _StubSessionDB:
    def __init__(self):
        self.queued = []

    def queue_token_counts(self, session_id, **kwargs):
        self.queued.append((session_id, kwargs))


def test_provider_reported_cost_persisted_as_actual_delta(tmp_path, monkeypatch):
    """A provider-reported usage.cost prices the call as status="actual" and the
    delta is accumulated into actual_cost_usd, not estimated_cost_usd (#105215)."""
    a = _agent(tmp_path, monkeypatch)
    try:
        stub = _StubSessionDB()
        a._session_db = stub
        a._session_db_created = True
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=7, total_tokens=107,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
                                completion_tokens_details=None, cost=0.00002510375)
        from agent import turn_usage
        turn_usage.record_response_usage(
            a, SimpleNamespace(usage=usage), messages=[{"role": "user", "content": "hi"}],
            api_call_count=1, api_duration=0.2, compression_attempts=0, max_compression_attempts=3)
        assert a.session_cost_status == "actual"
        assert a.session_cost_source == "provider_cost_api"
        assert abs(a.session_estimated_cost_usd - 0.00002510375) < 1e-12
        (sid, kw), = stub.queued
        assert sid == "t"
        assert kw["cost_status"] == "actual"
        assert abs(kw["actual_cost_usd"] - 0.00002510375) < 1e-15
        assert kw["estimated_cost_usd"] is None
    finally:
        a.close()
