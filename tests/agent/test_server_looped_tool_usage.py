"""Server-looped built-in tools (native web_search, x_search, ...) make the provider re-read the prompt on
every internal pass and report the SUM as input tokens. Billing keeps that sum; context accounting must
not, or one search turn on a 139K session reads as 561K and forces a needless compaction."""
from types import SimpleNamespace


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    return AIAgent(api_key="k", base_url="https://inference-api.nousresearch.com/v1", provider="nous",
                   api_mode="chat_completions", model="anthropic/claude-fable-5.1", session_id="t", platform="cli",
                   quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])


def _response(prompt, output):
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=1_629, total_tokens=prompt + 1_629,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
                            completion_tokens_details=None)
    return SimpleNamespace(usage=usage, output=output, id="resp-1", model="anthropic/claude-fable-5.1")


def _record(agent, response):
    from agent import turn_usage
    turn_usage.record_response_usage(agent, response, messages=[{"role": "user", "content": "hi"}], api_call_count=1,
                                     api_duration=0.2, compression_attempts=0, max_compression_attempts=3)


SINGLE_PASS_PROMPT = 138_742
SUMMED_PROMPT = 560_908
WEB_SEARCH_TURN = [SimpleNamespace(type="web_search_call", status="completed"),
                   SimpleNamespace(type="message", role="assistant", content=[])]


def test_summed_prompt_never_reaches_the_compressor(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        _record(a, _response(SINGLE_PASS_PROMPT, [SimpleNamespace(type="message", role="assistant", content=[])]))
        anchor_before = a._usage_anchor
        _record(a, _response(SUMMED_PROMPT, WEB_SEARCH_TURN))
        assert a.context_compressor.last_prompt_tokens == SINGLE_PASS_PROMPT
        assert a._usage_anchor is anchor_before
        assert a._last_prompt_size_tokens == SINGLE_PASS_PROMPT
    finally:
        a.close()


def test_summed_prompt_is_still_billed(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        _record(a, _response(SUMMED_PROMPT, WEB_SEARCH_TURN))
        assert a.session_prompt_tokens == SUMMED_PROMPT
    finally:
        a.close()


def test_single_pass_prompt_still_drives_the_compressor(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        _record(a, _response(SINGLE_PASS_PROMPT, [SimpleNamespace(type="function_call", name="read_file")]))
        assert a.context_compressor.last_prompt_tokens == SINGLE_PASS_PROMPT
    finally:
        a.close()


def test_detection_reads_dict_and_object_items():
    from agent.codex_responses_adapter import response_ran_server_looped_tools
    assert response_ran_server_looped_tools(SimpleNamespace(output=[{"type": "x_search_call"}]))
    assert response_ran_server_looped_tools(SimpleNamespace(output=WEB_SEARCH_TURN))
    assert not response_ran_server_looped_tools(SimpleNamespace(output=[{"type": "function_call"}]))
    assert not response_ran_server_looped_tools(SimpleNamespace(output=None))
    assert not response_ran_server_looped_tools(SimpleNamespace(choices=[]))
