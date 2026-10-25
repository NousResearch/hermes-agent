"""A billed response is accounted even when the turn leaves ``check_api_response`` early.

Refusals (``finish_reason == "content_filter"``) and length-stopped responses both return
before the success path, and used to skip ``record_response_usage``: the provider bills
the call, but the session counters and state.db never saw it.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent

_DEAD_LOCAL = "http://127.0.0.1:9"


def _response(content, finish_reason, prompt, completion, cached=0):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=None), finish_reason=finish_reason, index=0,
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )
    return SimpleNamespace(id="chatcmpl-test", choices=[choice], model="gpt-4o", usage=usage)


@pytest.fixture
def loop(tmp_path, monkeypatch):
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(var, _DEAD_LOCAL)
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    monkeypatch.setattr("agent.title_generator.start_title_upgrade", lambda *a, **k: None)
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "sess-early-exit-usage"
    with patch("agent.process_bootstrap.OpenAI"), patch("agent.model_metadata.fetch_model_metadata", return_value={}):
        agent = AIAgent(
            api_key="test-key", base_url=f"{_DEAD_LOCAL}/v1", provider="openai", model="gpt-4o",
            quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
            session_db=db, session_id=sid,
        )
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._disable_streaming = True

    def run(script):
        pending = list(script)
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = lambda **_kw: pending.pop(0)
        result = agent.run_conversation("hello")
        assert not pending, "every scripted response must be requested"
        db.flush_token_counts()
        return result, db.get_session(sid)

    yield SimpleNamespace(agent=agent, run=run)
    agent.close()
    db.close()


def test_refused_response_is_accounted(loop):
    result, row = loop.run([_response("I can't help with that.", "content_filter", 1000, 50, cached=600)])

    assert result["failed"]
    assert row["api_call_count"] == loop.agent.session_api_calls == 1
    assert (row["input_tokens"], row["cache_read_tokens"], row["output_tokens"]) == (400, 600, 50)


def test_length_stopped_response_is_accounted_alongside_its_continuation(loop):
    result, row = loop.run([
        _response("A long answer that got cut", "length", 1000, 4096),
        _response(" and the rest.", "stop", 5200, 30),
    ])

    assert result["final_response"].endswith("and the rest.")
    assert row["api_call_count"] == loop.agent.session_api_calls == 2
    assert (row["input_tokens"], row["output_tokens"]) == (6200, 4126)
