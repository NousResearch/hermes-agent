from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from agent.session_stats import SessionStatsSnapshot, collect_session_stats, render_stats_terminal


class StubDB:
    def __init__(self, messages):
        self._messages = messages

    def get_messages(self, session_id):
        return list(self._messages)


def test_collect_session_stats_aggregates_runtime_and_db_data():
    agent = SimpleNamespace(
        model="gpt-5.4",
        provider="openrouter",
        api_mode="chat_completions",
        base_url="https://openrouter.ai/api/v1",
        session_prompt_tokens=1200,
        session_completion_tokens=300,
        session_total_tokens=1500,
        session_api_calls=4,
        session_cache_read_tokens=400,
        session_cache_write_tokens=50,
    )
    compressor = SimpleNamespace(
        last_prompt_tokens=600,
        last_completion_tokens=120,
        last_total_tokens=720,
        context_length=128000,
        threshold_tokens=96000,
        compression_count=2,
        summarization_count=2,
        estimated_tokens_saved=900,
        summary_model="gemini-3-flash-preview",
        context_source="openrouter_metadata",
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "tool plan", "tool_calls": [{"id": "1", "function": {"name": "read_file"}}, {"id": "2", "function": {"name": "search_files"}}]},
        {"role": "tool", "tool_name": "read_file", "content": "..."},
        {"role": "tool", "tool_name": "read_file", "content": "..."},
        {"role": "tool", "tool_name": "search_files", "content": "..."},
    ]
    db = StubDB(messages)
    model_info = {
        "context_length": 128000,
        "max_completion_tokens": 8192,
        "pricing": {"prompt": "1.25", "completion": "10", "cache_read": "0.125", "cache_write": "1.25"},
        "name": "GPT 5.4",
    }

    stats = collect_session_stats(agent=agent, compressor=compressor, session_db=db, session_id="s1", model_info=model_info)

    assert stats.model == "gpt-5.4"
    assert stats.provider == "openrouter"
    assert stats.total_tokens == 1500
    assert stats.api_calls == 4
    assert stats.cache_read_tokens == 400
    assert stats.cache_write_tokens == 50
    assert stats.cache_hit_rate == pytest.approx(400 / 1200)
    assert stats.context_current_tokens == 600
    assert stats.context_max_tokens == 128000
    assert stats.context_threshold_tokens == 96000
    assert stats.context_source == "openrouter_metadata"
    assert stats.max_completion_tokens == 8192
    assert stats.message_count == 7
    assert stats.user_message_count == 1
    assert stats.assistant_message_count == 2
    assert stats.system_message_count == 1
    assert stats.tool_message_count == 3
    assert stats.tool_calls_total == 2
    assert stats.tool_calls_by_name["read_file"] == 1
    assert stats.tool_calls_by_name["search_files"] == 1
    assert stats.top_tools[0][1] == 1
    assert stats.compression_count == 2
    assert stats.summarization_count == 2
    assert stats.estimated_tokens_saved == 900
    assert stats.summary_model == "gemini-3-flash-preview"
    assert stats.model_display_name == "GPT 5.4"
    assert stats.pricing_known is True
    assert stats.input_cost_per_million == 1.25
    assert stats.output_cost_per_million == 10.0


def test_collect_session_stats_handles_missing_data_gracefully():
    agent = SimpleNamespace()

    stats = collect_session_stats(agent=agent, compressor=None, session_db=None, session_id="s1", model_info=None)

    assert stats.total_tokens == 0
    assert stats.api_calls == 0
    assert stats.cache_hit_rate is None
    assert stats.tool_calls_total == 0
    assert stats.tool_calls_by_name == {}
    assert stats.top_tools == []
    assert stats.pricing_known is False


def test_collect_session_stats_falls_back_to_tool_results_when_assistant_calls_missing():
    agent = SimpleNamespace()
    db = StubDB([
        {"role": "assistant", "content": "working"},
        {"role": "tool", "tool_name": "read_file", "content": "..."},
        {"role": "tool", "tool_name": "search_files", "content": "..."},
    ])

    stats = collect_session_stats(agent=agent, compressor=None, session_db=db, session_id="s1", model_info=None)

    assert stats.tool_calls_total == 2
    assert stats.tool_calls_by_name == {"read_file": 1, "search_files": 1}


def test_collect_session_stats_zero_prompt_tokens_avoids_division_by_zero():
    agent = SimpleNamespace(
        session_prompt_tokens=0,
        session_completion_tokens=10,
        session_total_tokens=10,
        session_api_calls=1,
        session_cache_read_tokens=99,
        session_cache_write_tokens=0,
    )

    stats = collect_session_stats(agent=agent, compressor=None, session_db=None, session_id="s1", model_info=None)

    assert stats.cache_hit_rate is None


def test_snapshot_is_frozen():
    stats = SessionStatsSnapshot()
    with pytest.raises(FrozenInstanceError):
        stats.total_tokens = 123


def test_render_stats_terminal_includes_key_sections():
    stats = SessionStatsSnapshot(
        model="gpt-5.4",
        provider="openrouter",
        api_mode="chat_completions",
        base_url="https://openrouter.ai/api/v1",
        prompt_tokens=1200,
        completion_tokens=300,
        total_tokens=1500,
        api_calls=4,
        cache_read_tokens=400,
        cache_write_tokens=50,
        cache_hit_rate=0.3333,
        context_current_tokens=600,
        context_max_tokens=128000,
        context_threshold_tokens=96000,
        context_source="openrouter_metadata",
        max_completion_tokens=8192,
        message_count=7,
        user_message_count=1,
        assistant_message_count=2,
        system_message_count=1,
        tool_message_count=3,
        tool_calls_total=4,
        top_tools=[("read_file", 2), ("search_files", 1)],
        compression_count=2,
        summarization_count=2,
        estimated_tokens_saved=900,
        summary_model="gemini-3-flash-preview",
        model_display_name="GPT 5.4",
        pricing_known=True,
        input_cost_per_million=1.25,
        output_cost_per_million=10.0,
        cache_read_cost_per_million=0.125,
        cache_write_cost_per_million=1.25,
    )

    text = render_stats_terminal(stats)

    assert "Session Diagnostics" in text
    assert "Prompt Cache" in text
    assert "33.3%" in text
    assert "read_file" in text
    assert "Model Pricing" in text


def test_render_stats_terminal_handles_empty_snapshot():
    text = render_stats_terminal(SessionStatsSnapshot())

    assert "Session Diagnostics" in text
    assert "Total" in text
