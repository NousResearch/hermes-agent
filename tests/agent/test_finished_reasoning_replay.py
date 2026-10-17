"""Regression coverage for opt-in finished-turn reasoning replay trimming (#138807)."""

from agent.turn_context import _preflight_request_tokens, build_api_messages


class _Agent:
    api_mode = "chat_completions"
    provider = "custom"
    model = "reasoning-model"
    base_url = "https://gateway.example/v1"
    ephemeral_system_prompt = None
    _current_turn_timestamp = 10_000.0
    _strip_finished_reasoning_replay_flag = True
    tools = None

    @staticmethod
    def _copy_reasoning_content_for_api(source, target):
        target.pop("reasoning_content", None)
        if "reasoning_content" in source:
            target["reasoning_content"] = source["reasoning_content"]
        elif source.get("tool_calls"):
            target["reasoning_content"] = " "

    @staticmethod
    def _should_sanitize_tool_calls():
        return False


def _history():
    return [
        {"role": "user", "content": "first request"},
        {
            "role": "assistant", "content": "used a tool",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
            "reasoning_content": "finished tool reasoning " * 200,
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "tool output"},
        {
            "role": "assistant", "content": "finished answer",
            "reasoning_content": "finished answer reasoning " * 200,
            "reasoning_details": [{"type": "reasoning.text", "text": "finished detail " * 200}],
        },
        {"role": "assistant", "content": "", "reasoning_content": "thinking-only history " * 200},
        {"role": "user", "content": "current request"},
        {"role": "assistant", "content": "live tool round", "reasoning_content": "current turn reasoning " * 200},
    ]


def test_opt_in_strips_only_finished_reasoning_and_preflight_matches_wire_shape():
    agent = _Agent()
    history = _history()

    wire, _ = build_api_messages(
        agent, history, current_turn_user_idx=5, ext_prefetch_cache="", plugin_user_context="",
        moa_config=None, active_system_prompt="",
    )

    assert wire[1]["reasoning_content"] == " "
    assert "reasoning_content" not in wire[3]
    assert "reasoning_details" not in wire[3]
    assert wire[4]["reasoning_content"].startswith("thinking-only history")
    assert wire[6]["reasoning_content"].startswith("current turn reasoning")
    assert history[1]["reasoning_content"].startswith("finished tool reasoning")
    assert history[3]["reasoning_content"].startswith("finished answer reasoning")

    trimmed = _preflight_request_tokens(agent, history, "")
    untrimmed_agent = _Agent()
    untrimmed_agent._strip_finished_reasoning_replay_flag = False
    untrimmed = _preflight_request_tokens(untrimmed_agent, history, "")
    assert trimmed < untrimmed


def test_config_opt_in_reaches_agent_reasoning_policy(tmp_path, monkeypatch):
    """The production config reader enables the request-only trim policy."""
    from run_agent import AIAgent

    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  strip_finished_reasoning_replay: true\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    agent = object.__new__(AIAgent)
    agent.api_mode = "chat_completions"
    agent._strip_finished_reasoning_replay_flag = agent._read_strip_finished_reasoning_replay_from_config()
    assert agent._strip_finished_reasoning_replay() is True
