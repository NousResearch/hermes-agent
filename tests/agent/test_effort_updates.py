"""Mid-conversation effort markers (port of anomalyco/opencode#48513)."""
from pathlib import Path

from agent.anthropic_adapter import build_anthropic_kwargs
from agent.effort_updates import (
    ANTHROPIC_MID_CONVERSATION_EFFORT_BETA, EFFORT_UPDATE_KEY, effort_update, make_effort_update_message,
    record_effort_switch,
)


def _wire_copy(history):
    """What turn_context hands the transport: display_* popped, marker payload carried top-level."""
    out = []
    for msg in history:
        api_msg = {k: v for k, v in msg.items() if k not in ("display_kind", "display_metadata")}
        if (update := effort_update(msg)) is not None:
            api_msg[EFFORT_UPDATE_KEY] = update
        out.append(api_msg)
    return out


HISTORY = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
    make_effort_update_message("low", "high"),
    {"role": "user", "content": "again"},
]


def test_anthropic_lowers_markers_in_band_and_freezes_top_level_effort():
    kwargs = build_anthropic_kwargs("claude-fable-5-1", _wire_copy(HISTORY), None, None, {"enabled": True, "effort": "low"})
    assert kwargs["output_config"] == {"effort": "high"}  # baseline the cached prefix was built with
    assert [m["role"] for m in kwargs["messages"]] == ["user", "assistant", "system", "user"]
    assert kwargs["messages"][2] == {"role": "system", "content": [], "output_config": {"effort": "low"}}
    assert ANTHROPIC_MID_CONVERSATION_EFFORT_BETA in kwargs["extra_headers"]["anthropic-beta"]

    # Unsupported model / third-party endpoint / reverted history: markers stripped, plain top-level effort.
    for model, base_url, effort in (
        ("claude-sonnet-4-6", None, "low"),
        ("claude-fable-5-1", "https://api.moonshot.cn/anthropic", "low"),
        ("claude-fable-5-1", None, "high"),
    ):
        kwargs = build_anthropic_kwargs(model, _wire_copy(HISTORY), None, None, {"enabled": True, "effort": effort}, base_url=base_url)
        assert [m["role"] for m in kwargs["messages"]] == ["user", "assistant", "user"]
        assert kwargs["output_config"] == {"effort": effort}
        assert "extra_headers" not in kwargs


def test_record_effort_switch_reads_session_baseline_then_last_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB

    db = SessionDB(db_path=Path(tmp_path) / "state.db")
    db.create_session("s", "cli", model="claude-fable-5-1", model_config={"reasoning_config": {"enabled": True, "effort": "high"}})

    class Agent:
        _session_db = db
        session_id = "s"
        _session_init_model_config = {"reasoning_config": {"enabled": True, "effort": "low"}}  # re-created agent
        reasoning_config = {"enabled": True, "effort": "low"}

    agent = Agent()
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert record_effort_switch(agent, messages) is True
    assert effort_update(messages[-1]) == {"effort": "low", "previous": "high"}
    assert messages[-1]["display_kind"] == "hidden"
    messages += [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert record_effort_switch(agent, messages) is False  # unchanged effort
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    assert record_effort_switch(agent, messages) is True
    assert effort_update(messages[-1]) == {"effort": "high", "previous": "low"}  # previous from the marker, not the DB
    # Never between a tool_use and its result, never on an empty history, never with reasoning off.
    assert record_effort_switch(agent, [{"role": "assistant", "content": "", "tool_calls": [{"id": "x"}]}]) is False
    assert record_effort_switch(agent, []) is False
    agent.reasoning_config = {"enabled": False}
    assert record_effort_switch(agent, messages + [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]) is False
