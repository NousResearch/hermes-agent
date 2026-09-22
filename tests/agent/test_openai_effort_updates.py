"""OpenAI Responses lowering for durable reasoning-effort markers."""
from pathlib import Path

from agent.effort_updates import EFFORT_UPDATE_KEY, effort_update, make_effort_update_message, record_effort_switch
from agent.transports.codex import ResponsesApiTransport


def _wire_copy(history):
    out = []
    for message in history:
        api_message = {key: value for key, value in message.items() if key not in ("display_kind", "display_metadata")}
        if (update := effort_update(message)) is not None:
            api_message[EFFORT_UPDATE_KEY] = dict(update)
        out.append(api_message)
    return out


def test_openai_effort_update_pins_baseline_and_fails_closed():
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        make_effort_update_message("low", "high"),
        {"role": "user", "content": "continue"},
    ]
    common = {
        "messages": _wire_copy(history),
        "reasoning_config": {"enabled": True, "effort": "low"},
        "context_management": [{"type": "compaction", "compact_threshold": 4_000}],
        "base_url": "https://chatgpt.com/backend-api/codex",
        "is_codex_backend": True,
    }

    transport = ResponsesApiTransport()
    supported = transport.preflight_kwargs(transport.build_kwargs(model="gpt-5.6-luna-1", **common))
    assert supported["reasoning"]["effort"] == "high"
    assert supported["input"][-2:] == [
        {"type": "configuration_update", "reasoning": {"effort": "low"}},
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "continue"}],
        },
    ]
    assert "context_management" not in supported

    custom = transport.build_kwargs(
        model="future-reasoner", **{
            **common, "base_url": "https://llm.internal.example.com/v1",
            "is_codex_backend": False, "capabilities": {"reasoning_effort_updates": True},
        }
    )
    assert any(item.get("type") == "configuration_update" for item in custom["input"])

    unsupported = transport.build_kwargs(model="gpt-5.5", **common)
    assert unsupported["reasoning"]["effort"] == "low"
    assert all(item.get("type") != "configuration_update" for item in unsupported["input"])
    assert unsupported["context_management"] == common["context_management"]


def test_official_gpt6_tiers_preserve_none_to_max_updates():
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        make_effort_update_message("none", "max"),
        {"role": "user", "content": "continue"},
    ]
    transport = ResponsesApiTransport()

    for model in ("gpt-6-luna", "gpt-6-terra", "gpt-6-sol"):
        route = {
            "model": model,
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
        }
        kwargs = transport.build_kwargs(
            messages=_wire_copy(history),
            reasoning_config={"enabled": True, "effort": "none"},
            context_management=[{"type": "compaction", "compact_threshold": 4_000}],
            **route,
        )
        prepared = transport.preflight_kwargs(kwargs)
        assert prepared["reasoning"]["effort"] == "max"
        assert prepared["input"][-2:] == [
            {"type": "configuration_update", "reasoning": {"effort": "none"}},
            {"role": "user", "content": "continue"},
        ]
        assert "context_management" not in prepared

    proxy = transport.build_kwargs(
        model="gpt-6-luna",
        messages=_wire_copy(history),
        reasoning_config={"enabled": True, "effort": "none"},
        context_management=[{"type": "compaction", "compact_threshold": 4_000}],
        provider="custom",
        base_url="https://api.openai.com.proxy.invalid/v1",
    )
    assert all(item.get("type") != "configuration_update" for item in proxy["input"])
    assert "context_management" in proxy


def test_official_astra_keeps_low_floor_while_gpt6_family_expands():
    transport = ResponsesApiTransport()
    route = {
        "model": "gpt-6-astra",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
    }
    kwargs = transport.build_kwargs(
        messages=[{"role": "user", "content": "hello"}],
        reasoning_config={"enabled": True, "effort": "none"},
        **route,
    )
    assert transport.preflight_kwargs(kwargs)["reasoning"]["effort"] == "low"


def test_model_change_starts_a_new_effort_lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB

    db = SessionDB(db_path=Path(tmp_path) / "state.db")
    db.create_session(
        "s", "cli", model="gpt-5.6-luna-1",
        model_config={"reasoning_config": {"enabled": True, "effort": "high"}},
    )

    class Agent:
        _session_db = db
        session_id = "s"
        _session_init_model_config = {"reasoning_config": {"enabled": True, "effort": "high"}}
        reasoning_config = {"enabled": True, "effort": "high"}
        provider = "openai-codex"
        model = "gpt-5.6-luna-1"
        base_url = "https://chatgpt.com/backend-api/codex"

    agent = Agent()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert record_effort_switch(agent, messages) is False
    agent.reasoning_config = {"enabled": True, "effort": "low"}
    assert record_effort_switch(agent, messages) is True
    old_lineage = effort_update(messages[-1])["lineage"]

    messages += [{"role": "user", "content": "next"}, {"role": "assistant", "content": "answer"}]
    agent.model = "gpt-5.6-sol-1"
    agent.reasoning_config = {"enabled": True, "effort": "max"}
    assert record_effort_switch(agent, messages) is True
    reset = effort_update(messages[-1])
    assert reset["reset"] is True
    assert reset["effort"] == reset["previous"] == "max"
    assert reset["lineage"] != old_lineage

    kwargs = ResponsesApiTransport().build_kwargs(
        model=agent.model, messages=_wire_copy(messages), reasoning_config=agent.reasoning_config,
        base_url=agent.base_url, provider=agent.provider, is_codex_backend=True,
    )
    assert kwargs["reasoning"]["effort"] == "max"
    assert all(item.get("type") != "configuration_update" for item in kwargs["input"])
