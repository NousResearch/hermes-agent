"""Test JIT Context Engine loading and execution within Hermes Agent."""

from agent.agent_init import _select_context_engine
from plugins.context_engine import load_context_engine


def test_jit_context_engine_loaded_by_name():
    engine = load_context_engine("jit")
    assert engine is not None
    assert engine.name == "jit"
    assert engine.threshold_tokens <= 35000


def test_agent_init_selects_jit_engine():
    cfg = {"context": {"engine": "jit"}}
    engine = _select_context_engine(cfg)
    assert engine is not None
    assert engine.name == "jit"


def test_jit_engine_wire_message_selection():
    engine = load_context_engine("jit")
    assert engine is not None

    messages = [{"role": "system", "content": "Base system prompt."}]
    for i in range(1, 15):
        messages.append({"role": "user", "content": f"User query {i}"})
        messages.append({"role": "assistant", "content": f"Assistant response {i}"})

    selected = engine.select_context(messages)
    assert len(selected) > 0
    assert selected[0]["role"] == "system"

    # User turns must be capped to last 4
    users = [m["content"] for m in selected if m["role"] == "user"]
    assert users == ["User query 11", "User query 12", "User query 13", "User query 14"]
