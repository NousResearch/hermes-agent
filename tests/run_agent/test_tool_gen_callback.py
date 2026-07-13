from run_agent import AIAgent


def test_fire_tool_gen_started_passes_generation_key_to_aware_callback():
    agent = object.__new__(AIAgent)
    calls = []

    def callback(name, *, generation_key=None):
        calls.append((name, generation_key))

    agent.tool_gen_callback = callback

    agent._fire_tool_gen_started("read_file", generation_key="chat:0")

    assert calls == [("read_file", "chat:0")]


def test_fire_tool_gen_started_falls_back_for_legacy_callback():
    agent = object.__new__(AIAgent)
    calls = []

    def callback(name):
        calls.append(name)

    agent.tool_gen_callback = callback

    agent._fire_tool_gen_started("read_file", generation_key="chat:0")

    assert calls == ["read_file"]
