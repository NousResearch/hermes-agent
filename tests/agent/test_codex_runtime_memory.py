from types import SimpleNamespace

from agent.codex_runtime import (
    _build_codex_app_server_memory_preamble,
    run_codex_app_server_turn,
)


class FakeMemoryStore:
    def __init__(self):
        self.loaded = False

    def load_from_disk(self):
        self.loaded = True

    def format_for_system_prompt(self, target):
        if target == "memory":
            return "Memory: User prefers Claude CLI via Max."
        if target == "user":
            return "User: Erik."
        return ""


def test_codex_app_server_memory_preamble_includes_builtin_memory():
    store = FakeMemoryStore()
    agent = SimpleNamespace(
        _memory_store=store,
        _memory_enabled=True,
        _user_profile_enabled=True,
        _memory_manager=None,
    )

    preamble = _build_codex_app_server_memory_preamble(agent, "hello")

    assert store.loaded is True
    assert "Hermes persistent memory" in preamble
    assert "User prefers Claude CLI via Max" in preamble
    assert "User: Erik" in preamble


def test_codex_app_server_turn_prepends_memory_to_codex_input():
    store = FakeMemoryStore()
    captured = {}

    class FakeCodexSession:
        def run_turn(self, *, user_input):
            captured["user_input"] = user_input
            return SimpleNamespace(
                final_text="ok",
                projected_messages=[],
                tool_iterations=0,
                interrupted=False,
                error=None,
                should_retire=False,
                thread_id="thread-1",
                turn_id="turn-1",
            )

    agent = SimpleNamespace(
        _codex_session=FakeCodexSession(),
        _memory_store=store,
        _memory_enabled=True,
        _user_profile_enabled=False,
        _memory_manager=None,
        _iters_since_skill=0,
        _skill_nudge_interval=0,
        valid_tool_names=set(),
    )

    result = run_codex_app_server_turn(
        agent,
        user_message="What were we working on?",
        original_user_message="What were we working on?",
        messages=[],
        effective_task_id="task",
    )

    assert result["final_response"] == "ok"
    assert "User prefers Claude CLI via Max" in captured["user_input"]
    assert "[Current user message]\nWhat were we working on?" in captured["user_input"]
