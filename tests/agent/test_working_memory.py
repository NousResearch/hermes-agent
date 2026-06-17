from types import SimpleNamespace

from agent.conversation_loop import _build_request_time_ephemeral_system_prompt
from agent.system_prompt import build_system_prompt
from agent.working_memory import WorkingMemory, build_working_memory_ephemeral_prompt
from tools.todo_tool import TodoStore


def _agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
        ephemeral_system_prompt=None,
        _working_memory=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_working_memory_observes_latest_user_goal():
    wm = WorkingMemory()
    wm.observe_user_turn("Continue the Hermes human-like memory architecture work")

    rendered = wm.format_for_ephemeral_system_prompt()

    assert "WORKING MEMORY (ephemeral, request-time only)" in rendered
    assert "Current goal: Continue the Hermes human-like memory architecture work" in rendered
    assert "durable memory" in rendered


def test_working_memory_includes_active_todos_only():
    store = TodoStore()
    store.write([
        {"id": "done", "content": "already finished", "status": "completed"},
        {"id": "active", "content": "implement request-time overlay", "status": "in_progress"},
        {"id": "next", "content": "run focused tests", "status": "pending"},
    ])
    wm = WorkingMemory(current_goal="PR1")
    agent = _agent(_working_memory=wm, _todo_store=store)

    rendered = build_working_memory_ephemeral_prompt(agent)

    assert "Current goal: PR1" in rendered
    assert "[in_progress] active: implement request-time overlay" in rendered
    assert "[pending] next: run focused tests" in rendered
    assert "already finished" not in rendered


def test_working_memory_is_not_in_cached_system_prompt_path():
    wm = WorkingMemory(current_goal="must stay ephemeral")
    agent = _agent(_working_memory=wm)

    cached_prompt = build_system_prompt(agent)
    request_time_prompt = _build_request_time_ephemeral_system_prompt(agent)

    assert "must stay ephemeral" not in cached_prompt
    assert "WORKING MEMORY" in request_time_prompt
    assert "must stay ephemeral" in request_time_prompt


def test_request_time_ephemeral_combines_existing_prompt_and_working_memory():
    wm = WorkingMemory(current_goal="active task")
    agent = _agent(ephemeral_system_prompt="existing ephemeral", _working_memory=wm)

    rendered = _build_request_time_ephemeral_system_prompt(agent)

    assert rendered.startswith("existing ephemeral")
    assert "WORKING MEMORY" in rendered
    assert "Current goal: active task" in rendered
