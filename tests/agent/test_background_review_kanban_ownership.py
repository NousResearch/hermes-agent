"""The background-review fork never acts as the dispatcher-owned Kanban worker.

A Kanban worker's review fork inherits ``HERMES_KANBAN_TASK``/``HERMES_KANBAN_RUN_ID`` through
``os.environ``. The fork runs AFTER the worker's own turn (often after ``kanban_complete``) on a
small iteration budget; if it exhausts that budget while still counted as the task's owner, the
turn finalizer records ``timed_out`` on the card and the gateway sends a false "dispatcher will
retry" notice. The fork's turn must therefore run outside the dispatcher-owned identity.
"""

from unittest.mock import patch

from agent.delegation_context import is_dispatcher_owned_worker_context, owned_kanban_task


class _SyncThread:
    def __init__(self, *, target=None, daemon=None, name=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def _make_agent_stub(agent_cls):
    import datetime as _dt

    agent = object.__new__(agent_cls)
    agent.model = "test-model"
    agent.platform = "cli"
    agent.provider = "openai"
    agent.session_id = "sess-kanban"
    agent.quiet_mode = True
    agent._memory_store = None
    agent._memory_enabled = False
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 0
    agent._skill_nudge_interval = 5
    agent.background_review_callback = None
    agent.status_callback = None
    agent._cached_system_prompt = None
    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.enabled_toolsets = ["skills", "kanban"]
    agent.disabled_toolsets = []
    return agent


def test_review_fork_turn_does_not_own_the_workers_kanban_task(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    # Precondition: the worker process itself IS the owner.
    assert is_dispatcher_owned_worker_context()
    assert owned_kanban_task() == "t_worker"

    import run_agent

    seen = {}

    def _capture_run_conversation(self, *, user_message, **kwargs):
        seen["owned"] = is_dispatcher_owned_worker_context()
        seen["task"] = owned_kanban_task()
        return {"final_response": "Nothing to save."}

    agent = _make_agent_stub(run_agent.AIAgent)
    with patch.object(run_agent.AIAgent, "__init__", lambda self, *a, **k: None), \
         patch.object(run_agent.AIAgent, "run_conversation", _capture_run_conversation), \
         patch.object(run_agent.AIAgent, "shutdown_memory_provider", lambda self: None), \
         patch.object(run_agent.AIAgent, "close", lambda self: None), \
         patch("threading.Thread", _SyncThread):
        agent._spawn_background_review(messages_snapshot=[], review_memory=False, review_skills=True)

    assert seen, "review fork run_conversation was not reached"
    assert seen["owned"] is False
    assert seen["task"] == ""
    # The worker's own identity is restored once the fork returns.
    assert is_dispatcher_owned_worker_context()
    assert owned_kanban_task() == "t_worker"
