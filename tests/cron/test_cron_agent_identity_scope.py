"""The cron principal (``job["origin"]``) is hook-only attribution: it must not re-scope the cron
agent's external memory provider, stamp gateway origin on the cron session row, or become the
agent's own ``_user_id`` etc. — cron keeps running under the provider default scope, as before.

Builds the REAL AIAgent through ``cron.scheduler._construct_cron_agent`` with an origin carrying a
user id and records what ``MemoryManager.initialize_all`` hands the memory provider.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.inline_tool_executors import principal_hook_fields
from agent.memory_provider import MemoryProvider
from cron.scheduler import _CronAgentSetup, _construct_cron_agent
from run_agent import _gateway_origin_json

_IDENTITY_KEYS = ("user_id", "user_name", "chat_id", "chat_type", "thread_id")
_JOB = {
    "id": "j-origin", "name": "origin", "prompt": "hi",
    "origin": {
        "platform": "discord", "chat_id": "c1", "chat_type": "channel",
        "user_id": "222", "user_name": "wife", "thread_id": "t9",
    },
}


class _RecordingProvider(MemoryProvider):
    def __init__(self):
        self._init_kwargs = {}

    @property
    def name(self) -> str:
        return "recording"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._init_kwargs = dict(kwargs)

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def sync_turn(self, user_content, assistant_content, *, session_id=""):
        pass

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({})

    def shutdown(self):
        pass


def _build_cron_agent(monkeypatch, provider, session_db=None):
    from run_agent import AIAgent

    client = SimpleNamespace(chat=SimpleNamespace(completions=MagicMock()))
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **_kw: client)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **k: [])
    # Operator config: an external memory provider is configured (default scope = provider default).
    monkeypatch.setattr("hermes_cli.config.load_config_readonly",
                        lambda *a, **k: {"memory": {"provider": "recording"}})
    monkeypatch.setattr("plugins.memory.load_memory_provider", lambda *a, **k: provider)
    setup = _CronAgentSetup(
        model="test-model",
        runtime={"api_key": "k", "base_url": "https://example.invalid/v1", "provider": "custom"},
    )
    return _construct_cron_agent(
        AIAgent, _JOB, {}, setup, workdir=None, session_id="cron-sess", session_db=session_db)


def test_cron_memory_provider_not_scoped_to_job_creator(monkeypatch):
    provider = _RecordingProvider()
    agent = _build_cron_agent(monkeypatch, provider)
    assert agent._memory_manager is not None and provider in agent._memory_manager.providers
    assert provider._init_kwargs.get("platform") == "cron"
    for key in _IDENTITY_KEYS:
        assert key not in provider._init_kwargs, provider._init_kwargs
    # The creator still reaches the tool hooks as the principal.
    fields = principal_hook_fields(agent)
    assert fields["user_id"] == "222" and fields["chat_id"] == "c1" and fields["platform"] == "cron"


def test_cron_mem0_scope_is_provider_default_not_job_creator(monkeypatch):
    from plugins.memory.mem0 import _DEFAULT_USER_ID, Mem0MemoryProvider

    provider = Mem0MemoryProvider()
    monkeypatch.setattr("plugins.memory.mem0._load_config",
                        lambda: {"api_key": "k", "agent_id": "hermes", "mode": "platform"})
    monkeypatch.setattr(Mem0MemoryProvider, "_create_backend", lambda self: MagicMock())
    monkeypatch.setattr(Mem0MemoryProvider, "is_available", lambda self: True)
    _build_cron_agent(monkeypatch, provider)
    # mem0 scopes every search/add to _user_id; cron must stay on the provider default.
    assert provider._user_id == _DEFAULT_USER_ID


def test_cron_session_row_not_stamped_with_job_creator(monkeypatch):
    session_db = MagicMock()
    session_db.get_session_title.return_value = None
    agent = _build_cron_agent(monkeypatch, _RecordingProvider(), session_db=session_db)
    assert _gateway_origin_json(agent) is None
    agent._ensure_db_session()
    kwargs = session_db.create_session.call_args.kwargs
    assert kwargs["user_id"] is None and kwargs["chat_id"] is None and kwargs["origin_json"] is None, kwargs
