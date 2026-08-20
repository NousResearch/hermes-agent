"""Regression test: ``agent_workspace`` must follow the active kanban board.

Memory providers document a ``{workspace}`` placeholder for their bank/namespace
templates (e.g. ``bank_id_template: hermes-{workspace}``), fed from the
``agent_workspace`` kwarg passed to ``MemoryProvider.initialize()``.  That kwarg
used to be hardcoded to ``"hermes"``, so the placeholder could never distinguish
anything: kanban workers running on different boards all shared a single memory
bank even when operators configured a per-workspace template.

``agent_workspace`` now follows ``HERMES_KANBAN_BOARD``.  That env var is pinned
by the kanban dispatcher for worker processes *and* by ``_pin_kanban_board_env()``
in ``cmd_chat`` at chat boot, so a plain ``hermes`` CLI chat resolves
``agent_workspace`` to the current board slug (``"default"`` on a pristine
install), not to ``"hermes"``.  The ``"hermes"`` fallback only covers contexts
where the var is genuinely unset — SDK/library construction and some gateway
paths.
"""

import json
import os

from agent.memory_provider import MemoryProvider
from run_agent import AIAgent


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")

    def close(self):
        pass


class RecordingProvider(MemoryProvider):
    """Minimal provider that records what initialize() receives."""

    def __init__(self, name="recording"):
        self._name = name
        self._init_kwargs = {}
        self._init_session_id = None

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._init_session_id = session_id
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


def _make_agent_with_recording_provider(monkeypatch, tmp_path):
    """Build an AIAgent whose only memory provider is a RecordingProvider."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)

    # agent_init imports both of these lazily inside init_agent(), so patching
    # the module attributes is what actually takes effect.
    import hermes_cli.config
    import plugins.memory

    monkeypatch.setattr(
        hermes_cli.config,
        "load_config_readonly",
        lambda *a, **kw: {"memory": {"provider": "recording"}},
    )
    provider = RecordingProvider()
    monkeypatch.setattr(
        plugins.memory, "load_memory_provider", lambda *a, **kw: provider
    )

    agent = AIAgent(
        api_key="test-key",
        base_url="http://test",
        provider="openrouter",
        api_mode="chat_completions",
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=False,
    )
    return agent, provider


def test_agent_workspace_defaults_to_hermes(monkeypatch, tmp_path):
    """No board pinned (plain CLI run) -> previous behavior is preserved."""
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    _agent, provider = _make_agent_with_recording_provider(monkeypatch, tmp_path)

    assert provider._init_kwargs.get("agent_workspace") == "hermes"


def test_agent_workspace_follows_kanban_board(monkeypatch, tmp_path):
    """A kanban worker's pinned board scopes the provider's workspace."""
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "my-project-board")
    _agent, provider = _make_agent_with_recording_provider(monkeypatch, tmp_path)

    assert provider._init_kwargs.get("agent_workspace") == "my-project-board", (
        "agent_workspace must follow HERMES_KANBAN_BOARD so memory providers "
        "can scope storage per board via the {workspace} placeholder"
    )


def test_agent_workspace_follows_chat_boot_pin(monkeypatch, tmp_path):
    """A real ``hermes`` chat boot pins the board, so the workspace is its slug.

    ``cmd_chat`` calls ``_pin_kanban_board_env()`` before constructing the agent,
    which writes ``get_current_board()`` into the environment — ``"default"`` on a
    pristine install.  This locks in that documented CLI-chat behavior end to end
    so a future change to the pin path cannot silently move operators' bank ids
    (e.g. ``bank_id_template: hermes-{workspace}``) without a failing test.
    """
    from hermes_cli.main import _pin_kanban_board_env

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    # setenv-then-delenv so monkeypatch records a restore entry even when the var
    # starts unset -- the pin below writes straight to os.environ.
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "")
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)

    _pin_kanban_board_env()
    assert os.environ["HERMES_KANBAN_BOARD"] == "default"

    _agent, provider = _make_agent_with_recording_provider(monkeypatch, tmp_path)

    assert provider._init_kwargs.get("agent_workspace") == "default"
