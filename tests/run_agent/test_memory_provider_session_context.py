"""Regression tests for threading ``session_workdir`` and
``session_title_hint`` into external memory-provider ``initialize()``
kwargs.

External memory providers need two context values they cannot reconstruct
from the public ``MemoryProvider`` surface:

- ``session_workdir``: the LOGICAL session working directory (never the raw
  process CWD). Only Hermes core can resolve it, via
  ``agent.runtime_cwd.resolve_agent_cwd()``.
- ``session_title_hint``: an explicit session-title hint threaded from the
  gateway host BEFORE the session-DB row exists (e.g. ``pending_title`` in
  the lazy-session flow), so provider-side title handling does not depend
  on write ordering against the session DB.

Both are asserted end-to-end through ``AIAgent.__init__`` ->
``agent.agent_init.init_agent`` -> ``MemoryManager.initialize_all`` ->
``provider.initialize(**kwargs)``, exactly the path a real memory provider
plugin observes.
"""

from unittest.mock import patch


class RecordingMemoryProvider:
    name = "recording"

    def __init__(self):
        self.init_kwargs = None
        self.init_session_id = None

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.init_session_id = session_id
        self.init_kwargs = dict(kwargs)

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        pass


def _make_agent_with_recording_provider(provider, *, session_title_hint=None, session_id="sess-adapter"):
    cfg = {"memory": {"provider": "recording"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.runtime_cwd.resolve_agent_cwd", return_value=__import__("pathlib").Path("/fake/logical/workdir")),
    ):
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id=session_id,
            platform="cli",
            session_title_hint=session_title_hint,
        )


def test_session_workdir_is_threaded_to_memory_provider_initialize():
    """A memory provider must receive the LOGICAL session workdir (never the
    raw process CWD) as ``session_workdir`` — the one piece of context an
    external provider cannot reconstruct itself (``agent.runtime_cwd`` is an
    internal Hermes module, outside the public ``MemoryProvider`` surface).
    """
    provider = RecordingMemoryProvider()
    _make_agent_with_recording_provider(provider)

    assert provider.init_kwargs is not None
    assert provider.init_kwargs["session_workdir"] == "/fake/logical/workdir"


def test_session_title_hint_overrides_session_db_lookup():
    """``session_title_hint`` (threaded through ``AIAgent.__init__`` ->
    ``init_agent``) must win over a session-DB title lookup and must reach
    the provider as ``session_title`` even when no session DB is wired.
    This is the lazy-session case: the DB row does not exist yet at
    agent-build time, so the host passes the intended title explicitly.
    """
    provider = RecordingMemoryProvider()
    _make_agent_with_recording_provider(provider, session_title_hint="Group: room-42")

    assert provider.init_kwargs is not None
    assert provider.init_kwargs["session_title"] == "Group: room-42"


def test_missing_session_title_hint_omits_session_title_key():
    """No hint and no session DB -> no ``session_title`` kwarg is sent at all
    (not an empty string), so providers can tell "no title known" apart from
    "empty title".
    """
    provider = RecordingMemoryProvider()
    _make_agent_with_recording_provider(provider, session_title_hint=None)

    assert provider.init_kwargs is not None
    assert "session_title" not in provider.init_kwargs
