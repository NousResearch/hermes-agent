"""Regression tests for optional context-engine compression kwargs."""

from unittest.mock import MagicMock

import pytest

from run_agent import AIAgent


def _make_agent_with_engine(engine):
    agent = object.__new__(AIAgent)
    agent.context_compressor = engine
    agent.session_id = "sess-1"
    agent.model = "test-model"
    agent.platform = "cli"
    agent.logs_dir = MagicMock()
    agent.quiet_mode = True
    setattr(agent, "status_callback", None)
    agent._todo_store = MagicMock()
    agent._todo_store.format_for_injection.return_value = ""
    agent._memory_manager = None
    agent._session_db = None
    agent._cached_system_prompt = None
    setattr(agent, "_compression_feasibility_checked", True)
    setattr(agent, "compression_in_place", False)
    setattr(agent, "tools", [])
    setattr(agent, "event_callback", None)
    agent.log_prefix = ""
    agent._vprint = lambda *a, **kw: None
    agent._emit_status = lambda *a, **kw: None
    agent._emit_warning = lambda *a, **kw: None
    agent._last_flushed_db_idx = 0
    agent._invalidate_system_prompt = lambda *a, **kw: None
    agent._build_system_prompt = lambda *a, **kw: "new-system-prompt"
    agent.commit_memory_session = lambda *a, **kw: None
    return agent


def test_compress_context_skips_optional_kwargs_for_legacy_engine():
    """Older plugins without focus_topic/force in compress() signature don't crash."""
    captured_kwargs = []

    class _StrictOldPluginEngine:
        name = "strict-old"
        compression_count = 0
        last_prompt_tokens = 0
        last_completion_tokens = 0

        def compress(self, messages, current_tokens=None):
            captured_kwargs.append({"current_tokens": current_tokens})
            return [messages[0], messages[-1]]

    agent = _make_agent_with_engine(_StrictOldPluginEngine())
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]

    compressed, _ = agent._compress_context(
        messages,
        "system prompt",
        approx_tokens=100,
        focus_topic="database schema",
        force=True,
    )

    assert compressed == [messages[0], messages[-1]]
    assert captured_kwargs == [{"current_tokens": 100}]


def test_compress_context_passes_optional_kwargs_to_kwargs_engine():
    """Engines accepting **kwargs receive supported focus and force hints."""
    captured_kwargs = []

    class _KwargsEngine:
        name = "kwargs"
        compression_count = 0
        last_prompt_tokens = 0
        last_completion_tokens = 0

        def compress(self, messages, **kwargs):
            captured_kwargs.append(kwargs)
            return list(messages)

    agent = _make_agent_with_engine(_KwargsEngine())
    messages = [{"role": "user", "content": "hello"}]

    agent._compress_context(
        messages,
        "system prompt",
        approx_tokens=1234,
        focus_topic="database schema",
        force=True,
    )

    assert captured_kwargs == [
        {"current_tokens": 1234, "focus_topic": "database schema", "force": True}
    ]


def test_compress_context_preserves_focus_when_engine_lacks_force():
    """Engines updated for focus_topic but not force keep their focus hint."""
    captured_kwargs = []

    class _FocusOnlyEngine:
        name = "focus-only"
        compression_count = 0
        last_prompt_tokens = 0
        last_completion_tokens = 0

        def compress(self, messages, current_tokens=None, focus_topic=None):
            captured_kwargs.append(
                {"current_tokens": current_tokens, "focus_topic": focus_topic}
            )
            return list(messages)

    agent = _make_agent_with_engine(_FocusOnlyEngine())
    messages = [{"role": "user", "content": "hello"}]

    agent._compress_context(
        messages,
        "system prompt",
        approx_tokens=4321,
        focus_topic="database schema",
        force=True,
    )

    assert captured_kwargs == [
        {"current_tokens": 4321, "focus_topic": "database schema"}
    ]


def test_compress_context_does_not_mask_internal_type_errors():
    """Internal TypeErrors should not be retried as signature mismatch."""

    class _FailingEngine:
        name = "failing"
        compression_count = 0
        last_prompt_tokens = 0
        last_completion_tokens = 0

        def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
            raise TypeError("internal compression bug")

    agent = _make_agent_with_engine(_FailingEngine())

    with pytest.raises(TypeError, match="internal compression bug"):
        agent._compress_context(
            [{"role": "user", "content": "hello"}],
            "system prompt",
            approx_tokens=1234,
            focus_topic="database schema",
            force=True,
        )
