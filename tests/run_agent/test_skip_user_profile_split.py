"""Regression test for the skip_memory / skip_user_profile split.

Background: ``skip_memory=True`` was an all-or-nothing gate that disabled BOTH
MEMORY.md (agent notes) and USER.md (user knowledge) AND the external memory
provider. Some consumers want finer per-agent control — retain MEMORY.md for
agent self-improvement notes but suppress USER.md to avoid cross-agent user-
persona accumulation. The new ``skip_user_profile`` param adds that surface
without breaking existing callers.

Composition matrix:
  skip_memory=False, skip_user_profile=False  → both honour config
  skip_memory=False, skip_user_profile=True   → MEMORY.md only
  skip_memory=True,  skip_user_profile=*      → both disabled

This file pins each cell of the matrix plus a backward-compat regression
(omitting ``skip_user_profile`` from the call site behaves identically to
the pre-split behavior).
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from run_agent import AIAgent


def _make_tool_defs():
    return [{"type": "function", "function": {"name": "web_search",
             "description": "search", "parameters": {"type": "object", "properties": {}}}}]


def _mock_client(api_key="key-1234567890", base_url="https://example.com/v1"):
    c = MagicMock()
    c.api_key = api_key
    c.base_url = base_url
    c._default_headers = None
    return c


def _construct(skip_memory: bool, skip_user_profile: bool, *, both_enabled_in_config: bool = True) -> AIAgent:
    """Construct an AIAgent with both memory flags enabled in config but the
    skip flags varying per test. Patches the heavy provider-resolution path
    so __init__ runs to completion."""
    config = {
        "memory": {
            "memory_enabled": both_enabled_in_config,
            "user_profile_enabled": both_enabled_in_config,
        },
    }
    with patch("hermes_cli.config.load_config", return_value=config), \
         patch("agent.auxiliary_client.resolve_provider_client",
               return_value=(_mock_client(), "fake-model")), \
         patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI", return_value=MagicMock()), \
         patch("tools.memory_tool.MemoryStore") as MockStore:
        # Make the MemoryStore mock satisfy the load_from_disk() call without
        # actually touching ~/.hermes/memories — keeps the test hermetic.
        MockStore.return_value.load_from_disk = MagicMock()

        return AIAgent(
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_key="test-key",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=skip_memory,
            skip_user_profile=skip_user_profile,
        )


def test_neither_skip_set_honours_config():
    """skip_memory=False, skip_user_profile=False → both flags follow config."""
    agent = _construct(skip_memory=False, skip_user_profile=False)
    assert agent._memory_enabled is True
    assert agent._user_profile_enabled is True
    # _memory_store is constructed when at least one is True.
    assert agent._memory_store is not None


def test_skip_user_profile_only_retains_memory():
    """skip_memory=False, skip_user_profile=True → MEMORY.md kept, USER.md off.

    This is the new capability — fine-grained per-agent disable that the
    pre-split code couldn't express. MemoryStore still constructed because
    _memory_enabled is True; USER.md writes get gated downstream by
    _user_profile_enabled.
    """
    agent = _construct(skip_memory=False, skip_user_profile=True)
    assert agent._memory_enabled is True
    assert agent._user_profile_enabled is False
    assert agent._memory_store is not None  # at least one store still active


def test_skip_memory_overrides_skip_user_profile_false():
    """skip_memory=True (with skip_user_profile=False) → both off, MemoryStore not built.

    skip_memory is the all-or-nothing gate; skip_user_profile cannot override
    its disable behavior. Composition rule: skip_memory wins.
    """
    agent = _construct(skip_memory=True, skip_user_profile=False)
    assert agent._memory_enabled is False
    assert agent._user_profile_enabled is False
    assert agent._memory_store is None


def test_skip_memory_overrides_skip_user_profile_true():
    """skip_memory=True, skip_user_profile=True → both off. Same as the
    skip_user_profile=False sibling — proves skip_memory wins regardless."""
    agent = _construct(skip_memory=True, skip_user_profile=True)
    assert agent._memory_enabled is False
    assert agent._user_profile_enabled is False
    assert agent._memory_store is None


def test_backward_compat_skip_user_profile_defaults_false():
    """Existing callers (constructing AIAgent without passing skip_user_profile)
    must observe identical behavior to before the split — both flags honour
    config when skip_memory is False.
    """
    config = {
        "memory": {"memory_enabled": True, "user_profile_enabled": True},
    }
    with patch("hermes_cli.config.load_config", return_value=config), \
         patch("agent.auxiliary_client.resolve_provider_client",
               return_value=(_mock_client(), "fake-model")), \
         patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI", return_value=MagicMock()), \
         patch("tools.memory_tool.MemoryStore") as MockStore:
        MockStore.return_value.load_from_disk = MagicMock()

        # Note: NO skip_user_profile argument — exercising the default.
        agent = AIAgent(
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_key="test-key",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )
        assert agent._memory_enabled is True
        assert agent._user_profile_enabled is True


def test_user_profile_disabled_in_config_stays_disabled():
    """Config-driven user_profile_enabled=False is preserved when
    skip_user_profile=False — the new param doesn't FORCE user profile on,
    it can only suppress.
    """
    config = {
        "memory": {"memory_enabled": True, "user_profile_enabled": False},
    }
    with patch("hermes_cli.config.load_config", return_value=config), \
         patch("agent.auxiliary_client.resolve_provider_client",
               return_value=(_mock_client(), "fake-model")), \
         patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI", return_value=MagicMock()), \
         patch("tools.memory_tool.MemoryStore") as MockStore:
        MockStore.return_value.load_from_disk = MagicMock()

        agent = AIAgent(
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_key="test-key",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            skip_user_profile=False,  # explicit False — should NOT force user_profile on
        )
        assert agent._memory_enabled is True
        assert agent._user_profile_enabled is False
        # MemoryStore still constructed because _memory_enabled is True.
        assert agent._memory_store is not None
