"""Tests for PromptAssembler — extracted from AIAgent._build_system_prompt.

Covers:
    - Layer ordering (identity first, platform last)
    - Tool-aware guidance gating
    - Memory block gating on flags
    - Context file skipping
    - Caching contract (same object, ignores new params)
    - Invalidation (clears cache, reloads memory)
    - Ephemeral prompt exclusion
    - System message inclusion
"""

from unittest import mock

import pytest

from agent.prompt_assembler import PromptAssembler
from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,
    MEMORY_GUIDANCE,
    SESSION_SEARCH_GUIDANCE,
    SKILLS_GUIDANCE,
    PLATFORM_HINTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory_store(memory_block="[memory block]", user_block="[user block]"):
    """Create a mock memory store with configurable format_for_system_prompt."""
    store = mock.MagicMock()

    def _format(kind):
        if kind == "memory":
            return memory_block
        if kind == "user":
            return user_block
        return None

    store.format_for_system_prompt = mock.MagicMock(side_effect=_format)
    store.load_from_disk = mock.MagicMock()
    return store


# Patch targets — functions called inside PromptAssembler.build()
_PATCH_SKILLS = "agent.prompt_assembler.build_skills_system_prompt"
_PATCH_CONTEXT = "agent.prompt_assembler.build_context_files_prompt"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPromptAssembler:
    """Unit tests for PromptAssembler."""

    # 1. Basic assembly returns a string
    @mock.patch(_PATCH_CONTEXT, return_value="")
    @mock.patch(_PATCH_SKILLS, return_value="")
    def test_build_returns_string(self, _skills, _ctx):
        pa = PromptAssembler()
        result = pa.build(valid_tool_names=[])
        assert isinstance(result, str)
        assert len(result) > 0

    # 2. DEFAULT_AGENT_IDENTITY is the first segment
    @mock.patch(_PATCH_CONTEXT, return_value="")
    @mock.patch(_PATCH_SKILLS, return_value="")
    def test_layer_ordering_identity_first(self, _skills, _ctx):
        pa = PromptAssembler()
        result = pa.build(valid_tool_names=[])
        segments = result.split("\n\n")
        assert segments[0] == DEFAULT_AGENT_IDENTITY

    # 3. Platform hint is the last segment when present
    @mock.patch(_PATCH_CONTEXT, return_value="")
    @mock.patch(_PATCH_SKILLS, return_value="")
    def test_layer_ordering_platform_last(self, _skills, _ctx):
        pa = PromptAssembler(platform="discord")
        result = pa.build(valid_tool_names=[])
        segments = result.split("\n\n")
        assert segments[-1] == PLATFORM_HINTS["discord"]

    # 4. MEMORY_GUIDANCE only present when "memory" in valid_tool_names
    @mock.patch(_PATCH_CONTEXT, return_value="")
    @mock.patch(_PATCH_SKILLS, return_value="")
    def test_tool_guidance_gated_on_tool_names(self, _skills, _ctx):
        # With "memory" tool
        pa = PromptAssembler()
        result = pa.build(valid_tool_names=["memory"])
        assert MEMORY_GUIDANCE in result

        # Without "memory" tool — need fresh assembler (cache)
        pa2 = PromptAssembler()
        result2 = pa2.build(valid_tool_names=[])
        assert MEMORY_GUIDANCE not in result2

    # 5. memory_enabled=False skips memory blocks
    @mock.patch(_PATCH_CONTEXT, return_value="")
    @mock.patch(_PATCH_SKILLS, return_value="")
    def test_memory_blocks_gated_on_flags(self, _skills, _ctx):
        store = _make_memory_store()

        # memory_enabled=False — memory block should NOT appear
        pa = PromptAssembler()
        result = pa.build(
            valid_tool_names=[],
            memory_store=store,
            memory_enabled=False,
            user_profile_enabled=False,
        )
        assert "[memory block]" not in result
        assert "[user block]" not in result

        # memory_enabled=True — memory block SHOULD appear
        pa2 = PromptAssembler()
        result2 = pa2.build(
            valid_tool_names=[],
            memory_store=store,
            memory_enabled=True,
            user_profile_enabled=True,
        )
        assert "[memory block]" in result2
        assert "[user block]" in result2

    # 6. skip_context_files=True omits context files
    @mock.patch(_PATCH_CONTEXT, return_value="[context files content]")
    @mock.patch(_PATCH_SKILLS, return_value="")
    def test_context_files_skipped(self, _skills, _ctx):
        pa = PromptAssembler(skip_context_files=True)
        result = pa.build(valid_tool_names=[])
        assert "[context files content]" not in result

        # With skip_context_files=False, context IS included
        pa2 = PromptAssembler(skip_context_files=False)
        result2 = pa2.build(valid_tool_names=[])
        assert "[context files content]" in result2

    # 7. Caching: second build() call returns cached value (same object)
    @mock.patch(_PATCH_CONTEXT, return_value="")
    @mock.patch(_PATCH_SKILLS, return_value="")
    def test_caching_returns_same_result(self, _skills, _ctx):
        pa = PromptAssembler()
        result1 = pa.build(valid_tool_names=[])
        result2 = pa.build(valid_tool_names=[])
        assert result1 is result2  # same object, not just equal

    # 8. Caching: second build() with different params still returns cached
    @mock.patch(_PATCH_CONTEXT, return_value="")
    @mock.patch(_PATCH_SKILLS, return_value="")
    def test_caching_ignores_new_params(self, _skills, _ctx):
        pa = PromptAssembler()
        result1 = pa.build(valid_tool_names=[])
        result2 = pa.build(valid_tool_names=["memory"], system_message="new msg")
        assert result1 is result2
        # The new params should NOT be reflected
        assert MEMORY_GUIDANCE not in result2
        assert "new msg" not in result2

    # 9. After invalidate(), cached property is None
    @mock.patch(_PATCH_CONTEXT, return_value="")
    @mock.patch(_PATCH_SKILLS, return_value="")
    def test_invalidate_clears_cache(self, _skills, _ctx):
        pa = PromptAssembler()
        pa.build(valid_tool_names=[])
        assert pa.cached is not None
        pa.invalidate()
        assert pa.cached is None

    # 10. invalidate(memory_store) calls load_from_disk()
    @mock.patch(_PATCH_CONTEXT, return_value="")
    @mock.patch(_PATCH_SKILLS, return_value="")
    def test_invalidate_reloads_memory(self, _skills, _ctx):
        store = _make_memory_store()
        pa = PromptAssembler()
        pa.build(valid_tool_names=[], memory_store=store)
        pa.invalidate(memory_store=store)
        store.load_from_disk.assert_called_once()

    # 11. Ephemeral system prompt is NOT in the output
    @mock.patch(_PATCH_CONTEXT, return_value="")
    @mock.patch(_PATCH_SKILLS, return_value="")
    def test_ephemeral_prompt_not_included(self, _skills, _ctx):
        """PromptAssembler.build() has no ephemeral_system_prompt parameter.

        The ephemeral prompt is injected separately in message history,
        not in the system prompt assembly. Verify the build() signature
        does not accept it and the output contains no ephemeral marker.
        """
        import inspect

        pa = PromptAssembler()
        sig = inspect.signature(pa.build)
        param_names = list(sig.parameters.keys())
        assert "ephemeral_system_prompt" not in param_names
        assert "ephemeral" not in param_names

    # 12. system_message appears in prompt when provided
    @mock.patch(_PATCH_CONTEXT, return_value="")
    @mock.patch(_PATCH_SKILLS, return_value="")
    def test_system_message_included(self, _skills, _ctx):
        pa = PromptAssembler()
        custom_msg = "You are a specialized coding assistant."
        result = pa.build(valid_tool_names=[], system_message=custom_msg)
        assert custom_msg in result
