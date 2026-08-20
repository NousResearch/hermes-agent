"""Tests for ``memory.suppress_builtin_when_external`` in system prompt assembly.

Regression coverage for #28796: when an external memory provider returns a
non-empty system-prompt block AND the user opted in via
``memory.suppress_builtin_when_external``, the built-in MEMORY.md / USER.md
blocks must not be injected — they'd duplicate the same content the external
provider already covers.

The default (opt-out absent or false) preserves the historical additive
behavior, so existing deployments keep both blocks.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import system_prompt as sp


def _make_agent(*, builtin: str, external: str, suppress: bool):
    """Build a stub AIAgent exposing only the fields ``build_system_prompt_parts`` reads."""
    mem_store = MagicMock()
    mem_store.format_for_system_prompt.side_effect = lambda kind: {
        "memory": "MEMORY:" + builtin if builtin else "",
        "user": "USER:" + builtin if builtin else "",
    }[kind]

    mem_mgr = MagicMock()
    mem_mgr.build_system_prompt.return_value = external

    return SimpleNamespace(
        # Identity / context
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=set(),
        provider="openrouter",
        model="test/model",
        platform="cli",
        # Tool-use enforcement (the function reads this — set to "false" so we
        # don't take any extra branches)
        _tool_use_enforcement="false",
        # Memory state
        _memory_store=mem_store,
        _memory_enabled=True,
        _user_profile_enabled=True,
        _memory_manager=mem_mgr,
        _memory_suppress_builtin_when_external=suppress,
        # Session/identity (consumed by the timestamp line)
        pass_session_id=False,
        session_id="sess-test",
        # Kanban worker guidance (None → branch skipped)
        _kanban_worker_guidance=None,
    )


@pytest.fixture
def stub_ra():
    """Stub the lazy ``_ra()`` indirection so we don't pull run_agent helpers."""
    stub = MagicMock()
    stub.load_soul_md.return_value = ""
    stub.build_environment_hints.return_value = ""
    stub.build_context_files_prompt.return_value = ""
    stub.build_nous_subscription_prompt.return_value = ""
    stub.build_skills_system_prompt.return_value = ""
    stub.get_toolset_for_tool.return_value = None
    with patch.object(sp, "_ra", return_value=stub):
        yield stub


class TestSuppressBuiltinWhenExternal:
    def test_default_keeps_both_blocks(self, stub_ra):
        """No flag set → built-in memory + external block both injected (back-compat)."""
        agent = _make_agent(builtin="abc", external="EXTERNAL_CTX", suppress=False)
        parts = sp.build_system_prompt_parts(agent)
        vol = parts["volatile"]
        assert "MEMORY:abc" in vol
        assert "USER:abc" in vol
        assert "EXTERNAL_CTX" in vol

    def test_suppression_drops_builtin_when_external_has_content(self, stub_ra):
        """Flag on + external provider has content → built-in blocks suppressed."""
        agent = _make_agent(builtin="abc", external="EXTERNAL_CTX", suppress=True)
        parts = sp.build_system_prompt_parts(agent)
        vol = parts["volatile"]
        assert "MEMORY:abc" not in vol
        assert "USER:abc" not in vol
        assert "EXTERNAL_CTX" in vol

    def test_suppression_inactive_when_external_empty(self, stub_ra):
        """Flag on but external returns empty → built-in still injected.

        Otherwise a misconfigured external provider would silently strip the
        user's MEMORY.md / USER.md from every system prompt.
        """
        agent = _make_agent(builtin="abc", external="", suppress=True)
        parts = sp.build_system_prompt_parts(agent)
        vol = parts["volatile"]
        assert "MEMORY:abc" in vol
        assert "USER:abc" in vol

    def test_suppression_safe_when_no_external_provider(self, stub_ra):
        """No external provider at all → flag is harmless."""
        agent = _make_agent(builtin="abc", external="", suppress=True)
        agent._memory_manager = None
        parts = sp.build_system_prompt_parts(agent)
        vol = parts["volatile"]
        assert "MEMORY:abc" in vol
        assert "USER:abc" in vol

    def test_external_provider_exception_does_not_break_builtin(self, stub_ra):
        """If the external provider's build_system_prompt raises, fall back cleanly.

        The suppression flag must not trigger (because _ext_mem_block ends
        up empty) and the built-in block must still be injected.
        """
        agent = _make_agent(builtin="abc", external="", suppress=True)
        agent._memory_manager.build_system_prompt.side_effect = RuntimeError("boom")
        parts = sp.build_system_prompt_parts(agent)
        vol = parts["volatile"]
        assert "MEMORY:abc" in vol
        assert "USER:abc" in vol
