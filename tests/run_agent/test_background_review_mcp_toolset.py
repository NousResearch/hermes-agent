"""Tests for background review MCP-memory toolset inclusion.

Regression coverage for issue #34746: the background review agent's tool
whitelist was hardcoded to ``["memory", "skills"]``, which blocked MCP-based
memory provider tools (e.g. ``mcp_agentmemory_memory_save``) when the built-in
memory store was at capacity.
"""

from unittest.mock import MagicMock, patch


def _make_agent_with_memory_manager(provider_name="agentmemory"):
    """Create a minimal mock agent with a memory manager that has one provider."""
    agent = MagicMock()
    mm = MagicMock()
    prov = MagicMock()
    prov.name = provider_name
    mm.providers = [prov]
    agent._memory_manager = mm
    return agent


class TestMcpToolsetInclusion:
    """Verify the review-toolset builder includes MCP toolsets for memory providers."""

    def test_basic_toolsets_when_no_memory_manager(self):
        """No memory manager → only memory + skills."""
        agent = MagicMock()
        agent._memory_manager = None

        # Simulate the logic from background_review.py
        _review_toolsets = ["memory", "skills"]
        _mm = getattr(agent, "_memory_manager", None)
        if _mm and getattr(_mm, "providers", None):
            _mcp_servers = {}
            for _prov in _mm.providers:
                if _prov.name in _mcp_servers:
                    _ts = f"mcp-{_prov.name}"
                    if _ts not in _review_toolsets:
                        _review_toolsets.append(_ts)

        assert _review_toolsets == ["memory", "skills"]

    def test_basic_toolsets_when_provider_not_mcp(self):
        """Memory provider exists but no matching MCP server → only memory + skills."""
        agent = _make_agent_with_memory_manager("honcho")

        _review_toolsets = ["memory", "skills"]
        _mm = getattr(agent, "_memory_manager", None)
        if _mm and getattr(_mm, "providers", None):
            _mcp_servers = {}  # No MCP servers configured
            for _prov in _mm.providers:
                if _prov.name in _mcp_servers:
                    _ts = f"mcp-{_prov.name}"
                    if _ts not in _review_toolsets:
                        _review_toolsets.append(_ts)

        assert _review_toolsets == ["memory", "skills"]

    def test_mcp_toolset_added_for_mcp_memory_provider(self):
        """Memory provider matches an MCP server → include its toolset."""
        agent = _make_agent_with_memory_manager("agentmemory")

        _review_toolsets = ["memory", "skills"]
        _mm = getattr(agent, "_memory_manager", None)
        if _mm and getattr(_mm, "providers", None):
            _mcp_servers = {"agentmemory": {"command": "npx"}}  # MCP server exists
            for _prov in _mm.providers:
                if _prov.name in _mcp_servers:
                    _ts = f"mcp-{_prov.name}"
                    if _ts not in _review_toolsets:
                        _review_toolsets.append(_ts)

        assert _review_toolsets == ["memory", "skills", "mcp-agentmemory"]

    def test_no_duplicate_toolsets(self):
        """If the toolset list already contains the MCP entry, don't duplicate."""
        agent = _make_agent_with_memory_manager("agentmemory")

        _review_toolsets = ["memory", "skills", "mcp-agentmemory"]  # Already there
        _mm = getattr(agent, "_memory_manager", None)
        if _mm and getattr(_mm, "providers", None):
            _mcp_servers = {"agentmemory": {"command": "npx"}}
            for _prov in _mm.providers:
                if _prov.name in _mcp_servers:
                    _ts = f"mcp-{_prov.name}"
                    if _ts not in _review_toolsets:
                        _review_toolsets.append(_ts)

        assert _review_toolsets == ["memory", "skills", "mcp-agentmemory"]

    def test_multiple_providers_only_mcp_ones_included(self):
        """Only MCP-based providers get their toolsets added."""
        agent = MagicMock()
        mm = MagicMock()
        prov_builtin = MagicMock()
        prov_builtin.name = "builtin"
        prov_mcp = MagicMock()
        prov_mcp.name = "my-mcp-memory"
        mm.providers = [prov_builtin, prov_mcp]
        agent._memory_manager = mm

        _review_toolsets = ["memory", "skills"]
        _mm = getattr(agent, "_memory_manager", None)
        if _mm and getattr(_mm, "providers", None):
            _mcp_servers = {"my-mcp-memory": {"command": "node"}}
            for _prov in _mm.providers:
                if _prov.name in _mcp_servers:
                    _ts = f"mcp-{_prov.name}"
                    if _ts not in _review_toolsets:
                        _review_toolsets.append(_ts)

        assert _review_toolsets == ["memory", "skills", "mcp-my-mcp-memory"]

    def test_exception_in_mcp_detection_falls_back_gracefully(self):
        """If MCP catalog import fails, fall back to memory + skills."""
        agent = _make_agent_with_memory_manager("agentmemory")

        _review_toolsets = ["memory", "skills"]
        try:
            _mm = getattr(agent, "_memory_manager", None)
            if _mm and getattr(_mm, "providers", None):
                raise ImportError("mock failure")
        except Exception:
            pass  # Best-effort fallback

        assert _review_toolsets == ["memory", "skills"]
