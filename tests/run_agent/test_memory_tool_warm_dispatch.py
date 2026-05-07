"""Regression test: AIAgent._invoke_tool must forward warm-tier args to memory_tool.

History: an early version of the warm-tier wiring left the per-agent memory
dispatch path (run_agent.py, the ``elif function_name == "memory":`` branch
inside ``_invoke_tool`` / the sequential executor) hardcoded to forward only
hot-tier kwargs (action, target, content, old_text). Warm-tier args (query,
top_k, category, tags, fact_id, helpful, tier) were silently dropped, and
``memory(action="recall", query="...")`` from a real agent always returned
``{"error": "query is required for recall.", "success": false}`` even though
the same call worked when made via direct python or via tools.registry.dispatch().

These tests guard against that regression by patching tools.memory_tool.memory_tool
and asserting that EVERY warm-tier kwarg the agent sees in function_args
arrives at the underlying tool function.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


WARM_KWARGS = {
    "tier": "warm",
    "query": "tanium developer API MCP",
    "top_k": 7,
    "category": "tanium",
    "tags": "tds,mcp",
    "fact_id": 42,
    "helpful": True,
}


def _make_agent():
    """Create a minimal AIAgent with memory off (we don't want hot-tier load
    to interfere) and skip_memory=True so _memory_store is None — the bypass
    still runs."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_id="test-session-memwarm",
            skip_context_files=True,
            skip_memory=True,
        )
    return agent


# ---------------------------------------------------------------------------
# The minimum-viable regression: recall reaches memory_tool with query set
# ---------------------------------------------------------------------------

class TestMemoryRecallForwardsQuery:
    def test_invoke_tool_recall_forwards_query(self):
        """_invoke_tool must pass `query` through to memory_tool for recall."""
        agent = _make_agent()

        captured: Dict[str, Any] = {}

        def _stub_memory_tool(**kwargs) -> str:
            captured.update(kwargs)
            return '{"success": true, "results": [], "count": 0}'

        with patch("tools.memory_tool.memory_tool", side_effect=_stub_memory_tool):
            result = agent._invoke_tool(
                function_name="memory",
                function_args={
                    "action": "recall",
                    "query": "tanium developer API MCP",
                    "top_k": 5,
                },
                effective_task_id="t1",
            )

        # The exact failure mode this test guards against: result containing
        # the "query is required for recall" message would mean the args
        # didn't reach the warm dispatcher.
        assert "query is required" not in result, result
        # Hard assertion: memory_tool actually received the query
        assert captured.get("query") == "tanium developer API MCP", captured
        assert captured.get("action") == "recall", captured
        assert captured.get("top_k") == 5, captured

    def test_invoke_tool_forwards_all_warm_kwargs(self):
        """Every warm-tier kwarg in function_args must reach memory_tool."""
        agent = _make_agent()

        captured: Dict[str, Any] = {}

        def _stub_memory_tool(**kwargs) -> str:
            captured.update(kwargs)
            return '{"success": true}'

        function_args = {"action": "recall", **WARM_KWARGS}

        with patch("tools.memory_tool.memory_tool", side_effect=_stub_memory_tool):
            agent._invoke_tool(
                function_name="memory",
                function_args=function_args,
                effective_task_id="t1",
            )

        for key, expected in WARM_KWARGS.items():
            assert captured.get(key) == expected, (
                f"warm kwarg {key!r} not forwarded — "
                f"expected {expected!r}, got {captured.get(key)!r}"
            )

    def test_invoke_tool_hot_path_still_works(self):
        """Hot-tier `add` path must still reach memory_tool with target/content."""
        agent = _make_agent()

        captured: Dict[str, Any] = {}

        def _stub_memory_tool(**kwargs) -> str:
            captured.update(kwargs)
            return '{"success": true}'

        with patch("tools.memory_tool.memory_tool", side_effect=_stub_memory_tool):
            agent._invoke_tool(
                function_name="memory",
                function_args={
                    "action": "add",
                    "target": "user",
                    "content": "User prefers concise responses.",
                },
                effective_task_id="t1",
            )

        assert captured.get("action") == "add"
        assert captured.get("target") == "user"
        assert captured.get("content") == "User prefers concise responses."
        # Tier defaults to "hot" when not set
        assert captured.get("tier") == "hot"


# ---------------------------------------------------------------------------
# End-to-end: recall actually returns query results from a real warm store
# ---------------------------------------------------------------------------

class TestMemoryRecallEndToEnd:
    def test_recall_returns_results_from_real_warm_store(
        self, tmp_path, monkeypatch,
    ):
        """A full _invoke_tool('memory', recall) round-trip should return rows
        from a real WarmStore — not the 'query is required' error."""
        # Isolate HERMES_HOME so we don't touch the user's real warm DB
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_constants
        if hasattr(hermes_constants, "_HERMES_HOME_CACHE"):
            hermes_constants._HERMES_HOME_CACHE = None

        from tools.memory_warm import get_warm_store, reset_warm_store_for_testing
        reset_warm_store_for_testing()

        warm = get_warm_store(db_path=tmp_path / "warm.db")
        warm.add(
            content="The Tanium developer API MCP runs at git.corp.tanium.com.",
            category="tanium",
            tags="mcp,api",
        )

        agent = _make_agent()

        try:
            result_str = agent._invoke_tool(
                function_name="memory",
                function_args={
                    "action": "recall",
                    "query": "Tanium developer API MCP",
                    "top_k": 5,
                },
                effective_task_id="t1",
            )
        finally:
            reset_warm_store_for_testing()
            if hasattr(hermes_constants, "_HERMES_HOME_CACHE"):
                hermes_constants._HERMES_HOME_CACHE = None

        import json
        result = json.loads(result_str)
        assert result.get("success") is True, result
        assert "query is required" not in (result.get("error") or ""), result
        assert result.get("count", 0) >= 1, result
