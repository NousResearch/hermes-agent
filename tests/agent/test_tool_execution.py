"""Tests for ToolExecutionMixin -- tool dispatch methods extracted from AIAgent."""

import pytest


class TestToolExecutionMixinImport:
    def test_mixin_importable(self):
        from agent.tool_execution import ToolExecutionMixin
        assert ToolExecutionMixin is not None

    def test_has_execute_sequential(self):
        from agent.tool_execution import ToolExecutionMixin
        assert hasattr(ToolExecutionMixin, '_execute_tool_calls_sequential')

    def test_has_execute_concurrent(self):
        from agent.tool_execution import ToolExecutionMixin
        assert hasattr(ToolExecutionMixin, '_execute_tool_calls_concurrent')

    def test_has_invoke_tool(self):
        from agent.tool_execution import ToolExecutionMixin
        assert hasattr(ToolExecutionMixin, '_invoke_tool')

    def test_has_apply_steer(self):
        from agent.tool_execution import ToolExecutionMixin
        assert hasattr(ToolExecutionMixin, '_apply_pending_steer_to_tool_results')
