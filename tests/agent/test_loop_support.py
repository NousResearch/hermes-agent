"""Tests for LoopSupportMixin -- iteration/display methods extracted from AIAgent."""

import pytest


class TestLoopSupportMixinImport:
    def test_mixin_importable(self):
        from agent.loop_support import LoopSupportMixin
        assert LoopSupportMixin is not None

    def test_has_safe_print(self):
        from agent.loop_support import LoopSupportMixin
        assert hasattr(LoopSupportMixin, '_safe_print')

    def test_has_vprint(self):
        from agent.loop_support import LoopSupportMixin
        assert hasattr(LoopSupportMixin, '_vprint')

    def test_has_should_start_quiet_spinner(self):
        from agent.loop_support import LoopSupportMixin
        assert hasattr(LoopSupportMixin, '_should_start_quiet_spinner')

    def test_has_should_emit_quiet_tool_messages(self):
        from agent.loop_support import LoopSupportMixin
        assert hasattr(LoopSupportMixin, '_should_emit_quiet_tool_messages')

    def test_has_emit_status(self):
        from agent.loop_support import LoopSupportMixin
        assert hasattr(LoopSupportMixin, '_emit_status')

    def test_has_emit_warning(self):
        from agent.loop_support import LoopSupportMixin
        assert hasattr(LoopSupportMixin, '_emit_warning')

    def test_has_handle_max_iterations(self):
        from agent.loop_support import LoopSupportMixin
        assert hasattr(LoopSupportMixin, '_handle_max_iterations')

    def test_has_interruptible_api_call(self):
        from agent.loop_support import LoopSupportMixin
        assert hasattr(LoopSupportMixin, '_interruptible_api_call')

    def test_has_wrap_verbose(self):
        from agent.loop_support import LoopSupportMixin
        assert hasattr(LoopSupportMixin, '_wrap_verbose')

    def test_has_touch_activity(self):
        from agent.loop_support import LoopSupportMixin
        assert hasattr(LoopSupportMixin, '_touch_activity')
