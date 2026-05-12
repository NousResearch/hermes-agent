"""Tests for SteerMixin -- steering/event methods extracted from AIAgent."""

import pytest


class TestSteerMixinImport:
    def test_mixin_importable(self):
        from agent.steer import SteerMixin
        assert SteerMixin is not None

    def test_has_steer(self):
        from agent.steer import SteerMixin
        assert hasattr(SteerMixin, 'steer')

    def test_has_interrupt(self):
        from agent.steer import SteerMixin
        assert hasattr(SteerMixin, 'interrupt')

    def test_has_clear_interrupt(self):
        from agent.steer import SteerMixin
        assert hasattr(SteerMixin, 'clear_interrupt')

    def test_has_is_interrupted(self):
        from agent.steer import SteerMixin
        assert hasattr(SteerMixin, 'is_interrupted')

    def test_has_drain_pending_steer(self):
        from agent.steer import SteerMixin
        assert hasattr(SteerMixin, '_drain_pending_steer')

    def test_has_fire_tool_gen_started(self):
        from agent.steer import SteerMixin
        assert hasattr(SteerMixin, '_fire_tool_gen_started')

    def test_has_fire_reasoning_delta(self):
        from agent.steer import SteerMixin
        assert hasattr(SteerMixin, '_fire_reasoning_delta')

    def test_has_emit_interim_assistant_message(self):
        from agent.steer import SteerMixin
        assert hasattr(SteerMixin, '_emit_interim_assistant_message')

    def test_has_normalize_interim_visible_text(self):
        from agent.steer import SteerMixin
        assert hasattr(SteerMixin, '_normalize_interim_visible_text')

    def test_has_capture_rate_limits(self):
        from agent.steer import SteerMixin
        assert hasattr(SteerMixin, '_capture_rate_limits')

    def test_has_usage_summary_for_api_request_hook(self):
        from agent.steer import SteerMixin
        assert hasattr(SteerMixin, '_usage_summary_for_api_request_hook')
