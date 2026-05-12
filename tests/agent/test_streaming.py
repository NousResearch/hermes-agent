"""Tests for StreamingMixin -- streaming API call methods extracted from AIAgent.

The mixin is designed to be composed into AIAgent, so functional tests
use a FakeAgent that provides the minimum required attributes.
"""

import pytest


class TestStreamingMixinImport:
    """Verify all 15 streaming methods are present on the mixin."""

    def test_mixin_importable(self):
        from agent.streaming import StreamingMixin
        assert StreamingMixin is not None

    def test_mixin_has_interruptible_streaming(self):
        from agent.streaming import StreamingMixin
        assert hasattr(StreamingMixin, '_interruptible_streaming_api_call')

    def test_mixin_has_reset_stream_tracking(self):
        from agent.streaming import StreamingMixin
        assert hasattr(StreamingMixin, '_reset_stream_delivery_tracking')

    def test_mixin_has_fire_delta(self):
        from agent.streaming import StreamingMixin
        assert hasattr(StreamingMixin, '_fire_stream_delta')

    def test_mixin_has_stream_consumers(self):
        from agent.streaming import StreamingMixin
        assert hasattr(StreamingMixin, '_has_stream_consumers')

    def test_mixin_has_record_streamed_text(self):
        from agent.streaming import StreamingMixin
        assert hasattr(StreamingMixin, '_record_streamed_assistant_text')

    def test_mixin_has_interim_content_check(self):
        from agent.streaming import StreamingMixin
        assert hasattr(StreamingMixin, '_interim_content_was_streamed')

    def test_mixin_has_diag_init(self):
        from agent.streaming import StreamingMixin
        assert hasattr(StreamingMixin, '_stream_diag_init')

    def test_mixin_has_diag_capture(self):
        from agent.streaming import StreamingMixin
        assert hasattr(StreamingMixin, '_stream_diag_capture_response')

    def test_mixin_has_log_stream_retry(self):
        from agent.streaming import StreamingMixin
        assert hasattr(StreamingMixin, '_log_stream_retry')

    def test_mixin_has_emit_stream_drop(self):
        from agent.streaming import StreamingMixin
        assert hasattr(StreamingMixin, '_emit_stream_drop')

    def test_mixin_has_emit_auxiliary_failure(self):
        from agent.streaming import StreamingMixin
        assert hasattr(StreamingMixin, '_emit_auxiliary_failure')

    def test_mixin_has_compute_stale_timeout(self):
        from agent.streaming import StreamingMixin
        assert hasattr(StreamingMixin, '_compute_non_stream_stale_timeout')

    def test_mixin_has_run_codex_stream(self):
        from agent.streaming import StreamingMixin
        assert hasattr(StreamingMixin, '_run_codex_stream')

    def test_mixin_has_run_codex_fallback(self):
        from agent.streaming import StreamingMixin
        assert hasattr(StreamingMixin, '_run_codex_create_stream_fallback')


class TestStreamingMixinFunctional:
    """Functional tests for self-contained streaming methods."""

    def _make_agent(self, **kwargs):
        from agent.streaming import StreamingMixin

        class FakeAgent(StreamingMixin):
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)
            def _emit_warning(self, msg):
                pass
            def _summarize_api_error(self, exc):
                return str(exc)
            def _strip_think_blocks(self, text):
                return text

        return FakeAgent(**kwargs)

    def test_reset_stream_tracking(self):
        agent = self._make_agent(
            _current_streamed_assistant_text="old",
            stream_delta_callback=None,
            _stream_callback=None,
        )
        agent._reset_stream_delivery_tracking()
        assert agent._current_streamed_assistant_text == ""

    def test_has_stream_consumers_false(self):
        agent = self._make_agent(stream_delta_callback=None, _stream_callback=None)
        assert not agent._has_stream_consumers()

    def test_has_stream_consumers_true_with_callback(self):
        agent = self._make_agent(
            stream_delta_callback=lambda *a: None,
            _stream_callback=None,
        )
        assert agent._has_stream_consumers()

    def test_record_streamed_text_accumulates(self):
        agent = self._make_agent()
        agent._record_streamed_assistant_text("hello ")
        agent._record_streamed_assistant_text("world")
        assert agent._current_streamed_assistant_text == "hello world"

    def test_diag_init_returns_dict(self):
        from agent.streaming import StreamingMixin
        diag = StreamingMixin._stream_diag_init()
        assert isinstance(diag, dict)
        assert "started_at" in diag

    def test_emit_auxiliary_failure_does_not_crash(self):
        agent = self._make_agent(log_prefix="[test] ", platform="cli")
        agent._emit_auxiliary_failure("test task", RuntimeError("boom"))
        # Should not raise

    def test_fire_stream_delta_no_callback(self):
        """_fire_stream_delta with no consumers should not crash."""
        agent = self._make_agent(stream_delta_callback=None, _stream_callback=None)
        agent._fire_stream_delta("hello")
        # Should not raise
