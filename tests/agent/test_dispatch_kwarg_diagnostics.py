"""Tests for the unexpected-kwarg TypeError dispatch diagnostic (#60821).

When the provider SDK rejects a kwarg at dispatch, the failure handler
renders the api_kwargs keys plus the llm_request middleware trace so the
injector (a plugin replacing the request dict, or stale kwargs crossing an
api_mode switch) is identifiable from one log line.
"""

from agent.conversation_loop import _format_unexpected_kwarg_diagnostics


class TestFormatUnexpectedKwargDiagnostics:
    def test_renders_sorted_kwargs_keys(self):
        rendered = _format_unexpected_kwarg_diagnostics(
            {"system": "x", "messages": [], "model": "m"}, []
        )
        assert "api_kwargs keys: [messages, model, system]" in rendered

    def test_renders_middleware_trace_entries(self):
        trace = [
            {"name": "skill-enforce", "source": "plugin"},
            {"source": "plugin"},
            "raw-entry",
        ]
        rendered = _format_unexpected_kwarg_diagnostics({"model": "m"}, trace)
        assert "skill-enforce" in rendered
        assert "plugin" in rendered
        assert "raw-entry" in rendered

    def test_no_trace_reads_as_none_applied(self):
        rendered = _format_unexpected_kwarg_diagnostics({"model": "m"}, [])
        assert "none applied" in rendered

    def test_unavailable_kwargs_do_not_crash(self):
        rendered = _format_unexpected_kwarg_diagnostics(None, None)
        assert "<unavailable>" in rendered
        assert "none applied" in rendered

    def test_the_60821_shape_is_identifiable(self):
        """The exact #60821 payload: Anthropic-shape 'system' on
        chat_completions kwargs, injected by a plugin middleware."""
        rendered = _format_unexpected_kwarg_diagnostics(
            {"model": "tencent/Hy3", "messages": [], "system": "reminder",
             "stream": True, "tools": []},
            [{"name": "skill-enforce"}],
        )
        assert "system" in rendered
        assert "skill-enforce" in rendered
