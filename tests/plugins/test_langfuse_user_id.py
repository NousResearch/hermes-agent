"""Tests for the Langfuse plugin's end-user attribution (`user_id`), alongside the
existing `session_id` grouping. Mirrors test_langfuse_plugin.py's TestTurnTraceIsolation
fake client, but records the full trace_context so assertions can target `user_id`
specifically, not just `trace_id`.
"""
from __future__ import annotations

import importlib
import sys


class TestUserIdAttribution:
    def _fresh_plugin(self):
        sys.modules.pop("plugins.observability.langfuse", None)
        return importlib.import_module("plugins.observability.langfuse")

    @staticmethod
    def _fake_client(contexts):
        class _Span:
            def update(self, **kw):
                pass

            def end(self, **kw):
                pass

            def start_observation(self, **kw):
                return _Span()

        class _RootCM:
            def __enter__(self):
                return _Span()

            def __exit__(self, *exc):
                return False

        class _Client:
            def create_trace_id(self, seed=None):
                return f"trace::{seed}"

            def start_as_current_observation(self, **kw):
                contexts.append(kw.get("trace_context", {}))
                return _RootCM()

            def flush(self):
                pass

        return _Client()

    def test_user_id_is_set_on_trace_context_when_present(self, monkeypatch):
        mod = self._fresh_plugin()
        contexts: list = []
        monkeypatch.setattr(mod, "_get_langfuse", lambda: self._fake_client(contexts))
        monkeypatch.setattr(mod, "_end_observation", lambda *a, **k: None)
        mod._TRACE_STATE.clear()

        mod.on_pre_llm_request(
            task_id="t-uid", session_id="sess-uid", model="m", provider="p", api_mode="chat",
            api_call_count=1, request_messages=[{"role": "user", "content": "hi"}],
            turn_id="turn-uid", api_request_id="req-uid", user_id="alice@example.com",
        )

        assert len(contexts) == 1
        assert contexts[0].get("user_id") == "alice@example.com"
        assert contexts[0].get("session_id") == "sess-uid"

    def test_absent_user_id_is_omitted_from_trace_context(self, monkeypatch):
        """No user_id -> key absent entirely, not an empty string (mirrors session_id's
        own ``if session_id`` guard -- an empty-string Langfuse user_id is still a value)."""
        mod = self._fresh_plugin()
        contexts: list = []
        monkeypatch.setattr(mod, "_get_langfuse", lambda: self._fake_client(contexts))
        monkeypatch.setattr(mod, "_end_observation", lambda *a, **k: None)
        mod._TRACE_STATE.clear()

        mod.on_pre_llm_request(
            task_id="t-nouid", session_id="sess-nouid", model="m", provider="p", api_mode="chat",
            api_call_count=1, request_messages=[{"role": "user", "content": "hi"}],
            turn_id="turn-nouid", api_request_id="req-nouid",
        )

        assert len(contexts) == 1
        assert "user_id" not in contexts[0]
