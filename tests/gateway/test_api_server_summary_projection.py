"""The cached AI summary must survive the client-facing session projection.

The hermes-sweeper review of PR #49519 found that the POST /summarize
response returned the summary, but the read path did not: every GET and
list went through ``APIServerAdapter._session_response``, which filtered
the session row down to a fixed key list that did not include the
summary columns. A client that fetched a session therefore never saw
the summary the API had just written.

These tests pin the projection in both directions — the fields appear
when they are set, and they are simply absent before the first summary
has been generated.
"""

from __future__ import annotations

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


class TestSessionResponseProjectsSummaryFields:
    def test_summary_fields_are_surfaced_when_present(self, adapter):
        """A session that has been summarized exposes all three fields."""
        payload = adapter._session_response(
            {
                "id": "s1",
                "title": "hello",
                "summary": "The user asked about the sidebar.",
                "summary_updated_at": 1_700_000_000.0,
                "summary_model": "aux-model-x",
            }
        )

        assert payload["summary"] == "The user asked about the sidebar."
        assert payload["summary_updated_at"] == 1_700_000_000.0
        assert payload["summary_model"] == "aux-model-x"

    def test_summary_fields_are_absent_when_never_summarized(self, adapter):
        """Before the first summary the fields are omitted, not nulled.

        The projection copies only the keys present in the row, so an
        unsummarized session simply has no summary keys at all. A client
        can therefore tell "not summarized yet" from "summarized to
        nothing" — the latter is not a state the code can produce.
        """
        payload = adapter._session_response({"id": "s1", "title": "hello"})

        assert "summary" not in payload
        assert "summary_updated_at" not in payload
        assert "summary_model" not in payload

    def test_the_projection_never_drops_the_other_session_fields(self):
        """Adding the summary columns must not narrow the projection.

        The safe-key list is the whole client contract; a key dropped
        from it silently disappears from every list response.
        """
        safe_keys = (
            "id", "source", "user_id", "model", "title", "started_at", "ended_at",
            "end_reason", "message_count", "tool_call_count", "input_tokens",
            "output_tokens", "cache_read_tokens", "cache_write_tokens",
            "reasoning_tokens", "estimated_cost_usd", "actual_cost_usd",
            "api_call_count", "parent_session_id", "last_active", "preview",
            "_lineage_root_id", "pinned", "archived", "hidden",
        )
        # The projection is a comprehension over a literal tuple; assert on
        # the rendered source so a future edit to that tuple is visible
        # here rather than only in a diff.
        import inspect

        source = inspect.getsource(APIServerAdapter._session_response)
        for key in safe_keys:
            assert f'"{key}"' in source, f"{key} was dropped from the projection"

    def test_system_prompt_and_model_config_stay_redacted(self, adapter):
        """The summary columns must not become a side door for the
        system prompt or the model config snapshot."""
        payload = adapter._session_response(
            {
                "id": "s1",
                "summary": "a summary",
                "system_prompt": "you are a helpful assistant with secrets",
                "model_config": '{"temperature": 0.7}',
            }
        )

        assert "system_prompt" not in payload
        assert "model_config" not in payload
        assert payload["has_system_prompt"] is True
        assert payload["has_model_config"] is True
        assert payload["is_internal_child"] is False


class TestSummarizeRouteIsRegistered:
    def test_the_route_is_reachable(self):
        """POST /api/sessions/{session_id}/summarize must be routed.

        The handler was reviewed as unusable when the route was not
        registered, so the registration itself is worth pinning.
        """
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        routes = adapter._build_routes() if hasattr(adapter, "_build_routes") else None
        if routes is None:
            # Newer layouts build the route table inline; fall back to
            # scanning the source for the registration.
            import inspect

            source = inspect.getsource(type(adapter))
            assert "/api/sessions/{session_id}/summarize" in source
            return
        assert any(
            method == "POST" and path == "/api/sessions/{session_id}/summarize"
            for method, path, *_ in routes
        )
