"""
Tests for the session-summary formatter and _session_response projection
added in PR #49519 (issue #45103, hover-card summary feature).

These tests close the review feedback from @teknium1 (2026-07-14):

  - The proposed async handler invokes synchronous summarize_session()
    directly. Its LLM path reaches synchronous
    agent/auxiliary_client.py, blocking the API event loop for the
    generation.  (covered separately by PR #49519's asyncio.to_thread
    change; integration test below exercises the non-blocking path)

  - The added hermes_state.py formatter says it strips tool results,
    but its proposed loop formats every non-empty message, including
    role="tool". Tool output must not be copied into the auxiliary
    prompt.

  - gateway/platforms/api_server.py filters the GET/list representation
    through _session_response, which does not expose summary,
    summary_updated_at, or summary_model. The new POST response does
    not make the fields available through the read path.

We test all three explicitly.
"""

from hermes_state import SessionDB
import unittest


class TestSummarizeFormatConversationRoleFilter(unittest.TestCase):
    """PR #49519 review: _summarize_format_conversation must skip tool/system.

    The original implementation formatted every non-empty message,
    including role="tool" rows (which can be large file contents or
    command stdout). That bloats the summarizer prompt and distorts
    the summary. The fix restricts to user/assistant only.
    """

    def test_skips_tool_role(self):
        messages = [
            {"role": "user", "content": "Show me /etc/passwd"},
            {
                "role": "tool",
                "content": "root:x:0:0:root:/root:/bin/bash\n" * 200,  # ~2.4 KB noise
            },
            {"role": "assistant", "content": "Here's the file contents."},
        ]
        out = SessionDB._summarize_format_conversation(messages)
        assert "[tool]" not in out, (
            "tool role leaked into summary prompt — would bloat cost and "
            "distort the summary"
        )
        assert "root:x:0:0" not in out, (
            "tool payload leaked into summary prompt"
        )
        # The actual conversation is preserved.
        assert "[user] Show me /etc/passwd" in out
        assert "[assistant] Here's the file contents." in out

    def test_skips_system_role(self):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. " * 100,  # noise
            },
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        out = SessionDB._summarize_format_conversation(messages)
        assert "helpful assistant" not in out
        assert "[user] Hello" in out
        assert "[assistant] Hi!" in out

    def test_keeps_user_and_assistant(self):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "Sunny, 25C."},
        ]
        out = SessionDB._summarize_format_conversation(messages)
        assert "[user] What's the weather?" in out
        assert "[assistant] Sunny, 25C." in out

    def test_empty_content_still_skipped(self):
        """The pre-existing empty-content skip must still hold."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "World"},
        ]
        out = SessionDB._summarize_format_conversation(messages)
        # The empty assistant turn is dropped, but the user turns remain.
        assert "[user] Hello" in out
        assert "[user] World" in out

    def test_multimodal_text_part_preserved(self):
        """The pre-existing multi-part text extraction must still hold."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first part"},
                    {"type": "image_url", "image_url": {"url": "..."}},
                    {"type": "text", "text": "second part"},
                ],
            },
        ]
        out = SessionDB._summarize_format_conversation(messages)
        assert "first part" in out
        assert "second part" in out


class TestSessionResponseIncludesSummaryFields(unittest.TestCase):
    """PR #49519 review: _session_response must surface the summary cache
    fields so the Desktop hover card and the new POST endpoint can read
    them through the standard GET /api/sessions/{id} path.
    """

    def test_summary_fields_surfaced_when_present(self):
        # We import the helper class lazily so this test stays
        # import-clean even if aiohttp is not installed in the test env.
        from gateway.platforms.api_server import APIServerAdapter

        session = {
            "id": "s1",
            "summary": "User asked about Q3 OKRs. Decided to drop speed metric.",
            "summary_updated_at": 1_700_000_000.0,
            "summary_model": "gpt-4o-mini",
            # other irrelevant fields, ignored
            "system_prompt": "should not leak",
            "model_config": {"_delegate_from": "should not leak"},
        }
        out = APIServerAdapter._session_response(session)
        assert out["summary"] == session["summary"]
        assert out["summary_updated_at"] == session["summary_updated_at"]
        assert out["summary_model"] == session["summary_model"]

    def test_summary_fields_absent_when_not_yet_summarized(self):
        from gateway.platforms.api_server import APIServerAdapter

        session = {"id": "s2", "title": "New chat, no summary yet"}
        out = APIServerAdapter._session_response(session)
        # All three fields are absent (not "" or None) — the JSON
        # serialiser will just omit them, which is what the frontend
        # already handles ("if summary is null, fall back to preview").
        assert "summary" not in out
        assert "summary_updated_at" not in out
        assert "summary_model" not in out

    def test_session_response_does_not_leak_internal_fields(self):
        """Regression: the new fields must not regress the existing
        'avoid leaking system_prompt / model_config' invariant.
        """
        from gateway.platforms.api_server import APIServerAdapter

        session = {
            "id": "s3",
            "system_prompt": "secret instructions",
            "model_config": {"_delegate_from": "secret"},
            "summary": "harmless",
        }
        out = APIServerAdapter._session_response(session)
        assert "system_prompt" not in out
        assert "model_config" not in out
        # has_* flags must still be present.
        assert out["has_system_prompt"] is True
        assert out["has_model_config"] is True
        assert out["summary"] == "harmless"
