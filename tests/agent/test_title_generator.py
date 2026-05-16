"""Tests for agent.title_generator — auto-generated session titles."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from agent.title_generator import (
    generate_title,
    auto_title_session,
    maybe_auto_title,
)


class TestGenerateTitle:
    """Unit tests for generate_title()."""

    def test_returns_title_on_success(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Debugging Python Import Errors"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title(["help me fix this import"])
            assert title == "Debugging Python Import Errors"

    def test_strips_quotes(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '"Setting Up Docker Environment"'

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title(["how do I set up docker"])
            assert title == "Setting Up Docker Environment"

    def test_strips_title_prefix(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Title: Kubernetes Pod Debugging"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title(["my pod keeps crashing"])
            assert title == "Kubernetes Pod Debugging"

    def test_truncates_long_titles(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A" * 100

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title(["question"])
            assert len(title) == 60
            assert title.endswith("...")

    def test_returns_none_on_empty_response(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert generate_title(["question"]) is None

    def test_returns_none_on_exception(self):
        with patch("agent.title_generator.call_llm", side_effect=RuntimeError("no provider")):
            assert generate_title(["question"]) is None

    def test_invokes_failure_callback_on_exception(self):
        """failure_callback must fire so the user sees a warning (issue #15775)."""
        captured = []

        def _cb(task, exc):
            captured.append((task, exc))

        exc = RuntimeError("openrouter 402: credits exhausted")
        with patch("agent.title_generator.call_llm", side_effect=exc):
            result = generate_title(["question"], failure_callback=_cb)

        assert result is None
        assert len(captured) == 1
        assert captured[0][0] == "title generation"
        assert captured[0][1] is exc

    def test_failure_callback_errors_are_swallowed(self):
        """A broken callback must not crash title generation."""

        def _bad_cb(task, exc):
            raise ValueError("callback bug")

        with patch("agent.title_generator.call_llm", side_effect=RuntimeError("nope")):
            # Should return None without re-raising the callback error
            assert generate_title(["q"], failure_callback=_bad_cb) is None

    def test_no_callback_matches_legacy_behavior(self):
        """Omitting failure_callback preserves the silent-None return."""
        with patch("agent.title_generator.call_llm", side_effect=RuntimeError("nope")):
            assert generate_title(["q"]) is None

    def test_uses_llm_client_directly(self):
        """When llm_client is provided, use it directly — no call_llm."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Direct Client Title"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent.title_generator.call_llm") as mock_call_llm:
            title = generate_title(["hello"], llm_client=mock_client)
            assert title == "Direct Client Title"
            mock_call_llm.assert_not_called()

    def test_llm_client_passes_model(self):
        """When llm_client and model are both provided, model is passed through."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Titled"
        mock_client.chat.completions.create.return_value = mock_response

        generate_title(["hi"], llm_client=mock_client, model="my-model")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "my-model"
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["temperature"] == 0.3

    def test_llm_client_falls_back_to_auxiliary(self):
        """When llm_client is None, falls back to call_llm."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Fallback Title"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title(["hello"], llm_client=None)
            assert title == "Fallback Title"

    def test_llm_client_error_returns_none(self):
        """When llm_client raises, return None gracefully."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("connection refused")

        title = generate_title(["hello"], llm_client=mock_client)
        assert title is None

    def test_truncates_long_messages(self):
        """Long user messages should be truncated in the LLM request."""
        captured_kwargs = {}

        def mock_call_llm(**kwargs):
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "Short Title"
            return resp

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm):
            generate_title(["x" * 1000])

        # The user content in the messages should be truncated
        user_content = captured_kwargs["messages"][1]["content"]
        assert len(user_content) < 650  # 500 + formatting

    def test_filters_noise_messages(self):
        """System notes and compaction summaries should be filtered out."""
        captured_kwargs = {}

        def mock_call_llm(**kwargs):
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "Real Topic"
            return resp

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm):
            generate_title([
                "[CONTEXT COMPACTION — REFERENCE ONLY] earlier turns...",
                "[System note: skill invoked]",
                "actual user message about topic",
            ])

        user_content = captured_kwargs["messages"][1]["content"]
        assert "CONTEXT COMPACTION" not in user_content
        assert "System note" not in user_content
        assert "actual user message" in user_content

    def test_returns_none_for_empty_input(self):
        assert generate_title([]) is None

    def test_uses_main_runtime_provider(self):
        """When main_runtime is set, its provider should be resolved."""
        captured_kwargs = {}

        def mock_call_llm(**kwargs):
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "Title"
            return resp

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm):
            generate_title(
                ["question"],
                main_runtime={"provider": "openrouter", "model": "test-model"},
            )

        assert captured_kwargs["provider"] == "openrouter"
        assert captured_kwargs["model"] == "test-model"

    def test_explicit_provider_overrides_main_runtime(self):
        """Explicit provider param should take priority over main_runtime."""
        captured_kwargs = {}

        def mock_call_llm(**kwargs):
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "Title"
            return resp

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm):
            generate_title(
                ["question"],
                provider="anthropic",
                model="claude-test",
                main_runtime={"provider": "openrouter", "model": "other"},
            )

        assert captured_kwargs["provider"] == "anthropic"
        assert captured_kwargs["model"] == "claude-test"


class TestAutoTitleSession:
    """Tests for auto_title_session() — the sync worker function."""

    def test_skips_if_no_session_db(self):
        auto_title_session(None, "sess-1", ["hi"], "hello")  # should not crash

    def test_skips_if_title_exists(self):
        db = MagicMock()
        db.get_session_title.return_value = "Existing Title"

        with patch("agent.title_generator.generate_title") as gen:
            auto_title_session(db, "sess-1", ["hi"], "hello")
            gen.assert_not_called()

    def test_generates_and_sets_title(self):
        db = MagicMock()
        db.get_session_title.return_value = None

        with patch("agent.title_generator.generate_title", return_value="New Title"):
            auto_title_session(db, "sess-1", ["hi"], "hello")
            db.set_session_title.assert_called_once_with("sess-1", "New Title")

    def test_invokes_title_callback_after_setting_title(self):
        db = MagicMock()
        db.get_session_title.return_value = None
        seen = []
        with patch("agent.title_generator.generate_title", return_value="Readable Session"):
            auto_title_session(
                db,
                "sess-1",
                ["hello"],
                "hi there",
                title_callback=seen.append,
            )
        db.set_session_title.assert_called_once_with("sess-1", "Readable Session")
        assert seen == ["Readable Session"]

    def test_skips_if_generation_fails(self):
        db = MagicMock()
        db.get_session_title.return_value = None

        with patch("agent.title_generator.generate_title", return_value=None):
            auto_title_session(db, "sess-1", ["hi"], "hello")
            db.set_session_title.assert_not_called()


class TestMaybeAutoTitle:
    """Tests for maybe_auto_title() — the fire-and-forget entry point."""

    def test_skips_if_not_first_exchange(self):
        """Should not fire for conversations with more than 3 user messages."""
        db = MagicMock()
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response 1"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "response 2"},
            {"role": "user", "content": "third"},
            {"role": "assistant", "content": "response 3"},
            {"role": "user", "content": "fourth"},
            {"role": "assistant", "content": "response 4"},
        ]

        with patch("agent.title_generator.auto_title_session") as mock_auto:
            maybe_auto_title(db, "sess-1", "fourth", "response 4", history)
            import time
            time.sleep(0.1)
            mock_auto.assert_not_called()

    def test_fires_on_first_exchange(self):
        """Should fire a background thread for the first exchange."""
        db = MagicMock()
        db.get_session_title.return_value = None
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        with patch("agent.title_generator.auto_title_session") as mock_auto:
            maybe_auto_title(db, "sess-1", "hello", "hi there", history)
            import time
            time.sleep(0.3)
            mock_auto.assert_called_once()
            # Verify the user_messages list includes the current message
            call_args = mock_auto.call_args
            assert "hello" in call_args[0][2]  # user_messages is 3rd positional arg

    def test_forwards_failure_callback_to_worker(self):
        """maybe_auto_title must forward failure_callback into the thread."""
        db = MagicMock()
        db.get_session_title.return_value = None
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        def _cb(task, exc):
            pass

        with patch("agent.title_generator.auto_title_session") as mock_auto:
            maybe_auto_title(db, "sess-1", "hello", "hi there", history, failure_callback=_cb)
            import time
            time.sleep(0.3)
            mock_auto.assert_called_once()
            assert mock_auto.call_args[1]["failure_callback"] is _cb

    def test_skips_if_no_response(self):
        db = MagicMock()
        maybe_auto_title(db, "sess-1", "hello", "", [])  # empty response

    def test_skips_if_no_session_db(self):
        maybe_auto_title(None, "sess-1", "hello", "response", [])  # no db

    def test_fires_on_second_exchange(self):
        """Should still fire for the second exchange (user_msg_count == 2)."""
        db = MagicMock()
        db.get_session_title.return_value = None
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response 1"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "response 2"},
        ]

        with patch("agent.title_generator.auto_title_session") as mock_auto:
            maybe_auto_title(db, "sess-1", "second", "response 2", history)
            import time
            time.sleep(0.3)
            mock_auto.assert_called_once()

    def test_collects_user_messages_from_history(self):
        """Should collect all user messages from history for context."""
        db = MagicMock()
        db.get_session_title.return_value = None
        history = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "response 1"},
            {"role": "user", "content": "follow up"},
            {"role": "assistant", "content": "response 2"},
        ]

        with patch("agent.title_generator.auto_title_session") as mock_auto:
            maybe_auto_title(db, "sess-1", "follow up", "response 2", history)
            import time
            time.sleep(0.3)
            call_args = mock_auto.call_args
            user_msgs = call_args[0][2]
            assert "first question" in user_msgs
            assert "follow up" in user_msgs
