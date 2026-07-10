"""Tests for agent.title_generator — auto-generated session titles."""

import pytest
from unittest.mock import MagicMock, patch


from agent.title_generator import (
    generate_title,
    regenerate_title,
    _condense_history,
    auto_title_session,
    maybe_auto_title,
    maybe_retitle_session,
    _title_language,
)
from hermes_state import SessionDB


class TestGenerateTitle:
    """Unit tests for generate_title()."""

    def test_returns_title_on_success(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Debugging Python Import Errors"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title("help me fix this import", "Sure, let me check...")
            assert title == "Debugging Python Import Errors"

    def test_default_prompt_matches_user_language(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Some Title"

        with patch("agent.title_generator.call_llm", return_value=mock_response) as llm:
            generate_title("質問です", "回答です")

        system_prompt = llm.call_args.kwargs["messages"][0]["content"]
        assert "same language the user is writing in" in system_prompt

    def test_configured_language_pins_prompt(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Some Title"

        with (
            patch("agent.title_generator.call_llm", return_value=mock_response) as llm,
            patch("agent.title_generator._title_language", return_value="Japanese"),
        ):
            generate_title("hello", "hi")

        system_prompt = llm.call_args.kwargs["messages"][0]["content"]
        assert "Write the title in Japanese" in system_prompt
        assert "same language the user" not in system_prompt

    def test_title_language_reads_config(self):
        cfg = {"auxiliary": {"title_generation": {"language": "  French "}}}

        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert _title_language() == "French"
        with patch("hermes_cli.config.load_config", return_value={}):
            assert _title_language() == ""
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("bad config")):
            assert _title_language() == ""

    def test_default_timeout_delegates_to_auxiliary_config(self):
        captured_kwargs = {}

        def mock_call_llm(**kwargs):
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "Configured Timeout"
            return resp

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm):
            assert generate_title("question", "answer") == "Configured Timeout"

        assert captured_kwargs["task"] == "title_generation"
        assert captured_kwargs["timeout"] is None

    def test_explicit_timeout_still_overrides_config(self):
        captured_kwargs = {}

        def mock_call_llm(**kwargs):
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "Explicit Timeout"
            return resp

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm):
            assert generate_title("question", "answer", timeout=123.0) == "Explicit Timeout"

        assert captured_kwargs["timeout"] == 123.0

    def test_strips_quotes(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '"Setting Up Docker Environment"'

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title("how do I set up docker", "First install...")
            assert title == "Setting Up Docker Environment"

    def test_strips_think_blocks(self):
        """Reasoning-model output wrapped in <think>...</think> must not
        leak into the session title."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "<think>The user wants a title. I'll summarize the topic "
            "concisely.</think>Debugging Python Import Errors"
        )

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title("help me fix this import", "Sure...")
            assert title == "Debugging Python Import Errors"
            assert "<think>" not in title
            assert "summarize" not in title

    def test_strips_unterminated_think_block(self):
        """An unterminated <think> block (no close tag) must still be
        stripped so the leaked reasoning doesn't become the title."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "<think>Let me reason about a good title for this session"
        )

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title("hello", "hi there")
            # Everything from the unterminated open tag onward is stripped,
            # leaving nothing → None.
            assert title is None

    def test_strips_title_prefix(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Title: Kubernetes Pod Debugging"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title("my pod keeps crashing", "Let me look...")
            assert title == "Kubernetes Pod Debugging"

    def test_truncates_long_titles(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A" * 100

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title("question", "answer")
            assert len(title) == 80
            assert title.endswith("...")

    def test_returns_none_on_empty_response(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert generate_title("question", "answer") is None

    def test_returns_none_on_exception(self):
        with patch("agent.title_generator.call_llm", side_effect=RuntimeError("no provider")):
            assert generate_title("question", "answer") is None

    def test_invokes_failure_callback_on_exception(self):
        """failure_callback must fire so the user sees a warning (issue #15775)."""
        captured = []

        def _cb(task, exc):
            captured.append((task, exc))

        exc = RuntimeError("openrouter 402: credits exhausted")
        with patch("agent.title_generator.call_llm", side_effect=exc):
            result = generate_title("question", "answer", failure_callback=_cb)

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
            assert generate_title("q", "a", failure_callback=_bad_cb) is None

    def test_no_callback_matches_legacy_behavior(self):
        """Omitting failure_callback preserves the silent-None return."""
        with patch("agent.title_generator.call_llm", side_effect=RuntimeError("nope")):
            assert generate_title("q", "a") is None

    def test_truncates_long_messages(self):
        """Long user/assistant messages should be truncated in the LLM request."""
        captured_kwargs = {}

        def mock_call_llm(**kwargs):
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "Short Title"
            return resp

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm):
            generate_title("x" * 1000, "y" * 1000)

        # The user content in the messages should be truncated
        user_content = captured_kwargs["messages"][1]["content"]
        assert len(user_content) < 1100  # 500 + 500 + formatting

    def test_skips_when_title_generation_disabled(self):
        """auxiliary.title_generation.enabled=false disables automatic titles."""
        config = {"auxiliary": {"title_generation": {"enabled": False}}}

        with (
            patch("hermes_cli.config.load_config_readonly", return_value=config),
            patch("agent.title_generator.call_llm") as mock_call_llm,
        ):
            assert generate_title("question", "answer") is None

        mock_call_llm.assert_not_called()


class TestAutoTitleSession:
    """Tests for auto_title_session() — the sync worker function."""

    def test_skips_if_no_session_db(self):
        auto_title_session(None, "sess-1", "hi", "hello")  # should not crash

    def test_skips_if_title_exists(self):
        db = MagicMock()
        db.get_session_title.return_value = "Existing Title"

        with patch("agent.title_generator.generate_title") as gen:
            auto_title_session(db, "sess-1", "hi", "hello")
            gen.assert_not_called()

    def test_generates_and_sets_title(self):
        db = MagicMock()
        db.get_session_title.return_value = None
        db.set_auto_title_if_empty.return_value = True

        with patch("agent.title_generator.generate_title", return_value="New Title"):
            auto_title_session(db, "sess-1", "hi", "hello")
            db.set_auto_title_if_empty.assert_called_once_with("sess-1", "New Title")

    def test_does_not_overwrite_title_set_immediately_before_conditional_write(
        self, tmp_path
    ):
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        seen = []

        def generate_after_manual_title(*_args, **_kwargs):
            db.set_session_title("sess-1", "Manual Title")
            return "Auto Title"

        with patch(
            "agent.title_generator.generate_title",
            side_effect=generate_after_manual_title,
        ):
            auto_title_session(
                db,
                "sess-1",
                "hi",
                "hello",
                title_callback=seen.append,
            )

        assert db.get_session_title("sess-1") == "Manual Title"
        assert seen == []

    def test_invokes_title_callback_after_setting_title(self):
        db = MagicMock()
        db.get_session_title.return_value = None
        db.set_auto_title_if_empty.return_value = True
        seen = []
        with patch("agent.title_generator.generate_title", return_value="Readable Session"):
            auto_title_session(
                db,
                "sess-1",
                "hello",
                "hi there",
                title_callback=seen.append,
            )
        db.set_auto_title_if_empty.assert_called_once_with("sess-1", "Readable Session")
        assert seen == ["Readable Session"]

    def test_skips_if_generation_fails(self):
        db = MagicMock()
        db.get_session_title.return_value = None

        with patch("agent.title_generator.generate_title", return_value=None):
            auto_title_session(db, "sess-1", "hi", "hello")
            db.set_auto_title_if_empty.assert_not_called()

    def test_never_raises_when_body_throws(self):
        """Daemon-thread target must swallow ALL exceptions (e.g. the
        post-update stale-module ImportError) instead of spraying a raw
        traceback into the terminal via the default threading excepthook."""
        db = MagicMock()
        db.get_session_title.return_value = None

        with patch(
            "agent.title_generator._auto_title_session",
            side_effect=ImportError(
                "cannot import name 'set_conversation_context' from 'agent.portal_tags'"
            ),
        ):
            auto_title_session(db, "sess-1", "hi", "hello")  # must not raise

    def test_body_exception_routed_to_failure_callback(self):
        db = MagicMock()
        db.get_session_title.return_value = None
        seen = []

        boom = ImportError("stale module")
        with patch("agent.title_generator._auto_title_session", side_effect=boom):
            auto_title_session(
                db,
                "sess-1",
                "hi",
                "hello",
                failure_callback=lambda task, exc: seen.append((task, exc)),
            )
        assert seen == [("title generation", boom)]

    def test_failure_callback_errors_also_swallowed(self):
        db = MagicMock()
        db.get_session_title.return_value = None

        def bad_cb(task, exc):
            raise RuntimeError("callback itself broke")

        with patch(
            "agent.title_generator._auto_title_session",
            side_effect=ImportError("stale module"),
        ):
            auto_title_session(db, "sess-1", "hi", "hello", failure_callback=bad_cb)


class TestMaybeAutoTitle:
    """Tests for maybe_auto_title() — the fire-and-forget entry point."""

    def test_skips_if_not_first_exchange(self):
        """Should not fire for conversations with more than 2 user messages."""
        db = MagicMock()
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response 1"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "response 2"},
            {"role": "user", "content": "third"},
            {"role": "assistant", "content": "response 3"},
        ]

        with patch("agent.title_generator.auto_title_session") as mock_auto:
            maybe_auto_title(db, "sess-1", "third", "response 3", history)
            # Wait briefly for any thread to start
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
            import threading
            called = threading.Event()
            mock_auto.side_effect = lambda *a, **k: called.set()
            maybe_auto_title(db, "sess-1", "hello", "hi there", history)
            # Event-based wait: sleep-sync flaked when the daemon thread
            # wasn't scheduled within the fixed nap on a loaded runner.
            assert called.wait(timeout=10), "auto_title thread never ran"
            mock_auto.assert_called_once_with(
                db,
                "sess-1",
                "hello",
                "hi there",
                failure_callback=None,
                main_runtime=None,
                title_callback=None,
                runtime_validator=None,
            )

    def test_skips_when_title_generation_disabled(self):
        """Disabled title generation should not even start the background worker."""
        db = MagicMock()
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        config = {"auxiliary": {"title_generation": {"enabled": False}}}

        with (
            patch("hermes_cli.config.load_config_readonly", return_value=config),
            patch("agent.title_generator.auto_title_session") as mock_auto,
        ):
            maybe_auto_title(db, "sess-1", "hello", "hi there", history)

        mock_auto.assert_not_called()

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
            import threading
            called = threading.Event()
            mock_auto.side_effect = lambda *a, **k: called.set()
            maybe_auto_title(db, "sess-1", "hello", "hi there", history, failure_callback=_cb)
            assert called.wait(timeout=10), "auto_title thread never ran"
            mock_auto.assert_called_once_with(
                db,
                "sess-1",
                "hello",
                "hi there",
                failure_callback=_cb,
                main_runtime=None,
                title_callback=None,
                runtime_validator=None,
            )

    def test_skips_if_no_response(self):
        db = MagicMock()
        maybe_auto_title(db, "sess-1", "hello", "", [])  # empty response

    def test_skips_if_no_session_db(self):
        maybe_auto_title(None, "sess-1", "hello", "response", [])  # no db


class TestAutoTitleDuplicateHandling:
    """Duplicate auto-title handling and not-found hardening (#50537)."""

    def test_dedupes_duplicate_title_via_lineage(self):
        db = MagicMock()
        db.get_session_title.return_value = None
        # Atomic write path: collision raises ValueError, retry persists.
        db.set_auto_title_if_empty.side_effect = [ValueError("in use"), True]
        db.get_next_title_in_lineage.return_value = "Debugging Import Error #2"
        with patch(
            "agent.title_generator.generate_title",
            return_value="Debugging Import Error",
        ):
            seen = []
            auto_title_session(db, "sess-1", "hi", "hello", title_callback=seen.append)
        db.get_next_title_in_lineage.assert_called_once_with("Debugging Import Error")
        assert db.set_auto_title_if_empty.call_args_list[-1][0] == (
            "sess-1",
            "Debugging Import Error #2",
        )
        # callback fires with the actually-persisted (deduped) title
        assert seen == ["Debugging Import Error #2"]

    def test_dedupes_duplicate_title_via_lineage_legacy_store(self):
        # Store without set_auto_title_if_empty: same dedup via the plain
        # set_session_title fallback.
        db = MagicMock(
            spec=["get_session_title", "set_session_title", "get_next_title_in_lineage"]
        )
        db.get_session_title.return_value = None
        db.set_session_title.side_effect = [ValueError("in use"), True]
        db.get_next_title_in_lineage.return_value = "Debugging Import Error #2"
        with patch(
            "agent.title_generator.generate_title",
            return_value="Debugging Import Error",
        ):
            seen = []
            auto_title_session(db, "sess-1", "hi", "hello", title_callback=seen.append)
        assert db.set_session_title.call_args_list[-1][0] == (
            "sess-1",
            "Debugging Import Error #2",
        )
        assert seen == ["Debugging Import Error #2"]

    def test_swallows_value_error_without_lineage_support(self):
        # No get_next_title_in_lineage -> ValueError propagates out of the
        # persist helper but auto_title_session still swallows it (no crash).
        db = MagicMock(spec=["get_session_title", "set_session_title"])
        db.get_session_title.return_value = None
        db.set_session_title.side_effect = ValueError("in use")
        with patch(
            "agent.title_generator.generate_title", return_value="Dup Title"
        ):
            auto_title_session(db, "sess-1", "hi", "hello")  # must not raise

    def test_manual_title_race_skips_without_callback(self):
        # Atomic predicate fails (manual /title landed while generation was in
        # flight) -> nothing persisted, no callback fired.
        from agent.title_generator import _persist_session_title
        db = MagicMock()
        db.set_auto_title_if_empty.return_value = False
        assert _persist_session_title(db, "sess-1", "Some Title") is None
        db.set_session_title.assert_not_called()

    def test_not_found_raises_runtime_error_internally(self):
        # Legacy store (no atomic write): set_session_title returning False
        # (session vanished) -> RuntimeError in the persist helper, swallowed
        # by auto_title_session, no callback.
        from agent.title_generator import _persist_session_title
        db = MagicMock(spec=["get_session_title", "set_session_title"])
        db.set_session_title.return_value = False
        with pytest.raises(RuntimeError):
            _persist_session_title(db, "missing", "Some Title")


class TestRuntimeValidator:
    """runtime_validator gating (#19027): a stale background title request
    must not fire when the session's model/provider changed after spawn."""

    def test_skips_when_validator_returns_false(self):
        with patch("agent.title_generator.call_llm") as mock_llm:
            title = generate_title(
                "question", "answer",
                runtime_validator=lambda: False,
            )
            assert title is None
            mock_llm.assert_not_called()

    def test_allows_when_validator_returns_true(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Validated Title"

        with patch("agent.title_generator.call_llm", return_value=mock_response) as mock_llm:
            title = generate_title(
                "question", "answer",
                runtime_validator=lambda: True,
            )
            assert title == "Validated Title"
            mock_llm.assert_called_once()

    def test_broken_validator_fails_open(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Resilient Title"

        def _bad_validator():
            raise RuntimeError("validator gone")

        with patch("agent.title_generator.call_llm", return_value=mock_response) as mock_llm:
            title = generate_title(
                "question", "answer",
                runtime_validator=_bad_validator,
            )
            assert title == "Resilient Title"
            mock_llm.assert_called_once()

    def test_forwards_runtime_validator_to_worker(self):
        db = MagicMock()
        db.get_session_title.return_value = None
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        def _v():
            return True

        with patch("agent.title_generator.auto_title_session") as mock_auto:
            import threading
            called = threading.Event()
            mock_auto.side_effect = lambda *a, **k: called.set()
            maybe_auto_title(db, "sess-1", "hello", "hi there", history, runtime_validator=_v)
            assert called.wait(timeout=10), "auto_title thread never ran"
            kwargs = mock_auto.call_args.kwargs
            assert kwargs["runtime_validator"] is _v


class TestCondenseHistory:
    """Tests for _condense_history() — the whole-conversation renderer."""

    def test_empty_history_returns_empty(self):
        assert _condense_history([]) == ""
        assert _condense_history(None) == ""

    def test_skips_system_and_tool_roles(self):
        history = [
            {"role": "system", "content": "you are an agent"},
            {"role": "user", "content": "hello"},
            {"role": "tool", "content": "tool output"},
            {"role": "assistant", "content": "hi there"},
        ]
        out = _condense_history(history)
        assert "you are an agent" not in out
        assert "tool output" not in out
        assert "User: hello" in out
        assert "Assistant: hi there" in out

    def test_short_history_not_elided(self):
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        out = _condense_history(history)
        assert "omitted" not in out
        assert out.count("User:") == 2

    def test_long_history_keeps_head_and_tail_with_elision(self):
        # 10 exchanges = 20 messages; head_turns=1 (2 msgs) + tail_turns=3 (6 msgs)
        history = []
        for i in range(10):
            history.append({"role": "user", "content": f"question {i}"})
            history.append({"role": "assistant", "content": f"answer {i}"})
        out = _condense_history(history)
        # Opening turn preserved (anchors intent)
        assert "question 0" in out
        assert "answer 0" in out
        # Latest turns preserved (detect drift)
        assert "question 9" in out
        assert "answer 9" in out
        # A middle turn is gone
        assert "question 5" not in out
        # Elision marker present
        assert "omitted" in out

    def test_truncates_long_messages(self):
        history = [
            {"role": "user", "content": "x" * 1000},
            {"role": "assistant", "content": "y" * 1000},
        ]
        out = _condense_history(history)
        # each message truncated to per_message (400) + ellipsis, not full 1000
        assert "x" * 401 not in out
        assert "…" in out


class TestRegenerateTitle:
    """Tests for regenerate_title() — whole-conversation, sticky re-assessment."""

    def _resp(self, text):
        r = MagicMock()
        r.choices = [MagicMock()]
        r.choices[0].message.content = text
        return r

    def test_returns_none_on_empty_history(self):
        # No LLM call should happen when there's no transcript.
        with patch("agent.title_generator.call_llm") as llm:
            assert regenerate_title([], "Some Title") is None
            llm.assert_not_called()

    def test_keeps_current_title_when_unchanged(self):
        history = [
            {"role": "user", "content": "help me draft the USCIS RFE response"},
            {"role": "assistant", "content": "Here's the outline..."},
            {"role": "user", "content": "now write the PDF"},
            {"role": "assistant", "content": "Generating the PDF..."},
        ]
        # Model, seeing the whole conversation, returns the existing title verbatim.
        with patch("agent.title_generator.call_llm", return_value=self._resp("USCIS RFE Response")):
            out = regenerate_title(history, "USCIS RFE Response")
            assert out == "USCIS RFE Response"

    def test_whole_conversation_passed_to_model_not_just_last_exchange(self):
        """The USCIS-RFE bug: a localized 'write the PDF' detour must not be the
        only thing the model sees. The opening intent must reach the prompt."""
        history = [
            {"role": "user", "content": "help me draft the USCIS RFE response gist"},
            {"role": "assistant", "content": "Here's the outline of the RFE response..."},
            {"role": "user", "content": "looks good, keep going"},
            {"role": "assistant", "content": "Continuing the RFE draft..."},
            {"role": "user", "content": "now produce the PDF of it"},
            {"role": "assistant", "content": "Rendering the PDF now..."},
        ]
        captured = {}

        def _cap(**kwargs):
            captured.update(kwargs)
            return self._resp("USCIS RFE Response")

        with patch("agent.title_generator.call_llm", side_effect=_cap):
            regenerate_title(history, "USCIS RFE Response")

        user_block = captured["messages"][1]["content"]
        system_block = captured["messages"][0]["content"]
        # Current title is handed to the model
        assert "USCIS RFE Response" in user_block
        # Opening intent (the real gist) is present, not just the PDF detour
        assert "RFE response gist" in user_block
        # The PDF detour is present too (tail), but as context, not the sole input
        assert "PDF" in user_block
        # Prompt instructs whole-conversation, keep-biased assessment
        assert "WHOLE" in system_block
        assert "UNCHANGED" in system_block

    def test_returns_new_title_on_genuine_drift(self):
        history = [
            {"role": "user", "content": "help me draft the USCIS RFE response"},
            {"role": "assistant", "content": "Here's the outline..."},
            {"role": "user", "content": "actually forget that, let's debug my docker setup"},
            {"role": "assistant", "content": "Let's look at your Dockerfile..."},
            {"role": "user", "content": "the container won't start"},
            {"role": "assistant", "content": "Check the entrypoint..."},
        ]
        with patch("agent.title_generator.call_llm", return_value=self._resp("Debugging Docker Setup")):
            out = regenerate_title(history, "USCIS RFE Response")
            assert out == "Debugging Docker Setup"

    def test_pinned_language_prompt(self):
        history = [
            {"role": "user", "content": "hola"},
            {"role": "assistant", "content": "hola, como estas"},
        ]
        captured = {}

        def _cap(**kwargs):
            captured.update(kwargs)
            return self._resp("Saludo")

        with (
            patch("agent.title_generator.call_llm", side_effect=_cap),
            patch("agent.title_generator._title_language", return_value="Spanish"),
        ):
            regenerate_title(history, "Greeting")

        system_block = captured["messages"][0]["content"]
        assert "Write the title in Spanish" in system_block

    def test_returns_none_on_exception(self):
        history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        with patch("agent.title_generator.call_llm", side_effect=RuntimeError("no provider")):
            assert regenerate_title(history, "Title") is None

    def test_invokes_failure_callback_on_exception(self):
        history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        captured = []
        exc = RuntimeError("boom")
        with patch("agent.title_generator.call_llm", side_effect=exc):
            regenerate_title(history, "Title", failure_callback=lambda t, e: captured.append((t, e)))
        assert captured == [("title regeneration", exc)]

    def test_rejects_conversational_prose_instead_of_truncating(self):
        """Regression: the model answered the "should this change?" question in
        PROSE ("The title remains accurate. The conversation is still about …")
        instead of returning a title. The old code sanitized + truncated it at
        80 chars and stored the sentence AS the title, which then became the
        Discord thread name. Prose must be rejected → None → keep current title.
        """
        history = [
            {"role": "user", "content": "run hermes update"},
            {"role": "assistant", "content": "Starting the triage..."},
            {"role": "user", "content": "restart and verify"},
            {"role": "assistant", "content": "Gateway restarted cleanly."},
        ]
        prose = (
            "The title remains accurate. The conversation is still about "
            "triaging and executing the hermes update"
        )
        with patch("agent.title_generator.call_llm", return_value=self._resp(prose)):
            out = regenerate_title(history, "Hermes Update Triage")
        assert out is None

    def test_rejects_overlong_prose_not_truncate(self):
        """A >80-char blob must be rejected (None), never truncated into a title."""
        history = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        with patch("agent.title_generator.call_llm", return_value=self._resp("A" * 100)):
            assert regenerate_title(history, "Existing Title") is None

    def test_accepts_abbreviation_titles_with_internal_dots(self):
        """The prose guard must NOT false-reject legit titles whose dotted
        component is an abbreviation ("U.S. Visa Renewal") — only lowercase-
        word sentence breaks count as prose."""
        history = [
            {"role": "user", "content": "help with my visa"},
            {"role": "assistant", "content": "Sure..."},
            {"role": "user", "content": "timeline?"},
            {"role": "assistant", "content": "Here..."},
        ]
        with patch("agent.title_generator.call_llm", return_value=self._resp("U.S. Visa Renewal Timeline")):
            assert regenerate_title(history, "Visa Help") == "U.S. Visa Renewal Timeline"


class TestLooksLikeTitle:
    """Unit tests for the _looks_like_title prose-rejection shape guard."""

    @pytest.mark.parametrize("text", [
        "USCIS RFE Response",
        "Debugging Python Import Errors",
        "Setting Up Docker Environment",
        "Kubernetes Pod Debugging",
        "U.S. Visa Renewal Timeline",   # uppercase/single-letter abbreviation dots
        "Q3 Financial Review",
    ])
    def test_accepts_real_titles(self, text):
        from agent.title_generator import _looks_like_title
        assert _looks_like_title(text) is True

    @pytest.mark.parametrize("text", [
        "",
        "The title remains accurate. The conversation is still about triaging",  # sentence break
        "A" * 100,                                                                # >80 chars
        "one two three four five six seven eight nine ten eleven",                # >10 words
        "Fixing the bug. Then shipping it",                                       # mid-sentence period
        "Line one\nLine two",                                                     # internal newline
    ])
    def test_rejects_prose_and_garbage(self, text):
        from agent.title_generator import _looks_like_title
        assert _looks_like_title(text) is False


class TestMaybeRetitleSession:
    """Tests for maybe_retitle_session() — the periodic re-title gate."""

    def _history(self, n_user):
        h = []
        for i in range(n_user):
            h.append({"role": "user", "content": f"q{i}"})
            h.append({"role": "assistant", "content": f"a{i}"})
        return h

    def test_skips_before_third_user_turn(self):
        db = MagicMock()
        with patch("agent.title_generator.regenerate_title") as regen:
            maybe_retitle_session(db, "s1", "q", "a", self._history(2), every_n_turns=6)
            import time
            time.sleep(0.1)
            regen.assert_not_called()

    def test_skips_off_cadence(self):
        db = MagicMock()
        # 4 user turns, every_n_turns=6 -> 4 % 6 != 0 -> skip
        with patch("agent.title_generator.regenerate_title") as regen:
            maybe_retitle_session(db, "s1", "q", "a", self._history(4), every_n_turns=6)
            import time
            time.sleep(0.1)
            regen.assert_not_called()

    def test_fires_on_cadence_and_uses_regenerate_title(self):
        db = MagicMock()
        db.get_session_title.return_value = "Old Title"
        history = self._history(6)  # 6 % 6 == 0 -> fire
        with patch("agent.title_generator.regenerate_title", return_value="New Title") as regen:
            maybe_retitle_session(db, "s1", "q", "a", history, every_n_turns=6)
            import time
            time.sleep(0.3)
            regen.assert_called_once()
            # regenerate_title must receive the full history + current title,
            # NOT just the last user/assistant message.
            args, kwargs = regen.call_args
            assert args[0] == history
            assert args[1] == "Old Title"
        db.set_session_title.assert_called_once_with("s1", "New Title")

    def test_no_db_write_when_title_unchanged(self):
        db = MagicMock()
        db.get_session_title.return_value = "Same Title"
        history = self._history(6)
        with patch("agent.title_generator.regenerate_title", return_value="Same Title"):
            maybe_retitle_session(db, "s1", "q", "a", history, every_n_turns=6)
            import time
            time.sleep(0.3)
        db.set_session_title.assert_not_called()

    def test_callback_fires_on_change(self):
        db = MagicMock()
        db.get_session_title.return_value = "Old"
        history = self._history(6)
        seen = []
        with patch("agent.title_generator.regenerate_title", return_value="Brand New"):
            maybe_retitle_session(
                db, "s1", "q", "a", history, every_n_turns=6, title_callback=seen.append
            )
            import time
            time.sleep(0.3)
        assert seen == ["Brand New"]
