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




    def test_title_language_reads_config(self):
        cfg = {"auxiliary": {"title_generation": {"language": "  French "}}}

        with patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg):
            assert _title_language() == "French"
        with patch("hermes_cli.config.load_config", return_value={}), patch("hermes_cli.config.load_config_readonly", return_value={}):
            assert _title_language() == ""
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("bad config")), \
         patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("bad config")):
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
            assert generate_title("question") == "Configured Timeout"

        assert captured_kwargs["task"] == "title_generation"
        assert captured_kwargs["timeout"] is None



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
            title = generate_title("help me fix this import")
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
            title = generate_title("hello")
            # Everything from the unterminated open tag onward is stripped,
            # leaving nothing → None.
            assert title is None


    def test_truncates_long_titles(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A" * 100

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title("question")
            assert len(title) == 80
            assert title.endswith("...")

    def test_rejects_answer_shaped_output(self):
        """A model that ignores the titling task and answers the user's
        message returns a full sentence; without a word bound the whole
        reply (truncated mid-sentence) became the session title.
        Regression for the can1357/oh-my-pi#7306 bug class."""
        answer = (
            "I don't have context on a \"registration system\" - that's not "
            "something I recognize from this conversation, and I don't see "
            "any prior discussion or code about it here"
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = answer

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert generate_title("how does the registration system work?", "...") is None

    def test_rejects_many_short_words(self):
        """13 short words stays under the 80-char cap but is not a title."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "one two three four five six seven eight nine ten eleven twelve thirteen"
        )

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert generate_title("question", "answer") is None

    def test_accepts_normal_title(self):
        """A normal 3-7 word title is unaffected by the answer-shape guard."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Investigate the title resolver bug"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert generate_title("question", "answer") == "Investigate the title resolver bug"



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











class TestAutoTitleSession:
    """Tests for auto_title_session() — the sync worker function."""




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
                title_callback=lambda title, source: seen.append(title),
            )

        assert db.get_session_title("sess-1") == "Manual Title"
        assert seen == []

    def test_invokes_title_callback_after_setting_title(self):
        db = MagicMock()
        db.get_session_title_source.return_value = None
        db.set_auto_title.return_value = True
        seen = []
        with patch("agent.title_generator.generate_title", return_value="Readable Session"):
            auto_title_session(
                db,
                "sess-1",
                "hello",
                title_callback=lambda title, source: seen.append((title, source)),
            )
        db.set_auto_title.assert_called_once_with(
            "sess-1", "Readable Session", source="llm"
        )
        # The stage reaches the consumer, so one that spends a rate-limited
        # remote call per title can take this and skip the derived one.
        assert seen == [("Readable Session", "llm")]

    def test_upgrades_a_derived_title_but_not_an_llm_one(self, tmp_path):
        """The instant title is provisional; a model title is final.

        This is the "session renames itself" guard: re-running the titler on a
        session that already has an LLM title must be a no-op.
        """
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        db.set_auto_title("sess-1", "fix the flaky auth test", source="derived")

        with patch("agent.title_generator.generate_title", return_value="Fix flaky auth test"):
            auto_title_session(db, "sess-1", "fix the flaky auth test")
        assert db.get_session_title("sess-1") == "Fix flaky auth test"

        with patch("agent.title_generator.generate_title", return_value="Totally Different"):
            auto_title_session(db, "sess-1", "fix the flaky auth test")
        assert db.get_session_title("sess-1") == "Fix flaky auth test"



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
                failure_callback=lambda task, exc: seen.append((task, exc)),
            )
        assert seen == [("title generation", boom)]



class TestMaybeAutoTitle:
    """Tests for maybe_auto_title() — the fire-and-forget entry point."""

    def test_skips_if_not_first_exchange(self):
        """Should not fire once the conversation is past its opening turn."""
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
            maybe_auto_title(db, "sess-1", "third", history)
            # Wait briefly for any thread to start
            import time
            time.sleep(0.1)
            mock_auto.assert_not_called()

    def test_fires_on_first_exchange(self):
        """Should fire a background thread for the opening message."""
        db = MagicMock()
        db.get_session_title.return_value = None
        history = [
            {"role": "user", "content": "hello"},
        ]

        with patch("agent.title_generator.auto_title_session") as mock_auto:
            import threading
            called = threading.Event()
            mock_auto.side_effect = lambda *a, **k: called.set()
            maybe_auto_title(db, "sess-1", "hello", history)
            # Event-based wait: sleep-sync flaked when the daemon thread
            # wasn't scheduled within the fixed nap on a loaded runner.
            assert called.wait(timeout=10), "auto_title thread never ran"
            mock_auto.assert_called_once_with(
                db,
                "sess-1",
                "hello",
                failure_callback=None,
                main_runtime=None,
                title_callback=None,
                runtime_validator=None,
            )

    def test_writes_instant_title_before_the_model_runs(self, tmp_path):
        """The derived title lands synchronously — no LLM, no waiting."""
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        with patch("agent.title_generator.auto_title_session"):
            maybe_auto_title(
                db, "sess-1", "fix the flaky auth test in login", []
            )
        assert db.get_session_title("sess-1") == "fix the flaky auth test in login"
        assert db.get_session_title_source("sess-1") == "derived"

    def test_skips_machine_authored_opening_messages(self, tmp_path):
        """A compaction handoff is not a user request and must not title."""
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        with patch("agent.title_generator.auto_title_session") as mock_auto:
            maybe_auto_title(
                db,
                "sess-1",
                "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted",
                [],
            )
        assert db.get_session_title("sess-1") is None
        mock_auto.assert_not_called()

    @pytest.mark.parametrize(
        "opener",
        [
            "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted",
            "[CONTEXT SUMMARY]: the user was refactoring the auth module",
            "[System note: the user switched models]",
            "[Runtime note: resumed from checkpoint]",
        ],
    )
    def test_skips_every_shape_of_machine_authored_opener(self, tmp_path, opener):
        """A session named after our own scaffolding is named after us."""
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        with patch("agent.title_generator.auto_title_session") as mock_auto:
            maybe_auto_title(db, "sess-1", opener, [])
        assert db.get_session_title("sess-1") is None
        mock_auto.assert_not_called()

    def test_a_multimodal_turn_counts_as_a_real_question(self, tmp_path):
        """"Here's a screenshot, fix the login" is a question, parts list or not.

        Judging a turn by `content` alone reads a multimodal one as machinery
        and undercounts the conversation, so a session deep into its history
        looks like it is still on its opening turn.
        """
        from agent.title_generator import _is_real_user_turn

        assert _is_real_user_turn(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
                    {"type": "text", "text": "fix the login button"},
                ],
            }
        )
        # An image with no words is not a question we can name anything after.
        assert not _is_real_user_turn(
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}
        )

    def test_titles_on_a_later_turn_when_the_opener_was_not_titleable(self, tmp_path):
        """A session whose opener couldn't be titled gets named by a later turn.

        The opener here is a compaction handoff, so turn one leaves the session
        nameless. Nothing used to reconsider it: the guard that stops re-titling
        a named session also stopped the nameless one from ever asking again.
        """
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        history = [
            {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY] x"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "sure"},
        ]
        with patch("agent.title_generator.auto_title_session"):
            maybe_auto_title(db, "sess-1", "fix the flaky auth test", history)
        assert db.get_session_title("sess-1") == "fix the flaky auth test"

    def test_leaves_an_already_titled_session_alone_on_later_turns(self, tmp_path):
        """The retry is for nameless sessions only; a named one asks nothing."""
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        db.set_session_title("sess-1", "Existing name")
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "sure"},
        ]
        with patch("agent.title_generator.auto_title_session") as mock_auto:
            maybe_auto_title(db, "sess-1", "and now something else", history)
        assert db.get_session_title("sess-1") == "Existing name"
        mock_auto.assert_not_called()

    def test_instant_title_declines_a_name_collision(self, tmp_path):
        """A colliding derived title is skipped, not scanned into 'hi #2'.

        Common openers collide constantly, and the lineage scan that resolves
        the collision runs inline on the turn. The model's title lands moments
        later, so the session is named either way.
        """
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="taken", source="cli")
        db.set_session_title("taken", "hi")
        db.create_session(session_id="sess-1", source="cli")
        with patch("agent.title_generator.auto_title_session"):
            maybe_auto_title(db, "sess-1", "hi", [])
        assert db.get_session_title("sess-1") is None






class TestAutoTitleDuplicateHandling:
    """Duplicate auto-title handling and not-found hardening (#50537)."""

    def test_background_stage_names_a_collision_the_instant_stage_declined(
        self, tmp_path
    ):
        """The lineage scan the turn skipped happens here instead.

        The inline stage declines a collision to stay off the critical path, and
        the model can still come back empty. Between them the session would be
        left nameless, so the background stage spends the scan the turn wouldn't.
        """
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="taken", source="cli")
        db.set_session_title("taken", "hi")
        db.create_session(session_id="sess-1", source="cli")
        with patch("agent.title_generator.generate_title", return_value=None):
            auto_title_session(db, "sess-1", "hi")
        assert db.get_session_title("sess-1") == "hi #2"

    def test_dedupes_duplicate_title_via_lineage(self):
        db = MagicMock()
        db.get_session_title_source.return_value = None
        # Atomic write path: collision raises ValueError, retry persists.
        db.set_auto_title.side_effect = [ValueError("in use"), True]
        db.get_next_title_in_lineage.return_value = "Debugging Import Error #2"
        with patch(
            "agent.title_generator.generate_title",
            return_value="Debugging Import Error",
        ):
            seen = []
            auto_title_session(
                db,
                "sess-1",
                "hi",
                title_callback=lambda title, _source: seen.append(title),
            )
        db.get_next_title_in_lineage.assert_called_once_with("Debugging Import Error")
        assert db.set_auto_title.call_args_list[-1][0] == (
            "sess-1",
            "Debugging Import Error #2",
        )
        # callback fires with the actually-persisted (deduped) title
        assert seen == ["Debugging Import Error #2"]



    def test_manual_title_race_skips_without_callback(self):
        # Precedence check fails (manual /title landed while generation was in
        # flight) -> nothing persisted, no callback fired.
        from agent.title_generator import _persist_session_title
        db = MagicMock()
        db.set_auto_title.return_value = False
        assert (
            _persist_session_title(db, "sess-1", "Some Title", source="llm") is None
        )
        db.set_session_title.assert_not_called()



class TestRuntimeValidator:
    """runtime_validator gating (#19027): a stale background title request
    must not fire when the session's model/provider changed after spawn."""



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
            maybe_auto_title(db, "sess-1", "hello", history, runtime_validator=_v)
            assert called.wait(timeout=10), "auto_title thread never ran"
            kwargs = mock_auto.call_args.kwargs
            assert kwargs["runtime_validator"] is _v


class TestModelSwitchMarkerNotTitleable:
    """Regression: a model-switch marker must never become the session title.

    ``_append_model_switch_marker`` (tui_gateway/server.py) persists its notice
    with ``role="user"`` because strict OpenAI-compatible providers reject a
    system message that is not first (#48338). Titling therefore has to
    recognise it as machine-authored, or switching models before asking the
    first real question titles the session
    "[System: The active model for this chat has…".
    """

    MARKER = (
        "[System: The active model for this chat has changed to "
        "deepseek-v4-flash via provider 94mei. From this point forward, use "
        "this runtime metadata when answering questions about what "
        "model/provider is active.]"
    )

    def test_marker_prefix_matches_gateway_constant(self):
        """The guard must stay in sync with the gateway's marker builder."""
        from tui_gateway.server import _MODEL_SWITCH_MARKER_PREFIX
        from agent.title_generator import _MACHINE_PREFIXES

        assert _MODEL_SWITCH_MARKER_PREFIX in _MACHINE_PREFIXES
        assert self.MARKER.startswith(_MODEL_SWITCH_MARKER_PREFIX)

    def test_marker_is_not_titleable(self):
        from agent.title_generator import is_titleable_user_message

        assert is_titleable_user_message(self.MARKER) is False

    def test_derive_title_is_unguarded_by_design(self):
        """``derive_title`` is a dumb formatter; the guard lives in the callers.

        Documents the contract deliberately: every caller checks
        ``is_titleable_user_message`` first, so ``derive_title`` itself is
        allowed to format a marker. If a future caller forgets that check, the
        marker leaks into the title — which is exactly the bug this class
        guards against.
        """
        from agent.title_generator import derive_title

        assert derive_title(self.MARKER) is not None

    def test_unrelated_system_bracket_text_still_titleable(self):
        """The guard is narrow: real user text starting "[System:" still titles."""
        from agent.title_generator import is_titleable_user_message

        assert is_titleable_user_message("[System: my own note] how do I ...") is True

    def test_real_question_after_marker_still_titles(self):
        """The marker must not consume the session's one titling opportunity.

        The marker is a role="user" row, so counting it made the first real
        question look like turn 2 — and titling bailed out entirely, leaving
        the session permanently untitled.
        """
        db = MagicMock()
        db.get_session_title.return_value = None
        db.get_session_title_source.return_value = None
        history = [
            {"role": "user", "content": self.MARKER},
            {"role": "user", "content": "南京市秦淮区 小时级天气预报"},
        ]

        with patch("agent.title_generator.auto_title_session") as mock_auto:
            import threading

            called = threading.Event()
            mock_auto.side_effect = lambda *a, **k: called.set()
            maybe_auto_title(db, "sess-1", "南京市秦淮区 小时级天气预报", history)
            assert called.wait(timeout=10), "auto_title never ran after marker"

    def test_instant_title_skips_marker_uses_real_message(self):
        from agent.title_generator import apply_instant_title

        db = MagicMock()
        db.get_session_title_source.return_value = None

        assert apply_instant_title(db, "sess-1", self.MARKER) is None
        assert apply_instant_title(db, "sess-1", "南京市秦淮区 小时级天气预报") == (
            "南京市秦淮区 小时级天气预报"
        )


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
