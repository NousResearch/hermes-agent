"""Tests for agent.title_generator — auto-generated session titles."""

import pytest
from unittest.mock import MagicMock, patch


from agent.title_generator import (
    generate_title,
    auto_title_session,
    maybe_auto_title,
    _title_language,
    _retitle_config,
    _retitle_enabled,
    _condense_history,
    _looks_like_title,
    regenerate_title,
    retitle_session,
    maybe_auto_retitle,
    MAX_TITLE_INPUT_CHARS,
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
        db.get_session_title_source.return_value = None
        seen = []

        boom = ImportError("stale module")
        with patch("agent.title_generator.generate_title", side_effect=boom):
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


class TestRetitleConfig:
    """Unit tests for _retitle_config() and _retitle_enabled()."""

    DEFAULT_KEYS = {
        "enabled": True,
        "auto_at_turn": 10,
        "turns_window": 10,
        "slash_command": True,
        "cli_command": True,
        "touch_platform_names": False,
        "provider": "",
        "model": "",
        "base_url": "",
        "api_key": "",
        "timeout": 30,
        "max_concurrency": 2,
        "prefer_fast_model": None,
    }

    def test_retitle_config_returns_defaults_when_block_missing(self):
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch("hermes_cli.config.load_config_readonly", return_value={}):
            cfg = _retitle_config()
        for key, expected in self.DEFAULT_KEYS.items():
            assert cfg[key] == expected, f"default for {key} lost"

    def test_retitle_config_merges_partial_user_overrides(self):
        user_cfg = {
            "auxiliary": {
                "title_generation": {
                    "retitle": {
                        "enabled": False,
                        "auto_at_turn": 20,
                        # None must be treated as "use default"
                        "provider": None,
                    }
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=user_cfg), \
             patch("hermes_cli.config.load_config_readonly", return_value=user_cfg):
            cfg = _retitle_config()

        assert cfg["enabled"] is False
        assert cfg["auto_at_turn"] == 20
        # None override was skipped → default preserved
        assert cfg["provider"] == ""
        # Untouched keys keep defaults
        assert cfg["turns_window"] == 10
        assert cfg["slash_command"] is True
        assert cfg["timeout"] == 30
        assert cfg["max_concurrency"] == 2
        assert cfg["prefer_fast_model"] is None

    def test_retitle_config_returns_empty_dict_on_config_error(self):
        with patch("hermes_cli.config.load_config",
                   side_effect=RuntimeError("bad config")), \
             patch("hermes_cli.config.load_config_readonly",
                   side_effect=RuntimeError("bad config")):
            assert _retitle_config() == {}

    def test_retitle_enabled_true_by_default(self):
        with patch("hermes_cli.config.load_config", return_value={}), \
             patch("hermes_cli.config.load_config_readonly", return_value={}):
            assert _retitle_enabled() is True

    def test_retitle_enabled_false_when_disabled(self):
        user_cfg = {
            "auxiliary": {
                "title_generation": {
                    "retitle": {"enabled": False}
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=user_cfg), \
             patch("hermes_cli.config.load_config_readonly", return_value=user_cfg):
            assert _retitle_enabled() is False

    def test_retitle_enabled_true_when_config_broken(self):
        with patch("hermes_cli.config.load_config",
                   side_effect=RuntimeError("bad config")), \
             patch("hermes_cli.config.load_config_readonly",
                   side_effect=RuntimeError("bad config")):
            assert _retitle_enabled() is True


class TestCondenseHistory:
    """Unit tests for _condense_history()."""

    def test_returns_empty_for_none_history(self):
        assert _condense_history(None) == ""

    def test_returns_empty_for_empty_history(self):
        assert _condense_history([]) == ""

    def test_includes_recent_user_and_assistant_turns_with_labels(self):
        history = [
            {"role": "user", "content": "How do I center a div?"},
            {"role": "assistant", "content": "Use flexbox with justify-content."},
        ]
        result = _condense_history(history)
        assert "User: How do I center a div?" in result
        assert "Assistant: Use flexbox with justify-content." in result

    def test_skips_tool_messages(self):
        history = [
            {"role": "user", "content": "Run the tests"},
            {"role": "tool", "content": "pytest output goes here"},
            {"role": "assistant", "content": "Tests passed."},
        ]
        result = _condense_history(history)
        assert "pytest output" not in result
        assert "User: Run the tests" in result
        assert "Assistant: Tests passed." in result

    def test_skips_machine_authored_user_turns(self):
        history = [
            {"role": "user", "content": "[System note: session resumed]"},
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "I can't check weather directly."},
        ]
        result = _condense_history(history)
        assert "[System note:" not in result
        assert "User: What's the weather like?" in result

    def test_skips_empty_assistant_content(self):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "   "},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = _condense_history(history)
        lines = result.split("\n")
        assistant_lines = [ln for ln in lines if ln.startswith("Assistant:")]
        assert len(assistant_lines) == 1
        assert "Assistant: Hello!" in result

    def test_truncates_individual_message_bodies_to_200_chars(self):
        long_body = "a" * 500
        history = [
            {"role": "user", "content": long_body},
            {"role": "assistant", "content": long_body},
        ]
        result = _condense_history(history)
        for line in result.split("\n"):
            # strip label prefix
            if line.startswith("User: "):
                body = line[len("User: "):]
            elif line.startswith("Assistant: "):
                body = line[len("Assistant: "):]
            else:
                continue
            assert len(body) <= 200

    def test_returns_only_last_turns_window_pairs(self):
        history = []
        for i in range(20):
            history.append({"role": "user", "content": f"user-msg-{i}"})
            history.append({"role": "assistant", "content": f"assistant-msg-{i}"})
        result = _condense_history(history, turns_window=5)
        # window=5 => last 10 messages: user-msg-15..19, assistant-msg-15..19
        assert "user-msg-0" not in result
        assert "user-msg-14" not in result
        assert "user-msg-15" in result
        assert "user-msg-19" in result
        assert "assistant-msg-19" in result

    def test_final_output_capped_at_max_title_input_chars(self):
        history = []
        chunk = "word " * 40  # ~200 chars → truncated to 200 per body
        for i in range(200):
            history.append({"role": "user", "content": chunk})
            history.append({"role": "assistant", "content": chunk})
        result = _condense_history(history, turns_window=100)
        assert len(result) <= MAX_TITLE_INPUT_CHARS

    def test_multimodal_user_content_flattens_to_text(self):
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here's a screenshot,"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                    {"type": "text", "text": "fix the login."},
                ],
            },
            {"role": "assistant", "content": "Sure, checking now."},
        ]
        result = _condense_history(history)
        assert "User:" in result
        assert "Here's a screenshot," in result
        assert "fix the login." in result


class TestLooksLikeTitle:
    """Unit tests for _looks_like_title() quality gate."""

    def test_rejects_empty(self):
        assert _looks_like_title("") is False

    def test_rejects_none(self):
        assert _looks_like_title(None) is False

    def test_rejects_single_word(self):
        assert _looks_like_title("Hello") is False

    def test_rejects_thirteen_word_prose(self):
        text = "one two three four five six seven eight nine ten eleven twelve thirteen"
        assert _looks_like_title(text) is False

    def test_rejects_here_is_prefix(self):
        assert _looks_like_title("Here is a summary") is False

    def test_rejects_about_prefix(self):
        assert _looks_like_title("about databases and connection pools") is False

    def test_rejects_this_conversation_prefix(self):
        assert _looks_like_title("This conversation is about Postgres") is False

    def test_rejects_the_conversation_prefix(self):
        assert _looks_like_title("The conversation covers Postgres") is False

    def test_accepts_concise_two_words(self):
        assert _looks_like_title("Friendly greeting") is True

    def test_accepts_seven_words(self):
        assert (
            _looks_like_title(
                "Debugging Postgres connection pool exhaustion during migration"
            )
            is True
        )

    def test_accepts_the_prefix_when_not_conversation(self):
        assert _looks_like_title("The Postgres pool issue") is True


class TestRegenerateTitle:
    """Unit tests for regenerate_title()."""

    def test_returns_none_for_empty_condensed(self):
        assert regenerate_title("") is None
        assert regenerate_title(None) is None
        assert regenerate_title("   \n  ") is None

    def test_returns_none_when_retitle_disabled(self):
        with patch("agent.title_generator._retitle_enabled", return_value=False):
            assert regenerate_title("User: hi\nAssistant: hello") is None

    def test_calls_call_llm_with_condensed_and_title_generation_task(self):
        captured = {}

        def mock_call_llm(**kwargs):
            captured.update(kwargs)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = '{"title": "Debugging Postgres pool"}'
            return resp

        condensed = "User: Postgres pool is exhausted\nAssistant: Let's check settings"
        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm), \
             patch("agent.title_generator._retitle_enabled", return_value=True):
            title = regenerate_title(condensed)

        assert title == "Debugging Postgres pool"
        assert captured["task"] == "title_generation"
        assert captured["max_tokens"] == 64
        assert captured["temperature"] == 0.3
        messages = captured["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == condensed
        # response_format schema is passed through
        assert "response_format" in captured["extra_body"]

    def test_extracts_json_response_via_extract_title_text(self):
        def mock_call_llm(**kwargs):
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = '{"title": "Fix login button on mobile"}'
            return resp

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm), \
             patch("agent.title_generator._retitle_enabled", return_value=True):
            assert regenerate_title("User: fix mobile login") == "Fix login button on mobile"

    def test_rejects_prose_via_looks_like_title(self):
        def mock_call_llm(**kwargs):
            resp = MagicMock()
            resp.choices = [MagicMock()]
            # 15 words of prose — should be rejected by _looks_like_title
            resp.choices[0].message.content = (
                "This conversation is a long summary about postgres and how the "
                "user tried to fix it eventually."
            )
            return resp

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm), \
             patch("agent.title_generator._retitle_enabled", return_value=True):
            assert regenerate_title("User: pg\nAssistant: ok") is None

    def test_returns_none_on_call_llm_exception(self):
        def mock_call_llm(**kwargs):
            raise RuntimeError("upstream 500")

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm), \
             patch("agent.title_generator._retitle_enabled", return_value=True):
            assert regenerate_title("User: hi\nAssistant: hey") is None

    def test_respects_pinned_language(self):
        captured = {}

        def mock_call_llm(**kwargs):
            captured.update(kwargs)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = '{"title": "ポストグレス プール修正"}'
            return resp

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm), \
             patch("agent.title_generator._retitle_enabled", return_value=True), \
             patch("agent.title_generator._title_language", return_value="Japanese"):
            regenerate_title("User: pg\nAssistant: hai")

        system_prompt = captured["messages"][0]["content"]
        assert "Japanese" in system_prompt

    def test_truncates_condensed_to_max_title_input_chars(self):
        captured = {}

        def mock_call_llm(**kwargs):
            captured.update(kwargs)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = '{"title": "Long convo title"}'
            return resp

        big = "x" * 5000
        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm), \
             patch("agent.title_generator._retitle_enabled", return_value=True):
            regenerate_title(big)

        user_content = captured["messages"][1]["content"]
        assert len(user_content) == MAX_TITLE_INPUT_CHARS


class TestRetitleSession:
    """Unit tests for retitle_session() — the sync worker."""

    def test_returns_none_when_session_db_is_none(self):
        assert retitle_session(None, "sess-1", []) is None

    def test_returns_none_when_session_id_is_empty(self):
        db = MagicMock()
        assert retitle_session(db, "", []) is None
        assert retitle_session(db, None, []) is None

    def test_skips_when_user_title_and_not_forced(self):
        db = MagicMock()
        db.get_session_title_source.return_value = "user"
        with patch("agent.title_generator._persist_session_title") as mock_persist, \
             patch("agent.title_generator.regenerate_title") as mock_regen:
            result = retitle_session(db, "sess-1", [{"role": "user", "content": "hi"}])
        assert result is None
        mock_persist.assert_not_called()
        mock_regen.assert_not_called()

    def test_proceeds_when_user_title_and_force_is_true(self):
        db = MagicMock()
        db.get_session_title_source.return_value = "user"
        db.get_conversation_root.return_value = "sess-1"
        with patch("agent.title_generator._condense_history", return_value="User: hi\nAssistant: hello"), \
             patch("agent.title_generator.regenerate_title", return_value="Cool title") as mock_regen, \
             patch("agent.title_generator._persist_session_title", return_value="Cool title") as mock_persist, \
             patch("agent.aux_accounting.set_accounting_context"), \
             patch("agent.portal_tags.set_conversation_context"):
            result = retitle_session(db, "sess-1", [{"role": "user", "content": "hi"}], force=True)
        assert result == "Cool title"
        mock_regen.assert_called_once()
        mock_persist.assert_called_once()
        # Persist called with source="llm"
        _, kwargs = mock_persist.call_args
        assert kwargs.get("source") == "llm"

    def test_condenses_history_and_persists_llm_title(self):
        db = MagicMock()
        db.get_session_title_source.return_value = "llm"
        db.get_conversation_root.return_value = "sess-1"
        history = [
            {"role": "user", "content": "postgres pool exhausted"},
            {"role": "assistant", "content": "let's check settings"},
        ]
        with patch("agent.title_generator._condense_history", return_value="User: postgres\nAssistant: check") as mock_cond, \
             patch("agent.title_generator.regenerate_title", return_value="Debugging Postgres pool") as mock_regen, \
             patch("agent.title_generator._persist_session_title", return_value="Debugging Postgres pool") as mock_persist, \
             patch("agent.aux_accounting.set_accounting_context"), \
             patch("agent.portal_tags.set_conversation_context"):
            result = retitle_session(db, "sess-1", history, turns_window=7)

        assert result == "Debugging Postgres pool"
        mock_cond.assert_called_once_with(history, turns_window=7)
        mock_regen.assert_called_once_with("User: postgres\nAssistant: check")
        args, kwargs = mock_persist.call_args
        assert args[0] is db
        assert args[1] == "sess-1"
        assert args[2] == "Debugging Postgres pool"
        assert kwargs["source"] == "llm"

    def test_returns_none_when_condensed_empty(self):
        db = MagicMock()
        db.get_session_title_source.return_value = "llm"
        db.get_conversation_root.return_value = "sess-1"
        with patch("agent.title_generator._condense_history", return_value=""), \
             patch("agent.title_generator.regenerate_title") as mock_regen, \
             patch("agent.title_generator._persist_session_title") as mock_persist, \
             patch("agent.aux_accounting.set_accounting_context"), \
             patch("agent.portal_tags.set_conversation_context"):
            result = retitle_session(db, "sess-1", [])
        assert result is None
        mock_regen.assert_not_called()
        mock_persist.assert_not_called()

    def test_returns_none_when_regenerate_returns_none(self):
        db = MagicMock()
        db.get_session_title_source.return_value = "llm"
        db.get_conversation_root.return_value = "sess-1"
        with patch("agent.title_generator._condense_history", return_value="User: hi"), \
             patch("agent.title_generator.regenerate_title", return_value=None), \
             patch("agent.title_generator._persist_session_title") as mock_persist, \
             patch("agent.aux_accounting.set_accounting_context"), \
             patch("agent.portal_tags.set_conversation_context"):
            result = retitle_session(db, "sess-1", [{"role": "user", "content": "hi"}])
        assert result is None
        mock_persist.assert_not_called()

    def test_returns_none_when_persist_returns_none(self):
        db = MagicMock()
        db.get_session_title_source.return_value = "llm"
        db.get_conversation_root.return_value = "sess-1"
        with patch("agent.title_generator._condense_history", return_value="User: hi"), \
             patch("agent.title_generator.regenerate_title", return_value="Some title"), \
             patch("agent.title_generator._persist_session_title", return_value=None), \
             patch("agent.aux_accounting.set_accounting_context"), \
             patch("agent.portal_tags.set_conversation_context"):
            result = retitle_session(db, "sess-1", [{"role": "user", "content": "hi"}])
        assert result is None

    def test_never_raises_on_internal_exception(self):
        db = MagicMock()
        db.get_session_title_source.return_value = "llm"
        db.get_conversation_root.return_value = "sess-1"
        with patch("agent.title_generator._condense_history", side_effect=RuntimeError("boom")), \
             patch("agent.aux_accounting.set_accounting_context"), \
             patch("agent.portal_tags.set_conversation_context"):
            # Must not raise
            result = retitle_session(db, "sess-1", [{"role": "user", "content": "hi"}])
        assert result is None

    def test_sets_accounting_and_conversation_context(self):
        db = MagicMock()
        db.get_session_title_source.return_value = "llm"
        db.get_conversation_root.return_value = "root-sess"
        with patch("agent.title_generator._condense_history", return_value=""), \
             patch("agent.aux_accounting.set_accounting_context") as mock_acc, \
             patch("agent.portal_tags.set_conversation_context") as mock_conv:
            retitle_session(db, "sess-1", [])
        mock_conv.assert_called_once_with("root-sess")
        mock_acc.assert_called_once_with(db, "sess-1")

    def test_falls_back_to_session_id_when_conversation_root_raises(self):
        db = MagicMock()
        db.get_session_title_source.return_value = "llm"
        db.get_conversation_root.side_effect = RuntimeError("db locked")
        with patch("agent.title_generator._condense_history", return_value=""), \
             patch("agent.aux_accounting.set_accounting_context"), \
             patch("agent.portal_tags.set_conversation_context") as mock_conv:
            retitle_session(db, "sess-1", [])
        mock_conv.assert_called_once_with("sess-1")


class TestMaybeAutoRetitle:
    """Tests for maybe_auto_retitle() — one-shot fire-and-forget trigger."""

    def _make_history(self, n_user_turns):
        """Build a history with n real user turns interleaved with assistant."""
        history = []
        for i in range(n_user_turns):
            history.append({"role": "user", "content": f"user message {i} is a real question"})
            history.append({"role": "assistant", "content": f"assistant reply {i}"})
        return history

    def test_returns_early_when_session_db_none(self):
        with patch("agent.title_generator.threading.Thread") as mock_thread:
            maybe_auto_retitle(None, "sess-1", self._make_history(10))
            mock_thread.assert_not_called()

    def test_returns_early_when_session_id_empty(self):
        db = MagicMock()
        with patch("agent.title_generator.threading.Thread") as mock_thread:
            maybe_auto_retitle(db, "", self._make_history(10))
            mock_thread.assert_not_called()

    def test_returns_early_when_disabled_by_config(self):
        db = MagicMock()
        with patch("agent.title_generator._retitle_enabled", return_value=False), \
             patch("agent.title_generator.threading.Thread") as mock_thread:
            maybe_auto_retitle(db, "sess-1", self._make_history(10))
            mock_thread.assert_not_called()

    def test_returns_early_when_auto_at_turn_is_zero(self):
        db = MagicMock()
        with patch("agent.title_generator._retitle_enabled", return_value=True), \
             patch("agent.title_generator.threading.Thread") as mock_thread:
            maybe_auto_retitle(db, "sess-1", self._make_history(10), auto_at_turn=0)
            mock_thread.assert_not_called()

    def test_fires_thread_at_exactly_10_user_turns(self):
        db = MagicMock()
        with patch("agent.title_generator._retitle_enabled", return_value=True), \
             patch("agent.title_generator._retitle_config", return_value={"touch_platform_names": False}), \
             patch("agent.title_generator.threading.Thread") as mock_thread:
            maybe_auto_retitle(db, "sess-1", self._make_history(10), auto_at_turn=10)
            assert mock_thread.call_count == 1
            _, kwargs = mock_thread.call_args
            assert kwargs["target"] is retitle_session
            mock_thread.return_value.start.assert_called_once()

    def test_does_not_fire_at_9_user_turns(self):
        db = MagicMock()
        with patch("agent.title_generator._retitle_enabled", return_value=True), \
             patch("agent.title_generator._retitle_config", return_value={"touch_platform_names": False}), \
             patch("agent.title_generator.threading.Thread") as mock_thread:
            maybe_auto_retitle(db, "sess-1", self._make_history(9), auto_at_turn=10)
            mock_thread.assert_not_called()

    def test_does_not_fire_at_11_user_turns(self):
        # Exactly-N semantics — one-shot trigger, not "N or more".
        db = MagicMock()
        with patch("agent.title_generator._retitle_enabled", return_value=True), \
             patch("agent.title_generator._retitle_config", return_value={"touch_platform_names": False}), \
             patch("agent.title_generator.threading.Thread") as mock_thread:
            maybe_auto_retitle(db, "sess-1", self._make_history(11), auto_at_turn=10)
            mock_thread.assert_not_called()

    def test_thread_is_daemon_with_correct_name(self):
        db = MagicMock()
        with patch("agent.title_generator._retitle_enabled", return_value=True), \
             patch("agent.title_generator._retitle_config", return_value={"touch_platform_names": False}), \
             patch("agent.title_generator.threading.Thread") as mock_thread:
            maybe_auto_retitle(db, "sess-1", self._make_history(10), auto_at_turn=10)
            _, kwargs = mock_thread.call_args
            assert kwargs["daemon"] is True
            assert kwargs["name"] == "auto-retitle"
            # kwargs passed to retitle_session
            call_kwargs = kwargs["kwargs"]
            assert call_kwargs["force"] is False
            assert "turns_window" in call_kwargs
            assert "touch_platform_names" in call_kwargs

    def test_passes_touch_platform_names_from_config(self):
        db = MagicMock()
        with patch("agent.title_generator._retitle_enabled", return_value=True), \
             patch("agent.title_generator._retitle_config", return_value={"touch_platform_names": True}), \
             patch("agent.title_generator.threading.Thread") as mock_thread:
            maybe_auto_retitle(db, "sess-1", self._make_history(10), auto_at_turn=10)
            _, kwargs = mock_thread.call_args
            assert kwargs["kwargs"]["touch_platform_names"] is True

    def test_passes_turns_window_to_thread_kwargs(self):
        db = MagicMock()
        with patch("agent.title_generator._retitle_enabled", return_value=True), \
             patch("agent.title_generator._retitle_config", return_value={"touch_platform_names": False}), \
             patch("agent.title_generator.threading.Thread") as mock_thread:
            maybe_auto_retitle(db, "sess-1", self._make_history(10), auto_at_turn=10, turns_window=15)
            _, kwargs = mock_thread.call_args
            assert kwargs["kwargs"]["turns_window"] == 15
