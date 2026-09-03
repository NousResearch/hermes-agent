"""Tests for agent.title_generator — auto-generated session titles."""

import pytest
from unittest.mock import MagicMock, patch


from agent.title_generator import (
    generate_title,
    auto_title_session,
    maybe_auto_title,
    _title_language,
    _strip_image_envelope,
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






class TestImageEnvelopeStripping:
    """#82339: titling must never be poisoned by the vision description that
    gateway/CLI image enrichment prepends to the user's message."""

    def test_strips_tui_gateway_envelope(self):
        msg = (
            "[The user attached an image:\n"
            "A close-up view of a browser's search or address bar with a list "
            "of autocomplete suggestions.\n"
            "]\n"
            "[You can examine it with vision_analyze using image_url: /tmp/x.png]\n"
            "小狼毫输入法候选词排序问题"
        )
        assert _strip_image_envelope(msg) == "小狼毫输入法候选词排序问题"

    def test_strips_cli_envelope(self):
        msg = (
            "[The user attached an image. Here's what it contains:\n"
            "The desktop app interface showing a sidebar with a session titled "
            "'修复搜索自动补全下拉'.\n"
            "]\n"
            "[If you need a closer look, use vision_analyze with image_url: /tmp/x.png]\n"
            "Rime 候选词排序问题"
        )
        assert _strip_image_envelope(msg) == "Rime 候选词排序问题"

    def test_strips_analysis_failed_variants(self):
        tui_fail = (
            "[The user attached an image but analysis failed.]\n"
            "[You can examine it with vision_analyze using image_url: /tmp/x.png]\n"
            "What is this?"
        )
        assert _strip_image_envelope(tui_fail) == "What is this?"
        cli_fail = (
            "[The user attached an image but it couldn't be analyzed. You can "
            "try examining it with vision_analyze using image_url: /tmp/x.png]\n"
            "What is this?"
        )
        assert _strip_image_envelope(cli_fail) == "What is this?"

    def test_strips_multiple_images(self):
        msg = (
            "[The user attached an image:\nfirst screenshot\n]\n"
            "[You can examine it with vision_analyze using image_url: /tmp/a.png]\n\n"
            "[The user attached an image:\nsecond screenshot\n]\n"
            "[You can examine it with vision_analyze using image_url: /tmp/b.png]\n\n"
            "Fix the login flow"
        )
        assert _strip_image_envelope(msg) == "Fix the login flow"

    def test_strips_envelope_when_description_contains_brackets(self):
        # The vision model echoes code/text that may contain ']' (e.g.
        # ``array[0] = 1``). The envelope must strip to its real closing
        # bracket, not stop at the first ']' inside the description
        # (#82339 follow-up from triage review).
        msg = (
            "[The user attached an image:\n"
            "A screenshot of code: array[0] = 1\n"
            "]\n"
            "[You can examine it with vision_analyze using image_url: /tmp/x.png]\n"
            "Fix the bug"
        )
        assert _strip_image_envelope(msg) == "Fix the bug"

    def test_strips_envelope_with_multiple_bracketed_tokens(self):
        msg = (
            "[The user attached an image:\n"
            "code: a[1], b[2], c[3]\n"
            "]\n"
            "[You can examine it with vision_analyze using image_url: /tmp/x.png]\n"
            "Fix it"
        )
        assert _strip_image_envelope(msg) == "Fix it"

    def test_strips_cli_envelope_when_description_contains_brackets(self):
        msg = (
            "[The user attached an image. Here's what it contains:\n"
            "a struct with fields f[0], f[1]\n"
            "]\n"
            "[If you need a closer look, use vision_analyze with image_url: /tmp/x.png]\n"
            "Rime 候选词排序问题"
        )
        assert _strip_image_envelope(msg) == "Rime 候选词排序问题"

    def test_derive_title_not_poisoned_by_envelope_with_brackets(self):
        from agent.title_generator import derive_title

        enriched = (
            "[The user attached an image:\n"
            "array[0] = 1\n"
            "]\n"
            "[You can examine it with vision_analyze using image_url: /tmp/x.png]\n"
            "Fix the indexing bug"
        )
        assert derive_title(enriched) == "Fix the indexing bug"

    def test_keeps_user_text_that_mentions_images_mid_sentence(self):
        # A user who literally types about an image (not an enrichment envelope
        # at the start of the message) must keep their text.
        msg = "I took a screenshot of [The user attached an image bug] earlier"
        assert _strip_image_envelope(msg) == msg

    def test_keeps_image_ref_directive(self):
        # The persisted @image:<path> form is not descriptive and must survive.
        msg = "@image:/tmp/x.png 修复输入法候选词排序"
        assert _strip_image_envelope(msg) == msg

    def test_derive_title_not_poisoned_by_envelope(self):
        from agent.title_generator import derive_title

        enriched = (
            "[The user attached an image:\n"
            "A close-up view of a browser's search bar.\n"
            "]\n"
            "[You can examine it with vision_analyze using image_url: /tmp/x.png]\n"
            "小狼毫输入法候选词排序问题"
        )
        assert derive_title(enriched) == "小狼毫输入法候选词排序问题"
        assert derive_title("小狼毫输入法候选词排序问题") == "小狼毫输入法候选词排序问题"

    def test_is_titleable_user_message_true_with_envelope(self):
        # The enriched message still carries user intent underneath the
        # envelope: it must remain titleable (the envelope alone is not the
        # reason to skip titling).
        from agent.title_generator import is_titleable_user_message

        enriched = (
            "[The user attached an image:\nsome description\n]\n"
            "[You can examine it with vision_analyze using image_url: /tmp/x.png]\n"
            "小狼毫输入法候选词排序问题"
        )
        assert is_titleable_user_message(enriched) is True

    def test_preserves_bare_envelope_quoted_as_entire_message(self):
        # If the user's *entire* message is an envelope-shaped string
        # (e.g. quoting one in a bug report), stripping eats everything
        # and leaves titling with nothing. The guard returns the original
        # unmodified. Real enrichment messages always have user content
        # below the envelope, so this never fires in practice.
        msg = (
            "[The user attached an image:\n"
            "A screenshot.\n"
            "]\n"
        )
        assert _strip_image_envelope(msg) == msg

    def test_bare_envelope_produces_no_title(self):
        # A message that is *only* an envelope-shaped quote (pasted export,
        # bug report) must not be titled after the machine's own wording —
        # deriving a title from it would name the session "[The user
        # attached an image:" instead of leaving it untitled. Stripping
        # leaves nothing, and the fallback (returning the original) would
        # still title off the envelope's first line, so _summarize_user_message
        # must collapse it to "" and derive_title to None (#82356, Enough1122
        # point 1).
        from agent.title_generator import derive_title, _summarize_user_message

        msg = (
            "[The user attached an image:\n"
            "A screenshot of the login form.\n"
            "]\n"
            "[You can examine it with vision_analyze using image_url: /tmp/x.png]"
        )
        assert _strip_image_envelope(msg) == msg
        assert _summarize_user_message(msg) == ""
        assert derive_title(msg) is None

    def test_keeps_first_person_attached_image_quote(self):
        # A user who opens their message with a first-person "I attached an
        # image ..." quote (not a machine envelope — the envelope always says
        # "the user attached an image") must keep the full message. The looser
        # "attached an image" match would swallow this and drop the subject
        # (#82356 symmetric edge: fixing "leading envelope eaten" must not
        # also eat a genuine first-person statement).
        msg = "[I attached an image to the report] please review the numbers"
        assert _strip_image_envelope(msg) == msg  # not swallowed
        from agent.title_generator import derive_title

        assert derive_title(msg).startswith("[I attached an image")

    def test_strips_envelope_even_with_short_user_request(self):
        # A short user request like "Fix it" after an envelope is a real
        # enrichment scenario — we strip normally. The guard only fires
        # when *nothing* remains, not when the remainder is short.
        msg = (
            "[The user attached an image:\n"
            "A screenshot.\n"
            "]\n"
            "[You can examine it with vision_analyze using image_url: /tmp/x.png]\n"
            "Fix it"
        )
        assert _strip_image_envelope(msg) == "Fix it"

    def test_matches_looser_wording_variants(self):
        # The regex keys on "attached an image" as the semantic signal,
        # not on one exact phrase — so a third call-site wording (or even
        # a capitalisation change) doesn't silently reintroduce the bug.
        # Variant: "User attached an image — see below:"
        msg = (
            "[User attached an image — see below:\n"
            "A screenshot of code: array[0] = 1\n"
            "]\n"
            "[Examine with vision_analyze: /tmp/x.png]\n"
            "Fix the indexing bug"
        )
        assert _strip_image_envelope(msg) == "Fix the indexing bug"

    def test_matches_case_insensitively(self):
        # Case variations shouldn't defeat the strip.
        msg = (
            "[the user attached an image:\n"
            "screenshot\n"
            "]\n"
            "[you can examine it with vision_analyze]\n"
            "修复输入法候选词排序"
        )
        assert _strip_image_envelope(msg) == "修复输入法候选词排序"


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
