"""Tests for agent.title_generator — auto-generated session titles."""

import pytest
from unittest.mock import MagicMock, patch


from agent.title_generator import (
    MAX_TITLE_INPUT_CHARS,
    build_title_input,
    derive_title,
    generate_title,
    auto_title_session,
    maybe_auto_title,
    wait_for_title_upgrades,
    _title_language,
)
from hermes_state import SessionDB


class TestGenerateTitle:
    """Unit tests for generate_title()."""

    @pytest.mark.parametrize(
        ("instruction", "paste_preview", "expected_parts"),
        [
            ("@file:/tmp/composer-pastes/pasted_content_1.txt", "Quarterly incident analysis for the database migration",
             ["Quarterly incident analysis"]),
            ("Analyze this", "Quarterly incident analysis for the database migration", ["Analyze this", "Quarterly incident analysis"]),
            ("Prepare the deployment follow-up", "Quarterly incident analysis", ["Prepare the deployment follow-up", "Quarterly incident analysis"]),
        ],
    )
    def test_generated_paste_preview_reaches_the_shared_title_input(self, instruction, paste_preview, expected_parts):
        """A Desktop large paste stays an @file attachment for the turn, but its preview informs BOTH title
        paths (derive_title instant + generate_title model input) through the one shared input."""
        title_input = build_title_input(instruction, paste_preview)

        assert all(part in title_input for part in expected_parts)
        assert "@file:" not in title_input
        # Paste-only opener (just the generated ref): the instant title is the paste's topic, not the path.
        lead = paste_preview if instruction.startswith("@file:") else instruction
        assert derive_title(instruction, paste_preview).startswith(lead[:12])

    def test_expanded_paste_ref_footer_does_not_demote_the_preview(self):
        """The titler receives the opener AFTER @-reference expansion: the generated ref carries a
        `--- Context Warnings ---` (or `--- Attached Context ---`) footer, which must not turn a
        paste-only opener into "instruction + trailing preview" (live wire finding on #114984)."""
        ref = "@file:/home/u/.hermes/attachments/pasted_content_2026-09-18_14-09-43-735_d0ee85.txt"
        preview = "Quarterly incident analysis for the database cluster"
        for footer in (f"\n\n--- Context Warnings ---\n- {ref}: path is outside the allowed workspace",
                       "\n\n--- Attached Context ---\n\n### file: pasted_content.txt\n" + preview):
            title_input = build_title_input(ref + footer, preview)

            assert title_input.startswith(preview)
            assert "---" not in title_input and "@file:" not in title_input
            assert derive_title(ref + footer, preview).startswith("Quarterly incident analysis")

    def test_title_input_budget_and_manual_attachments_stay_unread(self):
        title_input = build_title_input("Describe the release plan", "p" * MAX_TITLE_INPUT_CHARS)

        assert len(title_input) == MAX_TITLE_INPUT_CHARS
        assert title_input.startswith("Describe the release plan")
        assert title_input.endswith("p" * 20)
        # No preview => an ordinary manual attachment ref is never read for titling.
        assert build_title_input("Summarize @file:notes.txt", None) == "Summarize @file:notes.txt"




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

    def test_generate_title_disables_reasoning(self):
        """The titling pass must explicitly disable thinking (#91927).

        With the aux default reasoning_effort "" (provider default), Gemini
        bills internal thought tokens against max_tokens=64, the JSON payload
        never lands, and the prose fallback stores the opening fence
        ("```json") as the title. Enforce the module's documented
        thinking-disabled contract at the call site.
        """
        captured_kwargs = {}

        def mock_call_llm(**kwargs):
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = '{"title": "Reasoning Off"}'
            return resp

        with patch("agent.title_generator.call_llm", side_effect=mock_call_llm):
            assert generate_title("question") == "Reasoning Off"

        assert captured_kwargs.get("reasoning_config") == {"enabled": False}

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            # #83903: a token cap cutting the JSON mid-value or right after the fence opener must not
            # persist the fragment; the derived title survives instead.
            ('{"title":"Investigate and fix the login butt', None),
            ("```json", None),
            ('{"title"', None),
            # Legit titles the structural check must keep: emphasized/quoted prose, non-Latin, numeric.
            ("*Fix the login flow*", "*Fix the login flow*"),
            ("修复登录按钮", "修复登录按钮"),
            ("42", "42"),
            ('```json\n{"title": "Fix login button"', "Fix login button"),
            # Bracket/brace-prefixed prose and a literal fence inside a sentence are titles, not
            # truncated JSON — a provider that ignores response_format still gets its title kept.
            ("[WIP] Fix login flow", "[WIP] Fix login flow"),
            ("Fix ``` rendering in chat", "Fix ``` rendering in chat"),
        ],
    )
    def test_truncated_structured_output_never_becomes_the_title(self, content, expected):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert generate_title("login is broken") == expected

    def test_json_in_reasoning_content_is_used_but_reasoning_prose_is_not(self):
        """#82291: glm-5/minimax under json_schema return content='' with the JSON in reasoning_content /
        reasoning. That payload titles the session; chain-of-thought prose never does."""
        def response(content, **reasoning):
            resp = MagicMock(spec=["choices"])
            resp.choices = [MagicMock(spec=["message"])]
            resp.choices[0].message = MagicMock(spec=["content", *reasoning])
            resp.choices[0].message.content = content
            for k, v in reasoning.items():
                setattr(resp.choices[0].message, k, v)
            return resp

        cases = [
            (response("", reasoning_content='{"title": "Check FFmpeg on this machine"}'), "Check FFmpeg on this machine"),
            (response(None, reasoning='{"title": "Check FFmpeg on this machine"}'), "Check FFmpeg on this machine"),
            (response("", reasoning_content="The user wants ffmpeg checked. A short title would be"), None),
        ]
        for resp, expected in cases:
            with patch("agent.title_generator.call_llm", return_value=resp):
                assert generate_title("check ffmpeg") == expected

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

    @pytest.mark.parametrize("echo", [
        "Fix login button on mobile",
        "fix login button on mobile",
        '"Fix login button on mobile"',
        "(Fix login button on mobile)",
        "[Fix login button on mobile]",
        "Postgres connection pool exhaustion",
        "Code changes",
    ])
    def test_rejects_prompt_example_echo(self, echo):
        """A model that parrots one of the prompt's own example titles back
        must be rejected — a canned example says nothing about the session.
        Port of QwenLM/qwen-code#9709 (their #9706 bug class)."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = echo

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert generate_title("help me with something unrelated") is None

    def test_friendly_greeting_example_is_allowed(self):
        """'Friendly greeting' is prescribed output for bare greetings, not an
        echo failure — it must pass the guard."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Friendly greeting"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert generate_title("hey there!") == "Friendly greeting"

    def test_topical_title_resembling_example_passes(self):
        """The guard is exact-match only: a genuinely topical title that merely
        resembles an example must not be rejected."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Fix login button on desktop"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert generate_title("the login button is broken on desktop") == (
                "Fix login button on desktop"
            )



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

    def test_kanban_worker_is_named_after_its_card_without_the_llm_thread(self, tmp_path, monkeypatch):
        """A worker's session takes the board card's title synchronously; no auxiliary model call (#111166)."""
        from hermes_cli import kanban_db, kanban_db_connect

        with kanban_db_connect.connect_closing(board="default") as conn:
            task_id = kanban_db.create_task(conn, title="Fix flaky worker startup", board="default")
            conn.commit()
        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="kanban")

        with patch("agent.title_generator.auto_title_session") as mock_auto:
            maybe_auto_title(db, "sess-1", f"work kanban task {task_id}", [])

        assert db.get_session_title("sess-1") == "Fix flaky worker startup"
        assert db.get_session_title_source("sess-1") == "llm"
        mock_auto.assert_not_called()

    def test_kanban_worker_with_an_overlong_card_title_is_still_named(self, tmp_path, monkeypatch):
        """Cards have no length cap; the store rejects past MAX_TITLE_LENGTH, so the card title is trimmed, not dropped."""
        from hermes_cli import kanban_db, kanban_db_connect

        card = "Investigate why the swap modal intermittently fails to render its confirmation step on mobile Safari after a retry"
        assert len(card) > SessionDB.MAX_TITLE_LENGTH
        with kanban_db_connect.connect_closing(board="default") as conn:
            task_id = kanban_db.create_task(conn, title=card, board="default")
            conn.commit()
        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
        db = SessionDB(tmp_path / "state.db")
        for sid in ("sess-1", "sess-2"):  # a retried card must still get the ``#N`` suffix within the cap
            db.create_session(session_id=sid, source="kanban")
            with patch("agent.title_generator.auto_title_session"):
                maybe_auto_title(db, sid, f"work kanban task {task_id}", [])

        first, second = db.get_session_title("sess-1"), db.get_session_title("sess-2")
        assert first and first.startswith(card[:40]) and first.endswith("…")
        assert second == f"{first} #2"
        assert len(second) <= SessionDB.MAX_TITLE_LENGTH

    def test_kanban_worker_with_unreadable_card_falls_back_to_the_task_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_missing")
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="kanban")

        with patch("agent.title_generator.auto_title_session") as mock_auto:
            maybe_auto_title(db, "sess-1", "work kanban task t_missing", [])

        assert db.get_session_title("sess-1") == "Kanban task t_missing"
        mock_auto.assert_not_called()

    def test_delegated_child_of_a_worker_is_not_named_after_the_card(self, tmp_path, monkeypatch):
        """A delegate_task child inherits ``HERMES_KANBAN_TASK`` but is not the card's session;
        it takes the ordinary title path instead of the parent's card title (#112817)."""
        from agent.delegation_context import delegated_child_context

        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="child-1", source="kanban")

        with delegated_child_context(), patch("agent.title_generator.auto_title_session") as mock_auto:
            maybe_auto_title(db, "child-1", "research the auth flow for the parent", [])

        assert db.get_session_title("child-1") != "Kanban task t_parent"
        mock_auto.assert_called_once()

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

    def test_model_upgrade_disabled_keeps_derived_title_and_spawns_no_thread(self, tmp_path):
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        config = {
            "auxiliary": {"title_generation": {
                "enabled": True, "model_upgrade_enabled": False,
            }}
        }
        with patch("hermes_cli.config.load_config_readonly", return_value=config), \
             patch("agent.memory_provider.spawn_context_thread") as thread, \
             patch("agent.title_generator.call_llm") as call_llm:
            maybe_auto_title(db, "sess-1", "repair startup memory routing", [])
        assert db.get_session_title("sess-1") == "repair startup memory routing"
        assert db.get_session_title_source("sess-1") == "derived"
        thread.assert_not_called()
        call_llm.assert_not_called()
        # The toggle only silences the automatic upgrade: an explicit ``generate_title`` call
        # (``hermes sessions retitle-skills``) still asks the model.
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = '{"title": "Repair startup memory routing"}'
        with patch("hermes_cli.config.load_config_readonly", return_value=config), \
             patch("agent.title_generator.call_llm", return_value=resp):
            assert generate_title("repair startup memory routing") == "Repair startup memory routing"

    def test_enabled_false_still_disables_derived_and_model_titles(self, tmp_path):
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        config = {
            "auxiliary": {"title_generation": {
                "enabled": False, "model_upgrade_enabled": True,
            }}
        }
        with patch("hermes_cli.config.load_config_readonly", return_value=config), \
             patch("agent.memory_provider.spawn_context_thread") as thread, \
             patch("agent.title_generator.call_llm") as call_llm:
            maybe_auto_title(db, "sess-1", "repair startup memory routing", [])
        assert db.get_session_title("sess-1") is None
        thread.assert_not_called()
        call_llm.assert_not_called()

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

    @pytest.mark.parametrize("title, provisional", [
        ("Friendly greeting", True),
        ("'Friendly greeting in chat'", True),
        ("Friendly greeting card design", False),
        ("Friendly greetings and pleasantries", False),
    ])
    def test_only_the_exact_greeting_placeholder_is_provisional(self, title, provisional):
        """A topical title that merely starts with the phrase keeps its ``llm`` rank."""
        from agent.title_generator import _is_provisional_greeting_title
        assert _is_provisional_greeting_title(title) is provisional

    def test_upgrades_a_provisional_greeting_on_a_substantive_second_turn(self, tmp_path):
        """A bare "hi" opener leaves only placeholders (instant slice / the model's greeting title);
        the next real request must still be allowed to name the session."""
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        answers = iter(["Friendly greeting", "Debug scheduler failures"])

        def stub_call_llm(**kwargs):
            resp = MagicMock()
            resp.choices[0].message.content = next(answers)
            resp.choices[0].message.reasoning = None
            return resp

        history = [{"role": "user", "content": "hi how are you"}]
        with patch("agent.title_generator.call_llm", side_effect=stub_call_llm), \
                patch("agent.title_generator._auto_title_enabled", return_value=True), \
                patch("agent.title_generator._model_title_upgrade_enabled", return_value=True):
            maybe_auto_title(db, "sess-1", "hi how are you", history)
            wait_for_title_upgrades(10)
            assert db.get_session_title_source("sess-1") == "derived"
            history += [{"role": "assistant", "content": "Well, thanks."},
                        {"role": "user", "content": "help me debug the scheduler"}]
            maybe_auto_title(db, "sess-1", "help me debug the scheduler", history)
            wait_for_title_upgrades(10)

        assert db.get_session_title("sess-1") == "Debug scheduler failures"
        assert db.get_session_title_source("sess-1") == "llm"

    def test_a_placeholder_title_stops_retrying_after_the_third_turn(self, tmp_path):
        """A derived name gets turns 2-3 to upgrade, not a model call on every later turn."""
        db = SessionDB(tmp_path / "state.db")
        db.create_session(session_id="sess-1", source="cli")
        db.set_auto_title("sess-1", "hi", source="derived")
        history = [{"role": "user", "content": f"turn {n}"} for n in range(3)]
        with patch("agent.title_generator.auto_title_session") as mock_auto, \
                patch("agent.title_generator._auto_title_enabled", return_value=True), \
                patch("agent.title_generator._model_title_upgrade_enabled", return_value=True):
            maybe_auto_title(db, "sess-1", "and now the real question", history)
            wait_for_title_upgrades(10)
            assert mock_auto.call_count == 1  # turn 3: the placeholder still gets a model shot
            history.append({"role": "user", "content": "turn 3"})
            maybe_auto_title(db, "sess-1", "and now the real question", history)
            wait_for_title_upgrades(10)
        assert mock_auto.call_count == 1  # turn 4: capped, no call

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


class TestCliModelSwitchNoteNotTitleable:
    """Regression: the CLI/gateway ``/model`` note must never become the title.

    ``cli._pending_model_switch_note`` (hermes_cli/cli_model_switch_mixin.py,
    mirrored in gateway/slash_commands_model.py) is injected with
    ``role="user"`` for the same reason as the gateway marker above, so titling
    has to recognise it too. Seen in the wild: with the LLM titler failing, a
    session's ``derived`` title became
    "[Note: model was just switched from muse-spark-1.3-contributor to…", which
    is also what the status bar renders in its session-title slot.
    """

    NOTE = (
        "[Note: model was just switched from muse-spark-1.3-contributor to "
        "deepseek-v4-flash via OpenCode Go. Adjust your self-identification "
        "accordingly.]"
    )

    def test_note_is_not_titleable(self):
        from agent.title_generator import is_titleable_user_message

        assert is_titleable_user_message(self.NOTE) is False

    def test_guard_covers_the_one_turn_variant(self):
        """``--once`` adds a sentence mid-note; the prefix must still match."""
        from agent.title_generator import is_titleable_user_message

        one_turn = (
            "[Note: model was just switched from gpt-6-astra to glm-5 via "
            "OpenCode Go. This override applies to the next turn only. "
            "Adjust your self-identification accordingly.]"
        )
        assert is_titleable_user_message(one_turn) is False

    def test_unrelated_note_bracket_text_still_titleable(self):
        """The guard is narrow: real user text starting "[Note:" still titles."""
        from agent.title_generator import is_titleable_user_message

        assert is_titleable_user_message("[Note: mine] why does the poller stall?") is True

    def test_instant_title_skips_note_uses_real_message(self):
        from agent.title_generator import apply_instant_title

        db = MagicMock()
        db.get_session_title_source.return_value = None

        assert apply_instant_title(db, "sess-1", self.NOTE) is None
        assert apply_instant_title(db, "sess-1", "how do i queue a tg group?") == (
            "how do i queue a tg group?"
        )
