"""Tests for agent.title_generator — auto-generated session titles."""

from contextvars import ContextVar

import pytest
from unittest.mock import MagicMock, patch


from agent.title_generator import (
    generate_title,
    choose_topic_icon,
    auto_title_session,
    maybe_auto_title,
    _title_language,
)
from hermes_state import SessionDB


class TestGenerateTitle:
    """Unit tests for generate_title()."""



    def test_default_prompt_matches_user_language(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Some Title"

        with patch("agent.title_generator.call_llm", return_value=mock_response) as llm:
            generate_title("質問です", "回答です")

        system_prompt = llm.call_args.kwargs["messages"][0]["content"]
        assert "same language the user is writing in" in system_prompt
        assert "1-3 words" in system_prompt
        assert "named project" in system_prompt
        assert "Fixing" in system_prompt
        assert "Do not include emoji" in system_prompt

    def test_compact_preferences_and_name_aliases_are_configurable(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Planning Flow"
        config = {
            "auxiliary": {
                "title_generation": {
                    "max_words": 2,
                    "max_characters": 24,
                    "name_aliases": {
                        "project atlas": "ProjectAtlas",
                        "atlas app": "ProjectAtlas",
                    },
                }
            }
        }

        with (
            patch("hermes_cli.config.load_config_readonly", return_value=config),
            patch("agent.title_generator.call_llm", return_value=mock_response) as llm,
        ):
            assert generate_title("Update the ATLAS APP", "I will inspect it") == "ProjectAtlas"

        system_prompt = llm.call_args.kwargs["messages"][0]["content"]
        assert "1-2 words" in system_prompt
        assert "24 characters" in system_prompt
        assert '\"project atlas\": \"ProjectAtlas\"' in system_prompt
        assert '\"atlas app\": \"ProjectAtlas\"' in system_prompt

    def test_name_aliases_match_whole_terms_not_substrings(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Atlassian Migration"
        config = {
            "auxiliary": {
                "title_generation": {
                    "name_aliases": {"atlas": "ProjectAtlas"},
                }
            }
        }

        with (
            patch("hermes_cli.config.load_config_readonly", return_value=config),
            patch("agent.title_generator.call_llm", return_value=mock_response),
        ):
            title = generate_title("Plan the Atlassian migration", "I will inspect it")

        assert title == "Atlassian Migration"

    def test_name_aliases_do_not_match_across_exchange_boundary(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Vajra"
        config = {
            "auxiliary": {
                "title_generation": {
                    "name_aliases": {
                        "atlas app": "ProjectAtlas",
                        "vajra": "Vajra",
                    },
                }
            }
        }

        with (
            patch("hermes_cli.config.load_config_readonly", return_value=config),
            patch("agent.title_generator.call_llm", return_value=mock_response),
        ):
            title = generate_title(
                "Please leave this as atlas",
                "app support for Vajra is available",
            )

        assert title == "Vajra"

    def test_canonical_alias_takes_precedence_over_character_preference(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Atlas"
        config = {
            "auxiliary": {
                "title_generation": {
                    "max_characters": 12,
                    "name_aliases": {"atlas app": "ProjectAtlasLongName"},
                }
            }
        }

        with (
            patch("hermes_cli.config.load_config_readonly", return_value=config),
            patch("agent.title_generator.call_llm", return_value=mock_response),
        ):
            title = generate_title("Open the atlas app", "I will inspect it")

        assert title == "ProjectAtlasLongName"

    def test_name_alias_after_prompt_snippet_is_still_enforced(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Planning Flow"
        config = {
            "auxiliary": {
                "title_generation": {
                    "name_aliases": {"atlas app": "ProjectAtlas"},
                }
            }
        }

        with (
            patch("hermes_cli.config.load_config_readonly", return_value=config),
            patch("agent.title_generator.call_llm", return_value=mock_response),
        ):
            title = generate_title("x" * 550 + " atlas app", "I will inspect it")

        assert title == "ProjectAtlas"

    def test_name_aliases_ignore_hidden_skill_scaffolding(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Title Leak"
        config = {
            "auxiliary": {
                "title_generation": {
                    "name_aliases": {"worktree": "Worktree"},
                }
            }
        }
        expanded_skill_message = (
            "/work scaffolding with hidden instructions about a fresh worktree"
        )

        with (
            patch("hermes_cli.config.load_config_readonly", return_value=config),
            patch("agent.title_generator.call_llm", return_value=mock_response),
            patch(
                "agent.title_generator._summarize_user_message",
                return_value="/work — fix the title leak",
            ),
        ):
            title = generate_title(expanded_skill_message, "I will fix the title leak")

        assert title == "Title Leak"

    def test_configured_character_limit_is_enforced(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A" * 40
        config = {
            "auxiliary": {
                "title_generation": {
                    "max_words": 2,
                    "max_characters": 24,
                }
            }
        }

        with (
            patch("hermes_cli.config.load_config_readonly", return_value=config),
            patch("agent.title_generator.call_llm", return_value=mock_response),
        ):
            title = generate_title("question", "answer")

        assert title is not None
        assert len(title) == 24
        assert title.endswith("...")

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
            assert generate_title("question", "answer") == "Configured Timeout"

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


    def test_truncates_long_titles(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A" * 100

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            title = generate_title("question", "answer")
            assert title is not None
            assert len(title) == 40
            assert title.endswith("...")



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











class TestChooseTopicIcon:
    def test_returns_exact_allowed_emoji(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "🚀"

        with patch("agent.title_generator.call_llm", return_value=mock_response) as llm:
            result = choose_topic_icon(
                "ProjectAtlas",
                "Let's improve the ProjectAtlas planning workflow",
                ["📊", "🚀", "🛠️"],
            )

        assert result == "🚀"
        prompt = llm.call_args.kwargs["messages"][0]["content"]
        assert "ranked" in prompt
        assert "specific, playful visual metaphors" in prompt
        assert "📊" in prompt and "🚀" in prompt and "🛠️" in prompt
        assert llm.call_args.kwargs["temperature"] == 0.7

    def test_extracts_single_allowed_emoji_from_wrapped_response(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Emoji: 💳"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert choose_topic_icon("Finance", "credit card benefits", ["🚀", "💳"]) == "💳"

    def test_matches_allowed_variation_selector_form(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "⚡"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert choose_topic_icon("ProjectBolt", "fast analysis", ["⚡️", "💡"]) == "⚡️"

    def test_prefers_ranked_candidate_that_was_not_used_recently(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "🚀 📊 🧪"

        with patch("agent.title_generator.call_llm", return_value=mock_response) as llm:
            assert choose_topic_icon(
                "Launch",
                "compare metrics",
                ["🚀", "📊", "🧪"],
                recent_emojis=["🚀"],
            ) == "📊"

        prompt = llm.call_args.kwargs["messages"][0]["content"]
        assert "used recently" in prompt
        assert "🚀" in prompt

    def test_excludes_recent_icons_from_large_candidate_pool(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "💻"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert choose_topic_icon(
                "Developer Tools",
                "debug the agent",
                ["💻", "🎨", "🧪", "🔭", "🛠️"],
                recent_emojis=["💻"],
            ) is None

    def test_compound_emoji_wins_over_overlapping_component(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "👮‍♂️ ⚡"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert choose_topic_icon(
                "Safety",
                "police safety workflow",
                ["👮", "👮‍♂️", "♂️", "⚡️"],
            ) == "👮‍♂️"
            assert choose_topic_icon(
                "Safety",
                "police safety workflow",
                ["👮", "👮‍♂️", "♂️", "⚡️"],
                recent_emojis=["👮‍♂️"],
            ) == "⚡️"

    def test_falls_back_to_top_ranked_candidate_when_all_were_recent(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "💻 🤖"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert choose_topic_icon(
                "Developer Tools",
                "debug the agent",
                ["💻", "🤖"],
                recent_emojis=["💻", "🤖"],
            ) == "💻"

    def test_rejects_response_without_an_allowed_candidate(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "🛸"

        with patch("agent.title_generator.call_llm", return_value=mock_response):
            assert choose_topic_icon("Launch", "ship it", ["🚀", "📊"]) is None

    def test_skips_without_allowed_icons(self):
        with patch("agent.title_generator.call_llm") as llm:
            assert choose_topic_icon("Launch", "ship it", []) is None
        llm.assert_not_called()


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

    def test_copies_context_into_background_worker(self):
        """Profile-scoped ContextVars must survive the bare title thread."""
        db = MagicMock()
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        marker = ContextVar("title_profile_marker", default="default")
        token = marker.set("secondary")
        seen = []

        try:
            with patch("agent.title_generator.auto_title_session") as mock_auto:
                import threading

                called = threading.Event()

                def _capture(*_args, **_kwargs):
                    seen.append(marker.get())
                    called.set()

                mock_auto.side_effect = _capture
                maybe_auto_title(db, "sess-1", "hello", "hi there", history)
                assert called.wait(timeout=10), "auto_title thread never ran"
        finally:
            marker.reset(token)

        assert seen == ["secondary"]

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



    def test_manual_title_race_skips_without_callback(self):
        # Atomic predicate fails (manual /title landed while generation was in
        # flight) -> nothing persisted, no callback fired.
        from agent.title_generator import _persist_session_title
        db = MagicMock()
        db.set_auto_title_if_empty.return_value = False
        assert _persist_session_title(db, "sess-1", "Some Title") is None
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
            maybe_auto_title(db, "sess-1", "hello", "hi there", history, runtime_validator=_v)
            assert called.wait(timeout=10), "auto_title thread never ran"
            kwargs = mock_auto.call_args.kwargs
            assert kwargs["runtime_validator"] is _v
