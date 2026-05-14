"""Tests for the Discord thread auto-rename feature.

Covers:
  * Title sanitization (quotes, punctuation, length, whitespace).
  * Config gating (env vars, enabled/disabled).
  * "Looks like raw prompt" detection so user-set names are not stomped.
  * Discord API failure fallback (no exception, returns None).
  * Repeat-rename prevention via the ``is_renamed`` callback.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.discord_thread_rename import (
    DEFAULT_MAX_LENGTH,
    DELAY_AFTER_FIRST_ASSISTANT_RESPONSE,
    DISCORD_MAX_THREAD_NAME,
    DiscordAutoRenameConfig,
    load_config_from_env,
    looks_like_raw_prompt,
    maybe_auto_rename_thread,
    sanitize_title,
    strip_discord_mentions,
)


# --------------------------------------------------------------------------
# sanitize_title
# --------------------------------------------------------------------------


class TestSanitizeTitle:
    def test_strips_quotes_and_punctuation(self):
        assert sanitize_title('"Auto-renommage des threads Discord."') == \
            "Auto-renommage des threads Discord"

    def test_strips_title_prefix(self):
        assert sanitize_title("Title: Refactor login flow") == "Refactor login flow"

    def test_collapses_whitespace_and_newlines(self):
        assert sanitize_title("Refactor   the\nlogin\tflow") == "Refactor the login flow"

    def test_returns_empty_for_empty_input(self):
        assert sanitize_title("") == ""
        assert sanitize_title("   ") == ""
        assert sanitize_title("...") == ""

    def test_truncates_to_max_length_on_word_boundary(self):
        title = "This is a long generated title that exceeds the limit by a lot"
        out = sanitize_title(title, max_length=30)
        assert len(out) <= 30
        assert out.endswith("\u2026")
        assert " " in out  # word boundary cut, not mid-word

    def test_clamps_max_length_to_discord_limit(self):
        # Even with a huge config cap, never exceed Discord's 100-char limit.
        long = "x" * 200
        out = sanitize_title(long, max_length=500)
        assert len(out) <= DISCORD_MAX_THREAD_NAME

    def test_handles_curly_quotes(self):
        assert sanitize_title("\u201cFix login bug\u201d") == "Fix login bug"

    def test_returns_short_unchanged(self):
        assert sanitize_title("Hello world") == "Hello world"


# --------------------------------------------------------------------------
# load_config_from_env
# --------------------------------------------------------------------------


class TestLoadConfig:
    def test_defaults_enabled(self):
        cfg = load_config_from_env(env={})
        assert cfg.enabled is True
        assert cfg.max_length == DEFAULT_MAX_LENGTH
        assert cfg.delay == DELAY_AFTER_FIRST_ASSISTANT_RESPONSE

    def test_disabled_via_env(self):
        cfg = load_config_from_env(env={"DISCORD_AUTO_RENAME_THREADS": "false"})
        assert cfg.enabled is False

    def test_max_length_override(self):
        cfg = load_config_from_env(env={"DISCORD_AUTO_RENAME_MAX_LENGTH": "40"})
        assert cfg.max_length == 40

    def test_invalid_max_length_falls_back(self):
        cfg = load_config_from_env(env={"DISCORD_AUTO_RENAME_MAX_LENGTH": "abc"})
        assert cfg.max_length == DEFAULT_MAX_LENGTH

    def test_unknown_delay_falls_back_to_default(self):
        cfg = load_config_from_env(env={"DISCORD_AUTO_RENAME_DELAY": "ater_first_yolo"})
        assert cfg.delay == DELAY_AFTER_FIRST_ASSISTANT_RESPONSE

    def test_effective_max_length_clamps_to_discord_limit(self):
        cfg = DiscordAutoRenameConfig(enabled=True, max_length=500)
        assert cfg.effective_max_length == DISCORD_MAX_THREAD_NAME

    def test_effective_max_length_replaces_zero_with_default(self):
        cfg = DiscordAutoRenameConfig(enabled=True, max_length=0)
        assert cfg.effective_max_length == DEFAULT_MAX_LENGTH


# --------------------------------------------------------------------------
# looks_like_raw_prompt
# --------------------------------------------------------------------------


class TestLooksLikeRawPrompt:
    def test_exact_match(self):
        assert looks_like_raw_prompt("hello world", "hello world") is True

    def test_truncated_with_ellipsis(self):
        prompt = "lorsque l'agent Hermes ouvre un thread dans Discord il le renomme"
        # adapter does prompt[:77] + "..."
        seeded = prompt[:77] + "..."
        assert looks_like_raw_prompt(seeded, prompt) is True

    def test_user_set_name_returns_false(self):
        assert looks_like_raw_prompt("Customer billing thread", "fix the broken login") is False

    def test_empty_inputs(self):
        assert looks_like_raw_prompt("", "anything") is False
        assert looks_like_raw_prompt("anything", "") is False


# --------------------------------------------------------------------------
# maybe_auto_rename_thread
# --------------------------------------------------------------------------


def _fake_thread(name: str, edit: AsyncMock | None = None):
    thread = MagicMock()
    thread.id = 1234567
    thread.name = name
    thread.edit = edit if edit is not None else AsyncMock()
    return thread


@pytest.mark.asyncio
async def test_renames_thread_on_happy_path():
    cfg = DiscordAutoRenameConfig(enabled=True)
    thread = _fake_thread("fix the login bug please")
    on_renamed = MagicMock()

    result = await maybe_auto_rename_thread(
        thread,
        user_message="fix the login bug please",
        assistant_response="Sure, looking into it.",
        config=cfg,
        on_renamed=on_renamed,
        title_generator=lambda u, a: "Fix the broken login flow",
    )

    assert result == "Fix the broken login flow"
    thread.edit.assert_awaited_once()
    kwargs = thread.edit.await_args.kwargs
    assert kwargs["name"] == "Fix the broken login flow"
    assert "reason" in kwargs
    on_renamed.assert_called_once_with("Fix the broken login flow")


@pytest.mark.asyncio
async def test_skips_when_disabled():
    cfg = DiscordAutoRenameConfig(enabled=False)
    thread = _fake_thread("anything")

    result = await maybe_auto_rename_thread(
        thread, "u", "a", cfg,
        title_generator=lambda u, a: "Generated",
    )
    assert result is None
    thread.edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_when_already_renamed():
    cfg = DiscordAutoRenameConfig(enabled=True)
    thread = _fake_thread("fix the login bug please")

    result = await maybe_auto_rename_thread(
        thread, "fix the login bug please", "ok", cfg,
        is_renamed=lambda: True,
        title_generator=lambda u, a: "Generated Title",
    )
    assert result is None
    thread.edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_when_thread_name_no_longer_matches_prompt():
    """User (or another agent) already renamed the thread — leave it alone."""
    cfg = DiscordAutoRenameConfig(enabled=True)
    thread = _fake_thread("Customer billing thread")

    result = await maybe_auto_rename_thread(
        thread,
        user_message="fix the broken login",
        assistant_response="ok",
        config=cfg,
        title_generator=lambda u, a: "Generated Title",
    )
    assert result is None
    thread.edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_does_not_raise_when_discord_edit_fails():
    """Forbidden / HTTPException must be swallowed and logged — never bubble."""
    cfg = DiscordAutoRenameConfig(enabled=True)
    edit = AsyncMock(side_effect=RuntimeError("403 Forbidden: missing perms"))
    thread = _fake_thread("fix the login bug", edit=edit)
    on_renamed = MagicMock()

    result = await maybe_auto_rename_thread(
        thread, "fix the login bug", "ok", cfg,
        on_renamed=on_renamed,
        title_generator=lambda u, a: "Fix Login Bug",
    )

    assert result is None
    edit.assert_awaited_once()
    on_renamed.assert_not_called()


@pytest.mark.asyncio
async def test_skips_when_title_generator_returns_empty():
    cfg = DiscordAutoRenameConfig(enabled=True)
    thread = _fake_thread("fix the login bug")

    result = await maybe_auto_rename_thread(
        thread, "fix the login bug", "ok", cfg,
        title_generator=lambda u, a: "",
    )
    assert result is None
    thread.edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_when_user_or_assistant_message_missing():
    cfg = DiscordAutoRenameConfig(enabled=True)
    thread = _fake_thread("fix the login bug")
    result = await maybe_auto_rename_thread(
        thread, "", "response", cfg,
        title_generator=lambda u, a: "Title",
    )
    assert result is None
    thread.edit.assert_not_awaited()


# --------------------------------------------------------------------------
# strip_discord_mentions  — must mirror DiscordAdapter._auto_create_thread
# so the captured prompt matches the seeded thread name.
# --------------------------------------------------------------------------


class TestStripDiscordMentions:
    def test_strips_user_mention(self):
        assert strip_discord_mentions("hey <@1234> please help") == "hey please help"

    def test_strips_nick_mention(self):
        assert strip_discord_mentions("yo <@!9999>") == "yo"

    def test_strips_role_mention(self):
        assert strip_discord_mentions("ping <@&5550> team") == "ping team"

    def test_strips_channel_mention(self):
        assert strip_discord_mentions("see <#42> for context") == "see for context"

    def test_strips_mixed_mentions(self):
        out = strip_discord_mentions("<@1> talk to <@&2> in <#3> about <@!4> thanks")
        assert out == "talk to in about thanks"

    def test_collapses_whitespace(self):
        assert strip_discord_mentions("foo\n\n  bar\tbaz") == "foo bar baz"

    def test_empty_input(self):
        assert strip_discord_mentions("") == ""
        assert strip_discord_mentions("<@1>") == ""


@pytest.mark.asyncio
async def test_renames_thread_when_prompt_contained_mentions():
    """Regression: a prompt like '<@&5550> fix the bug' is normalized to
    'fix the bug' when seeding the thread name; the captured prompt must
    be normalized the same way so looks_like_raw_prompt matches.
    """
    cfg = DiscordAutoRenameConfig(enabled=True)
    # The thread was auto-created from "fix the bug" (mentions stripped).
    thread = _fake_thread("fix the bug")

    # The captured prompt has been normalized via strip_discord_mentions
    # by the adapter before storage; we feed it the cleaned form here.
    captured = strip_discord_mentions("<@&5550> fix the bug")
    assert captured == "fix the bug"

    result = await maybe_auto_rename_thread(
        thread,
        user_message=captured,
        assistant_response="On it.",
        config=cfg,
        title_generator=lambda u, a: "Fix the broken login",
    )
    assert result == "Fix the broken login"
    thread.edit.assert_awaited_once()


@pytest.mark.asyncio
async def test_title_generation_runs_off_event_loop():
    """generate_title can block on an LLM call — confirm we hand it off
    to a worker thread via asyncio.to_thread instead of running it
    inline on the Discord event loop.
    """
    import threading
    cfg = DiscordAutoRenameConfig(enabled=True)
    thread = _fake_thread("fix the login bug")
    main_thread_id = threading.get_ident()
    seen_thread_ids: list[int] = []

    def slow_blocking_generator(_user, _assistant):
        seen_thread_ids.append(threading.get_ident())
        return "Fix Login Bug"

    result = await maybe_auto_rename_thread(
        thread, "fix the login bug", "ok", cfg,
        title_generator=slow_blocking_generator,
    )

    assert result == "Fix Login Bug"
    assert seen_thread_ids, "title_generator was not called"
    # The generator must have run on a worker thread, not the caller's
    # (event-loop) thread.
    assert seen_thread_ids[0] != main_thread_id


@pytest.mark.asyncio
async def test_marks_renamed_only_after_successful_edit():
    """on_renamed must fire exactly once after a successful edit, not before."""
    cfg = DiscordAutoRenameConfig(enabled=True)

    renamed_state = {"done": False}

    async def fake_edit(name, reason):
        # Simulate edit succeeding.
        return None

    thread = _fake_thread("fix the login bug", edit=AsyncMock(side_effect=fake_edit))

    def is_renamed():
        return renamed_state["done"]

    def on_renamed(_title):
        renamed_state["done"] = True

    out1 = await maybe_auto_rename_thread(
        thread, "fix the login bug", "ok", cfg,
        is_renamed=is_renamed, on_renamed=on_renamed,
        title_generator=lambda u, a: "Fix Login Bug",
    )
    assert out1 == "Fix Login Bug"
    assert renamed_state["done"] is True

    # Second call must short-circuit on is_renamed=True.
    out2 = await maybe_auto_rename_thread(
        thread, "fix the login bug", "ok", cfg,
        is_renamed=is_renamed, on_renamed=on_renamed,
        title_generator=lambda u, a: "Different Title",
    )
    assert out2 is None
    # edit was called only once across both invocations
    assert thread.edit.await_count == 1
