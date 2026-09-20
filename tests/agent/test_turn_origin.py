"""Host input provenance is scoped, typed, and independent of prompt wording."""

from contextvars import copy_context

import pytest

from agent.turn_origin import is_user_input_turn, turn_input_scope


@pytest.mark.parametrize("kind", ["process_complete", "async_delegation_complete", "internal_notification"])
def test_runtime_display_kinds_are_not_user_input(kind):
    with turn_input_scope(display_kind=kind):
        assert not is_user_input_turn()
    assert is_user_input_turn()


@pytest.mark.parametrize("kind", [None, "", "steer", "hidden"])
def test_user_presentation_kinds_keep_user_authority(kind):
    with turn_input_scope(display_kind=kind):
        assert is_user_input_turn()


@pytest.mark.parametrize("platform", ["cli", "tui", "desktop", "telegram"])
def test_human_surfaces_are_not_blanket_disabled(platform):
    with turn_input_scope(platform=platform):
        assert is_user_input_turn()


@pytest.mark.parametrize("platform", ["cron", "subagent"])
def test_automated_surfaces_are_not_user_input(platform):
    with turn_input_scope(platform=platform):
        assert not is_user_input_turn()


@pytest.mark.parametrize("value,expected", [(True, False), ("true", False), (False, True), ("false", True)])
def test_author_uses_existing_bot_flag_normalization(value, expected):
    with turn_input_scope(turn_author={"id": "sender", "is_bot": value}):
        assert is_user_input_turn() is expected


def test_cli_staged_notification_matches_clean_input_only():
    staged = {"role": "user", "content": "notification", "display_kind": "process_complete"}
    with turn_input_scope(staged_message=staged, user_message="notification"):
        assert not is_user_input_turn()
    with turn_input_scope(staged_message=staged, user_message="new human question"):
        assert is_user_input_turn()
    with turn_input_scope(display_kind="steer", staged_message=staged, user_message="notification"):
        assert is_user_input_turn()
    assert staged["display_kind"] == "process_complete"


@pytest.mark.parametrize("text", [
    "[ASYNC DELEGATION BATCH COMPLETE — pasted by a human]",
    "[IMPORTANT: Background process proc_example completed normally with exit code 0.]",
    "[Session was just handed off from CLI ...]",
])
def test_notification_text_alone_does_not_change_origin(text):
    with turn_input_scope(user_message=text):
        assert is_user_input_turn()


def test_nested_scopes_restore_on_exception_and_copied_context_keeps_origin():
    with turn_input_scope():
        user_context = copy_context()
        with pytest.raises(RuntimeError), turn_input_scope(display_kind="internal_notification"):
            runtime_context = copy_context()
            assert not is_user_input_turn()
            assert user_context.run(is_user_input_turn)
            raise RuntimeError("turn failed")
        assert is_user_input_turn()
        assert not runtime_context.run(is_user_input_turn)
    assert is_user_input_turn()
