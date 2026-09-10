"""Typed Telegram slash-command normalization characterization."""

from types import SimpleNamespace

import pytest

from hermes_cli.telegram_command_normalization import (
    TelegramCommandAttemptStatus,
    normalize_telegram_command_attempt,
)
from hermes_cli.telegram_command_projection import (
    TelegramMenuOmissionReason,
    build_telegram_command_projection,
)


def _command(name: str, description: str | None = None, **overrides):
    values = {
        "name": name,
        "description": description or f"Run {name}",
        "aliases": (),
        "command_id": None,
        "visibility": None,
        "hidden": False,
        "debug": False,
        "available": True,
        "unsupported_surfaces": (),
        "supported_surfaces": (),
        "cli_only": False,
        "gateway_only": False,
        "presentation_overrides": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)

def test_command_addressed_to_current_bot_equals_unaddressed_command():
    projection = build_telegram_command_projection(
        [_command("status", command_id="session.status")]
    )

    plain = normalize_telegram_command_attempt(
        "/status Mixed CASE --Flag", projection, bot_username="HermesBot"
    )
    addressed = normalize_telegram_command_attempt(
        "/status@hermesbot Mixed CASE --Flag",
        projection,
        bot_username="@HermesBot",
    )

    assert plain.status is TelegramCommandAttemptStatus.KNOWN_COMMAND
    assert addressed.status is TelegramCommandAttemptStatus.KNOWN_COMMAND
    assert (
        addressed.command_id,
        addressed.canonical_name,
        addressed.raw_arguments,
        addressed.canonical_input,
    ) == (
        plain.command_id,
        plain.canonical_name,
        plain.raw_arguments,
        plain.canonical_input,
    )


def test_foreign_or_unprovable_bot_target_fails_closed():
    projection = build_telegram_command_projection([_command("status")])

    foreign = normalize_telegram_command_attempt(
        "/status@OtherBot", projection, bot_username="HermesBot"
    )
    unknown_owner = normalize_telegram_command_attempt(
        "/status@HermesBot", projection
    )

    assert foreign.status is TelegramCommandAttemptStatus.NOT_FOR_THIS_BOT
    assert unknown_owner.status is TelegramCommandAttemptStatus.NOT_FOR_THIS_BOT
    assert foreign.command_id is None
    assert unknown_owner.command_id is None


def test_unknown_slash_attempt_never_becomes_ordinary_text():
    projection = build_telegram_command_projection([_command("status")])

    attempt = normalize_telegram_command_attempt(
        "/definitely_unknown@HermesBot payload",
        projection,
        bot_username="hermesbot",
    )

    assert attempt.status is TelegramCommandAttemptStatus.UNKNOWN_COMMAND
    assert attempt.canonical_input == "/definitely_unknown payload"
    assert attempt.command_id is None


def test_ordinary_text_paths_and_code_are_not_commands():
    projection = build_telegram_command_projection([_command("status")])

    for text in ("hello", "/usr/local/bin/hermes", "// comment", "/module.py"):
        assert (
            normalize_telegram_command_attempt(text, projection).status
            is TelegramCommandAttemptStatus.NOT_COMMAND
        )
    assert (
        normalize_telegram_command_attempt("/", projection).status
        is TelegramCommandAttemptStatus.INVALID_COMMAND
    )


def test_duplicate_stable_identity_fails_closed():
    with pytest.raises(ValueError, match="duplicate Telegram command identity"):
        build_telegram_command_projection(
            [
                _command("one", command_id="session.same"),
                _command("two", command_id="session.same"),
            ]
        )


def test_alias_or_sanitized_token_collision_fails_closed():
    with pytest.raises(ValueError, match="Telegram token collision"):
        build_telegram_command_projection(
            [
                _command("foo-bar", command_id="one"),
                _command("foo_bar", command_id="two"),
            ]
        )


def test_overlong_native_name_is_omitted_but_remains_typed():
    long_name = "a" * 33
    projection = build_telegram_command_projection([_command(long_name)])

    assert projection.native_payload == ()
    assert projection.omissions[0].reason is TelegramMenuOmissionReason.NATIVE_NAME_INVALID
    assert (
        normalize_telegram_command_attempt(f"/{long_name}", projection).status
        is TelegramCommandAttemptStatus.KNOWN_COMMAND
    )


