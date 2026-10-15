"""`/new -project` request parsing and the commands that must stay non-destructive.

Issue #138777: opening the project picker must never reset, interrupt or retitle the
conversation, and a malformed flag must never silently become an ordinary reset or a title.
"""

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.project_picker import (
    is_nondestructive_new,
    new_request_for_event,
    parse_new_request,
)
from gateway.session import SessionSource


def _event(text: str, platform: Platform = Platform.TELEGRAM) -> MessageEvent:
    source = SessionSource(
        platform=platform, chat_id="c1", user_id="u1", chat_type="dm"
    )
    return MessageEvent(text=text, source=source)


@pytest.mark.parametrize("args", ["-project", "--project", "  --PROJECT  "])
def test_project_flag_opens_the_picker(args):
    request = parse_new_request(args, strict_flags=True)
    assert request.kind == "project"


@pytest.mark.parametrize("args", ["", "   ", "My release notes", "dash - inside"])
def test_plain_titles_and_blank_stay_ordinary(args):
    request = parse_new_request(args, strict_flags=True)
    assert request.kind == "plain" and request.title == args.strip()


@pytest.mark.parametrize(
    "args",
    [
        "-projects",
        "--project=alpha",
        "-project alpha",
        "alpha --project",
        "--projet",
        "-p",
        "--foo bar",
    ],
)
def test_malformed_or_unknown_flags_are_rejected_on_telegram(args):
    request = parse_new_request(args, strict_flags=True)
    assert request.kind == "invalid"
    assert request.token


def test_unknown_flag_keeps_legacy_title_on_other_platforms():
    # Only the project flag family is intercepted off-Telegram: nothing else changes there.
    assert parse_new_request("--foo bar", strict_flags=False).kind == "plain"
    assert parse_new_request("-project", strict_flags=False).kind == "project"
    assert parse_new_request("-project alpha", strict_flags=False).kind == "invalid"


def test_event_helpers_use_the_platform_to_choose_strictness():
    assert new_request_for_event(_event("/new --foo")).kind == "invalid"
    assert new_request_for_event(_event("/new --foo", Platform.DISCORD)).kind == "plain"
    assert is_nondestructive_new(_event("/new -project"))
    assert is_nondestructive_new(_event("/new --nope"))
    assert not is_nondestructive_new(_event("/new Release notes"))
    assert not is_nondestructive_new(_event("/new"))
    assert not is_nondestructive_new(_event("/model -project"))


def test_ios_smart_dash_is_the_same_flag():
    # iOS auto-corrects ``--`` to an em dash; get_command_args() normalizes it back.
    assert new_request_for_event(_event("/new \u2014project")).kind == "project"
