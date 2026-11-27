"""A room member answers to its display name, not only its ASCII handle."""

from __future__ import annotations

import pytest

from gateway.hosted_room_discussion import DiscussionMember, resolve_mentions


def _member(member_id: str, handle: str, display_name: str) -> DiscussionMember:
    return DiscussionMember(member_id, member_id, handle, display_name, {"kind": "local", "profile": member_id})


@pytest.fixture()
def roster() -> tuple[DiscussionMember, ...]:
    return (
        _member("lead", "lead", "여름"),
        _member("builder", "builder", "도윤"),
        _member("reviewer", "reviewer", "하람"),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("@여름 이거 봐", ["lead"]),
        ("@도윤 리뷰 부탁", ["builder"]),
        ("@builder 핸들도 그대로", ["builder"]),
        # A name that is not a member's is not a mention: the room-wide default applies.
        ("@없는이름 확인", ["lead", "builder", "reviewer"]),
    ],
)
def test_mentions_resolve_by_display_name(roster, text, expected):
    assert [member.handle for member in resolve_mentions((text,), roster)] == expected


def test_everyone_still_reaches_the_whole_roster(roster):
    assert len(resolve_mentions(("@everyone 다들 봐",), roster)) == len(roster)
