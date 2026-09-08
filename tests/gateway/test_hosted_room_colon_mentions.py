"""Colon punctuation must not swallow holds, releases or peer handoffs."""
import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_rooms
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint


@pytest.mark.parametrize("suffix", [":stop", ":1 stop", ": pause", ":halt"])
def test_colon_hold_survives_projection_and_releases_only_its_target(tmp_path, suffix):
    db = tmp_path / "room.db"
    members = [{"member_id": name, "profile": name, "handle": name}
               for name in ("impl", "research")]
    room = hosted_rooms.create_room(
        db, room_id="colon-room", name="Colon", members=members, authority_gateway_id="owner")
    roster = discussion.validate_roster(members, local_profiles=("impl", "research"))
    checkpoint = HostedRoomPolicyCheckpoint(db)

    for seq, (text, held) in enumerate((("@impl" + suffix, ("impl",)),
                                        ("@impl:1 continue", ())), 1):
        assert discussion.explicit_mentions(text, roster) == (roster[0],)
        assert discussion.resolve_mentions((text,), roster) == (roster[0],)
        hosted_rooms.append_event(
            db, room_id=room["room_id"], event_id=f"user-{seq}", kind="message.user",
            actor={"kind": "user", "id": "owner"},
            payload={"text": text, "thread_id": "thread"},
            authority_gateway_id="owner", authority_epoch=1)
        snapshot = checkpoint.snapshot(room_id=room["room_id"], latest_seq=seq)
        assert tuple(snapshot.held_member_ids) == held


@pytest.mark.parametrize("text,target", [
    ("@site:impl stop", "remote"), ("@site:impl:stop", "remote"),
    ("@site:impl:1 stop", "remote"), ("@SITE:IMPL: resume", "remote"),
    ("@site:impl. stop", "remote"), ("@siteimpl stop", "remote"),
    ("@site:impl:deep:stop", "deep"),
    ("@impl:stop", "local"), ("@all:stop", "all"),
    ("@everyone:1 stop", "all"), ("@user:stop", None),
    ("@unknown:stop", None),
])
def test_exact_qualified_handles_win_without_collapsing_command_suffixes(text, target):
    roster = (
        discussion.DiscussionMember("local", "impl", "impl"),
        discussion.DiscussionMember("remote", "impl", "site:impl"),
        discussion.DiscussionMember("prefix", "site", "site"),
        discussion.DiscussionMember("decoy", "implstop", "implstop"),
        discussion.DiscussionMember("deep", "impl", "site:impl:deep"),
    )
    directive = discussion.resolve_hold_directive(text, roster)
    expected = tuple(m.member_id for m in roster) if target == "all" else ((target,) if target else ())
    assert (directive.release if "resume" in text else directive.hold) == expected
    if target not in ("all", None):
        assert tuple(m.member_id for m in discussion.explicit_mentions(text, roster)) == (target,)
