from pathlib import Path
from gateway import hosted_room_discussion as discussion, hosted_room_driver as driver
from tests.gateway.test_hosted_room_discussion import (
    LOCAL_PROFILES, room_db as room_db_fixture, _append_user, _next_task, _settle_next, _events,
)


room_db = room_db_fixture

def _fixed_prompt_bytes(room: dict) -> int:
    """The bytes of the first member's prompt with no messages: what every message line adds to."""
    validated = discussion.validate_room(room, local_profiles=LOCAL_PROFILES)
    return len(discussion._build_prompt(
        room=validated, member=validated.members[0], messages=(), watermark=0, seen_through_seq=0, reply_required=True).encode("utf-8"))


def test_omission_notice_only_uses_budget_the_kept_messages_leave(room_db: tuple[Path, dict]):
    db, room = room_db
    fixed = _fixed_prompt_bytes(room)
    line = len(b"  User (user): ") + 1
    newest = discussion.MAX_USER_TEXT_BYTES - 64
    # Leave fewer bytes than the notice after the two newest messages.
    middle = driver.MAX_PROMPT_BYTES - fixed - 1 - (line + newest) - line - 20
    for index, text in enumerate(("o" * 100, "m" * middle, "@research " + "n" * (newest - 10))):
        _append_user(db, event_id=f"user-{index}", text=text)

    task = _next_task(room, db)
    prompt = task.payload["prompt"]
    assert len(prompt.encode("utf-8")) <= driver.MAX_PROMPT_BYTES
    assert "m" * middle in prompt
    assert "Earlier content omitted" not in prompt


def test_a_cut_newest_message_still_says_content_was_omitted(room_db: tuple[Path, dict], monkeypatch):
    """Seam: a smaller driver prompt limit, so the newest message alone overflows it (a member reply
    full of quoted control frames grows past it in production)."""
    db, room = room_db
    fixed = _fixed_prompt_bytes(room)
    monkeypatch.setattr(driver, "MAX_PROMPT_BYTES", fixed + 400)
    _append_user(db, event_id="user-0", text="older")
    _append_user(db, event_id="user-1", text="@research " + "n" * 1000)

    prompt = _next_task(room, db).payload["prompt"]
    assert len(prompt.encode("utf-8")) <= fixed + 400
    assert "n" * 200 in prompt and "older" not in prompt
    assert "Earlier content omitted" in prompt


def test_a_bots_own_replies_never_appear_in_its_next_prompt(room_db: tuple[Path, dict]):
    """The watermark normally covers them; the prompt also drops any that sit past it."""
    db, room = room_db
    _append_user(db, event_id="user-1", text="@build Draft the plan.")
    _settle_next(room, db, text="My own earlier draft.")
    _append_user(db, event_id="user-2", text="@build Now the details.")
    checked = discussion.validate_room(room, local_profiles=LOCAL_PROFILES)
    messages = [event for event in discussion._validated_events(_events(db), room=checked)
                if event.kind in {"message.user", "message.member"}]
    build, review = (next(m for m in checked.members if m.member_id == f"member-{name}")
                     for name in ("build", "review"))
    prompt = discussion._build_prompt(room=checked, member=build, messages=messages, watermark=0,
                                      seen_through_seq=messages[-1].seq)
    assert "Now the details." in prompt and "Draft the plan." in prompt
    assert "My own earlier draft." not in prompt
    assert "My own earlier draft." in discussion._build_prompt(
        room=checked, member=review, messages=messages, watermark=0, seen_through_seq=messages[-1].seq)
