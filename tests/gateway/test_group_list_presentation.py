"""Text Group Chat lists remain readable, bounded and command-prefix aware."""

import pytest

from gateway.hosted_room_messaging import MAX_ROOM_CHOICES, format_room_list


def sample_rooms(count):
    return [
        {
            "room_id": f"synthetic-room-{i}", "name": f"Planning {i}",
            "messaging_ref": i, "_room_mode": "remote", "members": [],
        }
        for i in range(1, count + 1)
    ]


@pytest.mark.parametrize("command", ["/group", "!group"])
def test_group_entries_have_blank_lines_and_keep_navigation(command):
    from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin

    rendered = format_room_list(None, rooms=sample_rooms(2), rooms_command=command)
    blocks = rendered.split("\n\n")
    assert "Group Chats" in blocks[0]
    assert "1. Planning 1" in blocks[1] and "2. Planning 2" not in blocks[1]
    assert "2. Planning 2" in blocks[2] and "1. Planning 1" not in blocks[2]
    assert f"`{command} <number>`" in rendered
    assert f"Help: `{command} help`" in rendered
    if command == "!group":
        assert "/group" not in rendered
    whatsapp = WhatsAppBehaviorMixin().format_message(rendered)
    assert len(whatsapp.split("\n\n")) == len(blocks)
    assert "*1. Planning 1*" in whatsapp


def test_group_page_spacing_preserves_limits_and_safe_names():
    rooms = sample_rooms(MAX_ROOM_CHOICES + 1)
    rooms[0]["name"] = "**unsafe**\n@all `!group 9 stop` " + "long " * 200
    rendered = format_room_list(None, rooms=rooms, rooms_command="!group")
    assert "@all" not in rendered and "`!group 9 stop`" not in rendered
    assert len(rendered) < 4096
    assert rendered.count(" · connected · ") == MAX_ROOM_CHOICES
    assert f"{MAX_ROOM_CHOICES + 1}. Planning" not in rendered
    assert "\n\nGo to page 2: `!group list 2`" in rendered
    last = format_room_list(None, rooms=rooms, rooms_command="!group", page=2)
    assert f"{MAX_ROOM_CHOICES + 1}. Planning" in last
    assert "\n\nGo to page 1: `!group list 1`" in last
