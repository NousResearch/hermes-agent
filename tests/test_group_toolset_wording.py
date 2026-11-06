"""The /toolsets display uses the same group-chat name as the rest of the product."""
from hermes_cli.cli_info_lines import toolsets_lines


def test_group_toolset_display_uses_group_chat():
    line, = (line for line in toolsets_lines(["bot_room"]) if "bot_room" in line)
    description = line.split(" - ", 1)[1]
    assert description.startswith("Group chat")
    assert "room" not in description
