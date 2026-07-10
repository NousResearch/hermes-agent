"""Regression test for Discord self-delete capability (issue #62157).

A Discord bot must be able to delete its OWN messages from a normal chat
session. The `discord` (core) toolset must expose `delete_own_message`;
it must NOT require the `discord_admin` toolset. Discord allows a bot to
delete its own messages with just the bot token, so this is a real
capability gap, not a Discord limitation.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.discord_tool as discord_tool


def test_delete_own_message_is_registered():
    assert "delete_own_message" in discord_tool._ACTIONS


def test_delete_own_message_is_core_not_admin():
    # Core => exposed on the `discord` toolset a normal chat session uses.
    assert "delete_own_message" in discord_tool._CORE_ACTION_NAMES
    assert "delete_own_message" in discord_tool._CORE_ACTIONS
    # It must NOT require the admin toolset.
    assert "delete_own_message" not in discord_tool._ADMIN_ACTIONS


def test_delete_own_message_required_params():
    assert discord_tool._REQUIRED_PARAMS.get("delete_own_message") == [
        "channel_id",
        "message_id",
    ]


def test_delete_own_message_delegates_to_delete():
    # Same REST call (DELETE /channels/{cid}/messages/{mid}) as delete_message;
    # it is the classification (core vs admin) that was the gap.
    assert (
        discord_tool._ACTIONS["delete_own_message"]
        is not discord_tool._ACTIONS["delete_message"]
    )
