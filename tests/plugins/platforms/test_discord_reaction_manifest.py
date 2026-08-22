import json

from plugins.platforms.discord.reaction_manifest import (
    REACTION_ACTIONS_MARKER,
    choose_reaction_message_index,
    extract_reaction_manifest,
    manifest_actions,
    manifest_discord_messages,
)


def test_extract_reaction_manifest_absent_returns_original_text():
    text, manifest = extract_reaction_manifest("plain message")

    assert text == "plain message"
    assert manifest is None


def test_extract_reaction_manifest_invalid_json_fails_open():
    original = f"hello {REACTION_ACTIONS_MARKER} {{not json"

    text, manifest = extract_reaction_manifest(original)

    assert text == original
    assert manifest is None


def test_extract_reaction_manifest_actions_strips_valid_marker():
    payload = {"actions": [{"emoji": "\u2705", "action": "dismiss"}]}

    text, manifest = extract_reaction_manifest(
        "Dismiss shortcuts\n"
        f"{REACTION_ACTIONS_MARKER} {json.dumps(payload)}"
    )

    assert text == "Dismiss shortcuts"
    assert manifest_actions(manifest) == payload["actions"]


def test_extract_reaction_manifest_discord_messages_without_top_level_actions():
    payload = {
        "discord_messages": [
            {"content": "Search queries", "actions": [{"emoji": "\U0001f50e"}]},
            {"text": "Dismiss shortcuts", "actions": [{"emoji": "\u2705"}]},
        ]
    }

    text, manifest = extract_reaction_manifest(
        f"fallback {REACTION_ACTIONS_MARKER} {json.dumps(payload)}"
    )

    assert text == "fallback"
    assert manifest_discord_messages(manifest) == [
        {"content": "Search queries", "actions": [{"emoji": "\U0001f50e"}]},
        {"content": "Dismiss shortcuts", "actions": [{"emoji": "\u2705"}]},
    ]


def test_choose_reaction_message_index_prefers_action_anchor():
    sent = [
        {"content": "Search queries:\n1. find receipts", "message_id": "1"},
        {"content": "Dismiss shortcuts:\n2. archive", "message_id": "2"},
    ]

    assert choose_reaction_message_index({"emoji": "\u2705", "anchor": "Dismiss shortcuts:"}, sent) == 1
    assert choose_reaction_message_index({"emoji": "\U0001f50e", "action": "search"}, sent) == 0
