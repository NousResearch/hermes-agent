"""Integration coverage for operator cards with the real discord.py types."""

import importlib
import sys

from gateway.operator_cards import OperatorCard


def _load_real_discord_adapter():
    """Replace the gateway test shim with the installed discord.py package."""
    for module_name in tuple(sys.modules):
        if module_name == "discord" or module_name.startswith("discord."):
            sys.modules.pop(module_name)
    sys.modules.pop("plugins.platforms.discord.adapter", None)
    discord_module = importlib.import_module("discord")
    adapter_module = importlib.import_module("plugins.platforms.discord.adapter")
    return discord_module, adapter_module


def test_operator_card_serializes_through_real_discord_embed():
    discord_module, adapter_module = _load_real_discord_adapter()
    card = OperatorCard.from_mapping(
        {
            "kind": "operator_card",
            "version": 1,
            "card_type": "approval",
            "title": "Approve source sync",
            "severity": "needs_review",
            "summary": "A second source is ready to normalize.",
            "fields": [{"label": "Issue", "value": "OE-222"}],
            "actions": [
                {"id": "approve", "label": "Approve", "style": "success"}
            ],
            "links": [
                {
                    "label": "Open issue",
                    "url": "https://linear.app/example/issue/OE-222",
                }
            ],
            "state_ref": "oe-222:approval:1",
        }
    )

    embed = adapter_module._build_operator_card_embed(card)
    serialized = embed.to_dict()

    assert isinstance(embed, discord_module.Embed)
    assert serialized["title"] == "Approve source sync"
    assert serialized["description"] == "A second source is ready to normalize."
    assert serialized["fields"] == [
        {"inline": False, "name": "Issue", "value": "OE-222"},
        {"inline": False, "name": "Actions", "value": "Approve"},
        {
            "inline": False,
            "name": "Links",
            "value": "[Open issue](https://linear.app/example/issue/OE-222)",
        },
    ]
    assert serialized["footer"]["text"] == "Approval · Needs review"
    assert serialized["color"] == 0xF1C40F
