"""The agent-facing projection of an *inbound* Block Kit payload must stay allowlist-filtered.

These cases live outside ``tests/gateway/test_slack.py`` because that file is already over its
repo-wide FILE_LINES cap, and new behaviour belongs in a topical sibling.
"""

from plugins.platforms.slack.block_payload import _serialize_slack_blocks_for_agent


def test_block_kit_dump_bounds_depth_instead_of_leaking_the_raw_blocks():
    """A deeply nested block tree must not reach the ``repr()`` fallback.

    ``repr(inspectable)`` is the *pre-filter* list: it carries the ``url`` field the allowlist
    deliberately strips, so bounding the walk is what keeps the payload filtered.
    """
    secret_url = "https://secret.example/redact-me"
    node: object = "leaf"
    for _ in range(2500):
        node = {"text": node}

    payload = _serialize_slack_blocks_for_agent(
        [{"type": "section", "url": secret_url, "text": node}]
    )

    assert secret_url not in payload


def test_block_kit_dump_keeps_real_depth_intact():
    """Body of the guard above: ordinary nesting still renders its allowlisted text."""
    node: object = "leaf"
    for _ in range(20):
        node = {"text": node}

    payload = _serialize_slack_blocks_for_agent([{"type": "section", "text": node}])

    assert "leaf" in payload


def test_block_kit_dump_still_describes_bot_ui_blocks():
    """Negative case kept next to its siblings: UI-heavy bot blocks are why the dump exists."""
    payload = _serialize_slack_blocks_for_agent(
        [
            {"type": "section", "text": {"type": "mrkdwn", "text": "Deploy failed"}},
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "action_id": "rollback",
                        "text": {"type": "plain_text", "text": "Roll back"},
                    }
                ],
            },
        ]
    )

    assert "Deploy failed" in payload
    assert "rollback" in payload
    assert "Roll back" in payload
