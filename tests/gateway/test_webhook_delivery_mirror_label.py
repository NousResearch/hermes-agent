"""The webhook delivery mirror label bounds the delivery and closes it (#153865).

The mirror is a USER-role row (alternation-safe, see ``WebhookAdapter._mirror_delivery``)
and mirror metadata is dropped at the SQLite boundary, so the text label is the model's only
signal about where the delivery came from. When the user replies in the same chat before an
agent turn, ``_merge_consecutive_users`` folds the reply into the mirror row with a bare
``"\\n\\n"``. The label therefore must carry a boundary: the delivery is closed with
``[Webhook delivery end: <route>]``, so the merged turn still shows where the delivery ends
and the reply begins.
"""

from agent.agent_runtime_helpers import _merge_consecutive_users
from gateway.platforms.webhook import _webhook_mirror_message


def test_mirror_label_attributes_and_closes_before_merged_user_reply():
    mirror = _webhook_mirror_message("mail-indi", "Important email: Meeting moved to 3pm")
    first_line = mirror.split("\n", 1)[0]
    assert first_line == "[Webhook delivery: mail-indi]"
    assert mirror.endswith("[Webhook delivery end: mail-indi]")

    merged, repairs = _merge_consecutive_users([
        {"role": "user", "content": mirror},
        {"role": "user", "content": "Check out this reel"},
    ])
    assert repairs == 1 and len(merged) == 1
    content = merged[0]["content"]
    end = content.index("[Webhook delivery end: mail-indi]")
    assert content.index("Important email: Meeting moved to 3pm") < end < content.index("Check out this reel")


def test_mirror_label_keeps_multiline_route_inside_delimiters():
    mirror = _webhook_mirror_message(
        "mail\nindi", "Important email: Meeting moved to 3pm"
    )
    assert mirror.splitlines() == [
        "[Webhook delivery: mail indi]",
        "Important email: Meeting moved to 3pm",
        "[Webhook delivery end: mail indi]",
    ]

    merged, repairs = _merge_consecutive_users([
        {"role": "user", "content": mirror},
        {"role": "user", "content": "Check out this reel"},
    ])
    assert repairs == 1
    assert merged[0]["content"].endswith(
        "[Webhook delivery end: mail indi]\n\nCheck out this reel"
    )


def test_mirror_label_collapses_line_breaks_but_keeps_other_route_whitespace():
    # U+3000 is part of the route name that routing matches on; only line breaks go.
    mirror = _webhook_mirror_message(
        "\n朝の\u3000ルート\n\n   Daily", "Important email: Meeting moved to 3pm"
    )
    opener, _body, closer = mirror.splitlines()
    assert opener == "[Webhook delivery: 朝の\u3000ルート Daily]"
    assert closer == "[Webhook delivery end: 朝の\u3000ルート Daily]"
