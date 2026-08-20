"""Mention rendering: the agent must be able to tell who was tagged.

WhatsApp puts mentions in the body as opaque numeric ids and carries the
identities out-of-band. The old behaviour stripped only the bot's own mention
and left every other id as a bare number, which inverted the signal the agent
reads: a message addressed to the bot lost its marker, while a message
addressed to a human kept an ambiguous "@<digits>".

Observed in production before this fix: in one group the agent replied to
every message that tagged a third party and stayed silent on the one message
that actually tagged it.

These are behaviour contracts about the *relationship* between the mention
metadata and the rendered text, not snapshots of exact wording.
"""

from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig


# Synthetic ids/names — shaped like real WhatsApp LIDs, tied to no real account.
BOT_LID = "100000000000001"
ALICE_LID = "100000000000002"
BOB_LID = "100000000000003"
GROUP_ID = "120363000000000000@g.us"


def _make_adapter():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._message_handler = AsyncMock()
    adapter._mention_patterns = []
    return adapter


def _msg(body, mentioned_ids=None, mentioned_names=None, **overrides):
    data = {
        "isGroup": True,
        "body": body,
        "chatId": GROUP_ID,
        "senderId": f"{BOB_LID}@lid",
        "senderName": "Bob",
        "mentionedIds": list(mentioned_ids or []),
        "mentionedNames": dict(mentioned_names or {}),
        "botIds": [f"{BOT_LID}@lid", f"{BOT_LID}@s.whatsapp.net"],
        "quotedParticipant": "",
    }
    data.update(overrides)
    return data


def test_tagging_another_person_is_not_rendered_as_self():
    """The regression: Bob tags Alice, the agent must not read it as its own."""
    adapter = _make_adapter()
    data = _msg(
        f"@{ALICE_LID} did you see this?",
        mentioned_ids=[f"{ALICE_LID}@lid"],
        mentioned_names={ALICE_LID: "Alice"},
    )

    rendered = adapter._clean_bot_mention_text(data["body"], data)

    assert "Alice" in rendered
    # The raw id must not survive — that is what the agent misread as "me".
    assert ALICE_LID not in rendered
    assert adapter._MENTION_SELF_LABEL not in rendered


def test_tagging_the_bot_is_rendered_as_self():
    adapter = _make_adapter()
    data = _msg(f"@{BOT_LID} how many levels does it have?",
                mentioned_ids=[f"{BOT_LID}@lid"])

    rendered = adapter._clean_bot_mention_text(data["body"], data)

    assert adapter._MENTION_SELF_LABEL in rendered
    assert BOT_LID not in rendered


def test_bot_and_human_tagged_together_keeps_both_distinguishable():
    """Multi-tag: being tagged alongside someone must not erase either party."""
    adapter = _make_adapter()
    data = _msg(
        f"@{BOT_LID} @{ALICE_LID} what time?",
        mentioned_ids=[f"{BOT_LID}@lid", f"{ALICE_LID}@lid"],
        mentioned_names={ALICE_LID: "Alice"},
    )

    rendered = adapter._clean_bot_mention_text(data["body"], data)

    assert adapter._MENTION_SELF_LABEL in rendered
    assert "Alice" in rendered
    assert ALICE_LID not in rendered
    assert BOT_LID not in rendered


def test_unresolved_mention_is_marked_as_another_person():
    """Name lookup can fail; it must degrade to 'not me', never to silence."""
    adapter = _make_adapter()
    unknown = "100000000000009"
    data = _msg(f"@{unknown} coming?", mentioned_ids=[f"{unknown}@lid"])

    rendered = adapter._clean_bot_mention_text(data["body"], data)

    assert adapter._MENTION_OTHER_SUFFIX.strip() in rendered
    assert adapter._MENTION_SELF_LABEL not in rendered


def test_message_without_mentions_is_unchanged():
    adapter = _make_adapter()
    data = _msg("no mentions in this one")

    assert adapter._clean_bot_mention_text(data["body"], data) == data["body"]


def test_rendering_never_empties_a_mention_only_message():
    """A body that is only a mention must stay non-empty.

    Empty bodies are dropped upstream as 'no content', so a bare '@bot' ping
    would vanish instead of waking the agent.
    """
    adapter = _make_adapter()
    data = _msg(f"@{BOT_LID}", mentioned_ids=[f"{BOT_LID}@lid"])

    rendered = adapter._clean_bot_mention_text(data["body"], data)

    assert rendered.strip()


def test_mention_detection_still_gates_on_bot_identity():
    """The existing ingestion gate must keep working after the rewrite."""
    adapter = _make_adapter()

    tagged_bot = _msg("hi", mentioned_ids=[f"{BOT_LID}@lid"])
    tagged_human = _msg("hi", mentioned_ids=[f"{ALICE_LID}@lid"])

    assert adapter._message_mentions_bot(tagged_bot) is True
    assert adapter._message_mentions_bot(tagged_human) is False


def test_sender_name_resolves_a_self_referential_tag():
    """Fallback path: no group metadata, but the sender tagged themselves."""
    adapter = _make_adapter()
    data = _msg(
        f"@{BOB_LID} that's me",
        mentioned_ids=[f"{BOB_LID}@lid"],
        mentioned_names={},
    )

    rendered = adapter._clean_bot_mention_text(data["body"], data)

    assert "Bob" in rendered
    assert BOB_LID not in rendered
