from datetime import UTC, datetime

import pytest

from gateway.becky_loop_summarizer import (
    VisibleMessage,
    chunk_visible_messages,
    extract_visible_messages,
)


def _visible_message(index: int, text: str) -> VisibleMessage:
    return VisibleMessage(
        ref=f"m{index:06d}",
        role="user",
        text=text,
        occurred_at=datetime(2026, 8, 13, tzinfo=UTC),
    )


def test_extract_visible_messages_keeps_only_active_public_turns() -> None:
    """Dropping a transcript filter must never expose internal turns to a model."""
    first_timestamp = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)
    second_timestamp = datetime(2026, 8, 13, 20, 1, tzinfo=UTC)
    transcript = [
        {
            "role": "system",
            "content": "private system prompt",
            "timestamp": first_timestamp,
        },
        {"role": "tool", "content": "tool result", "timestamp": first_timestamp},
        {"role": "cron", "content": "cron instruction", "timestamp": first_timestamp},
        {
            "role": "approval",
            "content": "approval prompt",
            "timestamp": first_timestamp,
        },
        {
            "role": "progress",
            "content": "progress update",
            "timestamp": first_timestamp,
        },
        {
            "role": "delegate",
            "content": "delegate context",
            "timestamp": first_timestamp,
        },
        {
            "role": "assistant",
            "type": "branch-marker",
            "content": "branch marker",
            "timestamp": first_timestamp,
        },
        {
            "role": "user",
            "content": "inactive",
            "active": False,
            "timestamp": first_timestamp,
        },
        {
            "role": "assistant",
            "content": "deleted",
            "deleted": True,
            "timestamp": first_timestamp,
        },
        {"role": "user", "content": 42, "timestamp": first_timestamp},
        {"role": "assistant", "content": "missing timestamp"},
        {"role": "user", "content": "invalid timestamp", "timestamp": "not-a-date"},
        {
            "id": "database-message-1",
            "role": "user",
            "content": "  Tell Becky hidden-handle\nnow.  ",
            "timestamp": first_timestamp,
        },
        {
            "source_id": "telegram-message-2",
            "role": "assistant",
            "content": "  I will do that.  ",
            "timestamp": second_timestamp,
        },
    ]

    visible = extract_visible_messages(transcript, {"hidden-handle"})

    assert visible == [
        VisibleMessage(
            ref="m000001",
            role="user",
            text="Tell Becky [REDACTED] now.",
            occurred_at=first_timestamp,
        ),
        VisibleMessage(
            ref="m000002",
            role="assistant",
            text="I will do that.",
            occurred_at=second_timestamp,
        ),
    ]


def test_extract_visible_messages_omits_rewound_entries() -> None:
    """Rewound transcript rows must not be reintroduced to a model prompt."""
    timestamp = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)

    visible = extract_visible_messages(
        [
            {"role": "user", "content": "keep this", "timestamp": timestamp},
            {
                "role": "assistant",
                "content": "rewound turn",
                "rewound": True,
                "timestamp": timestamp,
            },
            {
                "role": "user",
                "content": "also rewound",
                "is_rewound": True,
                "timestamp": timestamp,
            },
        ],
        set(),
    )

    assert [message.text for message in visible] == ["keep this"]


def test_extract_visible_messages_redacts_hidden_values_in_url_credentials() -> None:
    """Short or percent-encoded hidden values cannot leak through URL syntax."""
    timestamp = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)

    visible = extract_visible_messages(
        [
            {
                "role": "user",
                "content": (
                    "An idyllic note remains. https://id@example.test/path?"
                    "access_token=a%20b&note=idyllic"
                ),
                "timestamp": timestamp,
            }
        ],
        {"id", "a b"},
    )

    assert visible[0].text == (
        "An idyllic note remains. https://[REDACTED]@example.test/path?"
        "access_token=[REDACTED]&note=idyllic"
    )


def test_chunk_visible_messages_accepts_exact_message_limit() -> None:
    """Exactly 2,000 messages remain within the count boundary."""
    messages = [_visible_message(index, "x" * 1_024) for index in range(2_000)]

    chunks = chunk_visible_messages(messages)

    assert [message for chunk in chunks for message in chunk] == messages
    assert all(
        sum(len(message.text.encode("utf-8")) for message in chunk) <= 48 * 1024
        for chunk in chunks
    )


def test_chunk_visible_messages_accepts_exact_two_mebibytes_of_utf8_text() -> None:
    """The byte limit counts UTF-8 bytes, not Python character count."""
    messages = [_visible_message(index, "é" * 1_024) for index in range(1_024)]

    chunks = chunk_visible_messages(messages)

    assert (
        sum(len(message.text.encode("utf-8")) for message in messages)
        == 2 * 1024 * 1024
    )
    assert [message for chunk in chunks for message in chunk] == messages


@pytest.mark.parametrize(
    "messages",
    [
        [_visible_message(index, "x") for index in range(2_001)],
        [_visible_message(1, "x" * ((2 * 1024 * 1024) + 1))],
    ],
)
def test_chunk_visible_messages_rejects_excess_outer_limits(
    messages: list[VisibleMessage],
) -> None:
    """A one-message or one-byte limit overage is never sent to a provider."""
    with pytest.raises(ValueError, match="conversation_too_large"):
        chunk_visible_messages(messages)


def test_chunk_visible_messages_respects_48_kib_boundaries_without_splitting() -> None:
    """A message at the boundary remains whole and starts no partial chunk."""
    messages = [
        _visible_message(1, "a" * (20 * 1024)),
        _visible_message(2, "b" * (28 * 1024)),
        _visible_message(3, "c"),
    ]

    chunks = chunk_visible_messages(messages)

    assert [[message.ref for message in chunk] for chunk in chunks] == [
        ["m000001", "m000002"],
        ["m000003"],
    ]
    assert all(
        sum(len(message.text.encode("utf-8")) for message in chunk) <= 48 * 1024
        for chunk in chunks
    )
