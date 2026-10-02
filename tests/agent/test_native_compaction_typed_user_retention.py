"""Retain the newest plaintext ask on the actual Codex typed-content path.

These synthetic oversized asks require a native checkpoint; they do not model
normal bounded tool receipts or establish a historical session failure.
"""

from copy import deepcopy

import pytest

from agent.codex_responses_adapter import _chat_messages_to_responses_input
from agent.model_metadata import estimate_tokens_rough
from agent.native_compaction import (
    RETAINED_USER_MESSAGE_TOKEN_BUDGET,
    prune_pre_checkpoint_items,
)


_CHECKPOINT = {"type": "compaction", "encrypted_content": "synthetic_checkpoint"}


@pytest.mark.parametrize("issuer", ["codex_backend", None], ids=["typed-codex", "string-control"])
@pytest.mark.parametrize("checkpoint", [True, False], ids=["checkpoint", "no-checkpoint"])
@pytest.mark.parametrize("multipart", [False, True], ids=["single-text", "multipart"])
def test_converter_retains_newest_oversized_ask_not_completed_older_ask(issuer, checkpoint, multipart):
    prefix = "CURRENT ASK: inspect this input. "
    body = "x" * (RETAINED_USER_MESSAGE_TOKEN_BUDGET * 4)
    current = prefix + body
    content = [{"type": "text", "text": prefix}, {"type": "text", "text": body}] if multipart else current
    messages = [
        {"role": "user", "content": "OLDER ASK: already completed"},
        {"role": "assistant", "content": "Completed the older ask."},
        {"role": "user", "content": content},
        {
            "role": "assistant",
            "content": "Continuing the current ask.",
            **({"codex_reasoning_items": [_CHECKPOINT]} if checkpoint else {}),
        },
    ]
    original = deepcopy(messages)

    converted = _chat_messages_to_responses_input(
        messages, current_issuer_kind=issuer, native_compaction_eligible=True,
    )
    users = [item for item in converted if item.get("role") == "user"]
    texts = [
        "".join(part["text"] for part in item["content"])
        if isinstance(item["content"], list) else item["content"]
        for item in users
    ]

    if checkpoint:
        assert converted[0] == _CHECKPOINT
        assert len(users) == 1  # An older completed ask must not become the sole anchor.
        assert texts[0].startswith("CURRENT ASK:")
        assert current.startswith(texts[0]) and texts[0] != current
        assert 0 < estimate_tokens_rough(texts[0]) <= RETAINED_USER_MESSAGE_TOKEN_BUDGET
        assert converted[-1]["role"] == "assistant"
    else:
        assert texts == [messages[0]["content"], current]
    for item, text in zip(users, texts):
        if issuer == "codex_backend" or (multipart and text.startswith("CURRENT ASK:")):
            assert isinstance(item["content"], list)
            assert all(part["type"] == "input_text" for part in item["content"])
        else:
            assert isinstance(item["content"], str)
    assert messages == original


@pytest.mark.parametrize(
    "content,budget,expected",
    [
        ([{"type": "input_text", "text": "abcd"}], 2, "whole"),
        ([{"type": "input_text", "text": "abcdefgh"}], 1, ("abcd",)),
        ([{"type": "input_text", "text": "abcd"},
          {"type": "input_text", "text": "efghijkl"},
          {"type": "input_text", "text": "trailing"}], 2, ("abcd", "efgh")),
        ([{"type": "input_text", "text": "你好世界"}], 2, ("你好",)),
        ([{"type": "input_text", "text": "абвгдежз"}], 2, ("абвг",)),
        ([{"type": "input_text", "text": "🌍🌍🌍"}], 2, ("🌍🌍",)),
        ([{"type": "input_text", "text": "abcd" + " " * 40}], 2, ("abcd    ",)),
        ([{"type": "input_text", "text": "a"},
          {"type": "input_text", "text": "b"},
          {"type": "input_text", "text": "c"}], 2, ("a", "b")),
        ([{"type": "input_text", "text": ""},
          {"type": "input_text", "text": "abcdefgh"}], 2, ("", "abcd")),
        ([], 1, "older"),
        ([{"type": "input_text", "text": ""}], 1, "older"),
        ([{"type": "input_text", "text": None}], 1, "older"),
        ([{"type": "unknown", "value": "not text"}], 1, "older"),
        ([{"type": "unknown", "text": "CURRENT" * 20}], 2, "none"),
        ([{"type": "input_text", "text": "CURRENT" * 20},
          {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}], 2, "none"),
        ([{"type": "input_text", "text": "abcd"},
          {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}], 1, "whole"),
        ([{"type": "input_text", "text": "CURRENT"}], 0, "none"),
    ],
    ids=[
        "fits-whole", "single-boundary", "multipart-boundary", "cjk", "cyrillic", "emoji",
        "whitespace-budget", "part-rounding", "empty-leading-part", "empty-list", "empty-text",
        "malformed-text", "unknown-part", "oversized-unsupported", "oversized-mixed-image",
        "mixed-image-fits", "zero-budget",
    ],
)
def test_typed_retention_is_budgeted_copy_safe_and_conservative(content, budget, expected):
    older = {"role": "user", "content": "old"}
    current = {
        "type": "message", "role": "user", "id": "current-user",
        "metadata": {"source": "synthetic"}, "content": deepcopy(content),
    }
    for index, part in enumerate(current["content"]):
        part["metadata"] = {"index": index, "annotations": ["preserve"]}
    tail = {"role": "assistant", "content": "untouched tail"}
    items = [older, current, _CHECKPOINT, tail]
    original = deepcopy(items)

    pruned = prune_pre_checkpoint_items(items, retained_user_token_budget=budget)

    assert pruned[0] is _CHECKPOINT and pruned[-1] is tail
    users = [item for item in pruned if item.get("role") == "user"]
    if expected == "older":
        assert users == [older]  # Unmeasurable placeholders retain the existing skip policy.
    elif expected == "none":
        assert users == []  # Never substitute an older ask for an oversized unsupported one.
    else:
        assert users and users[-1].get("id") == current["id"]
        if expected == "whole":
            assert users[-1] is current
        else:
            assert users == [{**current, "content": [
                {**current["content"][index], "text": text}
                for index, text in enumerate(expected)
            ]}]
            assert users[0] is not current
            assert sum(max(1, estimate_tokens_rough(part["text"]))
                       for part in users[0]["content"]) <= budget
    assert items == original
