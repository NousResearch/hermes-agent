"""Retain the newest typed (input_text) ask at a native checkpoint boundary."""

from copy import deepcopy

import pytest

from agent.codex_responses_adapter import _chat_messages_to_responses_input
from agent.model_metadata import estimate_tokens_rough
from agent.native_compaction import (
    RETAINED_USER_MESSAGE_TOKEN_BUDGET,
    prune_pre_checkpoint_items,
)


_CHECKPOINT = {"type": "compaction", "encrypted_content": "synthetic_checkpoint"}


def test_converter_retains_newest_oversized_typed_ask_not_older_ask():
    current = "CURRENT ASK: inspect this input. " + "x" * (RETAINED_USER_MESSAGE_TOKEN_BUDGET * 4)
    messages = [
        {"role": "user", "content": "OLDER ASK: already completed"},
        {"role": "assistant", "content": "Completed the older ask."},
        {"role": "user", "content": current},
        {
            "role": "assistant",
            "content": "Continuing the current ask.",
            "codex_reasoning_items": [_CHECKPOINT],
        },
    ]
    original = deepcopy(messages)

    converted = _chat_messages_to_responses_input(
        messages, current_issuer_kind="codex_backend", native_compaction_eligible=True,
    )
    users = [item for item in converted if item.get("role") == "user"]
    assert all(part["type"] == "input_text" for item in users for part in item["content"])
    texts = ["".join(part["text"] for part in item["content"]) for item in users]

    assert converted[0] == _CHECKPOINT
    assert len(texts) == 1 and texts[0].startswith("CURRENT ASK:")
    assert current.startswith(texts[0]) and texts[0] != current
    assert 0 < estimate_tokens_rough(texts[0]) <= RETAINED_USER_MESSAGE_TOKEN_BUDGET
    assert messages == original


_IMAGE = {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}


def _user(*parts):
    return {"type": "message", "role": "user", "id": "current-user",
            "content": [{**part, "metadata": {"index": index}} for index, part in enumerate(parts)]}


def _text(text):
    return {"type": "input_text", "text": text}


@pytest.mark.parametrize(
    "current,budget,expected",
    [
        (_user(_text("abcd"), _text("efghijkl"), _text("trailing")), 2, ("abcd", "efgh")),
        (_user(_text("你好世界")), 2, ("你好",)),
        (_user(_text("CURRENT" * 20), _IMAGE), 2, None),
        # Content-less item measured only via output_text.
        ({"type": "message", "role": "user", "output_text": "CURRENT" * 20}, 2, None),
    ],
    ids=["multipart-boundary", "cjk-estimator-budget", "oversized-mixed-image", "oversized-output-text-only"],
)
def test_typed_boundary_is_budgeted_and_never_substitutes_older_ask(current, budget, expected):
    older = {"role": "user", "content": "old"}
    items = [older, current, _CHECKPOINT]
    original = deepcopy(items)

    users = [i for i in prune_pre_checkpoint_items(items, retained_user_token_budget=budget)
             if i.get("role") == "user"]

    if expected is None:
        assert users == []  # Never substitute an older ask for an oversized mixed one.
    else:
        assert users == [{**current, "content": [
            {**current["content"][index], "text": text} for index, text in enumerate(expected)
        ]}]
    assert items == original
