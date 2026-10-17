"""Multimodal text must use the same budgeting policy as plain text."""

from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor, _estimate_msg_budget_tokens
from agent.image_token_cost import image_cost_context
from agent.model_metadata import estimate_messages_tokens_rough, estimate_tokens_rough
from gateway.platforms.api_server import _normalize_multimodal_content


_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)


def _image(kind="image_url"):
    if kind == "image":
        return {"type": kind, "source": {"type": "base64", "media_type": "image/png", "data": _PNG.split(",")[1]}}
    return {"type": kind, "image_url": {"url": _PNG} if kind == "image_url" else _PNG}


@pytest.mark.parametrize("image_price", [1500, 4000])
@pytest.mark.parametrize(
    "texts,image_types,role,sidecar,uses_sidecar",
    [
        pytest.param(["a" * 12000], ["image_url"], "assistant", None, False, id="ascii"),
        pytest.param(["\u4e2d\u6587" * 6000], ["image_url"], "assistant", None, False, id="chinese"),
        pytest.param(["\uac00" * 12000], ["input_image"], "assistant", None, False, id="korean"),
        pytest.param(["\u0416" * 12000], ["image"], "assistant", None, False, id="cyrillic"),
        pytest.param(["a", "bc", "", "de", "f"], ["image_url"], "assistant", None, False, id="short-split"),
        pytest.param(["", ""], ["image_url"], "assistant", None, False, id="empty-text"),
        pytest.param([], ["image_url", "input_image", "image"], "assistant", None, False, id="images-only"),
        pytest.param([], [], "assistant", None, False, id="empty-list"),
        pytest.param(["\u4e2d\u6587"], [], "assistant", None, False, id="text-only-list"),
        pytest.param(["\u4e2d\u6587"], ["image_url"], "user", "wire text", True, id="user-sidecar"),
        pytest.param(["\u4e2d\u6587"], ["image_url"], "assistant", "wire text", True, id="assistant-sidecar"),
        pytest.param(["\u4e2d\u6587"], ["image_url"], "tool", "wire text", False, id="tool-ignores-sidecar"),
        pytest.param(["\u4e2d\u6587"], ["image_url"], "user", "", False, id="empty-sidecar"),
        pytest.param(["\u4e2d\u6587"], ["image_url"], "user", ["wire text"], False, id="nonstring-sidecar"),
    ],
)
def test_content_budget_preserves_text_policy_and_other_charges(
    texts, image_types, role, sidecar, uses_sidecar, image_price
):
    content = [{"type": "text", "text": text} for text in texts]
    content.extend(_image(kind) for kind in image_types)
    message = {"role": role, "content": content, "api_content": sidecar}
    if role == "assistant":
        message.update(
            tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "search", "arguments": '{"q":"text"}'}}],
            reasoning="duplicate thinking",
            reasoning_content="thinking on the wire",
            reasoning_details=[{"type": "reasoning.text", "text": "thinking on the wire"}],
            codex_message_items=[{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}],
        )
    snapshot = deepcopy(message)
    envelope = {**message, "content": "", "api_content": None}
    # Round each text block through the shared estimator, including short/empty blocks.
    content_cost = (
        estimate_tokens_rough(sidecar) if uses_sidecar
        else sum(map(estimate_tokens_rough, texts)) + len(image_types) * image_price
    )
    with image_cost_context(image_price):
        for charge_thinking in (False, True):
            overhead = _estimate_msg_budget_tokens(envelope, charge_stale_thinking=charge_thinking)
            assert _estimate_msg_budget_tokens(message, charge_stale_thinking=charge_thinking) == overhead + content_cost
    assert message == snapshot


@pytest.mark.parametrize(
    "with_image,position,large",
    [(False, "old", True), (True, "old", True), (True, "latest-user", True), (True, "latest-assistant", True), (True, "old", False)],
    ids=["old-plain", "old-image", "latest-user-image", "latest-assistant-image", "small-old-image"],
)
def test_compression_summarizes_over_budget_history_but_preserves_required_tail(with_image, position, large):
    document = "\u4e2d\u6587" * (10000 if large else 100)
    parts = [{"type": "text", "text": document}]
    if with_image:
        parts.append(_image())
    document_role = "assistant" if position == "latest-assistant" else "user"
    document_message = {"role": document_role, "content": _normalize_multimodal_content(parts)}
    assert isinstance(document_message["content"], list if with_image else str)
    latest_question = document_message if position == "latest-user" else {"role": "user", "content": "Current question"}
    latest_answer = document_message if position == "latest-assistant" else {"role": "assistant", "content": "Current answer"}
    # Keep the small-document control above the normal compression trigger too.
    earlier_repeats = 10000 if large else 15000
    history = [
        {"role": "user", "content": "Initial task"},
        {"role": "assistant", "content": "Initial answer"},
        {"role": "user", "content": "Analyze an earlier document"},
        {"role": "assistant", "content": "\u524d\u6587" * earlier_repeats},
        {"role": "user", "content": "Analyze another earlier document"},
        {"role": "assistant", "content": "\u65e7\u7b54" * earlier_repeats},
        {"role": "user", "content": "Continue"},
        {"role": "assistant", "content": "Continue acknowledged"},
        document_message if position == "old" else {"role": "user", "content": "A small old request"},
        latest_answer if position == "latest-assistant" else {"role": "assistant", "content": "The optional document was reviewed"},
        latest_question,
    ]
    if position != "latest-assistant":
        history.append(latest_answer)
    snapshot = deepcopy(history)
    compressor = ContextCompressor("test/offline", config_context_length=65536, quiet_mode=True)
    with image_cost_context(1500):
        before = estimate_messages_tokens_rough(history)
        assert before > compressor.threshold_tokens
        if large:
            assert estimate_tokens_rough(document) > compressor._tail_soft_ceiling(compressor.tail_token_budget)
        with patch.object(compressor, "_generate_summary", return_value="Earlier documents were reviewed.") as summary:
            result = compressor.compress(history, current_tokens=before)
    assert summary.called
    assert compressor.compression_count == 1
    assert history == snapshot
    for required in (latest_question, latest_answer):
        contents = [row.get("content") for row in result if row.get("role") == required["role"]]
        expected = required["content"]
        if isinstance(expected, list):
            # Summary placement may add blocks around the original carried content.
            assert any(
                content[start:start + len(expected)] == expected
                for content in contents if isinstance(content, list)
                for start in range(len(content) - len(expected) + 1)
            )
        else:
            assert expected in contents
    retained = document in json.dumps(result, ensure_ascii=False)
    assert retained == (position != "old" or not large)
    if not retained:
        assert any(document in json.dumps(call.args[0], ensure_ascii=False) for call in summary.call_args_list)
