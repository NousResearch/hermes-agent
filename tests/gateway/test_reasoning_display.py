from gateway.run import (
    _extract_gateway_display_reasoning,
    _format_gateway_display_reasoning,
    _prepend_gateway_display_reasoning,
)


def test_extract_display_reasoning_prefers_last_reasoning():
    result = {
        "last_reasoning": "direct reasoning",
        "messages": [
            {"role": "assistant", "reasoning_content": "fallback reasoning"},
        ],
    }

    assert _extract_gateway_display_reasoning(result) == "direct reasoning"


def test_extract_display_reasoning_falls_back_to_same_turn_reasoning_content():
    result = {
        "messages": [
            {"role": "user", "content": "previous"},
            {"role": "assistant", "reasoning_content": "stale reasoning"},
            {"role": "user", "content": "current"},
            {"role": "assistant", "content": "final"},
            {"role": "assistant", "reasoning_content": "current reasoning"},
        ]
    }

    assert _extract_gateway_display_reasoning(result) == "current reasoning"


def test_extract_display_reasoning_does_not_cross_user_boundary():
    result = {
        "messages": [
            {"role": "assistant", "reasoning_content": "stale reasoning"},
            {"role": "user", "content": "current"},
            {"role": "assistant", "content": "final"},
        ]
    }

    assert _extract_gateway_display_reasoning(result) is None


def test_format_reasoning_uses_chinese_prefix_for_feishu_and_compacts():
    reasoning = "\n".join(f"line {i}" for i in range(20))

    rendered = _format_gateway_display_reasoning("feishu", reasoning)

    assert rendered.startswith("🧠 **思考过程**")
    assert "```" in rendered
    assert "_... (5 more lines)_" in rendered


def test_format_reasoning_uses_compact_weixin_block_without_code_fence():
    rendered = _format_gateway_display_reasoning("weixin", "reasoning")

    assert rendered == "🧠 **思考过程**\nreasoning"


def test_prepend_reasoning_marks_changed_and_avoids_duplicates():
    response, changed = _prepend_gateway_display_reasoning(
        "weixin",
        "final answer",
        "reasoning",
    )

    assert changed is True
    assert response.startswith("🧠 **思考过程**")
    assert response.endswith("final answer")

    duplicated, changed_again = _prepend_gateway_display_reasoning(
        "weixin",
        response,
        "reasoning",
    )
    assert changed_again is False
    assert duplicated == response
