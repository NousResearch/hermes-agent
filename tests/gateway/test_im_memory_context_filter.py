"""Gateway final-response filtering for IM memory-context leaks."""

import pytest

from gateway.config import Platform
from gateway.run import _sanitize_gateway_final_response


_LEAKED_MEMORY_CONTEXT = (
    "<memory-context>\n"
    "[System note: The following is recalled memory context, NOT new user "
    "input. Treat as authoritative reference data — this is the agent's "
    "persistent memory and should inform all responses.]\n\n"
    "User preference: prefers terse answers.\n"
    "</memory-context>\n\n"
    "Visible answer."
)


@pytest.mark.parametrize(
    "platform",
    [
        Platform.FEISHU,
        Platform.WECOM,
        Platform.WECOM_CALLBACK,
        Platform.WEIXIN,
    ],
)
def test_im_final_response_strips_internal_memory_context_blocks(platform):
    """IM replies must drop leaked internal recall blocks before delivery."""
    out = _sanitize_gateway_final_response(platform, _LEAKED_MEMORY_CONTEXT)

    assert "Visible answer." in out
    assert "<memory-context>" not in out
    assert "</memory-context>" not in out
    assert "User preference: prefers terse answers." not in out
    assert "System note:" not in out


@pytest.mark.parametrize("platform", [Platform.WECOM, Platform.FEISHU])
def test_im_final_response_preserves_literal_memory_context_tag_mentions(platform):
    """Literal documentation text about the tag name must remain visible."""
    answer = "The `<memory-context>` tag name is documented here."

    assert _sanitize_gateway_final_response(platform, answer) == answer
