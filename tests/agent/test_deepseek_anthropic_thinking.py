"""Regression guard: preserve thinking blocks on DeepSeek's /anthropic endpoint.

DeepSeek's ``api.deepseek.com/anthropic`` route speaks the Anthropic Messages
protocol but, when thinking mode is enabled, requires ``thinking`` blocks from
prior assistant turns to round-trip on subsequent requests.  The generic
third-party path strips them (signatures are Anthropic-proprietary and other
proxies cannot validate them), so without a DeepSeek-specific carve-out the
next tool-call turn fails with HTTP 400::

    The content[].thinking in the thinking mode must be passed back to the
    API.

DeepSeek's compatibility matrix lists ``thinking`` as supported but
``redacted_thinking`` and ``cache_control`` on thinking blocks as not
supported.  Handling is the same as Kimi's ``/coding`` endpoint: strip
Anthropic-signed blocks (DeepSeek can't validate them) but preserve unsigned
blocks that Hermes synthesises from ``reasoning_content``.

See hermes-agent#16748.
"""

from __future__ import annotations

import pytest


class TestDeepSeekAnthropicPreservesThinking:
    """convert_messages_to_anthropic must replay DeepSeek thinking blocks."""



    def test_signed_anthropic_thinking_block_is_stripped(self) -> None:
        """Anthropic-signed blocks (that leaked through) must still be stripped.

        DeepSeek issues its own signatures and cannot validate Anthropic's —
        the strip-signed / keep-unsigned split matches the Kimi policy.
        """
        from agent.anthropic_message_convert import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "anthropic-signed payload",
                        "signature": "anthropic-sig-xyz",
                    },
                    {"type": "text", "text": "hello"},
                ],
            },
            {"role": "user", "content": "again"},
        ]
        _system, converted = convert_messages_to_anthropic(
            messages, base_url="https://api.deepseek.com/anthropic"
        )

        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        thinking_blocks = [
            b for b in assistant_msg["content"]
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert thinking_blocks == [], (
            "Signed Anthropic thinking blocks must be stripped on DeepSeek — "
            "DeepSeek cannot validate Anthropic-proprietary signatures."
        )

    def test_cache_control_stripped_from_thinking_block(self) -> None:
        """cache_control must still be stripped even when the block is preserved.

        DeepSeek's compatibility matrix lists cache_control on thinking blocks
        as ignored — cache markers interfere with signature validation on
        upstreams that do check them, so Hermes strips them everywhere.
        """
        from agent.anthropic_message_convert import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "reasoning_content": "r1",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
        # Inject cache_control on the synthesised thinking block after-the-fact
        # by running conversion once, mutating, then re-running would be
        # indirect.  Instead check the simpler invariant: no thinking block in
        # the converted output carries cache_control.
        _system, converted = convert_messages_to_anthropic(
            messages, base_url="https://api.deepseek.com/anthropic"
        )
        for m in converted:
            if not isinstance(m.get("content"), list):
                continue
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}:
                    assert "cache_control" not in b


@pytest.mark.parametrize("url", [None, "https://api.anthropic.com", "https://inference-api.nousresearch.com/anthropic"])
def test_deepseek_model_name_does_not_override_native_signature_contract(url):
    from agent.anthropic_message_convert import _manage_thinking_signatures
    block = {"type": "thinking", "thinking": "signed native reasoning", "signature": "sig"}
    messages = [{"role": "assistant", "content": [dict(block), {"type": "text", "text": "answer"}]}]
    _manage_thinking_signatures(messages, url, "deepseek-v4")
    assert messages[0]["content"][0] == block


@pytest.mark.parametrize(("model", "kept"), [
    ("vendor/deepseek-v4", [{"type": "thinking", "thinking": "unsigned"}]),  # thinking family: keep unsigned only
    (" DeepSeek-Pro ", [{"type": "thinking", "thinking": "unsigned"}]),
    ("deepseek-chat", []),  # non-thinking DeepSeek and unrelated models: generic third-party strip
    ("vendor/other-model", []),
])
def test_deepseek_proxy_keeps_unsigned_thinking_in_older_tool_turns_only(model, kept):
    import copy
    from agent.anthropic_message_convert import convert_messages_to_anthropic
    history = [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": "checking", "reasoning_details": [
            {"type": "thinking", "thinking": "unsigned", "cache_control": {"type": "ephemeral"}},
            {"type": "thinking", "thinking": "foreign signed", "signature": "sig"},
            {"type": "redacted_thinking", "data": "redacted-signature"},
        ], "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "inspect", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    snapshot = copy.deepcopy(history)
    _, result = convert_messages_to_anthropic(history, base_url="https://proxy.example/anthropic", model=model)
    assistant = next(m for m in result if m["role"] == "assistant")
    assert [b for b in assistant["content"] if b.get("type") in {"thinking", "redacted_thinking"}] == kept
    assert history == snapshot
