"""Regression guard: preserve thinking blocks on DeepSeek's /anthropic endpoint.

DeepSeek's ``api.deepseek.com/anthropic`` route speaks the Anthropic Messages
protocol but, when thinking mode is enabled, requires ``thinking`` blocks from
prior assistant turns to round-trip on subsequent requests.  The generic
third-party path strips them (signatures are Anthropic-proprietary and other
proxies cannot validate them), so without a DeepSeek-specific carve-out the
next tool-call turn fails with HTTP 400::

    The content[].thinking in the thinking mode must be passed back to the
    API.

DeepSeek signs its OWN thinking blocks — the signature is a UUID, not an
Anthropic-proprietary one — and still requires them replayed.  Measured live
against ``api.deepseek.com/anthropic`` with the same assistant-turn-with-
tool_use shape Hermes sends: thinking block present (signed, unsigned, or with
empty ``thinking`` text) -> 200; block absent -> the 400 above.  Removing the
block because it carried a truthy ``signature`` (the #17510 policy) was
therefore the bug: it turned every multi-step tool loop into a 400 and a
silent provider fallback.

``redacted_thinking`` is the one shape DeepSeek cannot parse (422 "unknown
variant ``redacted_thinking``"), so that variant is still dropped.  Relays
fronting a DeepSeek id keep the old strip-signed / keep-unsigned policy —
they cannot verify Anthropic signatures.

See hermes-agent#47685, #17992, #16748.
"""

from __future__ import annotations

import pytest


class TestDeepSeekAnthropicPreservesThinking:
    """convert_messages_to_anthropic must replay DeepSeek thinking blocks."""


    def test_presigned_deepseek_thinking_block_is_replayed(self) -> None:
        """DeepSeek's own signed blocks must round-trip.

        DeepSeek puts a UUID in ``signature``; the server accepts and requires
        those blocks back (measured: 200 with the block, 400 without it).
        """
        from agent.anthropic_message_convert import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "deepseek-signed payload",
                        "signature": "0188d6a3-ad7b-43df-954b-6c4c60f7699b",
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
        assert len(thinking_blocks) == 1, (
            "DeepSeek's own signed thinking blocks must be replayed — dropping "
            "them makes the next tool-loop call 400 "
            "'content[].thinking ... must be passed back to the API'."
        )
        assert thinking_blocks[0]["thinking"] == "deepseek-signed payload"

    def test_redacted_thinking_is_dropped_for_deepseek(self) -> None:
        """redacted_thinking is the one variant DeepSeek rejects (422)."""
        from agent.anthropic_message_convert import convert_messages_to_anthropic

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "Emx0ZXN0"},
                    {"type": "thinking", "thinking": "kept", "signature": "uuid-1"},
                    {"type": "text", "text": "hello"},
                ],
            },
            {"role": "user", "content": "again"},
        ]
        _system, converted = convert_messages_to_anthropic(
            messages, base_url="https://api.deepseek.com/anthropic"
        )
        assistant_msg = next(m for m in converted if m["role"] == "assistant")
        types = [b.get("type") for b in assistant_msg["content"] if isinstance(b, dict)]
        assert "redacted_thinking" not in types
        assert "thinking" in types

    def test_relay_with_deepseek_model_keeps_strip_signed_policy(self) -> None:
        """A relay cannot verify Anthropic signatures: unchanged #17510 policy."""
        from agent.anthropic_message_convert import _manage_thinking_signatures

        messages = [{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "unsigned"},
                {"type": "thinking", "thinking": "foreign signed", "signature": "sig"},
                {"type": "redacted_thinking", "data": "blob"},
                {"type": "text", "text": "answer"},
            ],
        }]
        _manage_thinking_signatures(messages, "https://proxy.example/anthropic", "vendor/deepseek-v4")
        types = [b.get("type") for b in messages[0]["content"]]
        assert types.count("thinking") == 1
        assert messages[0]["content"][0]["thinking"] == "unsigned"

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
