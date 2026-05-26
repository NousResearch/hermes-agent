"""Wire-payload snapshot tests for prompt caching.

Each test asserts the **literal expected output dict** of
`apply_anthropic_cache_control()` for a fixed input. The values were
captured by running the function against the current `main` and locking
them in. Any byte change in the future signals either:

  - An intentional behavior change (update the expected dict)
  - An unintentional regression (fix the code)

These snapshots are the canary for the PR-1 refactor: after the strategy
abstraction lands, the new code path must produce byte-identical output
for the same inputs.
"""

from __future__ import annotations

from agent.prompt_caching import apply_anthropic_cache_control


# ── Fixed inputs ────────────────────────────────────────────────────────

SYSTEM_PLUS_3 = [
    {"role": "system", "content": "You are a helpful agent."},
    {"role": "user", "content": "First question."},
    {"role": "assistant", "content": "First answer."},
    {"role": "user", "content": "Second question."},
]


# ── Snapshots ───────────────────────────────────────────────────────────


def test_snapshot_native_5m_system_plus_three():
    """Native layout, 5m TTL: marker on inner content blocks; system + last 3."""
    result = apply_anthropic_cache_control(
        SYSTEM_PLUS_3, cache_ttl="5m", native_anthropic=True
    )
    assert result == [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful agent.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "First question.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "First answer.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Second question.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]


def test_snapshot_native_1h_system_plus_three():
    """Native layout, 1h TTL: marker dict includes ttl field."""
    result = apply_anthropic_cache_control(
        SYSTEM_PLUS_3, cache_ttl="1h", native_anthropic=True
    )
    marker = {"type": "ephemeral", "ttl": "1h"}
    assert result == [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful agent.", "cache_control": marker}]},
        {"role": "user", "content": [{"type": "text", "text": "First question.", "cache_control": marker}]},
        {"role": "assistant", "content": [{"type": "text", "text": "First answer.", "cache_control": marker}]},
        {"role": "user", "content": [{"type": "text", "text": "Second question.", "cache_control": marker}]},
    ]


def test_snapshot_envelope_5m_system_plus_three():
    """Envelope layout (native_anthropic=False), 5m TTL.

    Behavior matches native for non-tool messages with string content —
    `_apply_cache_marker` wraps string content into a list and puts the
    marker on the inner block regardless of layout. The native/envelope
    distinction only matters for `role==tool` messages and empty/None
    content (see snapshot tests below)."""
    result = apply_anthropic_cache_control(
        SYSTEM_PLUS_3, cache_ttl="5m", native_anthropic=False
    )
    marker = {"type": "ephemeral"}
    assert result == [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful agent.", "cache_control": marker}]},
        {"role": "user", "content": [{"type": "text", "text": "First question.", "cache_control": marker}]},
        {"role": "assistant", "content": [{"type": "text", "text": "First answer.", "cache_control": marker}]},
        {"role": "user", "content": [{"type": "text", "text": "Second question.", "cache_control": marker}]},
    ]


def test_snapshot_envelope_1h_system_plus_three():
    """Envelope layout, 1h TTL."""
    result = apply_anthropic_cache_control(
        SYSTEM_PLUS_3, cache_ttl="1h", native_anthropic=False
    )
    marker = {"type": "ephemeral", "ttl": "1h"}
    assert result == [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful agent.", "cache_control": marker}]},
        {"role": "user", "content": [{"type": "text", "text": "First question.", "cache_control": marker}]},
        {"role": "assistant", "content": [{"type": "text", "text": "First answer.", "cache_control": marker}]},
        {"role": "user", "content": [{"type": "text", "text": "Second question.", "cache_control": marker}]},
    ]


def test_snapshot_native_with_tool_messages():
    """Tool messages get the marker at the envelope level on native layout
    (rather than on inner content). Asserts the layout distinction lands."""
    messages = [
        {"role": "system", "content": "Sys prompt."},
        {"role": "user", "content": "Use the tool."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "tool result"},
    ]
    result = apply_anthropic_cache_control(messages, cache_ttl="5m", native_anthropic=True)
    marker = {"type": "ephemeral"}
    assert result == [
        {"role": "system", "content": [{"type": "text", "text": "Sys prompt.", "cache_control": marker}]},
        {"role": "user", "content": [{"type": "text", "text": "Use the tool.", "cache_control": marker}]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
            # Empty-string content → envelope-level marker (per _apply_cache_marker).
            "cache_control": marker,
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "tool result",
            # native_anthropic=True → tool messages get envelope-level marker.
            "cache_control": marker,
        },
    ]


def test_snapshot_envelope_with_tool_messages_skips_tool_marker():
    """On envelope layout, tool messages are SKIPPED entirely — the function
    moves on to the next message looking for somewhere to place the marker.
    This catches the OpenRouter-vs-native difference at the wire-format level."""
    messages = [
        {"role": "system", "content": "Sys prompt."},
        {"role": "user", "content": "Use the tool."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "tool result"},
    ]
    result = apply_anthropic_cache_control(messages, cache_ttl="5m", native_anthropic=False)
    marker = {"type": "ephemeral"}
    # On envelope layout, tool messages get NO marker. The 4-cap counts the
    # last 3 non-system messages by index, so the tool message is included
    # in that selection but receives no marker (_apply_cache_marker returns
    # early for tool role on envelope layout).
    assert result == [
        {"role": "system", "content": [{"type": "text", "text": "Sys prompt.", "cache_control": marker}]},
        {"role": "user", "content": [{"type": "text", "text": "Use the tool.", "cache_control": marker}]},
        {
            "role": "assistant",
            # Empty-string content → envelope-level marker (per _apply_cache_marker).
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}],
            "cache_control": marker,
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "tool result",
            # NO cache_control — envelope layout skips tool messages.
        },
    ]
