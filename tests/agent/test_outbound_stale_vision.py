"""Send-path eviction of stale vision_analyze / screenshot tool payloads.

Issue #89296: compression only retires older image-bearing tool results when
prune/compress fires, so OpenAI-style screenshots are re-serialized on every
later turn until a 413. ``evict_stale_outbound_tool_images`` is the
unconditional per-call chokepoint.
"""

from __future__ import annotations

from agent.agent_runtime_helpers import sanitize_api_messages
from agent.context_compressor import (
    _IMAGE_EVICTION_BATCH,
    _MAX_KEEP_TOOL_IMAGES,
    _OUTBOUND_IMAGE_LIMIT,
    _tool_content_has_images,
    evict_stale_outbound_tool_images,
)


def _image_tool(i: int, *, blob: str = "A" * 80) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": "vision_analyze",
                        "arguments": f'{{"image_url":"shot{i}.png"}}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": [
                {"type": "text", "text": f"Image attached natively shot {i}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{blob}{i}"},
                },
            ],
        },
    ]


def _history_with_screenshots(n: int) -> list[dict]:
    msgs: list[dict] = [{"role": "user", "content": "look at these"}]
    for i in range(n):
        msgs.extend(_image_tool(i))
    msgs.append({"role": "user", "content": "compare them"})
    return msgs


def _image_bearing_tool_ids(messages: list[dict]) -> list[str]:
    return [
        m["tool_call_id"]
        for m in messages
        if m.get("role") == "tool" and _tool_content_has_images(m.get("content"))
    ]


class TestOutboundStaleVisionEviction:
    def test_sanitize_alone_keeps_every_screenshot(self):
        """The previous send chokepoint does not close #89296 by itself."""
        history = _history_with_screenshots(5)
        sanitized = sanitize_api_messages(history)
        assert _image_bearing_tool_ids(sanitized) == [f"call_{i}" for i in range(5)]

    def test_nothing_is_evicted_below_the_provider_limit(self):
        """The common case must be append-only: no rewrite, so the cached prefix survives."""
        history = _history_with_screenshots(_OUTBOUND_IMAGE_LIMIT)
        outbound = sanitize_api_messages(history)
        assert evict_stale_outbound_tool_images(outbound) == 0
        assert _image_bearing_tool_ids(outbound) == [
            f"call_{i}" for i in range(_OUTBOUND_IMAGE_LIMIT)
        ]

    def test_eviction_retires_a_batch_once_over_the_limit(self):
        n = _OUTBOUND_IMAGE_LIMIT + 1
        history = _history_with_screenshots(n)
        outbound = sanitize_api_messages(history)
        pruned = evict_stale_outbound_tool_images(outbound)
        assert pruned == _IMAGE_EVICTION_BATCH
        kept = _image_bearing_tool_ids(outbound)
        assert kept == [f"call_{i}" for i in range(_IMAGE_EVICTION_BATCH, n)]

        oldest = next(m for m in outbound if m.get("tool_call_id") == "call_0")
        assert isinstance(oldest["content"], list)
        assert not _tool_content_has_images(oldest["content"])
        assert any(
            isinstance(part, dict)
            and part.get("type") == "text"
            and "screenshot removed" in str(part.get("text", ""))
            for part in oldest["content"]
        )

    def test_frontier_holds_between_batch_advances(self):
        """The rewritten set must not move on every new image.

        A frontier that advances one step per image edits an already-cached row each
        turn, so Anthropic re-writes the entire prompt-cache prefix instead of reading
        it — orders of magnitude more expensive than the image tokens reclaimed. Asserted
        over a span wide enough that a per-image frontier cannot pass by coincidence.
        """
        from agent.conversation_loop import _clone_message_for_send

        def surviving(n: int) -> list:
            msgs = [_clone_message_for_send(m) for m in _history_with_screenshots(n)]
            evict_stale_outbound_tool_images(msgs)
            return _image_bearing_tool_ids(msgs)

        # Span three batch windows: a fixed one-batch retire holds the frontier but stops
        # enforcing the limit after the first window, which the count assertion catches.
        span = range(_MAX_KEEP_TOOL_IMAGES + 1, _OUTBOUND_IMAGE_LIMIT + 3 * _IMAGE_EVICTION_BATCH)
        kept = [surviving(n) for n in span]
        assert all(len(k) <= _OUTBOUND_IMAGE_LIMIT for k in kept), [len(k) for k in kept]
        frontier = [k[0] for k in kept]
        moves = sum(a != b for a, b in zip(frontier, frontier[1:]))
        assert moves == 3, (
            f"frontier moved {moves} times over {len(span)} images (frontier={frontier}); "
            "each move rewrites a cached row and restarts the prefix"
        )

    def test_does_not_rewrite_persisted_history(self):
        from agent.conversation_loop import _clone_message_for_send

        n = _OUTBOUND_IMAGE_LIMIT + 1
        history = _history_with_screenshots(n)
        outbound = [_clone_message_for_send(m) for m in history]
        evict_stale_outbound_tool_images(outbound)
        assert _image_bearing_tool_ids(history) == [f"call_{i}" for i in range(n)]
        assert _image_bearing_tool_ids(outbound) == [
            f"call_{i}" for i in range(_IMAGE_EVICTION_BATCH, n)
        ]

    def test_user_uploads_are_not_evicted(self):
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,USERUPLOAD"},
                    },
                ],
            }
        ]
        for i in range(_MAX_KEEP_TOOL_IMAGES + 2):
            history.extend(_image_tool(i))
        outbound = sanitize_api_messages(history)
        evict_stale_outbound_tool_images(outbound)
        user = next(m for m in outbound if m.get("role") == "user")
        assert user["content"][1]["image_url"]["url"].endswith("USERUPLOAD")
