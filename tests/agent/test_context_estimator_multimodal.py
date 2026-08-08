"""Tests for multimodal-aware context estimation (issue #76411).

Reproduces the base64 inflation: the legacy estimator counts base64 image
bytes as text (``len(str(payload)) // 4``), inflating the estimate ~4x
and shifting watchdog tiers. These tests pin the properties of the
multimodal-aware walker:

- base64 byte size must NOT drive the estimate proportionally;
- text around images is still counted;
- distinct images still contribute (per-image cost);
- data URLs are not ignored entirely — bounded visual cost;
- pure-text payloads keep the exact legacy estimate (backward compat);
- both Chat Completions (``image_url``) and Responses (``input_image``)
  shapes are interpreted;
- the estimate selects the expected watchdog tier.

See contribution-queue/issues/76411-multimodal-context-estimator.md for
the design and the PR-declaration boundary (estimator only; payload
deduplication is a separate follow-up).
"""

from __future__ import annotations

import copy


# ── payload builders ──────────────────────────────────────────────────────


def _b64(chars: str) -> str:
    """A data-URL image payload. Real base64 content is not required for
    the estimator; the walker must not depend on decodable bytes."""
    return "data:image/png;base64," + chars


def _chat_payload(*, text: str = "", images: list[str] | None = None) -> dict:
    """Chat Completions payload with optional multimodal content parts."""
    images = images or []
    content: list = []
    if text:
        content.append({"type": "text", "text": text})
    for url in images:
        content.append({"type": "image_url", "image_url": {"url": url}})
    if len(content) == 1 and content[0]["type"] == "text":
        content = text  # plain string content for pure-text shape
    return {"model": "gpt-5.5", "messages": [{"role": "user", "content": content}]}


def _codex_payload(*, text: str = "", images: list[str] | None = None) -> dict:
    """Responses API payload with input_text / input_image items."""
    images = images or []
    inp: list = []
    if text:
        inp.append({"type": "input_text", "text": text})
    for url in images:
        inp.append({"type": "input_image", "image_url": url})
    return {"model": "gpt-5.5", "input": inp}


# ── core properties ───────────────────────────────────────────────────────


def test_base64_size_does_not_drive_estimate_linear():
    """Growing base64 size must NOT change the estimate at all.

    Same structure, only base64 size differs -> identical estimate. This is
    the reproduction: the legacy estimator returns ~chars/4, so a
    100_000-char data URL estimates ~25k tokens from the image alone.
    """
    from agent.chat_completion_helpers import estimate_request_context_tokens

    small = _chat_payload(text="describe", images=[_b64("a" * 100)])
    large = _chat_payload(text="describe", images=[_b64("a" * 100_000)])
    small_est = estimate_request_context_tokens(small)
    large_est = estimate_request_context_tokens(large)
    # A 1000x byte growth must produce ZERO change: base64 is never text.
    assert large_est == small_est, (
        f"base64 inflation: small={small_est} large={large_est}"
    )


def test_irrelevant_metadata_not_counted_in_multimodal_mode():
    """Multimodal mode must count the same fields the legacy path counted.

    Review finding: the walker must not start counting ``model`` or random
    metadata just because an image is present — that would change the
    textual semantics purely from the image's existence.
    """
    from agent.chat_completion_helpers import estimate_request_context_tokens

    base = _codex_payload(text="describe", images=[_b64("a" * 1000)])
    with_metadata = dict(base)
    with_metadata["irrelevant_metadata"] = "x" * 100_000
    assert estimate_request_context_tokens(with_metadata) == estimate_request_context_tokens(
        base
    )


def test_text_around_image_still_counted():
    from agent.chat_completion_helpers import estimate_request_context_tokens

    img = [_b64("a" * 1000)]
    with_text = _chat_payload(text="describe this in detail please", images=img)
    without_text = _chat_payload(text="", images=img)
    assert estimate_request_context_tokens(with_text) > estimate_request_context_tokens(
        without_text
    )


def test_distinct_images_still_contribute():
    from agent.chat_completion_helpers import estimate_request_context_tokens

    one = _chat_payload(text="describe", images=[_b64("a" * 1000)])
    two = _chat_payload(text="describe", images=[_b64("a" * 1000), _b64("b" * 1000)])
    assert estimate_request_context_tokens(two) > estimate_request_context_tokens(one)


def test_data_url_not_ignored_gets_bounded_cost():
    from agent.chat_completion_helpers import estimate_request_context_tokens

    no_image = _chat_payload(text="describe", images=[])
    with_image = _chat_payload(text="describe", images=[_b64("a" * 1000)])
    # Image contributes a bounded visual cost, so the estimate grows.
    assert estimate_request_context_tokens(with_image) > estimate_request_context_tokens(
        no_image
    )


def test_pure_text_payload_unchanged():
    """Backward compat: a text-only payload yields the exact legacy estimate."""
    from agent.chat_completion_helpers import estimate_request_context_tokens

    text = "hello world this is a test message"
    payload = _chat_payload(text=text)
    # Legacy semantics for a dict with `messages`: sum of str(item) // 4.
    legacy = sum(len(str(item)) for item in payload["messages"]) // 4
    assert estimate_request_context_tokens(payload) == legacy


def test_short_text_fragments_match_single_text_block():
    """Short fragments must not lose division remainders vs one block.

    Uses plain string content parts so structural metadata is identical on
    both sides — this isolates remainder accumulation and does not couple
    the assertion to whether typed parts also count the value ``"text"``.
    Per-fragment ``// 4`` would score 100 single-char strings as 0 text
    tokens; a single 100-char string scores 25. Accumulating before the
    single final division makes both shapes match (within one token of
    combined-division remainder).
    """
    from agent.chat_completion_helpers import estimate_request_context_tokens

    image_part = {"type": "image_url", "image_url": {"url": _b64("a" * 1000)}}
    fragmented = {
        "messages": [
            {
                "role": "user",
                "content": (["a"] * 100) + [image_part],
            }
        ],
    }
    combined = {
        "messages": [
            {
                "role": "user",
                "content": ["a" * 100, image_part],
            }
        ],
    }
    fragmented_est = estimate_request_context_tokens(fragmented)
    combined_est = estimate_request_context_tokens(combined)
    assert abs(fragmented_est - combined_est) <= 1, (
        f"short fragments diverged from combined block: "
        f"fragmented={fragmented_est} combined={combined_est}"
    )


def test_chat_and_responses_shapes_both_handled():
    from agent.chat_completion_helpers import estimate_request_context_tokens

    url = _b64("a" * 10_000)  # ~2500 tokens if counted as text
    chat = _chat_payload(text="describe", images=[url])
    codex = _codex_payload(text="describe", images=[url])
    chat_est = estimate_request_context_tokens(chat)
    codex_est = estimate_request_context_tokens(codex)
    # Both shapes must understand the image as a bounded visual cost, NOT
    # count the base64 (~2500 tokens) as text.
    assert chat_est < 500, f"chat shape inflated: {chat_est}"
    assert codex_est < 500, f"responses shape inflated: {codex_est}"


# ── watchdog tier selection ───────────────────────────────────────────────


def test_large_image_does_not_select_giant_tier():
    """A single large image must select the LOWEST watchdog tier.

    Legacy would estimate ~125k tokens -> 1200s floor (giant tier). The
    multimodal scan must stay bounded: image cost (170) + tiny text.
    """
    from agent.chat_completion_helpers import (
        estimate_request_context_tokens,
        openai_codex_stale_timeout_floor,
    )

    big = _codex_payload(text="describe", images=[_b64("a" * 500_000)])
    est = estimate_request_context_tokens(big)
    # Concrete expected tier: < 10k tokens -> no floor engaged (0.0).
    assert est < 10_000, f"giant tier triggered by single image: est={est}"
    assert openai_codex_stale_timeout_floor(est) == 0.0, (
        f"expected lowest tier, got floor for est={est}"
    )


# ── hardening ─────────────────────────────────────────────────────────────


def test_unknown_multimodal_part_uses_fallback_not_zero():
    """Unknown parts contribute their full text (never dropped to zero)."""
    from agent.chat_completion_helpers import estimate_request_context_tokens

    image_part = {"type": "image_url", "image_url": {"url": _b64("a" * 1000)}}
    without_audio = {
        "model": "gpt-5.5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    image_part,
                ],
            }
        ],
    }
    with_audio = {
        "model": "gpt-5.5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    image_part,
                    {"type": "audio", "input_audio": {"data": "A" * 10_000}},
                ],
            }
        ],
    }
    increase = (
        estimate_request_context_tokens(with_audio)
        - estimate_request_context_tokens(without_audio)
    )
    # Audio data (10_000 chars) must contribute ~10_000 // 4 tokens, not be
    # dropped or truncated to a token or two.
    assert increase >= 10_000 // 4, f"audio text undercounted: increase={increase}"


def test_unknown_wrapper_does_not_recount_base64_as_text():
    """Structured image nested in an unknown wrapper must not scale with bytes.

    The walker recurses into arbitrary wrappers and recognizes the structured
    multimodal part inside — base64 transport under the part is never
    stringified as text. A bare data-URL string (no structured part) is
    deliberately treated as text; see
    ``test_plain_text_data_url_remains_text``.
    """
    from agent.chat_completion_helpers import estimate_request_context_tokens

    def wrap(url: str) -> dict:
        return {
            "model": "gpt-5.5",
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "custom": {
                            "image": {
                                "type": "input_image",
                                "image_url": url,
                            }
                        }
                    },
                }
            ],
        }

    small = wrap(_b64("a" * 100))
    large = wrap(_b64("a" * 100_000))
    small_est = estimate_request_context_tokens(small)
    large_est = estimate_request_context_tokens(large)
    assert large_est == small_est, (
        f"base64 re-counted through unknown wrapper: "
        f"small={small_est} large={large_est}"
    )


def test_plain_text_data_url_remains_text():
    """A data URL used as plain string content is text, not an image.

    Only structured multimodal parts switch estimation modes. A user
    (or tool result) that pastes a data URL as a string must keep the
    exact legacy char/4 cost — otherwise ~25k text tokens collapse to 170.
    """
    from agent.chat_completion_helpers import estimate_request_context_tokens

    payload = {
        "messages": [
            {
                "role": "user",
                "content": "data:image/png;base64," + "A" * 100_000,
            }
        ]
    }
    legacy = sum(len(str(item)) for item in payload["messages"]) // 4
    assert estimate_request_context_tokens(payload) == legacy


def test_corrupt_base64_does_not_break_estimator():
    """Corrupt base64 still gets the same bounded visual cost as valid data URLs."""
    from agent.chat_completion_helpers import estimate_request_context_tokens

    bad = _chat_payload(text="describe", images=["data:image/png;base64,!!!not-valid!!!"])
    good = _chat_payload(text="describe", images=[_b64("a" * 1000)])
    # Estimator never decodes image bytes, so corrupt and well-formed data
    # URLs of the same shape must cost the same.
    assert estimate_request_context_tokens(bad) == estimate_request_context_tokens(good)


def test_http_image_url_bounded_fallback_without_download():
    from agent.chat_completion_helpers import estimate_request_context_tokens

    payload = {
        "model": "gpt-5.5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/huge.jpg"},
                    }
                ],
            }
        ],
    }
    # Must not attempt a download; bounded fallback.
    est = estimate_request_context_tokens(payload)
    assert 0 < est < 100_000


def test_same_image_twice_counted_twice_until_payload_dedup():
    """No estimator-side dedup: two occurrences count two visual costs.

    While the payload still transmits the image twice, counting it once
    would underestimate the work actually sent to the provider.
    """
    from agent.chat_completion_helpers import estimate_request_context_tokens

    url = _b64("a" * 1000)
    one = _codex_payload(text="describe", images=[url])
    two = _codex_payload(text="describe", images=[url, url])
    assert estimate_request_context_tokens(two) > estimate_request_context_tokens(one)


def test_metadata_and_tools_still_contribute():
    from agent.chat_completion_helpers import estimate_request_context_tokens

    base = _codex_payload(text="describe", images=[_b64("a" * 1000)])
    with_tools = dict(base)
    with_tools["tools"] = [{"name": "t", "description": "d" * 500}]
    assert estimate_request_context_tokens(with_tools) > estimate_request_context_tokens(base)


def test_tool_schema_type_image_does_not_trigger_multimodal_mode():
    """Tool schemas with type=image must keep the exact legacy estimate.

    tools are opaque metadata, not multimodal content. A schema dict with
    ``"type": "image"`` must not false-trigger multimodal mode and drop a
    large description to a single fixed image cost.
    """
    from agent.chat_completion_helpers import estimate_request_context_tokens

    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "image", "description": "x" * 10_000}],
    }
    legacy = (
        sum(len(str(item)) for item in payload["messages"])
        + len(str(payload["tools"]))
    ) // 4
    assert estimate_request_context_tokens(payload) == legacy


def test_tools_keep_legacy_cost_when_messages_are_multimodal():
    """When messages carry a real image, tools still use opaque legacy cost."""
    from agent.chat_completion_helpers import estimate_request_context_tokens

    base = _chat_payload(text="describe", images=[_b64("a" * 1_000)])
    with_tools = dict(base)
    with_tools["tools"] = [
        {
            "name": "large_tool",
            "description": "x" * 10_000,
        }
    ]
    increase = (
        estimate_request_context_tokens(with_tools)
        - estimate_request_context_tokens(base)
    )
    # -1 accommodates only the combined-division remainder.
    assert increase >= len(str(with_tools["tools"])) // 4 - 1


def test_instructions_keep_legacy_cost_when_input_is_multimodal():
    """Responses instructions stay opaque even when input has an image."""
    from agent.chat_completion_helpers import estimate_request_context_tokens

    base = _codex_payload(text="describe", images=[_b64("a" * 1_000)])
    with_instructions = dict(base)
    with_instructions["instructions"] = "y" * 10_000
    increase = (
        estimate_request_context_tokens(with_instructions)
        - estimate_request_context_tokens(base)
    )
    assert increase >= len(str(with_instructions["instructions"])) // 4 - 1


def test_deeply_nested_image_does_not_fall_back_to_base64_counting():
    """A structured image nested beyond any depth cap must still be detected.

    The scan is iterative (explicit stack), so no depth limit can silently
    send the payload back to the legacy path where base64 is text. Nest the
    whole multimodal part — a bare data-URL string is text, not an image.
    """
    from agent.chat_completion_helpers import estimate_request_context_tokens

    def deeply_wrap(url: str, depth: int = 200) -> dict:
        node: object = {
            "type": "image_url",
            "image_url": {"url": url},
        }
        for _ in range(depth):
            node = {"nested": node}
        return {
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": node}],
        }

    small = deeply_wrap(_b64("a" * 100))
    large = deeply_wrap(_b64("a" * 100_000))
    assert estimate_request_context_tokens(small) == estimate_request_context_tokens(large)


def test_deep_text_subtree_is_not_silently_dropped():
    """Text nested deep inside a multimodal payload still contributes."""
    from agent.chat_completion_helpers import estimate_request_context_tokens

    def deeply_wrap(text: str, depth: int = 200) -> object:
        node: object = text
        for _ in range(depth):
            node = {"nested": node}
        return node

    image_only = {
        "model": "gpt-5.5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _b64("a" * 1000)}},
                ],
            }
        ],
    }
    with_deep_text = {
        "model": "gpt-5.5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _b64("a" * 1000)}},
                    deeply_wrap("x" * 10_000),
                ],
            }
        ],
    }
    increase = (
        estimate_request_context_tokens(with_deep_text)
        - estimate_request_context_tokens(image_only)
    )
    assert increase >= 10_000 // 4, f"deep text subtree dropped: increase={increase}"


def test_huge_image_estimate_bounded():
    from agent.chat_completion_helpers import estimate_request_context_tokens

    giant = _codex_payload(text="describe", images=[_b64("a" * 2_000_000)])
    est = estimate_request_context_tokens(giant)
    assert est < 100_000, f"2MB image inflated estimate to {est}"


def test_estimator_does_not_mutate_payload():
    from agent.chat_completion_helpers import estimate_request_context_tokens

    payload = _chat_payload(text="describe", images=[_b64("a" * 1000)])
    before = copy.deepcopy(payload)
    estimate_request_context_tokens(payload)
    assert payload == before
