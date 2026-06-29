"""INV-1 per-adapter leak tests — the `_lcm_summary` structural tag MUST NOT
reach any provider request body, on ANY adapter.

The LCM engine tags the summary message it assembles with an internal
``_lcm_summary: True`` key (so compaction-stats classification is structural).
That tag rides the active-context message list that becomes the next-turn
request for *whatever provider is active*. This file proves, per adapter, that
the tag is stripped (OpenAI-shape) or never copied (allowlist-rebuild adapters)
before serialization — closing INV-1 across every message-serializing path
(spec §0.7). It also serves as the INV-2 cache proof: identical request bytes
with/without the tag → identical cache key.

Spec: plans/2026-06-22_structural-summary-tagging-and-degrade-observability-SPEC.md
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest


def _tagged_summary_messages() -> List[Dict[str, Any]]:
    """A realistic active-context list with a tagged LCM summary row."""
    return [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "earlier user turn"},
        {
            "role": "assistant",
            "content": "[Session Arc Summary (d1, node 7)]\nrolled up earlier turns\n[Expand for details: x]",
            "_lcm_summary": True,
        },
        {"role": "user", "content": "latest user turn"},
    ]


def _untagged_summary_messages() -> List[Dict[str, Any]]:
    msgs = _tagged_summary_messages()
    for m in msgs:
        m.pop("_lcm_summary", None)
    return msgs


def _assert_no_lcm_summary_anywhere(obj: Any) -> None:
    """Deep-walk a serialized request structure asserting no `_lcm_summary` key."""
    if isinstance(obj, dict):
        assert "_lcm_summary" not in obj, f"_lcm_summary leaked into request dict: {obj!r}"
        for v in obj.values():
            _assert_no_lcm_summary_anywhere(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _assert_no_lcm_summary_anywhere(v)


# ── OpenAI-shape (the `_`-prefix strip path) ────────────────────────────────

def test_openai_shape_strips_lcm_summary():
    from agent.transports.chat_completions import ChatCompletionsTransport
    t = ChatCompletionsTransport()
    out = t.convert_messages(_tagged_summary_messages())
    _assert_no_lcm_summary_anywhere(out)
    # the summary row's CONTENT must survive (only the internal key is dropped)
    blob = repr(out)
    assert "Session Arc Summary" in blob, "the summary content must survive the strip"


# ── Anthropic (allowlist rebuild) ───────────────────────────────────────────

def test_anthropic_adapter_drops_lcm_summary():
    from agent.anthropic_adapter import convert_messages_to_anthropic
    _system, anthropic_messages = convert_messages_to_anthropic(_tagged_summary_messages())
    _assert_no_lcm_summary_anywhere(anthropic_messages)


def test_anthropic_request_byte_identical_tagged_vs_untagged():
    """INV-2 cache proof: the rebuilt Anthropic messages are byte-identical with
    or without the tag → the cache key cannot shift."""
    from agent.anthropic_adapter import convert_messages_to_anthropic
    _s1, tagged = convert_messages_to_anthropic(_tagged_summary_messages())
    _s2, untagged = convert_messages_to_anthropic(_untagged_summary_messages())
    assert tagged == untagged, "tag must not perturb the serialized Anthropic request (cache safety)"


# ── Gemini CloudCode: provider REMOVED upstream (#50492, google-gemini-cli +
#    google-antigravity OAuth providers deleted for account-ban risk). The
#    agent.gemini_cloudcode_adapter module no longer exists, so its LCM-summary
#    guard is retired here during the 2026-06-29 parity sync. The surviving
#    Gemini *native* adapter is still covered below.


# ── Gemini Native (allowlist rebuild) ───────────────────────────────────────

def test_gemini_native_drops_lcm_summary():
    from agent.gemini_native_adapter import _build_gemini_contents
    contents, _system = _build_gemini_contents(_tagged_summary_messages())
    _assert_no_lcm_summary_anywhere(contents)


# ── Bedrock (allowlist rebuild) — in-scope to prove the invariant generally ──

def test_bedrock_adapter_drops_lcm_summary():
    try:
        from agent.bedrock_adapter import convert_messages_to_converse
    except Exception:  # pragma: no cover - adapter import optional
        pytest.skip("bedrock adapter not importable in this environment")
    _system, converse_messages = convert_messages_to_converse(_tagged_summary_messages())
    _assert_no_lcm_summary_anywhere(converse_messages)
