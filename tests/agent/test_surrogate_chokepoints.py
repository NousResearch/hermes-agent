"""Lone-surrogate chokepoint regression tests.

One class of bug, many crash sites: a lone UTF-16 surrogate (U+D800–U+DFFF)
in model output or a request payload crashes whichever consumer encodes it
first — oneshot stdout (#80366), Telegram's utf16_len check (#55309), Signal
formatting (#55143), provider request bodies / tool descriptions (#50959),
and NIM responses that bypass the Ollama-era sanitize pass (#19819).

The fix is owned at chokepoints, not leaf sites:

* ``finalize_turn`` scrubs ``final_response`` once, where model text leaves
  the conversation loop (covers oneshot, gateway, API server, subagents).
* ``_sanitize_gateway_final_response`` scrubs at the gateway delivery
  boundary for chat surfaces (defense in depth for legacy/plugin paths).
* ``run_conversation`` walks the fully-built ``api_kwargs`` with
  ``_sanitize_structure_surrogates`` so tool descriptions and every other
  request-body leaf are json-encodable before any provider sees them.
"""

from types import SimpleNamespace

import json

import pytest

from agent.message_sanitization import (
    _sanitize_structure_surrogates,
    _sanitize_surrogates,
)
from agent.turn_finalizer import finalize_turn
from tests.agent.test_turn_finalizer_final_response_persistence import FakeAgent

LONE_HIGH = "\ud83d"  # unpaired high surrogate (half of an emoji pair)
LONE_LOW = "\udce7"  # the exact code point reported in #19819


# ---------------------------------------------------------------------------
# Chokepoint 1: model text leaving the conversation loop (finalize_turn)
# ---------------------------------------------------------------------------


def test_finalize_turn_scrubs_lone_surrogate_from_final_response(monkeypatch):
    """#80366 / #19819: the returned final_response must be valid Unicode.

    ``final_response`` is often raw SDK content (``assistant_message.content``)
    — not the sanitized history copy — so without the chokepoint a lone
    surrogate reaches oneshot stdout / gateway delivery and crashes there.
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    dirty = f"answer {LONE_HIGH} and {LONE_LOW} here"
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": dirty},
    ]

    result = finalize_turn(
        agent,
        final_response=dirty,
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="t",
        turn_id="tid",
        user_message="q",
        original_user_message="q",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    final = result["final_response"]
    # Encodable everywhere a delivery surface needs it to be.
    final.encode("utf-8")
    final.encode("utf-16-le")
    assert "\ufffd" in final
    assert "answer " in final and " here" in final


def test_finalize_turn_leaves_non_string_final_response_alone(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=1,
        interrupted=True,
        failed=False,
        messages=[{"role": "user", "content": "q"}],
        conversation_history=[],
        effective_task_id="t",
        turn_id="tid",
        user_message="q",
        original_user_message="q",
        _should_review_memory=False,
        _turn_exit_reason="interrupted",
    )
    assert result["final_response"] is None


# ---------------------------------------------------------------------------
# Chokepoint 2: gateway delivery boundary (mock platform send)
# ---------------------------------------------------------------------------


def test_gateway_final_response_sanitized_for_chat_surfaces():
    """#55309 / #55143: chat-surface replies must survive utf16_len/encode."""
    from gateway.platforms.base import utf16_len
    from gateway.run import _sanitize_gateway_final_response

    dirty = f"Here is your answer {LONE_HIGH} done"
    cleaned = _sanitize_gateway_final_response("telegram", dirty)

    # The raw text raises; the sanitized boundary output must not.
    with pytest.raises(UnicodeEncodeError):
        utf16_len(dirty)
    assert utf16_len(cleaned) > 0
    cleaned.encode("utf-8")
    assert "\ufffd" in cleaned
    assert "Here is your answer" in cleaned


def test_gateway_raw_text_surfaces_keep_passthrough():
    """Programmatic surfaces keep raw text — their JSON consumers escape
    surrogates safely, and byte fidelity matters there."""
    from gateway.run import _sanitize_gateway_final_response

    dirty = f"raw {LONE_HIGH} text"
    assert _sanitize_gateway_final_response("local", dirty) == dirty


# ---------------------------------------------------------------------------
# Chokepoint 3: outbound API request assembly (#50959)
# ---------------------------------------------------------------------------


def test_api_kwargs_walk_makes_tool_descriptions_json_safe():
    """A surrogate anywhere in the request body (tool descriptions included)
    must be scrubbed by one structure walk so json.dumps cannot fail."""
    api_kwargs = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "session_search",
                    "description": f"±5 message window {LONE_HIGH} around the match",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"description": f"nested {LONE_LOW} leaf"},
                        },
                    },
                },
            }
        ],
        "extra_body": {"note": f"deep {LONE_HIGH}"},
    }

    assert _sanitize_structure_surrogates(api_kwargs) is True
    encoded = json.dumps(api_kwargs)  # would raise UnicodeEncodeError-class failure before
    assert "\\ud83d" not in encoded.lower()
    desc = api_kwargs["tools"][0]["function"]["description"]
    assert desc.startswith("±5 message window")
    assert "\ufffd" in desc
    # Second walk is a no-op on the now-clean payload.
    assert _sanitize_structure_surrogates(api_kwargs) is False


def test_conversation_loop_sanitizes_api_kwargs_after_build():
    """Wiring pin: the structure walk runs on the fully-built api_kwargs
    (after _build_api_kwargs, before any transport/provider sees it)."""
    import inspect

    import agent.conversation_loop as cl

    src = inspect.getsource(cl.run_conversation)
    build_idx = src.index("api_kwargs = agent._build_api_kwargs(api_messages)")
    sanitize_idx = src.index("_sanitize_structure_surrogates(api_kwargs)")
    perform_idx = src.index("def _perform_api_call")
    assert build_idx < sanitize_idx < perform_idx


# ---------------------------------------------------------------------------
# Helper semantics shared by every chokepoint
# ---------------------------------------------------------------------------


def test_sanitize_surrogates_preserves_valid_astral_pairs():
    """Valid non-BMP text (proper emoji, CJK extension chars) is untouched."""
    text = "ok 😀 你好 𝕏"
    assert _sanitize_surrogates(text) == text
