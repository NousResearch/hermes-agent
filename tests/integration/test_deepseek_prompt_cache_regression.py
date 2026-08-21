"""Integration regression: DeepSeek prompt-cache hits across multi-step turns.

Proves against the real DeepSeek API (https://api.deepseek.com) that a
byte-identical conversation prefix yields provider-reported cache hits on
every request after the first, and that Hermes' canonical usage mapping
(``normalize_usage``) surfaces them as ``cache_read_tokens > 0`` — the
accounting signal that feeds ``agent.session_cache_read_tokens`` (the
DeepSeek native top-level ``prompt_cache_hit_tokens`` shape, #61871).

Two levels of proof:

1. ``test_multi_step_tool_turn_reports_cache_read_tokens`` — a direct
   multi-step LLM sequence with a tool call in the middle (turn 1 forces a
   tool call → 2 requests; turn 2 appends a follow-up message). Deterministic
   prefix control; asserts raw provider usage AND the Hermes canonical
   mapping on every request after the first.
2. ``test_agent_loop_second_turn_hits_cache`` — the real agent loop
   (``AIAgent.chat``), two consecutive turns; asserts the session-level
   ``session_cache_read_tokens`` accumulator.

Key-gated: skipped unless ``DEEPSEEK_API_KEY`` is available in the process
env or in ``$LOCALAPPDATA/hermes/.env`` / ``~/.hermes/.env``. The key is
captured at module import time because ``tests/conftest.py`` blanks
credential env vars before each test body runs.

Run with (integration tests are deselected by default via addopts):
    pytest tests/integration/test_deepseek_prompt_cache_regression.py -m integration -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent.usage_pricing import normalize_usage

LIVE_MODEL = "deepseek-v4-flash"
LIVE_BASE_URL = (
    os.environ.get("DEEPSEEK_BASE_URL", "").strip() or "https://api.deepseek.com"
)

# .env candidates in the same precedence order Hermes itself uses for
# profile secrets: the active profile home ($LOCALAPPDATA/hermes on Windows)
# first, then the classic ~/.hermes home.
_ENV_FILE_CANDIDATES: list[Path] = []
_local_appdata = os.environ.get("LOCALAPPDATA")
if _local_appdata:
    _ENV_FILE_CANDIDATES.append(Path(_local_appdata) / "hermes" / ".env")
_ENV_FILE_CANDIDATES.append(Path.home() / ".hermes" / ".env")


def _resolve_secret(key: str) -> str:
    """Resolve a secret from the process env, then from Hermes .env files.

    Runs at module import time only: tests/conftest.py blanks credential
    env vars before each test body runs, so import time is the only window
    where the key is visible. Never prints or logs the value.
    """
    value = os.environ.get(key, "").strip()
    if value:
        return value
    for env_file in _ENV_FILE_CANDIDATES:
        if not env_file.exists():
            continue
        # Repo-native .env parser (utf-8-sig aware, handles quoting/BOM the
        # same way Hermes' own secret loading does).
        from agent.secret_scope import load_env_file

        value = load_env_file(env_file).get(key, "").strip()
        if value:
            return value
    return ""


DEEPSEEK_API_KEY = _resolve_secret("DEEPSEEK_API_KEY")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DEEPSEEK_API_KEY, reason="DEEPSEEK_API_KEY not configured"
    ),
]

# Long enough that the shared request prefix spans several of DeepSeek's
# 64-token context-cache blocks from the very first request (mirrors the TS
# harness's packages/core/agent-loop/tests/request-cache.e2e.ts system prompt).
SYSTEM_PROMPT = (
    "You are a terse coding assistant used in an automated cache test. "
    "Always follow instructions literally and exactly. When the user asks you to look "
    "something up, call the lookup tool with the requested key and wait for its result "
    "before answering. Never invent a value the tool has not returned. After the tool "
    "returns, answer with a single short sentence that repeats the returned value "
    "verbatim. Do not add explanations, do not use markdown, do not ask follow-up "
    "questions. If the user asks anything else, answer in one short sentence."
)

LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look up the stored value for a key.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The key to look up."}
            },
            "required": ["key"],
            "additionalProperties": False,
        },
    },
}

CACHE_E2E_VALUE = "azure-falcon-42"


def _live_client():
    from openai import OpenAI

    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=LIVE_BASE_URL, timeout=120.0)


def _canonical_usage(response) -> int:
    """Hermes' canonical cache-read bucket for a raw chat-completions response."""
    return normalize_usage(response.usage, provider="deepseek").cache_read_tokens


def _assert_healthy_reply(reply: str, turn_label: str) -> None:
    """AIAgent returns an error-sentinel string (not an exception) when the
    underlying API call fails past retries; a naive truthiness assert misses
    it. Same checker as tests/run_agent/test_sequential_chats_live.py."""
    assert reply and reply.strip(), f"{turn_label} returned empty: {reply!r}"
    lowered = reply.lower().strip()
    bad_markers = (
        "api call failed",
        "connection error",
        "client has been closed",
        "max retries",
    )
    for marker in bad_markers:
        assert marker not in lowered, (
            f"{turn_label} returned an error-sentinel string instead of a real "
            f"model reply — matched marker {marker!r}. Reply was: {reply!r}"
        )


# ────────────────────────────────────────────────────────────────────────────
# 1. Direct multi-step LLM sequence (tool call in the middle)
# ────────────────────────────────────────────────────────────────────────────


def test_multi_step_tool_turn_reports_cache_read_tokens():
    """Every request after the first must report a DeepSeek cache hit.

    Request 1: system + user → forces a tool call (multi-step turn begins).
    Request 2: same prefix + assistant tool call + tool result → the model
        answers with the tool value; this request's prefix is byte-identical
        to request 1's plus an append, so DeepSeek must report
        ``prompt_cache_hit_tokens > 0`` and Hermes' canonical mapping must
        surface ``cache_read_tokens > 0``.
    Request 3: turn 2 — follow-up over the same (longer) prefix → cache hit
        again.
    """
    client = _live_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                'Look up the key "deploy-color" with the lookup tool and '
                "tell me the value."
            ),
        },
    ]
    common = {
        "model": LIVE_MODEL,
        "tools": [LOOKUP_TOOL],
        "max_tokens": 1024,
        "timeout": 120,
    }

    # Request 1 — the prefix to be reused. We deliberately do NOT assert
    # cache_read == 0 here: DeepSeek's context cache can outlive a test run,
    # so a re-run may legitimately find this exact prefix warm. The
    # within-run proof lives in request 3, whose prefix embeds the unique
    # tool_call id minted by request 1 of THIS run.
    first = client.chat.completions.create(**common, messages=messages)
    first_message = first.choices[0].message
    assert first_message.tool_calls, (
        "DeepSeek did not return a tool call on request 1 — the multi-step "
        "turn never started"
    )
    first_tool_call = first_message.tool_calls[0]
    assert first_tool_call.function.name == "lookup"
    assert (
        json.loads(first_tool_call.function.arguments or "{}").get("key")
        == "deploy-color"
    )

    # Request 2 — the multi-step continuation: assistant tool call + result.
    messages.extend(
        [
            {
                "role": "assistant",
                "content": first_message.content or "",
                "tool_calls": [
                    {
                        "id": first_tool_call.id,
                        "type": "function",
                        "function": {
                            "name": first_tool_call.function.name,
                            "arguments": first_tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": first_tool_call.id,
                "content": json.dumps(
                    {"key": "deploy-color", "value": CACHE_E2E_VALUE},
                    separators=(",", ":"),
                ),
            },
        ]
    )
    second = client.chat.completions.create(**common, messages=messages)

    # Provider-level proof: DeepSeek's native top-level cache-hit field.
    raw_hit = getattr(second.usage, "prompt_cache_hit_tokens", 0) or 0
    assert raw_hit > 0, (
        "DeepSeek reported 0 prompt_cache_hit_tokens on request 2 despite a "
        "byte-identical prefix — the provider context cache did not hit"
    )
    # Hermes-observable proof: normalize_usage must map it to the canonical
    # cache_read_tokens bucket (#61871).
    assert _canonical_usage(second) > 0, (
        "normalize_usage mapped request 2 to cache_read_tokens == 0 even "
        "though the provider reported prompt_cache_hit_tokens > 0"
    )
    second_text = second.choices[0].message.content or ""
    assert CACHE_E2E_VALUE in second_text, (
        "the tool value did not make it through the multi-step turn into the "
        "final answer"
    )

    # Request 3 — turn 2 follow-up over the same (longer) prefix.
    messages.append(
        {"role": "user", "content": "Thanks. Repeat that value one more time."}
    )
    third = client.chat.completions.create(**common, messages=messages)
    assert _canonical_usage(third) > 0, (
        "request 3 (turn 2) reported cache_read_tokens == 0 — the turn "
        "boundary broke the shared prefix"
    )
    assert CACHE_E2E_VALUE in (third.choices[0].message.content or "")


# ────────────────────────────────────────────────────────────────────────────
# 2. Real agent loop
# ────────────────────────────────────────────────────────────────────────────


def test_agent_loop_second_turn_hits_cache():
    """Full Hermes agent loop: two consecutive turns against DeepSeek.

    The loop builds one byte-stable system prompt per session (cached on
    ``agent._cached_system_prompt``), so turn 2 reuses turn 1's prefix and
    must accumulate ``session_cache_read_tokens > 0`` — the same accounting
    the CLI/gateway surface in per-turn token reports.
    """
    from run_agent import AIAgent

    agent = AIAgent(
        model=LIVE_MODEL,
        provider="deepseek",
        api_key=DEEPSEEK_API_KEY,
        base_url=LIVE_BASE_URL,
        max_iterations=3,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        # No toolsets: each turn is a single model request, so the cache
        # signal is purely about the stable system-prompt + message prefix.
        disabled_toolsets=["*"],
    )

    r1 = agent.chat("Reply with the single word: ALPHA")
    _assert_healthy_reply(r1, "turn 1")
    assert agent.session_api_calls >= 1

    r2 = agent.chat("Reply with the single word: BETA")
    _assert_healthy_reply(r2, "turn 2")
    assert agent.session_api_calls >= 2, (
        "expected at least 2 API calls across the two turns"
    )
    assert agent.session_cache_read_tokens > 0, (
        "no cache-read tokens after the second agent-loop turn — the system "
        "prompt or message prefix changed between turns, which would break "
        "prompt caching for every real conversation"
    )
