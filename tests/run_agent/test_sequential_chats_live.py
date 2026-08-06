"""Live regression guardrail for the keepalive/transport bug class (#10933).

AlexKucera reported on Discord (2026-04-16) that after ``hermes update`` pulled
#10933, the FIRST chat in a session worked and EVERY subsequent chat failed
with ``APIConnectionError('Connection error.')`` whose cause was
``RuntimeError: Cannot send a request, as the client has been closed``.

The companion ``test_create_openai_client_reuse.py`` pins this contract at
object level with mocked ``OpenAI``. This file runs the same shape of
reproduction against a real provider so we have a true end-to-end smoke test
for any future keepalive / transport plumbing.

Opt-in — not part of default CI:
    HERMES_LIVE_TESTS=1 pytest tests/run_agent/test_sequential_chats_live.py -v

Requires ``OPENROUTER_API_KEY`` to be set (or sourced via ~/.hermes/.env).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


# Load ~/.hermes/.env so live runs pick up OPENROUTER_API_KEY without
# needing the runner to shell-source it first. Silent if the file is absent.
#
# Only the key this module needs is imported, and only when live runs are
# actually enabled. The earlier version copied EVERY key from the file into
# os.environ at import time, which broke unrelated tests: pytest imports every
# collected module during COLLECTION, before any fixture runs, so the writes
# landed ahead of the hermetic `_hermetic_environment` autouse fixture and were
# never scrubbed. The file here carries 23 keys, ~12 of which (SEARXNG_URL,
# TELEGRAM_*, WHATSAPP_*, HERMES_SPOTIFY_*) do not match the conftest
# credential-suffix filter. With SEARXNG_URL leaked, `_get_backend()` resolved
# to "searxng" and two web-backend tests failed on `assert 'searxng' ==
# 'firecrawl'` — in any run that merely COLLECTED this file, with zero tests
# from it executing, which is why an execution-order bisect found nothing.
_LIVE_ENV_KEYS = ("OPENROUTER_API_KEY",)


def _load_user_env(keys: tuple[str, ...] = _LIVE_ENV_KEYS) -> None:
    env_file = Path.home() / ".hermes" / ".env"
    if not env_file.exists():
        return
    wanted = set(keys)
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k not in wanted:
            continue
        v = v.strip().strip('"').strip("'")
        # Don't clobber an already-set env var — lets the caller override.
        os.environ.setdefault(k, v)


LIVE = os.environ.get("HERMES_LIVE_TESTS") == "1"

# Gate the env read on the live flag: a skipped module must not touch shared
# process state just by being imported.
if LIVE:
    _load_user_env()

OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")

pytestmark = [
    pytest.mark.skipif(not LIVE, reason="live-only — set HERMES_LIVE_TESTS=1"),
    pytest.mark.skipif(not OR_KEY, reason="OPENROUTER_API_KEY not configured"),
]

# Cheap, fast, tool-capable. Swap if it ever goes dark.
LIVE_MODEL = "google/gemini-2.5-flash"


def _make_live_agent():
    from run_agent import AIAgent

    return AIAgent(
        model=LIVE_MODEL,
        provider="openrouter",
        api_key=OR_KEY,
        base_url="https://openrouter.ai/api/v1",
        max_iterations=3,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        # All toolsets off so the agent just produces a single text reply
        # per turn — we want to test the HTTP client lifecycle, not tools.
        disabled_toolsets=["*"],
    )


def _looks_like_error_reply(reply: str) -> tuple[bool, str]:
    """AIAgent returns an error-sentinel string (not an exception) when the
    underlying API call fails past retries. A naive ``assert reply and
    reply.strip()`` misses this because the sentinel is truthy. This
    checker enumerates the known-bad shapes so the live test actually
    catches #10933 instead of rubber-stamping the error response.
    """
    lowered = reply.lower().strip()
    bad_substrings = (
        "api call failed",
        "connection error",
        "client has been closed",
        "cannot send a request",
        "max retries",
    )
    for marker in bad_substrings:
        if marker in lowered:
            return True, marker
    return False, ""


def _assert_healthy_reply(reply, turn_label: str) -> None:
    assert reply and reply.strip(), f"{turn_label} returned empty: {reply!r}"
    is_err, marker = _looks_like_error_reply(reply)
    assert not is_err, (
        f"{turn_label} returned an error-sentinel string instead of a real "
        f"model reply — matched marker {marker!r}. This is the exact shape "
        f"of #10933 (AlexKucera Discord report, 2026-04-16): the agent's "
        f"retry loop burned three attempts against a closed httpx transport "
        f"and surfaced 'API call failed after 3 retries: Connection error.' "
        f"to the user. Reply was: {reply!r}"
    )


def test_three_sequential_chats_across_client_rebuild():
    """Reproduces AlexKucera's exact failure shape end-to-end.

    Turn 1 always worked under #10933. Turn 2 was the one that failed
    because the shared httpx transport had been torn down between turns.
    Turn 3 is here as extra insurance against any lazy-init shape where
    the failure only shows up on call N>=3.

    We also deliberately trigger ``_replace_primary_openai_client`` between
    turn 2 and turn 3 — that is the real rebuild entrypoint (401 refresh,
    credential rotation, model switch) and is the path that actually
    stored the closed transport into ``self._client_kwargs`` in #10933.
    """
    agent = _make_live_agent()

    r1 = agent.chat("Respond with only the word: ONE")
    _assert_healthy_reply(r1, "turn 1")

    r2 = agent.chat("Respond with only the word: TWO")
    _assert_healthy_reply(r2, "turn 2")

    # Force a client rebuild through the real path — mimics 401 refresh /
    # credential rotation / model switch lifecycle.
    rebuilt = agent._replace_primary_openai_client(reason="regression_test_rebuild")
    assert rebuilt, "rebuild via _replace_primary_openai_client returned False"

    r3 = agent.chat("Respond with only the word: THREE")
    _assert_healthy_reply(r3, "turn 3 (post-rebuild)")
