"""Regression tests: stream read timeout must apply reasoning-model floor
BEFORE ``stream_kwargs["timeout"]`` is built.

In ``agent/chat_completion_helpers.py`` the stale-stream detector's
reasoning-model floor is computed *after* the httpx socket read timeout
is baked into ``stream_kwargs["timeout"]``.  Reasoning models
(e.g. DeepSeek V4 on opencode-go) that need 65–77s for their first
content token are killed at 120s (the socket read default) even when
the stale detector would later have tolerated 600s.

The fix in ``_call_chat_completions`` applies
``get_reasoning_stale_timeout_floor`` to ``_stream_read_timeout``
between the local/cloud blocks and the ``stream_kwargs`` builder,
so the socket read timeout and the stale detector agree.

These tests mirror the relevant sections of the production code
and assert the floor is applied — and never lowers an explicit config.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor


def _resolve_stream_read_timeout(
    model: str | None,
    base_url: str,
    provider_timeout_cfg: float | None = None,
    stream_read_env: str | None = None,
) -> float:
    """Mirror of the _stream_read_timeout resolution in
    ``agent/chat_completion_helpers.py:_call_chat_completions``
    (lines ~2066–2119), including the new reasoning-floor block.

    The production code is an inner function inside
    ``chat_completion_stream_request``; this mirror extracts the pure
    timeout-resolution logic so tests don't need a live agent + worker
    thread.
    """
    from agent.model_metadata import is_local_endpoint

    # Per-provider config wins (line 2066-2071).
    _base_timeout = (
        provider_timeout_cfg
        if provider_timeout_cfg is not None
        else float(os.getenv("HERMES_API_TIMEOUT", "1800.0"))
    )

    # Read timeout (line 2074-2107).
    if provider_timeout_cfg is not None:
        _stream_read_timeout = provider_timeout_cfg
    else:
        _stream_read_timeout = float(
            os.environ.get("HERMES_STREAM_READ_TIMEOUT", str(stream_read_env or "120.0"))
        )
        if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
            _stream_read_timeout = _base_timeout
        # The cloud reasoning block is skipped here because
        # _stream_stale_timeout is always None at this point in the
        # real code (it's resolved later).  Our new floor block runs
        # immediately after, covering the gap.

    # === NEW: reasoning-model floor (lines inserted by this PR) ===
    _reasoning_floor = get_reasoning_stale_timeout_floor(model)
    if _reasoning_floor is not None:
        _stream_read_timeout = max(_stream_read_timeout, _reasoning_floor)

    return _stream_read_timeout


# ── positive cases: reasoning floor applies ────────────────────────────


@pytest.mark.parametrize("model, expected_floor", [
    ("deepseek/deepseek-v4-flash", 600.0),
    ("deepseek/deepseek-v4-pro", 600.0),
    ("deepseek/deepseek-r1", 600.0),
    ("nvidia/nemotron-3-ultra-550b-a55b", 600.0),
    ("openai/o3", 600.0),
    ("openai/o3-mini", 300.0),
    ("anthropic/claude-opus-4-6", 240.0),
    ("x-ai/grok-4-fast-reasoning", 300.0),
])
def test_reasoning_floor_applied_to_stream_read_timeout(
    model, expected_floor, monkeypatch,
):
    """Cloud endpoint + default 120s + reasoning model -> floor raises read
    timeout so the socket doesn't kill the stream before the detector."""
    monkeypatch.delenv("HERMES_STREAM_READ_TIMEOUT", raising=False)
    monkeypatch.setenv("HERMES_API_TIMEOUT", "1800.0")

    timeout = _resolve_stream_read_timeout(
        model=model,
        base_url="https://api.example.com/v1",  # cloud endpoint
    )
    assert timeout == expected_floor, (
        f"Stream read timeout for {model} on cloud endpoint must be "
        f"raised to {expected_floor}s (floor) from 120s (default); "
        f"got {timeout}s — socket would kill the stream before the "
        f"stale detector could act."
    )


@pytest.mark.parametrize("model, expected_floor", [
    ("deepseek/deepseek-v4-flash", 600.0),
    ("openai/o3", 600.0),
])
def test_reasoning_floor_applies_even_with_env_read_timeout(
    model, expected_floor, monkeypatch,
):
    """User sets HERMES_STREAM_READ_TIMEOUT=180 — still below the 600s
    floor for DeepSeek V4.  The floor must raise it further."""
    monkeypatch.setenv("HERMES_STREAM_READ_TIMEOUT", "180")
    monkeypatch.setenv("HERMES_API_TIMEOUT", "1800.0")

    timeout = _resolve_stream_read_timeout(
        model=model,
        base_url="https://api.example.com/v1",
    )
    assert timeout == expected_floor, (
        f"Stream read timeout for {model} must be {expected_floor}s — "
        f"the floor raises the env-set 180s; got {timeout}s."
    )


# ── negative cases: floor never lowers ─────────────────────────────────


def test_reasoning_floor_never_lowers_provider_config():
    """Per-provider request_timeout_seconds=900 must stay 900s even
    when the floor would be lower (e.g. 600s for DeepSeek V4)."""
    timeout = _resolve_stream_read_timeout(
        model="deepseek/deepseek-v4-flash",
        base_url="https://api.example.com/v1",
        provider_timeout_cfg=900.0,
    )
    assert timeout == 900.0, (
        "Explicit provider config (900s) must NOT be lowered by the "
        f"reasoning floor (600s); got {timeout}s."
    )


def test_reasoning_floor_never_lowers_env_read_timeout():
    """User sets HERMES_STREAM_READ_TIMEOUT=900 — already above the
    600s floor; floor must not lower it."""
    with patch.dict(os.environ, {"HERMES_STREAM_READ_TIMEOUT": "900"}):
        timeout = _resolve_stream_read_timeout(
            model="deepseek/deepseek-v4-flash",
            base_url="https://api.example.com/v1",
        )
    assert timeout == 900.0, (
        "Explicit env var 900s must NOT be lowered by the floor; "
        f"got {timeout}s."
    )


def test_non_reasoning_model_keeps_default():
    """gpt-4o on cloud endpoint — 120s default unchanged (no floor)."""
    with patch.dict(os.environ, {}, clear=True):
        timeout = _resolve_stream_read_timeout(
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
        )
    assert timeout == 120.0, (
        f"Non-reasoning model must keep the 120s default; got {timeout}s."
    )


def test_non_reasoning_model_keeps_env_override():
    """gpt-4o with HERMES_STREAM_READ_TIMEOUT=300 stays at 300."""
    with patch.dict(os.environ, {"HERMES_STREAM_READ_TIMEOUT": "300"}):
        timeout = _resolve_stream_read_timeout(
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
        )
    assert timeout == 300.0


def test_local_endpoint_with_reasoning_model_still_bumps():
    """Local endpoint (Ollama) + reasoning model — local bump wins
    (1800s), floor stays out of the way because local timeout is already
    higher."""
    timeout = _resolve_stream_read_timeout(
        model="deepseek/deepseek-v4-flash",
        base_url="http://localhost:11434",
    )
    # Local endpoint bumps to HERMES_API_TIMEOUT=1800s; floor (600s)
    # is lower -> no effect (max keeps 1800).
    assert timeout == 1800.0, (
        f"Local endpoint + reasoning model: read timeout should be "
        f"1800s (local bump); floor 600s should not interfere; "
        f"got {timeout}s."
    )


def test_empty_base_url_no_crash():
    """Reasoning model without base_url must not crash — floor applied
    normally, local/cloud blocks skipped."""
    timeout = _resolve_stream_read_timeout(
        model="deepseek/deepseek-v4-flash",
        base_url="",
    )
    assert timeout == 600.0