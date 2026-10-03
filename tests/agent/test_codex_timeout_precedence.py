"""Codex watchdog timeout hierarchy and precedence (no 180s sleeps).

Token-bucket idle defaults, env overrides, Codex vs generic stale, and the
historical 300s auxiliary compression floor are contracts. This file calls
production helpers rather than reading source text.
"""

from __future__ import annotations

import sys
import types

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def test_event_idle_default_buckets_preserve_180_for_large_context():
    from agent.chat_completion_helpers import codex_event_idle_timeout_default

    assert codex_event_idle_timeout_default(0) == 12.0
    assert codex_event_idle_timeout_default(10_001) == 60.0
    assert codex_event_idle_timeout_default(50_001) == 120.0
    assert codex_event_idle_timeout_default(100_001) == 180.0


def test_openai_codex_stale_floor_historical_600_900_1200_unchanged():
    from agent.chat_completion_helpers import openai_codex_stale_timeout_floor

    assert openai_codex_stale_timeout_floor(10_000) == 0.0
    assert openai_codex_stale_timeout_floor(10_001) == 600.0
    assert openai_codex_stale_timeout_floor(50_001) == 900.0
    assert openai_codex_stale_timeout_floor(100_001) == 1200.0


def test_env_event_idle_override_beats_token_bucket(monkeypatch):
    from agent.chat_completion_helpers import resolve_codex_event_idle_timeout

    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "300")
    assert resolve_codex_event_idle_timeout(est_tokens=200_000) == 300.0
    monkeypatch.delenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS")
    assert resolve_codex_event_idle_timeout(est_tokens=200_000) == 180.0


def test_zero_env_disables_event_idle_not_hard_ceiling():
    from agent.chat_completion_helpers import (
        resolve_codex_event_idle_timeout,
        resolve_codex_hard_timeout,
    )

    assert resolve_codex_event_idle_timeout(est_tokens=12, env_value=0) is None
    assert resolve_codex_hard_timeout(env_value=0) is None
    assert resolve_codex_hard_timeout(env_value=1500) == 1500.0


def test_provider_stale_timeout_does_not_replace_codex_event_idle(monkeypatch):
    """providers.<id>.stale_timeout_seconds is the generic non-Codex detector.

    Codex Responses uses parsed-event idle. A 300s provider stale must not
    silently become the event-idle budget. Historical 300s lives on aux
    compression, not this watchdog.
    """
    from agent.chat_completion_helpers import resolve_codex_event_idle_timeout

    monkeypatch.setattr(
        "hermes_cli.timeouts.get_provider_stale_timeout",
        lambda provider, model=None: 300.0,
    )
    assert resolve_codex_event_idle_timeout(est_tokens=200_000) == 180.0


def test_historical_aux_compression_floor_remains_300():
    from agent.auxiliary_client import (
        _COMPRESSION_TIMEOUT_FLOOR_SECONDS,
        _effective_aux_timeout,
    )

    assert _COMPRESSION_TIMEOUT_FLOOR_SECONDS == 300.0
    assert _effective_aux_timeout("compression", None) >= 300.0
    assert _effective_aux_timeout("compression", 120.0) == 120.0
    assert _effective_aux_timeout("title", None) < 300.0
