"""Auxiliary tasks read ``fallback_providers`` (main-agent naming convention) in addition to
the legacy ``fallback_chain`` spelling.

Primary config key is ``fallback_providers``; ``fallback_chain`` remains recognized for
backward compatibility. Both resolve to the same per-task fallback chain used by
``_try_configured_fallback_chain`` and ``_ladder_provider_fallback``.

Tests cover:
  - ``fallback_providers`` is read as the primary source
  - ``fallback_chain`` still works (backward compatibility)
  - ``fallback_providers`` takes precedence when both are set
  - Empty/missing chain resolves to no fallback
  - Labels use ``fallback_providers[{i}]`` naming
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import agent.auxiliary_client as aux
from agent import auxiliary_error_policy as aux_errors


@pytest.fixture(autouse=True)
def _fresh_unhealthy_cache():
    aux._reset_aux_unhealthy_cache()
    yield
    aux._reset_aux_unhealthy_cache()


def _set_task_config(task: str, **kwargs):
    """Patch ``_get_auxiliary_task_config`` to return a fixed block for *task*."""
    patcher = patch.object(
        aux, "_get_auxiliary_task_config", return_value=dict(kwargs)
    )
    patcher.start()
    return patcher


def _client(base_url: str, create):
    client = MagicMock()
    client.base_url = base_url
    client.chat.completions.create = create
    return client


def _ok_response(text: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.choices[0].message.tool_calls = None
    return resp


# ── _get_auxiliary_fallback_chain ──────────────────────────────────────────────

def test_reads_fallback_providers_primary():
    """``fallback_providers`` is read as the primary source."""
    with _set_task_config("vision", fallback_providers=[
        {"provider": "openrouter", "model": "gemini-a"},
    ]):
        chain = aux._get_auxiliary_fallback_chain("vision")
    assert chain == [{"provider": "openrouter", "model": "gemini-a"}]


def test_reads_fallback_chain_legacy_when_no_providers():
    """Legacy ``fallback_chain`` still works when ``fallback_providers`` is absent."""
    with _set_task_config("vision", fallback_chain=[
        {"provider": "openrouter", "model": "gemini-a"},
    ]):
        chain = aux._get_auxiliary_fallback_chain("vision")
    assert chain == [{"provider": "openrouter", "model": "gemini-a"}]


def test_fallback_providers_wins_when_both_set():
    """When both are set, ``fallback_providers`` takes precedence."""
    with _set_task_config("vision", fallback_providers=[
        {"provider": "openrouter", "model": "gemini-a"},
    ], fallback_chain=[
        {"provider": "nous", "model": "gemini-b"},
    ]):
        chain = aux._get_auxiliary_fallback_chain("vision")
    assert chain == [{"provider": "openrouter", "model": "gemini-a"}]


def test_empty_chain_returns_none():
    """An empty list or absent key returns None."""
    with _set_task_config("vision", fallback_providers=[]):
        assert aux._get_auxiliary_fallback_chain("vision") is None
    with _set_task_config("vision"):
        assert aux._get_auxiliary_fallback_chain("vision") is None


def test_no_task_returns_none():
    assert aux._get_auxiliary_fallback_chain(None) is None
    assert aux._get_auxiliary_fallback_chain("") is None


# ── _fallback_chain_entry reads fallback_providers labels ─────────────────────

def test_entry_label_fallback_providers_index():
    """``_fallback_chain_entry`` resolves labels like ``fallback_providers[0](openrouter)``."""
    with _set_task_config("vision", fallback_providers=[
        {"provider": "openrouter", "model": "gemini-a"},
        {"provider": "nous", "model": "gemini-b"},
    ]):
        entry = aux._fallback_chain_entry("vision", "fallback_providers[1](nous)")
    assert entry == {"provider": "nous", "model": "gemini-b"}


def test_entry_label_fallback_chain_still_works():
    """Legacy ``fallback_chain[0](openrouter)`` labels still resolve."""
    with _set_task_config("vision", fallback_chain=[
        {"provider": "openrouter", "model": "gemini-a"},
    ]):
        entry = aux._fallback_chain_entry("vision", "fallback_chain[0](openrouter)")
    assert entry == {"provider": "openrouter", "model": "gemini-a"}


def test_entry_label_malformed_returns_none():
    with _set_task_config("vision", fallback_providers=[
        {"provider": "openrouter", "model": "gemini-a"},
    ]):
        assert aux._fallback_chain_entry("vision", "not-a-label") is None
        assert aux._fallback_chain_entry("vision", "fallback_providers[99](x)") is None


# ── _try_configured_fallback_chain walks fallback_providers ────────────────────

def test_configured_chain_walks_fallback_providers_in_order():
    """`_try_configured_fallback_chain` tries each entry in order, skipping failures."""
    lane_a = _client("http://127.0.0.1:1/v1", MagicMock(side_effect=Exception("down")))
    lane_b = _client("http://127.0.0.1:2/v1", MagicMock(return_value=_ok_response("served-by-b")))

    with _set_task_config("vision", fallback_providers=[
        {"provider": "openrouter", "model": "model-a"},
        {"provider": "openrouter", "model": "model-b"},
    ]):
        client, model, label = aux._try_configured_fallback_chain(
            task="vision", failed_provider="openrouter",
            failed_model="model-a", failed_base_url="http://127.0.0.1:1/v1",
        )

    assert client is lane_b
    assert model == "model-b"
    assert label == "fallback_providers[1](openrouter)"


def test_configured_chain_exhaustion_returns_none():
    """When every entry fails, returns (None, None, "")."""
    lane_a = _client("http://127.0.0.1:1/v1", MagicMock(side_effect=Exception("down")))
    lane_b = _client("http://127.0.0.1:2/v1", MagicMock(side_effect=Exception("down")))

    with _set_task_config("vision", fallback_providers=[
        {"provider": "openrouter", "model": "model-a"},
        {"provider": "openrouter", "model": "model-b"},
    ]):
        client, model, label = aux._try_configured_fallback_chain(
            task="vision", failed_provider="openrouter",
            failed_model="model-a", failed_base_url="http://127.0.0.1:1/v1",
        )

    assert client is None and model is None and label == ""


def test_no_chain_returns_none_immediately():
    with _set_task_config("vision"):
        assert aux._try_configured_fallback_chain(
            task="vision", failed_provider="openrouter", failed_model="model-a",
        ) == (None, None, "")


# ── Backward compatibility: fallback_chain still walked ──────────────────────────

def test_configured_chain_walks_legacy_fallback_chain():
    """Legacy ``fallback_chain`` entries are still walked when ``fallback_providers`` is absent."""
    lane_b = _client("http://127.0.0.1:2/v1", MagicMock(return_value=_ok_response("served-by-legacy")))

    with _set_task_config("vision", fallback_chain=[
        {"provider": "openrouter", "model": "model-a"},
        {"provider": "openrouter", "model": "model-b"},
    ]):
        client, model, label = aux._try_configured_fallback_chain(
            task="vision", failed_provider="openrouter",
            failed_model="model-a", failed_base_url="http://127.0.0.1:1/v1",
        )

    assert client is lane_b
    assert model == "model-b"
    # Label reflects the legacy spelling since that's what the config used
    assert label.startswith("fallback_chain[")


# ── _ladder_provider_fallback respects fallback_providers ──────────────────────

def test_ladder_provider_fallback_uses_configured_chain():
    """`_ladder_provider_fallback` calls `_try_configured_fallback_chain` which reads
    ``fallback_providers`` from task config."""
    healthy = MagicMock(name="healthy-fallback")
    route = aux._LadderRoute(
        None, "vision", "", False, "", "openrouter", "model-a",
        None, None, None, None, {"provider": "openrouter"}, None, None,
    )

    with _set_task_config("vision", fallback_providers=[
        {"provider": "openrouter", "model": "gemini-b"},
    ]), patch.object(aux, "_try_configured_fallback_chain",
                     return_value=(healthy, "gemini-b", "fallback_providers[0](openrouter)")):
        gen = aux._ladder_provider_fallback(
            aux_errors.Failure(
                Exception("rate limited"), "capacity error", lane=None, client=None
            ),
            route,
        )
        step = next(gen)
    assert step.kind == "fallback"
    assert step.args == (healthy, "gemini-b", "fallback_providers[0](openrouter)")


# ── Integration: call_llm uses fallback_providers chain ────────────────────────

def test_call_llm_falls_back_through_configured_providers():
    """`call_llm` falls back through ``fallback_providers`` entries when primary fails."""
    primary = _client("http://127.0.0.1:1/v1", MagicMock(side_effect=Exception("primary down")))
    lane_b = _client("http://127.0.0.1:2/v1", MagicMock(return_value=_ok_response("fallback-ok")))

    # Patch resolution so primary is the resolved client, then failure triggers fallback walk.
    with patch.object(aux, "_resolve_task_provider_model",
                      return_value=("openrouter", "model-a", None, None, None)), \
         patch.object(aux, "_get_cached_client", return_value=(primary, "model-a")), \
         patch.object(aux, "_try_configured_fallback_chain",
                      return_value=(lane_b, "model-b", "fallback_providers[0](openrouter)")), \
         patch.object(aux, "_try_payment_fallback", return_value=(None, None, "")):
        result = aux.call_llm(
            task="vision", messages=[{"role": "user", "content": "hi"}],
        )

    assert result.choices[0].message.content == "fallback-ok"
    assert lane_b.chat.completions.create.call_count == 1


# ── _fallback_entry_timeout reads per-entry timeout from fallback_providers ─────

def test_entry_timeout_from_fallback_providers_entry():
    """`_fallback_entry_timeout` reads a per-entry ``timeout`` from ``fallback_providers``."""
    with _set_task_config("vision", fallback_providers=[
        {"provider": "openrouter", "model": "gemini-a", "timeout": 45},
    ]):
        timeout = aux._fallback_entry_timeout("vision", "fallback_providers[0](openrouter)")
    assert timeout == 45.0


def test_entry_timeout_from_fallback_chain_entry_still_works():
    """Legacy ``fallback_chain`` entries also expose per-entry ``timeout``."""
    with _set_task_config("vision", fallback_chain=[
        {"provider": "openrouter", "model": "gemini-a", "timeout": 45},
    ]):
        timeout = aux._fallback_entry_timeout("vision", "fallback_chain[0](openrouter)")
    assert timeout == 45.0
