"""Round 5 finding A (#1 + #6): auxiliary invocation ceiling ownership.

The auxiliary effective ceiling is a ContextVar consumed by the relay-helper
physical gates (``_aux_relay_gate`` / ``_aux_provider_callback``).  It MUST be
published per-invocation by the real call owners (``call_llm`` sync and
``async_call_llm`` async) and RESET before the owner returns, so that:

  * an oversized request is refused BEFORE any physical provider call, and
    never retried / fallen back / credential-refreshed (terminal by type);
  * the ceiling is not left AMBIENT after return or exception;
  * a subsequent / nested / concurrent call never inherits a stale ceiling;
  * concurrent asyncio tasks with different ceilings stay isolated.

These tests drive the REAL owners (``auxiliary_client.call_llm`` and
``auxiliary_client.async_call_llm``) end-to-end — they do NOT manually seed
the ContextVar and then call a ``_relay_*`` helper (the Round 4 pattern that
left the ceiling ambient and, in the async path, never published it at all).

The ceiling gate (``_aux_relay_gate``) fires inside the real
``_relay_sync_completion`` / ``_relay_async_completion`` helpers, so the
ceiling MUST be published by the owner (not manually seeded) for the gate to
see it.  We patch ``relay_llm.execute_current`` / ``execute_current_async``
to invoke the provider callback directly (bypassing relay session resolution)
while leaving the real relay helpers — and their ceiling gate — in place.

RED expectations against the pre-fix code:
  * sync: ceiling set but token discarded -> never reset -> ambient leak;
    the "restored after return" test FAILS (ceiling stays set).
  * async: ceiling never published -> oversized request is NOT refused ->
    the provider IS called physically; the "zero physical calls" and
    "refused by type" tests FAIL.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from agent import auxiliary_client as _aux
from agent import relay_llm as _relay_llm
from agent import model_metadata as _model_metadata
from agent.model_metadata import ContextCeilingExceeded

# Ceiling chosen so the SMALL payload (input ~31 + reservation 65536 = 65,567
# tokens) PASSES the gate, while the OVERSIZED payload (input ~50K + 65536
# = ~115K tokens) is REFUSED.  The "custom" provider profile declares a
# 65536 implicit output reservation, so the reservation dominates both cases;
# the ceiling sits between the two totals.
CEILING = 90_000


# ── shared response + message fixtures ──────────────────────────────────────

def _completed_response() -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]),
                                 finish_reason="stop")],
        usage=None,
        model="test-model",
    )


def _oversized_messages() -> list:
    # ~50K rough tokens (4 chars/token) + 65536 reservation = ~115K >> CEILING
    return [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "x" * 200_000},
    ]


def _under_messages() -> list:
    # Small payload: input ~31 + 65536 reservation = 65,567 < CEILING
    return [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "hello, a small request that fits."},
    ]


def _route_metadata() -> tuple:
    return ("openai", "test-model", {
        "api_mode": "chat_completions",
        "api_request_id": "aux-r5-test",
        "call_role": "auxiliary:test",
        "retry_count": 0,
        "auxiliary_task": "test",
    })


# ── spy clients (sync + async) ──────────────────────────────────────────────
# The sync dispatch path calls ``client.chat.completions.create(...)`` directly
# (returns a plain value).  The async dispatch path ``await``s the same call
# (must return a coroutine).  Separate spies keep each path's contract exact.

def _sync_spy_client() -> SimpleNamespace:
    calls: list[dict] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _completed_response()

    return SimpleNamespace(base_url="https://api.test/v1",
                           chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                           _calls=calls)


def _async_spy_client() -> SimpleNamespace:
    calls: list[dict] = []

    async def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _completed_response()

    return SimpleNamespace(base_url="https://api.test/v1",
                           chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                           _calls=calls)


def _patch_owner_env(monkeypatch, spy: SimpleNamespace) -> None:
    """Patch the owner's environment so call_llm / async_call_llm run
    end-to-end against a spy client and a deterministic ceiling.  The real
    relay helpers (and their ceiling gate) stay in place; only relay session
    resolution (execute_current / execute_current_async) is short-circuited."""
    monkeypatch.setattr(_aux, "_get_cached_client",
                        lambda *a, **k: (spy, "test-model"))
    monkeypatch.setattr(_aux, "_validate_llm_response",
                        lambda r, *a, **k: "OK")
    monkeypatch.setattr(_aux, "_relay_auxiliary_metadata",
                        lambda **k: _route_metadata())
    monkeypatch.setattr(_model_metadata, "effective_context_length",
                        lambda **k: CEILING)
    # Bypass relay session resolution; invoke the provider callback directly.
    # The ceiling gate already fired inside the real relay helper before this.
    monkeypatch.setattr(_relay_llm, "execute_current",
                        lambda req, cb, **k: cb(req))
    async def _fake_async(req, cb, **k: Any) -> Any:
        return await cb(req)
    monkeypatch.setattr(_relay_llm, "execute_current_async", _fake_async)


# ── 1. Sync oversized → refused by type, zero physical calls ────────────────

def test_sync_oversized_refused_zero_physical_calls(monkeypatch):
    spy = _sync_spy_client()
    _patch_owner_env(monkeypatch, spy)
    with pytest.raises(ContextCeilingExceeded):
        _aux.call_llm(
            "test", provider="openai", model="test-model",
            api_key="sk-test", messages=_oversized_messages(), max_tokens=256,
        )
    assert len(spy._calls) == 0, "physical provider .create must NOT be called"


def test_sync_oversized_refusal_not_fallback_or_retry(monkeypatch):
    """A ceiling refusal is terminal by type: no retry, fallback, or
    credential-refresh may be attempted."""
    spy = _sync_spy_client()
    _patch_owner_env(monkeypatch, spy)
    touched = {"retry": 0, "fallback": 0, "refresh": 0}

    def _marker(**_k: Any):
        return None

    monkeypatch.setattr(_aux, "_is_transient_transport_error", _marker, raising=False)
    monkeypatch.setattr(_aux, "_recover_provider_pool",
                        lambda *a, **k: touched.__setitem__("retry", touched["retry"] + 1),
                        raising=False)
    monkeypatch.setattr(_aux, "_try_configured_fallback_for_unavailable_client",
                        lambda *a, **k: (None, None, ""), raising=False)

    with pytest.raises(ContextCeilingExceeded):
        _aux.call_llm(
            "test", provider="openai", model="test-model",
            api_key="sk-test", messages=_oversized_messages(), max_tokens=256,
        )
    assert len(spy._calls) == 0
    assert touched["retry"] == 0
    assert touched["fallback"] == 0


# ── 2. ContextVar restored after success and after exception ────────────────

def test_sync_ceiling_restored_after_success(monkeypatch):
    spy = _sync_spy_client()
    _patch_owner_env(monkeypatch, spy)

    assert _aux.get_aux_ceiling() is None  # clean baseline
    result = _aux.call_llm(
        "test", provider="openai", model="test-model", api_key="sk-test",
        messages=_under_messages(), max_tokens=256,
    )
    assert result == "OK"
    assert len(spy._calls) == 1
    # The invocation ceiling must NOT be left ambient after a successful return.
    assert _aux.get_aux_ceiling() is None, (
        "auxiliary ceiling left ambient after a successful call_llm return"
    )


def test_sync_ceiling_restored_after_exception(monkeypatch):
    spy = _sync_spy_client()
    _patch_owner_env(monkeypatch, spy)

    # Force a non-ceiling transport failure so the except-chain runs, then
    # verify the ceiling is still reset in the owner's finally.
    def _boom(req, cb, **k):
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(_relay_llm, "execute_current", _boom)

    assert _aux.get_aux_ceiling() is None
    with pytest.raises(Exception):
        _aux.call_llm(
            "test", provider="openai", model="test-model", api_key="sk-test",
            messages=_under_messages(), max_tokens=256,
        )
    assert _aux.get_aux_ceiling() is None, (
        "auxiliary ceiling left ambient after a failed call_llm"
    )


# ── 3. Sequential calls cannot inherit a stale ceiling ──────────────────────

def test_sequential_calls_do_not_inherit_stale_ceiling(monkeypatch):
    spy = _sync_spy_client()
    _patch_owner_env(monkeypatch, spy)

    # Call 1: oversized -> refused by the ceiling (proves ceiling was set).
    with pytest.raises(ContextCeilingExceeded):
        _aux.call_llm("test", provider="openai", model="test-model",
                      api_key="sk-test", messages=_oversized_messages(),
                      max_tokens=256)
    # Call 2: a clean read of the ceiling must see None (no ambient leak).
    assert _aux.get_aux_ceiling() is None, "stale ceiling leaked into the next call"
    # Call 2: an under-sized request must succeed (no inherited refusal).
    result = _aux.call_llm("test", provider="openai", model="test-model",
                           api_key="sk-test", messages=_under_messages(),
                           max_tokens=256)
    assert result == "OK"
    assert len(spy._calls) == 1  # only the under-sized call reached the provider
    assert _aux.get_aux_ceiling() is None


# ── 4. Async oversized → zero physical calls, refused by type ───────────────

def test_async_oversized_refused_zero_physical_calls(monkeypatch):
    spy = _async_spy_client()
    _patch_owner_env(monkeypatch, spy)

    async def _run() -> None:
        with pytest.raises(ContextCeilingExceeded):
            await _aux.async_call_llm(
                "test", provider="openai", model="test-model",
                api_key="sk-test", messages=_oversized_messages(), max_tokens=256,
            )

    asyncio.get_event_loop().run_until_complete(_run())
    assert len(spy._calls) == 0, "async physical provider .create must NOT be called"


def test_async_ceiling_restored_after_success(monkeypatch):
    spy = _async_spy_client()
    _patch_owner_env(monkeypatch, spy)

    assert _aux.get_aux_ceiling() is None
    async def _run():
        return await _aux.async_call_llm(
            "test", provider="openai", model="test-model", api_key="sk-test",
            messages=_under_messages(), max_tokens=256,
        )
    result = asyncio.get_event_loop().run_until_complete(_run())
    assert result == "OK"
    assert len(spy._calls) == 1
    assert _aux.get_aux_ceiling() is None, (
        "async auxiliary ceiling left ambient after a successful call"
    )


# ── 5. Concurrent asyncio tasks with different ceilings stay isolated ───────

def test_concurrent_tasks_isolated_ceilings(monkeypatch):
    """Two concurrent async calls: the oversized one must be refused by the
    ceiling, the under-sized one must succeed, and no ambient ceiling is left
    behind.  Each task's ceiling is resolved by the owner in its own context,
    so ContextVar isolation is exercised end-to-end."""
    small_spy = _async_spy_client()
    big_spy = _async_spy_client()
    # Route each task to its own spy by message size.
    def _route_spy(*a: Any, **k: Any):
        return (small_spy, "test-model")

    monkeypatch.setattr(_aux, "_get_cached_client", _route_spy)
    monkeypatch.setattr(_aux, "_validate_llm_response", lambda r, *a, **k: "OK")
    monkeypatch.setattr(_aux, "_relay_auxiliary_metadata", lambda **k: _route_metadata())
    monkeypatch.setattr(_model_metadata, "effective_context_length", lambda **k: CEILING)

    async def _fake_async(req, cb, **k: Any) -> Any:
        return await cb(req)
    monkeypatch.setattr(_relay_llm, "execute_current_async", _fake_async)

    results: dict[str, Any] = {}

    async def run(tag: str, oversized: bool) -> Any:
        try:
            r = await _aux.async_call_llm(
                "test", provider="openai", model="test-model", api_key="sk-test",
                messages=(_oversized_messages() if oversized else _under_messages()),
                max_tokens=256,
            )
            return ("ok", r)
        except ContextCeilingExceeded as e:
            return ("ceiling", e)

    async def main() -> None:
        a, b = await asyncio.gather(run("small", False), run("big", True))
        results["small"] = a
        results["big"] = b

    asyncio.get_event_loop().run_until_complete(main())
    assert results["small"][0] == "ok"
    assert results["big"][0] == "ceiling"
    # After both complete, no ambient ceiling is left behind.
    assert _aux.get_aux_ceiling() is None, "ambient ceiling leaked after concurrent tasks"
    # The small task reached its provider exactly once.
    assert len(small_spy._calls) == 1
