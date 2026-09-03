"""Round 5 finding B (P1 from review 5049973836): auxiliary fallback
per-destination ceiling rebinding.

The auxiliary effective ceiling is a ContextVar published ONCE per
``call_llm`` / ``async_call_llm`` for the INITIAL destination (``_call_llm_impl``
/ ``_async_call_llm_impl`` resolve ``effective_context_length`` for the initial
``final_model`` and publish it via ``set_aux_ceiling``).  The relay-helper
physical gates (``_aux_relay_gate``) read that ambient value.  Before the fix,
the fallback candidates (``_call_fallback_candidate_sync`` /
``_call_fallback_candidate_async``) called the SAME relay helpers WITHOUT
rebinding the ceiling to the fallback destination — so a fallback to a
different provider/model/base_url inherited the INITIAL destination's ceiling.

Discriminating schedule (large initial → small fallback):

* Initial destination ceiling 900K, fallback destination ceiling 128K.
* A ~200K request passes the initial 900K gate, the initial provider then
  fails transiently, and the fallback candidate is invoked.
* BUGGY (no rebind): the fallback gate reads the ambient 900K ceiling and
  DISPATCHES the request to the 128K fallback model — the pre-I/O bound is
  gone (a ~200K request physically reaches a 128K model).
* FIXED (rebind): the fallback gate reads the rebound 128K ceiling and REFUSES
  the ~200K request — the fallback model is never physically called.

These tests drive the REAL owners (``call_llm`` / ``async_call_llm``)
end-to-end: the initial provider fails with a transient transport error, the
fallback chain returns a destination spy, and the REAL
``_call_fallback_candidate_*`` helper runs the REAL relay gate.
``effective_context_length`` is patched to resolve a DIFFERENT ceiling per
destination (keyed on model) so the gate sees whichever ceiling the owner
publishes for the destination it is dispatching to.

RED expectations against the pre-fix code (no rebind):
  * sync/async large→small: the fallback provider IS physically called (stale
    900K gate passes the ~200K request) and ``effective_context_length`` is
    NEVER called for the fallback destination.

GREEN expectations after the fix (rebind per destination with scoped
token/reset):
  * sync/async large→small: the fallback provider is NOT physically called;
    the gate fired under the 128K ceiling; ``effective_context_length`` WAS
    called for the fallback destination; no ambient ceiling is left behind
    after the call returns.
  * the ceiling resolver for the fallback carries the destination credential
    (api_key) so capability resolution is credential-aware.
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

# Per-destination ceilings (the resolver returns these keyed on model).
INITIAL_MODEL = "initial-model"
FALLBACK_MODEL = "fallback-model"
INITIAL_LARGE = 900_000   # large initial destination
FALLBACK_SMALL = 128_000  # small fallback destination


def _completed_response() -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]),
                                 finish_reason="stop")],
        usage=None,
        model="test-model",
    )


def _request_between_ceilings() -> list:
    # 600K chars → budget total ≈ 154K tokens (verified empirically).  This
    # sits in the discriminating band (128K, 900K): it PASSES the initial
    # 900K gate (so the initial call is attempted and then fails transiently)
    # and is REFUSED by the fallback destination's 128K ceiling.
    #   * BUGGY (no rebind): fallback gate reads the ambient 900K initial
    #     ceiling → 154K < 900K → DISPATCHES to the 128K model (defect).
    #   * FIXED (rebind):    fallback gate reads the rebound 128K ceiling →
    #     154K > 128K → REFUSES (correct pre-I/O bound).
    return [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "x" * 600_000},
    ]


def _sync_spy_client(fail_first: bool = True) -> SimpleNamespace:
    # fail_first=True (default): the INITIAL provider always fails with a
    # transient transport error — so BOTH the first attempt AND the
    # same-provider retry fail, forcing fall-through to the fallback candidate
    # (the async path unconditionally retries once on the same provider before
    # escalating to the fallback candidate; a fail-once spy would let that
    # retry succeed and never reach the fallback under test).
    calls: list[dict] = []
    state = {"count": 0}

    def create(**kwargs: Any) -> Any:
        state["count"] += 1
        if fail_first:
            raise RuntimeError("connection error")  # transient → triggers fallback
        calls.append(kwargs)
        return _completed_response()

    return SimpleNamespace(base_url="https://api.test/v1", api_key="sk-fallback",
                           chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                           _calls=calls,
                           _hermes_fallback_destination=None)


def _async_spy_client(fail_first: bool = True) -> SimpleNamespace:
    calls: list[dict] = []
    state = {"count": 0}

    async def create(**kwargs: Any) -> Any:
        state["count"] += 1
        if fail_first:
            raise RuntimeError("connection error")
        calls.append(kwargs)
        return _completed_response()

    return SimpleNamespace(base_url="https://api.test/v1",
                           chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                           _calls=calls,
                           _hermes_fallback_destination=None)


def _patch_fallback_env(
    monkeypatch,
    initial_spy: SimpleNamespace,
    fallback_spy: SimpleNamespace,
    initial_ceiling: int,
    fallback_ceiling: int,
    resolver_calls: list,
) -> None:
    """Patch the owner env so the REAL fallback path runs:
    initial provider fails transiently → fallback chain returns fallback_spy
    → REAL _call_fallback_candidate_* runs the REAL relay gate.

    ``effective_context_length`` resolves a DIFFERENT ceiling per destination
    (keyed on model) and records every call so tests can assert whether the
    fallback destination was resolved.
    """
    def _get_cached_client(provider, model, *a, **k):
        if model == FALLBACK_MODEL:
            return (fallback_spy, FALLBACK_MODEL)
        return (initial_spy, INITIAL_MODEL)

    monkeypatch.setattr(_aux, "_get_cached_client", _get_cached_client, raising=False)
    monkeypatch.setattr(_aux, "_is_transient_transport_error", lambda e: True, raising=False)
    monkeypatch.setattr(_aux, "_transient_retry_count", lambda: 0, raising=False)
    monkeypatch.setattr(_aux, "_is_connection_error", lambda e: True, raising=False)
    for name in ("_is_payment_error", "_is_auth_error", "_is_rate_limit_error",
                 "_is_model_incompatible_error", "_is_invalid_aux_response_error"):
        monkeypatch.setattr(_aux, name, lambda e: False, raising=False)

    monkeypatch.setattr(_aux, "_try_configured_fallback_chain",
                        lambda *a, **k: (fallback_spy, FALLBACK_MODEL, "configured"),
                        raising=False)
    monkeypatch.setattr(_aux, "_try_main_fallback_chain",
                        lambda *a, **k: (None, None, ""), raising=False)
    monkeypatch.setattr(_aux, "_try_payment_fallback",
                        lambda *a, **k: (None, None, ""), raising=False)
    monkeypatch.setattr(_aux, "_try_main_agent_model_fallback",
                        lambda *a, **k: (None, None, ""), raising=False)

    # Attach the fallback destination directly on the spy for determinism.
    fallback_spy._hermes_fallback_destination = _aux._FallbackDestination(
        provider="openai", base_url="https://api.test/v1",
        api_mode="chat_completions", model=FALLBACK_MODEL,
    )
    monkeypatch.setattr(_aux, "_fallback_entry_api_key",
                        lambda entry: "sk-fallback", raising=False)

    def _ecl(model="", base_url="", api_key="", provider="", **k):
        resolver_calls.append({"model": model, "base_url": base_url, "api_key": api_key})
        if model == FALLBACK_MODEL:
            return fallback_ceiling
        return initial_ceiling

    monkeypatch.setattr(_model_metadata, "effective_context_length", _ecl, raising=False)

    monkeypatch.setattr(_aux, "_validate_llm_response", lambda r, *a, **k: "OK", raising=False)
    monkeypatch.setattr(_aux, "_relay_auxiliary_metadata",
                        lambda **k: ("openai", "test-model", {
                            "api_mode": "chat_completions",
                            "api_request_id": "aux-r5b-test",
                            "call_role": "auxiliary:test",
                            "retry_count": 0,
                            "auxiliary_task": "test",
                        }), raising=False)
    monkeypatch.setattr(_relay_llm, "execute_current", lambda req, cb, **k: cb(req), raising=False)
    async def _fake_async(req, cb, **k: Any) -> Any:
        return await cb(req)
    monkeypatch.setattr(_relay_llm, "execute_current_async", _fake_async, raising=False)

    # The async fallback path converts the sync fallback client to an async
    # client via _to_async_client (which builds a REAL AsyncOpenAI and would
    # issue actual HTTP I/O).  Short-circuit it to return the fallback spy
    # directly, so the relay helper's default create (client.chat.completions.
    # create) resolves to the spy's async create.  (The initial path passes
    # create=_acreate explicitly, so it is unaffected.)
    monkeypatch.setattr(_aux, "_to_async_client",
                        lambda client, model, is_vision=False: (client, model),
                        raising=False)


# ── 1. Sync large→small: fallback refused under its OWN ceiling ─────────────

def test_sync_fallback_large_to_small_refused_under_fallback_ceiling(monkeypatch):
    """A ~200K request that legitimately passes the initial 900K ceiling must
    be REFUSED by the fallback destination's 128K ceiling — not physically
    dispatched to a 128K model under the stale 900K gate."""
    initial_spy = _sync_spy_client(fail_first=True)
    fallback_spy = _sync_spy_client(fail_first=False)
    resolver_calls: list = []
    _patch_fallback_env(monkeypatch, initial_spy, fallback_spy,
                        initial_ceiling=INITIAL_LARGE, fallback_ceiling=FALLBACK_SMALL,
                        resolver_calls=resolver_calls)

    with pytest.raises(ContextCeilingExceeded):
        _aux.call_llm(
            "test", provider="openai", model=INITIAL_MODEL,
            api_key="sk-initial", messages=_request_between_ceilings(), max_tokens=None,
        )

    assert len(fallback_spy._calls) == 0, (
        "fallback provider physically called under the initial destination's "
        "stale ceiling — the per-destination rebinding is missing"
    )
    assert any(c["model"] == FALLBACK_MODEL for c in resolver_calls), (
        "effective_context_length was never called for the fallback "
        "destination — the ceiling was not rebound per-destination"
    )


# ── 2. Async large→small: same contract on the async path ────────────────────

def test_async_fallback_large_to_small_refused_under_fallback_ceiling(monkeypatch):
    initial_spy = _async_spy_client(fail_first=True)
    fallback_spy = _async_spy_client(fail_first=False)
    resolver_calls: list = []
    _patch_fallback_env(monkeypatch, initial_spy, fallback_spy,
                        initial_ceiling=INITIAL_LARGE, fallback_ceiling=FALLBACK_SMALL,
                        resolver_calls=resolver_calls)

    async def _run() -> None:
        with pytest.raises(ContextCeilingExceeded):
            await _aux.async_call_llm(
                "test", provider="openai", model=INITIAL_MODEL,
                api_key="sk-initial", messages=_request_between_ceilings(), max_tokens=None,
            )

    asyncio.get_event_loop().run_until_complete(_run())
    assert len(fallback_spy._calls) == 0, (
        "async fallback provider physically called under the initial "
        "destination's stale ceiling"
    )
    assert any(c["model"] == FALLBACK_MODEL for c in resolver_calls), (
        "async effective_context_length was never called for the fallback "
        "destination — the ceiling was not rebound per-destination"
    )


# ── 3. Ceiling restored after the fallback attempt (token/reset) ────────────

def test_fallback_ceiling_restored_after_attempt(monkeypatch):
    """After the fallback attempt completes (ceiling-refused), the ambient
    ceiling must be reset — not left at the fallback's rebound value nor the
    initial's stale value."""
    initial_spy = _sync_spy_client(fail_first=True)
    fallback_spy = _sync_spy_client(fail_first=False)
    resolver_calls: list = []
    _patch_fallback_env(monkeypatch, initial_spy, fallback_spy,
                        initial_ceiling=INITIAL_LARGE, fallback_ceiling=FALLBACK_SMALL,
                        resolver_calls=resolver_calls)

    assert _aux.get_aux_ceiling() is None  # clean baseline
    with pytest.raises(ContextCeilingExceeded):
        _aux.call_llm(
            "test", provider="openai", model=INITIAL_MODEL,
            api_key="sk-initial", messages=_request_between_ceilings(), max_tokens=None,
        )
    assert _aux.get_aux_ceiling() is None, (
        "auxiliary ceiling left ambient after the fallback attempt — the "
        "scoped token was not reset"
    )


# ── 4. Fallback resolution carries the destination credential ───────────────

def test_fallback_resolution_carries_credential_context(monkeypatch):
    """The fallback destination's ceiling resolution must carry the
    destination's credential (api_key) so capability resolution is
    credential-aware — matching the existing _candidate_context_window() path
    that already carries api_key for authenticated endpoint probing."""
    initial_spy = _sync_spy_client(fail_first=True)
    fallback_spy = _sync_spy_client(fail_first=False)
    resolver_calls: list = []
    _patch_fallback_env(monkeypatch, initial_spy, fallback_spy,
                        initial_ceiling=INITIAL_LARGE, fallback_ceiling=FALLBACK_SMALL,
                        resolver_calls=resolver_calls)

    with pytest.raises(ContextCeilingExceeded):
        _aux.call_llm(
            "test", provider="openai", model=INITIAL_MODEL,
            api_key="sk-initial", messages=_request_between_ceilings(), max_tokens=None,
        )

    fb_calls = [c for c in resolver_calls if c["model"] == FALLBACK_MODEL]
    assert fb_calls, "fallback destination was never resolved"
    assert fb_calls[0]["api_key"], (
        "fallback destination ceiling resolution did not carry the "
        "destination credential (api_key)"
    )
