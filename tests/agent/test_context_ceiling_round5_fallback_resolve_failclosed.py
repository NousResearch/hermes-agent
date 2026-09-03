"""Round 5 finding C (remaining P1 from andrexibiza's re-review of
8e109d375): auxiliary fallback ceiling-authority resolution must FAIL CLOSED.

Context authority is destination-scoped.  Once any authority-bearing
destination projection changes (provider, model, base URL, or credential),
the SOURCE destination's ambient ceiling must NOT authorize the new physical
provider attempt.

The rebind helpers (``_rebind_aux_ceiling_for_fallback`` / ``_rebind_aux_
ceiling_for_fallback_async`` in ``agent/auxiliary_context_ceiling.py``)
resolve the fallback destination's own ceiling and scope it for the attempt.
The defect: when that resolution FAILS -- the resolver raises, or it returns
``None`` / an unusable (non-positive / non-int) value -- the helpers did NOT
refuse.  They fell through to the ambient (SOURCE) ceiling:

* sync:  ``except Exception: _ceiling = None`` → ``return contextlib.nullcontext()``
* async: ``except Exception: _ceiling = None`` → ``yield None; return``

So the physical fallback attempt continued under the initial destination's
ceiling -- a different provider/model/credential authorized by an authority
that does not belong to it.  That is the P1.

The fix must fail CLOSED before provider I/O, using the EXISTING terminal-by-
type local refusal ``ContextCeilingExceeded`` (the fallback / transient-retry
/ credential-refresh chains MUST let it propagate: no provider I/O, no
fallback, no retry, no credential refresh).  It is deliberately NOT a provider
error (not an auth error), so it is never converted into a provider failure
that would trigger an unauthorized fallback/retry cycle.

RED expectations against the pre-fix code:
  * sync resolver raises / returns None → the fallback provider IS physically
    called (under the source destination's ambient ceiling); ``call_llm``
    returns a response instead of raising ``ContextCeilingExceeded``.
  * async resolver raises / returns None → same fail-open on the async path.

GREEN expectations after the fix (fail closed, typed, terminal, pre-I/O):
  * sync resolver raises / returns None → ``call_llm`` raises
    ``ContextCeilingExceeded``; the fallback provider is NOT physically
    called (I/O count == 0); the resolver WAS consulted for the fallback
    destination (proving we reached the rebind); the ambient ceiling is not
    left behind (no token leak / no context leak).
  * async: same contract on the async path.
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

INITIAL_MODEL = "initial-model"
FALLBACK_MODEL = "fallback-model"
INITIAL_LARGE = 900_000   # large initial (source) destination ceiling

# A request that PASSES the initial 900K ceiling (so the initial attempt is
# made, fails transiently, and falls through to the fallback) but would be
# refused by a small fallback ceiling.  Under the BUGGY fail-open behaviour
# the fallback gate reads the ambient 900K ceiling and DISPATCHES.
def _request_between_ceilings() -> list:
    return [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "x" * 600_000},  # ~154K tokens (verified)
    ]


def _completed_response() -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]),
                                 finish_reason="stop")],
        usage=None,
        model="test-model",
    )


def _sync_spy_client(fail_first: bool = True) -> SimpleNamespace:
    calls: list[dict] = []
    state = {"count": 0}

    def create(**kwargs: Any) -> Any:
        state["count"] += 1
        if fail_first:
            raise RuntimeError("connection error")  # transient → fallback
        calls.append(kwargs)
        return _completed_response()

    return SimpleNamespace(base_url="https://api.test/v1", api_key="sk-fallback",
                           chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                           _calls=calls, _hermes_fallback_destination=None)


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
                           _calls=calls, _hermes_fallback_destination=None)


def _patch_fallback_env(
    monkeypatch,
    initial_spy: SimpleNamespace,
    fallback_spy: SimpleNamespace,
    resolver_mode: str,      # "raise" | "none"
    resolver_calls: list,
) -> None:
    """Patch the owner env so the REAL fallback path runs, and the fallback
    destination's ceiling resolver either RAISES or returns None (the two
    unresolvable-authority cases).  The initial destination resolves fine so
    the initial attempt is made and then fails transiently, forcing fall-
    through to the fallback candidate under test."""
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

    fallback_spy._hermes_fallback_destination = _aux._FallbackDestination(
        provider="openai", base_url="https://api.test/v1",
        api_mode="chat_completions", model=FALLBACK_MODEL,
    )
    monkeypatch.setattr(_aux, "_fallback_entry_api_key",
                        lambda entry: "sk-fallback", raising=False)

    def _ecl(model="", base_url="", api_key="", provider="", **k):
        resolver_calls.append({"model": model, "base_url": base_url, "api_key": api_key})
        if model == FALLBACK_MODEL:
            if resolver_mode == "raise":
                raise RuntimeError("resolver blew up (capability probe failed)")
            return None  # unusable / no usable destination ceiling
        return INITIAL_LARGE

    monkeypatch.setattr(_model_metadata, "effective_context_length", _ecl, raising=False)

    monkeypatch.setattr(_aux, "_validate_llm_response", lambda r, *a, **k: "OK", raising=False)
    monkeypatch.setattr(_aux, "_relay_auxiliary_metadata",
                        lambda **k: ("openai", "test-model", {
                            "api_mode": "chat_completions",
                            "api_request_id": "aux-r5c-test",
                            "call_role": "auxiliary:test",
                            "retry_count": 0,
                            "auxiliary_task": "test",
                        }), raising=False)
    monkeypatch.setattr(_relay_llm, "execute_current", lambda req, cb, **k: cb(req), raising=False)
    async def _fake_async(req, cb, **k: Any) -> Any:
        return await cb(req)
    monkeypatch.setattr(_relay_llm, "execute_current_async", _fake_async, raising=False)
    monkeypatch.setattr(_aux, "_to_async_client",
                        lambda client, model, is_vision=False: (client, model),
                        raising=False)


# ── 1. Sync: resolver RAISES → fail closed, no fallback I/O ─────────────────

def test_sync_resolver_raises_fails_closed_no_fallback_io(monkeypatch):
    initial_spy = _sync_spy_client(fail_first=True)
    fallback_spy = _sync_spy_client(fail_first=False)
    resolver_calls: list = []
    _patch_fallback_env(monkeypatch, initial_spy, fallback_spy,
                        resolver_mode="raise", resolver_calls=resolver_calls)

    assert _aux.get_aux_ceiling() is None  # clean baseline
    with pytest.raises(ContextCeilingExceeded):
        _aux.call_llm(
            "test", provider="openai", model=INITIAL_MODEL,
            api_key="sk-initial", messages=_request_between_ceilings(), max_tokens=None,
        )

    assert len(fallback_spy._calls) == 0, (
        "FAIL OPEN: the unresolved fallback destination was physically called "
        "under the source destination's ambient ceiling — authority is "
        "destination-scoped and must not authorize a different provider"
    )
    assert any(c["model"] == FALLBACK_MODEL for c in resolver_calls), (
        "the fallback destination's ceiling was never consulted — the rebind "
        "path was not reached (or short-circuited before the fail-closed check)"
    )
    assert _aux.get_aux_ceiling() is None, (
        "auxiliary ceiling left ambient after the failed fallback attempt — "
        "a token/context leak (no scoped token should have been set or left)"
    )


# ── 2. Sync: resolver returns None → fail closed, no fallback I/O ────────────

def test_sync_resolver_none_fails_closed_no_fallback_io(monkeypatch):
    initial_spy = _sync_spy_client(fail_first=True)
    fallback_spy = _sync_spy_client(fail_first=False)
    resolver_calls: list = []
    _patch_fallback_env(monkeypatch, initial_spy, fallback_spy,
                        resolver_mode="none", resolver_calls=resolver_calls)

    assert _aux.get_aux_ceiling() is None
    with pytest.raises(ContextCeilingExceeded):
        _aux.call_llm(
            "test", provider="openai", model=INITIAL_MODEL,
            api_key="sk-initial", messages=_request_between_ceilings(), max_tokens=None,
        )

    assert len(fallback_spy._calls) == 0, (
        "FAIL OPEN: the resolver returned None (no usable ceiling) yet the "
        "fallback provider was still physically called under the source ceiling"
    )
    assert any(c["model"] == FALLBACK_MODEL for c in resolver_calls)
    assert _aux.get_aux_ceiling() is None, "context/token leak after the failed attempt"


# ── 3. Async: resolver RAISES → fail closed, no fallback I/O ─────────────────

def test_async_resolver_raises_fails_closed_no_fallback_io(monkeypatch):
    initial_spy = _async_spy_client(fail_first=True)
    fallback_spy = _async_spy_client(fail_first=False)
    resolver_calls: list = []
    _patch_fallback_env(monkeypatch, initial_spy, fallback_spy,
                        resolver_mode="raise", resolver_calls=resolver_calls)

    assert _aux.get_aux_ceiling() is None

    async def _run() -> None:
        with pytest.raises(ContextCeilingExceeded):
            await _aux.async_call_llm(
                "test", provider="openai", model=INITIAL_MODEL,
                api_key="sk-initial", messages=_request_between_ceilings(), max_tokens=None,
            )

    asyncio.get_event_loop().run_until_complete(_run())
    assert len(fallback_spy._calls) == 0, (
        "FAIL OPEN (async): unresolved fallback destination physically called "
        "under the source destination's ambient ceiling"
    )
    assert any(c["model"] == FALLBACK_MODEL for c in resolver_calls)
    assert _aux.get_aux_ceiling() is None, "async context/token leak after the failed attempt"


# ── 4. Async: resolver returns None → fail closed, no fallback I/O ───────────

def test_async_resolver_none_fails_closed_no_fallback_io(monkeypatch):
    initial_spy = _async_spy_client(fail_first=True)
    fallback_spy = _async_spy_client(fail_first=False)
    resolver_calls: list = []
    _patch_fallback_env(monkeypatch, initial_spy, fallback_spy,
                        resolver_mode="none", resolver_calls=resolver_calls)

    assert _aux.get_aux_ceiling() is None

    async def _run() -> None:
        with pytest.raises(ContextCeilingExceeded):
            await _aux.async_call_llm(
                "test", provider="openai", model=INITIAL_MODEL,
                api_key="sk-initial", messages=_request_between_ceilings(), max_tokens=None,
            )

    asyncio.get_event_loop().run_until_complete(_run())
    assert len(fallback_spy._calls) == 0, (
        "FAIL OPEN (async): resolver returned None yet the fallback provider "
        "was still physically called under the source ceiling"
    )
    assert any(c["model"] == FALLBACK_MODEL for c in resolver_calls)
    assert _aux.get_aux_ceiling() is None, "async context/token leak after the failed attempt"


# ── 5. Classification: the failure is a TYPED LOCAL refusal, not a provider error ─
#
# Distinguishes "failure to resolve destination authority" (this test) from
# "a resolved ceiling that rejects" and from "ordinary provider failure."
# A provider error would NOT be ContextCeilingExceeded and, if misclassified,
# would be swallowed by the credential-refresh / fallback `except Exception`
# chain.  Proving the type is ContextCeilingExceeded (terminal by type) is
# the deterministic classification contract.

def test_sync_failure_is_typed_local_refusal_not_provider_error(monkeypatch):
    initial_spy = _sync_spy_client(fail_first=True)
    fallback_spy = _sync_spy_client(fail_first=False)
    _patch_fallback_env(monkeypatch, initial_spy, fallback_spy,
                        resolver_mode="raise", resolver_calls=[])

    with pytest.raises(ContextCeilingExceeded) as exc_info:
        _aux.call_llm(
            "test", provider="openai", model=INITIAL_MODEL,
            api_key="sk-initial", messages=_request_between_ceilings(), max_tokens=None,
        )

    # It must be the TYPED local ceiling refusal, not a wrapped provider error.
    assert isinstance(exc_info.value, ContextCeilingExceeded)
    assert not isinstance(exc_info.value, (TypeError, ValueError))
    # Deterministic: it is NOT an auth error → never routed to credential refresh.
    assert _aux._is_auth_error(exc_info.value) is False
    # The reason must distinguish the unresolvable-authority refusal from the
    # resolved-ceiling-exceeded refusal (and from any provider failure).
    reason = getattr(exc_info.value, "reason", "")
    assert "could not be resolved" in reason, (
        f"fail-closed refusal reason does not identify an unresolvable "
        f"destination authority: {reason!r}"
    )
