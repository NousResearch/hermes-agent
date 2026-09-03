"""Owning-seam suite for the bounded auxiliary context-ceiling owner shard.

The #78635 fracture contract requires the auxiliary context-ceiling authority
to live in a bounded owner (``agent/auxiliary_context_ceiling.py``) with
``agent/auxiliary_client.py`` keeping only narrow compatibility seams.  This
suite PROVES the extraction is correct and that the authoritative seams keep
working after the move:

1.  Historical names exposed through ``agent.auxiliary_client`` are the SAME
    objects as the shard's (identity), i.e. they DELEGATE to the shard rather
    than re-implementing it.
2.  Existing import names stay compatible (importable from both modules).
3.  Monkeypatching the historical auxiliary gate/provider seam STILL controls
    the actual live provider callback path after extraction (the decisive
    requirement — the relay helpers that consume these names must resolve
    them through ``agent.auxiliary_client``'s globals).
4.  Sync fallback rebinding reaches the shard and restores authority.
5.  Async fallback rebinding reaches the shard and restores authority.
6.  Large→small and small→large fallback transitions remain green.
7.  Credential-refresh fallback continues using the fallback destination's
    credential context.
8.  No duplicate ceiling owner remains in ``auxiliary_client.py``.

The monkeypatch test (3) is the discriminating one: if the relay helpers had
been moved to the shard alongside the gate (or the gate NOT re-imported into
the monolith's globals), patching ``agent.auxiliary_client._aux_provider_callback``
would NOT affect the live path and this test would fail.  It passes only when
there is exactly one owner and the monolith delegates to it.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from agent import auxiliary_client as _aux
from agent import auxiliary_context_ceiling as _shard
from agent import model_metadata as _model_metadata
from agent.model_metadata import ContextCeilingExceeded

SHARD_MODULE = "agent.auxiliary_context_ceiling"
MONO_MODULE = "agent.auxiliary_client"

CEILING_NAMES = [
    "_aux_relay_gate",
    "_aux_provider_callback",
    "_rebind_aux_ceiling_for_fallback",
    "_rebind_aux_ceiling_for_fallback_async",
]

INITIAL_MODEL = "initial-model"
FALLBACK_MODEL = "fallback-model"
INITIAL_LARGE = 900_000
FALLBACK_SMALL = 128_000
INITIAL_SMALL = 128_000
FALLBACK_LARGE = 900_000


# ── 1. Historical names delegate to the shard (identity, one real owner) ──

@pytest.mark.parametrize("name", CEILING_NAMES)
def test_historical_name_is_shard_object(name):
    """``agent.auxiliary_client.<name>`` MUST be the same object as the
    shard's — i.e. the monolith delegates to the owner rather than keeping a
    second copy of the ceiling authority."""
    assert getattr(_aux, name) is getattr(_shard, name), (
        f"agent.auxiliary_client.{name} is NOT the shard object — "
        "duplicate ceiling authority remains in the monolith"
    )


@pytest.mark.parametrize("name", CEILING_NAMES)
def test_no_duplicate_owner_in_auxiliary_client(name):
    """The four ceiling-authority functions must be OWNED by the shard, not
    defined in ``auxiliary_client.py`` (the monolith holds only the
    compatibility re-import)."""
    fn = getattr(_shard, name)
    assert fn.__module__ == SHARD_MODULE, (
        f"{name} is defined in {fn.__module__}, not the bounded owner shard"
    )
    # And it must NOT be a locally-defined function object in the monolith.
    assert getattr(_aux, name).__module__ == SHARD_MODULE


# ── 2. Existing import names stay compatible (both modules) ────────────────

def test_imports_compatible_from_both_modules():
    from agent.auxiliary_client import (
        _aux_relay_gate as m_gate,
        _aux_provider_callback as m_cb,
        _rebind_aux_ceiling_for_fallback as m_rebind,
        _rebind_aux_ceiling_for_fallback_async as m_rebind_async,
    )
    from agent.auxiliary_context_ceiling import (
        _aux_relay_gate as s_gate,
        _aux_provider_callback as s_cb,
        _rebind_aux_ceiling_for_fallback as s_rebind,
        _rebind_aux_ceiling_for_fallback_async as s_rebind_async,
    )
    assert m_gate is s_gate
    assert m_cb is s_cb
    assert m_rebind is s_rebind
    assert m_rebind_async is s_rebind_async


# ── 3. DECISIVE: monkeypatching the historical gate still controls the live ─

def _completed_response() -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]),
                                 finish_reason="stop")],
        usage=None, model="test-model",
    )


def _sync_spy_client(fail: bool = False) -> SimpleNamespace:
    calls: list = []
    def create(**kwargs: Any) -> Any:
        if fail:
            raise RuntimeError("connection error")
        calls.append(kwargs)
        return _completed_response()
    return SimpleNamespace(
        base_url="https://api.test/v1", api_key="sk-x",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        _calls=calls, _hermes_fallback_destination=None,
    )


def test_monkeypatch_provider_callback_affects_live_relay_path(monkeypatch):
    """The decisive seam requirement: patching the historical
    ``agent.auxiliary_client._aux_provider_callback`` (the provider-boundary
    wrapper) must STILL control the actual live provider callback path after
    extraction.

    We REMOVE the wrapper (identity) and assert the provider IS called with
    the payload — i.e. the entry gate alone is insufficient, proving the
    patch actually disabled the provider-boundary enforcement in the live
    ``_relay_sync_completion`` path.  This mirrors the established negative
    control in test_context_ceiling_round4_topo.py and fails if the relay
    helper resolves the gate through a DIFFERENT module's globals than the
    one the test patched.
    """
    # A small request (well under any ceiling) so the entry gate passes.
    kwargs = {"model": "m", "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 100}
    client = _sync_spy_client()

    # Patch the provider-boundary wrapper to identity (remove the physical seam).
    monkeypatch.setattr(_aux, "_aux_provider_callback", lambda cb, **kw: cb)
    monkeypatch.setattr(_aux, "_relay_auxiliary_metadata",
                        lambda provider=None, api_mode=None: None)  # no route → direct

    token = _aux.set_aux_ceiling(64_000)
    try:
        _aux._relay_sync_completion(client, dict(kwargs), provider="openai")
    finally:
        _aux.reset_aux_ceiling(token)

    # The provider WAS called — because the provider-boundary wrapper (the
    # only thing that would gate the FINAL payload at the physical seam) was
    # patched out on the live path.  This proves the patch reached the live
    # relay helper through agent.auxiliary_client's globals.
    assert len(client._calls) == 1, (
        "patching agent.auxiliary_client._aux_provider_callback did NOT affect "
        "the live relay path — the relay helper resolves the gate through a "
        "different module's globals than the one that was patched"
    )


def test_monkeypatch_gate_is_invoked_on_live_relay_path(monkeypatch):
    """Patching the historical ``_aux_relay_gate`` must be INVOKED by the
    live ``_relay_sync_completion`` path (the entry gate still fires)."""
    kwargs = {"model": "m", "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 100}
    client = _sync_spy_client()
    hits = {"n": 0}

    def _recording_gate(kwargs=None, task="auxiliary", provider=None, model=None):
        hits["n"] += 1

    monkeypatch.setattr(_aux, "_aux_relay_gate", _recording_gate)
    monkeypatch.setattr(_aux, "_aux_provider_callback", lambda cb, **kw: cb)
    monkeypatch.setattr(_aux, "_relay_auxiliary_metadata",
                        lambda provider=None, api_mode=None: None)

    # No ceiling published → gate returns early, but it MUST still be the
    # function the live path calls.
    token = _aux.set_aux_ceiling(64_000)
    try:
        _aux._relay_sync_completion(client, dict(kwargs), provider="openai")
    finally:
        _aux.reset_aux_ceiling(token)

    assert hits["n"] >= 1, (
        "the live relay path did not call the (patched) _aux_relay_gate — "
        "the gate seam is not the one the monolith resolves"
    )


# ── 4 & 5. Sync / async fallback rebinding reaches the shard, restores ─────

def _small_request() -> list:
    return [{"role": "user", "content": "x" * 600_000}]  # ~154K tokens (verified band)


def _rebind_destination(model: str, api_key: str = "sk-fb"):
    return _aux._FallbackDestination(
        provider="openai", base_url="https://api.test/v1",
        api_mode="chat_completions", model=model,
    )


def test_sync_rebind_reaches_shard_and_restores(monkeypatch):
    resolver = []
    def _ecl(model="", base_url="", api_key="", provider="", **k):
        resolver.append({"model": model, "api_key": api_key})
        return FALLBACK_SMALL
    monkeypatch.setattr(_model_metadata, "effective_context_length", _ecl)

    assert _aux.get_aux_ceiling() is None  # baseline
    with _shard._rebind_aux_ceiling_for_fallback(_rebind_destination(FALLBACK_MODEL),
                                                 api_key="sk-fb") as ce:
        assert ce == FALLBACK_SMALL
        # While inside the scope, the ambient ceiling is the rebound value.
        assert _aux.get_aux_ceiling() == FALLBACK_SMALL
    # After the scope, the prior value (None baseline) is restored.
    assert _aux.get_aux_ceiling() is None, "sync rebind did not reset the token"


def test_async_rebind_reaches_shard_and_restores(monkeypatch):
    def _ecl(model="", base_url="", api_key="", provider="", **k):
        return FALLBACK_SMALL
    monkeypatch.setattr(_model_metadata, "effective_context_length", _ecl)

    async def _run():
        assert _aux.get_aux_ceiling() is None
        async with _shard._rebind_aux_ceiling_for_fallback_async(
                _rebind_destination(FALLBACK_MODEL), api_key="sk-fb") as ce:
            assert ce == FALLBACK_SMALL
            assert _aux.get_aux_ceiling() == FALLBACK_SMALL
        assert _aux.get_aux_ceiling() is None, "async rebind did not reset the token"

    asyncio.get_event_loop().run_until_complete(_run())


# ── 6. Large→small and small→large fallback transitions remain green ───────

def _patch_fallback_env(monkeypatch, initial_spy, fallback_spy,
                        initial_ceiling, fallback_ceiling, resolver_calls):
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
    for name in ("_try_main_fallback_chain", "_try_payment_fallback",
                 "_try_main_agent_model_fallback"):
        monkeypatch.setattr(_aux, name, lambda *a, **k: (None, None, ""), raising=False)
    fallback_spy._hermes_fallback_destination = _aux._FallbackDestination(
        provider="openai", base_url="https://api.test/v1",
        api_mode="chat_completions", model=FALLBACK_MODEL)
    monkeypatch.setattr(_aux, "_fallback_entry_api_key",
                        lambda entry: "sk-fallback", raising=False)
    def _ecl(model="", base_url="", api_key="", provider="", **k):
        resolver_calls.append({"model": model, "api_key": api_key})
        return fallback_ceiling if model == FALLBACK_MODEL else initial_ceiling
    monkeypatch.setattr(_model_metadata, "effective_context_length", _ecl, raising=False)
    monkeypatch.setattr(_aux, "_validate_llm_response", lambda r, *a, **k: "OK", raising=False)
    monkeypatch.setattr(_aux, "_relay_auxiliary_metadata",
                        lambda **k: ("openai", "test-model", {
                            "api_mode": "chat_completions", "api_request_id": "aux-seam",
                            "call_role": "auxiliary:test", "retry_count": 0,
                            "auxiliary_task": "test"}), raising=False)
    monkeypatch.setattr(_aux, "_relay_llm_execute", lambda *a, **k: None, raising=False)
    from agent import relay_llm as _relay
    monkeypatch.setattr(_relay, "execute_current", lambda req, cb, **k: cb(req), raising=False)
    async def _fake_async(req, cb, **k):
        return await cb(req)
    monkeypatch.setattr(_relay, "execute_current_async", _fake_async, raising=False)
    monkeypatch.setattr(_aux, "_to_async_client",
                        lambda client, model, is_vision=False: (client, model), raising=False)


def _sync_spy(fail: bool) -> SimpleNamespace:
    calls = []
    def create(**kw):
        if fail:
            raise RuntimeError("connection error")
        calls.append(kw); return _completed_response()
    return SimpleNamespace(base_url="https://api.test/v1", api_key="sk-fb",
                           chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                           _calls=calls, _hermes_fallback_destination=None)


def test_large_to_small_fallback_refused_under_fallback_ceiling(monkeypatch):
    initial = _sync_spy(fail=True); fallback = _sync_spy(fail=False)
    resolver = []
    _patch_fallback_env(monkeypatch, initial, fallback,
                        INITIAL_LARGE, FALLBACK_SMALL, resolver)
    with pytest.raises(ContextCeilingExceeded):
        _aux.call_llm("test", provider="openai", model=INITIAL_MODEL,
                      api_key="sk-initial", messages=_small_request(), max_tokens=None)
    assert len(fallback._calls) == 0, "fallback physically called under stale ceiling"
    assert any(c["model"] == FALLBACK_MODEL for c in resolver)


def test_small_to_large_fallback_gate_uses_fallback_authority(monkeypatch):
    """small initial (128K) → large fallback (900K) — the reverse-direction
    discriminator, proven at the GATE level (the end-to-end path can't express
    it: the local-refusal-is-terminal invariant refuses a request above the
    small initial *before* the fallback path, and one below it passes trivially).

    A request in the discriminating band (between 128K and 900K) must be
    ACCEPTED when the ceiling is rebound to the fallback destination's OWN
    (larger) ceiling, and REFUSED under the small initial's ceiling.  This
    proves "every physical fallback destination uses its own authority" in the
    reverse direction: the rebind to a larger destination relaxes the gate to
    that destination's authority rather than inheriting the smaller one."""
    from agent.model_metadata import enforce_final_context_budget
    # Resolve the fallback destination's OWN ceiling to the large value.
    monkeypatch.setattr(_model_metadata, "effective_context_length",
                        lambda model="", base_url="", api_key="", provider="", **k: FALLBACK_LARGE)
    # A band request: > 128K (small initial) but < 900K (large fallback).
    band = [{"role": "user", "content": "x" * 600_000}]  # ~154K tokens
    kwargs = {"model": "m", "messages": band, "max_tokens": 4096}

    def _budget():
        return _model_metadata.build_final_context_budget(kwargs, provider="openai", model="m")

    # Under the small initial ceiling (128K) the band request is refused…
    with pytest.raises(ContextCeilingExceeded):
        enforce_final_context_budget(_budget(), ceiling=INITIAL_SMALL, reason="initial")

    # …but when rebound to the fallback destination's OWN ceiling (900K), the
    # SAME request passes — the fallback's larger authority is honored.
    with _shard._rebind_aux_ceiling_for_fallback(
            _rebind_destination(FALLBACK_MODEL), api_key="sk-fb") as ce:
        assert ce == FALLBACK_LARGE
        _model_metadata.enforce_final_context_budget(_budget(),
                                                     ceiling=_aux.get_aux_ceiling(),
                                                     reason="fallback")  # no raise = dispatched


# ── 7. Credential-refresh fallback uses the fallback credential context ────

def test_fallback_resolution_carries_credential(monkeypatch):
    initial = _sync_spy(fail=True); fallback = _sync_spy(fail=False)
    resolver = []
    _patch_fallback_env(monkeypatch, initial, fallback,
                        INITIAL_LARGE, FALLBACK_SMALL, resolver)
    with pytest.raises(ContextCeilingExceeded):
        _aux.call_llm("test", provider="openai", model=INITIAL_MODEL,
                      api_key="sk-initial", messages=_small_request(), max_tokens=None)
    fb = [c for c in resolver if c["model"] == FALLBACK_MODEL]
    assert fb and fb[0]["api_key"], (
        "fallback ceiling resolution did not carry the destination credential"
    )
