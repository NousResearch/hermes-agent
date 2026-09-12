"""Regression tests for issue #87654.

Before the fix, `_get_cached_client()` stored the `_AuxProbeClientStub`
returned during `aux_probe_mode()` under the real cache key:

- 1st `check_vision_requirements()` in a process → `True` (and poisons the key).
- Every later check → cache hit → `_compat_model()` touches `stub._client`
  → `RuntimeError` → check catches `Exception` → `False`, sticky for the
  process lifetime. Vision tools vanish from every subsequent session.

The plain `isinstance` guard additionally missed adapter-wrapped stubs
(`CodexAuxiliaryClient` / `AnthropicAuxiliaryClient` keep the leaf in
`_real_client`), and an already-poisoned key never self-repaired.

These tests cover all three gaps without network or API keys.
"""

from __future__ import annotations

import types

import pytest

from agent import auxiliary_client as ac


@pytest.fixture
def clean_cache(monkeypatch):
    monkeypatch.setattr(ac, "_client_cache", {})
    yield ac._client_cache


def _stub(**kwargs):
    return ac._AuxProbeClientStub(api_key="k", base_url="https://example.invalid/v1", **kwargs)


def _resolve_kwargs(**overrides):
    args = dict(provider="nvidia", model="google/diffusiongemma-26b-a4b-it", is_vision=True)
    args.update(overrides)
    return args


def test_direct_stub_never_cached(clean_cache, monkeypatch):
    """A probe answer is returned but must not enter the cache."""
    monkeypatch.setattr(
        ac, "resolve_provider_client", lambda *a, **k: (_stub(), "probe-model")
    )
    client, _ = ac._get_cached_client(**_resolve_kwargs())
    assert isinstance(client, ac._AuxProbeClientStub)  # probe still gets its answer
    assert clean_cache == {}


def test_wrapped_stub_never_cached(clean_cache, monkeypatch):
    """Adapter-wrapped stubs must not enter the cache either (#87654 review)."""
    try:
        wrapped = ac.CodexAuxiliaryClient(_stub(), "m")
    except Exception:
        # If the adapter ever needs more than api_key/base_url slots, fall back
        # to a minimal namespace carrying the leaf in _real_client.
        wrapped = types.SimpleNamespace(_real_client=_stub())
    assert ac._is_probe_stub_client(wrapped)
    monkeypatch.setattr(ac, "resolve_provider_client", lambda *a, **k: (wrapped, "m"))
    client, _ = ac._get_cached_client(**_resolve_kwargs())
    assert client is wrapped
    assert clean_cache == {}


def test_poisoned_key_self_repairs(clean_cache, monkeypatch):
    """A key poisoned before the fix heals on next resolve (no restart needed)."""
    sentinel, sentinel_model = object(), "real-model"
    monkeypatch.setattr(
        ac, "resolve_provider_client", lambda *a, **k: (sentinel, sentinel_model)
    )
    kwargs = _resolve_kwargs()
    key = ac._client_cache_key(
        kwargs["provider"],
        async_mode=False,
        base_url=None,
        api_key=None,
        api_mode=None,
        main_runtime=None,
        is_vision=kwargs["is_vision"],
        task=None,
        model=kwargs["model"],
    )
    clean_cache[key] = (_stub(), "probe-model", None)  # pre-fix poisoned state
    client, _ = ac._get_cached_client(**kwargs)
    assert client is sentinel
    assert clean_cache[key][0] is sentinel
