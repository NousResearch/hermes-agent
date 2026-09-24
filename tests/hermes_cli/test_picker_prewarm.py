"""Tests for the /model picker background cache prewarm.

``prewarm_picker_cache_async()`` warms the provider-models disk cache off the
user's critical path so the first ``/model`` open in a session is fast instead
of blocking ~1-2s on serial /v1/models fetches. These pin the two contracts
that matter: it runs the warm path exactly once per process (no thread leak),
and it delegates to ``list_authenticated_providers`` to do the warming.
"""

from __future__ import annotations

from unittest.mock import patch

import hermes_cli.model_switch as ms
from hermes_cli import model_switch_providers


def _reset_guard():
    model_switch_providers._picker_prewarm_done.clear()


def test_prewarm_runs_list_authenticated_providers_once():
    """First call spawns a thread that calls list_authenticated_providers;
    the warm side effect is delegated there (which disk-caches per provider)."""
    _reset_guard()
    with patch.object(ms, "list_authenticated_providers", return_value=[]) as mock_list:
        t = model_switch_providers.prewarm_picker_cache_async()
        assert t is not None, "first call must spawn a prewarm thread"
        t.join(timeout=10)
        assert not t.is_alive(), "prewarm thread should finish promptly"
        mock_list.assert_called_once()
    _reset_guard()


def test_prewarm_guard_is_once_per_process():
    """The process-level Event guard must make repeat calls no-ops so a
    long-lived process never leaks one OS thread per call."""
    _reset_guard()
    with patch.object(ms, "list_authenticated_providers", return_value=[]):
        t1 = model_switch_providers.prewarm_picker_cache_async()
        assert t1 is not None
        t1.join(timeout=10)
        # Subsequent calls return None (guard set) — no new thread.
        assert model_switch_providers.prewarm_picker_cache_async() is None
        assert model_switch_providers.prewarm_picker_cache_async() is None
    _reset_guard()


def test_prewarm_warms_the_active_custom_endpoint_for_the_next_open(monkeypatch):
    """End-to-end regression for #72762: the active custom endpoint must be
    warm by the time the user opens ``/model``, not just first-class
    ``PROVIDER_REGISTRY`` providers.

    The cache is keyed purely on ``base_url`` (see ``cached_fetch_api_models``
    in ``hermes_cli/models.py``), so this is not specific to any named
    provider — the fixture below stands in for any OpenAI-compatible custom
    endpoint a user might configure (an LLM gateway, Kilo Code, Together AI,
    a self-hosted vLLM/SGLang server, ...).

    Runs the real ``list_authenticated_providers()`` (not mocked, unlike the
    two tests above) through the prewarm thread against a fake
    ``load_picker_context()`` config with one active custom provider, then
    replays the exact kwargs the plain CLI ``/model`` handler passes twice in
    a row (``probe_custom_providers=False, probe_current_custom_provider=True``),
    simulating two ``/model`` opens in one session.

    We deliberately do NOT assert on *which* of (prewarm thread, first
    foreground open) wins the race to perform the live fetch — that's a
    thread-scheduling detail, not the contract. What must hold regardless of
    scheduling: across the warm-up plus two foreground opens, the endpoint is
    ever probed live at most once, and every open after that first probe is
    served from the disk cache with zero additional network calls.
    """
    import hermes_cli.inventory as inventory_mod
    import hermes_cli.models as models_mod

    _reset_guard()

    base_url = "https://api.example-gateway.test/v1"
    ctx = inventory_mod.ConfigContext(
        current_provider="custom:example-gateway",
        current_model="",  # avoid the unrelated current-model-always-shown guarantee (line ~3104)
        current_base_url=base_url,
        user_providers={},
        custom_providers=[
            {
                "name": "example-gateway",
                "base_url": base_url,
                "api_key": "sk-gateway-key",
            }
        ],
        excluded_providers=[],
    )
    monkeypatch.setattr(inventory_mod, "load_picker_context", lambda: ctx)

    calls = []

    def fake_fetch_api_models(api_key, url, **kwargs):
        calls.append((api_key, url))
        return ["gateway-model-a", "gateway-model-b"]

    monkeypatch.setattr(models_mod, "fetch_api_models", fake_fetch_api_models)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    def open_picker():
        return ms.list_authenticated_providers(
            current_provider=ctx.current_provider,
            current_base_url=ctx.current_base_url,
            current_model=ctx.current_model,
            user_providers=ctx.user_providers,
            custom_providers=ctx.custom_providers,
            excluded_providers=ctx.excluded_providers,
            # Exact kwargs cli.py's plain (no-args, no --refresh) /model
            # handler passes: probe only the active custom endpoint,
            # everything else from the warm disk cache.
            probe_custom_providers=False,
            probe_current_custom_provider=True,
        )

    t = model_switch_providers.prewarm_picker_cache_async()
    assert t is not None
    t.join(timeout=10)

    first_open = open_picker()
    assert len(calls) == 1, (
        "the endpoint must be live-probed at most once across boot prewarm "
        "plus the first /model open, however the race between them resolves"
    )
    row = next(p for p in first_open if p.get("api_url") == base_url)
    assert row["models"] == ["gateway-model-a", "gateway-model-b"]

    second_open = open_picker()
    assert len(calls) == 1, (
        "a second /model open in the same session must be served entirely "
        "from the disk cache — this is the #72762 regression: previously "
        "every open re-probed the endpoint live"
    )
    row2 = next(p for p in second_open if p.get("api_url") == base_url)
    assert row2["models"] == ["gateway-model-a", "gateway-model-b"]

    _reset_guard()


