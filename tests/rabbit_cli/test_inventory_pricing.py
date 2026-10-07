"""Tests for inventory._apply_pricing — the pricing enrichment that

feeds the desktop GUI model picker (and onboarding) so it can show $/Mtok
columns + Free badges, the same way the `rabbit model` CLI picker does.
"""

from threading import Event
from time import monotonic

import pytest

import rabbit_cli.inventory as inv
import rabbit_cli.models as models_mod
from rabbit_cli import models_pricing


def _patch_pricing(monkeypatch, *, pricing):
    monkeypatch.setattr(models_pricing, "get_pricing_for_provider", lambda slug, **kw: pricing.get(slug, {}))


def test_apply_pricing_formats_per_model_prices(monkeypatch):
    """Each model gets formatted input/output/cache + a free flag."""
    _patch_pricing(
        monkeypatch,
        pricing={
            "openrouter": {
                "a/paid": {"prompt": "0.000003", "completion": "0.000015", "input_cache_read": "0.0000003"},
                "b/free": {"prompt": "0", "completion": "0"},
            }
        },
    )
    rows = [{"slug": "openrouter", "models": ["a/paid", "b/free"]}]
    inv._apply_pricing(rows)

    pricing = rows[0]["pricing"]
    assert pricing["a/paid"] == {"input": "$3.00", "output": "$15.00", "cache": "$0.30", "free": False}
    assert pricing["b/free"]["free"] is True
    assert pricing["b/free"]["input"] == "free"


def test_apply_pricing_marks_zero_cost_models_free(monkeypatch):
    """Zero prompt+completion prices flag ``free``; the ``original`` hint is inert."""
    _patch_pricing(
        monkeypatch,
        pricing={
            "openrouter": {
                "a/free": {
                    "prompt": "0",
                    "completion": "0",
                    "original": {
                        "prompt": "0.000002",
                        "completion": "0.00001",
                    },
                },
                "b/natively-free": {
                    "prompt": "0",
                    "completion": "0",
                },
            }
        },
    )
    rows = [{"slug": "openrouter", "models": ["a/free", "b/natively-free"]}]
    inv._apply_pricing(rows)
    free = rows[0]["pricing"]["a/free"]
    assert free["free"] is True
    assert free["input"] == "free"
    native = rows[0]["pricing"]["b/natively-free"]
    assert native["free"] is True
    # No sale chrome in this build.
    assert "discount_percent" not in free
    assert "was_input" not in free


def test_model_options_cold_pricing_fetch_runs_off_the_request_path(monkeypatch):
    """A cold pricing endpoint must not delay the first picker payload."""
    fetch_started = Event()
    release_fetch = Event()

    def fake_pricing(_slug, *, force_refresh=False, cached_only=False):
        if cached_only:
            return {}
        fetch_started.set()
        release_fetch.wait(timeout=5)
        return {}

    row = {
        "slug": "openrouter",
        "name": "OpenRouter",
        "models": ["vendor/model"],
        "total_models": 1,
        "is_current": True,
        "is_user_defined": False,
        "source": "built-in",
    }
    monkeypatch.setattr(models_pricing, "get_pricing_for_provider", fake_pricing)
    monkeypatch.setattr(
        "rabbit_cli.model_switch.list_authenticated_providers",
        lambda **_kwargs: [row],
    )
    monkeypatch.setattr(inv, "_moa_provider_row", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(inv, "_apply_capabilities", lambda _rows, **_kwargs: None)
    monkeypatch.setattr(inv, "_apply_featured", lambda _rows, **_kwargs: None)
    monkeypatch.setattr(inv, "_pricing_prewarm_threads", {})

    try:
        started_at = monotonic()
        payload = inv.build_model_options_payload(
            inv.ConfigContext(
                current_provider="openrouter",
                current_model="vendor/model",
                current_base_url="",
                user_providers={},
                custom_providers=[],
            )
        )
        elapsed = monotonic() - started_at
        assert payload["providers"][0]["slug"] == "openrouter"
        assert "pricing" not in payload["providers"][0]
        assert elapsed < 2.0, f"cold picker blocked for {elapsed:.2f}s"
        assert fetch_started.wait(timeout=1), "pricing should prewarm in the background"
    finally:
        threads = list(inv._pricing_prewarm_threads.values())
        release_fetch.set()
        for thread in threads:
            thread.join(timeout=2)


def test_prewarm_preserves_context_and_runs_once_per_profile(tmp_path, monkeypatch):
    """Concurrent multiplex profiles retain their own home and secret scope."""
    from agent.secret_scope import (
        current_secret_scope,
        reset_secret_scope,
        set_secret_scope,
    )
    from rabbit_constants import (
        rabbit_home_key,
        reset_rabbit_home_override,
        set_rabbit_home_override,
    )

    monkeypatch.setattr(inv, "_pricing_prewarm_threads", {})
    release = Event()
    started = {"a": Event(), "b": Event()}
    observed = {}

    def capture_context(_rows):
        scope = current_secret_scope()
        label = scope["PROFILE_MARKER"]
        observed[label] = (rabbit_home_key(), dict(scope))
        started[label].set()
        release.wait(timeout=5)

    monkeypatch.setattr(inv, "_apply_pricing", capture_context)

    threads = []
    try:
        for label in ("a", "b"):
            home = tmp_path / label
            home_token = set_rabbit_home_override(str(home))
            secret_token = set_secret_scope({"PROFILE_MARKER": label})
            try:
                threads.append(inv._prewarm_pricing_async([{"models": []}]))
            finally:
                reset_secret_scope(secret_token)
                reset_rabbit_home_override(home_token)

        assert threads[0] is not threads[1]
        assert started["a"].wait(timeout=1)
        assert started["b"].wait(timeout=1)
        assert observed["a"] == (
            rabbit_home_key(tmp_path / "a"),
            {"PROFILE_MARKER": "a"},
        )
        assert observed["b"] == (
            rabbit_home_key(tmp_path / "b"),
            {"PROFILE_MARKER": "b"},
        )
    finally:
        release.set()
        for thread in threads:
            if thread is not None:
                thread.join(timeout=2)


def test_prewarm_deduplicates_inflight_scope_and_cleans_up(monkeypatch):
    """Rapid opens share one worker, then a completed scope can run again."""
    monkeypatch.setattr(inv, "_pricing_prewarm_threads", {})
    started = Event()
    release = Event()
    calls = []

    def blocked_prewarm(_rows):
        calls.append(None)
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(inv, "_apply_pricing", blocked_prewarm)
    rows = [{"slug": "openrouter", "models": ["vendor/model"]}]

    first = inv._prewarm_pricing_async(rows)
    try:
        assert started.wait(timeout=1)
        second = inv._prewarm_pricing_async(rows)
        assert second is first
        assert len(calls) == 1
    finally:
        release.set()
        first.join(timeout=2)

    assert not first.is_alive()
    assert inv._pricing_prewarm_threads == {}

    retry = inv._prewarm_pricing_async(rows)
    retry.join(timeout=2)
    assert retry is not first
    assert len(calls) == 2
    assert inv._pricing_prewarm_threads == {}


def _stub_deepinfra_endpoint_scope(monkeypatch, active_endpoint):
    """Point the deepinfra pricing scope at a mutable endpoint."""
    monkeypatch.setattr(
        models_mod, "_deepinfra_catalog_url",
        lambda: (active_endpoint["value"], active_endpoint["value"] + "/models"),
    )


def test_prewarm_endpoint_rotation_starts_a_new_worker(tmp_path, monkeypatch):
    """A live endpoint-A worker must not suppress endpoint B for its profile."""
    from rabbit_constants import (
        reset_rabbit_home_override,
        set_rabbit_home_override,
    )

    endpoint_a = "https://endpoint-a.example"
    endpoint_b = "https://endpoint-b.example"
    active_endpoint = {"value": endpoint_a}
    started = {endpoint_a: Event(), endpoint_b: Event()}
    release_a = Event()
    monkeypatch.setattr(inv, "_pricing_prewarm_threads", {})
    _stub_deepinfra_endpoint_scope(monkeypatch, active_endpoint)

    def prewarm(_rows):
        endpoint = active_endpoint["value"]
        started[endpoint].set()
        if endpoint == endpoint_a:
            release_a.wait(timeout=5)

    monkeypatch.setattr(inv, "_apply_pricing", prewarm)

    token = set_rabbit_home_override(str(tmp_path / "profile"))
    threads = []
    try:
        threads.append(
            inv._prewarm_pricing_async([{"slug": "deepinfra", "models": ["a/model"]}])
        )
        assert started[endpoint_a].wait(timeout=1)

        active_endpoint["value"] = endpoint_b
        threads.append(
            inv._prewarm_pricing_async([{"slug": "deepinfra", "models": ["b/model"]}])
        )

        assert threads[0] is not threads[1]
        assert started[endpoint_b].wait(timeout=1)
        threads[1].join(timeout=2)
        assert not threads[1].is_alive()
    finally:
        release_a.set()
        for thread in threads:
            if thread is not None:
                thread.join(timeout=2)
        reset_rabbit_home_override(token)


def test_prewarm_endpoint_scope_ignores_the_current_provider(tmp_path, monkeypatch):
    """An endpoint-scoped provider keys its worker on its own endpoint even
    while another provider is current."""
    from rabbit_constants import (
        reset_rabbit_home_override,
        set_rabbit_home_override,
    )

    endpoint_a = "https://endpoint-a.example"
    endpoint_b = "https://endpoint-b.example"
    active_endpoint = {"value": endpoint_a}
    started = {endpoint_a: Event(), endpoint_b: Event()}
    release_a = Event()
    monkeypatch.setattr(inv, "_pricing_prewarm_threads", {})
    _stub_deepinfra_endpoint_scope(monkeypatch, active_endpoint)

    def prewarm(_rows):
        endpoint = active_endpoint["value"]
        started[endpoint].set()
        if endpoint == endpoint_a:
            release_a.wait(timeout=5)

    monkeypatch.setattr(inv, "_apply_pricing", prewarm)

    token = set_rabbit_home_override(str(tmp_path / "profile"))
    threads = []
    try:
        threads.append(
            inv._prewarm_pricing_async(
                [{"slug": "deepinfra", "models": ["a/model"]}],
                current_provider="openrouter",
                current_base_url="https://openrouter.ai/api/v1",
            )
        )
        assert started[endpoint_a].wait(timeout=1)

        active_endpoint["value"] = endpoint_b
        threads.append(
            inv._prewarm_pricing_async(
                [{"slug": "deepinfra", "models": ["b/model"]}],
                current_provider="openrouter",
                current_base_url="https://openrouter.ai/api/v1",
            )
        )

        assert threads[0] is not threads[1]
        assert started[endpoint_b].wait(timeout=1)
        threads[1].join(timeout=2)
        assert not threads[1].is_alive()
    finally:
        release_a.set()
        for thread in threads:
            if thread is not None:
                thread.join(timeout=2)
        reset_rabbit_home_override(token)


def test_cached_only_pricing_returns_a_warm_value_without_fetching(monkeypatch):
    """Cache-only picker reads preserve pricing once the prewarm completes."""
    cache_key = "https://openrouter.ai/api"
    expected = {"vendor/model": {"prompt": "0.000001", "completion": "0.000002"}}
    monkeypatch.setattr(models_pricing, "_pricing_cache", {cache_key: expected})
    monkeypatch.setattr(models_pricing, "_pricing_cache_retry_after", {})
    monkeypatch.setattr(models_pricing, "_pricing_provider_cache_keys", {})
    monkeypatch.setattr(models_pricing, "fetch_models_with_pricing",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("network fetch started")),
    )

    assert models_pricing.get_pricing_for_provider(
        "openrouter", cached_only=True
    ) == expected


def test_custom_provider_with_proven_upstream_reuses_canonical_pricing_source(monkeypatch):
    """A custom row borrows a priced source only when its endpoint proves that source."""
    expected = {"vendor/model": {"prompt": "0.000001", "completion": "0.000002"}}
    fetch_calls = []

    def fetcher(*, force_refresh=False):
        fetch_calls.append(force_refresh)
        return expected

    cached_calls = []
    monkeypatch.setitem(models_pricing._PRICING_FETCHERS, "openrouter", fetcher)
    monkeypatch.setattr(
        models_pricing,
        "_cached_only_pricing",
        lambda provider: cached_calls.append(provider) or expected,
    )

    assert models_pricing.get_pricing_for_provider(
        "custom:openrouter", base_url="https://openrouter.ai/api/v1"
    ) == expected
    assert fetch_calls == [False]
    assert models_pricing.get_pricing_for_provider(
        "custom:unrelated-name", base_url="https://openrouter.ai/api/v1", cached_only=True
    ) == expected
    assert cached_calls == ["openrouter"]
    assert (
        models_pricing.pricing_cache_scope("custom:unrelated-name", base_url="https://openrouter.ai/api/v1")
        == models_pricing.pricing_cache_scope("openrouter")
    )


def test_custom_provider_pricing_fails_closed_without_a_proven_upstream(monkeypatch):
    """A shadowed canonical slug or empty suffix never leaks canonical prices to a proxy."""
    monkeypatch.setitem(
        models_pricing._PRICING_FETCHERS,
        "openrouter",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("custom proxy fetched OpenRouter pricing")),
    )
    monkeypatch.setattr(models_pricing, "_cached_only_pricing",
        lambda _provider: (_ for _ in ()).throw(AssertionError("custom proxy read canonical cache")),
    )

    assert models_pricing.get_pricing_for_provider(
        "custom:openrouter", base_url="https://proxy.example/v1"
    ) == {}
    assert models_pricing.get_pricing_for_provider("custom:") == {}
    assert models_pricing.pricing_cache_scope("custom:openrouter", base_url="https://proxy.example/v1") == ""


@pytest.mark.parametrize("base_url", [
    "https://openrouter.ai.attacker.invalid/v1",
    "https://not-openrouter.ai/v1",
    "https://ai-gateway.vercel.sh.attacker.invalid/v1",
])
def test_custom_provider_pricing_rejects_lookalike_upstream_hosts(monkeypatch, base_url):
    """A hostname merely containing a priced provider's hostname is still an untrusted proxy."""
    monkeypatch.setitem(
        models_pricing._PRICING_FETCHERS,
        "openrouter",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("lookalike host fetched OpenRouter pricing")),
    )

    assert models_pricing.get_pricing_for_provider(
        "custom:any-name", base_url=base_url
    ) == {}


def test_custom_provider_alias_uses_proven_canonical_upstream(monkeypatch):
    """A Vercel-named row on the real AI Gateway uses its canonical pricing cache."""
    expected = {"vendor/model": {"prompt": "0.000001", "completion": "0.000002"}}
    monkeypatch.setattr(models_pricing, "_cached_only_pricing", lambda provider: expected if provider == "ai-gateway" else {})

    assert models_pricing.get_pricing_for_provider(
        "custom:vercel", base_url="https://ai-gateway.vercel.sh/v1", cached_only=True
    ) == expected
    assert (
        models_pricing.pricing_cache_scope("custom:vercel", base_url="https://ai-gateway.vercel.sh/v1")
        == models_pricing.pricing_cache_scope("ai-gateway")
    )


def test_apply_pricing_uses_custom_row_endpoint_identity(monkeypatch):
    """The picker prices an actual OpenRouter row but leaves a canonical-named proxy unpriced."""
    expected = {"vendor/model": {"prompt": "0.000003", "completion": "0.000015"}}
    calls = []

    def pricing(provider, **kwargs):
        calls.append((provider, kwargs))
        return expected if kwargs.get("base_url") == "https://openrouter.ai/api/v1" else {}

    monkeypatch.setattr(models_pricing, "get_pricing_for_provider", pricing)
    rows = [
        {"slug": "custom:openrouter", "api_url": "https://openrouter.ai/api/v1", "models": ["vendor/model"]},
        {"slug": "custom:openrouter", "api_url": "https://proxy.example/v1", "models": ["vendor/model"]},
    ]

    inv._apply_pricing(rows, cached_only=True)

    assert rows[0]["pricing"]["vendor/model"] == {
        "input": "$3.00", "output": "$15.00", "cache": None, "free": False,
    }
    assert "pricing" not in rows[1]
    assert calls == [
        ("custom:openrouter", {"base_url": "https://openrouter.ai/api/v1", "cached_only": True}),
        ("custom:openrouter", {"base_url": "https://proxy.example/v1", "cached_only": True}),
    ]


def test_prewarm_cache_scope_uses_custom_row_endpoint_identity(monkeypatch):
    """Prewarm resolves a custom row with the same endpoint evidence as pricing lookup."""
    observed = []
    monkeypatch.setattr(inv, "_pricing_prewarm_threads", {})
    monkeypatch.setattr(
        models_pricing,
        "pricing_cache_scope",
        lambda provider, **kwargs: observed.append((provider, kwargs)) or "openrouter-scope",
    )
    monkeypatch.setattr(inv, "_apply_pricing", lambda _rows: None)

    thread = inv._prewarm_pricing_async([
        {"slug": "custom:unrelated-name", "api_url": "https://openrouter.ai/api/v1", "models": ["vendor/model"]},
    ])
    thread.join(timeout=2)

    assert observed == [(
        "custom:unrelated-name",
        {"base_url": "https://openrouter.ai/api/v1", "current_provider": "", "current_base_url": ""},
    )]


def test_cached_only_dynamic_pricing_is_profile_scoped(tmp_path, monkeypatch):
    """Alternating profiles read the endpoint each profile warmed."""
    from rabbit_constants import (
        reset_rabbit_home_override,
        set_rabbit_home_override,
    )

    endpoint_a = "https://profile-a.example"
    endpoint_b = "https://profile-b.example"
    expected_a = {"a/model": {"prompt": "1", "completion": "2"}}
    expected_b = {"b/model": {"prompt": "3", "completion": "4"}}
    cache = {endpoint_a: expected_a, endpoint_b: expected_b}
    active_endpoint = {"value": endpoint_a}
    _stub_deepinfra_endpoint_scope(monkeypatch, active_endpoint)
    monkeypatch.setattr(models_mod, "_deepinfra_catalog_cache", cache)
    monkeypatch.setattr(
        models_pricing, "_fetch_deepinfra_pricing",
        lambda **_kwargs: cache[models_mod._deepinfra_catalog_url()[0]],
    )
    monkeypatch.setitem(
        models_pricing._PRICING_FETCHERS, "deepinfra",
        lambda **_kwargs: cache[models_mod._deepinfra_catalog_url()[0]],
    )

    def in_profile(home, endpoint, *, cached_only):
        token = set_rabbit_home_override(str(home))
        active_endpoint["value"] = endpoint
        try:
            return models_pricing.get_pricing_for_provider(
                "deepinfra", cached_only=cached_only
            )
        finally:
            reset_rabbit_home_override(token)

    assert in_profile(tmp_path / "a", endpoint_a, cached_only=False) == expected_a
    assert in_profile(tmp_path / "b", endpoint_b, cached_only=False) == expected_b
    assert in_profile(tmp_path / "a", endpoint_b, cached_only=True) == expected_b
    assert in_profile(tmp_path / "b", endpoint_a, cached_only=True) == expected_a
