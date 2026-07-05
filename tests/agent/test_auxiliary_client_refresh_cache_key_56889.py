"""Regression tests for #56889: Nous credential refresh must evict the stale
expired-credential client under the SAME cache key call_llm looked it up with.

Background
----------
PR #56889 added the model name to the auxiliary client cache key so that
concurrent MoA fan-out calls to the same provider but different models don't
share (and cross-close) a single httpx client. The regression: on the default
Nous config ``call_llm`` looks the client up with ``resolved_model=None`` (the
key's model element is ``""``) but is handed back the resolved provider default
(e.g. ``"Hermes-4-405B"``) as ``final_model``. On a 401 the refresh path was
keyed on that resolved ``final_model`` instead of the ``None`` used at lookup,
so the fresh client landed under a DIFFERENT key. The stale expired-credential
client under ``""`` was never overwritten and stayed immortal: every auxiliary
call 401s -> forced portal round-trip -> retry, forever.

These are self-contained white-box tests: no network. The runtime-API resolver
and the openai-client factory are patched, then the REAL
``_refresh_nous_auxiliary_client`` is called and the subsequent default-config
lookup (``model=None``) is asserted to return the fresh client.
"""

import inspect

import pytest

import agent.auxiliary_client as ac


NOUS_BASE_URL = "https://inference-api.nousresearch.com/v1"


class _FakeClient:
    """Minimal stand-in for an OpenAI client with a closable connection pool."""

    def __init__(self, tag):
        self.tag = tag
        self.api_key = "key-" + tag
        self.base_url = NOUS_BASE_URL
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _clean_client_cache():
    """Isolate the module-level client cache around each test."""
    ac._client_cache.clear()
    yield
    ac._client_cache.clear()


def _default_config_lookup_key():
    """The cache key call_llm builds for a default-Nous acquisition.

    Mirrors ``_get_cached_client("nous", resolved_model=None, async_mode=False,
    ...)``: the model element of the key is ``None`` -> ``""``.
    """
    return ac._client_cache_key(
        "nous",
        async_mode=False,
        base_url=None,
        api_key=None,
        api_mode=None,
        main_runtime=None,
        is_vision=False,
        task=None,
        model=None,
    )


def _patch_refresh_externals(monkeypatch, fresh_client):
    monkeypatch.setattr(
        ac,
        "_resolve_nous_runtime_api",
        lambda *, force_refresh=False: ("fresh-key", NOUS_BASE_URL),
    )
    monkeypatch.setattr(
        ac,
        "_create_openai_client",
        lambda *, api_key, base_url, **kwargs: fresh_client,
    )


def test_refresh_evicts_stale_client_under_default_config_lookup_key(monkeypatch):
    """After a 401 refresh, the default-config lookup must return the fresh client.

    Fails on unfixed main: the refresh stores under ``model="Hermes-4-405B"``
    while the acquisition stored under ``model=None`` -> ``""``, so the stale
    client is never evicted.
    """
    lookup_key = _default_config_lookup_key()
    stale = _FakeClient("stale-expired-creds")
    # _get_cached_client stores (client, default_model, bound_loop). The stored
    # default_model is the resolved provider default even though the lookup
    # model was None.
    ac._client_cache[lookup_key] = (stale, "Hermes-4-405B", None)

    fresh = _FakeClient("fresh")
    _patch_refresh_externals(monkeypatch, fresh)

    # The 401 handler calls refresh with the RESOLVED model (final_model), which
    # on the default config is the provider default "Hermes-4-405B" -- NOT the
    # None the client was actually looked up with.
    refreshed_client, refreshed_model = ac._refresh_nous_auxiliary_client(
        cache_provider="nous",
        model="Hermes-4-405B",
        async_mode=False,
        base_url=None,
        api_key=None,
        api_mode=None,
        main_runtime=None,
        is_vision=False,
    )
    assert refreshed_client is fresh
    assert refreshed_model == "Hermes-4-405B"

    # The next auxiliary call repeats the SAME default-config lookup (model=None).
    # This is a pure cache hit in both the fixed and unfixed worlds (the "" key
    # always holds something), so no real client is built.
    served, _ = ac._get_cached_client(
        "nous",
        None,
        async_mode=False,
        base_url=None,
        api_key=None,
        api_mode=None,
        main_runtime=None,
    )
    assert served is not stale, (
        "stale expired-credential client is still served after refresh: the "
        "refresh stored the fresh client under a different cache key than the "
        "lookup key (#56889)"
    )
    assert served is fresh

    # The stale client must be fully gone from the cache, not merely shadowed by
    # a second entry under a different model key.
    assert not any(entry[0] is stale for entry in ac._client_cache.values()), (
        "stale client still present in the cache under some key after refresh"
    )


@pytest.mark.skipif(
    "lookup_model" not in inspect.signature(ac._refresh_nous_auxiliary_client).parameters,
    reason="fix not applied: _refresh_nous_auxiliary_client has no lookup_model parameter",
)
def test_refresh_preserves_explicit_model_key(monkeypatch):
    """An explicit-model acquisition (MoA fan-out) must refresh under that model.

    Guards the #56889 per-model keying: when the client was looked up under an
    explicit model, the refresh must key on that same explicit model and must
    NOT clobber the shared default-config ("") entry.
    """
    explicit = "Hermes-4-70B"
    lookup_key = ac._client_cache_key(
        "nous",
        async_mode=False,
        base_url=None,
        api_key=None,
        api_mode=None,
        main_runtime=None,
        is_vision=False,
        task=None,
        model=explicit,
    )
    stale = _FakeClient("stale-explicit")
    ac._client_cache[lookup_key] = (stale, explicit, None)

    fresh = _FakeClient("fresh-explicit")
    _patch_refresh_externals(monkeypatch, fresh)

    ac._refresh_nous_auxiliary_client(
        cache_provider="nous",
        model=explicit,
        lookup_model=explicit,
        async_mode=False,
        base_url=None,
        api_key=None,
        api_mode=None,
        main_runtime=None,
        is_vision=False,
    )

    served, _ = ac._get_cached_client(
        "nous",
        explicit,
        async_mode=False,
        base_url=None,
        api_key=None,
        api_mode=None,
        main_runtime=None,
    )
    assert served is fresh
    # The default-config ("") key was never created by an explicit-model refresh.
    assert _default_config_lookup_key() not in ac._client_cache
