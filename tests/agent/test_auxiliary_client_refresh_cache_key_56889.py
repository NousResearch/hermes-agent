"""Regression tests for #56889 and #58894: a Nous credential refresh must evict
the stale expired-credential client under the SAME cache key call_llm looked it
up with, across every dimension the lookup keyed on -- the model (#56889) and,
for auto-provider acquisitions, the task (#58894).

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

PR #58894 carries that same discipline one dimension further: for
``provider == "auto"`` the task also participates in the key, so the primary
acquisition site AND the 401 refresh must both fold the task in, or an
auto-provider client refreshed on a 401 lands under the ``task=""`` key while
the stale entry survives under the task-scoped key -- the immortal-client bug
reappears for the common auto path.

These are self-contained white-box tests: no network. The unit tests patch the
runtime-API resolver and the openai-client factory, then call the REAL
``_refresh_nous_auxiliary_client`` and assert the follow-up lookup returns the
fresh client. The end-to-end tests go further: they drive the REAL ``call_llm``
/ ``async_call_llm`` through the REAL module cache (patching only the client
*factories*, never ``_get_cached_client`` itself) so that the acquisition and
refresh cache keys are exercised together -- the only way to catch an
acquisition site that stops threading a keyed dimension.
"""

import inspect
from unittest.mock import AsyncMock, MagicMock

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


def test_refresh_evicts_stale_auto_client_under_task_scoped_key(monkeypatch):
    """An auto-provider refresh must carry the lookup task into the new key.

    ``_client_cache_key`` folds the task into the key ONLY for ``provider ==
    "auto"`` (auto can resolve through a task-specific fallback policy), so
    ``_get_cached_client("auto", ..., task=task)`` stores the client under a
    task-scoped key. The 401 refresh path must carry that same lookup task
    through ``_refresh_nous_auxiliary_client`` into ``_client_cache_key`` -- the
    ``task`` dimension of the #56889 keying, one element over from the model
    dimension. Dropping it lands the fresh client under the ``task=""`` key and
    leaves the stale expired-credential client immortal under the task-scoped
    key, so every auto-provider auxiliary call for that task keeps 401ing
    (#58894).

    Discriminating without a signature error on unfixed code: ``lookup_task`` is
    forwarded only when the parameter exists. On unfixed code the task is never
    carried, the fresh client lands under the wrong key, and the task-scoped
    lookup still serves the stale client -> the first assertion fails (rather
    than erroring on an unknown kwarg).
    """
    task = "agent"
    resolved_model = "Hermes-4-405B"
    # The key _get_cached_client("auto", resolved_model, ..., task=task) built
    # when the (now stale) client was acquired: task participates for "auto".
    lookup_key = ac._client_cache_key(
        "auto",
        async_mode=False,
        base_url=None,
        api_key=None,
        api_mode=None,
        main_runtime=None,
        is_vision=False,
        task=task,
        model=resolved_model,
    )
    stale = _FakeClient("stale-auto-task")
    ac._client_cache[lookup_key] = (stale, resolved_model, None)

    fresh = _FakeClient("fresh-auto-task")
    _patch_refresh_externals(monkeypatch, fresh)

    refresh_kwargs = dict(
        cache_provider="auto",
        model=resolved_model,
        lookup_model=resolved_model,
        async_mode=False,
        base_url=None,
        api_key=None,
        api_mode=None,
        main_runtime=None,
        is_vision=False,
    )
    if "lookup_task" in inspect.signature(ac._refresh_nous_auxiliary_client).parameters:
        refresh_kwargs["lookup_task"] = task
    refreshed_client, _ = ac._refresh_nous_auxiliary_client(**refresh_kwargs)
    assert refreshed_client is fresh

    # The next auto-provider call for this task repeats the SAME task-scoped
    # lookup. It is a pure cache hit in both worlds (that key always holds
    # something -- stale when unfixed, fresh when fixed), so no real client is
    # built.
    served, _ = ac._get_cached_client(
        "auto",
        resolved_model,
        async_mode=False,
        base_url=None,
        api_key=None,
        api_mode=None,
        main_runtime=None,
        task=task,
    )
    assert served is not stale, (
        "stale auto-provider client is still served after a task-scoped refresh: "
        "the refresh dropped the task dimension of the cache key (#58894)"
    )
    assert served is fresh
    # The stale client must be gone entirely, not merely shadowed by a second
    # entry under the task="" key.
    assert not any(entry[0] is stale for entry in ac._client_cache.values()), (
        "stale auto client still present in the cache under some key after refresh"
    )


class _Auth401(Exception):
    """A 401 the auth-error classifier recognizes (``status_code`` attribute)."""

    status_code = 401


def _nous_mock_client(*, async_mode, raises=None, returns=None):
    """A stand-in OpenAI client whose ``chat.completions.create`` 401s or returns."""
    client = MagicMock()
    client.base_url = NOUS_BASE_URL
    create = AsyncMock() if async_mode else MagicMock()
    if raises is not None:
        create.side_effect = raises
    else:
        create.return_value = returns
    client.chat.completions.create = create
    return client


def test_call_llm_auto_provider_evicts_stale_client_end_to_end(monkeypatch):
    """End-to-end: a default auto-provider 401 must evict the stale client.

    The integration guard the unit refresh tests structurally cannot give: it
    runs the REAL primary acquisition (``_get_cached_client`` at call_llm's
    acquisition site) and the REAL 401 refresh against the REAL module cache,
    patching only the client *factories* -- never ``_get_cached_client``, whose
    wholesale patching in the pre-existing call_llm 401 tests is exactly why the
    acquisition-vs-refresh key divergence went unseen. The stale client is
    acquired under the auto+task cache key; the 401 refresh must land the fresh
    client under that SAME key and evict the stale one. If the acquisition site
    stops threading ``task`` (the #58894 regression) the refresh rebuilds a
    divergent key, the stale expired-credential client survives, and the
    stale-absence assertion fails.
    """
    task = "compression"
    stale = _nous_mock_client(async_mode=False, raises=_Auth401("stale creds"))
    fresh = _nous_mock_client(async_mode=False, returns={"ok": True})

    # Force the default auto path and make the primary acquisition build `stale`.
    monkeypatch.setattr(
        ac, "_resolve_task_provider_model",
        lambda *a, **k: ("auto", None, None, None, None),
    )
    monkeypatch.setattr(
        ac, "resolve_provider_client",
        lambda *a, **k: (stale, "nous-model"),
    )
    # The 401 refresh rebuilds a fresh client from refreshed runtime creds.
    monkeypatch.setattr(
        ac, "_resolve_nous_runtime_api",
        lambda *, force_refresh=False: ("fresh-key", NOUS_BASE_URL),
    )
    monkeypatch.setattr(
        ac, "_create_openai_client",
        lambda *, api_key, base_url, **kwargs: fresh,
    )
    monkeypatch.setattr(ac, "_validate_llm_response", lambda resp, _task: resp)

    result = ac.call_llm(task=task, messages=[{"role": "user", "content": "hi"}])

    assert result == {"ok": True}
    assert stale.chat.completions.create.call_count == 1
    assert fresh.chat.completions.create.call_count == 1
    # The stale expired-credential client must be gone from the cache, not merely
    # shadowed by the fresh client under a divergent (task-dropped) key.
    assert not any(entry[0] is stale for entry in ac._client_cache.values()), (
        "stale auto-provider client survived the 401 refresh: the acquisition "
        "site dropped the task dimension so the refresh keyed the fresh client "
        "under a different cache entry (#58894)"
    )
    assert any(entry[0] is fresh for entry in ac._client_cache.values())


@pytest.mark.asyncio
async def test_async_call_llm_auto_provider_evicts_stale_client_end_to_end(monkeypatch):
    """Async twin of the end-to-end auto-provider eviction guard.

    Passing a non-None ``main_runtime`` also pins the async acquisition site's
    ``main_runtime`` threading: for ``provider == "auto"`` the runtime is part of
    the key, so if the async acquisition rebuilds without it (while the refresh
    passes it) the fresh client again lands under a divergent key -- the same bug
    class one element over. Reverting either the ``task`` or the ``main_runtime``
    kwarg at the async acquisition site fails this test.
    """
    task = "session_search"
    main_runtime = {"provider": "nous", "model": "Hermes-4-405B"}
    stale = _nous_mock_client(async_mode=True, raises=_Auth401("stale creds"))
    fresh = _nous_mock_client(async_mode=True, returns={"ok": True})

    monkeypatch.setattr(
        ac, "_resolve_task_provider_model",
        lambda *a, **k: ("auto", None, None, None, None),
    )
    monkeypatch.setattr(
        ac, "resolve_provider_client",
        lambda *a, **k: (stale, "nous-model"),
    )
    monkeypatch.setattr(
        ac, "_resolve_nous_runtime_api",
        lambda *, force_refresh=False: ("fresh-key", NOUS_BASE_URL),
    )
    # Async refresh builds a sync client then wraps it; patch the wrap to `fresh`.
    monkeypatch.setattr(
        ac, "_create_openai_client",
        lambda *, api_key, base_url, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(ac, "_to_async_client", lambda *a, **k: (fresh, "nous-model"))
    monkeypatch.setattr(ac, "_validate_llm_response", lambda resp, _task: resp)

    result = await ac.async_call_llm(
        task=task,
        messages=[{"role": "user", "content": "hi"}],
        main_runtime=main_runtime,
    )

    assert result == {"ok": True}
    assert stale.chat.completions.create.await_count == 1
    assert fresh.chat.completions.create.await_count == 1
    assert not any(entry[0] is stale for entry in ac._client_cache.values()), (
        "stale auto-provider async client survived the 401 refresh: the async "
        "acquisition site dropped the task/main_runtime dimension so the refresh "
        "keyed the fresh client under a different cache entry (#58894)"
    )
    assert any(entry[0] is fresh for entry in ac._client_cache.values())
