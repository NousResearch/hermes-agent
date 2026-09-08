"""A Nous 401 refresh must replace the auxiliary client under the SAME cache key
``call_llm`` acquired it with (model dimension #56889, task dimension #58894).
Otherwise the expired client is never evicted and every auxiliary call 401s,
refreshes, and retries forever (#91023).

End-to-end through the REAL call_llm / async_call_llm and the REAL module
cache; only the client factories are patched.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import agent.auxiliary_client as ac


NOUS_BASE_URL = "https://inference-api.nousresearch.com/v1"


@pytest.fixture(autouse=True)
def _clean_client_cache():
    ac._client_cache.clear()
    yield
    ac._client_cache.clear()


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
        lambda *, force_refresh=False, stale_access_token=None: ("fresh-key", NOUS_BASE_URL),
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
        lambda *, force_refresh=False, stale_access_token=None: ("fresh-key", NOUS_BASE_URL),
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
