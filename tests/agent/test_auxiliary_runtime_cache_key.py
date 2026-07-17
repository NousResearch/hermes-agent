"""Regression coverage for implicit live-runtime auxiliary cache keys.

#49151/#49156 is specifically the ``provider='auto'`` path where callers omit
``main_runtime`` after a mid-session model switch.  This is distinct from
#56889, which isolates callers that pass different explicit ``model=`` values.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import MagicMock, patch

import pytest

import agent.auxiliary_client as aux


@pytest.fixture(autouse=True)
def _clean_aux_state():
    aux.shutdown_cached_clients()
    aux.clear_runtime_main()
    yield
    aux.shutdown_cached_clients()
    aux.clear_runtime_main()


def _runtime(model: str, *, provider: str = "custom:llama-swap") -> dict:
    return {
        "provider": provider,
        "model": model,
        "base_url": "http://llama-swap.test/v1",
        "api_key": "local-key",
        "api_mode": "chat_completions",
        "auth_mode": "api_key",
    }


def test_implicit_auto_cache_rebuilds_after_runtime_model_switch():
    """A /model switch must not reuse the old implicit-auto cache entry."""
    built = []

    def fake_resolve(_provider, _model, _async_mode, *, main_runtime, **_kwargs):
        client = MagicMock(name=f"client-{main_runtime['model']}")
        built.append((client, dict(main_runtime)))
        return client, main_runtime["model"]

    with patch.object(aux, "resolve_provider_client", side_effect=fake_resolve):
        aux.set_runtime_main(**_runtime("qwen35b-code"))
        first_client, first_model = aux._get_cached_client("auto")

        aux.set_runtime_main(**_runtime("qwen27b-code"))
        second_client, second_model = aux._get_cached_client("auto")

    assert first_model == "qwen35b-code"
    assert second_model == "qwen27b-code"
    assert second_client is not first_client
    assert [runtime["model"] for _, runtime in built] == [
        "qwen35b-code",
        "qwen27b-code",
    ]


def test_implicit_runtime_cache_key_covers_full_connection_and_auth_surface():
    """Provider/endpoint/credential/wire/auth changes all isolate auto clients."""
    base = _runtime("same-model")
    variants = [
        {**base, "provider": "custom:other"},
        {**base, "base_url": "https://other.test/v1"},
        {**base, "api_key": "other-key"},
        {**base, "api_mode": "codex_responses"},
        {**base, "auth_mode": "entra_id", "api_key": lambda: "token"},
    ]

    aux.set_runtime_main(**base)
    baseline = aux._client_cache_key("auto", async_mode=False)
    keys = []
    for variant in variants:
        aux.set_runtime_main(**variant)
        keys.append(aux._client_cache_key("auto", async_mode=False))

    assert all(key != baseline for key in keys)
    assert len(set(keys)) == len(keys)


def test_implicit_runtime_is_isolated_between_concurrent_session_contexts():
    """Concurrent gateway sessions must not read each other's live runtime."""
    barrier = Barrier(2)

    def session(model: str):
        aux.set_runtime_main(**_runtime(model))
        barrier.wait()
        normalized = aux._normalize_main_runtime(None)
        return normalized["model"], aux._client_cache_key("auto", async_mode=False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(session, "session-a-model")
        second = pool.submit(session, "session-b-model")
        model_a, key_a = first.result()
        model_b, key_b = second.result()

    assert model_a == "session-a-model"
    assert model_b == "session-b-model"
    assert key_a != key_b


def test_context_without_runtime_does_not_fall_back_to_other_session_globals():
    """A fresh context must not inherit another session's compatibility mirrors."""
    aux.set_runtime_main(**_runtime("other-session-model"))

    def fresh_context():
        return aux._normalize_main_runtime(None)

    import contextvars

    assert contextvars.Context().run(fresh_context) == {}


def test_legacy_patched_globals_are_visible_only_without_an_active_runtime():
    """Direct legacy patches work, but never override context-local session state."""
    with patch.object(aux, "_RUNTIME_MAIN_PROVIDER", "custom:legacy"), patch.object(
        aux, "_RUNTIME_MAIN_MODEL", "legacy-model"
    ), patch.object(
        aux, "_RUNTIME_MAIN_BASE_URL", "https://legacy.test/v1"
    ):
        assert aux._normalize_main_runtime(None)["model"] == "legacy-model"

        aux.set_runtime_main(**_runtime("active-session-model"))
        runtime = aux._normalize_main_runtime(None)

    assert runtime["model"] == "active-session-model"
    assert runtime["base_url"] == "http://llama-swap.test/v1"


def test_concurrent_vision_probes_use_each_sessions_endpoint_and_model():
    """Vision auto-routing must not mix custom endpoints across sessions."""
    barrier = Barrier(2)

    def fake_resolve(provider, model, **kwargs):
        barrier.wait()
        client = MagicMock()
        client.probed_base_url = kwargs.get("explicit_base_url")
        return client, model

    def probe(model: str, base_url: str):
        runtime = _runtime(model)
        runtime["base_url"] = base_url
        aux.set_runtime_main(**runtime)
        provider, client, resolved_model = aux.resolve_vision_provider_client()
        assert client is not None
        return provider, resolved_model, client.probed_base_url

    with patch.object(
        aux, "_resolve_task_provider_model", return_value=("auto", None, None, None, None)
    ), patch.object(aux, "_main_model_supports_vision", return_value=True), patch.object(
        aux, "resolve_provider_client", side_effect=fake_resolve
    ):
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(probe, "vision-a", "https://a.test/v1")
            second = pool.submit(probe, "vision-b", "https://b.test/v1")
            result_a = first.result()
            result_b = second.result()

    assert result_a == ("custom:llama-swap", "vision-a", "https://a.test/v1")
    assert result_b == ("custom:llama-swap", "vision-b", "https://b.test/v1")


def test_explicit_model_cache_isolation_remains_independent_of_runtime_key():
    """#56889 remains covered: explicit model values isolate non-auto clients."""
    first = aux._client_cache_key(
        "openrouter", async_mode=False, model="anthropic/claude-opus-4.8"
    )
    second = aux._client_cache_key(
        "openrouter", async_mode=False, model="openai/gpt-5.5"
    )

    assert first != second


def test_unhashable_callable_runtime_api_keys_are_safe_secret_free_discriminators():
    """Callable token providers remain cacheable without leaking returned tokens."""

    class TokenProvider(list):
        def __init__(self, token: str):
            super().__init__()
            self.token = token

        def __call__(self) -> str:
            return self.token

    first_provider = TokenProvider("first-super-secret-token")
    second_provider = TokenProvider("second-super-secret-token")

    first = aux._client_cache_key(
        "auto", async_mode=False, main_runtime={**_runtime("same"), "api_key": first_provider}
    )
    second = aux._client_cache_key(
        "auto", async_mode=False, main_runtime={**_runtime("same"), "api_key": second_provider}
    )

    hash(first)
    hash(second)
    assert first != second
    rendered = repr((first, second))
    assert "first-super-secret-token" not in rendered
    assert "second-super-secret-token" not in rendered


def test_fifo_eviction_closes_oldest_sync_client_once_after_65_entries():
    """The bounded cache closes, rather than merely drops, its FIFO sync victim."""
    clients = []

    def fake_resolve(_provider, model, _async_mode, **_kwargs):
        client = MagicMock(name=f"client-{model}")
        clients.append(client)
        return client, model

    with patch.object(aux, "resolve_provider_client", side_effect=fake_resolve):
        for index in range(65):
            aux._get_cached_client("custom", model=f"model-{index}")

    assert len(aux._client_cache) == 64
    clients[0].close.assert_called_once_with()
    for client in clients[1:]:
        client.close.assert_not_called()

    aux.shutdown_cached_clients()
    clients[0].close.assert_called_once_with()
    for client in clients[1:]:
        client.close.assert_called_once_with()
