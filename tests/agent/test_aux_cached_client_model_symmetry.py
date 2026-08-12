"""Cold-path / warm-path model-resolution symmetry in ``_get_cached_client``.

Regression coverage for a cache-order-dependent bug: the cache-HIT paths ran the
caller-passed ``model`` through :func:`_compat_model` (which drops a
vendor-namespaced id for a client that cannot accept one), but the cache-MISS
(cold) path returned the raw ``model`` unchanged. The observable symptom was a
first auxiliary call failing with a provider 404 on a namespaced model id while
every subsequent call with identical arguments succeeded.

These are behavior contracts on the (client, resolved_model) pair, not snapshots
of any particular model list.
"""

import pytest

import agent.auxiliary_client as ac


class _NonSlashClient:
    """A client whose endpoint rejects ``vendor/model`` ids (e.g. Anthropic)."""

    base_url = "https://api.anthropic.com"


class _OpenRouterClient:
    """A client whose endpoint REQUIRES ``vendor/model`` ids."""

    base_url = "https://openrouter.ai/api/v1"


@pytest.fixture
def stub_resolver(monkeypatch):
    """Replace the network-touching resolver and isolate the module cache."""
    monkeypatch.setattr(ac, "_client_cache", {})

    def _install(client, default_model):
        monkeypatch.setattr(
            ac,
            "resolve_provider_client",
            lambda *a, **k: (client, default_model),
        )

    yield _install


def test_cold_and_warm_paths_agree_for_namespaced_model(stub_resolver):
    """The first call and the second must resolve the SAME model id.

    This is the core invariant: whether the client cache is cold or warm is an
    internal performance detail and must never change what gets sent on the wire.
    """
    stub_resolver(_NonSlashClient(), "claude-haiku-4-5")

    _, cold = ac._get_cached_client("anthropic", model="anthropic/claude-haiku-4-5")
    _, warm = ac._get_cached_client("anthropic", model="anthropic/claude-haiku-4-5")

    assert cold == warm


def test_cold_path_drops_namespace_for_a_client_that_cannot_accept_it(stub_resolver):
    """A non-slash client gets the resolver's normalized default, not the raw id."""
    stub_resolver(_NonSlashClient(), "claude-haiku-4-5")

    _, resolved = ac._get_cached_client("anthropic", model="anthropic/claude-haiku-4-5")

    assert "/" not in resolved
    assert resolved == "claude-haiku-4-5"


def test_caller_wins_is_preserved_for_a_slash_capable_client(stub_resolver):
    """OpenRouter-style aggregator slugs must survive untouched on BOTH paths.

    Guards against over-correcting: the fix must not strip namespaces that the
    endpoint actually requires.
    """
    stub_resolver(_OpenRouterClient(), "openai/gpt-5.4")

    _, cold = ac._get_cached_client("openrouter", model="anthropic/claude-sonnet-4.5")
    _, warm = ac._get_cached_client("openrouter", model="anthropic/claude-sonnet-4.5")

    assert cold == "anthropic/claude-sonnet-4.5"
    assert warm == cold


def test_plain_model_override_still_wins_over_the_default(stub_resolver):
    """A non-namespaced caller override is unaffected by the compat guard."""
    stub_resolver(_NonSlashClient(), "claude-haiku-4-5")

    _, cold = ac._get_cached_client("anthropic", model="claude-opus-4-1")
    _, warm = ac._get_cached_client("anthropic", model="claude-opus-4-1")

    assert cold == "claude-opus-4-1"
    assert warm == cold


def test_no_model_argument_falls_back_to_the_resolver_default(stub_resolver):
    """Omitting ``model`` returns the provider default on both paths."""
    stub_resolver(_NonSlashClient(), "claude-haiku-4-5")

    _, cold = ac._get_cached_client("anthropic")
    _, warm = ac._get_cached_client("anthropic")

    assert cold == "claude-haiku-4-5"
    assert warm == cold
