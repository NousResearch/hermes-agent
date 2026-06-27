"""Regression: the auxiliary client cache key must isolate per-model transports.

Root cause (fixed 2026-06-27): ``_client_cache_key`` did not distinguish
clients that pick a *different transport wrapper per model* on the same
endpoint. Copilot routes GPT-5.x through the Responses API (CodexAuxiliaryClient
wrapper) and Claude / gpt-5-mini through Chat Completions, but ``api_mode`` is
empty for all of them at key-construction time. So two Copilot models sharing
one base_url+api_key collided on a single cache entry: whichever ran first
(e.g. a ``gpt-5.5`` MoA reference) installed its Responses-API wrapper, and the
next model (e.g. a ``claude-opus-4.8`` aggregator) reused it and got
``400 unsupported_api_for_model``. This made every "two same-provider models
with different transports" MoA preset fail deterministically.

The fix folds a *transport discriminator* into the key — non-empty ONLY where
the wrapper genuinely differs per model (Copilot). Normal OpenAI-compatible
providers keep their single-client-per-endpoint reuse (one client, model
swapped at call time), so this must NOT split their cache entries.

These are behavior-contract tests (assert the invariant), not snapshots.
"""

from agent.auxiliary_client import _client_cache_key, _transport_discriminator


def _key(provider, model, **kw):
    return _client_cache_key(provider, model=model, async_mode=False, **kw)


# ── The bug: Copilot transports must not collide ─────────────────────────

def test_copilot_responses_and_chat_models_get_distinct_keys():
    """The exact MoA failure mode: gpt-5.5 (Responses) and claude-opus-4.8
    (Chat) on one Copilot base_url+key must NOT share a cache entry."""
    base = "https://api.githubcopilot.com"
    key = "tok-abc"
    k_gpt = _key("copilot", "gpt-5.5", base_url=base, api_key=key)
    k_opus = _key("copilot", "claude-opus-4.8", base_url=base, api_key=key)
    assert k_gpt != k_opus, (
        "gpt-5.5 (Responses) and claude-opus-4.8 (Chat) collided — the wrapper "
        "would leak between them (400 unsupported_api_for_model)"
    )


def test_transport_discriminator_classifies_copilot_models():
    assert _transport_discriminator("copilot", "gpt-5.5") == "responses"
    assert _transport_discriminator("copilot", "gpt-5.4") == "responses"
    assert _transport_discriminator("copilot", "claude-opus-4.8") == "chat"
    assert _transport_discriminator("copilot", "gpt-5-mini") == "chat"  # documented exception


# ── The constraint: normal providers must keep one-client reuse ──────────

def test_non_copilot_models_share_key_preserving_client_reuse():
    """GMI / OpenRouter / any normal endpoint serves all models through one
    client; two models on the same endpoint MUST map to the same cache key so
    the client is built once and reused (model swapped at call time)."""
    base = "https://api.gmi-serving.com/v1"
    k_a = _key("gmi", "google/gemini-3.1-flash-lite-preview", base_url=base, api_key="gmi-key")
    k_b = _key("gmi", "openai/gpt-5.4-mini", base_url=base, api_key="gmi-key")
    assert k_a == k_b, "normal-provider models must reuse one cached client"
    assert _transport_discriminator("gmi", "anything") == ""


def test_same_copilot_model_is_cache_stable():
    base = "https://api.githubcopilot.com"
    a = _key("copilot", "gpt-5.5", base_url=base, api_key="tok")
    b = _key("copilot", "gpt-5.5", base_url=base, api_key="tok")
    assert a == b


# ── End-to-end through the real cache: client identity is the true contract ──
# These mirror the existing GMI reuse test (TestAuxiliaryPoolAwareness::
# test_cached_gmi_client_keeps_explicit_slash_model_override) but exercise the
# Copilot collision the fix targets. Asserting on the *client object* (built vs
# reused) is a stronger behavior contract than asserting on the key tuple — it
# proves the actual runtime effect (no wrapper leak), not the key's shape.

def test_copilot_two_transports_build_two_clients_no_collision():
    """Two Copilot models with different transports on one base_url+key must
    each get their OWN cached client — the gpt-5.5 Responses wrapper must NOT
    be reused for claude-opus-4.8 (which caused 400 unsupported_api_for_model).
    """
    import agent.auxiliary_client as aux
    from unittest.mock import patch

    base = "https://api.githubcopilot.com"
    # Distinct sentinels so we can prove which client each call returns.
    responses_client = object()
    chat_client = object()

    def fake_resolve(provider, model, *a, **kw):
        # Mimic the real per-model transport split without network.
        if model and model.startswith("gpt-5") and not model.startswith("gpt-5-mini"):
            return responses_client, model
        return chat_client, model

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=fake_resolve):
        aux.shutdown_cached_clients()
        try:
            c_ref, _ = aux._get_cached_client("copilot", "gpt-5.5", base_url=base, api_key="k")
            c_agg, _ = aux._get_cached_client("copilot", "claude-opus-4.8", base_url=base, api_key="k")
        finally:
            aux.shutdown_cached_clients()

    assert c_ref is responses_client
    assert c_agg is chat_client
    assert c_ref is not c_agg, "Copilot transports collided — the bug is back"


def test_copilot_same_model_reuses_one_client():
    """The cache must still WORK for Copilot: repeated calls for the same model
    reuse one client (build happens once), so the fix doesn't defeat caching."""
    import agent.auxiliary_client as aux
    from unittest.mock import patch

    base = "https://api.githubcopilot.com"
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(object(), "gpt-5.5"),
    ) as mock_resolve:
        aux.shutdown_cached_clients()
        try:
            c1, _ = aux._get_cached_client("copilot", "gpt-5.5", base_url=base, api_key="k")
            c2, _ = aux._get_cached_client("copilot", "gpt-5.5", base_url=base, api_key="k")
        finally:
            aux.shutdown_cached_clients()

    assert c1 is c2
    assert mock_resolve.call_count == 1, "same Copilot model rebuilt — caching defeated"
