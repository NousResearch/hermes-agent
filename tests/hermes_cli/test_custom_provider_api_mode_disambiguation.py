"""Regression tests for issue #102725.

Two named ``custom_providers`` entries that share one ``base_url`` and serve
one model id but use different ``api_mode`` values must not collapse to the
first config entry during identity recovery. The persisted ``api_mode``
(stored alongside ``base_url`` in the session's ``gateway_runtime``)
disambiguates them; without the hint the historical first-match behavior is
preserved.
"""

import hermes_cli.runtime_provider as rp

URL = "https://example.com/v1"
MODEL = "model-x"

CONFIG = {
    "custom_providers": [
        {
            "name": "provider-chat",
            "base_url": URL,
            "api_key": "sk-chat",
            "api_mode": "chat_completions",
            "model": MODEL,
            "models": {MODEL: {"context_length": 1000000}},
        },
        {
            "name": "provider-responses",
            "base_url": URL,
            "api_key": "sk-responses",
            "api_mode": "codex_responses",
            "model": MODEL,
            "models": {MODEL: {"context_length": 1000000}},
        },
    ]
}


def _cfg(monkeypatch, config=None):
    monkeypatch.setattr(rp, "load_config", lambda: config if config is not None else CONFIG)


def test_url_lookup_prefers_matching_api_mode(monkeypatch):
    _cfg(monkeypatch)
    assert (
        rp.find_custom_provider_identity(URL, api_mode="codex_responses")
        == "custom:provider-responses"
    )
    assert (
        rp.find_custom_provider_identity(URL, api_mode="chat_completions")
        == "custom:provider-chat"
    )


def test_url_lookup_without_hint_keeps_first_match(monkeypatch):
    _cfg(monkeypatch)
    assert rp.find_custom_provider_identity(URL) == "custom:provider-chat"


def test_url_lookup_with_unmatched_hint_returns_none(monkeypatch):
    """A usable hint that matches no candidate must NOT fall back to the first
    entry (which may be a different wire mode) — reviewer follow-up on #102748.
    Historical first-wins applies only when no usable ``api_mode`` is given."""
    _cfg(monkeypatch)
    assert rp.find_custom_provider_identity(URL, api_mode="anthropic_messages") is None


def test_model_lookup_prefers_matching_api_mode(monkeypatch):
    _cfg(monkeypatch)
    assert (
        rp.find_custom_provider_identity_by_model(MODEL, api_mode="codex_responses")
        == "custom:provider-responses"
    )
    assert (
        rp.find_custom_provider_identity_by_model(MODEL, api_mode="chat_completions")
        == "custom:provider-chat"
    )


def test_model_lookup_without_hint_keeps_first_match(monkeypatch):
    _cfg(monkeypatch)
    assert rp.find_custom_provider_identity_by_model(MODEL) == "custom:provider-chat"


def test_canonical_identity_uses_api_mode_hint(monkeypatch):
    _cfg(monkeypatch)
    assert (
        rp.canonical_custom_identity(
            base_url=URL, model=MODEL, api_mode="codex_responses"
        )
        == "custom:provider-responses"
    )
    assert (
        rp.canonical_custom_identity(
            base_url=URL, model=MODEL, api_mode="chat_completions"
        )
        == "custom:provider-chat"
    )
    # Model-only recovery (no base_url survived) is disambiguated too.
    assert (
        rp.canonical_custom_identity(model=MODEL, api_mode="codex_responses")
        == "custom:provider-responses"
    )


def test_canonical_identity_without_hint_keeps_first_match(monkeypatch):
    _cfg(monkeypatch)
    assert (
        rp.canonical_custom_identity(base_url=URL, model=MODEL)
        == "custom:provider-chat"
    )


def test_implicit_chat_entry_matches_chat_hint(monkeypatch):
    """An entry with no explicit ``api_mode`` behaves as chat_completions."""
    _cfg(
        monkeypatch,
        {
            "custom_providers": [
                {"name": "implicit-chat", "base_url": URL, "models": {MODEL: {}}},
                {
                    "name": "explicit-responses",
                    "base_url": URL,
                    "api_mode": "codex_responses",
                    "models": {MODEL: {}},
                },
            ]
        },
    )
    assert (
        rp.find_custom_provider_identity(URL, api_mode="chat_completions")
        == "custom:implicit-chat"
    )
    assert (
        rp.find_custom_provider_identity(URL, api_mode="codex_responses")
        == "custom:explicit-responses"
    )


def test_providers_dict_shape_with_transport(monkeypatch):
    """New-style ``providers:`` entries using ``transport`` are honored."""
    _cfg(
        monkeypatch,
        {
            "providers": {
                "ep-chat": {"api": URL, "transport": "chat_completions"},
                "ep-responses": {"api": URL, "transport": "codex_responses"},
            }
        },
    )
    assert (
        rp.find_custom_provider_identity(URL, api_mode="codex_responses")
        == "custom:ep-responses"
    )
    assert (
        rp.find_custom_provider_identity(URL, api_mode="chat_completions")
        == "custom:ep-chat"
    )


def test_healed_slug_resolves_back_with_right_mode(monkeypatch):
    """The healed identity must route through the intended entry's mode."""
    _cfg(monkeypatch)
    slug = rp.canonical_custom_identity(
        base_url=URL, model=MODEL, api_mode="codex_responses"
    )
    assert slug == "custom:provider-responses"
    entry = rp._get_named_custom_provider(slug)
    assert entry is not None
    assert entry["api_mode"] == "codex_responses"
    assert entry["api_key"] == "sk-responses"


def test_hint_with_all_entries_broken_returns_none_not_first(monkeypatch):
    """Reviewer follow-up (#102748): when ``api_mode`` is set but every matching
    entry throws during mode probing, recovery must return None — never the
    first sibling, which may be a different wire mode."""
    _cfg(
        monkeypatch,
        {
            "custom_providers": [
                {
                    "name": "provider-chat",
                    "base_url": URL,
                    "api_key": "sk-chat",
                    "api_mode": "chat_completions",
                    "model": MODEL,
                    "models": {MODEL: {"context_length": 1000000}},
                },
                {
                    "name": "provider-responses",
                    "base_url": URL,
                    "api_key": "sk-responses",
                    "api_mode": "codex_responses",
                    "model": MODEL,
                    "models": {MODEL: {"context_length": 1000000}},
                },
            ]
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider_custom._effective_custom_entry_api_mode",
        lambda entry: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )
    assert rp.find_custom_provider_identity(URL, api_mode="codex_responses") is None
    assert (
        rp.find_custom_provider_identity_by_model(MODEL, api_mode="codex_responses")
        is None
    )
