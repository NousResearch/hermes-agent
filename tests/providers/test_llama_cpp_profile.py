"""Tests for the llama-cpp provider profile."""

from __future__ import annotations

import sys

import pytest


def _clear_provider_caches():
    """Force providers/__init__.py to re-discover on next list_providers()."""
    import providers as _pkg

    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._discovered = False
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("plugins.model_providers")
            or mod.startswith("_hermes_user_provider")
        ):
            del sys.modules[mod]


@pytest.fixture(autouse=True)
def _isolated_registry():
    _clear_provider_caches()
    yield
    _clear_provider_caches()


def test_llama_cpp_registers():
    from providers import get_provider_profile

    profile = get_provider_profile("llama-cpp")
    assert profile is not None
    assert profile.name == "llama-cpp"


def test_llama_cpp_aliases_resolve():
    from providers import get_provider_profile

    for alias in ("llamacpp", "llama.cpp", "llama_cpp", "llama-server"):
        profile = get_provider_profile(alias)
        assert profile is not None, f"alias {alias!r} did not resolve"
        assert profile.name == "llama-cpp", (
            f"alias {alias!r} resolved to {profile.name!r}"
        )


def test_llama_cpp_default_base_url_matches_launcher_script():
    """The plugin's base_url must line up with scripts/start-llama-server.sh
    so users get zero-config UX out of the box."""
    from providers import get_provider_profile

    profile = get_provider_profile("llama-cpp")
    assert profile.base_url == "http://127.0.0.1:8088/v1"


def test_llama_cpp_env_vars():
    from providers import get_provider_profile

    profile = get_provider_profile("llama-cpp")
    assert "LLAMA_CPP_API_KEY" in profile.env_vars
    assert "LLAMA_CPP_BASE_URL" in profile.env_vars


def test_llama_cpp_uses_chat_completions_api_mode():
    """llama-server speaks OpenAI chat-completions; no native adapter needed."""
    from providers import get_provider_profile

    profile = get_provider_profile("llama-cpp")
    assert profile.api_mode == "chat_completions"


def test_custom_no_longer_claims_llamacpp_aliases():
    """The llama.cpp aliases moved off the `custom` profile onto the dedicated
    `llama-cpp` profile. Make sure custom didn't keep them."""
    from providers import get_provider_profile

    custom = get_provider_profile("custom")
    assert custom is not None
    for alias in ("llamacpp", "llama.cpp", "llama-cpp", "llama_cpp"):
        assert alias not in custom.aliases, (
            f"alias {alias!r} should have moved off `custom` onto `llama-cpp`"
        )


def test_fetch_models_returns_none_when_server_offline():
    """The override must swallow connection errors — local server may not be up."""
    from providers import get_provider_profile

    profile = get_provider_profile("llama-cpp")
    # 127.0.0.1:1 is guaranteed-closed on every CI box.
    profile.base_url = "http://127.0.0.1:1/v1"
    try:
        result = profile.fetch_models(timeout=0.5)
    finally:
        profile.base_url = "http://127.0.0.1:8088/v1"
    assert result is None


def test_llama_cpp_in_canonical_providers():
    """Plugin must auto-extend hermes_cli.models.CANONICAL_PROVIDERS so the
    `hermes model` TUI picker shows llama.cpp without further wiring."""
    from hermes_cli.models import CANONICAL_PROVIDERS

    slugs = [p.slug for p in CANONICAL_PROVIDERS]
    assert "llama-cpp" in slugs
    entry = next(p for p in CANONICAL_PROVIDERS if p.slug == "llama-cpp")
    assert entry.label == "llama.cpp"
    assert "local" in entry.tui_desc.lower()


def test_llama_cpp_in_provider_registry():
    """auth.PROVIDER_REGISTRY must contain llama-cpp so runtime resolution +
    dashboard /api/model/options can authenticate (or skip auth) cleanly."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    assert "llama-cpp" in PROVIDER_REGISTRY
    cfg = PROVIDER_REGISTRY["llama-cpp"]
    assert cfg.auth_type == "api_key"
    assert cfg.inference_base_url == "http://127.0.0.1:8088/v1"
    assert cfg.api_key_env_vars == ("LLAMA_CPP_API_KEY",)
    assert cfg.base_url_env_var == "LLAMA_CPP_BASE_URL"


def test_resolve_provider_accepts_all_aliases():
    """`hermes --provider <alias>` and `HERMES_INFERENCE_PROVIDER=<alias>` must
    resolve every llama.cpp alias to the canonical `llama-cpp` slug."""
    from hermes_cli.auth import resolve_provider

    for alias in ("llama-cpp", "llamacpp", "llama.cpp", "llama_cpp", "llama-server"):
        assert resolve_provider(alias) == "llama-cpp", (
            f"alias {alias!r} should resolve to 'llama-cpp'"
        )


def test_picker_surfaces_llama_cpp_when_current_even_without_server(monkeypatch):
    """Dashboard /api/model/options + `hermes model` must show llama.cpp when
    it's the user's current provider, even if the local server is offline —
    so users don't lose access to it after restart."""
    monkeypatch.delenv("LLAMA_CPP_API_KEY", raising=False)
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)

    from hermes_cli.model_switch import list_authenticated_providers

    rows = list_authenticated_providers(
        current_provider="llama-cpp",
        current_model="my-loaded-model",
    )
    matches = [r for r in rows if r["slug"] == "llama-cpp"]
    assert len(matches) == 1, f"expected 1 llama-cpp row, got {len(matches)}"
    row = matches[0]
    assert row["is_current"] is True
    assert row["models"] == ["my-loaded-model"]
    assert row["source"] == "built-in"


def test_picker_does_not_clutter_when_server_offline_and_not_current(monkeypatch):
    """With no env vars, no current selection, and no live server, the picker
    must NOT inject a llama-cpp row — keeps the list tidy for users who don't
    use llama.cpp."""
    monkeypatch.delenv("LLAMA_CPP_API_KEY", raising=False)
    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)

    from hermes_cli.model_switch import list_authenticated_providers

    rows = list_authenticated_providers()
    assert "llama-cpp" not in [r["slug"] for r in rows]
