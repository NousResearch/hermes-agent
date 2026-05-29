"""Task 9 (cpf-zkw.9): real base_url_source/key_source provenance on the typed
ResolvedProvider, and mtime/env-fingerprinted memoization of resolution.

Provenance (plan §2): each resolution records WHERE its base_url and api_key
came from (explicit | config.base_url | custom_provider:<name> | registry-default
| env:<VAR> | no-key-required | none | pool…). These can diverge (config base_url
+ host-derived env key) — the custom/openrouter terminal paths stamp them
distinctly; OAuth/pool/process paths derive from the shared `source` label.

Memoization (plan §4 Task 9): resolve_runtime_provider is a pure offline
function of (args + config.yaml + env + pool). Repeated calls (the gateway runs
one per message) reuse a memoized result keyed on those inputs, but an edited
config.yaml or changed env invalidates it on the next call (live reload
preserved). Pool-bearing results are never cached (rotation stays live).
"""

from __future__ import annotations

import pytest

from hermes_cli import runtime_provider as rp


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    rp.clear_resolution_memo()
    yield
    rp.clear_resolution_memo()


def _write_config(text: str):
    from hermes_cli.config import get_config_path

    get_config_path().write_text(text)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_explicit_base_url_and_key_provenance():
    resolved = rp.resolve_runtime_provider_object(
        requested="custom",
        explicit_base_url="http://localhost:1234",
        explicit_api_key="sk-explicit",
    )
    assert resolved.provider == "custom"
    assert resolved.base_url_source == "explicit"
    assert resolved.key_source == "explicit"


def test_divergent_base_url_and_key_provenance():
    """base_url from explicit endpoint, key host-derived from a vendor env var —
    the two sources genuinely differ and must be recorded distinctly."""
    import os

    os.environ["DEEPSEEK_API_KEY"] = "sk-deepseek"
    try:
        rp.clear_resolution_memo()
        resolved = rp.resolve_runtime_provider_object(
            requested="custom",
            explicit_base_url="https://api.deepseek.com/v1",
        )
        assert resolved.provider == "custom"
        assert resolved.base_url_source == "explicit"
        assert resolved.key_source == "env:host-derived"
        assert resolved.api_key == "sk-deepseek"
    finally:
        del os.environ["DEEPSEEK_API_KEY"]


def test_config_base_url_no_key_provenance():
    """A config.yaml bare custom endpoint with no key: base_url from config,
    key is the local-server placeholder."""
    _write_config(
        "model:\n  default: m\n  provider: custom\n  base_url: http://localhost:1234\n"
    )
    rp.clear_resolution_memo()
    resolved = rp.resolve_runtime_provider_object(requested="custom")
    assert resolved.provider == "custom"
    assert resolved.base_url_source == "config.base_url"
    assert resolved.key_source == "no-key-required"


def test_openrouter_registry_default_base_url_provenance():
    """Explicit openrouter request with an explicit key (bypasses the pool):
    base_url falls back to the registry default, key is explicit."""
    resolved = rp.resolve_runtime_provider_object(
        requested="openrouter", explicit_api_key="sk-or"
    )
    assert resolved.provider == "openrouter"
    assert resolved.base_url_source == "registry-default"
    assert resolved.key_source == "explicit"


def test_provenance_never_empty():
    """Every resolution populates both provenance fields (no inert placeholder)."""
    resolved = rp.resolve_runtime_provider_object(
        requested="custom", explicit_base_url="http://localhost:1234"
    )
    assert resolved.base_url_source
    assert resolved.key_source


def test_provenance_in_object_but_not_in_legacy_as_dict():
    """as_dict() stays byte-compatible (no provenance keys), but the typed
    object's mapping reads DO expose provenance — no silent-None footgun."""
    resolved = rp.resolve_runtime_provider_object(
        requested="custom", explicit_base_url="http://localhost:1234"
    )
    # Legacy dict export stays narrow (byte-compat).
    assert "base_url_source" not in resolved.as_dict()
    assert "key_source" not in resolved.as_dict()
    # But the object exposes provenance via both attribute and mapping reads —
    # .get("base_url_source") must NOT silently return None (the footgun).
    assert resolved.base_url_source == "explicit"
    assert resolved.get("base_url_source") == "explicit"
    assert "base_url_source" in resolved
    assert resolved["key_source"] == resolved.key_source


# ---------------------------------------------------------------------------
# Memoization
# ---------------------------------------------------------------------------

def test_repeated_resolution_is_memoized():
    _write_config(
        "model:\n  default: m\n  provider: custom\n  base_url: http://localhost:1234\n"
    )
    rp.clear_resolution_memo()
    a = rp.resolve_runtime_provider_object(requested="custom")
    b = rp.resolve_runtime_provider_object(requested="custom")
    assert a is b, "identical inputs should return the memoized object"


def test_config_edit_invalidates_memo():
    """Editing config.yaml takes effect on the next resolve — live reload."""
    _write_config(
        "model:\n  default: m\n  provider: custom\n  base_url: http://localhost:1111\n"
    )
    rp.clear_resolution_memo()
    first = rp.resolve_runtime_provider_object(requested="custom")
    assert first.base_url == "http://localhost:1111/v1"

    # Rewrite with a different base_url; mtime/size change must bust the memo.
    import os
    import time

    cfg = __import__("hermes_cli.config", fromlist=["get_config_path"]).get_config_path()
    # Force a distinct mtime even on coarse-resolution filesystems.
    new_mtime = os.stat(cfg).st_mtime + 10
    _write_config(
        "model:\n  default: m\n  provider: custom\n  base_url: http://localhost:2222\n"
    )
    os.utime(cfg, (new_mtime, new_mtime))

    second = rp.resolve_runtime_provider_object(requested="custom")
    assert second.base_url == "http://localhost:2222/v1", (
        "edited config.yaml must invalidate the memo (live reload)"
    )
    assert second is not first


def test_env_change_invalidates_memo(monkeypatch):
    rp.clear_resolution_memo()
    first = rp.resolve_runtime_provider_object(
        requested="custom", explicit_base_url="https://api.deepseek.com/v1"
    )
    assert first.key_source == "no-key-required"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    second = rp.resolve_runtime_provider_object(
        requested="custom", explicit_base_url="https://api.deepseek.com/v1"
    )
    assert second.key_source == "env:host-derived", (
        "changed env must invalidate the memo and re-resolve the key"
    )
    assert second is not first


def test_oauth_portal_results_are_not_memoized(monkeypatch):
    """OAuth/portal/process providers return an expiring token snapshot (no
    pool); memoizing would freeze it and disable the resolver's proactive
    refresh. Such results MUST re-resolve every call (review C1)."""
    calls = {"n": 0}

    def _fake_impl(**_kwargs):
        calls["n"] += 1
        return {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": f"token-{calls['n']}",  # fresh token each resolve
            "source": "hermes-auth-store",
            "requested_provider": "openai-codex",
        }

    monkeypatch.setattr(rp, "_resolve_runtime_provider_impl", _fake_impl)
    rp.clear_resolution_memo()
    a = rp.resolve_runtime_provider_object(requested="openai-codex")
    b = rp.resolve_runtime_provider_object(requested="openai-codex")
    assert a is not b, "OAuth/portal results must not be memoized"
    assert calls["n"] == 2, "the resolver (token refresh) must run every call"
    assert b.api_key == "token-2"


def test_is_memoizable_allowlist():
    from hermes_cli.provider_resolution import ResolvedProvider

    def _mk(source, pool=None):
        return ResolvedProvider(
            provider="custom", requested_provider="custom",
            api_mode="chat_completions", base_url="http://h/v1", api_key="k",
            base_url_source=source, key_source=source,
            credential_pool=pool, extra={"source": source},
        )

    # Static sources → cacheable.
    for s in ("explicit", "env", "config", "env/config", "direct-alias",
              "registry-default", "azure-explicit", "custom_provider:beans"):
        assert rp._is_memoizable(_mk(s)), s
    # Dynamic/expiring sources → never cacheable.
    for s in ("portal", "hermes-auth-store", "qwen-cli", "oauth",
              "google-oauth", "process", "pool", "pool:abc"):
        assert not rp._is_memoizable(_mk(s)), s
    # Pool object present → never cacheable regardless of source.
    assert not rp._is_memoizable(_mk("explicit", pool=object()))


def test_clear_resolution_memo():
    _write_config(
        "model:\n  default: m\n  provider: custom\n  base_url: http://localhost:1234\n"
    )
    rp.clear_resolution_memo()
    a = rp.resolve_runtime_provider_object(requested="custom")
    rp.clear_resolution_memo()
    b = rp.resolve_runtime_provider_object(requested="custom")
    assert a is not b, "cleared memo forces a fresh resolution"
    assert a.base_url == b.base_url
