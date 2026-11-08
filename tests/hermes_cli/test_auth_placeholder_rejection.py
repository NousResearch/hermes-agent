"""Placeholder credentials must read as "not configured", never as a usable secret.

``.env.example`` ships ``your_key_here`` for four providers (Xiaomi, Upstage, Ramp Router,
Nebius) and ``your_google_ai_studio_key_here`` / ``your_gemini_key_here`` /
``your_ollama_key_here`` for three more; the quickstart and MCP/skill references use
``sk-xxx`` / ``ghp_xxx`` / ``hf_xxx``. ``has_usable_secret`` only knew the exact string
``your_api_key_here``, so every other shipped placeholder was treated as a real credential and
sent upstream — an opaque 401 instead of failing loud at the read point.

Regression for the placeholder shapes; the pooled-key case covers the sibling resolution path.
"""

import pytest

from hermes_cli.auth import has_usable_secret, AuthError


@pytest.mark.parametrize("value, usable", [
    # every placeholder shape shipped by this repo's own .env.example
    ("your_key_here", False),
    ("your_google_ai_studio_key_here", False),
    ("your_gemini_key_here", False),
    ("your_ollama_key_here", False),
    ("your_api_key_here", False),
    # the x-run convention used in quickstart / MCP / skill references
    ("sk-xxx", False),
    ("sk-XXXX", False),
    ("ghp_xxxxxxxxxxxxxxxxxxxx", False),
    ("xxxx xxxx xxxx xxxx", False),
    ("hf_xxxx", False),
    ("XXXXXXXX", False),
    # real-looking keys keep resolving
    ("sk-or-v1-abc123", True),
    ("ghp_real_token_here", True),
    ("hf_real_token", True),
    ("xai-real-key-123", True),
    ("sk-test-1234567890abcdef", True),
])
def test_shipped_placeholders_are_not_usable_secrets(value, usable):
    assert has_usable_secret(value) is usable


def test_placeholder_keys_resolve_as_unconfigured(tmp_path, monkeypatch):
    """Production entry point: a placeholder in the env fails loud at the read point, and a pooled
    placeholder (the sibling resolution path) behaves exactly like no credential at all."""
    import uuid

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "your_key_here")

    from agent.credential_pool import AUTH_TYPE_API_KEY, SOURCE_MANUAL, PooledCredential, load_pool
    from hermes_cli.runtime_provider import resolve_runtime_provider

    with pytest.raises(AuthError, match="No usable credentials found for provider 'xai'"):
        resolve_runtime_provider(requested="xai")

    pool = load_pool("openrouter")
    pool.add_entry(PooledCredential(
        provider="openrouter", id=uuid.uuid4().hex[:6], label="pasted-example",
        auth_type=AUTH_TYPE_API_KEY, priority=0, source=SOURCE_MANUAL,
        access_token="your_key_here", base_url="https://openrouter.ai/api/v1",
    ))
    runtime = resolve_runtime_provider(requested="openrouter")
    assert runtime.get("api_key") == "", "a pooled .env.example placeholder was used as a key"
