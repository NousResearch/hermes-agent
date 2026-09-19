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


class TestHasUsableSecretPlaceholderRejection:
    """has_usable_secret must reject common placeholder patterns."""

    @pytest.mark.parametrize("placeholder", [
        # every placeholder shape shipped by this repo's own .env.example
        "your_key_here",
        "your_google_ai_studio_key_here",
        "your_gemini_key_here",
        "your_ollama_key_here",
        "your_api_key_here",
        # the x-run convention used in quickstart / MCP / skill references
        "sk-xxx",
        "sk-XXXX",
        "ghp_xxxxxxxxxxxxxxxxxxxx",
        "xxxx xxxx xxxx xxxx",
        "hf_xxxx",
        "XXXXXXXX",
    ])
    def test_rejects_placeholder_patterns(self, placeholder):
        assert not has_usable_secret(placeholder)

    @pytest.mark.parametrize("real_key", [
        "sk-or-v1-abc123",
        "ghp_real_token_here",
        "hf_real_token",
        "xai-real-key-123",
        "sk-test-1234567890abcdef",
    ])
    def test_accepts_real_looking_keys(self, real_key):
        assert has_usable_secret(real_key)


class TestRuntimeRejectsPlaceholderApiKey:
    """Placeholder env vars must raise AuthError with guidance at resolution time."""

    def test_xai_placeholder_raises(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "your_key_here")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        with pytest.raises(AuthError, match="No usable credentials found for provider 'xai'"):
            resolve_runtime_provider(requested="xai")

    def test_openrouter_placeholder_fallback_to_empty_key(self, monkeypatch):
        """OpenRouter silently falls back to free-tier when the env key is a placeholder."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-xxx")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider(requested="openrouter")
        assert runtime.get("api_key") == ""
        assert runtime.get("provider") == "openrouter"

    def test_pooled_placeholder_is_not_used_as_a_credential(self, tmp_path, monkeypatch):
        """The credential pool is the sibling path: a pooled placeholder must behave exactly like
        no credential at all, never like a configured key."""
        import uuid

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        from agent.credential_pool import AUTH_TYPE_API_KEY, SOURCE_MANUAL, PooledCredential, load_pool

        pool = load_pool("openrouter")
        pool.add_entry(PooledCredential(
            provider="openrouter", id=uuid.uuid4().hex[:6], label="pasted-example",
            auth_type=AUTH_TYPE_API_KEY, priority=0, source=SOURCE_MANUAL,
            access_token="your_key_here", base_url="https://openrouter.ai/api/v1",
        ))

        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider(requested="openrouter")
        assert runtime.get("api_key") == "", "a pooled .env.example placeholder was used as a key"
