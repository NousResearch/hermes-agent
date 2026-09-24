"""Built-in provider metadata must not become a custom endpoint."""

from hermes_cli.providers import resolve_provider_full, resolve_user_provider


def test_metadata_only_provider_block_keeps_builtin_provider():
    configured = {
        "openai-codex": {
            "name": "Codex",
            "models": ["gpt-6-sol-900k"],
            "stale_timeout_seconds": 900,
        }
    }

    assert resolve_user_provider("openai-codex", configured) is None
    resolved = resolve_provider_full("openai-codex", configured)
    assert resolved is not None
    assert resolved.source != "user-config"
    assert resolved.transport == "codex_responses"


def test_endpoint_provider_block_still_overrides_builtin_provider():
    configured = {
        "openai-codex": {
            "name": "Local Codex-compatible endpoint",
            "base_url": "http://127.0.0.1:9999/v1",
            "transport": "openai_chat",
        }
    }

    resolved = resolve_user_provider("openai-codex", configured)
    assert resolved is not None
    assert resolved.source == "user-config"
    assert resolved.base_url == "http://127.0.0.1:9999/v1"
