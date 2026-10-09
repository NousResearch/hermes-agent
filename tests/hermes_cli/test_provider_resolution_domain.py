"""Provider resolution consumes the canonical provider-domain contract."""

from providers import ResolvedProvider
from providers.routing import InvocationRequest, resolve_invocation_route

from hermes_cli.providers import (
    get_provider,
    resolve_custom_provider,
    resolve_provider_full,
    resolve_user_provider,
)


def test_profile_resolution_returns_resolved_provider(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    resolved = get_provider("vercel", allow_network=False)

    assert isinstance(resolved, ResolvedProvider)
    assert resolved.id == "ai-gateway"
    assert resolved.display_name == "Vercel AI Gateway"
    assert resolved.api_mode == "chat_completions"
    assert resolved.env_vars == ("AI_GATEWAY_API_KEY",)
    assert resolved.base_url == "https://ai-gateway.vercel.sh/v1"
    assert resolved.base_url_env_var == "AI_GATEWAY_BASE_URL"
    assert resolved.is_aggregator is True
    assert resolved.is_routing_aggregator is True


def test_profile_resolution_preserves_non_routing_aggregator(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    resolved = get_provider("opencode", allow_network=False)

    assert isinstance(resolved, ResolvedProvider)
    assert resolved.id == "opencode-zen"
    assert resolved.is_aggregator is True
    assert resolved.is_routing_aggregator is False


def test_user_provider_resolution_uses_canonical_fields() -> None:
    resolved = resolve_user_provider(
        "lan",
        {
            "lan": {
                "name": "LAN",
                "base_url": "http://127.0.0.1:8000/v1",
                "key_env": "LAN_KEY",
                "transport": "anthropic_messages",
            }
        },
    )

    assert isinstance(resolved, ResolvedProvider)
    assert resolved.id == "lan"
    assert resolved.display_name == "LAN"
    assert resolved.api_mode == "anthropic_messages"
    assert resolved.env_vars == ("LAN_KEY",)
    assert resolved.source == "user-config"


def test_user_provider_resolves_canonical_named_custom_identity() -> None:
    resolved = resolve_user_provider(
        "custom:ollama",
        {
            "ollama": {
                "name": "Ollama",
                "base_url": "https://ollama.internal/v1",
                "key_env": "OLLAMA_API_KEY",
            }
        },
    )

    assert isinstance(resolved, ResolvedProvider)
    assert resolved.id == "ollama"
    assert resolved.display_name == "Ollama"
    assert resolved.source == "user-config"


def test_custom_provider_resolution_uses_canonical_fields() -> None:
    resolved = resolve_custom_provider(
        "custom:lab",
        [
            {
                "name": "Lab",
                "provider_key": "lab",
                "base_url": "http://lab.invalid/v1",
                "key_env": "LAB_KEY",
            }
        ],
    )

    assert isinstance(resolved, ResolvedProvider)
    assert resolved.id == "custom:lab"
    assert resolved.display_name == "Lab"
    assert resolved.api_mode == "chat_completions"
    assert resolved.env_vars == ("LAB_KEY",)


def test_raw_user_provider_still_precedes_profile_alias(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    resolved = resolve_provider_full(
        "openai",
        user_providers={
            "openai": {
                "name": "Direct OpenAI",
                "base_url": "https://api.openai.com/v1",
                "key_env": "OPENAI_API_KEY",
                "transport": "codex_responses",
            }
        },
    )

    assert isinstance(resolved, ResolvedProvider)
    assert resolved.id == "openai"
    assert resolved.display_name == "Direct OpenAI"
    assert resolved.api_mode == "codex_responses"
    assert resolved.source == "user-config"


def test_api_mode_reads_canonical_resolved_provider(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert resolve_invocation_route(InvocationRequest(provider="minimax")).api_mode == "anthropic_messages"
    assert resolve_invocation_route(InvocationRequest(provider="openai-codex")).api_mode == "codex_responses"
