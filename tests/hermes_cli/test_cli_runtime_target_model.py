from hermes_cli.auth import AuthError
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
from hermes_cli import runtime_provider as rp


class _DummyCLI(CLIAgentSetupMixin):
    def __init__(
        self,
        *,
        requested_provider: str = "opencode-go",
        model: str = "deepseek-v4-flash",
        fallback_model: list[dict[str, str]] | None = None,
    ) -> None:
        self.requested_provider = requested_provider
        self.model = model

        self._explicit_api_key = None
        self._explicit_base_url = None
        self._fallback_model = fallback_model or []

        self.api_key = "old-key"
        self.base_url = "https://opencode.ai/zen/go"
        self.provider = requested_provider
        self.api_mode = "anthropic_messages"
        self.acp_command = None
        self.acp_args = []

        self.agent = None
        self._active_agent_route_signature = None

    def _normalize_model_for_provider(self, _provider: str) -> bool:
        return False


def _configure_opencode_go(monkeypatch) -> None:
    monkeypatch.setattr(
        rp,
        "resolve_provider",
        lambda *args, **kwargs: "opencode-go",
    )
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "opencode-go",
            "default": "minimax-m2.5",
            "api_mode": "anthropic_messages",
            "base_url": "https://opencode.ai/zen/go",
        },
    )
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-opencode-go-key")
    monkeypatch.delenv("OPENCODE_GO_BASE_URL", raising=False)


def test_runtime_resolution_uses_cli_model_override(monkeypatch) -> None:
    _configure_opencode_go(monkeypatch)

    cli = _DummyCLI(model="deepseek-v4-flash")

    assert cli._ensure_runtime_credentials() is True
    assert cli.api_mode == "chat_completions"
    assert cli.base_url == "https://opencode.ai/zen/go/v1"


def test_fallback_runtime_resolution_uses_fallback_model(monkeypatch) -> None:
    _configure_opencode_go(monkeypatch)

    real_resolve_runtime_provider = rp.resolve_runtime_provider

    def resolve_with_primary_auth_failure(**kwargs):
        if kwargs.get("requested") == "broken-primary":
            raise AuthError(
                "Primary provider credentials unavailable.",
                provider="broken-primary",
                code="missing_api_key",
            )

        return real_resolve_runtime_provider(**kwargs)

    monkeypatch.setattr(
        rp,
        "resolve_runtime_provider",
        resolve_with_primary_auth_failure,
    )

    cli = _DummyCLI(
        requested_provider="broken-primary",
        model="minimax-m2.5",
        fallback_model=[
            {
                "provider": "opencode-go",
                "model": "deepseek-v4-flash",
            }
        ],
    )

    assert cli._ensure_runtime_credentials() is True
    assert cli.requested_provider == "opencode-go"
    assert cli.model == "deepseek-v4-flash"
    assert cli.api_mode == "chat_completions"
    assert cli.base_url == "https://opencode.ai/zen/go/v1"
