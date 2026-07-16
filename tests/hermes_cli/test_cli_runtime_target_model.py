from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


class _DummyCLI(CLIAgentSetupMixin):
    def __init__(self) -> None:
        self.requested_provider = "opencode-go"
        self.model = "deepseek-v4-flash"

        self._explicit_api_key = None
        self._explicit_base_url = None
        self._fallback_model = []

        self.api_key = "old-key"
        self.base_url = "https://opencode.ai/zen/go"
        self.provider = "opencode-go"
        self.api_mode = "anthropic_messages"
        self.acp_command = None
        self.acp_args = []

        self.agent = None
        self._active_agent_route_signature = None

    def _normalize_model_for_provider(self, _provider: str) -> bool:
        return False


def test_runtime_resolution_receives_cli_model_override(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_resolve_runtime_provider(**kwargs):
        captured.update(kwargs)

        return {
            "api_key": "test-key",
            "base_url": "https://opencode.ai/zen/go/v1",
            "provider": "opencode-go",
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )

    cli = _DummyCLI()

    assert cli._ensure_runtime_credentials() is True
    assert captured["target_model"] == "deepseek-v4-flash"
    assert cli.api_mode == "chat_completions"
    assert cli.base_url == "https://opencode.ai/zen/go/v1"
