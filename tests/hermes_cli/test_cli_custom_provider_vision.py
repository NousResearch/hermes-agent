"""CLI regression coverage for named custom-provider native vision."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


MODEL = "qwen3.8-max-preview"
PROVIDER = "custom:qwen-token-plan"


class _RuntimeCLI(CLIAgentSetupMixin):
    def __init__(self, *, model: str, provider: str):
        self.model = model
        self.requested_provider = provider
        self.provider = provider
        self.api_key = None
        self.base_url = None
        self.api_mode = "chat_completions"
        self.acp_command = None
        self.acp_args = []
        self.agent = None
        self._fallback_model = []
        self._explicit_api_key = None
        self._explicit_base_url = None

    def _normalize_model_for_provider(self, _provider: str) -> bool:
        return False


def _write_profile_config(hermes_home) -> None:
    (hermes_home / "config.yaml").write_text(
        """
model:
  default: ollama-cloud/glm-5.2
  provider: ollama-cloud
providers:
  qwen-token-plan:
    base_url: https://qwen-token-plan.example/v1
    api_key: test-key
    models:
      qwen3.8-max-preview:
        supports_vision: true
agent:
  image_input_mode: auto
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _resolve_cli_route():
    from hermes_cli._parser import build_top_level_parser
    from hermes_constants import get_hermes_home

    _write_profile_config(get_hermes_home())
    parser, _subparsers, _chat = build_top_level_parser()
    args, _unknown = parser.parse_known_args(
        ["-m", MODEL, "--provider", PROVIDER, "chat"]
    )
    cli = _RuntimeCLI(model=args.model, provider=args.provider)
    assert cli._ensure_runtime_credentials() is True
    return cli, cli._resolve_turn_agent_config("inspect the image")


def test_cli_named_custom_provider_routes_declared_model_natively():
    from agent.image_routing import decide_image_input_mode
    from hermes_cli.config import load_config

    cli, route = _resolve_cli_route()

    assert cli.provider == PROVIDER
    assert route["runtime"]["provider"] == PROVIDER
    assert decide_image_input_mode(cli.provider, cli.model, load_config()) == "native"

    sentinel_agent = SimpleNamespace()
    cli.agent = sentinel_agent
    assert cli._ensure_runtime_credentials() is True
    assert cli.agent is sentinel_agent


@pytest.mark.parametrize(
    ("resolved_provider", "requested_provider", "expected_provider"),
    [
        ("custom", "custom:qwen-token-plan", "custom:qwen-token-plan"),
        ("custom", "custom", "custom"),
        ("custom", "", "custom"),
        ("openrouter", "custom:qwen-token-plan", "openrouter"),
    ],
)
def test_runtime_identity_only_preserves_namespaced_custom_provider(
    monkeypatch,
    resolved_provider,
    requested_provider,
    expected_provider,
):
    runtime = {
        "provider": resolved_provider,
        "requested_provider": requested_provider,
        "api_key": "test-key",
        "base_url": "https://provider.example/v1",
        "api_mode": "chat_completions",
    }
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: runtime,
    )
    cli = _RuntimeCLI(model=MODEL, provider="custom")
    cli.api_key = runtime["api_key"]
    cli.base_url = runtime["base_url"]

    assert cli._ensure_runtime_credentials() is True
    assert cli.provider == expected_provider


def test_namespaced_identity_change_rebuilds_existing_agent(monkeypatch):
    runtime = {
        "provider": "custom",
        "requested_provider": PROVIDER,
        "api_key": "test-key",
        "base_url": "https://qwen-token-plan.example/v1",
        "api_mode": "chat_completions",
    }
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: runtime,
    )
    cli = _RuntimeCLI(model=MODEL, provider="custom")
    cli.api_key = runtime["api_key"]
    cli.base_url = runtime["base_url"]
    cli.agent = SimpleNamespace()

    assert cli._ensure_runtime_credentials() is True
    assert cli.provider == PROVIDER
    assert cli.agent is None


def test_cli_named_custom_provider_reaches_agent_and_tool_vision_gates():
    from agent.auxiliary_client import reset_runtime_main, set_runtime_main
    from run_agent import AIAgent
    from tools.vision_tools import _should_use_native_vision_fast_path

    cli, route = _resolve_cli_route()
    runtime = route["runtime"]

    agent = AIAgent.__new__(AIAgent)
    agent.provider = runtime["provider"]
    agent.model = route["model"]
    assert agent._model_supports_vision() is True

    token = set_runtime_main(
        runtime["provider"],
        route["model"],
        base_url=runtime["base_url"],
        api_key=runtime["api_key"],
        api_mode=runtime["api_mode"],
    )
    try:
        assert _should_use_native_vision_fast_path() is True
    finally:
        reset_runtime_main(token)
