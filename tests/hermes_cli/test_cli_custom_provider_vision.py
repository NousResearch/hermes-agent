"""End-to-end CLI coverage for named custom-provider vision routing.

Adapted from the independently reproduced CLI regression in #69896.  Unlike
that PR, these tests keep the live transport canonical (``provider=custom``)
and assert that the separate requested identity reaches each capability gate.
"""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


MODEL = "qwen3.8-max-preview"
REQUESTED_PROVIDER = "custom:qwen-token-plan"


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
        self._credential_pool = None
        self.service_tier = None

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
        ["-m", MODEL, "--provider", REQUESTED_PROVIDER, "chat"]
    )
    cli = _RuntimeCLI(model=args.model, provider=args.provider)
    assert cli._ensure_runtime_credentials() is True
    return cli, cli._resolve_turn_agent_config("inspect the image")


def test_real_cli_args_keep_transport_and_capability_identities_separate():
    from agent.image_routing import decide_image_input_mode
    from hermes_cli.config import load_config

    cli, route = _resolve_cli_route()
    runtime = route["runtime"]

    assert cli.provider == "custom"
    assert cli.requested_provider == REQUESTED_PROVIDER
    assert runtime["provider"] == "custom"
    assert runtime["requested_provider"] == REQUESTED_PROVIDER
    assert decide_image_input_mode(
        runtime["provider"],
        route["model"],
        load_config(),
        requested_provider=runtime["requested_provider"],
    ) == "native"

    # A credential refresh with the same two identities must retain the live
    # agent instead of treating canonicalization as a provider switch.
    sentinel_agent = SimpleNamespace()
    cli.agent = sentinel_agent
    assert cli._ensure_runtime_credentials() is True
    assert cli.agent is sentinel_agent


def test_named_identity_reaches_agent_and_vision_tool_native_gates():
    from agent.auxiliary_client import reset_runtime_main, set_runtime_main
    from run_agent import AIAgent
    from tools.vision_tools import _should_use_native_vision_fast_path

    _cli, route = _resolve_cli_route()
    runtime = route["runtime"]
    token = set_runtime_main(
        runtime["provider"],
        route["model"],
        requested_provider=runtime["requested_provider"],
        base_url=runtime["base_url"],
        api_key=runtime["api_key"],
        api_mode=runtime["api_mode"],
    )
    try:
        agent = AIAgent.__new__(AIAgent)
        agent.provider = runtime["provider"]
        agent.requested_provider = runtime["requested_provider"]
        agent.model = route["model"]

        assert agent._model_supports_vision() is True
        assert _should_use_native_vision_fast_path() is True
    finally:
        reset_runtime_main(token)
