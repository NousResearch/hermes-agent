"""Regression tests: switch_model() must reconcile request_overrides['extra_body']
on every live /model switch, for both agent-init and live-switch provider
identity conventions.

Root-cause bugs fixed together:

1. ``agent.provider`` carries the entry's identity DIFFERENTLY depending on
   when it's resolved. At agent-init (hermes_cli/runtime_provider.py::
   _resolve_named_custom_runtime) it's always the bare canonical "custom",
   with the real identity only on ``agent.requested_provider``. On a live
   ``/model`` switch (hermes_cli/model_switch.py's pure switch_model(), which
   every gateway/TUI/CLI switch path calls before mutating the live agent)
   ``target_provider`` — and so ``agent.provider`` after the swap — is the
   entry's OWN raw identity instead (e.g. "vllm"), never "custom". Matching
   extra_body only recognized the first convention, so two providers:
   entries sharing (base_url, model) — e.g. a vLLM endpoint listed twice as
   "vllm" / "vllm-no-think", the latter pinning
   extra_body.chat_template_kwargs.enable_thinking for a hybrid-thinking
   model — were indistinguishable and the WRONG entry's extra_body could be
   applied regardless of which was actually selected, or (for the live-
   switch convention) extra_body silently stopped resolving at all.

2. switch_model() never touched request_overrides['extra_body'] at all, so a
   provider switched away from would leave its extra_body stuck on every
   request for the rest of the session even after switching to one with none
   configured.
"""

from unittest.mock import MagicMock, patch


def _make_agent(provider="vllm", requested_provider=None, extra_body=None):
    """Minimal AIAgent with just enough surface for switch_model()'s
    non-anthropic (openai-client) branch, mirroring
    tests/run_agent/test_switch_model_reapplies_headers.py."""
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.model = "unsloth/Qwen3.6-35B-A3B-NVFP4"
    agent.provider = provider
    agent.requested_provider = requested_provider if requested_provider is not None else provider
    agent.base_url = "http://192.168.15.115:8000/v1"
    agent.api_key = "no-key-required"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True
    agent._config_context_length = None
    agent._client_kwargs = {"api_key": "no-key-required", "base_url": agent.base_url}
    agent.request_overrides = dict(extra_body and {"extra_body": extra_body} or {})
    agent._custom_provider_extra_body_keys = set(extra_body or {})
    agent._credential_pool = None
    agent._credential_pool_entry_id = None
    agent._transport_cache = {}
    agent.context_compressor = None
    agent._primary_runtime = {}
    agent._cached_system_prompt = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._is_anthropic_oauth = False
    agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
    agent._ensure_lmstudio_runtime_loaded = MagicMock(return_value=None)
    agent._lmstudio_load_was_unverified = MagicMock(return_value=False)
    agent._effective_lmstudio_context_length = MagicMock(return_value=None)
    agent._create_openai_client = MagicMock(return_value=MagicMock())
    agent._apply_client_headers_for_base_url = MagicMock()
    return agent


_CUSTOM_PROVIDERS = [
    {
        "name": "vLLM",
        "provider_key": "vllm",
        "base_url": "http://192.168.15.115:8000/v1",
        "model": "unsloth/Qwen3.6-35B-A3B-NVFP4",
    },
    {
        "name": "vLLM No-Think",
        "provider_key": "vllm-no-think",
        "base_url": "http://192.168.15.115:8000/v1",
        "model": "unsloth/Qwen3.6-35B-A3B-NVFP4",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
]


@patch("agent.model_metadata.get_model_context_length", return_value=131_072)
@patch("hermes_cli.config.get_custom_provider_context_length", return_value=None)
@patch("hermes_cli.config.get_compatible_custom_providers", return_value=_CUSTOM_PROVIDERS)
@patch("hermes_cli.config.load_config", return_value={})
def test_switch_to_no_think_variant_picks_up_its_extra_body(
    mock_cfg, mock_cps, mock_ctx_len_cp, mock_ctx_len
):
    """Real invocation convention: new_provider is the entry's OWN identity
    ("vllm-no-think"), matching what hermes_cli/model_switch.py's
    target_provider actually resolves to for a named providers: entry — NOT
    the "custom" canonical form used at agent-init."""
    agent = _make_agent(provider="vllm")

    agent.switch_model(
        "unsloth/Qwen3.6-35B-A3B-NVFP4",
        "vllm-no-think",
        base_url="http://192.168.15.115:8000/v1",
    )

    assert agent.request_overrides.get("extra_body") == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


@patch("agent.model_metadata.get_model_context_length", return_value=131_072)
@patch("hermes_cli.config.get_custom_provider_context_length", return_value=None)
@patch("hermes_cli.config.get_compatible_custom_providers", return_value=_CUSTOM_PROVIDERS)
@patch("hermes_cli.config.load_config", return_value={})
def test_switch_away_from_no_think_variant_clears_stale_extra_body(
    mock_cfg, mock_cps, mock_ctx_len_cp, mock_ctx_len
):
    agent = _make_agent(
        provider="vllm-no-think",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    agent.switch_model(
        "unsloth/Qwen3.6-35B-A3B-NVFP4",
        "vllm",
        base_url="http://192.168.15.115:8000/v1",
    )

    assert agent.request_overrides.get("extra_body") is None


@patch("agent.model_metadata.get_model_context_length", return_value=131_072)
@patch("hermes_cli.config.get_custom_provider_context_length", return_value=None)
@patch("hermes_cli.config.get_compatible_custom_providers", return_value=_CUSTOM_PROVIDERS)
@patch("hermes_cli.config.load_config", return_value={})
def test_agent_init_time_custom_convention_still_resolves_correctly(
    mock_cfg, mock_cps, mock_ctx_len_cp, mock_ctx_len
):
    """The OTHER convention — agent.provider == "custom" (bare canonical,
    as set at agent-init by hermes_cli/runtime_provider.py) with the real
    identity only on agent.requested_provider — must keep working too."""
    agent = _make_agent(provider="custom", requested_provider="vllm")

    agent.switch_model(
        "unsloth/Qwen3.6-35B-A3B-NVFP4",
        "custom",
        base_url="http://192.168.15.115:8000/v1",
    )
    agent.requested_provider = "vllm-no-think"

    agent.switch_model(
        "unsloth/Qwen3.6-35B-A3B-NVFP4",
        "custom",
        base_url="http://192.168.15.115:8000/v1",
    )

    assert agent.request_overrides.get("extra_body") == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
