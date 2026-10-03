"""Tests that switch_model does not inherit stale context_length overrides."""

from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.models_local import LMStudioLoadResult
from run_agent import AIAgent
from hermes_cli.route_identity import normalize_route_base_url
from agent.context_compressor import ContextCompressor


class _StubStartupCompressor:
    def __init__(self, *args, **kwargs):
        self.context_length = kwargs.get("config_context_length") or 272_000
        self.config_context_length = kwargs.get("config_context_length")
        self.threshold_tokens = int(self.context_length * 0.95)
        self.threshold_percent = 0.95

    def get_tool_schemas(self):
        return []

    def on_session_start(self, *args, **kwargs):
        return None


def test_route_url_normalization_preserves_path_slash_before_query():
    """A path slash before a query changes OpenAI SDK URL joining."""
    assert normalize_route_base_url(
        "https://example.com/v1/?tenant=large"
    ) != normalize_route_base_url("https://example.com/v1?tenant=large")














def _make_direct_start_agent(
    cfg: dict, *, model: str, provider: str, base_url: str
) -> AIAgent:
    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.agent_init.ContextCompressor", new=_StubStartupCompressor),
    ):
        return AIAgent(
            model=model,
            provider=provider,
            api_key="fake-test-token",
            base_url=base_url,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def _make_agent_with_compressor(config_context_length=None) -> AIAgent:
    """Build a minimal AIAgent with a context_compressor, skipping __init__."""
    agent = AIAgent.__new__(AIAgent)

    # Primary model settings
    agent.model = "primary-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "sk-primary"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True

    # Store the initial config_context_length override used at agent construction.
    agent._config_context_length = config_context_length

    # Context compressor with primary model values
    compressor = ContextCompressor(
        model="primary-model",
        threshold_percent=0.50,
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-primary",
        provider="openrouter",
        quiet_mode=True,
        config_context_length=config_context_length,
    )
    agent.context_compressor = compressor

    # For switch_model
    agent._primary_runtime = {}

    return agent


@patch("agent.model_metadata.get_model_context_length", return_value=131_072)
def test_switch_model_clears_previous_config_context_length(mock_ctx_len):
    """Switching models must not reuse the previous model.context_length override."""
    agent = _make_agent_with_compressor(config_context_length=32_768)

    assert agent.context_compressor.model == "primary-model"
    assert agent.context_compressor.context_length == 32_768  # From config override

    # Switch model
    agent.switch_model("new-model", "openrouter", api_key="sk-new", base_url="https://openrouter.ai/api/v1")

    # Verify the old config override is not passed to the new model.
    mock_ctx_len.assert_called_once()
    call_kwargs = mock_ctx_len.call_args.kwargs
    assert call_kwargs.get("config_context_length") is None

    # Verify compressor was updated from the newly resolved model metadata.
    assert agent.context_compressor.model == "new-model"
    assert agent.context_compressor.context_length == 131_072




def test_switch_model_reapplies_checked_auxiliary_compression_limit():
    """A model switch re-probes the auxiliary summariser eagerly: the trigger is clamped to its window
    before the first compaction on the new model, and un-clamped again when a later switch fits (#114707)."""
    agent = _make_agent_with_compressor(config_context_length=200_000)
    agent.compression_enabled = True
    agent._compression_feasibility_checked = False
    agent._aux_compression_context_length_config = None
    agent._custom_providers = []
    agent._compression_warning = None
    agent._print_fn = None
    agent.log_prefix = ""
    agent.suppress_status_output = False
    agent._stream_consumers = []
    agent._executing_tools = False
    agent._mute_post_response = False
    agent.status_callback = None
    agent.tool_progress_callback = None
    agent.tools = []
    agent._emit_status = lambda _message: None
    agent._create_openai_client = lambda *_args, **_kwargs: MagicMock()

    aux_client = MagicMock(base_url="https://aux.example/v1", api_key="aux-key")
    with (
        patch(
            "agent.model_metadata.get_model_context_length",
            side_effect=[400_000, 80_000],
        ),
        patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(aux_client, "auxiliary-model"),
        ),
    ):
        agent.switch_model(
            "larger-main-model",
            "openrouter",
            api_key="sk-new",
            base_url="https://openrouter.ai/api/v1",
        )

    assert agent.context_compressor.context_length == 400_000
    assert agent.context_compressor.threshold_tokens == 80_000
    assert agent._compression_feasibility_checked is True
    # A same-runtime window correction (overflow-reported limit) keeps the aux ceiling.
    agent.context_compressor.update_model(
        model="larger-main-model", context_length=1_000_000, base_url="https://openrouter.ai/api/v1",
        api_key="sk-new", provider="openrouter", api_mode="chat_completions",
    )
    assert agent.context_compressor.threshold_tokens == 80_000
    # Switching to a runtime whose aux fits restores the main-model trigger (no one-way ratchet).
    with (
        patch("agent.model_metadata.get_model_context_length", side_effect=[400_000, 400_000]),
        patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(aux_client, "auxiliary-model")),
    ):
        agent.switch_model(
            "other-main-model", "openrouter", api_key="sk-new", base_url="https://openrouter.ai/api/v1",
        )
    assert agent.context_compressor.threshold_tokens == 300_000


def test_switch_model_omitted_base_url_preserves_direct_openai_capability():
    """A same-provider switch resolves capabilities from the retained URL."""
    agent = _make_agent_with_compressor(config_context_length=None)
    agent.provider = "openai"
    agent.model = "gpt-5.6"
    agent.base_url = "https://api.openai.com/v1"
    agent.runtime_capabilities = {"native_compaction": True}
    agent._create_openai_client = lambda *_args, **_kwargs: MagicMock()

    with patch("agent.model_metadata.get_model_context_length", return_value=128_000):
        agent.switch_model("gpt-5.6", "openai", api_key="sk-new")

    assert agent.base_url == "https://api.openai.com/v1"
    assert agent.runtime_capabilities == {"native_compaction": True}


def test_cross_provider_switch_to_default_openai_preserves_native_capability():
    agent = _make_agent_with_compressor(config_context_length=None)
    agent.provider = "openrouter"
    agent.model = "gpt-5.5"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.runtime_capabilities = {"native_compaction": False}
    agent._create_openai_client = lambda *_args, **_kwargs: MagicMock()

    with patch("agent.model_metadata.get_model_context_length", return_value=128_000):
        agent.switch_model("gpt-5.6", "openai", api_key="sk-new")

    assert agent.base_url == "https://api.openai.com/v1"
    assert agent.runtime_capabilities == {"native_compaction": True}


def test_direct_start_model_override_does_not_inherit_profile_context_length():
    """A CLI ``--model`` startup override must not inherit another model's window."""
    cfg = {
        "model": {
            "default": "kimi-k3",
            "provider": "custom:kimi-coding-1m",
            "base_url": "https://api.kimi.com/coding",
            "context_length": 1_048_576,
        },
        "custom_providers": [
            {
                "name": "kimi-coding-1m",
                "base_url": "https://api.kimi.com/coding",
                "models": {"kimi-k3": {"context_length": 1_048_576}},
            }
        ],
    }
    agent = _make_direct_start_agent(
        cfg,
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert agent.context_compressor.config_context_length is None
    assert agent.context_compressor.context_length == 272_000


def test_direct_start_preserves_context_for_normalized_default_model_alias():
    """Equivalent vendor-prefixed defaults still own their explicit window."""
    cfg = {
        "model": {
            "default": "openai/gpt-5.6-sol",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "context_length": 272_000,
        }
    }

    agent = _make_direct_start_agent(
        cfg,
        model="gpt-5.6-sol",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert agent.context_compressor.config_context_length == 272_000
    assert agent.context_compressor.context_length == 272_000
































































def test_lmstudio_switch_uses_destination_context_and_verified_runtime(monkeypatch):
    agent = _make_agent_with_compressor(config_context_length=32_768)
    calls = []

    def fake_load_config():
        return {}

    def fake_compatible(_cfg):
        return [{"name": "lmstudio", "base_url": "http://127.0.0.1:1234/v1"}]

    def fake_provider_context(*, model, base_url, custom_providers):
        assert model == "lmstudio/new-model"
        assert base_url == "http://127.0.0.1:1234/v1"
        return 120_000

    def fake_lmstudio_load(self, config_context_length=None):
        calls.append(config_context_length)
        return LMStudioLoadResult(100_000)

    monkeypatch.setattr("hermes_cli.config.load_config", fake_load_config)

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", fake_load_config)
    monkeypatch.setattr("hermes_cli.config.get_compatible_custom_providers", fake_compatible)
    monkeypatch.setattr("hermes_cli.config.get_custom_provider_context_length", fake_provider_context)
    monkeypatch.setattr(AIAgent, "_ensure_lmstudio_runtime_loaded", fake_lmstudio_load)

    with patch("agent.model_metadata.get_model_context_length", return_value=100_000) as mock_ctx_len:
        agent.switch_model(
            "lmstudio/new-model",
            "lmstudio",
            api_key="",
            base_url="http://127.0.0.1:1234/v1",
        )

    assert calls == [120_000]
    call_kwargs = mock_ctx_len.call_args.kwargs
    assert call_kwargs.get("config_context_length") == 100_000
    assert agent._config_context_length == 120_000
    assert agent.context_compressor.context_length == 100_000


def test_later_lmstudio_failure_restores_runtime_capabilities(monkeypatch):
    agent = _make_agent_with_compressor(config_context_length=32_768)
    agent.runtime_capabilities = {"native_compaction": True}
    original_client = agent.client

    monkeypatch.setattr(
        AIAgent,
        "_ensure_lmstudio_runtime_loaded",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated LM Studio failure")
        ),
    )

    with pytest.raises(RuntimeError, match="simulated LM Studio failure"):
        agent.switch_model(
            "new-model",
            "openrouter",
            api_key="sk-new",
            base_url="https://openrouter.ai/api/v1",
        )

    assert agent.model == "primary-model"
    assert agent.provider == "openrouter"
    assert agent.client is original_client
    assert agent.runtime_capabilities == {"native_compaction": True}


# ═══════════════════════════════════════════════════════════════════════
# Level 2: switch_model re-resolves _ollama_num_ctx (#110239)
# ═══════════════════════════════════════════════════════════════════════

def _make_ollama_agent(
    model: str = "modelA",
    base_url: str = "http://127.0.0.1:11434/v1",
    ollama_num_ctx: int | None = 40000,
    config_context_length: int | None = 40000,
) -> AIAgent:
    """Build a minimal AIAgent with Ollama-like settings, skipping __init__."""
    agent = AIAgent.__new__(AIAgent)
    agent.model = model
    agent.provider = "custom"
    agent.base_url = base_url
    agent.api_key = "ollama"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True
    agent._ollama_num_ctx = ollama_num_ctx
    agent._config_context_length = config_context_length
    agent._primary_runtime = {}
    agent.runtime_capabilities = {}

    compressor = ContextCompressor(
        model=model,
        threshold_percent=0.50,
        base_url=base_url,
        api_key="ollama",
        provider="custom",
        quiet_mode=True,
        config_context_length=config_context_length,
    )
    agent.context_compressor = compressor
    return agent


@patch("agent.model_metadata.get_model_context_length", return_value=131072)
def test_switch_ollama_model_recomputes_num_ctx_from_config(mock_ctx_len):
    """Switching between two Ollama models must re-resolve _ollama_num_ctx."""
    agent = _make_ollama_agent(model="modelA", ollama_num_ctx=40000)
    assert agent._ollama_num_ctx == 40000

    # Simulate config.yaml with per-model context_length overrides
    new_cfg = {
        "model": {},
        "providers": {
            "ollama-local": {
                "base_url": "http://127.0.0.1:11434/v1",
                "models": {
                    "modelB": {"context_length": 131072},
                },
            },
        },
    }

    def fake_load_config():
        return new_cfg

    with (
        patch("hermes_cli.config.load_config", side_effect=fake_load_config),
        patch("hermes_cli.config.load_config_readonly", side_effect=fake_load_config),
        patch("hermes_cli.config.get_compatible_custom_providers", return_value=[]),
        patch("hermes_cli.config.get_custom_provider_context_length", return_value=131072),
        patch("agent.agent_init.query_ollama_num_ctx", return_value=131072),
        patch("agent.agent_init.is_local_endpoint", return_value=True),
    ):
        agent.switch_model(
            "modelB", "custom", api_key="ollama",
            base_url="http://127.0.0.1:11434/v1",
        )

    # _ollama_num_ctx should now reflect modelB's configured value
    assert agent._ollama_num_ctx == 131072


@patch("agent.model_metadata.get_model_context_length", return_value=128000)
def test_switch_to_non_ollama_clears_num_ctx(mock_ctx_len):
    """Switching from Ollama to a non-local provider must clear _ollama_num_ctx."""
    agent = _make_ollama_agent(model="modelA", ollama_num_ctx=40000)
    assert agent._ollama_num_ctx == 40000

    with (
        patch("hermes_cli.config.load_config", return_value={"model": {}}),
        patch("hermes_cli.config.load_config_readonly", return_value={"model": {}}),
        patch("hermes_cli.config.get_compatible_custom_providers", return_value=[]),
        patch("hermes_cli.config.get_custom_provider_context_length", return_value=None),
        patch("agent.agent_init.is_local_endpoint", return_value=False),
    ):
        agent.switch_model(
            "gpt-5.6", "openai", api_key="sk-new",
            base_url="https://api.openai.com/v1",
        )

    # Non-Ollama endpoint: _ollama_num_ctx must be cleared
    assert agent._ollama_num_ctx is None


@patch("agent.model_metadata.get_model_context_length", return_value=65536)
def test_switch_ollama_explicit_num_ctx_override_respected(mock_ctx_len):
    """An explicit model.ollama_num_ctx override must be used after switch."""
    agent = _make_ollama_agent(model="modelA", ollama_num_ctx=40000)
    assert agent._ollama_num_ctx == 40000

    # Config with an explicit ollama_num_ctx override on model section
    new_cfg = {
        "model": {"ollama_num_ctx": 32768},
        "providers": {},
    }

    def fake_load_config():
        return new_cfg

    with (
        patch("hermes_cli.config.load_config", side_effect=fake_load_config),
        patch("hermes_cli.config.load_config_readonly", side_effect=fake_load_config),
        patch("hermes_cli.config.get_compatible_custom_providers", return_value=[]),
        patch("hermes_cli.config.get_custom_provider_context_length", return_value=65536),
        patch("agent.agent_init.is_local_endpoint", return_value=True),
    ):
        agent.switch_model(
            "modelB", "custom", api_key="ollama",
            base_url="http://127.0.0.1:11434/v1",
        )

    # Explicit override in model.ollama_num_ctx wins over auto-detection
    assert agent._ollama_num_ctx == 32768


@patch("agent.model_metadata.get_model_context_length", return_value=131072)
def test_switch_ollama_num_ctx_capped_to_config_context_length(mock_ctx_len):
    """Auto-detected num_ctx must be capped to the config context_length."""
    agent = _make_ollama_agent(model="modelA", ollama_num_ctx=40000)

    # Ollama reports 131072 but config caps at 65536
    new_cfg = {"model": {}, "providers": {}}

    def fake_load_config():
        return new_cfg

    with (
        patch("hermes_cli.config.load_config", side_effect=fake_load_config),
        patch("hermes_cli.config.load_config_readonly", side_effect=fake_load_config),
        patch("hermes_cli.config.get_compatible_custom_providers", return_value=[]),
        patch("hermes_cli.config.get_custom_provider_context_length", return_value=65536),
        patch("agent.agent_init.query_ollama_num_ctx", return_value=131072),
        patch("agent.agent_init.is_local_endpoint", return_value=True),
    ):
        agent.switch_model(
            "modelB", "custom", api_key="ollama",
            base_url="http://127.0.0.1:11434/v1",
        )

    # num_ctx capped to the config context_length (no explicit ollama_num_ctx override)
    assert agent._ollama_num_ctx == 65536
