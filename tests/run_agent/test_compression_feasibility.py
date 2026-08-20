"""Tests for _check_compression_model_feasibility() — warns when the
auxiliary compression model's context is smaller than the main model's
compression threshold.

Two-phase design:
  1. __init__  → runs the check, prints via _vprint (CLI), stores warning
  2. run_conversation (first call) → replays stored warning through
     status_callback (gateway platforms)
"""

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.context_compressor import ContextCompressor


@pytest.fixture(autouse=True)
def _stable_aux_provider_config():
    """Keep feasibility tests independent from the developer's config.yaml."""
    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("auto", None, None, None, None),
    ):
        yield


def _make_agent(
    *,
    compression_enabled: bool = True,
    threshold_percent: float = 0.50,
    main_context: int = 200_000,
) -> AIAgent:
    """Build a minimal AIAgent with a compressor, skipping __init__."""
    agent = AIAgent.__new__(AIAgent)
    agent.model = "test-main-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "sk-test"
    agent.api_mode = "chat_completions"
    agent.quiet_mode = True
    agent.log_prefix = ""
    agent.compression_enabled = compression_enabled
    agent._print_fn = None
    agent.suppress_status_output = False
    agent._stream_consumers = []
    agent._executing_tools = False
    agent._mute_post_response = False
    agent.status_callback = None
    agent.tool_progress_callback = None
    agent._compression_warning = None
    agent._aux_compression_context_length_config = None
    agent._custom_providers = []
    agent.tools = []

    compressor = MagicMock(spec=ContextCompressor)
    compressor.context_length = main_context
    compressor.threshold_tokens = int(main_context * threshold_percent)
    compressor.summary_target_ratio = 0.20
    compressor.tail_token_budget = int(
        compressor.threshold_tokens * compressor.summary_target_ratio
    )
    agent.context_compressor = compressor

    return agent


# ── Core warning logic ──────────────────────────────────────────────


@patch("agent.model_metadata.get_model_context_length", return_value=80_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_small_aux_window_keeps_threshold_and_stashes_it(mock_get_client, mock_ctx_len):
    """aux >= 64K floor but < threshold → threshold UNCHANGED; the aux window
    is stashed for _generate_summary's per-call input fit instead. Lowering
    the session threshold (the old behaviour) silently shrank the main
    model's usable window, while the summariser request — bounded by
    per-message truncation and _SUMMARY_INPUT_MAX_CHARS — never approached
    the threshold it was compared against."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    # threshold = 100,000 — aux has 80,000 (above 64K floor, below threshold)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "google/gemini-3-flash-preview")

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    agent._check_compression_model_feasibility()

    # No user-facing warning: the per-call bound makes the model just work.
    assert messages == []
    assert agent._compression_warning is None
    # Threshold and every threshold-derived budget stay untouched.
    assert agent.context_compressor.threshold_tokens == 100_000
    assert agent.context_compressor.tail_token_budget == 20_000
    # The aux window is stashed for the per-call fit.
    assert agent.context_compressor.summary_model_context_length == 80_000


@patch("agent.model_metadata.get_model_context_length", return_value=32_768)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_rejects_aux_below_minimum_context(mock_get_client, mock_ctx_len):
    """Hard floor: aux context < MINIMUM_CONTEXT_LENGTH (64K) → session
    refuses to start (ValueError), mirroring the main-model rejection."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "tiny-aux-model")

    agent._emit_status = lambda msg: None

    with pytest.raises(ValueError) as exc_info:
        agent._check_compression_model_feasibility()

    err = str(exc_info.value)
    assert "tiny-aux-model" in err
    assert "32,768" in err
    assert "64,000" in err
    assert "below the minimum" in err




def test_feasibility_check_passes_live_main_runtime():
    """Compression feasibility should probe using the live session runtime."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    agent.model = "gpt-5.4"
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_key = "codex-token"
    agent.api_mode = "codex_responses"

    mock_client = MagicMock()
    mock_client.base_url = "https://chatgpt.com/backend-api/codex"
    mock_client.api_key = "codex-token"

    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(mock_client, "gpt-5.4")) as mock_get_client, \
         patch("agent.model_metadata.get_model_context_length", return_value=200_000):
        agent._emit_status = lambda msg: None
        agent._check_compression_model_feasibility()

    mock_get_client.assert_called_once_with(
        "compression",
        main_runtime={
            "model": "gpt-5.4",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-token",
            "api_mode": "codex_responses",
            "auth_mode": "",
        },
    )


@patch("agent.model_metadata.get_model_context_length", return_value=1_000_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_feasibility_check_passes_config_context_length(mock_get_client, mock_ctx_len):
    """auxiliary.compression.context_length from config is forwarded to
    get_model_context_length so custom endpoints that lack /models still
    report the correct context window (fixes #8499)."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.85)
    agent._aux_compression_context_length_config = 1_000_000
    mock_client = MagicMock()
    mock_client.base_url = "http://custom-endpoint:8080/v1"
    mock_client.api_key = "sk-custom"
    mock_get_client.return_value = (mock_client, "custom/big-model")

    agent._emit_status = lambda msg: None
    agent._check_compression_model_feasibility()

    mock_ctx_len.assert_called_once_with(
        "custom/big-model",
        base_url="http://custom-endpoint:8080/v1",
        api_key="sk-custom",
        config_context_length=1_000_000,
        provider="openrouter",
        custom_providers=[],
    )




def test_init_feasibility_check_uses_aux_context_override_from_config():
    """Lazy feasibility check should cache and forward auxiliary.compression.context_length.

    NB: feasibility check is deferred from AIAgent.__init__ to the first
    actual compression attempt (saves ~400ms cold startup on short sessions
    that never trigger compression). The test drives the check explicitly
    via ``agent._check_compression_model_feasibility()`` to assert the
    config-override threading.
    """

    class _StubCompressor:
        def __init__(self, *args, **kwargs):
            self.context_length = 200_000
            self.threshold_tokens = 100_000
            self.threshold_percent = 0.50

        def get_tool_schemas(self):
            return []

        def on_session_start(self, *args, **kwargs):
            return None

    cfg = {
        "auxiliary": {
            "compression": {
                "context_length": 1_000_000,
            },
        },
    }
    mock_client = MagicMock()
    mock_client.base_url = "http://custom-endpoint:8080/v1"
    mock_client.api_key = "sk-custom"

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent.ContextCompressor", new=_StubCompressor),
        patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(mock_client, "custom/big-model")),
        patch("agent.model_metadata.get_model_context_length", return_value=1_000_000) as mock_ctx_len,
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

        # Config override is captured eagerly in __init__ (still needed
        # because the threshold-derivation logic at construction time
        # consults it).
        assert agent._aux_compression_context_length_config == 1_000_000

        # The expensive feasibility probe is deferred. Drive it manually
        # to validate the call shape still forwards the override correctly.
        agent._check_compression_model_feasibility()

    mock_ctx_len.assert_called_once_with(
        "custom/big-model",
        base_url="http://custom-endpoint:8080/v1",
        api_key="sk-custom",
        config_context_length=1_000_000,
        provider="",
        custom_providers=[],
    )


@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_warns_when_no_auxiliary_provider(mock_get_client):
    """Warning emitted when no auxiliary provider is configured."""
    agent = _make_agent()
    mock_get_client.return_value = (None, None)

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    agent._check_compression_model_feasibility()

    assert len(messages) == 1
    assert "No auxiliary LLM provider" in messages[0]
    assert agent._compression_warning is not None


def test_no_unavailable_warning_when_configured_fallback_chain_resolves():
    """Primary compression provider can be down if configured fallback works."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    fallback_client = MagicMock()
    fallback_client.base_url = "https://chatgpt.com/backend-api/codex"
    fallback_client.api_key = "codex-oauth-token"

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("ollama-cloud", "deepseek-v4-flash:cloud", None, None, None),
    ), patch(
        "agent.auxiliary_client.get_text_auxiliary_client",
        return_value=(None, None),
    ), patch(
        "agent.auxiliary_client._try_configured_fallback_for_unavailable_client",
        return_value=(fallback_client, "gpt-5.4-mini", "fallback_chain[0](openai-codex)"),
    ) as mock_fallback, patch(
        "agent.model_metadata.get_model_context_length",
        return_value=200_000,
    ) as mock_ctx_len:
        agent._check_compression_model_feasibility()

    assert messages == []
    assert agent._compression_warning is None
    mock_fallback.assert_called_once_with("compression", "ollama-cloud")
    mock_ctx_len.assert_called_once()
    assert mock_ctx_len.call_args.args == ("gpt-5.4-mini",)
    assert mock_ctx_len.call_args.kwargs["provider"] == "openai-codex"










# ── Two-phase: __init__ + run_conversation replay ───────────────────


@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_warning_stored_for_gateway_replay(mock_get_client):
    """__init__ stores the warning; _replay sends it through status_callback.

    The small-aux-window case no longer warns (the per-call input fit makes
    it work silently), so the replay path is exercised with the remaining
    startup warning: no auxiliary provider available at all."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    mock_get_client.return_value = (None, None)

    # Phase 1: __init__ — _emit_status prints (CLI) but callback is None
    vprint_messages = []
    agent._emit_status = lambda msg: vprint_messages.append(msg)
    agent._check_compression_model_feasibility()

    assert len(vprint_messages) == 1  # CLI got it
    assert agent._compression_warning is not None  # stored for replay

    # Phase 2: gateway wires callback post-init, then run_conversation replays
    callback_events = []
    agent.status_callback = lambda ev, msg: callback_events.append((ev, msg))
    agent._replay_compression_warning()

    assert any(
        ev == "lifecycle" and "No auxiliary LLM provider" in msg
        for ev, msg in callback_events
    )


@patch("agent.model_metadata.get_model_context_length", return_value=200_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_no_replay_when_no_warning(mock_get_client, mock_ctx_len):
    """_replay_compression_warning is a no-op when there's no stored warning."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "big-model")

    agent._emit_status = lambda msg: None
    agent._check_compression_model_feasibility()

    assert agent._compression_warning is None

    callback_events = []
    agent.status_callback = lambda ev, msg: callback_events.append((ev, msg))
    agent._replay_compression_warning()

    assert len(callback_events) == 0






# ── Large-context main model: same no-lower semantics ───────────────────────


@patch("agent.model_metadata.get_model_context_length", return_value=300_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_large_context_main_also_keeps_threshold(mock_get_client, mock_ctx_len):
    """A 1M-window main model with a 300K aux stays at its 500K threshold —
    the stash-and-fit semantics do not depend on the small-context floor."""
    agent = _make_agent(main_context=1_000_000, threshold_percent=0.50)
    # threshold = 500,000 — aux has 300,000
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "google/gemini-3-flash-preview")

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    agent._check_compression_model_feasibility()

    assert messages == []
    assert agent.context_compressor.threshold_tokens == 500_000
    assert agent.context_compressor.summary_model_context_length == 300_000




