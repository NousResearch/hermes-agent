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
    agent.tools = []

    compressor = MagicMock(spec=ContextCompressor)
    compressor.context_length = main_context
    compressor.threshold_tokens = int(main_context * threshold_percent)
    agent.context_compressor = compressor

    return agent


# ── Core warning logic ──────────────────────────────────────────────


@patch("agent.model_metadata.get_model_context_length", return_value=80_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_auto_corrects_threshold_when_aux_context_below_threshold(mock_get_client, mock_ctx_len):
    """Auto-correction: aux >= 64K floor but < threshold → lower threshold
    to aux_context so compression still works this session."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    # threshold = 100,000 — aux has 80,000 (above 64K floor, below threshold)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "google/gemini-3-flash-preview")

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    agent._check_compression_model_feasibility()

    assert len(messages) == 1
    assert "Compression model" in messages[0]
    assert "80,000" in messages[0]        # aux context
    assert "100,000" in messages[0]       # old threshold
    assert "Auto-lowered" in messages[0]
    # Actionable persistence guidance included
    assert "config.yaml" in messages[0]
    assert "auxiliary:" in messages[0]
    assert "compression:" in messages[0]
    assert "threshold:" in messages[0]
    # Warning stored for gateway replay
    assert agent._compression_warning is not None
    # Threshold on the live compressor was actually lowered to aux_context.
    assert agent.context_compressor.threshold_tokens == 80_000


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


@patch("agent.model_metadata.get_model_context_length", return_value=200_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_no_warning_when_aux_context_sufficient(mock_get_client, mock_ctx_len):
    """No warning when aux model context >= main model threshold."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    # threshold = 100,000 — aux has 200,000 (sufficient)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "google/gemini-2.5-flash")

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    agent._check_compression_model_feasibility()

    assert len(messages) == 0
    assert agent._compression_warning is None


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
    )


@patch("agent.model_metadata.get_model_context_length", return_value=128_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_feasibility_check_ignores_invalid_context_length(mock_get_client, mock_ctx_len):
    """Non-integer context_length in config is silently ignored."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    agent._aux_compression_context_length_config = None
    mock_client = MagicMock()
    mock_client.base_url = "http://custom:8080/v1"
    mock_client.api_key = "sk-test"
    mock_get_client.return_value = (mock_client, "custom/model")

    agent._emit_status = lambda msg: None
    agent._check_compression_model_feasibility()

    mock_ctx_len.assert_called_once_with(
        "custom/model",
        base_url="http://custom:8080/v1",
        api_key="sk-test",
        config_context_length=None,
        provider="openrouter",
    )


def test_init_feasibility_check_uses_aux_context_override_from_config():
    """Real AIAgent init should cache and forward auxiliary.compression.context_length."""

    class _StubCompressor:
        def __init__(self, *args, **kwargs):
            self.context_length = 200_000
            self.threshold_tokens = 100_000
            self.threshold_percent = 0.50

        def get_tool_schemas(self):
            return []

        def init(self, config_args, *args, **kwargs):
            return None

        def on_session_start(self, config_args, *args, **kwargs):
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
        patch("hermes_cli.config.load_config", return_value=cfg),
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

    assert agent._aux_compression_context_length_config == 1_000_000
    mock_ctx_len.assert_called_once_with(
        "custom/big-model",
        base_url="http://custom-endpoint:8080/v1",
        api_key="sk-custom",
        config_context_length=1_000_000,
        provider="",
    )


def _init_agent_with_context(config, context_length):
    class _StubCompressor:
        def __init__(self, *args, **kwargs):
            self.context_length = context_length
            self.threshold_tokens = max(1, context_length // 2)
            self.threshold_percent = 0.50

        def get_tool_schemas(self):
            return []

        def init(self, config_args, *args, **kwargs):
            return None

        def on_session_start(self, config_args, *args, **kwargs):
            return None

    with (
        patch("hermes_cli.config.load_config", return_value=config),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent.ContextCompressor", new=_StubCompressor),
    ):
        return AIAgent(
            api_key="[REDACTED]",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def test_default_still_rejects_context_below_64k():
    with pytest.raises(ValueError) as exc_info:
        _init_agent_with_context({}, 32_768)

    err = str(exc_info.value)
    assert "32,768" in err
    assert "64,000" in err


def test_low_context_mode_accepts_context_at_configured_minimum_when_compression_disabled():
    agent = _init_agent_with_context(
        {
            "agent": {"low_context_mode": True, "low_context_minimum": 4096},
            "compression": {"enabled": False},
        },
        4096,
    )

    assert agent.low_context_mode is True
    assert agent.low_context_minimum == 4096
    assert agent.compression_enabled is False


def test_low_context_mode_rejects_context_below_configured_minimum():
    with pytest.raises(ValueError) as exc_info:
        _init_agent_with_context(
            {
                "agent": {"low_context_mode": True, "low_context_minimum": 4096},
                "compression": {"enabled": False},
            },
            4095,
        )

    err = str(exc_info.value)
    assert "4,095" in err
    assert "4,096" in err


def test_low_context_mode_requires_compression_disabled():
    with pytest.raises(ValueError) as exc_info:
        _init_agent_with_context(
            {
                "agent": {"low_context_mode": True, "low_context_minimum": 4096},
                "compression": {"enabled": True},
            },
            4096,
        )

    err = str(exc_info.value)
    assert "agent.low_context_mode" in err
    assert "compression.enabled=false" in err


def test_low_context_mode_rejects_invalid_minimum():
    with pytest.raises(ValueError) as exc_info:
        _init_agent_with_context(
            {
                "agent": {"low_context_mode": True, "low_context_minimum": "4K"},
                "compression": {"enabled": False},
            },
            4096,
        )

    err = str(exc_info.value)
    assert "agent.low_context_minimum" in err
    assert "4K" in err


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


def test_skips_check_when_compression_disabled():
    """No check performed when compression is disabled."""
    agent = _make_agent(compression_enabled=False)

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    agent._check_compression_model_feasibility()

    assert len(messages) == 0
    assert agent._compression_warning is None


@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_exception_does_not_crash(mock_get_client):
    """Exceptions in the check are caught — never blocks startup."""
    agent = _make_agent()
    mock_get_client.side_effect = RuntimeError("boom")

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    # Should not raise
    agent._check_compression_model_feasibility()

    # No user-facing message (error is debug-logged)
    assert len(messages) == 0


@patch("agent.model_metadata.get_model_context_length", return_value=100_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_exact_threshold_boundary_no_warning(mock_get_client, mock_ctx_len):
    """No warning when aux context exactly equals the threshold."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "test-model")

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    agent._check_compression_model_feasibility()

    assert len(messages) == 0


@patch("agent.model_metadata.get_model_context_length", return_value=99_999)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_just_below_threshold_auto_corrects(mock_get_client, mock_ctx_len):
    """Auto-correct fires when aux context is one token below the threshold
    (and above the 64K hard floor)."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "small-model")

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    agent._check_compression_model_feasibility()

    assert len(messages) == 1
    assert "small-model" in messages[0]
    assert "Auto-lowered" in messages[0]
    assert agent.context_compressor.threshold_tokens == 99_999


# ── Two-phase: __init__ + run_conversation replay ───────────────────


@patch("agent.model_metadata.get_model_context_length", return_value=80_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_warning_stored_for_gateway_replay(mock_get_client, mock_ctx_len):
    """__init__ stores the warning; _replay sends it through status_callback."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "google/gemini-3-flash-preview")

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
        ev == "lifecycle" and "Auto-lowered" in msg
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


def test_replay_without_callback_is_noop():
    """_replay_compression_warning doesn't crash when status_callback is None."""
    agent = _make_agent()
    agent._compression_warning = "some warning"
    agent.status_callback = None

    # Should not raise
    agent._replay_compression_warning()


@patch("agent.model_metadata.get_model_context_length", return_value=80_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_run_conversation_clears_warning_after_replay(mock_get_client, mock_ctx_len):
    """After replay in run_conversation, _compression_warning is cleared
    so the warning is not sent again on subsequent turns."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "small-model")

    agent._emit_status = lambda msg: None
    agent._check_compression_model_feasibility()

    assert agent._compression_warning is not None

    # Simulate what run_conversation does
    callback_events = []
    agent.status_callback = lambda ev, msg: callback_events.append((ev, msg))
    if agent._compression_warning:
        agent._replay_compression_warning()
        agent._compression_warning = None  # as in run_conversation

    assert len(callback_events) == 1

    # Second turn — nothing replayed
    callback_events.clear()
    if agent._compression_warning:
        agent._replay_compression_warning()
        agent._compression_warning = None

    assert len(callback_events) == 0
