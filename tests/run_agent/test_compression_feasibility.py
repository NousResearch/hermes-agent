"""Tests for _check_compression_model_feasibility() — warns when the
auxiliary compression model's context is smaller than the main model's
compression threshold.

Two-phase design:
  1. __init__  → runs the check, prints via _vprint (CLI), stores warning
  2. run_conversation (first call) → replays stored warning through
     status_callback (gateway platforms)
"""

from unittest.mock import MagicMock, patch

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

    compressor = MagicMock(spec=ContextCompressor)
    compressor.context_length = main_context
    compressor.threshold_tokens = int(main_context * threshold_percent)
    agent.context_compressor = compressor

    return agent


# ── Core warning logic ──────────────────────────────────────────────


@patch("agent.model_metadata.get_model_context_length", return_value=32_768)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_warns_when_aux_context_below_threshold(mock_get_client, mock_ctx_len):
    """Warning emitted when aux model context < main model threshold."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    # threshold = 100,000 — aux has only 32,768
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "google/gemini-3-flash-preview")

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    agent._check_compression_model_feasibility()

    assert len(messages) == 1
    assert "Compression model" in messages[0]
    assert "32,768" in messages[0]
    assert "100,000" in messages[0]
    assert "will not be possible" in messages[0]
    # Actionable fix guidance included
    assert "Fix options" in messages[0]
    assert "auxiliary:" in messages[0]
    assert "compression:" in messages[0]
    assert "threshold:" in messages[0]
    # Warning stored for gateway replay
    assert agent._compression_warning is not None


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
def test_just_below_threshold_warns(mock_get_client, mock_ctx_len):
    """Warning fires when aux context is one token below the threshold."""
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


# ── Two-phase: __init__ + run_conversation replay ───────────────────


@patch("agent.model_metadata.get_model_context_length", return_value=32_768)
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
        ev == "lifecycle" and "will not be possible" in msg
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


@patch("agent.model_metadata.get_model_context_length", return_value=32_768)
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
