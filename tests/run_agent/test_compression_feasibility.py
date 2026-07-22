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
    # Persisted successfully (isolated per-test HERMES_HOME is writable),
    # so the message confirms config.yaml was updated permanently rather
    # than telling the user to edit it themselves.
    assert "config.yaml" in messages[0]
    assert "updated compression.threshold" in messages[0]
    # Warning stored for gateway replay
    assert agent._compression_warning is not None
    # Threshold on the live compressor was actually lowered to aux_context.
    assert agent.context_compressor.threshold_tokens == 80_000
    # Every threshold-derived budget must move with it. Keeping the original
    # 20K tail here would protect 25% of the lowered threshold instead of the
    # configured 20%, and larger real-world mismatches can make the tail's 1.5x
    # soft ceiling wider than the entire compression trigger.
    assert agent.context_compressor.tail_token_budget == 16_000

    # Persisted value actually landed on disk, one point under the exact
    # aux-derived ratio (40%) as a safety margin.
    from hermes_cli import config as hermes_config
    persisted_cfg = hermes_config.load_config()
    assert persisted_cfg["compression"]["threshold"] == 0.39


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
    assert len(callback_events) == 0


# ── Persistence of the auto-lowered threshold to config.yaml (#15962) ──
#
# These exercise the real config.yaml read/merge/write path (not mocks) so
# an integration bug in the reused hermes_cli.config machinery would show
# up here, per AGENTS.md's "E2E validation, not just green unit mocks."


@patch("agent.model_metadata.get_model_context_length", return_value=80_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_autolower_persists_corrected_ratio_to_config_yaml(mock_get_client, mock_ctx_len):
    """E2E: successful persistence writes the corrected ratio to the real
    (per-test-isolated) config.yaml, and a subsequent fresh load reflects
    it — proving the fix: the correction survives a session restart
    instead of being recomputed from the same stale ratio every time."""
    from hermes_cli import config as hermes_config

    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "google/gemini-3-flash-preview")

    # Seed an existing user config.yaml with an unrelated key + the stale
    # (too-high) threshold ratio, to prove the write is minimal-diff and
    # doesn't clobber sibling keys.
    config_path = hermes_config.get_config_path()
    config_path.write_text(
        "model:\n  default: some/model\ncompression:\n  threshold: 0.50\n",
        encoding="utf-8",
    )

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)
    agent._check_compression_model_feasibility()

    assert "updated compression.threshold" in messages[0]

    # Fresh load (simulating a new session reading config.yaml from
    # scratch) sees the corrected ratio, not the original 0.50.
    fresh_cfg = hermes_config.load_config()
    assert fresh_cfg["compression"]["threshold"] == 0.39
    # Sibling key preserved — minimal-diff write, not a full rewrite.
    assert fresh_cfg["model"]["default"] == "some/model"


@patch("agent.model_metadata.get_model_context_length", return_value=80_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_autolower_persist_preserves_comments_and_formatting(mock_get_client, mock_ctx_len):
    """E2E: the persisted write must survive as a genuine round-trip edit —
    user comments, key ordering, and quoting in config.yaml must all still
    be present afterward, not just sibling *values*.

    This is the regression test for the review finding on #65934: the
    original implementation went through hermes_cli.config.atomic_config_write,
    which re-serialises the whole parsed dict via plain yaml.safe_dump and
    silently drops comments even though sibling *keys* survive (which is
    all the prior test actually checked). The fix routes through
    utils.atomic_roundtrip_yaml_update (ruamel.yaml round-trip), which
    edits the loaded CommentedMap in place instead of re-dumping a plain
    dict, so comments/ordering/quoting all survive.
    """
    from hermes_cli import config as hermes_config

    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "google/gemini-3-flash-preview")

    config_path = hermes_config.get_config_path()
    config_path.write_text(
        "# top-of-file user comment — must survive\n"
        "model:\n"
        "  default: some/model  # inline comment on the model key\n"
        "compression:\n"
        "  # comment directly above the threshold key\n"
        "  threshold: 0.50\n",
        encoding="utf-8",
    )

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)
    agent._check_compression_model_feasibility()

    assert "updated compression.threshold" in messages[0]

    written = config_path.read_text(encoding="utf-8")
    assert "# top-of-file user comment — must survive" in written
    assert "# inline comment on the model key" in written
    assert "# comment directly above the threshold key" in written
    # The corrected value landed, in place of the stale one.
    assert "threshold: 0.39" in written
    assert "threshold: 0.5" not in written


@patch("agent.model_metadata.get_model_context_length", return_value=80_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_autolower_managed_scope_blocked_falls_back_to_in_memory(mock_get_client, mock_ctx_len, tmp_path, monkeypatch):
    """When compression.threshold is pinned by the managed scope, the
    write must not crash and must fall back to in-memory-only correction
    with the original 'edit config.yaml yourself' warning text."""
    from hermes_cli import managed_scope

    managed_dir = tmp_path / "managed"
    managed_dir.mkdir()
    (managed_dir / "config.yaml").write_text(
        "compression:\n  threshold: 0.50\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))
    managed_scope.invalidate_managed_cache()

    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "google/gemini-3-flash-preview")

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    # Should not raise even though the key is managed-pinned.
    agent._check_compression_model_feasibility()

    assert len(messages) == 1
    # In-memory correction still applied — session still works.
    assert agent.context_compressor.threshold_tokens == 80_000
    # Old-style "edit config.yaml yourself" guidance is preserved, not the
    # "updated permanently" success text.
    assert "updated compression.threshold" not in messages[0]
    assert "To make this permanent, edit config.yaml" in messages[0]

    managed_scope.invalidate_managed_cache()


@patch("agent.model_metadata.get_model_context_length", return_value=80_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_autolower_write_failure_degrades_gracefully(mock_get_client, mock_ctx_len):
    """A failing config.yaml write (e.g. read-only filesystem, permission
    error) must not raise and must fall back to in-memory-only correction
    with the original warning text."""
    from agent import conversation_compression as cc_module

    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "google/gemini-3-flash-preview")

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    with patch.object(
        cc_module,
        "_persist_autolowered_threshold",
        side_effect=OSError("Read-only file system"),
    ):
        # The exception is caught by the outer try/except in
        # check_compression_model_feasibility (mirrors any other
        # unexpected error in this best-effort probe) — should not raise
        # and should not block the in-memory correction that already ran.
        agent._check_compression_model_feasibility()

    # In-memory correction still landed before persistence was attempted.
    assert agent.context_compressor.threshold_tokens == 80_000


def test_persist_autolowered_threshold_readonly_write_returns_false(tmp_path, monkeypatch):
    """Unit-level check of the helper itself: a write that raises OSError
    (simulating a read-only filesystem) returns False rather than raising."""
    from hermes_cli import config as hermes_config
    from agent.conversation_compression import _persist_autolowered_threshold

    config_path = hermes_config.get_config_path()
    config_path.write_text("compression:\n  threshold: 0.50\n", encoding="utf-8")

    with patch(
        "utils.atomic_roundtrip_yaml_update",
        side_effect=OSError("Read-only file system"),
    ):
        result = _persist_autolowered_threshold(39)

    assert result is False
    # Original file untouched since the write never completed.
    assert "0.50" in config_path.read_text(encoding="utf-8")
