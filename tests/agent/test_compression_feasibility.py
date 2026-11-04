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


@pytest.mark.parametrize("main_context,aux_context", [(1_000_000, 512_000), (400_000, 80_000)])
def test_aux_sync_keeps_lean_tail_policy(main_context, aux_context):
    """Lowering only the trigger must not change window-relative retention."""
    agent = _make_agent(main_context=main_context)
    compressor = agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=main_context,
        threshold_percent=0.85, quiet_mode=True,
    )
    before = compressor.tail_token_budget
    agent._emit_status = lambda message: None
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")), \
         patch("agent.model_metadata.get_model_context_length", return_value=aux_context):
        agent._check_compression_model_feasibility()
        assert compressor.threshold_tokens == aux_context
        assert compressor.tail_token_budget == before
        # Repeated feasibility and subsequent model recalibration retain policy.
        agent._check_compression_model_feasibility()
        assert compressor.tail_token_budget == before
        compressor.update_model("test-main-model", context_length=main_context)
        assert compressor.tail_token_budget == before


def test_aux_sync_legacy_tail_follows_lowered_threshold():
    """Explicit legacy retention follows the current trigger, not its old cache."""
    agent = _make_agent(main_context=1_000_000)
    compressor = agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=1_000_000,
        threshold_percent=0.85, tail_mode="legacy", quiet_mode=True,
    )
    before = compressor.tail_token_budget
    agent._emit_status = lambda message: None
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")), \
         patch("agent.model_metadata.get_model_context_length", return_value=512_000):
        agent._check_compression_model_feasibility()
    assert compressor.threshold_tokens == 512_000
    assert compressor.tail_token_budget < before
    assert compressor.tail_token_budget == int(compressor.threshold_tokens * compressor.summary_target_ratio)


def test_fallback_activation_on_never_probed_session_stays_lazy():
    """A session that never ran the feasibility probe does not resolve an auxiliary client while a
    fallback is being activated; the compaction-time probe still owns the first verdict (#114707)."""
    from agent.chat_completion_helpers import _update_fallback_context_compressor

    agent = _make_agent(main_context=200_000)
    agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=200_000, threshold_percent=0.50, quiet_mode=True,
    )
    agent._config_context_length = None
    agent.model = "fallback-model"
    with patch("agent.auxiliary_client.get_text_auxiliary_client") as aux_client, \
         patch("agent.model_metadata.get_model_context_length", return_value=1_000_000):
        _update_fallback_context_compressor(agent)
    aux_client.assert_not_called()
    assert getattr(agent, "_compression_feasibility_checked", False) is False


def test_fallback_activation_reprobes_aux_ceiling_and_keeps_it_durable():
    """Every main-runtime change re-probes the summariser and the clamp survives later window
    corrections; a failed probe leaves the latch unset for the lazy compaction-time probe (#114707)."""
    from agent.chat_completion_helpers import _update_fallback_context_compressor

    agent = _make_agent(main_context=200_000)
    compressor = agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=200_000, threshold_percent=0.50, quiet_mode=True,
    )
    notices = 0

    def _count(_message):
        nonlocal notices
        notices += 1

    agent._emit_status = _count
    agent._config_context_length = None
    agent._compression_feasibility_checked = True  # probed on the primary; aux fit there
    agent.model = "fallback-model"
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")), \
         patch("agent.model_metadata.get_model_context_length", side_effect=[1_000_000, 80_000]):
        _update_fallback_context_compressor(agent)
    assert compressor.context_length == 1_000_000
    assert compressor.threshold_tokens == 80_000
    assert agent._compression_feasibility_checked is True
    # Same-runtime window correction (provider-reported limit) keeps the ceiling.
    compressor.update_model(
        "fallback-model", context_length=800_000, base_url=agent.base_url, api_key=agent.api_key,
        provider=agent.provider, api_mode=agent.api_mode,
    )
    assert compressor.threshold_tokens == 80_000
    # An unchanged verdict is not re-announced on the next runtime change (fallback/restore cycles).
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")), \
         patch("agent.model_metadata.get_model_context_length", side_effect=[1_000_000, 80_000]):
        agent.base_url = "https://other-route.example/v1"  # runtime change, identical verdict text
        _update_fallback_context_compressor(agent)
    assert compressor.threshold_tokens == 80_000
    assert notices == 1


def test_unclamp_clears_stale_clamp_warning():
    """When a re-probe finds the summariser fits again, the stale 'auto-lowered' text must not survive
    for ``replay_compression_warning`` to resend on a session that is no longer clamped (#114707)."""
    from agent.chat_completion_helpers import _update_fallback_context_compressor

    agent = _make_agent(main_context=200_000)
    compressor = agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=200_000, threshold_percent=0.50, quiet_mode=True,
    )
    agent._emit_status = lambda message: None
    agent._config_context_length = None
    agent._compression_feasibility_checked = True
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    for main_ctx, aux_ctx, label in ((1_000_000, 80_000, "big"), (100_000, 80_000, "small")):
        agent.model = label
        with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")), \
             patch("agent.model_metadata.get_model_context_length", side_effect=[main_ctx, aux_ctx]):
            _update_fallback_context_compressor(agent)
    assert compressor._aux_context_ceiling is None
    assert compressor.threshold_tokens == 75_000
    assert agent._compression_warning is None
    assert agent._last_feasibility_notice is None


def test_near_threshold_probe_clamps_before_first_compaction():
    """A fresh instance probes once its request first reaches the smallest window any summariser may
    have, so the aux clamp lands before the first compaction fires on the main-window threshold (#114707);
    requests below that stay probe-free (#28957)."""
    from agent.conversation_compression_feasibility import ensure_compression_feasibility_checked
    from agent.model_metadata import MINIMUM_CONTEXT_LENGTH

    agent = _make_agent(main_context=1_000_000)
    compressor = agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=1_000_000, threshold_percent=0.75, quiet_mode=True,
    )
    agent._emit_status = lambda message: None
    agent._compression_feasibility_checked = False
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")) as aux_client, \
         patch("agent.model_metadata.get_model_context_length", return_value=80_000):
        ensure_compression_feasibility_checked(agent, MINIMUM_CONTEXT_LENGTH - 1)
        aux_client.assert_not_called()
        assert agent._compression_feasibility_checked is False
        ensure_compression_feasibility_checked(agent, MINIMUM_CONTEXT_LENGTH)
        ensure_compression_feasibility_checked(agent, 200_000)
    assert aux_client.call_count == 1
    assert agent._compression_feasibility_checked is True
    assert compressor.threshold_tokens == 80_000
    assert compressor.should_compress(200_000) is True


def test_a_moved_compression_route_reprobes_a_warm_agent_and_drops_the_old_ceiling():
    """A gateway session's side tasks follow the profile, so a warm agent can meet a new compression
    route mid-session. The ceiling installed for the old summariser's window is dropped and the new
    route is probed, as a main-model switch does; an unchanged route is not probed again."""
    from agent.auxiliary_task_config import session_auxiliary_scope
    from agent.conversation_compression_feasibility import ensure_compression_feasibility_checked

    agent = _make_agent(main_context=1_000_000)
    compressor = agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=1_000_000, threshold_percent=0.75, quiet_mode=True,
    )
    agent._emit_status = lambda message: None
    agent._compression_feasibility_checked = False
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    old = {"compression": {"provider": "custom", "model": "retired", "base_url": "http://localhost/v1"}}
    new = {"compression": {"provider": "custom", "model": "replacement", "base_url": "http://localhost/v1"}}
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")) as aux_client, \
         patch("agent.model_metadata.get_model_context_length", side_effect=[80_000, 900_000]):
        with session_auxiliary_scope(old):
            ensure_compression_feasibility_checked(agent, 200_000)
            ensure_compression_feasibility_checked(agent, 200_000)
        assert aux_client.call_count == 1
        assert compressor.threshold_tokens == 80_000
        with session_auxiliary_scope(new):
            ensure_compression_feasibility_checked(agent, 200_000)
    assert aux_client.call_count == 2
    assert compressor._aux_context_ceiling is None
    assert compressor.threshold_tokens == 750_000


@pytest.mark.parametrize("first_probe", ["main_model_switch", "first_compaction"])
def test_a_route_change_reprobes_after_any_path_that_latched_the_probe(first_probe):
    """A model switch (``revalidate``) and the first compaction (lazy probe) latch the probe too: they record
    the route they measured, so a later compression-route change still re-probes."""
    from agent.auxiliary_task_config import session_auxiliary_scope
    from agent.conversation_compression_feasibility import (
        ensure_compression_feasibility_checked, lazy_compression_feasibility_check, revalidate_compression_feasibility,
    )

    agent = _make_agent(main_context=1_000_000)
    compressor = agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=1_000_000, threshold_percent=0.75, quiet_mode=True,
    )
    agent._emit_status = lambda message: None
    agent._compression_feasibility_checked = False
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    old = {"compression": {"provider": "custom", "model": "small", "base_url": "http://localhost/v1"}}
    new = {"compression": {"provider": "custom", "model": "large", "base_url": "http://localhost/v1"}}
    first = revalidate_compression_feasibility if first_probe == "main_model_switch" else lazy_compression_feasibility_check
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")) as aux_client, \
         patch("agent.model_metadata.get_model_context_length", side_effect=[80_000, 900_000]):
        with session_auxiliary_scope(old):
            first(agent)
        assert compressor.threshold_tokens == 80_000
        with session_auxiliary_scope(new):
            ensure_compression_feasibility_checked(agent, 200_000)
    assert aux_client.call_count == 2
    assert compressor._aux_context_ceiling is None
    assert compressor.threshold_tokens == 750_000


def test_a_moved_providers_entry_reprobes_a_warm_agent():
    """The compression task names a ``providers:`` entry; moving that entry to another server moves the
    summariser even though the task block itself is unchanged."""
    from agent.auxiliary_task_config import session_auxiliary_scope
    from agent.conversation_compression_feasibility import ensure_compression_feasibility_checked

    agent = _make_agent(main_context=1_000_000)
    agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=1_000_000, threshold_percent=0.75, quiet_mode=True,
    )
    agent._emit_status = lambda message: None
    agent._compression_feasibility_checked = False
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    task = {"compression": {"provider": "work-gateway", "model": "small"}}
    entries = [{"base_url": "http://old.invalid/v1"}, {"base_url": "http://new.invalid/v1"}]
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")) as aux_client, \
         patch("agent.model_metadata.get_model_context_length", side_effect=[80_000, 900_000]), \
         patch("hermes_cli.runtime_provider_custom._get_named_custom_provider", side_effect=lambda *a, **k: entries[0]):
        with session_auxiliary_scope(task):
            ensure_compression_feasibility_checked(agent, 200_000)
            entries.pop(0)
            ensure_compression_feasibility_checked(agent, 200_000)
    assert aux_client.call_count == 2


def test_a_replacement_summariser_is_measured_with_its_own_context_length_pin():
    """``auxiliary.compression.context_length`` belongs to the configured summariser: a replacement with
    its own pin is measured with that pin, not the one captured when the agent was built."""
    from agent.auxiliary_task_config import session_auxiliary_scope
    from agent.conversation_compression_feasibility import ensure_compression_feasibility_checked

    agent = _make_agent(main_context=1_000_000)
    compressor = agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=1_000_000, threshold_percent=0.75, quiet_mode=True,
    )
    agent._emit_status = lambda message: None
    agent._compression_feasibility_checked = False
    agent._aux_compression_context_length_config = 80_000  # captured at agent start, for the old summariser
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    old = {"compression": {"provider": "custom", "model": "small", "base_url": "http://localhost/v1",
                           "context_length": 80_000}}
    new = {"compression": {"provider": "custom", "model": "large", "base_url": "http://localhost/v1",
                           "context_length": 900_000}}
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")):
        with session_auxiliary_scope(old):
            ensure_compression_feasibility_checked(agent, 200_000)
        assert compressor.threshold_tokens == 80_000
        with session_auxiliary_scope(new):
            ensure_compression_feasibility_checked(agent, 200_000)
    assert compressor._aux_context_ceiling is None
    assert compressor.threshold_tokens == 750_000


def test_an_unreadable_compression_route_never_stops_the_main_request():
    """A side-task entry the route check cannot read (``base_url: 123``) must not raise out of the main
    request's preflight; the probe is simply re-run."""
    from agent.conversation_compression_feasibility import ensure_compression_feasibility_checked

    agent = _make_agent(main_context=1_000_000)
    agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=1_000_000, threshold_percent=0.75, quiet_mode=True,
    )
    agent._compression_feasibility_checked = True
    agent._compression_feasibility_route = ("custom", "small", "http://localhost/v1", "")
    with patch("agent.auxiliary_task_config._get_auxiliary_task_config", side_effect=AttributeError("strip")):
        ensure_compression_feasibility_checked(agent, 1_000)
    assert agent._compression_feasibility_checked is False


def test_the_probe_reads_provider_entries_as_configured_now(monkeypatch, tmp_path):
    """A provider entry's per-model ``context_length`` is read from config.yaml when the compression model
    is measured, not from the entry list captured at agent start: an edit that shrinks the summariser's
    pin is what the probe uses."""
    from agent.conversation_compression import check_compression_model_feasibility
    import json
    home = tmp_path / "home"
    home.mkdir()
    entry = {"name": "work-gateway", "base_url": "http://gateway.invalid/v1",
             "models": {"small": {"context_length": 80_000}}}
    (home / "config.yaml").write_text(json.dumps({"custom_providers": [entry]}))
    monkeypatch.setenv("HERMES_HOME", str(home))

    agent = _make_agent(main_context=1_000_000)
    compressor = agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=1_000_000, threshold_percent=0.75, quiet_mode=True,
    )
    agent._emit_status = lambda message: None
    agent._custom_providers = [{**entry, "models": {"small": {"context_length": 900_000}}}]  # agent start
    client = MagicMock(base_url="http://gateway.invalid/v1", api_key="test-key")
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "small")):
        check_compression_model_feasibility(agent)
    assert compressor.threshold_tokens == 80_000


def test_a_bypass_session_measures_with_the_entries_from_agent_start(monkeypatch, tmp_path):
    """Inside a bypass session (its side tasks carry their own saved section) the probe never reads the
    profile's provider entries; it keeps the list captured at agent start."""
    from agent.auxiliary_task_config import session_auxiliary_scope
    from agent.conversation_compression import check_compression_model_feasibility
    import json
    home = tmp_path / "home"
    home.mkdir()
    entry = {"name": "work-gateway", "base_url": "http://gateway.invalid/v1",
             "models": {"small": {"context_length": 80_000}}}
    (home / "config.yaml").write_text(json.dumps({"custom_providers": [entry]}))
    monkeypatch.setenv("HERMES_HOME", str(home))

    agent = _make_agent(main_context=1_000_000)
    compressor = agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=1_000_000, threshold_percent=0.75, quiet_mode=True,
    )
    agent._emit_status = lambda message: None
    agent._custom_providers = [{**entry, "models": {"small": {"context_length": 900_000}}}]  # agent start
    client = MagicMock(base_url="http://gateway.invalid/v1", api_key="test-key")
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "small")), \
         session_auxiliary_scope({"compression": {}}):
        check_compression_model_feasibility(agent)
    assert compressor.threshold_tokens == 750_000


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
    # 200K main is under the 512K small-context limit and 80K/200K = 40% sits
    # below the 75% floor — a `threshold:` suggestion would be raised back to
    # 75% and ignored (#67422), so the message must not offer one and must
    # explain the recomputed trigger instead (0.75 * 200K = 150K).
    assert "threshold:" not in messages[0]
    assert "150,000" in messages[0]
    # Warning stored for gateway replay
    assert agent._compression_warning is not None
    # Threshold on the live compressor was actually lowered to aux_context.
    assert agent.context_compressor.threshold_tokens == 80_000
    # Every threshold-derived budget must move with it. Keeping the original
    # 20K tail here would protect 25% of the lowered threshold instead of the
    # configured 20%, and larger real-world mismatches can make the tail's 1.5x
    # soft ceiling wider than the entire compression trigger.
    assert agent.context_compressor.tail_token_budget == 16_000


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

    assert "tiny-aux-model" in str(exc_info.value)




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
            "session_id": "",
        },
    )


@patch("agent.model_metadata.get_model_context_length", return_value=1_000_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_feasibility_check_passes_config_context_length(mock_get_client, mock_ctx_len):
    """auxiliary.compression.context_length from config is forwarded to
    get_model_context_length so custom endpoints that lack /models still
    report the correct context window (fixes #8499)."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.85)
    mock_client = MagicMock()
    mock_client.base_url = "http://custom-endpoint:8080/v1"
    mock_client.api_key = "sk-custom"
    mock_get_client.return_value = (mock_client, "custom/big-model")

    agent._emit_status = lambda msg: None
    from agent.auxiliary_task_config import session_auxiliary_scope
    with session_auxiliary_scope({"compression": {"context_length": 1_000_000}}):
        agent._check_compression_model_feasibility()

    mock_ctx_len.assert_called_once_with(
        "custom/big-model",
        base_url="http://custom-endpoint:8080/v1",
        api_key="sk-custom",
        config_context_length=1_000_000,
        provider="openrouter",
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
        ev == "lifecycle" and str(msg) == str(agent._compression_warning)
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






# ── #67422: threshold suggestion must survive the small-context floor ────────




@patch("agent.model_metadata.get_model_context_length", return_value=300_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_threshold_suggestion_kept_for_large_context_main(mock_get_client, mock_ctx_len):
    """Main window >= 512K has no floor — any suggestion is honored, so the
    `threshold:` option stays even below 75%."""
    agent = _make_agent(main_context=1_000_000, threshold_percent=0.50)
    # threshold = 500,000 — aux has 300,000
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "google/gemini-3-flash-preview")

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    agent._check_compression_model_feasibility()

    assert len(messages) == 1
    assert "threshold: 0.30" in messages[0]






def _warm_agent_on(home, entries):
    """A warm agent whose compression task names the ``work-gateway`` custom provider in *home*."""
    import json
    (home / "config.yaml").write_text(json.dumps({
        "auxiliary": {"compression": {"provider": "work-gateway", "model": "small"}},
        "custom_providers": entries}))
    agent = _make_agent(main_context=1_000_000)
    agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=1_000_000, threshold_percent=0.75, quiet_mode=True,
    )
    agent._emit_status = lambda message: None
    agent._compression_feasibility_checked = False
    return agent


def test_a_changed_provider_context_pin_reprobes_a_warm_agent(monkeypatch, tmp_path):
    """Only the provider entry's per-model ``context_length`` changes (900K to 80K): the next request
    re-probes and the trigger lands on the new pin, through the same path a turn takes."""
    import json
    from agent.conversation_compression_feasibility import ensure_compression_feasibility_checked
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    entry = {"name": "work-gateway", "base_url": "http://gateway.invalid/v1",
             "models": {"small": {"context_length": 900_000}}}
    agent = _warm_agent_on(home, [entry])
    client = MagicMock(base_url="http://gateway.invalid/v1", api_key="test-key")
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "small")):
        ensure_compression_feasibility_checked(agent, 200_000)
        assert agent.context_compressor.threshold_tokens == 750_000
        entry["models"]["small"]["context_length"] = 80_000
        (home / "config.yaml").write_text(json.dumps({
            "auxiliary": {"compression": {"provider": "work-gateway", "model": "small"}},
            "custom_providers": [entry]}))
        ensure_compression_feasibility_checked(agent, 200_000)
    assert agent.context_compressor.threshold_tokens == 80_000


def test_a_route_edited_during_the_probe_is_probed_again(monkeypatch, tmp_path):
    """The latch records the route the probe measured: an edit that lands while the probe runs is
    probed on the next request instead of being labelled with the earlier measurement."""
    from agent.auxiliary_task_config import session_auxiliary_scope
    from agent.conversation_compression_feasibility import ensure_compression_feasibility_checked
    import agent.conversation_compression_feasibility as feas
    agent = _make_agent(main_context=1_000_000)
    agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=1_000_000, threshold_percent=0.75, quiet_mode=True,
    )
    agent._emit_status = lambda message: None
    agent._compression_feasibility_checked = False
    task = {"compression": {"provider": "custom", "model": "small", "base_url": "http://localhost/v1"}}
    probes = []
    real_probe = feas._probe

    def probe_then_edit(a):
        probes.append(task["compression"]["model"])
        real_probe(a)
        task["compression"]["model"] = "large"  # the edit lands while the probe runs
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")), \
         patch("agent.model_metadata.get_model_context_length", side_effect=[80_000, 900_000]), \
         patch.object(feas, "_probe", side_effect=probe_then_edit), session_auxiliary_scope(task):
        ensure_compression_feasibility_checked(agent, 200_000)
        ensure_compression_feasibility_checked(agent, 200_000)
    assert probes == ["small", "large"]

