"""The auxiliary-window compaction clamp must be reversible, not a ratchet.

``check_compression_model_feasibility`` caps a session's compaction trigger at
the auxiliary compression model's context window (the durable
``_aux_context_ceiling`` from #114707) so the summariser is never handed a
region it cannot ingest. The probe only ever lowered: once a small aux model
had installed a ceiling, pointing ``auxiliary.compression`` at a larger model
in the same session never lifted it, so a 272K session kept compacting at the
small model's window (~1.6x more often than configured) until restart.

These tests assert the relation both ways: after every probe the live trigger
is ``min(configured_trigger, aux_context)`` for the CURRENT aux model, where
the configured trigger is whatever the compressor's own derivation produces.
"""

from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.conversation_compression import check_compression_model_feasibility
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH

MAIN_CTX = 272_000
BIG_AUX_CTX = 272_000
SMALL_AUX_CTX = 128_000


class _Agent:
    """Minimal agent surface used by the feasibility check."""

    def __init__(self, compressor):
        self.context_compressor = compressor
        self.compression_enabled = True
        self.model = compressor.model
        self.provider = compressor.provider
        self.base_url = "https://main.example/v1"
        self._compression_warning = None
        self._aux_compression_context_length_config = None
        self._custom_providers = []
        self.status_callback = None
        self.emitted: list[str] = []

    def _current_main_runtime(self):
        return {"model": self.model, "provider": self.provider}

    def _emit_diagnostic_status(self, msg):
        self.emitted.append(msg)


def _compressor(threshold_percent=0.5, **kwargs):
    return ContextCompressor(
        model="main-model",
        provider="test-provider",
        threshold_percent=threshold_percent,
        config_context_length=MAIN_CTX,
        quiet_mode=True,
        **kwargs,
    )


def _configured(compressor):
    """The trigger the compressor's own derivation installs, with no aux ceiling."""
    return compressor.preview_threshold_tokens(
        compressor.model, compressor.context_length, compressor.provider
    )


class _AuxClient:
    base_url = "https://aux.example/v1"
    api_key = "aux-key"


def _run_check(agent, aux_ctx, aux_model="aux-model"):
    """Drive the real feasibility check with a stubbed aux route."""
    with (
        patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(_AuxClient(), aux_model),
        ),
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("aux-provider", aux_model, "", "", ""),
        ),
        patch("agent.model_metadata.get_model_context_length", return_value=aux_ctx),
    ):
        check_compression_model_feasibility(agent)


class TestClampIsReversible:
    def test_small_aux_model_lowers_the_trigger(self):
        compressor = _compressor()
        agent = _Agent(compressor)
        _run_check(agent, SMALL_AUX_CTX)
        assert compressor.threshold_tokens == SMALL_AUX_CTX < _configured(compressor)

    def test_switching_back_to_a_large_aux_model_restores_the_trigger(self):
        """The bug: this used to stay pinned at the small model's window."""
        compressor = _compressor()
        agent = _Agent(compressor)
        configured = compressor.threshold_tokens
        _run_check(agent, SMALL_AUX_CTX)
        assert compressor.threshold_tokens == SMALL_AUX_CTX

        _run_check(agent, BIG_AUX_CTX)
        assert compressor.threshold_tokens == configured
        assert compressor._aux_context_ceiling is None

    def test_restored_trigger_survives_a_same_runtime_recompute(self):
        """A lifted ceiling must not resurface on the next window correction."""
        compressor = _compressor()
        agent = _Agent(compressor)
        _run_check(agent, SMALL_AUX_CTX)
        _run_check(agent, BIG_AUX_CTX)
        compressor.update_model(compressor.model, MAIN_CTX, provider=compressor.provider)
        assert compressor.threshold_tokens == _configured(compressor)

    def test_repeated_checks_are_idempotent(self):
        compressor = _compressor()
        agent = _Agent(compressor)
        configured = _configured(compressor)
        for _ in range(3):
            _run_check(agent, SMALL_AUX_CTX)
            assert compressor.threshold_tokens == SMALL_AUX_CTX
        for _ in range(3):
            _run_check(agent, BIG_AUX_CTX)
            assert compressor.threshold_tokens == configured

    def test_trigger_always_fits_the_current_aux_window(self):
        """The invariant, stated directly: trigger == min(configured, aux)."""
        compressor = _compressor()
        agent = _Agent(compressor)
        configured = _configured(compressor)
        for aux_ctx in (
            SMALL_AUX_CTX, BIG_AUX_CTX, 200_000, BIG_AUX_CTX, MINIMUM_CONTEXT_LENGTH, BIG_AUX_CTX,
        ):
            _run_check(agent, aux_ctx)
            assert compressor.threshold_tokens == min(configured, aux_ctx)

    def test_hard_rejection_keeps_the_previous_ceiling(self):
        """An aux below the minimum is refused before anything is lifted."""
        compressor = _compressor()
        agent = _Agent(compressor)
        _run_check(agent, SMALL_AUX_CTX)
        with pytest.raises(ValueError):
            _run_check(agent, MINIMUM_CONTEXT_LENGTH - 1)
        assert compressor.threshold_tokens == SMALL_AUX_CTX


class TestClampDoesNotCorruptDerivation:
    def test_clamp_never_writes_a_sub_floor_threshold_percent(self):
        """A raw ``clamped/context`` ratio bypasses the small-window floor."""
        compressor = _compressor()
        agent = _Agent(compressor)
        _run_check(agent, SMALL_AUX_CTX)
        floor = ContextCompressor._effective_threshold_percent(MAIN_CTX, 0.0)
        assert compressor.threshold_percent >= floor

    @pytest.mark.parametrize("tail_mode", ["lean", "legacy"])
    def test_tail_budget_follows_the_trigger_in_both_directions(self, tail_mode):
        compressor = _compressor(tail_mode=tail_mode)
        agent = _Agent(compressor)
        unclamped_tail = compressor.tail_token_budget

        _run_check(agent, SMALL_AUX_CTX)
        assert compressor.tail_token_budget <= compressor.threshold_tokens

        _run_check(agent, BIG_AUX_CTX)
        assert compressor.tail_token_budget == unclamped_tail

    def test_restore_honours_a_threshold_edit_made_while_clamped(self):
        """Restoring re-derives from config instead of replaying a snapshot."""
        compressor = _compressor()
        agent = _Agent(compressor)
        _run_check(agent, SMALL_AUX_CTX)

        compressor._config_threshold_percent = 0.90
        _run_check(agent, BIG_AUX_CTX)
        assert compressor.threshold_tokens == _configured(compressor)
        assert compressor.threshold_tokens > _configured(_compressor())

    def test_restore_honours_per_model_override_and_output_reservation(self):
        compressor = _compressor(model_thresholds={"main-model": 0.9}, max_tokens=32_000)
        agent = _Agent(compressor)
        configured = compressor.threshold_tokens
        _run_check(agent, SMALL_AUX_CTX)
        _run_check(agent, BIG_AUX_CTX)
        assert compressor.threshold_tokens == configured == _configured(compressor)

    def test_restore_respects_the_absolute_threshold_cap(self):
        compressor = _compressor()
        agent = _Agent(compressor)
        _run_check(agent, SMALL_AUX_CTX)

        compressor.threshold_tokens_cap = 100_000
        _run_check(agent, BIG_AUX_CTX)
        assert compressor.threshold_tokens == _configured(compressor) <= 100_000
