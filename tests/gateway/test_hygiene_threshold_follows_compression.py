"""Gateway session hygiene follows a RAISED ``compression.threshold`` (#118984).

The pre-agent hygiene net is a safety net: it must fire at or AFTER the agent's own
compressor, never before it.  It was pinned at a hard-coded ratio while the agent's
trigger came from ``compression.threshold``, so any ratio above the net's default was
silently capped at the net (the agent's compressor never ran and the session compacted
short of the ratio the operator asked for).

These tests pin both halves of the contract:

1. ``compression.threshold`` lifts the net (and never lowers it).
2. The net's decision point on a 1M-token window: a ratio of 0.95 leaves the
   850K-950K band to the agent's compressor, while the net still fires above it.
"""

import asyncio
import types
from unittest.mock import patch

import pytest

import gateway.run_turn as run_turn
from gateway.run import GatewayRunner

DEFAULT_NET_PCT = 0.85
WINDOW = 1_000_000


def _hygiene_settings(threshold_pct: float = DEFAULT_NET_PCT):
    """Same shape ``_hmwa_hygiene_settings`` builds (GatewayRunner._HygieneSettings)."""
    return GatewayRunner._HygieneSettings(
        model="anthropic/claude-sonnet-4.6", threshold_pct=threshold_pct, compression_enabled=True,
        hard_msg_limit=5000, timeout_seconds=30.0, total_ceiling_seconds=600.0,
        max_turn_hold_seconds=10.0, failure_cooldown_seconds=300.0, config_context_length=None,
        provider=None, base_url=None, api_key=None, data={},
    )


def _read_config(hs, compression):
    GatewayRunner._hmwa_hygiene_read_config(hs, {"compression": compression})
    return hs


class TestHygieneNetRatioFollowsCompressionThreshold:
    def test_raised_threshold_lifts_the_net(self):
        """compression.threshold: 0.95 must raise the net off its 0.85 default."""
        hs = _read_config(_hygiene_settings(), {"enabled": True, "threshold": 0.95})
        assert hs.threshold_pct == pytest.approx(0.95)

    def test_shipped_default_keeps_the_net_above_the_compressor(self):
        """The default 0.50 must not pull the net down to the agent's trigger."""
        hs = _read_config(_hygiene_settings(), {"enabled": True, "threshold": 0.50})
        assert hs.threshold_pct == pytest.approx(DEFAULT_NET_PCT)

    def test_lower_threshold_never_lowers_the_net(self):
        """A ratio below the net keeps the net a net (it is a floor, not a mirror)."""
        hs = _read_config(_hygiene_settings(), {"enabled": True, "threshold": 0.30})
        assert hs.threshold_pct == pytest.approx(DEFAULT_NET_PCT)

    def test_string_ratio_is_accepted_like_the_agent_side(self):
        """The agent resolves the ratio with float(); the net accepts the same shape."""
        hs = _read_config(_hygiene_settings(), {"enabled": True, "threshold": "0.97"})
        assert hs.threshold_pct == pytest.approx(0.97)

    @pytest.mark.parametrize("bad", [None, "not-a-number", True, {"nested": 1}])
    def test_invalid_ratio_keeps_the_net_default(self, bad):
        hs = _read_config(_hygiene_settings(), {"enabled": True, "threshold": bad})
        assert hs.threshold_pct == pytest.approx(DEFAULT_NET_PCT)

    def test_missing_compression_section_keeps_defaults(self):
        hs = _hygiene_settings()
        GatewayRunner._hmwa_hygiene_read_config(hs, {})
        assert hs.threshold_pct == pytest.approx(DEFAULT_NET_PCT)
        assert hs.compression_enabled is True


class _PlanRunner:
    """Minimal runner exposing only what ``_hmwa_hygiene_plan`` touches."""

    _session_db = None
    _HygienePlan = run_turn.GatewayTurnMixin._HygienePlan

    def __init__(self):
        self.compressed_sessions = []

    async def _session_has_compression_in_flight(self, session_key):
        self.compressed_sessions.append(session_key)
        return False


def _plan_for(prompt_tokens: int, threshold_pct: float):
    hs = _hygiene_settings(threshold_pct)

    async def _fake_context_length(model, **kwargs):
        return WINDOW

    runner = _PlanRunner()
    with patch(
        "agent.model_metadata.get_model_context_length_async",
        new=_fake_context_length,
    ):
        plan = asyncio.run(
            run_turn.GatewayTurnMixin._hmwa_hygiene_plan(
                runner, hs, [{"role": "user", "content": "x" * 40}],
                types.SimpleNamespace(last_prompt_tokens=prompt_tokens, session_id="sess-1"),
                "session-key",
            )
        )
    return plan, hs


class TestHygieneNetDecisionPoint:
    def test_widened_threshold_leaves_the_band_to_the_agent_compressor(self):
        """0.95 x 1M = 950K: 900K of real usage must NOT trigger the net.

        Pre-fix the net resolved 0.85 (850K) and compressed here, so the agent's own
        compressor — the surface the operator configured — never ran.
        """
        hs = _read_config(_hygiene_settings(), {"enabled": True, "threshold": 0.95})
        plan, _ = _plan_for(900_000, hs.threshold_pct)
        assert plan.needs_compress is False, "net pre-empted the agent's 0.95 trigger at 900K"

    def test_net_still_fires_above_the_widened_threshold(self):
        """The net stays a net: 960K > 0.95 x 1M, so it compresses."""
        hs = _read_config(_hygiene_settings(), {"enabled": True, "threshold": 0.95})
        plan, _ = _plan_for(960_000, hs.threshold_pct)
        assert plan.needs_compress is True

    def test_default_ratio_still_fires_at_the_net(self):
        """Unchanged default behaviour: 0.50 for the agent, 0.85 for the net."""
        hs = _read_config(_hygiene_settings(), {"enabled": True, "threshold": 0.50})
        plan, _ = _plan_for(900_000, hs.threshold_pct)
        assert plan.needs_compress is True
