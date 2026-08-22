"""RED/GREEN tests for the main-orchestrator usage-aware routing module.

``agent/orchestrator_usage_routing.py`` decides, per turn, whether the main
interactive orchestrator (CLI or gateway) should run on its configured
primary native provider/model (openai-codex / gpt-5.6-sol) or fall over to
a secondary native provider/model (google-antigravity / gemini-3.1-pro-high,
the "Agy" fallback) when the cached primary-provider usage reading is low.

This module must stay a pure, side-effect-free decision function: it is fed
an already-resolved ``ProviderUsage`` reading (from
``agent.delegation_usage_cache.build_usage_view``) plus already-known "what
provider/model is currently active" state, and returns a deterministic
``OrchestratorRouteDecision``. It never calls the cache's refresh entry
points itself.
"""

from __future__ import annotations

import math

import pytest

from agent.delegation_routing import CLAUDE_P_PROVIDER, ProviderUsage

# Imported lazily inside fixtures/tests below where useful so the module's
# ImportError (in the RED phase, before the module exists) is captured by
# the specific test rather than collection-time failure for the whole file.


PRIMARY_PROVIDER = "openai-codex"
PRIMARY_MODEL = "gpt-5.6-sol"
FALLBACK_PROVIDER = "google-antigravity"
FALLBACK_MODEL = "gemini-3.1-pro-high"


def _valid_config(**overrides):
    cfg = {
        "enabled": True,
        "primary_provider": PRIMARY_PROVIDER,
        "primary_model": PRIMARY_MODEL,
        "fallback_provider": FALLBACK_PROVIDER,
        "fallback_model": FALLBACK_MODEL,
        "switch_at_remaining_percent": 10,
        "restore_above_remaining_percent": 10,
        "usage_ttl_seconds": 900,
        "usage_stale_seconds": 7200,
        "primary_usage_window_prefixes": ["Session", "Weekly"],
    }
    cfg.update(overrides)
    return cfg


def _usage(remaining, freshness, age_seconds=None):
    return ProviderUsage(
        provider=PRIMARY_PROVIDER,
        remaining_percent=remaining,
        freshness=freshness,
        age_seconds=age_seconds,
    )


class TestParseConfig:
    def test_parses_valid_block(self):
        from agent.orchestrator_usage_routing import parse_orchestrator_routing_config

        parsed = parse_orchestrator_routing_config({"agent": {"orchestrator_usage_routing": _valid_config()}})
        assert parsed.enabled is True
        assert parsed.primary_provider == PRIMARY_PROVIDER
        assert parsed.primary_model == PRIMARY_MODEL
        assert parsed.fallback_provider == FALLBACK_PROVIDER
        assert parsed.fallback_model == FALLBACK_MODEL
        assert parsed.switch_at_remaining_percent == 10
        assert parsed.restore_above_remaining_percent == 10
        assert parsed.usage_ttl_seconds == 900
        assert parsed.usage_stale_seconds == 7200
        assert tuple(parsed.primary_usage_window_prefixes) == ("Session", "Weekly")

    def test_absent_block_is_inert(self):
        from agent.orchestrator_usage_routing import parse_orchestrator_routing_config

        parsed = parse_orchestrator_routing_config({})
        assert parsed.enabled is False

    def test_absent_block_is_inert_none_config(self):
        from agent.orchestrator_usage_routing import parse_orchestrator_routing_config

        parsed = parse_orchestrator_routing_config(None)
        assert parsed.enabled is False

    def test_enabled_false_short_circuits_before_validating_rest(self):
        """Mirrors delegation_routing.load_route_catalog: enabled=false must
        be inert even when the rest of the block is garbage."""
        from agent.orchestrator_usage_routing import parse_orchestrator_routing_config

        cfg = {
            "agent": {
                "orchestrator_usage_routing": {
                    "enabled": False,
                    "primary_provider": "claude-p",  # would be a hard error if validated
                    "switch_at_remaining_percent": "not-a-number",
                    "usage_stale_seconds": 1,
                    "usage_ttl_seconds": 900,
                }
            }
        }
        parsed = parse_orchestrator_routing_config(cfg)
        assert parsed.enabled is False

    def test_rejects_claude_p_as_primary(self):
        from agent.orchestrator_usage_routing import (
            OrchestratorRoutingConfigError,
            parse_orchestrator_routing_config,
        )

        cfg = {"agent": {"orchestrator_usage_routing": _valid_config(primary_provider=CLAUDE_P_PROVIDER)}}
        with pytest.raises(OrchestratorRoutingConfigError):
            parse_orchestrator_routing_config(cfg)

    def test_rejects_claude_p_as_fallback(self):
        from agent.orchestrator_usage_routing import (
            OrchestratorRoutingConfigError,
            parse_orchestrator_routing_config,
        )

        cfg = {"agent": {"orchestrator_usage_routing": _valid_config(fallback_provider="claude-p")}}
        with pytest.raises(OrchestratorRoutingConfigError):
            parse_orchestrator_routing_config(cfg)

    def test_rejects_unsupported_native_provider(self):
        from agent.orchestrator_usage_routing import (
            OrchestratorRoutingConfigError,
            parse_orchestrator_routing_config,
        )

        cfg = {"agent": {"orchestrator_usage_routing": _valid_config(primary_provider="anthropic")}}
        with pytest.raises(OrchestratorRoutingConfigError):
            parse_orchestrator_routing_config(cfg)

    @pytest.mark.parametrize("bad", [-1, 101, math.nan, math.inf, "abc", None])
    def test_rejects_bad_switch_percent(self, bad):
        from agent.orchestrator_usage_routing import (
            OrchestratorRoutingConfigError,
            parse_orchestrator_routing_config,
        )

        if bad is None:
            # None is allowed to fall back to default elsewhere in this repo's
            # idiom, but this field is required by the contract here — assert
            # a clear error either way (missing vs invalid both reported).
            cfg = {"agent": {"orchestrator_usage_routing": _valid_config()}}
            del cfg["agent"]["orchestrator_usage_routing"]["switch_at_remaining_percent"]
        else:
            cfg = {"agent": {"orchestrator_usage_routing": _valid_config(switch_at_remaining_percent=bad)}}
        with pytest.raises(OrchestratorRoutingConfigError):
            parse_orchestrator_routing_config(cfg)

    def test_rejects_bad_restore_percent(self):
        from agent.orchestrator_usage_routing import (
            OrchestratorRoutingConfigError,
            parse_orchestrator_routing_config,
        )

        cfg = {"agent": {"orchestrator_usage_routing": _valid_config(restore_above_remaining_percent=200)}}
        with pytest.raises(OrchestratorRoutingConfigError):
            parse_orchestrator_routing_config(cfg)

    def test_rejects_stale_less_than_ttl(self):
        from agent.orchestrator_usage_routing import (
            OrchestratorRoutingConfigError,
            parse_orchestrator_routing_config,
        )

        cfg = {
            "agent": {
                "orchestrator_usage_routing": _valid_config(
                    usage_ttl_seconds=1000, usage_stale_seconds=500
                )
            }
        }
        with pytest.raises(OrchestratorRoutingConfigError):
            parse_orchestrator_routing_config(cfg)

    def test_rejects_non_positive_ttl(self):
        from agent.orchestrator_usage_routing import (
            OrchestratorRoutingConfigError,
            parse_orchestrator_routing_config,
        )

        cfg = {"agent": {"orchestrator_usage_routing": _valid_config(usage_ttl_seconds=0)}}
        with pytest.raises(OrchestratorRoutingConfigError):
            parse_orchestrator_routing_config(cfg)

    def test_rejects_non_int_ttl(self):
        from agent.orchestrator_usage_routing import (
            OrchestratorRoutingConfigError,
            parse_orchestrator_routing_config,
        )

        cfg = {"agent": {"orchestrator_usage_routing": _valid_config(usage_ttl_seconds="soon")}}
        with pytest.raises(OrchestratorRoutingConfigError):
            parse_orchestrator_routing_config(cfg)

    def test_missing_required_key_raises(self):
        from agent.orchestrator_usage_routing import (
            OrchestratorRoutingConfigError,
            parse_orchestrator_routing_config,
        )

        cfg = _valid_config()
        del cfg["primary_provider"]
        with pytest.raises(OrchestratorRoutingConfigError):
            parse_orchestrator_routing_config({"agent": {"orchestrator_usage_routing": cfg}})

    def test_wrong_type_block_raises(self):
        from agent.orchestrator_usage_routing import (
            OrchestratorRoutingConfigError,
            parse_orchestrator_routing_config,
        )

        with pytest.raises(OrchestratorRoutingConfigError):
            parse_orchestrator_routing_config({"agent": {"orchestrator_usage_routing": "not-a-mapping"}})


class TestDecision:
    def _config(self, **overrides):
        from agent.orchestrator_usage_routing import parse_orchestrator_routing_config

        return parse_orchestrator_routing_config({"agent": {"orchestrator_usage_routing": _valid_config(**overrides)}})

    def _decide(self, *, current_provider, usage, config=None, current_model=None):
        from agent.orchestrator_usage_routing import decide_orchestrator_route

        return decide_orchestrator_route(
            current_provider=current_provider,
            usage=usage,
            config=config or self._config(),
            current_model=current_model,
        )

    def test_remaining_exactly_at_threshold_switches_to_agy(self):
        decision = self._decide(
            current_provider=PRIMARY_PROVIDER,
            usage=_usage(10.0, "fresh", age_seconds=10),
        )
        assert decision.provider == FALLBACK_PROVIDER
        assert decision.model == FALLBACK_MODEL

    def test_remaining_below_threshold_switches_to_agy(self):
        decision = self._decide(
            current_provider=PRIMARY_PROVIDER,
            usage=_usage(3.0, "fresh", age_seconds=10),
        )
        assert decision.provider == FALLBACK_PROVIDER
        assert decision.model == FALLBACK_MODEL

    def test_stale_low_reading_while_on_codex_stays_on_codex(self):
        decision = self._decide(
            current_provider=PRIMARY_PROVIDER,
            usage=_usage(3.0, "stale", age_seconds=5000),
        )
        assert decision.provider == PRIMARY_PROVIDER
        assert decision.model == PRIMARY_MODEL
        assert decision.switched is False

    def test_remaining_above_threshold_fresh_stays_on_codex(self):
        decision = self._decide(
            current_provider=PRIMARY_PROVIDER,
            usage=_usage(50.0, "fresh", age_seconds=10),
        )
        assert decision.provider == PRIMARY_PROVIDER
        assert decision.model == PRIMARY_MODEL

    def test_on_agy_fresh_exactly_ten_does_not_recover(self):
        decision = self._decide(
            current_provider=FALLBACK_PROVIDER,
            usage=_usage(10.0, "fresh", age_seconds=10),
        )
        assert decision.provider == FALLBACK_PROVIDER
        assert decision.model == FALLBACK_MODEL

    def test_on_agy_fresh_good_reading_recovers_to_codex(self):
        decision = self._decide(
            current_provider=FALLBACK_PROVIDER,
            usage=_usage(50.0, "fresh", age_seconds=10),
        )
        assert decision.provider == PRIMARY_PROVIDER
        assert decision.model == PRIMARY_MODEL

    def test_on_agy_stale_good_reading_does_not_recover(self):
        decision = self._decide(
            current_provider=FALLBACK_PROVIDER,
            usage=_usage(50.0, "stale", age_seconds=5000),
        )
        assert decision.provider == FALLBACK_PROVIDER
        assert decision.model == FALLBACK_MODEL

    def test_on_agy_stale_low_reading_stays_agy(self):
        decision = self._decide(
            current_provider=FALLBACK_PROVIDER,
            usage=_usage(5.0, "stale", age_seconds=5000),
        )
        assert decision.provider == FALLBACK_PROVIDER
        assert decision.model == FALLBACK_MODEL

    def test_unknown_at_startup_no_prior_route_stays_codex(self):
        decision = self._decide(
            current_provider=None,
            usage=_usage(None, "unknown"),
        )
        assert decision.provider == PRIMARY_PROVIDER
        assert decision.model == PRIMARY_MODEL

    def test_unknown_while_currently_agy_stays_agy(self):
        decision = self._decide(
            current_provider=FALLBACK_PROVIDER,
            usage=_usage(None, "unknown"),
        )
        assert decision.provider == FALLBACK_PROVIDER
        assert decision.model == FALLBACK_MODEL

    def test_unknown_while_currently_codex_stays_codex(self):
        decision = self._decide(
            current_provider=PRIMARY_PROVIDER,
            usage=_usage(None, "unknown"),
        )
        assert decision.provider == PRIMARY_PROVIDER
        assert decision.model == PRIMARY_MODEL

    def test_disabled_always_keeps_current_regardless_of_usage(self):
        cfg = self._config(enabled=False)
        # Even a very low usage reading must not switch when disabled.
        decision = self._decide(
            current_provider=PRIMARY_PROVIDER,
            current_model=PRIMARY_MODEL,
            usage=_usage(0.0, "fresh", age_seconds=1),
            config=cfg,
        )
        assert decision.provider == PRIMARY_PROVIDER
        assert decision.model == PRIMARY_MODEL
        assert decision.switched is False

    def test_disabled_stays_on_agy_if_that_was_current(self):
        cfg = self._config(enabled=False)
        decision = self._decide(
            current_provider=FALLBACK_PROVIDER,
            current_model=FALLBACK_MODEL,
            usage=_usage(0.0, "fresh", age_seconds=1),
            config=cfg,
        )
        assert decision.provider == FALLBACK_PROVIDER
        assert decision.model == FALLBACK_MODEL
        assert decision.switched is False

    def test_reason_string_has_no_credential_looking_content(self):
        decision = self._decide(
            current_provider=PRIMARY_PROVIDER,
            usage=_usage(3.0, "fresh", age_seconds=10),
        )
        for banned in ("sk-", "Bearer ", "api_key", "token", "Authorization"):
            assert banned.lower() not in decision.reason.lower()

    def test_decision_never_targets_claude_p_even_with_bad_current_provider(self):
        """Defense in depth: even if some caller passes claude-p as the
        'current' provider (a misconfiguration bug elsewhere), the decision
        must never *select* claude-p as the target."""
        decision = self._decide(
            current_provider=CLAUDE_P_PROVIDER,
            usage=_usage(50.0, "fresh", age_seconds=10),
        )
        assert decision.provider != CLAUDE_P_PROVIDER


class TestNonBlockingCacheUsage:
    """Prove the decision path only ever reads through build_usage_view and
    never calls the network-touching refresh/fetch entry points itself."""

    def test_decision_path_never_calls_fetch_or_refresh(self, monkeypatch):
        import agent.account_usage as account_usage_mod
        import agent.delegation_usage_cache as cache_mod

        def _boom(*args, **kwargs):
            raise AssertionError("routing decision must not call this directly")

        monkeypatch.setattr(account_usage_mod, "fetch_account_usage", _boom)
        monkeypatch.setattr(cache_mod, "refresh_provider_now", _boom)

        from agent.orchestrator_usage_routing import decide_orchestrator_route

        config = self._config()
        # Exercise decide_orchestrator_route across the interesting branches;
        # none of them may touch fetch_account_usage / refresh_provider_now.
        for provider, remaining, freshness in (
            (PRIMARY_PROVIDER, 3.0, "fresh"),
            (PRIMARY_PROVIDER, 50.0, "fresh"),
            (FALLBACK_PROVIDER, 50.0, "fresh"),
            (FALLBACK_PROVIDER, 50.0, "stale"),
            (None, None, "unknown"),
        ):
            decide_orchestrator_route(
                current_provider=provider,
                usage=_usage(remaining, freshness, age_seconds=10),
                config=config,
            )

    def _config(self):
        from agent.orchestrator_usage_routing import parse_orchestrator_routing_config

        return parse_orchestrator_routing_config({"agent": {"orchestrator_usage_routing": _valid_config()}})
