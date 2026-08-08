"""Regression tests for #81209: CLI/oneshot must consult fallback_providers
at resolution time, not only after the session starts.

Before the fix, ``_run_agent`` called ``resolve_runtime_provider`` bare,
so a quota-exhausted primary (429) raised *before* ``AIAgent`` was
constructed and the ``fallback_model=_fb`` wiring that handles
mid-session failures was never reached.  The gateway already had this
behaviour via ``_try_resolve_fallback_provider``; the helper introduced
here brings the oneshot path to parity.
"""

from unittest.mock import patch

import pytest

from hermes_cli import oneshot


@pytest.fixture
def cfg_with_fallback():
    return {
        "model": {"default": "primary-model"},
        "fallback_providers": [
            {"provider": "anthropic", "model": "haiku"},
            {"provider": "openai", "model": "gpt-4o-mini"},
        ],
    }


@pytest.fixture
def cfg_no_fallback():
    return {
        "model": {"default": "primary-model"},
    }


class TestResolveRuntimeWithFallback:
    def test_primary_success_short_circuits(self, cfg_with_fallback):
        primary_runtime = {"provider": "openai", "api_key": "k1"}
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=primary_runtime,
        ) as resolve:
            runtime, err = oneshot._resolve_runtime_with_fallback(
                effective_provider="openai",
                effective_model="gpt-4o",
                explicit_base_url=None,
                cfg=cfg_with_fallback,
            )

        assert runtime is primary_runtime
        assert err is None
        # Primary success must not touch fallback entries.
        assert resolve.call_count == 1

    def test_primary_quota_failure_invokes_fallback(self, cfg_with_fallback):
        primary_err = RuntimeError("Codex provider quota exhausted (429)")
        fallback_runtime = {"provider": "anthropic", "api_key": "k2"}

        # First call raises, second call (fallback entry) succeeds.
        call_log = []

        def fake_resolve(*, requested, target_model=None, **kwargs):
            call_log.append(requested)
            if requested == "openai":
                raise primary_err
            return fallback_runtime

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=fake_resolve,
        ), patch(
            "hermes_cli.fallback_config.resolve_entry_api_key",
            return_value="resolved-key",
        ):
            runtime, err = oneshot._resolve_runtime_with_fallback(
                effective_provider="openai",
                effective_model="gpt-4o",
                explicit_base_url=None,
                cfg=cfg_with_fallback,
            )

        assert err is None
        # The fallback entry's ``model`` is now stamped onto a copy of
        # ``fallback_runtime`` so the helper always returns a fresh dict;
        # compare by content rather than identity.
        assert runtime == {**fallback_runtime, "model": "haiku"}
        # First call: primary; second call: first fallback entry.
        assert call_log == ["openai", "anthropic"]

    def test_primary_failure_no_fallback_chain_returns_primary_error(
        self, cfg_no_fallback
    ):
        primary_err = RuntimeError("primary down")

        def fake_resolve(**kwargs):
            raise primary_err

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=fake_resolve,
        ):
            runtime, err = oneshot._resolve_runtime_with_fallback(
                effective_provider="openai",
                effective_model="gpt-4o",
                explicit_base_url=None,
                cfg=cfg_no_fallback,
            )

        assert runtime is None
        # Operator gets the primary's error message (not a fallback that
        # was never configured in the first place).
        assert err is primary_err

    def test_all_fallbacks_exhausted_returns_primary_error(
        self, cfg_with_fallback
    ):
        primary_err = RuntimeError("primary quota")

        def fake_resolve(*, requested, **kwargs):
            # Primary (first call) raises the original primary error;
            # subsequent fallback entries raise their own distinct errors.
            if requested == "openai":
                raise primary_err
            raise RuntimeError(f"{requested} also down")

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=fake_resolve,
        ):
            runtime, err = oneshot._resolve_runtime_with_fallback(
                effective_provider="openai",
                effective_model="gpt-4o",
                explicit_base_url=None,
                cfg=cfg_with_fallback,
            )

        # Three calls: primary + 2 fallback entries, all fail.
        assert runtime is None
        # Primary error wins (operator-facing) — not the second fallback's.
        assert err is primary_err

    def test_second_fallback_succeeds_when_first_also_fails(
        self, cfg_with_fallback
    ):
        primary_err = RuntimeError("primary quota")
        openai_runtime = {"provider": "openai", "api_key": "openai-key"}

        # Distinguish the primary call (first) from the fallback entry call
        # (third) — they both target openai, so the side_effect needs a
        # per-call gate.
        state = {"calls": 0}

        def fake_resolve(*, requested, **kwargs):
            state["calls"] += 1
            if state["calls"] == 1:
                # First call: primary's configured provider.
                assert requested == "openai"
                raise primary_err
            if requested == "anthropic":
                # Second call: first fallback entry, also fails.
                raise RuntimeError("anthropic auth invalid")
            # Third call: openai fallback entry — succeeds.
            assert state["calls"] == 3
            return openai_runtime

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=fake_resolve,
        ), patch(
            "hermes_cli.fallback_config.resolve_entry_api_key",
            return_value="key",
        ):
            runtime, err = oneshot._resolve_runtime_with_fallback(
                effective_provider="openai",
                effective_model="gpt-4o",
                explicit_base_url=None,
                cfg=cfg_with_fallback,
            )

        assert err is None
        assert runtime == {**openai_runtime, "model": "gpt-4o-mini"}
        assert state["calls"] == 3

    def test_fallback_model_is_injected_into_runtime(self, cfg_with_fallback):
        """The fallback entry's ``model`` must be carried into the returned
        runtime dict so AIAgent constructs against the fallback's model
        rather than the originally-requested primary model (#81209)."""
        primary_err = RuntimeError("primary quota")
        fallback_runtime = {"provider": "anthropic", "api_key": "anth-key"}

        def fake_resolve(*, requested, **kwargs):
            if requested == "openai":
                raise primary_err
            return fallback_runtime

        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=fake_resolve,
        ), patch(
            "hermes_cli.fallback_config.resolve_entry_api_key",
            return_value="key",
        ):
            runtime, err = oneshot._resolve_runtime_with_fallback(
                effective_provider="openai",
                effective_model="gpt-4o",
                explicit_base_url=None,
                cfg=cfg_with_fallback,
            )

        assert err is None
        # ``cfg_with_fallback`` declares the first fallback entry with
        # model ``haiku``; that key must appear on the returned runtime so
        # AIAgent constructs against the fallback's model instead of the
        # originally-requested primary model (#81209).
        assert runtime.get("model") == "haiku"