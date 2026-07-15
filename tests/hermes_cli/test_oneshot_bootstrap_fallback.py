"""Oneshot (-z) bootstrap must honour fallback_providers on primary AuthError.

Repro (#oneshot-codex-fallback): HERMES_HOME with model.provider=openai-codex,
every Codex credential exhausted/missing, and fallback_providers configured.
``hermes -z "..." --cli`` fails immediately with "No Codex credentials stored"
because ``hermes_cli.oneshot._run_agent`` resolves the provider via
``resolve_runtime_provider()`` BEFORE building the agent, and that bootstrap
call re-raises AuthError without consulting the fallback chain. The interactive
CLI (``_ensure_runtime_credentials``) and the cron scheduler both already fall
through to fallback_providers on the same failure; oneshot must match.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

import hermes_cli.oneshot as oneshot

# Imported at module scope ON PURPOSE: runtime_provider snapshots
# ``hermes_cli.config.load_config`` into its own namespace at import time.
# If its first import happens inside a test whose fixture has monkeypatched
# config_mod.load_config, the mock gets baked into runtime_provider and
# LEAKS past monkeypatch teardown into later tests (the integration test
# below resolves through the real runtime_provider stack).
import hermes_cli.runtime_provider  # noqa: F401
from hermes_cli.auth import AuthError

NVIDIA_RUNTIME = {
    "provider": "custom",
    "api_mode": "chat_completions",
    "base_url": "https://integrate.api.nvidia.com/v1",
    "api_key": "nvapi-test",
    "source": "custom-provider:nvidia",
}

NVIDIA2_RUNTIME = {
    "provider": "custom",
    "api_mode": "chat_completions",
    "base_url": "https://integrate2.api.nvidia.com/v1",
    "api_key": "nvapi-test-2",
    "source": "custom-provider:nvidia2",
}


def _codex_auth_error() -> AuthError:
    return AuthError(
        "No Codex credentials stored. Run `hermes auth` to authenticate.",
        provider="openai-codex",
        code="codex_auth_missing",
        relogin_required=True,
    )


class FakeAgent:
    """Stands in for run_agent.AIAgent; records construction kwargs."""

    instances: list["FakeAgent"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.suppress_status_output = False
        self.stream_delta_callback = None
        self.tool_gen_callback = None
        FakeAgent.instances.append(self)

    def run_conversation(self, prompt):
        return {"final_response": "ok"}


@pytest.fixture
def oneshot_env(monkeypatch):
    """Isolate _run_agent from config files, tools, sessions, and AIAgent."""
    FakeAgent.instances = []
    config = {
        "model": {"provider": "openai-codex", "model": "gpt-5.5-codex"},
        "fallback_providers": [
            {"provider": "custom:nvidia", "model": "moonshotai/kimi-k2.5"},
            {"provider": "custom:nvidia2", "model": "qwen/qwen3.5-coder"},
        ],
    }

    import hermes_cli.config as config_mod
    import hermes_cli.tools_config as tools_config_mod
    import run_agent as run_agent_mod

    monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(tools_config_mod, "_get_platform_tools", lambda *a, **k: set())
    monkeypatch.setattr(run_agent_mod, "AIAgent", FakeAgent)
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    return config


def _patch_resolver(monkeypatch, runtimes_by_provider):
    """resolve_runtime_provider stub: raise AuthError for the primary (codex)
    request, return a canned runtime for known fallback providers."""
    calls = []

    def fake_resolve(requested=None, **kwargs):
        calls.append(requested)
        if not requested or requested in ("openai-codex", "codex"):
            raise _codex_auth_error()
        runtime = runtimes_by_provider.get(requested)
        if runtime is None:
            raise AuthError(
                f"No credentials for {requested}.", provider=str(requested)
            )
        return dict(runtime)

    import hermes_cli.runtime_provider as rp_mod

    monkeypatch.setattr(rp_mod, "resolve_runtime_provider", fake_resolve)
    return calls


class TestOneshotBootstrapFallback:
    def test_primary_auth_error_falls_back_to_first_configured_provider(
        self, monkeypatch, oneshot_env
    ):
        calls = _patch_resolver(monkeypatch, {"custom:nvidia": NVIDIA_RUNTIME})

        response, result = oneshot._run_agent("hello")

        assert response == "ok"
        assert calls[0] in (None, "openai-codex")
        assert "custom:nvidia" in calls
        agent = FakeAgent.instances[-1]
        assert agent.kwargs["api_key"] == "nvapi-test"
        assert agent.kwargs["base_url"] == "https://integrate.api.nvidia.com/v1"
        assert agent.kwargs["provider"] == "custom"

    def test_fallback_switches_model_to_fallback_entry_model(
        self, monkeypatch, oneshot_env
    ):
        _patch_resolver(monkeypatch, {"custom:nvidia": NVIDIA_RUNTIME})

        oneshot._run_agent("hello")

        agent = FakeAgent.instances[-1]
        # The configured codex model can't run on the fallback provider —
        # the fallback entry's own model must be used.
        assert agent.kwargs["model"] == "moonshotai/kimi-k2.5"

    def test_fallback_chain_iterates_past_failing_entries(
        self, monkeypatch, oneshot_env
    ):
        # First fallback also has no credentials; second resolves.
        calls = _patch_resolver(monkeypatch, {"custom:nvidia2": NVIDIA2_RUNTIME})

        response, _ = oneshot._run_agent("hello")

        assert response == "ok"
        assert calls == [None, "custom:nvidia", "custom:nvidia2"] or calls == [
            "openai-codex",
            "custom:nvidia",
            "custom:nvidia2",
        ]
        agent = FakeAgent.instances[-1]
        assert agent.kwargs["api_key"] == "nvapi-test-2"
        assert agent.kwargs["model"] == "qwen/qwen3.5-coder"

    def test_auth_error_propagates_when_no_fallback_configured(
        self, monkeypatch, oneshot_env
    ):
        oneshot_env["fallback_providers"] = []
        _patch_resolver(monkeypatch, {})

        with pytest.raises(AuthError, match="No Codex credentials stored"):
            oneshot._run_agent("hello")
        assert FakeAgent.instances == []

    def test_original_auth_error_propagates_when_all_fallbacks_fail(
        self, monkeypatch, oneshot_env
    ):
        _patch_resolver(monkeypatch, {})  # every fallback raises AuthError too

        with pytest.raises(AuthError, match="No Codex credentials stored"):
            oneshot._run_agent("hello")
        assert FakeAgent.instances == []

    def test_non_auth_error_is_not_swallowed_by_fallback(
        self, monkeypatch, oneshot_env
    ):
        import hermes_cli.runtime_provider as rp_mod

        def broken_resolve(requested=None, **kwargs):
            raise ValueError("config parse exploded")

        monkeypatch.setattr(rp_mod, "resolve_runtime_provider", broken_resolve)

        with pytest.raises(ValueError, match="config parse exploded"):
            oneshot._run_agent("hello")
        assert FakeAgent.instances == []

    def test_explicit_model_and_provider_pair_propagates_auth_error(
        self, monkeypatch, oneshot_env
    ):
        # The caller pinned BOTH --model and --provider: that exact pair is
        # the contract. No fallback may be attempted; AuthError propagates.
        calls = _patch_resolver(monkeypatch, {"custom:nvidia": NVIDIA_RUNTIME})

        with pytest.raises(AuthError, match="No Codex credentials stored"):
            oneshot._run_agent(
                "hello", model="gpt-5.5-codex", provider="openai-codex"
            )

        assert calls == ["openai-codex"]  # fallback chain never consulted
        assert FakeAgent.instances == []

    def test_explicit_provider_with_env_model_propagates_auth_error(
        self, monkeypatch, oneshot_env
    ):
        # --provider with the model pinned via HERMES_INFERENCE_MODEL is the
        # same deliberate pair as --provider plus --model (it is the only way
        # --provider passes early validation without --model). No fallback
        # may be attempted; AuthError propagates.
        monkeypatch.setenv("HERMES_INFERENCE_MODEL", "gpt-5.5-codex")
        calls = _patch_resolver(monkeypatch, {"custom:nvidia": NVIDIA_RUNTIME})

        with pytest.raises(AuthError, match="No Codex credentials stored"):
            oneshot._run_agent("hello", provider="openai-codex")

        assert calls == ["openai-codex"]  # fallback chain never consulted
        assert FakeAgent.instances == []

    def test_env_provider_and_env_model_propagates_auth_error(
        self, monkeypatch, oneshot_env
    ):
        # HERMES_INFERENCE_PROVIDER + HERMES_INFERENCE_MODEL pin the pair just
        # as deliberately as --provider + --model. No fallback may be
        # attempted; AuthError propagates.
        monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openai-codex")
        monkeypatch.setenv("HERMES_INFERENCE_MODEL", "gpt-5.5-codex")
        calls = _patch_resolver(monkeypatch, {"custom:nvidia": NVIDIA_RUNTIME})

        with pytest.raises(AuthError, match="No Codex credentials stored"):
            oneshot._run_agent("hello")

        assert calls in ([None], ["openai-codex"])  # fallback never consulted
        assert FakeAgent.instances == []

    def test_fallback_passes_entry_model_as_target_model(
        self, monkeypatch, oneshot_env
    ):
        # api_mode is derived from target_model inside resolve_runtime_provider,
        # so each fallback attempt must pass the entry's OWN model — not the
        # primary's — and the resolved api_mode must reach the agent.
        seen_kwargs = []

        def fake_resolve(requested=None, **kwargs):
            seen_kwargs.append({"requested": requested, **kwargs})
            if not requested or requested in ("openai-codex", "codex"):
                raise _codex_auth_error()
            return dict(NVIDIA_RUNTIME)

        import hermes_cli.runtime_provider as rp_mod

        monkeypatch.setattr(rp_mod, "resolve_runtime_provider", fake_resolve)

        oneshot._run_agent("hello")

        assert seen_kwargs[0]["target_model"] == "gpt-5.5-codex"
        fallback_call = seen_kwargs[1]
        assert fallback_call["requested"] == "custom:nvidia"
        assert fallback_call["target_model"] == "moonshotai/kimi-k2.5"
        agent = FakeAgent.instances[-1]
        assert agent.kwargs["api_mode"] == "chat_completions"

    @pytest.mark.parametrize("exc_type", [ValueError, TypeError])
    def test_non_auth_error_inside_fallback_walk_propagates(
        self, monkeypatch, oneshot_env, exc_type
    ):
        # Only AuthError means "try the next entry". A ValueError/TypeError
        # raised while resolving a fallback is a real bug and must surface.
        def fake_resolve(requested=None, **kwargs):
            if not requested or requested in ("openai-codex", "codex"):
                raise _codex_auth_error()
            raise exc_type("fallback resolver blew up")

        import hermes_cli.runtime_provider as rp_mod

        monkeypatch.setattr(rp_mod, "resolve_runtime_provider", fake_resolve)

        with pytest.raises(exc_type, match="fallback resolver blew up"):
            oneshot._run_agent("hello")
        assert FakeAgent.instances == []

    def test_original_auth_error_identity_and_attributes_preserved(
        self, monkeypatch, oneshot_env
    ):
        # When the whole chain fails, the ORIGINAL AuthError object must
        # propagate — same instance, attributes intact, no wrapping.
        sentinel = _codex_auth_error()

        def fake_resolve(requested=None, **kwargs):
            if not requested or requested in ("openai-codex", "codex"):
                raise sentinel
            raise AuthError("no creds", provider=str(requested), code="missing")

        import hermes_cli.runtime_provider as rp_mod

        monkeypatch.setattr(rp_mod, "resolve_runtime_provider", fake_resolve)

        with pytest.raises(AuthError) as excinfo:
            oneshot._run_agent("hello")

        assert excinfo.value is sentinel
        assert excinfo.value.provider == "openai-codex"
        assert excinfo.value.code == "codex_auth_missing"
        assert excinfo.value.relogin_required is True

    def test_logs_never_contain_auth_error_message(
        self, monkeypatch, oneshot_env, caplog
    ):
        # AuthError messages can embed credential fragments (tokens echoed by
        # upstream 401 bodies). Only the structured provider/code fields may
        # be logged — never str(exc).
        secret = "sk-SUPER-SECRET-TOKEN-12345"

        def fake_resolve(requested=None, **kwargs):
            if not requested or requested in ("openai-codex", "codex"):
                raise AuthError(
                    f"401 from upstream: token {secret} rejected",
                    provider="openai-codex",
                    code="codex_auth_missing",
                )
            if requested == "custom:nvidia":
                raise AuthError(
                    f"key {secret} invalid for nvidia",
                    provider="custom:nvidia",
                    code="invalid_key",
                )
            return dict(NVIDIA2_RUNTIME)

        import hermes_cli.runtime_provider as rp_mod

        monkeypatch.setattr(rp_mod, "resolve_runtime_provider", fake_resolve)

        with caplog.at_level(logging.DEBUG):
            oneshot._run_agent("hello")

        assert secret not in caplog.text
        assert "codex_auth_missing" in caplog.text  # safe code IS logged
        assert "custom:nvidia2" in caplog.text  # chosen fallback IS logged


class TestOneshotBootstrapFallbackIntegration:
    """End-to-end through the real config stack: a real ``config.yaml`` inside
    the per-test temp HERMES_HOME (conftest's hermetic fixture), real
    ``load_config``, real ``get_fallback_chain`` and real
    ``resolve_runtime_provider``. Only ``AIAgent`` is faked (no network).
    """

    def test_codex_auth_failure_falls_back_using_real_config(self, monkeypatch):
        home = Path(os.environ["HERMES_HOME"])
        (home / "config.yaml").write_text(
            "\n".join(
                [
                    "model:",
                    "  provider: openai-codex",
                    "  model: gpt-5.5-codex",
                    "providers:",
                    "  nvidia:",
                    "    base_url: https://integrate.api.nvidia.com/v1",
                    "    api_key: nvapi-integration-test",
                    "fallback_providers:",
                    "  - provider: custom:nvidia",
                    "    model: moonshotai/kimi-k2.5",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        FakeAgent.instances = []
        import run_agent as run_agent_mod

        monkeypatch.setattr(run_agent_mod, "AIAgent", FakeAgent)

        # No Codex credentials exist in the temp HERMES_HOME, so the primary
        # (openai-codex) raises AuthError and the chain must kick in.
        response, _ = oneshot._run_agent("hello")

        assert response == "ok"
        agent = FakeAgent.instances[-1]
        assert agent.kwargs["provider"] == "custom"
        assert agent.kwargs["base_url"] == "https://integrate.api.nvidia.com/v1"
        assert agent.kwargs["api_key"] == "nvapi-integration-test"
        assert agent.kwargs["model"] == "moonshotai/kimi-k2.5"
