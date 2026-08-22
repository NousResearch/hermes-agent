"""Regression for #90089 Part 2 — provider/model fallback atomicity.

The reported manual-run path intermittently ignored a job's pinned
``zai/glm-4.5-air`` route when primary provider resolution failed. A fallback
must select one configured provider/model pair; preserving one pinned axis while
swapping the other creates a synthetic route that was never configured or
validated.

These tests assert both axes. Incompatible explicit pins fail closed, while
unpinned jobs and partially pinned jobs may use only a configured fallback route
that satisfies every explicit pin. Provider aliases are compared canonically.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import run_job


def _base_job(**overrides):
    job = {
        "id": "pin-90089",
        "name": "pin test 90089",
        "prompt": "hello",
        "model": None,
        "provider": None,
        "provider_snapshot": None,
        "model_snapshot": None,
        "base_url": None,
    }
    job.update(overrides)
    return job


def _run_with_fallback(
    job,
    *,
    primary_provider,
    primary_raises,
    fallback_provider,
    fallback_model,
    tmp_path,
    config_model="global-default-model",
):
    """Drive primary-resolution failure through the configured fallback route.

    Returns ``(success, output, error, model_used, provider_used)`` where the
    last two values are the exact pair passed to ``AIAgent``.
    """
    config_yaml = f"model:\n  default: {config_model}\n  provider: {primary_provider}\n"
    config_yaml += (
        "fallback_providers:\n"
        f"  - provider: {fallback_provider}\n"
        f"    model: {fallback_model}\n"
    )
    (tmp_path / "config.yaml").write_text(config_yaml)

    captured_agent_kwargs: dict = {}

    class _FakeAgent:
        def __init__(self, **kwargs):
            captured_agent_kwargs.update(kwargs)
            self.session_id = None

        def run_conversation(self, *a, **kw):
            return {"final_response": "ok"}

        def close(self):
            pass

    fake_db = MagicMock()

    call_count = [0]

    def _resolve(**kwargs):
        call_count[0] += 1
        # First call (primary resolution) always raises to trigger the
        # fallback chain. Subsequent calls (fallback entries) succeed.
        if call_count[0] == 1 and primary_raises:
            raise primary_raises
        return {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "provider": fallback_provider,
            "api_mode": "chat_completions",
        }

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._get_hermes_home", return_value=tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             side_effect=_resolve,
         ), \
         patch("run_agent.AIAgent", _FakeAgent):
        try:
            success, output, final_response, error = run_job(job)
        except Exception as exc:
            return (
                False,
                "",
                str(exc),
                captured_agent_kwargs.get("model", ""),
                captured_agent_kwargs.get("provider", ""),
            )

    return (
        success,
        output,
        error,
        captured_agent_kwargs.get("model", ""),
        captured_agent_kwargs.get("provider", ""),
    )


class TestProviderModelFallbackAtomicity:
    """Fallback may select only a route compatible with every explicit pin."""

    def test_incompatible_pinned_pair_fails_closed_on_auth_error(self, tmp_path):
        from hermes_cli.auth import AuthError

        job = _base_job(model="glm-4.5-air", provider="zai")
        success, _output, error, model_used, provider_used = _run_with_fallback(
            job,
            primary_provider="zai",
            primary_raises=AuthError("zai token expired"),
            fallback_provider="lmstudio",
            fallback_model="qwen3.8",
            tmp_path=tmp_path,
        )

        assert success is False
        assert "pinned route" in error
        assert model_used == ""
        assert provider_used == ""

    def test_unpinned_job_uses_the_configured_fallback_pair(self, tmp_path):
        from hermes_cli.auth import AuthError

        job = _base_job(
            model=None,
            provider=None,
            provider_snapshot=None,
            model_snapshot=None,
        )
        success, _output, error, model_used, provider_used = _run_with_fallback(
            job,
            primary_provider="zai",
            primary_raises=AuthError("zai token expired"),
            fallback_provider="lmstudio",
            fallback_model="qwen3.8",
            tmp_path=tmp_path,
        )

        assert success is True, error
        assert model_used == "qwen3.8"
        assert provider_used == "lmstudio"

    def test_model_only_pin_allows_a_matching_configured_route(self, tmp_path):
        from hermes_cli.auth import AuthError

        job = _base_job(model="shared-model", provider=None)
        success, _output, error, model_used, provider_used = _run_with_fallback(
            job,
            primary_provider="zai",
            primary_raises=AuthError("zai token expired"),
            fallback_provider="openrouter",
            fallback_model="shared-model",
            tmp_path=tmp_path,
        )

        assert success is True, error
        assert model_used == "shared-model"
        assert provider_used == "openrouter"

    def test_provider_only_pin_accepts_a_canonical_alias_match(self, tmp_path):
        from hermes_cli.auth import AuthError

        job = _base_job(model=None, provider="z-ai")
        success, _output, error, model_used, provider_used = _run_with_fallback(
            job,
            primary_provider="z-ai",
            primary_raises=AuthError("zai token expired"),
            fallback_provider="zai",
            fallback_model="glm-4.5-air",
            tmp_path=tmp_path,
        )

        assert success is True, error
        assert model_used == "glm-4.5-air"
        assert provider_used == "zai"

    def test_xai_oauth_pin_allows_same_service_api_key_route(
        self, tmp_path
    ):
        from hermes_cli.auth import AuthError

        job = _base_job(model="grok-4.5", provider="xai-oauth")
        success, _output, error, model_used, provider_used = (
            _run_with_fallback(
                job,
                primary_provider="xai-oauth",
                primary_raises=AuthError("xai oauth token expired"),
                fallback_provider="xai",
                fallback_model="grok-4.5",
                tmp_path=tmp_path,
            )
        )

        assert success is True, error
        assert model_used == "grok-4.5"
        assert provider_used == "xai"

    def test_incompatible_pinned_pair_fails_closed_on_network_error(
        self, tmp_path
    ):
        import httpx

        job = _base_job(model="glm-4.5-air", provider="zai")
        success, _output, error, model_used, provider_used = _run_with_fallback(
            job,
            primary_provider="zai",
            primary_raises=httpx.ConnectError("DNS resolution failed"),
            fallback_provider="lmstudio",
            fallback_model="qwen3.8",
            tmp_path=tmp_path,
        )

        assert success is False
        assert "pinned route" in error
        assert model_used == ""
        assert provider_used == ""
