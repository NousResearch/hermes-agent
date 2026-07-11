"""Provider-neutral runtime routing contracts for scheduled jobs."""

from unittest.mock import MagicMock, patch

from hermes_cli.auth import AuthError


def test_cron_auth_fallback_uses_fallback_runtime_and_model(tmp_path):
    from cron.scheduler import run_job

    primary = {
        "default": "gpt-5.4",
        "provider": "openai-codex",
        "runtime": "codex_app_server",
    }
    fallback = {
        "provider": "anthropic",
        "model": "claude-opus-4-6",
        "runtime": "claude_agent_sdk",
    }
    (tmp_path / "config.yaml").write_text(
        "model:\n"
        "  default: gpt-5.4\n"
        "  provider: openai-codex\n"
        "  runtime: codex_app_server\n"
        "fallback_providers:\n"
        "  - provider: anthropic\n"
        "    model: claude-opus-4-6\n"
        "    runtime: claude_agent_sdk\n",
        encoding="utf-8",
    )
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise AuthError("Codex subscription unavailable", provider="openai-codex")
        return {
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "api_mode": "anthropic_messages",
            "runtime": "claude_agent_sdk",
            "base_url": "",
            "api_key": "",
            "command": None,
            "args": [],
        }

    fake_db = MagicMock()
    fake_agent = MagicMock()
    fake_agent.run_conversation.return_value = {"final_response": "ok"}

    with (
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB", return_value=fake_db),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=resolve,
        ),
        patch("run_agent.AIAgent", return_value=fake_agent) as agent_cls,
    ):
        success, _output, final_response, error = run_job(
            {"id": "claude-fallback", "name": "test", "prompt": "hello"}
        )

    assert success is True
    assert final_response == "ok"
    assert error is None
    assert calls == [
        {
            "requested": "openai-codex",
            "target_model": "gpt-5.4",
            "route_config": primary,
        },
        {
            "requested": "anthropic",
            "target_model": "claude-opus-4-6",
            "route_config": fallback,
        },
    ]
    assert agent_cls.call_args.kwargs["model"] == "claude-opus-4-6"
    assert agent_cls.call_args.kwargs["runtime"] == "claude_agent_sdk"
    assert agent_cls.call_args.kwargs["fallback_model"] == [fallback]


def test_cron_explicit_alternate_provider_does_not_inherit_claude_runtime(
    tmp_path,
):
    from cron.scheduler import run_job

    (tmp_path / "config.yaml").write_text(
        "model:\n"
        "  default: claude-opus-4-6\n"
        "  provider: anthropic\n"
        "  runtime: claude_agent_sdk\n",
        encoding="utf-8",
    )
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        return {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "runtime": "hermes",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "subscription-token",
            "command": None,
            "args": [],
        }

    fake_agent = MagicMock()
    fake_agent.run_conversation.return_value = {"final_response": "ok"}
    with (
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB", return_value=MagicMock()),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=resolve,
        ),
        patch("run_agent.AIAgent", return_value=fake_agent),
    ):
        success, _output, _final_response, error = run_job(
            {
                "id": "codex-explicit",
                "name": "test",
                "prompt": "hello",
                "provider": "openai-codex",
                "model": "gpt-5.4",
            }
        )

    assert success is True
    assert error is None
    assert calls == [
        {
            "requested": "openai-codex",
            "target_model": "gpt-5.4",
            "route_config": {
                "provider": "openai-codex",
                "model": "gpt-5.4",
            },
        }
    ]
