"""Provider-routing regressions for the standalone batch runner."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import batch_runner
import pytest


def test_batch_cli_has_no_openrouter_or_model_default():
    signature = inspect.signature(batch_runner.main)

    assert signature.parameters["base_url"].default is None
    assert signature.parameters["model"].default is None


def test_explicit_api_key_and_base_url_take_precedence(monkeypatch):
    def unexpected_resolution(**_kwargs):
        raise AssertionError("a complete explicit runtime must not consult config")

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        unexpected_resolution,
    )

    runtime = batch_runner._resolve_batch_runtime(
        {
            "model": "gpt-5.5",
            "provider": None,
            "api_mode": None,
            "api_key": "sk-explicit",
            "base_url": "https://gateway.example/v1",
        }
    )

    assert runtime == {
        "model": "gpt-5.5",
        "provider": None,
        "api_mode": None,
        "api_key": "sk-explicit",
        "base_url": "https://gateway.example/v1",
        "credential_pool": None,
        "source": "command-line",
    }


def test_missing_runtime_fields_inherit_configured_codex(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": {
                "provider": "openai-codex",
                "default": "gpt-5.5",
                "base_url": "https://chatgpt.com/backend-api/codex",
            }
        },
    )

    def resolve_runtime_provider(**kwargs):
        calls.append(kwargs)
        return {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "api_key": "jwt-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "credential_pool": "pool",
            "source": "hermes-auth-store",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        resolve_runtime_provider,
    )

    runtime = batch_runner._resolve_batch_runtime(
        {
            "model": None,
            "provider": None,
            "api_mode": None,
            "api_key": None,
            "base_url": None,
        }
    )

    assert calls == [
        {
            "requested": None,
            "explicit_api_key": None,
            "explicit_base_url": None,
            "target_model": "gpt-5.5",
        }
    ]
    assert runtime["model"] == "gpt-5.5"
    assert runtime["provider"] == "openai-codex"
    assert runtime["api_mode"] == "codex_responses"
    assert runtime["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert runtime["api_key"] == "jwt-token"
    assert runtime["credential_pool"] == "pool"


def test_real_runtime_resolution_inherits_codex_from_temp_hermes_home(
    tmp_path,
    monkeypatch,
):
    """Exercise the real config, auth store, and provider resolver together."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  provider: openai-codex\n"
        "  default: gpt-5.5\n"
        "  base_url: https://chatgpt.com/backend-api/codex\n",
        encoding="utf-8",
    )
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "active_provider": "openai-codex",
                "providers": {
                    "openai-codex": {
                        "tokens": {
                            "access_token": "test-codex-access-token",
                            "refresh_token": "test-codex-refresh-token",
                        },
                        "last_refresh": "2026-07-23T00:00:00Z",
                        "auth_mode": "chatgpt",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex-home"))

    runtime = batch_runner._resolve_batch_runtime(
        {
            "model": None,
            "provider": None,
            "api_mode": None,
            "api_key": None,
            "base_url": None,
        }
    )

    assert runtime["model"] == "gpt-5.5"
    assert runtime["provider"] == "openai-codex"
    assert runtime["api_mode"] == "codex_responses"
    assert runtime["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert runtime["api_key"] == "test-codex-access-token"


def test_partial_explicit_runtime_uses_configured_provider(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"provider": "openai", "default": "gpt-5.5"}},
    )
    captured = {}

    def resolve_runtime_provider(**kwargs):
        captured.update(kwargs)
        return {
            "provider": "openai",
            "api_mode": "codex_responses",
            "api_key": kwargs["explicit_api_key"],
            "base_url": "https://api.openai.com/v1",
            "source": "explicit",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        resolve_runtime_provider,
    )

    runtime = batch_runner._resolve_batch_runtime(
        {
            "model": None,
            "provider": None,
            "api_mode": None,
            "api_key": "sk-explicit",
            "base_url": None,
        }
    )

    assert captured["requested"] == "openai"
    assert captured["explicit_api_key"] == "sk-explicit"
    assert runtime["base_url"] == "https://api.openai.com/v1"


def test_explicit_base_url_takes_precedence_over_configured_provider(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"provider": "openai-codex", "default": "gpt-5.5"}},
    )
    captured = {}

    def resolve_runtime_provider(**kwargs):
        captured.update(kwargs)
        return {
            "provider": "custom",
            "api_mode": "chat_completions",
            "api_key": "openrouter-key",
            "base_url": kwargs["explicit_base_url"],
            "source": "explicit",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        resolve_runtime_provider,
    )

    runtime = batch_runner._resolve_batch_runtime(
        {
            "model": None,
            "provider": None,
            "api_mode": None,
            "api_key": None,
            "base_url": "https://openrouter.ai/api/v1",
        }
    )

    assert captured["requested"] == "custom"
    assert captured["explicit_base_url"] == "https://openrouter.ai/api/v1"
    assert runtime["api_key"] == "openrouter-key"
    assert runtime["api_mode"] == "chat_completions"


def test_real_explicit_openrouter_url_does_not_load_configured_codex_auth(
    tmp_path,
    monkeypatch,
):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  provider: openai-codex\n"
        "  default: gpt-5.5\n"
        "  base_url: https://chatgpt.com/backend-api/codex\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex-home"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    runtime = batch_runner._resolve_batch_runtime(
        {
            "model": "anthropic/claude-sonnet-4.6",
            "provider": None,
            "api_mode": None,
            "api_key": None,
            "base_url": "https://openrouter.ai/api/v1",
        }
    )

    assert runtime["model"] == "anthropic/claude-sonnet-4.6"
    assert runtime["provider"] == "custom"
    assert runtime["api_mode"] == "chat_completions"
    assert runtime["base_url"] == "https://openrouter.ai/api/v1"
    assert runtime["api_key"] == "test-openrouter-key"


def test_api_key_alone_without_provider_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "gpt-5.5"}},
    )

    with pytest.raises(ValueError, match="--api-key without --base-url"):
        batch_runner._resolve_batch_runtime(
            {
                "model": None,
                "provider": None,
                "api_mode": None,
                "api_key": "sk-opaque",
                "base_url": None,
            }
        )


def test_process_prompt_passes_resolved_provider_transport_and_pool(monkeypatch):
    captured = {}
    pool = object()

    monkeypatch.setattr(
        batch_runner,
        "_resolve_batch_runtime",
        lambda _config: {
            "model": "gpt-5.5",
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "api_key": "jwt-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "credential_pool": pool,
            "source": "hermes-auth-store",
        },
    )
    monkeypatch.setattr(batch_runner, "sample_toolsets_from_distribution", lambda _name: [])

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_conversation(self, _prompt, task_id=None):
            assert task_id == "task_0"
            return {
                "messages": [],
                "completed": True,
                "partial": False,
                "api_calls": 1,
            }

        def _convert_to_trajectory_format(self, _messages, _prompt, _completed):
            return []

    monkeypatch.setattr(batch_runner, "AIAgent", FakeAgent)

    result = batch_runner._process_single_prompt(
        0,
        {"prompt": "hello"},
        0,
        {"distribution": "default", "max_iterations": 2},
    )

    assert result["success"] is True
    assert captured["model"] == "gpt-5.5"
    assert captured["provider"] == "openai-codex"
    assert captured["api_mode"] == "codex_responses"
    assert captured["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert captured["api_key"] == "jwt-token"
    assert captured["credential_pool"] is pool


def test_worker_resolves_runtime_once_and_reuses_it(monkeypatch, tmp_path):
    resolutions = []
    runtimes_seen = []
    runtime = {
        "model": "gpt-5.5",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "api_key": "jwt-token",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "credential_pool": object(),
        "source": "hermes-auth-store",
    }
    config = {"distribution": "default", "max_iterations": 2}

    monkeypatch.setattr(batch_runner, "_WORKER_CONFIG", {})
    monkeypatch.setattr(batch_runner, "_WORKER_RUNTIME", None)

    def resolve_batch_runtime(received):
        resolutions.append(received)
        return runtime

    def process_single_prompt(
        prompt_index,
        _prompt_data,
        _batch_num,
        _config,
        resolved_runtime=None,
    ):
        runtimes_seen.append(resolved_runtime)
        return {
            "success": False,
            "prompt_index": prompt_index,
            "error": "test stop",
            "trajectory": None,
            "tool_stats": {},
            "reasoning_stats": {},
            "toolsets_used": [],
        }

    monkeypatch.setattr(batch_runner, "_resolve_batch_runtime", resolve_batch_runtime)
    monkeypatch.setattr(batch_runner, "_process_single_prompt", process_single_prompt)

    batch_runner._initialize_batch_worker(config)
    for batch_num in (0, 1):
        result = batch_runner._process_batch_worker(
            (
                batch_num,
                [(batch_num, {"prompt": f"prompt {batch_num}"})],
                str(tmp_path),
                set(),
                config,
            )
        )
        assert result["model"] == "gpt-5.5"

    assert resolutions == [config]
    assert runtimes_seen == [runtime, runtime]


def test_statistics_model_prefers_effective_worker_runtime(monkeypatch):
    monkeypatch.setattr(
        batch_runner,
        "_configured_batch_model_and_provider",
        lambda: ("configured-model", "openai-codex"),
    )

    assert (
        batch_runner._effective_model_for_statistics(
            [{"model": "gpt-5.5"}],
            requested_model=None,
        )
        == "gpt-5.5"
    )
    assert (
        batch_runner._effective_model_for_statistics([], requested_model=None)
        == "configured-model"
    )


def test_agent_banner_reports_effective_client_url(monkeypatch, capsys):
    from run_agent import AIAgent

    routed_client = SimpleNamespace(
        api_key="jwt-token-value",
        base_url="https://chatgpt.com/backend-api/codex/",
        _custom_headers={},
        default_headers={},
        _default_headers={},
    )
    monkeypatch.setattr(
        "agent.auxiliary_client.resolve_provider_client",
        lambda *_args, **_kwargs: (routed_client, "gpt-5.5"),
    )
    monkeypatch.setattr("hermes_logging.setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("run_agent.OpenAI", lambda **kwargs: SimpleNamespace(**kwargs))

    AIAgent(
        base_url="https://stale.example/v1",
        api_key=None,
        model="gpt-5.5",
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )

    output = capsys.readouterr().out
    assert "Using effective base URL: https://chatgpt.com/backend-api/codex/" in output
    assert "Using effective base URL: https://stale.example/v1" not in output
