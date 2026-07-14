"""Runtime consumers must not downgrade process-pin violations to defaults."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

import run_agent
from agent.context_compressor import ContextCompressor
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import config as config_module
from hermes_cli import managed_scope


def _pin_violation() -> config_module.PinnedEffectiveConfigError:
    return config_module.PinnedEffectiveConfigError("synthetic pinned drift")


def _activate_test_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    exact = {
        "model": {"default": "gpt-5.6-sol", "provider": "openai-codex"},
        "gateway": {"api_server": {"max_concurrent_runs": 1}},
    }
    raw = yaml.safe_dump(exact, sort_keys=True).encode("utf-8")
    config_path.write_bytes(raw)
    monkeypatch.setattr(config_module, "_PINNED_EFFECTIVE_CONFIG", None)
    monkeypatch.setattr(config_module, "get_config_path", lambda: config_path)
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: None)
    config_module.pin_effective_config_projection(
        config_path=config_path,
        raw_bytes=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        exact_mapping=exact,
    )


def _install_pinned_filesystem_fault(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    if operation == "close":
        real_close = config_module.os.close

        def close_then_fail(fd: int) -> None:
            real_close(fd)
            raise OSError("synthetic close failure")

        monkeypatch.setattr(config_module.os, "close", close_then_fail)
        return

    def fail_operation(*_args, **_kwargs):
        raise OSError(f"synthetic {operation} failure")

    monkeypatch.setattr(config_module.os, operation, fail_operation)


def test_agent_init_propagates_pinned_effective_config_violation() -> None:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", side_effect=_pin_violation()),
        pytest.raises(
            config_module.PinnedEffectiveConfigError,
            match="synthetic pinned drift",
        ),
    ):
        run_agent.AIAgent(
            model="gpt-5.6-sol",
            provider="openai-codex",
            api_mode="codex_responses",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="test-token",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def test_agent_init_stops_on_one_shot_custom_provider_pin_violation() -> None:
    load_results = iter([{}, _pin_violation(), {}])

    def one_shot_violation():
        result = next(load_results)
        if isinstance(result, BaseException):
            raise result
        return result

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch.object(
            run_agent.AIAgent,
            "_apply_user_default_headers",
            return_value=None,
        ),
        patch("hermes_cli.config.load_config", side_effect=one_shot_violation),
        pytest.raises(
            config_module.PinnedEffectiveConfigError,
            match="synthetic pinned drift",
        ),
    ):
        run_agent.AIAgent(
            model="gpt-5.6-sol",
            provider="openai-codex",
            api_mode="codex_responses",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="test-token",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def test_api_run_cap_does_not_widen_after_pinned_config_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: (_ for _ in ()).throw(_pin_violation()),
    )

    with pytest.raises(
        config_module.PinnedEffectiveConfigError,
        match="synthetic pinned drift",
    ):
        APIServerAdapter._resolve_max_concurrent_runs()


@pytest.mark.parametrize(
    ("operation", "reported_action"),
    [
        ("lstat", "path inspection"),
        ("open", "open"),
        ("fstat", "fstat"),
        ("read", "read"),
        ("close", "close"),
    ],
)
def test_pinned_filesystem_fault_cannot_downgrade_api_or_agent_config(
    operation: str,
    reported_action: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _activate_test_pin(tmp_path, monkeypatch)
    _install_pinned_filesystem_fault(monkeypatch, operation)

    expected = {
        "model": {"default": "gpt-5.6-sol", "provider": "openai-codex"},
        "gateway": {"api_server": {"max_concurrent_runs": 1}},
    }
    # Semantic readers never touch the filesystem once pinned, so a transient
    # fault cannot be swallowed into defaults or a wider concurrency policy.
    assert config_module.load_config() == expected
    assert config_module.load_config_readonly() == expected
    assert config_module.read_raw_config() == expected
    assert APIServerAdapter._resolve_max_concurrent_runs() == 1

    with pytest.raises(
        config_module.PinnedEffectiveConfigError,
        match=f"filesystem {reported_action} failed",
    ):
        config_module.attest_pinned_effective_config_projection()

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        pytest.raises(
            config_module.PinnedEffectiveConfigError,
            match=f"filesystem {reported_action} failed",
        ),
    ):
        run_agent.AIAgent(
            model="gpt-5.6-sol",
            provider="openai-codex",
            api_mode="codex_responses",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="test-token",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def test_pinned_read_and_close_failures_are_both_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _activate_test_pin(tmp_path, monkeypatch)
    real_close = config_module.os.close

    def fail_read(*_args, **_kwargs):
        raise OSError("synthetic read failure")

    def close_then_fail(fd: int) -> None:
        real_close(fd)
        raise OSError("synthetic close failure")

    monkeypatch.setattr(config_module.os, "read", fail_read)
    monkeypatch.setattr(config_module.os, "close", close_then_fail)

    with pytest.raises(
        config_module.PinnedEffectiveConfigError,
        match="read and cleanup failed",
    ) as exc_info:
        config_module.attest_pinned_effective_config_projection()

    combined = exc_info.value.__cause__
    assert isinstance(combined, ExceptionGroup)
    assert [type(item) for item in combined.exceptions] == [
        config_module.PinnedEffectiveConfigError,
        config_module.PinnedEffectiveConfigError,
    ]
    assert "filesystem read failed" in str(combined.exceptions[0])
    assert "filesystem close failed" in str(combined.exceptions[1])


def _make_test_agent() -> run_agent.AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = run_agent.AIAgent(
            model="test-model",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-token",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "Stable system prompt"
    agent.compression_enabled = False
    return agent


def test_agent_turn_rechecks_pin_before_prologue() -> None:
    agent = _make_test_agent()

    with (
        patch(
            "hermes_cli.config.attest_pinned_effective_config_projection",
            side_effect=_pin_violation(),
        ),
        patch("agent.conversation_loop.build_turn_context") as build_context,
        pytest.raises(
            config_module.PinnedEffectiveConfigError,
            match="synthetic pinned drift",
        ),
    ):
        agent.run_conversation("must not enter the turn prologue")

    build_context.assert_not_called()


def test_agent_loop_rechecks_pin_before_every_model_call() -> None:
    agent = _make_test_agent()
    agent.client = MagicMock()

    with (
        patch(
            "hermes_cli.config.attest_pinned_effective_config_projection",
            side_effect=[None, None, _pin_violation()],
        ),
        pytest.raises(
            config_module.PinnedEffectiveConfigError,
            match="synthetic pinned drift",
        ),
    ):
        agent.run_conversation("must not reach the provider")

    agent.client.chat.completions.create.assert_not_called()


def test_codex_app_server_rechecks_pin_at_dispatch_boundary() -> None:
    agent = _make_test_agent()
    agent.api_mode = "codex_app_server"
    agent._run_codex_app_server_turn = MagicMock()

    with (
        patch(
            "hermes_cli.config.attest_pinned_effective_config_projection",
            side_effect=[None, _pin_violation()],
        ),
        pytest.raises(
            config_module.PinnedEffectiveConfigError,
            match="synthetic pinned drift",
        ),
    ):
        agent.run_conversation("must not reach the app-server provider")

    agent._run_codex_app_server_turn.assert_not_called()


def _configure_codex_iteration_summary_agent(
    agent: run_agent.AIAgent,
) -> None:
    agent.api_mode = "codex_responses"
    agent.provider = "openai-codex"
    agent.model = "gpt-5.6-sol"
    agent._build_api_kwargs = MagicMock(return_value={"input": []})
    agent._run_codex_stream = MagicMock(return_value=object())
    transport = MagicMock()
    transport.normalize_response.return_value.content = ""
    agent._get_transport = MagicMock(return_value=transport)


def test_iteration_summary_rechecks_pin_before_first_provider_call() -> None:
    agent = _make_test_agent()
    _configure_codex_iteration_summary_agent(agent)

    with (
        patch(
            "hermes_cli.config.attest_pinned_effective_config_projection",
            side_effect=_pin_violation(),
        ),
        pytest.raises(
            config_module.PinnedEffectiveConfigError,
            match="synthetic pinned drift",
        ),
    ):
        agent._handle_max_iterations(
            [{"role": "user", "content": "finish the bounded plan"}],
            90,
        )

    agent._run_codex_stream.assert_not_called()


def test_iteration_summary_rechecks_pin_before_retry_provider_call() -> None:
    agent = _make_test_agent()
    _configure_codex_iteration_summary_agent(agent)

    with (
        patch(
            "hermes_cli.config.attest_pinned_effective_config_projection",
            side_effect=[None, _pin_violation()],
        ),
        pytest.raises(
            config_module.PinnedEffectiveConfigError,
            match="synthetic pinned drift",
        ),
    ):
        agent._handle_max_iterations(
            [{"role": "user", "content": "finish the bounded plan"}],
            90,
        )

    agent._run_codex_stream.assert_called_once()


def test_context_compression_rechecks_pin_before_compressor() -> None:
    agent = MagicMock()
    messages = [{"role": "user", "content": "large context"}]

    with (
        patch(
            "hermes_cli.config.attest_pinned_effective_config_projection",
            side_effect=_pin_violation(),
        ),
        pytest.raises(
            config_module.PinnedEffectiveConfigError,
            match="synthetic pinned drift",
        ),
    ):
        run_agent.AIAgent._compress_context(
            agent,
            messages,
            "Stable system prompt",
        )

    agent.context_compressor.compress.assert_not_called()


def test_post_tool_compression_propagates_pinned_config_violation() -> None:
    agent = _make_test_agent()
    tool_call = SimpleNamespace(
        id="call_pinned_compression",
        type="function",
        function=SimpleNamespace(name="execute_code", arguments="{}"),
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[tool_call]),
                finish_reason="tool_calls",
            )
        ],
        model="test-model",
        usage=None,
    )
    agent.client.chat.completions.create.return_value = response
    agent.valid_tool_names.add("execute_code")

    def enable_post_tool_compression(*_args, **_kwargs) -> None:
        agent.compression_enabled = True

    agent._execute_tool_calls = MagicMock(
        side_effect=enable_post_tool_compression
    )
    agent.compression_enabled = False
    agent.context_compressor.last_prompt_tokens = 1_000
    agent.context_compressor.should_compress = MagicMock(return_value=True)
    agent._compress_context = MagicMock(side_effect=_pin_violation())

    with (
        patch(
            "hermes_cli.config.attest_pinned_effective_config_projection",
            return_value=None,
        ),
        pytest.raises(
            config_module.PinnedEffectiveConfigError,
            match="synthetic pinned drift",
        ),
    ):
        agent.run_conversation("execute one tool, then compact")

    agent._execute_tool_calls.assert_called_once()
    agent._compress_context.assert_called_once()


def test_compression_rechecks_pin_immediately_before_auxiliary_provider() -> None:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        compressor = ContextCompressor(model="test-model", quiet_mode=True)

    with (
        patch(
            "hermes_cli.config.attest_pinned_effective_config_projection",
            side_effect=_pin_violation(),
        ),
        patch("agent.context_compressor.call_llm") as call_llm,
        pytest.raises(
            config_module.PinnedEffectiveConfigError,
            match="synthetic pinned drift",
        ),
    ):
        compressor._generate_summary(
            [
                {"role": "user", "content": "large context"},
                {"role": "assistant", "content": "working"},
            ]
        )

    call_llm.assert_not_called()
