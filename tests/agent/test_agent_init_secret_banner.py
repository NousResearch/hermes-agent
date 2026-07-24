"""Credential banners must not expose raw API-key/token slices on stdout.

``print()`` bypasses ``RedactingFormatter`` (logging-only), so these tests call
``init_agent`` itself and verify the two client-construction paths that emit a
credential banner.
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from agent.redact import mask_secret

_SECRET = "credential-ABCDEFGH-middle-SECRET99"


def _new_agent():
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._base_url = ""
    agent._base_url_lower = ""
    agent._base_url_hostname = ""
    return agent


def _common_init_patches(stack: ExitStack) -> None:
    stack.enter_context(
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None))
    )
    stack.enter_context(patch("run_agent.get_tool_definitions", return_value=[]))
    stack.enter_context(
        patch("agent.azure_identity_adapter.is_token_provider", return_value=False)
    )
    stack.enter_context(
        patch(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            return_value="test-model",
        )
    )
    stack.enter_context(patch("agent.credential_pool.load_pool", return_value=MagicMock()))
    stack.enter_context(patch("hermes_cli.config.load_config", return_value={}))
    stack.enter_context(
        patch("hermes_cli.config.get_compatible_custom_providers", return_value=[])
    )
    stack.enter_context(patch("agent.iteration_budget.IterationBudget"))
    stack.enter_context(patch("hermes_cli.config.cfg_get", return_value=None))


def _assert_secret_is_masked(output: str, banner: str) -> None:
    assert banner in output
    assert mask_secret(_SECRET) in output
    assert _SECRET not in output
    assert _SECRET[:8] not in output
    assert _SECRET[-8:] not in output
    assert f"{_SECRET[:8]}...{_SECRET[-4:]}" not in output


def test_anthropic_init_banner_masks_token(capsys: pytest.CaptureFixture[str]) -> None:
    from agent.agent_init import init_agent

    agent = _new_agent()
    with ExitStack() as stack:
        _common_init_patches(stack)
        stack.enter_context(
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="")
        )
        stack.enter_context(
            patch("agent.anthropic_adapter._is_oauth_token", return_value=False)
        )
        init_agent(
            agent,
            base_url="https://api.anthropic.com",
            api_key=_SECRET,
            provider="anthropic",
            model="test-model",
            skip_context_files=True,
            skip_memory=True,
            quiet_mode=False,
        )

    _assert_secret_is_masked(capsys.readouterr().out, "Using token:")


def test_openai_wire_init_banner_masks_api_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent.agent_init import init_agent

    agent = _new_agent()
    with ExitStack() as stack:
        _common_init_patches(stack)
        stack.enter_context(patch("run_agent.OpenAI", return_value=MagicMock()))
        init_agent(
            agent,
            base_url="https://api.openai.com/v1",
            api_key=_SECRET,
            provider="openai",
            model="test-model",
            skip_context_files=True,
            skip_memory=True,
            quiet_mode=False,
        )

    _assert_secret_is_masked(capsys.readouterr().out, "Using API key:")
