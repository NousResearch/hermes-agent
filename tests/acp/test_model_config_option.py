"""Regression tests for #105383: ACP model picker missing in Zed 1.18+.

``session/new`` must advertise the model selector BOTH as the v1.3.0
``configOptions`` (``SessionConfigOptionSelect``) AND the legacy ``models``
(``SessionModelState``); ``session/set_config_option`` with id ``model``
must switch the session model like the legacy ``session/set_model``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from acp.schema import SessionConfigOptionSelect, SessionModelState
from acp_adapter.model_catalog import ACP_MAX_MODELS_PER_PROVIDER
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


def _manager(**agent_kwargs):
    defaults = {
        "model": "gpt-5.4",
        "provider": "openai-codex",
        "base_url": "https://api.openai.com/v1",
    }
    defaults.update(agent_kwargs)
    return SessionManager(agent_factory=lambda: SimpleNamespace(**defaults))


def _payload():
    return {
        "providers": [
            {"slug": "anthropic", "name": "Anthropic", "models": ["claude-sonnet-4-6"]},
            {"slug": "openai-codex", "name": "OpenAI Codex", "models": [{"id": "gpt-5.4"}]},
        ],
    }


def _patched_inventory(payload):
    picker_context = MagicMock()
    picker_context.with_overrides.return_value = picker_context
    return (
        patch("hermes_cli.inventory.load_picker_context", return_value=picker_context),
        patch("hermes_cli.inventory.build_models_payload", return_value=payload),
    )


@pytest.mark.asyncio
async def test_new_session_advertises_model_config_option_alongside_legacy():
    agent = HermesACPAgent(session_manager=_manager())
    ctx_patch, payload_patch = _patched_inventory(_payload())
    with ctx_patch, payload_patch:
        resp = await agent.new_session(cwd="/tmp")
        forked = await agent.fork_session(cwd="/tmp", session_id=resp.session_id)

    for response in (resp, forked):
        # Legacy field preserved for older clients.
        assert isinstance(response.models, SessionModelState)
        assert response.models.current_model_id == "openai-codex:gpt-5.4"
        # v1.3.0 shape for Zed 1.18+: same selector, same ids.
        assert isinstance(response.config_options, list) and len(response.config_options) == 1
        option = response.config_options[0]
        assert isinstance(option, SessionConfigOptionSelect)
        assert option.id == "model"
        assert option.type == "select"
        assert option.current_value == response.models.current_model_id
        assert [item.value for item in option.options] == [
            model.model_id for model in response.models.available_models
        ]


@pytest.mark.asyncio
async def test_set_config_option_model_switches_session_and_persists():
    agent = HermesACPAgent(session_manager=_manager())
    ctx_patch, payload_patch = _patched_inventory(_payload())
    with ctx_patch, payload_patch:
        resp = await agent.new_session(cwd="/tmp")
        before = resp.config_options[0].current_value
        update = await agent.set_config_option(
            "model", resp.session_id, "anthropic:claude-sonnet-4-6",
        )
        resumed = await agent.resume_session(cwd="/tmp", session_id=resp.session_id)

    state = agent.session_manager.get_session(resp.session_id)
    assert state.model == "claude-sonnet-4-6"
    assert update.config_options[0].current_value != before
    assert update.config_options[0].current_value.endswith("claude-sonnet-4-6")
    assert update.config_options[0].current_value in [
        item.value for item in update.config_options[0].options
    ]
    # Switch survives resume; legacy field agrees with the new shape.
    assert resumed.config_options[0].current_value == update.config_options[0].current_value
    assert resumed.models.current_model_id == update.config_options[0].current_value
