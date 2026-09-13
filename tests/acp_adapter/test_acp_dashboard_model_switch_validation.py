"""ACP ``session/set_model`` and the dashboard main slot validate through ``switch_model``.

Both surfaces used to accept any string (``parse_model_input`` + ``detect_provider_for_model``
for ACP; bare provider/model normalization for ``POST /api/model/set``), so a model no catalog
knew — or a provider with no credentials — was handed to the session / written to config.yaml
and only failed at inference time. They now share the CLI/gateway/TUI ``/model`` pipeline: a
rejection from ``switch_model`` is a rejection on these surfaces too, and an acceptance carries
the resolved (provider, model) — an explicit ``provider:model`` prefix is honoured as
``--provider`` (#59089), never re-detected.
"""

from __future__ import annotations

import types

import pytest

from hermes_cli.model_switch import ModelSwitchResult


def _acp_agent():
    from acp_adapter.server import HermesACPAgent
    made: dict = {}

    class _SM:
        def _make_agent(self, **kw):
            made.update(kw)
            return types.SimpleNamespace(provider=kw.get("requested_provider"), model=kw.get("model"))

        def save_session(self, sid):
            pass

    return HermesACPAgent(session_manager=_SM()), made


def _state():
    return types.SimpleNamespace(
        session_id="s1", cwd=".", model="claude-sonnet-5",
        agent=types.SimpleNamespace(provider="anthropic", base_url="https://api.anthropic.com", api_key="k"))


def test_acp_and_dashboard_reject_what_switch_model_rejects(monkeypatch):
    rejected = ModelSwitchResult(success=False, error_message="Unknown provider 'notaprovider'.")
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_kw: rejected)

    agent, made = _acp_agent()
    state = _state()
    with pytest.raises(ValueError, match="Unknown provider"):
        agent._switch_model(state, "notaprovider:whatever")
    assert made == {} and state.model == "claude-sonnet-5"  # session untouched

    from fastapi import HTTPException
    from hermes_cli.web_server_config import _apply_model_assignment_sync
    with pytest.raises(HTTPException) as exc:
        _apply_model_assignment_sync("main", "notaprovider", "whatever", "", "")
    assert exc.value.status_code == 400 and "Unknown provider" in exc.value.detail


def test_acp_explicit_provider_prefix_becomes_explicit_provider(monkeypatch):
    seen: dict = {}

    def _switch(**kw):
        seen.update(kw)
        return ModelSwitchResult(success=True, new_model=kw["raw_input"], target_provider=kw["explicit_provider"])

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _switch)
    agent, made = _acp_agent()
    old, new_provider, model = agent._switch_model(_state(), "anthropic:claude-sonnet-5", keep_endpoint=True)
    assert (seen["explicit_provider"], seen["raw_input"]) == ("anthropic", "claude-sonnet-5")
    assert (old, new_provider, model) == ("anthropic", "anthropic", "claude-sonnet-5")
    assert made["requested_provider"] == "anthropic" and made["base_url"] == "https://api.anthropic.com"
