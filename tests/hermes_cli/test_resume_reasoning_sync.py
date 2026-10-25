"""The CLI's display effort follows the attached gateway session, not just the launch profile.

A resumed session can own a reasoning level (a ``/reasoning`` pin frozen into its policy). The
CLI's own re-resolve reads only ``config.yaml``, so before this sync the status bar and
``/reasoning`` reported the profile level while the next turn ran the session's. The gateway is
the owner of session state: ``session.info.reasoning_effort`` is adopted after the resolve.
"""

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.cli_model_switch_mixin import (
    CLIModelSwitchMixin,
    _sync_cli_reasoning_from_session,
)

_CFG = {
    "agent": {"reasoning_effort": "medium", "reasoning_overrides": {"glm-5.3-flash": "high"}},
    "model": {"provider": "custom:ark", "default": "deepseek-v4-flash"},
}
_MAX = {"enabled": True, "effort": "max"}


def _agent(info):
    return SimpleNamespace(
        _state=lambda: SimpleNamespace(info=info),
        switch_model=lambda **_kw: True,
    )


def _cli(model="deepseek-v4-flash", reasoning=None, agent=None):
    return SimpleNamespace(
        model=model, provider="custom:ark", requested_provider="custom:ark", api_key="k",
        base_url="https://ark.example/v3", api_mode="chat_completions",
        _explicit_api_key=None, _explicit_base_url=None, _explicit_model_override=False,
        _credential_pool=None, agent=agent,
        reasoning_config=reasoning or {"enabled": True, "effort": "high"},
        _reasoning_user_set=False,
        _console_print=lambda *_a, **_k: None,
    )


def test_sync_adopts_the_sessions_own_effort():
    cli = _cli(agent=_agent({"reasoning_effort": "max"}))
    _sync_cli_reasoning_from_session(cli)
    assert cli.reasoning_config == _MAX
    assert cli._reasoning_user_set is False


def test_sync_keeps_the_resolved_value_when_the_session_has_no_own_effort():
    cli = _cli(agent=_agent({}))
    before = dict(cli.reasoning_config)
    _sync_cli_reasoning_from_session(cli)
    assert cli.reasoning_config == before


def test_sync_is_a_no_op_without_an_agent():
    cli = _cli(agent=None)
    before = dict(cli.reasoning_config)
    _sync_cli_reasoning_from_session(cli)
    assert cli.reasoning_config == before


def test_resume_restores_the_sessions_pinned_effort_after_the_profile_resolve():
    import cli as cli_mod

    with patch.dict(cli_mod.CLI_CONFIG, _CFG):
        cli = _cli(reasoning={"enabled": True, "effort": "medium"},
                   agent=_agent({"reasoning_effort": "max"}))
        CLIModelSwitchMixin._restore_session_model(
            cli,
            {"model": "glm-5.3-flash", "model_config": {"gateway_runtime": {
                "provider": "custom:ark", "base_url": "https://ark.example/v3",
                "api_mode": "chat_completions"}}},
            quiet=True,
        )
        # The profile resolve for glm-5.3-flash would be high; the session's own max wins.
        assert cli.reasoning_config == _MAX


def test_apply_info_adopts_effort_without_writing_the_session():
    from hermes_cli.gateway_agent import GatewayAgent

    agent = GatewayAgent.__new__(GatewayAgent)
    agent._reasoning_config = {"enabled": True, "effort": "high"}
    calls = []
    agent.connection = SimpleNamespace(call=lambda *a, **k: calls.append((a, k)))
    agent.context_compressor = SimpleNamespace(context_length=0)

    agent._apply_info({"model": "m", "provider": "p", "context_length": 100,
                       "reasoning_effort": "max"})

    assert agent._reasoning_config == _MAX
    assert calls == []
