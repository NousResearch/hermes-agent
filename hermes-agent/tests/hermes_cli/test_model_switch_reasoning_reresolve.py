"""The CLI-level ``reasoning_config`` follows the model on every path that moves ``cli.model``.

Startup resolves the effort once for the launch model; a lazily built agent (first message after
``/model``, ``--resume``, ``/new``) inherits ``cli.reasoning_config`` verbatim. Leaving it stale sends
the launch model's effort to the new model — always-thinking models that accept only their own level
set reject that with a non-retryable HTTP 400 (#112921, #96012).
"""

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin

_CFG = {
    "agent": {"reasoning_effort": "medium", "reasoning_overrides": {"glm-5.3-flash": "high"}},
    "model": {"provider": "custom:ark", "default": "deepseek-v4-flash"},
}
_MEDIUM = {"enabled": True, "effort": "medium"}
_HIGH = {"enabled": True, "effort": "high"}


def _cli(model, reasoning_config, agent=None):
    return SimpleNamespace(
        model=model, provider="custom:ark", requested_provider="custom:ark", api_key="k",
        base_url="https://ark.example/v3", api_mode="chat_completions", _explicit_api_key=None,
        _explicit_base_url=None, agent=agent, _explicit_model_override=False, _credential_pool=None,
        reasoning_config=reasoning_config, _console_print=lambda *_a, **_k: None)


def _result(model):
    return SimpleNamespace(new_model=model, target_provider="custom:ark", api_key="k",
                           base_url="https://ark.example/v3", api_mode="chat_completions", success=True)


def test_switch_and_session_restore_re_resolve_effort_before_the_agent_exists():
    import cli as cli_mod

    with patch.dict(cli_mod.CLI_CONFIG, _CFG):
        # /model glm-5.3-flash before the first message: per-model override wins.
        cli = _cli("deepseek-v4-flash", _MEDIUM)
        assert CLIModelSwitchMixin._stage_and_swap_model(cli, _result("glm-5.3-flash"), "deepseek-v4-flash")
        assert cli.reasoning_config == _HIGH

        # Control: switching back to a model without an override lands on the global effort.
        assert CLIModelSwitchMixin._stage_and_swap_model(cli, _result("deepseek-v4-flash"), "glm-5.3-flash")
        assert cli.reasoning_config == _MEDIUM

        # --resume of a session stored on the override model, agent not built yet.
        cli = _cli("deepseek-v4-flash", _MEDIUM)
        CLIModelSwitchMixin._restore_session_model(
            cli, {"model": "glm-5.3-flash", "model_config": {"gateway_runtime": {
                "provider": "custom:ark", "base_url": "https://ark.example/v3", "api_mode": "chat_completions"}}},
            quiet=True)
        assert cli.model == "glm-5.3-flash"
        assert cli.reasoning_config == _HIGH


def test_failed_swap_and_new_session_keep_the_effort_with_the_route():
    import cli as cli_mod
    from hermes_cli.cli_session_mixin import CLISessionMixin

    def _boom(**_kw):
        raise RuntimeError("boom")

    with patch.dict(cli_mod.CLI_CONFIG, _CFG):
        # A failed in-place swap rolls reasoning_config back with the rest of the CLI route.
        cli = _cli("deepseek-v4-flash", _MEDIUM, agent=SimpleNamespace(switch_model=_boom))
        assert not CLIModelSwitchMixin._stage_and_swap_model(cli, _result("glm-5.3-flash"), "deepseek-v4-flash")
        assert (cli.model, cli.reasoning_config) == ("deepseek-v4-flash", _MEDIUM)

        # /new on a session whose config default carries an override: the default model's effort,
        # not the bare global key, and a /reasoning session override does not carry over.
        with patch.dict(cli_mod.CLI_CONFIG["model"], {"default": "glm-5.3-flash"}):
            cli = _cli("glm-5.3-flash", {"enabled": False})
            cli.conversation_history, cli.session_id, cli._session_db, cli._pending_title = [], "old", None, None
            cli._resumed, cli._notify_session_boundary = False, lambda *_a, **_k: None
            CLISessionMixin.new_session(cli, silent=True)
        assert cli.reasoning_config == _HIGH
