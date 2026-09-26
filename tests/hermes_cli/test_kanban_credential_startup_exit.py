"""Credential rejection before a turn must retain the Kanban exit contract."""
from types import SimpleNamespace

import pytest

import cli
from hermes_cli.auth import AuthError, CODEX_RATE_LIMITED_CODE
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


@pytest.mark.parametrize("quiet", [False, True])
@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("kind,worker_exit", [("revoked", 78), ("quota", 75), ("unknown", 1)])
def test_credential_resolution_exit(monkeypatch, quiet, worker, kind, worker_exit):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    if worker:
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fixture")
    errors = {
        "revoked": AuthError("fixture revoked", provider="openai-codex", code="invalid_grant", relogin_required=True),
        "quota": AuthError("fixture quota", provider="openai-codex", code=CODEX_RATE_LIMITED_CODE),
        "unknown": RuntimeError("fixture transport"),
    }
    def resolve(**kwargs):
        raise errors[kind]
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    shell = CLIAgentSetupMixin.__new__(CLIAgentSetupMixin)
    shell.model = "fixture"
    shell.requested_provider = "openai-codex"
    shell._explicit_api_key = shell._explicit_base_url = None
    shell._fallback_model = []
    shell.tool_progress_mode = "off"
    shell.session_id = "fixture-session"
    shell._last_turn_result = None
    shell._claim_active_session = lambda *a, **k: True
    shell._show_security_advisories = lambda: None
    shell._print_exit_summary = lambda **k: None
    shell.console = SimpleNamespace(print=lambda *a, **k: None)
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
    shell._secret_capture_callback = lambda *a, **k: None
    shell.chat = CLIChatTurnMixin.chat.__get__(shell)
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda q, i: (q, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda imgs: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda c: None)
    with pytest.raises(SystemExit) as exc:
        cli._run_single_query_mode(shell, "fixture", None, quiet, True)
    assert exc.value.code == (worker_exit if worker else 1)


@pytest.mark.parametrize("fallback", [False, True])
def test_successful_resolution_clears_prior_failure(monkeypatch, fallback):
    shell = CLIAgentSetupMixin.__new__(CLIAgentSetupMixin)
    shell.model = "fixture"
    shell.requested_provider = "openai-codex"
    shell._explicit_api_key = shell._explicit_base_url = None
    shell._fallback_model = [{"provider": "custom", "model": "fixture"}]
    shell._credentials_terminal = shell._credentials_rate_limited = True
    shell.api_key = "fixture-key"
    shell.base_url = "http://fixture.invalid/v1"
    shell.provider = "custom"
    shell.api_mode = "chat_completions"
    shell.acp_command = None
    shell.acp_args = []
    shell.agent = None
    shell._maybe_print_free_tier_available_notice = lambda: None
    shell._normalize_model_for_provider = lambda provider: False
    calls = []
    def resolve(**kwargs):
        calls.append(kwargs)
        if fallback and kwargs["requested"] == "openai-codex":
            raise AuthError("revoked", relogin_required=True)
        return {"provider": "custom", "api_mode": "chat_completions",
                "api_key": shell.api_key, "base_url": shell.base_url}
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    monkeypatch.setattr("hermes_cli.fallback_config.resolve_entry_api_key", lambda entry: "fixture-key")
    assert shell._ensure_runtime_credentials() is True
    assert shell._credentials_terminal is False
    assert shell._credentials_rate_limited is False
    assert calls[-1]["target_model"] == "fixture"
    if fallback:
        assert calls[-1]["requested"] == "custom"


@pytest.mark.parametrize("result,expected", [
    ({"completed": True}, 0),
    ({"interrupted": True}, 130),
    ({"failed": True, "failure_reason": "unknown"}, 1),
])
def test_turn_result_takes_precedence_over_startup_flags(monkeypatch, result, expected):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fixture")
    assert cli._single_query_exit_code(
        result, credentials_terminal=True, credentials_rate_limited=True) == expected
