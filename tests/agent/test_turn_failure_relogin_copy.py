"""Re-login copy for an ``external_process`` provider must name a command that works (#121290).

``copilot-acp`` signs in through the CLI subprocess it drives; ``hermes auth add copilot-acp
--type oauth`` exits with "not implemented for auth type oauth yet", so every copy surface
(``oauth_relogin_command``, ``relogin_command_hint`` and the chat's non-retryable auth notice)
must point at ``hermes model`` — its external-process flow runs the CLI's own login.
"""

from types import SimpleNamespace

from agent.turn_failure_copy import nonretryable_copy, oauth_relogin_command, relogin_command_hint


def test_copilot_acp_relogin_command_is_the_model_picker() -> None:
    assert oauth_relogin_command("copilot-acp") == "hermes model"
    assert "auth add" not in oauth_relogin_command("copilot-acp")


def test_copilot_acp_relogin_hint_never_names_auth_add() -> None:
    assert "auth add" not in relogin_command_hint("copilot-acp")


def test_copilot_acp_chat_auth_notice_names_a_working_command() -> None:
    classified = SimpleNamespace(is_auth=True)
    copy = nonretryable_copy(
        classified, provider="copilot-acp", model="gpt-5",
        summary="Copilot ACP session/prompt failed: Authentication required",
    )
    assert "Sign in again: `hermes model`." in copy
    assert "auth add" not in copy


def test_implemented_oauth_providers_and_nous_keep_their_commands() -> None:
    assert oauth_relogin_command("nous") == "hermes portal"
    assert oauth_relogin_command("openai-codex") == "hermes auth add openai-codex --type oauth"
    assert oauth_relogin_command("xai-oauth") == "hermes auth add xai-oauth --type oauth"
