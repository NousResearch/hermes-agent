from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_gquota_uses_chat_console_when_tui_is_live():
    from agent.google_oauth import GoogleOAuthError
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.console = MagicMock()
    cli._app = object()

    live_console = MagicMock()

    with patch("cli.ChatConsole", return_value=live_console), \
         patch("agent.google_oauth.get_valid_access_token", side_effect=GoogleOAuthError("No Google OAuth credentials found")), \
         patch("agent.google_oauth.load_credentials", return_value=None), \
         patch("agent.google_code_assist.retrieve_user_quota"):
        cli._handle_gquota_command("/gquota")

    assert live_console.print.call_count == 2
    cli.console.print.assert_not_called()


def test_gquota_resolves_project_context_before_quota_lookup():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.console = MagicMock()
    cli._app = None
    cli.current_model = "gemini-3.1-pro-preview"

    bucket = SimpleNamespace(
        model_id="gemini-3.1-pro-preview",
        token_type="daily",
        remaining_fraction=0.5,
    )
    ctx = SimpleNamespace(
        project_id="resolved-project",
        managed_project_id="managed-project",
        source="discovered",
    )

    with patch("agent.google_oauth.get_valid_access_token", return_value="access-token"), \
         patch("agent.google_oauth.load_credentials", return_value=SimpleNamespace(project_id="")), \
         patch("agent.google_oauth.resolve_project_id_from_env", return_value=""), \
         patch("agent.google_oauth.update_project_ids") as update_project_ids, \
         patch("agent.google_code_assist.resolve_project_context", return_value=ctx) as resolve_project_context, \
         patch("agent.google_code_assist.retrieve_user_quota", return_value=[bucket]) as retrieve_user_quota:
        cli._handle_gquota_command("/gquota")

    resolve_project_context.assert_called_once_with(
        "access-token",
        env_project_id="",
        user_agent_model="gemini-3.1-pro-preview",
    )
    update_project_ids.assert_called_once_with(
        project_id="resolved-project",
        managed_project_id="managed-project",
    )
    retrieve_user_quota.assert_called_once_with(
        "access-token",
        project_id="resolved-project",
    )
    rendered = "\n".join(str(call.args[0]) for call in cli.console.print.call_args_list if call.args)
    assert "resolved-project" in rendered
    assert "source: discovered" in rendered


def test_gquota_falls_back_to_auto_quota_when_project_discovery_fails():
    from agent.google_code_assist import CodeAssistError
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.console = MagicMock()
    cli._app = None
    cli.current_model = "gemini-3.1-pro-preview"

    bucket = SimpleNamespace(
        model_id="gemini-3.1-pro-preview",
        token_type="daily",
        remaining_fraction=0.25,
    )

    with patch("agent.google_oauth.get_valid_access_token", return_value="access-token"), \
         patch("agent.google_oauth.load_credentials", return_value=SimpleNamespace(project_id="")), \
         patch("agent.google_oauth.resolve_project_id_from_env", return_value=""), \
         patch("agent.google_oauth.update_project_ids") as update_project_ids, \
         patch("agent.google_code_assist.resolve_project_context", side_effect=CodeAssistError("discovery failed")) as resolve_project_context, \
         patch("agent.google_code_assist.retrieve_user_quota", return_value=[bucket]) as retrieve_user_quota:
        cli._handle_gquota_command("/gquota")

    resolve_project_context.assert_called_once_with(
        "access-token",
        env_project_id="",
        user_agent_model="gemini-3.1-pro-preview",
    )
    update_project_ids.assert_not_called()
    retrieve_user_quota.assert_called_once_with(
        "access-token",
        project_id="",
    )
    rendered = "\n".join(str(call.args[0]) for call in cli.console.print.call_args_list if call.args)
    assert "Project discovery failed" in rendered
    assert "project: (auto / free-tier)" in rendered
