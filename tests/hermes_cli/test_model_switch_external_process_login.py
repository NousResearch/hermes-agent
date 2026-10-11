"""An external-process switch surfaces the CLI's own login state as a warning.

The bug this guards: the switch path accepted a provider whose CLI login was dead because
``auth._external_process_auth_evidence`` can only prove the *binary resolves*. The selection was
saved to config and the failure only appeared at the first request, as a provider failure with no
hint that the fix is a CLI login. ``hermes model`` already gates on ``setup_status()``; the
mid-session switch had no equivalent read.

A warning, not a refusal: the login can be fixed afterwards (or die mid-session), so the switch
still succeeds and persists.
"""
from unittest.mock import patch

from hermes_cli.model_switch import switch_model
from providers.base import ProviderProfile


def _switch(profile, status_patch=None):
    accepted = {"accepted": True, "persist": True, "recognized": True, "message": None}
    with patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("providers.get_provider_profile", return_value=profile), \
         patch("hermes_cli.model_switch.get_authenticated_provider_slugs", return_value=[]), \
         patch("hermes_cli.models_validate.validate_requested_model", return_value=accepted), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch("hermes_cli.model_switch._check_hermes_model_warning", return_value=""), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value={"api_key": "external-process", "base_url": profile.base_url,
                             "api_mode": "chat_completions"}):
        return switch_model(raw_input="claude-sonnet-5[1m]", current_provider="ollama-cloud",
                            current_model="deepseek-v4.1-flash", current_base_url="https://ollama.com/v1",
                            user_providers={}, custom_providers=[])


def _profile(status):
    prof = ProviderProfile(
        name="proc-provider", display_name="Proc Provider", auth_type="external_process",
        base_url="process://proc-provider", process_command="/bin/true",
        fallback_models=("claude-sonnet-5[1m]",))
    prof.setup_status = lambda **_: status
    return prof


def test_logged_out_switch_warns_with_the_cli_instruction():
    detail = ("Claude Code is installed but has no usable login in the environment Hermes runs it "
              "in. Run `claude auth login` as the user Hermes runs as, set CLAUDE_CODE_OAUTH_TOKEN "
              "(from `claude setup-token`) in Hermes' environment, or point "
              "CLAUDE_SUBSCRIPTION_DIRECTSDK_CONFIG_DIR at a logged-in config directory, then try again.")
    result = _switch(_profile({"available": True, "logged_in": False, "plan": "", "detail": detail,
                               "login_command": ["claude", "auth", "login"]}))
    assert result.success, result.error_message
    assert "no usable login" in result.warning_message
    assert "claude auth login" in result.warning_message


def test_logged_in_switch_has_no_login_warning():
    result = _switch(_profile({"available": True, "logged_in": True, "plan": "Claude Max",
                               "detail": "", "login_command": None}))
    assert result.success, result.error_message
    assert "login" not in result.warning_message.lower()


def test_profile_without_setup_status_is_silent():
    """``ProviderProfile.setup_status`` defaults to None — nothing to report beyond the binary."""
    result = _switch(_profile(None))
    assert result.success, result.error_message
    assert result.warning_message == ""
