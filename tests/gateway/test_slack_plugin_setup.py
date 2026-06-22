"""Tests for the Slack plugin's interactive_setup wizard.

These cover the home-channel save logic that previously lived in
``hermes_cli/setup.py::_setup_slack`` before the Slack adapter migrated to a
bundled plugin (#41112). ``interactive_setup`` lazy-imports its CLI helpers
from ``hermes_cli.config`` (get_env_value / save_env_value) and
``hermes_cli.cli_output`` (prompt / prompt_yes_no / print_*), so we patch those
source modules.
"""
import hermes_cli.config as config_mod
import hermes_cli.cli_output as cli_output_mod
from plugins.platforms.slack.adapter import interactive_setup


def _patch_setup_io(monkeypatch, prompts, saved):
    """Wire interactive_setup's lazy-imported CLI helpers to test doubles."""
    prompt_iter = iter(prompts)
    monkeypatch.setattr(config_mod, "get_env_value", lambda key: "")
    monkeypatch.setattr(config_mod, "save_env_value", lambda k, v: saved.update({k: v}))
    monkeypatch.setattr(cli_output_mod, "prompt", lambda *_a, **_kw: next(prompt_iter))
    monkeypatch.setattr(cli_output_mod, "prompt_yes_no", lambda *_a, **_kw: False)
    for name in ("print_header", "print_info", "print_success", "print_warning"):
        monkeypatch.setattr(cli_output_mod, name, lambda *_a, **_kw: None)
    # Manifest writing reaches out to hermes_cli.slack_cli + filesystem; stub it.
    import hermes_cli.slack_cli as slack_cli_mod
    monkeypatch.setattr(slack_cli_mod, "_build_full_manifest", lambda **_kw: {"display_information": {}})


def test_interactive_setup_saves_home_channel(monkeypatch, tmp_path):
    """interactive_setup() saves SLACK_HOME_CHANNEL when the user provides one."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    saved = {}
    # prompts: bot token, app token, allowed users (empty), home channel
    _patch_setup_io(
        monkeypatch,
        ["xoxb-test-token", "xapp-test-token", "", "C01ABC2DE3F"],
        saved,
    )

    interactive_setup()

    assert saved.get("SLACK_HOME_CHANNEL") == "C01ABC2DE3F"


def test_interactive_setup_home_channel_empty_not_saved(monkeypatch, tmp_path):
    """interactive_setup() does not save SLACK_HOME_CHANNEL when left blank."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    saved = {}
    _patch_setup_io(
        monkeypatch,
        ["xoxb-test-token", "xapp-test-token", "", ""],
        saved,
    )

    interactive_setup()

    assert "SLACK_HOME_CHANNEL" not in saved
