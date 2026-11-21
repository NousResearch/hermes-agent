"""Slack's hint stops forbidding tables once a Block Kit renderer that draws tables is on.

The base Slack hint says tables are NOT supported, which is true for plain mrkdwn. With
``platforms.slack.extra.markdown_blocks`` or ``rich_blocks`` on, GFM pipe tables render as real
Slack tables — but a model still told "no tables" hand-builds fixed-width tables inside ``` fences,
which Slack shows as raw monospace text. Live chats and Slack-delivered cron jobs share the hint.
"""

import types

import pytest

from agent.system_prompt import platform_hint
from gateway.platform_registry import platform_registry
from gateway.session_context import _VAR_MAP
from hermes_cli.plugins import discover_plugins

SUPERSEDES = "the no-tables advice above does not apply"


def _agent(platform):
    return types.SimpleNamespace(platform=platform, _platform_hint_overrides={})


@pytest.fixture
def slack_config(tmp_path, monkeypatch):
    """Write a real config.yaml under a temp HERMES_HOME; the loader itself is not mocked."""
    from hermes_cli import config as _cfgmod

    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(_cfgmod, "get_config_path", lambda: home / "config.yaml")
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
    discover_plugins()

    def write(yaml_text: str) -> None:
        (home / "config.yaml").write_text(yaml_text)

    return write


@pytest.mark.parametrize("flag", ["markdown_blocks", "rich_blocks"])
def test_table_rendering_flag_supersedes_no_tables_advice(slack_config, flag):
    slack_config(f"platforms:\n  slack:\n    extra:\n      {flag}: true\n")
    hint = platform_hint(_agent("slack"))
    base = platform_registry.get("slack").platform_hint
    assert hint.startswith(base.rstrip())
    assert hint.index(SUPERSEDES) > hint.index("tables are NOT supported")
    assert "GFM pipe table" in hint


def test_without_a_table_renderer_the_base_hint_stands(slack_config):
    slack_config("platforms:\n  slack:\n    extra:\n      markdown_blocks: false\n")
    assert platform_hint(_agent("slack")) == platform_registry.get("slack").platform_hint


def test_both_renderers_on_add_the_note_once(slack_config):
    slack_config("gateway:\n  platforms:\n    slack:\n      extra:\n        rich_blocks: true\n"
                 "platforms:\n  slack:\n    extra:\n      markdown_blocks: 'true'\n")
    assert platform_hint(_agent("slack")).count(SUPERSEDES) == 1


def test_slack_delivered_cron_job_gets_the_note_too(slack_config):
    slack_config("platforms:\n  slack:\n    extra:\n      markdown_blocks: true\n")
    var = _VAR_MAP["HERMES_CRON_AUTO_DELIVER_PLATFORM"]
    token = var.set("slack")
    try:
        assert SUPERSEDES in platform_hint(_agent("cron"))
    finally:
        var.reset(token)
