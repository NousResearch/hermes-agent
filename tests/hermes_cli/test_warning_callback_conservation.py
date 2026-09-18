"""Concrete CLI sinks own filtering; generic agent callbacks remain observable."""
from types import SimpleNamespace

import pytest
import yaml

from agent.status_output import StatusOutputMixin
from hermes_cli.cli_stream_mixin import CLIStreamMixin


class Agent(StatusOutputMixin):
    suppress_status_output = True
    platform = "cli"

    def _touch_activity(self, *args):
        pass


class CLI(CLIStreamMixin):
    _spinner_text = ""

    def _invalidate(self):
        pass


@pytest.mark.parametrize("setting", [None, False, True])
def test_cli_notice_and_wait_callbacks_keep_default_output(tmp_path, monkeypatch, setting):
    import cli
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(
        {} if setting is None else {"display": {"suppress_warning_notifications": setting}}))
    printed = []
    monkeypatch.setattr(cli, "_cprint", printed.append)
    surface = CLI()
    agent = Agent()
    surface.agent = agent
    agent.notice_callback = surface._on_notice
    agent.thinking_callback = surface._on_thinking
    agent._emit_notice(SimpleNamespace(level="warn", text="arbitrary warning wording"))
    agent._emit_diagnostic_wait("arbitrary retry wording")
    surface._flush_credit_notices()
    assert bool(printed) is not (setting is True)
    assert surface._spinner_text == ("" if setting is True else "arbitrary retry wording")
    agent._emit_notice(SimpleNamespace(level="info", text="ordinary progress"))
    agent._emit_wait_notice("⚠ requested progress")
    surface._flush_credit_notices()
    assert "ordinary progress" in printed[-1]
    assert surface._spinner_text == "⚠ requested progress"
