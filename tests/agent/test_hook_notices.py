"""A hook's ``notice`` / ``systemMessage`` reaches the user and never the model.

Port of MiniMax-AI/minimax-code#376 (Claude-Code hook protocol's ``systemMessage``). Before, a
plugin or shell hook could only speak to the model (``context``, a block message, a transformed
result); a formatter that wanted to say "reformatted 3 files" had to stay silent or spend model
context. The text rides the ``AgentNotice`` / status channel of the agent whose ``session_id`` the
hook payload names, and is never appended to ``messages``.
"""

from __future__ import annotations

import sys
import pytest

from agent import hook_notices, shell_hooks
from agent.status_output import StatusOutputMixin
from hermes_cli import plugins


class _Agent(StatusOutputMixin):
    """The slice of AIAgent the notice path touches: identity, driver callbacks, status printing."""

    log_prefix = ""
    suppress_status_output = True  # no terminal in the test; the status_callback is what we read

    def __init__(self, session_id: str, platform: str):
        self.session_id = session_id
        self.platform = platform
        self.messages: list = []
        self.notices: list = []
        self.statuses: list = []
        self.notice_callback = self.notices.append
        self.status_callback = lambda kind, text: self.statuses.append((kind, text))


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    plugins._plugin_manager = plugins.PluginManager()
    shell_hooks.reset_for_tests()
    yield plugins._plugin_manager
    shell_hooks.reset_for_tests()


def test_plugin_hook_notice_is_shown_to_the_session_owner_and_kept_out_of_model_context(manager):
    """Both dialects, both driver families: a non-CLI agent gets a self-expiring ``AgentNotice``;
    the CLI agent gets a ``hook_notice`` status line. Neither touches ``messages``; an agent whose
    session the payload does not name sees nothing."""
    desktop = _Agent("sess-desktop", platform="desktop")
    cli = _Agent("sess-cli", platform="cli")
    other = _Agent("sess-other", platform="desktop")
    for a in (desktop, cli, other):
        hook_notices.register_agent(a)

    manager._hooks.setdefault("post_tool_call", []).append(
        lambda **kw: {"systemMessage": "Formatted 3 files\x1b[31m"} if kw["tool_name"] == "write_file" else {"notice": "lint ok"})

    manager.invoke_hook("post_tool_call", tool_name="write_file", args={}, result="ok", session_id="sess-desktop")
    manager.invoke_hook("post_tool_call", tool_name="terminal", args={}, result="ok", session_id="sess-cli")

    [notice] = desktop.notices
    assert notice.text == "⚑ hook post_tool_call: Formatted 3 files"  # ANSI stripped
    assert (notice.level, notice.kind) == ("info", "ttl") and notice.ttl_ms
    assert cli.statuses == [("hook_notice", "⚑ hook post_tool_call: lint ok")] and cli.notices == []
    assert other.notices == [] and other.statuses == []
    assert desktop.messages == cli.messages == []


def test_shell_hook_system_message_rides_any_event_and_never_counts_as_a_directive(tmp_path, manager):
    """A Claude-Code style script's ``systemMessage`` surfaces from ``post_tool_call`` (an event
    with no directive parser) — and a fail-closed ``pre_tool_call`` gate that died with only a
    notice still blocks: a notice is not a decision."""
    script = tmp_path / "notice.py"
    script.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "print(json.dumps({'systemMessage': 'checks complete for ' + payload['hook_event_name']}))\n"
        "sys.exit(3 if payload['hook_event_name'] == 'pre_tool_call' else 0)\n")
    cmd = f"{sys.executable} {script}"
    agent = _Agent("sess-shell", platform="desktop")
    hook_notices.register_agent(agent)
    cfg = {"hooks": {"post_tool_call": [{"command": cmd}],
                     "pre_tool_call": [{"command": cmd, "fail_closed": True}]}}
    assert len(shell_hooks.register_from_config(cfg, accept_hooks=True)) == 2

    manager.invoke_hook("post_tool_call", tool_name="terminal", args={}, result="ok", session_id="sess-shell")
    block = plugins.get_pre_tool_call_block_message(tool_name="terminal", args={"command": "x"}, session_id="sess-shell")

    assert [n.text for n in agent.notices][0] == "⚑ hook post_tool_call: checks complete for post_tool_call"
    assert block and "exited 3 with no directive" in block
    assert agent.messages == []
