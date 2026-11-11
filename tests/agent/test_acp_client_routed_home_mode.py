"""A routed profile's ``terminal.home_mode`` governs the HOME of the ACP agent process it starts.

One multiplexed gateway serves several profiles; an external ACP agent CLI launched for a turn routed
to another profile is one of that profile's tool subprocesses, so its HOME follows that profile's
setting, not the launch profile's process environment.
"""

import os
import sys

import pytest

import tools.terminal_tool as tt
from agent.acp_client import ACPClient

# Answers every prompt with the HOME it was started with.
_AGENT = r'''
import json, os, sys
def send(o): print(json.dumps(o), flush=True)
for line in sys.stdin:
    msg = json.loads(line)
    m, i = msg.get("method"), msg.get("id")
    if m == "session/new":
        send({"jsonrpc": "2.0", "id": i, "result": {"sessionId": "s1"}})
    elif m == "session/prompt":
        send({"jsonrpc": "2.0", "method": "session/update", "params": {"update": {
            "sessionUpdate": "agent_message_chunk", "content": {"text": os.environ.get("HOME", "")}}}})
        send({"jsonrpc": "2.0", "id": i, "result": {"stopReason": "end_turn"}})
    elif i is not None:
        send({"jsonrpc": "2.0", "id": i, "result": {}})
'''


@pytest.fixture
def bee(monkeypatch, tmp_path):
    """A second profile whose config asks for the profile home, under a launch env that sets no mode."""
    for name in [k for k in os.environ if k.startswith("TERMINAL_")]:
        monkeypatch.delenv(name)
    monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", True)
    home = tmp_path / ".hermes" / "profiles" / "bee"
    (home / "home").mkdir(parents=True)
    (home / "config.yaml").write_text("terminal:\n  backend: local\n  home_mode: profile\n", encoding="utf-8")
    return home


@pytest.mark.platforms("posix")
def test_routed_profile_home_mode_reaches_the_acp_agent_process(bee, tmp_path):
    import gateway.run as gw

    script = tmp_path / "home_reporting_acp_agent.py"
    script.write_text(_AGENT, encoding="utf-8")
    client = ACPClient(command=sys.executable, args=[str(script)], acp_cwd=str(tmp_path))

    with gw._profile_runtime_scope(bee):
        completion = client.chat.completions.create(messages=[{"role": "user", "content": "hi"}], timeout=30)

    assert completion.choices[0].message.content == str(bee / "home")
