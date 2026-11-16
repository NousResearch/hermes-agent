"""A bare ``message_agent`` target naming both a local bot and a registered peer (#145241).

The refusal must point at routes that actually work, and the collision check must not
re-read every profile's friendly names for an ordinary bare-peer send.
"""

import json

from tests.tools.test_bot_mode_dm import _FakeAgent, _capture_spawn, _managed_home, _rename
from tools import bot_mode_dm, bot_mode_probe, bot_relay


def test_refusal_never_offers_a_handle_another_peer_owns(tmp_path, monkeypatch):
    """With peers 'ops' and 'aaa', local 'aaa' answers to a peer name too: only @bbb is a
    route that reaches a local bot."""
    calls = _capture_spawn(monkeypatch)
    monkeypatch.setattr(bot_relay, "_hermes_cli", lambda: "hermes")
    home = _managed_home(tmp_path, teammates=("aaa", "bbb"), peers=("ops", "aaa"))
    _rename(home, "aaa", title="Ops")
    _rename(home, "bbb", title="Ops")

    result = json.loads(bot_mode_dm.message_agent_tool(target="ops", message="ping", agent=_FakeAgent(home)))

    assert "ambiguous" in result["error"]
    assert "or @bbb to reach it" in result["error"]
    assert "@aaa" not in result["error"]
    assert calls == []


def test_bare_peer_send_reads_local_aliases_once(tmp_path, monkeypatch):
    """A peer with no local twin is the common case; its collision check reads the alias map once."""
    _capture_spawn(monkeypatch)
    monkeypatch.setattr(bot_relay, "_hermes_cli", lambda: "hermes")
    monkeypatch.setattr("socket.gethostname", lambda: "eri-mac.local")
    home = _managed_home(tmp_path, teammates=("researcher",), peers=("ops",))
    reads = []
    real = bot_mode_probe.local_alias_map
    monkeypatch.setattr(bot_mode_probe, "local_alias_map", lambda root: reads.append(root) or real(root))

    result = json.loads(bot_mode_dm.message_agent_tool(target="ops", message="ping", agent=_FakeAgent(home)))

    assert result["status"] == "queued"
    assert len(reads) == 1
