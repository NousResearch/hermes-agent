"""Mattermost CLI destinations must preserve channel/root and authorization."""
import argparse
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

CHANNEL = "a" * 26
ROOT = "b" * 26


@pytest.fixture
def offline_sender(tmp_path, monkeypatch):
    # Exercise real plugin discovery/configuration in the isolated test home.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("MATTERMOST_URL", "https://mattermost.invalid")
    monkeypatch.setenv("MATTERMOST_TOKEN", "offline-fixture-token")
    from hermes_cli.plugins import discover_plugins
    discover_plugins(force=True)
    from tools import send_message_tool as send
    from plugins.platforms.mattermost import adapter
    # Only the HTTP transport is mocked; sender, resolver and guard are real.
    response = MagicMock()
    response.status = 201
    response.json = AsyncMock(return_value={"id": "c" * 26})
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post.return_value = response
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("aiohttp.ClientSession", MagicMock(return_value=session))
    monkeypatch.setattr("gateway.mirror.mirror_to_session", lambda *a, **kw: False)
    original_guard = send._authorize_relay_target
    guard = MagicMock(wraps=original_guard)
    monkeypatch.setattr(send, "_authorize_relay_target", guard)
    return send, adapter, session, guard


@pytest.mark.parametrize("threaded,deny", [(False, False), (True, False), (True, True)])
def test_cli_destination_reaches_payload_and_guard(offline_sender, capsys, threaded, deny):
    send, _adapter, session, guard = offline_sender
    if deny:
        guard.side_effect = None
        guard.return_value = "Fixture authorization denial"
    from hermes_cli.send_cmd import cmd_send
    target = f"mattermost:{CHANNEL}" + (f":{ROOT}" if threaded else "")
    args = argparse.Namespace(to=target, message="Resume the existing scenario", file=None,
                              subject=None, json=True, quiet=False, list_targets=False)
    with pytest.raises(SystemExit) as exc:
        cmd_send(args)
    result = json.loads(capsys.readouterr().out)
    expected_root = ROOT if threaded else None
    guard.assert_called_once_with("mattermost", CHANNEL, expected_root,
                                  native_token="offline-fixture-token")
    if deny:
        assert exc.value.code == 1
        assert "Fixture authorization denial" in result["error"]
        session.post.assert_not_called()
    else:
        assert exc.value.code == 0
        assert result["success"]
        url = session.post.call_args.args[0]
        payload = session.post.call_args.kwargs["json"]
        assert url == "https://mattermost.invalid/api/v4/posts"
        assert payload["channel_id"] == CHANNEL
        assert payload.get("root_id") == expected_root
        assert payload["message"] == args.message
        assert ("root_id" in payload) == threaded


@pytest.mark.parametrize("target_ref", [
    CHANNEL + ":" + ROOT + ":extra", CHANNEL + ":", CHANNEL + ":" + ROOT[:-1],
    CHANNEL.upper() + ":" + ROOT, CHANNEL + ":" + ROOT.upper(),
    CHANNEL + ":" + ROOT + "/", "short", CHANNEL + "\n:" + ROOT,
])
def test_malformed_destination_never_reaches_network(offline_sender, target_ref):
    send, _adapter, session, guard = offline_sender
    result = json.loads(send.send_message_tool({"action": "send",
                       "target": "mattermost:" + target_ref, "message": "offline"}))
    assert result.get("error")
    session.post.assert_not_called()
    guard.assert_not_called()
