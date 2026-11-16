"""Relay discovery and mailbox ownership are profile scoped."""
import json
from types import SimpleNamespace

from tools import bot_mode_dm, bot_mode_probe, bot_relay


def test_named_sender_uses_own_remote_roster_and_outbox(tmp_path, monkeypatch):
    root = tmp_path / 'state'
    sender = root / 'profiles' / 'sender'
    sender.mkdir(parents=True)
    (sender / 'profile.yaml').write_text('ui_meta:\n  hermes-bots:\n    shape: cloud\n')
    own = dict(profile='remote', handle='remote', connection_id='own', title='Owned remote')
    other = dict(profile='remote', handle='remote', connection_id='foreign', title='Foreign remote')
    bot_relay.write_remote_roster(sender, [own])
    bot_relay.write_remote_roster(root, [other])
    agent = SimpleNamespace(_session_db=SimpleNamespace(db_path=str(sender / 'state.db')),
                            _session_title_hint='Bot Chat')
    commands = []
    monkeypatch.setattr(bot_mode_dm, '_spawn_delivery', lambda command, *a, **kw:
                        commands.append(command) or json.dumps({'status': 'sent'}))
    result = json.loads(bot_mode_dm.message_agent_tool('remote', 'hello', agent=agent))
    assert result['status'] == 'sent'
    envelopes = bot_relay.claim_pending_envelopes(sender)
    assert len(envelopes) == 1 and envelopes[0]['target_connection'] == 'own'
    assert envelopes[0]['message'] == 'Message from 🤖 sender (@sender): hello'
    assert bot_relay.claim_pending_envelopes(root) == []
    section = bot_mode_probe.get_bot_mode_protocol_section(sender, force_refresh=True)
    assert 'Owned remote' in section and 'Foreign remote' not in section
