"""`hermes groups` asks the running gateway, shows exactly what a code would allow, and confirms."""
from types import SimpleNamespace

import pytest

from hermes_cli import groups_cmd

SHARED = {'code': 'ABCD-EFGH', 'expires_in': 500, 'profile': 'default', 'bot': 'default',
          'platform': 'telegram', 'kind': 'shared', 'chat': 'Team', 'chat_id': '-100',
          'user': 'Alice', 'user_id': '42', 'admins': 'group_allow_admin_from'}


@pytest.fixture
def gateway(monkeypatch, tmp_path):
    calls, answers = [], {}
    monkeypatch.setattr(groups_cmd, '_homes', lambda: [tmp_path])

    def query(home, verb, *, params=None, timeout=None):
        calls.append((verb, params))
        answer = answers.get(params['action'])
        return answer(params) if callable(answer) else answer
    monkeypatch.setattr('gateway.control_socket.query_gateway_control', query)
    return SimpleNamespace(calls=calls, answers=answers)


def run(action, **args):
    return groups_cmd.groups_command(SimpleNamespace(groups_action=action, **args))


def test_allow_describes_the_audience_and_needs_a_yes(gateway, monkeypatch, capsys):
    gateway.answers.update(describe=SHARED, allow={'grant': 'abcd1234', **SHARED})
    monkeypatch.setattr('hermes_cli.cli_output.prompt_yes_no', lambda question, default: False)
    assert run('allow', code='abcd-efgh', yes=False) == 1
    out = capsys.readouterr().out
    assert 'shared chat "Team"' in out and 'Everyone in that chat' in out and 'Not allowed.' in out
    assert "People on the Bot's group_allow_admin_from list" in out
    assert [params['action'] for _, params in gateway.calls] == ['describe']
    assert run('allow', code='abcd-efgh', yes=True) == 0
    assert 'hermes groups revoke abcd1234' in capsys.readouterr().out
    assert gateway.calls[-1] == ('group-chats', {'action': 'allow', 'code': 'abcd-efgh'})


def test_private_description_names_the_only_person(gateway, capsys):
    gateway.answers.update(describe={**SHARED, 'kind': 'private'}, allow={'grant': 'abcd1234'})
    assert run('allow', code='ABCDEFGH', yes=True) == 0
    out = capsys.readouterr().out
    assert 'private chat with Alice (user ID 42)' in out and 'Everyone' not in out


@pytest.mark.parametrize(('answer', 'text'), [
    (None, 'No running Hermes gateway answered'),
    ({'error': 'unknown_code'}, 'expires after 10 minutes'),
    ({'error': 'something_new'}, 'Refused: something_new'),
])
def test_allow_failures_are_explained(gateway, capsys, answer, text):
    gateway.answers['describe'] = answer
    assert run('allow', code='ABCDEFGH', yes=True) == 1
    assert text in capsys.readouterr().out


def test_chats_lists_every_reachable_gateway_and_revoke_reports(gateway, capsys):
    remembered = [{'group': 'Research', 'bot': 'Ada', 'command': 'npm test', 'context': 'Local, folder /w', 'uses': 2}]
    gateway.answers['list'] = {'chats': [{**SHARED, 'grant': 'abcd1234', 'created_at': 0, 'remembered': remembered}]}
    assert run('chats') == 0
    out = capsys.readouterr().out
    assert 'abcd1234  telegram shared chat "Team", profile "default"' in out
    assert 'always allowed in "Research" for Ada: npm test (Local, folder /w)' in out
    gateway.answers['list'] = {'chats': []}
    assert run('chats') == 0
    assert 'No chats can control' in capsys.readouterr().out
    gateway.answers['revoke'] = lambda params: (
        {'revoked': 'abcd1234'} if params['grant'] == 'abcd' else {'error': 'unknown_grant'})
    assert run('revoke', chat='abcd') == 0
    assert 'Revoked abcd1234' in capsys.readouterr().out
    assert run('revoke', chat='ffff') == 1
    assert 'hermes groups chats' in capsys.readouterr().out


def test_a_second_gateway_is_asked_when_the_first_does_not_hold_the_code(monkeypatch, tmp_path, capsys):
    homes = [tmp_path / 'profile', tmp_path / 'root']
    monkeypatch.setattr(groups_cmd, '_homes', lambda: homes)

    def query(home, verb, *, params=None, timeout=None):
        if home == homes[0]:
            return {'error': 'unknown_code'}
        return SHARED if params['action'] == 'describe' else {'grant': 'abcd1234'}
    monkeypatch.setattr('gateway.control_socket.query_gateway_control', query)
    assert run('allow', code='ABCDEFGH', yes=True) == 0
    assert run(None) == 1
    assert 'Usage: hermes groups' in capsys.readouterr().out


def test_parser_registers_the_subcommands():
    import argparse
    from hermes_cli.subcommands.groups import build_groups_parser
    parser = argparse.ArgumentParser()
    build_groups_parser(parser.add_subparsers(dest='command'), cmd_groups=lambda args: 0)
    args = parser.parse_args(['groups', 'allow', 'ABCD-EFGH', '--yes'])
    assert (args.groups_action, args.code, args.yes) == ('allow', 'ABCD-EFGH', True)
    assert parser.parse_args(['groups', 'revoke', 'abcd']).chat == 'abcd'
    from hermes_cli import main
    assert 'groups' in main._BUILTIN_SUBCOMMANDS
