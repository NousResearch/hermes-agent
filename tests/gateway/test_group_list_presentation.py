"""M73 presentation invariants adapted to canonical names/counts, not selectors."""
from types import SimpleNamespace

import pytest

from gateway.group_chat_slash import GroupChatSlashCommandsMixin, parse_group_args
from gateway.hosted_room_messaging_runtime import InvalidPageError, render_inventory, safe_name


@pytest.mark.parametrize('args', ['list 0', 'list -1', 'list +1', 'list 1.0', 'list ١',
                                 'list 2 extra', 'list 99999', 'send 1', 'stop', '@all'])
def test_only_bounded_list_and_help_arguments(args):
    with pytest.raises(InvalidPageError):
        parse_group_args(args)


def test_safe_names_order_count_spacing_and_actual_prefix():
    names = ['Zulu', '@everyone <@123> `code` **b** MEDIA:/private/file',
             '[link](https://secret.invalid)', 'A\nB\x00C\u202eD'] + [f'Row {i}' for i in range(6)]
    rows = [dict(name=name, member_count=i) for i, name in enumerate(names)]
    body = render_inventory(rows, 1, '!')
    assert body.startswith('Groups — page 1 of 2\n\n• Zulu — 0 members\n')
    assert body.count('\n• ') == 8
    assert 'Next: !group list 2' in body and '/group' not in body
    assert '1 member\n' in body
    assert all(token not in body for token in ('@', '`', '**', 'MEDIA:', '[link]', 'https:', '\x00', '\u202e'))
    assert '\n\nNext:' in body
    last = render_inventory(rows, 2, '!')
    assert last.count('\n• ') == 2 and 'Previous: !group list 1' in last
    assert 'Next:' not in last
    assert safe_name(' \n ') == 'Unnamed group'
    assert len(safe_name('X' * 200)) == 72
    for page in (0, -1, True, 3, '1'):
        with pytest.raises(InvalidPageError):
            render_inventory(rows, page, '/')
    assert parse_group_args('') == parse_group_args('list') == parse_group_args('list 1') == 1
    assert parse_group_args('help') is None


def test_read_buckets_bound_live_keys_and_expire_without_writes(monkeypatch):
    from gateway import group_chat_slash as slash
    runner = GroupChatSlashCommandsMixin()
    now = [10.0]
    monkeypatch.setattr(slash.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(slash, '_GROUP_CHAT_RATE_BUCKET_CAP', 2)
    a = SimpleNamespace(recipient_json='recipient-a')
    for _ in range(slash._GROUP_CHAT_READ_RATE_LIMIT):
        assert runner._group_chat_rate_limit_denial(a) is None
    assert runner._group_chat_rate_limit_denial(a)
    assert runner._group_chat_rate_limit_denial(SimpleNamespace(recipient_json='recipient-b')) is None
    assert runner._group_chat_rate_limit_denial(SimpleNamespace(recipient_json='recipient-c'))
    assert len(runner._group_chat_command_rate_buckets) == 2
    now[0] += slash._GROUP_CHAT_RATE_WINDOW_SECONDS
    assert runner._group_chat_rate_limit_denial(a) is None
    assert len(runner._group_chat_command_rate_buckets) == 1
