"""The Inbox: a finished cron run filed by ``deliver=inbox`` is listed, marked read, and continued once
as the continuing client's own interactive chat (gateway/session_inbox.py, cron/inbox.py)."""
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

# Import-time update recovery stats the checkout's git dir; collection runs before the home I/O guard.
import gateway.session_managed_worker
from gateway.session_contract import SessionActor


def _connection(owner, *, desktop_local=True, capabilities=('session:read', 'session:control', 'session:submit')):
    actor = SessionActor('desktop-user', owner.profile_id, frozenset(capabilities), 't1')
    return SimpleNamespace(authority=owner, actor=actor, transport=SimpleNamespace(desktop_local=desktop_local))


def _filed_run(owner, tmp_path, *, ended=True):
    """A cron run session committed exactly as ``session_cron._create`` freezes it, then filed."""
    from cron.inbox import INBOX_MARKER
    from gateway.session_local import commit_local_session
    from gateway.session_policy import build_policy
    from persistence.authority.bindings import local_session_id
    sid = local_session_id(owner.profile_id, 'cron-owner', 'cron:job-1:fire-1')
    policy = build_policy({'source': 'cli', 'request_id': 'cron:job-1:fire-1', 'toolsets': [], 'cwd': str(tmp_path)},
                          {})
    policy = replace(policy, source='cron', platform='cron', request_json=json.dumps(
        {'cron_job': {'id': 'job-1'}, 'extra_prompt': None, 'request_id': 'cron:job-1:fire-1'}))
    route = f'agent:main:local:dm:{sid}'
    origin = {'platform': 'local', 'chat_id': sid, 'user_id': 'cron-owner', 'chat_type': 'dm', 'profile': None}
    entry = {'session_key': route, 'session_id': sid, 'created_at': '2026-01-01T00:00:00',
             'updated_at': '2026-01-01T00:00:00', 'origin': origin, 'platform': 'local'}
    commit_local_session(owner.persistence, epoch=owner.epoch, receipt={
        'profile_id': owner.profile_id, 'principal_id': 'cron-owner', 'request_id': 'cron:job-1:fire-1',
        'session_id': sid, 'route': route, 'entry': entry, 'policy': asdict(policy)})
    owner.persistence.sessions.update_system_prompt(sid, 'You are a cron job.\n\nPlatform: cron')
    owner.persistence.sessions.patch_model_config(sid, {INBOX_MARKER: {
        'job_id': 'job-1', 'job_name': 'standup', 'delivered_at': 1.0, 'preview': 'Remember standup'}})
    owner.persistence.sessions.set_read(sid, False)
    if ended:
        owner.persistence.sessions.end(sid, 'cron_complete')
    return sid


@pytest.mark.asyncio
async def test_inbox_lists_a_filed_run_unread_and_marks_it_read(owner, tmp_path):
    from gateway.session_inbox import list_items, mark_read
    sid = _filed_run(owner, tmp_path)
    connection = _connection(owner)

    listed = await list_items(connection, None, {})
    assert listed['unread_count'] == 1
    assert listed['items'] == [{'session_id': sid, 'job_id': 'job-1', 'job_name': 'standup', 'delivered_at': 1.0,
                                'preview': 'Remember standup', 'unread': True, 'running': False}]

    assert await mark_read(connection, None, {'session_id': sid}) == {'session_id': sid, 'unread': False}
    assert (await list_items(connection, None, {}))['unread_count'] == 0


@pytest.mark.asyncio
async def test_continuing_a_finished_item_rebuilds_it_once_as_the_desktop_chat(owner, tmp_path, monkeypatch):
    """The first continuation re-resolves the cron policy as a fresh Desktop chat's (desktop platform, the
    Desktop's own toolsets incl. desktop_ui) and clears the cron prompt snapshot; a second is a no-op."""
    from gateway.session_contract import SessionRef
    from gateway.session_inbox import continue_if_inbox
    from gateway.session_policy import restore_policy
    import gateway.session_policy_refresh as refresh
    published = []
    monkeypatch.setattr(refresh, '_publish', lambda authority, ref, moved=False: published.append(ref.session_id))
    sid = _filed_run(owner, tmp_path)
    ref = SessionRef(owner.profile_id, sid)

    await continue_if_inbox(_connection(owner), ref)
    policy = restore_policy(owner.persistence.bindings.receipt(sid).policy)
    assert (policy.source, policy.platform) == ('gui', 'desktop')
    assert 'desktop_ui' in policy.toolsets
    assert 'cron_job' not in json.loads(policy.request_json)
    assert owner.persistence.sessions.get(sid).system_prompt is None  # the one cache break
    assert published == [sid]

    stored = owner.persistence.bindings.receipt(sid).policy
    await continue_if_inbox(_connection(owner), ref)
    assert owner.persistence.bindings.receipt(sid).policy == stored and published == [sid]


@pytest.mark.asyncio
async def test_a_running_item_stays_on_its_cron_policy(owner, tmp_path):
    from gateway.session_contract import SessionRef
    from gateway.session_inbox import continue_if_inbox
    sid = _filed_run(owner, tmp_path, ended=False)
    await continue_if_inbox(_connection(owner), SessionRef(owner.profile_id, sid))
    assert owner.persistence.bindings.receipt(sid).policy['source'] == 'cron'


@pytest.mark.asyncio
async def test_inbox_list_reads_only_the_newest_window_of_runs(owner, tmp_path, monkeypatch):
    """The listing cost is bounded: only the newest LIST_MAX marked runs are read, however many exist."""
    from gateway import session_inbox
    sid = _filed_run(owner, tmp_path)
    read = []
    real_item = session_inbox._item
    monkeypatch.setattr(session_inbox, '_item', lambda store, s: read.append(s) or real_item(store, s))
    monkeypatch.setattr(session_inbox, 'LIST_MAX', 1)
    monkeypatch.setattr(owner.persistence.sessions, 'session_ids_with_model_config_key',
                        lambda key: (sid, 'older-1', 'older-2'))
    listed = await session_inbox.list_items(_connection(owner), None, {'limit': 1})
    assert read == [sid] and [item['session_id'] for item in listed['items']] == [sid]


@pytest.mark.asyncio
async def test_a_continued_item_never_badges_the_inbox_again(owner, tmp_path, monkeypatch):
    """Once the user replies in an item it is their chat: its own turns postdate the read watermark,
    which must not turn it back into an unread Inbox result."""
    from gateway.session_contract import SessionRef
    from gateway.session_inbox import continue_if_inbox, list_items, mark_read
    import gateway.session_policy_refresh as refresh
    monkeypatch.setattr(refresh, '_publish', lambda authority, ref, moved=False: None)
    sid = _filed_run(owner, tmp_path)
    await mark_read(_connection(owner), None, {'session_id': sid})
    await continue_if_inbox(_connection(owner), SessionRef(owner.profile_id, sid))
    from persistence.conversation.messages import NewMessage
    owner.persistence.messages.append(sid, NewMessage(role='user', content='thanks', timestamp=4e9))
    listed = await list_items(_connection(owner), None, {})
    assert listed['unread_count'] == 0 and listed['items'][0]['unread'] is False


@pytest.mark.asyncio
async def test_a_continued_item_keeps_an_explicit_mark_unread(owner, tmp_path, monkeypatch):
    from gateway.session_contract import SessionRef
    from gateway.session_inbox import continue_if_inbox, list_items, mark_read
    import gateway.session_policy_refresh as refresh
    monkeypatch.setattr(refresh, '_publish', lambda authority, ref, moved=False: None)
    sid = _filed_run(owner, tmp_path)
    await continue_if_inbox(_connection(owner), SessionRef(owner.profile_id, sid))
    await mark_read(_connection(owner), None, {'session_id': sid, 'read': False})
    assert (await list_items(_connection(owner), None, {}))['unread_count'] == 1


@pytest.mark.asyncio
async def test_a_conversion_stopped_before_its_policy_commit_finishes_on_the_next_reply(
        owner, tmp_path, monkeypatch):
    """A stop between the policy commit and the prompt/tool clear must not strand a half-converted chat
    (desktop policy, cron prompt): the clear lands first, so a stop there leaves the item on its cron
    policy and the next reply converts it fully."""
    from gateway.session_contract import SessionRef
    from gateway import session_inbox
    import gateway.session_policy_refresh as refresh
    monkeypatch.setattr(refresh, '_publish', lambda authority, ref, moved=False: None)
    sid = _filed_run(owner, tmp_path)
    ref = SessionRef(owner.profile_id, sid)
    real_clear = session_inbox._clear_prompt_snapshot

    def stopped(store, session_id):
        raise RuntimeError('gateway stopped')
    monkeypatch.setattr(session_inbox, '_clear_prompt_snapshot', stopped)
    with pytest.raises(RuntimeError):
        await session_inbox.continue_if_inbox(_connection(owner), ref)
    monkeypatch.setattr(session_inbox, '_clear_prompt_snapshot', real_clear)
    await session_inbox.continue_if_inbox(_connection(owner), ref)
    assert owner.persistence.bindings.receipt(sid).policy['source'] == 'gui'
    assert owner.persistence.sessions.get(sid).system_prompt is None


@pytest.mark.asyncio
async def test_two_first_replies_at_once_both_proceed(owner, tmp_path, monkeypatch):
    """The second first-reply loses the policy compare-and-swap to the first; the item is converted, so
    its message is admitted after the first instead of failing ``revision_conflict``."""
    import asyncio
    import threading
    from gateway.session_contract import SessionRef
    from gateway import session_inbox
    import gateway.session_policy_refresh as refresh
    monkeypatch.setattr(refresh, '_publish', lambda authority, ref, moved=False: None)
    sid = _filed_run(owner, tmp_path)
    ref = SessionRef(owner.profile_id, sid)
    both_read = threading.Barrier(2, timeout=20)  # both callers read the cron policy before either commits
    real_policy = session_inbox._interactive_policy

    def gated_policy(*args):
        both_read.wait()
        return real_policy(*args)
    monkeypatch.setattr(session_inbox, '_interactive_policy', gated_policy)
    await asyncio.wait_for(asyncio.gather(session_inbox.continue_if_inbox(_connection(owner), ref),
                                          session_inbox.continue_if_inbox(_connection(owner), ref)), 30)
    assert owner.persistence.bindings.receipt(sid).policy['source'] == 'gui'


@pytest.mark.asyncio
@pytest.mark.parametrize('identity,continues', [
    ({'provider': 'local', 'native_bootstrap': True}, False),  # CLI/TUI native ticket
    ({'user_id': 'server-internal', 'provider': 'server-internal'}, False),  # dashboard's embedded TUI
    ({'user_id': 'u1', 'provider': 'nous'}, True),  # Web app tab's socket ticket
])
async def test_only_a_desktop_renderer_continues_an_item_without_the_desktop_declaration(
        owner, tmp_path, monkeypatch, identity, continues):
    """A terminal client (native ticket without ``desktop_local``, or the embedded TUI's internal
    credential) holds the operator capability too: it must not turn the run into a Desktop chat. The
    same capability on a Web app socket does."""
    from gateway.session_contract import SessionRef
    from gateway.session_inbox import continue_if_inbox
    import gateway.session_policy_refresh as refresh
    monkeypatch.setattr(refresh, '_publish', lambda authority, ref, moved=False: None)
    sid = _filed_run(owner, tmp_path)
    connection = _connection(owner, desktop_local=False, capabilities=(
        'session:read', 'session:control', 'session:submit', 'session:operator'))
    connection.transport.auth_identity = identity
    await continue_if_inbox(connection, SessionRef(owner.profile_id, sid))
    assert owner.persistence.bindings.receipt(sid).policy['source'] == ('gui' if continues else 'cron')


@pytest.mark.parametrize('source,title,given,expected', [
    ('gui', 'standup chat', None, 'inbox'),
    ('gui', 'standup chat', 'origin', 'inbox'),
    ('gui', 'Bot Chat', None, None),
    ('tui', 'standup chat', None, None),
    ('gui', 'standup chat', 'telegram', 'telegram'),
])
def test_default_deliver_is_inbox_only_from_a_desktop_chat_that_is_not_bot_chat(
        source, title, given, expected, monkeypatch):
    from cron import inbox
    from gateway import session_policy
    monkeypatch.setattr(session_policy, 'active_policy', lambda: SimpleNamespace(source=source))
    monkeypatch.setattr(inbox, '_in_bot_chat', lambda: title == 'Bot Chat')
    assert inbox.default_deliver(given) == (expected if expected is not None else given)
