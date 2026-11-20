"""A named profile whose ``profiles/<name>`` folder is a link (a symlink, or a Windows junction) chats and is
listed under its own name. Regression for #142944.

The multiplexer serves a linked profile at its resolved home, outside ``profiles/``. That path no longer read as
``profiles/<name>``, so the gateway named the home ``default``: the session snapshot Desktop routes by sent every
prompt to the default profile's authority (``permission_denied``), and the Bots roster listed the profile as a
second ``default``. Driven through Desktop's host-scoped socket against a real in-process multiplexer
(``test_hosted_group_members._gateway``); only the turn executor is stubbed.
"""
from contextlib import contextmanager

from gateway import session_finite
from tests.gateway.test_hosted_group_members import _gateway
from tests.waiting import wait_until


@contextmanager
def _desktop(tmp_path, monkeypatch):
    """Desktop's host-scoped socket on a multiplexer serving ``default``, ``alpha`` and a linked ``linked``."""
    from gateway.session_controls import AuthorityConnection
    with _gateway(tmp_path, monkeypatch, served=('alpha',), linked=('linked',)) as (runner, homes, call):
        connection = AuthorityConnection(runner.session_authority, object(),
                                         {'user_id': 'desktop', 'profile_scope': 'host'})

        def request(active, method, **params):
            return call(connection.dispatch({'id': 1, 'method': method, 'params': {'profile': active, **params}}))
        try:
            yield request, homes
        finally:
            call(connection.close())


def test_a_linked_profile_chats_under_the_name_it_is_served_under(tmp_path, monkeypatch):
    ran = []

    async def execute(authority, ref, row):
        ran.append(authority.profile_id)
        return 'reply'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    with _desktop(tmp_path, monkeypatch) as (request, homes):
        created = request('linked', 'session.create', request_id='r', source='gui')
        info = created['result']['info']
        # Desktop adopts this label and routes every later request of the session by it.
        assert info['profile_name'] == 'linked', info
        submitted = request(info['profile_name'], 'prompt.submit', session_id=created['result']['session_id'],
                            text='hi', submission_id='s')
        assert submitted.get('result', {}).get('status') == 'queued', submitted
        wait_until(lambda: ran, what="the linked profile's turn ran")
        assert ran == [str(homes['linked'].resolve())]


def test_the_bots_roster_lists_a_linked_profile_once_under_its_name(tmp_path, monkeypatch):
    with _desktop(tmp_path, monkeypatch) as (request, _homes):
        roster = request('default', 'profiles.list', include_sessions=False)
        assert sorted(row['name'] for row in roster['result']['profiles']) == ['alpha', 'default', 'linked']
