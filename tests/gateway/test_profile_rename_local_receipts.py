"""Regression for #143333: a renamed profile keeps its gateway-owned chats.

``hermes profile rename`` moves ``profiles/<old>`` to ``profiles/<new>``. The authority serving a profile is
owned by its resolved home path (``gateway/run_runtime.py``), and each LOCAL creation receipt names that
owner, the ``agent:<old>:`` route and origin profile, and a frozen credential reference scoped to the home;
its legacy bindings, API bindings and history-control bindings name the same owner. A rename that left them
behind stranded every Desktop/Webapp/CLI chat of the profile: restore refused each with ``profile_mismatch``,
and ``hermes profile migrate-identity`` reported success without repairing it.
"""
import asyncio
import json
import os
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner

LEGACY_SID = '20261022_101500_a1b2c3'
HISTORY_SID = '20261021_090000_d4e5f6'
API_SID = 'api-chat'
_CONFIG = ('model:\n  default: frozen-model\n  provider: custom\n  base_url: http://127.0.0.1:9/v1\n'
           '  api_key: sk-rename-test\n')


def _actor(authority, transport='socket'):
    from gateway.session_contract import Principal
    return Principal('owner', authority.profile_id,
                     frozenset({'session:create', 'session:read', 'session:control'}), transport)


def _own_api_and_history(authority):
    """An API session, and an unrouted CLI chat whose sidebar control the owner claimed (pin/archive)."""
    from gateway.session_api import bind_api_session
    from hermes_state_mutation_binding import authorize_history
    authority.db.create_session(API_SID, source='api_server')
    bind_api_session(authority, API_SID)
    authority.db.create_session(HISTORY_SID, source='cli')
    assert authority.db._execute_write(lambda conn: authorize_history(conn, _actor(authority), HISTORY_SID, claim=True))


def _assert_api_and_history(authority, name):
    from gateway.session_api import restore_api_session
    from gateway.session_contract import SessionRef
    from hermes_state_mutation_binding import authorize_history
    authority.sessions.pop(API_SID, None)
    assert restore_api_session(authority, API_SID) == SessionRef(authority.profile_id, API_SID)
    assert authority.sessions[API_SID].route.startswith(f'agent:{name}:api_server:')
    with authority.db._read_ctx() as conn:
        assert authorize_history(conn, _actor(authority), HISTORY_SID)


def _create_chats(tmp_path, authority, home):
    """A native Desktop chat, an adopted pre-gateway TUI chat and one imported from another machine's
    profile of the same name (its receipt names that other home), whose ids it returns; plus the API
    session and claimed history of ``_own_api_and_history``."""
    from gateway.run import _profile_runtime_scope
    from gateway.session_local import create_local_session
    from gateway.session_local_migration import bind_native_transport, resolve_local_target
    from hermes_state_local import POLICY_PREFIX, local_receipt
    _own_api_and_history(authority)
    with _profile_runtime_scope(home, {}):
        native = create_local_session(authority, _actor(authority), {
            'request_id': 'desktop-chat', 'source': 'gui', 'cwd': str(tmp_path), 'toolsets': []}).session_id
        foreign = create_local_session(authority, _actor(authority), {
            'request_id': 'imported-chat', 'source': 'gui', 'cwd': str(tmp_path), 'toolsets': []}).session_id
        native_actor = _actor(authority, 'native')
        bind_native_transport(authority, native_actor, {
            'provider': 'local', 'profile_id': authority.profile_id, 'instance_id': authority.instance_id})
        authority.db.create_session(LEGACY_SID, source='tui', model='frozen-model',
                                    model_config={'model': 'frozen-model'})
        resolve_local_target(authority, native_actor, LEGACY_SID)
    assert local_receipt(authority.db, native)['policy']['config_secret_ref'].startswith('profile-config-v1:')
    assert 'legacy_session_id' in local_receipt(authority.db, LEGACY_SID)
    receipt = local_receipt(authority.db, foreign)
    receipt['profile_id'] = str(tmp_path / 'elsewhere' / '.hermes' / 'profiles' / 'alpha')
    authority.db._execute_write(lambda conn: conn.execute(
        'UPDATE state_meta SET value=? WHERE key=?', (json.dumps(receipt), POLICY_PREFIX + foreign)))
    return native, LEGACY_SID, foreign


def _close(runner):
    for authority in getattr(runner, 'session_authorities', None) or ():
        authority.db.close()


def _assert_restored(authority, home, name, sids):
    from gateway.run import _profile_runtime_scope
    from gateway.session_contract import SessionRef
    from gateway.session_local_recovery import restore_local_session
    from gateway.session_policy import restore_policy
    from gateway.session_policy_credentials import recover_config_secrets
    from hermes_state_local import local_receipt
    for sid in sids:
        assert sid in authority.sessions, f'{sid} was not restored by recovery'
        with _profile_runtime_scope(home, {}):
            assert restore_local_session(authority, sid) == SessionRef(authority.profile_id, sid)
        assert authority.sessions[sid].route.startswith(f'agent:{name}:local:dm:')
        policy = restore_policy(local_receipt(authority.db, sid)['policy'])
        if policy.config_secret_ref:
            with _profile_runtime_scope(home, {}):
                assert recover_config_secrets(authority, policy)[('model', 'api_key')] == 'sk-rename-test'


@pytest.mark.asyncio
@pytest.mark.parametrize('names, linked', [(('beta',), False), (('beta', 'gamma'), False),
                                           (('beta', 'alpha'), False), (('beta',), True)],
                         ids=['one-rename', 'two-renames', 'renamed-back', 'symlinked-profiles-dir'])
async def test_a_renamed_profile_restores_its_local_chats_and_their_credentials(tmp_path, monkeypatch, names, linked):
    """The CLI's own rename (no live multiplexer): the next gateway boot restores every chat of the
    renamed profile, credentials included, and still refuses a receipt that names another home. With
    ``profiles/`` a symlink, every owner is spelled through its target."""
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_local_recovery import restore_local_session
    from hermes_cli.profiles import rename_profile
    from hermes_state_local import local_receipt
    from hermes_state_runtime import RuntimeStoreError

    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    (homes[1][1] / 'config.yaml').write_text(_CONFIG, encoding='utf-8')
    if linked:
        os.rename(root / 'profiles', tmp_path / 'profiles-store')
        os.symlink(tmp_path / 'profiles-store', root / 'profiles', target_is_directory=True)
    process_ownership.reserve([home for _, home in homes])
    first = _runner(root, homes)
    try:
        await initialize_gateway_runtime(first)
        authority = first.session_authorities.for_home(homes[1][1])
        native, legacy, foreign = _create_chats(tmp_path, authority, homes[1][1])
        foreign_receipt = local_receipt(authority.db, foreign)
    finally:
        _close(first)
        for _, home in homes:
            process_ownership.release(home)

    with ExitStack() as stack:
        stack.enter_context(patch('hermes_cli.profiles.check_alias_collision', return_value='skip'))
        stack.enter_context(patch('hermes_cli.profiles._live_default_multiplexer', return_value=False))
        old = 'alpha'
        for new in names:
            rename_profile(old, new)
            old = new
    renamed = [('default', root), (old, root / 'profiles' / old)]
    process_ownership.reserve([home for _, home in renamed])
    second = _runner(root, renamed)
    try:
        await initialize_gateway_runtime(second)
        restored = second.session_authorities.for_home(renamed[1][1])
        _assert_restored(restored, renamed[1][1], old, [native, legacy])
        _assert_api_and_history(restored, old)
        assert foreign not in restored.sessions
        with pytest.raises(RuntimeStoreError) as refused:
            restore_local_session(restored, foreign)
        assert refused.value.reason == 'profile_mismatch'
        assert local_receipt(restored.db, foreign)['profile_id'] == foreign_receipt['profile_id']
    finally:
        _close(second)
        for _, home in renamed:
            process_ownership.release(home)


@pytest.mark.asyncio
async def test_migrate_identity_restores_the_chats_of_an_already_served_renamed_profile(tmp_path, monkeypatch):
    """A live multiplexer already serving the new name (a rescan won the race, or the user retries with
    ``hermes profile migrate-identity``) refused those chats when it served them. The control verb
    re-owns them and brings them back on that live authority, without a restart or a re-serve."""
    from gateway.run_profile_reconcile import migrate_profile_identity_verb
    from gateway.run_runtime import (initialize_gateway_runtime, release_profile_home, reserve_profile_home,
                                     serve_profile_runtime, unserve_profile_runtime)
    from gateway.runtime_ownership import process_ownership

    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    alpha = homes[1][1]
    (alpha / 'config.yaml').write_text(_CONFIG, encoding='utf-8')
    beta = root / 'profiles' / 'beta'
    process_ownership.reserve([home for _, home in homes])
    runner = _runner(root, homes)
    try:
        await initialize_gateway_runtime(runner)
        native, legacy, _foreign = _create_chats(tmp_path, runner.session_authorities.for_home(alpha), alpha)
        # What `hermes profile rename` does to a live multiplexer: unserve the old name, move the
        # directory, and serve the new one.
        assert await unserve_profile_runtime(runner, alpha)
        release_profile_home(runner, alpha)
        os.rename(alpha, beta)
        reserve_profile_home(runner, 'beta', beta)
        authority = await serve_profile_runtime(runner, 'beta', beta)
        assert native not in authority.sessions  # served before its identity moved

        answer = await asyncio.to_thread(migrate_profile_identity_verb(runner), {'old': 'alpha', 'new': 'beta'})
        assert answer['ok'] is True, answer
        _assert_restored(authority, beta.resolve(), 'beta', [native, legacy])
    finally:
        _close(runner)
        for home in (root, alpha, beta):
            process_ownership.release(home)
