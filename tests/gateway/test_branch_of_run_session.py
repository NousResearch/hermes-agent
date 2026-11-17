"""A branch of a run session (a cron fire) is an ordinary chat, not another run.

``hermes_state_mutation_branch._child_policy`` used to copy the parent's frozen policy verbatim, and
a cron policy freezes WHICH fire it is (``cron_job`` / ``extra_prompt`` / the fire's ``request_id``)
beside ``source``/``platform``. The child therefore looked like a run session: on its next submit,
``gateway.session_cron.execute`` compared the typed admission with that frozen fire, matched none of
its three fields, and settled ``cron_session_read_only`` — the "this chat belongs to a scheduled job"
card, on a chat the user had just created by branching. See NousResearch/hermes-agent#151857.

These tests drive the real receipt commit over a real ``SessionDB``: a cron run session branched by
a person must come out as the requesting person's interactive chat, while every other branch keeps
exactly the parent's policy (no surface is inferred for ordinary chats).
"""
from __future__ import annotations

import json

import pytest

# Import-time code in this chain reads the checkout's manifest (``hermes_bootstrap``); do it at
# collection, which runs before ``tests/home_io_guard`` installs — as tests/gateway/
# test_cron_admission_error_surface.py does for the same reason.
import agent.conversation_loop


def _run_policy_payload(**overrides):
    """A ``gateway.local_policy.v1`` receipt's policy, shaped as ``gateway/session_cron._create``
    freezes it: the fire's job, its extra prompt, and the fire's own request id."""
    policy = {
        'source': 'cron',
        'platform': 'cron',
        'cwd': '/work/job',
        'model': 'some-model',
        'toolsets': ['plm'],
        'skills_prompt': None,
        'config_json': json.dumps({'model': {'default': 'some-model'},
                                   'desktop_local': True,
                                   'follow_profile_config': True}),
        'request_json': json.dumps({'cron_job': {'id': 'job-1', 'name': 'nightly'},
                                    'extra_prompt': None,
                                    'request_id': 'cron:job-1:fire-1'}),
        'terminal_json': json.dumps({'TERMINAL_ENV': 'local', 'TERMINAL_CWD': '/work/job'}),
        'edit_approval_policy': 'ask',
        'editor_mcp_json': None,
        'config_secret_ref': None,
        'credential_ref': None,
        'kanban_json': None,
        'safe_mode': False,
        'ignore_user_config': False,
        'coding_posture': False,
        'client_env_json': None,
        'client_env_remote': False,
        'client_env_rules_json': None,
    }
    policy.update(overrides)
    return policy


def _chat_policy_payload():
    """The same receipt for an ordinary Desktop chat: already interactive, already the caller's."""
    return _run_policy_payload(source='gui', platform='desktop',
                               request_json=json.dumps({'source': 'gui', 'cwd': '/work/job',
                                                        'model': 'some-model'}),
                               config_json=json.dumps({'model': {'default': 'some-model'}}))


def _seed(db, policy, principal='cron-owner', title='nightly'):
    """A parent local session at its real identity: ``local_identity(profile, principal, request)``.

    Returns ``(session_id, request_id)``.
    """
    from hermes_state_keys import cron_request_id, local_identity
    from hermes_state_local import POLICY_PREFIX
    request_id = cron_request_id('job-1', 'fire-1')
    sid = local_identity('default', principal, request_id)
    db.create_session(sid, source=policy['source'], model=policy['model'])
    db.set_session_title(sid, title)
    entry = {'session_key': f'agent:main:local:dm:{sid}', 'session_id': sid, 'updated_at': 0.0,
             'origin': {'platform': 'local', 'chat_id': sid, 'user_id': principal, 'chat_type': 'dm',
                        'profile': None}}
    receipt = {'profile_id': 'default', 'principal_id': principal, 'request_id': request_id,
               'session_id': sid, 'route': entry['session_key'], 'entry': entry, 'policy': policy}

    def _do(conn):
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                     (POLICY_PREFIX + sid, json.dumps(receipt)))
        conn.execute("UPDATE sessions SET session_key=?,chat_id=?,user_id=?,origin_json=? WHERE id=?",
                     (entry['session_key'], sid, principal, json.dumps(entry['origin']), sid))
    db._execute_write(_do)
    db.append_message(sid, 'user', 'summarize the inbox')
    db.append_message(sid, 'assistant', 'Inbox summary: nothing urgent.')
    return sid


@pytest.fixture()
def db(tmp_path, monkeypatch):
    # The branch commit imports gateway/agent modules whose payload bootstrap reads HERMES_HOME;
    # point it at the temp home first, as every gateway test does (tests/home_io_guard.py).
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from hermes_state import SessionDB
    database = SessionDB(tmp_path / 'state.db')
    yield database
    database.close()


def _branch(db, parent, *, principal=None, source=None):
    from hermes_state_runtime import begin_runtime_epoch, mutate_runtime_session
    epoch = begin_runtime_epoch(db, instance_id='test')
    context = {'principal_id': principal, 'source': source}
    return mutate_runtime_session(
        db, epoch=epoch, principal_id='sid:S-1-5-21-1-2-3-1001', session_id=parent,
        request_id='branch:1', expected_revision=0, expected_generation=0, operation='branch',
        payload={}, _branch_context=context)


def test_branch_of_a_run_session_is_an_interactive_chat(db):
    """The child of a branched cron run is the caller's chat: interactive surface, no frozen fire."""
    parent = _seed(db, _run_policy_payload())
    caller = 'sid:S-1-5-21-1-2-3-1001'

    result = _branch(db, parent, principal=caller, source='gui')
    child = result['branched_session_id']

    from hermes_state_local import local_receipt
    receipt = local_receipt(db, child)
    policy, request = receipt['policy'], json.loads(receipt['policy']['request_json'])

    # Not a run: the executor's cron branch (and its read-only refusal) can never apply to it.
    assert policy['source'] == 'gui'
    assert policy['platform'] == 'desktop'
    assert 'cron_job' not in request and 'extra_prompt' not in request
    # The fire's admission identity is not carried over: the child is its own creation.
    assert 'cron:job-1' not in receipt['policy']['request_json']
    assert 'request_id' not in request
    assert receipt['request_id'] != 'cron:job-1:fire-1'
    # Owned by the person who branched it, not by the job scheduler.
    assert receipt['principal_id'] == caller
    assert receipt['entry']['origin']['user_id'] == caller
    row = db.get_session(child)
    assert row['source'] == 'gui'
    assert row['created_source'] == 'desktop'
    assert row['user_id'] == caller
    # Everything needed to CONTINUE the run's work survives: route, tools, folder, transcript.
    assert policy['toolsets'] == ['plm']
    assert policy['cwd'] == '/work/job'
    assert policy['model'] == 'some-model'
    assert row['message_count'] == 2
    # A frozen config must not re-read the profile on the child's first turn.
    config = json.loads(policy['config_json'])
    assert 'follow_profile_config' not in config and 'desktop_local' not in config


def test_branch_of_a_run_session_never_keeps_the_scheduler_as_owner(db):
    """No caller identity -> refuse. Minting a chat owned by ``cron-owner`` would be nobody's chat."""
    parent = _seed(db, _run_policy_payload())
    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError):
        _branch(db, parent, principal=None, source='gui')
    with pytest.raises(RuntimeStoreError):
        _branch(db, parent, principal='someone', source='telegram')


def test_branch_of_an_interactive_chat_keeps_its_policy(db):
    """An ordinary chat's branch is unchanged: no surface is invented, ownership is the parent's."""
    parent = _seed(db, _chat_policy_payload(), principal='owner-a', title='chat')
    result = _branch(db, parent, principal='owner-a', source='cli')

    from hermes_state_local import local_receipt
    receipt = local_receipt(db, result['branched_session_id'])
    assert receipt['policy']['source'] == 'gui'
    assert receipt['policy']['platform'] == 'desktop'
    assert receipt['policy']['request_json'] == _chat_policy_payload()['request_json']
    assert receipt['principal_id'] == 'owner-a'
    assert db.get_session(result['branched_session_id'])['source'] == 'gui'