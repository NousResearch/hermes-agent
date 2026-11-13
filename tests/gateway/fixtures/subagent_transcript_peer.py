"""Authenticated owner fixture: a viewer reads a delegated child's stored transcript (``session.transcript``
with ``subagent_id``; adapted from #140477's ``subagent.transcript`` fixture).

Real gateway runner, session authority, ticketed WebSocket and the profile's own ``state.db``
under a temp HERMES_HOME; no inference. Child rows are written the way a delegated child writes
them (``source='subagent'``, ``_delegate_from`` + ``_subagent_id`` markers, messages as it runs).
Every reply lands in ``receipt.json`` for the test to judge.
"""
import asyncio
import json
import os
from pathlib import Path
import sys
import traceback
import weakref


def seed(db, target):
    """The conversation's children as the store holds them; returns the row ids per child."""
    def child(sid, parent, subagent_id, *, source='subagent', **cfg):
        db.create_session(sid, source=source, parent_session_id=parent,
                          model_config={'_delegate_from': parent, '_subagent_id': subagent_id, **cfg})

    # A foreign conversation's child that happens to draw the same subagent_id, stored FIRST.
    db.create_session('foreign-parent', source='cli')
    child('foreign-done', 'foreign-parent', 'sa-1-done')
    child('foreign-child', 'foreign-parent', 'sa-0-foreign')
    child('live-child', target, 'sa-0-live')
    child('done-child', target, 'sa-1-done')
    child('empty-child', target, 'sa-2-empty')
    # The finished child compressed once: the continuation inherits its model_config.
    db.append_message('done-child', role='user', content='done goal')
    db.end_session('done-child', 'compression')
    db.create_session('done-child-2', source='subagent', parent_session_id='done-child',
                      model_config={'_delegate_from': target, '_subagent_id': 'sa-1-done'})
    # An orchestrator child's own worker, delegated from the child's first segment.
    child('nested-child', 'done-child', 'sa-0-nested')
    # A running orchestrator child and its own running worker (the rebuilt-parent case).
    child('orch-child', target, 'sa-1-orch')
    child('deep-child', 'orch-child', 'sa-0-deep')
    # A branch of this conversation is another conversation: its children are not ours.
    db.create_session('branch', source='gui', parent_session_id=target, model_config={'_branched_from': target})
    child('branch-child', 'branch', 'sa-0-branch')
    rows = {}
    # The delegating conversation's own rows never join a child's transcript.
    db.append_message(target, role='user', content='PARENT_PRIVATE')
    # A queued child: its row is reserved with no model_config (``_reserve_child_transcript``), the markers
    # still in memory, and the delegating conversation then rotates through compression.
    db.create_session('rotating-parent', source='cli')
    db.append_message('rotating-parent', role='user', content='DELEGATOR_PRIVATE')
    db.create_session('queued-child', source='subagent', parent_session_id='rotating-parent')
    db.end_session('rotating-parent', 'compression')
    for sid, texts in [('live-child', ['live goal', 'live step one']),
                       ('done-child-2', ['done summary', 'done final']),
                       ('nested-child', ['nested goal', 'nested final']), ('deep-child', ['deep goal']),
                       ('foreign-done', ['FOREIGN_PRIVATE']),
                       ('foreign-child', ['FOREIGN_PRIVATE']), ('branch-child', ['BRANCH_PRIVATE'])]:
        rows[sid] = [db.append_message(sid, role='user' if i == 0 else 'assistant', content=text)
                     for i, text in enumerate(texts)]
    return rows


async def probe():
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_agent.gateway.session_ancillary import physical_target
    from hermes_agent.gateway.session_authority import initialize_session_authority
    from hermes_agent.gateway.run_api import start_gateway_api, stop_gateway_api
    from hermes_agent.gateway.session_contract import SessionRef
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
    from tests.gateway.fixtures.local_recovery_probe import rpc
    from hermes_agent.tools.delegate_tool_child_run import _register_child
    from hermes_agent.tools.delegate_tool_registry import _unregister_subagent
    from websockets.asyncio.client import connect

    home = Path(os.environ['HERMES_HOME'])
    (home / 'config.yaml').write_text('gateway:\n  multiplex_profiles: false\nmodel:\n  provider: custom\n  default: no-inference\n  base_url: http://127.0.0.1:1/v1\nauxiliary:\n  title_generation:\n    enabled: false\n')
    runner = GatewayRunner()
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='subagent-transcript')
    api = await start_gateway_api(runner)
    web_server.app.state.auth_required = True
    receipt = {}

    class Agent:
        pass

    def socket(actor):
        ticket = mint_ticket(user_id=actor, provider='fixture')
        return connect(api.api_origin.replace('http:', 'ws:') + '/api/ws?ticket=' + ticket)

    async def transcript(ws, sid, subagent_id, **extra):
        reply = await rpc(ws, 'session.transcript', session_id=sid, subagent_id=subagent_id, **extra)
        return reply.get('result', reply.get('error'))

    from hermes_agent.gateway import session_stored_reads
    full_reads, window = [0], session_stored_reads._window

    def counted(db, target, latest, after, *args, **kwargs):
        full_reads[0] += after is None  # a read from the first row
        return window(db, target, latest, after, *args, **kwargs)
    session_stored_reads._window = counted
    try:
        async with socket('owner') as ws, socket('peer_operator') as peer:
            created = await rpc(ws, 'session.create', request_id='transcript-owner', source='gui', toolsets=[])
            sid = created['result']['session_id']
            live = authority.sessions[sid]
            target = physical_target(authority, SessionRef(str(home), sid))
            receipt['rows'] = seed(authority.db, target)
            parent = Agent()
            parent.session_id = target
            with runner._agent_cache_lock:
                runner._agent_cache[live.route] = (parent, 'fixture')
            running = Agent()
            running._subagent_id, running.session_id = 'sa-0-live', 'live-child'
            running._delegate_parent_ref = weakref.ref(parent)
            running._parent_session_id = target
            _register_child(running, parent, 'live goal', owner_session_id=None,
                            owner_transport=None, owner_session_record=None)
            # Queued in the roster, its reserved row parented on a delegator segment that has since rotated.
            queued = Agent()
            queued._subagent_id, queued.session_id = 'sa-3-queued', 'queued-child'
            queued._delegate_parent_ref, queued._parent_session_id = weakref.ref(parent), 'rotating-parent'
            _register_child(queued, parent, 'queued goal', owner_session_id=None,
                            owner_transport=None, owner_session_record=None)
            try:
                receipt['queued_child'] = await transcript(ws, sid, 'sa-3-queued')
            finally:
                _unregister_subagent('sa-3-queued', agent=queued)
            try:
                roster = await rpc(ws, 'subagent.list', session_id=sid)
                receipt['list'] = roster.get('result', roster.get('error'))
                first = receipt['live'] = await transcript(ws, sid, 'sa-0-live')
                receipt['empty_cursor_probe'] = await transcript(ws, sid, 'sa-2-empty')
                receipt['peer_live'] = await transcript(peer, sid, 'sa-0-live')
                # The child writes on while the viewer polls from the cursor it was handed.
                receipt['rows']['live-child'].append(authority.db.append_message(
                    'live-child', role='assistant', content='live step two'))
                polled_from = full_reads[0]
                receipt['live_after'] = await transcript(ws, sid, 'sa-0-live', cursor=first['next_cursor'])
                receipt['live_caught_up'] = await transcript(ws, sid, 'sa-0-live',
                                                             cursor=receipt['live_after']['next_cursor'])
                receipt['full_reads_while_polling'] = full_reads[0] - polled_from
                # The child compacts in place (the store write in-place compression commits): its id stays,
                # the carried tail gets fresh row ids and the rows the viewer holds become archived.
                authority.db.archive_and_compact('live-child', [
                    {'role': 'user', 'content': 'live summary'}, {'role': 'assistant', 'content': 'live step two'}],
                    tail_count=1)
                receipt['live_compacted'] = await transcript(ws, sid, 'sa-0-live',
                                                             cursor=receipt['live_caught_up']['next_cursor'])
                receipt['live_compacted_caught_up'] = await transcript(
                    ws, sid, 'sa-0-live', cursor=receipt['live_compacted']['next_cursor'])
                # A reaction rewrites only a held row's display_metadata (no new row, no deactivation).
                reacted = next(m['row_id'] for m in receipt['live_compacted']['rows']
                               if m.get('text') == 'live summary')
                authority.db.set_message_reaction('live-child', reacted, '\U0001f525')
                receipt['live_reacted'] = await transcript(
                    ws, sid, 'sa-0-live', cursor=receipt['live_compacted_caught_up']['next_cursor'])
                # Cursors the owner never issued (another child's, the parent's own, junk) are invalid_params,
                # never a server error.
                parent_cursor = (await rpc(ws, 'session.transcript', session_id=sid))['result']['next_cursor']
                receipt['odd_cursors'] = [await transcript(ws, sid, 'sa-0-live', cursor=bad) for bad in (
                    receipt['empty_cursor_probe']['next_cursor'], parent_cursor, '\u00b2.0.0.0.live-child')]
                # The parent agent is rebuilt (/model, a policy refresh): the orchestrator is still found by
                # its durable owner id, and its worker (owner id: the orchestrator) by its spawn ancestry.
                old_parent, rebuilt = parent, Agent()
                rebuilt.session_id = target
                with runner._agent_cache_lock:
                    runner._agent_cache[live.route] = (rebuilt, 'fixture')
                orch, deep = Agent(), Agent()
                orch._subagent_id, orch.session_id, orch._parent_session_id = 'sa-1-orch', 'orch-child', target
                deep._subagent_id, deep.session_id, deep._parent_session_id = 'sa-0-deep', 'deep-child', 'orch-child'
                orch._delegate_parent_ref, deep._delegate_parent_ref = weakref.ref(old_parent), weakref.ref(orch)
                orch._session_db = deep._session_db = authority.db
                deep._delegate_depth = 2
                for agent, goal in ((orch, 'orch goal'), (deep, 'deep goal')):
                    _register_child(agent, agent._delegate_parent_ref(), goal, owner_session_id=None,
                                    owner_transport=None, owner_session_record=None)
                receipt['rebuilt_orch'] = await transcript(ws, sid, 'sa-1-orch')
                receipt['rebuilt_deep'] = await transcript(ws, sid, 'sa-0-deep')
                rebuilt_list = await rpc(ws, 'subagent.list', session_id=sid)
                receipt['rebuilt_list'] = sorted(r['subagent_id'] for r in rebuilt_list['result']['subagents'])
            finally:
                _unregister_subagent('sa-0-live', agent=running)
                for name in ('sa-1-orch', 'sa-0-deep'):
                    _unregister_subagent(name)
            # A child read before it wrote anything, then appended and compacted before the next poll.
            receipt['empty'] = await transcript(ws, sid, 'sa-2-empty')
            for i, text in enumerate(['empty goal', 'empty step']):
                authority.db.append_message('empty-child', role='user' if i == 0 else 'assistant', content=text)
            authority.db.archive_and_compact('empty-child', [{'role': 'user', 'content': 'empty summary'}])
            receipt['empty_compacted'] = await transcript(ws, sid, 'sa-2-empty', cursor=receipt['empty']['next_cursor'])
            # Ended: gone from the registry, still readable from the store.
            receipt['live_ended'] = await transcript(ws, sid, 'sa-0-live')
            for name in ('sa-1-done', 'sa-0-nested', 'sa-0-foreign', 'sa-0-branch', 'sa-9-unknown'):
                receipt[name] = await transcript(ws, sid, name)
            # A reaction on a row of the segment before the rotation: shown archived, owned by the ancestor.
            ancestor = next(m['row_id'] for m in receipt['sa-1-done']['rows'] if m.get('text') == 'done goal')
            authority.db.set_message_reaction('done-child-2', ancestor, '\U0001f44d')
            receipt['done_ancestor_reacted'] = await transcript(ws, sid, 'sa-1-done',
                                                                cursor=receipt['sa-1-done']['next_cursor'])
            receipt['unknown_session'] = await transcript(ws, 'no-such-session', 'sa-1-done')
            receipt['foreign_profile'] = await transcript(ws, sid, 'sa-1-done', profile='foreign')
            receipt['bad_cursor'] = await transcript(ws, sid, 'sa-1-done', cursor=-1)
            described = await rpc(ws, 'runtime.describe')
            receipt['capabilities'] = described['result']['capabilities']
    finally:
        await stop_gateway_api(api)
    (home / 'receipt.json').write_text(json.dumps(receipt))


if __name__ == '__main__':
    status = 0
    try:
        asyncio.run(probe())
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
