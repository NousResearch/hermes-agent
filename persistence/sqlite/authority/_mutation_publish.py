"""The mutation commands that publish a new physical target or route inside the receipt write: compress commits a
prepared summary, branch an independent local route with its copied transcript and frozen policy, reset a fresh
target for a local conversation."""

import json
import time
import uuid

from persistence.authority.bindings import local_session_id
from persistence.core.errors import AuthorityRefusal
from session_rules.route_entries import Platform, SessionEntry, SessionSource, _now, build_session_key

from .._schema import TITLE_SOURCE_DERIVED, TITLE_SOURCE_USER
from ._runtime import _json
from ._runtime_guards import require_idle, require_target_advanceable
from ._runtime_picks import validate_prepared


def _profile_from_session_key(session_key):
    """Profile namespace encoded in an ``agent:<ns>:...`` gateway session key, else None: ``"default"`` for
    ``main``, ``"main"`` for the marked ``main~``, else the slot is the profile id."""
    if not session_key:
        return None
    parts = str(session_key).split(":")
    if len(parts) < 2 or parts[0] != "agent":
        return None
    namespace = parts[1] or "main"
    if namespace == "main":
        return "default"
    return "main" if namespace == "main~" else namespace


def compress_in_transaction(holder, conn, session_id, payload, prepared):
    snapshot = validate_prepared(holder, conn, session_id, prepared)
    policy = snapshot['receipt']['policy']
    target = snapshot['target']
    from ..workers._compression import archive_on_connection, publish_on_connection
    # No worker/turn/compactor can be active: validate_prepared checks those
    # obligations on this exact connection. No separate transaction runs.
    if prepared['in_place']:
        child = target
        archive_on_connection(holder, conn, target, prepared['messages'])
        affected = {session_id, target}
    else:
        child = uuid.uuid4().hex
        publish_on_connection(holder, conn, parent_session_id=target, child_session_id=child,
            source=policy['source'], model=policy['model'], messages=prepared['messages'],
            model_config={'_compressed_from': target}, cwd=policy['cwd'], require_compression_lease=False)
        affected = set()  # Rotation publication already advances the logical revision.
    conn.execute('UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', (session_id,))
    generation = conn.execute('SELECT runtime_generation FROM sessions WHERE id=?', (session_id,)).fetchone()[0]
    return affected, {'target_session_id': child, 'previous_target_session_id': target,
                   'execution_generation': generation, 'message_count': len(prepared['messages']),
                   'summary': prepared.get('summary')}


def _branch_boundary(conn, target, through):
    """Highest row id the branch copies. ``through`` must name an active row of the physical
    target (after compression that is the continuation, whose ids the viewer was shown); rows
    appended later, even in this same commit window, have larger ids and stay out. The tool
    results that directly follow an assistant boundary answer its calls, so they belong to it."""
    if through is None:
        return conn.execute('SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id=? AND active=1',
                            (target,)).fetchone()[0]
    row = conn.execute('SELECT role FROM messages WHERE id=? AND session_id=? AND active=1',
                       (through, target)).fetchone()
    if row is None:
        raise AuthorityRefusal('invalid_params')
    if row['role'] == 'assistant':
        following = conn.execute('SELECT id, role FROM messages WHERE session_id=? AND active=1 AND id>? '
                                 'ORDER BY id', (target, through))
        for later in following:
            if later['role'] != 'tool':
                break
            through = later['id']
    return through


def _child_policy(policy, child, prepared):
    """The parent's frozen policy, except editor MCP servers: the gateway prepared the child's own
    selection under the child's own scope (gateway/session_local_mcp.py ``prepare_branch_editor``).
    Copying the parent's would share its scope, borrowed credentials and live connections.
    A branch starts at Default (``ask``), as main's ACP ``fork_session`` did."""
    if 'edit_approval_policy' in policy:
        policy = {**policy, 'edit_approval_policy': 'ask'}
    if prepared is None:
        if policy.get('editor_mcp_json'):
            raise AuthorityRefusal('runtime_coordination_required')
        return policy
    # Prepared against the parent's selection and the child id this commit derives; anything else moved.
    if child != prepared['child'] or policy.get('editor_mcp_json') != prepared['parent_editor_mcp_json']:
        raise AuthorityRefusal('revision_conflict')
    return {**policy, 'editor_mcp_json': prepared['editor_mcp_json'], 'toolsets': prepared['toolsets'],
            'config_secret_ref': prepared['config_secret_ref']}


def branch_in_transaction(holder, conn, session_id, payload, prepared=None):
    from ..conversation._store.local import POLICY_PREFIX
    from ..conversation._store.local_lineage import validate_local_lineage
    row = conn.execute('SELECT value FROM state_meta WHERE key=?',
                       (POLICY_PREFIX + session_id,)).fetchone()
    if row is None:
        raise AuthorityRefusal('runtime_coordination_required')
    saved = json.loads(row[0])
    target = validate_local_lineage(conn, saved)
    require_idle(holder, conn, list({session_id, target}))
    request_id = prepared['request_id'] if prepared else 'branch:' + uuid.uuid4().hex
    child = local_session_id(saved['profile_id'], saved['principal_id'], request_id)
    policy = _child_policy(saved['policy'], child, prepared)
    # The child keeps the parent's owning profile: a served secondary's origin names it
    # (``local_source``), and restore compares the stored origin against that exactly.
    source = SessionSource(Platform.LOCAL, child, user_id=saved['principal_id'], chat_type='dm',
                           profile=saved['entry']['origin'].get('profile'))
    route = build_session_key(source, profile=_profile_from_session_key(saved['route']))
    now = _now()
    entry = SessionEntry(route, child, now, now, origin=source, platform=Platform.LOCAL)
    parent = conn.execute('SELECT * FROM sessions WHERE id=?', (target,)).fetchone()
    from ..conversation._store.compression import _publish_child_session_row
    _publish_child_session_row(holder, conn, parent, parent_session_id=target,
        child_session_id=child, source=policy['source'], model=policy['model'],
        model_config={'_branched_from': target}, system_prompt=None,
        cwd=policy['cwd'], profile_name=parent['profile_name'])
    conn.execute('UPDATE sessions SET session_key=?,chat_id=?,origin_json=?,system_prompt_hash=? WHERE id=?',
                 (route, child, _json(source.to_dict()), parent['system_prompt_hash'], child))
    from ..conversation._store.messages import _clone_message_rows, _tail_rows_after_watermark
    ids, tools = _tail_rows_after_watermark(holder, conn,
        'SELECT id,tool_calls FROM messages WHERE session_id=? AND active=1 AND id<=? ORDER BY id',
        [target, _branch_boundary(conn, target, payload.get('through_message_id'))])
    _clone_message_rows(holder, conn, ids, session_id=child)
    conn.execute('UPDATE sessions SET message_count=?,tool_call_count=? WHERE id=?', (len(ids), tools, child))
    from ..conversation._store.titles import _set_session_title_in_transaction
    if payload.get('title'):
        _set_session_title_in_transaction(holder, conn, child, payload['title'], source=TITLE_SOURCE_USER)
    else:
        from .._schema import next_title_in_lineage
        # Untitled branch: the next name in the parent's lineage ("my session" → "my session #2"),
        # same rule as the legacy TUI /branch, so the child is its own titled sidebar row.
        _set_session_title_in_transaction(holder, conn, child, next_title_in_lineage(conn, parent['title'] or 'branch'),
                                          source=TITLE_SOURCE_DERIVED)
    receipt = dict(profile_id=saved['profile_id'], principal_id=saved['principal_id'],
                   request_id=request_id, session_id=child, route=route, entry=entry.to_dict(), policy=policy)
    # The copied policy holds the parent's picks; the child's refresh must keep them as picks, not defaults.
    # Always written: a receipt without it predates the field, and picks_on reads that one's lineage.
    from ._runtime_picks import picks_on
    receipt['picks'] = sorted(picks_on(conn, session_id))
    conn.execute("INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('',?,?,?)",
                 (route, _json(entry.to_dict()), now.timestamp()))
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (POLICY_PREFIX + child, _json(receipt)))
    return {session_id}, {'branched_session_id': child, 'copied_messages': len(ids)}


def reset_in_transaction(holder, conn, session_id, payload):
    from ..conversation._store.local import POLICY_PREFIX
    from ..conversation._store.local_lineage import advance_local_target, validate_local_lineage
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                         (POLICY_PREFIX + session_id,)).fetchone()
    if saved is None:
        raise AuthorityRefusal('runtime_coordination_required')
    receipt = json.loads(saved[0])
    target = validate_local_lineage(conn, receipt)
    from ..workers._runtime_workers import discard_orphan_workers_on_reset, runtime_lineage
    discard_orphan_workers_on_reset(conn, runtime_lineage(conn, session_id))
    require_target_advanceable(holder, conn, list({session_id, target}))
    parent = conn.execute('SELECT * FROM sessions WHERE id=?', (target,)).fetchone()
    policy = receipt['policy']
    previous = SessionEntry.from_dict(receipt['entry'])
    now = _now()
    child_id = uuid.uuid4().hex
    entry = SessionEntry(previous.session_key, child_id, now, now,
        origin=previous.origin, platform=previous.platform, chat_type=previous.chat_type,
        display_name=previous.display_name, is_fresh_reset=True)
    from ..conversation._store.compression import _publish_child_session_row
    from ..conversation._store.messages import _bump_conversation_generation
    _publish_child_session_row(holder, conn, parent, parent_session_id=target,
        child_session_id=child_id, source=policy['source'], model=policy['model'],
        model_config={'_reset_from': target}, system_prompt=None,
        cwd=policy['cwd'], profile_name=parent['profile_name'])
    conn.execute("UPDATE sessions SET ended_at=?,end_reason='session_reset' WHERE id=?", (time.time(), target))
    _bump_conversation_generation(holder, conn, target, 'session_reset')
    advance_local_target(conn, target, child_id, entry=entry.to_dict())
    conn.execute('UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', (session_id,))
    generation = conn.execute('SELECT runtime_generation FROM sessions WHERE id=?', (session_id,)).fetchone()[0]
    # advance_local_target already advances the logical owner's revision.
    return set(), {'target_session_id': child_id, 'previous_target_session_id': target,
                   'execution_generation': generation}
