"""An independent local route, transcript and frozen policy in one receipt commit."""
import json
import uuid

from hermes_state_local import POLICY_PREFIX
from hermes_state_local_lineage import validate_local_lineage
from hermes_state_write_guards import REVIEW_LEASE_IGNORE
from hermes_state_mutation_guards import require_idle
from hermes_state_runtime import RuntimeStoreError, _json
from hermes_state_titles import lineage_title_on_conn


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
        raise RuntimeStoreError('invalid_params')
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
    Copying the parent's would share its scope, borrowed credentials and live connections."""
    if prepared is None:
        if policy.get('editor_mcp_json'):
            raise RuntimeStoreError('runtime_coordination_required')
        return policy
    # Prepared against the parent's selection and the child id this commit derives; anything else moved.
    if child != prepared['child'] or policy.get('editor_mcp_json') != prepared['parent_editor_mcp_json']:
        raise RuntimeStoreError('revision_conflict')
    return {**policy, 'editor_mcp_json': prepared['editor_mcp_json'], 'toolsets': prepared['toolsets'],
            'config_secret_ref': prepared['config_secret_ref']}


def branch_in_transaction(db, conn, session_id, payload, prepared=None):
    row = conn.execute('SELECT value FROM state_meta WHERE key=?',
                       (POLICY_PREFIX + session_id,)).fetchone()
    if row is None:
        raise RuntimeStoreError('runtime_coordination_required')
    saved = json.loads(row[0])
    target = validate_local_lineage(conn, saved)
    # A copy cannot invalidate a background review's snapshot: the branch lands beside it.
    require_idle(db, conn, list({session_id, target}), review_lease_policy=REVIEW_LEASE_IGNORE)
    # Late: the routing entry/key are gateway types (the gateway imports this store at module level).
    from gateway.session import SessionEntry, SessionSource
    from gateway.config import Platform
    from hermes_state_keys import local_identity, profile_from_session_key
    request_id = prepared['request_id'] if prepared else 'branch:' + uuid.uuid4().hex
    child = local_identity(saved['profile_id'], saved['principal_id'], request_id)
    policy = _child_policy(saved['policy'], child, prepared)
    # The child keeps the parent's owning profile: a served secondary's origin names it
    # (``local_source``), and restore compares the stored origin against that exactly.
    source = SessionSource(Platform.LOCAL, child, user_id=saved['principal_id'], chat_type='dm',
                           profile=saved['entry']['origin'].get('profile'))
    from gateway.session import build_session_key
    route = build_session_key(source, profile=profile_from_session_key(saved['route']))
    from gateway.session_lifecycle import _now
    now = _now()
    entry = SessionEntry(route, child, now, now, origin=source, platform=Platform.LOCAL)
    parent = conn.execute('SELECT * FROM sessions WHERE id=?', (target,)).fetchone()
    db._publish_child_session_row(conn, parent, parent_session_id=target,
        child_session_id=child, source=policy['source'], model=policy['model'],
        model_config={'_branched_from': target}, system_prompt=None,
        cwd=policy['cwd'], profile_name=parent['profile_name'])
    conn.execute('UPDATE sessions SET session_key=?,chat_id=?,origin_json=?,system_prompt_hash=? WHERE id=?',
                 (route, child, _json(source.to_dict()), parent['system_prompt_hash'], child))
    ids, tools = db._tail_rows_after_watermark(conn,
        'SELECT id,tool_calls FROM messages WHERE session_id=? AND active=1 AND id<=? ORDER BY id',
        [target, _branch_boundary(conn, target, payload.get('through_message_id'))])
    db._clone_message_rows(conn, ids, session_id=child)
    conn.execute('UPDATE sessions SET message_count=?,tool_call_count=? WHERE id=?', (len(ids), tools, child))
    if payload.get('title'):
        db._set_session_title_in_transaction(conn, child, payload['title'], source=db.TITLE_SOURCE_USER)
    else:
        # Untitled branch: the next name in the parent's lineage ("my session" → "my session #2"),
        # same rule as the legacy TUI /branch, so the child is its own titled sidebar row.
        db._set_session_title_in_transaction(conn, child, lineage_title_on_conn(conn, parent['title'] or 'branch'),
                                             source=db.TITLE_SOURCE_DERIVED)
    receipt = dict(profile_id=saved['profile_id'], principal_id=saved['principal_id'],
                   request_id=request_id, session_id=child, route=route, entry=entry.to_dict(), policy=policy)
    conn.execute("INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('',?,?,?)",
                 (route, _json(entry.to_dict()), now.timestamp()))
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (POLICY_PREFIX + child, _json(receipt)))
    return {session_id}, {'branched_session_id': child, 'copied_messages': len(ids)}
