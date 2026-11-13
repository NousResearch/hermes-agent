"""An independent local route, transcript and frozen policy in one receipt commit."""
import json
import uuid

from hermes_state_local import POLICY_PREFIX
from hermes_state_local_lineage import validate_local_lineage
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


def _child_policy(policy, child, prepared, context):
    """The parent's frozen policy, except editor MCP servers: the gateway prepared the child's own
    selection under the child's own scope (gateway/session_local_mcp.py ``prepare_branch_editor``).
    Copying the parent's would share its scope, borrowed credentials and live connections.
    A branch starts at Default (``ask``), as main's ACP ``fork_session`` did.

    A branch of an autonomous run (``cron``) is rebased onto an interactive surface (see
    ``_rebase_run_policy``); every other policy is the parent's, as it was.
    """
    if 'edit_approval_policy' in policy:
        policy = {**policy, 'edit_approval_policy': 'ask'}
    if prepared is not None and 'parent_editor_mcp_json' in prepared:
        # An editor session's branch was prepared under the child's own scope: the child id this
        # commit derives and the parent's selection must both still hold.
        if child != prepared['child'] or policy.get('editor_mcp_json') != prepared['parent_editor_mcp_json']:
            raise RuntimeStoreError('revision_conflict')
        policy = {**policy, 'editor_mcp_json': prepared['editor_mcp_json'], 'toolsets': prepared['toolsets'],
                  'config_secret_ref': prepared['config_secret_ref']}
    elif policy.get('editor_mcp_json'):
        # Only the prepared path may fork an editor session (it owns the child's server scope).
        raise RuntimeStoreError('runtime_coordination_required')
    return _rebase_run_policy(policy, context)


# Surfaces a branch of an autonomous run may be rebased onto. Only interactive local surfaces: the
# child is a chat on this host, never a messaging platform's conversation and never another run.
BRANCH_SURFACES = frozenset({'gui', 'tui', 'cli'})


def _is_run_policy(policy):
    """Whether *policy* drives an autonomous run — a cron fire — rather than an interactive chat."""
    return policy.get('source') == 'cron' or policy.get('platform') == 'cron'


def _branch_principal(parent_policy, parent_principal, context):
    """The principal the branch child is created for.

    A branch of an autonomous run (``cron``) is rebased onto an interactive surface, and the fire's
    ``cron-owner`` stamp is the scheduler's identity, not a person: the child belongs to the caller
    who asked for it, which the gateway passes as ``context['principal_id']``. Every other branch is
    already the caller's (the mutation's authorization refused anyone else), so the parent's stands.
    """
    if not _is_run_policy(parent_policy):
        return parent_principal
    requester = (context or {}).get('principal_id')
    if not isinstance(requester, str) or not requester:
        # No caller identity to hand the child to: refuse rather than mint a chat nobody owns.
        raise RuntimeStoreError('invalid_params', field='principal_id',
                                detail='a branch of a run needs the requesting principal')
    return requester


def _rebase_run_policy(policy, context):
    """*policy* retargeted at the surface the branch was requested from; any other policy unchanged.

    A run policy freezes WHICH fire it is (``cron_job``, ``extra_prompt``, the fire's ``request_id``)
    beside its route. Copied verbatim into a branch, the child was a run session too: its executor
    accepted only the parent's frozen fire, so every typed message settled ``cron_session_read_only``.
    The child keeps what continuing the run's work needs — its route, its toolsets, its frozen config
    and its folders — and takes the branching surface's identity instead of the scheduler's.
    """
    if not _is_run_policy(policy):
        return policy
    from gateway.session_policy import CREATE_FIELDS, LOCAL_DESKTOP_CHAT, SURFACES
    source = (context or {}).get('source')
    if source not in BRANCH_SURFACES or source not in SURFACES:
        raise RuntimeStoreError('invalid_params', field='source',
                                detail=f'not a branch destination surface: {source!r}')
    # Creation fields only, minus the identity fields a rebased child must not inherit: the request
    # id here is the FIRE's (``cron:<job>:<execution>``) and names the parent's admission, not this
    # child's creation.
    request = {key: value for key, value in json.loads(policy['request_json']).items()
               if key in CREATE_FIELDS and key not in {'request_id', 'source'}}
    request['source'] = source
    request['cwd'] = policy['cwd']
    # The child continues the run's frozen route, so it must not re-read the profile's on first turn.
    config = json.loads(policy['config_json'])
    for mark in ('follow_profile_config', LOCAL_DESKTOP_CHAT):
        config.pop(mark, None)
    return {**policy, 'source': source, 'platform': SURFACES[source],
            'request_json': json.dumps(request, sort_keys=True), 'config_json': json.dumps(config)}


def branch_in_transaction(db, conn, session_id, payload, prepared=None, context=None):
    row = conn.execute('SELECT value FROM state_meta WHERE key=?',
                       (POLICY_PREFIX + session_id,)).fetchone()
    if row is None:
        raise RuntimeStoreError('runtime_coordination_required')
    saved = json.loads(row[0])
    target = validate_local_lineage(conn, saved)
    require_idle(db, conn, list({session_id, target}))
    # Late: the routing entry/key are gateway types (the gateway imports this store at module level).
    from gateway.session import SessionEntry, SessionSource
    from gateway.config import Platform
    from hermes_state_keys import local_identity, profile_from_session_key
    request_id = prepared['request_id'] if prepared else 'branch:' + uuid.uuid4().hex
    # A run's branch belongs to whoever asked for it, not to the job that produced the parent: the
    # child is an ordinary chat the caller owns, ready for its first typed message. Any other branch
    # keeps the parent's owner, which is already the caller (authorize refused everyone else).
    principal = _branch_principal(saved['policy'], saved['principal_id'], context)
    child = local_identity(saved['profile_id'], principal, request_id)
    policy = _child_policy(saved['policy'], child, prepared, context)
    # The child keeps the parent's owning profile: a served secondary's origin names it
    # (``local_source``), and restore compares the stored origin against that exactly.
    source = SessionSource(Platform.LOCAL, child, user_id=principal, chat_type='dm',
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
    conn.execute('UPDATE sessions SET session_key=?,chat_id=?,user_id=?,origin_json=?,system_prompt_hash=? WHERE id=?',
                 (route, child, principal, _json(source.to_dict()), parent['system_prompt_hash'], child))
    if _is_run_policy(saved['policy']):
        # ``_publish_child_session_row`` stamped ``created_source`` from the PARENT's, which for a
        # run is the scheduler. The child was created by this surface, so that is its provenance.
        from gateway.session_policy import created_surface
        conn.execute('UPDATE sessions SET created_source=? WHERE id=?',
                     (created_surface(policy['source']), child))
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
    receipt = dict(profile_id=saved['profile_id'], principal_id=principal,
                   request_id=request_id, session_id=child, route=route, entry=entry.to_dict(), policy=policy)
    # The copied policy holds the parent's picks; the child's refresh must keep them as picks, not defaults.
    # Always written: a receipt without it predates the field, and picks_on reads that one's lineage.
    from hermes_state_mutation_prepared import picks_on
    receipt['picks'] = sorted(picks_on(conn, session_id))
    conn.execute("INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('',?,?,?)",
                 (route, _json(entry.to_dict()), now.timestamp()))
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (POLICY_PREFIX + child, _json(receipt)))
    return {session_id}, {'branched_session_id': child, 'copied_messages': len(ids)}
