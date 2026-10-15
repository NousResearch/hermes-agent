"""Editor MCP servers are session state the gateway owns; an ACP connection only requests them.

Only a fingerprint and the selected schema manifest are durable. Editors must
resupply the exact transport specification after owner restart, including argv:
credentials are not reliably distinguishable from ordinary command arguments.
A different specification is a ``session.mutate editor_mcp`` (control capability,
revision/generation fenced, idle only), never an implicit swap on attach.

The frozen manifest is part of the session's prompt-cache prefix, so it only changes
at a turn boundary: through that mutation while the session is idle, or through the
re-freeze the drain runs at the start of an admission (``refreeze_editor_mcp``).
Inside an execution it never changes.
"""
import asyncio
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import hmac
import json
import logging
import threading
import uuid

from hermes_state_runtime import RuntimeStoreError, classify_loc, publish_client_value

logger = logging.getLogger(__name__)

PRIVATE_KEY = ('editor_mcp',)
_HELD_LOCK = threading.Lock()
_RELIST_TIMEOUT = 5  # seconds an admission start waits on a held server's tools/list


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def fingerprint(servers):
    return hashlib.sha256(_json(servers).encode()).hexdigest()


def _spec_problem(exc):
    """Why a server spec is unusable, without the spec's values.

    Pydantic's message quotes ``input_value``, which can be the whole spec with its env values,
    so a schema error is its failing locs and error types. Any other message is one line, and
    only when it may be echoed."""
    from pydantic import ValidationError
    if not isinstance(exc, ValidationError):
        return publish_client_value(' '.join(str(exc).split()))
    problems = []
    for error in exc.errors(include_url=False, include_input=False)[:3]:
        loc = classify_loc('.'.join(str(part) for part in error.get('loc') or ())) or 'spec'
        kind = classify_loc(str(error.get('type') or '')) or 'invalid'
        problems.append(f'{loc} ({kind})')
    return ', '.join(problems)


def validate_servers(servers):
    from acp.schema import McpServerStdio, McpServerHttp, McpServerSse
    from pydantic import ValidationError
    kinds = {'stdio': McpServerStdio, 'http': McpServerHttp, 'sse': McpServerSse}
    if not isinstance(servers, list):
        raise RuntimeStoreError('invalid_params', field='editor.mcp_servers',
                                detail=f'must be a list, got {type(servers).__name__}')
    if len(servers) > 16:
        raise RuntimeStoreError('invalid_params', field='editor.mcp_servers',
                                detail=f'at most 16 servers, got {len(servers)}')
    if len(_json(servers)) > 65536:
        raise RuntimeStoreError('invalid_params', field='editor.mcp_servers',
                                detail='the serialized server list exceeds 65536 bytes')
    names = set()
    try:
        for spec in servers:
            kind = spec.get('type', 'stdio')
            model = kinds[kind].model_validate(spec)
            if not model.name or model.name in names:
                raise ValueError('duplicate server')
            names.add(model.name)
    except (ValidationError, KeyError, AttributeError, TypeError, ValueError) as exc:
        raise RuntimeStoreError('invalid_params', field='editor.mcp_servers',
                                detail=f'unusable server spec: {_spec_problem(exc)}') from exc


def private_editor_request(request, private):
    editor = request.get('editor') or {}
    servers = editor.get('mcp_servers') or []
    if not servers:
        return
    if private is None:
        raise RuntimeStoreError('launch_credentials_unavailable')
    private[PRIVATE_KEY] = json.loads(_json(servers))
    request['editor'] = {**editor, 'mcp_servers': {'fingerprint': fingerprint(servers)}}


def private_mutation_payload(operation, payload):
    """``(store payload, servers)`` of an ``editor_mcp`` or a ``branch`` naming servers: the list
    stays in owner memory; the receipt digest and every store write see only its fingerprint."""
    if operation == 'editor_mcp':
        if not isinstance(payload, dict) or set(payload) != {'mcp_servers'}:
            raise RuntimeStoreError('invalid_params')
    elif operation != 'branch' or not isinstance(payload, dict) or 'mcp_servers' not in payload:
        return payload, None
    servers = payload['mcp_servers']
    validate_servers(servers)
    servers = json.loads(_json(servers))
    return {**payload, 'mcp_servers': {'fingerprint': fingerprint(servers)}}, servers


def session_scope(authority, sid):
    """A session's own ledger scope: never shared with another session (a fork included) or profile."""
    return 'editor-session:' + hashlib.sha256(f'{authority.profile_id}:{sid}'.encode()).hexdigest()


def _server_names(scope, servers):
    """Index -> ledger name: the editor's own server name (the model sees ``mcp__editor_github_<tag>__x``)
    plus a per-session tag, so two sessions borrowing same-named servers never adopt one connection."""
    from tools.mcp_tool_schema import sanitize_mcp_name_component
    return {i: f"editor_{sanitize_mcp_name_component(spec['name'])}_"
               f"{hashlib.sha256(f'{scope}:{i}'.encode()).hexdigest()[:8]}" for i, spec in enumerate(servers)}


def _frozen_names(data, servers):
    """The names a session's manifest was frozen under; manifests without ``names`` predate them."""
    if 'names' in data:
        return {int(i): name for i, name in data['names'].items()}
    prefix = hashlib.sha256(data['scope'].encode()).hexdigest()[:20]
    return {i: f'editor_{prefix}_{i}' for i in range(len(servers))}


def _frozen_toolset(data, toolset):
    """Whether *toolset* is one of the editor toolsets frozen in *data* (a manifest or None)."""
    if not data:
        return False
    if 'names' in data:
        return toolset in {f'mcp-{name}' for name in data['names'].values()}
    return toolset.startswith(f"mcp-editor_{hashlib.sha256(data['scope'].encode()).hexdigest()[:20]}_")


def _configs(servers, names, cwd):
    configs = {}
    for i, name in names.items():
        spec = servers[i]
        if spec.get('type', 'stdio') == 'stdio':
            config = {'command': spec['command'], 'args': spec['args'],
                      'env': {e['name']: e['value'] for e in spec.get('env', [])}, 'cwd': cwd}
        else:
            config = {'url': spec['url'], 'headers': {e['name']: e['value'] for e in spec.get('headers', [])}}
            if spec['type'] == 'sse':
                config['transport'] = 'sse'
        # One bounded eager discovery; no lazy or late schema refresh.
        config.update(connect_timeout=5, tool_timeout=30, lazy=False)
        configs[name] = config
    return configs


class _Held:
    """One session scope's editor connections. ``users`` executions borrow them right now; between
    executions they stay connected only while the session has a viewer (``_viewed``)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.users = 0
        self.fingerprint = None


def _held(authority, scope):
    with _HELD_LOCK:
        table = getattr(authority, '_editor_mcp_held', None)
        if table is None:
            table = authority._editor_mcp_held = {}
        return table.setdefault(scope, _Held())


def _viewed(authority, scope):
    return any(live.subscribers and session_scope(authority, sid) == scope
               for sid, live in list(getattr(authority, 'sessions', {}).items()))


def _connected(scope, names):
    from tools.mcp_tool import _servers
    from tools.mcp_tool_scope import _server_key
    # Inside the scope the ledger keys are ``(scope, name)``, never bare names.
    return [i for i, name in names.items()
            if _server_key(name, scope) in _servers and _servers[_server_key(name, scope)].session]


def _disconnect(held, scope):
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    shutdown_mcp_servers(scope=scope)
    held.fingerprint = None


@contextmanager
def _borrowed(authority, scope, servers, names, cwd, *, partial=False):
    """Yield ``(schemas, connected indexes)`` inside the session's tool scope. Only ``partial`` (a
    selection being frozen) may leave a server out; an execution needs every server its manifest
    was frozen with. A held connection is reused; the last borrower closes it unless a viewer remains."""
    from tools.registry import registry, session_tool_scope
    from tools.mcp_tool_discovery import register_mcp_servers
    held = _held(authority, scope)
    # The cwd too: stdio servers run in it, so a committed workspace move respawns them there.
    wanted = fingerprint([servers, cwd])
    with session_tool_scope(scope):
        with held.lock:
            if held.fingerprint not in (None, wanted):
                # A committed editor_mcp swapped the credentials (or a workspace move the cwd): no
                # execution may still run on the old ones.
                if held.users:
                    raise RuntimeStoreError('session_busy')
                _disconnect(held, scope)
            connected = _connected(scope, names)
            if len(connected) != len(names):
                register_mcp_servers(_configs(servers, names, cwd))
                connected = _connected(scope, names)
            held.fingerprint = wanted
            held.users += 1
        try:
            if len(connected) != len(names) and not partial:
                raise RuntimeStoreError('acp_mcp_discovery_failed')
            toolsets = {f'mcp-{names[i]}' for i in connected}
            tools = [n for n in registry.get_all_tool_names() if registry.get_toolset_for_tool(n) in toolsets]
            yield {n: registry.get_schema(n) for n in sorted(tools)}, connected
        finally:
            with held.lock:
                held.users -= 1
                if not held.users and not _viewed(authority, scope):
                    _disconnect(held, scope)


def release_editor_mcp(authority, session_ids):
    """The last viewer left (or the sessions retired): close their held editor connections. One an
    execution still borrows is closed by that execution when it ends."""
    table = getattr(authority, '_editor_mcp_held', None) or {}
    for sid in session_ids:
        scope = session_scope(authority, sid)
        held = table.get(scope)
        if held is None:
            continue
        with held.lock:
            if not held.users and held.fingerprint is not None and not _viewed(authority, scope):
                _disconnect(held, scope)


def rebind_editor_mcp(authority, sid, policy, servers):
    """``(policy, manifest changed)``: *policy* with its editor selection freshly discovered from
    *servers* under *sid*'s own scope (blocking). A broken server is left out of the frozen selection
    (main skipped it too), so no later execution respawns it."""
    old = json.loads(policy.editor_mcp_json or 'null')
    toolsets = {name for name in policy.toolsets if not _frozen_toolset(old, name)}
    data = None
    if servers:
        scope = session_scope(authority, sid)
        held = _held(authority, scope)
        with held.lock:
            if held.users:
                raise RuntimeStoreError('session_busy')
            # Fresh discovery: a held connection keeps the schemas it was first frozen with.
            _disconnect(held, scope)
        names = _server_names(scope, servers)
        with _borrowed(authority, scope, servers, names, policy.cwd, partial=True) as (schemas, connected):
            for i in sorted(set(names) - set(connected)):
                logger.warning("Editor MCP server %r failed discovery; session %s continues without it",
                               servers[i].get('name'), sid)
        names = {i: names[i] for i in connected}
        data = {'scope': scope, 'fingerprint': fingerprint(servers), 'schemas': schemas,
                'names': {str(i): name for i, name in names.items()}}
        toolsets |= {f'mcp-{name}' for name in names.values()}
    rebound = replace(policy, editor_mcp_json=_json(data) if data else None, toolsets=tuple(sorted(toolsets)))
    changed = ((old or {}).get('schemas', {}) != (data or {}).get('schemas', {})
               or rebound.toolsets != policy.toolsets)
    return rebound, changed


def bind_editor_mcp(authority, sid, policy, servers):
    if not servers:
        return policy
    policy, _ = rebind_editor_mcp(authority, sid, policy, servers)
    attach_editor_mcp(authority, policy, servers)
    return policy


def attach_editor_mcp(authority, policy, servers):
    validate_servers(servers)
    data = json.loads(policy.editor_mcp_json or '{}')
    if not data or not hmac.compare_digest(data['fingerprint'], fingerprint(servers)):
        raise RuntimeStoreError('acp_mcp_policy_conflict')
    borrowed = getattr(authority, '_local_editor_mcp', None)
    if borrowed is None:
        borrowed = authority._local_editor_mcp = {}
    borrowed[data['scope']] = json.loads(_json(servers))


def resume_editor_mcp(authority, ref, editor):
    """Call after ordinary resume authorization, before attaching the viewer. An empty or omitted
    list keeps the session's servers: ACP load always carries one, so a second editor without MCP
    config must not strip the first one's tools for every viewer. Removal is only the fenced
    ``editor_mcp`` change with ``[]``."""
    if editor is None:
        return
    servers = editor.get('mcp_servers') if isinstance(editor, dict) else None
    if servers == []:
        return
    from gateway.session_local_editor import validate_editor
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    if policy is None:
        raise RuntimeStoreError('acp_mcp_policy_conflict')
    validate_editor(policy.source, editor)
    if servers:
        attach_editor_mcp(authority, policy, servers)


def retire_editor_mcp(authority, sid):
    """An empty selection committed: *sid*'s held editor connections close now (blocking), viewer or not."""
    scope = session_scope(authority, sid)
    held = _held(authority, scope)
    with held.lock:
        _disconnect(held, scope)


def install_editor_mcp(authority, sid, servers):
    """After a committed change: *sid*'s borrowed credentials follow its committed policy (memory only)."""
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    policy = restore_policy(local_receipt(authority.db, sid)['policy'])
    data = json.loads(policy.editor_mcp_json or '{}')
    borrowed = getattr(authority, '_local_editor_mcp', None)
    if borrowed is None:
        borrowed = authority._local_editor_mcp = {}
    if not data:
        borrowed.pop(session_scope(authority, sid), None)
    elif hmac.compare_digest(data['fingerprint'], fingerprint(servers)):
        borrowed[data['scope']] = json.loads(_json(servers))


def announce_tools_changed(live, session_id, result):
    """One ``session.tools_changed`` on the session's stream when a change altered the manifest:
    server names and the generation only, never a transport specification."""
    if live is not None and result.get('tools_changed'):
        live.event_stream.publish(session_id, {'mcp_servers': result['mcp_servers'],
            'execution_generation': result['execution_generation']}, event_type='session.tools_changed')


async def prepare_editor_mcp(authority, live, payload, prepared, *, servers):
    """``session.mutate editor_mcp``: discover the new list and freeze it into the policy the next
    turn runs under. Equal schemas leave the manifest, and so the prompt-cache prefix, as it was;
    only the fingerprint and the borrowed credentials swap."""
    from gateway.session_mutation_policy import _rebind
    from gateway.session_policy import restore_policy
    receipt = prepared['snapshot']['receipt']
    old = restore_policy(receipt['policy'])
    if old.source != 'acp':
        raise RuntimeStoreError('invalid_params')
    policy, changed = await asyncio.to_thread(rebind_editor_mcp, authority, receipt['session_id'], old, servers)
    names = json.loads(policy.editor_mcp_json or '{}').get('names', {})
    return _rebind(authority, prepared, old, policy, tools_changed=changed,
                   mcp_servers=[servers[int(i)]['name'] for i in sorted(names, key=int)])


def _live_tools_changed(scope, names):
    """Whether a held server's live ``tools/list`` differs from the tools its connection was
    discovered with. Editor servers ignore ``tools/list_changed`` (the manifest is frozen), so
    the admission start asks; an unanswered list (bounded) counts as unchanged."""
    from tools import mcp_tool_loop
    from tools.mcp_tool import _paginate_full_list, _servers
    from tools.mcp_tool_scope import _server_key

    def dump(tools):
        return _json([tool.model_dump(mode='json') for tool in tools])

    held = [server for server in (_servers.get(_server_key(name, scope)) for name in names.values())
            if server is not None]

    async def relist(server):
        async with server._rpc_lock:
            if server.session is None:
                return None
            return await _paginate_full_list(server.session.list_tools, 'tools', server.name)

    async def relist_all():
        # Every held server at once: one bound for the whole check, not one per server.
        tasks = [asyncio.ensure_future(relist(server)) for server in held]
        try:
            if tasks:
                await asyncio.wait(tasks, timeout=_RELIST_TIMEOUT)
        finally:
            for task in tasks:
                task.cancel()  # an unanswered list (bounded) counts as unchanged
        answered = []
        for server, task in zip(held, tasks):
            settled = task.done() and not task.cancelled()
            error = task.exception() if settled else None
            if settled and error is None:
                answered.append((server, task.result()))
            else:
                logger.warning('Editor MCP server %r: tools/list for the drift check failed', server.name,
                               exc_info=error)
        return answered
    try:
        # The loop's own bound is a backstop; relist_all settles within _RELIST_TIMEOUT.
        answered = mcp_tool_loop._run_on_mcp_loop(relist_all, timeout=_RELIST_TIMEOUT + 1)
    except (RuntimeError, TimeoutError, InterruptedError):
        logger.warning('Editor MCP drift check of %s failed', scope, exc_info=True)
        return False
    return any(tools is not None and dump(tools) != dump(server._tools) for server, tools in answered)


async def refreeze_editor_mcp(authority, ref, live):
    """The drain, once the FIFO head is authorized and before it is claimed (claim gate and mutation
    lock held, nothing executing), re-checks the editor schemas; a drifted manifest is re-frozen through the
    ``editor_mcp`` writer path, so the turn runs on it instead of failing ``acp_mcp_schema_changed``.
    Best-effort: the FIFO must not pause on it, so a failed re-check only logs and the turn then
    reports real drift as before."""
    if not getattr(authority, '_local_editor_mcp', None):
        return  # no session of this owner borrows editor servers: nothing to re-check
    try:
        await _refreeze(authority, ref)
    except Exception:  # health: allow BLE001 -- the probe spawns arbitrary editor servers; nothing it raises may kill the drain
        logger.warning('Editor MCP re-check of session %s failed; the turn runs on its frozen manifest',
                       ref.session_id, exc_info=True)


async def _refreeze(authority, ref):
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    # The committed policy the claim just restored (``restore_local_session``).
    receipt = await asyncio.to_thread(local_receipt, authority.db, ref.session_id)
    policy = restore_policy(receipt['policy'])
    if not policy.editor_mcp_json:
        return
    data = json.loads(policy.editor_mcp_json)
    servers = authority._local_editor_mcp.get(data['scope'])
    if servers is None:
        return  # the execution reports launch_credentials_unavailable

    def probe():
        names = _frozen_names(data, servers)
        with _borrowed(authority, data['scope'], servers, names, policy.cwd, partial=True) as (schemas, connected):
            complete = len(connected) == len(names)
            # A held connection keeps the registry schemas it was discovered with: ask the server.
            return schemas, complete, complete and _live_tools_changed(data['scope'], names)
    schemas, complete, drifted = await asyncio.to_thread(probe)
    # A server that is down right now is the execution's discovery failure, never a narrowed manifest.
    if not complete or (schemas == data['schemas'] and not drifted):
        return
    await _commit_refreeze(authority, ref, servers)


async def _commit_refreeze(authority, ref, servers):
    """The ``editor_mcp`` change on the owning principal's behalf, through ``session.mutate``'s own
    writer (the drain holds the session's locks, and a schema re-check needs no capability)."""
    from gateway.session_mutations import write_mutation
    from hermes_state_local import local_receipt
    owner = await asyncio.to_thread(local_receipt, authority.db, ref.session_id)
    row = await asyncio.to_thread(authority.db.get_session, ref.session_id)
    payload, servers = private_mutation_payload('editor_mcp', {'mcp_servers': servers})
    params = {'session_id': ref.session_id, 'request_id': 'editor-mcp-refreeze:' + uuid.uuid4().hex,
              'expected_revision': row['runtime_revision'], 'expected_generation': row['runtime_generation'],
              'operation': 'editor_mcp', 'payload': payload}
    await write_mutation(authority, ref, params, servers, principal_id=owner['principal_id'])


def _child_secret_ref(authority, child, parent, policy):
    """A durable config-secret reference names the policy identity, which the child's own editor
    selection changed; the frozen secrets themselves are the parent's."""
    from gateway.session_policy_credentials import PREFIX, config_reference, recover_config_secrets
    if parent.config_secret_ref is None or not parent.config_secret_ref.startswith(PREFIX):
        return policy
    secrets = recover_config_secrets(authority, parent, skip=())
    return replace(policy, config_secret_ref=config_reference(authority, child, policy, secrets))


async def prepare_branch_editor(authority, ref, servers, gate):
    """``(prepared, servers)`` for a fork of an editor session: the child's selection under its OWN
    scope, from the fork's list or else a copy of the parent's borrowed servers. The child never
    shares the parent's scope or connections. ``(None, None)`` when no editor servers are involved.

    Discovery starts every server, so *gate* (the branch's prepare-only fences: receipt, revision,
    generation, idle) runs first; an exact retry's durable receipt comes back as ``prepared``."""
    from gateway.session_policy import restore_policy
    from hermes_state_keys import local_identity
    from hermes_state_local import local_receipt
    try:
        saved = await asyncio.to_thread(local_receipt, authority.db, ref.session_id)
    except RuntimeStoreError:
        return None, None  # not a local session: the branch transaction refuses it
    parent = restore_policy(saved['policy'])
    if servers is None:
        if not parent.editor_mcp_json:
            return None, None
        servers = getattr(authority, '_local_editor_mcp', {}).get(json.loads(parent.editor_mcp_json)['scope'])
        if servers is None:
            raise RuntimeStoreError('launch_credentials_unavailable')
        servers = json.loads(_json(servers))
    elif parent.source != 'acp':
        raise RuntimeStoreError('invalid_params')
    gated = await gate()
    if 'snapshot' not in gated:
        return gated, servers
    request_id = 'branch:' + uuid.uuid4().hex
    child = local_identity(saved['profile_id'], saved['principal_id'], request_id)
    policy, _ = await asyncio.to_thread(rebind_editor_mcp, authority, child, parent, servers)
    policy = _child_secret_ref(authority, child, parent, policy)
    return {**gated, 'request_id': request_id, 'child': child, 'parent_editor_mcp_json': parent.editor_mcp_json,
            'editor_mcp_json': policy.editor_mcp_json, 'toolsets': list(policy.toolsets),
            'config_secret_ref': policy.config_secret_ref}, servers


@contextmanager
def editor_mcp_scope(authority, policy):
    if not policy.editor_mcp_json:
        yield
        return
    data = json.loads(policy.editor_mcp_json)
    servers = getattr(authority, '_local_editor_mcp', {}).get(data['scope'])
    if servers is None:
        raise RuntimeStoreError('launch_credentials_unavailable')
    with _borrowed(authority, data['scope'], servers, _frozen_names(data, servers), policy.cwd) as (schemas, _):
        if schemas != data['schemas']:
            # Drift is re-frozen by the drain before a claim; inside an execution the manifest never changes.
            raise RuntimeStoreError('acp_mcp_schema_changed')
        yield
