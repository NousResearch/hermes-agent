"""Borrow editor transports for one canonical execution, never for a viewer.

Only a fingerprint and the selected schema manifest are durable. Editors must
resupply the exact transport specification after owner restart, including argv:
credentials are not reliably distinguishable from ordinary command arguments.
"""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import hmac
import json

from hermes_state_runtime import RuntimeStoreError

PRIVATE_KEY = ('editor_mcp',)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def fingerprint(servers):
    return hashlib.sha256(_json(servers).encode()).hexdigest()


def validate_servers(servers):
    from acp.schema import McpServerStdio, McpServerHttp, McpServerSse
    from pydantic import ValidationError
    kinds = {'stdio': McpServerStdio, 'http': McpServerHttp, 'sse': McpServerSse}
    if not isinstance(servers, list) or len(servers) > 16 or len(_json(servers)) > 65536:
        raise RuntimeStoreError('invalid_params')
    names = set()
    try:
        for spec in servers:
            kind = spec.get('type', 'stdio')
            model = kinds[kind].model_validate(spec)
            if not model.name or model.name in names:
                raise ValueError('duplicate server')
            names.add(model.name)
    except (ValidationError, KeyError, AttributeError, TypeError, ValueError) as exc:
        raise RuntimeStoreError('invalid_params') from exc


def private_editor_request(request, private):
    editor = request.get('editor') or {}
    servers = editor.get('mcp_servers') or []
    if not servers:
        return
    if private is None:
        raise RuntimeStoreError('launch_credentials_unavailable')
    private[PRIVATE_KEY] = json.loads(_json(servers))
    request['editor'] = {**editor, 'mcp_servers': {'fingerprint': fingerprint(servers)}}


def _configs(scope, servers, cwd):
    configs = {}
    prefix = hashlib.sha256(scope.encode()).hexdigest()[:20]
    for i, spec in enumerate(servers):
        name = f'editor_{prefix}_{i}'
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


@contextmanager
def _registered(scope, servers, cwd):
    from tools.registry import registry, session_tool_scope
    from tools.mcp_tool_discovery import register_mcp_servers
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.mcp_tool import _servers
    from tools.mcp_tool_scope import _server_key
    configs = _configs(scope, servers, cwd)
    with session_tool_scope(scope):
        try:
            register_mcp_servers(configs)
            # Inside the scope the ledger keys are ``(scope, name)``, never bare names.
            keys = [_server_key(name, scope) for name in configs]
            if any(key not in _servers or _servers[key].session is None for key in keys):
                raise RuntimeStoreError('acp_mcp_discovery_failed')
            names = [n for n in registry.get_all_tool_names()
                     if registry.get_toolset_for_tool(n) in {f'mcp-{k}' for k in configs}]
            yield {n: registry.get_schema(n) for n in sorted(names)}, tuple(f'mcp-{k}' for k in configs)
        finally:
            shutdown_mcp_servers(scope=scope)


def bind_editor_mcp(authority, sid, policy, servers):
    if not servers:
        return policy
    scope = 'editor-session:' + hashlib.sha256(f'{authority.profile_id}:{sid}'.encode()).hexdigest()
    with _registered(scope, servers, policy.cwd) as (schemas, toolsets):
        data = {'scope': scope, 'fingerprint': fingerprint(servers), 'schemas': schemas}
    policy = replace(policy, editor_mcp_json=_json(data),
                     toolsets=tuple(sorted(set(policy.toolsets) | set(toolsets))))
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
    """Call after ordinary resume authorization, before attaching the viewer."""
    if editor is None:
        return
    from gateway.session_local_editor import validate_editor
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    if policy is None:
        raise RuntimeStoreError('acp_mcp_policy_conflict')
    validate_editor(policy.source, editor)
    servers = editor.get('mcp_servers', [])
    if servers:
        attach_editor_mcp(authority, policy, servers)


@contextmanager
def editor_mcp_scope(authority, policy):
    if not policy.editor_mcp_json:
        yield
        return
    data = json.loads(policy.editor_mcp_json)
    servers = getattr(authority, '_local_editor_mcp', {}).get(data['scope'])
    if servers is None:
        raise RuntimeStoreError('launch_credentials_unavailable')
    with _registered(data['scope'], servers, policy.cwd) as (schemas, _):
        if schemas != data['schemas']:
            raise RuntimeStoreError('acp_mcp_schema_changed')
        yield
