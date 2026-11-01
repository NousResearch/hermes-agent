"""Canonical coding checkout ownership, stored with the frozen local launch policy.

The client receipt stays verbatim in request_json; server identity and managed paths
live in the policy, never in a sidecar session or a rewritten create request.
"""
import json
import asyncio
from dataclasses import replace
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError

_KEY = 'canonical_coding_workspace_v1'


def binding_request(params):
    """Validate the typed receipt and its explicit launch-directory identity."""
    raw = params.get('coding_workspace')
    if raw is None:
        return None
    from pydantic import ValidationError
    from tui_gateway.contracts.common import CodingWorkspaceBinding
    try:
        binding = CodingWorkspaceBinding.model_validate_json(json.dumps(raw), strict=True)
    except ValidationError as exc:
        raise RuntimeStoreError('invalid_params', field='coding_workspace') from exc
    if (not binding.requestId or len(binding.requestId) > 256 or binding.cwd != params.get('cwd')
            or params.get('request_id', binding.requestId) != binding.requestId
            or params.get('row_cwd', binding.cwd) != binding.cwd
            or binding.artifactsPath is not None):
        raise RuntimeStoreError('invalid_params', field='coding_workspace', detail='binding must match cwd and request_id; artifacts are server-owned')
    return raw


def request_id(params):
    binding = params.get('coding_workspace')
    return params['request_id'] if 'request_id' in params else (binding or {}).get('requestId')


def launch_cwd(params, cwd, home):
    if not binding_request(params) and params.get('cwd_explicit') is False and 'cwd' in params:
        from gateway.session_policy import _profile_workspace_cwd
        return _profile_workspace_cwd(home) or cwd
    return cwd


def _stat(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino]


def _git_path(cwd, flag):
    from tui_gateway.coding_workspaces import _git_output
    value = _git_output(cwd, ['rev-parse', '--path-format=absolute', flag]).removesuffix('\n')
    return str(Path(value).resolve(strict=True))


def _identity(binding):
    from tui_gateway.coding_workspaces import _directory, _git_output
    cwd = _directory(binding['cwd'])
    if cwd != binding['cwd']:
        raise ValueError('Workspace path moved or became an alias')
    result: dict = {'cwd': _stat(cwd)}
    if binding.get('repoRoot'):
        trees = _git_output(cwd, ['worktree', 'list', '--porcelain', '-z'])
        primary = trees.split('\0', 1)[0].removeprefix('worktree ')
        if primary != binding['repoRoot']:
            raise ValueError('Git repository differs from prepared checkout')
        for key, flag in (('top', '--show-toplevel'), ('git', '--git-dir'), ('common', '--git-common-dir')):
            path = _git_path(cwd, flag)
            result[key] = [path, *_stat(path)]
        if result['top'][0] != cwd:
            raise ValueError('Workspace must be the prepared checkout root')
    else:
        from hermes_cli import web_git
        code, _out, err = web_git._git(cwd, ['rev-parse', '--is-inside-work-tree'])
        if not code or not err.startswith('fatal: not a git repository (or any'):
            raise ValueError('Folder repository identity changed')
    return result


def prepare_binding(authority, actor, params):
    """Off-loop pre-mint filesystem and Projects checks; no session or registry writes."""
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if 'session:create' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    binding = binding_request(params)
    if binding is None:
        return None
    from gateway.session_local_recovery import local_identity
    sid = local_identity(authority.profile_id, actor.subject, request_id(params))
    if authority.db.get_session(sid) is not None:
        return None  # exact retry is judged by the original request, not current filesystem state
    from hermes_cli import projects_db as pdb
    home = Path(authority.profile_id)
    try:
        identity = _identity(binding)
        root = binding.get('repoRoot') or binding['cwd']
        with pdb.connect_closing(home / 'projects.db') as conn:
            project = pdb.project_for_path(conn, root)
            if project is None or project.id != binding.get('projectId'):
                raise ValueError('Workspace does not belong to the selected project')
            source = binding.get('sourcePath') or binding['cwd']
            if binding.get('repoRoot'):
                source_binding = {**binding, 'cwd': source}
                if _identity(source_binding)['common'] != identity['common']:
                    raise ValueError('Source belongs to another repository')
            elif source != binding['cwd']:
                raise ValueError('Folder source differs from prepared checkout')
            claim = pdb.get_workspace_claim(conn, binding['requestId'])
            if binding.get('mode') == 'worktree' and (not claim or any(
                    claim[key] != binding[field] for key, field in (('root', 'repoRoot'), ('path', 'cwd'), ('branch', 'branch')))):
                raise ValueError('Workspace claim differs from prepared checkout')
            if claim and claim['state'] == 'bound':
                raise ValueError('Workspace request already belongs to a session')
        artifacts = home / 'cache' / 'session-artifacts' / sid
        return {'binding': {**binding, 'artifactsPath': str(artifacts)}, 'identity': identity, 'session_id': sid}
    except (OSError, ValueError) as exc:
        raise RuntimeStoreError('invalid_params', field='coding_workspace', detail=str(exc)) from exc


def bind_policy(policy, prepared):
    if prepared is None:
        return policy
    terminal = json.loads(policy.terminal_json)
    if (str(terminal.get('TERMINAL_ENV') or 'local').strip().lower() != 'local'
            or terminal.get('TERMINAL_CWD') != prepared['binding']['cwd'] or policy.cwd != prepared['binding']['cwd']):
        raise RuntimeStoreError('invalid_params', field='coding_workspace', detail='a coding workspace requires its local terminal cwd')
    from tui_gateway.coding_workspaces import workspace_instructions
    config = json.loads(policy.config_json)
    config[_KEY] = prepared
    instructions = workspace_instructions(prepared['binding'])
    return replace(policy, config_json=json.dumps(config),
                   skills_prompt='\n\n'.join(p for p in (policy.skills_prompt, instructions) if p))


def workspace_binding(policy):
    if policy is None:
        return None
    return (json.loads(policy.config_json).get(_KEY) or {}).get('binding')


def verify_policy(policy, expected=None, *, task_id=None, session_key=None):
    """No branch constraint: pin the checkout inode and Git metadata ownership instead."""
    binding = workspace_binding(policy)
    if not binding:
        if expected is not None:
            raise RuntimeStoreError('invalid_params', field='coding_workspace', detail='session has no coding workspace')
        return None
    try:
        terminal = json.loads(policy.terminal_json)
        if (policy.cwd != binding['cwd'] or terminal.get('TERMINAL_CWD') != policy.cwd
                or str(terminal.get('TERMINAL_ENV') or 'local').strip().lower() != 'local'
                or (expected is not None and expected != binding['cwd'])):
            raise ValueError('Session or terminal workspace changed')
        prepared = json.loads(policy.config_json)[_KEY]
        saved = prepared['identity']
        if _identity(binding) != saved:
            raise ValueError('Prepared checkout moved or was replaced')
        _verify_terminal(binding, task_id or prepared['session_id'], session_key)
    except (OSError, ValueError) as exc:
        raise RuntimeStoreError('workspace_unavailable', field='coding_workspace', detail=str(exc)) from exc
    return binding


def _terminal_cwd_within(cwd, root):
    """Observed cd state may vary, but must resolve to an existing checkout directory."""
    try:
        path = Path(cwd)
        resolved = path.resolve(strict=True)
        return path.is_absolute() and resolved.is_dir() and resolved.is_relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False


def _verify_terminal(binding, task_id, session_key=None):
    from tools.terminal_tool import get_session_cwd, resolve_task_overrides
    overrides = resolve_task_overrides(task_id)
    # Configured task identity stays pinned; only observed persistent cd state may vary.
    if (overrides.get('cwd', binding['cwd']) != binding['cwd']
            or overrides.get('env_type', 'local') != 'local'):
        raise ValueError('Terminal workspace changed')
    for key in (task_id, session_key):
        observed = get_session_cwd(key) if key else None
        if observed is not None and not _terminal_cwd_within(observed, binding['cwd']):
            raise ValueError('Terminal workspace changed')


def terminal_identity(authority, session_id):
    """Resolve only this owned conversation's live transcript and route, never foreground state."""
    from gateway.session_contract import SessionRef
    live = authority.sessions[session_id]
    return {'task_id': authority.physical_target(SessionRef(authority.profile_id, session_id)),
            'session_key': live.route}


async def verify_admission(authority, ref):
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    if workspace_binding(policy):
        await asyncio.to_thread(verify_policy, policy, **terminal_identity(authority, ref.session_id))


def _probe_terminal(authority, policy, session_id, expected):
    identity = terminal_identity(authority, session_id)
    binding = verify_policy(policy, expected, **identity)
    if binding is None:
        raise RuntimeStoreError('invalid_params', field='coding_workspace', detail='session has no coding workspace')
    from gateway.session_policy import policy_scope
    from tools.approval_context import reset_current_session_key, set_current_session_key
    from tools.terminal_tool import terminal_tool
    with policy_scope(policy, authority=authority):
        token = set_current_session_key(identity['session_key'])
        try:
            result = json.loads(terminal_tool('pwd', task_id=identity['task_id'], timeout=10))
        finally:
            reset_current_session_key(token)
    if result.get('exit_code') != 0 or not _terminal_cwd_within(
            result.get('output', '').removesuffix('\n'), binding['cwd']):
        raise RuntimeStoreError('workspace_unavailable', field='coding_workspace', detail='terminal workspace changed')
    return {'cwd': binding['cwd'], 'gatewayCwd': binding['cwd']}


def _authorized_policy(connection, ref):
    authority = connection.authority
    authority.authorize(connection.actor, ref, 'session:read')
    source = authority.sessions[ref.session_id].source
    from gateway.config import Platform
    if source.platform != Platform.LOCAL or source.user_id != connection.actor.subject:
        raise RuntimeStoreError('permission_denied')
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, source)
    if not workspace_binding(policy):
        raise RuntimeStoreError('invalid_params', field='coding_workspace', detail='session has no coding workspace')
    return policy


async def verify(connection, ref, params):
    policy = _authorized_policy(connection, ref)
    return await asyncio.to_thread(_probe_terminal, connection.authority, policy, ref.session_id, params.get('cwd'))


def _references(policy, params):
    binding = verify_policy(policy)
    from tui_gateway.coding_workspaces import remap_workspace_reference_text, remap_workspace_references
    try:
        base = params.get('reference_cwd')
        if not base and any(not Path(path).expanduser().is_absolute() for path in params['paths']):
            raise ValueError('Original reference CWD is required')
        result: dict = {'paths': remap_workspace_references(binding, params['paths'], base)}
        if 'text' in params:
            result['text'] = remap_workspace_reference_text(binding, params['text'], base)
        return result
    except (OSError, ValueError) as exc:
        raise RuntimeStoreError('invalid_params', field='references', detail=str(exc)) from exc


async def references(connection, ref, params):
    policy = _authorized_policy(connection, ref)
    return await asyncio.to_thread(_references, policy, params)


def handlers(connection):
    from functools import partial
    return {'session.workspace.verify': partial(verify, connection),
            'session.workspace.references': partial(references, connection)}


def check_receipt_owner(conn, receipt):
    """Called in the durable creation transaction: two principals cannot own one claim."""
    prepared = json.loads(receipt['policy']['config_json']).get(_KEY)
    if not prepared:
        return
    from hermes_state_local import POLICY_PREFIX
    request = prepared['binding']['requestId']
    rows = conn.execute('SELECT value FROM state_meta WHERE key GLOB ?', (POLICY_PREFIX + '*',))
    for (raw,) in rows:
        try:
            other = json.loads(raw)
            bound = json.loads(other['policy']['config_json']).get(_KEY)
            other_request = bound['binding']['requestId'] if bound else None
            other_session = other['session_id'] if bound else None
        except (ValueError, TypeError, KeyError, AttributeError):
            # Malformed unrelated receipts name no owner (as in retirement scans).
            continue
        if other_request == request and other_session != receipt['session_id']:
            raise RuntimeStoreError('admission_conflict')


def finish_binding(authority, session_id):
    """Retryable post-commit projection: only the durable owner may mark the project claim bound."""
    from hermes_state_local import local_receipt
    from gateway.session_policy import restore_policy
    policy = restore_policy(local_receipt(authority.db, session_id)['policy'])
    binding = workspace_binding(policy)
    if not binding:
        return
    artifacts = Path(binding['artifactsPath'])
    if artifacts.resolve() != artifacts:
        raise RuntimeStoreError('workspace_unavailable', field='coding_workspace', detail='managed artifacts path changed')
    artifacts.mkdir(parents=True, exist_ok=True)
    from hermes_cli import projects_db as pdb
    with pdb.connect_closing(Path(authority.profile_id) / 'projects.db') as conn:
        pdb.set_workspace_claim_state(conn, binding['requestId'], 'bound')
