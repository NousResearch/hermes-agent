"""Bound local slash commands; prompt directives use the client's durable submit.

Desktop and Ink both submit ``skill``/``send`` results with their own input ID.
Executing those here would double-submit and bypass their retry identity.
"""
import asyncio
from contextlib import contextmanager
from pathlib import Path
import shlex

from hermes_state_runtime import RuntimeStoreError


# Reviewed gateway handlers only: listing a command is not permission to run it. Each read was
# run through its real handler against a local session (website/docs/developer-guide/
# gateway-command-parity.md); a handler that reads messaging-only state or writes config is not here.
_READ_COMMANDS = frozenset({'help', 'commands', 'status', 'context', 'version', 'whoami',
                            'usage', 'insights', 'profile', 'diff', 'memory', 'kanban', 'topup',
                            'suggestions', 'bundles'})
_CONTROL_COMMANDS = frozenset({'title'})

# Kanban verbs that only read the board. ``create`` would also subscribe the LOCAL chat to
# messaging notifications, and every other verb mutates the board.
_KANBAN_READS = frozenset({'', 'help', '--help', '-h', '?', 'list', 'ls', 'show', 'stats', 'assignees',
                           'context', 'runs', 'log', 'diagnostics', 'diag'})


def _first_token(arg):
    tokens = iter(shlex.split(arg) if arg.strip() else [])
    for token in tokens:
        if token == '--board':
            next(tokens, None)
        elif not token.startswith('--board='):
            return token.lower()
    return ''


# The read-only argument forms of a read command whose other subcommands write: ``/usage reset``
# redeems a credit, ``/memory approve|approval`` applies writes / edits config.yaml,
# ``/suggestions accept|catalog`` schedules cron jobs delivering to this (non-messaging) chat.
_READ_ARGUMENTS = {
    'usage': lambda arg: not arg.strip(),
    'memory': lambda arg: arg.strip().lower() in {'', 'pending'},
    'suggestions': lambda arg: not arg.strip(),
    'kanban': lambda arg: _first_token(arg) in _KANBAN_READS,
}


def command_verdict(name, arg=''):
    """``read`` / ``control`` / ``refused`` for a canonical registry command on a local session."""
    if name in _CONTROL_COMMANDS:
        return 'control'
    if name in _READ_COMMANDS and _READ_ARGUMENTS.get(name, lambda _arg: True)(arg):
        return 'read'
    return 'refused'


def _parse(params, dispatch):
    allowed = {'session_id', 'profile', 'name', 'arg'} if dispatch else {'session_id', 'profile', 'command'}
    if (set(params) - allowed or any(not isinstance(v, str) for v in params.values())):
        raise RuntimeStoreError('invalid_params')
    text = params.get('name', '') if dispatch else params.get('command', '')
    parts = text.strip().lstrip('/').split(None, 1)
    if not parts or (dispatch and len(parts) != 1):
        raise RuntimeStoreError('invalid_params')
    return parts[0], params.get('arg', '') if dispatch else (parts[1] if len(parts) > 1 else '')


def _resolve_alias(name, arg):
    from gateway.run import _load_gateway_config
    from hermes_cli.commands import resolve_command
    quick = _load_gateway_config().get('quick_commands') or {}
    seen = set()
    while resolve_command(name) is None and name in quick:
        if name in seen:
            raise RuntimeStoreError('invalid_params')
        seen.add(name)
        entry = quick[name]
        # Shell/plugin execution has no canonical approval or retry contract here.
        if not isinstance(entry, dict) or entry.get('type') != 'alias':
            raise RuntimeStoreError('unsupported_command')
        target = entry.get('target')
        if not isinstance(target, str) or not target.strip():
            raise RuntimeStoreError('invalid_params')
        parts = target.strip().lstrip('/').split(None, 1)
        name = parts[0]
        arg = ' '.join(p for p in (*parts[1:], arg) if p)
    return name, arg


def _skill_directive(name, arg, route):
    from agent.skill_commands import (
        build_skill_invocation_message, describe_skill_invocation, get_skill_commands,
    )
    key = '/' + name
    skills = get_skill_commands()
    if key not in skills:
        raise RuntimeStoreError('unsupported_command')
    message = build_skill_invocation_message(key, arg, task_id=route)
    if not message:
        raise RuntimeStoreError('command_unavailable')
    return {'type': 'skill', 'name': skills[key]['name'], 'message': message,
            'display': describe_skill_invocation(message, separator=' ')}


@contextmanager
def _session_workspace(runner, source):
    """The session's frozen terminal policy (``TERMINAL_CWD`` = its launch cwd), so ``/diff``
    reads the caller's checkout instead of the gateway process's configured directory."""
    import json
    from agent.runtime_cwd import reset_session_cwd, set_session_cwd
    from gateway.session_policy import policy_for_source
    from tools.terminal_scope import reset_terminal_scope, set_terminal_scope
    policy = policy_for_source(runner, source)
    if policy is None:
        yield
        return
    cwd_token = set_session_cwd(policy.cwd)
    terminal_token = set_terminal_scope(json.loads(policy.terminal_json))
    try:
        yield
    finally:
        reset_terminal_scope(terminal_token)
        reset_session_cwd(cwd_token)


async def execute_command(connection, ref, params, *, dispatch=False):
    authority, actor = connection.authority, connection.actor
    authority.authorize(actor, ref, 'session:read')
    name, arg = _parse(params, dispatch)
    from hermes_cli.commands import resolve_command
    from hermes_cli.profiles import profile_matches_home
    home = Path(authority.profile_id)
    if params.get('profile') and not profile_matches_home(params['profile'], home):
        raise RuntimeStoreError('profile_mismatch')

    from gateway.run import _profile_runtime_scope
    def resolve():
        with _profile_runtime_scope(home):
            return _resolve_alias(name, arg)
    name, arg = await asyncio.to_thread(resolve)
    definition = resolve_command(name)
    canonical = definition.name if definition else name
    verdict = command_verdict(canonical, arg) if definition else None
    capability = ('session:read' if verdict == 'read' else
                  'session:control' if definition else 'session:submit')
    authority.authorize(actor, ref, capability)
    if verdict == 'refused':
        raise RuntimeStoreError('unsupported_command')
    from gateway.session_local import authorize_local_source
    live = authority.sessions[ref.session_id]
    if authorize_local_source(authority.runner, live.source) is not True:
        raise RuntimeStoreError('permission_denied')
    if capability != 'session:read':
        authority._require_admission_open()
    if definition is None:
        def build():
            with _profile_runtime_scope(home):
                return _skill_directive(name, arg, live.route)
        return await asyncio.to_thread(build)

    from gateway.platforms.base import MessageEvent
    event = MessageEvent(text='/' + canonical + (' ' + arg if arg else ''), source=live.source)
    # Use the registry's existing busy contract, not a local approximation.
    with _profile_runtime_scope(home), _session_workspace(authority.runner, live.source):
        if authority._handle(ref).execution_state != 'idle':
            output = await authority.runner._dispatch_busy_slash_command(event, definition, live.route, live.source)
        else:
            handler = authority.runner._command_handler_table((canonical,))[canonical]
            output = await handler(event)
    return {'type': 'exec', 'output': str(output or '')}
