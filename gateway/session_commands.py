"""Bound local slash commands; prompt directives use the client's durable submit.

Desktop and Ink both submit ``skill``/``send`` results with their own input ID.
Executing those here would double-submit and bypass their retry identity.
"""
import asyncio
from contextlib import contextmanager
import os
from pathlib import Path
import shlex

from hermes_state_runtime import RuntimeStoreError


# Reviewed gateway handlers only: listing a command is not permission to run it. Each read was
# run through its real handler against a local session (website/docs/developer-guide/
# gateway-command-parity.md); a handler that reads messaging-only state or writes config is not here.
_READ_COMMANDS = frozenset({'help', 'commands', 'status', 'context', 'version', 'whoami',
                            'usage', 'insights', 'profile', 'diff', 'memory', 'kanban', 'topup',
                            'suggestions', 'bundles', 'skills'})
_CONTROL_COMMANDS = frozenset({'title'})
# Authoring/workflow commands whose messaging handler (``_command_handler_table``) runs as-is for a
# local session: goal/loop/heartbeat state on the session's row, the review/refine forks of its
# cached agent, the /debug summary upload, /blueprint's job creation. A turn the handler would have
# pushed onto the adapter FIFO comes back as a ``send`` directive (gateway/session_command_sink.py).
_HANDLER_CONTROLS = frozenset({'goal', 'subgoal', 'loop', 'heartbeat', 'review', 'refine', 'debug', 'curator',
                               'login', 'autoprune'})
# Idle-path commands the messaging gateway answers by rewriting the turn (``_hm_cmd_<name>``,
# gateway/run_inbound.py): the rewritten text is returned as the client's ``send`` directive.
_PROMPT_COMMANDS = frozenset({'plan', 'learn', 'init', 'moa', 'blueprint', 'egress'})
# Per-session settings: the messaging handler itself, made session-correct for a local session
# (gateway/session_setting_commands.py). Admission-fenced like any control.
_SETTING_COMMANDS = frozenset({'reasoning', 'voice', 'approvals', 'codex-runtime', 'footer', 'busy', 'verbose',
                               'yolo', 'fast', 'personality'})
# Typed `/approve` / `/deny`: the messaging handler resolves THIS session's waiting approval (its
# route is the session key), the same waiter `approval.respond` answers by prompt id.
_APPROVAL_COMMANDS = frozenset({'approve', 'deny'})
# Reports with no messaging handler: the classic CLI's own text, built by the owner for this
# session (gateway/session_info_reads.py).
_INFO_READS = frozenset({'toolsets', 'platforms'})
# Commands that move the session's goal/loop/heartbeat state: every viewer's cards repaint after them.
_AUTOMATION_COMMANDS = frozenset({'goal', 'subgoal', 'loop', 'heartbeat'})

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
    'skills': lambda arg: _first_token(arg) in {'', 'pending', 'diff'},
    'usage': lambda arg: not arg.strip(),
    'memory': lambda arg: arg.strip().lower() in {'', 'pending'},
    'suggestions': lambda arg: not arg.strip(),
    'kanban': lambda arg: _first_token(arg) in _KANBAN_READS,
}


# The write forms of a command whose reads are above: the messaging handler's own review verbs,
# applied for this session's profile (`/skills approve|reject` the staged queue, `/skills approval`
# its config.yaml switch, as main's CLI wrote it).
_CONTROL_ARGUMENTS = {
    'skills': lambda arg: _first_token(arg) in {'approve', 'apply', 'reject', 'deny', 'drop', 'approval', 'mode'},
}


def command_verdict(name, arg=''):
    """``read`` / ``control`` / ``refused`` for a canonical registry command on a local session."""
    if (name in _CONTROL_COMMANDS or name in _HANDLER_CONTROLS or name in _PROMPT_COMMANDS
            or name in _SETTING_COMMANDS or name in _APPROVAL_COMMANDS
            or _CONTROL_ARGUMENTS.get(name, lambda _a: False)(arg)):
        return 'control'
    if name in _INFO_READS or (name in _READ_COMMANDS and _READ_ARGUMENTS.get(name, lambda _arg: True)(arg)):
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
        if not isinstance(entry, dict) or entry.get('type') not in {'alias', 'exec'}:
            raise RuntimeStoreError('unsupported_command')
        if entry['type'] == 'exec':
            return name, arg
        target = entry.get('target')
        if not isinstance(target, str) or not target.strip():
            raise RuntimeStoreError('invalid_params')
        parts = target.strip().lstrip('/').split(None, 1)
        name = parts[0]
        arg = ' '.join(p for p in (*parts[1:], arg) if p)
    return name, arg


def _runner_command(name):
    """``('exec', shell)`` for a ``type: exec`` quick command, ``('plugin', None)`` for a plugin
    slash command, else None (read in the owning profile's scope)."""
    from gateway.run import _load_gateway_config
    from hermes_cli.plugins import get_plugin_command_handler
    entry = (_load_gateway_config().get('quick_commands') or {}).get(name)
    if isinstance(entry, dict) and entry.get('type') == 'exec':
        return 'exec', str(entry.get('command') or '')
    return ('plugin', None) if get_plugin_command_handler(name.replace('_', '-')) else None


async def _run_runner_command(runner, live, name, arg, kind, shell):
    """The messaging gateway's own quick-command / plugin-command dispatch for this session
    (``run_inbound_hooks``): the operator's slash access policy, the 30 s sanitized-env exec with
    redacted output, the plugin handler under this session's env scope."""
    from agent.i18n import t
    from gateway.platforms.base import MessageEvent
    if kind == 'exec':
        if not shell:
            return t('gateway.quick_command.no_command', command=name)
        refused = runner._check_slash_access(live.source, name)
        if refused:
            return refused
        from gateway.session_policy import policy_for_source
        policy = policy_for_source(runner, live.source)
        cwd = getattr(policy, 'cwd', None) or None
        if cwd is None or not await asyncio.to_thread(os.path.isdir, cwd):
            # Never fall back to the gateway's own folder: a relative command would act there.
            raise RuntimeStoreError('cwd_unavailable')
        return await runner._hm_run_exec_quick_command(name, shell, cwd=cwd)
    # Plugin-only: the runner's quick commands are the launch profile's under multiplex, and this
    # session's own were already resolved above in its profile (``_resolve_alias``/``_runner_command``).
    event = MessageEvent(text='/' + name + (' ' + arg if arg else ''), source=live.source)
    _handled, output, _command = await runner._hm_dispatch_plugin_command(event, live.source, name)
    return output


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


async def _tui_read(authority, actor, ref, name, arg):
    """Main's TUI sidecar text for a TUI session's /status, /context or /cron read, else None."""
    from gateway.session_local import authorize_local_source
    from gateway.session_policy import policy_for_source
    from gateway.session_tui_reads import run_tui_read, tui_read
    live = authority.sessions[ref.session_id]
    if authorize_local_source(authority.runner, live.source) is not True:
        return None
    policy = policy_for_source(authority.runner, live.source)
    render = tui_read(policy, name, arg)
    if render is None:
        return None
    authority.authorize(actor, ref, 'session:read')
    return await run_tui_read(authority, ref, live, policy, render, arg)


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
    if definition is not None and (tui := await _tui_read(authority, actor, ref, canonical, arg)) is not None:
        return tui
    runner_command = None
    if definition is None:
        def lookup():
            with _profile_runtime_scope(home):
                return _runner_command(name)
        runner_command = await asyncio.to_thread(lookup)
    capability = ('session:read' if verdict == 'read' else
                  'session:approve' if canonical in _APPROVAL_COMMANDS else
                  'session:control' if definition or runner_command else 'session:submit')
    authority.authorize(actor, ref, capability)
    if verdict == 'refused':
        raise RuntimeStoreError('unsupported_command')
    from gateway.session_local import authorize_local_source
    live = authority.sessions[ref.session_id]
    if authorize_local_source(authority.runner, live.source) is not True:
        raise RuntimeStoreError('permission_denied')
    if capability != 'session:read':
        authority._require_admission_open()
    if runner_command is not None:
        with _profile_runtime_scope(home), _session_workspace(authority.runner, live.source):
            output = await _run_runner_command(authority.runner, live, name, arg, *runner_command)
        return {'type': 'exec', 'output': _text(output)}
    if definition is None:
        def build():
            with _profile_runtime_scope(home), _session_workspace(authority.runner, live.source):
                return _skill_directive(name, arg, live.route)
        return await asyncio.to_thread(build)

    if canonical in _INFO_READS:
        from gateway.session_info_reads import run_info_read
        return await run_info_read(authority, live, canonical)

    from gateway.platforms.base import MessageEvent
    from gateway.session_command_sink import capture
    typed = '/' + canonical + (' ' + arg if arg else '')
    event = MessageEvent(text=typed, source=live.source)
    runner = authority.runner
    prompt = None
    # Use the registry's existing busy contract, not a local approximation.
    with _profile_runtime_scope(home), _session_workspace(runner, live.source), capture() as sink:
        if authority._handle(ref).execution_state != 'idle':
            output = await runner._dispatch_busy_slash_command(event, definition, live.route, live.source)
        elif canonical in _PROMPT_COMMANDS:
            handled, output = await getattr(runner, '_hm_cmd_' + canonical.replace('-', '_'))(
                event, live.source, live.route)
            if not handled:
                prompt = event.text
        else:
            handler = runner._command_handler_table((canonical,))[canonical]
            output = await handler(event)
    if canonical in _APPROVAL_COMMANDS:
        # Settle the answered prompt for every viewer now (`approval.settled`), as approval.respond does.
        handle = authority._handle(ref)
        live.controls.snapshot(ref.session_id, handle.execution_generation
                               if handle.execution_state == 'running' else None)
    if canonical in _AUTOMATION_COMMANDS:
        from gateway.session_control_actions import publish
        publish(authority, ref.session_id)
    prompt = prompt if prompt is not None else sink.prompt
    if prompt:
        # The client submits the turn with its own durable input id; the gateway never does.
        notice = '\n'.join(filter(None, [*sink.notices, str(output or '')]))
        return {'type': 'send', 'message': prompt, 'display': sink.display or typed,
                **({'notice': notice} if notice else {})}
    return {'type': 'exec', 'output': '\n'.join(filter(None, [*sink.notices, _text(output)]))}


def _text(output):
    """A handler reply as text: ``/blueprint``'s result object carries it in ``.text``."""
    return str(getattr(output, 'text', output) or '')
