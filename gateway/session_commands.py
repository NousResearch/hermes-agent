"""Bound local slash commands; prompt directives use the client's durable submit.

Desktop and Ink both submit ``skill``/``send`` results with their own input ID.
Executing those here would double-submit and bypass their retry identity.
"""
import asyncio
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import shlex

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)


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


# The setting forms a local session cannot deliver: `/voice tts` asks the owner to voice every reply,
# and a local session's audio plays only in its client. Read like the handler reads it (no shlex): an
# unbalanced quote in any other `/voice` argument still reaches the handler.
_REFUSED_ARGUMENTS = {
    'voice': lambda arg: arg.strip().lower() == 'tts',
}


def command_verdict(name, arg=''):
    """``read`` / ``control`` / ``refused`` for a canonical registry command on a local session."""
    if _REFUSED_ARGUMENTS.get(name, lambda _a: False)(arg):
        return 'refused'
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
    """``('exec', shell)`` for a ``type: exec`` quick command, ``('plugin', registered)`` for a plugin
    slash command, else None (read in the owning profile's scope). The exact registered name wins over
    Telegram's ``_`` -> ``-`` form, as main's CLI looked it up: ``foo_bar`` is its own command."""
    from gateway.run import _load_gateway_config
    from hermes_cli.plugins import get_plugin_command_handler
    entry = (_load_gateway_config().get('quick_commands') or {}).get(name)
    if isinstance(entry, dict) and entry.get('type') == 'exec':
        return 'exec', str(entry.get('command') or '')
    return next((('plugin', registered) for registered in dict.fromkeys((name, name.replace('_', '-')))
                 if get_plugin_command_handler(registered)), None)


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
    # session's own were already resolved above in its profile (``_resolve_alias``/``_runner_command``),
    # so is the registered name (``shell``) the handler is looked up by.
    from hermes_cli.plugins import get_plugin_command_handler
    handler = get_plugin_command_handler(shell)
    if handler is None:
        raise RuntimeStoreError('unsupported_command')
    event = MessageEvent(text='/' + name + (' ' + arg if arg else ''), source=live.source)
    # The session's own entry, so HERMES_SESSION_ID is bound as on its turns: a background job
    # the handler starts records this session as its parent.
    entry = await asyncio.to_thread(runner.session_store.lookup_by_session_key, live.route)
    try:
        # The argument as typed, like main's CLI: no messaging dash repair.
        return await runner._hm_run_plugin_handler(handler, event, live.source, entry, args=arg)
    except Exception as e:  # main's CLI printed it; messaging (run_inbound_hooks) still only logs it
        logger.warning("Plugin command dispatch failed: %s", e, exc_info=True)
        return t('cli.plugin.command_error', error=str(e) or type(e).__name__)


def _skill_directive(name, arg, route):
    """A local session's ``/skill`` (its plugin skills included, as on main's interactive surfaces),
    stacked ``/skill-a /skill-b do X`` too: built here, so no client loads plugins to resolve one."""
    from agent.i18n import t
    from agent.skill_commands import (
        build_skill_invocation_message, build_stacked_skill_invocation_message, describe_skill_invocation,
        get_interactive_skill_commands, resolve_slash_key, split_stacked_skill_commands,
    )
    skills = get_interactive_skill_commands()
    key = resolve_slash_key(name, skills)
    if key is None:
        raise RuntimeStoreError('unsupported_command')
    extra, instruction = split_stacked_skill_commands(arg, interactive=True)
    stacked = build_stacked_skill_invocation_message([key, *extra], instruction, task_id=route) if extra else None
    message = stacked[0] if stacked else build_skill_invocation_message(key, arg, task_id=route)
    if not message:
        raise RuntimeStoreError('command_unavailable')
    missing = stacked[2] if stacked else []
    return {'type': 'skill', 'name': ', '.join(stacked[1]) if stacked else skills[key]['name'], 'message': message,
            'display': describe_skill_invocation(message, separator=' '),
            **({'notice': t('cli.skills.skipped_missing', names=', '.join(missing))} if missing else {})}


async def _tui_read(authority, actor, ref, name, arg, before):
    """Main's TUI sidecar text for a TUI session's /status, /context or /cron read, else None;
    ``before`` (its ``pre_command``) runs once the read is accepted."""
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
    await before()
    return await run_tui_read(authority, ref, live, policy, render, arg)


# Messaging's running-agent escape hatches (hooks.md ``pre_command``: ``/stop``, approval controls):
# while a turn runs no plugin observer sits in front of them. Every other command that runs fires.
_ESCAPE_HATCHES = frozenset({'stop'}) | _APPROVAL_COMMANDS
_PRE_COMMAND_PENDING = 32  # one session's queued deliveries; past it a delivery is dropped (logged)
_PRE_COMMAND_TAILS = {}  # (profile, session id) -> (that session's last queued delivery, pending count)


async def _on_own_thread(fn):
    """``fn()`` on a daemon thread of its own (under the caller's contextvars), awaited on the loop: never
    the loop's shared executor, and no shared worker a callback waiting out its bound
    (``plugins.hook_callback_timeout``, enforced by the hook dispatcher: abandoned worker, suppression,
    cap) holds from another session. One session runs one delivery at a time, so this is at most one
    thread per session with a delivery in flight."""
    from agent.memory_provider import spawn_context_thread
    loop = asyncio.get_running_loop()
    done = loop.create_future()

    def settle():
        if not done.done():  # else the delivery was cancelled
            done.set_result(None)

    def run():
        try:
            fn()
        except Exception:  # fire_pre_command_hook never raises: anything here is a bug, never the caller's
            logger.exception('pre_command delivery failed')
        finally:
            try:
                loop.call_soon_threadsafe(settle)
            except RuntimeError:  # the loop closed (gateway shutdown): nobody awaits it any more
                pass
    try:
        spawn_context_thread(run, name='hermes-pre-command').start()
    except RuntimeError as exc:  # no thread available: this delivery is skipped, as the dispatcher skips
        logger.warning('pre_command delivery skipped: %s', exc)
        return
    await done


def _pre_command(authority, ref, canonical, alias, arg):
    """Queue main's ``pre_command`` observer for a recognized command typed in a local session, fired
    here for every surface (classic CLI, Ink, Desktop), so no client loads plugin code to fire it; the
    delivery task, or None when not fired. One session's deliveries fire in arrival order; the hook
    dispatcher bounds each callback, and one still running for this session (hung) is skipped for its
    later commands, never run out of order, while other sessions' deliveries go on. ``surface`` and
    ``platform`` keep main's ``cli`` on every local surface (Ink and Desktop fired it through the CLI
    slash worker). During a turn every command that still runs fires, as main's CLI did, except the
    escape hatches."""
    from gateway.run import _profile_runtime_scope
    from gateway.session_local import authorize_local_source
    live = authority.sessions[ref.session_id]
    if ((canonical in _ESCAPE_HATCHES and authority._handle(ref).execution_state != 'idle')
            or authorize_local_source(authority.runner, live.source) is not True):
        return None
    home, key = Path(authority.profile_id), (authority.profile_id, ref.session_id)
    previous, pending = _PRE_COMMAND_TAILS.get(key, (None, 0))
    if previous is not None and previous.get_loop() is not asyncio.get_running_loop():
        previous, pending = None, 0  # a retired loop's delivery (an in-process restart) never resolves on this one
    if pending >= _PRE_COMMAND_PENDING:
        logger.warning('pre_command for /%s dropped: %d deliveries already queued for this session', canonical, pending)
        return None

    def fire():  # bound to the session it fires for: a key-less inject_message from the hook lands there
        from gateway.session_context import scoped_session_key
        from hermes_cli.plugins import fire_pre_command_hook
        with _profile_runtime_scope(home), scoped_session_key(live.route):
            fire_pre_command_hook(surface='cli', command=canonical, alias_used=alias, args_raw=arg,
                                  session_key=ref.session_id, platform='cli')

    async def deliver():
        if previous is not None:
            await asyncio.wait({previous})
        await _on_own_thread(fire)

    def settle(done):
        tail, count = _PRE_COMMAND_TAILS.get(key, (None, 0))
        if tail is done:
            _PRE_COMMAND_TAILS.pop(key)
        elif count:
            _PRE_COMMAND_TAILS[key] = (tail, count - 1)

    task = asyncio.ensure_future(deliver())
    _PRE_COMMAND_TAILS[key] = (task, pending + 1)
    task.add_done_callback(settle)
    return task


async def _pre_command_before(authority, ref, canonical, alias, arg):
    """``pre_command`` before the gateway runs the command itself (``slash.exec``/``command.dispatch``)."""
    task = _pre_command(authority, ref, canonical, alias, arg)
    if task is not None:
        await asyncio.wait({task})


async def observe_command(connection, ref, params):
    """``command.observe``: a recognized command the classic CLI or Ink runs itself on this session.
    ``pre_command`` fires here, in the session's profile, after the reply: the client never waits on a
    plugin and never loads one. Plain data only (the command name as typed and its arguments)."""
    from hermes_cli.commands import resolve_command
    from hermes_cli.profiles import profile_matches_home
    authority, actor = connection.authority, connection.actor
    authority.authorize(actor, ref, 'session:read')
    name, arg = _parse(params, True)
    if params.get('profile') and not profile_matches_home(params['profile'], Path(authority.profile_id)):
        raise RuntimeStoreError('profile_mismatch')
    definition = resolve_command(name)
    if definition is None:
        raise RuntimeStoreError('unsupported_command')
    return {'queued': _pre_command(authority, ref, definition.name, name.lower(), arg) is not None}


@contextmanager
def _accepting(before):
    """``before`` as the busy dispatcher's acceptance hook: a mid-run variant that refuses its arguments
    (``/goal <new text>``) never reaches it, so a refused command fires no ``pre_command``."""
    from gateway.run_busy import BUSY_ACCEPTED
    token = BUSY_ACCEPTED.set(before)
    try:
        yield
    finally:
        BUSY_ACCEPTED.reset(token)


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

    fired = []

    async def before():  # pre_command, once, for a recognized command that is about to run
        if not fired:
            fired.append(canonical)
            await _pre_command_before(authority, ref, canonical, name.lower(), arg)
    if definition is not None and (tui := await _tui_read(authority, actor, ref, canonical, arg, before)) is not None:
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
    if definition is not None and (canonical in _INFO_READS or authority._handle(ref).execution_state == 'idle'):
        await before()  # mid-run the busy dispatcher fires it once it accepts the command (``BUSY_ACCEPTED``)
    if runner_command is not None:
        with _profile_runtime_scope(home), _session_workspace(authority.runner, live.source):
            output = await _run_runner_command(authority.runner, live, name, arg, *runner_command)
        return {'type': 'exec', 'output': _text(output)}
    if definition is None:
        def build():  # in the session's folder: its trusted project skills, as its catalog lists them
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
    with (_profile_runtime_scope(home), _session_workspace(runner, live.source), capture() as sink,
          _accepting(before)):
        if authority._handle(ref).execution_state != 'idle':
            output = await runner._dispatch_busy_slash_command(event, definition, live.route, live.source)
        elif canonical in _PROMPT_COMMANDS:
            await before()  # a no-op unless the turn ended since the check above
            handled, output = await getattr(runner, '_hm_cmd_' + canonical.replace('-', '_'))(
                event, live.source, live.route)
            if not handled:
                prompt = event.text
        else:
            await before()
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
