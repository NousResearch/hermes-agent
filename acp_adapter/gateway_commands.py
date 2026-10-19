"""Editor slash commands on the gateway path; each one is an existing gateway verb.

Main served these in-process (``acp_adapter/commands.py``). The reports are the owner's own
``slash.exec`` reads, ``/model``, ``/compress`` and ``/reset`` the retained ``session.mutate``
controls, ``/steer`` the owner's ``session.steer``; ``/queue`` and an idle ``/steer`` become the
next prompt, which the session's FIFO runs after any turn in flight. ``/title <name>`` is the
``rename`` mutation and ``/undo`` / ``/retry`` the ``rewind`` one, as in the classic transport
client (``hermes_cli/gateway_chat_commands.py``). Every other registry command goes to the owner's
``slash.exec`` like there: it runs what it serves for this session and refuses the rest, and that
refusal reaches the editor as a reply naming the command, never a failed prompt.
"""
from __future__ import annotations

from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError


class CommandRefused(GatewayClientError):
    """A refusal the editor shows as the command's reply (the prompt itself ends normally)."""


async def run_command(agent, session_id, command, arg):
    """``(reply, None)`` for a command answered here, or ``(None, text)`` when ``text`` is to be
    submitted as the prompt. A refusal raises ``CommandRefused`` (or the owner's RPC error).

    ``ROUTES`` (below) is the one table of what ACP answers itself; every other registry command,
    the reads included, is the owner's ``slash.exec`` answer (``_forward``)."""
    route = ROUTES.get(command)
    if route is None:
        return await _forward(agent, await agent._client(), session_id, command, arg)
    return await route[1](agent, session_id, command, arg)


async def _branch(agent, session_id, command, arg):
    raise CommandRefused("/branch: use your editor's fork session action (ACP session/fork)")


async def _help(agent, session_id, command, arg):
    from acp_adapter.gateway_session_meta import help_text
    return help_text(), None


async def _prompt(agent, session_id, command, arg):
    if not arg:
        return f"Usage: /{command} <{'guidance' if command == 'steer' else 'prompt'}>", None
    steered = await _steer(agent, session_id, arg) if command == 'steer' else None
    return (steered, None) if steered else (None, arg)


async def _title(agent, session_id, command, arg):
    client = await agent._client()
    if not arg:
        return await _forward(agent, client, session_id, command, arg)
    return await _rename(agent, client, session_id, arg), None


async def _info(agent, session_id, command, arg):
    client = await agent._client()
    if command == 'model' and arg:
        return await _mutate(agent, client, session_id, command, arg), None
    info = await client.rpc('session.info', session_id=session_id)
    return (_tools_text(info) if command == 'tools' else _model_text('Current model', info)), None


async def _mutation(agent, session_id, command, arg):
    return await _mutate(agent, await agent._client(), session_id, command, arg), None


async def _rewind_command(agent, session_id, command, arg):
    return await _rewind(agent, await agent._client(), session_id, command, arg)


async def _discard_unknown(agent, session_id, _command, admission_id):
    """Acknowledge one exact lost turn and refresh its FIFO snapshot."""
    if not admission_id:
        return 'Usage: /discard <lost admission id>', None
    # Pending-only refreshes: adopting the whole snapshot would advance its reply cursor past
    # messages and approvals the editor's event pump has not projected yet.
    await agent._refresh_unknown(session_id)
    lost = next((row for row in agent._snapshots[session_id].get('pending', [])
                 if row.get('admission_id') == admission_id and row.get('status') == 'unknown'), None)
    if lost is None:
        return (f'No unknown (lost) admission with id {admission_id}; refresh the session and try again.', None)
    client = await agent._client()
    await client.rpc('prompt.resolve_unknown', session_id=session_id, admission_id=admission_id,
                     execution_generation=lost['execution_generation'])
    await agent._refresh_unknown(session_id)
    return 'Discarded the lost turn. Queued messages can continue.', None


async def _forward(agent, client, session_id, command, arg):
    """The owner's ``slash.exec`` answer: its report, a turn to submit (a skill body or the turn a
    command built, after its notice), or the shared refusal for a command it does not serve."""
    import acp
    from hermes_cli.gateway_chat_commands import refusal
    try:
        reply = await client.rpc('slash.exec', session_id=session_id, command=f'/{command} {arg}'.strip())
    except GatewayRPCError as exc:
        if str(exc) != 'unsupported_command':
            raise
        raise CommandRefused(refusal(command)) from exc
    if reply.get('type') in ('skill', 'send'):
        if reply.get('notice') and agent._conn:
            await agent._conn.session_update(session_id=agent._editor_id(session_id),
                                             update=acp.update_agent_message_text(reply['notice'] + '\n'))
        return None, reply['message']
    return str(reply.get('output') or reply.get('notice') or f'/{command}: no output'), None


async def _rename(agent, client, session_id, title):
    payload = {'title': title}
    result = await agent._mutations.apply(client, session_id, 'rename', payload)
    agent._mutations.acknowledge(session_id, 'rename', payload)
    return f"Session title set: {result.get('title') or title}"


async def _rewind(agent, client, session_id, command, arg):
    """``/undo [N]`` reports the removed message (an editor has no composer to put it back in);
    ``/retry`` resubmits the last user turn's lossless text as this prompt."""
    from agent.i18n import t
    from hermes_cli.gateway_chat_commands import _rewind_to
    retry = command == 'retry'
    if not retry and arg and not (arg.isdigit() and int(arg) > 0):
        raise CommandRefused('Usage: /undo [N]')
    turns = 1 if retry else int(arg or 1)
    target = _rewind_to(turns, retry=retry)

    def payload(snapshot):
        # The target check runs on the CAS snapshot before anything is archived: its refusal (an
        # image turn /retry cannot resend) is the command's reply. Transport errors stay failures.
        try:
            return target(snapshot)
        except GatewayClientError as exc:
            raise CommandRefused(str(exc)) from exc
    result = await agent._mutations.apply(client, session_id, 'rewind', payload)
    if result.get('status') == 'nothing' or not result.get('rewound_count'):
        agent._mutations.acknowledge(session_id, 'rewind', payload)
        return t('cli.session.retry_no_user_message' if retry else 'cli.session.undo_no_user_message'), None
    # Retained until the refresh lands: a repeat after a lost refresh replays this rewind instead of
    # deriving a new target from the already-shortened transcript (which would drop the turn before).
    agent._snapshots[session_id] = await client.rpc('session.resume', session_id=session_id)
    agent._mutations.acknowledge(session_id, 'rewind', payload)
    content = (result.get('target_message') or {}).get('content')
    if retry:
        from agent.context_compressor import retryable_user_text
        try:
            text = retryable_user_text(content)
        except ValueError as exc:
            raise CommandRefused(f'/retry: {exc}') from exc
        if not text.strip():
            return t('cli.session.retry_no_user_message'), None
        return None, text
    removed = content if isinstance(content, str) else ''
    count = result['rewound_count']
    return (f"Undid {turns} turn{'s' if turns != 1 else ''} ({count} message(s)).\n"
            + (f"Removed message: {removed}" if removed else '')).rstrip(), None


async def _steer(agent, session_id, text):
    """Guidance for this editor's running turn; None when there is none to steer (main ran an idle
    steer as a plain prompt, and a turn that ended meanwhile leaves the text for the next one).

    The generation steered is our own admission's (``prompt.receipt``), never the session's latest:
    while our prompt still waits behind another client's turn, that turn is not ours to redirect."""
    admission_id = agent._admissions.get(session_id)
    if admission_id is None:
        return None
    client = await agent._client()
    receipt = await client.rpc('prompt.receipt', session_id=session_id, admission_id=admission_id)
    if receipt.get('status') != 'started':
        return None
    try:
        result = await client.rpc('session.steer', session_id=session_id, text=text,
                                  execution_generation=receipt['execution_generation'])
    except GatewayRPCError as exc:
        # Only a turn that ended meanwhile (its generation moved on) leaves the text for the next
        # prompt; any other refusal (permission, unsupported control, not ready) is the answer.
        if str(exc) == 'stale_generation':
            return None
        raise
    if result.get('status') != 'queued':
        return None
    return f"⏩ Steer queued for the active turn: {text[:80]}{'...' if len(text) > 80 else ''}"


async def _mutate(agent, client, session_id, command, arg):
    from hermes_cli.gateway_mutations import slash_mutation
    operation, payload = ('reset', {}) if command in {'new', 'clear'} else slash_mutation('/' + command, arg)
    confirm = agent._model_confirmer(session_id) if operation == 'model' else None
    result = await agent._mutations.apply(client, session_id, operation, payload, confirm=confirm)
    # Answers that changed nothing (declined, deferred to the next turn, a preview, nothing to remove).
    unchanged = _UNCHANGED.get(result.get('status'))
    if unchanged is not None:
        reply = unchanged(result)
    else:
        agent._snapshots[session_id] = await client.rpc('session.resume', session_id=session_id)
        if operation in {'reset', 'compress'}:
            await _refresh_context(agent, client, session_id, operation)
        reply = {'model': lambda: _model_text('Model switched to', result),
                 'compress': lambda: _summary_text(result) or 'Context compressed.',
                 'reset': lambda: 'Conversation history cleared.'}[operation]()
    agent._mutations.acknowledge(session_id, operation, payload)
    return reply


def _cancelled_text(_result):
    from agent.i18n import t
    return t('cli.model.switch_cancelled')


# A mutation answer that committed nothing, so the editor's snapshot is still current: a declined
# guarded pick, a pick during a running turn (stored for the next turn), a read-only compress
# preview, and a compress with nothing removable (the owner's shared report).
_UNCHANGED = {
    'cancelled': _cancelled_text,
    'deferred': lambda result: deferred_pick_text(result),
    'preview': lambda result: '\n'.join(result['lines']),
    'noop': lambda result: _summary_text(result) or 'No changes from compression.',
}


async def _refresh_context(agent, client, session_id, operation):
    """The editor's context indicator after /reset or /compress. Both are session boundaries that
    evict the owner's agent, so ``session.stats`` has no window until the next turn builds one: the
    window shown stays (the model is unchanged) and its occupancy is the new transcript's, nothing
    after a reset, the compressed messages' estimate (the summary's own figure) after a compress."""
    stats = await client.rpc('session.stats', session_id=session_id)
    if stats.get('context_length') is None:
        from agent.model_metadata import estimate_request_tokens_rough
        messages = [m for m in agent._snapshots[session_id].get('messages') or [] if not m.get('archived')]
        stats = {**stats, 'context_length': (agent._usage.get(session_id) or {}).get('context_length'),
                 'context_tokens': estimate_request_tokens_rough(messages) if operation == 'compress' else 0}
    await agent._project_usage(session_id, agent._editor_id(session_id), stats)


def deferred_pick_text(result):
    return f"{result.get('model') or 'The picked model'} applies from your next message."


def refused_pick_text(update):
    """The owner's ``session.updated`` for a deferred pick it dropped at the turn boundary."""
    current = update.get('model') or 'the current model'
    if update.get('provider'):
        current += f" ({update['provider']})"
    return (f"Model switch to {update.get('target_model') or 'the picked model'} was not applied "
            f"({update.get('reason') or 'refused'}); still using {current}.")


def _summary_text(result):
    summary = result.get('summary') or {}
    return '\n'.join(filter(None, (summary.get('headline'), summary.get('token_line'), summary.get('note'))))


def _model_text(label, fields):
    lines = [f"{label}: {fields.get('model') or 'unknown'}"]
    if fields.get('provider'):
        lines.append(f"Provider: {fields['provider']}")
    return '\n'.join(lines)


def _tools_text(info):
    """The session's tools by toolset (``session.info`` ``tools``: toolset -> tool names)."""
    groups = info.get('tools') or {}
    names = sorted({name for tools in groups.values() for name in tools})
    if not names:
        toolsets = info.get('enabled_toolsets') or []
        return f"Enabled toolsets: {', '.join(toolsets)}" if toolsets else "No tools available."
    lines = [f"Available tools ({len(names)}):"]
    lines.extend(f"  {toolset}: {', '.join(sorted(tools))}" for toolset, tools in sorted(groups.items()) if tools)
    return '\n'.join(lines)


# What ACP answers itself: command -> (the parity table's ACP cell, handler). Dispatch reads this and
# ``scripts/gen_gateway_command_parity.py`` documents it, so a route cannot change without the table.
# Any command not here goes to the owner's ``slash.exec`` (``_forward``). ``/clear`` is main's
# "new session + clear screen"; here it is ``/new``'s reset of this session.
ROUTES = {
    'branch': ('ACP fork_session', _branch),
    'discard': ('prompt.resolve_unknown', _discard_unknown),
    'help': ('local', _help),
    'steer': ('session.steer', _prompt),
    'queue': ('prompt.submit', _prompt),
    'title': ('session.mutate rename (bare: slash.exec)', _title),
    'undo': ('session.mutate rewind', _rewind_command),
    'retry': ('session.mutate rewind + resubmit', _rewind_command),
    'tools': ('session.info', _info),
    'model': ('session.mutate model', _info),
    'compress': ('session.mutate compress', _mutation),
    'new': ('session.mutate reset', _mutation),
    'clear': ('session.mutate reset', _mutation),
}
