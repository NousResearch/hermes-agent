"""Editor slash commands on the gateway path; each one is an existing gateway verb.

Main served these in-process (``acp_adapter/commands.py``). The reports are the owner's own
``slash.exec`` reads, ``/model``, ``/compress`` and ``/reset`` the retained ``session.mutate``
controls, ``/steer`` the owner's ``session.steer``; ``/queue`` and an idle ``/steer`` become the
next prompt, which the session's FIFO runs after any turn in flight.
"""
from __future__ import annotations

from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError

# Reports the owner answers for a local session (``gateway/session_commands.py::_READ_COMMANDS``).
_READS = frozenset({'context', 'usage', 'status', 'version'})
_MUTATIONS = frozenset({'model', 'compress', 'new', 'branch'})
_PROMPTS = frozenset({'steer', 'queue'})
COMMANDS = _READS | _MUTATIONS | _PROMPTS | {'help', 'tools'}


async def run_command(agent, session_id, command, arg):
    """``(reply, None)`` for a command answered here, or ``(None, text)`` when ``text`` is to be
    submitted as the prompt. Unsupported commands are refused before any gateway call."""
    from acp_adapter.gateway_session_meta import help_text
    if command not in COMMANDS:
        raise GatewayClientError('unsupported_command')
    if command == 'branch':
        raise GatewayClientError('use_acp_fork_session')
    if command == 'help':
        return help_text(), None
    if command in _PROMPTS:
        if not arg:
            return f"Usage: /{command} <{'guidance' if command == 'steer' else 'prompt'}>", None
        steered = await _steer(agent, session_id, arg) if command == 'steer' else None
        return (steered, None) if steered else (None, arg)
    client = await agent._client()
    if command in _READS:
        result = await client.rpc('slash.exec', session_id=session_id, command=f'/{command} {arg}'.strip())
        return str(result.get('output') or result.get('notice') or ''), None
    if command == 'tools' or (command == 'model' and not arg):
        info = await client.rpc('session.info', session_id=session_id)
        return (_tools_text(info) if command == 'tools' else _model_text('Current model', info)), None
    return await _mutate(agent, client, session_id, command, arg), None


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
    operation, payload = ('reset', {}) if command == 'new' else slash_mutation('/' + command, arg)
    confirm = agent._model_confirmer(session_id) if operation == 'model' else None
    result = await agent._mutations.apply(client, session_id, operation, payload, confirm=confirm)
    status = result.get('status')
    if status == 'cancelled':
        from agent.i18n import t
        reply = t('cli.model.switch_cancelled')
    elif status == 'preview':
        # Read-only report: nothing changed, so the editor's snapshot is still current.
        reply = '\n'.join(result['lines'])
    elif status == 'noop':
        # Nothing removable (an empty session too): the owner's shared report, nothing committed.
        reply = _summary_text(result) or 'No changes from compression.'
    else:
        agent._snapshots[session_id] = await client.rpc('session.resume', session_id=session_id)
        if operation in {'reset', 'compress'}:
            await _refresh_context(agent, client, session_id, operation)
        reply = {'model': lambda: _model_text('Model switched to', result),
                 'compress': lambda: _summary_text(result) or 'Context compressed.',
                 'reset': lambda: 'Conversation history cleared.'}[operation]()
    agent._mutations.acknowledge(session_id, operation, payload)
    return reply


async def _refresh_context(agent, client, session_id, operation):
    """The editor's context indicator after /reset or /compress. Both are session boundaries that
    evict the owner's agent, so ``session.stats`` has no window until the next turn builds one: the
    window shown stays (the model is unchanged) and its occupancy is the new transcript's, nothing
    after a reset, the compressed messages' estimate (the summary's own figure) after a compress."""
    stats = await client.rpc('session.stats', session_id=session_id)
    if stats.get('context_length') is None:
        from agent.model_metadata import estimate_request_tokens_rough
        messages = agent._snapshots[session_id].get('messages') or []
        stats = {**stats, 'context_length': (agent._usage.get(session_id) or {}).get('context_length'),
                 'context_tokens': estimate_request_tokens_rough(messages) if operation == 'compress' else 0}
    await agent._project_usage(session_id, agent._editor_id(session_id), stats)


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
