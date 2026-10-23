"""Main's TUI text for the session reads its sidecar answered itself: /status, /context.

On main the TUI's ``slash.exec`` never reached the messaging gateway's handlers: ``/status`` was
``session.status`` (``Hermes TUI Status``), ``/context`` was ``_format_live_context_output`` (the
conversation counts, then the desktop gauge's category table). A TUI session on the shared owner
gets the same lines here, read from the owner's live agent and stored row; every other surface keeps
the gateway's own handlers (``gateway/session_commands.py``). ``/cron``, the classic CLI's handler
that main's TUI ran in its slash worker, is gateway/session_cron_commands.py.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections import Counter

# The TUI surfaces whose /status and /context read as main's TUI sidecar did.
_TUI_SOURCES = frozenset({'tui'})


def tui_read(policy, name: str, arg: str):
    """The renderer for a TUI session's read of *name*, or None to keep the gateway handler."""
    if policy is None or policy.source not in _TUI_SOURCES:
        return None
    if name == 'status':
        return _status
    if name == 'context':
        return _context
    return None


def _stored_id(authority, ref) -> str:
    with contextlib.suppress(Exception):
        from hermes_state_local import local_receipt
        return local_receipt(authority.db, ref.session_id)['entry']['session_id'] or ref.session_id
    return ref.session_id


def _status(authority, ref, live, policy, agent, arg) -> str:
    """``tui_gateway/methods_session.py`` ``session.status`` on main, for the owner's session."""
    from hermes_cli.status_report import build_status_fields, status_lines
    from tui_gateway import server
    stored = _stored_id(authority, ref)
    row = {}
    with contextlib.suppress(Exception):
        row = authority.db.get_session(stored) or {}
    running = authority._handle(ref).execution_state != 'idle'
    fields = build_status_fields(
        stored, agent, row, model=policy.current_model(authority) if agent is None else None,
        provider=policy.provider if agent is None else None,
        tokens=getattr(agent, 'session_total_tokens', 0) if agent is not None else None,
        agent_running=running, home=authority.profile_id)
    project = server._project_info_for_cwd(policy.cwd)
    return '\n'.join([
        'Hermes TUI Status', '', *status_lines(fields, 'session_id', 'path'),
        *([f"Project: {project['name']}"] if project else []),
        *status_lines(fields, 'title', 'model', 'created', 'last_activity', 'tokens', 'agent_running')])


def _context(authority, ref, live, policy, agent, arg) -> str:
    """Main's ``_format_live_context_output``: counts by role, model/provider, then the category
    table off the live agent (``render_context_breakdown_lines(grid=False)``), else the stored usage."""
    stored = _stored_id(authority, ref)
    rows = []
    with contextlib.suppress(Exception):
        rows = authority.db.get_messages_as_conversation(stored, include_ancestors=True, include_row_ids=True)
    from gateway.session_controls import display_projection
    messages = display_projection(rows, authority.profile_id) if rows else []
    breakdown: list[str] = []
    if agent is not None:
        with contextlib.suppress(Exception):
            from agent.context_breakdown import (
                compute_context_details, compute_session_context_breakdown, render_context_breakdown_lines)
            conversation = authority.db.get_messages_as_conversation(stored, include_ancestors=True)
            payload = compute_session_context_breakdown(agent, conversation)
            details = None
            if arg.strip().lower() in {'all', 'full', 'details'}:
                with contextlib.suppress(Exception):
                    details = compute_context_details(agent)
            breakdown = render_context_breakdown_lines(payload, details=details, grid=False)
    lines = [f'Conversation: {len(messages)} messages' if messages else 'Conversation is empty (no messages yet).']
    roles = Counter(str(msg.get('role') or 'unknown') for msg in messages)
    lines.append('  ' + ', '.join(f'{r}: {roles.get(r, 0)}' for r in ('user', 'assistant', 'tool', 'system')))
    if model := (getattr(agent, 'model', '') if agent is not None else '') or policy.current_model(authority):
        lines.append(f'Model: {model}')
    # Main read the provider off the compute-host metadata mirror, which a local session never
    # filled, so its TUI always printed ``auto`` here; kept as main's text.
    lines.append('Provider: auto')
    if breakdown:
        return '\n'.join([*lines, '', *breakdown])
    compressor = getattr(agent, 'context_compressor', None) if agent is not None else None
    used, maximum = (int(getattr(compressor, name, 0) or 0) for name in ('last_prompt_tokens', 'context_length'))
    if used and maximum:
        lines.append(f'Context usage: {used:,} / {maximum:,} tokens ({used / maximum * 100:.1f}%)')
    elif used:
        lines.append(f'Context usage: {used:,} tokens')
    if count := int(getattr(compressor, 'compression_count', 0) or 0):
        lines.append(f'Compressions: {count:,}')
    return '\n'.join(lines)


async def run_tui_read(authority, ref, live, policy, render, arg) -> dict:
    """Render off the owner loop, inside the owning profile and the session's workspace."""
    from gateway.session_authorities import owner_scope
    from gateway.session_commands import _session_workspace
    agent = authority.agent(ref)

    def build():
        with owner_scope(authority), _session_workspace(authority.runner, live.source):
            return render(authority, ref, live, policy, agent, arg)
    return {'type': 'exec', 'output': await asyncio.to_thread(build)}


async def context_breakdown(connection, ref, params) -> dict:
    """``session.context_breakdown`` on the owner: the category estimate (``compute_session_context_
    breakdown``) and context-file manifest of this session's resident agent over its stored
    transcript, built in the session's workspace. Before the agent exists: empty categories plus the
    session's stored usage (the contract's pre-build shape)."""
    from gateway.session_busy_controls import authorize
    from hermes_state_runtime import RuntimeStoreError
    if set(params) - {'session_id', 'profile'} or not isinstance(ref.session_id, str) or not ref.session_id:
        raise RuntimeStoreError('invalid_params')
    authorize(connection, ref, params, 'session:read')
    authority = connection.authority
    live = authority.sessions.get(ref.session_id)
    if live is None:
        raise RuntimeStoreError('not_found')
    agent = authority.agent(ref)
    if agent is None:
        from gateway.session_usage import stats
        usage = await stats(connection, ref, {'session_id': ref.session_id})
        usage = usage.get('usage') if isinstance(usage.get('usage'), dict) else usage
        return {'categories': [], 'context_max': int(usage.get('context_max') or 0),
                'context_percent': int(usage.get('context_percent') or 0),
                'context_used': int(usage.get('context_used') or 0), 'estimated_total': 0,
                'context_estimated': bool(usage.get('context_estimated', False)),
                'context_source': str(usage.get('context_source') or 'provider_usage'),
                'model': str(usage.get('model') or '')}
    from gateway.session_authorities import owner_scope
    from gateway.session_commands import _session_workspace
    stored = _stored_id(authority, ref)

    def build():
        from agent.context_breakdown import compute_session_context_breakdown
        from agent.context_file_sources import context_file_sources_for_agent
        with owner_scope(authority), _session_workspace(authority.runner, live.source):
            # A revision probe is two indexed reads; the rebuild walks the whole transcript and the
            # system prompt. Reuse the payload while neither the transcript nor the agent moved.
            key = (id(agent), _agent_context_marks(agent),
                   authority.db.conversation_revision(stored, include_ancestors=True))
            cached = live.context_breakdown
            if cached is not None and cached[0] == key:
                return dict(cached[1])
            conversation = authority.db.get_messages_as_conversation(stored, include_ancestors=True)
            payload = compute_session_context_breakdown(agent, conversation)
            payload['context_files'] = context_file_sources_for_agent(agent)
            live.context_breakdown = (key, payload)
            return dict(payload)
    return await asyncio.to_thread(build)


def _agent_context_marks(agent) -> tuple:
    """What the breakdown reads off the agent besides the transcript: model, toolset size, window,
    measured prompt tokens, compressions and the usage anchors."""
    comp = getattr(agent, 'context_compressor', None)
    return (getattr(agent, 'model', None), getattr(agent, 'provider', None), len(getattr(agent, 'tools', None) or ()),
            *(getattr(comp, name, None) for name in ('context_length', 'last_prompt_tokens', 'compression_count')),
            repr(getattr(agent, '_turn_base_usage_anchor', None)), repr(getattr(agent, '_usage_anchor', None)))
