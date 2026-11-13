"""Rediscover persisted parked goals on the existing heartbeat ticker."""
import logging

logger = logging.getLogger('gateway.run')


def candidates(authority):
    from hermes_cli.goals import GoalState
    from hermes_cli.goals_waits import parked
    from gateway.session_goal_decisions import outstanding
    from gateway.session_local_recovery import owned_receipt
    from hermes_state_local import POLICY_PREFIX
    with authority.db._read_ctx() as conn:
        rows = conn.execute("SELECT key,value FROM state_meta WHERE key LIKE 'goal:%'").fetchall()
        if not rows:
            return []
        locals_ = conn.execute('SELECT key FROM state_meta WHERE key LIKE ?', (POLICY_PREFIX + '%',)).fetchall()
        parsed = {}
        decisions = {}
        for key, raw in rows:
            try:
                state = GoalState.from_json(raw)
            except (ValueError, TypeError, AttributeError, KeyError, OverflowError, RecursionError):
                logger.debug('Skipping unreadable persisted goal %s', key, exc_info=True)
                continue
            parsed[key] = (raw, state)
            decisions[key] = outstanding(authority, conn, key[len('goal:'):], state)
    # A local creation receipt is the authority, not a viewer or in-memory watch.
    local_entries = {}
    for row in locals_:
        owner = row[0][len(POLICY_PREFIX):]
        try:
            _receipt, _source, _route, entry = owned_receipt(authority, owner)
        except (ValueError, KeyError, TypeError):
            continue
        local_entries[entry.session_id] = (entry, _source)
    entries = {e.session_id: e for e in authority.runner.session_store.list_sessions()}
    entries.update({sid: pair[0] for sid, pair in local_entries.items()})
    result = []
    for key, (raw, state) in parsed.items():
        state._persisted_raw = raw
        entry = entries.get(key[len('goal:'):])
        if (entry is not None and entry.origin is not None and not entry.suspended
                and (parked(state) or decisions[key] is not None)):
            source = local_entries[entry.session_id][1] if entry.session_id in local_entries else authority.runner._restored_source(entry)
            if source.platform.value == 'local':
                live = authority.sessions.get(source.chat_id)
                if live is not None:
                    source = live.source
            result.append((entry, source, state))
    return result


def has_goal_duty(authority):
    """Eligible persisted parked duty, including future timers, holds unmanaged idle exit."""
    from gateway.session_completion_notes import _closed_by_user
    from gateway.session_local_recovery import owned_receipt
    from gateway.session_policy import restore_policy
    from gateway.session_viewer_close import CLOSE_PREFIX, ORPHAN_REASONS, _PendingClose
    from gateway.session_viewer_automation import viewer_driven
    for entry, source, _state in candidates(authority):
        owner = source.chat_id if source.platform.value == 'local' else authority.logical_owner(entry.session_id)
        if viewer_driven(authority, owner) or _closed_by_user(authority, owner):
            continue
        close = authority.db.get_meta(CLOSE_PREFIX + owner)
        if close and _PendingClose.from_record(close).reason not in ORPHAN_REASONS:
            continue
        current = authority.runner.session_store.lookup_by_session_key(entry.session_key)
        if current is not None and (current.suspended or current.session_id != entry.session_id):
            continue
        if source.platform.value == 'local':
            receipt, *_rest = owned_receipt(authority, owner)
            if restore_policy(receipt['policy']).source not in {'cli', 'tui', 'gui'}:
                continue
        elif authority.runner._delivery_adapter_for(source) is None:
            continue
        return True
    return False


def silent_completion_text(session_id, event):
    """Actual completion evidence when its successful parent turn intentionally returned no text."""
    from gateway.session_results import execution_result
    from hermes_cli.goals import load_goal
    from hermes_cli.goals_workers import completed, completion_ids
    captured = execution_result.get() or {}
    result = captured.get('terminal_result', captured.get('result'))
    if (event is None or not event.internal or event.metadata.get('display_kind') != 'async_delegation_complete'
            or not isinstance(result, dict) or result.get('failed') or result.get('interrupted')):
        return ''
    state = load_goal(session_id)
    if state is None or state.status != 'active' or not state.waiting_workers:
        return ''
    if completed(session_id, state.waiting_workers) & completion_ids(session_id):
        return 'Delegation result received; the parent returned no final text:\n' + event.text
    return ''


async def poll_one(authority, entry, source, state):
    from gateway.session_goal_waits import admit_wait
    from gateway.session_goal_decisions import outstanding
    authority._require_admission_open()  # a scan may have awaited across profile retirement
    if source.platform.value == 'local':
        from gateway.session_local_recovery import restore_local_session
        restore_local_session(authority, source.chat_id)
        source = authority.sessions[source.chat_id].source
    with authority.db._read_ctx() as conn:
        duty = outstanding(authority, conn, entry.session_id, state)
    await admit_wait(authority, entry, source, state, after=duty['after'] if duty else None,
                     prompt=duty['prompt'] if duty else None)


async def poll_goal_waits(runner):
    from gateway.session_authorities import async_owner_scope, serving_authorities
    from hermes_cli import goals
    from hermes_cli.goals_waits import ready
    from hermes_cli.goals_waits import parked
    for authority in serving_authorities(runner):
        try:
            async with async_owner_scope(authority):
                found = await runner._run_in_executor_with_context(candidates, authority)
                for entry, source, state in found:
                    if not parked(state) or ready(state, goals.time.time()):
                        try:
                            await poll_one(authority, entry, source, state)
                        except Exception:
                            logger.debug('Parked goal admission deferred for %s', entry.session_id, exc_info=True)
        except Exception:
            logger.debug('Parked goal scan deferred', exc_info=True)
