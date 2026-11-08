"""Closeout on the authenticated owner; the normal primary admission does all remote work."""
import asyncio
from pathlib import Path

from hermes_cli.conversation_closeout_jobs import PRESERVATION_WAIT_SECONDS
from hermes_state_runtime import RuntimeStoreError


def canonical_settled(authority, ref):
    from hermes_cli.conversation_closeout_jobs import _segments
    from tools.async_delegation import has_live_for_session
    from tools.process_registry import process_registry
    from tools.process_registry_awaiting import awaits_notification
    live = authority.sessions.get(ref.session_id)
    if live is None:
        return False
    if live.task is not None and not live.task.done():
        return False
    try:
        with authority.db._read_ctx() as conn:
            ids = _segments(conn, authority.db, ref.session_id)
            for sid in ids:
                if conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status!='terminal' "
                    "UNION ALL SELECT 1 FROM worker_executions WHERE session_id=? AND status!='terminal' LIMIT 1",
                    (sid, sid)).fetchone():
                    return False
        if any(has_live_for_session(session_key=live.route, parent_session_id=sid) for sid in ids):
            return False
        agent = authority.agent(ref)
        owners = set(ids) | set(getattr(agent, '_process_owner_task_ids', ()))
        return not awaits_notification(process_registry, owners)
    except Exception:  # health: allow BLE001 -- settlement probes include plugins; fail closed without logging private exception payloads
        return False


async def closeout(connection, ref, params):
    from gateway.session_mutations import _owner_ref
    from hermes_cli.conversation_closeout_jobs import begin_closeout, old_identity
    from hermes_constants import profile_name_for_home
    from hermes_cli.profiles import profile_matches_home
    authority = connection.authority
    if params.get('profile') and not profile_matches_home(params['profile'], Path(authority.profile_id)):
        raise RuntimeStoreError('profile_mismatch')
    owner = _owner_ref(authority, ref, {'operation': 'archive'})
    authority.authorize(connection.actor, owner, 'session:control')
    authority.authorize(connection.actor, owner, 'session:submit')
    if owner.session_id not in connection.subscriptions:
        raise RuntimeStoreError('permission_denied')
    authority._require_admission_open()
    live = authority.sessions[owner.session_id]

    async def submit(prompt, operation):
        return await connection.submit(owner, {'session_id': owner.session_id, 'text': prompt,
            'submission_id': 'closeout-review-' + operation, 'queued': True})

    def launch(coroutine):
        task = asyncio.create_task(coroutine)
        jobs = authority.__dict__.setdefault('_closeout_jobs', set())
        jobs.add(task)
        task.add_done_callback(jobs.discard)

    from gateway.session_viewer_close import _physical_target
    old = old_identity(authority.db, _physical_target(authority, owner.session_id),
        profile=profile_name_for_home(authority.db.db_path.parent) or 'default')
    return begin_closeout(authority.db, old, params['request_id'], params['intent'], submit=submit,
        settled=lambda: canonical_settled(authority, owner), launch=launch, epoch=authority.epoch,
        wait_seconds=PRESERVATION_WAIT_SECONDS,
        publish=lambda result: live.event_stream.publish(owner.session_id, result, event_type='session.closeout.updated'))
