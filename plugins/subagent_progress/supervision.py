"""Independent supervision and parent review; reports and heartbeats never renew."""
import json
import time
import logging
from .transactions import lease_transaction

REVIEW_GUIDANCE = (
    "For review_required checkpoints, independently inspect evidence against the ORIGINAL goal, then use "
    "review_subagent_progress(checkpoint_id, decision='approve'|'steer'|'stop', reason, evidence_checked). "
    "Load it with tool_describe if deferred. Only approve verified useful progress; approval grants a fresh "
    "configured timeout from NOW. Steering, self-reports, activity and silence never renew. "
    "If a supervision alert has no checkpoint, inspect the child and use delegate_task steer/stop; "
    "do not fabricate a report or approve an alert. Treat missing evidence as uncertainty, not proof of failure. "
)


def owned_child(subagent_id, parent):
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, _resolve_session_lineage
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
        owner_sid = str(record.get('owner_agent_session_id') or '') if record else ''
        parent_sid = str(getattr(parent, 'session_id', '') or '')
        # Review authority is the direct owning conversation, not the ancestor tree.
        # Durable ownership survives agent rebuilds and verified compression rotations.
        if (not owner_sid or not parent_sid or
                _resolve_session_lineage(owner_sid, parent) != _resolve_session_lineage(parent_sid, parent)):
            raise ValueError("No live child owned by this parent conversation.")
        child = record['agent']
        if child is parent:
            raise ValueError('A child cannot review itself.')
    if getattr(parent, '_interrupt_requested', False) or getattr(child, '_interrupt_requested', False):
        raise ValueError("A stopped parent or child cannot renew its deadline.")
    return child


class Supervision:
    def __init__(self, plugin):
        self.plugin = plugin
        self.alerted = {}
        self.handle = None
        with plugin.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS reviews (checkpoint INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS supervision_checks (subagent TEXT PRIMARY KEY, checked REAL, payload TEXT)')

    def start(self):
        from agent.periodic_scheduler import schedule
        with self.plugin.lock:
            if self.handle is None and not self.plugin.closed:
                self.handle = schedule(self.tick, 30)

    def close(self):
        with self.plugin.lock:
            self.plugin.closed = True
            if self.handle is not None:
                self.handle.cancel()
            for future in list(self.plugin.delivery_futures):
                future.cancel()
            self.plugin.pending_delivery.clear()

    def review(self, args, **_):
        try:
            allowed = {'checkpoint_id', 'decision', 'reason', 'evidence_checked'}
            if not isinstance(args, dict) or set(args) - allowed:
                raise ValueError('Unknown review fields.')
            checkpoint, decision = args.get('checkpoint_id'), args.get('decision')
            reason, evidence = args.get('reason'), args.get('evidence_checked', [])
            if type(checkpoint) is not int or decision not in {'approve', 'steer', 'stop'}:
                raise ValueError('A checkpoint_id and approve/steer/stop decision are required.')
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
                raise ValueError('Explain the evidence-based decision in 1..1000 characters.')
            if not isinstance(evidence, list) or len(evidence) > 6 or any(
                    not isinstance(p, str) or not p.strip() or len(p) > 500 for p in evidence):
                raise ValueError('evidence_checked must be at most six bounded evidence references.')
            if decision == 'approve' and not evidence:
                raise ValueError('Approval requires independently checked evidence references.')
            plugin = self.plugin
            parent = plugin.current_child()
            with plugin.lock:
                with plugin.db() as db:
                    row = db.execute('SELECT * FROM reports WHERE id=?', (checkpoint,)).fetchone()
                if not row or json.loads(row['payload']).get('kind') != 'milestone':
                    raise ValueError('Only a milestone checkpoint may be reviewed.')
                child = owned_child(row['subagent'], parent)
                lease = getattr(child, '_delegate_reviewed_deadline', None)
                if lease is None:
                    raise ValueError('This child has no enabled parent-reviewed deadline.')
                with lease_transaction(lease), plugin.db() as db:
                    if plugin.closed:
                        raise ValueError('Plugin has been unloaded.')
                    if db.execute('SELECT 1 FROM reviews WHERE checkpoint=?', (checkpoint,)).fetchone():
                        raise ValueError('This checkpoint has already been reviewed; it cannot renew twice.')
                    result = lease.review(checkpoint, approve=decision == 'approve')
                    result.update(success=True, decision=decision, reason=reason, evidence_checked=evidence,
                                  parent_session_id=str(getattr(parent, 'session_id', '')), created=time.time())
                    db.execute('INSERT INTO reviews VALUES (?, ?)', (checkpoint, json.dumps(result, ensure_ascii=False)))
                # External control is irreversible; only request it after durable commit.
                if decision != 'approve':
                    from tools.delegate_tool_registry import steer_subagent, interrupt_subagent
                    result['control_requested'] = (steer_subagent(row['subagent'], reason) if decision == 'steer'
                                                   else interrupt_subagent(row['subagent']))
            return json.dumps(result, ensure_ascii=False)
        except (ValueError, TypeError) as exc:
            return json.dumps({'success': False, 'error': str(exc)})

    def tick(self):
        """Isolate the complete observation/persistence/delivery path per child."""
        plugin = self.plugin
        if plugin.closed:
            return False
        plugin.retry_notifications()
        with plugin.db() as db:
            owners = [dict(row) for row in db.execute("SELECT * FROM children WHERE status='running'")]
        live_ids = {o['subagent'] for o in owners}
        self.alerted = {k: v for k, v in self.alerted.items() if k in live_ids}
        for owner in owners:
            try:
                self.check_child(owner)
            except Exception:
                logging.getLogger("subagent_progress").exception("Supervision check failed: %s", owner['subagent'])

    def check_child(self, owner):
        from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock
        plugin, sid = self.plugin, owner['subagent']
        if plugin.closed:
            return
        with _active_subagents_lock:
            record = _active_subagents.get(sid)
            child = record.get('agent') if record else None
        lease = getattr(child, '_delegate_reviewed_deadline', None)
        if lease is None:
            return
        state = lease.snapshot()
        activity = child.get_activity_summary()
        observation = {**state, 'current_tool': activity.get('current_tool'),
                       'api_call_count': activity.get('api_call_count'),
                       'last_activity_ts': activity.get('last_activity_ts')}
        with plugin.lock:
            if plugin.closed:
                return
            with plugin.db() as db:
                db.execute('INSERT OR REPLACE INTO supervision_checks VALUES (?, ?, ?)',
                           (sid, time.time(), json.dumps(observation)))
            if state['closed'] or state['remaining_seconds'] > state['timeout_seconds'] / 2:
                return
            generation = state['renewals']
            if self.alerted.get(sid) == generation:
                return
            with plugin.db() as db:
                last = db.execute('SELECT id,payload FROM reports WHERE subagent=? ORDER BY id DESC LIMIT 1', (sid,)).fetchone()
                payload = {'subagent_id': sid, 'goal': owner['goal'], 'status': 'running', 'kind': 'supervision_alert',
                           'completed': 'Runtime supervision check; progress is NOT inferred from activity.',
                           'evidence': json.loads(last['payload']).get('evidence', []) if last else [],
                           'next_step': 'Parent inspect progress and evidence; only a real milestone may renew.',
                           'blocker': 'At least half of the approved time window elapsed; independent review requested.',
                           'needs_decision': True, 'observation': observation, 'window_generation': generation}
                alert_id = db.execute('INSERT INTO reports(subagent,parent,payload,created) VALUES (?,?,?,?)',
                                      (sid, owner['parent'], json.dumps(payload), time.time())).lastrowid
            self.alerted[sid] = generation
            payload['checkpoint_id'] = alert_id
            parent_ref = getattr(child, '_delegate_parent_ref', None)
            parent = parent_ref() if callable(parent_ref) else None
            if not plugin.closed:
                plugin.notify(parent, owner, payload, alert_id)


def register_review(ctx, supervisor):
    schema = {'name': 'review_subagent_progress', 'description': (
        'Owning parent only: review a child milestone against its original goal and inspected evidence. '
        'approve renews the configured timeout from now; steer or stop do NOT renew. '
        'Silence, duplicate/stale approvals and child self-reports cannot extend a deadline.'),
        'parameters': {'type': 'object', 'additionalProperties': False, 'properties': {
            'checkpoint_id': {'type': 'integer'}, 'decision': {'type': 'string', 'enum': ['approve', 'steer', 'stop']},
            'reason': {'type': 'string', 'maxLength': 1000},
            'evidence_checked': {'type': 'array', 'maxItems': 6, 'items': {'type': 'string', 'maxLength': 500}}},
            'required': ['checkpoint_id', 'decision', 'reason']}}
    ctx.register_tool(name='review_subagent_progress', toolset='subagent_progress', schema=schema,
                      handler=supervisor.review, check_fn=lambda: True, emoji='🔎')
