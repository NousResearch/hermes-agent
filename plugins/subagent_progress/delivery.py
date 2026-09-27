"""Delivery receipts and bounded retries; only SDK injection can wake a parent."""
import json
import logging
import re
import sys

LOG = logging.getLogger('subagent_progress')
WAKE_PREFIX = '[SUBAGENT CHECKPOINT ID: '


class DeliveryMixin:
    def fresh(self, db, report_id, *, unconsumed=False):
        row = db.execute('SELECT r.*, c.status AS child_status FROM reports r JOIN children c '
                         'ON c.subagent=r.subagent WHERE r.id=?', (report_id,)).fetchone()
        if not row or self.closed or (unconsumed and row['consumed']):
            return False
        payload = json.loads(row['payload'])
        kind = payload.get('kind')
        if kind == 'terminal_checkpoint':
            latest = db.execute("SELECT MAX(id) FROM reports WHERE subagent=? AND "
                                "json_extract(payload,'$.kind')='terminal_checkpoint'", (row['subagent'],)).fetchone()[0]
            return row['child_status'] != 'running' and latest == report_id
        if row['child_status'] != 'running':
            return False
        from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock
        with _active_subagents_lock:
            record = _active_subagents.get(row['subagent'])
            child = record.get('agent') if record else None
        if child is None or getattr(child, '_interrupt_requested', False):
            return False
        lease = getattr(child, '_delegate_reviewed_deadline', None)
        state = lease.snapshot() if lease is not None else None
        if state and state['closed']:
            return False
        latest = db.execute("SELECT MAX(id) FROM reports WHERE subagent=? AND "
                            "json_extract(payload,'$.kind')='milestone'", (row['subagent'],)).fetchone()[0]
        if kind == 'milestone':
            return latest == report_id and not db.execute('SELECT 1 FROM reviews WHERE checkpoint=?', (report_id,)).fetchone()
        if kind == 'supervision_alert':
            reviewed = db.execute('SELECT 1 FROM reviews v JOIN reports r ON v.checkpoint=r.id '
                                  'WHERE r.subagent=? AND json_extract(v.payload,\'$.created\')>=?',
                                  (row['subagent'], row['created'])).fetchone()
            return bool(state and state['renewals'] == payload.get('window_generation')
                        and (latest is None or latest < report_id) and not reviewed)
        return False

    def dispatch(self, *, event, **_):
        metadata = getattr(event, 'metadata', None) or {}
        if (not getattr(event, 'internal', False) or not metadata.get('hermes_plugin_injection') or
                metadata.get('hermes_plugin_id') != 'subagent-progress'):
            return None
        # SDK prefixes tool-role content; other plugin content is not ours.
        match = re.fullmatch(r'(?:\[tool\] )?\[SUBAGENT CHECKPOINT ID: (\d+)\]', event.text)
        if not match:
            return None
        try:
            with self.lock, self.db() as db:
                if self.fresh(db, int(match[1]), unconsumed=True):
                    return None
        except Exception:
            LOG.exception('Checkpoint consumption validation failed; refusing wake')
        return {'action': 'skip', 'reason': 'Subagent checkpoint is stale, reviewed, consumed or closed'}

    def receipt(self, table, report_id, value):
        with self.lock, self.db() as db:
            db.execute(f'INSERT OR REPLACE INTO {table} VALUES (?, ?)', (report_id, json.dumps(value)))

    def request_wake(self, owner, payload, report_id):
        with self.lock:
            if self.closed or not owner['route'] or not (payload.get('needs_decision') or payload.get('review_required')):
                return False
            job = self.pending_delivery.setdefault(report_id, {'owner': owner, 'payload': payload,
                'display_done': False, 'display_attempts': 0, 'display_pending': False,
                'accepted': False, 'wake_pending': False, 'attempts': 0})
            if job['accepted'] or job['wake_pending'] or job['attempts'] >= 3:
                return False
            with self.db() as db:
                if not self.fresh(db, report_id, unconsumed=True):
                    return False
            job['attempts'] += 1
            job['wake_pending'] = True
            def delivered(accepted):
                with self.lock:
                    if self.closed:
                        return
                    job['wake_pending'] = False
                    job['accepted'] = job['accepted'] or bool(accepted)
                    self.receipt('wake_deliveries', report_id, {'accepted': job['accepted'],
                                 'attempts': job['attempts'], 'pending': False})
            try:
                scheduled = bool(self.ctx.inject_message(f'{WAKE_PREFIX}{report_id}]', role='tool',
                    session_key=owner['route'], expected_session_id=owner['parent'], on_delivery=delivered))
            except TypeError:
                job['attempts'] = 3
                LOG.exception('Upgrade Hermes core: inject_message requires expected_session_id and on_delivery')
                scheduled = False
            except Exception:
                LOG.exception('Checkpoint wake scheduling failed: %s', report_id)
                scheduled = False
            if not scheduled and not job['accepted']:
                delivered(False)
            self.receipt('wake_deliveries', report_id, {'accepted': job['accepted'], 'attempts': job['attempts'],
                                                      'pending': job['wake_pending'], 'scheduled': scheduled})
            return scheduled

    def notify(self, parent, owner, payload, report_id):
        with self.lock:
            if self.closed:
                return False, False
            job = self.pending_delivery.setdefault(report_id, {'owner': owner, 'payload': payload,
                'display_done': False, 'display_attempts': 0, 'display_pending': False,
                'accepted': False, 'wake_pending': False, 'attempts': 0})
            wake = self.request_wake(owner, payload, report_id)
            if job['display_done'] or job['display_pending'] or job['display_attempts'] >= 3:
                return False, wake
            job['display_attempts'] += 1
            gateway = sys.modules.get('gateway.run')
            runner_ref = getattr(gateway, '_gateway_runner_ref', None)
            runner = runner_ref() if callable(runner_ref) else None
            if runner is not None and owner['route']:
                from agent.async_utils import safe_schedule_threadsafe
                loop = getattr(runner, '_gateway_loop', None)
                if loop is None or not loop.is_running():
                    return False, wake
                job['display_pending'] = True
                future = safe_schedule_threadsafe(self.deliver_notice(runner, owner, payload, report_id), loop,
                    logger=LOG, log_message='Subagent notification scheduling failed')
                if future is None:
                    job['display_pending'] = False
                    return False, wake
                self.delivery_futures.add(future)
                def done(completed):
                    with self.lock:
                        self.delivery_futures.discard(completed)
                        job['display_pending'] = False
                        if not completed.cancelled():
                            try:
                                job['display_done'] = bool(completed.result())
                            except Exception:
                                LOG.exception('Checkpoint display failed: %s', report_id)
                future.add_done_callback(done)
                return True, wake
            if parent is not None:
                parent._emit_warning(self.notice(payload))
                job['display_done'] = True
                self.receipt('deliveries', report_id, {'success': True, 'transport': 'cli'})
            return job['display_done'], wake

    async def deliver_notice(self, runner, owner, payload, report_id):
        entry = await runner.async_session_store.lookup_by_session_key(owner['route'])
        if self.closed or entry is None or entry.origin is None:
            return False
        if entry.session_id != owner['parent']:
            session_db = getattr(runner, '_session_db', None)
            resolve = getattr(runner, '_resolve_compression_lineage_target', None)
            tip = await resolve(session_db, entry, owner['parent']) if session_db and resolve else None
            if tip != entry.session_id:
                LOG.warning('checkpoint=%s display refused: parent session changed', report_id)
                return False
        if self.closed:
            return False
        source = runner._restored_source(entry)
        if not runner._is_user_authorized_for_source(source, allow_adapter_delegation=False):
            return False
        adapter = runner._delivery_adapter_for(source)
        if adapter is None:
            return False
        from gateway.run import _redact_gateway_user_facing_secrets
        result = await adapter.send(source.chat_id, _redact_gateway_user_facing_secrets(self.notice(payload)),
                                    metadata=runner._thread_metadata_for_source(source))
        receipt = {'success': bool(getattr(result, 'success', False)), 'message_id': getattr(result, 'message_id', None)}
        self.receipt('deliveries', report_id, receipt)
        return receipt['success']

    def retry_notifications(self):
        with self.lock:
            if self.closed:
                return
            for report_id, job in list(self.pending_delivery.items()):
                try:
                    with self.db() as db:
                        fresh = self.fresh(db, report_id, unconsumed=True)
                    if not fresh:
                        self.pending_delivery.pop(report_id, None)
                        continue
                    self.notify(None, job['owner'], job['payload'], report_id)
                except Exception:
                    LOG.exception('Checkpoint retry failed: %s', report_id)
