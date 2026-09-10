"""Concrete worker compression APIs; scope and receipts stay in RuntimeSessionStore."""


class RuntimeSessionCompressionMixin:
    def _append_compression_messages(self, session_id, messages, compression_lock_holder,
                                     turn_lease_holder, turn_lease_ttl_seconds):
        self._session(session_id)
        result = self._apply('compression.append', dict(messages=messages,
            compression_lock_holder=compression_lock_holder, turn_lease_holder=turn_lease_holder,
            turn_lease_ttl_seconds=turn_lease_ttl_seconds))
        for message, annotation in zip(messages, result['annotations'], strict=True):
            message.update(annotation)
        return result['count']

    def get_session(self, session_id):
        if session_id != self.scope['session_id']:
            from agent.runtime_session_store import WorkerPersistenceError
            ids = self._apply('compression.lineage', {'target': self.scope['session_id']})['readable_ids']
            if session_id not in ids:
                raise WorkerPersistenceError('permission_denied')
        return self._apply('compression.context', {'target': session_id})['session']

    def get_compression_lineage(self, session_id):
        return self._apply('compression.lineage', {'target': session_id})['lineage']

    def get_conversation_root(self, session_id):
        return self._apply('compression.lineage', {'target': session_id})['root']

    def get_compression_tip(self, session_id):
        return self._apply('compression.lineage', {'target': session_id})['tip']

    def resolve_resume_session_id(self, session_id):
        return self._apply('compression.lineage', {'target': session_id})['resume']

    def declared_scope_identity(self, session_id):
        return tuple(self._apply('compression.lineage', {'target': session_id})['identity'])

    def is_explicit_fork_child(self, session_id):
        return self.declared_scope_identity(session_id)[0]

    def latest_conversation_boundary(self, session_key, source):
        return self._apply('compression.boundary', {'session_key': session_key, 'source': source})['value']

    def get_messages_as_conversation(self, session_id, include_ancestors=False, include_inactive=False,
                                     repair_alternation=False, include_row_ids=False, include_compacted=False):
        return self._apply('compression.history', dict(target=session_id, include_ancestors=include_ancestors,
            include_inactive=include_inactive, repair_alternation=repair_alternation,
            include_row_ids=include_row_ids, include_compacted=include_compacted))['messages']

    def get_active_message_watermark(self, session_id):
        self._session(session_id)
        return self._apply('compression.watermark', {})['value']

    def archive_and_compact(self, session_id, compacted_messages, model_config_patch=None,
                            watermark=None, lock_holder=None, tail_count=0):
        self._session(session_id)
        result = self._apply('compression.archive', dict(messages=compacted_messages,
            model_config_patch=model_config_patch, watermark=watermark, lock_holder=lock_holder,
            tail_count=tail_count))
        for message, row_id in zip(compacted_messages, result['row_ids'], strict=True):
            message['_row_id'] = row_id
        return result['value']

    def publish_compression_child(self, *, parent_session_id, child_session_id, source, messages,
            model=None, model_config=None, system_prompt=None, cwd=None, profile_name=None,
            compression_lock_holder=None, require_compression_lease=True, require_lease_refresh=False,
            lease_ttl_seconds=300.0, watermark=None, watermark_ceiling=None):
        self._session(parent_session_id)
        result = self._apply('compression.publish', dict(child_session_id=child_session_id, source=source,
            messages=messages, model=model, model_config=model_config, system_prompt=system_prompt,
            cwd=cwd, profile_name=profile_name, compression_lock_holder=compression_lock_holder,
            require_compression_lease=require_compression_lease, require_lease_refresh=require_lease_refresh,
            lease_ttl_seconds=lease_ttl_seconds, watermark=watermark, watermark_ceiling=watermark_ceiling))
        for message, row_id in zip(messages, result['row_ids'], strict=True):
            message['_row_id'] = row_id

    def _compression_receipt_journal(self, candidate, result):
        assignment = result.get('worker_assignment')
        if assignment:
            from agent.runtime_session_store import WorkerPersistenceError
            if assignment['parent'] != self.scope['session_id']:
                raise WorkerPersistenceError('outbox_scope_mismatch')
            candidate = dict(candidate, scope=dict(self.scope, session_id=assignment['session_id']))
        return candidate

    def try_acquire_compression_lock(self, session_id, holder, ttl_seconds=300.0):
        self._session(session_id)
        return self._apply('compression.lock.acquire', {'holder': holder, 'ttl_seconds': ttl_seconds})['value']

    def refresh_compression_lock(self, session_id, holder, ttl_seconds=300.0):
        self._session(session_id)
        return self._apply('compression.lock.renew', {'holder': holder, 'ttl_seconds': ttl_seconds})['value']

    def reopen_orphaned_compression_session(self, session_id):
        self._session(session_id)
        return self._apply('compression.reopen', {})['value']

    def release_compression_lock(self, session_id, holder):
        self._apply('compression.cleanup', {'target': session_id, 'holder': holder})

    def get_compression_lock_holder(self, session_id):
        self._session(session_id)
        return self._apply('compression.lock.holder', {})['value']

    def record_compression_failure_cooldown(self, session_id, cooldown_until, error=None):
        self._session(session_id)
        self._apply('compression.cooldown.record', {'cooldown_until': cooldown_until, 'error': error})

    def restore_compression_failure_cooldown_row(self, session_id, snapshot):
        self._session(session_id)
        self._apply('compression.cooldown.restore', {'snapshot': snapshot})

    def clear_compression_failure_cooldown(self, session_id):
        self._session(session_id)
        self._apply('compression.cooldown.clear', {})

    def set_compression_fallback_streak(self, session_id, streak):
        self._session(session_id)
        self._apply('compression.fallback', {'value': max(0, int(streak))})

    def set_compression_ineffective_count(self, session_id, count):
        self._session(session_id)
        self._apply('compression.ineffective', {'value': max(0, int(count))})

    def set_compression_recovery_deadline(self, session_id, deadline):
        self._session(session_id)
        try:
            value = max(0.0, float(deadline or 0.0))
        except (TypeError, ValueError):
            value = 0.0
        self._apply('compression.recovery', {'value': value or None})
