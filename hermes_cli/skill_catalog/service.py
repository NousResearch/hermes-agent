"""Profile-owned catalog orchestration; hub bundles never enter this service."""
import time
from pathlib import Path

from hermes_constants import hermes_home_key
from hermes_cli.skill_catalog.registry import snapshot
from hermes_cli.skill_catalog.validation import CatalogError, _reference, validate_request
from hermes_cli.skill_catalog.worker import call, deadline_for


class Service:
    def __init__(self, home):
        self.home = Path(home)

    def entries(self):
        return snapshot(scope=hermes_home_key(self.home))

    def resolve(self, identifier):
        try:
            _reference(identifier)
        except (ValueError, TypeError):
            raise CatalogError('invalid_identifier', 'Invalid catalog identifier.') from None
        source, item = identifier.split(':')
        entry = next((e for e in self.entries() if e.source_id == source), None)
        if entry is None:
            raise CatalogError('not_found', 'Catalog source is not registered.')
        if not entry.authorized():
            raise CatalogError('consent_required', 'Catalog consent is required.')
        return entry, item

    def sources(self, *, deadline=None):
        deadline = deadline_for('availability', deadline)
        rows = []
        for entry in self.entries():
            try:
                state = call(entry, 'availability', {}, deadline=deadline)
            except CatalogError as exc:
                state = dict(state='unavailable', message=exc.message)
            if state['state'] == 'hidden':
                continue
            rows.append(dict(id=entry.source_id, label=entry.descriptor.label, searchable=True,
                provider=dict(id=entry.descriptor.name, plugin_id=entry.plugin_key), **state))
        return rows

    def read(self, method, identifier, payload=None, *, deadline=None):
        deadline = deadline_for(method, deadline)
        entry, item = self.resolve(identifier)
        request = dict(payload or {}, id=item)
        return call(entry, method, request, deadline=deadline)

    def review(self, identifier, action, version=None, *, deadline=None):
        deadline = deadline_for('review', deadline)
        result = self.read('review', identifier, dict(action=action, version=version), deadline=deadline)
        entry, _ = self.resolve(identifier)
        entry.mutations.reconcile(target=result['target_relative_path'], deadline=deadline)
        if result['identifier'] != identifier or result['action'] != action:
            raise CatalogError('invalid_response', 'Catalog review identity changed.')
        return result

    def confirm(self, identifier, action, review_token, acknowledgements, *, deadline=None):
        deadline = deadline_for('confirm', deadline)
        entry, item = self.resolve(identifier)
        payload = validate_request('confirm', dict(action=action, id=item, review_token=review_token,
                                                   acknowledgements=acknowledgements, admission=None))
        authority = call(entry, 'reviewed', dict(review_token=review_token), deadline=deadline)
        plan = authority['review']
        if plan['identifier'] != identifier or plan['action'] != action:
            raise CatalogError('review_changed', 'Confirmation does not match its review.')
        if authority['kind'] == 'completed':
            # Completed replay never reserves, rechecks files or republishes ownership.
            return authority['operation']
        if set(acknowledgements) != set(plan['acknowledgements_required']):
            raise CatalogError('acknowledgement_required', 'Acknowledge the exact review findings.')
        if authority['kind'] == 'unstarted' and (
                not plan['confirmable'] or plan['expires_at_ms'] <= int(time.time() * 1000)):
            raise CatalogError('review_expired', 'Review again before confirming.')
        entry.mutations.reconcile(target=plan['target_relative_path'], deadline=deadline)
        if time.monotonic() >= deadline:
            raise CatalogError('timeout', 'Catalog request timed out.')
        operation_id = authority['operation']['operation_id'] if authority['kind'] == 'pending' else None
        admission = entry.mutations.projection.reserve(entry, plan, operation_id=operation_id,
            recovery=authority['kind'] == 'pending', deadline=deadline)
        payload['admission'] = admission
        result = call(entry, 'confirm', payload, deadline=deadline)
        if result['identifier'] != identifier or result['action'] != action:
            raise CatalogError('invalid_response', 'Catalog operation identity changed.')
        return result

    def operation(self, identifier, operation_id, *, deadline=None):
        deadline = deadline_for('operation', deadline)
        entry, _ = self.resolve(identifier)
        result = call(entry, 'operation', dict(operation_id=operation_id), deadline=deadline)
        if result['identifier'] != identifier or result['operation_id'] != operation_id:
            raise CatalogError('invalid_response', 'Catalog operation identity changed.')
        if result['state'] in ('completed', 'failed'):
            try:
                entry.mutations.reconcile(operation_id=operation_id, deadline=deadline)
            except CatalogError:
                # A read still reports the terminal provider journal and finalizing state.
                pass
        return dict(result, settlement=entry.mutations.settlement(operation_id))
