"""Owning-context mutation facade, including legacy native receipt admission."""
import hashlib
import time
import threading
from pathlib import Path

from hermes_constants import get_hermes_home, hermes_home_key
from hermes_cli.skill_catalog import registry
from hermes_cli.skill_catalog.projection import Projection
from hermes_cli.skill_catalog.validation import CatalogError, validate_result, _payload


class ContextMutations:
    def __init__(self, ctx):
        self.ctx = ctx
        self.home = Path(getattr(ctx._manager, 'scope_key', None) or get_hermes_home())
        self._permits = {}
        self._permit_lock = threading.Lock()

    def _entry(self, identifier=None):
        rows = registry.snapshot(scope=hermes_home_key(self.home))
        own = [row for row in rows if row.plugin_key == self.ctx.plugin_id
               and (identifier is None or identifier.split(':')[0] == row.source_id)]
        if len(own) == 1:
            return own[0]
        if own or any(row.plugin_key == self.ctx.plugin_id for row in rows):
            raise CatalogError('ownership_conflict', 'No unique owning catalog is available.')
        doc = Projection(self.home).read()
        if any(row['plugin_key'] == self.ctx.plugin_id for row in doc['registrations'].values()):
            raise CatalogError('provider_unavailable', 'Catalog is active in another serving process.')
        return None

    def admit_native(self, native_receipt, approved_choices):
        identifier = native_receipt.get('identifier') if type(native_receipt) is dict else None
        entry = self._entry(identifier)
        if entry is None:
            return None
        _payload(native_receipt, 256 * 1024)
        _payload(approved_choices, 256 * 1024)
        review = validate_result('review', dict(ok=True, data=native_receipt), entry)
        if not review['confirmable'] or not entry.authorized():
            raise CatalogError('review_required', 'A current confirmable review is required.')
        if type(approved_choices) is not dict:
            raise CatalogError('invalid_request', 'Invalid native confirmation choices.')
        from hermes_cli.skill_catalog.validation import validate_request
        validate_request('confirm', dict(id=review['identifier'].split(':')[1], action=review['action'],
            review_token=review['review_token'], acknowledgements=approved_choices.get('acknowledgements', []),
            admission=None))
        if set(approved_choices.get('acknowledgements', [])) != set(review['acknowledgements_required']):
            raise CatalogError('acknowledgement_required', 'Acknowledge the exact review findings.')
        for field, value in approved_choices.items():
            if field == 'acknowledgements':
                continue
            if field not in ('accept_partial', 'preserve_modified', 'accept_sensitive', 'accept_warnings') or type(value) is not bool:
                raise CatalogError('invalid_request', 'Invalid native confirmation choices.')
        # Stable nonsecret operation binding, shared with generic reservation admission.
        # No native token or exceptional choices are stored by core.
        key = native_receipt.get('native_receipt_id', review['review_id'])
        if type(key) is not str:
            raise CatalogError('invalid_request', 'Invalid native receipt identity.')
        operation_id = hashlib.sha256((hermes_home_key(self.home) + '\0' + entry.source_id + '\0' + key).encode()).hexdigest()
        try:
            admission = entry.mutations.projection.reserve(entry, review, operation_id=operation_id)
        except CatalogError as exc:
            if exc.code not in ('review_changed', 'review_expired'):
                raise
            from hermes_cli.skill_catalog.worker import call
            try:
                pending = call(entry, 'operation', dict(operation_id=operation_id))
            except CatalogError:
                raise exc from None
            if (pending['operation_id'] != operation_id or pending['state'] not in ('pending', 'unknown')
                    or pending['identifier'] != review['identifier'] or pending['action'] != review['action']):
                raise CatalogError('review_changed', 'Native pending journal does not match this review.') from None
            admission = entry.mutations.projection.reserve(entry, review, operation_id=operation_id, recovery=True)
        return dict(admission=admission, operation_id=operation_id)

    def _mutations(self, payload):
        if payload.get('permit_id') is not None:
            with self._permit_lock:
                facade = self._permits.get(payload['permit_id'])
            if facade is None:
                raise CatalogError('ownership_conflict', 'Catalog permit is unavailable.')
            return facade
        registration_id = payload.get('registration_id')
        identifier = (payload.get('result') or {}).get('identifier')
        source = identifier.split(':')[0] if type(identifier) is str else None
        rows = [row for row in registry.snapshot(scope=hermes_home_key(self.home))
                if row.plugin_key == self.ctx.plugin_id
                and (source is None or row.source_id == source)
                and (registration_id is None or row.registration_id == registration_id)]
        if len(rows) != 1 or not rows[0].authorized():
            raise CatalogError('provider_unavailable', 'Owning catalog is unavailable.')
        return rows[0].mutations

    def bind(self, payload):
        return self._mutations(payload).bind(payload)

    def acquire(self, payload):
        facade = self._mutations(payload)
        from hermes_cli.skill_catalog.validation import _admission
        admission = {key: payload[key] for key in ('registration_id', 'reservation_id',
            'expected_path_revision', 'expected_installation_revision')}
        with facade.projection.transaction() as doc:
            row = doc['reservations'].get(payload['reservation_id'])
            if row is None:
                raise CatalogError('ownership_conflict', 'Reservation is absent.')
            admission.update(generation=row['generation'], review_id=row['review_id'])
        acquired = facade.acquire(dict(admission=_admission(admission), operation_id=payload['operation_id']))
        return self._remember_permit(facade, acquired)

    def _remember_permit(self, facade, acquired):
        with self._permit_lock:
            self._permits[acquired['permit_id']] = facade
        return acquired

    def validate(self, payload):
        return self._mutations(payload).validate(payload)

    def settle(self, payload):
        return self._mutations(payload).settle(payload)

    def release(self, payload):
        with self._permit_lock:
            facade = self._permits.get(payload['permit_id'])
        if facade is None:
            return True
        result = facade.release(payload)
        with self._permit_lock:
            self._permits.pop(payload['permit_id'], None)
        return result

    def _recovery_mutations(self, operation_id):
        from hermes_cli.skill_catalog.projection import _valid_token
        _valid_token(operation_id)
        rows = [row for row in registry.snapshot(scope=hermes_home_key(self.home))
                if row.plugin_key == self.ctx.plugin_id and row.authorized()]
        doc = Projection(self.home).read()
        owners = {row['registration_id'] for row in doc['reservations'].values()
                  if row['operation_id'] == operation_id}
        matches = [row.mutations for row in rows if row.registration_id in owners]
        if len(matches) != 1:
            raise CatalogError('review_changed', 'No pending operation is bound to this catalog.')
        return matches[0]

    def recovery_acquire(self, payload):
        facade = self._recovery_mutations(payload['operation_id'])
        return self._remember_permit(facade, facade.recovery_acquire(payload))

    def recover(self, operation_id):
        return self._recovery_mutations(operation_id).recover(operation_id)

    def commit(self, admission, operation_id):
        return self._mutations(admission).commit(admission, operation_id)
