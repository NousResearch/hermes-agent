"""Single-thread permit owner and writer-side context manager.

The owner retains the admission gate until an explicit release. Caller timeouts
never revoke a writer. The committing process owns the exclusion descriptors.
"""
import copy
import queue
import threading
import time
import uuid
from agent.memory_provider import spawn_context_thread
from hermes_cli.skill_catalog.locks import writer_hold
from hermes_cli.skill_catalog.projection import _target, _conflict
from hermes_cli.skill_catalog.validation import CatalogError, validate_result


class _Command:
    def __init__(self, method, payload):
        self.method, self.payload = method, payload
        self.done = threading.Event()
        self.result = None
        self.error = None


class Mutations:
    """One facade per live registration; all gate ownership stays on its worker."""

    def __init__(self, entry, projection):
        self.entry, self.projection = entry, projection
        self._queue = queue.Queue(maxsize=8)
        self._permit = None
        self._accepted = {}
        self._closed = False
        # Capture this exact Popen, not host.alive (a broken channel is not death)
        # or a host's replaceable current process. Never release an orphan writer.
        from hermes_cli.plugin_host import is_hosted_object
        provider = getattr(entry, 'provider', None)
        slot = object.__getattribute__(provider, '__hermes_plugin_host_ref__') if is_hosted_object(provider) else None
        self._writer_process = slot._host._proc if slot is not None else None
        self._thread = spawn_context_thread(self._run, name='skill-catalog-permits')
        self._thread.start()

    def _run(self):
        while True:
            try:
                command = self._queue.get(timeout=0.1 if self._permit is not None else None)
            except queue.Empty:
                try:
                    self._release_dead_writer()
                except CatalogError:
                    # Failed publication retains the durable bound reservation.
                    # The worker must still accept close/retry commands.
                    pass
                continue
            try:
                self._release_dead_writer()
                command.result = getattr(self, '_do_' + command.method)(command.payload)
            except Exception as exc:  # health: allow BLE001 -- relay the exact exception to the waiting caller, never kill the lock owner
                command.error = exc
            finally:
                command.done.set()
            if command.method == 'close' and command.error is None:
                return

    def _release_dead_writer(self):
        process = self._writer_process
        if process is not None and process.poll() is not None:
            self.entry.stopped = True
            if self._permit is not None:
                self._do_release(dict(permit_id=self._permit['id']))

    def _send(self, method, payload):
        if self._closed and method != 'close':
            raise CatalogError('provider_unavailable', 'Catalog mutations are stopped.')
        command = _Command(method, copy.deepcopy(payload))
        try:
            self._queue.put_nowait(command)
        except queue.Full:
            raise CatalogError('busy', 'Catalog mutation queue is busy.') from None
        if not command.done.wait(3.5):
            raise CatalogError('outcome_unknown', 'Catalog mutation outcome is unknown.')
        if command.error is not None:
            raise command.error
        return command.result

    def bind(self, payload):
        return self.projection.bind(self.entry, payload)

    def acquire(self, payload):
        return self._send('acquire', payload)

    def validate(self, payload):
        return self._send('validate', payload)

    def settle(self, payload):
        return self._send('settle', payload)

    def release(self, payload):
        result = self._send('release', payload)
        try:
            self.reconcile()
        except CatalogError:
            # The reservation remains unknown. A later explicit read retries.
            pass
        return result

    def reconcile(self, *, operation_id=None, target=None, deadline=None):
        end = min(deadline, time.monotonic() + 3) if deadline is not None else time.monotonic() + 3
        return self._send('reconcile', dict(operation_id=operation_id, target=target, deadline=end))

    def settlement(self, operation_id):
        return self._send('settlement', operation_id)

    def close(self):
        if not self._closed:
            self._send('close', None)
            self._thread.join(3.5)

    def _do_acquire(self, payload):
        if self._permit is not None:
            raise CatalogError('busy', 'Catalog commit is active.')
        admission, operation_id = payload['admission'], payload['operation_id']
        transaction = self.projection.transaction()
        doc = transaction.__enter__()
        try:
            row = self.projection.binding(self.entry, doc, admission, operation_id)
            if row.get('recovery'):
                raise CatalogError('review_changed', 'Pending recovery requires the journal recovery hold.')
            if row.get('local_result') is not None:
                raise CatalogError('operation_in_progress', 'Local commit already completed. Reconcile only.')
            permit_id = uuid.uuid4().hex
            self._permit = dict(id=permit_id, transaction=transaction, doc=doc,
                                admission=admission, operation_id=operation_id, settled=False)
            return dict(permit_id=permit_id, target_relative_path=row['target_relative_path'],
                        expected_path_revision=row['expected_path_revision'],
                        expected_installation_revision=row['expected_installation_revision'])
        except BaseException as exc:
            transaction.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def _current(self, permit_id):
        permit = self._permit
        if permit is None or permit['id'] != permit_id:
            raise _conflict()
        self.projection._owner(self.entry, permit['doc'])
        return permit

    def _do_validate(self, payload):
        permit = self._current(payload['permit_id'])
        self.projection.binding(self.entry, permit['doc'], permit['admission'], permit['operation_id'])
        return True

    def _do_release(self, payload):
        if self._permit is None:
            return True
        permit = self._permit
        if permit['id'] != payload['permit_id']:
            raise _conflict()
        row = permit['doc']['reservations'].get(permit['admission']['reservation_id'])
        if row is not None and not permit['settled']:
            row['state'] = 'unknown'
        self._permit = None
        permit['transaction'].__exit__(None, None, None)
        return True

    def _do_close(self, _payload):
        if self._permit is not None:
            raise CatalogError('busy', 'A writer still owns a catalog commit.')
        self._closed = True

    def _do_reconcile(self, payload):
        from hermes_cli.skill_catalog.projection import _overlap
        from hermes_cli.skill_catalog.worker import call
        if self._permit is not None:
            raise CatalogError('operation_in_progress', 'A catalog writer is still active.')
        deadline = payload['deadline']
        if time.monotonic() >= deadline:
            raise CatalogError('timeout', 'Catalog request timed out.')
        with self.projection.transaction(deadline=deadline) as doc:
            rows = [(key, copy.deepcopy(row)) for key, row in doc['reservations'].items()
                    if row['registration_id'] == self.entry.registration_id
                    and (row['state'] == 'unknown' or row.get('local_result') is not None)
                    and (payload['operation_id'] is None or row['operation_id'] == payload['operation_id'])
                    and (payload['target'] is None or _overlap(row['target_relative_path'], payload['target']))]
            for reservation_id, row in rows:
                self._reconcile_row(doc, reservation_id, row, call, deadline)

    def _reconcile_row(self, doc, reservation_id, row, call, deadline):
        if row['operation_id'] is None:
            raise CatalogError('ownership_state_unavailable', 'Catalog operation binding is unavailable.')
        try:
            with writer_hold(self.projection.home, [row['target_relative_path']], deadline=time.monotonic()):
                try:
                    result = call(self.entry, 'operation', dict(operation_id=row['operation_id']), deadline=deadline)
                except CatalogError:
                    raise CatalogError('ownership_state_unavailable', 'Catalog operation cannot be reconciled.') from None
                if result['operation_id'] != row['operation_id']:
                    raise _conflict()
                if result['state'] in ('completed', 'failed'):
                    payload = dict(reservation_id=reservation_id, operation_id=row['operation_id'],
                        expected_path_revision=row['expected_path_revision'], result=result)
                    revision = self._settle_doc(doc, payload, result)
                    self.projection.save(doc)
                    self._accepted[row['operation_id']] = (payload, revision)
        except CatalogError as exc:
            if exc.code == 'busy':
                raise CatalogError('operation_in_progress', 'A catalog writer is still active.') from None
            raise

    def _do_settlement(self, operation_id):
        if operation_id in self._accepted:
            return 'settled'
        if self._permit is not None:
            rows = self._permit['doc']['reservations']
        else:
            rows = self.projection.read()['reservations']
        return 'reconciling' if any(row['operation_id'] == operation_id for row in rows.values()) else 'settled'

    def _do_settle(self, payload):
        result = validate_result('operation', dict(ok=True, data=payload['result']), self.entry)
        if result['operation_id'] != payload['operation_id']:
            raise _conflict()
        if 'permit_id' in payload:
            permit = self._current(payload['permit_id'])
            if (payload['reservation_id'], payload['operation_id'], payload['expected_path_revision']) != (
                    permit['admission']['reservation_id'], permit['operation_id'],
                    permit['admission']['expected_path_revision']):
                raise _conflict()
        key = result['operation_id']
        fingerprint = {k: v for k, v in payload.items() if k != 'permit_id'}
        if key in self._accepted:
            prior, revision = self._accepted[key]
            self.projection._owner(self.entry, self._permit['doc'] if self._permit else self.projection.read())
            if fingerprint != prior:
                raise _conflict()
            return revision
        if 'permit_id' in payload:
            permit = self._current(payload['permit_id'])
            candidate = copy.deepcopy(permit['doc'])
            revision = self._settle_doc(candidate, payload, result)
            # A failed atomic publication must not leak into release's later save.
            self.projection.save(candidate)
            permit['doc'].clear()
            permit['doc'].update(candidate)
            permit['settled'] = True
        else:
            if self._permit is not None:
                raise CatalogError('busy', 'Catalog commit is active.')
            with self.projection.transaction() as doc:
                revision = self._settle_doc(doc, payload, result)
        if result['state'] in ('completed', 'failed'):
            self._accepted[key] = (fingerprint, revision)
        return revision

    def _settle_doc(self, doc, payload, result):
        self.projection._owner(self.entry, doc)
        row = doc['reservations'].get(payload['reservation_id'])
        if row is None:
            raise _conflict()
        if (row['registration_id'] != self.entry.registration_id or row['generation'] != self.entry.generation
                or row['operation_id'] != payload['operation_id']
                or row['expected_path_revision'] != payload['expected_path_revision']
                or row['identifier'] != result['identifier'] or row['action'] != result['action']):
            raise _conflict()
        path = row['target_relative_path']
        if doc['path_revisions'].get(path, 0) != row['expected_path_revision']:
            raise _conflict()
        pending = result['state'] == 'pending'
        if pending and 'permit_id' not in payload:
            raise _conflict()
        if result['state'] == 'unknown':
            row['state'] = 'unknown'
            return row['expected_path_revision']
        prior = row.get('local_result')
        if prior is not None and pending:
            if prior != result:
                raise _conflict()
            return row['expected_path_revision']
        if pending:
            row['local_result'] = result
        if pending or result['state'] == 'completed':
            self._project(doc, row, result, advance=not pending)
        elif prior is not None:
            # Failed remote reconciliation does not undo committed local files.
            self._project(doc, row, prior, advance=True)
        revision = doc['path_revisions'].get(path, 0)
        if not pending:
            doc['reservations'].pop(payload['reservation_id'])
        return revision

    def _project(self, doc, row, result, *, advance):
        path = row['target_relative_path']
        revision = row['expected_path_revision'] + int(advance)
        if result['managed_state'] in ('installed', 'pending'):
            if result['installation_revision'] is None or not _target(self.projection.home, path).exists():
                raise _conflict()
            doc['targets'][path] = dict(registration_id=self.entry.registration_id,
                identifier=row['identifier'], installation_revision=result['installation_revision'],
                managed_state=result['managed_state'], path_revision=revision)
        elif result['managed_state'] in ('removed', 'not_installed'):
            doc['targets'].pop(path, None)
        doc['path_revisions'][path] = revision

    def _do_recovery_acquire(self, payload):
        from hermes_cli.skill_catalog.projection import _admission, _valid_token
        operation_id = _valid_token(payload['operation_id'])
        if self._permit is not None:
            raise CatalogError('busy', 'Catalog commit is active.')
        transaction = self.projection.transaction()
        doc = transaction.__enter__()
        try:
            self.projection._owner(self.entry, doc)
            rows = [(key, row) for key, row in doc['reservations'].items()
                    if row['registration_id'] == self.entry.registration_id
                    and row['operation_id'] == operation_id]
            if len(rows) != 1:
                raise CatalogError('review_changed', 'No pending operation is bound to this catalog.')
            key, row = rows[0]
            admission = _admission(key, row)
            self.projection.binding(self.entry, doc, admission, operation_id)
            permit_id = uuid.uuid4().hex
            self._permit = dict(id=permit_id, transaction=transaction, doc=doc,
                                admission=admission, operation_id=operation_id, settled=False)
            return dict(permit_id=permit_id, admission=admission,
                        target_relative_path=row['target_relative_path'], identifier=row['identifier'],
                        action=row['action'], local_result=row.get('local_result'),
                        expected_path_revision=row['expected_path_revision'],
                        expected_installation_revision=row['expected_installation_revision'])
        except BaseException as exc:
            transaction.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def recovery_acquire(self, payload):
        return self._send('recovery_acquire', payload)

    def recover(self, operation_id):
        from hermes_cli.skill_catalog.recovery import recover
        acquired = self.recovery_acquire(dict(operation_id=operation_id))
        return recover(self, self.projection.home, operation_id, acquired)

    def commit(self, admission, operation_id):
        from hermes_cli.skill_catalog.writer import commit
        acquired = self.acquire(dict(admission=admission, operation_id=operation_id))
        return commit(self, self.projection.home, admission, operation_id, acquired)
