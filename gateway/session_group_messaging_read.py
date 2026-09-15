"""Explicit native-owner consent for independently authenticated inventory reads.

This is not messaging command admission, Home consent or a room-control grant.
The only owner substitution is the existing groups.list filter, behind a live,
revocable attestation which retains the actual messaging actor.
"""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import uuid

from gateway.config import Platform
from gateway.session_contract import Principal
from gateway.session_group_messaging_identity import home_thread_from_source, is_private_source, trusted_person
from hermes_state_runtime import RuntimeStoreError, _epoch

BINDING_METHODS = {
    'groups.messaging.read.grant': 'session:operator',
    'groups.messaging.read.revoke': 'session:operator',
}
BINDING_FIELDS = {
    'groups.messaging.read.grant': {'request_id', 'recipient', 'expected_generation'},
    'groups.messaging.read.revoke': {'request_id', 'recipient', 'expected_generation', 'binding_id'},
}
_PREFIX = 'gateway.messaging.read.v1.'
_RECIPIENT_FIELDS = frozenset({'platform', 'user_id', 'chat_id', 'thread_id', 'scope_id',
                              'transport_profile', 'runtime_profile'})
_MAX_GENERATION = 2**63 - 2
MAX_PAGE_SIZE = 8
MAX_INVENTORY_OFFSET = 4096
_MAX_BINDINGS = 4096
_MAX_REQUESTS = 16384


def _text(value, maximum):
    if (type(value) is not str or not 1 <= len(value) <= maximum or value != value.strip()
            or any(unicodedata.category(c).startswith('C') for c in value)):
        raise RuntimeStoreError('invalid_params')
    return value


def _recipient(value):
    if type(value) is not dict or set(value) != _RECIPIENT_FIELDS:
        raise RuntimeStoreError('invalid_params')
    result = {}
    for key, field in value.items():
        result[key] = None if key in {'thread_id', 'scope_id'} and field is None else _text(
            field, 64 if key in {'platform', 'transport_profile', 'runtime_profile'} else 256)
    try:
        Platform(result['platform'])
    except ValueError as exc:
        raise RuntimeStoreError('invalid_params') from exc
    if result['runtime_profile'] != 'default':
        raise RuntimeStoreError('profile_mismatch')
    return result


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def _key(kind, value):
    return _PREFIX + kind + '.' + hashlib.sha256(_json(value).encode()).hexdigest()


def _binding_id(value):
    if type(value) is not str or re.fullmatch(r'mr-[0-9a-f]{32}', value) is None:
        raise RuntimeStoreError('invalid_params')
    return value


def _load(conn, key):
    row = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    if row is None:
        return None
    try:
        value = json.loads(row[0])
        if type(value) is not dict:
            raise ValueError('invalid stored binding')
        return value
    except (TypeError, ValueError) as exc:
        raise RuntimeStoreError('permission_denied') from exc


def _binding(conn, recipient, profile_id):
    record = _load(conn, _key('binding', recipient))
    if record is None:
        return None
    fields = {'version', 'recipient', 'profile_id', 'owner', 'scope', 'binding_id', 'generation', 'active'}
    if (set(record) != fields or type(record['version']) is not int or record['version'] != 1
            or record['recipient'] != recipient or record['profile_id'] != profile_id
            or record['scope'] != 'groups.list' or type(record['active']) is not bool
            or type(record['generation']) is not int or not 1 <= record['generation'] <= _MAX_GENERATION
            or type(record['owner']) is not str or not 1 <= len(record['owner']) <= 1024):
        raise RuntimeStoreError('permission_denied')
    _binding_id(record['binding_id'])
    return record


def _response(record):
    return {field: record[field] for field in ('binding_id', 'generation', 'active')}


def prepare_native_binding(connection, method, params):
    """Freeze native provenance and immutable request before the dispatch await."""
    from gateway.session_group_peers import _native_owner
    operation = _native_owner(connection)
    if method not in BINDING_FIELDS or set(params) != BINDING_FIELDS[method]:
        raise RuntimeStoreError('invalid_params')
    recipient = _recipient(params['recipient'])
    request_id = _text(params['request_id'], 128)
    generation = params['expected_generation']
    if type(generation) is not int or not 0 <= generation < _MAX_GENERATION:
        raise RuntimeStoreError('invalid_params')
    binding_id = _binding_id(params['binding_id']) if method.endswith('.revoke') else None
    owner = _text(operation.actor.subject, 1024)
    intent = dict(method=method, recipient=recipient, expected_generation=generation,
                  binding_id=binding_id, owner=owner, profile_id=operation.profile_id)
    return operation, _key('request', [owner, request_id]), _json(intent)


def commit_native_binding(prepared):
    operation, request_key, intent_json = prepared
    intent = json.loads(intent_json)
    recipient, owner = intent['recipient'], intent['owner']
    binding_key = _key('binding', recipient)

    def write(conn):
        operation.require_current(conn)
        current = _binding(conn, recipient, operation.profile_id)
        prior = _load(conn, request_key)
        if prior is not None:
            if prior.get('intent') != intent:
                raise RuntimeStoreError('admission_conflict')
            if current is None or prior.get('state') != current:
                raise RuntimeStoreError('messaging_read_stale')
            return _response(current)
        if current is not None and current['owner'] != owner:
            raise RuntimeStoreError('permission_denied')
        if (current['generation'] if current else 0) != intent['expected_generation']:
            raise RuntimeStoreError('messaging_read_stale')
        granting = intent['method'].endswith('.grant')
        if granting:
            if current is not None and current['active']:
                raise RuntimeStoreError('admission_conflict')
        elif (current is None or not current['active'] or current['binding_id'] != intent['binding_id']):
            raise RuntimeStoreError('messaging_read_stale')
        counts = {kind: conn.execute('SELECT COUNT(*) FROM state_meta WHERE key LIKE ?',
                    (_PREFIX + kind + '.%',)).fetchone()[0] for kind in ('binding', 'request')}
        if ((current is None and counts['binding'] >= _MAX_BINDINGS)
                or counts['request'] >= _MAX_REQUESTS
                # Reserve a receipt for every binding's future revoke; a full
                # enrollment ledger must never strand an active permission.
                or (granting and counts['request'] + counts['binding'] + 2 > _MAX_REQUESTS)):
            raise RuntimeStoreError('messaging_read_capacity')
        record = dict(version=1, recipient=recipient, owner=owner, profile_id=operation.profile_id,
            scope='groups.list', binding_id='mr-' + uuid.uuid4().hex if granting else current['binding_id'],
            generation=intent['expected_generation'] + 1, active=granting)
        operation.require_current(conn)
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     (binding_key, _json(record)))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                     (request_key, _json(dict(intent=intent, state=record))))
        operation.require_current(conn)
        return _response(record)

    return operation.db._execute_write(write)


def _receiver(runner, event):
    """M73 registered-receiver policy, without its legacy primary fallback."""
    source = event.source
    if not trusted_person(event) or not is_private_source(source):
        raise RuntimeStoreError('permission_denied')
    owner = runner._transport_owner(source)
    if owner is None:
        raise RuntimeStoreError('permission_denied')
    adapter, profile = owner
    config = getattr(adapter, 'config', None)
    if config is None:
        raise RuntimeStoreError('permission_denied')
    primary = _text(getattr(runner, '_primary_profile_name', None), 64)
    recipient = _recipient(dict(platform=source.platform.value, user_id=source.user_id,
        chat_id=source.chat_id, thread_id=home_thread_from_source(source), scope_id=source.scope_id,
        transport_profile=profile if profile is not None else primary,
        runtime_profile=source.profile if source.profile is not None else primary))
    from gateway.slash_access import policy_from_extra
    policy = policy_from_extra(config.extra if isinstance(config.extra, dict) else {}, 'dm')
    if (runner._is_user_authorized_for_source(source) is not True
            or not policy.can_run(source.user_id, 'group')):
        raise RuntimeStoreError('permission_denied')
    # Retain the actual source principal, not its delegating native owner.
    subject = 'messaging:' + json.dumps([source.profile, source.platform.value, source.chat_id,
                                        source.thread_id, source.user_id], separators=(',', ':'))
    return adapter, config, policy, recipient, primary, subject


@dataclass(frozen=True)
class _InventoryRead:
    runner: object
    authority: object
    actor: Principal
    registry: object
    db: object
    home: Path
    profile_id: str
    epoch: int
    instance_id: str
    service: object
    checker: object
    event: object
    source: object
    adapter: object
    config: object
    policy: object
    recipient_json: str
    primary: str
    authorization_home: object
    state_json: str | None = None

    def _require_context(self):
        from gateway.session_authorities import active_authority, served_profile_name
        from gateway.session_hosted_service import CanonicalHostedRoomService
        from hermes_constants import get_hermes_home
        a = self.authority
        if (getattr(self.runner, 'session_authorities', None) is not self.registry or self.registry is None
                or self.registry.for_home(self.home) is not a or active_authority(self.runner) is not a
                or getattr(self.runner, 'session_authority', None) is not a or a.runner is not self.runner
                or a.profile_id != self.profile_id or a.db is not self.db or a.epoch != self.epoch
                or a.instance_id != self.instance_id or Path(get_hermes_home()).resolve() != self.home
                or served_profile_name(self.home) != 'default'
                or Path(self.db.db_path).resolve() != self.home / 'state.db'):
            raise RuntimeStoreError('profile_mismatch')
        if (getattr(a, 'hosted_room_service', None) is not self.service
                or not isinstance(self.service, CanonicalHostedRoomService)
                or self.service.authority is not a or Path(self.service.db_path).resolve() != self.home / 'state.db'
                or not callable(self.checker) or self.service.authorize_room != self.checker):
            raise RuntimeStoreError('permission_denied')
        if (self.event.source is not self.source
                or getattr(self.source, '_authorization_profile_home', None) != self.authorization_home):
            raise RuntimeStoreError('permission_denied')
        adapter, config, policy, recipient, primary, subject = _receiver(self.runner, self.event)
        if (adapter is not self.adapter or config is not self.config or policy != self.policy
                or _json(recipient) != self.recipient_json or primary != self.primary
                or self.actor.subject != subject or self.actor.profile_id != self.profile_id
                or self.actor.capabilities != frozenset({'session:read'})):
            raise RuntimeStoreError('permission_denied')
        return recipient

    def require_current(self):
        recipient = self._require_context()
        with self.db._read_ctx() as conn:
            # The read connection/lock may have waited behind a native write.
            self._require_context()
            state = self._consent(conn, recipient)
            self._require_context()
            return state

    def _consent(self, conn, recipient):
        if Path(conn.execute('PRAGMA database_list').fetchone()[2]).resolve() != self.home / 'state.db':
            raise RuntimeStoreError('profile_mismatch')
        _epoch(conn, self.epoch)
        if conn.execute('SELECT instance_id FROM runtime_epoch WHERE singleton=1').fetchone()[0] != self.instance_id:
            raise RuntimeStoreError('stale_epoch')
        state = _binding(conn, recipient, self.profile_id)
        if state is None or not state['active'] or (self.state_json is not None and _json(state) != self.state_json):
            raise RuntimeStoreError('permission_denied')
        return state

    def project(self, result):
        state = self.require_current()
        projected = []
        for room in result['rooms']:
            # A page may have waited in a worker after its first owner check.
            self.checker(state['owner'], room['room_id'])
            name = ''.join(' ' if unicodedata.category(c).startswith('C') else c for c in room['name'])
            projected.append(dict(name=' '.join(name.split())[:72], member_count=len(room['members'])))
        self.require_current()
        return dict(rooms=projected, next_offset=result['next_offset'])


def _attest_inventory(runner, event):
    from gateway.session_authorities import active_authority
    authority = active_authority(runner)
    if authority is None:
        raise RuntimeStoreError('permission_denied')
    # Capture the runtime before source authorization or a DB read can block.
    registry, db, profile_id = getattr(runner, 'session_authorities', None), authority.db, authority.profile_id
    epoch, instance = authority.epoch, authority.instance_id
    service = getattr(authority, 'hosted_room_service', None)
    checker = getattr(service, 'authorize_room', None)
    source = event.source
    authorization_home = getattr(source, '_authorization_profile_home', None)
    adapter, config, policy, recipient, primary, subject = _receiver(runner, event)
    context = _InventoryRead(runner, authority,
        Principal(subject, profile_id, frozenset({'session:read'}), 'messaging-inventory'),
        registry, db, Path(profile_id).resolve(), profile_id, epoch, instance, service, checker,
        event, source, adapter, config, policy, _json(recipient), primary, authorization_home)
    state = context.require_current()
    return replace(context, state_json=_json(state))


def validate_page(params):
    if type(params) is not dict or set(params) != {'limit', 'offset'}:
        raise RuntimeStoreError('invalid_params')
    limit, offset = params['limit'], params['offset']
    if (type(limit) is not int or not 1 <= limit <= MAX_PAGE_SIZE or type(offset) is not int
            or not 0 <= offset < MAX_INVENTORY_OFFSET or offset + limit > MAX_INVENTORY_OFFSET):
        raise RuntimeStoreError('invalid_params')


async def read_messaging_inventory_page(runner, event, *, limit=8, offset=0):
    """Return only bounded names/member counts and the canonical raw-page cursor."""
    from gateway.session_group_controls import dispatch_group_control
    params = dict(limit=limit, offset=offset)
    validate_page(params)
    context = _attest_inventory(runner, event)
    return await dispatch_group_control(context, 'groups.list', params)
