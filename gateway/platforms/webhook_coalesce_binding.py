"""The accepted coalesced binding is checked against the actual producer snapshot."""
import hashlib
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field

from hermes_state_runtime import RuntimeStoreError


@dataclass(frozen=True)
class Binding:
    profile_id: str
    store_id: str
    route: str
    connector: str
    destination: dict
    authority: object = field(default=None, compare=False, repr=False)

    def to_wire(self):
        return dict(profile_id=self.profile_id, store_id=self.store_id, route=self.route,
                    connector=self.connector, destination=self.destination)

    @classmethod
    def from_wire(cls, value):
        if (not isinstance(value, dict) or set(value) != {'profile_id', 'store_id', 'route', 'connector', 'destination'}
                or any(not isinstance(value[key], str) for key in ('profile_id', 'store_id', 'route', 'connector'))
                or not isinstance(value['destination'], dict)):
            raise RuntimeStoreError('invalid_params')
        return cls(**value)


expected_binding: ContextVar[Binding | None] = ContextVar('webhook_coalesced_binding', default=None)


def ensure_store_identity(db):
    rows = db._read_all("SELECT value FROM state_meta WHERE key='store_instance_id'")
    if rows:
        return rows[0]['value']

    def write(conn):
        conn.execute("INSERT OR IGNORE INTO state_meta(key,value) VALUES('store_instance_id',?)", (uuid.uuid4().hex,))
        return conn.execute("SELECT value FROM state_meta WHERE key='store_instance_id'").fetchone()[0]

    return db._execute_write(write)


def binding_from(adapter, authority, store_id, route, destination):
    return Binding(authority.profile_id, store_id, route,
                   hashlib.sha256(adapter.token.encode()).hexdigest(), destination, authority)


def validate_snapshot(adapter, envelope):
    expected = expected_binding.get()
    if expected is None:
        return
    from gateway.session_authorities import active_authority
    runner = adapter.gateway_runner or adapter._message_handler.__self__
    authority = active_authority(runner)
    if (authority is not expected.authority or authority is None
            or binding_from(adapter, authority, expected.store_id, envelope['webhook_route'],
                            envelope['webhook_delivery']) != expected):
        raise RuntimeStoreError('admission_conflict')
