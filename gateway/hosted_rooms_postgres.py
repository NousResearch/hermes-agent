"""PostgreSQL schema for shared hosted-room coordination."""
from hermes_cli.postgres_util import connect
from hermes_state_backend import resolve_database_settings


def open_database(path):
    settings = resolve_database_settings(path)
    if settings.backend != 'postgres':
        return None
    return connect(settings, 'hosted_rooms', initialize=_initialize)


def _initialize(connection):
    from gateway import hosted_rooms, hosted_room_policy_checkpoint, hosted_room_replicas, hosted_room_driver
    for statement in hosted_rooms._SCHEMA_DDL:
        connection.execute(statement.replace(' REAL', ' DOUBLE PRECISION'))
    connection.execute('CREATE INDEX IF NOT EXISTS idx_hosted_room_events_cursor ON hosted_room_events(room_id, seq)')
    for statement in hosted_room_policy_checkpoint._SCHEMA_DDL:
        connection.execute(statement.replace(' REAL', ' DOUBLE PRECISION'))
    hosted_room_replicas._initialize_replica_schema(connection)
    hosted_room_driver._initialize_schema(connection)
