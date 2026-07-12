from __future__ import annotations

import json
import os
from copy import deepcopy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gateway import canonical_writer_db as writer_db
from gateway.canonical_writer_db import CanonicalWriterDB, QueryResult
from gateway.canonical_writer_postgres_backend import (
    CANONICAL_WRITER_MIGRATION_OWNER,
    CANONICAL_WRITER_ROLE,
    EXPECTED_HELPER_ROUTINE_SIGNATURES,
    EXPECTED_ROUTINE_SIGNATURES,
    PRODUCTION_CATALOG_SHA256,
    PRODUCTION_STATEMENT_CATALOG,
    PostgresCanonicalWriterBackend,
)
from gateway.canonical_writer_protocol import CanonicalWriterOperation
from gateway.canonical_writer_boundary import DEFAULT_SOCKET_PATH
from scripts.canonical_brain_event_projector import read_events
from scripts.canonical_writer_bootstrap import (
    build_service,
    export_projection_events,
    load_service_config,
)
from scripts.canonical_writer_service import DispatchContext, PeerCredentials


EVENT_ID_1 = "11111111-1111-4111-8111-111111111111"
EVENT_ID_2 = "22222222-2222-4222-8222-222222222222"
EVENT_ID_3 = "33333333-3333-4333-8333-333333333333"


class _ProjectionScopeMixin:
    @contextmanager
    def projection_export_scope(self):
        self.snapshot_entries = getattr(self, "snapshot_entries", 0) + 1
        try:
            yield self
        except BaseException:
            self.snapshot_failures = getattr(self, "snapshot_failures", 0) + 1
            raise
        finally:
            self.snapshot_exits = getattr(self, "snapshot_exits", 0) + 1


class _ProjectionProtocolSession:
    def __init__(self, pages):
        self.pages = pages
        self.queries = []
        self.closed = False

    def query(self, sql, *, maximum_rows):
        self.queries.append((sql, maximum_rows))
        if sql.startswith("BEGIN"):
            return QueryResult((), (), "BEGIN")
        if sql.startswith("SELECT pg_catalog.pg_advisory_xact_lock_shared"):
            return QueryResult((), ((None,),), "SELECT 1")
        if sql == "COMMIT":
            return QueryResult((), (), "COMMIT")
        if sql == "ROLLBACK":
            return QueryResult((), (), "ROLLBACK")
        if "writer_projection_read_events" in sql:
            response = self.pages.pop(0)
            envelope = json.dumps({"ok": True, "result": response})
            return QueryResult(("response",), ((envelope,),), "SELECT 1")
        pytest.fail(f"unexpected projection protocol query: {sql}")

    def close(self):
        self.closed = True


def _config_value(tmp_path):
    return {
        "service": {
            "socket_path": str(DEFAULT_SOCKET_PATH),
            "gateway_unit": "hermes-cloud-gateway.service",
            "gateway_uid": os.getuid() + 1,
            "writer_uid": os.getuid(),
            "writer_gid": os.getgid(),
            "projector_gid": os.getgid() + 1,
            "owner_discord_user_ids": ["owner-1"],
            "connection_timeout_seconds": 20,
            "max_connections": 4,
        },
        "database": {
            "host": "db.internal",
            "port": 5432,
            "database": "canonical",
            "user": "canonical_writer",
            "ca_file": str(tmp_path / "server-ca.pem"),
            "credential_file": str(tmp_path / "database-password"),
            "connect_timeout_seconds": 3,
            "io_timeout_seconds": 7,
        },
        "privileges": {
            "schema": "canonical_brain",
            "table_grants": [],
            "routine_identities": [
                {
                    "signature": signature,
                    "owner": CANONICAL_WRITER_MIGRATION_OWNER,
                    "security_definer": True,
                    "language": "plpgsql",
                    "configuration": [
                        "search_path=pg_catalog, canonical_brain"
                    ],
                    "definition_sha256": "a" * 64,
                }
                for signature in EXPECTED_ROUTINE_SIGNATURES
            ],
            "helper_routine_identities": [
                {
                    "signature": signature,
                    "owner": CANONICAL_WRITER_MIGRATION_OWNER,
                    "security_definer": False,
                    "language": "sql",
                    "configuration": [
                        "search_path=pg_catalog, canonical_brain"
                    ],
                    "definition_sha256": "b" * 64,
                }
                for signature in EXPECTED_HELPER_ROUTINE_SIGNATURES
            ],
            "schema_privileges": ["USAGE"],
            "database_privileges": ["CONNECT"],
            "role_memberships": [CANONICAL_WRITER_ROLE],
            "private_schema_identity_sha256": "c" * 64,
            "deployment_lock_key": 4_841_739_663_211_427_921,
        },
    }


def _write_config(tmp_path, value, *, mode=0o400):
    path = tmp_path / "writer.json"
    if path.exists():
        path.chmod(0o600)
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(mode)
    return path


def _load(path):
    return load_service_config(
        path,
        _expected_owner_uid=os.getuid(),
        _require_root_owned_parents=False,
    )


def test_loads_explicit_secret_free_config_and_pins_routine_catalog(tmp_path):
    config = _load(_write_config(tmp_path, _config_value(tmp_path)))

    assert config.gateway_uid != config.writer_uid
    assert config.owner_discord_user_ids == frozenset({"owner-1"})
    assert config.projector_gid != config.writer_gid
    assert config.database.credential.path == tmp_path / "database-password"
    assert not hasattr(config.database, "password")
    assert config.privileges.executable_routines == EXPECTED_ROUTINE_SIGNATURES
    assert {
        identity.signature
        for identity in config.privileges.dependency_routine_identities
    } == set(EXPECTED_HELPER_ROUTINE_SIGNATURES)
    assert config.privileges.table_grants == ()
    assert config.privileges.sequence_grants == ()
    assert config.privileges.schema_privileges == ("USAGE",)
    assert config.privileges.database_privileges == ("CONNECT",)
    assert config.privileges.role_memberships == (CANONICAL_WRITER_ROLE,)
    assert config.privileges.private_schema_identity_sha256 == "c" * 64
    assert config.privileges.deployment_lock_key == 4_841_739_663_211_427_921


@pytest.mark.parametrize("mode", [0o600, 0o640, 0o644, 0o660])
def test_config_rejects_mutable_or_world_readable_modes(tmp_path, mode):
    path = _write_config(tmp_path, _config_value(tmp_path), mode=mode)

    with pytest.raises(ValueError, match="mode"):
        _load(path)


def test_config_rejects_symlink_secret_fields_unknown_fields_and_same_uid(tmp_path):
    value = _config_value(tmp_path)
    original = _write_config(tmp_path, value)
    link = tmp_path / "link.json"
    link.symlink_to(original)
    with pytest.raises(ValueError, match="symlink"):
        _load(link)

    value = _config_value(tmp_path)
    value["database"]["password"] = "must-never-be-here"
    with pytest.raises(ValueError, match="secret"):
        _load(_write_config(tmp_path, value))

    value = _config_value(tmp_path)
    value["service"]["shell_command"] = "/bin/sh"
    with pytest.raises(ValueError, match="unknown"):
        _load(_write_config(tmp_path, value))

    value = _config_value(tmp_path)
    value["service"]["gateway_uid"] = value["service"]["writer_uid"]
    with pytest.raises(ValueError, match="distinct"):
        _load(_write_config(tmp_path, value))


def test_config_rejects_operator_override_of_immutable_routine_catalog(tmp_path):
    value = _config_value(tmp_path)
    value["privileges"]["executable_routines"] = ["public.anything()"]

    with pytest.raises(ValueError, match="unknown"):
        _load(_write_config(tmp_path, value))


@pytest.mark.parametrize(
    "field,value,reason",
    [
        (
            "table_grants",
            [{"table": "canonical_brain.events", "privileges": ["UPDATE"]}],
            "must be empty",
        ),
        ("schema_privileges", ["USAGE", "CREATE"], "exactly USAGE"),
        ("database_privileges", ["CONNECT", "TEMP"], "exactly CONNECT"),
        ("role_memberships", [], "dedicated writer role"),
        (
            "role_memberships",
            [CANONICAL_WRITER_ROLE, "cloudsqlsuperuser"],
            "dedicated writer role",
        ),
    ],
)
def test_config_hard_pins_zero_direct_grants_and_exact_role_scope(
    tmp_path,
    field,
    value,
    reason,
):
    config = _config_value(tmp_path)
    config["privileges"][field] = value

    with pytest.raises(ValueError, match=reason):
        _load(_write_config(tmp_path, config))


def test_config_requires_exact_non_executable_helper_identity_set(tmp_path):
    config = _config_value(tmp_path)
    config["privileges"]["helper_routine_identities"].pop()
    with pytest.raises(ValueError, match="pinned helper catalog"):
        _load(_write_config(tmp_path, config))

    config = _config_value(tmp_path)
    config["privileges"]["helper_routine_identities"][0][
        "security_definer"
    ] = True
    with pytest.raises(ValueError, match="SECURITY INVOKER"):
        _load(_write_config(tmp_path, config))

    config = _config_value(tmp_path)
    config["privileges"]["routine_identities"][0]["owner"] = "other_owner"
    with pytest.raises(ValueError, match="pinned migration owner"):
        _load(_write_config(tmp_path, config))

    config = _config_value(tmp_path)
    config["privileges"]["helper_routine_identities"][0]["configuration"] = [
        "search_path=public"
    ]
    with pytest.raises(ValueError, match="safe search_path"):
        _load(_write_config(tmp_path, config))


def test_config_rejects_sequence_grant_surface_entirely(tmp_path):
    config = _config_value(tmp_path)
    config["privileges"]["sequence_grants"] = []

    with pytest.raises(ValueError, match="unknown"):
        _load(_write_config(tmp_path, config))


def test_config_rejects_deployment_lock_key_drift(tmp_path):
    value = _config_value(tmp_path)
    value["privileges"]["deployment_lock_key"] += 1

    with pytest.raises(ValueError, match="pinned writer lock"):
        _load(_write_config(tmp_path, value))


@pytest.mark.parametrize("digest", [None, "c" * 63, "C" * 64, "g" * 64])
def test_config_requires_exact_private_schema_identity_digest(tmp_path, digest):
    value = _config_value(tmp_path)
    if digest is None:
        del value["privileges"]["private_schema_identity_sha256"]
    else:
        value["privileges"]["private_schema_identity_sha256"] = digest

    with pytest.raises(ValueError, match="private_schema_identity_sha256"):
        _load(_write_config(tmp_path, value))


def test_config_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "writer.json"
    path.write_text('{"service":{},"service":{}}', encoding="utf-8")
    path.chmod(0o400)

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        _load(path)


class _FakeDatabase:
    def __init__(self, *, config, privilege_policy, statements):
        self.config = config
        self.privilege_policy = privilege_policy
        self.statement_names = statements.names
        self.statement_catalog_sha256 = statements.sha256
        self.attested = False
        self.calls = []

    def startup_attest(self):
        self.attested = True

    def query_fixed(self, statement_name, parameters):
        assert self.attested
        self.calls.append((statement_name, parameters))
        response = json.dumps({"ok": True, "result": {"pong": True}})
        return QueryResult(("response",), ((response,),), "SELECT 1")


def test_build_service_attests_before_exposing_typed_dispatch(tmp_path):
    config = _load(_write_config(tmp_path, _config_value(tmp_path)))

    bootstrap = build_service(config, _database_factory=_FakeDatabase)
    response = bootstrap.server.dispatcher.dispatch(
        CanonicalWriterOperation.PING,
        {},
        DispatchContext(
            request_id="request-1",
            sequence=1,
            deadline_unix_ms=9999999999999,
            idempotency_key=None,
            peer=PeerCredentials(10, config.gateway_uid, 20),
            runtime={"platform": "discord", "thread_id": "thread-1"},
        ),
    )

    assert bootstrap.database.attested is True
    assert bootstrap.database.statement_catalog_sha256 == PRODUCTION_CATALOG_SHA256
    assert response.status == "ok"
    assert response.result == {"pong": True}
    assert bootstrap.database.calls[0][0] == "op_ping"


def test_build_service_rejects_runtime_identity_drift(tmp_path):
    value = deepcopy(_config_value(tmp_path))
    value["service"]["writer_uid"] = os.getuid() + 10
    config = _load(_write_config(tmp_path, value))

    with pytest.raises(PermissionError, match="UID/GID"):
        build_service(config, _database_factory=_FakeDatabase)


def _snapshot_backend(tmp_path, monkeypatch, pages):
    config = _load(_write_config(tmp_path, _config_value(tmp_path)))
    sessions = []
    attested_sessions = []

    def session_factory(_config):
        session = _ProjectionProtocolSession(pages)
        sessions.append(session)
        return session

    def collect_attestation(session, **_kwargs):
        attested_sessions.append(session)
        return object()

    monkeypatch.setattr(
        writer_db,
        "_collect_privilege_attestation",
        collect_attestation,
    )
    monkeypatch.setattr(
        writer_db,
        "validate_privilege_attestation",
        lambda *_args, **_kwargs: None,
    )
    database = CanonicalWriterDB(
        config=config.database,
        privilege_policy=config.privileges,
        statements=PRODUCTION_STATEMENT_CATALOG,
        _session_factory=session_factory,
    )
    database.startup_attest()
    return (
        PostgresCanonicalWriterBackend(database),
        sessions,
        attested_sessions,
    )


def test_projection_export_requires_snapshot_scope_and_leaves_no_partial_file(
    tmp_path,
):
    class _Backend:
        @staticmethod
        def projector_read(_request, _runtime):
            pytest.fail("ordinary per-page query must not be used")

    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=_Backend(),
    )
    target = tmp_path / "canonical-events.json"

    with pytest.raises(RuntimeError, match="projection snapshot"):
        export_projection_events(bootstrap, target, limit=10)

    assert target.exists() is False
    assert list(tmp_path.glob(".canonical-events.json.tmp.*")) == []


def test_multi_page_export_uses_one_attested_serializable_database_snapshot(
    tmp_path,
    monkeypatch,
):
    pages = [
        {
            "events": [{"event_id": EVENT_ID_1}],
            "has_more": True,
            "next_after_event_id": EVENT_ID_1,
        },
        {
            "events": [{"event_id": EVENT_ID_2}],
            "has_more": False,
            "next_after_event_id": EVENT_ID_2,
        },
    ]
    backend, sessions, attested_sessions = _snapshot_backend(
        tmp_path,
        monkeypatch,
        pages,
    )
    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=backend,
    )
    target = tmp_path / "canonical-events.json"

    assert export_projection_events(bootstrap, target, limit=10) == 2

    assert len(sessions) == 2
    assert attested_sessions == sessions
    snapshot = sessions[1]
    sql = [query for query, _maximum_rows in snapshot.queries]
    assert sql.count("BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY") == 1
    assert sum("writer_projection_read_events" in query for query in sql) == 2
    assert sql.count("COMMIT") == 1
    assert "ROLLBACK" not in sql
    assert snapshot.closed is True


def test_database_snapshot_rolls_back_when_projection_validation_fails(
    tmp_path,
    monkeypatch,
):
    pages = [
        {
            "events": [{"event_id": EVENT_ID_1}, "invalid-row"],
            "has_more": False,
        }
    ]
    backend, sessions, attested_sessions = _snapshot_backend(
        tmp_path,
        monkeypatch,
        pages,
    )
    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=backend,
    )
    target = tmp_path / "canonical-events.json"
    original = '{"events":[]}\n'
    target.write_text(original, encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid row"):
        export_projection_events(bootstrap, target, limit=10)

    assert attested_sessions == sessions
    snapshot = sessions[1]
    sql = [query for query, _maximum_rows in snapshot.queries]
    assert sql.count("BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY") == 1
    assert sql.count("ROLLBACK") == 1
    assert "COMMIT" not in sql
    assert snapshot.closed is True
    assert target.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".canonical-events.json.tmp.*")) == []


def test_privileged_projection_export_paginates_and_feeds_pure_projector(tmp_path):
    pages = [
        {
            "events": [
                {"event_id": EVENT_ID_1, "case_id": "case:1"},
                {"event_id": EVENT_ID_2, "case_id": "case:2"},
            ],
            "has_more": True,
            "next_after_event_id": EVENT_ID_2,
        },
        {
            "events": [{"event_id": EVENT_ID_3, "case_id": "case:3"}],
            "has_more": False,
            "next_after_event_id": EVENT_ID_3,
        },
    ]
    calls = []

    class _Backend(_ProjectionScopeMixin):
        @staticmethod
        def projector_read(request, runtime):
            calls.append((request, runtime))
            return pages[len(calls) - 1]

    backend = _Backend()
    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=backend,
    )
    target = tmp_path / "canonical-events.json"

    count = export_projection_events(bootstrap, target, limit=3)
    projected_rows = read_events(target, limit=10)

    assert count == 3
    assert projected_rows == [
        {"event_id": EVENT_ID_1, "case_id": "case:1"},
        {"event_id": EVENT_ID_2, "case_id": "case:2"},
        {"event_id": EVENT_ID_3, "case_id": "case:3"},
    ]
    assert [
        (request.case_id, request.after_event_id, request.limit)
        for request, _runtime in calls
    ] == [
        ("", "", 3),
        ("", EVENT_ID_2, 1),
    ]
    assert all(
        runtime.platform == "writer_service" and runtime.service_internal is True
        for _request, runtime in calls
    )
    assert backend.snapshot_entries == 1
    assert backend.snapshot_exits == 1
    assert getattr(backend, "snapshot_failures", 0) == 0


def test_projection_export_failure_removes_partial_file_and_preserves_target(tmp_path):
    target = tmp_path / "canonical-events.json"
    original = '{"events":[{"event_id":"old"}]}\n'
    target.write_text(original, encoding="utf-8")

    class _Backend(_ProjectionScopeMixin):
        @staticmethod
        def projector_read(_request, _runtime):
            return {
                "events": [{"event_id": EVENT_ID_1}, "invalid-row"],
                "has_more": False,
            }

    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=_Backend(),
    )

    with pytest.raises(RuntimeError, match="invalid row"):
        export_projection_events(bootstrap, target, limit=10)

    assert target.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".canonical-events.json.tmp.*")) == []
    assert bootstrap.backend.snapshot_entries == 1
    assert bootstrap.backend.snapshot_exits == 1
    assert bootstrap.backend.snapshot_failures == 1


def test_projection_export_rejects_nonadvancing_cursor_and_cleans_up(tmp_path):
    calls = 0

    class _Backend(_ProjectionScopeMixin):
        @staticmethod
        def projector_read(_request, _runtime):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "events": [{"event_id": EVENT_ID_1}],
                    "has_more": True,
                    "next_after_event_id": EVENT_ID_1,
                }
            return {
                "events": [{"event_id": EVENT_ID_2}],
                "has_more": True,
                "next_after_event_id": EVENT_ID_1,
            }

    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=_Backend(),
    )
    target = tmp_path / "canonical-events.json"

    with pytest.raises(RuntimeError, match="cursor did not advance"):
        export_projection_events(bootstrap, target, limit=10)

    assert calls == 2
    assert target.exists() is False
    assert list(tmp_path.glob(".canonical-events.json.tmp.*")) == []


def test_projection_export_rejects_page_larger_than_requested_bound(tmp_path):
    class _Backend(_ProjectionScopeMixin):
        @staticmethod
        def projector_read(request, _runtime):
            assert request.limit == 1
            return {
                "events": [
                    {"event_id": EVENT_ID_1},
                    {"event_id": EVENT_ID_2},
                ],
                "has_more": False,
            }

    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=_Backend(),
    )
    target = tmp_path / "canonical-events.json"

    with pytest.raises(RuntimeError, match="exceeded its page limit"):
        export_projection_events(bootstrap, target, limit=1)

    assert target.exists() is False
    assert list(tmp_path.glob(".canonical-events.json.tmp.*")) == []


def test_projection_export_rejects_truncation_at_global_limit(tmp_path):
    target = tmp_path / "canonical-events.json"
    original = '{"events":[]}\n'
    target.write_text(original, encoding="utf-8")

    class _Backend(_ProjectionScopeMixin):
        @staticmethod
        def projector_read(request, _runtime):
            assert request.limit == 1
            return {
                "events": [{"event_id": EVENT_ID_1}],
                "has_more": True,
                "next_after_event_id": EVENT_ID_1,
            }

    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=_Backend(),
    )

    with pytest.raises(RuntimeError, match="global event limit"):
        export_projection_events(bootstrap, target, limit=1)

    assert target.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".canonical-events.json.tmp.*")) == []


def test_pure_projector_rejects_a_smaller_partial_read_limit(tmp_path):
    target = tmp_path / "canonical-events.json"
    target.write_text(
        json.dumps({
            "events": [
                {"event_id": EVENT_ID_1},
                {"event_id": EVENT_ID_2},
            ]
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="projector limit"):
        read_events(target, limit=1)


def test_projection_export_rejects_duplicate_event_ids_across_pages(tmp_path):
    calls = 0

    class _Backend(_ProjectionScopeMixin):
        @staticmethod
        def projector_read(_request, _runtime):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "events": [{"event_id": EVENT_ID_1}],
                    "has_more": True,
                    "next_after_event_id": EVENT_ID_1,
                }
            return {
                "events": [{"event_id": EVENT_ID_1}],
                "has_more": False,
                "next_after_event_id": EVENT_ID_1,
            }

    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=_Backend(),
    )
    target = tmp_path / "canonical-events.json"

    with pytest.raises(RuntimeError, match="duplicate or noncanonical event_id"):
        export_projection_events(bootstrap, target, limit=10)

    assert calls == 2
    assert target.exists() is False
    assert list(tmp_path.glob(".canonical-events.json.tmp.*")) == []
