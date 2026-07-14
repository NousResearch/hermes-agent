from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

import pytest


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    ROOT
    / "scripts"
    / "sql"
    / "canonical_writer_foundation_phase_b_preflight_v1.sql"
)
ROOT_PASSWORD = "fixture-only-cloudsqladmin-password"
WRITER_PASSWORD = "fixture-only-writer-password"
ADMIN_PASSWORD = "fixture-only-phase-b-admin-password"
WRITER_LOGIN = "muncho_canary_writer_login"
ADMIN = "muncho_canary_admin_1234567890abcdef"


def _run(
    command: list[str],
    *,
    stdin: bytes = b"",
    accepted: frozenset[int] = frozenset({0}),
) -> bytes:
    completed = subprocess.run(
        command,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode not in accepted:
        sanitized = completed.stdout
        for secret in (ROOT_PASSWORD, WRITER_PASSWORD, ADMIN_PASSWORD):
            sanitized = sanitized.replace(secret.encode(), b"<redacted>")
        pytest.fail(
            f"postgres fixture command failed ({completed.returncode}): "
            + sanitized.decode("utf-8", errors="replace")
        )
    return completed.stdout


def _psql(
    container: str,
    sql: bytes,
    *,
    user: str = "cloudsqladmin",
    password: str = ROOT_PASSWORD,
    database: str = "muncho_canary_brain",
) -> bytes:
    return _run(
        [
            "docker",
            "exec",
            "-e",
            f"PGPASSWORD={password}",
            "-i",
            container,
            "psql",
            "-h",
            "127.0.0.1",
            "-X",
            "-qAt",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            user,
            "-d",
            database,
        ],
        stdin=sql,
    )


def _preflight(container: str) -> dict[str, object]:
    raw = _psql(
        container,
        ARTIFACT_PATH.read_bytes(),
        user=WRITER_LOGIN,
        password=WRITER_PASSWORD,
    )
    lines = [line for line in raw.decode("utf-8").splitlines() if line]
    assert len(lines) == 1
    value = json.loads(lines[0])
    assert isinstance(value, dict)
    digest = value.pop("receipt_sha256")
    unsigned_text = value.pop("unsigned_receipt_jsonb_text")
    assert isinstance(unsigned_text, str)
    assert digest == hashlib.sha256(unsigned_text.encode("utf-8")).hexdigest()
    assert json.loads(unsigned_text) == value
    value["unsigned_receipt_jsonb_text"] = unsigned_text
    value["receipt_sha256"] = digest
    return value


def _wait_for_postgres(container: str) -> None:
    for _attempt in range(100):
        ready = subprocess.run(
            [
                "docker",
                "exec",
                "-e",
                f"PGPASSWORD={ROOT_PASSWORD}",
                container,
                "psql",
                "-h",
                "127.0.0.1",
                "-X",
                "-qAt",
                "-U",
                "cloudsqladmin",
                "-d",
                "postgres",
                "-c",
                "SELECT 1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if ready.returncode == 0 and ready.stdout == b"1\n":
            return
        time.sleep(0.1)
    pytest.fail("postgres:18 fixture did not become ready")


def _create_foundation(container: str) -> None:
    _psql(
        container,
        f"""
CREATE ROLE cloudsqlsuperuser
    NOLOGIN INHERIT NOSUPERUSER CREATEDB CREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
CREATE ROLE canonical_brain_migration_owner
    NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
CREATE ROLE canonical_brain_writer
    NOLOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
CREATE ROLE {WRITER_LOGIN}
    LOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1
    PASSWORD '{WRITER_PASSWORD}';
CREATE ROLE muncho_legacy_source
    NOLOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
GRANT canonical_brain_writer TO {WRITER_LOGIN}
    WITH ADMIN FALSE, INHERIT TRUE, SET FALSE;
REVOKE CONNECT, TEMPORARY ON DATABASE postgres FROM PUBLIC;
REVOKE CONNECT, TEMPORARY ON DATABASE template1 FROM PUBLIC;
REVOKE CONNECT, TEMPORARY ON DATABASE cloudsqladmin FROM PUBLIC;
CREATE DATABASE muncho_canary_brain OWNER cloudsqlsuperuser;
""".encode(),
        database="postgres",
    )
    _psql(
        container,
        b"""
REVOKE ALL PRIVILEGES ON DATABASE muncho_canary_brain FROM PUBLIC;
GRANT CONNECT ON DATABASE muncho_canary_brain TO canonical_brain_writer;

CREATE TABLE public.canonical_event_log (
    event_id uuid NOT NULL,
    schema_version text NOT NULL,
    event_type text NOT NULL,
    occurred_at timestamptz NOT NULL,
    case_id text NOT NULL,
    source jsonb NOT NULL,
    actor jsonb NOT NULL,
    subject jsonb NOT NULL,
    evidence jsonb NOT NULL,
    decision jsonb NOT NULL,
    status jsonb NOT NULL,
    next_action jsonb NOT NULL,
    safety jsonb NOT NULL,
    payload jsonb NOT NULL,
    PRIMARY KEY (event_id)
);
ALTER TABLE public.canonical_event_log
    OWNER TO canonical_brain_migration_owner;
REVOKE ALL ON TABLE public.canonical_event_log FROM PUBLIC;

CREATE SCHEMA canonical_brain
    AUTHORIZATION canonical_brain_migration_owner;
CREATE FUNCTION canonical_brain.writer_ping(request jsonb, runtime jsonb)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, canonical_brain
AS $function$
BEGIN
    RETURN pg_catalog.jsonb_build_object(
        'ok', true, 'service', 'canonical_writer', 'protocol', 'v1'
    );
END
$function$;
ALTER FUNCTION canonical_brain.writer_ping(jsonb,jsonb)
    OWNER TO canonical_brain_migration_owner;
REVOKE ALL ON FUNCTION canonical_brain.writer_ping(jsonb,jsonb) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION canonical_brain.writer_ping(jsonb,jsonb)
    TO canonical_brain_writer;

CREATE SCHEMA canonical_brain_legacy_quarantine
    AUTHORIZATION muncho_legacy_source;
CREATE TABLE canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1 (
    event_id uuid NOT NULL PRIMARY KEY,
    schema_version text NOT NULL,
    event_type text NOT NULL,
    occurred_at timestamptz NOT NULL,
    case_id text NOT NULL,
    source jsonb NOT NULL DEFAULT '{}'::jsonb,
    actor jsonb NOT NULL DEFAULT '{}'::jsonb,
    subject jsonb NOT NULL DEFAULT '{}'::jsonb,
    evidence jsonb NOT NULL DEFAULT '[]'::jsonb,
    decision jsonb NOT NULL DEFAULT '{}'::jsonb,
    status jsonb NOT NULL DEFAULT '{}'::jsonb,
    next_action jsonb NOT NULL DEFAULT '{}'::jsonb,
    safety jsonb NOT NULL DEFAULT '{}'::jsonb,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    inserted_at timestamptz NOT NULL DEFAULT now(),
    idempotency_key text,
    source_spool text,
    spool_line_number integer,
    raw_event_sha256 text
);
CREATE INDEX legacy_case_id_idx ON
    canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1 (case_id);
CREATE INDEX legacy_event_type_idx ON
    canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1 (event_type);
CREATE INDEX legacy_occurred_at_idx ON
    canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1 (occurred_at);
CREATE UNIQUE INDEX legacy_idempotency_key_idx ON
    canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1
    (idempotency_key) WHERE idempotency_key IS NOT NULL;
ALTER TABLE canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1
    OWNER TO muncho_legacy_source;
REVOKE ALL ON TABLE
    canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1
    FROM PUBLIC;

REVOKE ALL ON SCHEMA public FROM PUBLIC,
    canonical_brain_writer, muncho_canary_writer_login;
GRANT USAGE ON SCHEMA public TO canonical_brain_migration_owner;
REVOKE ALL ON SCHEMA canonical_brain FROM PUBLIC,
    canonical_brain_writer, muncho_canary_writer_login;
GRANT USAGE ON SCHEMA canonical_brain TO canonical_brain_writer;
REVOKE ALL ON SCHEMA canonical_brain_legacy_quarantine FROM PUBLIC,
    canonical_brain_writer, muncho_canary_writer_login;
REVOKE ALL ON TABLE public.canonical_event_log
    FROM canonical_brain_writer, muncho_canary_writer_login;
""",
    )


def _canonical14_columns() -> list[dict[str, object]]:
    names_and_types = (
        ("event_id", "uuid"),
        ("schema_version", "text"),
        ("event_type", "text"),
        ("occurred_at", "timestamp with time zone"),
        ("case_id", "text"),
        ("source", "jsonb"),
        ("actor", "jsonb"),
        ("subject", "jsonb"),
        ("evidence", "jsonb"),
        ("decision", "jsonb"),
        ("status", "jsonb"),
        ("next_action", "jsonb"),
        ("safety", "jsonb"),
        ("payload", "jsonb"),
    )
    type_oids = {
        "uuid": "2950",
        "text": "25",
        "timestamp with time zone": "1184",
        "jsonb": "3802",
    }
    return [
        {
            "acl": [],
            "name": name,
            "type": type_name,
            "type_oid": type_oids[type_name],
            "identity": "",
            "is_local": True,
            "not_null": True,
            "generated": "",
            "position": position,
            "acl_is_null": True,
            "has_default": False,
            "has_missing": False,
            "array_dimensions": 0,
            "inheritance_count": 0,
            "options_are_empty": True,
            "statistics_target": None,
            "fdw_options_are_empty": True,
            "storage_is_type_default": True,
            "collation_is_type_default": True,
            "default_expression_sha256": None,
        }
        for position, (name, type_name) in enumerate(names_and_types, start=1)
    ]


@pytest.mark.integration
def test_writer_preflight_observes_pristine_recovery_and_catalog_drift():
    if shutil.which("docker") is None:
        pytest.skip("docker is unavailable")
    probe = subprocess.run(
        ["docker", "image", "inspect", "postgres:18"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("the local postgres:18 image is unavailable")

    container = "cw-phase-b-preflight-" + uuid.uuid4().hex[:12]
    try:
        _run([
            "docker",
            "run",
            "-d",
            "--name",
            container,
            "-e",
            "POSTGRES_USER=cloudsqladmin",
            "-e",
            f"POSTGRES_PASSWORD={ROOT_PASSWORD}",
            "postgres:18",
        ])
        _wait_for_postgres(container)
        _create_foundation(container)

        assert _psql(
            container,
            b"""
SELECT pg_catalog.has_schema_privilege(
           'muncho_canary_writer_login', 'public', 'USAGE'
       ), pg_catalog.has_schema_privilege(
           'muncho_canary_writer_login', 'canonical_brain', 'USAGE'
       ), pg_catalog.has_schema_privilege(
           'muncho_canary_writer_login',
           'canonical_brain_legacy_quarantine', 'USAGE'
       ), pg_catalog.has_table_privilege(
           'muncho_canary_writer_login',
           'public.canonical_event_log', 'SELECT'
       );
""",
        ) == b"f|t|f|f\n"

        pristine = _preflight(container)
        assert pristine["schema"] == (
            "muncho-canonical-writer-foundation-phase-b-db-preflight.v1"
        )
        assert pristine["preflight"] is True
        assert pristine["terminal"] is False
        assert pristine["secret_material_recorded"] is False
        serialized_pristine = json.dumps(pristine, sort_keys=True)
        assert ROOT_PASSWORD not in serialized_pristine
        assert WRITER_PASSWORD not in serialized_pristine
        assert ADMIN_PASSWORD not in serialized_pristine
        assert pristine["database_owner"] == "cloudsqlsuperuser"
        assert pristine["bootstrap_role_absent"] is True
        assert pristine["bootstrap_login_absent"] is True
        assert pristine["temporary_admin_roles"] == []
        assert {row["name"] for row in pristine["roles"]} == {
            "canonical_brain_migration_owner",
            "canonical_brain_writer",
            WRITER_LOGIN,
        }
        assert pristine["memberships"] == [{
            "grantor": "cloudsqladmin",
            "member_role": WRITER_LOGIN,
            "set_option": False,
            "admin_option": False,
            "granted_role": "canonical_brain_writer",
            "inherit_option": True,
        }]

        replayed = json.loads(json.dumps(pristine, sort_keys=True))
        replay_unsigned = replayed.pop("unsigned_receipt_jsonb_text")
        replay_digest = replayed.pop("receipt_sha256")
        assert hashlib.sha256(replay_unsigned.encode("utf-8")).hexdigest() == (
            replay_digest
        )
        assert json.loads(replay_unsigned) == replayed

        namespaces = {
            namespace["name"]: namespace
            for namespace in pristine["namespaces"]
        }
        assert set(namespaces) == {
            "public",
            "canonical_brain",
            "canonical_brain_legacy_quarantine",
        }
        assert namespaces["canonical_brain"]["owner"] == (
            "canonical_brain_migration_owner"
        )
        assert namespaces["canonical_brain_legacy_quarantine"]["owner"] == (
            "muncho_legacy_source"
        )
        public_acl = {
            (row["grantee"], row["privilege"])
            for row in namespaces["public"]["acl"]
        }
        assert ("canonical_brain_migration_owner", "USAGE") in public_acl
        assert not any(
            grantee in {"PUBLIC", "canonical_brain_writer", WRITER_LOGIN}
            for grantee, _privilege in public_acl
        )
        canonical_acl = {
            (row["grantee"], row["privilege"])
            for row in namespaces["canonical_brain"]["acl"]
        }
        assert ("canonical_brain_writer", "USAGE") in canonical_acl
        assert not any(grantee == "PUBLIC" for grantee, _privilege in canonical_acl)
        legacy_schema_acl = {
            (row["grantee"], row["privilege"])
            for row in namespaces["canonical_brain_legacy_quarantine"]["acl"]
        }
        assert {grantee for grantee, _privilege in legacy_schema_acl} == {
            "muncho_legacy_source"
        }

        event = pristine["event_log"]
        assert event["cardinality"] == 1
        assert re.fullmatch(r"[1-9][0-9]*", event["identity"]["oid"])
        assert event["identity"]["owner"] == "canonical_brain_migration_owner"
        assert event["identity"]["relation_kind"] == "r"
        assert event["identity"]["persistence"] == "p"
        assert event["identity"]["namespace_oid"] == namespaces["public"]["oid"]
        assert event["identity"]["is_partition"] is False
        assert event["identity"]["access_method"] == "heap"
        assert event["identity"]["tablespace_oid"] == "0"
        assert event["identity"]["row_security"] is False
        assert event["identity"]["force_row_security"] is False
        assert event["identity"]["replica_identity"] == "d"
        assert event["identity"]["options_are_empty"] is True
        assert event["identity"]["attribute_slots"] == 14
        assert event["identity"]["columns"] == _canonical14_columns()
        assert {
            row["grantee"] for row in event["identity"]["relation_acl"]
        } == {"canonical_brain_migration_owner"}
        assert len(event["identity"]["constraints"]) == 1
        event_constraint = event["identity"]["constraints"][0]
        assert event_constraint["type"] == "p"
        assert event_constraint["column_numbers"] == [1]
        assert event_constraint["column_names"] == ["event_id"]
        assert len(event["identity"]["indexes"]) == 1
        event_index = event["identity"]["indexes"][0]
        assert event_index["oid"] == event_constraint["index_oid"]
        assert event_index["constraint_oids"] == [event_constraint["oid"]]
        assert event_index["owner"] == "canonical_brain_migration_owner"
        assert event_index["access_method"] == "btree"
        assert event_index["primary"] is True
        assert event_index["unique"] is True
        assert event_index["valid"] is True
        assert event_index["ready"] is True
        assert event_index["live"] is True
        assert event_index["key_columns"] == [{
            "name": "event_id",
            "position": 1,
            "attribute_number": 1,
        }]
        for field in (
            "user_triggers",
            "rules",
            "policies",
            "inheritance",
        ):
            assert event["identity"][field] == []

        ping = pristine["writer_ping"]
        assert ping["cardinality"] == 1
        assert len(ping["routines"]) == 1
        routine = ping["routines"][0]
        assert routine["namespace_oid"] == namespaces["canonical_brain"]["oid"]
        assert routine["owner"] == "canonical_brain_migration_owner"
        assert routine["language"] == "plpgsql"
        assert routine["kind"] == "f"
        assert routine["return_type"] == {"name": "jsonb", "schema": "pg_catalog"}
        assert routine["argument_types"] == [
            {"name": "jsonb", "schema": "pg_catalog", "position": 1},
            {"name": "jsonb", "schema": "pg_catalog", "position": 2},
        ]
        assert routine["returns_set"] is False
        assert routine["security_definer"] is True
        assert routine["volatility"] == "v"
        assert routine["configuration_count"] == 1
        assert routine["configuration_is_exact"] is True
        assert re.fullmatch(r"[0-9a-f]{64}", routine["implementation_sha256"])

        archive = pristine["legacy_archive"]
        assert archive["cardinality"] == 1
        assert re.fullmatch(r"[1-9][0-9]*", archive["identity"]["oid"])
        assert archive["identity"]["owner"] == "muncho_legacy_source"
        assert archive["identity"]["relation_kind"] == "r"
        assert archive["identity"]["persistence"] == "p"
        assert archive["identity"]["owner_superuser"] is False
        assert archive["identity"]["owner_create_database"] is False
        assert archive["identity"]["owner_create_role"] is False
        assert archive["identity"]["owner_replication"] is False
        assert archive["identity"]["owner_bypass_row_security"] is False
        assert archive["identity"]["owner_connection_limit"] == -1
        assert archive["identity"]["owner_validity_is_unbounded"] is True
        assert archive["identity"]["owner_configuration_is_empty"] is True
        assert archive["identity"]["namespace_oid"] == (
            namespaces["canonical_brain_legacy_quarantine"]["oid"]
        )
        assert archive["identity"]["is_partition"] is False
        assert archive["identity"]["access_method"] == "heap"
        assert archive["identity"]["row_security"] is False
        assert archive["identity"]["force_row_security"] is False
        assert archive["identity"]["attribute_slots"] == 19
        assert len(archive["identity"]["columns"]) == 19
        assert {
            row["name"] for row in archive["identity"]["columns"]
        } == {
            "event_id", "schema_version", "event_type", "occurred_at",
            "case_id", "source", "actor", "subject", "evidence",
            "decision", "status", "next_action", "safety", "payload",
            "inserted_at", "idempotency_key", "source_spool",
            "spool_line_number", "raw_event_sha256",
        }
        archive_columns = {
            row["name"]: row for row in archive["identity"]["columns"]
        }
        assert archive_columns["source"]["has_default"] is True
        assert re.fullmatch(
            r"[0-9a-f]{64}",
            archive_columns["source"]["default_expression_sha256"],
        )
        assert archive_columns["idempotency_key"]["has_default"] is False
        assert {
            row["grantee"] for row in archive["identity"]["relation_acl"]
        } == {"muncho_legacy_source"}
        assert len(archive["identity"]["constraints"]) == 1
        assert archive["identity"]["constraints"][0]["column_names"] == [
            "event_id"
        ]
        assert len(archive["identity"]["indexes"]) == 5
        assert {
            row["key_columns"][0]["name"]
            for row in archive["identity"]["indexes"]
        } == {
            "event_id",
            "case_id",
            "event_type",
            "occurred_at",
            "idempotency_key",
        }
        assert sum(
            row["predicate_present"]
            for row in archive["identity"]["indexes"]
        ) == 1
        for field in (
            "user_triggers",
            "rules",
            "policies",
            "inheritance",
        ):
            assert archive["identity"][field] == []

        target = pristine["target_database"]
        assert target["name"] == "muncho_canary_brain"
        assert target["owner"] == "cloudsqlsuperuser"
        assert target["allow_connections"] is True
        assert target["effective_public_connect"] is False
        assert target["effective_public_temporary"] is False
        assert {
            (entry["grantor"], entry["grantee"], entry["privilege"])
            for entry in target["acl"]
        } >= {
            ("cloudsqlsuperuser", "cloudsqlsuperuser", "CONNECT"),
            ("cloudsqlsuperuser", "cloudsqlsuperuser", "CREATE"),
            ("cloudsqlsuperuser", "cloudsqlsuperuser", "TEMPORARY"),
            ("cloudsqlsuperuser", "canonical_brain_writer", "CONNECT"),
        }
        other = {
            database["name"]: database
            for database in pristine["other_connectable_databases"]
        }
        assert {"cloudsqladmin", "postgres", "template1"} <= set(other)
        assert all(
            database["effective_public_connect"] is False
            and database["effective_public_temporary"] is False
            for database in other.values()
        )
        managed = pristine["managed_cloudsqladmin"]
        assert managed["role_cardinality"] == 1
        assert managed["role"]["name"] == "cloudsqladmin"
        assert managed["role"]["superuser"] is True
        assert {
            row["database"] for row in managed["database_privileges"]
        } == {
            "cloudsqladmin",
            "muncho_canary_brain",
            "postgres",
            "template1",
        }
        assert all(
            row["effective_connect"] is True
            and row["effective_temporary"] is True
            for row in managed["database_privileges"]
        )
        assert _preflight(container) == pristine

        before_drop_pk = _preflight(container)
        _psql(
            container,
            b"""
ALTER TABLE public.canonical_event_log
    DROP CONSTRAINT canonical_event_log_pkey;
""",
        )
        dropped_pk = _preflight(container)
        assert dropped_pk["receipt_sha256"] != before_drop_pk["receipt_sha256"]
        assert dropped_pk["event_log"]["identity"]["constraints"] == []
        assert dropped_pk["event_log"]["identity"]["indexes"] == []
        _psql(
            container,
            b"""
ALTER TABLE public.canonical_event_log
    ADD CONSTRAINT canonical_event_log_pkey PRIMARY KEY (event_id);
""",
        )

        before_event_acl = _preflight(container)
        _psql(
            container,
            b"GRANT INSERT ON TABLE public.canonical_event_log TO PUBLIC;",
        )
        event_acl_drift = _preflight(container)
        assert event_acl_drift["receipt_sha256"] != (
            before_event_acl["receipt_sha256"]
        )
        assert (
            "PUBLIC",
            "INSERT",
        ) in {
            (row["grantee"], row["privilege"])
            for row in event_acl_drift["event_log"]["identity"]["relation_acl"]
        }
        _psql(
            container,
            b"REVOKE INSERT ON TABLE public.canonical_event_log FROM PUBLIC;",
        )

        before_public_acl = _preflight(container)
        _psql(container, b"GRANT CREATE ON SCHEMA public TO PUBLIC;")
        public_schema_drift = _preflight(container)
        assert public_schema_drift["receipt_sha256"] != (
            before_public_acl["receipt_sha256"]
        )
        public_schema = next(
            row for row in public_schema_drift["namespaces"]
            if row["name"] == "public"
        )
        assert ("PUBLIC", "CREATE") in {
            (row["grantee"], row["privilege"])
            for row in public_schema["acl"]
        }
        _psql(container, b"REVOKE CREATE ON SCHEMA public FROM PUBLIC;")

        before_canonical_owner = _preflight(container)
        _psql(
            container,
            b"ALTER SCHEMA canonical_brain OWNER TO muncho_legacy_source;",
        )
        canonical_owner_drift = _preflight(container)
        assert canonical_owner_drift["receipt_sha256"] != (
            before_canonical_owner["receipt_sha256"]
        )
        canonical_namespace = next(
            row for row in canonical_owner_drift["namespaces"]
            if row["name"] == "canonical_brain"
        )
        assert canonical_namespace["owner"] == "muncho_legacy_source"
        assert canonical_owner_drift["writer_ping"]["routines"][0][
            "namespace_oid"
        ] == canonical_namespace["oid"]
        _psql(
            container,
            b"""
ALTER SCHEMA canonical_brain OWNER TO canonical_brain_migration_owner;
REVOKE ALL ON SCHEMA canonical_brain FROM PUBLIC,
    canonical_brain_writer, muncho_canary_writer_login;
GRANT USAGE ON SCHEMA canonical_brain TO canonical_brain_writer;
""",
        )

        before_legacy_acl = _preflight(container)
        _psql(
            container,
            b"""
GRANT SELECT, UPDATE ON TABLE
    canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1
    TO PUBLIC;
""",
        )
        legacy_acl_drift = _preflight(container)
        assert legacy_acl_drift["receipt_sha256"] != (
            before_legacy_acl["receipt_sha256"]
        )
        assert {
            (row["grantee"], row["privilege"])
            for row in legacy_acl_drift["legacy_archive"]["identity"][
                "relation_acl"
            ]
        } >= {("PUBLIC", "SELECT"), ("PUBLIC", "UPDATE")}
        _psql(
            container,
            b"""
REVOKE SELECT, UPDATE ON TABLE
    canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1
    FROM PUBLIC;
""",
        )

        before_legacy_owner = _preflight(container)
        _psql(
            container,
            b"""
ALTER SCHEMA canonical_brain_legacy_quarantine
    OWNER TO canonical_brain_migration_owner;
""",
        )
        legacy_owner_drift = _preflight(container)
        assert legacy_owner_drift["receipt_sha256"] != (
            before_legacy_owner["receipt_sha256"]
        )
        legacy_namespace = next(
            row for row in legacy_owner_drift["namespaces"]
            if row["name"] == "canonical_brain_legacy_quarantine"
        )
        assert legacy_namespace["owner"] == "canonical_brain_migration_owner"
        assert legacy_owner_drift["legacy_archive"]["identity"]["owner"] == (
            "muncho_legacy_source"
        )
        _psql(
            container,
            b"""
ALTER SCHEMA canonical_brain_legacy_quarantine
    OWNER TO muncho_legacy_source;
REVOKE ALL ON SCHEMA canonical_brain_legacy_quarantine FROM PUBLIC,
    canonical_brain_writer, muncho_canary_writer_login;
""",
        )

        before_rls = _preflight(container)
        _psql(
            container,
            b"""
ALTER TABLE public.canonical_event_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY event_drift_policy ON public.canonical_event_log USING (true);
""",
        )
        rls_drift = _preflight(container)
        assert rls_drift["receipt_sha256"] != before_rls["receipt_sha256"]
        assert rls_drift["event_log"]["identity"]["row_security"] is True
        assert [
            row["name"]
            for row in rls_drift["event_log"]["identity"]["policies"]
        ] == ["event_drift_policy"]
        _psql(
            container,
            b"""
DROP POLICY event_drift_policy ON public.canonical_event_log;
ALTER TABLE public.canonical_event_log DISABLE ROW LEVEL SECURITY;
""",
        )

        before_trigger = _preflight(container)
        _psql(
            container,
            b"""
CREATE FUNCTION canonical_brain.event_drift_trigger()
RETURNS trigger LANGUAGE plpgsql AS $function$
BEGIN
    RETURN NEW;
END
$function$;
CREATE TRIGGER event_drift_trigger
BEFORE INSERT ON public.canonical_event_log
FOR EACH ROW EXECUTE FUNCTION canonical_brain.event_drift_trigger();
""",
        )
        trigger_drift = _preflight(container)
        assert trigger_drift["receipt_sha256"] != before_trigger["receipt_sha256"]
        assert [
            row["name"]
            for row in trigger_drift["event_log"]["identity"]["user_triggers"]
        ] == ["event_drift_trigger"]
        _psql(
            container,
            b"""
DROP TRIGGER event_drift_trigger ON public.canonical_event_log;
DROP FUNCTION canonical_brain.event_drift_trigger();
""",
        )

        before_rule = _preflight(container)
        _psql(
            container,
            b"""
CREATE RULE event_drift_rule AS
    ON INSERT TO public.canonical_event_log DO INSTEAD NOTHING;
""",
        )
        rule_drift = _preflight(container)
        assert rule_drift["receipt_sha256"] != before_rule["receipt_sha256"]
        assert [
            row["name"]
            for row in rule_drift["event_log"]["identity"]["rules"]
        ] == ["event_drift_rule"]
        _psql(
            container,
            b"DROP RULE event_drift_rule ON public.canonical_event_log;",
        )

        before_inheritance = _preflight(container)
        _psql(
            container,
            b"""
CREATE TABLE public.event_drift_parent (event_id uuid);
ALTER TABLE public.canonical_event_log INHERIT public.event_drift_parent;
""",
        )
        inheritance_drift = _preflight(container)
        assert inheritance_drift["receipt_sha256"] != (
            before_inheritance["receipt_sha256"]
        )
        assert len(
            inheritance_drift["event_log"]["identity"]["inheritance"]
        ) == 1
        assert inheritance_drift["event_log"]["identity"]["columns"][0][
            "inheritance_count"
        ] == 1
        _psql(
            container,
            b"""
ALTER TABLE public.canonical_event_log
    NO INHERIT public.event_drift_parent;
DROP TABLE public.event_drift_parent;
""",
        )

        before_legacy_index = _preflight(container)
        _psql(
            container,
            b"DROP INDEX canonical_brain_legacy_quarantine.legacy_case_id_idx;",
        )
        legacy_index_drift = _preflight(container)
        assert legacy_index_drift["receipt_sha256"] != (
            before_legacy_index["receipt_sha256"]
        )
        assert len(legacy_index_drift["legacy_archive"]["identity"]["indexes"]) == 4
        _psql(
            container,
            b"""
CREATE INDEX legacy_case_id_idx ON
    canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1 (case_id);
""",
        )

        original_event_oid = event["identity"]["oid"]
        _psql(
            container,
            b"""
ALTER TABLE public.canonical_event_log RENAME TO canonical_event_log_original;
CREATE TABLE public.canonical_event_log (
    event_id uuid NOT NULL,
    schema_version text NOT NULL,
    event_type text NOT NULL,
    occurred_at timestamptz NOT NULL,
    case_id text NOT NULL,
    source jsonb NOT NULL,
    actor jsonb NOT NULL,
    subject jsonb NOT NULL,
    evidence jsonb NOT NULL,
    decision jsonb NOT NULL,
    status jsonb NOT NULL,
    next_action jsonb NOT NULL,
    safety jsonb NOT NULL,
    payload jsonb NOT NULL
);
ALTER TABLE public.canonical_event_log OWNER TO muncho_legacy_source;
REVOKE ALL ON TABLE public.canonical_event_log FROM PUBLIC;
""",
        )
        identity_drift = _preflight(container)["event_log"]["identity"]
        assert identity_drift["oid"] != original_event_oid
        assert identity_drift["owner"] == "muncho_legacy_source"
        assert identity_drift["columns"] == _canonical14_columns()
        _psql(
            container,
            b"""
DROP TABLE public.canonical_event_log;
ALTER TABLE public.canonical_event_log_original
    RENAME TO canonical_event_log;
""",
        )

        _psql(
            container,
            b"GRANT CONNECT, TEMPORARY ON DATABASE postgres TO PUBLIC;",
            database="postgres",
        )
        public_drift = {
            database["name"]: database
            for database in _preflight(container)["other_connectable_databases"]
        }["postgres"]
        assert public_drift["effective_public_connect"] is True
        assert public_drift["effective_public_temporary"] is True
        assert {
            (entry["grantee"], entry["privilege"])
            for entry in public_drift["acl"]
        } >= {("PUBLIC", "CONNECT"), ("PUBLIC", "TEMPORARY")}
        _psql(
            container,
            b"REVOKE CONNECT, TEMPORARY ON DATABASE postgres FROM PUBLIC;",
            database="postgres",
        )

        _psql(
            container,
            f"""
CREATE ROLE {ADMIN}
    LOGIN INHERIT NOSUPERUSER CREATEDB CREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1
    PASSWORD '{ADMIN_PASSWORD}';
GRANT cloudsqlsuperuser TO {ADMIN}
    WITH ADMIN FALSE, INHERIT TRUE, SET TRUE;
CREATE ROLE canonical_brain_canary_bootstrap
    NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
GRANT CONNECT ON DATABASE muncho_canary_brain
    TO canonical_brain_canary_bootstrap;
GRANT canonical_brain_canary_bootstrap TO {ADMIN}
    WITH ADMIN TRUE, INHERIT FALSE, SET FALSE;
""".encode(),
        )
        role_recovery = _preflight(container)
        assert role_recovery["bootstrap_role_absent"] is False
        assert role_recovery["bootstrap_login_absent"] is True
        assert [row["name"] for row in role_recovery["temporary_admin_roles"]] == [
            ADMIN
        ]
        assert {
            (row["granted_role"], row["member_role"], row["grantor"])
            for row in role_recovery["memberships"]
        } >= {
            ("cloudsqlsuperuser", ADMIN, "cloudsqladmin"),
            ("canonical_brain_canary_bootstrap", ADMIN, "cloudsqladmin"),
        }

        _psql(
            container,
            b"""
CREATE ROLE canonical_brain_canary_bootstrap_login
    LOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
GRANT canonical_brain_canary_bootstrap
    TO canonical_brain_canary_bootstrap_login
    WITH ADMIN FALSE, INHERIT TRUE, SET FALSE;
""",
        )
        login_recovery = _preflight(container)
        assert login_recovery["bootstrap_role_absent"] is False
        assert login_recovery["bootstrap_login_absent"] is False
        bootstrap_roles = {
            row["name"]: row for row in login_recovery["roles"]
        }
        assert bootstrap_roles["canonical_brain_canary_bootstrap"] == {
            "oid": bootstrap_roles["canonical_brain_canary_bootstrap"]["oid"],
            "name": "canonical_brain_canary_bootstrap",
            "inherits": False,
            "can_login": False,
            "superuser": False,
            "create_role": False,
            "replication": False,
            "create_database": False,
            "connection_limit": -1,
            "bypass_row_security": False,
            "configuration_is_empty": True,
            "validity_is_unbounded": True,
        }
        persistent = next(
            row for row in login_recovery["memberships"]
            if row["granted_role"] == "canonical_brain_canary_bootstrap"
            and row["member_role"]
            == "canonical_brain_canary_bootstrap_login"
        )
        assert persistent == {
            "grantor": "cloudsqladmin",
            "member_role": "canonical_brain_canary_bootstrap_login",
            "set_option": False,
            "admin_option": False,
            "granted_role": "canonical_brain_canary_bootstrap",
            "inherit_option": True,
        }
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
