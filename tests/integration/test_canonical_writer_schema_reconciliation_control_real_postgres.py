from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid

import pytest

from gateway import (
    canonical_writer_schema_reconciliation_control_bootstrap as control_bootstrap,
)
from gateway.canonical_writer_db import QueryResult
from gateway.canonical_writer_schema_reconciliation_db import (
    _AUTHORITY_OPEN_RECEIPT_COLUMNS,
    _AUTHORITY_OPEN_RECEIPT_SQL,
    _POST_DELETE_AUTHORITY_COLUMNS,
    _post_delete_authority_absence_sql,
)


pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
INSTALL = ROOT / "scripts/sql/canonical_writer_schema_reconciliation_control_v1.sql"
RETIRE = (
    ROOT
    / "scripts/sql/canonical_writer_schema_reconciliation_control_retire_v1.sql"
)
IMAGE = "postgres:18"
DATABASE = "muncho_canary_brain"
CONTROL = "muncho_canary_control_1234567890abcdef"
RUNTIME = "muncho_canary_reconciler_1234567890abcdef"
EXECUTOR = "canonical_brain_schema_reconciler"
MARKER = "canonical-private-row-must-never-be-exported"
PLAN = "a" * 64
INTENT = "b" * 64
TRUTH = "c" * 64
HELPER_DEFINITION_SHA256 = (
    "2fc8e5334784ae791398033c8b460ccb170324f4c8cc1b3d0387f907e9cf7737"
)


def _run(
    command: list[str],
    *,
    stdin: str = "",
    accepted: frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=180,
    )
    if completed.returncode not in accepted:
        pytest.fail(
            "postgres:18 fixture command failed "
            f"({completed.returncode}): {completed.stdout[-6000:]}"
        )
    return completed


def _psql(
    container: str,
    sql: str,
    *,
    user: str = "cloudsqladmin",
    database: str = DATABASE,
    accepted: frozenset[int] = frozenset({0}),
) -> str:
    result = _run(
        [
            "docker",
            "exec",
            "-i",
            container,
            "psql",
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
        accepted=accepted,
    )
    return result.stdout


def _wait(container: str) -> None:
    consecutive = 0
    for _ in range(120):
        probe = subprocess.run(
            [
                "docker", "exec", container, "psql", "-X", "-qAt",
                "-U", "cloudsqladmin", "-d", "postgres", "-c", "SELECT 1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode == 0 and probe.stdout == b"1\n":
            consecutive += 1
            if consecutive == 2:
                return
        else:
            consecutive = 0
        time.sleep(0.1)
    pytest.fail("postgres:18 fixture did not become ready")


def _setup(container: str) -> None:
    _psql(
        container,
        f"""
CREATE ROLE postgres NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
CREATE ROLE cloudsqlsuperuser NOLOGIN NOINHERIT SUPERUSER CREATEDB CREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
CREATE ROLE canonical_brain_migration_owner
    NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
CREATE ROLE canonical_brain_writer
    NOLOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
CREATE ROLE {CONTROL}
    LOGIN INHERIT NOSUPERUSER CREATEDB CREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
GRANT cloudsqlsuperuser TO {CONTROL}
    WITH ADMIN FALSE, INHERIT TRUE, SET TRUE;
CREATE DATABASE {DATABASE} OWNER cloudsqlsuperuser;
REVOKE ALL ON DATABASE {DATABASE} FROM PUBLIC;
REVOKE ALL ON DATABASE postgres FROM PUBLIC;
REVOKE ALL ON DATABASE template1 FROM PUBLIC;
""",
        database="postgres",
    )
    _psql(
        container,
        """
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO canonical_brain_migration_owner;
CREATE TABLE public.canonical_event_log (
    event_id uuid PRIMARY KEY,
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
ALTER TABLE public.canonical_event_log
    OWNER TO canonical_brain_migration_owner;

CREATE SCHEMA canonical_brain
    AUTHORIZATION canonical_brain_migration_owner;
SET ROLE canonical_brain_migration_owner;
CREATE TABLE canonical_brain.writer_capability_consumptions (
    consume_id uuid PRIMARY KEY
);
CREATE TABLE canonical_brain.writer_capability_grants (
    approval_id text PRIMARY KEY
);
CREATE TABLE canonical_brain.writer_capability_revocation_scopes (
    scope_type text NOT NULL,
    session_key_sha256 text NOT NULL,
    capability_epoch_sha256 text NOT NULL,
    plan_id text NOT NULL,
    PRIMARY KEY (
        scope_type, session_key_sha256, capability_epoch_sha256, plan_id
    )
);
CREATE TABLE canonical_brain.writer_capability_revocations (
    approval_id text PRIMARY KEY
);
CREATE TABLE canonical_brain.writer_event_provenance (
    event_id uuid PRIMARY KEY
);
CREATE TABLE canonical_brain.writer_public_routeback_targets (
    channel_id text PRIMARY KEY
);
CREATE TABLE canonical_brain.writer_routeback_authorizations (
    authorization_id text PRIMARY KEY
);
CREATE TABLE canonical_brain.writer_routeback_lifecycle_terminals (
    lifecycle_id text PRIMARY KEY
);
CREATE TABLE canonical_brain.writer_routeback_terminals (
    authorization_id text PRIMARY KEY
);
CREATE FUNCTION canonical_brain._contains_forbidden_dm_ref(value jsonb)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
SET search_path = pg_catalog, canonical_brain
AS $function$
    SELECT false
$function$;
RESET ROLE;

CREATE SCHEMA canonical_brain_legacy_quarantine AUTHORIZATION postgres;
SET ROLE postgres;
CREATE TABLE canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1 (
    event_id uuid PRIMARY KEY
);
CREATE TABLE canonical_brain_legacy_quarantine.reconciliation_receipts (
    receipt_id uuid PRIMARY KEY
);
REVOKE ALL ON SCHEMA canonical_brain_legacy_quarantine FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA canonical_brain_legacy_quarantine
    FROM PUBLIC;
RESET ROLE;

INSERT INTO public.canonical_event_log VALUES (
    '00000000-0000-0000-0000-000000000001',
    'v1', 'case.opened', '2026-07-17T00:00:00Z', 'case:test',
    '{}'::jsonb, '{}'::jsonb, '{}'::jsonb, '[]'::jsonb, '{}'::jsonb,
    '{}'::jsonb, '{}'::jsonb, '{}'::jsonb,
    '{"private_marker":"canonical-private-row-must-never-be-exported"}'::jsonb
);
""",
    )


def _observe(container: str) -> tuple[str, ...]:
    raw = _psql(
        container,
        """
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SELECT row_count, canonical14_sha256, relation_receipts::text,
       quarantine_anchor_receipts::text, observation_sha256
  FROM canonical_brain_reconciliation.
       observe_missing_discord_routeback_helper_v1();
ROLLBACK;
""",
        user=RUNTIME,
    )
    lines = [line for line in raw.splitlines() if line]
    assert len(lines) == 1
    assert MARKER not in raw
    fields = tuple(lines[0].split("|"))
    assert len(fields) == 5
    expected = hashlib.sha256(
        "\n".join((
            "canonical-writer-schema-reconciliation-control-observation-v1",
            *fields[:4],
        )).encode("utf-8")
    ).hexdigest()
    assert fields[4] == expected
    return fields


def _apply(
    container: str,
    *,
    plan: str = PLAN,
    intent: str = INTENT,
    truth: str = TRUTH,
    accepted: frozenset[int] = frozenset({0}),
) -> str:
    return _psql(
        container,
        f"""
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SET LOCAL muncho.schema_reconciliation_plan_sha256 = '{plan}';
SET LOCAL muncho.schema_reconciliation_authorized_intent_sha256 = '{intent}';
SET LOCAL muncho.schema_reconciliation_truth_receipt_sha256 = '{truth}';
SELECT applied::text, plan_sha256, authorized_intent_sha256,
       canonical_truth_receipt_sha256, observation_sha256,
       helper_definition_sha256, receipt_sha256
  FROM canonical_brain_reconciliation.
       apply_missing_discord_routeback_helper_v1();
COMMIT;
""",
        user=RUNTIME,
        accepted=accepted,
    )


def _authority(container: str) -> dict[str, bool]:
    raw = _psql(container, _AUTHORITY_OPEN_RECEIPT_SQL, user=RUNTIME)
    lines = [line for line in raw.splitlines() if line]
    assert len(lines) == 1
    values = lines[0].split("|")
    assert len(values) == len(_AUTHORITY_OPEN_RECEIPT_COLUMNS)
    assert set(values) <= {"t", "f"}
    return {
        name: value == "t"
        for name, value in zip(
            _AUTHORITY_OPEN_RECEIPT_COLUMNS, values, strict=True
        )
    }


def _assert_authority_exact(container: str) -> None:
    receipt = _authority(container)
    assert receipt and all(receipt.values()), {
        name: exact for name, exact in receipt.items() if not exact
    }


def _authority_drift(
    container: str,
    *,
    mutation: str,
    expected_false: frozenset[str],
    restore: str,
) -> None:
    _psql(container, mutation)
    observed = _authority(container)
    assert expected_false <= {
        name for name, exact in observed.items() if not exact
    }
    _psql(container, restore)
    _assert_authority_exact(container)


def _wait_for_runtime_advisory_lock(container: str) -> None:
    for _ in range(100):
        value = _psql(
            container,
            f"""
SELECT count(*) FROM pg_locks AS lock
JOIN pg_stat_activity AS activity ON activity.pid = lock.pid
WHERE lock.locktype = 'advisory' AND lock.granted
  AND lock.mode = 'ExclusiveLock' AND activity.usename = '{RUNTIME}';
""",
        ).strip()
        if value == "1":
            return
        time.sleep(0.02)
    pytest.fail("runtime observer did not acquire the deployment lock")


class _PersistentPsqlSession:
    """One real backend so session advisory locks survive query calls."""

    def __init__(self, container: str, user: str) -> None:
        self.container = container
        self.username = user
        self._sequence = 0
        self._closed = False
        self._process = subprocess.Popen(
            [
                "docker", "exec", "-i", container, "psql", "-X", "-qAt",
                "-P", "null=__MUNCHO_NULL__", "-v", "ON_ERROR_STOP=1",
                "-U", user, "-d", DATABASE,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def _execute(self, sql: str) -> list[str]:
        assert not self._closed
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._sequence += 1
        marker = f"__MUNCHO_DONE_{self._sequence}_{uuid.uuid4().hex}__"
        statement = sql.rstrip()
        if not statement.endswith(";"):
            statement += ";"
        self._process.stdin.write(statement + "\n")
        self._process.stdin.write("\\echo " + marker + "\n")
        self._process.stdin.flush()
        lines: list[str] = []
        while True:
            line = self._process.stdout.readline()
            if line == "":
                pytest.fail(
                    "persistent PostgreSQL session exited before marker: "
                    + "\n".join(lines[-40:])
                )
            value = line.rstrip("\n")
            if value == marker:
                return lines
            lines.append(value)

    def raw(self, sql: str) -> list[str]:
        return self._execute(sql)

    def query(self, sql: str, *, maximum_rows: int) -> QueryResult:
        lines = self._execute(sql)
        if sql in {
            control_bootstrap._FOUNDATION_OBSERVATION_SET_LOCK_TIMEOUT_SQL,
            "SET lock_timeout = '15s'",
            "SET statement_timeout = '2min'",
        }:
            assert maximum_rows == 0
            return QueryResult((), (), "SET")
        if sql == control_bootstrap._FOUNDATION_OBSERVATION_RESET_LOCK_TIMEOUT_SQL:
            assert maximum_rows == 0
            return QueryResult((), (), "RESET")
        if sql == "RESET statement_timeout":
            assert maximum_rows == 0
            return QueryResult((), (), "RESET")
        if sql.startswith("SELECT pg_catalog.pg_advisory_lock_shared("):
            return QueryResult(
                ("pg_advisory_lock_shared",), (("",),), "SELECT 1"
            )
        if sql.startswith("SELECT pg_catalog.pg_advisory_unlock_shared("):
            assert lines == ["t"]
            return QueryResult(
                ("pg_advisory_unlock_shared",), (("t",),), "SELECT 1"
            )
        if sql == control_bootstrap._FOUNDATION_OBSERVATION_SQL:
            rows = [line for line in lines if line]
            assert len(rows) == 1
            values = tuple(
                None if value == "__MUNCHO_NULL__" else value
                for value in rows[0].split("|")
            )
            return QueryResult(
                control_bootstrap._FOUNDATION_OBSERVATION_COLUMNS,
                (values,),
                "COMMIT",
            )
        if sql.lstrip().startswith("WITH executor AS ("):
            rows = [line for line in lines if line]
            assert len(rows) == 1
            values = tuple(rows[0].split("|"))
            return QueryResult(
                _POST_DELETE_AUTHORITY_COLUMNS,
                (values,),
                "SELECT 1",
            )
        normalized = sql.strip().rstrip(";").upper()
        if normalized.startswith("BEGIN"):
            return QueryResult((), (), "BEGIN")
        if normalized == "COMMIT":
            return QueryResult((), (), "COMMIT")
        if normalized == "ROLLBACK":
            return QueryResult((), (), "ROLLBACK")
        raise AssertionError(f"unrecognized persistent query: {sql[:160]}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            try:
                self._process.stdin.write("\\q\n")
                self._process.stdin.flush()
                self._process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=10)


def _wait_for_advisory_waiter(container: str, user: str) -> None:
    for _ in range(200):
        waiting = _psql(
            container,
            f"""
SELECT pg_catalog.count(*)
  FROM pg_catalog.pg_stat_activity AS activity
 WHERE activity.usename = '{user}'
   AND activity.wait_event_type = 'Lock'
   AND activity.wait_event = 'advisory';
""",
        ).strip()
        if waiting == "1":
            return
        time.sleep(0.02)
    pytest.fail("persistent observer did not wait on the advisory lock")


def test_fixed_control_install_apply_and_retire_on_real_postgresql_18() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker is unavailable")
    if subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode:
        pytest.skip("postgres:18 image is unavailable")

    container = "muncho-control-pg18-" + uuid.uuid4().hex[:12]
    try:
        _run([
            "docker", "run", "-d", "--name", container,
            "-e", "POSTGRES_USER=cloudsqladmin",
            "-e", "POSTGRES_PASSWORD=fixture-only",
            IMAGE,
        ])
        _wait(container)
        _setup(container)

        install_sql = INSTALL.read_text(encoding="utf-8")
        partial_states = (
            (
                f"CREATE ROLE {EXECUTOR} NOLOGIN NOINHERIT;",
                f"SELECT to_regrole('{EXECUTOR}') IS NOT NULL, "
                "to_regnamespace('canonical_brain_reconciliation') IS NULL;",
                f"DROP ROLE {EXECUTOR};",
            ),
            (
                "CREATE SCHEMA canonical_brain_reconciliation "
                "AUTHORIZATION canonical_brain_migration_owner;",
                f"SELECT to_regrole('{EXECUTOR}') IS NULL, "
                "to_regnamespace('canonical_brain_reconciliation') IS NOT NULL;",
                "DROP SCHEMA canonical_brain_reconciliation;",
            ),
        )
        for create_sql, state_sql, cleanup_sql in partial_states:
            _psql(container, create_sql)
            partial = _psql(
                container,
                install_sql,
                user=CONTROL,
                accepted=frozenset({3}),
            )
            assert "control bootstrap preflight failed" in partial
            assert _psql(container, state_sql).strip() == "t|t"
            _psql(container, cleanup_sql)

        _psql(container, install_sql, user=CONTROL)

        creator_edge = _psql(
            container,
            f"""
SELECT granted.rolname, member.rolname, grantor.rolname,
       membership.admin_option, membership.inherit_option,
       membership.set_option
  FROM pg_catalog.pg_auth_members AS membership
  JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
  JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
  JOIN pg_catalog.pg_roles AS grantor ON grantor.oid = membership.grantor
 WHERE granted.rolname = '{EXECUTOR}';
""",
        ).strip()
        assert creator_edge == (
            f"{EXECUTOR}|{CONTROL}|cloudsqladmin|t|f|f"
        )

        # Install is absent-only; exact-installed adoption belongs to the
        # external bootstrap journal and never replays privileged SQL.
        replay = _psql(
            container,
            install_sql,
            user=CONTROL,
            accepted=frozenset({3}),
        )
        assert "control bootstrap preflight failed" in replay

        _psql(container, f"DROP ROLE {CONTROL};")
        assert _psql(
            container,
            f"SELECT count(*) FROM pg_auth_members WHERE roleid = "
            f"'{EXECUTOR}'::regrole;",
        ).strip() == "0"

        # Vanilla PostgreSQL cannot model Cloud SQL's provider-only GRANT
        # authority while keeping pg_roles/pg_auth_members byte-equivalent.
        # The SUPERUSER surrogate is confined to the sealed artifact above;
        # every product/runtime assertion below sees the live Cloud role shape.
        _psql(
            container,
            """
ALTER ROLE cloudsqlsuperuser LOGIN NOSUPERUSER CREATEDB CREATEROLE
    NOREPLICATION NOBYPASSRLS;
""",
        )

        _psql(
            container,
            f"""
CREATE ROLE {RUNTIME}
    LOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
GRANT {EXECUTOR} TO {RUNTIME}
    WITH ADMIN FALSE, INHERIT TRUE, SET TRUE;
""",
        )
        assert _psql(
            container,
            f"""
SELECT grantor.rolname, membership.admin_option,
       membership.inherit_option, membership.set_option
  FROM pg_auth_members AS membership
  JOIN pg_roles AS granted ON granted.oid = membership.roleid
  JOIN pg_roles AS member ON member.oid = membership.member
  JOIN pg_roles AS grantor ON grantor.oid = membership.grantor
 WHERE granted.rolname = '{EXECUTOR}' AND member.rolname = '{RUNTIME}';
""",
        ).strip() == "cloudsqladmin|f|t|t"
        assert _psql(
            container,
            f"SELECT has_database_privilege('{RUNTIME}', '{DATABASE}', 'TEMP');",
        ).strip() == "f"
        _assert_authority_exact(container)

        before = _observe(container)
        assert before[0] == "1"
        relations = json.loads(before[2])
        anchors = json.loads(before[3])
        assert len(relations) == 10
        assert [item["relation"] for item in relations][0] == (
            "public.canonical_event_log"
        )
        assert len(anchors) == 3
        assert all(item["owner"] == "postgres" for item in anchors)

        # Even if TEMP is adversarially added after authority preflight, every
        # load-bearing name in the owner routine remains catalog-qualified.
        _psql(
            container,
            f"GRANT TEMPORARY ON DATABASE {DATABASE} TO {RUNTIME};",
        )
        direct_acl_drift = _authority(container)
        assert direct_acl_drift[
            "temporary_executor_has_zero_shared_dependencies"
        ] is False
        shadowed = _psql(
            container,
            """
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
CREATE TEMP TABLE canonical_event_log (event_id text, payload text);
CREATE TEMP TABLE writer_capability_grants (approval_id text);
CREATE FUNCTION pg_temp.sha256(value bytea)
RETURNS bytea LANGUAGE sql IMMUTABLE
AS $function$ SELECT '\\x00'::bytea $function$;
SELECT row_count, canonical14_sha256, relation_receipts::text,
       quarantine_anchor_receipts::text, observation_sha256
  FROM canonical_brain_reconciliation.
       observe_missing_discord_routeback_helper_v1();
ROLLBACK;
""",
            user=RUNTIME,
        )
        shadow_lines = [line for line in shadowed.splitlines() if line]
        assert len(shadow_lines) == 1
        assert tuple(shadow_lines[0].split("|")) == before
        _psql(
            container,
            f"REVOKE TEMPORARY ON DATABASE {DATABASE} FROM {RUNTIME};",
        )
        _assert_authority_exact(container)

        applied = [line for line in _apply(container).splitlines() if line]
        assert len(applied) == 1
        fields = applied[0].split("|")
        assert fields[:4] == ["true", PLAN, INTENT, TRUTH]
        assert fields[4] == before[4]
        assert fields[5] == HELPER_DEFINITION_SHA256
        assert fields[6] == hashlib.sha256(
            "\n".join((
                "canonical-writer-schema-reconciliation-control-apply-v1",
                "true", PLAN, INTENT, TRUTH, before[4],
                HELPER_DEFINITION_SHA256,
            )).encode("utf-8")
        ).hexdigest()
        assert _observe(container) == before

        idempotent = [line for line in _apply(container).splitlines() if line]
        assert idempotent[0].split("|")[0] == "false"
        assert idempotent[0].split("|")[5] == HELPER_DEFINITION_SHA256

        acl = _psql(
            container,
            f"""
WITH executor AS (
    SELECT oid FROM pg_roles WHERE rolname = '{EXECUTOR}'
), dependencies AS (
    SELECT * FROM pg_shdepend, executor
     WHERE refclassid = 'pg_authid'::regclass
       AND refobjid = executor.oid
)
SELECT count(*) FILTER (WHERE deptype = 'a'),
       count(*) FILTER (WHERE deptype = 'o'),
       count(*) FILTER (WHERE deptype NOT IN ('a'))
  FROM dependencies;
""",
        ).strip()
        assert acl == "4|0|0"
        assert _psql(
            container,
            """
SELECT count(*) FROM pg_proc AS routine
CROSS JOIN LATERAL aclexplode(COALESCE(
    routine.proacl, acldefault('f', routine.proowner)
)) AS acl
WHERE routine.oid IN (
    'canonical_brain_reconciliation.observe_missing_discord_routeback_helper_v1()'::regprocedure,
    'canonical_brain_reconciliation.apply_missing_discord_routeback_helper_v1()'::regprocedure
) AND acl.grantee = 0;
""",
        ).strip() == "0"

        _authority_drift(
            container,
            mutation="""
GRANT EXECUTE ON FUNCTION canonical_brain_reconciliation.
observe_missing_discord_routeback_helper_v1() TO PUBLIC;
""",
            expected_false=frozenset({"control_routine_acl_surface_exact"}),
            restore="""
REVOKE EXECUTE ON FUNCTION canonical_brain_reconciliation.
observe_missing_discord_routeback_helper_v1() FROM PUBLIC;
""",
        )
        _authority_drift(
            container,
            mutation="""
ALTER FUNCTION canonical_brain_reconciliation.
apply_missing_discord_routeback_helper_v1() OWNER TO cloudsqladmin;
""",
            expected_false=frozenset({
                "executor_routine_acl_exact",
                "control_routine_attributes_exact",
                "control_routine_acl_surface_exact",
            }),
            restore="""
ALTER FUNCTION canonical_brain_reconciliation.
apply_missing_discord_routeback_helper_v1()
    OWNER TO canonical_brain_migration_owner;
SET ROLE canonical_brain_migration_owner;
REVOKE ALL PRIVILEGES ON FUNCTION canonical_brain_reconciliation.
apply_missing_discord_routeback_helper_v1() FROM PUBLIC;
REVOKE ALL PRIVILEGES ON FUNCTION canonical_brain_reconciliation.
apply_missing_discord_routeback_helper_v1()
    FROM canonical_brain_schema_reconciler;
GRANT EXECUTE ON FUNCTION canonical_brain_reconciliation.
apply_missing_discord_routeback_helper_v1()
    TO canonical_brain_schema_reconciler;
RESET ROLE;
""",
        )
        _authority_drift(
            container,
            mutation="""
ALTER FUNCTION canonical_brain_reconciliation.
observe_missing_discord_routeback_helper_v1()
    SET search_path = public;
""",
            expected_false=frozenset({"control_routine_attributes_exact"}),
            restore="""
ALTER FUNCTION canonical_brain_reconciliation.
observe_missing_discord_routeback_helper_v1()
    SET search_path = pg_catalog, pg_temp;
""",
        )
        _authority_drift(
            container,
            mutation="""
UPDATE pg_catalog.pg_proc AS routine
   SET prosrc = routine.prosrc || E'\\n-- adversarial-control-prosrc-drift'
  FROM pg_catalog.pg_namespace AS namespace
 WHERE namespace.oid = routine.pronamespace
   AND namespace.nspname = 'canonical_brain_reconciliation'
   AND routine.proname = 'observe_missing_discord_routeback_helper_v1'
   AND routine.pronargs = 0;
""",
            expected_false=frozenset({"control_routine_attributes_exact"}),
            restore="""
UPDATE pg_catalog.pg_proc AS routine
   SET prosrc = pg_catalog.left(
       routine.prosrc,
       pg_catalog.length(routine.prosrc)
       - pg_catalog.length(E'\\n-- adversarial-control-prosrc-drift')
   )
  FROM pg_catalog.pg_namespace AS namespace
 WHERE namespace.oid = routine.pronamespace
   AND namespace.nspname = 'canonical_brain_reconciliation'
   AND routine.proname = 'observe_missing_discord_routeback_helper_v1'
   AND routine.pronargs = 0
   AND pg_catalog.right(
       routine.prosrc,
       pg_catalog.length(E'\\n-- adversarial-control-prosrc-drift')
   ) = E'\\n-- adversarial-control-prosrc-drift';
""",
        )
        _authority_drift(
            container,
            mutation="""
SET ROLE canonical_brain_migration_owner;
CREATE FUNCTION canonical_brain_reconciliation.
observe_missing_discord_routeback_helper_v1(value text)
RETURNS text LANGUAGE sql IMMUTABLE
SET search_path = pg_catalog, pg_temp
AS $function$ SELECT value $function$;
RESET ROLE;
""",
            expected_false=frozenset({
                "control_routine_attributes_exact",
                "control_inventory_and_ownership_exact",
                "control_routine_acl_surface_exact",
            }),
            restore="""
SET ROLE canonical_brain_migration_owner;
DROP FUNCTION canonical_brain_reconciliation.
observe_missing_discord_routeback_helper_v1(text);
RESET ROLE;
""",
        )
        _authority_drift(
            container,
            mutation=f"GRANT CREATE ON SCHEMA canonical_brain_reconciliation "
                     f"TO {EXECUTOR};",
            expected_false=frozenset({
                "executor_schema_acl_exact",
                "control_schema_acl_surface_exact",
            }),
            restore=f"REVOKE CREATE ON SCHEMA "
                    f"canonical_brain_reconciliation FROM {EXECUTOR};",
        )
        _authority_drift(
            container,
            mutation=f"ALTER ROLE {EXECUTOR} INHERIT;",
            expected_false=frozenset({"executor_role_attributes_exact"}),
            restore=f"ALTER ROLE {EXECUTOR} NOINHERIT;",
        )
        _authority_drift(
            container,
            mutation=f"""
CREATE ROLE adversarial_privilege_bridge NOLOGIN NOINHERIT;
GRANT canonical_brain_migration_owner TO adversarial_privilege_bridge;
GRANT adversarial_privilege_bridge TO {EXECUTOR};
""",
            expected_false=frozenset({
                "provider_executor_edge_exact",
                "recursive_authority_closure_exact",
                "privileged_roles_unreachable",
            }),
            restore=f"""
REVOKE adversarial_privilege_bridge FROM {EXECUTOR};
REVOKE canonical_brain_migration_owner
    FROM adversarial_privilege_bridge;
DROP ROLE adversarial_privilege_bridge;
""",
        )
        _authority_drift(
            container,
            mutation=f"""
REVOKE {EXECUTOR} FROM {RUNTIME};
GRANT {EXECUTOR} TO {RUNTIME}
    WITH ADMIN FALSE, INHERIT TRUE, SET FALSE;
""",
            expected_false=frozenset({"provider_executor_edge_exact"}),
            restore=f"""
REVOKE {EXECUTOR} FROM {RUNTIME};
GRANT {EXECUTOR} TO {RUNTIME}
    WITH ADMIN FALSE, INHERIT TRUE, SET TRUE;
""",
        )
        assert _observe(container) == before

        # A present-but-wrong helper is never overwritten.
        _psql(
            container,
            """
CREATE OR REPLACE FUNCTION
canonical_brain._discord_guild_routeback_target_valid(value jsonb)
RETURNS boolean LANGUAGE sql IMMUTABLE
SET search_path = pg_catalog, canonical_brain
AS $function$ SELECT false $function$;
""",
        )
        drift = _apply(container, accepted=frozenset({3}))
        assert "schema reconciliation target helper drifted" in drift
        assert _observe(container) == before
        _psql(
            container,
            "DROP FUNCTION "
            "canonical_brain._discord_guild_routeback_target_valid(jsonb);",
        )
        alternate = [
            line
            for line in _apply(
                container,
                plan="d" * 64,
                intent="e" * 64,
                truth="f" * 64,
            ).splitlines()
            if line
        ][0].split("|")
        assert alternate[:4] == ["true", "d" * 64, "e" * 64, "f" * 64]
        assert alternate[5] == HELPER_DEFINITION_SHA256
        assert _observe(container) == before

        # Binding values never select a routine, object, or DDL.  Invalid
        # values fail before mutation; different valid values create the same
        # release-pinned helper bytes.
        _psql(
            container,
            "DROP FUNCTION "
            "canonical_brain._discord_guild_routeback_target_valid(jsonb);",
        )
        invalid_binding = _apply(
            container,
            plan="g" * 64,
            accepted=frozenset({3}),
        )
        assert "schema reconciliation fixed apply binding failed" in (
            invalid_binding
        )
        assert _psql(
            container,
            "SELECT count(*) FROM pg_proc AS routine "
            "JOIN pg_namespace AS namespace ON namespace.oid = "
            "routine.pronamespace WHERE namespace.nspname = "
            "'canonical_brain' AND routine.proname = "
            "'_discord_guild_routeback_target_valid';",
        ).strip() == "0"
        assert _observe(container) == before

        holder = subprocess.Popen(
            [
                "docker", "exec", "-i", container, "psql", "-X", "-qAt",
                "-v", "ON_ERROR_STOP=1", "-U", RUNTIME, "-d", DATABASE,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert holder.stdin is not None
        holder.stdin.write("""
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SELECT observation_sha256
  FROM canonical_brain_reconciliation.
       observe_missing_discord_routeback_helper_v1();
SELECT pg_sleep(1.25);
ROLLBACK;
""")
        holder.stdin.close()
        _wait_for_runtime_advisory_lock(container)
        started = time.monotonic()
        concurrent = [
            line
            for line in _apply(
                container,
                plan="1" * 64,
                intent="2" * 64,
                truth="3" * 64,
            ).splitlines()
            if line
        ][0].split("|")
        elapsed = time.monotonic() - started
        assert concurrent[0] == "true"
        assert concurrent[5] == HELPER_DEFINITION_SHA256
        assert elapsed >= 0.8
        assert holder.wait(timeout=30) == 0
        assert holder.stdout is not None
        holder_output = holder.stdout.read()
        assert before[4] in holder_output
        assert _observe(container) == before

        _psql(container, f"DROP ROLE {RUNTIME};")
        # Re-enable the explicitly scoped provider-authority surrogate only
        # for sealed retirement, then remove it again after the artifact.
        _psql(
            container,
            """
ALTER ROLE cloudsqlsuperuser NOLOGIN SUPERUSER CREATEDB CREATEROLE
    NOREPLICATION NOBYPASSRLS;
""",
        )
        _psql(
            container,
            f"""
CREATE ROLE {CONTROL}
    LOGIN INHERIT NOSUPERUSER CREATEDB CREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
GRANT cloudsqlsuperuser TO {CONTROL}
    WITH ADMIN FALSE, INHERIT TRUE, SET TRUE;
CREATE ROLE rogue_reconciler NOLOGIN;
GRANT {EXECUTOR} TO rogue_reconciler
    WITH ADMIN FALSE, INHERIT TRUE, SET FALSE;
""",
        )
        member_drift = _psql(
            container,
            RETIRE.read_text(encoding="utf-8"),
            user=CONTROL,
            accepted=frozenset({3}),
        )
        assert "control retire preflight failed" in member_drift
        _psql(
            container,
            f"REVOKE {EXECUTOR} FROM rogue_reconciler; DROP ROLE rogue_reconciler;",
        )

        _psql(
            container,
            f"CREATE DATABASE executor_owned_drift OWNER {EXECUTOR};",
            database="postgres",
        )
        ownership_drift = _psql(
            container,
            RETIRE.read_text(encoding="utf-8"),
            user=CONTROL,
            accepted=frozenset({3}),
        )
        assert "control retire preflight failed" in ownership_drift
        _psql(
            container,
            "DROP DATABASE executor_owned_drift;",
            database="postgres",
        )

        _psql(container, RETIRE.read_text(encoding="utf-8"), user=CONTROL)
        _psql(
            container,
            """
ALTER ROLE cloudsqlsuperuser LOGIN NOSUPERUSER CREATEDB CREATEROLE
    NOREPLICATION NOBYPASSRLS;
""",
        )
        terminal = _psql(
            container,
            f"""
SELECT to_regrole('{EXECUTOR}') IS NULL,
       to_regnamespace('canonical_brain_reconciliation') IS NULL,
       encode(sha256(convert_to(pg_get_functiondef(
           'canonical_brain._discord_guild_routeback_target_valid(jsonb)'::regprocedure
       ), 'UTF8')), 'hex');
""",
        ).strip()
        assert terminal == f"t|t|{HELPER_DEFINITION_SHA256}"
        _psql(container, f"DROP ROLE {CONTROL};")
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def test_control_bootstrap_observation_and_post_delete_lock_are_real_pg18() -> None:
    """Exercise the Python observation ABI and its pre-snapshot session lock."""

    if shutil.which("docker") is None:
        pytest.skip("docker is unavailable")
    if subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode:
        pytest.skip("postgres:18 image is unavailable")

    container = "muncho-control-observe-pg18-" + uuid.uuid4().hex[:10]
    writer = control_bootstrap.foundation.SQL_USER
    temporary_login = "muncho_canary_reconciler_deadbeefdeadbeef"

    def set_provider_live() -> None:
        _psql(
            container,
            """
ALTER ROLE cloudsqlsuperuser LOGIN NOSUPERUSER CREATEDB CREATEROLE
    NOREPLICATION NOBYPASSRLS;
""",
        )

    def set_provider_surrogate() -> None:
        _psql(
            container,
            """
ALTER ROLE cloudsqlsuperuser NOLOGIN SUPERUSER CREATEDB CREATEROLE
    NOREPLICATION NOBYPASSRLS;
""",
        )

    def observe(user: str, phase: str) -> dict[str, object]:
        session = _PersistentPsqlSession(container, user)
        try:
            return dict(control_bootstrap._observe_foundation(
                session,
                phase=phase,
                observed_at_unix=lambda: 1_000,
            ))
        finally:
            session.close()

    def reject_observation(user: str, phase: str = "post_cleanup") -> None:
        session = _PersistentPsqlSession(container, user)
        try:
            with pytest.raises(
                control_bootstrap.ControlBootstrapError,
                match=(
                    "schema_reconciliation_control_"
                    "database_observation_invalid"
                ),
            ):
                control_bootstrap._observe_foundation(
                    session,
                    phase=phase,
                    observed_at_unix=lambda: 1_000,
                )
        finally:
            session.close()

    try:
        _run([
            "docker", "run", "-d", "--name", container,
            "-e", "POSTGRES_USER=cloudsqladmin",
            "-e", "POSTGRES_PASSWORD=fixture-only",
            IMAGE,
        ])
        _wait(container)
        _setup(container)
        assert _psql(
            container,
            "SELECT current_setting('max_prepared_transactions');",
        ).strip() == "0"
        _psql(
            container,
            f"""
CREATE ROLE {writer}
    LOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
GRANT canonical_brain_writer TO {writer}
    WITH ADMIN FALSE, INHERIT TRUE, SET FALSE;
GRANT CONNECT ON DATABASE {DATABASE} TO canonical_brain_writer;
GRANT USAGE ON SCHEMA canonical_brain TO canonical_brain_writer;
""",
        )

        # The real Cloud role shape is mandatory for every product-path
        # observation.  The fixed artifact later gets a tightly scoped
        # vanilla-PG surrogate because Cloud's provider-only GRANT authority
        # has no catalog-equivalent outside Cloud SQL.
        set_provider_live()
        absent = observe(CONTROL, "before_install")
        assert absent["state"] == "absent"
        assert absent["control_admin_count"] == 1
        assert absent["max_prepared_transactions"] == 0
        assert absent["non_template_database_inventory_exact"] is True
        assert absent["all_connectable_database_inventory_exact"] is True
        assert absent["latent_provider_exception_exact"] is True

        set_provider_surrogate()
        _psql(
            container,
            INSTALL.read_text(encoding="utf-8"),
            user=CONTROL,
        )
        set_provider_live()
        installed = observe(CONTROL, "after_install")
        assert installed["state"] == "exact_installed"
        assert installed["executor_membership_count"] == 1
        assert installed["control_admin_forward_role_count"] == 2

        _psql(container, f"DROP ROLE {CONTROL};")
        post_cleanup = observe(writer, "post_cleanup")
        assert post_cleanup["state"] == "exact_installed"
        assert post_cleanup["control_admin_count"] == 0
        assert post_cleanup["executor_membership_count"] == 0

        # Exact-installed adoption has no replay and no creator edge.
        _psql(
            container,
            f"""
CREATE ROLE {CONTROL}
    LOGIN INHERIT NOSUPERUSER CREATEDB CREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
GRANT cloudsqlsuperuser TO {CONTROL}
    WITH ADMIN FALSE, INHERIT TRUE, SET TRUE;
""",
        )
        adopted = observe(CONTROL, "before_install")
        assert adopted["state"] == "exact_installed"
        assert adopted["executor_membership_count"] == 0
        assert adopted["control_admin_forward_role_count"] == 1
        _psql(container, f"DROP ROLE {CONTROL};")

        # Same-name overload, wrong fourth dependency, grantor-only role edge,
        # template privilege, owner attrs, and database owner all fail closed.
        _psql(
            container,
            """
SET ROLE canonical_brain_migration_owner;
CREATE FUNCTION canonical_brain_reconciliation.
observe_missing_discord_routeback_helper_v1(value text)
RETURNS text LANGUAGE sql IMMUTABLE
SET search_path = pg_catalog, pg_temp
AS $function$ SELECT value $function$;
RESET ROLE;
""",
        )
        reject_observation(writer)
        _psql(
            container,
            """
SET ROLE canonical_brain_migration_owner;
DROP FUNCTION canonical_brain_reconciliation.
observe_missing_discord_routeback_helper_v1(text);
RESET ROLE;
""",
        )

        _psql(
            container,
            f"""
REVOKE CONNECT ON DATABASE {DATABASE} FROM {EXECUTOR};
SET ROLE canonical_brain_migration_owner;
GRANT USAGE ON SCHEMA canonical_brain TO {EXECUTOR};
RESET ROLE;
""",
        )
        reject_observation(writer)
        _psql(
            container,
            f"""
SET ROLE canonical_brain_migration_owner;
REVOKE USAGE ON SCHEMA canonical_brain FROM {EXECUTOR};
RESET ROLE;
GRANT CONNECT ON DATABASE {DATABASE} TO {EXECUTOR};
""",
        )

        _psql(
            container,
            f"""
CREATE ROLE adversarial_parent NOLOGIN;
CREATE ROLE adversarial_member NOLOGIN;
GRANT adversarial_parent TO adversarial_member;
UPDATE pg_catalog.pg_auth_members
   SET grantor = '{EXECUTOR}'::regrole
 WHERE roleid = 'adversarial_parent'::regrole
   AND member = 'adversarial_member'::regrole;
""",
        )
        reject_observation(writer)
        _psql(
            container,
            """
UPDATE pg_catalog.pg_auth_members
   SET grantor = 'cloudsqladmin'::regrole
 WHERE roleid = 'adversarial_parent'::regrole
   AND member = 'adversarial_member'::regrole;
REVOKE adversarial_parent FROM adversarial_member;
DROP ROLE adversarial_parent;
DROP ROLE adversarial_member;
""",
        )

        _psql(container, "GRANT CONNECT ON DATABASE template1 TO PUBLIC;")
        reject_observation(writer)
        _psql(container, "REVOKE CONNECT ON DATABASE template1 FROM PUBLIC;")

        _psql(container, "ALTER ROLE canonical_brain_migration_owner LOGIN;")
        reject_observation(writer)
        _psql(container, "ALTER ROLE canonical_brain_migration_owner NOLOGIN;")

        _psql(
            container,
            f"ALTER DATABASE {DATABASE} OWNER TO cloudsqladmin;",
        )
        reject_observation(writer)
        _psql(
            container,
            f"ALTER DATABASE {DATABASE} OWNER TO cloudsqlsuperuser;",
        )
        assert observe(writer, "post_cleanup")["state"] == "exact_installed"

        # A committed overload is installed while holding the exclusive
        # session lock.  The Python observer must wait, then take a fresh
        # SERIALIZABLE snapshot and reject it; a pre-wait snapshot would have
        # incorrectly returned exact_installed.
        holder = _PersistentPsqlSession(container, "cloudsqladmin")
        holder.raw(f"""
BEGIN;
SELECT pg_catalog.pg_advisory_lock(
    {control_bootstrap.CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY}
);
SET ROLE canonical_brain_migration_owner;
CREATE FUNCTION canonical_brain_reconciliation.
observe_missing_discord_routeback_helper_v1(value integer)
RETURNS integer LANGUAGE sql IMMUTABLE
SET search_path = pg_catalog, pg_temp
AS $function$ SELECT value $function$;
RESET ROLE;
COMMIT;
""")
        concurrency: dict[str, object] = {}

        def run_control_observer() -> None:
            started = time.monotonic()
            try:
                observe(writer, "post_cleanup")
            except BaseException as exc:  # recorded for the main test thread
                concurrency["error"] = exc
            concurrency["elapsed"] = time.monotonic() - started

        thread = threading.Thread(target=run_control_observer, daemon=True)
        thread.start()
        _wait_for_advisory_waiter(container, writer)
        assert thread.is_alive()
        holder.close()
        thread.join(timeout=20)
        assert not thread.is_alive()
        assert isinstance(
            concurrency.get("error"),
            control_bootstrap.ControlBootstrapError,
        )
        assert float(concurrency["elapsed"]) > 0
        _psql(
            container,
            """
SET ROLE canonical_brain_migration_owner;
DROP FUNCTION canonical_brain_reconciliation.
observe_missing_discord_routeback_helper_v1(integer);
RESET ROLE;
""",
        )

        # Apply the fixed helper through the inert executor, then prove the
        # normal post-delete authority SQL under the live role shape.
        _psql(
            container,
            f"""
CREATE ROLE {RUNTIME}
    LOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
GRANT {EXECUTOR} TO {RUNTIME}
    WITH ADMIN FALSE, INHERIT TRUE, SET TRUE;
""",
        )
        applied = [line for line in _apply(container).splitlines() if line]
        assert applied and applied[0].split("|")[0] == "true"
        _psql(container, f"DROP ROLE {RUNTIME};")

        authority_sql = _post_delete_authority_absence_sql(temporary_login)
        authority_raw = _psql(container, authority_sql, user=writer)
        authority_lines = [line for line in authority_raw.splitlines() if line]
        assert len(authority_lines) == 1
        assert authority_lines[0].split("|") == [
            "t"
        ] * len(_POST_DELETE_AUTHORITY_COLUMNS)

        # Repeat the stale-snapshot race directly around the normal
        # _post_delete_authority_absence_sql: holder commits role drift while
        # observer waits; the post-lock receipt must contain the drift.
        holder = _PersistentPsqlSession(container, "cloudsqladmin")
        holder.raw(f"""
BEGIN;
SELECT pg_catalog.pg_advisory_lock(
    {control_bootstrap.CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY}
);
ALTER ROLE {EXECUTOR} INHERIT;
COMMIT;
""")
        normal_result: dict[str, object] = {}

        def run_post_delete_authority() -> None:
            session = _PersistentPsqlSession(container, writer)
            try:
                session.query("SET lock_timeout = '15s'", maximum_rows=0)
                session.query("SET statement_timeout = '2min'", maximum_rows=0)
                session.query(
                    "SELECT pg_catalog.pg_advisory_lock_shared("
                    + str(control_bootstrap.CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY)
                    + ")",
                    maximum_rows=1,
                )
                session.query("RESET lock_timeout", maximum_rows=0)
                session.query("RESET statement_timeout", maximum_rows=0)
                session.query(
                    "BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY",
                    maximum_rows=0,
                )
                receipt = session.query(authority_sql, maximum_rows=1)
                session.query("COMMIT", maximum_rows=0)
                session.query(
                    "SELECT pg_catalog.pg_advisory_unlock_shared("
                    + str(control_bootstrap.CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY)
                    + ")",
                    maximum_rows=1,
                )
                normal_result["receipt"] = receipt
            except BaseException as exc:
                normal_result["error"] = exc
            finally:
                session.close()

        normal_thread = threading.Thread(
            target=run_post_delete_authority,
            daemon=True,
        )
        normal_thread.start()
        _wait_for_advisory_waiter(container, writer)
        assert normal_thread.is_alive()
        holder.close()
        normal_thread.join(timeout=20)
        assert not normal_thread.is_alive()
        assert "error" not in normal_result
        result = normal_result["receipt"]
        assert isinstance(result, QueryResult)
        flags = dict(zip(result.columns, result.rows[0], strict=True))
        assert flags["persistent_executor_role_attributes_exact"] == "f"
        assert set(flags.values()) <= {"t", "f"}
        _psql(container, f"ALTER ROLE {EXECUTOR} NOINHERIT;")
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
