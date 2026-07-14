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
    ROOT / "scripts" / "sql" / "canonical_writer_foundation_phase_b_role_v1.sql"
)
ADMIN = "muncho_canary_admin_1234567890abcdef"
RECOVERY_ADMIN = "muncho_canary_admin_fedcba0987654321"
DRIFT_ADMIN = "muncho_canary_admin_1111111111111111"
ROOT_PASSWORD = "fixture-only-cloudsqladmin-password"
ADMIN_PASSWORD = "fixture-only-phase-b-admin-password"
REVISION = "a" * 40
OBSERVATION_SHA256 = "c" * 64
APPROVED_PLAN_SHA256 = "d" * 64


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
        for secret in (ROOT_PASSWORD.encode(), ADMIN_PASSWORD.encode()):
            sanitized = sanitized.replace(secret, b"<redacted>")
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
    accepted: frozenset[int] = frozenset({0}),
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
        accepted=accepted,
    )


def _admin_sql(name: str) -> bytes:
    return f"""
CREATE ROLE {name}
    LOGIN INHERIT NOSUPERUSER CREATEDB CREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1
    PASSWORD '{ADMIN_PASSWORD}';
GRANT cloudsqlsuperuser TO {name}
    WITH ADMIN FALSE, INHERIT TRUE, SET TRUE;
""".encode()


def _artifact_sql() -> bytes:
    raw = ARTIFACT_PATH.read_bytes()
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    settings = (
        "SET muncho.canonical_writer_phase_b_release_revision = "
        f"'{REVISION}';\n"
        "SET muncho.canonical_writer_phase_b_role_artifact_sha256 = "
        f"'{artifact_sha256}';\n"
        "SET muncho.canonical_writer_phase_b_initial_observation_sha256 = "
        f"'{OBSERVATION_SHA256}';\n"
        "SET muncho.canonical_writer_phase_b_approved_plan_sha256 = "
        f"'{APPROVED_PLAN_SHA256}';\n"
    ).encode()
    return settings + raw


def _receipt(raw: bytes) -> dict[str, object]:
    lines = [line for line in raw.decode("utf-8").splitlines() if line]
    assert len(lines) == 1
    value = json.loads(lines[0])
    assert isinstance(value, dict)
    digest = value.pop("receipt_sha256")
    rendered = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    assert digest == hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    value["receipt_sha256"] = digest
    return value


def _identity(container: str) -> bytes:
    return _psql(
        container,
        b"""
SELECT role.oid::text,
       role.rolcanlogin, role.rolinherit, role.rolsuper,
       role.rolcreatedb, role.rolcreaterole, role.rolreplication,
       role.rolbypassrls, role.rolconnlimit,
       role.rolvaliduntil IS NULL, role.rolconfig IS NULL
  FROM pg_catalog.pg_roles AS role
 WHERE role.rolname = 'canonical_brain_canary_bootstrap';
SELECT granted.rolname, member.rolname, grantor.rolname,
       membership.admin_option, membership.inherit_option,
       membership.set_option
  FROM pg_catalog.pg_auth_members AS membership
  JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
  JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
  JOIN pg_catalog.pg_roles AS grantor ON grantor.oid = membership.grantor
 WHERE granted.rolname = 'canonical_brain_canary_bootstrap'
    OR member.rolname = 'canonical_brain_canary_bootstrap'
 ORDER BY granted.rolname, member.rolname;
SELECT pg_catalog.pg_get_userbyid(acl.grantor), acl.privilege_type,
       acl.is_grantable
  FROM pg_catalog.pg_database AS database
  CROSS JOIN LATERAL pg_catalog.aclexplode(database.datacl) AS acl
 WHERE database.datname = 'muncho_canary_brain'
   AND acl.grantee = (
        SELECT oid FROM pg_catalog.pg_roles
         WHERE rolname = 'canonical_brain_canary_bootstrap'
   );
""",
    )


def test_phase_b_role_is_preterminal_crash_replayable_and_provider_retired():
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

    container = "cw-phase-b-role-" + uuid.uuid4().hex[:12]
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
                break
            time.sleep(0.1)
        else:
            pytest.fail("postgres:18 fixture did not become ready")

        _psql(
            container,
            b"""
CREATE ROLE cloudsqlsuperuser
    NOLOGIN CREATEROLE CREATEDB NOSUPERUSER
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
CREATE ROLE canonical_brain_writer
    NOLOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
REVOKE CONNECT, TEMPORARY ON DATABASE postgres FROM PUBLIC;
REVOKE CONNECT, TEMPORARY ON DATABASE template1 FROM PUBLIC;
CREATE DATABASE muncho_canary_brain OWNER cloudsqlsuperuser;
""",
            database="postgres",
        )
        _psql(
            container,
            b"""
REVOKE ALL PRIVILEGES ON DATABASE muncho_canary_brain FROM PUBLIC;
SET ROLE cloudsqlsuperuser;
GRANT CONNECT ON DATABASE muncho_canary_brain TO canonical_brain_writer;
CREATE SCHEMA canonical_brain_legacy_quarantine;
CREATE TABLE canonical_brain_legacy_quarantine.legacy_sentinel (
    sequence_no integer PRIMARY KEY,
    payload text NOT NULL
);
INSERT INTO canonical_brain_legacy_quarantine.legacy_sentinel
VALUES (1, 'legacy-preserved');
RESET ROLE;
""",
        )
        legacy_before = _psql(
            container,
            b"""
SELECT class.oid::text, pg_catalog.pg_get_userbyid(class.relowner),
       pg_catalog.count(row.*), pg_catalog.md5(pg_catalog.string_agg(
           row.sequence_no::text || ':' || row.payload, ',' ORDER BY row.sequence_no
       ))
  FROM pg_catalog.pg_class AS class
  JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = class.relnamespace
  CROSS JOIN canonical_brain_legacy_quarantine.legacy_sentinel AS row
 WHERE namespace.nspname = 'canonical_brain_legacy_quarantine'
   AND class.relname = 'legacy_sentinel'
 GROUP BY class.oid, class.relowner;
""",
        )
        _psql(container, _admin_sql(ADMIN))

        created = _receipt(_psql(
            container,
            _artifact_sql(),
            user=ADMIN,
            password=ADMIN_PASSWORD,
        ))
        assert created["schema"] == (
            "muncho-canonical-writer-foundation-phase-b-role-preterminal.v1"
        )
        assert created["preterminal"] is True
        assert created["role_outcome"] == "created"
        assert created["temporary_admin_delete_required"] is True
        assert created["secret_material_recorded"] is False
        assert created["approved_plan_sha256"] == APPROVED_PLAN_SHA256
        assert "safe_to_start" not in created
        assert created["temporary_auto_membership"] == {
            "granted_role": "canonical_brain_canary_bootstrap",
            "member_role": ADMIN,
            "grantor": "cloudsqladmin",
            "admin_option": True,
            "inherit_option": False,
            "set_option": False,
        }
        created_identity = _identity(container)
        assert re.fullmatch(
            rb"[1-9][0-9]*\|f\|f\|f\|f\|f\|f\|f\|-1\|t\|t\n"
            rb"canonical_brain_canary_bootstrap\|"
            + ADMIN.encode()
            + rb"\|cloudsqladmin\|t\|f\|f\n"
            rb"cloudsqlsuperuser\|CONNECT\|f\n",
            created_identity,
        )

        replay = _receipt(_psql(
            container,
            _artifact_sql(),
            user=ADMIN,
            password=ADMIN_PASSWORD,
        ))
        assert replay["role_outcome"] == "adopted_same_admin_predelete"
        assert replay["temporary_auto_membership"] == (
            created["temporary_auto_membership"]
        )
        assert _identity(container) == created_identity

        _psql(
            container,
            f"DROP ROLE {ADMIN};\n".encode(),
        )
        after_delete = _identity(container)
        assert re.fullmatch(
            rb"[1-9][0-9]*\|f\|f\|f\|f\|f\|f\|f\|-1\|t\|t\n"
            rb"cloudsqlsuperuser\|CONNECT\|f\n",
            after_delete,
        )
        assert _psql(
            container,
            b"""
SELECT pg_catalog.has_database_privilege(
           'canonical_brain_canary_bootstrap',
           'muncho_canary_brain', 'CONNECT'
       ), pg_catalog.has_database_privilege(
           'canonical_brain_writer',
           'muncho_canary_brain', 'CONNECT'
       ), pg_catalog.count(*)
  FROM pg_catalog.pg_roles
 WHERE rolname ~ '^muncho_canary_admin_[0-9a-f]{16}$';
""",
        ) == b"t|t|0\n"

        _psql(container, _admin_sql(RECOVERY_ADMIN))
        postdelete = _receipt(_psql(
            container,
            _artifact_sql(),
            user=RECOVERY_ADMIN,
            password=ADMIN_PASSWORD,
        ))
        assert postdelete["role_outcome"] == "adopted_zero_membership"
        assert postdelete["temporary_auto_membership"] is None
        _psql(
            container,
            f"DROP OWNED BY {RECOVERY_ADMIN};\nDROP ROLE {RECOVERY_ADMIN};\n".encode(),
        )
        assert _identity(container) == after_delete
        assert _psql(
            container,
            b"""
SELECT class.oid::text, pg_catalog.pg_get_userbyid(class.relowner),
       pg_catalog.count(row.*), pg_catalog.md5(pg_catalog.string_agg(
           row.sequence_no::text || ':' || row.payload, ',' ORDER BY row.sequence_no
       ))
  FROM pg_catalog.pg_class AS class
  JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = class.relnamespace
  CROSS JOIN canonical_brain_legacy_quarantine.legacy_sentinel AS row
 WHERE namespace.nspname = 'canonical_brain_legacy_quarantine'
   AND class.relname = 'legacy_sentinel'
 GROUP BY class.oid, class.relowner;
""",
        ) == legacy_before

        _psql(
            container,
            b"""
SET ROLE cloudsqlsuperuser;
REVOKE CONNECT ON DATABASE muncho_canary_brain
    FROM canonical_brain_canary_bootstrap;
RESET ROLE;
""" + _admin_sql(DRIFT_ADMIN),
        )
        failed = _psql(
            container,
            _artifact_sql(),
            user=DRIFT_ADMIN,
            password=ADMIN_PASSWORD,
            accepted=frozenset({3}),
        )
        assert b"phase-b existing bootstrap database ACL is drifted" in failed
        assert _psql(
            container,
            b"""
SELECT pg_catalog.count(*)
  FROM pg_catalog.pg_auth_members AS membership
  JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
 WHERE granted.rolname = 'canonical_brain_canary_bootstrap';
SELECT pg_catalog.has_database_privilege(
    'canonical_brain_canary_bootstrap', 'muncho_canary_brain', 'CONNECT'
);
""",
        ) == b"0\nf\n"
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
