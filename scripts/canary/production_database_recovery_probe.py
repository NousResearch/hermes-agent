#!/usr/bin/env python3
"""Release-bound, fixed private PostgreSQL recovery probe.

The module has two and only two commands.  ``preflight`` proves the sealed
stopped release, the dedicated canary VM/network, and the fixed psql binary
without accepting input.  ``probe`` emits that same proof as a short-lived
gate, accepts one bounded binary frame, and runs one hard-coded read-only SQL
transaction against the provider-read-back scratch private address.

The PostgreSQL password is never accepted through argv or an environment
variable.  It exists only in the received mutable frame and an anonymous
``memfd`` used as ``PGPASSFILE``; both are overwritten before return.  psql
stderr is discarded and stdout is read with a hard byte bound.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import ipaddress
import json
import os
import re
import selectors
import signal
import ssl
import stat
import struct
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, NoReturn, Sequence

from gateway import canonical_writer_production_cutover as cutover
from gateway import canonical_canary_host_identity as host_identity
from scripts.canary import writer_release


PREFLIGHT_SCHEMA = "muncho-production-database-recovery-probe-preflight.v1"
GATE_SCHEMA = "muncho-production-database-recovery-probe-gate.v1"
FRAME_SCHEMA = "muncho-production-database-recovery-secret-frame.v1"
FAILURE_SCHEMA = "muncho-production-database-recovery-probe-failure.v1"
FRAME_MAGIC = b"MRP1"
MAX_METADATA_BYTES = 192 * 1024
MAX_PASSWORD_BYTES = 4 * 1024
MAX_CA_BYTES = 128 * 1024
MAX_PSQL_OUTPUT_BYTES = 1024 * 1024
GATE_LIFETIME_SECONDS = 180

PSQL = Path("/usr/lib/postgresql/15/bin/psql")
EXPECTED_PSQL_SHA256 = "f79699639336f0ed369158e9e4085929d943427e25393725ae28f1f4d95690c4"
EXPECTED_NETWORK = "muncho-canary-vpc"
EXPECTED_SUBNETWORK = "muncho-canary-europe-west3"
EXPECTED_PRIVATE_IP = "10.90.0.2"
TLS_MODE = "verify-ca"
DATABASE_USER = "postgres"
DATABASE_PORT = "5432"

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCRATCH = re.compile(r"^muncho-recovery-[0-9a-f]{20}$")
_RFC1918 = tuple(
    ipaddress.ip_network(value)
    for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
_NETWORK_METADATA_PATHS = {
    "network": "/computeMetadata/v1/instance/network-interfaces/0/network",
    "subnetwork": "/computeMetadata/v1/instance/network-interfaces/0/subnetwork",
    "private_ip": "/computeMetadata/v1/instance/network-interfaces/0/ip",
}

# psql emits exactly three tuple-only lines: read-only state, bounded schema
# identity JSON, and bounded content identity JSON.  No payload values leave
# PostgreSQL.  md5() is used only to bound opaque event-envelope samples; the
# receipt itself hashes the complete bounded results with SHA-256 locally.
_READ_ONLY_SQL = r"""
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL lock_timeout = '5000ms';
SET LOCAL statement_timeout = '60000ms';
SELECT pg_catalog.current_setting('transaction_read_only');
WITH schema_rows AS (
  SELECT n.nspname AS schema_name,
         c.relname AS relation_name,
         c.relkind::text AS relation_kind,
         a.attnum AS ordinal_position,
         a.attname AS column_name,
         pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
         a.attnotnull AS not_null
    FROM pg_catalog.pg_namespace AS n
    JOIN pg_catalog.pg_class AS c ON c.relnamespace = n.oid
    JOIN pg_catalog.pg_attribute AS a ON a.attrelid = c.oid
   WHERE n.nspname IN ('canonical_brain', 'public')
     AND c.relkind IN ('r', 'p', 'v', 'm', 'S')
     AND a.attnum > 0
     AND NOT a.attisdropped
   ORDER BY n.nspname, c.relname, a.attnum
   LIMIT 1025
)
SELECT pg_catalog.json_build_object(
         'columns', COALESCE(pg_catalog.json_agg(schema_rows), '[]'::json),
         'column_count', pg_catalog.count(*)
       )::text
  FROM schema_rows;
WITH event_stats AS (
  SELECT pg_catalog.count(*)::bigint AS row_count,
         pg_catalog.min(event_id)::text AS minimum_event_id,
         pg_catalog.max(event_id)::text AS maximum_event_id,
         pg_catalog.min(occurred_at)::text AS minimum_occurred_at,
         pg_catalog.max(occurred_at)::text AS maximum_occurred_at
    FROM public.canonical_event_log
), head_rows AS (
  SELECT event_id::text AS event_id,
         pg_catalog.md5(ROW(
           schema_version, event_type, occurred_at, case_id, source, actor,
           subject, evidence, decision, status, next_action, safety, payload
         )::text) AS envelope_md5
    FROM public.canonical_event_log
   ORDER BY event_id
   LIMIT 64
), tail_rows AS (
  SELECT event_id::text AS event_id,
         pg_catalog.md5(ROW(
           schema_version, event_type, occurred_at, case_id, source, actor,
           subject, evidence, decision, status, next_action, safety, payload
         )::text) AS envelope_md5
    FROM public.canonical_event_log
   ORDER BY event_id DESC
   LIMIT 64
)
SELECT pg_catalog.json_build_object(
         'row_count', event_stats.row_count,
         'minimum_event_id', event_stats.minimum_event_id,
         'maximum_event_id', event_stats.maximum_event_id,
         'minimum_occurred_at', event_stats.minimum_occurred_at,
         'maximum_occurred_at', event_stats.maximum_occurred_at,
         'head', COALESCE((SELECT pg_catalog.json_agg(head_rows ORDER BY event_id)
                            FROM head_rows), '[]'::json),
         'tail', COALESCE((SELECT pg_catalog.json_agg(tail_rows ORDER BY event_id)
                            FROM tail_rows), '[]'::json)
       )::text
  FROM event_stats;
ROLLBACK;
""".strip()

_PSQL_ARGV = (
    str(PSQL),
    "-X",
    "--no-psqlrc",
    "--quiet",
    "--no-align",
    "--tuples-only",
    "--set=ON_ERROR_STOP=1",
    f"--command={_READ_ONLY_SQL}",
)


class RecoveryProbeError(RuntimeError):
    """Stable, secret-free remote probe failure."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RecoveryProbeError("recovery_probe_json_invalid") from exc


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha(_canonical(value))


def _is_private_ipv4(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return address.version == 4 and any(address in network for network in _RFC1918)


def _zeroize(value: bytearray | memoryview | None) -> None:
    if value is None:
        return
    try:
        view = value if isinstance(value, memoryview) else memoryview(value)
        view.cast("B")[:] = b"\x00" * view.nbytes
        if not isinstance(value, memoryview):
            view.release()
    except (TypeError, ValueError, BufferError):
        pass


def _revision_from_interpreter(executable: str | None = None) -> str:
    raw = os.path.normpath(executable or sys.executable)
    match = re.fullmatch(
        r"/opt/muncho-canary-releases/([0-9a-f]{40})/venv/bin/python(?:3(?:\.11)?)?",
        raw,
    )
    if match is None:
        raise RecoveryProbeError("recovery_probe_release_interpreter_invalid")
    return match.group(1)


def _root_executable_sha256(path: Path) -> str:
    if path != PSQL:
        raise RecoveryProbeError("recovery_probe_psql_identity_invalid")
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RecoveryProbeError("recovery_probe_psql_identity_invalid") from exc
    if (
        not path.is_absolute()
        or resolved != PSQL
        or not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != 0
        or before.st_gid != 0
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o022
        or not stat.S_IMODE(before.st_mode) & 0o111
        or not 0 < before.st_size <= 256 * 1024 * 1024
    ):
        raise RecoveryProbeError("recovery_probe_psql_identity_invalid")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reachable = path.lstat()
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(opened) or identity(before) != identity(after) or identity(before) != identity(reachable):
        raise RecoveryProbeError("recovery_probe_psql_identity_changed")
    observed_digest = digest.hexdigest()
    if observed_digest != EXPECTED_PSQL_SHA256:
        raise RecoveryProbeError("recovery_probe_psql_identity_invalid")
    return observed_digest


def _read_network_metadata_value(path: str) -> bytes:
    if path not in _NETWORK_METADATA_PATHS.values():
        raise RecoveryProbeError("recovery_probe_network_identity_unavailable")
    connection = http.client.HTTPConnection("169.254.169.254", 80, timeout=1.0)
    try:
        connection.request(
            "GET",
            path,
            headers={
                "Host": "metadata.google.internal",
                "Metadata-Flavor": "Google",
            },
        )
        response = connection.getresponse()
        raw = response.read(1025)
        if (
            response.status != 200
            or response.getheader("Metadata-Flavor") != "Google"
            or len(raw) > 1024
        ):
            raise RecoveryProbeError("recovery_probe_network_identity_unavailable")
        return raw
    except (OSError, http.client.HTTPException) as exc:
        raise RecoveryProbeError("recovery_probe_network_identity_unavailable") from exc
    finally:
        connection.close()


def _network_identity(
    reader: Callable[[str], bytes | str] | None = None,
) -> Mapping[str, str]:
    read = reader or _read_network_metadata_value
    values: dict[str, str] = {}
    for name, path in _NETWORK_METADATA_PATHS.items():
        # The canonical host collector intentionally allow-lists only its own
        # leaves, so use its bounded text validator with this module's fixed
        # network leaves and the same metadata transport.
        try:
            raw = read(path)
            values[name] = host_identity._bounded_identity_text(
                raw, label=f"GCE metadata {name}"
            )
        except Exception as exc:
            raise RecoveryProbeError("recovery_probe_network_identity_unavailable") from exc
    expected_network_suffix = f"/networks/{EXPECTED_NETWORK}"
    expected_subnetwork_suffix = f"/subnetworks/{EXPECTED_SUBNETWORK}"
    if (
        not values["network"].endswith(expected_network_suffix)
        or not values["subnetwork"].endswith(expected_subnetwork_suffix)
        or values["private_ip"] != EXPECTED_PRIVATE_IP
    ):
        raise RecoveryProbeError("recovery_probe_network_identity_invalid")
    try:
        address = ipaddress.ip_address(values["private_ip"])
    except ValueError as exc:
        raise RecoveryProbeError("recovery_probe_network_identity_invalid") from exc
    if not _is_private_ipv4(address):
        raise RecoveryProbeError("recovery_probe_network_identity_invalid")
    return {
        "network": EXPECTED_NETWORK,
        "subnetwork": EXPECTED_SUBNETWORK,
        "private_ip": EXPECTED_PRIVATE_IP,
        "network_identity_sha256": _sha_json(values),
    }


def _stopped_release_identity(revision: str) -> Mapping[str, str]:
    if _REVISION.fullmatch(revision) is None:
        raise RecoveryProbeError("recovery_probe_release_invalid")
    try:
        release = writer_release._validate_completed_release(
            writer_release._stopped_release_spec(revision)
        )
        states = writer_release._collect_service_states()
    except Exception as exc:
        raise RecoveryProbeError("recovery_probe_stopped_release_invalid") from exc
    if (
        len(states) != len(writer_release._STOPPED_SERVICE_UNITS)
        or any(item.get("state") not in {"absent", "disabled_inactive"} for item in states)
    ):
        raise RecoveryProbeError("recovery_probe_stopped_release_invalid")
    manifest_sha = release.get("release_manifest_file_sha256")
    if not isinstance(manifest_sha, str) or _SHA256.fullmatch(manifest_sha) is None:
        raise RecoveryProbeError("recovery_probe_stopped_release_invalid")
    return {
        "release_manifest_file_sha256": manifest_sha,
        "stopped_units_sha256": _sha_json(states),
    }


def collect_preflight(
    revision: str,
    *,
    host_observer: Callable[[], Mapping[str, str]] | None = None,
    network_reader: Callable[[str], bytes | str] | None = None,
    release_observer: Callable[[str], Mapping[str, str]] | None = None,
    psql_hasher: Callable[[Path], str] = _root_executable_sha256,
    descriptor_factory: Callable[[str], int] | None = None,
) -> Mapping[str, Any]:
    if _REVISION.fullmatch(revision or "") is None:
        raise RecoveryProbeError("recovery_probe_release_invalid")
    try:
        host = dict((host_observer or host_identity._observe_dedicated_canary_host)())
    except Exception as exc:
        raise RecoveryProbeError("recovery_probe_host_identity_invalid") from exc
    if (
        host.get("project_id") != host_identity.DEDICATED_CANARY_PROJECT_ID
        or host.get("zone") != host_identity.DEDICATED_CANARY_ZONE
        or host.get("instance_name") != host_identity.DEDICATED_CANARY_INSTANCE_NAME
        or host.get("instance_id") != host_identity.DEDICATED_CANARY_INSTANCE_ID
        or _SHA256.fullmatch(str(host.get("host_identity_sha256"))) is None
    ):
        raise RecoveryProbeError("recovery_probe_host_identity_invalid")
    network = _network_identity(network_reader)
    release = dict((release_observer or _stopped_release_identity)(revision))
    psql_sha = psql_hasher(PSQL)
    if (
        any(_SHA256.fullmatch(str(value)) is None for value in release.values())
        or psql_sha != EXPECTED_PSQL_SHA256
    ):
        raise RecoveryProbeError("recovery_probe_runtime_identity_invalid")
    descriptor: int | None = None
    try:
        descriptor = (descriptor_factory or _anonymous_descriptor)(
            "muncho-recovery-preflight"
        )
        observed_descriptor = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed_descriptor.st_mode)
            or observed_descriptor.st_nlink != 0
            or stat.S_IMODE(observed_descriptor.st_mode) != 0o600
        ):
            raise RecoveryProbeError(
                "recovery_probe_anonymous_secret_invalid"
            )
    except (OSError, RecoveryProbeError) as exc:
        raise RecoveryProbeError(
            "recovery_probe_anonymous_secret_unavailable"
        ) from exc
    finally:
        if descriptor is not None:
            _wipe_descriptor(descriptor)
    unsigned = {
        "schema": PREFLIGHT_SCHEMA,
        "ok": True,
        "release_revision": revision,
        "canary_instance_id": host_identity.DEDICATED_CANARY_INSTANCE_ID,
        "canary_host_identity_sha256": host["host_identity_sha256"],
        "canary_network": network["network"],
        "canary_subnetwork": network["subnetwork"],
        "canary_private_ip": network["private_ip"],
        "network_identity_sha256": network["network_identity_sha256"],
        "release_manifest_file_sha256": release["release_manifest_file_sha256"],
        "stopped_units_sha256": release["stopped_units_sha256"],
        "psql_executable": str(PSQL),
        "psql_executable_sha256": psql_sha,
        "database": cutover.DATABASE,
        "probe_contract_sha256": _sha_json(cutover.DATABASE_RECOVERY_PROBE_CONTRACT),
        "accepts_caller_target": False,
        "accepts_caller_sql": False,
        "accepts_caller_command": False,
    }
    return {**unsigned, "preflight_sha256": _sha_json(unsigned)}


def build_gate(preflight: Mapping[str, Any], *, now_unix: int | None = None) -> Mapping[str, Any]:
    now = int(time.time()) if now_unix is None else now_unix
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("schema") != PREFLIGHT_SCHEMA
        or type(now) is not int
        or now <= 0
    ):
        raise RecoveryProbeError("recovery_probe_preflight_invalid")
    unsigned = {
        "schema": GATE_SCHEMA,
        "ok": True,
        "release_revision": preflight["release_revision"],
        "preflight_sha256": preflight["preflight_sha256"],
        "challenge": os.urandom(32).hex(),
        "issued_at_unix": now,
        "expires_at_unix": now + GATE_LIFETIME_SECONDS,
    }
    return {**unsigned, "gate_sha256": _sha_json(unsigned)}


def _read_exact_mutable(stream: BinaryIO, size: int) -> bytearray:
    if type(size) is not int or size < 0:
        raise RecoveryProbeError("recovery_probe_frame_invalid")
    result = bytearray(size)
    view = memoryview(result)
    offset = 0
    try:
        while offset < size:
            count = stream.readinto(view[offset:])
            if not count:
                raise RecoveryProbeError("recovery_probe_frame_truncated")
            offset += count
    except BaseException:
        _zeroize(result)
        raise
    finally:
        view.release()
    return result


def _read_frame(stream: BinaryIO) -> tuple[Mapping[str, Any], bytearray]:
    header = _read_exact_mutable(stream, 12)
    metadata_raw: bytearray | None = None
    password: bytearray | None = None
    try:
        magic, metadata_size, password_size = struct.unpack(">4sII", header)
        if (
            magic != FRAME_MAGIC
            or not 2 <= metadata_size <= MAX_METADATA_BYTES
            or not 1 <= password_size <= MAX_PASSWORD_BYTES
        ):
            raise RecoveryProbeError("recovery_probe_frame_invalid")
        metadata_raw = _read_exact_mutable(stream, metadata_size)
        password = _read_exact_mutable(stream, password_size)
        if stream.read(1) != b"":
            raise RecoveryProbeError("recovery_probe_frame_trailing_data")
        try:
            metadata = json.loads(bytes(metadata_raw).decode("ascii", errors="strict"))
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise RecoveryProbeError("recovery_probe_frame_invalid") from exc
        if not isinstance(metadata, Mapping) or bytes(metadata_raw) != _canonical(metadata):
            raise RecoveryProbeError("recovery_probe_frame_invalid")
        return dict(metadata), password
    except BaseException:
        _zeroize(password)
        raise
    finally:
        _zeroize(header)
        _zeroize(metadata_raw)


def _validate_frame_metadata(
    metadata: Mapping[str, Any],
    *,
    revision: str,
    gate: Mapping[str, Any],
    now_unix: int,
) -> tuple[str, str]:
    fields = {
        "schema", "release_revision", "gate_sha256", "scratch_instance",
        "scratch_private_ip", "server_ca_pem", "server_ca_sha256", "tls_mode",
        "secret_resource", "secret_version",
    }
    private_ip = metadata.get("scratch_private_ip")
    ca_pem = metadata.get("server_ca_pem")
    try:
        address = ipaddress.ip_address(str(private_ip))
        ca_raw = str(ca_pem).encode("ascii", errors="strict")
        ssl.PEM_cert_to_DER_cert(str(ca_pem))
    except (ValueError, UnicodeError, ssl.SSLError) as exc:
        raise RecoveryProbeError("recovery_probe_frame_invalid") from exc
    if (
        set(metadata) != fields
        or metadata.get("schema") != FRAME_SCHEMA
        or metadata.get("release_revision") != revision
        or metadata.get("gate_sha256") != gate.get("gate_sha256")
        or metadata.get("scratch_instance") != cutover.database_recovery_scratch_instance(revision)
        or not _is_private_ipv4(address)
        or not 1 <= len(ca_raw) <= MAX_CA_BYTES
        or metadata.get("server_ca_sha256") != _sha(ca_raw)
        or metadata.get("tls_mode") != TLS_MODE
        or metadata.get("secret_resource") != "projects/adventico-ai-platform/secrets/ai-platform-db-password"
        or not isinstance(metadata.get("secret_version"), str)
        or re.fullmatch(r"[1-9][0-9]{0,18}", metadata["secret_version"]) is None
        or type(now_unix) is not int
        or not gate["issued_at_unix"] <= now_unix < gate["expires_at_unix"]
    ):
        raise RecoveryProbeError("recovery_probe_frame_invalid")
    return str(private_ip), str(ca_pem)


def _anonymous_descriptor(label: str) -> int:
    creator = getattr(os, "memfd_create", None)
    if not callable(creator) or not sys.platform.startswith("linux"):
        raise RecoveryProbeError("recovery_probe_anonymous_secret_unavailable")
    try:
        descriptor = creator(label, flags=getattr(os, "MFD_CLOEXEC", 0))
        os.fchmod(descriptor, 0o600)
        observed = os.fstat(descriptor)
    except OSError as exc:
        raise RecoveryProbeError("recovery_probe_anonymous_secret_unavailable") from exc
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 0
        or stat.S_IMODE(observed.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise RecoveryProbeError("recovery_probe_anonymous_secret_invalid")
    return descriptor


def _write_all(descriptor: int, value: bytearray | bytes) -> None:
    view = memoryview(value)
    offset = 0
    try:
        while offset < view.nbytes:
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short anonymous descriptor write")
            offset += written
        os.lseek(descriptor, 0, os.SEEK_SET)
    finally:
        view.release()


def _wipe_descriptor(descriptor: int) -> None:
    try:
        size = os.fstat(descriptor).st_size
        os.lseek(descriptor, 0, os.SEEK_SET)
        zeros = bytearray(min(size, 64 * 1024))
        remaining = size
        while remaining:
            chunk = min(remaining, len(zeros))
            _write_all_at_current(descriptor, memoryview(zeros)[:chunk])
            remaining -= chunk
        _zeroize(zeros)
        os.ftruncate(descriptor, 0)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_all_at_current(descriptor: int, view: memoryview) -> None:
    offset = 0
    try:
        while offset < view.nbytes:
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short descriptor overwrite")
            offset += written
    finally:
        view.release()


def _pgpass(private_ip: str, password: bytearray) -> bytearray:
    prefix = f"{private_ip}:{DATABASE_PORT}:{cutover.DATABASE}:{DATABASE_USER}:".encode("ascii")
    result = bytearray(prefix)
    for value in password:
        if value in {ord(":"), ord("\\")}:
            result.append(ord("\\"))
        if value in {0, 10, 13}:
            _zeroize(result)
            raise RecoveryProbeError("recovery_probe_password_shape_invalid")
        result.append(value)
    result.append(10)
    return result


def _bounded_psql(
    argv: Sequence[str],
    environment: Mapping[str, str],
    pass_fds: Sequence[int],
) -> bytes:
    descriptors = tuple(pass_fds)
    fields = {
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PGCONNECT_TIMEOUT",
        "PGDATABASE",
        "PGHOSTADDR",
        "PGOPTIONS",
        "PGPASSFILE",
        "PGPORT",
        "PGSSLMODE",
        "PGSSLROOTCERT",
        "PGUSER",
    }
    try:
        address = ipaddress.ip_address(str(environment.get("PGHOSTADDR")))
    except ValueError as exc:
        raise RecoveryProbeError("recovery_probe_psql_contract_invalid") from exc
    if (
        tuple(argv) != _PSQL_ARGV
        or len(descriptors) != 2
        or any(type(descriptor) is not int or descriptor <= 2 for descriptor in descriptors)
        or descriptors[0] == descriptors[1]
        or set(environment) != fields
        or any(type(value) is not str for value in environment.values())
        or environment.get("HOME") != "/nonexistent"
        or environment.get("LANG") != "C.UTF-8"
        or environment.get("LC_ALL") != "C.UTF-8"
        or environment.get("PATH") != "/usr/bin:/bin"
        or environment.get("PGCONNECT_TIMEOUT") != "15"
        or environment.get("PGDATABASE") != cutover.DATABASE
        or not _is_private_ipv4(address)
        or environment.get("PGOPTIONS")
        != "-c default_transaction_read_only=on -c statement_timeout=60000 -c lock_timeout=5000"
        or environment.get("PGPASSFILE") != f"/proc/self/fd/{descriptors[0]}"
        or environment.get("PGPORT") != DATABASE_PORT
        or environment.get("PGSSLMODE") != TLS_MODE
        or environment.get("PGSSLROOTCERT") != f"/proc/self/fd/{descriptors[1]}"
        or environment.get("PGUSER") != DATABASE_USER
    ):
        raise RecoveryProbeError("recovery_probe_psql_contract_invalid")
    try:
        process = subprocess.Popen(
            tuple(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=dict(environment),
            pass_fds=descriptors,
            shell=False,
            start_new_session=True,
            bufsize=0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RecoveryProbeError("recovery_probe_psql_unavailable") from exc
    if process.stdout is None:
        raise RecoveryProbeError("recovery_probe_psql_unavailable")
    output = bytearray()
    selector = selectors.DefaultSelector()
    deadline = time.monotonic() + 90.0
    try:
        selector.register(process.stdout.fileno(), selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RecoveryProbeError("recovery_probe_psql_timeout")
            if not selector.select(min(remaining, 1.0)):
                if process.poll() is not None:
                    break
                continue
            chunk = os.read(process.stdout.fileno(), 64 * 1024)
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > MAX_PSQL_OUTPUT_BYTES:
                raise RecoveryProbeError("recovery_probe_psql_output_oversized")
        try:
            returncode = process.wait(max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise RecoveryProbeError("recovery_probe_psql_timeout") from exc
        if returncode != 0:
            raise RecoveryProbeError("recovery_probe_psql_failed")
        return bytes(output)
    except BaseException:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            try:
                process.wait(5.0)
            except subprocess.TimeoutExpired:
                pass
        raise
    finally:
        selector.close()
        _zeroize(output)


PsqlExecutor = Callable[[Sequence[str], Mapping[str, str], Sequence[int]], bytes]
DescriptorFactory = Callable[[str], int]

_SCHEMA_ROW_FIELDS = frozenset(
    {
        "schema_name",
        "relation_name",
        "relation_kind",
        "ordinal_position",
        "column_name",
        "data_type",
        "not_null",
    }
)
_SAMPLE_ROW_FIELDS = frozenset({"event_id", "envelope_md5"})
_REQUIRED_EVENT_COLUMNS = frozenset(
    {
        "event_id",
        "schema_version",
        "event_type",
        "occurred_at",
        "case_id",
        "source",
        "actor",
        "subject",
        "evidence",
        "decision",
        "status",
        "next_action",
        "safety",
        "payload",
    }
)
_MD5 = re.compile(r"^[0-9a-f]{32}$")


def _canonical_uuid(value: Any) -> uuid.UUID:
    if type(value) is not str:
        raise RecoveryProbeError("recovery_probe_psql_output_invalid")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise RecoveryProbeError("recovery_probe_psql_output_invalid") from exc
    if str(parsed) != value:
        raise RecoveryProbeError("recovery_probe_psql_output_invalid")
    return parsed


def _timestamp(value: Any) -> datetime:
    if type(value) is not str or not value or len(value) > 64:
        raise RecoveryProbeError("recovery_probe_psql_output_invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RecoveryProbeError("recovery_probe_psql_output_invalid") from exc
    if parsed.tzinfo is None:
        raise RecoveryProbeError("recovery_probe_psql_output_invalid")
    return parsed


def _validate_schema_rows(schema: Mapping[str, Any]) -> None:
    rows = schema["columns"]
    seen_ordinals: set[tuple[str, str, int]] = set()
    seen_columns: set[tuple[str, str, str]] = set()
    event_columns: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _SCHEMA_ROW_FIELDS:
            raise RecoveryProbeError("recovery_probe_psql_output_invalid")
        schema_name = row.get("schema_name")
        relation_name = row.get("relation_name")
        relation_kind = row.get("relation_kind")
        ordinal = row.get("ordinal_position")
        column_name = row.get("column_name")
        data_type = row.get("data_type")
        not_null = row.get("not_null")
        if (
            type(schema_name) is not str
            or schema_name not in {"canonical_brain", "public"}
            or type(relation_name) is not str
            or not relation_name
            or len(relation_name) > 63
            or type(relation_kind) is not str
            or relation_kind not in {"r", "p", "v", "m", "S"}
            or type(ordinal) is not int
            or ordinal <= 0
            or type(column_name) is not str
            or not column_name
            or len(column_name) > 63
            or type(data_type) is not str
            or not data_type
            or len(data_type) > 256
            or type(not_null) is not bool
        ):
            raise RecoveryProbeError("recovery_probe_psql_output_invalid")
        ordinal_identity = (schema_name, relation_name, ordinal)
        column_identity = (schema_name, relation_name, column_name)
        if ordinal_identity in seen_ordinals or column_identity in seen_columns:
            raise RecoveryProbeError("recovery_probe_psql_output_invalid")
        seen_ordinals.add(ordinal_identity)
        seen_columns.add(column_identity)
        if schema_name == "public" and relation_name == "canonical_event_log":
            event_columns.add(column_name)
    if not _REQUIRED_EVENT_COLUMNS.issubset(event_columns):
        raise RecoveryProbeError("recovery_probe_psql_output_invalid")


def _validate_sample_rows(rows: list[Any], expected_length: int) -> list[uuid.UUID]:
    if len(rows) != expected_length:
        raise RecoveryProbeError("recovery_probe_psql_output_invalid")
    event_ids: list[uuid.UUID] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _SAMPLE_ROW_FIELDS:
            raise RecoveryProbeError("recovery_probe_psql_output_invalid")
        event_id = _canonical_uuid(row.get("event_id"))
        envelope_md5 = row.get("envelope_md5")
        if type(envelope_md5) is not str or _MD5.fullmatch(envelope_md5) is None:
            raise RecoveryProbeError("recovery_probe_psql_output_invalid")
        event_ids.append(event_id)
    if len(set(event_ids)) != len(event_ids) or event_ids != sorted(event_ids):
        raise RecoveryProbeError("recovery_probe_psql_output_invalid")
    return event_ids


def _parse_psql_output(raw: bytes) -> tuple[str, str, int]:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_PSQL_OUTPUT_BYTES:
        raise RecoveryProbeError("recovery_probe_psql_output_invalid")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise RecoveryProbeError("recovery_probe_psql_output_invalid") from exc
    lines = text.splitlines()
    if len(lines) != 3 or lines[0] != "on" or any(not line for line in lines):
        raise RecoveryProbeError("recovery_probe_psql_output_invalid")
    try:
        schema = json.loads(lines[1])
        content = json.loads(lines[2])
    except (ValueError, json.JSONDecodeError) as exc:
        raise RecoveryProbeError("recovery_probe_psql_output_invalid") from exc
    if (
        not isinstance(schema, Mapping)
        or set(schema) != {"columns", "column_count"}
        or not isinstance(schema.get("columns"), list)
        or type(schema.get("column_count")) is not int
        or schema["column_count"] != len(schema["columns"])
        or not 1 <= schema["column_count"] <= 1024
        or not isinstance(content, Mapping)
        or set(content) != {
            "row_count", "minimum_event_id", "maximum_event_id",
            "minimum_occurred_at", "maximum_occurred_at", "head", "tail",
        }
        or type(content.get("row_count")) is not int
        or content["row_count"] < 0
        or not isinstance(content.get("head"), list)
        or not isinstance(content.get("tail"), list)
        or len(content["head"]) > 64
        or len(content["tail"]) > 64
    ):
        raise RecoveryProbeError("recovery_probe_psql_output_invalid")
    _validate_schema_rows(schema)
    row_count = int(content["row_count"])
    sample_count = min(row_count, 64)
    head_ids = _validate_sample_rows(content["head"], sample_count)
    tail_ids = _validate_sample_rows(content["tail"], sample_count)
    extrema = (
        content["minimum_event_id"],
        content["maximum_event_id"],
        content["minimum_occurred_at"],
        content["maximum_occurred_at"],
    )
    if row_count == 0:
        if any(value is not None for value in extrema) or head_ids or tail_ids:
            raise RecoveryProbeError("recovery_probe_psql_output_invalid")
    else:
        minimum_event_id = _canonical_uuid(content["minimum_event_id"])
        maximum_event_id = _canonical_uuid(content["maximum_event_id"])
        minimum_occurred_at = _timestamp(content["minimum_occurred_at"])
        maximum_occurred_at = _timestamp(content["maximum_occurred_at"])
        if (
            minimum_event_id > maximum_event_id
            or minimum_occurred_at > maximum_occurred_at
            or head_ids[0] < minimum_event_id
            or head_ids[-1] > maximum_event_id
            or tail_ids[0] < minimum_event_id
            or tail_ids[-1] > maximum_event_id
        ):
            raise RecoveryProbeError("recovery_probe_psql_output_invalid")
    return _sha_json(schema), _sha_json(content), int(content["row_count"])


def run_probe_frame(
    stream: BinaryIO,
    *,
    revision: str,
    preflight: Mapping[str, Any],
    gate: Mapping[str, Any],
    clock: Callable[[], float] = time.time,
    psql_executor: PsqlExecutor = _bounded_psql,
    descriptor_factory: DescriptorFactory = _anonymous_descriptor,
) -> Mapping[str, Any]:
    metadata: Mapping[str, Any] | None = None
    password: bytearray | None = None
    pgpass: bytearray | None = None
    password_fd: int | None = None
    ca_fd: int | None = None
    try:
        metadata, password = _read_frame(stream)
        now = int(clock())
        private_ip, ca_pem = _validate_frame_metadata(
            metadata, revision=revision, gate=gate, now_unix=now
        )
        pgpass = _pgpass(private_ip, password)
        password_fd = descriptor_factory("muncho-recovery-pgpass")
        ca_fd = descriptor_factory("muncho-recovery-server-ca")
        _write_all(password_fd, pgpass)
        _write_all(ca_fd, ca_pem.encode("ascii", errors="strict"))
        environment = {
            "HOME": "/nonexistent",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PGCONNECT_TIMEOUT": "15",
            "PGDATABASE": cutover.DATABASE,
            "PGHOSTADDR": private_ip,
            "PGOPTIONS": "-c default_transaction_read_only=on -c statement_timeout=60000 -c lock_timeout=5000",
            "PGPASSFILE": f"/proc/self/fd/{password_fd}",
            "PGPORT": DATABASE_PORT,
            "PGSSLMODE": TLS_MODE,
            "PGSSLROOTCERT": f"/proc/self/fd/{ca_fd}",
            "PGUSER": DATABASE_USER,
        }
        output = psql_executor(_PSQL_ARGV, environment, (password_fd, ca_fd))
        schema_sha, content_sha, row_count = _parse_psql_output(output)
        unsigned = {
            "schema": cutover.DATABASE_RECOVERY_PROBE_RECEIPT_SCHEMA,
            "ok": True,
            "release_revision": revision,
            "scratch_instance": metadata["scratch_instance"],
            "database": cutover.DATABASE,
            "probe_contract_sha256": preflight["probe_contract_sha256"],
            "transaction_read_only": True,
            "schema_sha256": schema_sha,
            "content_sha256": content_sha,
            "canonical_event_row_count": row_count,
            "scratch_private_ip": private_ip,
            "server_ca_sha256": metadata["server_ca_sha256"],
            "tls_mode": TLS_MODE,
            "tls_ca_verified": True,
            "tls_hostname_verified": False,
            "canary_instance_id": preflight["canary_instance_id"],
            "canary_network": preflight["canary_network"],
            "canary_subnetwork": preflight["canary_subnetwork"],
            "canary_private_ip": preflight["canary_private_ip"],
            "release_manifest_file_sha256": preflight["release_manifest_file_sha256"],
            "stopped_units_sha256": preflight["stopped_units_sha256"],
            "psql_executable_sha256": preflight["psql_executable_sha256"],
            "probed_at_unix": now,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        return {**unsigned, "receipt_sha256": _sha_json(unsigned)}
    finally:
        _zeroize(password)
        _zeroize(pgpass)
        if password_fd is not None:
            _wipe_descriptor(password_fd)
        if ca_fd is not None:
            _wipe_descriptor(ca_fd)


def _failure(revision: str | None) -> Mapping[str, Any]:
    unsigned = {
        "schema": FAILURE_SCHEMA,
        "ok": False,
        "release_revision": revision if isinstance(revision, str) and _REVISION.fullmatch(revision) else None,
        "error_code": "production_database_recovery_probe_failed",
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha_json(unsigned)}


def _emit(value: Mapping[str, Any]) -> None:
    sys.stdout.buffer.write(_canonical(value) + b"\n")
    sys.stdout.buffer.flush()


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise RecoveryProbeError("recovery_probe_command_invalid")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(allow_abbrev=False)
    parser.add_argument("command", choices=("preflight", "probe"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    revision: str | None = None
    try:
        args = _parser().parse_args(argv)
        revision = _revision_from_interpreter()
        preflight = collect_preflight(revision)
        if args.command == "preflight":
            _emit(preflight)
            return 0
        gate = build_gate(preflight)
        _emit(gate)
        receipt = run_probe_frame(
            sys.stdin.buffer,
            revision=revision,
            preflight=preflight,
            gate=gate,
        )
        _emit(receipt)
        return 0
    except BaseException:
        _emit(_failure(revision))
        return 2


__all__ = [
    "EXPECTED_PSQL_SHA256",
    "FAILURE_SCHEMA",
    "FRAME_MAGIC",
    "FRAME_SCHEMA",
    "GATE_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "RecoveryProbeError",
    "build_gate",
    "collect_preflight",
    "main",
    "run_probe_frame",
]


if __name__ == "__main__":
    raise SystemExit(main())
