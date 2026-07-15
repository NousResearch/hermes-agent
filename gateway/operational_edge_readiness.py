"""Portable readiness contract for credential-scoped operational edges.

The live collector performs peer/MainPID and real-user operation probes.  This
module is the pure, publication-safe validator consumed by cron continuity and
cutover code.  It never probes a host, reads credentials, or treats output text
as a decision signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from gateway.operational_edge_catalog import (
    catalog_public_contract,
    operation_catalog,
    required_cron_operations,
)


OPERATIONAL_EDGE_READINESS_SCHEMA = "muncho-operational-edge-readiness.v2"
PROBE_PACKET_SCHEMA = "muncho-operational-edge-probe-packet.v1"
OPERATIONAL_EDGE_READINESS_PATH = Path(
    "/var/lib/muncho-operational-edge/readiness.json"
)
MAX_READINESS_BYTES = 4 * 1024 * 1024
READINESS_MAXIMUM_AGE_SECONDS = 120
COLLECTOR_CHILD_TIMEOUT_SECONDS = 15 * 60
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID = re.compile(r"^[0-9a-f]{12}$")
_UNIT = re.compile(r"^muncho-operational-edge-[a-z][a-z0-9_-]{0,31}\.service$")

JOB_FIELDS = frozenset(
    {
        "source_job_id",
        "operation_id",
        "domain",
        "service_unit",
        "service_uid",
        "service_gid",
        "socket_path",
        "socket_uid",
        "socket_gid",
        "socket_mode",
        "main_pid",
        "peer_round_trip",
        "probe_operation_id",
        "probe_return_code",
        "probe_packet_schema",
        "probe_packet_sha256",
        "meaningful_packet",
        "error_only_packet",
    }
)

READINESS_FIELDS = frozenset(
    {
        "schema",
        "release_revision",
        "boot_id_sha256",
        "observed_at_unix",
        "maximum_age_seconds",
        "collector_nonce",
        "catalog_sha256",
        "required_jobs",
        "required_job_count",
        "jobs",
        "job_count",
        "all_required_jobs_ready",
        "credential_values_read",
        "permissions_widened",
        "secret_material_recorded",
        "secret_digest_recorded",
        "receipt_sha256",
    }
)


class OperationalEdgeReadinessError(ValueError):
    """Stable, secret-free operational readiness validation failure."""


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
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_json_invalid"
        ) from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def operational_catalog_sha256() -> str:
    return _sha256(_canonical(catalog_public_contract()))


def _current_boot_id_sha256() -> str:
    try:
        raw = BOOT_ID_PATH.read_bytes()
        parsed = uuid.UUID(raw.decode("ascii", errors="strict").strip())
    except (OSError, UnicodeError, ValueError, AttributeError) as exc:
        raise OperationalEdgeReadinessError(
            "operational_edge_boot_identity_unavailable"
        ) from exc
    if parsed.int == 0:
        raise OperationalEdgeReadinessError(
            "operational_edge_boot_identity_invalid"
        )
    return _sha256((str(parsed) + "\n").encode("ascii"))


def _required(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise OperationalEdgeReadinessError(
            "operational_edge_required_jobs_invalid"
        )
    catalog = operation_catalog()
    result = dict(sorted(value.items()))
    if not result:
        raise OperationalEdgeReadinessError(
            "operational_edge_required_jobs_invalid"
        )
    for job_id, operation_id in result.items():
        operation = catalog.get(operation_id)
        if (
            _JOB_ID.fullmatch(job_id) is None
            or operation is None
            or operation.cron_source_job_id != job_id
        ):
            raise OperationalEdgeReadinessError(
                "operational_edge_required_jobs_invalid"
            )
    return result


def _job(value: Any, required: Mapping[str, str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != JOB_FIELDS:
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_job_invalid"
        )
    row = dict(value)
    job_id = row["source_job_id"]
    operation_id = row["operation_id"]
    operation = operation_catalog().get(operation_id)
    socket_path = Path(str(row["socket_path"]))
    if (
        required.get(job_id) != operation_id
        or operation is None
        or row["domain"] != operation.domain
        or _UNIT.fullmatch(str(row["service_unit"])) is None
        or row["service_unit"]
        != f"muncho-operational-edge-{operation.domain}.service"
        or any(
            type(row[name]) is not int or row[name] < 1
            for name in (
                "service_uid",
                "service_gid",
                "socket_uid",
                "socket_gid",
                "main_pid",
            )
        )
        or not socket_path.is_absolute()
        or ".." in socket_path.parts
        or socket_path
        != Path("/run/muncho-operational-edge") / operation.domain / "edge.sock"
        or row["socket_mode"] != "0660"
        or row["peer_round_trip"] is not True
        or row["probe_operation_id"] != (
            operation.probe_operation_id or operation.operation_id
        )
        or row["probe_return_code"] != 0
        or row["probe_packet_schema"] != PROBE_PACKET_SCHEMA
        or _SHA256.fullmatch(str(row["probe_packet_sha256"])) is None
        or row["meaningful_packet"] is not True
        or row["error_only_packet"] is not False
    ):
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_job_invalid"
        )
    return row


def build_operational_edge_readiness(
    *,
    revision: str,
    required_jobs: Mapping[str, str],
    jobs: Sequence[Mapping[str, Any]],
    boot_id_sha256: str | None = None,
    observed_at_unix: int | None = None,
    collector_nonce: str | None = None,
) -> dict[str, Any]:
    """Build one portable receipt from already-attested live observations."""

    if _REVISION.fullmatch(revision or "") is None:
        raise OperationalEdgeReadinessError(
            "operational_edge_release_revision_invalid"
        )
    required = _required(required_jobs)
    parsed = [_job(item, required) for item in jobs]
    parsed.sort(key=lambda item: item["source_job_id"])
    if [item["source_job_id"] for item in parsed] != sorted(required):
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_jobs_incomplete"
        )
    boot = boot_id_sha256 or _current_boot_id_sha256()
    observed = int(time.time()) if observed_at_unix is None else observed_at_unix
    nonce = collector_nonce or str(uuid.uuid4())
    try:
        parsed_nonce = uuid.UUID(nonce)
    except (ValueError, TypeError, AttributeError) as exc:
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_nonce_invalid"
        ) from exc
    if (
        _SHA256.fullmatch(boot or "") is None
        or type(observed) is not int
        or observed < 1
        or parsed_nonce.version != 4
        or str(parsed_nonce) != nonce
    ):
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_freshness_invalid"
        )
    unsigned = {
        "schema": OPERATIONAL_EDGE_READINESS_SCHEMA,
        "release_revision": revision,
        "boot_id_sha256": boot,
        "observed_at_unix": observed,
        "maximum_age_seconds": READINESS_MAXIMUM_AGE_SECONDS,
        "collector_nonce": nonce,
        "catalog_sha256": operational_catalog_sha256(),
        "required_jobs": [
            {"source_job_id": job_id, "operation_id": operation_id}
            for job_id, operation_id in required.items()
        ],
        "required_job_count": len(required),
        "jobs": parsed,
        "job_count": len(parsed),
        "all_required_jobs_ready": True,
        "credential_values_read": False,
        "permissions_widened": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha256(_canonical(unsigned))}


def collect_operational_edge_readiness(
    *,
    revision: str,
    required_jobs: Mapping[str, str],
    service_observer: Any,
    probe_runner: Any,
    boot_id_sha256: str | None = None,
    observed_at_unix: int | None = None,
) -> dict[str, Any]:
    """Collect exact live facts through injectable mechanical boundaries.

    ``service_observer(domain, unit)`` must return the exact UID/GID/socket/
    MainPID facts. ``probe_runner(operation_id, idempotency_key)`` must return
    one already signature-verified operational edge receipt.  Neither callback
    receives packet content for routing or chooses an operation.
    """

    required = _required(required_jobs)
    collector_nonce = str(uuid.uuid4())
    observed = int(time.time()) if observed_at_unix is None else observed_at_unix
    boot = _current_boot_id_sha256() if boot_id_sha256 is None else boot_id_sha256
    jobs: list[dict[str, Any]] = []
    for source_job_id, operation_id in required.items():
        operation = operation_catalog()[operation_id]
        unit = f"muncho-operational-edge-{operation.domain}.service"
        facts = service_observer(operation.domain, unit)
        if not isinstance(facts, Mapping) or set(facts) != {
            "service_uid", "service_gid", "socket_path", "socket_uid",
            "socket_gid", "socket_mode", "main_pid", "peer_round_trip",
        }:
            raise OperationalEdgeReadinessError(
                "operational_edge_service_observation_invalid"
            )
        probe_operation_id = operation.probe_operation_id or operation_id
        receipt = probe_runner(
            probe_operation_id,
            (
                f"operational-readiness:{revision}:"
                f"{collector_nonce}:{source_job_id}"
            ),
        )
        if not isinstance(receipt, Mapping):
            raise OperationalEdgeReadinessError(
                "operational_edge_probe_receipt_invalid"
            )
        stdout_b64 = receipt.get("stdout_b64")
        try:
            import base64

            stdout = base64.b64decode(stdout_b64, validate=True)
        except (TypeError, ValueError) as exc:
            raise OperationalEdgeReadinessError(
                "operational_edge_probe_receipt_invalid"
            ) from exc
        meaningful = bool(
            receipt.get("operation_id") == probe_operation_id
            and receipt.get("domain") == operation.domain
            and receipt.get("outcome") == "succeeded"
            and receipt.get("return_code") == 0
            and receipt.get("readback_verified") is True
            and receipt.get("service_pid") == facts.get("main_pid")
            and receipt.get("secret_material_recorded") is False
            and stdout.strip()
        )
        if not meaningful:
            raise OperationalEdgeReadinessError(
                "operational_edge_probe_not_meaningful"
            )
        packet = {
            "schema": PROBE_PACKET_SCHEMA,
            "source_job_id": source_job_id,
            "operation_id": operation_id,
            "probe_operation_id": probe_operation_id,
            "receipt_sha256": _sha256(_canonical(dict(receipt))),
            "stdout_sha256": _sha256(stdout),
            "return_code": 0,
            "meaningful_packet": True,
            "error_only_packet": False,
            "secret_material_recorded": False,
        }
        jobs.append(
            {
                "source_job_id": source_job_id,
                "operation_id": operation_id,
                "domain": operation.domain,
                "service_unit": unit,
                **dict(facts),
                "peer_round_trip": True,
                "probe_operation_id": probe_operation_id,
                "probe_return_code": 0,
                "probe_packet_schema": PROBE_PACKET_SCHEMA,
                "probe_packet_sha256": _sha256(_canonical(packet)),
                "meaningful_packet": True,
                "error_only_packet": False,
            }
        )
    return build_operational_edge_readiness(
        revision=revision,
        required_jobs=required,
        jobs=jobs,
        boot_id_sha256=boot,
        observed_at_unix=observed,
        collector_nonce=collector_nonce,
    )


def _validate_process_status(
    raw: str,
    *,
    expected_uid: int,
    expected_gid: int,
) -> None:
    fields: dict[str, list[str]] = {}
    for line in raw.splitlines():
        name, separator, value = line.partition(":")
        if separator and name in {"Uid", "Gid"}:
            fields[name] = value.split()
    try:
        uids = [int(value) for value in fields["Uid"]]
        gids = [int(value) for value in fields["Gid"]]
    except (KeyError, ValueError) as exc:
        raise OperationalEdgeReadinessError(
            "operational_edge_process_identity_invalid"
        ) from exc
    if (
        len(uids) != 4
        or len(gids) != 4
        or any(uid != expected_uid for uid in uids)
        or any(gid != expected_gid for gid in gids)
    ):
        raise OperationalEdgeReadinessError(
            "operational_edge_process_identity_invalid"
        )


def _process_identity(
    pid: int,
    *,
    expected_uid: int,
    expected_gid: int,
) -> None:
    try:
        raw = (Path("/proc") / str(pid) / "status").read_text(
            encoding="ascii", errors="strict"
        )
    except (OSError, UnicodeError) as exc:
        raise OperationalEdgeReadinessError(
            "operational_edge_process_identity_unavailable"
        ) from exc
    _validate_process_status(
        raw,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )


def _collector_probe_identity(
    configs: Mapping[str, Any],
) -> tuple[int, int, tuple[int, ...]]:
    identities = {
        (
            config.probe_uid,
            config.probe_gid,
            tuple(config.probe_supplementary_gids),
        )
        for config in configs.values()
    }
    if len(identities) != 1:
        raise OperationalEdgeReadinessError(
            "operational_edge_collector_identity_ambiguous"
        )
    uid, gid, supplementary_gids = next(iter(identities))
    socket_gids = tuple(sorted({config.socket_gid for config in configs.values()}))
    service_uids = {config.service_uid for config in configs.values()}
    if (
        type(uid) is not int
        or type(gid) is not int
        or uid < 1
        or gid < 1
        or uid in service_uids
        or supplementary_gids != socket_gids
    ):
        raise OperationalEdgeReadinessError(
            "operational_edge_collector_identity_invalid"
        )
    return uid, gid, supplementary_gids


def _require_current_collector_identity(
    configs: Mapping[str, Any],
    *,
    effective_uid: int,
    effective_gid: int,
    effective_supplementary_gids: Sequence[int],
) -> tuple[int, int, tuple[int, ...]]:
    uid, gid, supplementary_gids = _collector_probe_identity(configs)
    if (
        type(effective_uid) is not int
        or type(effective_gid) is not int
        or effective_uid == 0
        or effective_uid != uid
        or effective_gid != gid
        or tuple(sorted(effective_supplementary_gids))
        != supplementary_gids
    ):
        raise OperationalEdgeReadinessError(
            "operational_edge_collector_peer_unauthorized"
        )
    return uid, gid, supplementary_gids


def collect_operational_edge_readiness_live(
    *,
    revision: str,
    required_jobs: Mapping[str, str],
) -> dict[str, Any]:
    """Immediately re-observe systemd/PID/socket and execute every live probe."""

    from gateway.operational_edge_client import (
        OperationalEdgeClient,
        SystemctlMainPidProvider,
        load_operational_edge_client_configs,
    )

    configs = load_operational_edge_client_configs()
    _require_current_collector_identity(
        configs,
        effective_uid=os.geteuid(),
        effective_gid=os.getegid(),
        effective_supplementary_gids=os.getgroups(),
    )
    provider = SystemctlMainPidProvider()
    clients = {
        domain: OperationalEdgeClient(config, main_pid_provider=provider)
        for domain, config in configs.items()
    }

    def observe(domain: str, unit: str) -> Mapping[str, Any]:
        config = configs.get(domain)
        if config is None or config.service_unit != unit:
            raise OperationalEdgeReadinessError(
                "operational_edge_service_observation_invalid"
            )
        main_pid = provider.main_pid(unit)
        _process_identity(
            main_pid,
            expected_uid=config.service_uid,
            expected_gid=config.service_gid,
        )
        try:
            item = os.lstat(config.socket_path)
        except OSError as exc:
            raise OperationalEdgeReadinessError(
                "operational_edge_socket_unavailable"
            ) from exc
        if (
            not stat.S_ISSOCK(item.st_mode)
            or item.st_uid != config.service_uid
            or item.st_gid != config.socket_gid
            or stat.S_IMODE(item.st_mode) != 0o660
        ):
            raise OperationalEdgeReadinessError(
                "operational_edge_socket_identity_invalid"
            )
        return {
            "service_uid": config.service_uid,
            "service_gid": config.service_gid,
            "socket_path": str(config.socket_path),
            "socket_uid": item.st_uid,
            "socket_gid": item.st_gid,
            "socket_mode": "0660",
            "main_pid": main_pid,
            # The collector upgrades this to true only after client.invoke
            # verifies SO_PEERCRED == systemd MainPID and a signed receipt.
            "peer_round_trip": False,
        }

    def probe(operation_id: str, idempotency_key: str) -> Mapping[str, Any]:
        operation = operation_catalog()[operation_id]
        return clients[operation.domain].invoke(
            operation_id,
            {},
            idempotency_key=idempotency_key,
            timeout_seconds=min(60, operation.timeout_seconds),
        )

    return collect_operational_edge_readiness(
        revision=revision,
        required_jobs=required_jobs,
        service_observer=observe,
        probe_runner=probe,
    )


def _decode_dropped_collector_receipt(
    raw: bytes,
    *,
    revision: str,
    required_jobs: Mapping[str, str],
    expected_boot_id_sha256: str | None = None,
    now_unix: int | None = None,
) -> dict[str, Any]:
    if not raw or len(raw) > MAX_READINESS_BYTES:
        raise OperationalEdgeReadinessError(
            "operational_edge_dropped_collector_invalid"
        )

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("ascii", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise OperationalEdgeReadinessError(
            "operational_edge_dropped_collector_invalid"
        ) from exc
    if not isinstance(value, Mapping) or raw != _canonical(value) + b"\n":
        raise OperationalEdgeReadinessError(
            "operational_edge_dropped_collector_invalid"
        )
    return validate_operational_edge_readiness(
        value,
        revision=revision,
        required_jobs=required_jobs,
        expected_boot_id_sha256=(
            _current_boot_id_sha256()
            if expected_boot_id_sha256 is None
            else expected_boot_id_sha256
        ),
        now_unix=int(time.time()) if now_unix is None else now_unix,
    )


def _collect_operational_edge_readiness_as_service_peer(
    *,
    revision: str,
    required_jobs: Mapping[str, str],
    configs: Mapping[str, Any],
    interpreter: Path,
    runner: Any,
    expected_boot_id_sha256: str | None = None,
    now_unix: int | None = None,
) -> dict[str, Any]:
    """Run the live client in a child with irreversible non-root credentials."""

    uid, gid, supplementary_gids = _collector_probe_identity(configs)
    expected_release = (
        Path("/opt/adventico-ai-platform/hermes-agent-releases")
        / f"hermes-agent-{revision[:12]}"
    )
    expected_interpreter = expected_release / ".venv/bin/python"
    if (
        _REVISION.fullmatch(revision or "") is None
        or dict(required_jobs) != dict(required_cron_operations())
        or interpreter != expected_interpreter
    ):
        raise OperationalEdgeReadinessError(
            "operational_edge_dropped_collector_input_invalid"
        )
    try:
        completed = runner(
            [
                str(interpreter),
                "-I",
                "-B",
                "-m",
                "gateway.operational_edge_readiness",
                "--collect-child",
                "--revision",
                revision,
            ],
            cwd=str(expected_release),
            env={
                "HOME": "/var/lib/muncho-operational-edge",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "TZ": "UTC",
            },
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=COLLECTOR_CHILD_TIMEOUT_SECONDS,
            user=uid,
            group=gid,
            extra_groups=supplementary_gids,
            umask=0o077,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OperationalEdgeReadinessError(
            "operational_edge_dropped_collector_failed"
        ) from exc
    if (
        completed.returncode != 0
        or not isinstance(completed.stdout, bytes)
        or not isinstance(completed.stderr, bytes)
        or len(completed.stderr) > MAX_READINESS_BYTES
    ):
        raise OperationalEdgeReadinessError(
            "operational_edge_dropped_collector_failed"
        )
    return _decode_dropped_collector_receipt(
        completed.stdout,
        revision=revision,
        required_jobs=required_jobs,
        expected_boot_id_sha256=expected_boot_id_sha256,
        now_unix=now_unix,
    )


def collect_and_publish_operational_edge_readiness(
    *,
    revision: str,
    required_jobs: Mapping[str, str],
) -> dict[str, Any]:
    """Root-only production preflight: live probe, then exact publication."""

    if os.geteuid() != 0:
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_publisher_unauthorized"
        )
    from gateway.operational_edge_client import load_operational_edge_client_configs

    configs = load_operational_edge_client_configs()
    value = _collect_operational_edge_readiness_as_service_peer(
        revision=revision,
        required_jobs=required_jobs,
        configs=configs,
        interpreter=(
            Path("/opt/adventico-ai-platform/hermes-agent-releases")
            / f"hermes-agent-{revision[:12]}"
            / ".venv/bin/python"
        ),
        runner=subprocess.run,
    )
    _publish_mainpid_attestations(value)
    return publish_operational_edge_readiness(
        value,
        revision=revision,
        required_jobs=required_jobs,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="muncho-operational-edge-readiness")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collect-child", action="store_true")
    mode.add_argument("--publish", action="store_true")
    parser.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    if args.collect_child:
        if os.geteuid() == 0:
            raise OperationalEdgeReadinessError(
                "operational_edge_collector_peer_unauthorized"
            )
        value = collect_operational_edge_readiness_live(
            revision=args.revision,
            required_jobs=required_cron_operations(),
        )
    else:
        value = collect_and_publish_operational_edge_readiness(
            revision=args.revision,
            required_jobs=required_cron_operations(),
        )
    sys.stdout.buffer.write(_canonical(value) + b"\n")
    sys.stdout.buffer.flush()
    return 0


def _publish_mainpid_attestations(value: Mapping[str, Any]) -> None:
    if os.geteuid() != 0:
        raise OperationalEdgeReadinessError(
            "operational_edge_main_pid_publisher_unauthorized"
        )
    by_domain: dict[str, Mapping[str, Any]] = {}
    for row in value.get("jobs", []):
        if isinstance(row, Mapping):
            prior = by_domain.get(str(row.get("domain")))
            if prior is not None and (
                prior.get("main_pid") != row.get("main_pid")
                or prior.get("service_unit") != row.get("service_unit")
            ):
                raise OperationalEdgeReadinessError(
                    "operational_edge_main_pid_observation_conflict"
                )
            by_domain[str(row.get("domain"))] = row
    for domain, row in by_domain.items():
        unsigned = {
            "schema": "muncho-operational-edge-mainpid.v1",
            "domain": domain,
            "service_unit": row["service_unit"],
            "main_pid": row["main_pid"],
            "observed_at_unix": value["observed_at_unix"],
        }
        payload = _canonical(
            {**unsigned, "attestation_sha256": _sha256(_canonical(unsigned))}
        ) + b"\n"
        path = Path("/run/muncho-operational-edge") / domain / "mainpid.json"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=".mainpid.", suffix=".tmp"
        )
        try:
            os.fchmod(descriptor, 0o444)
            os.fchown(descriptor, 0, 0)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise


def validate_operational_edge_readiness(
    value: Any,
    *,
    revision: str,
    required_jobs: Mapping[str, str],
    expected_boot_id_sha256: str | None = None,
    now_unix: int | None = None,
) -> dict[str, Any]:
    """Validate exact job coverage and meaningful real-user probe evidence."""

    if (
        _REVISION.fullmatch(revision or "") is None
        or not isinstance(value, Mapping)
        or set(value) != READINESS_FIELDS
    ):
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_invalid"
        )
    required = _required(required_jobs)
    expected_required = [
        {"source_job_id": job_id, "operation_id": operation_id}
        for job_id, operation_id in required.items()
    ]
    jobs = value.get("jobs")
    if not isinstance(jobs, list):
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_invalid"
        )
    parsed = [_job(item, required) for item in jobs]
    job_ids = [item["source_job_id"] for item in parsed]
    unsigned = {name: item for name, item in value.items() if name != "receipt_sha256"}
    nonce = value.get("collector_nonce")
    try:
        parsed_nonce = uuid.UUID(str(nonce))
    except (ValueError, TypeError, AttributeError) as exc:
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_invalid"
        ) from exc
    boot_id = (
        _current_boot_id_sha256()
        if expected_boot_id_sha256 is None
        else expected_boot_id_sha256
    )
    now = int(time.time()) if now_unix is None else now_unix
    if (
        value.get("schema") != OPERATIONAL_EDGE_READINESS_SCHEMA
        or value.get("release_revision") != revision
        or _SHA256.fullmatch(str(boot_id or "")) is None
        or value.get("boot_id_sha256") != boot_id
        or type(now) is not int
        or type(value.get("observed_at_unix")) is not int
        or not 0 <= now - value["observed_at_unix"] <= READINESS_MAXIMUM_AGE_SECONDS
        or value.get("maximum_age_seconds") != READINESS_MAXIMUM_AGE_SECONDS
        or parsed_nonce.version != 4
        or str(parsed_nonce) != nonce
        or value.get("catalog_sha256") != operational_catalog_sha256()
        or value.get("required_jobs") != expected_required
        or type(value.get("required_job_count")) is not int
        or value.get("required_job_count") != len(required)
        or type(value.get("job_count")) is not int
        or value.get("job_count") != len(parsed)
        or job_ids != sorted(required)
        or len(job_ids) != len(set(job_ids))
        or value.get("all_required_jobs_ready") is not True
        or value.get("credential_values_read") is not False
        or value.get("permissions_widened") is not False
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or _SHA256.fullmatch(str(value.get("receipt_sha256") or "")) is None
        or value.get("receipt_sha256") != _sha256(_canonical(unsigned))
    ):
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_invalid"
        )
    return dict(value)


def load_operational_edge_readiness(
    path: Path = OPERATIONAL_EDGE_READINESS_PATH,
    *,
    revision: str,
    required_jobs: Mapping[str, str],
    expected_owner_uid: int = 0,
    expected_group_gid: int = 0,
) -> dict[str, Any]:
    """Stable-read the root-published live receipt and validate it exactly."""

    if (
        path != OPERATIONAL_EDGE_READINESS_PATH
        or type(expected_owner_uid) is not int
        or expected_owner_uid != 0
        or type(expected_group_gid) is not int
        or expected_group_gid != 0
    ):
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_path_invalid"
        )
    descriptor = -1
    try:
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != expected_owner_uid
            or before.st_gid != expected_group_gid
            or stat.S_IMODE(before.st_mode) != 0o400
            or not 0 < before.st_size <= MAX_READINESS_BYTES
        ):
            raise OperationalEdgeReadinessError(
                "operational_edge_readiness_file_invalid"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    except OperationalEdgeReadinessError:
        raise
    except OSError as exc:
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_file_unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
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
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or identity(before) != identity(opened)
        or identity(before) != identity(after)
    ):
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_file_changed"
        )
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = item
        return result

    try:
        value = json.loads(
            raw.decode("ascii", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_file_invalid"
        ) from exc
    if not isinstance(value, Mapping) or raw != _canonical(value) + b"\n":
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_file_invalid"
        )
    return validate_operational_edge_readiness(
        value,
        revision=revision,
        required_jobs=required_jobs,
        expected_boot_id_sha256=_current_boot_id_sha256(),
        now_unix=int(time.time()),
    )


def publish_operational_edge_readiness(
    value: Mapping[str, Any],
    *,
    revision: str,
    required_jobs: Mapping[str, str],
    path: Path = OPERATIONAL_EDGE_READINESS_PATH,
) -> dict[str, Any]:
    """Atomically publish the exact root:root 0400 canonical receipt."""

    if os.geteuid() != 0 or path != OPERATIONAL_EDGE_READINESS_PATH:
        raise OperationalEdgeReadinessError(
            "operational_edge_readiness_publisher_unauthorized"
        )
    trusted = validate_operational_edge_readiness(
        value,
        revision=revision,
        required_jobs=required_jobs,
        expected_boot_id_sha256=_current_boot_id_sha256(),
        now_unix=int(time.time()),
    )
    payload = _canonical(trusted) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        os.fchmod(descriptor, 0o400)
        os.fchown(descriptor, 0, 0)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return trusted


__all__ = [
    "JOB_FIELDS",
    "OPERATIONAL_EDGE_READINESS_SCHEMA",
    "OPERATIONAL_EDGE_READINESS_PATH",
    "PROBE_PACKET_SCHEMA",
    "OperationalEdgeReadinessError",
    "build_operational_edge_readiness",
    "collect_operational_edge_readiness",
    "collect_operational_edge_readiness_live",
    "collect_and_publish_operational_edge_readiness",
    "operational_catalog_sha256",
    "publish_operational_edge_readiness",
    "load_operational_edge_readiness",
    "validate_operational_edge_readiness",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the packaged child
    raise SystemExit(main())
