#!/usr/bin/env python3
"""Root read-only collector for the pre-freeze production truth snapshot.

The first PostgreSQL observation cannot depend on a FreezePlan because that
plan binds the observation itself.  This boundary breaks that cycle without
weakening authority: it verifies the exact target release and its separately
owner-approved unit inputs, invokes only the release-sealed
``observe_initial`` action, observes the three fixed systemd units, and emits
one short-lived, secret-free public receipt.  It does not stage host files,
stop services, or mutate PostgreSQL.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gateway import canonical_writer_production_cutover as cutover
from gateway import production_cron_continuity_package as cron_continuity
from gateway import production_cron_migration as cron_migration
from ops.muncho.runtime import mechanical_job_rail
from scripts.canary import package_production_cutover_artifacts as package


RECEIPT_SCHEMA = "muncho-production-cutover-initial-observations.v1"
STOPPED_RECEIPT_SCHEMA = "muncho-production-cutover-stopped-services.v1"
MAX_RESPONSE = 8 * 1024 * 1024
MAX_OBSERVATION_SKEW_SECONDS = 120
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class InitialCollectorError(RuntimeError):
    """Stable, secret-free initial collector failure."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8", errors="strict")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_unix(value: Any) -> int:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise InitialCollectorError("initial_collector_auxiliary_fact_invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise InitialCollectorError(
            "initial_collector_auxiliary_fact_invalid"
        ) from exc
    if parsed.tzinfo != timezone.utc or parsed.microsecond != 0:
        raise InitialCollectorError("initial_collector_auxiliary_fact_invalid")
    return int(parsed.timestamp())


def _decode(raw: bytes) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, item in items:
            if name in result:
                raise InitialCollectorError("initial_collector_duplicate_key")
            result[name] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except InitialCollectorError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise InitialCollectorError("initial_collector_response_invalid") from exc
    if not isinstance(value, Mapping) or raw != _canonical(value):
        raise InitialCollectorError("initial_collector_response_invalid")
    return value


def _boot_id(path: Path = BOOT_ID_PATH) -> str:
    descriptor: int | None = None
    try:
        before = os.lstat(path)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or not 0 <= before.st_size <= 256
        ):
            raise InitialCollectorError("initial_collector_boot_identity_invalid")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        raw = os.read(descriptor, 257)
        after = os.fstat(descriptor)
    except InitialCollectorError:
        raise
    except OSError as exc:
        raise InitialCollectorError("initial_collector_boot_unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or (before.st_size != 0 and len(raw) != before.st_size)
        or len(raw) > 256
        or not raw.strip()
    ):
        raise InitialCollectorError("initial_collector_boot_changed")
    return _sha(raw.strip())


def _initial_request(
    *,
    revision: str,
    target: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> Mapping[str, Any]:
    unsigned = {
        "schema": "muncho-production-cutover-initial-observation-request.v1",
        "action": "observe_initial",
        "release_revision": revision,
        "target": copy.deepcopy(dict(target)),
        "artifact": {
            "path": artifact["path"],
            "sha256": artifact["sha256"],
        },
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "request_sha256": _sha(_canonical(unsigned))}


def _invoke_observer(
    *,
    executable: Path,
    request: Mapping[str, Any],
    runner: Callable[..., Any],
) -> cutover.LegacySnapshot:
    try:
        completed = runner(
            (str(executable), "observe_initial"),
            input=_canonical(request) + b"\n",
            capture_output=True,
            check=False,
            timeout=900,
            env={
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "PYTHONNOUSERSITE": "1",
            },
            cwd="/",
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise InitialCollectorError("initial_collector_observer_failed") from exc
    stdout = getattr(completed, "stdout", None)
    stderr = getattr(completed, "stderr", None)
    if (
        getattr(completed, "returncode", 1) != 0
        or not isinstance(stdout, bytes)
        or not isinstance(stderr, bytes)
        or stderr
        or not stdout.endswith(b"\n")
        or len(stdout) > MAX_RESPONSE + 1
    ):
        raise InitialCollectorError("initial_collector_observer_failed")
    try:
        return cutover.LegacySnapshot.from_mapping(_decode(stdout[:-1]))
    except (TypeError, ValueError) as exc:
        raise InitialCollectorError("initial_collector_snapshot_invalid") from exc


def _mechanical_package_manifest(
    revision: str,
    host_facts: Mapping[str, Any],
) -> Mapping[str, Any]:
    package_value = mechanical_job_rail.build_package(
        revision=revision,
        host_facts=host_facts,
        expected_host_facts_sha256=str(host_facts["host_facts_sha256"]),
    )
    return mechanical_job_rail.package_public_manifest(package_value)


def _cron_continuity_derivation(
    revision: str,
    mechanical_job_package: Mapping[str, Any],
) -> cron_continuity.HostContinuityDerivation:
    return cron_continuity.derive_packaged_continuity_from_host(
        revision=revision,
        mechanical_job_package=mechanical_job_package,
        source_jobs_path=cron_migration.DEFAULT_JOBS_PATH,
    )


def collect_initial_observations(
    revision: str,
    *,
    release_root: Path | None = None,
    unit_inputs: Mapping[str, Any] | None = None,
    runner: Callable[..., Any] = subprocess.run,
    services: Any | None = None,
    clock: Callable[[], float] = time.time,
    boot_reader: Callable[[], str] = _boot_id,
    cron_inventory_collector: Callable[[], Mapping[str, Any]] = (
        cron_migration.inventory_jobs_file
    ),
    mechanical_host_facts_collector: Callable[[], Mapping[str, Any]] = (
        mechanical_job_rail.collect_host_facts
    ),
    mechanical_package_collector: Callable[
        [str, Mapping[str, Any]], Mapping[str, Any]
    ] = _mechanical_package_manifest,
    cron_continuity_collector: Callable[
        [str, Mapping[str, Any]],
        cron_continuity.HostContinuityDerivation,
    ] = _cron_continuity_derivation,
    require_root: bool = True,
) -> Mapping[str, Any]:
    """Collect the only facts that must exist before FreezePlan authoring."""

    if _REVISION.fullmatch(revision or "") is None:
        raise InitialCollectorError("initial_collector_revision_invalid")
    fixed_release = (
        cutover.PRODUCTION_RELEASE_BASE / f"hermes-agent-{revision[:12]}"
    )
    release = fixed_release if release_root is None else release_root
    if require_root:
        if (
            not sys.platform.startswith("linux")
            or os.geteuid() != 0
            or release != fixed_release
        ):
            raise InitialCollectorError("initial_collector_requires_linux_root")
        inputs = package.load_fixed_unit_inputs(revision=revision)
    else:
        if unit_inputs is None:
            raise InitialCollectorError("initial_collector_test_inputs_required")
        inputs = package._unit_inputs(unit_inputs, revision=revision)
    manifest = package.verify_release_artifacts(
        release,
        revision,
        release_address=fixed_release,
        unit_inputs=inputs,
    )
    if manifest["unit_inputs"] != inputs:
        raise InitialCollectorError("initial_collector_unit_inputs_drifted")
    observer = manifest["artifacts"]["production-observe"]
    if "observe_initial" not in observer["actions"]:
        raise InitialCollectorError("initial_collector_observer_action_missing")
    request = _initial_request(
        revision=revision,
        target=inputs["target"],
        artifact=observer,
    )
    boot_before = boot_reader()
    try:
        observed_cron_inventory = cron_migration.validate_inventory(
            cron_inventory_collector()
        )
        host_facts_candidate = mechanical_host_facts_collector()
        host_facts = mechanical_job_rail.validate_host_facts(
            host_facts_candidate,
            expected_sha256=str(
                host_facts_candidate.get("host_facts_sha256", "")
            ),
        )
        mechanical_package = mechanical_job_rail.validate_package_manifest(
            mechanical_package_collector(revision, host_facts),
            revision=revision,
            host_facts_sha256=host_facts["host_facts_sha256"],
        )
        continuity_derivation = cron_continuity_collector(
            revision,
            mechanical_package,
        )
        if not isinstance(
            continuity_derivation,
            cron_continuity.HostContinuityDerivation,
        ):
            raise TypeError("cron continuity derivation is invalid")
        cron_inventory = cron_migration.validate_inventory(
            continuity_derivation.inventory
        )
        if cron_inventory != observed_cron_inventory:
            raise ValueError("cron store changed during initial collection")
        cron_continuity_plan = (
            cron_continuity.validate_packaged_continuity_plan(
                continuity_derivation.build.plan,
                inventory=cron_inventory,
                expected_mechanical_job_package_manifest_sha256=(
                    mechanical_package["manifest_sha256"]
                ),
                require_executable=True,
            )
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        cron_migration.ProductionCronMigrationError,
        mechanical_job_rail.MechanicalJobRailError,
        cron_continuity.ProductionCronContinuityPackageError,
    ) as exc:
        raise InitialCollectorError(
            "initial_collector_auxiliary_fact_invalid"
        ) from exc
    snapshot = _invoke_observer(
        executable=(
            release / "ops/muncho/cutover/artifacts/production-observe"
        ),
        request=request,
        runner=runner,
    )
    boundary = (
        cutover.ProductionSystemdServiceBoundary(clock=clock)
        if services is None
        else services
    )
    gateway = boundary.observe_gateway()
    writer = boundary.observe_writer()
    connector = boundary.observe_connector()
    boot_after = boot_reader()
    observed_at = int(clock())
    observations = (
        snapshot.value["observed_at_unix"],
        gateway.value["observed_at_unix"],
        writer.value["observed_at_unix"],
        connector.value["observed_at_unix"],
        _utc_unix(cron_inventory["created_at"]),
        _utc_unix(host_facts["collected_at"]),
    )
    if (
        boot_before != boot_after
        or re.fullmatch(r"[0-9a-f]{64}", boot_after or "") is None
        or any(
            type(value) is not int
            or value > observed_at + 30
            or observed_at - value > MAX_OBSERVATION_SKEW_SECONDS
            for value in observations
        )
        or gateway.value["name"] != cutover.GATEWAY_UNIT
        or gateway.stopped
        or writer.value["name"] != cutover.WRITER_UNIT
        or not writer.stopped
        or connector.value["name"] != cutover.CONNECTOR_UNIT
        or not connector.stopped
    ):
        raise InitialCollectorError("initial_collector_live_state_invalid")
    unsigned = {
        "schema": RECEIPT_SCHEMA,
        "release_revision": revision,
        "target": copy.deepcopy(inputs["target"]),
        "artifacts": copy.deepcopy(manifest["plan_bindings"]),
        "gateway_before": gateway.to_mapping(),
        "writer_before": writer.to_mapping(),
        "connector_before": connector.to_mapping(),
        "initial_snapshot": snapshot.to_mapping(),
        "cron_inventory": copy.deepcopy(cron_inventory),
        "cron_continuity_plan": copy.deepcopy(cron_continuity_plan),
        "mechanical_job_host_facts": copy.deepcopy(host_facts),
        "mechanical_job_package": copy.deepcopy(mechanical_package),
        "observed_at_unix": observed_at,
        "source_boot_id_sha256": boot_after,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha(_canonical(unsigned))}


def collect_stopped_services(
    revision: str,
    *,
    freeze_plan: Mapping[str, Any] | None = None,
    freeze_approval: Mapping[str, Any] | None = None,
    services: Any | None = None,
    clock: Callable[[], float] = time.time,
    boot_reader: Callable[[], str] = _boot_id,
    require_root: bool = True,
) -> Mapping[str, Any]:
    """Collect the three stopped identities after final-tail capture."""

    if _REVISION.fullmatch(revision or "") is None:
        raise InitialCollectorError("stopped_collector_revision_invalid")
    if require_root:
        if not sys.platform.startswith("linux") or os.geteuid() != 0:
            raise InitialCollectorError("stopped_collector_requires_linux_root")
        plan_value = cutover._load_staged_json(
            cutover.STAGED_FREEZE_PLAN_PATH
        )
        approval_value = cutover._load_staged_json(
            cutover.STAGED_FREEZE_APPROVAL_PATH
        )
    else:
        if freeze_plan is None or freeze_approval is None:
            raise InitialCollectorError("stopped_collector_test_inputs_required")
        plan_value = freeze_plan
        approval_value = freeze_approval
    try:
        plan = cutover.FreezePlan.from_mapping(plan_value)
        approval = cutover.CutoverApproval.from_mapping(
            approval_value,
            plan=plan,
            now_unix=int(clock()),
        )
    except (TypeError, ValueError) as exc:
        raise InitialCollectorError("stopped_collector_authority_invalid") from exc
    if plan.value["release_revision"] != revision:
        raise InitialCollectorError("stopped_collector_revision_invalid")
    boundary = (
        cutover.ProductionSystemdServiceBoundary(clock=clock)
        if services is None
        else services
    )
    boot_before = boot_reader()
    gateway = boundary.observe_gateway()
    writer = boundary.observe_writer()
    connector = boundary.observe_connector()
    boot_after = boot_reader()
    observed_at = int(clock())
    expected = (
        cutover.ServiceObservation.from_mapping(plan.value["gateway_before"]),
        cutover.ServiceObservation.from_mapping(plan.value["writer_before"]),
        cutover.ServiceObservation.from_mapping(plan.value["connector_before"]),
    )
    observed = (gateway, writer, connector)
    if (
        boot_before != boot_after
        or re.fullmatch(r"[0-9a-f]{64}", boot_after or "") is None
        or any(not item.stopped for item in observed)
        or any(
            item.stable_identity() != expected_item.stable_identity()
            for item, expected_item in zip(observed, expected, strict=True)
        )
        or any(
            item.value["observed_at_unix"] > observed_at + 30
            or observed_at - item.value["observed_at_unix"]
            > MAX_OBSERVATION_SKEW_SECONDS
            for item in observed
        )
    ):
        raise InitialCollectorError("stopped_collector_live_state_invalid")
    unsigned = {
        "schema": STOPPED_RECEIPT_SCHEMA,
        "release_revision": revision,
        "freeze_plan_sha256": plan.sha256,
        "freeze_approval_sha256": approval.value["approval_sha256"],
        "gateway_stopped": gateway.to_mapping(),
        "writer_stopped": writer.to_mapping(),
        "connector_stopped": connector.to_mapping(),
        "observed_at_unix": observed_at,
        "source_boot_id_sha256": boot_after,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha(_canonical(unsigned))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect exact production cutover observations",
    )
    parser.add_argument("action", choices=("initial", "stopped"))
    parser.add_argument("--revision", required=True)
    arguments = parser.parse_args(argv)
    try:
        receipt = (
            collect_initial_observations(arguments.revision)
            if arguments.action == "initial"
            else collect_stopped_services(arguments.revision)
        )
    except (InitialCollectorError, OSError, TypeError, ValueError):
        print(
            '{"error_code":"initial_cutover_collection_failed","ok":false}',
            file=sys.stderr,
        )
        return 2
    print(_canonical(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
