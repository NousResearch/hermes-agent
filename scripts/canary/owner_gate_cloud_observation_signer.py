#!/usr/bin/env python3
"""Fixed executor-UID signer for one release-bound owner-gate Cloud report.

The immutable entrypoint accepts one canonical LF-terminated request on stdin.
It has no argv, environment, path, command, URL, token, or signer selection.
The request must bind the complete unsigned Cloud schema to the already signed
HOST observation, attached-service-account probe, inert terminal receipt, and
installed package lineage.  Only then is the report signed by the provisioned
cloud observation key and durably replay-protected.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO, Mapping, NoReturn, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_preflight as preflight
from scripts.canary import owner_gate_stage0 as stage0
from scripts.canary import storage_growth_trusted_collector as trusted


REQUEST_SCHEMA = "muncho-owner-gate-cloud-observation-signing-request.v1"
ATTESTATION_SCHEMA = "muncho-owner-gate-observation-attestation.v1"
EXECUTOR_UID = 29103
MAX_FRAME_BYTES = 1024 * 1024
FRESHNESS_SECONDS = 300
OWNER_RELEASE_BASE = Path("/opt/muncho-owner-gate/releases")
CLOUD_SIGNER_ENTRYPOINT = "bin/muncho-owner-gate-cloud-observation-signer"
CLOUD_SIGNER_SOURCE = "scripts/canary/owner_gate_cloud_observation_signer.py"
CLOUD_SIGNER_ASSET = (
    "ops/muncho/owner-gate/bin/muncho-owner-gate-cloud-observation-signer"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_TERMINAL_FIELDS = frozenset({
    "schema",
    "release_sha",
    "source_tree_oid",
    "package_sha256",
    "kit_release_id",
    "trusted_runner_path",
    "bundle_path",
    "pre_foundation_authority_sha256",
    "foundation_apply_receipt_sha256",
    "project_ancestry_evidence_sha256",
    "project_ancestry_chain_sha256",
    "resource_ancestor_chain",
    "operation_order",
    "transport_receipt_sha256",
    "cloud_verify_receipt_sha256",
    "cloud_preflight_receipt_sha256",
    "cloud_install_receipt_sha256",
    "cloud_install_receipt_file_sha256",
    "cloud_install_receipt",
    "cloud_install_signature_framing_validated",
    "cloud_install_signature_cryptographically_verified",
    "inert_cloud_bundle_installed",
    "host_filesystem_materialization_performed",
    "current_release_selected",
    "systemd_units_enabled",
    "service_activation_performed",
    "activation_performed",
    "activation_seal_created",
    "iam_binding_created",
    "caddy_cutover_performed",
    "cloud_mutation_performed",
    "cloud_control_plane_mutation_performed",
    "terminal_receipt_sha256",
})


class OwnerGateCloudObservationSignerError(RuntimeError):
    """Stable, secret-free target signer failure."""


def _error(code: str, exc: BaseException | None = None) -> NoReturn:
    del exc
    raise OwnerGateCloudObservationSignerError(code) from None


def _canonical(value: Any) -> bytes:
    try:
        return foundation.canonical_json_bytes(value)
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_cloud_signer_json_invalid", exc)


def _decode_canonical(raw: bytes, *, maximum: int, code: str) -> Mapping[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        _error(code)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in items:
            if not isinstance(name, str) or name in result:
                raise ValueError
            result[name] = value
        return result

    try:
        value = json.loads(
            raw.decode("ascii", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error(code, exc)
    if not isinstance(value, Mapping) or _canonical(value) != raw:
        _error(code)
    return dict(value)


def _read_regular(
    path: Path,
    *,
    maximum: int,
    uid: int,
    gid: int,
    mode: int,
) -> bytes:
    descriptor: int | None = None
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_uid != uid
            or opened.st_gid != gid
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != mode
            or not 0 < opened.st_size <= maximum
        ):
            _error("owner_gate_cloud_signer_file_invalid")
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        if (
            len(raw) != opened.st_size
            or len(raw) > maximum
            or (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_nlink,
                stat.S_IMODE(after.st_mode),
            )
            != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_nlink,
                stat.S_IMODE(opened.st_mode),
            )
        ):
            _error("owner_gate_cloud_signer_file_changed")
        return bytes(raw)
    except OwnerGateCloudObservationSignerError:
        raise
    except OSError as exc:
        _error("owner_gate_cloud_signer_file_unavailable", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _runtime_revision() -> str:
    if os.geteuid() != EXECUTOR_UID:
        _error("owner_gate_cloud_signer_uid_invalid")
    try:
        executable = Path(sys.executable).resolve(strict=True)
    except OSError as exc:
        _error("owner_gate_cloud_signer_runtime_invalid", exc)
    candidates = [
        parent for parent in executable.parents if parent.parent == OWNER_RELEASE_BASE
    ]
    if (
        len(candidates) != 1
        or _REVISION.fullmatch(candidates[0].name) is None
        or executable != (candidates[0] / "venv/bin/python").resolve(strict=True)
    ):
        _error("owner_gate_cloud_signer_runtime_invalid")
    return candidates[0].name


def _load_package(revision: str) -> Mapping[str, Any]:
    release = OWNER_RELEASE_BASE / revision
    try:
        state = release.lstat()
    except OSError as exc:
        _error("owner_gate_cloud_signer_release_invalid", exc)
    if (
        not stat.S_ISDIR(state.st_mode)
        or state.st_uid != 0
        or state.st_gid != 0
        or stat.S_IMODE(state.st_mode) != 0o555
    ):
        _error("owner_gate_cloud_signer_release_invalid")
    raw = _read_regular(
        release / "package-manifest.json",
        maximum=stage0.MAX_JSON_BYTES,
        uid=0,
        gid=0,
        mode=0o444,
    )
    package = _decode_canonical(
        raw,
        maximum=stage0.MAX_JSON_BYTES,
        code="owner_gate_cloud_signer_package_invalid",
    )
    inventory = {name: package.get(name) for name in stage0.INVENTORY_FIELDS}
    unsigned = {
        name: value for name, value in package.items() if name != "package_sha256"
    }
    collectors = package.get("collector_public_key_ids")
    required_entrypoints = package.get("required_entrypoints")
    runtime_sources = package.get("runtime_source_closure")
    payloads = package.get("payloads")
    if not isinstance(payloads, list):
        _error("owner_gate_cloud_signer_package_invalid")
    signer_payloads = [
        item
        for item in payloads
        if isinstance(item, Mapping)
        and item.get("release_relative")
        in {CLOUD_SIGNER_ENTRYPOINT, CLOUD_SIGNER_SOURCE}
    ]
    expected_payloads = {
        CLOUD_SIGNER_ENTRYPOINT: (CLOUD_SIGNER_ASSET, "0555"),
        CLOUD_SIGNER_SOURCE: (CLOUD_SIGNER_SOURCE, "0444"),
    }
    if (
        set(package) != stage0.MANIFEST_FIELDS
        or package.get("schema") != stage0.PACKAGE_SCHEMA
        or package.get("release_revision") != revision
        or package.get("release_root") != str(release)
        or package.get("package_inventory_sha256") != foundation.sha256_json(inventory)
        or package.get("package_sha256") != foundation.sha256_json(unsigned)
        or not isinstance(collectors, Mapping)
        or set(collectors) != {"network", "cloud", "host"}
        or any(not isinstance(value, str) for value in collectors.values())
        or any(_SHA256.fullmatch(str(value)) is None for value in collectors.values())
        or len(set(collectors.values())) != 3
        or not isinstance(required_entrypoints, list)
        or any(not isinstance(item, str) for item in required_entrypoints)
        or len(required_entrypoints) != len(set(required_entrypoints))
        or CLOUD_SIGNER_ENTRYPOINT not in required_entrypoints
        or not isinstance(runtime_sources, list)
        or any(not isinstance(item, str) for item in runtime_sources)
        or len(runtime_sources) != len(set(runtime_sources))
        or CLOUD_SIGNER_SOURCE not in runtime_sources
        or len(signer_payloads) != 2
        or any(
            item.get("source_relative")
            != expected_payloads[str(item["release_relative"])][0]
            or item.get("owner") != "root:root"
            or item.get("mode") != expected_payloads[str(item["release_relative"])][1]
            or _SHA256.fullmatch(str(item.get("sha256", ""))) is None
            or type(item.get("size")) is not int
            or item["size"] <= 0
            for item in signer_payloads
        )
        or package.get("release_owner") != "root:root"
        or package.get("release_directory_mode") != "0555"
        or package.get("immutable_after_install") is not True
        or package.get("offline_bootstrap") is not True
        or package.get("network_install_required") is not False
        or package.get("interpreter_source") != "pinned_debian_image_usr_bin_python3"
        or package.get("interpreter_version") != "3.11.2"
        or package.get("interpreter_hash_revalidated_before_each_service_start")
        is not True
        or package.get("generic_shell_entrypoint") is not False
        or package.get("local_gcloud_runtime_fallback") is not False
        or package.get("secret_material_recorded") is not False
        or package.get("secret_digest_recorded") is not False
        or package.get("activation_performed") is not False
        or package.get("cloud_mutation_performed") is not False
        or package.get("caller_self_hash_is_authority") is not False
    ):
        _error("owner_gate_cloud_signer_package_invalid")
    return package


def _load_public_key(path: Path, *, expected_id: str) -> Ed25519PublicKey:
    raw = _read_regular(path, maximum=32, uid=0, gid=0, mode=0o444)
    if len(raw) != 32 or hashlib.sha256(raw).hexdigest() != expected_id:
        _error("owner_gate_cloud_signer_public_key_invalid")
    try:
        return Ed25519PublicKey.from_public_bytes(raw)
    except ValueError as exc:
        _error("owner_gate_cloud_signer_public_key_invalid", exc)


def _host_spec(package: Mapping[str, Any]) -> foundation.OwnerGateSpec:
    collectors = package.get("collector_public_key_ids")
    image = package.get("interpreter_image")
    ancestors = package.get("resource_ancestor_chain")
    if (
        not isinstance(collectors, Mapping)
        or not isinstance(image, Mapping)
        or not isinstance(ancestors, list)
        or not ancestors
        or not isinstance(image.get("image_self_link"), str)
    ):
        _error("owner_gate_cloud_signer_package_invalid")
    prefix = "https://www.googleapis.com/compute/v1/"
    image_self_link = str(image["image_self_link"])
    organization = str(ancestors[-1])
    if (
        not image_self_link.startswith(prefix)
        or re.fullmatch(r"organizations/[1-9][0-9]{5,30}", organization) is None
    ):
        _error("owner_gate_cloud_signer_package_invalid")
    try:
        spec = foundation.OwnerGateSpec(
            release_revision=str(package["release_revision"]),
            source_tree_oid=str(package["source_tree_oid"]),
            boot_image_self_link=image_self_link.removeprefix(prefix),
            boot_image_numeric_id=str(image["image_numeric_id"]),
            package_inventory_sha256=str(package["package_inventory_sha256"]),
            interpreter_sha256=str(package["interpreter_sha256"]),
            network_collector_public_key_id=str(collectors["network"]),
            cloud_collector_public_key_id=str(collectors["cloud"]),
            host_collector_public_key_id=str(collectors["host"]),
            organization_id=organization.split("/", 1)[1],
            ancestry_evidence_sha256=str(package["project_ancestry_evidence_sha256"]),
        )
        spec.validate()
    except (KeyError, foundation.OwnerGateFoundationError) as exc:
        _error("owner_gate_cloud_signer_package_invalid", exc)
    return spec


def _validate_terminal(
    value: Any,
    *,
    package: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TERMINAL_FIELDS:
        _error("owner_gate_cloud_signer_terminal_invalid")
    terminal = dict(value)
    unsigned = {
        name: item
        for name, item in terminal.items()
        if name != "terminal_receipt_sha256"
    }
    false_fields = (
        "current_release_selected",
        "service_activation_performed",
        "activation_performed",
        "activation_seal_created",
        "iam_binding_created",
        "caddy_cutover_performed",
        "cloud_mutation_performed",
        "cloud_control_plane_mutation_performed",
    )
    lineage = (
        "pre_foundation_authority_sha256",
        "foundation_apply_receipt_sha256",
        "project_ancestry_evidence_sha256",
        "project_ancestry_chain_sha256",
    )
    if (
        terminal.get("schema") != "muncho-owner-gate-inert-cloud-bundle-terminal.v1"
        or terminal.get("release_sha") != package.get("release_revision")
        or terminal.get("source_tree_oid") != package.get("source_tree_oid")
        or terminal.get("package_sha256") != package.get("package_sha256")
        or any(terminal.get(name) != package.get(name) for name in lineage)
        or terminal.get("resource_ancestor_chain")
        != package.get("resource_ancestor_chain")
        or terminal.get("operation_order")
        != [
            "transport_exact_stage0_and_bundle",
            "cloud-verify",
            "cloud-preflight",
            "cloud-install",
        ]
        or terminal.get("cloud_install_signature_framing_validated") is not True
        or terminal.get("inert_cloud_bundle_installed") is not True
        or terminal.get("host_filesystem_materialization_performed") is not True
        or terminal.get("systemd_units_enabled") != []
        or any(terminal.get(name) is not False for name in false_fields)
        or terminal.get("terminal_receipt_sha256") != foundation.sha256_json(unsigned)
    ):
        _error("owner_gate_cloud_signer_terminal_invalid")
    return terminal


def _validate_host_binding(
    value: Any,
    *,
    phase: str,
    plan_sha256: str,
    package: Mapping[str, Any],
    host_public_key: Ed25519PublicKey,
    now_unix: int,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error("owner_gate_cloud_signer_host_invalid")
    host = dict(value)
    try:
        preflight._validate_host(
            host,
            spec=_host_spec(package),
            plan_sha256=plan_sha256,
            public_key=host_public_key,
            expected_public_key_id=str(package["collector_public_key_ids"]["host"]),
            mutation_binding_present=phase == "post_iam",
        )
    except preflight.OwnerGatePreflightError as exc:
        _error("owner_gate_cloud_signer_host_invalid", exc)
    release = host.get("release")
    host_collected = host.get("collected_at_unix")
    host_completed = host.get("completed_at_unix")
    host_fresh_through = host.get("fresh_through_unix")
    expected_probe = preflight.expected_effective_permission_probe(phase == "post_iam")
    lineage = (
        "pre_foundation_authority_sha256",
        "foundation_apply_receipt_sha256",
        "project_ancestry_evidence_sha256",
        "project_ancestry_chain_sha256",
    )
    receipt_fields = (
        "attached_sa_permission_probe_report_sha256",
        "cloud_signer_provisioning_receipt_sha256",
        "cloud_signer_readiness_sha256",
        "host_signer_provisioning_receipt_sha256",
        "host_signer_readiness_sha256",
    )
    if (
        host.get("schema") != preflight.HOST_OBSERVATION_SCHEMA
        or host.get("phase") != phase
        or host.get("plan_sha256") != plan_sha256
        or type(host_collected) is not int
        or not 0 <= now_unix - host_collected <= FRESHNESS_SECONDS
        or type(host_completed) is not int
        or type(host_fresh_through) is not int
        or not host_completed <= now_unix <= host_fresh_through
        or host.get("effective_permission_probe") != expected_probe
        or not isinstance(release, Mapping)
        or release.get("revision") != package.get("release_revision")
        or release.get("source_tree_oid") != package.get("source_tree_oid")
        or release.get("package_sha256") != package.get("package_sha256")
        or release.get("package_inventory_sha256")
        != package.get("package_inventory_sha256")
        or any(release.get(name) != package.get(name) for name in lineage)
        or release.get("resource_ancestor_chain")
        != package.get("resource_ancestor_chain")
        or any(
            _SHA256.fullmatch(str(release.get(name, ""))) is None
            for name in receipt_fields
        )
    ):
        _error("owner_gate_cloud_signer_host_invalid")
    return host


def _validate_request(
    value: Any,
    *,
    revision: str,
    package: Mapping[str, Any],
    host_public_key: Ed25519PublicKey,
    now_unix: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    fields = {
        "schema",
        "phase",
        "release_revision",
        "unsigned_observation",
        "terminal_receipt",
        "host_observation",
        "request_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        _error("owner_gate_cloud_signer_request_invalid")
    request = dict(value)
    unsigned_request = {
        name: item for name, item in request.items() if name != "request_sha256"
    }
    phase = request.get("phase")
    unsigned_observation = request.get("unsigned_observation")
    if (
        request.get("schema") != REQUEST_SCHEMA
        or phase not in {"inert", "post_iam"}
        or request.get("release_revision") != revision
        or request.get("request_sha256") != foundation.sha256_json(unsigned_request)
        or not isinstance(unsigned_observation, Mapping)
    ):
        _error("owner_gate_cloud_signer_request_invalid")
    unsigned_observation = dict(unsigned_observation)
    plan_sha256 = unsigned_observation.get("plan_sha256")
    collected = unsigned_observation.get("collected_at_unix")
    if (
        _SHA256.fullmatch(str(plan_sha256 or "")) is None
        or type(collected) is not int
        or not 0 <= now_unix - collected <= FRESHNESS_SECONDS
    ):
        _error("owner_gate_cloud_signer_request_stale")
    terminal = _validate_terminal(request.get("terminal_receipt"), package=package)
    host = _validate_host_binding(
        request.get("host_observation"),
        phase=str(phase),
        plan_sha256=str(plan_sha256),
        package=package,
        host_public_key=host_public_key,
        now_unix=now_unix,
    )
    release = host["release"]
    binding = unsigned_observation.get("release_binding")
    expected_binding = {
        "phase": phase,
        "release_revision": revision,
        "source_tree_oid": package["source_tree_oid"],
        "package_sha256": package["package_sha256"],
        "package_inventory_sha256": package["package_inventory_sha256"],
        "pre_foundation_authority_sha256": package["pre_foundation_authority_sha256"],
        "foundation_apply_receipt_sha256": package["foundation_apply_receipt_sha256"],
        "project_ancestry_evidence_sha256": package["project_ancestry_evidence_sha256"],
        "project_ancestry_chain_sha256": package["project_ancestry_chain_sha256"],
        "resource_ancestor_chain": package["resource_ancestor_chain"],
        "terminal_receipt_sha256": terminal["terminal_receipt_sha256"],
        "host_observation_report_sha256": host["report_sha256"],
        "host_observation_binding_sha256": host["observation_binding_sha256"],
        "attached_sa_permission_probe_report_sha256": release[
            "attached_sa_permission_probe_report_sha256"
        ],
        "cloud_signer_provisioning_receipt_sha256": release[
            "cloud_signer_provisioning_receipt_sha256"
        ],
        "cloud_signer_readiness_sha256": release["cloud_signer_readiness_sha256"],
        "host_signer_provisioning_receipt_sha256": release[
            "host_signer_provisioning_receipt_sha256"
        ],
        "host_signer_readiness_sha256": release["host_signer_readiness_sha256"],
        "effective_permission_probe_sha256": foundation.sha256_json(
            host["effective_permission_probe"]
        ),
    }
    if binding != expected_binding:
        _error("owner_gate_cloud_signer_lineage_invalid")
    try:
        preflight._validate_cloud_unsigned(
            unsigned_observation,
            plan_sha256=str(plan_sha256),
            mutation_binding_present=phase == "post_iam",
        )
    except preflight.OwnerGatePreflightError as exc:
        _error("owner_gate_cloud_signer_observation_invalid", exc)
    return request, unsigned_observation


def _decode_signature(value: Any) -> bytes:
    if not isinstance(value, str) or len(value) != 86 or "=" in value:
        _error("owner_gate_cloud_signer_signature_invalid")
    try:
        raw = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        _error("owner_gate_cloud_signer_signature_invalid", exc)
    if (
        len(raw) != 64
        or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != value
    ):
        _error("owner_gate_cloud_signer_signature_invalid")
    return raw


def _validate_response(
    value: Any,
    *,
    unsigned_observation: Mapping[str, Any],
    public_key: Ed25519PublicKey,
    expected_key_id: str,
    phase: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error("owner_gate_cloud_signer_response_invalid")
    response = dict(value)
    report_sha256 = foundation.sha256_json(unsigned_observation)
    attestation = response.get("attestation")
    if (
        set(response) != {*unsigned_observation, "report_sha256", "attestation"}
        or any(
            response.get(name) != item for name, item in unsigned_observation.items()
        )
        or response.get("report_sha256") != report_sha256
        or not isinstance(attestation, Mapping)
        or set(attestation) != {"schema", "public_key_id", "signature_ed25519_b64url"}
        or attestation.get("schema") != ATTESTATION_SCHEMA
        or attestation.get("public_key_id") != expected_key_id
    ):
        _error("owner_gate_cloud_signer_response_invalid")
    signed = {name: item for name, item in response.items() if name != "attestation"}
    try:
        public_key.verify(
            _decode_signature(attestation["signature_ed25519_b64url"]),
            _canonical(signed),
        )
        preflight._validate_cloud(
            response,
            plan_sha256=str(unsigned_observation["plan_sha256"]),
            public_key=public_key,
            expected_public_key_id=expected_key_id,
            mutation_binding_present=phase == "post_iam",
        )
    except (InvalidSignature, preflight.OwnerGatePreflightError) as exc:
        _error("owner_gate_cloud_signer_response_invalid", exc)
    return response


def _store_or_replay(
    response: Mapping[str, Any],
    *,
    request_sha256: str,
    config: Mapping[str, Any],
    public_key: Ed25519PublicKey,
    unsigned_observation: Mapping[str, Any],
    expected_key_id: str,
    phase: str,
) -> Mapping[str, Any]:
    _path, directory_fd = trusted._validate_replay_directory(config)
    name = f"owner-gate-cloud-{request_sha256}.json"
    try:
        lock_fd = os.open(
            ".owner-gate-cloud.lock",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        try:
            os.fchmod(lock_fd, 0o600)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            existing_raw = trusted._read_at(directory_fd, name, maximum=MAX_FRAME_BYTES)
            if existing_raw is not None:
                existing = _decode_canonical(
                    existing_raw,
                    maximum=MAX_FRAME_BYTES,
                    code="owner_gate_cloud_signer_replay_invalid",
                )
                checked = _validate_response(
                    existing,
                    unsigned_observation=unsigned_observation,
                    public_key=public_key,
                    expected_key_id=expected_key_id,
                    phase=phase,
                )
                if _canonical(checked) != _canonical(response):
                    _error("owner_gate_cloud_signer_replay_conflict")
                return checked
            trusted._write_at_atomic(directory_fd, name, _canonical(response))
            return response
        except trusted.TrustedObservationError as exc:
            _error("owner_gate_cloud_signer_replay_invalid", exc)
        finally:
            os.close(lock_fd)
    finally:
        os.close(directory_fd)


def sign_request(
    request_value: Mapping[str, Any],
    *,
    release_revision: str,
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate, sign, self-verify, and durably replay one exact request."""

    if (
        os.geteuid() != EXECUTOR_UID
        or _REVISION.fullmatch(release_revision) is None
        or type(now_unix) is not int
        or now_unix <= 0
    ):
        _error("owner_gate_cloud_signer_runtime_invalid")
    package = _load_package(release_revision)
    release = OWNER_RELEASE_BASE / release_revision
    host_key = _load_public_key(
        release / "trust/host-observation-attestation.pub",
        expected_id=str(package["collector_public_key_ids"]["host"]),
    )
    request, unsigned_observation = _validate_request(
        request_value,
        revision=release_revision,
        package=package,
        host_public_key=host_key,
        now_unix=now_unix,
    )
    try:
        config = trusted.load_cloud_attestor_config()
        private_key, public_key, key_id = trusted._load_private_key(config)
    except trusted.TrustedObservationError as exc:
        _error("owner_gate_cloud_signer_key_unavailable", exc)
    if key_id != package["collector_public_key_ids"]["cloud"]:
        _error("owner_gate_cloud_signer_key_invalid")
    report = {
        **unsigned_observation,
        "report_sha256": foundation.sha256_json(unsigned_observation),
    }
    signature = private_key.sign(_canonical(report))
    response = {
        **report,
        "attestation": {
            "schema": ATTESTATION_SCHEMA,
            "public_key_id": key_id,
            "signature_ed25519_b64url": base64
            .urlsafe_b64encode(signature)
            .rstrip(b"=")
            .decode("ascii"),
        },
    }
    checked = _validate_response(
        response,
        unsigned_observation=unsigned_observation,
        public_key=public_key,
        expected_key_id=key_id,
        phase=str(request["phase"]),
    )
    return _store_or_replay(
        checked,
        request_sha256=str(request["request_sha256"]),
        config=config,
        public_key=public_key,
        unsigned_observation=unsigned_observation,
        expected_key_id=key_id,
        phase=str(request["phase"]),
    )


def _read_stdin(stream: BinaryIO) -> Mapping[str, Any]:
    raw = bytearray(stream.read(MAX_FRAME_BYTES + 2))
    try:
        if (
            not raw
            or len(raw) > MAX_FRAME_BYTES + 1
            or raw[-1:] != b"\n"
            or b"\n" in raw[:-1]
        ):
            _error("owner_gate_cloud_signer_stdin_invalid")
        return _decode_canonical(
            bytes(raw[:-1]),
            maximum=MAX_FRAME_BYTES,
            code="owner_gate_cloud_signer_stdin_invalid",
        )
    finally:
        for index in range(len(raw)):
            raw[index] = 0


def _write_stdout(stream: BinaryIO, value: Mapping[str, Any]) -> None:
    raw = _canonical(value) + b"\n"
    if len(raw) > MAX_FRAME_BYTES + 1:
        _error("owner_gate_cloud_signer_stdout_invalid")
    stream.write(raw)
    stream.flush()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments:
        _error("owner_gate_cloud_signer_argv_invalid")
    revision = _runtime_revision()
    request = _read_stdin(sys.stdin.buffer)
    response = sign_request(
        request,
        release_revision=revision,
        now_unix=int(time.time()),
    )
    _write_stdout(sys.stdout.buffer, response)
    return 0


__all__ = [
    "OwnerGateCloudObservationSignerError",
    "main",
]
