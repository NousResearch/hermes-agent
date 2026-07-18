#!/usr/bin/env python3
"""Fixed-command owner IAP transport for the owner-gate stage-zero streams.

There is deliberately no generic remote-command surface in this module.  It
materializes one locally pinned receiver, streams the exact stage-zero kit and
signed bundle over stdin, compares every remote receipt byte-for-byte with a
locally derived receipt, and stops before any service activation or Cloud
mutation.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import selectors
import shlex
import signal
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Protocol, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_author_journal as author_journal
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_foundation_apply as foundation_apply
from scripts.canary import owner_gate_outer_stage0 as outer
from scripts.canary import owner_gate_preflight as owner_preflight
from scripts.canary import owner_gate_pre_foundation as pre_foundation
from scripts.canary import owner_gate_stage0 as cloud_stage0
from scripts.canary import owner_gate_trust as release_trust
from scripts.canary import trusted_signer_author as signer_author
from scripts.canary import trusted_signer_provisioning as signer_provisioning


TRANSPORT_RECEIPT_SCHEMA = "muncho-owner-gate-iap-stage0-transport.v2"
INERT_CLOUD_BUNDLE_TERMINAL_SCHEMA = (
    "muncho-owner-gate-inert-cloud-bundle-terminal.v1"
)
MAX_STDOUT_BYTES = 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024
MAX_CLOUD_RECEIPT_BYTES = 256 * 1024
MAX_OBSERVATION_FRAME_BYTES = 1024 * 1024
MAX_SEALER_BYTES = 128 * 1024 * 1024
MAX_STREAM_BYTES = (
    len(outer.TREE_STREAM_MAGIC)
    + 8
    + outer.MAX_MANIFEST_BYTES
    + outer.MAX_TREE_BYTES
)
_SHA256 = launcher._SHA256
_REVISION = launcher._RELEASE_SHA
_ED25519_SIGNATURE_B64URL = re.compile(r"^[A-Za-z0-9_-]{86}$")
_FOLDER_RESOURCE = re.compile(r"^folders/[1-9][0-9]{5,30}$")
_ORGANIZATION_RESOURCE = re.compile(
    r"^organizations/[1-9][0-9]{5,30}$"
)
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{5,30}$")
_ATTACHED_SA_PROBE_SCHEMA = (
    "muncho-owner-gate-attached-sa-permission-probe.v1"
)
_ATTACHED_SA_REQUEST_SCHEMA = (
    "muncho-owner-gate-attached-sa-permission-probe-request.v1"
)
_ATTACHED_SA_REQUEST_FIELDS = frozenset({
    "schema",
    "phase",
    "collected_at_unix",
    "plan_sha256",
    "cloud_install_receipt",
    "cloud_signer_provisioning_receipt_sha256",
    "cloud_signer_readiness_sha256",
    "host_signer_provisioning_receipt_sha256",
    "host_signer_readiness_sha256",
    "observation_binding_sha256",
    "request_sha256",
})
_ATTACHED_SA_PROBE_FIELDS = frozenset({
    "schema", "phase", "collected_at_unix", "completed_at_unix",
    "fresh_through_unix", "release_revision", "plan_sha256",
    "observation_binding_sha256", "attached_request_sha256",
    "source_tree_oid", "package_sha256", "package_inventory_sha256",
    "pre_foundation_authority_sha256", "foundation_apply_receipt_sha256",
    "project_ancestry_evidence_sha256", "project_ancestry_chain_sha256",
    "resource_ancestor_chain", "install_receipt_sha256",
    "cloud_signer_provisioning_receipt_sha256",
    "cloud_signer_readiness_sha256",
    "host_signer_provisioning_receipt_sha256",
    "host_signer_readiness_sha256", "runtime_instance_numeric_id",
    "runtime_service_account_email", "runtime_service_account_unique_id",
    "metadata_scopes", "effective_permission_probe",
    "target_instance_numeric_id", "target_disk_numeric_id",
    "numeric_targets_reverified", "inherited_bindings_evaluated",
    "conditional_bindings_evaluated", "metadata_token_acquired",
    "metadata_token_wiped", "owner_credential_values_read",
    "report_sha256", "attestation",
})


class StableOuterSealer(Protocol):
    def snapshot(self) -> tuple[bytes, str]: ...


@dataclass(frozen=True)
class RawFoundationChainArtifacts:
    """Untrusted raw signed foundation-A inputs for the public IAP boundary.

    This carrier deliberately contains paths, not a Python validation
    capability or decoded projection.  Constructing it confers no authority;
    the IAP constructor consumes each immutable artifact, cryptographically
    re-decodes foundation A, and loads foundation B only from the fixed
    privileged apply journal.
    """

    pre_foundation_authority_path: Path
    owner_reauthentication_receipt_path: Path
    network_evidence_path: Path
    network_collector_public_key_path: Path
    project_ancestry_evidence_path: Path
    project_ancestry_collector_public_key_path: Path
    release_public_key_path: Path

    def __post_init__(self) -> None:
        paths = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        duplicate_paths = len(paths) - len(set(paths))
        if any(
            not isinstance(path, Path)
            or not path.is_absolute()
            or ".." in path.parts
            or str(path) != os.path.normpath(str(path))
            or os.path.realpath(path) != str(path)
            for path in paths
        ) or (
            duplicate_paths != 0
            and not (
                duplicate_paths == 1
                and self.network_collector_public_key_path
                == self.project_ancestry_collector_public_key_path
                and all(
                    paths.count(path) == 1
                    for path in paths
                    if path != self.network_collector_public_key_path
                )
            )
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_raw_foundation_artifacts_invalid"
            )


@dataclass(frozen=True)
class _FoundationProjection:
    foundation_source_revision: str
    foundation_source_tree_oid: str
    pre_foundation_authority_sha256: str
    foundation_apply_receipt_sha256: str
    project_ancestry_evidence_sha256: str
    project_ancestry_chain_sha256: str
    resource_ancestor_chain: tuple[str, ...]
    interpreter_sha256: str
    owner_reauthentication_receipt_sha256: str


@dataclass(frozen=True)
class _BoundInertCloudBundle:
    source_tree_oid: str
    package_sha256: str
    interpreter_sha256: str
    bootstrap_pip_version: str
    bootstrap_pip_sha256: str
    kit_release_id: str
    trusted_runner_path: str
    bundle_path: str


_HOST_OBSERVATION_HANDOFF_MARKER = object()


@dataclass(frozen=True, init=False)
class OwnerGateHostObservationHandoff:
    """Opaque terminal plus signed HOST observation from the fixed composite."""

    terminal_receipt: Mapping[str, Any]
    host_observation: Mapping[str, Any]
    _marker: object

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "OwnerGateHostObservationHandoff":
        raise launcher.OwnerLauncherError(
            "owner_gate_host_observation_handoff_factory_required"
        )

    @classmethod
    def _create(
        cls,
        *,
        terminal_receipt: Mapping[str, Any],
        host_observation: Mapping[str, Any],
    ) -> "OwnerGateHostObservationHandoff":
        value = object.__new__(cls)
        object.__setattr__(value, "terminal_receipt", terminal_receipt)
        object.__setattr__(value, "host_observation", host_observation)
        object.__setattr__(value, "_marker", _HOST_OBSERVATION_HANDOFF_MARKER)
        return value


class _MutableFrameReader:
    """Read a mutable secret/request frame without retaining an immutable copy."""

    def __init__(self, frame: bytearray) -> None:
        if type(frame) is not bytearray:
            raise launcher.OwnerLauncherError("owner_gate_stage0_mutable_frame_invalid")
        self._frame = frame
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if not isinstance(size, int) or isinstance(size, bool):
            raise launcher.OwnerLauncherError("owner_gate_stage0_mutable_frame_invalid")
        remaining = len(self._frame) - self._offset
        count = remaining if size < 0 else min(size, remaining)
        start = self._offset
        self._offset += count
        return bytes(memoryview(self._frame)[start : start + count])


def _read_foundation_artifact(path: Path, *, maximum: int) -> bytes:
    allowed_modes = {0o400, 0o440, 0o444}
    if _is_fixed_author_journal_artifact(path):
        allowed_modes.add(0o600)
    try:
        return release_trust._read_immutable(
            path,
            maximum=maximum,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
            allowed_modes=frozenset(allowed_modes),
        )
    except release_trust.OwnerGateTrustError as exc:
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_foundation_chain_invalid"
        ) from None


def _is_fixed_author_journal_artifact(path: Path) -> bool:
    root = author_journal.DEFAULT_ROOT
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if (
        len(relative.parts) != 3
        or _REVISION.fullmatch(relative.parts[0]) is None
        or _SHA256.fullmatch(relative.parts[1]) is None
        or relative.parts[2]
        not in {
            "authority.json",
            "owner-reauth.json",
            "network-evidence.json",
            "ancestry-evidence.json",
        }
        or os.path.realpath(path) != str(path)
    ):
        return False
    for directory in (root, root / relative.parts[0], path.parent):
        try:
            item = directory.lstat()
        except OSError:
            return False
        if (
            stat.S_ISLNK(item.st_mode)
            or not stat.S_ISDIR(item.st_mode)
            or item.st_uid != os.geteuid()  # windows-footgun: ok
            or item.st_gid != os.getegid()  # windows-footgun: ok
            or stat.S_IMODE(item.st_mode) != 0o700
        ):
            return False
    return True


def _load_collector_public_key(path: Path) -> Ed25519PublicKey:
    raw = _read_foundation_artifact(path, maximum=16 * 1024)
    try:
        key = (
            Ed25519PublicKey.from_public_bytes(raw)
            if len(raw) == 32
            else serialization.load_pem_public_key(raw)
        )
    except (TypeError, ValueError) as exc:
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_foundation_chain_invalid"
        ) from None
    if not isinstance(key, Ed25519PublicKey):
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_foundation_chain_invalid"
        )
    return key


def _load_foundation_projection(
    artifacts: RawFoundationChainArtifacts,
) -> _FoundationProjection:
    if type(artifacts) is not RawFoundationChainArtifacts:
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_foundation_chain_invalid"
        )
    try:
        release_public_key = pre_foundation.load_pinned_public_key(
            artifacts.release_public_key_path,
            expected_uid=os.geteuid(),  # windows-footgun: ok — POSIX owner boundary
        )
        now_unix = int(time.time())
        if now_unix <= 0:
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_foundation_chain_invalid"
            )
        foundation_a = foundation_apply.decode_validated_foundation_a_chain(
            pre_foundation_authority_raw=_read_foundation_artifact(
                artifacts.pre_foundation_authority_path,
                maximum=foundation_apply.MAX_JSON_BYTES,
            ),
            owner_reauthentication_receipt_raw=_read_foundation_artifact(
                artifacts.owner_reauthentication_receipt_path,
                maximum=foundation_apply.MAX_JSON_BYTES,
            ),
            network_evidence_raw=_read_foundation_artifact(
                artifacts.network_evidence_path,
                maximum=foundation_apply.MAX_JSON_BYTES,
            ),
            project_ancestry_evidence_raw=_read_foundation_artifact(
                artifacts.project_ancestry_evidence_path,
                maximum=foundation_apply.MAX_JSON_BYTES,
            ),
            release_public_key=release_public_key,
            network_collector_public_key=_load_collector_public_key(
                artifacts.network_collector_public_key_path
            ),
            project_ancestry_collector_public_key=(
                _load_collector_public_key(
                    artifacts.project_ancestry_collector_public_key_path
                )
            ),
            now_unix=now_unix,
        )
        chain = foundation_apply.load_validated_foundation_apply_chain(
            foundation_a
        )
        authority = chain.foundation_a.authority
        interpreter = authority["interpreter_image"]
        ancestry_chain = tuple(
            item["resource_name"]
            for item in chain.foundation_a.ancestry_evidence.ordered_chain[1:]
        )
        projection = _FoundationProjection(
            foundation_source_revision=chain.foundation_source_revision,
            foundation_source_tree_oid=chain.foundation_source_tree_oid,
            pre_foundation_authority_sha256=(chain.pre_foundation_authority_sha256),
            foundation_apply_receipt_sha256=(chain.foundation_apply_receipt_sha256),
            project_ancestry_evidence_sha256=(
                chain.foundation_a.ancestry_evidence_sha256
            ),
            project_ancestry_chain_sha256=authority["ancestry_chain_sha256"],
            resource_ancestor_chain=ancestry_chain,
            interpreter_sha256=interpreter["interpreter_sha256"],
            owner_reauthentication_receipt_sha256=(
                chain.owner_reauthentication_receipt_sha256
            ),
        )
    except launcher.OwnerLauncherError:
        raise
    except (
        KeyError,
        TypeError,
        foundation_apply.OwnerGateFoundationApplyError,
        pre_foundation.OwnerGatePreFoundationError,
    ) as exc:
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_foundation_chain_invalid"
        ) from None
    if (
        _REVISION.fullmatch(projection.foundation_source_revision) is None
        or _REVISION.fullmatch(projection.foundation_source_tree_oid) is None
        or _SHA256.fullmatch(projection.pre_foundation_authority_sha256) is None
        or _SHA256.fullmatch(projection.foundation_apply_receipt_sha256) is None
        or _SHA256.fullmatch(projection.project_ancestry_evidence_sha256) is None
        or _SHA256.fullmatch(projection.project_ancestry_chain_sha256) is None
        or _SHA256.fullmatch(projection.interpreter_sha256) is None
        or _SHA256.fullmatch(projection.owner_reauthentication_receipt_sha256) is None
        or not projection.resource_ancestor_chain
        or len(projection.resource_ancestor_chain) > 31
        or len(projection.resource_ancestor_chain)
        != len(set(projection.resource_ancestor_chain))
        or _ORGANIZATION_RESOURCE.fullmatch(projection.resource_ancestor_chain[-1])
        is None
        or any(
            _FOLDER_RESOURCE.fullmatch(item) is None
            for item in projection.resource_ancestor_chain[:-1]
        )
        or interpreter.get("python_version") != "3.11.2"
    ):
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_foundation_chain_invalid"
        )
    return projection


@dataclass(frozen=True)
class _ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class _FixedOperation:
    name: str
    root_argv: tuple[str, ...]
    expected_stdout: bytes
    maximum_input_bytes: int
    timeout_seconds: float


def _canonical(value: Any) -> bytes:
    return outer.canonical_json_bytes(value)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _wipe(frame: bytearray) -> None:
    for index in range(len(frame)):
        frame[index] = 0


def _local_signer_public_identity(
    release_revision: str,
    *,
    role: str,
) -> tuple[Ed25519PublicKey, str]:
    if role not in {"cloud", "host"}:
        raise launcher.OwnerLauncherError("owner_gate_stage0_signer_public_key_invalid")
    try:
        raw = release_trust._read_immutable(
            signer_author._public_path(release_revision, role),
            maximum=32,
            expected_uid=os.geteuid(),  # windows-footgun: ok — owner authority boundary
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
        )
    except (
        release_trust.OwnerGateTrustError,
        signer_author.TrustedSignerAuthorError,
    ):
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_signer_public_key_invalid"
        ) from None
    if len(raw) != 32:
        raise launcher.OwnerLauncherError("owner_gate_stage0_signer_public_key_invalid")
    try:
        return Ed25519PublicKey.from_public_bytes(raw), _sha256(raw)
    except ValueError:
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_signer_public_key_invalid"
        ) from None


def _local_signer_public_key(
    release_revision: str,
    *,
    role: str,
    expected_key_id: str,
) -> Ed25519PublicKey:
    public_key, key_id = _local_signer_public_identity(
        release_revision,
        role=role,
    )
    if _SHA256.fullmatch(expected_key_id) is None or key_id != expected_key_id:
        raise launcher.OwnerLauncherError("owner_gate_stage0_signer_public_key_invalid")
    return public_key


def _signer_readiness_from_receipt(
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    unsigned = {
        "schema": signer_provisioning.READINESS_SCHEMA,
        "role": receipt["role"],
        "release_revision": receipt["release_revision"],
        "package_sha256": receipt["package_sha256"],
        "public_key_id": receipt["public_key_id"],
        "provisioning_receipt_sha256": receipt["receipt_sha256"],
        "private_public_identity_matched": True,
        "config_exact": True,
        "replay_directory_exact": True,
        "sudoers_exact": True,
        "offline_runtime_exact": True,
        "activation_seal_absent": True,
        "current_link_absent": True,
        "services_inactive_disabled": True,
        "activation_performed": False,
        "iam_mutation_performed": False,
    }
    return {**unsigned, "readiness_sha256": foundation.sha256_json(unsigned)}


def _validate_attached_sa_probe_stable_projection(
    probe: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    release_revision: str,
    plan: foundation.OwnerGateFoundationPlan,
    binding: _BoundInertCloudBundle,
    foundation_projection: _FoundationProjection,
    cloud_install_receipt: Mapping[str, Any],
    cloud_receipt: Mapping[str, Any],
    cloud_readiness: Mapping[str, Any],
    host_receipt: Mapping[str, Any],
    host_readiness: Mapping[str, Any],
    host_public_key: Ed25519PublicKey,
) -> Mapping[str, Any]:
    error_code = "owner_gate_attached_sa_probe_invalid"
    try:
        owner_preflight._strict(
            request,
            set(_ATTACHED_SA_REQUEST_FIELDS),
            "attached_sa_probe_request",
        )
        owner_preflight._strict(
            probe,
            set(_ATTACHED_SA_PROBE_FIELDS),
            "attached_sa_probe",
        )
        owner_preflight._verify_seal(probe, label="attached_sa_probe")
        owner_preflight._verify_attestation(
            probe,
            public_key=host_public_key,
            expected_public_key_id=str(plan.spec.host_collector_public_key_id),
            label="attached_sa_probe",
        )
    except owner_preflight.OwnerGatePreflightError:
        raise launcher.OwnerLauncherError(error_code) from None
    completed_at_unix = probe.get("completed_at_unix")
    fresh_through_unix = probe.get("fresh_through_unix")
    collected_at_unix = request.get("collected_at_unix")
    attestation = probe.get("attestation")
    expected_signer_lineage = {
        "cloud_signer_provisioning_receipt_sha256": cloud_receipt.get(
            "receipt_sha256"
        ),
        "cloud_signer_readiness_sha256": cloud_readiness.get(
            "readiness_sha256"
        ),
        "host_signer_provisioning_receipt_sha256": host_receipt.get(
            "receipt_sha256"
        ),
        "host_signer_readiness_sha256": host_readiness.get(
            "readiness_sha256"
        ),
    }
    request_binding = {
        "phase": request.get("phase"),
        "collected_at_unix": collected_at_unix,
        "plan_sha256": request.get("plan_sha256"),
        "cloud_install_receipt": request.get("cloud_install_receipt"),
        **expected_signer_lineage,
    }
    request_unsigned = {
        "schema": _ATTACHED_SA_REQUEST_SCHEMA,
        **request_binding,
        "observation_binding_sha256": foundation.sha256_json(
            request_binding
        ),
    }
    expected_release_lineage = {
        "pre_foundation_authority_sha256": (
            foundation_projection.pre_foundation_authority_sha256
        ),
        "foundation_apply_receipt_sha256": (
            foundation_projection.foundation_apply_receipt_sha256
        ),
        "project_ancestry_evidence_sha256": (
            foundation_projection.project_ancestry_evidence_sha256
        ),
        "project_ancestry_chain_sha256": (
            foundation_projection.project_ancestry_chain_sha256
        ),
    }
    if (
        probe.get("schema") != _ATTACHED_SA_PROBE_SCHEMA
        or request.get("schema") != _ATTACHED_SA_REQUEST_SCHEMA
        or request.get("phase") not in {"inert", "post_iam"}
        or type(plan) is not foundation.OwnerGateFoundationPlan
        or plan.spec.release_revision != release_revision
        or not plan.spec.final_release_bound
        or _REVISION.fullmatch(release_revision) is None
        or probe.get("phase") != request.get("phase")
        or type(collected_at_unix) is not int
        or collected_at_unix <= 0
        or probe.get("collected_at_unix") != collected_at_unix
        or type(completed_at_unix) is not int
        or type(fresh_through_unix) is not int
        or completed_at_unix < collected_at_unix
        or completed_at_unix > fresh_through_unix
        or fresh_through_unix
        != collected_at_unix + owner_preflight.HOST_OBSERVATION_FRESHNESS_SECONDS
        or request.get("observation_binding_sha256")
        != request_unsigned["observation_binding_sha256"]
        or request.get("request_sha256")
        != foundation.sha256_json(request_unsigned)
        or probe.get("release_revision") != release_revision
        or probe.get("plan_sha256") != plan.sha256
        or probe.get("plan_sha256") != request.get("plan_sha256")
        or probe.get("observation_binding_sha256")
        != request.get("observation_binding_sha256")
        or probe.get("attached_request_sha256")
        != request.get("request_sha256")
        or probe.get("source_tree_oid") != binding.source_tree_oid
        or probe.get("package_sha256") != binding.package_sha256
        or probe.get("package_inventory_sha256")
        != plan.spec.package_inventory_sha256
        or any(
            probe.get(name) != value
            for name, value in expected_release_lineage.items()
        )
        or probe.get("resource_ancestor_chain")
        != list(foundation_projection.resource_ancestor_chain)
        or request.get("cloud_install_receipt") != cloud_install_receipt
        or probe.get("install_receipt_sha256")
        != cloud_install_receipt.get("receipt_sha256")
        or any(
            request.get(name) != value or probe.get(name) != value
            for name, value in expected_signer_lineage.items()
        )
        or _NUMERIC_ID.fullmatch(
            str(probe.get("runtime_instance_numeric_id", ""))
        ) is None
        or probe.get("runtime_service_account_email")
        != (
            f"{foundation.SERVICE_ACCOUNT_NAME}@{foundation.PROJECT}."
            "iam.gserviceaccount.com"
        )
        or _NUMERIC_ID.fullmatch(
            str(probe.get("runtime_service_account_unique_id", ""))
        ) is None
        or probe.get("metadata_scopes")
        != list(foundation.OWNER_GATE_OAUTH_SCOPES)
        or probe.get("effective_permission_probe")
        != owner_preflight.expected_effective_permission_probe(
            request.get("phase") == "post_iam"
        )
        or probe.get("target_instance_numeric_id")
        != foundation.TARGET_INSTANCE_ID
        or probe.get("target_disk_numeric_id") != foundation.TARGET_DISK_ID
        or probe.get("numeric_targets_reverified")
        is not (request.get("phase") == "post_iam")
        or any(
            probe.get(name) is not True
            for name in (
                "inherited_bindings_evaluated",
                "conditional_bindings_evaluated",
                "metadata_token_acquired",
                "metadata_token_wiped",
            )
        )
        or probe.get("owner_credential_values_read") is not False
        or not isinstance(attestation, Mapping)
    ):
        raise launcher.OwnerLauncherError(error_code)
    stable = {
        name: item
        for name, item in probe.items()
        if name not in {"completed_at_unix", "report_sha256", "attestation"}
    }
    stable["attestation"] = {
        "schema": attestation["schema"],
        "public_key_id": attestation["public_key_id"],
    }
    return stable


def _select_stable_attached_sa_probe(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    **validation: Any,
) -> Mapping[str, Any]:
    first_stable = _validate_attached_sa_probe_stable_projection(
        first,
        **validation,
    )
    second_stable = _validate_attached_sa_probe_stable_projection(
        second,
        **validation,
    )
    if _canonical(first_stable) != _canonical(second_stable):
        raise launcher.OwnerLauncherError("owner_gate_attached_sa_probe_unstable")
    return (
        second
        if second["completed_at_unix"] >= first["completed_at_unix"]
        else first
    )


def _terminate(
    process: subprocess.Popen[bytes],
    *,
    kill_process_group: Callable[[int, int], None] = os.killpg,  # windows-footgun: ok — POSIX process boundary
) -> None:
    try:
        if process.poll() is None:
            try:
                kill_process_group(process.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                try:
                    process.terminate()
                except OSError:
                    pass
            try:
                process.wait(timeout=5.0)
            except (OSError, subprocess.SubprocessError):
                pass
        if process.poll() is None:
            try:
                kill_process_group(process.pid, signal.SIGKILL)  # windows-footgun: ok — POSIX process boundary
            except (OSError, ProcessLookupError):
                try:
                    process.kill()
                except OSError:
                    pass
            try:
                process.wait(timeout=5.0)
            except (OSError, subprocess.SubprocessError):
                pass
    finally:
        for name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, name, None)
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except OSError:
                pass


def _bounded_process_exchange(
    argv: Sequence[str],
    environment: Mapping[str, str],
    input_source: BinaryIO,
    *,
    maximum_input_bytes: int,
    maximum_stdout_bytes: int = MAX_STDOUT_BYTES,
    maximum_stderr_bytes: int = MAX_STDERR_BYTES,
    timeout_seconds: float,
    popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    process_terminator: Callable[[subprocess.Popen[bytes]], None] = _terminate,
) -> _ProcessResult:
    if (
        not argv
        or any(not isinstance(item, str) or not item for item in argv)
        or isinstance(maximum_input_bytes, bool)
        or not isinstance(maximum_input_bytes, int)
        or maximum_input_bytes < 0
        or maximum_input_bytes > MAX_STREAM_BYTES
        or not 0 < maximum_stdout_bytes <= MAX_STDOUT_BYTES
        or not 0 < maximum_stderr_bytes <= MAX_STDERR_BYTES
        or isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 0 < timeout_seconds <= 2_400
    ):
        raise launcher.OwnerLauncherError("owner_gate_stage0_iap_exchange_invalid")
    try:
        process = popen_factory(
            tuple(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(environment),
            shell=False,
            start_new_session=True,
            bufsize=0,
        )
    except (OSError, subprocess.SubprocessError):
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_iap_unavailable"
        ) from None
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process_terminator(process)
        raise launcher.OwnerLauncherError("owner_gate_stage0_iap_unavailable")
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    pending = memoryview(b"")
    input_count = 0
    input_eof = False
    input_open = True
    output_open = {"stdout": True, "stderr": True}
    deadline = time.monotonic() + float(timeout_seconds)
    try:
        descriptors = {
            "stdin": process.stdin.fileno(),
            "stdout": process.stdout.fileno(),
            "stderr": process.stderr.fileno(),
        }
        for descriptor in descriptors.values():
            os.set_blocking(descriptor, False)
        selector.register(descriptors["stdin"], selectors.EVENT_WRITE, "stdin")
        selector.register(descriptors["stdout"], selectors.EVENT_READ, "stdout")
        selector.register(descriptors["stderr"], selectors.EVENT_READ, "stderr")
        while True:
            if input_open and not pending and not input_eof:
                chunk = input_source.read(64 * 1024)
                if not isinstance(chunk, bytes):
                    raise launcher.OwnerLauncherError(
                        "owner_gate_stage0_iap_input_invalid"
                    )
                if chunk:
                    input_count += len(chunk)
                    if input_count > maximum_input_bytes:
                        raise launcher.OwnerLauncherError(
                            "owner_gate_stage0_iap_input_oversized"
                        )
                    pending = memoryview(chunk)
                else:
                    input_eof = True
                    selector.unregister(descriptors["stdin"])
                    process.stdin.close()
                    input_open = False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_iap_timeout"
                )
            for key, _mask in selector.select(min(remaining, 0.25)):
                if key.data == "stdin":
                    if not pending:
                        continue
                    try:
                        written = os.write(key.fd, pending[: 64 * 1024])
                    except BlockingIOError:
                        continue
                    except OSError:
                        raise launcher.OwnerLauncherError(
                            "owner_gate_stage0_iap_stdin_failed"
                        ) from None
                    if written <= 0:
                        raise launcher.OwnerLauncherError(
                            "owner_gate_stage0_iap_stdin_failed"
                        )
                    pending = pending[written:]
                    continue
                try:
                    chunk = os.read(key.fd, 64 * 1024)
                except BlockingIOError:
                    continue
                except OSError:
                    raise launcher.OwnerLauncherError(
                        "owner_gate_stage0_iap_output_failed"
                    ) from None
                target = stdout if key.data == "stdout" else stderr
                maximum = (
                    maximum_stdout_bytes
                    if key.data == "stdout"
                    else maximum_stderr_bytes
                )
                if chunk:
                    target.extend(chunk)
                    if len(target) > maximum:
                        raise launcher.OwnerLauncherError(
                            f"owner_gate_stage0_iap_{key.data}_oversized"
                        )
                else:
                    selector.unregister(key.fd)
                    getattr(process, key.data).close()
                    output_open[key.data] = False
            if process.poll() is not None and not any(output_open.values()):
                break
        if input_open or pending or not input_eof:
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_iap_stdin_failed"
            )
        try:
            returncode = process.wait(max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_iap_timeout"
            ) from None
        return _ProcessResult(returncode, bytes(stdout), bytes(stderr))
    except BaseException:
        process_terminator(process)
        raise
    finally:
        pending.release()
        selector.close()
        if process.poll() is not None:
            process_terminator(process)


class PinnedExactTreeStream:
    """Stable local stream identity checked before and after IAP transfer."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        purpose: str,
        release_id: str,
        expected_manifest_sha256: str,
    ) -> None:
        selected = os.path.abspath(os.fspath(path))
        if (
            os.path.realpath(selected) != selected
            or _SHA256.fullmatch(expected_manifest_sha256) is None
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_stream_identity_invalid"
            )
        self.path = selected
        self.purpose = purpose
        self.release_id = release_id
        self.expected_manifest_sha256 = expected_manifest_sha256
        self._fingerprint, self.manifest, self.manifest_raw = self._capture()

    @staticmethod
    def _read_exact(descriptor: int, size: int) -> bytes:
        raw = bytearray()
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_stream_identity_invalid"
                )
            raw.extend(chunk)
            remaining -= len(chunk)
        return bytes(raw)

    def _capture(self) -> tuple[tuple[Any, ...], Mapping[str, Any], bytes]:
        descriptor: int | None = None
        try:
            before = os.lstat(self.path)
            descriptor = os.open(
                self.path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(opened.st_mode)
                or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
                or opened.st_uid not in {0, os.getuid()}  # windows-footgun: ok — POSIX owner boundary
                or opened.st_nlink != 1
                or stat.S_IMODE(opened.st_mode) != 0o400
                or opened.st_size < len(outer.TREE_STREAM_MAGIC) + 9
                or opened.st_size > MAX_STREAM_BYTES
            ):
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_stream_identity_invalid"
                )
            magic = self._read_exact(descriptor, len(outer.TREE_STREAM_MAGIC))
            manifest_size = int.from_bytes(self._read_exact(descriptor, 8), "big")
            if magic != outer.TREE_STREAM_MAGIC or not 0 < manifest_size <= outer.MAX_MANIFEST_BYTES:
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_stream_identity_invalid"
                )
            manifest_raw = self._read_exact(descriptor, manifest_size)
            if _sha256(manifest_raw) != self.expected_manifest_sha256:
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_stream_identity_invalid"
                )
            try:
                decoded = json.loads(manifest_raw.decode("utf-8", errors="strict"))
                manifest = outer.validate_tree_stream_manifest(
                    decoded,
                    expected_purpose=self.purpose,
                    expected_release_id=self.release_id,
                )
            except (UnicodeError, ValueError, json.JSONDecodeError, outer.OwnerGateOuterStage0Error):
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_stream_identity_invalid"
                ) from None
            expected_size = (
                len(outer.TREE_STREAM_MAGIC)
                + 8
                + manifest_size
                + sum(item["size"] for item in manifest["files"])
            )
            after = os.fstat(descriptor)
            fingerprint = (
                opened.st_mode,
                opened.st_uid,
                opened.st_gid,
                opened.st_dev,
                opened.st_ino,
                opened.st_nlink,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                self.expected_manifest_sha256,
            )
            if (
                expected_size != opened.st_size
                or _canonical(manifest) != manifest_raw
                or (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                )
                != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
            ):
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_stream_identity_invalid"
                )
            return fingerprint, manifest, manifest_raw
        except OSError:
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_stream_identity_invalid"
            ) from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @property
    def size(self) -> int:
        return int(self._fingerprint[6])

    def open(self) -> BinaryIO:
        fingerprint, manifest, manifest_raw = self._capture()
        if (
            fingerprint != self._fingerprint
            or manifest != self.manifest
            or manifest_raw != self.manifest_raw
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_stream_changed"
            )
        try:
            descriptor = os.open(
                self.path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            opened_fingerprint = (
                opened.st_mode,
                opened.st_uid,
                opened.st_gid,
                opened.st_dev,
                opened.st_ino,
                opened.st_nlink,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                self.expected_manifest_sha256,
            )
            if opened_fingerprint != self._fingerprint:
                os.close(descriptor)
                descriptor = -1
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_stream_changed"
                )
            return os.fdopen(descriptor, "rb", closefd=True)
        except launcher.OwnerLauncherError:
            raise
        except OSError:
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_stream_changed"
            ) from None

    def assert_stable(self) -> None:
        fingerprint, manifest, manifest_raw = self._capture()
        if (
            fingerprint != self._fingerprint
            or manifest != self.manifest
            or manifest_raw != self.manifest_raw
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_stream_changed"
            )

    def member(self, relative: str) -> bytes:
        offset = len(outer.TREE_STREAM_MAGIC) + 8 + len(self.manifest_raw)
        selected: Mapping[str, Any] | None = None
        for item in self.manifest["files"]:
            if item["path"] == relative:
                selected = item
                break
            offset += item["size"]
        if selected is None:
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_stream_member_missing"
            )
        with self.open() as stream:
            stream.seek(offset)
            raw = stream.read(selected["size"])
        if (
            not isinstance(raw, bytes)
            or len(raw) != selected["size"]
            or _sha256(raw) != selected["sha256"]
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_stream_changed"
            )
        self.assert_stable()
        return raw


class TrustedOuterSealerSource:
    """Read the sealer only from the activated exact owner-support release."""

    def __init__(self, release_sha: str) -> None:
        if _REVISION.fullmatch(release_sha) is None:
            raise launcher.OwnerLauncherError("owner_gate_stage0_sealer_invalid")
        runtime = launcher.require_trusted_owner_runtime(release_sha)
        launcher.activate_trusted_owner_support(runtime, release_sha=release_sha)
        _root, source_root, _site = launcher._trusted_owner_support_paths(release_sha)
        self._release_sha = release_sha
        self._path = os.path.join(
            source_root,
            "scripts/canary/owner_gate_outer_stage0.py",
        )
        self._fingerprint, self._payload = self._capture()

    def _capture(self) -> tuple[tuple[Any, ...], bytes]:
        launcher.require_local_launcher_provenance(self._release_sha)
        fingerprint, payload = launcher._read_pinned_regular_file(
            self._path,
            maximum=MAX_SEALER_BYTES,
            unavailable_code="owner_gate_stage0_sealer_unavailable",
            invalid_code="owner_gate_stage0_sealer_invalid",
            changed_code="owner_gate_stage0_sealer_changed",
            allowed_owners=frozenset({0, os.getuid()}),  # windows-footgun: ok — POSIX owner boundary
        )
        if stat.S_IMODE(int(fingerprint[0])) not in {0o400, 0o444}:
            raise launcher.OwnerLauncherError("owner_gate_stage0_sealer_invalid")
        return fingerprint, payload

    def snapshot(self) -> tuple[bytes, str]:
        fingerprint, payload = self._capture()
        if fingerprint != self._fingerprint or payload != self._payload:
            raise launcher.OwnerLauncherError("owner_gate_stage0_sealer_changed")
        return payload, _sha256(payload)


def _tree_projection(manifest: Mapping[str, Any]) -> tuple[str, int]:
    directories = {item["path"]: item for item in manifest["directories"]}
    files = {item["path"]: item for item in manifest["files"]}
    projection: list[Mapping[str, Any]] = []
    for relative in sorted((*directories, *files)):
        if relative in directories:
            projection.append({
                "path": relative,
                "mode": "0555",
                "type": "directory",
            })
        else:
            item = files[relative]
            projection.append({
                "path": relative,
                "mode": item["mode"],
                "type": "file",
                "size": item["size"],
                "sha256": item["sha256"],
            })
    return outer.sha256_json(projection), len(projection)


def expected_tree_receipt(
    stream: PinnedExactTreeStream,
    *,
    receiver_self_sha256: str,
) -> Mapping[str, Any]:
    projection_sha256, projection_count = _tree_projection(stream.manifest)
    base = (
        outer.INCOMING_BASE
        if stream.purpose == "outer-stage0-kit"
        else outer.BUNDLE_INCOMING_BASE
    )
    unsigned = {
        "schema": outer.TREE_RECEIPT_SCHEMA,
        "purpose": stream.purpose,
        "release_id": stream.release_id,
        "stream_manifest_sha256": stream.expected_manifest_sha256,
        "transport_manifest_sha256": stream.manifest[
            "transport_manifest_sha256"
        ],
        "source_tree_projection_sha256": stream.manifest[
            "source_tree_projection_sha256"
        ],
        "receiver_self_sha256": receiver_self_sha256,
        "received_tree": {
            "path": str(base / stream.release_id),
            "uid": 0,
            "gid": 0,
            "mode": "0555",
            "projection_sha256": projection_sha256,
            "projection_count": projection_count,
        },
        "input_code_executed": False,
        "input_code_imported": False,
        "symlinks_received": False,
        "special_files_received": False,
        "extra_paths_received": False,
    }
    return {**unsigned, "receipt_sha256": outer.sha256_json(unsigned)}


def expected_seal_receipt(
    kit_stream: PinnedExactTreeStream,
    *,
    receiver_self_sha256: str,
) -> Mapping[str, Any]:
    manifest_raw = kit_stream.member("outer-stage0-manifest.json")
    if _sha256(manifest_raw) != kit_stream.release_id:
        raise launcher.OwnerLauncherError("owner_gate_stage0_kit_authority_invalid")
    try:
        manifest = outer.validate_manifest(
            json.loads(manifest_raw.decode("utf-8", errors="strict"))
        )
    except (UnicodeError, ValueError, json.JSONDecodeError, outer.OwnerGateOuterStage0Error):
        raise launcher.OwnerLauncherError(
            "owner_gate_stage0_kit_authority_invalid"
        ) from None
    if _canonical(manifest) != manifest_raw:
        raise launcher.OwnerLauncherError("owner_gate_stage0_kit_authority_invalid")
    files = {
        **{item["path"]: item for item in manifest["files"]},
        "outer-stage0-manifest.json": {
            "sha256": _sha256(manifest_raw),
            "size": len(manifest_raw),
            "mode": "0444",
        },
    }
    directories = {
        str(parent)
        for relative in files
        for parent in Path(relative).parents
        if str(parent) != "."
    }
    projection: list[Mapping[str, Any]] = []
    for relative in sorted((*directories, *files)):
        if relative in directories:
            projection.append({
                "path": relative,
                "type": "directory",
                "mode": "0555",
            })
        else:
            item = files[relative]
            projection.append({
                "path": relative,
                "type": "file",
                "mode": item["mode"],
                "size": item["size"],
                "sha256": item["sha256"],
            })
    release = outer.RELEASE_BASE / kit_stream.release_id
    unsigned = {
        "schema": outer.RECEIPT_SCHEMA,
        "kit_manifest_sha256": kit_stream.release_id,
        "kit_self_hash": manifest["kit_manifest_sha256"],
        "source_release_revision": manifest["source_release_revision"],
        "source_tree_oid": manifest["source_tree_oid"],
        "outer_sealer_sha256": receiver_self_sha256,
        "trusted_runner": str(release / outer.TRUSTED_RUNNER),
        "release": {
            "path": str(release),
            "uid": 0,
            "gid": 0,
            "mode": "0555",
            "projection_sha256": outer.sha256_json(projection),
            "projection_count": len(projection),
        },
        "incoming_payload_code_executed": False,
        "incoming_payload_imported": False,
        "network_fetch_performed": False,
        "generic_shell_runtime_added": False,
    }
    return {**unsigned, "receipt_sha256": outer.sha256_json(unsigned)}


def _decode_canonical_mapping(
    raw: bytes,
    *,
    maximum: int,
    error_code: str,
) -> Mapping[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        raise launcher.OwnerLauncherError(error_code)
    try:
        value = json.loads(
            raw.decode("ascii", errors="strict"),
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
        canonical = _canonical(value)
    except (UnicodeError, TypeError, ValueError, json.JSONDecodeError):
        raise launcher.OwnerLauncherError(error_code) from None
    if not isinstance(value, dict) or canonical != raw:
        raise launcher.OwnerLauncherError(error_code)
    return value


def _decode_canonical_stdout(
    raw: bytes,
    *,
    error_code: str,
) -> Mapping[str, Any]:
    if (
        type(raw) is not bytes
        or not raw.endswith(b"\n")
        or raw == b"\n"
        or b"\n" in raw[:-1]
    ):
        raise launcher.OwnerLauncherError(error_code)
    return _decode_canonical_mapping(
        raw[:-1],
        maximum=MAX_CLOUD_RECEIPT_BYTES,
        error_code=error_code,
    )


def _validate_self_hash(
    value: Mapping[str, Any],
    *,
    field: str,
    error_code: str,
) -> None:
    digest = value.get(field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if (
        _SHA256.fullmatch(str(digest or "")) is None
        or digest != outer.sha256_json(unsigned)
    ):
        raise launcher.OwnerLauncherError(error_code)


_VERIFY_RECEIPT_FIELDS = frozenset({
    "schema",
    "release_revision",
    "package_sha256",
    "verified",
    "incoming_payload_code_executed_before_verification",
    "receipt_sha256",
})
_PREFLIGHT_RECEIPT_FIELDS = frozenset({
    "schema",
    "release_revision",
    "python_version",
    "python_sha256",
    "openssl_version",
    "openssl_ed25519_rawin_verified",
    "systemd_version",
    "systemd_sysusers_available",
    "systemd_tmpfiles_available",
    "python_venv_available",
    "python_venv_without_pip_available",
    "bootstrap_pip_version",
    "bootstrap_pip_sha256",
    "executable_identities_sha256",
    "network_install_required",
    "cloud_mutation_performed",
    "activation_performed",
    "preflight_sha256",
})
_INSTALL_PHASES_WITHOUT_RECEIPT = (
    "reverify_bundle_and_runtime",
    "install_fixed_identities_and_directories",
    "generate_or_verify_authority_receipt_key",
    "install_root_owned_configuration_units_firewall_and_hosts",
    "bootstrap_and_verify_canonical_databases",
    "seal_and_publish_immutable_release",
)
_INSTALL_RECEIPT_FIELDS = frozenset({
    "schema",
    "release_revision",
    "package_sha256",
    "source_tree_oid",
    "pre_foundation_authority_sha256",
    "foundation_apply_receipt_sha256",
    "project_ancestry_evidence_sha256",
    "project_ancestry_chain_sha256",
    "resource_ancestor_chain",
    "installed_at_unix",
    "release_path",
    "release_tree_sha256",
    "transaction_prefix_sha256",
    "phase_evidence_sha256",
    "authority_receipt_public_key_sha256",
    "authority_receipt_public_key_id",
    "credential_id_sha256",
    "executor_hosts_receipt_sha256",
    "current_release_selected",
    "systemd_units_enabled",
    "activation_performed",
    "activation_seal_created",
    "iam_binding_created",
    "cloud_mutation_performed",
    "caddy_cutover_performed",
    "receipt_sha256",
    "signer_key_id",
    "signature_ed25519_b64url",
})
_HOST_RUNTIME_RECEIPT_FIELDS = frozenset({
    "schema",
    "release_revision",
    "package_sha256",
    "preflight_sha256",
    "release",
    "sudoers",
    "runtime_inventory_sha256",
    "runtime_interpreter",
    "host_attestor_entrypoint",
    "host_provisioner_entrypoint",
    "offline_runtime",
    "network_install_required",
    "generic_usr_bin_python3_runtime",
    "current_link_absent",
    "activation_seal_absent",
    "service_start_performed",
    "service_enablement_mutated",
    "iam_mutation_performed",
    "cloud_mutation_performed",
    "private_key_material_received",
    "private_key_digest_recorded",
    "receipt_sha256",
})
_INERT_TERMINAL_FIELDS = frozenset({
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


class OwnerGateStage0IapTransport(launcher.OwnerGateIapTransport):
    """Fixed root bootstrap operations over the already pinned IAP identity."""

    _SEALER_ROOT = "/run/muncho-owner-gate-stage0-bootstrap"

    def __init__(
        self,
        *,
        release_sha: str,
        owner_identity: launcher.GcloudOwnerAccessToken,
        gcloud_executable: launcher.TrustedGcloudExecutable,
        gcloud_configuration: launcher.PinnedGcloudConfiguration,
        foundation_artifacts: RawFoundationChainArtifacts,
        sealer_source: StableOuterSealer | None = None,
        host_identity: launcher.StableOwnerGateHostIdentity | None = None,
        known_hosts: launcher.StableKnownHosts | None = None,
        popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        timeout_seconds: float = 900.0,
        exchange: Callable[..., _ProcessResult] | None = None,
    ) -> None:
        projection = _load_foundation_projection(foundation_artifacts)
        super().__init__(
            release_sha=release_sha,
            owner_identity=owner_identity,
            gcloud_executable=gcloud_executable,
            gcloud_configuration=gcloud_configuration,
            host_identity=host_identity,
            known_hosts=known_hosts,
            popen_factory=popen_factory,
            timeout_seconds=timeout_seconds,
        )
        self._foundation = projection
        self._stage0_sealer_source = sealer_source or TrustedOuterSealerSource(
            release_sha
        )
        self._stage0_exchange = exchange or _bounded_process_exchange

    def _bind_inert_cloud_bundle(
        self,
        *,
        kit_stream: PinnedExactTreeStream,
        bundle_stream: PinnedExactTreeStream,
    ) -> _BoundInertCloudBundle:
        error_code = "owner_gate_stage0_stream_pair_invalid"
        if (
            not isinstance(kit_stream, PinnedExactTreeStream)
            or kit_stream.purpose != "outer-stage0-kit"
            or _SHA256.fullmatch(kit_stream.release_id or "") is None
            or kit_stream.manifest.get("release_id") != kit_stream.release_id
            or not isinstance(bundle_stream, PinnedExactTreeStream)
            or bundle_stream.purpose != "owner-gate-bundle"
            or bundle_stream.release_id != self._release_sha
            or bundle_stream.manifest.get("release_id") != self._release_sha
        ):
            raise launcher.OwnerLauncherError(error_code)
        kit_manifest_raw = kit_stream.member("outer-stage0-manifest.json")
        if _sha256(kit_manifest_raw) != kit_stream.release_id:
            raise launcher.OwnerLauncherError(error_code)
        try:
            kit_manifest = outer.validate_manifest(
                _decode_canonical_mapping(
                    kit_manifest_raw,
                    maximum=outer.MAX_MANIFEST_BYTES,
                    error_code=error_code,
                )
            )
        except (
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
            outer.OwnerGateOuterStage0Error,
        ):
            raise launcher.OwnerLauncherError(error_code) from None
        package_raw = bundle_stream.member("package-manifest.json")
        package = _decode_canonical_mapping(
            package_raw,
            maximum=cloud_stage0.MAX_JSON_BYTES,
            error_code=error_code,
        )
        if frozenset(package) != cloud_stage0.MANIFEST_FIELDS:
            raise launcher.OwnerLauncherError(error_code)
        _validate_self_hash(
            package,
            field="package_sha256",
            error_code=error_code,
        )
        try:
            bootstrap_pip = cloud_stage0._bootstrap_pip_artifact(package)
        except cloud_stage0.OwnerGateStage0Error:
            raise launcher.OwnerLauncherError(error_code) from None
        source_tree_oid = package.get("source_tree_oid")
        package_sha256 = package.get("package_sha256")
        interpreter_sha256 = package.get("interpreter_sha256")
        lineage = {
            "pre_foundation_authority_sha256": (
                self._foundation.pre_foundation_authority_sha256
            ),
            "foundation_apply_receipt_sha256": (
                self._foundation.foundation_apply_receipt_sha256
            ),
            "project_ancestry_evidence_sha256": (
                self._foundation.project_ancestry_evidence_sha256
            ),
            "project_ancestry_chain_sha256": (
                self._foundation.project_ancestry_chain_sha256
            ),
            "resource_ancestor_chain": list(
                self._foundation.resource_ancestor_chain
            ),
        }
        if (
            package.get("schema") != cloud_stage0.PACKAGE_SCHEMA
            or package.get("release_revision") != self._release_sha
            or self._foundation.foundation_source_revision
            != self._release_sha
            or self._foundation.foundation_source_tree_oid
            != source_tree_oid
            or _REVISION.fullmatch(str(source_tree_oid or "")) is None
            or kit_manifest.get("source_release_revision")
            != self._release_sha
            or kit_manifest.get("source_tree_oid") != source_tree_oid
            or _SHA256.fullmatch(str(package_sha256 or "")) is None
            or interpreter_sha256 != self._foundation.interpreter_sha256
            or package.get("release_root")
            != str(cloud_stage0.RELEASE_BASE / self._release_sha)
            or package.get("activation_performed") is not False
            or package.get("cloud_mutation_performed") is not False
            or package.get("caller_self_hash_is_authority") is not False
            or any(package.get(name) != value for name, value in lineage.items())
        ):
            raise launcher.OwnerLauncherError(error_code)
        kit_stream.assert_stable()
        bundle_stream.assert_stable()
        return _BoundInertCloudBundle(
            source_tree_oid=str(source_tree_oid),
            package_sha256=str(package_sha256),
            interpreter_sha256=str(interpreter_sha256),
            bootstrap_pip_version=str(bootstrap_pip["version"]),
            bootstrap_pip_sha256=str(bootstrap_pip["sha256"]),
            kit_release_id=kit_stream.release_id,
            trusted_runner_path=str(
                outer.RELEASE_BASE / kit_stream.release_id / outer.TRUSTED_RUNNER
            ),
            bundle_path=str(outer.BUNDLE_INCOMING_BASE / bundle_stream.release_id),
        )

    def _attest_remote_interpreter(self) -> Mapping[str, Any]:
        digest = self._foundation.interpreter_sha256
        expected = (
            (
                "python_link",
                ("/usr/bin/readlink", "--", "/usr/bin/python3"),
                b"python3.11\n",
            ),
            (
                "python_link_identity",
                (
                    "/usr/bin/stat",
                    "--format=%F|%u|%g|%a|%h",
                    "--",
                    "/usr/bin/python3",
                ),
                b"symbolic link|0|0|777|1\n",
            ),
            (
                "python_target_identity",
                (
                    "/usr/bin/stat",
                    "--format=%F|%u|%g|%a|%h",
                    "--",
                    "/usr/bin/python3.11",
                ),
                b"regular file|0|0|755|1\n",
            ),
            (
                "python_target_digest",
                ("/usr/bin/sha256sum", "--", "/usr/bin/python3.11"),
                f"{digest}  /usr/bin/python3.11\n".encode("ascii"),
            ),
        )
        for name, argv, stdout in expected:
            self._execute_empty(
                _FixedOperation(
                    f"{name}_before",
                    argv,
                    stdout,
                    0,
                    60.0,
                )
            )
        self._execute_empty(
            _FixedOperation(
                "python_version_after_digest",
                ("/usr/bin/python3", "--version"),
                b"Python 3.11.2\n",
                0,
                60.0,
            )
        )
        for name, argv, stdout in expected:
            self._execute_empty(
                _FixedOperation(
                    f"{name}_after",
                    argv,
                    stdout,
                    0,
                    60.0,
                )
            )
        unsigned = {
            "schema": "muncho-owner-gate-host-interpreter-attestation.v1",
            "path": "/usr/bin/python3",
            "link_target": "python3.11",
            "resolved_path": "/usr/bin/python3.11",
            "target_sha256": digest,
            "python_version": "3.11.2",
            "pre_foundation_authority_sha256": (
                self._foundation.pre_foundation_authority_sha256
            ),
            "foundation_apply_receipt_sha256": (
                self._foundation.foundation_apply_receipt_sha256
            ),
            "project_ancestry_evidence_sha256": (
                self._foundation.project_ancestry_evidence_sha256
            ),
            "project_ancestry_chain_sha256": (
                self._foundation.project_ancestry_chain_sha256
            ),
            "resource_ancestor_chain": list(
                self._foundation.resource_ancestor_chain
            ),
            "identity_stable_before_after_version_probe": True,
            "python_executed_before_digest_match": False,
        }
        return {**unsigned, "attestation_sha256": outer.sha256_json(unsigned)}

    @staticmethod
    def _sealer_paths(sealer_sha256: str) -> tuple[str, str, str]:
        if _SHA256.fullmatch(sealer_sha256) is None:
            raise launcher.OwnerLauncherError("owner_gate_stage0_sealer_invalid")
        directory = f"{OwnerGateStage0IapTransport._SEALER_ROOT}/{sealer_sha256}"
        staging = f"{directory}/.owner_gate_outer_stage0.py.uploading"
        final = f"{directory}/owner_gate_outer_stage0.py"
        return directory, staging, final

    def _root_argv(
        self,
        snapshot: tuple[Any, ...],
        operation: _FixedOperation,
    ) -> tuple[str, ...]:
        (
            prefix,
            account,
            _launcher_sha256,
            known_hosts,
            private_key,
            _public_key,
            host_identity,
            _server_host_key,
        ) = snapshot
        if not isinstance(host_identity, launcher.OwnerGateHostIdentitySnapshot):
            raise launcher.OwnerLauncherError(
                "owner_gate_iap_identity_receipt_invalid"
            )
        root_command = (
            "/usr/bin/sudo",
            "--non-interactive",
            "--",
            *operation.root_argv,
        )
        remote_command = shlex.join(root_command)
        ssh_flags = self._sealed_ssh_flags(
            known_hosts,
            private_key,
            host_identity.vm_numeric_id,
        )
        expected = (
            *prefix,
            "compute",
            "ssh",
            f"{launcher.OS_LOGIN_USERNAME}@{self._VM_NAME}",
            f"--project={launcher.PROJECT}",
            f"--zone={launcher.ZONE}",
            f"--account={account}",
            "--plain",
            "--tunnel-through-iap",
            "--quiet",
            f"--command={remote_command}",
            *ssh_flags,
        )
        argv = tuple(expected)
        if (
            account != self._OWNER_ACCOUNT
            or argv != expected
            or argv[len(prefix) : len(prefix) + 9]
            != (
                "compute",
                "ssh",
                f"{launcher.OS_LOGIN_USERNAME}@{self._VM_NAME}",
                f"--project={launcher.PROJECT}",
                f"--zone={launcher.ZONE}",
                f"--account={self._OWNER_ACCOUNT}",
                "--plain",
                "--tunnel-through-iap",
                "--quiet",
            )
            or argv[-len(ssh_flags) :] != ssh_flags
        ):
            raise launcher.OwnerLauncherError("owner_gate_stage0_iap_argv_invalid")
        return argv

    def _cloud_observation_signer_argv(
        self,
        snapshot: tuple[Any, ...],
    ) -> tuple[str, ...]:
        (
            prefix,
            account,
            _launcher_sha256,
            known_hosts,
            private_key,
            _public_key,
            host_identity,
            _server_host_key,
        ) = snapshot
        if not isinstance(host_identity, launcher.OwnerGateHostIdentitySnapshot):
            raise launcher.OwnerLauncherError("owner_gate_iap_identity_receipt_invalid")
        release = f"/opt/muncho-owner-gate/releases/{self._release_sha}"
        command = (
            "/usr/bin/sudo",
            "--non-interactive",
            "--user=muncho-storage-executor",
            "--",
            "/usr/bin/env",
            "-i",
            f"{release}/venv/bin/python",
            "-I",
            "-B",
            f"{release}/bin/muncho-owner-gate-cloud-observation-signer",
        )
        remote_command = shlex.join(command)
        ssh_flags = self._sealed_ssh_flags(
            known_hosts,
            private_key,
            host_identity.vm_numeric_id,
        )
        expected = (
            *prefix,
            "compute",
            "ssh",
            f"{launcher.OS_LOGIN_USERNAME}@{self._VM_NAME}",
            f"--project={launcher.PROJECT}",
            f"--zone={launcher.ZONE}",
            f"--account={account}",
            "--plain",
            "--tunnel-through-iap",
            "--quiet",
            f"--command={remote_command}",
            *ssh_flags,
        )
        argv = tuple(expected)
        if (
            account != self._OWNER_ACCOUNT
            or argv != expected
            or command[0:6]
            != (
                "/usr/bin/sudo",
                "--non-interactive",
                "--user=muncho-storage-executor",
                "--",
                "/usr/bin/env",
                "-i",
            )
            or argv[-len(ssh_flags) :] != ssh_flags
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_cloud_observation_signer_argv_invalid"
            )
        return argv

    def _exchange_cloud_observation_signer(
        self,
        input_source: BinaryIO,
        *,
        maximum_input_bytes: int,
    ) -> Mapping[str, Any]:
        before = self._authority_snapshot()
        argv = self._cloud_observation_signer_argv(before)
        environment = self._environment(before[0])
        try:
            result = self._stage0_exchange(
                argv,
                environment,
                input_source,
                maximum_input_bytes=maximum_input_bytes,
                maximum_stdout_bytes=MAX_OBSERVATION_FRAME_BYTES,
                maximum_stderr_bytes=MAX_STDERR_BYTES,
                timeout_seconds=self._timeout_seconds,
                popen_factory=self._popen_factory,
            )
        finally:
            if self._authority_snapshot() != before:
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_iap_authority_changed"
                )
        if (
            not isinstance(result, _ProcessResult)
            or result.returncode != 0
            or result.stderr != b""
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_cloud_observation_signer_failed"
            )
        return _decode_canonical_stdout(
            result.stdout,
            error_code="owner_gate_cloud_observation_signer_response_invalid",
        )

    def _exchange_fixed_operation(
        self,
        operation: _FixedOperation,
        input_source: BinaryIO,
        *,
        maximum_stdout_bytes: int = MAX_STDOUT_BYTES,
    ) -> _ProcessResult:
        before = self._authority_snapshot()
        argv = self._root_argv(before, operation)
        environment = self._environment(before[0])
        try:
            result = self._stage0_exchange(
                argv,
                environment,
                input_source,
                maximum_input_bytes=operation.maximum_input_bytes,
                maximum_stdout_bytes=maximum_stdout_bytes,
                maximum_stderr_bytes=MAX_STDERR_BYTES,
                timeout_seconds=operation.timeout_seconds,
                popen_factory=self._popen_factory,
            )
        finally:
            after = self._authority_snapshot()
            if after != before:
                raise launcher.OwnerLauncherError(
                    "owner_gate_stage0_iap_authority_changed"
                )
        if (
            not isinstance(result, _ProcessResult)
            or result.returncode != 0
            or result.stderr != b""
        ):
            raise launcher.OwnerLauncherError(
                f"owner_gate_stage0_iap_{operation.name}_failed"
            )
        return result

    def _execute(
        self,
        operation: _FixedOperation,
        input_source: BinaryIO,
    ) -> bytes:
        result = self._exchange_fixed_operation(operation, input_source)
        if result.stdout != operation.expected_stdout:
            raise launcher.OwnerLauncherError(
                f"owner_gate_stage0_iap_{operation.name}_failed"
            )
        return result.stdout

    def _execute_empty(self, operation: _FixedOperation) -> bytes:
        return self._execute(operation, io.BytesIO(b""))

    def _execute_canonical_receipt(
        self,
        operation: _FixedOperation,
        *,
        error_code: str,
    ) -> Mapping[str, Any]:
        result = self._exchange_fixed_operation(
            operation,
            io.BytesIO(b""),
            maximum_stdout_bytes=MAX_CLOUD_RECEIPT_BYTES,
        )
        return _decode_canonical_stdout(
            result.stdout,
            error_code=error_code,
        )

    def _materialize_sealer(self, payload: bytes, sha256: str) -> str:
        if (
            type(payload) is not bytes
            or not payload
            or len(payload) > MAX_SEALER_BYTES
            or _sha256(payload) != sha256
        ):
            raise launcher.OwnerLauncherError("owner_gate_stage0_sealer_invalid")
        directory, staging, final = self._sealer_paths(sha256)
        operations = (
            _FixedOperation(
                "sealer_directory",
                (
                    "/usr/bin/install",
                    "-d",
                    "-o",
                    "root",
                    "-g",
                    "root",
                    "-m",
                    "0700",
                    directory,
                ),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_stale_stage_remove",
                ("/bin/rm", "--force", "--", staging),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_stage",
                (
                    "/usr/bin/dd",
                    f"of={staging}",
                    "bs=65536",
                    "conv=fsync",
                    "oflag=excl,nofollow",
                    "status=none",
                ),
                b"",
                len(payload),
                120.0,
            ),
            _FixedOperation(
                "sealer_stage_owner",
                ("/bin/chown", "root:root", staging),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_stage_mode",
                ("/bin/chmod", "0400", staging),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_stage_sync",
                ("/usr/bin/sync", "-f", staging),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_stage_digest",
                ("/usr/bin/sha256sum", staging),
                f"{sha256}  {staging}\n".encode("ascii"),
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_publish",
                ("/bin/cp", "--no-clobber", "--reflink=never", staging, final),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_owner",
                ("/bin/chown", "root:root", final),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_mode",
                ("/bin/chmod", "0400", final),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_sync",
                ("/usr/bin/sync", "-f", final),
                b"",
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_digest",
                ("/usr/bin/sha256sum", final),
                f"{sha256}  {final}\n".encode("ascii"),
                0,
                60.0,
            ),
            _FixedOperation(
                "sealer_stage_remove",
                ("/bin/rm", "--force", "--", staging),
                b"",
                0,
                60.0,
            ),
        )
        for operation in operations:
            source = (
                io.BytesIO(payload)
                if operation.name == "sealer_stage"
                else io.BytesIO(b"")
            )
            self._execute(operation, source)
        return final

    def _receive_stream(
        self,
        stream: PinnedExactTreeStream,
        *,
        sealer_path: str,
        sealer_sha256: str,
    ) -> Mapping[str, Any]:
        expected = expected_tree_receipt(
            stream,
            receiver_self_sha256=sealer_sha256,
        )
        operation = _FixedOperation(
            f"receive_{stream.purpose.replace('-', '_')}",
            (
                "/usr/bin/python3",
                "-I",
                "-B",
                sealer_path,
                "stream-receive",
                "--purpose",
                stream.purpose,
                "--release-id",
                stream.release_id,
                "--expected-stream-manifest-sha256",
                stream.expected_manifest_sha256,
                "--expected-self-sha256",
                sealer_sha256,
            ),
            _canonical(expected) + b"\n",
            stream.size,
            self._timeout_seconds,
        )
        with stream.open() as source:
            self._execute(operation, source)
        stream.assert_stable()
        return expected

    def _seal_kit(
        self,
        stream: PinnedExactTreeStream,
        *,
        sealer_path: str,
        sealer_sha256: str,
    ) -> Mapping[str, Any]:
        expected = expected_seal_receipt(
            stream,
            receiver_self_sha256=sealer_sha256,
        )
        operation = _FixedOperation(
            "seal_kit",
            (
                "/usr/bin/python3",
                "-I",
                "-B",
                sealer_path,
                "seal",
                "--incoming",
                str(outer.INCOMING_BASE / stream.release_id),
                "--expected-manifest-sha256",
                stream.release_id,
            ),
            _canonical(expected) + b"\n",
            0,
            self._timeout_seconds,
        )
        self._execute_empty(operation)
        return expected

    def _validate_cloud_verify_receipt(
        self,
        value: Mapping[str, Any],
        *,
        binding: _BoundInertCloudBundle,
    ) -> Mapping[str, Any]:
        error_code = "owner_gate_stage0_cloud_verify_receipt_invalid"
        if (
            frozenset(value) != _VERIFY_RECEIPT_FIELDS
            or value.get("schema")
            != "muncho-owner-gate-stage0-bundle-verification.v1"
            or value.get("release_revision") != self._release_sha
            or value.get("package_sha256") != binding.package_sha256
            or value.get("verified") is not True
            or value.get("incoming_payload_code_executed_before_verification")
            is not False
        ):
            raise launcher.OwnerLauncherError(error_code)
        _validate_self_hash(
            value,
            field="receipt_sha256",
            error_code=error_code,
        )
        return value

    def _validate_cloud_preflight_receipt(
        self,
        value: Mapping[str, Any],
        *,
        binding: _BoundInertCloudBundle,
    ) -> Mapping[str, Any]:
        error_code = "owner_gate_stage0_cloud_preflight_receipt_invalid"
        true_flags = (
            "openssl_ed25519_rawin_verified",
            "systemd_sysusers_available",
            "systemd_tmpfiles_available",
            "python_venv_available",
            "python_venv_without_pip_available",
        )
        false_flags = (
            "network_install_required",
            "cloud_mutation_performed",
            "activation_performed",
        )
        if (
            frozenset(value) != _PREFLIGHT_RECEIPT_FIELDS
            or value.get("schema") != cloud_stage0.PREFLIGHT_SCHEMA
            or value.get("release_revision") != self._release_sha
            or value.get("python_version")
            != f"Python {cloud_stage0.PYTHON_VERSION}"
            or value.get("python_sha256") != binding.interpreter_sha256
            or not isinstance(value.get("openssl_version"), str)
            or not value["openssl_version"].startswith(
                cloud_stage0.OPENSSL_VERSION_PREFIX
            )
            or not isinstance(value.get("systemd_version"), str)
            or not value["systemd_version"].startswith("systemd 252 ")
            or value.get("bootstrap_pip_version")
            != binding.bootstrap_pip_version
            or value.get("bootstrap_pip_sha256")
            != binding.bootstrap_pip_sha256
            or _SHA256.fullmatch(
                str(value.get("executable_identities_sha256", ""))
            )
            is None
            or any(value.get(name) is not True for name in true_flags)
            or any(value.get(name) is not False for name in false_flags)
        ):
            raise launcher.OwnerLauncherError(error_code)
        _validate_self_hash(
            value,
            field="preflight_sha256",
            error_code=error_code,
        )
        return value

    def _validate_cloud_install_receipt(
        self,
        value: Mapping[str, Any],
        *,
        binding: _BoundInertCloudBundle,
    ) -> Mapping[str, Any]:
        error_code = "owner_gate_stage0_cloud_install_receipt_invalid"
        phase_evidence = value.get("phase_evidence_sha256")
        lineage = {
            "pre_foundation_authority_sha256": (
                self._foundation.pre_foundation_authority_sha256
            ),
            "foundation_apply_receipt_sha256": (
                self._foundation.foundation_apply_receipt_sha256
            ),
            "project_ancestry_evidence_sha256": (
                self._foundation.project_ancestry_evidence_sha256
            ),
            "project_ancestry_chain_sha256": (
                self._foundation.project_ancestry_chain_sha256
            ),
            "resource_ancestor_chain": list(
                self._foundation.resource_ancestor_chain
            ),
        }
        digest_fields = (
            "release_tree_sha256",
            "transaction_prefix_sha256",
            "authority_receipt_public_key_sha256",
            "authority_receipt_public_key_id",
            "credential_id_sha256",
            "executor_hosts_receipt_sha256",
        )
        false_flags = (
            "current_release_selected",
            "activation_performed",
            "activation_seal_created",
            "iam_binding_created",
            "cloud_mutation_performed",
            "caddy_cutover_performed",
        )
        signature = value.get("signature_ed25519_b64url")
        try:
            signature_raw = base64.urlsafe_b64decode(
                f"{signature}==".encode("ascii", errors="strict")
            )
        except (UnicodeError, ValueError):
            signature_raw = b""
        if (
            frozenset(value) != _INSTALL_RECEIPT_FIELDS
            or value.get("schema")
            != "muncho-owner-gate-offline-install-receipt.v1"
            or value.get("release_revision") != self._release_sha
            or value.get("package_sha256") != binding.package_sha256
            or value.get("source_tree_oid") != binding.source_tree_oid
            or any(value.get(name) != expected for name, expected in lineage.items())
            or type(value.get("installed_at_unix")) is not int
            or value["installed_at_unix"] <= 0
            or value.get("release_path")
            != str(cloud_stage0.RELEASE_BASE / self._release_sha)
            or any(
                _SHA256.fullmatch(str(value.get(name, ""))) is None
                for name in digest_fields
            )
            or not isinstance(phase_evidence, dict)
            or frozenset(phase_evidence)
            != frozenset(_INSTALL_PHASES_WITHOUT_RECEIPT)
            or any(
                _SHA256.fullmatch(str(phase_evidence.get(name, ""))) is None
                for name in _INSTALL_PHASES_WITHOUT_RECEIPT
            )
            or value.get("systemd_units_enabled") != []
            or any(value.get(name) is not False for name in false_flags)
            or value.get("signer_key_id")
            != value.get("authority_receipt_public_key_id")
            or not isinstance(signature, str)
            or _ED25519_SIGNATURE_B64URL.fullmatch(signature) is None
            or len(signature_raw) != 64
            or base64.urlsafe_b64encode(signature_raw)
            .rstrip(b"=")
            .decode("ascii")
            != signature
        ):
            raise launcher.OwnerLauncherError(error_code)
        unsigned = {
            key: item
            for key, item in value.items()
            if key
            not in {
                "receipt_sha256",
                "signer_key_id",
                "signature_ed25519_b64url",
            }
        }
        if _SHA256.fullmatch(str(value.get("receipt_sha256", ""))) is None or value[
            "receipt_sha256"
        ] != outer.sha256_json(unsigned):
            raise launcher.OwnerLauncherError(error_code)
        return value

    def _run_cloud_verify(
        self,
        binding: _BoundInertCloudBundle,
    ) -> Mapping[str, Any]:
        value = self._execute_canonical_receipt(
            _FixedOperation(
                "cloud_verify",
                (
                    "/usr/bin/python3",
                    "-I",
                    "-B",
                    binding.trusted_runner_path,
                    "cloud-verify",
                    "--bundle",
                    binding.bundle_path,
                ),
                b"",
                0,
                self._timeout_seconds,
            ),
            error_code="owner_gate_stage0_cloud_verify_receipt_invalid",
        )
        return self._validate_cloud_verify_receipt(value, binding=binding)

    def _run_cloud_preflight(
        self,
        binding: _BoundInertCloudBundle,
    ) -> Mapping[str, Any]:
        value = self._execute_canonical_receipt(
            _FixedOperation(
                "cloud_preflight",
                (
                    "/usr/bin/python3",
                    "-I",
                    "-B",
                    binding.trusted_runner_path,
                    "cloud-preflight",
                    "--bundle",
                    binding.bundle_path,
                ),
                b"",
                0,
                self._timeout_seconds,
            ),
            error_code="owner_gate_stage0_cloud_preflight_receipt_invalid",
        )
        return self._validate_cloud_preflight_receipt(value, binding=binding)

    def _run_cloud_install(
        self,
        binding: _BoundInertCloudBundle,
    ) -> Mapping[str, Any]:
        value = self._execute_canonical_receipt(
            _FixedOperation(
                "cloud_install",
                (
                    "/usr/bin/python3",
                    "-I",
                    "-B",
                    binding.trusted_runner_path,
                    "cloud-install",
                    "--bundle",
                    binding.bundle_path,
                ),
                b"",
                0,
                self._timeout_seconds,
            ),
            error_code="owner_gate_stage0_cloud_install_receipt_invalid",
        )
        return self._validate_cloud_install_receipt(value, binding=binding)

    def _validate_host_runtime_receipt(
        self,
        value: Mapping[str, Any],
        *,
        binding: _BoundInertCloudBundle,
    ) -> Mapping[str, Any]:
        error_code = "owner_gate_stage0_host_runtime_receipt_invalid"
        false_fields = (
            "network_install_required",
            "generic_usr_bin_python3_runtime",
            "service_start_performed",
            "service_enablement_mutated",
            "iam_mutation_performed",
            "cloud_mutation_performed",
            "private_key_material_received",
            "private_key_digest_recorded",
        )
        if (
            frozenset(value) != _HOST_RUNTIME_RECEIPT_FIELDS
            or value.get("schema") != "muncho-host-offline-trusted-runtime.v1"
            or value.get("release_revision") != self._release_sha
            or value.get("package_sha256") != binding.package_sha256
            or _SHA256.fullmatch(str(value.get("preflight_sha256", ""))) is None
            or _SHA256.fullmatch(str(value.get("runtime_inventory_sha256", ""))) is None
            or value.get("runtime_interpreter")
            != (
                f"/opt/muncho-trusted-observation/releases/"
                f"{self._release_sha}/venv/bin/python"
            )
            or value.get("host_attestor_entrypoint")
            != (
                f"/opt/muncho-trusted-observation/releases/"
                f"{self._release_sha}/bin/muncho-host-observation-attestor"
            )
            or value.get("host_provisioner_entrypoint")
            != (
                f"/opt/muncho-trusted-observation/releases/"
                f"{self._release_sha}/bin/muncho-host-trusted-signer-provision"
            )
            or not isinstance(value.get("release"), Mapping)
            or not isinstance(value.get("sudoers"), Mapping)
            or value.get("offline_runtime") is not True
            or value.get("current_link_absent") is not True
            or value.get("activation_seal_absent") is not True
            or any(value.get(name) is not False for name in false_fields)
        ):
            raise launcher.OwnerLauncherError(error_code)
        _validate_self_hash(
            value,
            field="receipt_sha256",
            error_code=error_code,
        )
        return value

    def _run_host_runtime_install(
        self,
        binding: _BoundInertCloudBundle,
    ) -> Mapping[str, Any]:
        value = self._execute_canonical_receipt(
            _FixedOperation(
                "host_runtime_install",
                (
                    "/usr/bin/python3",
                    "-I",
                    "-B",
                    binding.trusted_runner_path,
                    "host-install",
                    "--bundle",
                    binding.bundle_path,
                ),
                b"",
                0,
                self._timeout_seconds,
            ),
            error_code="owner_gate_stage0_host_runtime_receipt_invalid",
        )
        return self._validate_host_runtime_receipt(value, binding=binding)

    def _provision_signer(
        self,
        *,
        role: str,
        package_sha256: str,
        expected_key_id: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if role not in {"cloud", "host"}:
            raise launcher.OwnerLauncherError("owner_gate_stage0_signer_role_invalid")
        try:
            frame = signer_author.build_provisioning_envelope(
                role=role,
                release_revision=self._release_sha,
                package_sha256=package_sha256,
                owner_authorization_receipt_sha256=(
                    self._foundation.owner_reauthentication_receipt_sha256
                ),
            )
        except signer_author.TrustedSignerAuthorError:
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_signer_envelope_invalid"
            ) from None
        base = (
            "/opt/muncho-owner-gate/releases"
            if role == "cloud"
            else "/opt/muncho-trusted-observation/releases"
        )
        executable = f"{base}/{self._release_sha}/venv/bin/python"
        entrypoint = (
            f"{base}/{self._release_sha}/bin/muncho-{role}-trusted-signer-provision"
        )
        try:
            result = self._exchange_fixed_operation(
                _FixedOperation(
                    f"{role}_signer_provision",
                    (executable, "-I", "-B", entrypoint),
                    b"",
                    len(frame),
                    self._timeout_seconds,
                ),
                _MutableFrameReader(frame),
                maximum_stdout_bytes=MAX_CLOUD_RECEIPT_BYTES,
            )
            receipt = _decode_canonical_stdout(
                result.stdout,
                error_code="owner_gate_stage0_signer_receipt_invalid",
            )
        finally:
            _wipe(frame)
        public_key = _local_signer_public_key(
            self._release_sha,
            role=role,
            expected_key_id=expected_key_id,
        )
        try:
            checked = signer_provisioning._verify_receipt(
                receipt,
                public_key=public_key,
            )
        except signer_provisioning.TrustedSignerProvisioningError:
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_signer_receipt_invalid"
            ) from None
        if (
            checked.get("role") != role
            or checked.get("release_revision") != self._release_sha
            or checked.get("package_sha256") != package_sha256
            or checked.get("public_key_id") != expected_key_id
            or checked.get("owner_authorization_receipt_sha256")
            != self._foundation.owner_reauthentication_receipt_sha256
            or checked.get("private_key_material_recorded") is not False
            or checked.get("private_key_digest_recorded") is not False
            or checked.get("activation_performed") is not False
            or checked.get("iam_mutation_performed") is not False
            or checked.get("cloud_mutation_performed") is not False
            or checked.get("service_start_performed") is not False
            or checked.get("network_fetch_performed") is not False
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_signer_receipt_invalid"
            )
        return checked, _signer_readiness_from_receipt(checked)

    def _run_host_observation_dispatcher(
        self,
        *,
        operation_name: str,
        frame_value: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if operation_name not in {
            "attached_sa_probe_first",
            "attached_sa_probe_second",
            "host_observation",
        }:
            raise launcher.OwnerLauncherError(
                "owner_gate_host_observation_operation_invalid"
            )
        frame = bytearray(_canonical(frame_value) + b"\n")
        if len(frame) > MAX_OBSERVATION_FRAME_BYTES + 1:
            _wipe(frame)
            raise launcher.OwnerLauncherError(
                "owner_gate_host_observation_frame_oversized"
            )
        base = f"/opt/muncho-trusted-observation/releases/{self._release_sha}"
        try:
            result = self._exchange_fixed_operation(
                _FixedOperation(
                    operation_name,
                    (
                        f"{base}/venv/bin/python",
                        "-I",
                        "-B",
                        f"{base}/bin/muncho-host-observation-attestor",
                    ),
                    b"",
                    len(frame),
                    self._timeout_seconds,
                ),
                _MutableFrameReader(frame),
                maximum_stdout_bytes=MAX_OBSERVATION_FRAME_BYTES,
            )
            return _decode_canonical_stdout(
                result.stdout,
                error_code="owner_gate_host_observation_response_invalid",
            )
        finally:
            _wipe(frame)

    @staticmethod
    def _host_request(
        *,
        schema: str,
        phase: str,
        plan_sha256: str,
        collected_at_unix: int,
        cloud_install_receipt: Mapping[str, Any],
        cloud_receipt: Mapping[str, Any],
        cloud_readiness: Mapping[str, Any],
        host_receipt: Mapping[str, Any],
        host_readiness: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if schema not in {
            "muncho-owner-gate-host-observation-request.v1",
            "muncho-owner-gate-attached-sa-permission-probe-request.v1",
        }:
            raise launcher.OwnerLauncherError(
                "owner_gate_host_observation_request_invalid"
            )
        binding = {
            "phase": phase,
            "collected_at_unix": collected_at_unix,
            "plan_sha256": plan_sha256,
            "cloud_install_receipt": dict(cloud_install_receipt),
            "cloud_signer_provisioning_receipt_sha256": cloud_receipt["receipt_sha256"],
            "cloud_signer_readiness_sha256": cloud_readiness["readiness_sha256"],
            "host_signer_provisioning_receipt_sha256": host_receipt["receipt_sha256"],
            "host_signer_readiness_sha256": host_readiness["readiness_sha256"],
        }
        unsigned = {
            "schema": schema,
            **binding,
            "observation_binding_sha256": foundation.sha256_json(binding),
        }
        return {
            **unsigned,
            "request_sha256": foundation.sha256_json(unsigned),
        }

    def collect_owner_gate_host_observation(
        self,
        *,
        phase: str,
        plan: foundation.OwnerGateFoundationPlan,
        kit_stream: PinnedExactTreeStream,
        bundle_stream: PinnedExactTreeStream,
    ) -> OwnerGateHostObservationHandoff:
        """Install inertly, probe the attached SA twice, and author HOST v2."""

        if (
            type(plan) is not foundation.OwnerGateFoundationPlan
            or phase not in {"inert", "post_iam"}
            or plan.spec.release_revision != self._release_sha
            or not plan.spec.final_release_bound
            or _SHA256.fullmatch(str(plan.spec.cloud_collector_public_key_id or ""))
            is None
            or _SHA256.fullmatch(str(plan.spec.host_collector_public_key_id or ""))
            is None
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_host_observation_input_invalid"
            )
        binding = OwnerGateStage0IapTransport._bind_inert_cloud_bundle(
            self,
            kit_stream=kit_stream,
            bundle_stream=bundle_stream,
        )
        terminal = OwnerGateStage0IapTransport.transport_and_install_inert_cloud_bundle(
            self,
            kit_stream=kit_stream,
            bundle_stream=bundle_stream,
        )
        host_runtime = self._run_host_runtime_install(binding)
        cloud_receipt, cloud_readiness = self._provision_signer(
            role="cloud",
            package_sha256=binding.package_sha256,
            expected_key_id=str(plan.spec.cloud_collector_public_key_id),
        )
        host_receipt, host_readiness = self._provision_signer(
            role="host",
            package_sha256=binding.package_sha256,
            expected_key_id=str(plan.spec.host_collector_public_key_id),
        )
        collected_at_unix = int(time.time())
        if collected_at_unix <= 0:
            raise launcher.OwnerLauncherError(
                "owner_gate_host_observation_time_invalid"
            )
        host_request = self._host_request(
            schema="muncho-owner-gate-host-observation-request.v1",
            phase=phase,
            plan_sha256=plan.sha256,
            collected_at_unix=collected_at_unix,
            cloud_install_receipt=terminal["cloud_install_receipt"],
            cloud_receipt=cloud_receipt,
            cloud_readiness=cloud_readiness,
            host_receipt=host_receipt,
            host_readiness=host_readiness,
        )
        attached_request = self._host_request(
            schema=_ATTACHED_SA_REQUEST_SCHEMA,
            phase=phase,
            plan_sha256=plan.sha256,
            collected_at_unix=collected_at_unix,
            cloud_install_receipt=terminal["cloud_install_receipt"],
            cloud_receipt=cloud_receipt,
            cloud_readiness=cloud_readiness,
            host_receipt=host_receipt,
            host_readiness=host_readiness,
        )
        initial_host_public_key = _local_signer_public_key(
            self._release_sha,
            role="host",
            expected_key_id=str(plan.spec.host_collector_public_key_id),
        )
        probe_first = self._run_host_observation_dispatcher(
            operation_name="attached_sa_probe_first",
            frame_value=attached_request,
        )
        probe_second = self._run_host_observation_dispatcher(
            operation_name="attached_sa_probe_second",
            frame_value=attached_request,
        )
        selected_probe = _select_stable_attached_sa_probe(
            probe_first,
            probe_second,
            request=attached_request,
            release_revision=self._release_sha,
            plan=plan,
            binding=binding,
            foundation_projection=self._foundation,
            cloud_install_receipt=terminal["cloud_install_receipt"],
            cloud_receipt=cloud_receipt,
            cloud_readiness=cloud_readiness,
            host_receipt=host_receipt,
            host_readiness=host_readiness,
            host_public_key=initial_host_public_key,
        )
        frame_unsigned = {
            "schema": "muncho-owner-gate-host-observation-frame.v1",
            "request": host_request,
            "attached_sa_probe": selected_probe,
        }
        host_observation = self._run_host_observation_dispatcher(
            operation_name="host_observation",
            frame_value={
                **frame_unsigned,
                "frame_sha256": foundation.sha256_json(frame_unsigned),
            },
        )
        cloud_receipt_after, cloud_readiness_after = self._provision_signer(
            role="cloud",
            package_sha256=binding.package_sha256,
            expected_key_id=str(plan.spec.cloud_collector_public_key_id),
        )
        host_receipt_after, host_readiness_after = self._provision_signer(
            role="host",
            package_sha256=binding.package_sha256,
            expected_key_id=str(plan.spec.host_collector_public_key_id),
        )
        host_public_key = _local_signer_public_key(
            self._release_sha,
            role="host",
            expected_key_id=str(plan.spec.host_collector_public_key_id),
        )
        try:
            owner_preflight._validate_host(
                host_observation,
                spec=plan.spec,
                plan_sha256=plan.sha256,
                public_key=host_public_key,
                expected_public_key_id=str(plan.spec.host_collector_public_key_id),
                mutation_binding_present=phase == "post_iam",
            )
        except owner_preflight.OwnerGatePreflightError:
            raise launcher.OwnerLauncherError(
                "owner_gate_host_observation_response_invalid"
            ) from None
        release = host_observation.get("release")
        if (
            cloud_receipt_after != cloud_receipt
            or cloud_readiness_after != cloud_readiness
            or host_receipt_after != host_receipt
            or host_readiness_after != host_readiness
            or not isinstance(release, Mapping)
            or release.get("revision") != self._release_sha
            or release.get("source_tree_oid") != binding.source_tree_oid
            or release.get("package_sha256") != binding.package_sha256
            or release.get("package_inventory_sha256")
            != plan.spec.package_inventory_sha256
            or release.get("install_receipt_sha256")
            != terminal["cloud_install_receipt_sha256"]
            or release.get("install_receipt_file_sha256")
            != terminal["cloud_install_receipt_file_sha256"]
            or release.get("cloud_signer_provisioning_receipt_sha256")
            != cloud_receipt["receipt_sha256"]
            or release.get("cloud_signer_readiness_sha256")
            != cloud_readiness["readiness_sha256"]
            or release.get("host_signer_provisioning_receipt_sha256")
            != host_receipt["receipt_sha256"]
            or release.get("host_signer_readiness_sha256")
            != host_readiness["readiness_sha256"]
            or release.get("attached_sa_permission_probe_report_sha256")
            != selected_probe.get("report_sha256")
            or host_observation.get("observation_binding_sha256")
            != host_request["observation_binding_sha256"]
            or host_runtime.get("package_sha256") != binding.package_sha256
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_host_observation_lineage_invalid"
            )
        kit_stream.assert_stable()
        bundle_stream.assert_stable()
        terminal_copy = _decode_canonical_mapping(
            _canonical(terminal),
            maximum=MAX_OBSERVATION_FRAME_BYTES,
            error_code="owner_gate_host_observation_terminal_invalid",
        )
        host_copy = _decode_canonical_mapping(
            _canonical(host_observation),
            maximum=MAX_OBSERVATION_FRAME_BYTES,
            error_code="owner_gate_host_observation_response_invalid",
        )
        return OwnerGateHostObservationHandoff._create(
            terminal_receipt=terminal_copy,
            host_observation=host_copy,
        )

    def _sign_owner_gate_cloud_observation_on_target(
        self,
        *,
        phase: str,
        unsigned_observation: Mapping[str, Any],
        terminal_binding: OwnerGateHostObservationHandoff,
    ) -> Mapping[str, Any]:
        """Use only the fixed UID-29103 release signer for one bound report."""

        if (
            phase not in {"inert", "post_iam"}
            or not isinstance(unsigned_observation, Mapping)
            or type(terminal_binding) is not OwnerGateHostObservationHandoff
            or terminal_binding._marker is not _HOST_OBSERVATION_HANDOFF_MARKER
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_cloud_observation_signer_input_invalid"
            )
        terminal = terminal_binding.terminal_receipt
        host = terminal_binding.host_observation
        release_binding = unsigned_observation.get("release_binding")
        host_release = host.get("release") if isinstance(host, Mapping) else None
        host_attestation = (
            host.get("attestation") if isinstance(host, Mapping) else None
        )
        if (
            not isinstance(terminal, Mapping)
            or set(terminal) != _INERT_TERMINAL_FIELDS
            or terminal.get("schema") != INERT_CLOUD_BUNDLE_TERMINAL_SCHEMA
            or terminal.get("release_sha") != self._release_sha
            or terminal.get("terminal_receipt_sha256")
            != foundation.sha256_json({
                name: item
                for name, item in terminal.items()
                if name != "terminal_receipt_sha256"
            })
            or terminal.get("inert_cloud_bundle_installed") is not True
            or terminal.get("current_release_selected") is not False
            or terminal.get("systemd_units_enabled") != []
            or terminal.get("service_activation_performed") is not False
            or terminal.get("activation_performed") is not False
            or terminal.get("iam_binding_created") is not False
            or terminal.get("cloud_mutation_performed") is not False
            or not isinstance(host, Mapping)
            or not isinstance(host_release, Mapping)
            or not isinstance(host_attestation, Mapping)
            or not isinstance(release_binding, Mapping)
            or host.get("phase") != phase
            or host.get("plan_sha256") != unsigned_observation.get("plan_sha256")
            or release_binding.get("phase") != phase
            or release_binding.get("release_revision") != self._release_sha
            or release_binding.get("source_tree_oid") != terminal.get("source_tree_oid")
            or release_binding.get("package_sha256") != terminal.get("package_sha256")
            or release_binding.get("terminal_receipt_sha256")
            != terminal.get("terminal_receipt_sha256")
            or release_binding.get("host_observation_report_sha256")
            != host.get("report_sha256")
            or release_binding.get("host_observation_binding_sha256")
            != host.get("observation_binding_sha256")
            or release_binding.get("attached_sa_permission_probe_report_sha256")
            != host_release.get("attached_sa_permission_probe_report_sha256")
            or release_binding.get("cloud_signer_provisioning_receipt_sha256")
            != host_release.get("cloud_signer_provisioning_receipt_sha256")
            or release_binding.get("cloud_signer_readiness_sha256")
            != host_release.get("cloud_signer_readiness_sha256")
            or release_binding.get("host_signer_provisioning_receipt_sha256")
            != host_release.get("host_signer_provisioning_receipt_sha256")
            or release_binding.get("host_signer_readiness_sha256")
            != host_release.get("host_signer_readiness_sha256")
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_cloud_observation_terminal_binding_invalid"
            )
        package_sha256 = str(terminal["package_sha256"])
        cloud_public_key, cloud_key_id = _local_signer_public_identity(
            self._release_sha,
            role="cloud",
        )
        host_key_id = str(host_attestation.get("public_key_id", ""))
        if (
            _SHA256.fullmatch(package_sha256) is None
            or _SHA256.fullmatch(host_key_id) is None
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_cloud_observation_terminal_binding_invalid"
            )
        cloud_receipt, cloud_readiness = self._provision_signer(
            role="cloud",
            package_sha256=package_sha256,
            expected_key_id=cloud_key_id,
        )
        host_receipt, host_readiness = self._provision_signer(
            role="host",
            package_sha256=package_sha256,
            expected_key_id=host_key_id,
        )
        if (
            cloud_receipt.get("receipt_sha256")
            != host_release.get("cloud_signer_provisioning_receipt_sha256")
            or cloud_readiness.get("readiness_sha256")
            != host_release.get("cloud_signer_readiness_sha256")
            or host_receipt.get("receipt_sha256")
            != host_release.get("host_signer_provisioning_receipt_sha256")
            or host_readiness.get("readiness_sha256")
            != host_release.get("host_signer_readiness_sha256")
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_cloud_observation_signer_readiness_invalid"
            )
        snapshot = _canonical({
            "terminal_receipt": terminal,
            "host_observation": host,
            "unsigned_observation": unsigned_observation,
        })
        request_unsigned = {
            "schema": ("muncho-owner-gate-cloud-observation-signing-request.v1"),
            "phase": phase,
            "release_revision": self._release_sha,
            "unsigned_observation": dict(unsigned_observation),
            "terminal_receipt": dict(terminal),
            "host_observation": dict(host),
        }
        request = {
            **request_unsigned,
            "request_sha256": foundation.sha256_json(request_unsigned),
        }
        frame = bytearray(_canonical(request) + b"\n")
        if len(frame) > MAX_OBSERVATION_FRAME_BYTES + 1:
            _wipe(frame)
            raise launcher.OwnerLauncherError(
                "owner_gate_cloud_observation_signer_request_oversized"
            )
        result: Mapping[str, Any] | None = None
        exchange_failure: BaseException | None = None
        try:
            result = self._exchange_cloud_observation_signer(
                _MutableFrameReader(frame),
                maximum_input_bytes=len(frame),
            )
        except BaseException as exc:
            exchange_failure = exc
        finally:
            _wipe(frame)
        after_failure: BaseException | None = None
        try:
            cloud_receipt_after, cloud_readiness_after = self._provision_signer(
                role="cloud",
                package_sha256=package_sha256,
                expected_key_id=cloud_key_id,
            )
            host_receipt_after, host_readiness_after = self._provision_signer(
                role="host",
                package_sha256=package_sha256,
                expected_key_id=host_key_id,
            )
            if (
                cloud_receipt_after != cloud_receipt
                or cloud_readiness_after != cloud_readiness
                or host_receipt_after != host_receipt
                or host_readiness_after != host_readiness
            ):
                raise launcher.OwnerLauncherError(
                    "owner_gate_cloud_observation_signer_readiness_changed"
                )
        except BaseException as exc:
            after_failure = exc
        if exchange_failure is not None or after_failure is not None:
            raise launcher.OwnerLauncherError(
                "owner_gate_cloud_observation_signer_failed"
            ) from None
        assert result is not None
        returned_unsigned = {
            name: item
            for name, item in result.items()
            if name not in {"report_sha256", "attestation"}
        }
        if _canonical(returned_unsigned) != _canonical(
            unsigned_observation
        ) or snapshot != _canonical({
            "terminal_receipt": terminal_binding.terminal_receipt,
            "host_observation": terminal_binding.host_observation,
            "unsigned_observation": unsigned_observation,
        }):
            raise launcher.OwnerLauncherError(
                "owner_gate_cloud_observation_signer_response_invalid"
            )
        try:
            owner_preflight._validate_cloud(
                result,
                plan_sha256=str(unsigned_observation["plan_sha256"]),
                public_key=cloud_public_key,
                expected_public_key_id=cloud_key_id,
                mutation_binding_present=phase == "post_iam",
            )
        except (KeyError, owner_preflight.OwnerGatePreflightError):
            raise launcher.OwnerLauncherError(
                "owner_gate_cloud_observation_signer_response_invalid"
            ) from None
        return _decode_canonical_mapping(
            _canonical(result),
            maximum=MAX_OBSERVATION_FRAME_BYTES,
            error_code="owner_gate_cloud_observation_signer_response_invalid",
        )

    def transport_exact_stage0_and_bundle(
        self,
        *,
        kit_stream: PinnedExactTreeStream,
        bundle_stream: PinnedExactTreeStream,
    ) -> Mapping[str, Any]:
        binding = self._bind_inert_cloud_bundle(
            kit_stream=kit_stream,
            bundle_stream=bundle_stream,
        )
        interpreter = self._attest_remote_interpreter()
        sealer_payload, sealer_sha256 = self._stage0_sealer_source.snapshot()
        sealer_path = self._materialize_sealer(sealer_payload, sealer_sha256)
        kit_receiver = self._receive_stream(
            kit_stream,
            sealer_path=sealer_path,
            sealer_sha256=sealer_sha256,
        )
        seal = self._seal_kit(
            kit_stream,
            sealer_path=sealer_path,
            sealer_sha256=sealer_sha256,
        )
        bundle_receiver = self._receive_stream(
            bundle_stream,
            sealer_path=sealer_path,
            sealer_sha256=sealer_sha256,
        )
        final_sealer_payload, final_sealer_sha256 = (
            self._stage0_sealer_source.snapshot()
        )
        if (
            final_sealer_payload != sealer_payload
            or final_sealer_sha256 != sealer_sha256
        ):
            raise launcher.OwnerLauncherError("owner_gate_stage0_sealer_changed")
        host_identity = self._host_identity.snapshot()
        unsigned = {
            "schema": TRANSPORT_RECEIPT_SCHEMA,
            "release_sha": self._release_sha,
            "source_tree_oid": binding.source_tree_oid,
            "package_sha256": binding.package_sha256,
            "kit_release_id": binding.kit_release_id,
            "project": launcher.PROJECT,
            "zone": launcher.ZONE,
            "vm_name": self._VM_NAME,
            "vm_numeric_id": host_identity.vm_numeric_id,
            "owner_account": self._OWNER_ACCOUNT,
            "pre_foundation_authority_sha256": (
                self._foundation.pre_foundation_authority_sha256
            ),
            "foundation_apply_receipt_sha256": (
                self._foundation.foundation_apply_receipt_sha256
            ),
            "project_ancestry_evidence_sha256": (
                self._foundation.project_ancestry_evidence_sha256
            ),
            "project_ancestry_chain_sha256": (
                self._foundation.project_ancestry_chain_sha256
            ),
            "resource_ancestor_chain": list(
                self._foundation.resource_ancestor_chain
            ),
            "interpreter_attestation_sha256": interpreter[
                "attestation_sha256"
            ],
            "sealer_sha256": sealer_sha256,
            "sealer_remote_path": sealer_path,
            "kit_receiver_receipt_sha256": kit_receiver["receipt_sha256"],
            "kit_seal_receipt_sha256": seal["receipt_sha256"],
            "bundle_receiver_receipt_sha256": bundle_receiver[
                "receipt_sha256"
            ],
            "recursive_scp_used": False,
            "caller_controlled_remote_command_used": False,
            "caller_controlled_remote_path_used": False,
            "cloud_control_plane_mutation_performed": False,
            "host_filesystem_materialization_performed": True,
            "host_filesystem_materialization_roots": [
                self._SEALER_ROOT,
                str(outer.INCOMING_BASE),
                str(outer.BUNDLE_INCOMING_BASE),
                str(outer.RELEASE_BASE),
                str(outer.TRANSPORT_RECEIPT_BASE),
                str(outer.RECEIPT_BASE),
            ],
            "service_activation_performed": False,
        }
        return {**unsigned, "receipt_sha256": outer.sha256_json(unsigned)}

    def transport_and_install_inert_cloud_bundle(
        self,
        *,
        kit_stream: PinnedExactTreeStream,
        bundle_stream: PinnedExactTreeStream,
    ) -> Mapping[str, Any]:
        """Transfer and install the exact cloud bundle without activation."""

        binding = self._bind_inert_cloud_bundle(
            kit_stream=kit_stream,
            bundle_stream=bundle_stream,
        )
        transport = self.transport_exact_stage0_and_bundle(
            kit_stream=kit_stream,
            bundle_stream=bundle_stream,
        )
        verify = self._run_cloud_verify(binding)
        preflight = self._run_cloud_preflight(binding)
        install = self._run_cloud_install(binding)
        kit_stream.assert_stable()
        bundle_stream.assert_stable()
        if (
            transport.get("release_sha") != self._release_sha
            or transport.get("source_tree_oid") != binding.source_tree_oid
            or transport.get("package_sha256") != binding.package_sha256
            or transport.get("kit_release_id") != binding.kit_release_id
            or transport.get("receipt_sha256")
            != outer.sha256_json({
                key: item
                for key, item in transport.items()
                if key != "receipt_sha256"
            })
        ):
            raise launcher.OwnerLauncherError(
                "owner_gate_stage0_transport_receipt_invalid"
            )
        unsigned = {
            "schema": INERT_CLOUD_BUNDLE_TERMINAL_SCHEMA,
            "release_sha": self._release_sha,
            "source_tree_oid": binding.source_tree_oid,
            "package_sha256": binding.package_sha256,
            "kit_release_id": binding.kit_release_id,
            "trusted_runner_path": binding.trusted_runner_path,
            "bundle_path": binding.bundle_path,
            "pre_foundation_authority_sha256": (
                self._foundation.pre_foundation_authority_sha256
            ),
            "foundation_apply_receipt_sha256": (
                self._foundation.foundation_apply_receipt_sha256
            ),
            "project_ancestry_evidence_sha256": (
                self._foundation.project_ancestry_evidence_sha256
            ),
            "project_ancestry_chain_sha256": (
                self._foundation.project_ancestry_chain_sha256
            ),
            "resource_ancestor_chain": list(
                self._foundation.resource_ancestor_chain
            ),
            "operation_order": [
                "transport_exact_stage0_and_bundle",
                "cloud-verify",
                "cloud-preflight",
                "cloud-install",
            ],
            "transport_receipt_sha256": transport["receipt_sha256"],
            "cloud_verify_receipt_sha256": verify["receipt_sha256"],
            "cloud_preflight_receipt_sha256": preflight[
                "preflight_sha256"
            ],
            "cloud_install_receipt_sha256": install["receipt_sha256"],
            "cloud_install_receipt_file_sha256": _sha256(
                _canonical(install)
            ),
            "cloud_install_receipt": dict(install),
            "cloud_install_signature_framing_validated": True,
            "cloud_install_signature_cryptographically_verified": False,
            "inert_cloud_bundle_installed": True,
            "host_filesystem_materialization_performed": True,
            "current_release_selected": False,
            "systemd_units_enabled": [],
            "service_activation_performed": False,
            "activation_performed": False,
            "activation_seal_created": False,
            "iam_binding_created": False,
            "caddy_cutover_performed": False,
            "cloud_mutation_performed": False,
            "cloud_control_plane_mutation_performed": False,
        }
        return {
            **unsigned,
            "terminal_receipt_sha256": outer.sha256_json(unsigned),
        }


__all__ = [
    "MAX_STDERR_BYTES",
    "MAX_STDOUT_BYTES",
    "INERT_CLOUD_BUNDLE_TERMINAL_SCHEMA",
    "OwnerGateHostObservationHandoff",
    "OwnerGateStage0IapTransport",
    "PinnedExactTreeStream",
    "RawFoundationChainArtifacts",
    "StableOuterSealer",
    "TRANSPORT_RECEIPT_SCHEMA",
    "TrustedOuterSealerSource",
    "expected_seal_receipt",
    "expected_tree_receipt",
]
