#!/usr/bin/env python3
"""Pure deployment preflight for the privileged Canonical Brain writer.

The command consumes an already-collected JSON snapshot.  It never invokes
systemd, IAM, Secret Manager, Cloud SQL, or the network, so collection and
mutation authority remain outside this checker.
"""

from __future__ import annotations

import argparse
import json
import re
import stat
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from gateway.canonical_writer_db import (
    CanonicalEventLogIdentity,
    PrivilegeAttestation,
    PrivilegeAttestationError,
    RoutineIdentity,
    SequencePrivilegeGrant,
    TablePrivilegeGrant,
    WriterPrivilegePolicy,
    validate_privilege_attestation,
)
from gateway.canonical_writer_postgres_backend import (
    CANONICAL_WRITER_MIGRATION_OWNER,
    CANONICAL_WRITER_ROLE,
    CANONICAL_WRITER_SCHEMA,
    EXPECTED_HELPER_ROUTINE_SIGNATURES,
    EXPECTED_ROUTINE_SIGNATURES,
)
from gateway.canonical_writer_boundary import (
    DEFAULT_GATEWAY_UNIT,
    DEFAULT_SOCKET_PATH,
    DEFAULT_WRITER_UNIT,
)


_TRUE = {"1", "yes", "true", "on"}
_HARDENED_TRUE_PROPERTIES = (
    "NoNewPrivileges",
    "PrivateTmp",
    "PrivateDevices",
    "ProtectKernelTunables",
    "ProtectKernelModules",
    "ProtectKernelLogs",
    "ProtectControlGroups",
    "RestrictSUIDSGID",
    "LockPersonality",
    "MemoryDenyWriteExecute",
    "RestrictRealtime",
    "RestrictNamespaces",
)
_FORBIDDEN_IAM_MARKERS = ("cloudsql", "secretmanager")
_FORBIDDEN_BROAD_ROLES = {"roles/owner", "roles/editor"}
_MAX_GATEWAY_PROCESS_EVIDENCE_AGE_SECONDS = 30
_TRUSTED_SECRET_PROVISIONERS = {"root", "systemd"}
_WRITER_BOOTSTRAP_MODULE = "scripts.canonical_writer_bootstrap"
_GATEWAY_ENTRY_MODULE = "gateway.run"
_WRITER_EXPORT_UNIT = "muncho-canonical-writer-export.service"
_WRITER_EXPORT_TIMER = "muncho-canonical-writer-export.timer"
_REQUIRED_IMPORT_KINDS = {
    "application",
    "interpreter",
    "site_packages",
    "stdlib",
}
_ALLOWED_IMPORT_KINDS = _REQUIRED_IMPORT_KINDS | {"native_library"}
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MUTATION_ACCESS_KEYS = {"create_child", "rename", "replace", "write"}
_DANGEROUS_LOCAL_GROUP_NAMES = (
    "adm",
    "crontab",
    "disk",
    "docker",
    "incus",
    "kmem",
    "kvm",
    "libvirt",
    "lxd",
    "operator",
    "root",
    "shadow",
    "sudo",
    "systemd-journal",
    "wheel",
)
_ALLOWED_TIMER_SCHEDULE_KEYS = {
    "AccuracyUSec",
    "OnBootSec",
    "OnCalendar",
    "OnUnitActiveSec",
    "Persistent",
    "RandomizedDelayUSec",
}
_GATEWAY_ALLOWED_READ_WRITE_PATHS = {
    "/run/hermes-cloud-gateway",
    "/var/lib/hermes-gateway",
    "/var/log/hermes-gateway",
}


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[PreflightCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [asdict(check) for check in self.checks],
        }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _integer(value: Any, default: int = -1) -> int:
    try:
        if isinstance(value, bool):
            return default
        return int(value)
    except (OverflowError, TypeError, ValueError):
        return default


def _mode(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        raw = value.strip().casefold()
        try:
            return int(raw, 8)
        except ValueError:
            return -1
    return -1


def _enabled(value: Any) -> bool:
    if value is True:
        return True
    return isinstance(value, str) and value.strip().casefold() in _TRUE


def _disabled_or_empty(value: Any) -> bool:
    if value is None or value is False:
        return True
    return isinstance(value, str) and not value.strip()


def _access_is_denied(value: Any) -> bool:
    access = _mapping(value)
    return all(access.get(name) is False for name in ("read", "write", "execute"))


def _access_is_read_only(value: Any) -> bool:
    access = _mapping(value)
    return (
        access.get("read") is True
        and access.get("write") is False
        and access.get("execute") is False
    )


def _legacy_hermes_env_path(value: Any) -> bool:
    path = str(value or "").strip()
    return path == "~/.hermes/.env" or path.endswith("/.hermes/.env")


def _pgpass_path(value: Any) -> bool:
    path = str(value or "").strip()
    return path == "~/.pgpass" or path.endswith("/.pgpass")


def _empty_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and not value
    )


def _absolute_normalized_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or value.startswith("//"):
        return None
    if any(character in value for character in ("\x00", "\n", "\r")):
        return None
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        return None
    return value


def _path_is_within(path: str, root: str) -> bool:
    candidate = Path(path)
    boundary = Path(root)
    return candidate == boundary or boundary in candidate.parents


def _sha256(value: Any) -> str | None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        return None
    return value


def _revision(value: Any) -> str | None:
    if not isinstance(value, str) or _REVISION_RE.fullmatch(value) is None:
        return None
    return value


def _mutation_access_is_denied(value: Any) -> bool:
    access = _mapping(value)
    return set(access) == _MUTATION_ACCESS_KEYS and all(
        access.get(name) is False for name in _MUTATION_ACCESS_KEYS
    )


def _parent_chain_is_immutable(value: Any) -> bool:
    parent_chain = _mapping(value)
    return (
        set(parent_chain)
        == {
            "gateway_child_mutation_access",
            "gateway_mutation_access",
            "mount_read_only",
            "root_owned",
            "symlink_free",
        }
        and parent_chain.get("root_owned") is True
        and parent_chain.get("symlink_free") is True
        and parent_chain.get("mount_read_only") is True
        and _mutation_access_is_denied(
            parent_chain.get("gateway_mutation_access")
        )
        and _mutation_access_is_denied(
            parent_chain.get("gateway_child_mutation_access")
        )
    )


_PROTECTED_PATH_EVIDENCE_KEYS = {
    "digest_sha256",
    "exists",
    "gateway_child_mutation_access",
    "gateway_child_open_write_fds",
    "gateway_child_writable_mappings",
    "gateway_mutation_access",
    "gateway_open_write_fds",
    "gateway_writable_mappings",
    "immutable",
    "kind",
    "mode",
    "mount_read_only",
    "object_type",
    "owner_uid",
    "parent_chain",
    "path",
    "symlink",
    "writer_access",
}


def _protected_path_is_immutable(
    value: Any,
    *,
    expected_path: str,
    expected_kind: str,
    expected_digest: str,
    expected_object_type: str,
) -> bool:
    item = _mapping(value)
    mode = _mode(item.get("mode"))
    writer_access = _mapping(item.get("writer_access"))
    writer_execute_required = (
        expected_object_type == "directory" or expected_kind == "interpreter"
    )
    return (
        set(item) == _PROTECTED_PATH_EVIDENCE_KEYS
        and item.get("path") == expected_path
        and item.get("kind") == expected_kind
        and item.get("digest_sha256") == expected_digest
        and item.get("object_type") == expected_object_type
        and item.get("exists") is True
        and item.get("symlink") is False
        and item.get("immutable") is True
        and item.get("mount_read_only") is True
        and type(item.get("owner_uid")) is int
        and item.get("owner_uid") == 0
        and mode >= 0
        and mode & ~0o777 == 0
        and mode & 0o022 == 0
        and set(writer_access) == {"execute", "read", "write"}
        and writer_access.get("read") is True
        and writer_access.get("write") is False
        and (
            writer_access.get("execute") is True
            if writer_execute_required
            else type(writer_access.get("execute")) is bool
        )
        and _mutation_access_is_denied(item.get("gateway_mutation_access"))
        and _mutation_access_is_denied(
            item.get("gateway_child_mutation_access")
        )
        and _empty_sequence(item.get("gateway_open_write_fds"))
        and _empty_sequence(item.get("gateway_child_open_write_fds"))
        and _empty_sequence(item.get("gateway_writable_mappings"))
        and _empty_sequence(item.get("gateway_child_writable_mappings"))
        and _parent_chain_is_immutable(item.get("parent_chain"))
    )


_WRITER_DEPLOYMENT_POLICY_KEYS = {
    "artifact_digest_sha256",
    "artifact_root",
    "bind_paths",
    "bind_read_only_paths",
    "config_path",
    "exec_start",
    "import_paths",
    "interpreter",
    "module",
    "module_origin",
    "projection_export_directory",
    "read_write_paths",
    "revision",
    "runtime_directory",
    "unit_name",
    "working_directory",
}
_WRITER_DEPLOYMENT_ATTESTATION_KEYS = {
    "artifact",
    "complete",
    "import_closure",
    "mounts",
    "process",
    "unit",
}
_WRITER_UNIT_ATTESTATION_KEYS = {
    "alternate_exec_commands",
    "artifact_digest_sha256",
    "code_injection_environment_variable_names",
    "config_path",
    "environment_files",
    "environment_pythonhome",
    "environment_pythonpath",
    "exec_start",
    "group_gid",
    "interpreter",
    "module",
    "name",
    "revision",
    "user_uid",
    "working_directory",
}
_WRITER_MOUNT_ATTESTATION_KEYS = {
    "bind_paths",
    "bind_read_only_paths",
    "complete",
    "read_write_paths",
}
_WRITER_PROCESS_ATTESTATION_KEYS = {
    "artifact_digest_sha256",
    "bootstrap_module_origin",
    "cmdline",
    "complete",
    "deleted_code_mappings",
    "effective_import_paths",
    "executable_digest_sha256",
    "executable_path",
    "loaded_module_origins",
    "loaded_module_origins_complete",
    "mapped_executable_paths",
    "mapped_executable_paths_complete",
    "observed_at_unix",
    "pid",
    "process_start_time_ticks",
    "revision",
    "systemd_main_pid",
    "systemd_main_pid_start_time_ticks",
    "unexpected_import_origins",
    "unit_name",
    "writable_code_mappings",
}
_WRITER_IMPORT_POLICY_KEYS = {
    "digest_sha256",
    "kind",
    "object_type",
    "path",
}


def _writer_deployment_checks(
    value: Any,
    *,
    collected_at_unix: int,
    writer_uid: int,
    writer_gid: int,
) -> list[PreflightCheck]:
    """Validate a pinned, self-contained writer artifact and unit snapshot.

    The policy side is expected to come from deployment control, while the
    attestation side is host evidence.  Keeping both explicit makes the trust
    requirement visible and lets a future authenticated collector keep policy
    out of the gateway process.
    """

    deployment = _mapping(value)
    policy = _mapping(deployment.get("policy"))
    attestation = _mapping(deployment.get("attestation"))
    policy_shape_ok = (
        set(deployment) == {"attestation", "policy"}
        and set(policy) == _WRITER_DEPLOYMENT_POLICY_KEYS
        and set(attestation) == _WRITER_DEPLOYMENT_ATTESTATION_KEYS
    )

    policy_valid = False
    unit_exact = False
    artifact_exact = False
    artifact_immutable = False
    closure_complete = False
    closure_exact = False
    closure_immutable = False
    process_fresh = False
    process_exact = False
    process_code_closure = False
    mounts_exact = False
    mounts_minimal = False

    try:
        unit_name = str(policy.get("unit_name") or "")
        artifact_root = _absolute_normalized_path(policy.get("artifact_root"))
        revision = _revision(policy.get("revision"))
        artifact_digest = _sha256(policy.get("artifact_digest_sha256"))
        interpreter = _absolute_normalized_path(policy.get("interpreter"))
        module = str(policy.get("module") or "")
        module_origin = _absolute_normalized_path(policy.get("module_origin"))
        config_path = _absolute_normalized_path(policy.get("config_path"))
        working_directory = _absolute_normalized_path(
            policy.get("working_directory")
        )
        runtime_directory = _absolute_normalized_path(
            policy.get("runtime_directory")
        )
        export_directory = _absolute_normalized_path(
            policy.get("projection_export_directory")
        )
        exec_start = _strings(policy.get("exec_start"))
        read_write_paths = _strings(policy.get("read_write_paths"))
        bind_paths = _strings(policy.get("bind_paths"))
        bind_read_only_paths = _strings(policy.get("bind_read_only_paths"))

        raw_import_paths = policy.get("import_paths")
        if not isinstance(raw_import_paths, Sequence) or isinstance(
            raw_import_paths, (str, bytes, bytearray)
        ):
            raise ValueError("import path policy must be a sequence")
        import_policy: dict[str, tuple[str, str, str]] = {}
        import_policy_shape_ok = bool(raw_import_paths)
        for raw in raw_import_paths:
            item = _mapping(raw)
            path = _absolute_normalized_path(item.get("path"))
            kind = str(item.get("kind") or "")
            digest = _sha256(item.get("digest_sha256"))
            object_type = str(item.get("object_type") or "")
            import_policy_shape_ok = import_policy_shape_ok and (
                set(item) == _WRITER_IMPORT_POLICY_KEYS
                and path is not None
                and kind in _ALLOWED_IMPORT_KINDS
                and digest is not None
                and object_type in {"directory", "regular_file"}
                and path not in import_policy
            )
            if path is not None:
                import_policy[path] = (kind, digest or "", object_type)

        expected_exec = (
            interpreter or "",
            "-I",
            "-m",
            _WRITER_BOOTSTRAP_MODULE,
            "--config",
            config_path or "",
        )
        expected_write_paths = tuple(
            sorted({runtime_directory or "", export_directory or ""})
        )
        required_kinds = {item[0] for item in import_policy.values()}
        protected_paths_inside_artifact = bool(artifact_root) and all(
            _path_is_within(path, artifact_root or "")
            for path in import_policy
        )
        artifact_policy = import_policy.get(artifact_root or "")
        interpreter_policy = import_policy.get(interpreter or "")
        site_package_roots = tuple(
            path
            for path, expected in import_policy.items()
            if expected[0] == "site_packages"
            and expected[2] == "directory"
        )
        expected_module_relative = Path(*_WRITER_BOOTSTRAP_MODULE.split("."))
        expected_module_relative = expected_module_relative.with_suffix(".py")
        module_origin_exact = module_origin is not None and any(
            Path(module_origin) == Path(root) / expected_module_relative
            for root in site_package_roots
        )
        paths_do_not_overlap_writes = (
            artifact_root is not None
            and runtime_directory is not None
            and export_directory is not None
            and not _path_is_within(runtime_directory, artifact_root)
            and not _path_is_within(export_directory, artifact_root)
            and not _path_is_within(artifact_root, runtime_directory)
            and not _path_is_within(artifact_root, export_directory)
        )
        config_outside_mutable_and_artifact = (
            config_path is not None
            and artifact_root is not None
            and runtime_directory is not None
            and export_directory is not None
            and not _path_is_within(config_path, artifact_root)
            and not _path_is_within(config_path, runtime_directory)
            and not _path_is_within(config_path, export_directory)
            and not _path_is_within(artifact_root, config_path)
            and not _path_is_within(runtime_directory, config_path)
            and not _path_is_within(export_directory, config_path)
        )
        policy_valid = (
            policy_shape_ok
            and unit_name == DEFAULT_WRITER_UNIT
            and artifact_root is not None
            and revision is not None
            and Path(artifact_root).name == revision
            and artifact_digest is not None
            and interpreter is not None
            and _path_is_within(interpreter, artifact_root)
            and module == _WRITER_BOOTSTRAP_MODULE
            and module_origin_exact
            and config_path is not None
            and working_directory == artifact_root
            and runtime_directory == str(DEFAULT_SOCKET_PATH.parent)
            and export_directory is not None
            and len(Path(export_directory).parts) >= 4
            and exec_start == expected_exec
            and import_policy_shape_ok
            and required_kinds >= _REQUIRED_IMPORT_KINDS
            and protected_paths_inside_artifact
            and artifact_policy
            == ("application", artifact_digest, "directory")
            and interpreter_policy is not None
            and interpreter_policy[0] == "interpreter"
            and interpreter_policy[2] == "regular_file"
            and read_write_paths == expected_write_paths
            and not bind_paths
            and bind_read_only_paths == (artifact_root,)
            and paths_do_not_overlap_writes
            and config_outside_mutable_and_artifact
        )

        unit = _mapping(attestation.get("unit"))
        unit_exact = (
            set(unit) == _WRITER_UNIT_ATTESTATION_KEYS
            and unit.get("name") == unit_name
            and tuple(_strings(unit.get("exec_start"))) == exec_start
            and unit.get("interpreter") == interpreter
            and unit.get("module") == module
            and unit.get("config_path") == config_path
            and unit.get("working_directory") == working_directory
            and unit.get("revision") == revision
            and unit.get("artifact_digest_sha256") == artifact_digest
            and _empty_sequence(unit.get("alternate_exec_commands"))
            and _empty_sequence(unit.get("environment_files"))
            and _empty_sequence(
                unit.get("code_injection_environment_variable_names")
            )
            and _empty_sequence(unit.get("environment_pythonpath"))
            and _disabled_or_empty(unit.get("environment_pythonhome"))
            and type(unit.get("user_uid")) is int
            and unit.get("user_uid") == writer_uid
            and type(unit.get("group_gid")) is int
            and unit.get("group_gid") == writer_gid
        )

        artifact = _mapping(attestation.get("artifact"))
        artifact_exact = (
            artifact.get("path") == artifact_root
            and artifact.get("revision") == revision
            and artifact.get("digest_sha256") == artifact_digest
        )
        artifact_immutable = _protected_path_is_immutable(
            {
                key: artifact.get(key)
                for key in _PROTECTED_PATH_EVIDENCE_KEYS
            },
            expected_path=artifact_root or "",
            expected_kind="application",
            expected_digest=artifact_digest or "",
            expected_object_type="directory",
        ) and set(artifact) == _PROTECTED_PATH_EVIDENCE_KEYS | {"revision"}

        import_closure = _mapping(attestation.get("import_closure"))
        raw_observed_paths = import_closure.get("paths")
        closure_complete = (
            set(import_closure) == {"complete", "paths"}
            and import_closure.get("complete") is True
            and isinstance(raw_observed_paths, Sequence)
            and not isinstance(raw_observed_paths, (str, bytes, bytearray))
        )
        observed_paths: dict[str, Mapping[str, Any]] = {}
        observed_shape_ok = closure_complete
        if closure_complete:
            for raw in raw_observed_paths:
                item = _mapping(raw)
                path = item.get("path")
                observed_shape_ok = observed_shape_ok and (
                    isinstance(path, str) and path not in observed_paths
                )
                if isinstance(path, str):
                    observed_paths[path] = item
        closure_exact = (
            observed_shape_ok and set(observed_paths) == set(import_policy)
        )
        closure_immutable = closure_exact and all(
            _protected_path_is_immutable(
                observed_paths[path],
                expected_path=path,
                expected_kind=expected[0],
                expected_digest=expected[1],
                expected_object_type=expected[2],
            )
            for path, expected in import_policy.items()
        )

        process = _mapping(attestation.get("process"))
        process_shape_ok = (
            set(process) == _WRITER_PROCESS_ATTESTATION_KEYS
            and process.get("complete") is True
        )
        process_observed_at = _integer(process.get("observed_at_unix"))
        process_pid = _integer(process.get("pid"))
        process_main_pid = _integer(process.get("systemd_main_pid"))
        process_start = _integer(process.get("process_start_time_ticks"))
        process_main_start = _integer(
            process.get("systemd_main_pid_start_time_ticks")
        )
        process_fresh = (
            process_shape_ok
            and collected_at_unix >= 0
            and process_observed_at >= 0
            and 0
            <= collected_at_unix - process_observed_at
            <= _MAX_GATEWAY_PROCESS_EVIDENCE_AGE_SECONDS
        )
        process_exact = (
            process_shape_ok
            and process_pid > 1
            and process_pid == process_main_pid
            and process_start > 0
            and process_start == process_main_start
            and process.get("unit_name") == unit_name
            and tuple(_strings(process.get("cmdline"))) == exec_start
            and process.get("executable_path") == interpreter
            and process.get("executable_digest_sha256")
            == (interpreter_policy or ("", "", ""))[1]
            and process.get("revision") == revision
            and process.get("artifact_digest_sha256") == artifact_digest
            and process.get("bootstrap_module_origin") == module_origin
        )
        effective_import_paths = _strings(
            process.get("effective_import_paths")
        )
        loaded_module_origins = _strings(
            process.get("loaded_module_origins")
        )
        mapped_executable_paths = _strings(
            process.get("mapped_executable_paths")
        )
        expected_import_paths = tuple(
            sorted(
                path
                for path, expected in import_policy.items()
                if expected[0] in {"site_packages", "stdlib"}
            )
        )

        def covered_by_immutable_closure(path: str) -> bool:
            normalized = _absolute_normalized_path(path)
            return normalized is not None and any(
                (
                    _path_is_within(normalized, protected_path)
                    if expected[2] == "directory"
                    else normalized == protected_path
                )
                for protected_path, expected in import_policy.items()
            )

        process_code_closure = (
            process_shape_ok
            and effective_import_paths == expected_import_paths
            and process.get("loaded_module_origins_complete") is True
            and bool(loaded_module_origins)
            and module_origin in loaded_module_origins
            and len(set(loaded_module_origins)) == len(loaded_module_origins)
            and all(covered_by_immutable_closure(path) for path in loaded_module_origins)
            and process.get("mapped_executable_paths_complete") is True
            and interpreter in mapped_executable_paths
            and len(set(mapped_executable_paths)) == len(mapped_executable_paths)
            and all(
                covered_by_immutable_closure(path)
                for path in mapped_executable_paths
            )
            and _empty_sequence(process.get("unexpected_import_origins"))
            and _empty_sequence(process.get("deleted_code_mappings"))
            and _empty_sequence(process.get("writable_code_mappings"))
        )

        mounts = _mapping(attestation.get("mounts"))
        mounts_shape_ok = (
            set(mounts) == _WRITER_MOUNT_ATTESTATION_KEYS
            and mounts.get("complete") is True
        )
        observed_read_write = _strings(mounts.get("read_write_paths"))
        observed_bind = _strings(mounts.get("bind_paths"))
        observed_bind_read_only = _strings(
            mounts.get("bind_read_only_paths")
        )
        mounts_exact = (
            mounts_shape_ok
            and observed_read_write == read_write_paths
            and observed_bind == bind_paths
            and observed_bind_read_only == bind_read_only_paths
        )
        mounts_minimal = (
            mounts_exact
            and read_write_paths == expected_write_paths
            and all(_absolute_normalized_path(path) == path for path in read_write_paths)
            and not bind_paths
            and bind_read_only_paths == (artifact_root,)
            and paths_do_not_overlap_writes
            and config_outside_mutable_and_artifact
        )
    except (TypeError, ValueError):
        pass

    evidence_complete = (
        policy_shape_ok
        and attestation.get("complete") is True
        and closure_complete
        and _mapping(attestation.get("process")).get("complete") is True
        and _mapping(attestation.get("mounts")).get("complete") is True
    )
    return [
        PreflightCheck(
            "writer_deployment.evidence_complete",
            evidence_complete,
            "writer artifact, import closure, unit, and mount evidence must be explicitly complete",
        ),
        PreflightCheck(
            "writer_deployment.policy_valid",
            policy_valid,
            "deployment policy must pin one self-contained revision and digest with an isolated Python bootstrap",
        ),
        PreflightCheck(
            "writer_deployment.unit_exact",
            policy_valid and unit_exact,
            "writer unit must use the exact UID/GID, interpreter, -I module invocation, revision, digest, and config with no alternate commands or code-injection environment",
        ),
        PreflightCheck(
            "writer_deployment.artifact_exact",
            policy_valid and artifact_exact,
            "observed writer artifact path, revision, and digest must exactly match deployment policy",
        ),
        PreflightCheck(
            "writer_deployment.artifact_immutable",
            policy_valid and artifact_immutable,
            "writer artifact and parent chain must be root-owned, symlink-free, read-only, immutable, and mutation-inaccessible to gateway processes",
        ),
        PreflightCheck(
            "writer_deployment.import_closure_complete",
            closure_complete,
            "Python interpreter, application, stdlib, site-packages, and native import evidence must be explicitly complete",
        ),
        PreflightCheck(
            "writer_deployment.import_closure_exact",
            policy_valid and closure_exact,
            "observed Python import and venv closure must exactly match the pinned path and digest set",
        ),
        PreflightCheck(
            "writer_deployment.import_closure_immutable",
            policy_valid and closure_immutable,
            "every Python import path and parent chain must be immutable and mutation-inaccessible to the gateway and its children",
        ),
        PreflightCheck(
            "writer_deployment.process_fresh",
            process_fresh,
            "live writer process evidence must be complete and no more than 30 seconds older than the snapshot",
        ),
        PreflightCheck(
            "writer_deployment.process_exact",
            policy_valid and process_fresh and process_exact,
            "live systemd MainPID, start time, argv, executable, revision, and digests must match the pinned writer unit",
        ),
        PreflightCheck(
            "writer_deployment.process_code_closure",
            policy_valid and process_fresh and process_code_closure,
            "live Python imports and executable mappings must be complete, immutable-closure-contained, non-writable, and not deleted",
        ),
        PreflightCheck(
            "writer_deployment.mount_carveouts_exact",
            policy_valid and mounts_exact,
            "observed ReadWritePaths, BindPaths, and BindReadOnlyPaths must exactly match deployment policy",
        ),
        PreflightCheck(
            "writer_deployment.mount_carveouts_minimal",
            policy_valid and mounts_minimal,
            "writable carve-outs are limited to the exact socket runtime and projection export directories; writable bind mounts are forbidden",
        ),
    ]


_GATEWAY_DEPLOYMENT_POLICY_KEYS = {
    "artifact_digest_sha256",
    "artifact_root",
    "bind_paths",
    "bind_read_only_paths",
    "dynamic_python_discovery_paths",
    "dynamic_python_loading_mode",
    "exec_start",
    "import_paths",
    "interpreter",
    "module",
    "module_origin",
    "read_write_paths",
    "revision",
    "unit_name",
    "working_directory",
}
_GATEWAY_DEPLOYMENT_ATTESTATION_KEYS = {
    "complete",
    "mounts",
    "process",
    "unit",
}
_GATEWAY_UNIT_ATTESTATION_KEYS = {
    "alternate_exec_commands",
    "artifact_digest_sha256",
    "code_injection_environment_variable_names",
    "environment_files",
    "environment_pythonhome",
    "environment_pythonpath",
    "exec_start",
    "group_gid",
    "interpreter",
    "module",
    "module_origin",
    "name",
    "revision",
    "user_uid",
    "working_directory",
}
_GATEWAY_PROCESS_ATTESTATION_KEYS = {
    "artifact_digest_sha256",
    "cmdline",
    "complete",
    "deleted_code_mappings",
    "dynamic_python_discovery_complete",
    "dynamic_python_discovery_paths",
    "dynamic_python_loaded_origins",
    "dynamic_python_loading_mode",
    "dynamic_python_writable_paths",
    "effective_import_paths",
    "entry_module_origin",
    "executable_digest_sha256",
    "executable_path",
    "loaded_module_origins",
    "loaded_module_origins_complete",
    "mapped_executable_paths",
    "mapped_executable_paths_complete",
    "observed_at_unix",
    "pid",
    "process_start_time_ticks",
    "revision",
    "systemd_main_pid",
    "systemd_main_pid_start_time_ticks",
    "unexpected_import_origins",
    "unit_name",
    "writable_code_mappings",
}


def _gateway_deployment_checks(
    value: Any,
    *,
    writer_deployment: Any,
    collected_at_unix: int,
    gateway_uid: int,
    gateway_gid: int,
    gateway_process_snapshot: Any,
) -> list[PreflightCheck]:
    """Bind the authorized gateway MainPID to the writer's immutable release."""

    deployment = _mapping(value)
    policy = _mapping(deployment.get("policy"))
    attestation = _mapping(deployment.get("attestation"))
    writer_policy = _mapping(_mapping(writer_deployment).get("policy"))
    policy_shape_ok = (
        set(deployment) == {"attestation", "policy"}
        and set(policy) == _GATEWAY_DEPLOYMENT_POLICY_KEYS
        and set(attestation) == _GATEWAY_DEPLOYMENT_ATTESTATION_KEYS
    )
    policy_valid = False
    unit_exact = False
    process_fresh = False
    process_exact = False
    process_code_closure = False
    mounts_exact = False
    try:
        unit_name = str(policy.get("unit_name") or "")
        artifact_root = _absolute_normalized_path(policy.get("artifact_root"))
        revision = _revision(policy.get("revision"))
        artifact_digest = _sha256(policy.get("artifact_digest_sha256"))
        interpreter = _absolute_normalized_path(policy.get("interpreter"))
        module = str(policy.get("module") or "")
        module_origin = _absolute_normalized_path(policy.get("module_origin"))
        working_directory = _absolute_normalized_path(
            policy.get("working_directory")
        )
        exec_start = _strings(policy.get("exec_start"))
        read_write_paths = _strings(policy.get("read_write_paths"))
        bind_paths = _strings(policy.get("bind_paths"))
        bind_read_only_paths = _strings(policy.get("bind_read_only_paths"))
        dynamic_loading_mode = str(
            policy.get("dynamic_python_loading_mode") or ""
        )
        dynamic_discovery_paths = _strings(
            policy.get("dynamic_python_discovery_paths")
        )
        raw_import_paths = policy.get("import_paths")
        if not isinstance(raw_import_paths, Sequence) or isinstance(
            raw_import_paths, (str, bytes, bytearray)
        ):
            raise ValueError("gateway import path policy must be a sequence")
        import_policy: dict[str, tuple[str, str, str]] = {}
        import_shape_ok = bool(raw_import_paths)
        for raw in raw_import_paths:
            item = _mapping(raw)
            path = _absolute_normalized_path(item.get("path"))
            kind = str(item.get("kind") or "")
            digest = _sha256(item.get("digest_sha256"))
            object_type = str(item.get("object_type") or "")
            import_shape_ok = import_shape_ok and (
                set(item) == _WRITER_IMPORT_POLICY_KEYS
                and path is not None
                and kind in _ALLOWED_IMPORT_KINDS
                and digest is not None
                and object_type in {"directory", "regular_file"}
                and path not in import_policy
            )
            if path is not None:
                import_policy[path] = (kind, digest or "", object_type)
        site_package_roots = tuple(
            path
            for path, expected in import_policy.items()
            if expected[0] == "site_packages"
            and expected[2] == "directory"
        )
        expected_module_relative = Path(*_GATEWAY_ENTRY_MODULE.split("."))
        expected_module_relative = expected_module_relative.with_suffix(".py")
        module_origin_exact = module_origin is not None and any(
            Path(module_origin) == Path(root) / expected_module_relative
            for root in site_package_roots
        )
        writable_paths_safe = all(
            _absolute_normalized_path(path) == path
            and path in _GATEWAY_ALLOWED_READ_WRITE_PATHS
            and artifact_root is not None
            and not _path_is_within(path, artifact_root)
            and not _path_is_within(artifact_root, path)
            for path in read_write_paths
        )
        dynamic_paths_safe = (
            dynamic_loading_mode in {"disabled", "immutable_only"}
            and len(set(dynamic_discovery_paths))
            == len(dynamic_discovery_paths)
            and all(
                _absolute_normalized_path(path) == path
                and artifact_root is not None
                and _path_is_within(path, artifact_root)
                and all(
                    not _path_is_within(path, writable)
                    and not _path_is_within(writable, path)
                    for writable in read_write_paths
                )
                for path in dynamic_discovery_paths
            )
            and (
                not dynamic_discovery_paths
                if dynamic_loading_mode == "disabled"
                else bool(dynamic_discovery_paths)
            )
        )
        interpreter_policy = import_policy.get(interpreter or "")
        policy_valid = (
            policy_shape_ok
            and unit_name == DEFAULT_GATEWAY_UNIT
            and artifact_root is not None
            and revision is not None
            and artifact_digest is not None
            and interpreter is not None
            and module == _GATEWAY_ENTRY_MODULE
            and module_origin_exact
            and working_directory == artifact_root
            and exec_start
            == (interpreter, "-I", "-m", _GATEWAY_ENTRY_MODULE)
            and import_shape_ok
            and interpreter_policy is not None
            and interpreter_policy[0] == "interpreter"
            and interpreter_policy[2] == "regular_file"
            and policy.get("artifact_root")
            == writer_policy.get("artifact_root")
            and policy.get("revision") == writer_policy.get("revision")
            and policy.get("artifact_digest_sha256")
            == writer_policy.get("artifact_digest_sha256")
            and policy.get("interpreter") == writer_policy.get("interpreter")
            and policy.get("import_paths") == writer_policy.get("import_paths")
            and writable_paths_safe
            and dynamic_paths_safe
            and not bind_paths
            and bind_read_only_paths == (artifact_root,)
        )

        unit = _mapping(attestation.get("unit"))
        unit_exact = (
            set(unit) == _GATEWAY_UNIT_ATTESTATION_KEYS
            and unit.get("name") == unit_name
            and tuple(_strings(unit.get("exec_start"))) == exec_start
            and unit.get("interpreter") == interpreter
            and unit.get("module") == module
            and unit.get("module_origin") == module_origin
            and unit.get("working_directory") == working_directory
            and unit.get("revision") == revision
            and unit.get("artifact_digest_sha256") == artifact_digest
            and _empty_sequence(unit.get("alternate_exec_commands"))
            and _empty_sequence(unit.get("environment_files"))
            and _empty_sequence(
                unit.get("code_injection_environment_variable_names")
            )
            and _empty_sequence(unit.get("environment_pythonpath"))
            and _disabled_or_empty(unit.get("environment_pythonhome"))
            and type(unit.get("user_uid")) is int
            and unit.get("user_uid") == gateway_uid
            and type(unit.get("group_gid")) is int
            and unit.get("group_gid") == gateway_gid
        )

        process = _mapping(attestation.get("process"))
        live_gateway = _mapping(gateway_process_snapshot)
        process_shape_ok = (
            set(process) == _GATEWAY_PROCESS_ATTESTATION_KEYS
            and process.get("complete") is True
        )
        observed_at = _integer(process.get("observed_at_unix"))
        pid = _integer(process.get("pid"))
        main_pid = _integer(process.get("systemd_main_pid"))
        start = _integer(process.get("process_start_time_ticks"))
        main_start = _integer(
            process.get("systemd_main_pid_start_time_ticks")
        )
        process_fresh = (
            process_shape_ok
            and collected_at_unix >= 0
            and observed_at >= 0
            and 0
            <= collected_at_unix - observed_at
            <= _MAX_GATEWAY_PROCESS_EVIDENCE_AGE_SECONDS
        )
        process_exact = (
            process_shape_ok
            and pid > 1
            and pid == main_pid == _integer(live_gateway.get("pid"))
            and start > 0
            and start
            == main_start
            == _integer(live_gateway.get("process_start_time_ticks"))
            and process.get("unit_name") == unit_name
            and tuple(_strings(process.get("cmdline"))) == exec_start
            and process.get("executable_path") == interpreter
            and process.get("executable_digest_sha256")
            == (interpreter_policy or ("", "", ""))[1]
            and process.get("entry_module_origin") == module_origin
            and process.get("revision") == revision
            and process.get("artifact_digest_sha256") == artifact_digest
        )
        effective_import_paths = _strings(
            process.get("effective_import_paths")
        )
        loaded_origins = _strings(process.get("loaded_module_origins"))
        mapped_paths = _strings(process.get("mapped_executable_paths"))
        observed_dynamic_paths = _strings(
            process.get("dynamic_python_discovery_paths")
        )
        dynamic_loaded_origins = _strings(
            process.get("dynamic_python_loaded_origins")
        )
        expected_import_paths = tuple(
            sorted(
                path
                for path, expected in import_policy.items()
                if expected[0] in {"site_packages", "stdlib"}
            )
        )

        def covered_by_shared_release(path: str) -> bool:
            normalized = _absolute_normalized_path(path)
            return normalized is not None and any(
                (
                    _path_is_within(normalized, protected_path)
                    if expected[2] == "directory"
                    else normalized == protected_path
                )
                for protected_path, expected in import_policy.items()
            )

        process_code_closure = (
            process_shape_ok
            and effective_import_paths == expected_import_paths
            and process.get("loaded_module_origins_complete") is True
            and module_origin in loaded_origins
            and len(set(loaded_origins)) == len(loaded_origins)
            and all(covered_by_shared_release(path) for path in loaded_origins)
            and process.get("mapped_executable_paths_complete") is True
            and interpreter in mapped_paths
            and len(set(mapped_paths)) == len(mapped_paths)
            and all(covered_by_shared_release(path) for path in mapped_paths)
            and _empty_sequence(process.get("unexpected_import_origins"))
            and _empty_sequence(process.get("deleted_code_mappings"))
            and _empty_sequence(process.get("writable_code_mappings"))
            and process.get("dynamic_python_discovery_complete") is True
            and process.get("dynamic_python_loading_mode")
            == dynamic_loading_mode
            and observed_dynamic_paths == dynamic_discovery_paths
            and len(set(dynamic_loaded_origins))
            == len(dynamic_loaded_origins)
            and all(
                any(
                    _path_is_within(origin, discovery_path)
                    for discovery_path in dynamic_discovery_paths
                )
                and origin in loaded_origins
                for origin in dynamic_loaded_origins
            )
            and (
                not dynamic_loaded_origins
                if dynamic_loading_mode == "disabled"
                else True
            )
            and _empty_sequence(
                process.get("dynamic_python_writable_paths")
            )
        )

        mounts = _mapping(attestation.get("mounts"))
        mounts_exact = (
            set(mounts) == _WRITER_MOUNT_ATTESTATION_KEYS
            and mounts.get("complete") is True
            and _strings(mounts.get("read_write_paths"))
            == read_write_paths
            and _strings(mounts.get("bind_paths")) == bind_paths
            and _strings(mounts.get("bind_read_only_paths"))
            == bind_read_only_paths
        )
    except (TypeError, ValueError):
        pass

    evidence_complete = (
        policy_shape_ok
        and attestation.get("complete") is True
        and _mapping(attestation.get("process")).get("complete") is True
        and _mapping(attestation.get("mounts")).get("complete") is True
    )
    return [
        PreflightCheck(
            "gateway_deployment.evidence_complete",
            evidence_complete,
            "gateway unit, live process, import closure, and mount evidence must be explicitly complete",
        ),
        PreflightCheck(
            "gateway_deployment.shared_immutable_release",
            policy_valid,
            "gateway must execute the exact pinned gateway module from the writer-attested root-owned immutable release",
        ),
        PreflightCheck(
            "gateway_deployment.unit_exact",
            policy_valid and unit_exact,
            "gateway unit must have exact immutable ExecStart identity and no alternate command or code-injection environment",
        ),
        PreflightCheck(
            "gateway_deployment.process_fresh",
            process_fresh,
            "live gateway code evidence must be complete and no more than 30 seconds older than the snapshot",
        ),
        PreflightCheck(
            "gateway_deployment.process_exact",
            policy_valid and process_fresh and process_exact,
            "authorized gateway MainPID must execute the exact pinned argv, interpreter, module origin, revision, and digest",
        ),
        PreflightCheck(
            "gateway_deployment.process_code_closure",
            policy_valid and process_fresh and process_code_closure,
            "gateway imports and executable mappings must be complete, shared-release-contained, non-writable, and not deleted",
        ),
        PreflightCheck(
            "gateway_deployment.mounts_exact",
            policy_valid and mounts_exact,
            "gateway writable paths must exactly match policy and never contain or replace the immutable release",
        ),
    ]


_AUTHORITY_IDENTITY_KEYS = {
    "effective_gid",
    "effective_uid",
    "pid",
    "supplementary_gids",
}
_AUTHORITY_CHILDREN_KEYS = {"complete", "processes"}
_AUTHORITY_IDENTITIES_KEYS = {"gateway", "gateway_children", "writer"}
_GROUP_POLICY_KEYS = {
    "complete",
    "evaluated_dangerous_group_names",
    "gateway_child_dangerous_memberships",
    "gateway_dangerous_memberships",
    "unknown_privileged_gids",
    "writer_dangerous_memberships",
}
_AUTHORITY_DENIAL_KEYS = {
    "authorized_doas_commands",
    "authorized_polkit_actions",
    "authorized_sudo_commands",
    "authorized_systemd_dbus_actions",
    "can_create_transient_units",
    "can_invoke_polkit_actions",
    "can_manage_cron",
    "can_manage_timers",
    "can_manage_writer_units",
    "can_switch_to_writer_uid",
    "can_write_cron_paths",
    "can_write_systemd_unit_paths",
    "effective_capabilities",
}
_PRIVILEGED_INVENTORY_KEYS = {
    "complete",
    "gateway_child_writable_writer_unit_files",
    "gateway_writable_writer_unit_files",
    "writer_uid_at_jobs",
    "writer_uid_cron_entries",
    "writer_uid_process_executables",
    "writer_uid_service_units",
    "writer_uid_timer_units",
    "writer_uid_transient_units",
    "writer_uid_unattributed_processes",
}
_EXPORTER_POLICY_KEYS = {
    "artifact_digest_sha256",
    "artifact_root",
    "bind_paths",
    "bind_read_only_paths",
    "config_path",
    "enabled",
    "exec_start",
    "export_limit",
    "export_path",
    "interpreter",
    "module",
    "module_origin",
    "read_write_paths",
    "revision",
    "timer_name",
    "timer_schedule",
    "unit_name",
}
_EXPORTER_ATTESTATION_KEYS = {"complete", "enabled", "timer", "unit"}
_EXPORTER_UNIT_KEYS = {
    "alternate_exec_commands",
    "artifact_digest_sha256",
    "artifact_root",
    "bind_paths",
    "bind_read_only_paths",
    "code_injection_environment_variable_names",
    "environment_files",
    "exec_start",
    "group_gid",
    "module_origin",
    "name",
    "read_write_paths",
    "revision",
    "systemd_properties",
    "type",
    "user_uid",
}
_EXPORTER_TIMER_KEYS = {"enabled", "name", "schedule", "unit_name"}
_AUTHORITY_SURFACE_KEYS = {
    "collected_by_uid",
    "complete",
    "gateway_authority",
    "gateway_child_authority",
    "group_policy",
    "identities",
    "observed_at_unix",
    "privileged_execution_inventory",
    "projection_exporter",
}


def _strict_integers(value: Any) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ValueError("expected an integer sequence")
    if any(type(item) is not int for item in value):
        raise ValueError("expected an integer sequence")
    return tuple(value)


def _authority_identity(
    value: Any,
    *,
    expected_pid: int,
    expected_uid: int,
    expected_gid: int,
    expected_supplementary_gids: tuple[int, ...],
) -> bool:
    identity = _mapping(value)
    try:
        supplementary = _strict_integers(identity.get("supplementary_gids"))
    except ValueError:
        return False
    return (
        set(identity) == _AUTHORITY_IDENTITY_KEYS
        and type(identity.get("pid")) is int
        and identity.get("pid") == expected_pid
        and type(identity.get("effective_uid")) is int
        and identity.get("effective_uid") == expected_uid
        and type(identity.get("effective_gid")) is int
        and identity.get("effective_gid") == expected_gid
        and supplementary == expected_supplementary_gids
    )


def _authority_is_denied(value: Any) -> bool:
    authority = _mapping(value)
    boolean_names = {
        "can_create_transient_units",
        "can_invoke_polkit_actions",
        "can_manage_cron",
        "can_manage_timers",
        "can_manage_writer_units",
        "can_switch_to_writer_uid",
        "can_write_cron_paths",
        "can_write_systemd_unit_paths",
    }
    sequence_names = _AUTHORITY_DENIAL_KEYS - boolean_names
    return (
        set(authority) == _AUTHORITY_DENIAL_KEYS
        and all(authority.get(name) is False for name in boolean_names)
        and all(_empty_sequence(authority.get(name)) for name in sequence_names)
    )


def _timer_schedule_is_safe(value: Any) -> bool:
    schedule = _mapping(value)
    trigger_keys = {"OnBootSec", "OnCalendar", "OnUnitActiveSec"}
    shape_ok = (
        bool(schedule)
        and set(schedule) <= _ALLOWED_TIMER_SCHEDULE_KEYS
        and bool(set(schedule) & trigger_keys)
        and all(
            type(item) in {bool, int, str}
            and not (
                isinstance(item, str)
                and (
                    len(item) > 500
                    or any(ord(character) < 32 for character in item)
                )
            )
            for item in schedule.values()
        )
    )
    if not shape_ok:
        return False
    for name, item in schedule.items():
        if name == "Persistent" and type(item) is not bool:
            return False
        if name == "OnCalendar" and (
            not isinstance(item, str) or not item.strip()
        ):
            return False
        if name in {
            "AccuracyUSec",
            "OnBootSec",
            "OnUnitActiveSec",
            "RandomizedDelayUSec",
        } and not (
            (type(item) is int and item >= 0)
            or (isinstance(item, str) and bool(item.strip()))
        ):
            return False
    return True


def _writer_authority_surface_checks(
    value: Any,
    *,
    writer_deployment: Any,
    gateway_process_snapshot: Any,
    collected_at_unix: int,
    gateway_uid: int,
    gateway_gid: int,
    writer_uid: int,
    writer_gid: int,
    socket_group_gid: int,
    projector_gid: int,
) -> list[PreflightCheck]:
    """Validate root-collected local privilege and scheduled-exec evidence."""

    surface = _mapping(value)
    identities = _mapping(surface.get("identities"))
    group_policy = _mapping(surface.get("group_policy"))
    inventory = _mapping(surface.get("privileged_execution_inventory"))
    exporter = _mapping(surface.get("projection_exporter"))
    exporter_policy = _mapping(exporter.get("policy"))
    exporter_attestation = _mapping(exporter.get("attestation"))
    writer_policy = _mapping(_mapping(writer_deployment).get("policy"))
    writer_attestation = _mapping(
        _mapping(writer_deployment).get("attestation")
    )
    writer_process = _mapping(writer_attestation.get("process"))
    gateway_process = _mapping(gateway_process_snapshot)
    shape_ok = (
        set(surface) == _AUTHORITY_SURFACE_KEYS
        and set(identities) == _AUTHORITY_IDENTITIES_KEYS
        and set(group_policy) == _GROUP_POLICY_KEYS
        and set(inventory) == _PRIVILEGED_INVENTORY_KEYS
        and set(exporter) == {"attestation", "policy"}
        and set(exporter_policy) == _EXPORTER_POLICY_KEYS
        and set(exporter_attestation) == _EXPORTER_ATTESTATION_KEYS
    )
    observed_at = _integer(surface.get("observed_at_unix"))
    fresh_root_evidence = (
        shape_ok
        and surface.get("complete") is True
        and type(surface.get("collected_by_uid")) is int
        and surface.get("collected_by_uid") == 0
        and collected_at_unix >= 0
        and observed_at >= 0
        and 0
        <= collected_at_unix - observed_at
        <= _MAX_GATEWAY_PROCESS_EVIDENCE_AGE_SECONDS
    )

    identities_exact = False
    dangerous_groups_absent = False
    exporter_manifest_exact = False
    exporter_state_safe = False
    inventory_exact = False
    try:
        gateway_identity = identities.get("gateway")
        writer_identity = identities.get("writer")
        children = _mapping(identities.get("gateway_children"))
        raw_children = children.get("processes")
        if not isinstance(raw_children, Sequence) or isinstance(
            raw_children, (str, bytes, bytearray)
        ):
            raise ValueError("gateway child inventory must be a sequence")
        child_pids: set[int] = set()
        children_exact = (
            set(children) == _AUTHORITY_CHILDREN_KEYS
            and children.get("complete") is True
        )
        for raw_child in raw_children:
            child = _mapping(raw_child)
            child_pid = child.get("pid")
            children_exact = children_exact and (
                type(child_pid) is int
                and child_pid > 1
                and child_pid not in child_pids
                and child_pid
                not in {
                    _integer(gateway_process.get("pid")),
                    _integer(writer_process.get("pid")),
                }
                and _authority_identity(
                    child,
                    expected_pid=child_pid,
                    expected_uid=gateway_uid,
                    expected_gid=gateway_gid,
                    expected_supplementary_gids=(socket_group_gid,),
                )
            )
            if type(child_pid) is int:
                child_pids.add(child_pid)
        identities_exact = (
            gateway_uid > 0
            and gateway_gid > 0
            and writer_uid > 0
            and writer_gid > 0
            and len({gateway_uid, writer_uid}) == 2
            and gateway_gid
            not in {writer_gid, socket_group_gid, projector_gid}
            and _authority_identity(
                gateway_identity,
                expected_pid=_integer(gateway_process.get("pid")),
                expected_uid=gateway_uid,
                expected_gid=gateway_gid,
                expected_supplementary_gids=(socket_group_gid,),
            )
            and _authority_identity(
                writer_identity,
                expected_pid=_integer(writer_process.get("pid")),
                expected_uid=writer_uid,
                expected_gid=writer_gid,
                expected_supplementary_gids=(projector_gid,),
            )
            and children_exact
        )

        evaluated_names = _strings(
            group_policy.get("evaluated_dangerous_group_names")
        )
        unknown_gids = _strict_integers(
            group_policy.get("unknown_privileged_gids")
        )
        dangerous_groups_absent = (
            group_policy.get("complete") is True
            and evaluated_names == _DANGEROUS_LOCAL_GROUP_NAMES
            and _empty_sequence(
                group_policy.get("gateway_dangerous_memberships")
            )
            and _empty_sequence(
                group_policy.get("gateway_child_dangerous_memberships")
            )
            and _empty_sequence(
                group_policy.get("writer_dangerous_memberships")
            )
            and not unknown_gids
        )

        enabled = exporter_policy.get("enabled")
        if type(enabled) is not bool:
            raise ValueError("exporter enabled must be boolean")
        artifact_root = _absolute_normalized_path(
            exporter_policy.get("artifact_root")
        )
        interpreter = _absolute_normalized_path(
            exporter_policy.get("interpreter")
        )
        module_origin = _absolute_normalized_path(
            exporter_policy.get("module_origin")
        )
        config_path = _absolute_normalized_path(
            exporter_policy.get("config_path")
        )
        export_path = _absolute_normalized_path(
            exporter_policy.get("export_path")
        )
        export_limit = exporter_policy.get("export_limit")
        read_write_paths = _strings(
            exporter_policy.get("read_write_paths")
        )
        bind_paths = _strings(exporter_policy.get("bind_paths"))
        bind_read_only_paths = _strings(
            exporter_policy.get("bind_read_only_paths")
        )
        exec_start = _strings(exporter_policy.get("exec_start"))
        export_directory = _absolute_normalized_path(
            writer_policy.get("projection_export_directory")
        )
        expected_exec = (
            interpreter or "",
            "-I",
            "-m",
            _WRITER_BOOTSTRAP_MODULE,
            "--config",
            config_path or "",
            "--export-events",
            export_path or "",
            "--export-limit",
            str(export_limit),
        )
        exporter_manifest_exact = (
            artifact_root is not None
            and interpreter is not None
            and module_origin is not None
            and config_path is not None
            and export_path is not None
            and export_directory is not None
            and Path(export_path).parent == Path(export_directory)
            and type(export_limit) is int
            and 1 <= export_limit <= 1_000_000
            and exporter_policy.get("unit_name") == _WRITER_EXPORT_UNIT
            and exporter_policy.get("timer_name") == _WRITER_EXPORT_TIMER
            and exporter_policy.get("module") == _WRITER_BOOTSTRAP_MODULE
            and exporter_policy.get("artifact_root")
            == writer_policy.get("artifact_root")
            and exporter_policy.get("revision")
            == writer_policy.get("revision")
            and exporter_policy.get("artifact_digest_sha256")
            == writer_policy.get("artifact_digest_sha256")
            and exporter_policy.get("interpreter")
            == writer_policy.get("interpreter")
            and exporter_policy.get("module_origin")
            == writer_policy.get("module_origin")
            and exporter_policy.get("config_path")
            == writer_policy.get("config_path")
            and exec_start == expected_exec
            and read_write_paths == (export_directory,)
            and not bind_paths
            and bind_read_only_paths == (artifact_root,)
            and _timer_schedule_is_safe(
                exporter_policy.get("timer_schedule")
            )
        )

        unit = _mapping(exporter_attestation.get("unit"))
        timer = _mapping(exporter_attestation.get("timer"))
        if enabled:
            unit_systemd = _systemd_checks(
                _mapping(unit.get("systemd_properties"))
            )
            exporter_state_safe = (
                exporter_attestation.get("complete") is True
                and exporter_attestation.get("enabled") is True
                and set(unit) == _EXPORTER_UNIT_KEYS
                and unit.get("name") == _WRITER_EXPORT_UNIT
                and unit.get("type") == "oneshot"
                and _strings(unit.get("exec_start")) == exec_start
                and unit.get("user_uid") == writer_uid
                and unit.get("group_gid") == writer_gid
                and unit.get("artifact_root") == artifact_root
                and unit.get("revision") == writer_policy.get("revision")
                and unit.get("artifact_digest_sha256")
                == writer_policy.get("artifact_digest_sha256")
                and unit.get("module_origin") == module_origin
                and _strings(unit.get("read_write_paths"))
                == read_write_paths
                and _strings(unit.get("bind_paths")) == bind_paths
                and _strings(unit.get("bind_read_only_paths"))
                == bind_read_only_paths
                and _empty_sequence(unit.get("alternate_exec_commands"))
                and _empty_sequence(unit.get("environment_files"))
                and _empty_sequence(
                    unit.get("code_injection_environment_variable_names")
                )
                and all(check.passed for check in unit_systemd)
                and set(timer) == _EXPORTER_TIMER_KEYS
                and timer.get("name") == _WRITER_EXPORT_TIMER
                and timer.get("unit_name") == _WRITER_EXPORT_UNIT
                and timer.get("enabled") is True
                and timer.get("schedule")
                == exporter_policy.get("timer_schedule")
            )
        else:
            exporter_state_safe = (
                exporter_attestation.get("complete") is True
                and exporter_attestation.get("enabled") is False
                and not unit
                and not timer
            )

        expected_services = tuple(
            sorted(
                {
                    DEFAULT_WRITER_UNIT,
                    *({_WRITER_EXPORT_UNIT} if enabled else set()),
                }
            )
        )
        expected_timers = (_WRITER_EXPORT_TIMER,) if enabled else ()
        inventory_exact = (
            inventory.get("complete") is True
            and _strings(inventory.get("writer_uid_service_units"))
            == expected_services
            and _strings(inventory.get("writer_uid_timer_units"))
            == expected_timers
            and _empty_sequence(
                inventory.get("writer_uid_transient_units")
            )
            and _empty_sequence(inventory.get("writer_uid_cron_entries"))
            and _empty_sequence(inventory.get("writer_uid_at_jobs"))
            and _strings(inventory.get("writer_uid_process_executables"))
            == (str(writer_policy.get("interpreter") or ""),)
            and _empty_sequence(
                inventory.get("writer_uid_unattributed_processes")
            )
            and _empty_sequence(
                inventory.get("gateway_writable_writer_unit_files")
            )
            and _empty_sequence(
                inventory.get("gateway_child_writable_writer_unit_files")
            )
        )
    except (TypeError, ValueError):
        pass

    return [
        PreflightCheck(
            "writer_authority.root_evidence_fresh",
            fresh_root_evidence,
            "local authority evidence must be complete, root-collected, and no more than 30 seconds older than the snapshot",
        ),
        PreflightCheck(
            "writer_authority.identities_exact",
            fresh_root_evidence and identities_exact,
            "gateway, every observed gateway child, and writer must have exact effective UID/GID and only their dedicated supplementary group",
        ),
        PreflightCheck(
            "writer_authority.dangerous_groups_absent",
            fresh_root_evidence and dangerous_groups_absent,
            "gateway, children, and writer must have no known or unknown privileged local group membership",
        ),
        PreflightCheck(
            "writer_authority.gateway_denied",
            fresh_root_evidence
            and _authority_is_denied(surface.get("gateway_authority")),
            "gateway must have no systemd, transient-unit, timer, cron, UID-switch, sudo/doas, capability, or polkit authority over the writer boundary",
        ),
        PreflightCheck(
            "writer_authority.children_denied",
            fresh_root_evidence
            and _authority_is_denied(
                surface.get("gateway_child_authority")
            ),
            "gateway children must have no systemd, transient-unit, timer, cron, UID-switch, sudo/doas, capability, or polkit authority over the writer boundary",
        ),
        PreflightCheck(
            "writer_authority.exporter_manifest_exact",
            fresh_root_evidence and exporter_manifest_exact,
            "projection exporter policy must pin the same immutable artifact, exact one-shot argv, export path, and timer schedule",
        ),
        PreflightCheck(
            "writer_authority.exporter_state_safe",
            fresh_root_evidence
            and exporter_manifest_exact
            and exporter_state_safe,
            "projection exporter must be absent when disabled or exactly attested and hardened when enabled",
        ),
        PreflightCheck(
            "writer_authority.privileged_inventory_exact",
            fresh_root_evidence and inventory_exact,
            "writer UID services and timers must match the exact allow-list with no cron, at, transient, unattributed, or gateway-writable execution surface",
        ),
    ]


def _gateway_process_checks(
    value: Any,
    *,
    collected_at_unix: int,
) -> list[PreflightCheck]:
    process = _mapping(value)
    pid = _integer(process.get("pid"))
    systemd_main_pid = _integer(process.get("systemd_main_pid"))
    process_start_time = _integer(process.get("process_start_time_ticks"))
    main_pid_start_time = _integer(
        process.get("systemd_main_pid_start_time_ticks")
    )
    observed_at_unix = _integer(process.get("observed_at_unix"))
    evidence_age = collected_at_unix - observed_at_unix
    return [
        PreflightCheck(
            "gateway_process.evidence_fresh",
            process.get("complete") is True
            and collected_at_unix >= 0
            and observed_at_unix >= 0
            and 0 <= evidence_age <= _MAX_GATEWAY_PROCESS_EVIDENCE_AGE_SECONDS,
            "live gateway process evidence must be complete and no more than 30 seconds older than the snapshot",
        ),
        PreflightCheck(
            "gateway_process.linux",
            str(process.get("platform") or "").casefold() == "linux",
            "the privileged writer boundary requires a Linux gateway process",
        ),
        PreflightCheck(
            "gateway_process.exact_main_pid",
            pid > 1
            and pid == systemd_main_pid
            and process_start_time > 0
            and process_start_time == main_pid_start_time,
            "evidence PID and start time must match the current systemd MainPID",
        ),
        PreflightCheck(
            "gateway_process.nondumpable",
            process.get("dumpable") is False,
            "the live gateway process must report PR_GET_DUMPABLE=0",
        ),
        PreflightCheck(
            "gateway_process.core_disabled",
            _integer(process.get("core_soft_limit")) == 0
            and _integer(process.get("core_hard_limit")) == 0,
            "the live gateway process must have zero soft and hard core limits",
        ),
    ]


def _runtime_secret_source_checks(value: Any) -> list[PreflightCheck]:
    evidence = _mapping(value)
    legacy = _mapping(evidence.get("legacy_hermes_env"))
    pgpass = _mapping(evidence.get("pgpass"))
    cloud_sql_socket = _mapping(evidence.get("cloud_sql_unix_socket"))
    effective_env = _mapping(evidence.get("effective_gateway_env"))
    effective_env_shape_ok = set(effective_env) == {
        "complete",
        "values_included",
        "database_password_variable_names",
        "database_connection_secret_variable_names",
    }

    try:
        gateway_readable = _strings(
            evidence.get("gateway_readable_secret_files")
        )
        child_readable = _strings(
            evidence.get("gateway_child_readable_secret_files")
        )
        gateway_readable_database_credentials = _strings(
            evidence.get("gateway_readable_database_credential_files")
        )
        child_readable_database_credentials = _strings(
            evidence.get("gateway_child_readable_database_credential_files")
        )
        gateway_readable_environment_files = _strings(
            evidence.get("gateway_readable_systemd_environment_files")
        )
        child_readable_environment_files = _strings(
            evidence.get("gateway_child_readable_systemd_environment_files")
        )
        database_password_names = _strings(
            effective_env.get("database_password_variable_names")
        )
        database_connection_names = _strings(
            effective_env.get("database_connection_secret_variable_names")
        )
        raw_sources = evidence.get("sources")
        if not isinstance(raw_sources, Sequence) or isinstance(
            raw_sources, (str, bytes, bytearray)
        ):
            raise ValueError("runtime secret sources must be a sequence")
        sources = tuple(_mapping(item) for item in raw_sources)
        source_shape_ok = bool(sources) and all(
            bool(str(source.get("name") or "").strip())
            and str(source.get("provisioned_by") or "").casefold()
            in _TRUSTED_SECRET_PROVISIONERS
            and _access_is_denied(source.get("gateway_file_access"))
            and _access_is_denied(source.get("gateway_child_file_access"))
            for source in sources
        )
        unique_names = len(
            {str(source.get("name") or "").strip() for source in sources}
        ) == len(sources)
        source_names = {
            str(source.get("name") or "").strip().casefold()
            for source in sources
        }
    except ValueError:
        gateway_readable = ("invalid",)
        child_readable = ("invalid",)
        gateway_readable_database_credentials = ("invalid",)
        child_readable_database_credentials = ("invalid",)
        gateway_readable_environment_files = ("invalid",)
        child_readable_environment_files = ("invalid",)
        database_password_names = ("invalid",)
        database_connection_names = ("invalid",)
        source_shape_ok = False
        unique_names = False
        source_names = set()

    return [
        PreflightCheck(
            "runtime_secrets.evidence_complete",
            evidence.get("complete") is True,
            "runtime secret-source inventory must be explicitly complete",
        ),
        PreflightCheck(
            "runtime_secrets.legacy_env_absent",
            _legacy_hermes_env_path(legacy.get("path"))
            and legacy.get("exists") is False
            and _access_is_denied(legacy.get("gateway_file_access"))
            and _access_is_denied(legacy.get("gateway_child_file_access")),
            "the legacy ~/.hermes/.env must be absent and inaccessible to the gateway and its children",
        ),
        PreflightCheck(
            "runtime_secrets.no_readable_files",
            not gateway_readable and not child_readable,
            "no secret-bearing file may be readable by the gateway UID or its children",
        ),
        PreflightCheck(
            "runtime_secrets.pgpass_absent",
            _pgpass_path(pgpass.get("path"))
            and pgpass.get("exists") is False
            and _access_is_denied(pgpass.get("gateway_file_access"))
            and _access_is_denied(pgpass.get("gateway_child_file_access")),
            "~/.pgpass must be absent and inaccessible to the gateway and its children",
        ),
        PreflightCheck(
            "runtime_secrets.no_readable_database_credentials",
            not gateway_readable_database_credentials
            and not child_readable_database_credentials,
            "no database credential file may be readable by the gateway UID or its children",
        ),
        PreflightCheck(
            "runtime_secrets.no_readable_systemd_environment",
            not gateway_readable_environment_files
            and not child_readable_environment_files,
            "no systemd EnvironmentFile may be readable by the gateway UID or its children",
        ),
        PreflightCheck(
            "runtime_secrets.no_database_credential_fds",
            _empty_sequence(evidence.get("open_database_credential_fds"))
            and _empty_sequence(
                evidence.get("inherited_database_credential_fds")
            ),
            "the gateway and its children must have no open or inherited database credential descriptor",
        ),
        PreflightCheck(
            "runtime_secrets.no_cloud_sql_socket",
            bool(str(cloud_sql_socket.get("path") or "").strip())
            and _access_is_denied(cloud_sql_socket.get("gateway_access"))
            and _access_is_denied(
                cloud_sql_socket.get("gateway_child_access")
            )
            and _empty_sequence(cloud_sql_socket.get("open_fds"))
            and _empty_sequence(cloud_sql_socket.get("inherited_fds")),
            "the gateway and its children must have no Cloud SQL Unix-socket access or descriptor",
        ),
        PreflightCheck(
            "runtime_secrets.gateway_env_database_clean",
            effective_env_shape_ok
            and effective_env.get("complete") is True
            and effective_env.get("values_included") is False
            and not database_password_names
            and not database_connection_names,
            "effective gateway environment evidence must contain names only and no database password or connection-secret variable",
        ),
        PreflightCheck(
            "runtime_secrets.trusted_sources",
            source_shape_ok and unique_names,
            "each unique runtime secret must be root/systemd-provisioned and file-inaccessible to the gateway and its children",
        ),
        PreflightCheck(
            "runtime_secrets.discord_source_isolated",
            "discord_bot_token" in source_names,
            "the Discord bot token must have a root/systemd source that is file-inaccessible to gateway children",
        ),
    ]


def _table_grants(value: Any) -> tuple[TablePrivilegeGrant, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("table grants must be a sequence")
    grants = []
    for raw in value:
        item = _mapping(raw)
        privileges = item.get("privileges", ())
        if not isinstance(privileges, Sequence) or isinstance(
            privileges, (str, bytes, bytearray)
        ):
            raise ValueError("table privileges must be a sequence")
        grants.append(
            TablePrivilegeGrant(
                table=str(item.get("table") or ""),
                privileges=tuple(str(privilege) for privilege in privileges),
            )
        )
    return tuple(grants)


def _sequence_grants(value: Any) -> tuple[SequencePrivilegeGrant, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("sequence grants must be a sequence")
    grants = []
    for raw in value:
        item = _mapping(raw)
        privileges = item.get("privileges", ())
        if not isinstance(privileges, Sequence) or isinstance(
            privileges, (str, bytes, bytearray)
        ):
            raise ValueError("sequence privileges must be a sequence")
        grants.append(
            SequencePrivilegeGrant(
                sequence=str(item.get("sequence") or ""),
                privileges=tuple(str(privilege) for privilege in privileges),
            )
        )
    return tuple(grants)


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("expected a sequence of strings")
    if any(not isinstance(item, str) for item in value):
        raise ValueError("expected a sequence of strings")
    return tuple(value)


def _routine_identities(value: Any) -> tuple[RoutineIdentity, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("routine identities must be a sequence")
    identities = []
    for raw in value:
        item = _mapping(raw)
        identities.append(
            RoutineIdentity(
                signature=str(item.get("signature") or ""),
                owner=str(item.get("owner") or ""),
                security_definer=item.get("security_definer") is True,
                language=str(item.get("language") or ""),
                configuration=_strings(item.get("configuration", ())),
                definition_sha256=str(item.get("definition_sha256") or ""),
                owner_dangerous=item.get("owner_dangerous") is True,
            )
        )
    return tuple(identities)


def _canonical_event_log_identity(value: Any) -> CanonicalEventLogIdentity:
    item = _mapping(value)
    return CanonicalEventLogIdentity(
        table=str(item.get("table") or ""),
        owner=str(item.get("owner") or ""),
        owner_dangerous=item.get("owner_dangerous") is True,
        relation_kind=str(item.get("relation_kind") or ""),
        persistence=str(item.get("persistence") or ""),
        is_partition=item.get("is_partition") is True,
        access_method=str(item.get("access_method") or ""),
        tablespace_oid=int(item.get("tablespace_oid", -1)),
        row_security=item.get("row_security") is True,
        force_row_security=item.get("force_row_security") is True,
        replica_identity=str(item.get("replica_identity") or ""),
        relation_options=_strings(item.get("relation_options", ())),
        columns=_strings(item.get("columns", ())),
        constraints=_strings(item.get("constraints", ())),
        user_triggers=_strings(item.get("user_triggers", ())),
        rewrite_rules=_strings(item.get("rewrite_rules", ())),
        policies=_strings(item.get("policies", ())),
        inheritance=item.get("inheritance") is True,
        non_owner_acl_grants=_strings(
            item.get("non_owner_acl_grants", ())
        ),
        index_count=int(item.get("index_count", -1)),
        primary_index_exact=item.get("primary_index_exact") is True,
    )


def _parse_database_attestation(
    value: Any,
) -> tuple[str, WriterPrivilegePolicy, PrivilegeAttestation]:
    database = _mapping(value)
    expected_user = str(database.get("expected_user") or "")
    policy_raw = _mapping(database.get("policy"))
    attestation_raw = _mapping(database.get("attestation"))
    raw_schema_privileges = tuple(
        sorted(value.upper() for value in _strings(policy_raw.get("schema_privileges")))
    )
    raw_database_privileges = tuple(
        sorted(
            value.upper()
            for value in _strings(policy_raw.get("database_privileges"))
        )
    )
    raw_role_memberships = tuple(
        sorted(_strings(policy_raw.get("role_memberships")))
    )
    if raw_schema_privileges != ("USAGE",):
        raise ValueError("writer schema privileges must be exactly USAGE")
    if raw_database_privileges != ("CONNECT",):
        raise ValueError("writer database privileges must be exactly CONNECT")
    if raw_role_memberships != (CANONICAL_WRITER_ROLE,):
        raise ValueError("writer role membership does not match production")
    policy = WriterPrivilegePolicy(
        schema=str(policy_raw.get("schema") or ""),
        table_grants=_table_grants(policy_raw.get("table_grants", ())),
        sequence_grants=_sequence_grants(policy_raw.get("sequence_grants", ())),
        executable_routines=_strings(policy_raw.get("executable_routines", ())),
        routine_identities=_routine_identities(
            policy_raw.get("routine_identities", ())
        ),
        dependency_routine_identities=_routine_identities(
            policy_raw.get("helper_routine_identities", ())
        ),
        schema_privileges=raw_schema_privileges,
        database_privileges=raw_database_privileges,
        role_memberships=raw_role_memberships,
    )
    attested_schema_privileges = tuple(
        sorted(
            value.upper()
            for value in _strings(attestation_raw.get("schema_privileges"))
        )
    )
    attested_database_privileges = tuple(
        sorted(
            value.upper()
            for value in _strings(attestation_raw.get("database_privileges"))
        )
    )
    attested_role_memberships = tuple(
        sorted(_strings(attestation_raw.get("role_memberships")))
    )
    attestation = PrivilegeAttestation(
        role=str(attestation_raw.get("role") or ""),
        superuser=attestation_raw.get("superuser") is True,
        createdb=attestation_raw.get("createdb") is True,
        createrole=attestation_raw.get("createrole") is True,
        replication=attestation_raw.get("replication") is True,
        bypassrls=attestation_raw.get("bypassrls") is True,
        table_owner=attestation_raw.get("table_owner") is True,
        routine_owner=attestation_raw.get("routine_owner") is True,
        table_grants=_table_grants(attestation_raw.get("table_grants", ())),
        sequence_grants=_sequence_grants(
            attestation_raw.get("sequence_grants", ())
        ),
        executable_routines=_strings(
            attestation_raw.get("executable_routines", ())
        ),
        routine_identities=_routine_identities(
            attestation_raw.get("routine_identities", ())
        ),
        dependency_routine_identities=_routine_identities(
            attestation_raw.get("helper_routine_identities", ())
        ),
        schema_privileges=attested_schema_privileges,
        database_privileges=attested_database_privileges,
        role_memberships=attested_role_memberships,
        unexpected_privileges=_strings(
            attestation_raw.get("unexpected_privileges", ())
        ),
        public_acl_grants=_strings(attestation_raw.get("public_acl_grants", ())),
        canonical_non_owner_acl_grants=_strings(
            attestation_raw.get("canonical_non_owner_acl_grants", ())
        ),
        canonical_writer_role_inheritors=_strings(
            attestation_raw.get("canonical_writer_role_inheritors", ())
        ),
        canonical_event_log_identity=_canonical_event_log_identity(
            attestation_raw.get("canonical_event_log_identity")
        ),
    )
    if policy.schema != CANONICAL_WRITER_SCHEMA:
        raise ValueError("writer schema does not match production policy")
    if policy.table_grants or policy.sequence_grants:
        raise ValueError("writer role must have zero table and sequence grants")
    if policy.executable_routines != EXPECTED_ROUTINE_SIGNATURES:
        raise ValueError("writer executable routine set does not match production")
    if {
        identity.signature for identity in policy.dependency_routine_identities
    } != set(EXPECTED_HELPER_ROUTINE_SIGNATURES):
        raise ValueError("writer helper routine identity set does not match production")
    if policy.schema_privileges != ("USAGE",):
        raise ValueError("writer schema privileges must be exactly USAGE")
    if policy.database_privileges != ("CONNECT",):
        raise ValueError("writer database privileges must be exactly CONNECT")
    if policy.role_memberships != (CANONICAL_WRITER_ROLE,):
        raise ValueError("writer role membership does not match production")
    if any(
        identity.owner != CANONICAL_WRITER_MIGRATION_OWNER
        for identity in (
            policy.routine_identities + policy.dependency_routine_identities
        )
    ):
        raise ValueError("writer routine owner does not match production")
    if any(not identity.security_definer for identity in policy.routine_identities):
        raise ValueError("public writer routines must be SECURITY DEFINER")
    if any(
        identity.security_definer
        for identity in policy.dependency_routine_identities
    ):
        raise ValueError("dependency helper routines must be SECURITY INVOKER")
    if attested_schema_privileges != raw_schema_privileges:
        raise ValueError("attested schema privileges do not match production")
    if attested_database_privileges != raw_database_privileges:
        raise ValueError("attested database privileges do not match production")
    if attested_role_memberships != raw_role_memberships:
        raise ValueError("attested role memberships do not match production")
    return expected_user, policy, attestation


def _systemd_checks(properties: Mapping[str, Any]) -> list[PreflightCheck]:
    checks = [
        PreflightCheck(
            f"systemd.{name}",
            _enabled(properties.get(name)),
            "must be enabled",
        )
        for name in _HARDENED_TRUE_PROPERTIES
    ]
    checks.extend(
        (
            PreflightCheck(
                "systemd.ProtectSystem",
                str(properties.get("ProtectSystem") or "").casefold() == "strict",
                "must be strict",
            ),
            PreflightCheck(
                "systemd.ProtectHome",
                str(properties.get("ProtectHome") or "").casefold()
                in {"yes", "read-only", "tmpfs"},
                "must hide or make home read-only",
            ),
            PreflightCheck(
                "systemd.ProtectProc",
                str(properties.get("ProtectProc") or "").casefold() == "invisible",
                "must be invisible",
            ),
            PreflightCheck(
                "systemd.ProcSubset",
                str(properties.get("ProcSubset") or "").casefold() == "pid",
                "must be pid",
            ),
            PreflightCheck(
                "systemd.UMask",
                _mode(properties.get("UMask")) == 0o077,
                "must be 0077",
            ),
            PreflightCheck(
                "systemd.CapabilityBoundingSet",
                _disabled_or_empty(properties.get("CapabilityBoundingSet")),
                "must be empty",
            ),
            PreflightCheck(
                "systemd.AmbientCapabilities",
                _disabled_or_empty(properties.get("AmbientCapabilities")),
                "must be empty",
            ),
        )
    )
    raw_families = properties.get("RestrictAddressFamilies")
    if isinstance(raw_families, str):
        families = set(raw_families.split())
    elif isinstance(raw_families, Sequence):
        families = {str(item) for item in raw_families}
    else:
        families = set()
    allowed_families = {"AF_UNIX", "AF_INET", "AF_INET6"}
    families_ok = (
        "AF_UNIX" in families
        and families <= allowed_families
        and not any(item.startswith("~") for item in families)
    )
    checks.append(
        PreflightCheck(
            "systemd.RestrictAddressFamilies",
            families_ok,
            "must be an allow-list containing AF_UNIX and only TCP/IP families",
        )
    )
    return checks


def evaluate_snapshot(snapshot: Mapping[str, Any]) -> PreflightReport:
    """Evaluate deterministic deployment invariants from a collected snapshot."""

    gateway_uid = _integer(snapshot.get("gateway_uid"))
    gateway_gid = _integer(snapshot.get("gateway_gid"))
    writer_uid = _integer(snapshot.get("writer_uid"))
    writer_gid = _integer(snapshot.get("writer_gid"))
    projector_gid = _integer(snapshot.get("projector_gid"))
    helper = _mapping(snapshot.get("helper"))
    writer_socket = _mapping(snapshot.get("socket"))
    credential = _mapping(snapshot.get("credential"))
    projection_export = _mapping(snapshot.get("projection_export"))
    expected_group_gid = _integer(writer_socket.get("expected_group_gid"))
    socket_mode = _mode(writer_socket.get("mode"))
    credential_mode = _mode(credential.get("mode"))
    projection_export_mode = _mode(projection_export.get("mode"))
    collected_at_unix = _integer(snapshot.get("collected_at_unix"))
    raw_supplementary_gids = snapshot.get("writer_supplementary_gids")
    if isinstance(raw_supplementary_gids, Sequence) and not isinstance(
        raw_supplementary_gids,
        (str, bytes, bytearray),
    ):
        supplementary_gids = tuple(_integer(value) for value in raw_supplementary_gids)
    else:
        supplementary_gids = ()

    checks: list[PreflightCheck] = [
        PreflightCheck(
            "identity.distinct_uids",
            gateway_uid >= 0 and writer_uid >= 0 and gateway_uid != writer_uid,
            "gateway and writer must use distinct valid UIDs",
        ),
        PreflightCheck(
            "identity.projector_gid_isolated",
            writer_gid >= 0
            and projector_gid >= 0
            and projector_gid != writer_gid
            and projector_gid != expected_group_gid,
            "projector GID must differ from writer and socket client groups",
        ),
        PreflightCheck(
            "identity.writer_projector_membership",
            supplementary_gids == (projector_gid,),
            "writer must have exactly the dedicated projector supplementary GID",
        ),
        PreflightCheck(
            "helper.legacy_absent",
            helper.get("exists") is False,
            "legacy Cloud SQL helper must be removed from the runtime host",
        ),
        PreflightCheck(
            "helper.gateway_inaccessible",
            _access_is_denied(helper.get("gateway_access")),
            "gateway must have no read, write, or execute access to helper",
        ),
        PreflightCheck(
            "socket.writer_owned",
            _integer(writer_socket.get("owner_uid")) == writer_uid,
            "socket must be owned by the writer UID",
        ),
        PreflightCheck(
            "socket.dedicated_group",
            expected_group_gid >= 0
            and _integer(writer_socket.get("group_gid")) == expected_group_gid,
            "socket must use the declared dedicated client group",
        ),
        PreflightCheck(
            "socket.mode",
            socket_mode == 0o660 and socket_mode & stat.S_IRWXO == 0,
            "socket mode must be exactly 0660",
        ),
        PreflightCheck(
            "credential.writer_owned",
            _integer(credential.get("owner_uid")) == writer_uid,
            "credential must be owned by the writer UID",
        ),
        PreflightCheck(
            "credential.mode",
            credential_mode in {0o400, 0o600},
            "credential mode must be 0400 or 0600",
        ),
        PreflightCheck(
            "credential.gateway_inaccessible",
            _access_is_denied(credential.get("gateway_access")),
            "gateway must have no read, write, or execute access to credential",
        ),
        PreflightCheck(
            "projection_export.writer_owned",
            _integer(projection_export.get("owner_uid")) == writer_uid,
            "projection export must be owned by the writer UID",
        ),
        PreflightCheck(
            "projection_export.projector_group",
            projector_gid >= 0
            and _integer(projection_export.get("group_gid")) == projector_gid,
            "projection export must use the dedicated projector GID",
        ),
        PreflightCheck(
            "projection_export.mode",
            projection_export_mode == 0o640,
            "projection export mode must be exactly 0640",
        ),
        PreflightCheck(
            "projection_export.gateway_inaccessible",
            _access_is_denied(projection_export.get("gateway_access")),
            "gateway must have no access to projection export",
        ),
        PreflightCheck(
            "projection_export.projector_read_only",
            _access_is_read_only(projection_export.get("projector_access")),
            "projector must have read-only projection export access",
        ),
    ]
    checks.extend(
        _gateway_process_checks(
            snapshot.get("gateway_process"),
            collected_at_unix=collected_at_unix,
        )
    )
    checks.extend(
        _runtime_secret_source_checks(snapshot.get("runtime_secret_sources"))
    )
    checks.extend(
        _writer_deployment_checks(
            snapshot.get("writer_deployment"),
            collected_at_unix=collected_at_unix,
            writer_uid=writer_uid,
            writer_gid=writer_gid,
        )
    )
    checks.extend(
        _gateway_deployment_checks(
            snapshot.get("gateway_deployment"),
            writer_deployment=snapshot.get("writer_deployment"),
            collected_at_unix=collected_at_unix,
            gateway_uid=gateway_uid,
            gateway_gid=gateway_gid,
            gateway_process_snapshot=snapshot.get("gateway_process"),
        )
    )
    checks.extend(
        _writer_authority_surface_checks(
            snapshot.get("writer_authority_surface"),
            writer_deployment=snapshot.get("writer_deployment"),
            gateway_process_snapshot=snapshot.get("gateway_process"),
            collected_at_unix=collected_at_unix,
            gateway_uid=gateway_uid,
            gateway_gid=gateway_gid,
            writer_uid=writer_uid,
            writer_gid=writer_gid,
            socket_group_gid=expected_group_gid,
            projector_gid=projector_gid,
        )
    )
    checks.extend(_systemd_checks(_mapping(snapshot.get("systemd_properties"))))
    gateway_systemd = _systemd_checks(
        _mapping(snapshot.get("gateway_systemd_properties"))
    )
    checks.extend(
        PreflightCheck(
            "gateway_" + check.name,
            check.passed,
            check.detail,
        )
        for check in gateway_systemd
    )

    gateway_iam = _mapping(snapshot.get("gateway_iam"))
    try:
        roles = tuple(item.casefold() for item in _strings(gateway_iam.get("roles")))
        permissions = tuple(
            item.casefold() for item in _strings(gateway_iam.get("permissions"))
        )
    except ValueError:
        roles = ()
        permissions = ()
        iam_shape_ok = False
    else:
        iam_shape_ok = gateway_iam.get("complete") is True
    roles_ok = iam_shape_ok and not any(
        role in _FORBIDDEN_BROAD_ROLES
        or any(marker in role for marker in _FORBIDDEN_IAM_MARKERS)
        for role in roles
    )
    permissions_ok = iam_shape_ok and not any(
        permission.startswith(("cloudsql.", "secretmanager."))
        for permission in permissions
    )
    checks.extend(
        (
            PreflightCheck(
                "iam.gateway_evidence_complete",
                iam_shape_ok,
                "effective gateway IAM evidence must be explicitly complete",
            ),
            PreflightCheck(
                "iam.gateway_roles",
                roles_ok,
                "gateway must have no Cloud SQL, Secret Manager, owner, or editor role",
            ),
            PreflightCheck(
                "iam.gateway_permissions",
                permissions_ok,
                "gateway must have no effective Cloud SQL or Secret Manager permission",
            ),
        )
    )

    writer_iam = _mapping(snapshot.get("writer_iam"))
    try:
        writer_roles = tuple(
            item.casefold() for item in _strings(writer_iam.get("roles"))
        )
        writer_permissions = tuple(
            item.casefold() for item in _strings(writer_iam.get("permissions"))
        )
    except ValueError:
        writer_roles = ()
        writer_permissions = ()
        writer_iam_shape_ok = False
    else:
        writer_iam_shape_ok = writer_iam.get("complete") is True
    writer_roles_ok = writer_iam_shape_ok and not any(
        role in _FORBIDDEN_BROAD_ROLES
        or any(marker in role for marker in _FORBIDDEN_IAM_MARKERS)
        for role in writer_roles
    )
    writer_permissions_ok = writer_iam_shape_ok and not any(
        permission.startswith(("cloudsql.", "secretmanager."))
        for permission in writer_permissions
    )
    checks.extend(
        (
            PreflightCheck(
                "iam.writer_evidence_complete",
                writer_iam_shape_ok,
                "effective writer IAM evidence must be explicitly complete",
            ),
            PreflightCheck(
                "iam.writer_roles",
                writer_roles_ok,
                "writer must not depend on ambient Cloud SQL or Secret Manager roles",
            ),
            PreflightCheck(
                "iam.writer_permissions",
                writer_permissions_ok,
                "writer must not have ambient Cloud SQL or Secret Manager permissions",
            ),
        )
    )

    try:
        expected_user, policy, attestation = _parse_database_attestation(
            snapshot.get("database")
        )
        validate_privilege_attestation(
            attestation,
            policy,
            expected_user=expected_user,
        )
    except (PrivilegeAttestationError, TypeError, ValueError) as exc:
        database_ok = False
        database_detail = f"least-privilege attestation rejected: {exc}"
    else:
        database_ok = True
        database_detail = "least-privilege attestation matches exact policy"
    checks.append(
        PreflightCheck("database.least_privilege", database_ok, database_detail)
    )
    return PreflightReport(tuple(checks))


def _read_snapshot(path: str) -> Mapping[str, Any]:
    if path == "-":
        value = json.load(sys.stdin)
    else:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("snapshot root must be a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="snapshot JSON path, or - for stdin",
    )
    arguments = parser.parse_args(argv)
    try:
        report = evaluate_snapshot(_read_snapshot(arguments.input))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = PreflightReport(
            (PreflightCheck("snapshot.valid", False, f"invalid snapshot: {exc}"),)
        )
    print(json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")))
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
