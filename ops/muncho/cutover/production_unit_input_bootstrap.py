#!/usr/bin/python3
"""Root-only bootstrap for signed, non-secret production unit inputs.

This file is intentionally stdlib-only.  Auto-deploy extracts its exact Git
blob from the SHA-verified target clone into a root-owned temporary file before
execution, so the first release does not depend on the old active package or
on an owner-writable Python import path.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


STAGED_ROOT = Path("/var/lib/muncho-production-legacy-cutover/staged")
PLAN_PATH = STAGED_ROOT / "unit-input-plan.json"
APPROVAL_PATH = STAGED_ROOT / "unit-input-approval.json"
UNIT_INPUTS_PATH = STAGED_ROOT / "production-unit-inputs.json"
OPENSSL = Path("/usr/bin/openssl")

PLAN_SCHEMA = "muncho-production-cutover-unit-input-plan.v3"
PAYLOAD_SCHEMA = "muncho-production-cutover-unit-input-payload.v3"
APPROVAL_SCHEMA = "muncho-production-cutover-unit-input-approval.v3"
UNIT_INPUT_SCHEMA = "muncho-production-cutover-unit-inputs.v3"
RECEIPT_SCHEMA = "muncho-production-cutover-unit-input-staging.v3"
DISCORD_RECONCILIATION_INTENT_SCHEMA = (
    "muncho-production-discord-reconciliation-intent.v1"
)
DISCORD_RECONCILIATION_INTENT_PURPOSE = (
    "production_discord_policy_reconciliation"
)
REVISION = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
IDENTITY = re.compile(r"^[a-z_][a-z0-9_-]{0,63}$")
PLAN_FIELDS = frozenset({
    "schema", "release_revision", "unit_inputs", "owner_subject_sha256",
    "owner_public_key_ed25519_hex", "owner_key_id", "owner_runtime_attestation",
    "created_at_unix",
    "secret_material_recorded", "secret_digest_recorded", "plan_sha256",
})
RUNTIME_ATTESTATION_FIELDS = frozenset({
    "schema", "revision", "manifest_sha256", "tree_sha256",
    "interpreter_sha256", "pyvenv_cfg_sha256", "sys_path_sha256",
    "required_modules_sha256", "module_origins_release_local",
    "ambient_python_environment_present", "secret_material_recorded",
    "secret_digest_recorded", "attestation_sha256",
})
PAYLOAD_FIELDS = frozenset({
    "schema", "database_ip", "target", "gateway", "routeback", "mac_ops", "browser",
    "writer", "projector", "connector", "worker", "writer_client_group",
    "worker_client_group",
    "operational_edge_identities", "operational_edge_socket_groups",
    "writer_capability_public_key_id", "discord_edge_receipt_public_key_id",
    "operational_edge_key_foundation_sha256", "discord_reconciliation_intent",
    "operational_edge_receipt_public_key_ids", "release_owner_uid",
    "release_owner_gid", "bwrap_sha256",
    "shell_sha256", "secret_material_recorded", "secret_digest_recorded",
})
APPROVAL_FIELDS = frozenset({
    "schema", "purpose", "plan_sha256", "release_revision",
    "owner_subject_sha256", "owner_public_key_ed25519_hex", "owner_key_id",
    "nonce_sha256", "issued_at_unix", "expires_at_unix", "approved",
    "signature_ed25519_hex", "approval_sha256",
})
IDENTITY_FIELDS = frozenset({"user", "group", "uid", "gid"})
CLIENT_GROUP_FIELDS = frozenset({"group", "gid"})
DISCORD_RECONCILIATION_INTENT_FIELDS = frozenset({
    "schema", "purpose", "release_revision", "legacy_public_policy_sha256",
    "target_public_policy_sha256", "reviewed_reconciliation",
    "secret_material_recorded", "secret_digest_recorded",
})
TARGET_FIELDS = frozenset({
    "project", "zone", "vm", "database", "sql_instance", "sql_host",
    "tls_server_name", "port", "writer_login",
})
OPERATIONAL_EDGE_DOMAINS = frozenset({
    "adventico_email", "bitrix", "canonical", "github", "infrastructure",
    "skyvision_db", "skyvision_email", "skyvision_gitlab", "skyvision_panel",
})
SPKI_ED25519_PREFIX = bytes.fromhex("302a300506032b6570032100")


class BootstrapError(RuntimeError):
    """Stable, secret-free bootstrap failure."""


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


def _identity(item: os.stat_result) -> tuple[int, ...]:
    return (
        item.st_dev, item.st_ino, item.st_mode, item.st_nlink,
        item.st_uid, item.st_gid, item.st_size, item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _read_exact(
    path: Path,
    *,
    uid: int,
    gid: int,
    mode: int,
    maximum: int,
) -> bytes:
    descriptor: int | None = None
    try:
        before = os.lstat(path)
        if (
            path.resolve(strict=True) != path
            or stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != uid
            or before.st_gid != gid
            or stat.S_IMODE(before.st_mode) != mode
            or not 0 < before.st_size <= maximum
        ):
            raise BootstrapError("unit_input_bootstrap_file_identity_invalid")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(
                descriptor,
                min(64 * 1024, maximum + 1 - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        reachable = os.lstat(path)
    except BootstrapError:
        raise
    except OSError as exc:
        raise BootstrapError("unit_input_bootstrap_file_unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        len(raw) != before.st_size
        or len(raw) > maximum
        or _identity(before) != _identity(opened)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(reachable)
    ):
        raise BootstrapError("unit_input_bootstrap_file_changed")
    return bytes(raw)


def _decode(raw: bytes) -> Mapping[str, Any]:
    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise BootstrapError("unit_input_bootstrap_duplicate_key")
            result[name] = value
        return result

    def constant(_value):
        raise BootstrapError("unit_input_bootstrap_nonfinite_number")

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except BootstrapError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BootstrapError("unit_input_bootstrap_json_invalid") from exc
    if not isinstance(value, Mapping) or raw != _canonical(value):
        raise BootstrapError("unit_input_bootstrap_json_not_canonical")
    return value


def _self_hashed(
    value: Mapping[str, Any],
    *,
    fields: frozenset[str],
    digest_field: str,
    code: str,
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or SHA256.fullmatch(str(value.get(digest_field))) is None
    ):
        raise BootstrapError(code)
    unsigned = {key: item for key, item in value.items() if key != digest_field}
    if _sha(_canonical(unsigned)) != value[digest_field]:
        raise BootstrapError(code)
    return value


def _payload(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != PAYLOAD_FIELDS:
        raise BootstrapError("unit_input_bootstrap_payload_invalid")
    identities = []
    for name in (
        "gateway", "writer", "projector", "routeback", "connector",
        "mac_ops", "browser", "worker",
    ):
        item = value[name]
        if (
            not isinstance(item, Mapping)
            or set(item) != IDENTITY_FIELDS
            or IDENTITY.fullmatch(str(item["user"])) is None
            or IDENTITY.fullmatch(str(item["group"])) is None
            or type(item["uid"]) is not int
            or type(item["gid"]) is not int
            or item["uid"] <= 0
            or item["gid"] <= 0
        ):
            raise BootstrapError("unit_input_bootstrap_identity_invalid")
        identities.append(item)
    expected_identity_names = {
        "gateway": "ai-platform-brain",
        "writer": "muncho-canonical-writer",
        "projector": "muncho-projector",
        "routeback": "muncho-discord-egress",
        "connector": "muncho-discord-connector",
        "mac_ops": "muncho-mac-ops-edge",
        "browser": "muncho-capability-browser",
        "worker": "muncho-worker",
    }
    if any(
        value[role]["user"] != name or value[role]["group"] != name
        for role, name in expected_identity_names.items()
    ):
        raise BootstrapError("unit_input_bootstrap_identity_invalid")
    client_groups = []
    for field, expected_name in (
        ("writer_client_group", "muncho-writer-client"),
        ("worker_client_group", "muncho-worker-clients"),
    ):
        group = value[field]
        if (
            not isinstance(group, Mapping)
            or set(group) != CLIENT_GROUP_FIELDS
            or group.get("group") != expected_name
            or type(group.get("gid")) is not int
            or group["gid"] <= 0
        ):
            raise BootstrapError("unit_input_bootstrap_identity_invalid")
        client_groups.append(group)
    operational_identities = value.get("operational_edge_identities")
    operational_socket_groups = value.get("operational_edge_socket_groups")
    if (
        not isinstance(operational_identities, Mapping)
        or set(operational_identities) != OPERATIONAL_EDGE_DOMAINS
        or not isinstance(operational_socket_groups, Mapping)
        or set(operational_socket_groups) != OPERATIONAL_EDGE_DOMAINS
    ):
        raise BootstrapError("unit_input_bootstrap_identity_invalid")
    socket_gids: set[int] = set()
    for domain in sorted(OPERATIONAL_EDGE_DOMAINS):
        item = operational_identities[domain]
        socket = operational_socket_groups[domain]
        expected_name = f"muncho-edge-{domain}"
        expected_socket_group = f"muncho-edge-{domain}-c"
        if (
            not isinstance(item, Mapping)
            or set(item) != IDENTITY_FIELDS
            or item.get("user") != expected_name
            or item.get("group") != expected_name
            or type(item.get("uid")) is not int
            or type(item.get("gid")) is not int
            or item["uid"] <= 0
            or item["gid"] <= 0
            or not isinstance(socket, Mapping)
            or set(socket) != {"group", "gid"}
            or socket.get("group") != expected_socket_group
            or type(socket.get("gid")) is not int
            or socket["gid"] <= 0
        ):
            raise BootstrapError("unit_input_bootstrap_identity_invalid")
        identities.append(item)
        socket_gids.add(socket["gid"])
    target = value["target"]
    receipt_key_ids = value["operational_edge_receipt_public_key_ids"]
    reconciliation = value["discord_reconciliation_intent"]
    try:
        address = ipaddress.ip_address(str(target.get("sql_host")))
    except (AttributeError, ValueError) as exc:
        raise BootstrapError("unit_input_bootstrap_target_invalid") from exc
    if (
        not isinstance(target, Mapping)
        or set(target) != TARGET_FIELDS
        or target["project"] != "adventico-ai-platform"
        or target["zone"] != "europe-west3-a"
        or target["vm"] != "ai-platform-runtime-01"
        or target["database"] != "ai_platform_brain"
        or target["port"] != 5432
        or str(address) != target["sql_host"]
        or target["sql_host"] != value["database_ip"]
        or re.fullmatch(r"[a-z][a-z0-9-]{0,62}", str(target["sql_instance"]))
        is None
        or not isinstance(target["tls_server_name"], str)
        or len(target["tls_server_name"]) > 253
        or re.fullmatch(r"[A-Za-z0-9.-]+", target["tls_server_name"]) is None
        or IDENTITY.fullmatch(str(target["writer_login"])) is None
    ):
        raise BootstrapError("unit_input_bootstrap_target_invalid")
    if (
        value["schema"] != PAYLOAD_SCHEMA
        or not isinstance(value["database_ip"], str)
        or not value["database_ip"]
        or value["worker"]["user"] != "muncho-worker"
        or value["worker"]["group"] != "muncho-worker"
        or SHA256.fullmatch(
            str(value["writer_capability_public_key_id"])
        )
        is None
        or SHA256.fullmatch(
            str(value["discord_edge_receipt_public_key_id"])
        )
        is None
        or SHA256.fullmatch(
            str(value["operational_edge_key_foundation_sha256"])
        )
        is None
        or not isinstance(receipt_key_ids, Mapping)
        or set(receipt_key_ids) != OPERATIONAL_EDGE_DOMAINS
        or any(
            SHA256.fullmatch(str(key_id)) is None
            for key_id in receipt_key_ids.values()
        )
        or len(set(receipt_key_ids.values())) != len(receipt_key_ids)
        or value["writer_capability_public_key_id"]
        in set(receipt_key_ids.values())
        or value["discord_edge_receipt_public_key_id"]
        in (
            set(receipt_key_ids.values())
            | {value["writer_capability_public_key_id"]}
        )
        or not isinstance(reconciliation, Mapping)
        or set(reconciliation) != DISCORD_RECONCILIATION_INTENT_FIELDS
        or reconciliation.get("schema")
        != DISCORD_RECONCILIATION_INTENT_SCHEMA
        or reconciliation.get("purpose")
        != DISCORD_RECONCILIATION_INTENT_PURPOSE
        or REVISION.fullmatch(str(reconciliation.get("release_revision")))
        is None
        or SHA256.fullmatch(
            str(reconciliation.get("legacy_public_policy_sha256"))
        )
        is None
        or SHA256.fullmatch(
            str(reconciliation.get("target_public_policy_sha256"))
        )
        is None
        or reconciliation["legacy_public_policy_sha256"]
        == reconciliation["target_public_policy_sha256"]
        or reconciliation.get("reviewed_reconciliation") is not True
        or reconciliation.get("secret_material_recorded") is not False
        or reconciliation.get("secret_digest_recorded") is not False
        or type(value["release_owner_uid"]) is not int
        or type(value["release_owner_gid"]) is not int
        or value["release_owner_uid"] != value["gateway"]["uid"]
        or value["release_owner_gid"] != value["gateway"]["gid"]
        or SHA256.fullmatch(str(value["bwrap_sha256"])) is None
        or SHA256.fullmatch(str(value["shell_sha256"])) is None
        or value["bwrap_sha256"] == value["shell_sha256"]
        or value["secret_material_recorded"] is not False
        or value["secret_digest_recorded"] is not False
        or len({item["uid"] for item in identities}) != len(identities)
        or len(
            {item["gid"] for item in identities}
            | {item["gid"] for item in client_groups}
            | socket_gids
        )
        != len(identities) + len(client_groups) + len(OPERATIONAL_EDGE_DOMAINS)
    ):
        raise BootstrapError("unit_input_bootstrap_payload_invalid")
    return value


def _approval_payload(value: Mapping[str, Any]) -> bytes:
    return _canonical({
        key: item
        for key, item in value.items()
        if key not in {"signature_ed25519_hex", "approval_sha256"}
    })


def _verify_signature(
    public_hex: str,
    signature_hex: str,
    payload: bytes,
    *,
    openssl: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="muncho-unit-input-approval-") as root:
        directory = Path(root)
        public = directory / "owner.der"
        message = directory / "approval.bin"
        signature = directory / "approval.sig"
        public.write_bytes(SPKI_ED25519_PREFIX + bytes.fromhex(public_hex))
        message.write_bytes(payload)
        signature.write_bytes(bytes.fromhex(signature_hex))
        for path in (public, message, signature):
            path.chmod(0o400)
        try:
            result = subprocess.run(
                (
                    str(openssl), "pkeyutl", "-verify", "-pubin",
                    "-inkey", str(public), "-keyform", "DER", "-rawin",
                    "-in", str(message), "-sigfile", str(signature),
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise BootstrapError("unit_input_bootstrap_signature_unavailable") from exc
    if result.returncode != 0:
        raise BootstrapError("unit_input_bootstrap_signature_invalid")


def _install_exact(
    path: Path,
    payload: bytes,
    *,
    uid: int,
    gid: int,
) -> bool:
    created = False
    if not os.path.lexists(path):
        temporary = path.with_name(f".{path.name}.bootstrap.{os.getpid()}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = None
        try:
            descriptor = os.open(temporary, flags, 0o600)
            os.fchown(descriptor, uid, gid)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short unit-input bootstrap write")
                view = view[written:]
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.link(temporary, path, follow_symlinks=False)
            created = True
        except FileExistsError:
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    observed = _read_exact(
        path,
        uid=uid,
        gid=gid,
        mode=0o444,
        maximum=128 * 1024,
    )
    if observed != payload:
        raise BootstrapError("unit_input_bootstrap_conflict")
    return created


def bootstrap(
    *,
    plan_path: Path = PLAN_PATH,
    approval_path: Path = APPROVAL_PATH,
    output_path: Path = UNIT_INPUTS_PATH,
    openssl: Path = OPENSSL,
    now_unix: int | None = None,
    require_root: bool = True,
) -> Mapping[str, Any]:
    if require_root and (
        os.geteuid() != 0
        or not sys.platform.startswith("linux")
        or (plan_path, approval_path, output_path, openssl)
        != (PLAN_PATH, APPROVAL_PATH, UNIT_INPUTS_PATH, OPENSSL)
    ):
        raise BootstrapError("unit_input_bootstrap_boundary_invalid")
    uid = 0 if require_root else os.geteuid()
    gid = 0 if require_root else os.getegid()
    parent = os.lstat(output_path.parent)
    if (
        output_path.parent.resolve(strict=True) != output_path.parent
        or stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != uid
        or parent.st_gid != gid
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise BootstrapError("unit_input_bootstrap_directory_invalid")
    plan = _self_hashed(
        _decode(_read_exact(
            plan_path, uid=uid, gid=gid, mode=0o400, maximum=1024 * 1024,
        )),
        fields=PLAN_FIELDS,
        digest_field="plan_sha256",
        code="unit_input_bootstrap_plan_invalid",
    )
    payload = _payload(plan["unit_inputs"])
    public = plan["owner_public_key_ed25519_hex"]
    runtime_attestation = _self_hashed(
        plan["owner_runtime_attestation"],
        fields=RUNTIME_ATTESTATION_FIELDS,
        digest_field="attestation_sha256",
        code="unit_input_bootstrap_owner_runtime_invalid",
    )
    if (
        plan["schema"] != PLAN_SCHEMA
        or REVISION.fullmatch(str(plan["release_revision"])) is None
        or SHA256.fullmatch(str(plan["owner_subject_sha256"])) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(public)) is None
        or plan["owner_key_id"] != _sha(bytes.fromhex(public))
        or type(plan["created_at_unix"]) is not int
        or plan["created_at_unix"] <= 0
        or plan["secret_material_recorded"] is not False
        or plan["secret_digest_recorded"] is not False
        or payload["discord_reconciliation_intent"]["release_revision"]
        != plan["release_revision"]
        or runtime_attestation["schema"]
        != "muncho-production-owner-runtime-attestation.v1"
        or runtime_attestation["revision"] != plan["release_revision"]
        or any(
            SHA256.fullmatch(str(runtime_attestation[name])) is None
            for name in (
                "manifest_sha256", "tree_sha256", "interpreter_sha256",
                "pyvenv_cfg_sha256", "sys_path_sha256",
                "required_modules_sha256",
            )
        )
        or runtime_attestation["module_origins_release_local"] is not True
        or runtime_attestation["ambient_python_environment_present"] is not False
        or runtime_attestation["secret_material_recorded"] is not False
        or runtime_attestation["secret_digest_recorded"] is not False
    ):
        raise BootstrapError("unit_input_bootstrap_plan_invalid")
    approval = _self_hashed(
        _decode(_read_exact(
            approval_path, uid=uid, gid=gid, mode=0o400, maximum=1024 * 1024,
        )),
        fields=APPROVAL_FIELDS,
        digest_field="approval_sha256",
        code="unit_input_bootstrap_approval_invalid",
    )
    current = int(time.time()) if now_unix is None else now_unix
    if (
        approval["schema"] != APPROVAL_SCHEMA
        or approval["purpose"] != "production_cutover_unit_inputs"
        or approval["plan_sha256"] != plan["plan_sha256"]
        or approval["release_revision"] != plan["release_revision"]
        or approval["owner_subject_sha256"] != plan["owner_subject_sha256"]
        or approval["owner_public_key_ed25519_hex"] != public
        or approval["owner_key_id"] != plan["owner_key_id"]
        or SHA256.fullmatch(str(approval["nonce_sha256"])) is None
        or type(approval["issued_at_unix"]) is not int
        or type(approval["expires_at_unix"]) is not int
        or not approval["issued_at_unix"] <= current < approval["expires_at_unix"]
        or not 1 <= approval["expires_at_unix"] - approval["issued_at_unix"] <= 3600
        or approval["approved"] is not True
        or re.fullmatch(r"[0-9a-f]{128}", str(approval["signature_ed25519_hex"]))
        is None
    ):
        raise BootstrapError("unit_input_bootstrap_approval_invalid")
    _verify_signature(
        public,
        approval["signature_ed25519_hex"],
        _approval_payload(approval),
        openssl=openssl,
    )
    unit_inputs = {
        "schema": UNIT_INPUT_SCHEMA,
        "release_revision": plan["release_revision"],
        "authority_plan_sha256": plan["plan_sha256"],
        "authority_approval_sha256": approval["approval_sha256"],
        **{key: item for key, item in payload.items() if key != "schema"},
    }
    output = _canonical(unit_inputs) + b"\n"
    created = _install_exact(output_path, output, uid=uid, gid=gid)
    unsigned = {
        "schema": RECEIPT_SCHEMA,
        "path": str(output_path),
        "sha256": _sha(output),
        "release_revision": plan["release_revision"],
        "authority_plan_sha256": plan["plan_sha256"],
        "authority_approval_sha256": approval["approval_sha256"],
        "created": created,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha(_canonical(unsigned))}


def main(argv: Sequence[str] | None = None) -> int:
    if list(argv or sys.argv[1:]):
        print('{"error_code":"unit_input_bootstrap_argv_invalid","ok":false}')
        return 2
    try:
        receipt = bootstrap()
    except (BootstrapError, OSError, ValueError):
        print('{"error_code":"unit_input_bootstrap_failed","ok":false}')
        return 2
    print(_canonical(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
