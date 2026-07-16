"""Package the four role-local capability-canary producer services.

The bundle is deliberately mechanical.  It fixes process identities, paths,
keys, systemd hardening, socket access, and immutable native-evidence files.
It never inspects task text and never decides what an operation means.
"""

from __future__ import annotations

import argparse
import copy
import grp
import hashlib
import json
import os
import pwd
import re
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from gateway import canonical_capability_canary_e2e as evidence_contract
from gateway.canonical_capability_canary_producers import (
    BITRIX_OPERATIONAL_EDGE_SERVICE_UNIT,
    BITRIX_OPERATIONAL_EDGE_SOCKET_GROUP,
    BitrixOperationalEdgeNativeCollector,
    BitrixWriterNativeCollector,
    DEFAULT_CONFIG_ROOT,
    DEFAULT_FOUNDATION_PATH,
    DEFAULT_KEY_ROOT,
    DEFAULT_OWNER_PUBLIC_KEY_HEX_PIN_PATH,
    DEFAULT_OWNER_PUBLIC_KEY_SOURCE_SHA256_PIN_PATH,
    DEFAULT_PROBE_CATALOG_PATH,
    DEFAULT_RECEIPT_ROOT,
    DEFAULT_RUNTIME_ROOT,
    ENDPOINT_ROLES,
    PRODUCER_FOUNDATION_SCHEMA,
    PRODUCER_FOUNDATION_SSHSIG_NAMESPACE,
    PRODUCER_CONFIG_SCHEMA,
    PRODUCER_SERVICE_UNITS,
    PRODUCTION_PRE_CLEANUP_PUMP_SLOTS,
    PRODUCTION_OWNER_ID,
    RECEIPT_SLOTS,
    SLOT_FILENAME,
    SLOT_NATIVE_BINDING_KINDS,
    SLOT_ROLE,
    CapabilityProducerError,
    NativeEvidenceBinding,
    ProducerConfig,
    ProductionReceiptPump,
    _canonical_bytes,
    _digest,
    _fail,
    _load_private_key,
    _load_public_key,
    _publish_no_replace,
    _safe_id,
    _sha256_bytes,
    _sha256_json,
    _stable_read,
    _strict,
    _strict_json,
    _validate_full_canary_terminal,
    load_installed_producer_foundation,
    producer_foundation_sha256,
    producer_foundation_signature_payload,
    project_pinned_owner_public_key_source,
    seal_producer_foundation,
    validate_bitrix_operational_edge_contract,
    validate_producer_config_binding,
    validate_producer_foundation,
    validate_probe_catalog,
)


UNIT_BUNDLE_SCHEMA = "muncho-capability-canary-producer-unit-bundle.v1"
KEY_BOOTSTRAP_SCHEMA = "muncho-capability-canary-producer-key-bootstrap.v1"
FOUNDATION_PREPARE_REQUEST_SCHEMA = (
    "muncho-capability-producer-foundation-prepare-request.v2"
)
FOUNDATION_PREPARATION_SCHEMA = "muncho-capability-producer-foundation-preparation.v2"
FOUNDATION_INSTALL_REQUEST_SCHEMA = (
    "muncho-capability-producer-foundation-install-request.v1"
)
FOUNDATION_INSTALL_RECEIPT_SCHEMA = (
    "muncho-capability-producer-foundation-installation.v2"
)
PRODUCER_HOST_IDENTITY_SCHEMA = "muncho-production-capability-producer-host-identity.v2"
PRODUCER_IDENTITY_FOUNDATION_SCHEMA = (
    "muncho-production-capability-producer-identity-foundation.v2"
)
SERVICE_IDENTITY_FOUNDATION_SCHEMA = (
    "muncho-production-capability-service-identity-foundation.v1"
)
CAPABILITY_SERVICE_HOST_IDENTITY_SCHEMA = (
    "muncho-production-capability-service-host-identity.v2"
)
NATIVE_PUBLICATION_SCHEMA = "muncho-capability-canary-role-native-publication.v1"
PRODUCTION_DIFF_OBSERVATION_SCHEMA = "muncho-production-capability-production-diff.v1"
PRODUCTION_DIFF_CATEGORIES = (
    "code",
    "config",
    "identities_permissions",
    "jobs",
    "migration_assets",
)
GATEWAY_OBSERVER_SOURCE_PROJECTION_SCHEMA = (
    "muncho-capability-canary-gateway-observer-source-projection.v1"
)
GOAL_CONTINUATION_NATIVE_IDENTITY_SCHEMA = (
    "muncho-production-capability-goal-continuation-native-identity.v1"
)
GATEWAY_OBSERVER_SOURCE_PROJECTION_FILENAME = "gateway-observer-source-projection.json"
GATEWAY_OBSERVER_PROPOSAL_EVENT_TYPE = "capability.canary.gateway-evidence.proposed"
GATEWAY_OBSERVER_PROPOSAL_CORE_SCHEMA = (
    "muncho-production-capability-gateway-observer-proposal-core.v1"
)
GATEWAY_OBSERVER_PROPOSAL_IDENTITY_SCHEMA = (
    "muncho-capability-canary-model-proposal-core-identity.v1"
)
DEFAULT_NATIVE_ROOT = Path("/run/muncho-capability-canary-native")
PRODUCER_RECEIPT_WRITER_GROUP = "muncho-capability-receipt-writers"
DEFAULT_KEY_BOOTSTRAP_RECEIPT = DEFAULT_KEY_ROOT / "producer-key-bootstrap.json"
DEFAULT_FOUNDATION_CONTROL_ROOT = Path(
    "/var/lib/muncho-capability-canary-control/producer-foundation"
)
DEFAULT_FOUNDATION_PREPARATION_PATH = (
    DEFAULT_FOUNDATION_CONTROL_ROOT / "preparation.json"
)
DEFAULT_FOUNDATION_INSTALL_RECEIPT = (
    DEFAULT_FOUNDATION_CONTROL_ROOT / "installation.json"
)
DEFAULT_PRODUCER_IDENTITY_FOUNDATION_ROOT = (
    DEFAULT_FOUNDATION_CONTROL_ROOT / "producer-identities"
)
DEFAULT_TMPFILES_PATH = Path("/etc/tmpfiles.d/muncho-capability-canary-producers.conf")
PRODUCER_ROLE_ACCOUNTS = {
    "business_edge": ("muncho-cap-business", "muncho-cap-business"),
    "canonical_writer": ("muncho-cap-writer", "muncho-cap-writer"),
    "discord_edge": ("muncho-cap-discord", "muncho-cap-discord"),
    "gateway_observer": ("muncho-cap-observer", "muncho-cap-observer"),
}
PRODUCER_ROLE_NUMERIC_IDENTITIES = {
    "business_edge": (2109, 2212),
    "canonical_writer": (2110, 2213),
    "discord_edge": (2111, 2214),
    "gateway_observer": (2112, 2215),
}
PRODUCER_RECEIPT_WRITER_GID = 2216
PRODUCER_BITRIX_SOCKET_GID = 2211
CANONICAL_WRITER_SERVICE_GROUP = "muncho-canonical-writer"
CANONICAL_WRITER_SOCKET_GROUP = "muncho-writer-client"
DISCORD_ROUTEBACK_GROUP = "muncho-discord-egress"
DISCORD_CONNECTOR_GROUP = "muncho-discord-connector"
DISCORD_EDGE_RECEIPT_PUBLIC_KEY_PATH = Path(
    "/etc/muncho/keys/discord-edge-receipt-public.pem"
)
PRODUCER_OWNER_PUBLIC_KEY_COMMENT = "skyvision-mac-ops-emil-20260710"
PRODUCER_INACCESSIBLE_CREDENTIAL_PATHS = (
    "/etc/muncho/keys",
    "/etc/muncho/discord-edge-credentials",
    "/etc/muncho/discord-connector-credentials",
    "/etc/muncho/mac-ops-edge-credentials",
    "/var/lib/muncho-capability-canary/.hermes/auth.json",
    "/run/credentials/muncho-discord-egress.service",
    "/run/credentials/muncho-discord-connector.service",
    "/run/credentials/muncho-operational-edge-bitrix.service",
    "/run/credentials/muncho-mac-ops-edge.service",
    "/run/credentials/hermes-cloud-gateway.service",
)
GROUPADD = Path("/usr/sbin/groupadd")
USERADD = Path("/usr/sbin/useradd")
SYSTEMCTL = Path("/usr/bin/systemctl")
SYSTEMD_TMPFILES = Path("/usr/bin/systemd-tmpfiles")

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_IDENTITY_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,30}$")
_PRODUCER_IDENTITY_LOCK = threading.Lock()


def _production_diff_observation_path(run_root: Path) -> Path:
    return run_root / "production-diff.json"


def _gateway_observer_source_projection_path(run_root: Path) -> Path:
    return run_root / GATEWAY_OBSERVER_SOURCE_PROJECTION_FILENAME


@dataclass(frozen=True)
class ProducerRoleIdentity:
    role: str
    user: str
    group: str
    uid: int
    gid: int
    receipt_writer_gid: int
    bitrix_socket_gid: int | None = None

    @classmethod
    def from_mapping(cls, role: str, value: Any) -> "ProducerRoleIdentity":
        fields = {
            "user",
            "group",
            "uid",
            "gid",
            "receipt_writer_gid",
            "bitrix_socket_gid",
        }
        if role not in ENDPOINT_ROLES or not isinstance(value, Mapping):
            _fail("producer_role_identity_invalid")
        raw = dict(value)
        if set(raw) != fields:
            _fail("producer_role_identity_invalid")
        if (
            not isinstance(raw["user"], str)
            or _IDENTITY_RE.fullmatch(raw["user"]) is None
            or not isinstance(raw["group"], str)
            or _IDENTITY_RE.fullmatch(raw["group"]) is None
            or type(raw["uid"]) is not int
            or type(raw["gid"]) is not int
            or raw["uid"] <= 0
            or raw["gid"] <= 0
            or type(raw["receipt_writer_gid"]) is not int
            or raw["receipt_writer_gid"] <= 0
            or raw["receipt_writer_gid"] == raw["gid"]
        ):
            _fail("producer_role_identity_invalid")
        bitrix_gid = raw["bitrix_socket_gid"]
        if role in {"business_edge", "canonical_writer"}:
            if type(bitrix_gid) is not int or bitrix_gid <= 0:
                _fail("producer_role_identity_invalid")
        elif bitrix_gid is not None:
            _fail("producer_role_identity_invalid")
        return cls(
            role=role,
            user=raw["user"],
            group=raw["group"],
            uid=raw["uid"],
            gid=raw["gid"],
            receipt_writer_gid=raw["receipt_writer_gid"],
            bitrix_socket_gid=bitrix_gid,
        )


@dataclass(frozen=True)
class ProducerUnitIdentityContract:
    revision: str
    release_root: Path
    units: Mapping[str, bytes]
    service_identity_sha256s: Mapping[str, str]
    role_identities: Mapping[str, ProducerRoleIdentity]


@dataclass(frozen=True)
class ProducerKeyBootstrap:
    value: Mapping[str, Any]
    public_contracts: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class ProducerUnitBundle:
    revision: str
    units: Mapping[str, bytes]
    configs: Mapping[str, bytes]
    auxiliary_files: Mapping[str, bytes]
    manifest: Mapping[str, Any]


def producer_config_path(role: str) -> Path:
    if role not in ENDPOINT_ROLES:
        _fail("producer_role_invalid")
    return DEFAULT_CONFIG_ROOT / f"{role}.json"


def producer_private_key_source_path(role: str) -> Path:
    if role not in ENDPOINT_ROLES:
        _fail("producer_role_invalid")
    return DEFAULT_KEY_ROOT / f"{role}-private.pem"


def producer_private_key_projection_path(role: str) -> Path:
    unit = PRODUCER_SERVICE_UNITS.get(role)
    if unit is None:
        _fail("producer_role_invalid")
    return Path("/run/credentials") / unit / "producer-private-key"


def producer_public_key_path(role: str) -> Path:
    unit = PRODUCER_SERVICE_UNITS.get(role)
    if unit is None:
        _fail("producer_role_invalid")
    return Path("/run/credentials") / unit / "producer-public-key"


def producer_public_key_source_path(role: str) -> Path:
    if role not in ENDPOINT_ROLES:
        _fail("producer_role_invalid")
    return DEFAULT_KEY_ROOT / f"{role}-public.pem"


def producer_probe_catalog_projection_path(role: str) -> Path:
    if role not in {"canonical_writer", "discord_edge"}:
        _fail("producer_role_invalid")
    return Path("/run/credentials") / PRODUCER_SERVICE_UNITS[role] / "probe-catalog"


def producer_discord_edge_public_key_projection_path(role: str) -> Path:
    if role != "discord_edge":
        _fail("producer_role_invalid")
    return (
        Path("/run/credentials")
        / PRODUCER_SERVICE_UNITS[role]
        / "discord-edge-receipt-public-key"
    )


def producer_socket_path(role: str) -> Path:
    if role not in ENDPOINT_ROLES:
        _fail("producer_role_invalid")
    return DEFAULT_RUNTIME_ROOT / role / "producer.sock"


def _unit_bytes(lines: Sequence[str]) -> bytes:
    if any("\x00" in item or "\r" in item or "\n" in item for item in lines):
        _fail("producer_unit_invalid")
    return ("\n".join(lines) + "\n").encode("ascii", errors="strict")


def _release(revision: str, release_root: Path | None) -> tuple[Path, Path]:
    if _GIT_SHA_RE.fullmatch(revision or "") is None:
        _fail("producer_release_invalid")
    expected = Path("/opt/muncho-canary-releases") / revision
    release = expected if release_root is None else release_root
    if release != expected:
        _fail("producer_release_invalid")
    return release, release / "venv/bin/python"


def _role_identities(
    value: Mapping[str, Any],
) -> Mapping[str, ProducerRoleIdentity]:
    if not isinstance(value, Mapping) or set(value) != set(ENDPOINT_ROLES):
        _fail("producer_role_identity_invalid")
    identities = {
        role: ProducerRoleIdentity.from_mapping(role, value[role])
        for role in ENDPOINT_ROLES
    }
    endpoint_uids = [item.uid for item in identities.values()]
    endpoint_gids = [item.gid for item in identities.values()]
    if len(endpoint_uids) != len(set(endpoint_uids)) or len(endpoint_gids) != len(
        set(endpoint_gids)
    ):
        _fail("producer_role_identity_not_separated")
    bitrix_gids = {
        item.bitrix_socket_gid
        for item in identities.values()
        if item.bitrix_socket_gid is not None
    }
    if len(bitrix_gids) != 1:
        _fail("producer_bitrix_group_binding_invalid")
    receipt_gids = {item.receipt_writer_gid for item in identities.values()}
    if len(receipt_gids) != 1 or receipt_gids & bitrix_gids:
        _fail("producer_receipt_group_binding_invalid")
    return identities


def _run_account_command(argv: Sequence[str]) -> None:
    if not argv or Path(argv[0]) not in {GROUPADD, USERADD}:
        _fail("producer_identity_command_invalid")
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"},
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CapabilityProducerError("producer_identity_command_failed") from exc
    if completed.returncode != 0:
        _fail("producer_identity_command_failed")


def planned_producer_role_identities() -> Mapping[str, Mapping[str, Any]]:
    """Return the code-pinned producer principals; never learn IDs from NSS."""

    return {
        role: {
            "user": PRODUCER_ROLE_ACCOUNTS[role][0],
            "group": PRODUCER_ROLE_ACCOUNTS[role][1],
            "uid": PRODUCER_ROLE_NUMERIC_IDENTITIES[role][0],
            "gid": PRODUCER_ROLE_NUMERIC_IDENTITIES[role][1],
            "receipt_writer_gid": PRODUCER_RECEIPT_WRITER_GID,
            "bitrix_socket_gid": (
                PRODUCER_BITRIX_SOCKET_GID
                if role in {"business_edge", "canonical_writer"}
                else None
            ),
        }
        for role in ENDPOINT_ROLES
    }


def _optional_group_by_name(name: str) -> grp.struct_group | None:
    try:
        return grp.getgrnam(name)
    except KeyError:
        return None


def _optional_group_by_gid(gid: int) -> grp.struct_group | None:
    try:
        return grp.getgrgid(gid)
    except KeyError:
        return None


def _optional_user_by_name(name: str) -> pwd.struct_passwd | None:
    try:
        return pwd.getpwnam(name)
    except KeyError:
        return None


def _optional_user_by_uid(uid: int) -> pwd.struct_passwd | None:
    try:
        return pwd.getpwuid(uid)
    except KeyError:
        return None


def _passwd_slot_inventory(
    user_name: str,
    uid: int,
    gid: int,
) -> tuple[list[tuple[str, int, int, str, str]], list[str]]:
    """Return raw NSS account collisions and primary-GID users.

    ``grp.gr_mem`` contains supplementary members only on normal Linux NSS
    backends.  Checking it alone would miss an unrelated account whose
    ``pw_gid`` already points at a producer GID.  Lists deliberately preserve
    duplicate NSS rows; collapsing by name could hide an alias or duplicate
    account with different numeric authority.
    """

    try:
        entries = pwd.getpwall()
    except (KeyError, OSError) as exc:
        raise CapabilityProducerError(
            "producer_identity_inventory_unavailable"
        ) from exc
    slot_rows = sorted(
        (
            entry.pw_name,
            entry.pw_uid,
            entry.pw_gid,
            entry.pw_dir,
            entry.pw_shell,
        )
        for entry in entries
        if entry.pw_name == user_name or entry.pw_uid == uid
    )
    primary_names = sorted(entry.pw_name for entry in entries if entry.pw_gid == gid)
    if any(not isinstance(name, str) or not name for name in primary_names):
        _fail("producer_identity_inventory_invalid")
    return slot_rows, primary_names


def _primary_group_user_names(gid: int) -> list[str]:
    """Return raw, non-deduplicated NSS primary members for a group-only slot."""

    try:
        entries = pwd.getpwall()
    except (KeyError, OSError) as exc:
        raise CapabilityProducerError(
            "producer_identity_inventory_unavailable"
        ) from exc
    names = sorted(entry.pw_name for entry in entries if entry.pw_gid == gid)
    if any(not isinstance(name, str) or not name for name in names):
        _fail("producer_identity_inventory_invalid")
    return names


def _group_slot_inventory(
    group_name: str,
    gid: int,
) -> list[tuple[str, int, tuple[str, ...]]]:
    """Return every NSS group row colliding by fixed name or numeric GID."""

    try:
        entries = grp.getgrall()
    except (KeyError, OSError) as exc:
        raise CapabilityProducerError(
            "producer_identity_inventory_unavailable"
        ) from exc
    return sorted(
        (
            entry.gr_name,
            entry.gr_gid,
            tuple(entry.gr_mem),
        )
        for entry in entries
        if entry.gr_name == group_name or entry.gr_gid == gid
    )


def _producer_role_host_observation(
    role: str,
    *,
    allow_create_only_absence: bool,
) -> Mapping[str, Any]:
    user_name, group_name = PRODUCER_ROLE_ACCOUNTS[role]
    uid, gid = PRODUCER_ROLE_NUMERIC_IDENTITIES[role]
    user = _optional_user_by_name(user_name)
    uid_owner = _optional_user_by_uid(uid)
    group = _optional_group_by_name(group_name)
    gid_owner = _optional_group_by_gid(gid)
    passwd_slot_rows, primary_group_users = _passwd_slot_inventory(user_name, uid, gid)
    group_slot_rows = _group_slot_inventory(group_name, gid)
    expected_passwd_row = (
        user_name,
        uid,
        gid,
        "/nonexistent",
        "/usr/sbin/nologin",
    )
    expected_group_row = (group_name, gid, ())
    all_absent = all(item is None for item in (user, uid_owner, group, gid_owner)) and (
        passwd_slot_rows == [] and primary_group_users == [] and group_slot_rows == []
    )
    all_present_exact = (
        user is not None
        and uid_owner is not None
        and group is not None
        and gid_owner is not None
        and user.pw_name == user_name
        and user.pw_uid == uid
        and user.pw_gid == gid
        and user.pw_dir == "/nonexistent"
        and user.pw_shell == "/usr/sbin/nologin"
        and uid_owner.pw_name == user_name
        and group.gr_name == group_name
        and group.gr_gid == gid
        and gid_owner.gr_name == group_name
        and list(group.gr_mem) == []
        and passwd_slot_rows == [expected_passwd_row]
        and primary_group_users == [user_name]
        and group_slot_rows == [expected_group_row]
        and sorted(set(os.getgrouplist(user_name, gid))) == [gid]
    )
    group_present_user_absent = (
        user is None
        and uid_owner is None
        and group is not None
        and gid_owner is not None
        and group.gr_name == group_name
        and group.gr_gid == gid
        and gid_owner.gr_name == group_name
        and list(group.gr_mem) == []
        and passwd_slot_rows == []
        and primary_group_users == []
        and group_slot_rows == [expected_group_row]
    )
    if all_absent:
        state = "absent_create_only_slot"
        group_members: list[str] | None = None
        supplementary_group_ids: list[int] | None = None
    elif group_present_user_absent:
        # ``groupadd`` and ``useradd`` cannot be one kernel transaction.  This
        # exact empty-group state is the sole safe crash-recovery seam: the
        # pinned user name and UID are both still free, so a retry may execute
        # only the originally reviewed ``useradd`` command.  Every other
        # partial/colliding NSS shape remains fail-closed.
        state = "group_present_user_absent_create_only_slot"
        group_members = []
        supplementary_group_ids = None
    elif all_present_exact:
        state = "present_exact"
        group_members = []
        supplementary_group_ids = [gid]
    else:
        _fail("producer_identity_slot_collision_or_drift")
    if state != "present_exact" and not allow_create_only_absence:
        _fail("producer_identity_unavailable")
    return {
        "state": state,
        "user": user_name,
        "group": group_name,
        "uid": uid,
        "gid": gid,
        "home": "/nonexistent",
        "shell": "/usr/sbin/nologin",
        "group_members": group_members,
        "supplementary_group_ids": supplementary_group_ids,
        "create_only_eligible": True,
    }


def _producer_receipt_group_observation(
    *, allow_create_only_absence: bool
) -> Mapping[str, Any]:
    group = _optional_group_by_name(PRODUCER_RECEIPT_WRITER_GROUP)
    gid_owner = _optional_group_by_gid(PRODUCER_RECEIPT_WRITER_GID)
    primary_group_users = _primary_group_user_names(PRODUCER_RECEIPT_WRITER_GID)
    group_slot_rows = _group_slot_inventory(
        PRODUCER_RECEIPT_WRITER_GROUP,
        PRODUCER_RECEIPT_WRITER_GID,
    )
    expected_group_row = (
        PRODUCER_RECEIPT_WRITER_GROUP,
        PRODUCER_RECEIPT_WRITER_GID,
        (),
    )
    if (
        group is None
        and gid_owner is None
        and primary_group_users == []
        and group_slot_rows == []
    ):
        state = "absent_create_only_slot"
        members: list[str] | None = None
    elif (
        group is not None
        and gid_owner is not None
        and group.gr_name == PRODUCER_RECEIPT_WRITER_GROUP
        and group.gr_gid == PRODUCER_RECEIPT_WRITER_GID
        and gid_owner.gr_name == PRODUCER_RECEIPT_WRITER_GROUP
        and list(group.gr_mem) == []
        and primary_group_users == []
        and group_slot_rows == [expected_group_row]
    ):
        state = "present_exact"
        members = []
    else:
        _fail("producer_receipt_group_slot_collision_or_drift")
    if state != "present_exact" and not allow_create_only_absence:
        _fail("producer_receipt_group_unavailable")
    return {
        "state": state,
        "group": PRODUCER_RECEIPT_WRITER_GROUP,
        "gid": PRODUCER_RECEIPT_WRITER_GID,
        "members": members,
        "create_only_eligible": True,
    }


def producer_host_identity_receipt(
    plan_sha256: str,
    *,
    allow_create_only_absence: bool,
) -> Mapping[str, Any]:
    """Observe all fixed producer name and numeric slots without mutation."""

    _digest(plan_sha256, "producer_host_identity_invalid")
    unsigned = {
        "schema": PRODUCER_HOST_IDENTITY_SCHEMA,
        "plan_sha256": plan_sha256,
        "roles": {
            role: _producer_role_host_observation(
                role,
                allow_create_only_absence=allow_create_only_absence,
            )
            for role in ENDPOINT_ROLES
        },
        "receipt_writer_group": _producer_receipt_group_observation(
            allow_create_only_absence=allow_create_only_absence
        ),
        "planned_identities": planned_producer_role_identities(),
        "persistent_supplementary_memberships": False,
        "service_time_supplementary_groups_only": True,
        "create_only_eligible": True,
        "secret_material_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def _validate_producer_host_identity_receipt(
    value: Any,
    *,
    plan_sha256: str,
    require_present: bool,
) -> Mapping[str, Any]:
    code = "producer_host_identity_invalid"
    raw = _strict(
        value,
        (
            "schema",
            "plan_sha256",
            "roles",
            "receipt_writer_group",
            "planned_identities",
            "persistent_supplementary_memberships",
            "service_time_supplementary_groups_only",
            "create_only_eligible",
            "secret_material_recorded",
            "receipt_sha256",
        ),
        code,
    )
    roles = _strict(raw["roles"], ENDPOINT_ROLES, code)
    expected_identities = planned_producer_role_identities()
    role_fields = (
        "state",
        "user",
        "group",
        "uid",
        "gid",
        "home",
        "shell",
        "group_members",
        "supplementary_group_ids",
        "create_only_eligible",
    )
    for role in ENDPOINT_ROLES:
        item = _strict(roles[role], role_fields, code)
        expected = expected_identities[role]
        state = item["state"]
        if (
            state
            not in {
                "absent_create_only_slot",
                "group_present_user_absent_create_only_slot",
                "present_exact",
            }
            or (require_present and state != "present_exact")
            or item["user"] != expected["user"]
            or item["group"] != expected["group"]
            or item["uid"] != expected["uid"]
            or item["gid"] != expected["gid"]
            or item["home"] != "/nonexistent"
            or item["shell"] != "/usr/sbin/nologin"
            or item["create_only_eligible"] is not True
            or (
                state == "absent_create_only_slot"
                and (
                    item["group_members"] is not None
                    or item["supplementary_group_ids"] is not None
                )
            )
            or (
                state == "group_present_user_absent_create_only_slot"
                and (
                    item["group_members"] != []
                    or item["supplementary_group_ids"] is not None
                )
            )
            or (
                state == "present_exact"
                and (
                    item["group_members"] != []
                    or item["supplementary_group_ids"] != [item["gid"]]
                )
            )
        ):
            _fail(code)
    receipt_group = _strict(
        raw["receipt_writer_group"],
        ("state", "group", "gid", "members", "create_only_eligible"),
        code,
    )
    receipt_state = receipt_group["state"]
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    if (
        raw["schema"] != PRODUCER_HOST_IDENTITY_SCHEMA
        or raw["plan_sha256"] != plan_sha256
        or raw["planned_identities"] != expected_identities
        or raw["persistent_supplementary_memberships"] is not False
        or raw["service_time_supplementary_groups_only"] is not True
        or raw["create_only_eligible"] is not True
        or raw["secret_material_recorded"] is not False
        or receipt_state not in {"absent_create_only_slot", "present_exact"}
        or (require_present and receipt_state != "present_exact")
        or receipt_group["group"] != PRODUCER_RECEIPT_WRITER_GROUP
        or receipt_group["gid"] != PRODUCER_RECEIPT_WRITER_GID
        or receipt_group["create_only_eligible"] is not True
        or (
            receipt_state == "absent_create_only_slot"
            and receipt_group["members"] is not None
        )
        or (receipt_state == "present_exact" and receipt_group["members"] != [])
        or raw["receipt_sha256"] != _sha256_json(unsigned)
    ):
        _fail(code)
    return copy.deepcopy(raw)


def attest_producer_role_identities(
    expected: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    """Read-only proof of fixed principals with no persistent extra authority."""

    identities = _role_identities(expected)
    planned = planned_producer_role_identities()
    if {
        role: {
            "user": item.user,
            "group": item.group,
            "uid": item.uid,
            "gid": item.gid,
            "receipt_writer_gid": item.receipt_writer_gid,
            "bitrix_socket_gid": item.bitrix_socket_gid,
        }
        for role, item in identities.items()
    } != planned:
        _fail("producer_role_identity_not_fixed")
    observation = _validate_producer_host_identity_receipt(
        producer_host_identity_receipt(
            "0" * 64,
            allow_create_only_absence=False,
        ),
        plan_sha256="0" * 64,
        require_present=True,
    )
    return copy.deepcopy(observation["planned_identities"])


def attest_foundation_service_identities(
    plan: Any,
) -> Mapping[str, Mapping[str, Any]]:
    """Read-only proof that the two plan services and connector ACL are exact."""

    try:
        expected = {
            "mac_ops": (
                plan.identities.mac_ops_user,
                plan.identities.mac_ops_group,
                plan.identities.mac_ops_uid,
                plan.identities.mac_ops_gid,
                (),
            ),
            "connector": (
                plan.identities.connector_user,
                plan.identities.connector_group,
                plan.identities.connector_uid,
                plan.identities.connector_gid,
                (),
            ),
        }
    except AttributeError as exc:
        raise CapabilityProducerError("producer_plan_invalid") from exc
    observed: dict[str, Mapping[str, Any]] = {}
    for role, (user_name, group_name, uid, gid, members) in expected.items():
        try:
            user = pwd.getpwnam(user_name)
            group = grp.getgrnam(group_name)
            uid_owner = pwd.getpwuid(uid)
            gid_owner = grp.getgrgid(gid)
        except KeyError as exc:
            raise CapabilityProducerError(
                "producer_service_identity_unavailable"
            ) from exc
        passwd_slot_rows, primary_group_users = _passwd_slot_inventory(
            user_name,
            uid,
            gid,
        )
        group_slot_rows = _group_slot_inventory(group_name, gid)
        supplementary = sorted(set(os.getgrouplist(user_name, gid)))
        if (
            user.pw_uid != uid
            or user.pw_gid != gid
            or user.pw_dir != "/nonexistent"
            or user.pw_shell != "/usr/sbin/nologin"
            or uid_owner.pw_name != user_name
            or group.gr_gid != gid
            or gid_owner.gr_name != group_name
            or sorted(group.gr_mem) != sorted(members)
            or passwd_slot_rows
            != [(user_name, uid, gid, "/nonexistent", "/usr/sbin/nologin")]
            or primary_group_users != [user_name]
            or group_slot_rows != [(group_name, gid, tuple(sorted(members)))]
            or supplementary != [gid]
        ):
            _fail("producer_service_identity_drifted")
        observed[role] = {
            "user": user_name,
            "group": group_name,
            "uid": uid,
            "gid": gid,
            "group_members": sorted(members),
            "supplementary_group_ids": supplementary,
        }
    return copy.deepcopy(observed)


def _producer_identity_foundation_path(plan: Any) -> Path:
    revision = getattr(plan, "revision", None)
    plan_sha256 = getattr(plan, "sha256", None)
    if (
        not isinstance(revision, str)
        or _GIT_SHA_RE.fullmatch(revision) is None
        or not isinstance(plan_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None
    ):
        _fail("producer_plan_invalid")
    return (
        DEFAULT_PRODUCER_IDENTITY_FOUNDATION_ROOT
        / revision
        / plan_sha256
        / "foundation.json"
    )


def _validate_producer_identity_foundation_receipt(
    value: Any,
    *,
    plan: Any,
    full_plan: Any,
    plan_publication_receipt_sha256: str,
    receipt_path: Path,
) -> Mapping[str, Any]:
    code = "producer_identity_foundation_invalid"
    raw = _strict(
        value,
        (
            "schema",
            "operation",
            "revision",
            "capability_plan_sha256",
            "full_canary_plan_sha256",
            "full_canary_terminal_receipt_sha256",
            "original_full_canary_owner_approval_sha256",
            "plan_publication_receipt_sha256",
            "receipt_path",
            "planned_identities",
            "before",
            "after",
            "created",
            "create_only",
            "existing_identities_mutated",
            "retained_dormant_on_rollback",
            "mutation_performed",
            "persistent_supplementary_memberships",
            "service_time_supplementary_groups_only",
            "secret_material_recorded",
            "secret_digest_recorded",
            "receipt_sha256",
        ),
        code,
    )
    before = _validate_producer_host_identity_receipt(
        raw["before"],
        plan_sha256=plan.sha256,
        require_present=False,
    )
    _validate_producer_host_identity_receipt(
        raw["after"],
        plan_sha256=plan.sha256,
        require_present=True,
    )
    expected_created: list[str] = []
    for role in ENDPOINT_ROLES:
        state = before["roles"][role]["state"]
        if state == "absent_create_only_slot":
            expected_created.append(f"{role}_group")
        if state != "present_exact":
            expected_created.append(f"{role}_user")
    if before["receipt_writer_group"]["state"] == "absent_create_only_slot":
        expected_created.append("receipt_writer_group")
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    if (
        raw["schema"] != PRODUCER_IDENTITY_FOUNDATION_SCHEMA
        or raw["operation"] != "create_only_fixed_producer_principals"
        or raw["revision"] != plan.revision
        or raw["capability_plan_sha256"] != plan.sha256
        or raw["full_canary_plan_sha256"] != full_plan.sha256
        or raw["full_canary_terminal_receipt_sha256"]
        != plan.full_canary_terminal_receipt_sha256
        or raw["original_full_canary_owner_approval_sha256"]
        != plan.original_full_canary_owner_approval_sha256
        or raw["plan_publication_receipt_sha256"] != plan_publication_receipt_sha256
        or raw["receipt_path"] != str(receipt_path)
        or raw["planned_identities"] != planned_producer_role_identities()
        or raw["created"] != expected_created
        or raw["create_only"] is not True
        or raw["existing_identities_mutated"] is not False
        or raw["retained_dormant_on_rollback"] is not True
        or raw["mutation_performed"] is not bool(expected_created)
        or raw["persistent_supplementary_memberships"] is not False
        or raw["service_time_supplementary_groups_only"] is not True
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["receipt_sha256"] != _sha256_json(unsigned)
    ):
        _fail(code)
    for field in (
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "full_canary_terminal_receipt_sha256",
        "original_full_canary_owner_approval_sha256",
        "plan_publication_receipt_sha256",
        "receipt_sha256",
    ):
        _digest(raw[field], code)
    return copy.deepcopy(raw)


def load_producer_identity_foundation_receipt(
    *,
    plan: Any,
    full_plan: Any,
    plan_publication_receipt: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if plan_publication_receipt is None:
        from gateway.canonical_capability_canary_runtime import (
            load_bound_plan_publication_receipt,
        )

        publication = load_bound_plan_publication_receipt(plan)
    else:
        publication = copy.deepcopy(dict(plan_publication_receipt))
    publication_sha256 = _digest(
        publication.get("receipt_sha256"),
        "producer_identity_foundation_invalid",
    )
    receipt_path = _producer_identity_foundation_path(plan)
    raw, _item = _stable_read(
        receipt_path,
        maximum=2 * 1024 * 1024,
        uid=0,
        gid=0,
        mode=0o400,
    )
    return _validate_producer_identity_foundation_receipt(
        _strict_json(raw, "producer_identity_foundation_invalid"),
        plan=plan,
        full_plan=full_plan,
        plan_publication_receipt_sha256=publication_sha256,
        receipt_path=receipt_path,
    )


def _publish_producer_identity_foundation(path: Path, payload: bytes) -> None:
    current = DEFAULT_FOUNDATION_CONTROL_ROOT
    _ensure_exact_directory(current, uid=0, gid=0, mode=0o700)
    for part in path.parent.relative_to(current).parts:
        current /= part
        _ensure_exact_directory(current, uid=0, gid=0, mode=0o700)
    _publish_no_replace(
        path,
        payload,
        uid=0,
        gid=0,
        mode=0o400,
        parent_uid=0,
        parent_gid=0,
        parent_mode=0o700,
    )


def ensure_producer_role_identities(
    *,
    plan: Any,
    full_plan: Any,
    account_runner: Callable[[Sequence[str]], None] = _run_account_command,
    observer: Callable[..., Mapping[str, Any]] = producer_host_identity_receipt,
    publisher: Callable[[Path, bytes], None] = _publish_producer_identity_foundation,
    plan_publication_receipt: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Create only exact fixed producer principals and publish their receipt."""

    if (
        sys.platform != "linux" or os.geteuid() != 0
    ):  # windows-footgun: ok — Linux production/canary boundary
        _fail("producer_root_linux_required")
    if plan_publication_receipt is None:
        from gateway.canonical_capability_canary_runtime import (
            load_bound_plan_publication_receipt,
        )

        publication = load_bound_plan_publication_receipt(plan)
    else:
        publication = copy.deepcopy(dict(plan_publication_receipt))
    publication_sha256 = _digest(
        publication.get("receipt_sha256"),
        "producer_identity_foundation_invalid",
    )
    receipt_path = _producer_identity_foundation_path(plan)
    with _PRODUCER_IDENTITY_LOCK:
        if os.path.lexists(receipt_path):
            return load_producer_identity_foundation_receipt(
                plan=plan,
                full_plan=full_plan,
                plan_publication_receipt=publication,
            )
        before = _validate_producer_host_identity_receipt(
            observer(plan.sha256, allow_create_only_absence=True),
            plan_sha256=plan.sha256,
            require_present=False,
        )
        created: list[str] = []
        for role in ENDPOINT_ROLES:
            state = before["roles"][role]["state"]
            user, group = PRODUCER_ROLE_ACCOUNTS[role]
            uid, gid = PRODUCER_ROLE_NUMERIC_IDENTITIES[role]
            if state == "absent_create_only_slot":
                account_runner((
                    str(GROUPADD),
                    "--system",
                    "--gid",
                    str(gid),
                    "--",
                    group,
                ))
                created.append(f"{role}_group")
            if state != "present_exact":
                account_runner((
                    str(USERADD),
                    "--system",
                    "--uid",
                    str(uid),
                    "--gid",
                    group,
                    "--home-dir",
                    "/nonexistent",
                    "--no-create-home",
                    "--shell",
                    "/usr/sbin/nologin",
                    "--",
                    user,
                ))
                created.append(f"{role}_user")
        if before["receipt_writer_group"]["state"] == "absent_create_only_slot":
            account_runner((
                str(GROUPADD),
                "--system",
                "--gid",
                str(PRODUCER_RECEIPT_WRITER_GID),
                "--",
                PRODUCER_RECEIPT_WRITER_GROUP,
            ))
            created.append("receipt_writer_group")
        after = _validate_producer_host_identity_receipt(
            observer(plan.sha256, allow_create_only_absence=False),
            plan_sha256=plan.sha256,
            require_present=True,
        )
        unsigned = {
            "schema": PRODUCER_IDENTITY_FOUNDATION_SCHEMA,
            "operation": "create_only_fixed_producer_principals",
            "revision": plan.revision,
            "capability_plan_sha256": plan.sha256,
            "full_canary_plan_sha256": full_plan.sha256,
            "full_canary_terminal_receipt_sha256": (
                plan.full_canary_terminal_receipt_sha256
            ),
            "original_full_canary_owner_approval_sha256": (
                plan.original_full_canary_owner_approval_sha256
            ),
            "plan_publication_receipt_sha256": publication_sha256,
            "receipt_path": str(receipt_path),
            "planned_identities": planned_producer_role_identities(),
            "before": before,
            "after": after,
            "created": created,
            "create_only": True,
            "existing_identities_mutated": False,
            "retained_dormant_on_rollback": True,
            "mutation_performed": bool(created),
            "persistent_supplementary_memberships": False,
            "service_time_supplementary_groups_only": True,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        receipt = _validate_producer_identity_foundation_receipt(
            {**unsigned, "receipt_sha256": _sha256_json(unsigned)},
            plan=plan,
            full_plan=full_plan,
            plan_publication_receipt_sha256=publication_sha256,
            receipt_path=receipt_path,
        )
        publisher(receipt_path, _canonical_bytes(receipt))
        return copy.deepcopy(receipt)


def _ensure_exact_directory(path: Path, *, uid: int, gid: int, mode: int) -> None:
    if not path.is_absolute() or ".." in path.parts:
        _fail("producer_directory_invalid")
    try:
        item = path.lstat()
    except FileNotFoundError:
        try:
            path.mkdir(mode=mode)
            os.chown(path, uid, gid)
            os.chmod(path, mode)
            item = path.lstat()
        except OSError as exc:
            raise CapabilityProducerError("producer_directory_unavailable") from exc
    except OSError as exc:
        raise CapabilityProducerError("producer_directory_unavailable") from exc
    if (
        not stat.S_ISDIR(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != uid
        or item.st_gid != gid
        or stat.S_IMODE(item.st_mode) != mode
    ):
        _fail("producer_directory_identity_invalid")


def _require_exact_directory(path: Path, *, uid: int, gid: int, mode: int) -> None:
    try:
        item = path.lstat()
    except OSError as exc:
        raise CapabilityProducerError("producer_directory_unavailable") from exc
    if (
        not path.is_absolute()
        or ".." in path.parts
        or not stat.S_ISDIR(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != uid
        or item.st_gid != gid
        or stat.S_IMODE(item.st_mode) != mode
    ):
        _fail("producer_directory_identity_invalid")


def _render_unit(
    *,
    revision: str,
    release: Path,
    interpreter: Path,
    identity: ProducerRoleIdentity,
) -> bytes:
    role = identity.role
    unit = PRODUCER_SERVICE_UNITS[role]
    config = producer_config_path(role)
    private_source = producer_private_key_source_path(role)
    private_projection = producer_private_key_projection_path(role)
    public_source = producer_public_key_source_path(role)
    public_projection = producer_public_key_path(role)
    foundation_projection = Path("/run/credentials") / unit / "producer-foundation"
    owner_hex_projection = Path("/run/credentials") / unit / "owner-public-key-hex"
    owner_source_projection = (
        Path("/run/credentials") / unit / "owner-public-key-source-sha256"
    )
    supplementary_groups = [PRODUCER_RECEIPT_WRITER_GROUP]
    if role in {"business_edge", "canonical_writer"}:
        supplementary_groups.append(BITRIX_OPERATIONAL_EDGE_SOCKET_GROUP)
    if role == "canonical_writer":
        supplementary_groups.extend([
            CANONICAL_WRITER_SERVICE_GROUP,
            CANONICAL_WRITER_SOCKET_GROUP,
        ])
    if role == "discord_edge":
        supplementary_groups.extend([DISCORD_ROUTEBACK_GROUP, DISCORD_CONNECTOR_GROUP])
    supplementary = [f"SupplementaryGroups={' '.join(supplementary_groups)}"]
    ordering = (
        [
            f"After={BITRIX_OPERATIONAL_EDGE_SERVICE_UNIT}",
            f"Requires={BITRIX_OPERATIONAL_EDGE_SERVICE_UNIT}",
        ]
        if role in {"business_edge", "canonical_writer"}
        else []
    )
    return _unit_bytes([
        "# Role-local signer for the production-shaped capability canary.",
        f"# ReleaseRevision={revision}",
        f"# AuthorityRole={role}",
        "[Unit]",
        f"Description=Muncho capability canary producer ({role})",
        *ordering,
        "Before=hermes-cloud-gateway.service",
        "StartLimitIntervalSec=300s",
        "StartLimitBurst=5",
        f"AssertPathExists={interpreter}",
        f"AssertPathExists={release / '.codex-source-commit'}",
        f"AssertPathExists={config}",
        f"AssertPathExists={private_source}",
        f"AssertPathExists={public_source}",
        f"AssertPathExists={DEFAULT_FOUNDATION_PATH}",
        f"AssertPathExists={DEFAULT_OWNER_PUBLIC_KEY_HEX_PIN_PATH}",
        (f"AssertPathExists={DEFAULT_OWNER_PUBLIC_KEY_SOURCE_SHA256_PIN_PATH}"),
        *(
            [f"AssertPathExists={DEFAULT_PROBE_CATALOG_PATH}"]
            if role in {"canonical_writer", "discord_edge"}
            else []
        ),
        *(
            [f"AssertPathExists={DISCORD_EDGE_RECEIPT_PUBLIC_KEY_PATH}"]
            if role == "discord_edge"
            else []
        ),
        "",
        "[Service]",
        "Type=simple",
        f"User={identity.user}",
        f"Group={identity.group}",
        *supplementary,
        f"LoadCredential=producer-private-key:{private_source}",
        f"LoadCredential=producer-public-key:{public_source}",
        f"LoadCredential=producer-foundation:{DEFAULT_FOUNDATION_PATH}",
        (
            "LoadCredential=owner-public-key-hex:"
            f"{DEFAULT_OWNER_PUBLIC_KEY_HEX_PIN_PATH}"
        ),
        (
            "LoadCredential=owner-public-key-source-sha256:"
            f"{DEFAULT_OWNER_PUBLIC_KEY_SOURCE_SHA256_PIN_PATH}"
        ),
        *(
            [f"LoadCredential=probe-catalog:{DEFAULT_PROBE_CATALOG_PATH}"]
            if role in {"canonical_writer", "discord_edge"}
            else []
        ),
        *(
            [
                "LoadCredential=discord-edge-receipt-public-key:"
                f"{DISCORD_EDGE_RECEIPT_PUBLIC_KEY_PATH}"
            ]
            if role == "discord_edge"
            else []
        ),
        f"RuntimeDirectory=muncho-capability-canary-producers/{role}",
        "RuntimeDirectoryMode=0700",
        f"WorkingDirectory={release}",
        (
            f"ExecStart={interpreter} -I -B -m "
            "gateway.canonical_capability_canary_producers serve "
            f"--config {config} --foundation {foundation_projection} "
            f"--owner-public-key-hex-file {owner_hex_projection} "
            "--owner-public-key-source-sha256-file "
            f"{owner_source_projection}"
        ),
        "Restart=no",
        "RuntimeMaxSec=900s",
        "TimeoutStartSec=30s",
        "TimeoutStopSec=30s",
        "KillMode=control-group",
        "UMask=0077",
        "LimitCORE=0",
        "LimitNOFILE=1024",
        "Environment=HOME=/run/muncho-capability-canary-producers",
        "Environment=LANG=C.UTF-8",
        "Environment=LC_ALL=C.UTF-8",
        "Environment=PATH=/usr/bin:/bin",
        "Environment=PYTHONDONTWRITEBYTECODE=1",
        "Environment=PYTHONNOUSERSITE=1",
        "Environment=TZ=UTC",
        *(
            [
                "Environment=GATEWAY_RELAY_URL="
                "unix:///run/muncho-discord-connector/connector.sock"
            ]
            if role == "discord_edge"
            else []
        ),
        "UnsetEnvironment=PYTHONPATH PYTHONHOME BASH_ENV ENV CDPATH",
        "NoNewPrivileges=yes",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
        "LockPersonality=yes",
        "PrivateDevices=yes",
        "PrivateTmp=yes",
        "ProtectClock=yes",
        "ProtectControlGroups=yes",
        "ProtectHome=yes",
        "ProtectHostname=yes",
        "ProtectKernelLogs=yes",
        "ProtectKernelModules=yes",
        "ProtectKernelTunables=yes",
        "ProtectProc=invisible",
        "ProtectSystem=strict",
        "RemoveIPC=yes",
        "RestrictNamespaces=yes",
        "RestrictRealtime=yes",
        "RestrictSUIDSGID=yes",
        "SystemCallArchitectures=native",
        "RestrictAddressFamilies=AF_UNIX",
        "IPAddressDeny=any",
        f"ReadOnlyPaths={release}",
        f"ReadOnlyPaths={DEFAULT_NATIVE_ROOT / role}",
        f"ReadWritePaths={DEFAULT_RECEIPT_ROOT}",
        *(
            f"InaccessiblePaths=-{path}"
            for path in PRODUCER_INACCESSIBLE_CREDENTIAL_PATHS
        ),
        "InaccessiblePaths=-/opt/adventico-ai-platform/hermes-home/secrets",
        "InaccessiblePaths=-/opt/adventico-ai-platform/.hermes/secrets",
        f"InaccessiblePaths={DEFAULT_KEY_ROOT}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ])


def render_producer_unit_identity_contract(
    *,
    revision: str,
    role_identities: Mapping[str, Any],
    release_root: Path | None = None,
) -> ProducerUnitIdentityContract:
    release, interpreter = _release(revision, release_root)
    identities = _role_identities(role_identities)
    units: dict[str, bytes] = {}
    service_ids: dict[str, str] = {}
    for role in ENDPOINT_ROLES:
        unit = _render_unit(
            revision=revision,
            release=release,
            interpreter=interpreter,
            identity=identities[role],
        )
        path = f"/etc/systemd/system/{PRODUCER_SERVICE_UNITS[role]}"
        units[path] = unit
        service_ids[role] = _sha256_json({
            "schema": "muncho-capability-producer-service-identity.v1",
            "role": role,
            "unit": PRODUCER_SERVICE_UNITS[role],
            "unit_sha256": _sha256_bytes(unit),
            "revision": revision,
            "release_root": str(release),
            "uid": identities[role].uid,
            "gid": identities[role].gid,
            "receipt_writer_gid": identities[role].receipt_writer_gid,
            "socket_path": str(producer_socket_path(role)),
            "bitrix_socket_gid": identities[role].bitrix_socket_gid,
        })
    return ProducerUnitIdentityContract(
        revision=revision,
        release_root=release,
        units=units,
        service_identity_sha256s=service_ids,
        role_identities=identities,
    )


def bootstrap_producer_keys(
    *,
    role_identities: Mapping[str, Any],
    key_root: Path = DEFAULT_KEY_ROOT,
    receipt_path: Path = DEFAULT_KEY_BOOTSTRAP_RECEIPT,
    root_uid: int = 0,
    root_gid: int = 0,
) -> ProducerKeyBootstrap:
    """Create each independent role key once; publish only public metadata."""

    identities = _role_identities(role_identities)
    try:
        root_item = key_root.lstat()
    except OSError as exc:
        raise CapabilityProducerError("producer_key_root_invalid") from exc
    if (
        not stat.S_ISDIR(root_item.st_mode)
        or stat.S_ISLNK(root_item.st_mode)
        or root_item.st_uid != root_uid
        or root_item.st_gid != root_gid
        or stat.S_IMODE(root_item.st_mode) != 0o700
    ):
        _fail("producer_key_root_invalid")
    rows: list[dict[str, Any]] = []
    public_contracts: dict[str, Mapping[str, Any]] = {}
    for role in ENDPOINT_ROLES:
        private_path = key_root / f"{role}-private.pem"
        public_path = key_root / f"{role}-public.pem"
        if os.path.lexists(private_path):
            private, _private_digest = _load_private_key(
                private_path,
                uid=root_uid,
                gid=root_gid,
            )
        else:
            private = Ed25519PrivateKey.generate()
            private_raw = private.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            _publish_no_replace(
                private_path,
                private_raw,
                uid=root_uid,
                gid=root_gid,
                mode=0o400,
                parent_uid=root_uid,
                parent_gid=root_gid,
            )
        public = private.public_key()
        public_raw = public.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        _publish_no_replace(
            public_path,
            public_raw,
            uid=root_uid,
            gid=root_gid,
            mode=0o400,
            parent_uid=root_uid,
            parent_gid=root_gid,
        )
        observed_public, public_hex, public_file_sha = _load_public_key(
            public_path,
            uid=root_uid,
            gid=root_gid,
            mode=0o400,
        )
        if observed_public.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ) != public.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ):
            _fail("producer_key_pair_mismatch")
        key_id = _sha256_bytes(bytes.fromhex(public_hex))
        contract = {
            "key_id": key_id,
            "algorithm": "ed25519",
            "public_key_ed25519_hex": public_hex,
            "public_key_source_path": str(public_path),
            "public_key_projection_path": str(producer_public_key_path(role)),
            "public_key_file_sha256": public_file_sha,
        }
        public_contracts[role] = contract
        rows.append({
            "role": role,
            "private_key_source_path": str(private_path),
            "private_key_projection_path": str(
                producer_private_key_projection_path(role)
            ),
            **contract,
            "private_content_or_digest_recorded": False,
        })
    unsigned = {
        "schema": KEY_BOOTSTRAP_SCHEMA,
        "keys": rows,
        "roles": list(ENDPOINT_ROLES),
        "independent_key_ids": len({row["key_id"] for row in rows})
        == len(ENDPOINT_ROLES),
        "private_content_or_digest_recorded": False,
    }
    value = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
    _publish_no_replace(
        receipt_path,
        _canonical_bytes(value),
        uid=root_uid,
        gid=root_gid,
        mode=0o400,
        parent_uid=root_uid,
        parent_gid=root_gid,
    )
    return ProducerKeyBootstrap(
        value=copy.deepcopy(value),
        public_contracts=copy.deepcopy(public_contracts),
    )


def _parent_contract(path: Path) -> tuple[int, int, int]:
    try:
        item = path.lstat()
    except OSError as exc:
        raise CapabilityProducerError("producer_parent_unavailable") from exc
    mode = stat.S_IMODE(item.st_mode)
    if (
        not stat.S_ISDIR(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != 0
        or item.st_gid != 0
        or mode & 0o022
    ):
        _fail("producer_parent_identity_invalid")
    return item.st_uid, item.st_gid, mode


def _publish_root_file(path: Path, payload: bytes, *, mode: int = 0o400) -> None:
    parent_uid, parent_gid, parent_mode = _parent_contract(path.parent)
    _publish_no_replace(
        path,
        payload,
        uid=0,
        gid=0,
        mode=mode,
        parent_uid=parent_uid,
        parent_gid=parent_gid,
        parent_mode=parent_mode,
    )


def _plan_bitrix_contract(plan: Any) -> Mapping[str, Any]:
    try:
        value = plan.to_mapping()["bitrix_operational_edge"]
        revision = plan.revision
    except (AttributeError, KeyError, TypeError) as exc:
        raise CapabilityProducerError("producer_plan_invalid") from exc
    return validate_bitrix_operational_edge_contract(value, release_sha=revision)


def _discord_edge_evidence_contract(
    plan: Any,
    full_plan: Any,
) -> Mapping[str, Any]:
    try:
        edge_uid = full_plan.identities.edge_uid
        edge_gid = full_plan.identities.edge_gid
        writer_gid = full_plan.identities.writer_gid
        connector_uid = plan.identities.connector_uid
        connector_gid = plan.identities.connector_gid
    except AttributeError as exc:
        raise CapabilityProducerError("producer_plan_invalid") from exc
    _public, public_hex, file_sha256 = _load_public_key(
        DISCORD_EDGE_RECEIPT_PUBLIC_KEY_PATH,
        uid=0,
        gid=writer_gid,
        mode=0o440,
    )
    from gateway.canonical_capability_canary_producers import (
        validate_discord_edge_evidence_contract,
    )

    return validate_discord_edge_evidence_contract({
        "edge_service_unit": "muncho-discord-egress.service",
        "edge_socket_path": "/run/muncho-discord-egress/edge.sock",
        "edge_service_uid": edge_uid,
        "edge_service_gid": edge_gid,
        "receipt_public_key_path": str(DISCORD_EDGE_RECEIPT_PUBLIC_KEY_PATH),
        "receipt_public_key_id": _sha256_bytes(bytes.fromhex(public_hex)),
        "receipt_public_key_file_sha256": file_sha256,
        "connector_service_unit": "muncho-discord-connector.service",
        "connector_socket_path": ("/run/muncho-discord-connector/connector.sock"),
        "connector_service_uid": connector_uid,
        "connector_service_gid": connector_gid,
        "public_history_operation": "public.history.fetch",
        "direct_message_allowed": False,
        "token_or_token_digest_recorded": False,
    })


_SERVICE_IDENTITY_FOUNDATION_FIELDS = (
    "schema",
    "operation",
    "revision",
    "capability_plan_sha256",
    "full_canary_plan_sha256",
    "full_canary_terminal_receipt_sha256",
    "original_full_canary_owner_approval_sha256",
    "plan_publication_receipt_sha256",
    "receipt_path",
    "before",
    "after",
    "created",
    "create_only",
    "existing_identities_mutated",
    "retained_dormant_on_rollback",
    "mutation_performed",
    "secret_material_recorded",
    "secret_digest_recorded",
    "receipt_sha256",
)


def _validate_service_identity_foundation_binding(
    value: Any,
    *,
    revision: str,
    capability_plan_sha256: str,
    full_canary_plan_sha256: str,
    terminal_receipt_sha256: str,
    original_owner_approval_sha256: str,
    plan_publication_receipt_sha256: str,
) -> Mapping[str, Any]:
    code = "producer_service_identity_foundation_invalid"
    raw = _strict(value, _SERVICE_IDENTITY_FOUNDATION_FIELDS, code)
    before = _strict(raw["before"], ("mac_ops", "connector"), code)
    after = _strict(raw["after"], ("mac_ops", "connector"), code)
    expected_created: list[str] = []
    identity_fields = (
        "schema",
        "plan_sha256",
        "role",
        "state",
        "user",
        "group",
        "uid",
        "gid",
        "home",
        "shell",
        "group_members",
        "supplementary_group_ids",
        "create_only_eligible",
        "secret_material_recorded",
        "receipt_sha256",
    )
    allowed_before = {
        "present_exact",
        "group_present_user_absent_create_only_slot",
        "absent_create_only_slot",
    }
    for role in ("mac_ops", "connector"):
        before_item = _strict(before[role], identity_fields, code)
        after_item = _strict(after[role], identity_fields, code)
        for item in (before_item, after_item):
            unsigned_item = {
                key: nested for key, nested in item.items() if key != "receipt_sha256"
            }
            if (
                item["schema"] != CAPABILITY_SERVICE_HOST_IDENTITY_SCHEMA
                or item["plan_sha256"] != capability_plan_sha256
                or item["role"] != role
                or item["home"] != "/nonexistent"
                or item["shell"] != "/usr/sbin/nologin"
                or type(item["uid"]) is not int
                or type(item["gid"]) is not int
                or item["uid"] <= 0
                or item["gid"] <= 0
                or item["create_only_eligible"] is not True
                or item["secret_material_recorded"] is not False
                or item["receipt_sha256"] != _sha256_json(unsigned_item)
            ):
                _fail(code)
        if (
            before_item["state"] not in allowed_before
            or after_item["state"] != "present_exact"
            or any(
                before_item[field] != after_item[field]
                for field in ("user", "group", "uid", "gid", "home", "shell")
            )
            or after_item["group_members"] != []
            or after_item["supplementary_group_ids"] != [after_item["gid"]]
        ):
            _fail(code)
        if before_item["state"] == "absent_create_only_slot":
            if (
                before_item["group_members"] is not None
                or before_item["supplementary_group_ids"] is not None
            ):
                _fail(code)
            expected_created.append(f"{role}_group")
        elif before_item["state"] == "group_present_user_absent_create_only_slot":
            if (
                before_item["group_members"] != []
                or before_item["supplementary_group_ids"] is not None
            ):
                _fail(code)
        elif before_item["group_members"] != [] or before_item[
            "supplementary_group_ids"
        ] != [before_item["gid"]]:
            _fail(code)
        if before_item["state"] != "present_exact":
            expected_created.append(f"{role}_user")
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    if (
        raw["schema"] != SERVICE_IDENTITY_FOUNDATION_SCHEMA
        or raw["operation"] != "create_only_service_principals"
        or raw["revision"] != revision
        or raw["capability_plan_sha256"] != capability_plan_sha256
        or raw["full_canary_plan_sha256"] != full_canary_plan_sha256
        or raw["full_canary_terminal_receipt_sha256"] != terminal_receipt_sha256
        or raw["original_full_canary_owner_approval_sha256"]
        != original_owner_approval_sha256
        or raw["plan_publication_receipt_sha256"] != plan_publication_receipt_sha256
        or not isinstance(raw["receipt_path"], str)
        or not Path(raw["receipt_path"]).is_absolute()
        or ".." in Path(raw["receipt_path"]).parts
        or raw["created"] != expected_created
        or raw["create_only"] is not True
        or raw["existing_identities_mutated"] is not False
        or raw["retained_dormant_on_rollback"] is not True
        or raw["mutation_performed"] is not bool(expected_created)
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["receipt_sha256"] != _sha256_json(unsigned)
    ):
        _fail(code)
    for field in (
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "full_canary_terminal_receipt_sha256",
        "original_full_canary_owner_approval_sha256",
        "plan_publication_receipt_sha256",
        "receipt_sha256",
    ):
        _digest(raw[field], code)
    return copy.deepcopy(raw)


def prepare_producer_foundation(
    request_value: Any,
    *,
    plan: Any,
    full_plan: Any,
    role_identities: Mapping[str, Any] | None = None,
    key_root: Path = DEFAULT_KEY_ROOT,
    key_bootstrap_receipt: Path = DEFAULT_KEY_BOOTSTRAP_RECEIPT,
    preparation_path: Path = DEFAULT_FOUNDATION_PREPARATION_PATH,
    owner_public_key_hex_path: Path = DEFAULT_OWNER_PUBLIC_KEY_HEX_PIN_PATH,
    owner_public_key_source_sha256_path: Path = (
        DEFAULT_OWNER_PUBLIC_KEY_SOURCE_SHA256_PIN_PATH
    ),
    service_identity_foundation: Mapping[str, Any] | None = None,
    producer_identity_foundation: Mapping[str, Any] | None = None,
    plan_publication_receipt: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Prepare exact Cloud-generated endpoint keys for external owner signing."""

    request = _strict(
        request_value,
        (
            "schema",
            "revision",
            "capability_plan_sha256",
            "full_canary_plan_sha256",
            "full_canary_terminal_receipt",
            "full_canary_terminal_receipt_sha256",
            "original_full_canary_owner_approval_sha256",
            "owner_public_authority",
            "secret_material_recorded",
            "semantic_content_recorded",
        ),
        "producer_foundation_prepare_invalid",
    )
    terminal = _validate_full_canary_terminal(
        request["full_canary_terminal_receipt"],
        code="producer_foundation_prepare_invalid",
    )
    if (
        request["schema"] != FOUNDATION_PREPARE_REQUEST_SCHEMA
        or request["revision"] != getattr(plan, "revision", None)
        or request["revision"] != getattr(full_plan, "revision", None)
        or request["capability_plan_sha256"] != getattr(plan, "sha256", None)
        or request["full_canary_plan_sha256"] != getattr(full_plan, "sha256", None)
        or request["full_canary_terminal_receipt_sha256"] != terminal["receipt_sha256"]
        or request["original_full_canary_owner_approval_sha256"]
        != terminal["owner_approval_sha256"]
        or terminal["release_sha"] != request["revision"]
        or terminal["full_canary_plan_sha256"] != request["full_canary_plan_sha256"]
        or request["full_canary_terminal_receipt"]
        != getattr(plan, "full_canary_terminal_receipt", None)
        or request["secret_material_recorded"] is not False
        or request["semantic_content_recorded"] is not False
    ):
        _fail("producer_foundation_prepare_invalid")
    if plan_publication_receipt is None:
        from gateway.canonical_capability_canary_runtime import (
            load_bound_plan_publication_receipt,
        )

        publication = load_bound_plan_publication_receipt(plan)
    else:
        publication = copy.deepcopy(dict(plan_publication_receipt))
    publication_sha256 = _digest(
        publication.get("receipt_sha256"),
        "producer_identity_foundation_invalid",
    )
    if service_identity_foundation is None:
        from gateway.canonical_capability_canary_runtime import (
            ensure_service_identities_create_only,
        )

        service_foundation = ensure_service_identities_create_only(plan, full_plan)
    else:
        service_foundation = copy.deepcopy(dict(service_identity_foundation))
    service_foundation = _validate_service_identity_foundation_binding(
        service_foundation,
        revision=plan.revision,
        capability_plan_sha256=plan.sha256,
        full_canary_plan_sha256=full_plan.sha256,
        terminal_receipt_sha256=terminal["receipt_sha256"],
        original_owner_approval_sha256=terminal["owner_approval_sha256"],
        plan_publication_receipt_sha256=publication_sha256,
    )
    if producer_identity_foundation is None:
        identity_foundation = ensure_producer_role_identities(
            plan=plan,
            full_plan=full_plan,
            plan_publication_receipt=publication,
        )
    else:
        identity_foundation = _validate_producer_identity_foundation_receipt(
            producer_identity_foundation,
            plan=plan,
            full_plan=full_plan,
            plan_publication_receipt_sha256=publication_sha256,
            receipt_path=_producer_identity_foundation_path(plan),
        )
    identities_value = planned_producer_role_identities()
    if role_identities is not None and dict(role_identities) != identities_value:
        _fail("producer_role_identity_not_fixed")
    identities = _role_identities(identities_value)
    owner = project_pinned_owner_public_key_source(
        request["owner_public_authority"],
        expected_comment=PRODUCER_OWNER_PUBLIC_KEY_COMMENT,
    )
    if key_root == DEFAULT_KEY_ROOT:
        _ensure_exact_directory(key_root, uid=0, gid=0, mode=0o700)
    root_item = key_root.lstat()
    keys = bootstrap_producer_keys(
        role_identities=identities_value,
        key_root=key_root,
        receipt_path=key_bootstrap_receipt,
        root_uid=root_item.st_uid,
        root_gid=root_item.st_gid,
    )
    identity = render_producer_unit_identity_contract(
        revision=plan.revision,
        role_identities=identities_value,
    )
    endpoints = endpoint_contracts(
        identity_contract=identity,
        key_bootstrap=keys,
    )
    authorities = {
        role: {
            "key_id": keys.public_contracts[role]["key_id"],
            "algorithm": "ed25519",
            "public_key_ed25519_hex": keys.public_contracts[role][
                "public_key_ed25519_hex"
            ],
        }
        for role in ENDPOINT_ROLES
    }
    authorities["owner"] = {
        "key_id": owner["key_id"],
        "algorithm": "sshsig-ed25519-sha512",
        "public_key_ed25519_hex": owner["public_key_ed25519_hex"],
    }
    unsigned_foundation = {
        "schema": PRODUCER_FOUNDATION_SCHEMA,
        "release_sha": plan.revision,
        "capability_plan_sha256": plan.sha256,
        "full_canary_plan_sha256": full_plan.sha256,
        "full_canary_terminal_receipt": copy.deepcopy(dict(terminal)),
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "original_full_canary_owner_approval_sha256": terminal["owner_approval_sha256"],
        "service_identity_foundation_receipt_sha256": service_foundation[
            "receipt_sha256"
        ],
        "producer_identity_foundation_receipt_sha256": identity_foundation[
            "receipt_sha256"
        ],
        "owner_id": PRODUCTION_OWNER_ID,
        "owner_authority": {
            "owner_id": PRODUCTION_OWNER_ID,
            "key_id": owner["key_id"],
            "algorithm": "sshsig-ed25519-sha512",
            "public_key_ed25519_hex": owner["public_key_ed25519_hex"],
            "public_key_source": owner["public_key_source"],
            "public_key_source_sha256": owner["public_key_source_sha256"],
        },
        "authority_keys": authorities,
        "endpoints": endpoints,
        "bitrix_operational_edge_contract": _plan_bitrix_contract(plan),
        "discord_edge_evidence_contract": _discord_edge_evidence_contract(
            plan, full_plan
        ),
        "receipt_contract": {
            "base_root": str(DEFAULT_RECEIPT_ROOT),
            "run_directory_uid": 0,
            "run_directory_gid": next(iter(identities.values())).receipt_writer_gid,
            "run_directory_mode": 0o3770,
            "slot_filenames": dict(SLOT_FILENAME),
            "slot_roles": dict(SLOT_ROLE),
            "slot_native_binding_kinds": {
                slot: list(SLOT_NATIVE_BINDING_KINDS[slot]) for slot in RECEIPT_SLOTS
            },
        },
        "producer_protocol": "role_local_native_evidence_v1",
        "root_can_sign_non_observer_roles": False,
        "token_or_token_digest_recorded": False,
        "signature_namespace": PRODUCER_FOUNDATION_SSHSIG_NAMESPACE,
        "signature_algorithm": "sshsig-ed25519-sha512",
    }
    unsigned = {
        "schema": FOUNDATION_PREPARATION_SCHEMA,
        "revision": plan.revision,
        "capability_plan_sha256": plan.sha256,
        "full_canary_plan_sha256": full_plan.sha256,
        "full_canary_terminal_receipt": copy.deepcopy(dict(terminal)),
        "full_canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "original_full_canary_owner_approval_sha256": terminal["owner_approval_sha256"],
        "service_identity_foundation": service_foundation,
        "service_identity_foundation_receipt_sha256": service_foundation[
            "receipt_sha256"
        ],
        "producer_identity_foundation": identity_foundation,
        "producer_identity_foundation_receipt_sha256": identity_foundation[
            "receipt_sha256"
        ],
        "role_identities": identities_value,
        "key_bootstrap_receipt_sha256": keys.value["receipt_sha256"],
        "owner_public_key_ed25519_hex": owner["public_key_ed25519_hex"],
        "owner_public_key_source_sha256": owner["public_key_source_sha256"],
        "unsigned_foundation": unsigned_foundation,
        "signature_payload_sha256": _sha256_bytes(
            producer_foundation_signature_payload(unsigned_foundation)
        ),
        "secret_material_recorded": False,
        "semantic_content_recorded": False,
    }
    preparation = dict(
        validate_foundation_preparation(
            {
                **unsigned,
                "preparation_sha256": _sha256_json(unsigned),
            },
            expected_plan_publication_receipt_sha256=publication_sha256,
        )
    )
    if preparation_path == DEFAULT_FOUNDATION_PREPARATION_PATH:
        _ensure_exact_directory(preparation_path.parent, uid=0, gid=0, mode=0o700)
        _publish_root_file(
            owner_public_key_hex_path,
            (owner["public_key_ed25519_hex"] + "\n").encode("ascii"),
        )
        _publish_root_file(
            owner_public_key_source_sha256_path,
            (owner["public_key_source_sha256"] + "\n").encode("ascii"),
        )
        _publish_no_replace(
            preparation_path,
            _canonical_bytes(preparation),
            uid=0,
            gid=0,
            mode=0o400,
            parent_uid=0,
            parent_gid=0,
            parent_mode=0o700,
        )
    return copy.deepcopy(preparation)


def _daemon_reload() -> None:
    try:
        result = subprocess.run(
            (str(SYSTEMCTL), "daemon-reload"),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CapabilityProducerError("producer_daemon_reload_failed") from exc
    if result.returncode != 0:
        _fail("producer_daemon_reload_failed")


def _materialize_volatile_runtime() -> None:
    try:
        result = subprocess.run(
            (str(SYSTEMD_TMPFILES), "--create", str(DEFAULT_TMPFILES_PATH)),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CapabilityProducerError("producer_tmpfiles_failed") from exc
    if result.returncode != 0:
        _fail("producer_tmpfiles_failed")


def install_prepared_producer_foundation(
    request_value: Any,
    *,
    preparation: Mapping[str, Any],
    daemon_reload: Callable[[], None] = _daemon_reload,
    materialize_runtime: Callable[[], None] = _materialize_volatile_runtime,
) -> Mapping[str, Any]:
    """Verify the external owner signature and install its exact bundle."""

    preparation = validate_foundation_preparation(preparation)
    request = _strict(
        request_value,
        ("schema", "preparation_sha256", "owner_signature"),
        "producer_foundation_install_invalid",
    )
    if (
        request["schema"] != FOUNDATION_INSTALL_REQUEST_SCHEMA
        or request["preparation_sha256"] != preparation.get("preparation_sha256")
        or not isinstance(request["owner_signature"], str)
    ):
        _fail("producer_foundation_install_invalid")
    owner_hex = preparation["owner_public_key_ed25519_hex"]
    owner_source_sha256 = preparation["owner_public_key_source_sha256"]
    foundation = seal_producer_foundation(
        preparation["unsigned_foundation"],
        owner_signature=request["owner_signature"],
        pinned_owner_public_key_ed25519_hex=owner_hex,
        pinned_owner_public_key_source_sha256=owner_source_sha256,
    )
    identities = preparation["role_identities"]
    bundle = render_producer_units(
        foundation=foundation,
        pinned_owner_public_key_ed25519_hex=owner_hex,
        pinned_owner_public_key_source_sha256=owner_source_sha256,
        role_identities=identities,
    )
    _ensure_exact_directory(DEFAULT_CONFIG_ROOT, uid=0, gid=0, mode=0o755)
    _ensure_exact_directory(DEFAULT_RECEIPT_ROOT, uid=0, gid=0, mode=0o711)
    _ensure_exact_directory(DEFAULT_NATIVE_ROOT, uid=0, gid=0, mode=0o755)
    for role in ENDPOINT_ROLES:
        role_identity = ProducerRoleIdentity.from_mapping(role, identities[role])
        _ensure_exact_directory(
            DEFAULT_NATIVE_ROOT / role,
            uid=role_identity.uid,
            gid=role_identity.gid,
            mode=0o700,
        )
    _publish_root_file(DEFAULT_FOUNDATION_PATH, _canonical_bytes(foundation))
    for path_text, raw in bundle.units.items():
        _publish_root_file(Path(path_text), raw, mode=0o644)
    for path_text, raw in bundle.auxiliary_files.items():
        _publish_root_file(Path(path_text), raw, mode=0o644)
    for role in ENDPOINT_ROLES:
        path = producer_config_path(role)
        identity = ProducerRoleIdentity.from_mapping(role, identities[role])
        _publish_no_replace(
            path,
            bundle.configs[str(path)],
            uid=0,
            gid=identity.gid,
            mode=0o440,
            parent_uid=0,
            parent_gid=0,
            parent_mode=0o755,
        )
    materialize_runtime()
    daemon_reload()
    unsigned = {
        "schema": FOUNDATION_INSTALL_RECEIPT_SCHEMA,
        "revision": foundation["release_sha"],
        "capability_plan_sha256": foundation["capability_plan_sha256"],
        "full_canary_plan_sha256": foundation["full_canary_plan_sha256"],
        "full_canary_terminal_receipt": copy.deepcopy(
            dict(foundation["full_canary_terminal_receipt"])
        ),
        "full_canary_terminal_receipt_sha256": foundation[
            "full_canary_terminal_receipt_sha256"
        ],
        "original_full_canary_owner_approval_sha256": foundation[
            "original_full_canary_owner_approval_sha256"
        ],
        "service_identity_foundation_receipt_sha256": preparation[
            "service_identity_foundation_receipt_sha256"
        ],
        "producer_identity_foundation_receipt_sha256": preparation[
            "producer_identity_foundation_receipt_sha256"
        ],
        "preparation_sha256": preparation["preparation_sha256"],
        "foundation_sha256": producer_foundation_sha256(foundation),
        "unit_bundle_manifest_sha256": bundle.manifest["manifest_sha256"],
        "installed_units": sorted(bundle.units),
        "installed_configs": sorted(bundle.configs),
        "installed_auxiliary_files": sorted(bundle.auxiliary_files),
        "native_root_contract": bundle.manifest["native_root_contract"],
        "config_install_contract": bundle.manifest["config_install_contract"],
        "volatile_runtime_contract": bundle.manifest["volatile_runtime_contract"],
        "authority_key_lifecycle": bundle.manifest["authority_key_lifecycle"],
        "daemon_reload_completed": True,
        "volatile_runtime_materialized": True,
        "services_started": False,
        "secret_material_recorded": False,
    }
    receipt = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
    _publish_no_replace(
        DEFAULT_FOUNDATION_INSTALL_RECEIPT,
        _canonical_bytes(receipt),
        uid=0,
        gid=0,
        mode=0o400,
        parent_uid=0,
        parent_gid=0,
        parent_mode=0o700,
    )
    return copy.deepcopy(receipt)


def validate_foundation_preparation(
    value: Any,
    *,
    expected_plan_publication_receipt_sha256: str | None = None,
) -> Mapping[str, Any]:
    raw = _strict(
        value,
        (
            "schema",
            "revision",
            "capability_plan_sha256",
            "full_canary_plan_sha256",
            "full_canary_terminal_receipt",
            "full_canary_terminal_receipt_sha256",
            "original_full_canary_owner_approval_sha256",
            "service_identity_foundation",
            "service_identity_foundation_receipt_sha256",
            "producer_identity_foundation",
            "producer_identity_foundation_receipt_sha256",
            "role_identities",
            "key_bootstrap_receipt_sha256",
            "owner_public_key_ed25519_hex",
            "owner_public_key_source_sha256",
            "unsigned_foundation",
            "signature_payload_sha256",
            "secret_material_recorded",
            "semantic_content_recorded",
            "preparation_sha256",
        ),
        "producer_foundation_preparation_invalid",
    )
    unsigned = {key: item for key, item in raw.items() if key != "preparation_sha256"}
    foundation = raw["unsigned_foundation"]
    if not isinstance(foundation, Mapping):
        _fail("producer_foundation_preparation_invalid")
    identities = _role_identities(raw["role_identities"])
    owner = foundation.get("owner_authority")
    endpoints = foundation.get("endpoints")
    authorities = foundation.get("authority_keys")
    terminal = _validate_full_canary_terminal(
        raw["full_canary_terminal_receipt"],
        code="producer_foundation_preparation_invalid",
    )
    identity_foundation_value = raw["producer_identity_foundation"]
    if not isinstance(identity_foundation_value, Mapping):
        _fail("producer_foundation_preparation_invalid")
    service_foundation = _validate_service_identity_foundation_binding(
        raw["service_identity_foundation"],
        revision=raw["revision"],
        capability_plan_sha256=raw["capability_plan_sha256"],
        full_canary_plan_sha256=raw["full_canary_plan_sha256"],
        terminal_receipt_sha256=terminal["receipt_sha256"],
        original_owner_approval_sha256=terminal["owner_approval_sha256"],
        plan_publication_receipt_sha256=identity_foundation_value.get(
            "plan_publication_receipt_sha256"
        ),
    )
    identity_plan = SimpleNamespace(
        revision=raw["revision"],
        sha256=raw["capability_plan_sha256"],
        full_canary_terminal_receipt_sha256=terminal["receipt_sha256"],
        original_full_canary_owner_approval_sha256=terminal["owner_approval_sha256"],
    )
    producer_identity_foundation = _validate_producer_identity_foundation_receipt(
        identity_foundation_value,
        plan=identity_plan,
        full_plan=SimpleNamespace(sha256=raw["full_canary_plan_sha256"]),
        plan_publication_receipt_sha256=identity_foundation_value.get(
            "plan_publication_receipt_sha256"
        ),
        receipt_path=_producer_identity_foundation_path(identity_plan),
    )
    publication_sha256 = producer_identity_foundation["plan_publication_receipt_sha256"]
    if service_foundation["plan_publication_receipt_sha256"] != publication_sha256 or (
        expected_plan_publication_receipt_sha256 is not None
        and publication_sha256
        != _digest(
            expected_plan_publication_receipt_sha256,
            "producer_foundation_preparation_invalid",
        )
    ):
        _fail("producer_foundation_preparation_invalid")
    if (
        raw["schema"] != FOUNDATION_PREPARATION_SCHEMA
        or _GIT_SHA_RE.fullmatch(str(raw["revision"])) is None
        or raw["secret_material_recorded"] is not False
        or raw["semantic_content_recorded"] is not False
        or raw["preparation_sha256"] != _sha256_json(unsigned)
        or raw["signature_payload_sha256"]
        != _sha256_bytes(producer_foundation_signature_payload(foundation))
        or foundation.get("release_sha") != raw["revision"]
        or foundation.get("capability_plan_sha256") != raw["capability_plan_sha256"]
        or foundation.get("full_canary_plan_sha256") != raw["full_canary_plan_sha256"]
        or raw["full_canary_terminal_receipt_sha256"] != terminal["receipt_sha256"]
        or raw["original_full_canary_owner_approval_sha256"]
        != terminal["owner_approval_sha256"]
        or terminal["release_sha"] != raw["revision"]
        or terminal["full_canary_plan_sha256"] != raw["full_canary_plan_sha256"]
        or foundation.get("full_canary_terminal_receipt") != terminal
        or foundation.get("full_canary_terminal_receipt_sha256")
        != terminal["receipt_sha256"]
        or foundation.get("original_full_canary_owner_approval_sha256")
        != terminal["owner_approval_sha256"]
        or raw["service_identity_foundation_receipt_sha256"]
        != service_foundation["receipt_sha256"]
        or foundation.get("service_identity_foundation_receipt_sha256")
        != service_foundation["receipt_sha256"]
        or raw["producer_identity_foundation_receipt_sha256"]
        != producer_identity_foundation["receipt_sha256"]
        or foundation.get("producer_identity_foundation_receipt_sha256")
        != producer_identity_foundation["receipt_sha256"]
        or raw["role_identities"] != producer_identity_foundation["planned_identities"]
        or not isinstance(owner, Mapping)
        or owner.get("public_key_ed25519_hex") != raw["owner_public_key_ed25519_hex"]
        or owner.get("public_key_source_sha256")
        != raw["owner_public_key_source_sha256"]
        or not isinstance(endpoints, Mapping)
        or set(endpoints) != set(ENDPOINT_ROLES)
        or not isinstance(authorities, Mapping)
        or set(authorities) != {*ENDPOINT_ROLES, "owner"}
    ):
        _fail("producer_foundation_preparation_invalid")
    _digest(raw["capability_plan_sha256"], "producer_foundation_preparation_invalid")
    _digest(raw["full_canary_plan_sha256"], "producer_foundation_preparation_invalid")
    _digest(
        raw["key_bootstrap_receipt_sha256"], "producer_foundation_preparation_invalid"
    )
    _digest(
        raw["service_identity_foundation_receipt_sha256"],
        "producer_foundation_preparation_invalid",
    )
    _digest(
        raw["producer_identity_foundation_receipt_sha256"],
        "producer_foundation_preparation_invalid",
    )
    _digest(
        raw["owner_public_key_source_sha256"], "producer_foundation_preparation_invalid"
    )
    if re.fullmatch(r"[0-9a-f]{64}", str(raw["owner_public_key_ed25519_hex"])) is None:
        _fail("producer_foundation_preparation_invalid")
    for role, identity in identities.items():
        endpoint = endpoints[role]
        authority = authorities[role]
        if (
            not isinstance(endpoint, Mapping)
            or not isinstance(authority, Mapping)
            or endpoint.get("uid") != identity.uid
            or endpoint.get("gid") != identity.gid
            or endpoint.get("service_unit") != PRODUCER_SERVICE_UNITS[role]
            or endpoint.get("socket_path") != str(producer_socket_path(role))
            or endpoint.get("private_key_path")
            != str(producer_private_key_projection_path(role))
            or endpoint.get("public_key_path") != str(producer_public_key_path(role))
            or endpoint.get("key_id") != authority.get("key_id")
            or endpoint.get("public_key_ed25519_hex")
            != authority.get("public_key_ed25519_hex")
        ):
            _fail("producer_foundation_preparation_invalid")
    owner_authority = authorities["owner"]
    if (
        not isinstance(owner_authority, Mapping)
        or owner_authority.get("key_id") != owner.get("key_id")
        or owner_authority.get("public_key_ed25519_hex")
        != raw["owner_public_key_ed25519_hex"]
    ):
        _fail("producer_foundation_preparation_invalid")
    return copy.deepcopy(raw)


def load_foundation_preparation(
    path: Path = DEFAULT_FOUNDATION_PREPARATION_PATH,
) -> Mapping[str, Any]:
    raw, _item = _stable_read(
        path,
        maximum=2 * 1024 * 1024,
        uid=0,
        gid=0,
        mode=0o400,
    )
    return validate_foundation_preparation(
        _strict_json(raw, "producer_foundation_preparation_invalid")
    )


def _validate_key_bootstrap_receipt(value: Any) -> Mapping[str, Any]:
    raw = _strict(
        value,
        (
            "schema",
            "keys",
            "roles",
            "independent_key_ids",
            "private_content_or_digest_recorded",
            "receipt_sha256",
        ),
        "producer_key_bootstrap_invalid",
    )
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    rows = raw["keys"]
    if (
        raw["schema"] != KEY_BOOTSTRAP_SCHEMA
        or raw["roles"] != list(ENDPOINT_ROLES)
        or raw["independent_key_ids"] is not True
        or raw["private_content_or_digest_recorded"] is not False
        or raw["receipt_sha256"] != _sha256_json(unsigned)
        or not isinstance(rows, list)
        or len(rows) != len(ENDPOINT_ROLES)
    ):
        _fail("producer_key_bootstrap_invalid")
    for role, value in zip(ENDPOINT_ROLES, rows, strict=True):
        row = _strict(
            value,
            (
                "role",
                "private_key_source_path",
                "private_key_projection_path",
                "key_id",
                "algorithm",
                "public_key_ed25519_hex",
                "public_key_source_path",
                "public_key_projection_path",
                "public_key_file_sha256",
                "private_content_or_digest_recorded",
            ),
            "producer_key_bootstrap_invalid",
        )
        if (
            row["role"] != role
            or row["algorithm"] != "ed25519"
            or row["private_key_source_path"]
            != str(producer_private_key_source_path(role))
            or row["private_key_projection_path"]
            != str(producer_private_key_projection_path(role))
            or row["public_key_source_path"]
            != str(producer_public_key_source_path(role))
            or row["public_key_projection_path"] != str(producer_public_key_path(role))
            or row["private_content_or_digest_recorded"] is not False
        ):
            _fail("producer_key_bootstrap_invalid")
        _digest(row["key_id"], "producer_key_bootstrap_invalid")
        _digest(row["public_key_file_sha256"], "producer_key_bootstrap_invalid")
    return copy.deepcopy(raw)


def _load_install_receipt() -> Mapping[str, Any]:
    raw, _item = _stable_read(
        DEFAULT_FOUNDATION_INSTALL_RECEIPT,
        maximum=2 * 1024 * 1024,
        uid=0,
        gid=0,
        mode=0o400,
    )
    value = _strict_json(raw, "producer_install_receipt_invalid")
    receipt = _strict(
        value,
        (
            "schema",
            "revision",
            "capability_plan_sha256",
            "full_canary_plan_sha256",
            "full_canary_terminal_receipt",
            "full_canary_terminal_receipt_sha256",
            "original_full_canary_owner_approval_sha256",
            "service_identity_foundation_receipt_sha256",
            "producer_identity_foundation_receipt_sha256",
            "preparation_sha256",
            "foundation_sha256",
            "unit_bundle_manifest_sha256",
            "installed_units",
            "installed_configs",
            "installed_auxiliary_files",
            "native_root_contract",
            "config_install_contract",
            "volatile_runtime_contract",
            "authority_key_lifecycle",
            "daemon_reload_completed",
            "volatile_runtime_materialized",
            "services_started",
            "secret_material_recorded",
            "receipt_sha256",
        ),
        "producer_install_receipt_invalid",
    )
    unsigned = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    terminal = _validate_full_canary_terminal(
        receipt["full_canary_terminal_receipt"],
        code="producer_install_receipt_invalid",
    )
    if (
        receipt["schema"] != FOUNDATION_INSTALL_RECEIPT_SCHEMA
        or receipt["receipt_sha256"] != _sha256_json(unsigned)
        or receipt["daemon_reload_completed"] is not True
        or receipt["volatile_runtime_materialized"] is not True
        or receipt["services_started"] is not False
        or receipt["secret_material_recorded"] is not False
        or receipt["full_canary_terminal_receipt_sha256"] != terminal["receipt_sha256"]
        or receipt["original_full_canary_owner_approval_sha256"]
        != terminal["owner_approval_sha256"]
        or receipt["revision"] != terminal["release_sha"]
        or receipt["full_canary_plan_sha256"] != terminal["full_canary_plan_sha256"]
    ):
        _fail("producer_install_receipt_invalid")
    _digest(
        receipt["service_identity_foundation_receipt_sha256"],
        "producer_install_receipt_invalid",
    )
    _digest(
        receipt["producer_identity_foundation_receipt_sha256"],
        "producer_install_receipt_invalid",
    )
    return copy.deepcopy(receipt)


def validate_installed_producer_foundation(
    *,
    plan: Any,
    full_plan: Any,
) -> Mapping[str, Any]:
    """Read-only preflight for the exact durable installation."""

    installed = load_installed_producer_foundation()
    preparation = load_foundation_preparation()
    install_receipt = _load_install_receipt()
    producer_identity_foundation = load_producer_identity_foundation_receipt(
        plan=plan,
        full_plan=full_plan,
    )
    foundation = installed.value
    if (
        foundation["release_sha"] != getattr(plan, "revision", None)
        or foundation["release_sha"] != getattr(full_plan, "revision", None)
        or foundation["capability_plan_sha256"] != getattr(plan, "sha256", None)
        or foundation["full_canary_plan_sha256"] != getattr(full_plan, "sha256", None)
        or foundation["full_canary_terminal_receipt"]
        != getattr(plan, "full_canary_terminal_receipt", None)
        or foundation["full_canary_terminal_receipt_sha256"]
        != getattr(plan, "full_canary_terminal_receipt_sha256", None)
        or foundation["original_full_canary_owner_approval_sha256"]
        != getattr(plan, "original_full_canary_owner_approval_sha256", None)
        or preparation["unsigned_foundation"]
        != {key: item for key, item in foundation.items() if key != "owner_signature"}
        or producer_identity_foundation["receipt_sha256"]
        != preparation["producer_identity_foundation_receipt_sha256"]
    ):
        _fail("producer_foundation_plan_mismatch")
    bundle = render_producer_units(
        foundation=foundation,
        pinned_owner_public_key_ed25519_hex=(
            installed.pinned_owner_public_key_ed25519_hex
        ),
        pinned_owner_public_key_source_sha256=(
            installed.pinned_owner_public_key_source_sha256
        ),
        role_identities=preparation["role_identities"],
    )
    attest_producer_role_identities(preparation["role_identities"])
    attest_foundation_service_identities(plan)
    _require_exact_directory(DEFAULT_KEY_ROOT, uid=0, gid=0, mode=0o700)
    bootstrap_raw, _bootstrap_item = _stable_read(
        DEFAULT_KEY_BOOTSTRAP_RECEIPT,
        maximum=2 * 1024 * 1024,
        uid=0,
        gid=0,
        mode=0o400,
    )
    bootstrap = _validate_key_bootstrap_receipt(
        _strict_json(bootstrap_raw, "producer_key_bootstrap_invalid")
    )
    if (
        bootstrap["receipt_sha256"] != preparation["key_bootstrap_receipt_sha256"]
        or install_receipt.get("preparation_sha256")
        != preparation["preparation_sha256"]
        or install_receipt.get("service_identity_foundation_receipt_sha256")
        != preparation["service_identity_foundation_receipt_sha256"]
        or install_receipt.get("producer_identity_foundation_receipt_sha256")
        != preparation["producer_identity_foundation_receipt_sha256"]
        or install_receipt.get("foundation_sha256") != installed.sha256
        or install_receipt.get("full_canary_terminal_receipt")
        != foundation["full_canary_terminal_receipt"]
        or install_receipt.get("full_canary_terminal_receipt_sha256")
        != foundation["full_canary_terminal_receipt_sha256"]
        or install_receipt.get("original_full_canary_owner_approval_sha256")
        != foundation["original_full_canary_owner_approval_sha256"]
        or install_receipt.get("unit_bundle_manifest_sha256")
        != bundle.manifest["manifest_sha256"]
        or install_receipt.get("installed_units") != sorted(bundle.units)
        or install_receipt.get("installed_configs") != sorted(bundle.configs)
        or install_receipt.get("installed_auxiliary_files")
        != sorted(bundle.auxiliary_files)
        or install_receipt.get("native_root_contract")
        != bundle.manifest["native_root_contract"]
        or install_receipt.get("config_install_contract")
        != bundle.manifest["config_install_contract"]
        or install_receipt.get("volatile_runtime_contract")
        != bundle.manifest["volatile_runtime_contract"]
        or install_receipt.get("authority_key_lifecycle")
        != bundle.manifest["authority_key_lifecycle"]
    ):
        _fail("producer_install_receipt_mismatch")
    for role in ENDPOINT_ROLES:
        private, _private_file_sha256 = _load_private_key(
            producer_private_key_source_path(role), uid=0, gid=0
        )
        public, public_hex, public_file_sha256 = _load_public_key(
            producer_public_key_source_path(role), uid=0, gid=0, mode=0o400
        )
        private_hex = (
            private
            .public_key()
            .public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
            .hex()
        )
        observed_public_hex = public.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()
        endpoint = foundation["endpoints"][role]
        if (
            private_hex != public_hex
            or observed_public_hex != public_hex
            or endpoint["public_key_ed25519_hex"] != public_hex
            or endpoint["public_key_file_sha256"] != public_file_sha256
            or endpoint["key_id"] != _sha256_bytes(bytes.fromhex(public_hex))
        ):
            _fail("producer_key_installation_drifted")
    for path_text, expected in bundle.units.items():
        observed, _item = _stable_read(
            Path(path_text),
            maximum=512 * 1024,
            uid=0,
            gid=0,
            mode=0o644,
        )
        if observed != expected:
            _fail("producer_unit_installation_drifted")
    for path_text, expected in bundle.auxiliary_files.items():
        observed, _item = _stable_read(
            Path(path_text),
            maximum=512 * 1024,
            uid=0,
            gid=0,
            mode=0o644,
        )
        if observed != expected:
            _fail("producer_auxiliary_installation_drifted")
    for role in ENDPOINT_ROLES:
        identity = ProducerRoleIdentity.from_mapping(
            role, preparation["role_identities"][role]
        )
        path = producer_config_path(role)
        observed, _item = _stable_read(
            path,
            maximum=512 * 1024,
            uid=0,
            gid=identity.gid,
            mode=0o440,
        )
        if observed != bundle.configs[str(path)]:
            _fail("producer_config_installation_drifted")
        _require_exact_directory(
            DEFAULT_NATIVE_ROOT / role,
            uid=identity.uid,
            gid=identity.gid,
            mode=0o700,
        )
    _require_exact_directory(DEFAULT_NATIVE_ROOT, uid=0, gid=0, mode=0o755)
    _require_exact_directory(DEFAULT_RECEIPT_ROOT, uid=0, gid=0, mode=0o711)
    _require_exact_directory(
        Path("/run/muncho-capability-canary"), uid=0, gid=0, mode=0o700
    )
    _require_exact_directory(DEFAULT_RUNTIME_ROOT, uid=0, gid=0, mode=0o755)
    return {
        "schema": "muncho-capability-producer-installation-preflight.v2",
        "revision": foundation["release_sha"],
        "full_canary_terminal_receipt": copy.deepcopy(
            dict(foundation["full_canary_terminal_receipt"])
        ),
        "full_canary_terminal_receipt_sha256": foundation[
            "full_canary_terminal_receipt_sha256"
        ],
        "original_full_canary_owner_approval_sha256": foundation[
            "original_full_canary_owner_approval_sha256"
        ],
        "service_identity_foundation_receipt_sha256": preparation[
            "service_identity_foundation_receipt_sha256"
        ],
        "producer_identity_foundation_receipt_sha256": preparation[
            "producer_identity_foundation_receipt_sha256"
        ],
        "foundation_sha256": installed.sha256,
        "preparation_sha256": preparation["preparation_sha256"],
        "unit_bundle_manifest_sha256": bundle.manifest["manifest_sha256"],
        "ready": True,
        "mutation_performed": False,
    }


def endpoint_contracts(
    *,
    identity_contract: ProducerUnitIdentityContract,
    key_bootstrap: ProducerKeyBootstrap,
) -> Mapping[str, Mapping[str, Any]]:
    if set(key_bootstrap.public_contracts) != set(ENDPOINT_ROLES):
        _fail("producer_key_bootstrap_invalid")
    result: dict[str, Mapping[str, Any]] = {}
    for role in ENDPOINT_ROLES:
        identity = identity_contract.role_identities[role]
        key = key_bootstrap.public_contracts[role]
        result[role] = {
            "service_unit": PRODUCER_SERVICE_UNITS[role],
            "service_identity_sha256": identity_contract.service_identity_sha256s[role],
            "uid": identity.uid,
            "gid": identity.gid,
            "socket_path": str(producer_socket_path(role)),
            "private_key_path": str(producer_private_key_projection_path(role)),
            "public_key_path": str(producer_public_key_path(role)),
            "public_key_file_sha256": key["public_key_file_sha256"],
            "allowed_slots": [
                slot for slot in RECEIPT_SLOTS if SLOT_ROLE[slot] == role
            ],
            "key_id": key["key_id"],
            "algorithm": "ed25519",
            "public_key_ed25519_hex": key["public_key_ed25519_hex"],
        }
    return result


def _render_tmpfiles(
    identities: Mapping[str, ProducerRoleIdentity],
) -> bytes:
    lines = [
        "# Volatile roots for the signed capability-canary producer foundation.",
        "d /run/muncho-capability-canary 0700 root root -",
        f"d {DEFAULT_RUNTIME_ROOT} 0755 root root -",
        f"d {DEFAULT_NATIVE_ROOT} 0755 root root -",
    ]
    for role in ENDPOINT_ROLES:
        identity = identities[role]
        lines.extend((
            f"d {DEFAULT_RUNTIME_ROOT / role} 0700 {identity.user} {identity.group} -",
            f"d {DEFAULT_NATIVE_ROOT / role} 0700 {identity.user} {identity.group} -",
        ))
    return _unit_bytes(lines)


def _producer_config(
    *,
    role: str,
    foundation: Mapping[str, Any],
) -> bytes:
    endpoint = foundation["endpoints"][role]
    receipt = foundation["receipt_contract"]
    value = {
        "schema": PRODUCER_CONFIG_SCHEMA,
        "role": role,
        "foundation_sha256": producer_foundation_sha256(foundation),
        "release_sha": foundation["release_sha"],
        "capability_plan_sha256": foundation["capability_plan_sha256"],
        "full_canary_plan_sha256": foundation["full_canary_plan_sha256"],
        "service_unit": endpoint["service_unit"],
        "service_identity_sha256": endpoint["service_identity_sha256"],
        "service_uid": endpoint["uid"],
        "service_gid": endpoint["gid"],
        "root_client_uid": 0,
        "socket_path": endpoint["socket_path"],
        "receipt_base_root": receipt["base_root"],
        "receipt_directory_uid": receipt["run_directory_uid"],
        "receipt_directory_gid": receipt["run_directory_gid"],
        "receipt_directory_mode": receipt["run_directory_mode"],
        "private_key_path": endpoint["private_key_path"],
        "public_key_path": endpoint["public_key_path"],
        "allowed_slots": endpoint["allowed_slots"],
    }
    ProducerConfig.from_mapping(value)
    return _canonical_bytes(value)


def render_producer_units(
    *,
    foundation: Mapping[str, Any],
    pinned_owner_public_key_ed25519_hex: str,
    pinned_owner_public_key_source_sha256: str,
    role_identities: Mapping[str, Any],
) -> ProducerUnitBundle:
    trusted = validate_producer_foundation(
        foundation,
        pinned_owner_public_key_ed25519_hex=(pinned_owner_public_key_ed25519_hex),
        pinned_owner_public_key_source_sha256=(pinned_owner_public_key_source_sha256),
    )
    identity = render_producer_unit_identity_contract(
        revision=trusted["release_sha"],
        role_identities=role_identities,
    )
    configs: dict[str, bytes] = {}
    for role in ENDPOINT_ROLES:
        endpoint = trusted["endpoints"][role]
        observed_identity = identity.role_identities[role]
        if (
            endpoint["service_unit"] != PRODUCER_SERVICE_UNITS[role]
            or endpoint["service_identity_sha256"]
            != identity.service_identity_sha256s[role]
            or endpoint["uid"] != observed_identity.uid
            or endpoint["gid"] != observed_identity.gid
            or endpoint["socket_path"] != str(producer_socket_path(role))
            or endpoint["private_key_path"]
            != str(producer_private_key_projection_path(role))
            or endpoint["public_key_path"] != str(producer_public_key_path(role))
        ):
            _fail("producer_endpoint_foundation_mismatch")
        raw = _producer_config(role=role, foundation=trusted)
        config = ProducerConfig.from_mapping(
            _strict_json(raw, "producer_config_invalid")
        )
        validate_producer_config_binding(config, trusted)
        configs[str(producer_config_path(role))] = raw
    auxiliary_files = {
        str(DEFAULT_TMPFILES_PATH): _render_tmpfiles(identity.role_identities)
    }
    manifest_unsigned = {
        "schema": UNIT_BUNDLE_SCHEMA,
        "revision": trusted["release_sha"],
        "release_root": str(identity.release_root),
        "foundation_sha256": producer_foundation_sha256(trusted),
        "units": {
            path: _sha256_bytes(raw) for path, raw in sorted(identity.units.items())
        },
        "configs": {path: _sha256_bytes(raw) for path, raw in sorted(configs.items())},
        "auxiliary_files": {
            path: _sha256_bytes(raw) for path, raw in sorted(auxiliary_files.items())
        },
        "config_install_contract": {
            str(producer_config_path(role)): {
                "uid": 0,
                "gid": identity.role_identities[role].gid,
                "mode": 0o440,
            }
            for role in ENDPOINT_ROLES
        },
        "native_root_contract": {
            str(DEFAULT_NATIVE_ROOT): {"uid": 0, "gid": 0, "mode": 0o755},
            **{
                str(DEFAULT_NATIVE_ROOT / role): {
                    "uid": identity.role_identities[role].uid,
                    "gid": identity.role_identities[role].gid,
                    "mode": 0o700,
                }
                for role in ENDPOINT_ROLES
            },
        },
        "volatile_runtime_contract": {
            "/run/muncho-capability-canary": {
                "uid": 0,
                "gid": 0,
                "mode": 0o700,
                "activation_file_retired_independently": True,
            },
            str(DEFAULT_RUNTIME_ROOT): {"uid": 0, "gid": 0, "mode": 0o755},
            "tmpfiles_path": str(DEFAULT_TMPFILES_PATH),
            "recreated_at_boot_without_foundation_rotation": True,
        },
        "service_identity_sha256s": dict(identity.service_identity_sha256s),
        "bitrix_socket_access": {
            "group": BITRIX_OPERATIONAL_EDGE_SOCKET_GROUP,
            "group_gid": identity.role_identities["business_edge"].bitrix_socket_gid,
            "read_roles": ["business_edge", "canonical_writer"],
            "mutation_role": "canonical_writer",
            "denied_roles": ["discord_edge", "gateway_observer"],
        },
        "credential_inaccessibility_contract": {
            "paths": list(PRODUCER_INACCESSIBLE_CREDENTIAL_PATHS),
            "applies_to_roles": list(ENDPOINT_ROLES),
            "unit_hash_bound": True,
            "cleanup_observer_has_no_credential_read_access": True,
        },
        "receipt_writer_group": {
            "group": PRODUCER_RECEIPT_WRITER_GROUP,
            "gid": identity.role_identities["business_edge"].receipt_writer_gid,
            "persistent_members": [],
            "runtime_service_roles": list(ENDPOINT_ROLES),
            "authority_source": "systemd_supplementary_groups",
            "run_directory_uid": trusted["receipt_contract"]["run_directory_uid"],
            "run_directory_gid": trusted["receipt_contract"]["run_directory_gid"],
            "run_directory_mode": trusted["receipt_contract"]["run_directory_mode"],
            "cross_role_precreate_is_fail_closed_dos_only": True,
            "accepted_file_owner_must_match_role_uid_gid": True,
        },
        "authority_key_lifecycle": {
            "ownership": "owner_signed_foundation",
            "durable_across_canary_runs": True,
            "retired_per_run": False,
            "source_root_inaccessible_to_service": True,
            "private_keys_projected_by_systemd": True,
            "per_run_activation_readiness_retired_after_service_stop": True,
        },
        "private_key_content_or_digest_recorded": False,
        "token_or_token_digest_recorded": False,
    }
    manifest = {
        **manifest_unsigned,
        "manifest_sha256": _sha256_json(manifest_unsigned),
    }
    receipt_contract = trusted["receipt_contract"]
    if (
        receipt_contract["run_directory_uid"] != 0
        or receipt_contract["run_directory_gid"]
        != identity.role_identities["business_edge"].receipt_writer_gid
        or receipt_contract["run_directory_mode"] != 0o3770
    ):
        _fail("producer_receipt_group_binding_invalid")
    return ProducerUnitBundle(
        revision=trusted["release_sha"],
        units=copy.deepcopy(identity.units),
        configs=configs,
        auxiliary_files=auxiliary_files,
        manifest=manifest,
    )


class RoleOwnedNativePublicationCollector:
    """Read an exact immutable role-authored binding publication."""

    def __init__(
        self,
        *,
        role: str,
        uid: int,
        gid: int,
        root: Path = DEFAULT_NATIVE_ROOT,
        partial_kinds: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        if role not in ENDPOINT_ROLES or uid < 0 or gid < 0:
            _fail("native_publication_collector_invalid")
        self.role = role
        self.uid = uid
        self.gid = gid
        self.root = root
        self.partial_kinds = {
            key: tuple(value) for key, value in (partial_kinds or {}).items()
        }

    def collect(
        self,
        *,
        slot: str,
        payload: Mapping[str, Any],
    ) -> Sequence[NativeEvidenceBinding]:
        code = "native_publication_invalid"
        run_id = payload.get("run_id")
        if (
            SLOT_ROLE.get(slot) != self.role
            or not isinstance(run_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,239}", run_id) is None
        ):
            _fail(code)
        path = self.root / self.role / run_id / f"{slot}.json"
        raw, _item = _stable_read(
            path,
            maximum=2 * 1024 * 1024,
            uid=self.uid,
            gid=self.gid,
            mode=0o400,
        )
        value = _strict_json(raw, code)
        publication = _strict(
            value,
            (
                "schema",
                "role",
                "slot",
                "run_id",
                "release_sha",
                "fixture_sha256",
                "payload",
                "payload_sha256",
                "bindings",
                "publication_sha256",
            ),
            code,
        )
        unsigned = {
            key: item
            for key, item in publication.items()
            if key != "publication_sha256"
        }
        expected_kinds = self.partial_kinds.get(slot)
        if expected_kinds is None:
            from gateway.canonical_capability_canary_producers import (
                SLOT_NATIVE_BINDING_KINDS,
            )

            expected_kinds = SLOT_NATIVE_BINDING_KINDS[slot]
        bindings_raw = publication["bindings"]
        if (
            publication["schema"] != NATIVE_PUBLICATION_SCHEMA
            or publication["role"] != self.role
            or publication["slot"] != slot
            or publication["run_id"] != run_id
            or publication["release_sha"] != payload.get("release_sha")
            or publication["fixture_sha256"] != payload.get("fixture_sha256")
            or publication["payload"] != payload
            or publication["payload_sha256"] != _sha256_json(payload)
            or publication["publication_sha256"] != _sha256_json(unsigned)
            or not isinstance(bindings_raw, list)
        ):
            _fail(code)
        bindings = tuple(
            NativeEvidenceBinding.from_mapping(item) for item in bindings_raw
        )
        if tuple(item.kind for item in bindings) != tuple(expected_kinds):
            _fail(code)
        return bindings


def publish_role_native_publication(
    *,
    role: str,
    slot: str,
    payload: Mapping[str, Any],
    bindings: Sequence[NativeEvidenceBinding],
    uid: int,
    gid: int,
    root: Path = DEFAULT_NATIVE_ROOT,
) -> Mapping[str, Any]:
    """Publish one fixed role fact bundle; never select a slot from content."""

    if (
        role not in ENDPOINT_ROLES
        or SLOT_ROLE.get(slot) != role
        or os.geteuid() != uid  # windows-footgun: ok — Linux production/canary boundary
        or os.getegid() != gid  # windows-footgun: ok — Linux production/canary boundary
        or not isinstance(payload, Mapping)
    ):
        _fail("native_publication_invalid")
    value = copy.deepcopy(dict(payload))
    run_id = value.get("run_id")
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,239}", run_id) is None
        or not isinstance(value.get("release_sha"), str)
        or _GIT_SHA_RE.fullmatch(value["release_sha"]) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(value.get("fixture_sha256"))) is None
    ):
        _fail("native_publication_invalid")
    collected = tuple(bindings)
    publication_kinds = (
        ("canonical_writer_handoff_events",)
        if slot == "bitrix_writer"
        else SLOT_NATIVE_BINDING_KINDS[slot]
    )
    if (
        any(not isinstance(item, NativeEvidenceBinding) for item in collected)
        or tuple(item.kind for item in collected) != publication_kinds
    ):
        _fail("native_publication_invalid")
    role_root = root / role
    _require_exact_directory(role_root, uid=uid, gid=gid, mode=0o700)
    run_root = role_root / run_id
    _ensure_exact_directory(run_root, uid=uid, gid=gid, mode=0o700)
    unsigned = {
        "schema": NATIVE_PUBLICATION_SCHEMA,
        "role": role,
        "slot": slot,
        "run_id": run_id,
        "release_sha": value["release_sha"],
        "fixture_sha256": value["fixture_sha256"],
        "payload": value,
        "payload_sha256": _sha256_json(value),
        "bindings": [item.to_mapping() for item in collected],
    }
    publication = {**unsigned, "publication_sha256": _sha256_json(unsigned)}
    _publish_no_replace(
        run_root / f"{slot}.json",
        _canonical_bytes(publication),
        uid=uid,
        gid=gid,
        mode=0o400,
        parent_uid=uid,
        parent_gid=gid,
        parent_mode=0o700,
    )
    return copy.deepcopy(publication)


def load_role_native_publication_payload(
    *,
    role: str,
    slot: str,
    run_id: str,
    release_sha: str,
    fixture_sha256: str,
    uid: int,
    gid: int,
    root: Path = DEFAULT_NATIVE_ROOT,
) -> Mapping[str, Any]:
    path = root / role / run_id / f"{slot}.json"
    raw, _item = _stable_read(
        path,
        maximum=2 * 1024 * 1024,
        uid=uid,
        gid=gid,
        mode=0o400,
    )
    publication = _strict_json(raw, "native_publication_invalid")
    payload = publication.get("payload")
    if not isinstance(payload, Mapping):
        _fail("native_publication_invalid")
    collector = RoleOwnedNativePublicationCollector(
        role=role,
        uid=uid,
        gid=gid,
        root=root,
        partial_kinds=(
            {"bitrix_writer": ("canonical_writer_handoff_events",)}
            if slot == "bitrix_writer"
            else None
        ),
    )
    collector.collect(slot=slot, payload=payload)
    if (
        payload.get("run_id") != run_id
        or payload.get("release_sha") != release_sha
        or payload.get("fixture_sha256") != fixture_sha256
    ):
        _fail("native_publication_invalid")
    return copy.deepcopy(dict(payload))


class FixedNativePublicationPump:
    """Wait for exact role publications and invoke only their sealed slots."""

    def __init__(
        self,
        *,
        pump: ProductionReceiptPump,
        root: Path = DEFAULT_NATIVE_ROOT,
        poll_seconds: float = 0.05,
    ) -> None:
        if (
            not isinstance(pump, ProductionReceiptPump)
            or not root.is_absolute()
            or ".." in root.parts
            or not 0.01 <= poll_seconds <= 1.0
        ):
            _fail("native_publication_pump_invalid")
        self.pump = pump
        self.root = root
        self.poll_seconds = poll_seconds

    def _wait_payload(
        self,
        slot: str,
        *,
        deadline: float,
        cancel: threading.Event | None,
    ) -> Mapping[str, Any]:
        role = SLOT_ROLE.get(slot)
        if role not in ENDPOINT_ROLES:
            _fail("native_publication_pump_invalid")
        endpoint = self.pump.foundation["endpoints"][role]
        path = self.root / role / self.pump.readiness["run_id"] / f"{slot}.json"
        while True:
            if cancel is not None and cancel.is_set():
                _fail("native_publication_pump_cancelled")
            if time.monotonic() >= deadline:
                _fail("native_publication_pump_timeout")
            if os.path.lexists(path):
                return load_role_native_publication_payload(
                    role=role,
                    slot=slot,
                    run_id=self.pump.readiness["run_id"],
                    release_sha=self.pump.readiness["release_sha"],
                    fixture_sha256=self.pump.readiness["fixture_sha256"],
                    uid=endpoint["uid"],
                    gid=endpoint["gid"],
                    root=self.root,
                )
            time.sleep(self.poll_seconds)

    def _load_payload_if_present(
        self,
        slot: str,
        *,
        cancel: threading.Event | None,
    ) -> Mapping[str, Any] | None:
        """Load one fixed publication without waiting on another slot.

        Role-owned publications are independent peers.  A canonical-order
        blocking wait on slot N can otherwise prevent a ready slot N+1 from
        receiving its receipt, even when that receipt is what lets the peer
        finish slot N.  The path table remains fixed; only the polling order is
        fair.
        """

        role = SLOT_ROLE.get(slot)
        if role not in ENDPOINT_ROLES:
            _fail("native_publication_pump_invalid")
        if cancel is not None and cancel.is_set():
            _fail("native_publication_pump_cancelled")
        endpoint = self.pump.foundation["endpoints"][role]
        path = self.root / role / self.pump.readiness["run_id"] / f"{slot}.json"
        if not os.path.lexists(path):
            return None
        return load_role_native_publication_payload(
            role=role,
            slot=slot,
            run_id=self.pump.readiness["run_id"],
            release_sha=self.pump.readiness["release_sha"],
            fixture_sha256=self.pump.readiness["fixture_sha256"],
            uid=endpoint["uid"],
            gid=endpoint["gid"],
            root=self.root,
        )

    def pump_pre_cleanup(
        self,
        *,
        deadline: float,
        cancel: threading.Event | None = None,
    ) -> Mapping[str, Mapping[str, Any]]:
        return self.pump_slots(
            tuple(
                slot
                for slot in PRODUCTION_PRE_CLEANUP_PUMP_SLOTS
                if SLOT_ROLE[slot] in ENDPOINT_ROLES
            ),
            deadline=deadline,
            cancel=cancel,
        )

    def pump_slots(
        self,
        slots: Sequence[str],
        *,
        deadline: float,
        cancel: threading.Event | None = None,
    ) -> Mapping[str, Mapping[str, Any]]:
        """Pump a canonical-order subset of fixed pre-cleanup slots."""

        if not isinstance(slots, (list, tuple)) or not slots:
            _fail("native_publication_pump_slots_invalid")
        selected = tuple(slots)
        if any(not isinstance(slot, str) for slot in selected):
            _fail("native_publication_pump_slots_invalid")
        selected_set = set(selected)
        native_slots = tuple(
            slot
            for slot in PRODUCTION_PRE_CLEANUP_PUMP_SLOTS
            if SLOT_ROLE[slot] in ENDPOINT_ROLES
        )
        canonical = tuple(slot for slot in native_slots if slot in selected_set)
        if (
            selected != canonical
            or len(selected) != len(selected_set)
            or any(slot not in native_slots for slot in selected)
        ):
            _fail("native_publication_pump_slots_invalid")
        if not isinstance(deadline, (int, float)):
            _fail("native_publication_pump_slots_invalid")
        pending = list(selected)
        produced: dict[str, Mapping[str, Any]] = {}
        while pending:
            if cancel is not None and cancel.is_set():
                _fail("native_publication_pump_cancelled")
            if time.monotonic() >= deadline:
                _fail("native_publication_pump_timeout")
            progressed = False
            for slot in tuple(pending):
                payload = self._load_payload_if_present(slot, cancel=cancel)
                if payload is None:
                    continue
                produced[slot] = self.pump.produce(slot=slot, payload=payload)
                pending.remove(slot)
                progressed = True
            if not progressed:
                time.sleep(self.poll_seconds)
        # Preserve the sealed receipt order independently of peer arrival.
        return {slot: produced[slot] for slot in selected}

    def pump_cleanup(
        self,
        *,
        deadline: float,
        cancel: threading.Event | None = None,
    ) -> Mapping[str, Any]:
        return self.pump.produce_cleanup(
            self._wait_payload("cleanup", deadline=deadline, cancel=cancel)
        )

    def pump_cleanup_payload(
        self,
        *,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Ask the live observer producer to verify and sign cleanup facts.

        Native bindings are deliberately not accepted from the root caller.
        The gateway-observer producer must derive them inside its own process
        from source-owned cleanup receipts/readbacks.
        """

        return self.pump.produce_cleanup(payload)


_CANONICAL_WRITER_CASE_BY_SLOT = {
    "workspace_writer": "workspace_continuation",
    "worker_restart_checkpoint": "workspace_continuation",
    "capability_denials": "capability_denials",
    "database_reconciliation": "database_reconciliation",
    "bitrix_writer": "bitrix_boundary",
    "discord_writer": "discord_routeback",
    "failure_writer": "failure_recovery",
}

CANONICAL_ROUTEBACK_EVENT_BINDING_SCHEMA = (
    "muncho-production-capability-canonical-routeback-event-binding.v1"
)

_CANONICAL_EVENT_FIELDS = (
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
)
_ROUTEBACK_EVENT_BINDING_FIELDS = (
    "schema",
    "event_id",
    "event_type",
    "case_id",
    "event_sha256",
    "canonical_content_sha256",
    "source_refs",
    "returned_receipt",
    "payload_receipt",
    "route_back_receipt",
)
_PRIVATE_ROUTEBACK_EVENT_BINDING_FIELDS = (
    *_ROUTEBACK_EVENT_BINDING_FIELDS,
    "payload_dispatch_attempted",
    "route_back_dispatch_attempted",
)
_PUBLIC_ROUTE_BACK_RECEIPT_FIELDS = (
    "platform",
    "adapter_receipt",
    "receipt_readback_verified",
    "message_id",
    "channel_id",
    "content_sha256",
    "public_receipt_sha256",
)
_PRIVATE_ROUTE_BACK_RECEIPT_FIELDS = (
    "private_denial_receipt_sha256",
    "dispatch_attempted",
)


def _nested_contains(value: Any, expected: str) -> bool:
    if value == expected:
        return True
    if isinstance(value, Mapping):
        return any(_nested_contains(item, expected) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_nested_contains(item, expected) for item in value)
    return False


def _exact_routeback_event(
    events: Sequence[Mapping[str, Any]],
    *,
    binding_value: Any,
    expected_event_type: str,
    expected_case_id: str,
    expected_public_target: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate one exact immutable route-back event and its signed binding."""

    code = "canonical_writer_native_evidence_invalid"
    binding_fields = (
        _PRIVATE_ROUTEBACK_EVENT_BINDING_FIELDS
        if expected_event_type == "route_back.blocked"
        else _ROUTEBACK_EVENT_BINDING_FIELDS
    )
    binding = _strict(binding_value, binding_fields, code)
    if (
        binding["schema"] != CANONICAL_ROUTEBACK_EVENT_BINDING_SCHEMA
        or binding["event_type"] != expected_event_type
        or binding["case_id"] != expected_case_id
    ):
        _fail(code)
    event_id = _safe_id(binding["event_id"], code)
    _digest(binding["event_sha256"], code)
    _digest(binding["canonical_content_sha256"], code)
    source_refs = binding["source_refs"]
    if (
        not isinstance(source_refs, Mapping)
        or not source_refs
        or any(not isinstance(key, str) for key in source_refs)
    ):
        _fail(code)
    matches = [item for item in events if item.get("event_id") == event_id]
    if len(matches) != 1:
        _fail(code)
    event = _strict(matches[0], _CANONICAL_EVENT_FIELDS, code)
    source = _strict(
        event["source"],
        ("system", "component", "source_refs", "observed_session"),
        code,
    )
    actor = _strict(event["actor"], ("type", "id"), code)
    subject = _strict(event["subject"], ("type", "id"), code)
    decision = _strict(
        event["decision"],
        ("kind", "decided_by", "keyword_authority", "attestation"),
        code,
    )
    status = _strict(
        event["status"],
        ("state", "event_type", "summary"),
        code,
    )
    if (
        event["schema_version"] != "canonical_event.v1"
        or event["event_type"] != expected_event_type
        or event["case_id"] != expected_case_id
        or not isinstance(event["occurred_at"], str)
        or not event["occurred_at"]
        or source["system"] != "hermes_agent"
        or source["component"] != "canonical_writer"
        or source["source_refs"] != source_refs
        or not isinstance(source["observed_session"], Mapping)
        or actor != {"type": "service", "id": "canonical_writer"}
        or event["evidence"] != []
        or decision["kind"] != "typed_canonical_writer_operation"
        or decision["keyword_authority"] is not False
        or decision["attestation"] != "privileged_writer_receipt"
        or status["state"] != expected_event_type
        or status["event_type"] != expected_event_type
        or event["next_action"] != {}
        or _sha256_json(event) != binding["event_sha256"]
    ):
        _fail(code)

    payload = event["payload"]
    if expected_event_type == "route_back.sent":
        payload = _strict(
            payload,
            (
                "authorization_id",
                "receipt",
                "route_back",
                "idempotency_key",
                "summary",
                "canonical_content_sha256",
            ),
            code,
        )
        route_back = _strict(
            payload["route_back"],
            ("target_ref", "receipt", "execution_binding"),
            code,
        )
        receipt = _strict(
            payload["receipt"],
            _PUBLIC_ROUTE_BACK_RECEIPT_FIELDS,
            code,
        )
        route_back_receipt = _strict(
            route_back["receipt"],
            _PUBLIC_ROUTE_BACK_RECEIPT_FIELDS,
            code,
        )
        target_ref = route_back["target_ref"]
        execution_binding = _strict(
            route_back["execution_binding"],
            ("target_channel_id", "content_sha256"),
            code,
        )
        if not isinstance(target_ref, Mapping) or not isinstance(
            expected_public_target, Mapping
        ):
            _fail(code)
        catalog_target_type = expected_public_target.get("target_type")
        if catalog_target_type != "public_channel":
            _fail(code)
        expected_target_fields = {
            "target_type": "public_guild_channel",
            "guild_id": expected_public_target.get("guild_id"),
            "channel_id": expected_public_target.get("channel_id"),
        }
        expected_target_type = "public_guild_channel"
        safety = _strict(
            event["safety"],
            (
                "secret_value_recorded",
                "payment_credential_recorded",
                "business_mutation",
                "outbound",
            ),
            code,
        )
        for field in ("content_sha256", "public_receipt_sha256"):
            _digest(receipt[field], code)
        if (
            receipt["platform"] != "discord"
            or receipt["adapter_receipt"] is not True
            or receipt["receipt_readback_verified"] is not True
            or route_back_receipt != receipt
            or binding["returned_receipt"] != receipt
            or binding["payload_receipt"] != receipt
            or binding["route_back_receipt"] != receipt
            or subject["type"] != "route_back"
            or subject["id"] != receipt["channel_id"]
            or payload["authorization_id"] != payload["idempotency_key"]
            or any(
                target_ref.get(field) != expected
                for field, expected in expected_target_fields.items()
            )
            or target_ref.get("channel_type") != expected_target_type
            or receipt["channel_id"] != expected_target_fields["channel_id"]
            or execution_binding["target_channel_id"] != receipt["channel_id"]
            or execution_binding["content_sha256"] != receipt["content_sha256"]
            or decision["decided_by"] != "routeback_finalize_sent"
            or safety
            != {
                "secret_value_recorded": False,
                "payment_credential_recorded": False,
                "business_mutation": False,
                "outbound": True,
            }
        ):
            _fail(code)
    else:
        payload = _strict(
            payload,
            (
                "preclaim",
                "preclaim_block_id",
                "target_ref",
                "blocker_reason",
                "receipt",
                "partial_receipt",
                "dispatch_attempted",
                "route_back",
                "idempotency_key",
                "summary",
                "canonical_content_sha256",
            ),
            code,
        )
        route_back = _strict(
            payload["route_back"],
            (
                "preclaim",
                "target_ref",
                "delivery_state",
                "blocker_reason",
                "receipt",
                "partial_receipt",
                "dispatch_attempted",
            ),
            code,
        )
        receipt = _strict(
            payload["receipt"],
            _PRIVATE_ROUTE_BACK_RECEIPT_FIELDS,
            code,
        )
        route_back_receipt = _strict(
            route_back["receipt"],
            _PRIVATE_ROUTE_BACK_RECEIPT_FIELDS,
            code,
        )
        safety = _strict(
            event["safety"],
            (
                "secret_value_recorded",
                "payment_credential_recorded",
                "business_mutation",
                "outbound",
                "outbound_delivery_uncertain",
                "adapter_acceptance_observed",
            ),
            code,
        )
        _digest(receipt["private_denial_receipt_sha256"], code)
        if (
            payload["preclaim"] is not True
            or payload["blocker_reason"] != "discord_dm_target_forbidden"
            or payload["dispatch_attempted"] is not False
            or payload["partial_receipt"] != receipt
            or receipt["dispatch_attempted"] is not False
            or route_back["preclaim"] is not True
            or route_back["delivery_state"] != "not_attempted"
            or route_back["blocker_reason"] != "discord_dm_target_forbidden"
            or route_back["dispatch_attempted"] is not False
            or route_back["partial_receipt"] != receipt
            or route_back_receipt != receipt
            or binding["returned_receipt"] != receipt
            or binding["payload_receipt"] != receipt
            or binding["route_back_receipt"] != receipt
            or binding["payload_dispatch_attempted"] is not False
            or binding["route_back_dispatch_attempted"] is not False
            or subject["type"] != "route_back_preclaim"
            or subject["id"] != payload["preclaim_block_id"]
            or payload["idempotency_key"] != payload["preclaim_block_id"]
            or payload["target_ref"] != route_back["target_ref"]
            or decision["decided_by"] != "routeback_preclaim_blocked"
            or safety
            != {
                "secret_value_recorded": False,
                "payment_credential_recorded": False,
                "business_mutation": False,
                "outbound": False,
                "outbound_delivery_uncertain": False,
                "adapter_acceptance_observed": False,
            }
        ):
            _fail(code)
    if (
        payload["canonical_content_sha256"] != binding["canonical_content_sha256"]
        or not isinstance(payload["summary"], str)
        or not payload["summary"]
        or status["summary"] != payload["summary"][:500]
        or not isinstance(payload["idempotency_key"], str)
        or not payload["idempotency_key"]
    ):
        _fail(code)
    return copy.deepcopy(event)


def _exact_terminal_event(
    events: Sequence[Mapping[str, Any]],
    *,
    terminal: Mapping[str, Any],
    expected_case_id: str,
) -> Mapping[str, Any]:
    code = "canonical_writer_native_evidence_invalid"
    event_id = terminal.get("terminal_event_id")
    event_sha256 = terminal.get("terminal_event_sha256")
    if not isinstance(event_id, str) or not isinstance(event_sha256, str):
        _fail(code)
    _safe_id(event_id, code)
    _digest(event_sha256, code)
    matches = [item for item in events if item.get("event_id") == event_id]
    if len(matches) != 1 or matches[0].get("case_id") != expected_case_id:
        _fail(code)
    event = matches[0]
    payload = event.get("payload")
    observed_sha256 = event.get("content_sha256")
    if isinstance(payload, Mapping):
        observed_sha256 = payload.get("canonical_content_sha256")
    if observed_sha256 != event_sha256:
        _fail(code)
    return copy.deepcopy(dict(event))


class CanonicalWriterProjectionNativeCollector:
    """Read exact case events through the peer-authenticated writer socket."""

    def __init__(
        self,
        *,
        client: Any,
        catalog: Mapping[str, Any],
        release_sha: str,
        capability_plan_sha256: str,
        full_canary_plan_sha256: str,
        source_identity: Mapping[str, Any],
        maximum_pages: int = 16,
    ) -> None:
        if not callable(getattr(client, "call", None)):
            _fail("canonical_writer_native_collector_invalid")
        trusted = validate_probe_catalog(catalog)
        if (
            trusted["release_sha"] != release_sha
            or trusted["capability_plan_sha256"] != capability_plan_sha256
            or trusted["full_canary_plan_sha256"] != full_canary_plan_sha256
            or not isinstance(source_identity, Mapping)
            or not 1 <= maximum_pages <= 32
        ):
            _fail("canonical_writer_native_collector_invalid")
        self.client = client
        self.catalog = trusted
        self.release_sha = release_sha
        self.source_identity = copy.deepcopy(dict(source_identity))
        self.maximum_pages = maximum_pages

    @staticmethod
    def _required_values(slot: str, payload: Mapping[str, Any]) -> tuple[str, ...]:
        required: list[Any] = []
        terminal = payload.get("terminal_ctw")
        if isinstance(terminal, Mapping):
            required.extend((
                terminal.get("terminal_event_id"),
                terminal.get("terminal_event_sha256"),
            ))
        if slot == "workspace_writer":
            required.extend((
                payload.get("owner_grant_id"),
                payload.get("owner_grant_sha256"),
            ))
        elif slot == "worker_restart_checkpoint":
            required.extend((
                payload.get("checkpoint_event_id"),
                payload.get("checkpoint_event_sha256"),
            ))
        elif slot == "capability_denials":
            denials = payload.get("denials")
            if not isinstance(denials, list):
                _fail("canonical_writer_native_evidence_invalid")
            required.extend(
                item.get("receipt_sha256")
                for item in denials
                if isinstance(item, Mapping)
            )
            if len(required) != len(denials):
                _fail("canonical_writer_native_evidence_invalid")
        elif slot == "database_reconciliation":
            required.extend(
                payload.get(name)
                for name in (
                    "read_receipt_sha256",
                    "transaction_receipt_sha256",
                    "readback_receipt_sha256",
                )
            )
        elif slot == "bitrix_writer":
            required.extend(
                payload.get(name)
                for name in (
                    "selection_event_id",
                    "selection_event_sha256",
                    "blocked_event_id",
                    "blocked_receipt_sha256",
                    "mutation_denial_receipt_sha256",
                )
            )
        elif slot == "discord_writer":
            for name in ("sent_event", "blocked_event"):
                binding = payload.get(name)
                if not isinstance(binding, Mapping):
                    _fail("canonical_writer_native_evidence_invalid")
                required.extend((binding.get("event_id"), binding.get("event_sha256")))
        if (
            not required
            or any(
                not isinstance(item, str)
                or not item
                or len(item.encode("utf-8", errors="strict")) > 512
                for item in required
            )
            or len(required) != len(set(required))
        ):
            _fail("canonical_writer_native_evidence_invalid")
        return tuple(required)

    def _read_projection(
        self,
        *,
        case_id: str,
    ) -> tuple[list[Mapping[str, Any]], list[str]]:
        events: list[Mapping[str, Any]] = []
        request_ids: list[str] = []
        after_event_id = ""
        for _page in range(self.maximum_pages):
            call = self.client.call(
                "projection.read_events",
                {
                    "case_id": case_id,
                    "after_event_id": after_event_id,
                    "limit": 500,
                },
                runtime={"platform": "capability-canary-producer"},
            )
            request_id = getattr(call, "request_id", None)
            result = getattr(call, "result", None)
            if (
                not isinstance(request_id, str)
                or not request_id
                or not isinstance(result, Mapping)
                or set(result) - {"events", "has_more"}
                or not isinstance(result.get("events"), list)
            ):
                _fail("canonical_writer_native_evidence_invalid")
            rows = result["events"]
            if len(rows) > 500:
                _fail("canonical_writer_native_evidence_invalid")
            for row in rows:
                if (
                    not isinstance(row, Mapping)
                    or row.get("case_id") != case_id
                    or not isinstance(row.get("event_id"), str)
                    or not row["event_id"]
                    or any(prior.get("event_id") == row["event_id"] for prior in events)
                ):
                    _fail("canonical_writer_native_evidence_invalid")
                events.append(copy.deepcopy(dict(row)))
            request_ids.append(request_id)
            has_more = result.get("has_more")
            if has_more not in {None, True, False}:
                _fail("canonical_writer_native_evidence_invalid")
            if not rows or has_more is False or (has_more is None and len(rows) < 500):
                return events, request_ids
            after_event_id = str(rows[-1]["event_id"])
        _fail("canonical_writer_native_evidence_page_limit")

    def collect(
        self,
        *,
        slot: str,
        payload: Mapping[str, Any],
    ) -> Sequence[NativeEvidenceBinding]:
        code = "canonical_writer_native_evidence_invalid"
        objective = _CANONICAL_WRITER_CASE_BY_SLOT.get(slot)
        if objective is None or SLOT_ROLE.get(slot) != "canonical_writer":
            _fail(code)
        if (
            payload.get("run_id") != self.catalog["run_id"]
            or payload.get("release_sha") != self.release_sha
            or payload.get("fixture_sha256") != self.catalog["fixture_sha256"]
        ):
            _fail(code)
        case_id = self.catalog["case_ids"][objective]
        terminal = payload.get("terminal_ctw")
        if isinstance(terminal, Mapping) and terminal.get("case_id") != case_id:
            _fail(code)
        required = self._required_values(slot, payload)
        events, request_ids = self._read_projection(case_id=case_id)
        if not events:
            _fail(code)
        if slot == "discord_writer":
            terminal_event = _exact_terminal_event(
                events,
                terminal=payload["terminal_ctw"],
                expected_case_id=case_id,
            )
            sent_event = _exact_routeback_event(
                events,
                binding_value=payload.get("sent_event"),
                expected_event_type="route_back.sent",
                expected_case_id=case_id,
                expected_public_target=self.catalog["discord"]["public_target"],
            )
            blocked_event = _exact_routeback_event(
                events,
                binding_value=payload.get("blocked_event"),
                expected_event_type="route_back.blocked",
                expected_case_id=case_id,
            )
            projection = {
                "case_id": case_id,
                "terminal_event": terminal_event,
                "sent_event": sent_event,
                "blocked_event": blocked_event,
            }
        else:
            if any(not _nested_contains(events, expected) for expected in required):
                _fail(code)
            projection = {"case_id": case_id, "events": events}
        projection_sha256 = _sha256_json(projection)
        source_sha256 = _sha256_json(self.source_identity)
        verification = _sha256_json({
            "case_id": case_id,
            "payload_sha256": _sha256_json(payload),
            "projection_sha256": projection_sha256,
            "request_ids": request_ids,
            "required_values": list(required),
        })
        event_kind = {
            "workspace_writer": "canonical_writer_projection_events",
            "worker_restart_checkpoint": "canonical_writer_checkpoint_event",
            "capability_denials": "canonical_writer_capability_events",
            "database_reconciliation": "canonical_writer_database_events",
            "bitrix_writer": "canonical_writer_handoff_events",
            "discord_writer": "canonical_writer_routeback_events",
            "failure_writer": "canonical_writer_failure_events",
        }[slot]
        event_binding = NativeEvidenceBinding(
            kind=event_kind,
            source_identity_sha256=source_sha256,
            artifact_sha256=projection_sha256,
            verification_receipt_sha256=verification,
        )
        if slot == "workspace_writer":
            terminal_event_id = payload["terminal_ctw"]["terminal_event_id"]
            terminal_event = next(
                item for item in events if item.get("event_id") == terminal_event_id
            )
            return (
                NativeEvidenceBinding(
                    kind="canonical_writer_resume_bundle",
                    source_identity_sha256=source_sha256,
                    artifact_sha256=_sha256_json(terminal_event),
                    verification_receipt_sha256=verification,
                ),
                event_binding,
            )
        if slot == "database_reconciliation":
            return (
                event_binding,
                NativeEvidenceBinding(
                    kind="database_live_readback",
                    source_identity_sha256=source_sha256,
                    artifact_sha256=_digest(
                        payload.get("readback_receipt_sha256"), code
                    ),
                    verification_receipt_sha256=verification,
                ),
            )
        return (event_binding,)


class DiscordEdgeNativeCollector:
    """Reconcile one public edge send and prove the DM path stayed local."""

    def __init__(
        self,
        *,
        edge_client: Any,
        history_client: Any,
        edge_public_key: Any,
        catalog: Mapping[str, Any],
        contract: Mapping[str, Any],
    ) -> None:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
        from gateway.canonical_capability_canary_producers import (
            validate_discord_edge_evidence_contract,
        )

        trusted_catalog = validate_probe_catalog(catalog)
        trusted_contract = validate_discord_edge_evidence_contract(contract)
        if (
            not callable(getattr(edge_client, "reconcile", None))
            or not callable(getattr(history_client, "read", None))
            or not isinstance(edge_public_key, Ed25519PublicKey)
        ):
            _fail("discord_native_collector_invalid")
        self.edge_client = edge_client
        self.history_client = history_client
        self.edge_public_key = edge_public_key
        self.catalog = trusted_catalog
        self.contract = trusted_contract

    @staticmethod
    def _public_target(value: Mapping[str, Any]) -> Mapping[str, Any]:
        if value.get("target_type") != "public_channel":
            _fail("discord_native_evidence_invalid")
        return {
            "target_type": "public_guild_channel",
            "guild_id": value.get("guild_id"),
            "channel_id": value.get("channel_id"),
        }

    def collect(
        self,
        *,
        slot: str,
        payload: Mapping[str, Any],
    ) -> Sequence[NativeEvidenceBinding]:
        from gateway.discord_edge_protocol import (
            DiscordEdgeErrorCode,
            DiscordEdgeOperation,
            DiscordEdgeProtocolError,
            DiscordEdgeReceiptOutcome,
            DiscordEdgeReconciliationQuery,
            DiscordPublicTarget,
            verify_receipt,
        )

        code = "discord_native_evidence_invalid"
        if (
            slot != "discord_edge"
            or payload.get("run_id") != self.catalog["run_id"]
            or payload.get("release_sha") != self.catalog["release_sha"]
            or payload.get("fixture_sha256") != self.catalog["fixture_sha256"]
        ):
            _fail(code)
        catalog_discord = self.catalog["discord"]
        target_value = catalog_discord["public_target"]
        if any(
            payload.get(name) != target_value.get(name)
            for name in ("target_type", "guild_id", "channel_id")
        ):
            _fail(code)
        idempotency_key = catalog_discord["public_idempotency_key"]
        if payload.get("idempotency_key_sha256") != _sha256_bytes(
            idempotency_key.encode("utf-8")
        ):
            _fail(code)
        target = DiscordPublicTarget.from_mapping(self._public_target(target_value))
        query = DiscordEdgeReconciliationQuery(
            idempotency_key=idempotency_key,
            operation=DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
            target=target,
            request_sha256=_digest(payload.get("request_sha256"), code),
            content_sha256=_digest(payload.get("content_sha256"), code),
        )
        first = self.edge_client.reconcile(query, require_preconnected=False)
        receipt = verify_receipt(
            first.receipt,
            self.edge_public_key,
            expected_request=first.request,
            now_unix_ms=payload.get("observed_at_unix_ms"),
        )
        envelope = first.receipt.to_message()
        envelope_sha256 = _sha256_json(envelope)
        if (
            first.state != "verified"
            or first.replayed is not True
            or receipt.outcome is not DiscordEdgeReceiptOutcome.VERIFIED
            or receipt.discord_object_id != payload.get("platform_message_id")
            or receipt.bot_user_id != payload.get("routeback_bot_user_id")
            or receipt.adapter_accepted is not True
            or receipt.readback_verified is not True
            or receipt.content_sha256 != payload.get("content_sha256")
            or envelope_sha256 != payload.get("public_receipt_sha256")
        ):
            _fail(code)
        # This trusted synthetic collector is bound to the exact canary user;
        # it does not reuse Discord delivery/origin metadata as a sender.
        from gateway.discord_history_authority import CANARY_REQUESTER_USER_ID
        from gateway.session_context import clear_session_vars, set_session_vars

        history_tokens = set_session_vars(
            platform="discord",
            user_id=CANARY_REQUESTER_USER_ID,
        )
        try:
            history = self.history_client.read(
                channel_id=str(payload["channel_id"]),
                limit=25,
            )
        finally:
            clear_session_vars(history_tokens)
        messages = history.get("messages") if isinstance(history, Mapping) else None
        if not isinstance(messages, list):
            _fail(code)
        matches = [
            item
            for item in messages
            if isinstance(item, Mapping)
            and item.get("message_id") == payload.get("platform_message_id")
        ]
        if len(matches) != 1:
            _fail(code)
        message = matches[0]
        if (
            message.get("author_id") != payload.get("routeback_bot_user_id")
            or message.get("author_is_bot") is not True
            or message.get("content_truncated") is not False
            or not isinstance(message.get("content"), str)
            or _sha256_bytes(message["content"].encode("utf-8"))
            != payload.get("content_sha256")
        ):
            _fail(code)

        private_probe = {
            "target_type": catalog_discord["private_target_kind"],
            "guild_id": target.guild_id,
            "channel_id": target.channel_id,
        }
        try:
            DiscordPublicTarget.from_mapping(private_probe)
        except DiscordEdgeProtocolError as exc:
            if exc.code is not DiscordEdgeErrorCode.FORBIDDEN_TARGET:
                _fail(code)
        else:
            _fail(code)
        private_denial = {
            "probe_id": catalog_discord["private_probe_id"],
            "target_kind": "dm",
            "blocker_code": DiscordEdgeErrorCode.FORBIDDEN_TARGET.value,
            "dispatch_attempted": False,
        }
        private_denial_sha256 = _sha256_json(private_denial)
        if (
            payload.get("private_target_kind") != "dm"
            or payload.get("private_dispatch_attempted") is not False
            or payload.get("journal_unchanged_after_private_probe") is not True
            or payload.get("private_denial_receipt_sha256") != private_denial_sha256
        ):
            _fail(code)
        second = self.edge_client.reconcile(query, require_preconnected=False)
        second_envelope = second.receipt.to_message()
        if (
            second.state != first.state
            or second.replayed is not True
            or second.request.to_message() != first.request.to_message()
            or second_envelope != envelope
        ):
            _fail(code)
        edge_source = _sha256_json({
            "service_unit": self.contract["edge_service_unit"],
            "socket_path": self.contract["edge_socket_path"],
            "service_uid": self.contract["edge_service_uid"],
            "receipt_public_key_id": self.contract["receipt_public_key_id"],
            "peer_authorization": "exact_current_systemd_main_pid_each_call",
        })
        connector_source = _sha256_json({
            "service_unit": self.contract["connector_service_unit"],
            "socket_path": self.contract["connector_socket_path"],
            "service_uid": self.contract["connector_service_uid"],
            "operation": self.contract["public_history_operation"],
            "peer_authorization": "exact_current_systemd_main_pid_each_call",
        })
        reconciliation_sha256 = _sha256_json({
            "request": first.request.to_message(),
            "receipt": envelope,
            "second_receipt": second_envelope,
        })
        history_sha256 = _sha256_json(history)
        routeback_identity = {
            "bot_user_id": receipt.bot_user_id,
            "receipt_public_key_id": self.contract["receipt_public_key_id"],
            "edge_service_unit": self.contract["edge_service_unit"],
        }
        return (
            NativeEvidenceBinding(
                kind="discord_edge_signed_receipt",
                source_identity_sha256=edge_source,
                artifact_sha256=envelope_sha256,
                verification_receipt_sha256=_sha256_json({
                    "verified_receipt_id": receipt.receipt_id,
                    "outcome": receipt.outcome.value,
                    "request_sha256": receipt.request_sha256,
                }),
            ),
            NativeEvidenceBinding(
                kind="discord_edge_journal_readback",
                source_identity_sha256=edge_source,
                artifact_sha256=reconciliation_sha256,
                verification_receipt_sha256=_sha256_json({
                    "replayed": first.replayed,
                    "stable_after_dm_probe": True,
                }),
            ),
            NativeEvidenceBinding(
                kind="discord_public_readback",
                source_identity_sha256=connector_source,
                artifact_sha256=history_sha256,
                verification_receipt_sha256=_sha256_json(message),
            ),
            NativeEvidenceBinding(
                kind="discord_private_predispatch_denial",
                source_identity_sha256=edge_source,
                artifact_sha256=private_denial_sha256,
                verification_receipt_sha256=reconciliation_sha256,
            ),
            NativeEvidenceBinding(
                kind="routeback_bot_identity",
                source_identity_sha256=edge_source,
                artifact_sha256=_sha256_json(routeback_identity),
                verification_receipt_sha256=envelope_sha256,
            ),
        )


_CLEANUP_CREDENTIAL_BINDINGS = (
    "api_control",
    "bitrix_operational_edge_webhook",
    "discord_canonical_routeback_bot_token",
    "discord_public_session_bot_token",
    "mac_ops_gitlab",
    "openai_codex",
)

_CLEANUP_OBSERVER_UNIT = PRODUCER_SERVICE_UNITS["gateway_observer"]
_CLEANUP_NON_OBSERVER_SERVICE_UNITS = (
    "hermes-cloud-gateway.service",
    *(
        PRODUCER_SERVICE_UNITS[role]
        for role in reversed(ENDPOINT_ROLES)
        if role != "gateway_observer"
    ),
    "muncho-operational-edge-bitrix.service",
    "muncho-canonical-writer.service",
    "muncho-capability-browser.service",
    "muncho-isolated-worker.service",
    "muncho-isolated-worker.socket",
    "muncho-mac-ops-edge.service",
    "muncho-discord-connector.service",
    "muncho-discord-egress.service",
    "muncho-canonical-writer-phase-b-readiness.service",
)

_CLEANUP_DISCORD_CREDENTIAL_TOPOLOGY = {
    "connector_service_unit": "muncho-discord-connector.service",
    "connector_credential_lease": "discord_public_session_bot_token",
    "connector_credential_scope": "ordinary_public_ingress_and_session_replies",
    "routeback_service_unit": "muncho-discord-egress.service",
    "routeback_credential_lease": "discord_canonical_routeback_bot_token",
    "routeback_credential_scope": "canonical_public_routeback_only",
}


def _cleanup_role_identities(
    foundation: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    """Reconstruct the exact unit inputs from the signed foundation only."""

    code = "gateway_observer_cleanup_evidence_invalid"
    endpoints = foundation.get("endpoints")
    receipt = foundation.get("receipt_contract")
    bitrix = foundation.get("bitrix_operational_edge_contract")
    if not isinstance(endpoints, Mapping) or not isinstance(receipt, Mapping):
        _fail(code)
    try:
        receipt_gid = int(receipt["run_directory_gid"])
        bitrix_gid = int(bitrix["identity_bootstrap"]["socket_client_gid"])
    except (KeyError, TypeError, ValueError):
        _fail(code)
    result: dict[str, Mapping[str, Any]] = {}
    for role, (user, group) in PRODUCER_ROLE_ACCOUNTS.items():
        endpoint = endpoints.get(role)
        if not isinstance(endpoint, Mapping):
            _fail(code)
        result[role] = {
            "user": user,
            "group": group,
            "uid": endpoint.get("uid"),
            "gid": endpoint.get("gid"),
            "receipt_writer_gid": receipt_gid,
            "bitrix_socket_gid": (
                bitrix_gid if role in {"business_edge", "canonical_writer"} else None
            ),
        }
    _role_identities(result)
    return result


def _cleanup_service_state(unit: str) -> Mapping[str, Any]:
    from gateway.canonical_capability_canary_runtime import (
        collect_capability_service_state,
    )

    return collect_capability_service_state(unit)


def _cleanup_service_stopped(value: Mapping[str, Any]) -> bool:
    return (
        value.get("LoadState") in {"loaded", "not-found"}
        and value.get("ActiveState") in {"inactive", "failed"}
        and value.get("MainPID") == 0
        and value.get("UnitFileState") in {"disabled", ""}
        and value.get("DropInPaths") in {"", "[]"}
    )


def _cleanup_observer_live(value: Mapping[str, Any]) -> bool:
    return (
        value.get("LoadState") == "loaded"
        and value.get("ActiveState") == "active"
        and value.get("SubState") == "running"
        and type(value.get("MainPID")) is int
        and value["MainPID"] > 1
        and value.get("FragmentPath") == f"/etc/systemd/system/{_CLEANUP_OBSERVER_UNIT}"
        and value.get("UnitFileState") in {"disabled", ""}
        and value.get("DropInPaths") in {"", "[]"}
        and value.get("Type") == "simple"
        and value.get("NotifyAccess") in {"", "none"}
    )


def _read_cleanup_production_diff(
    run_root: Path,
    observer_gid: int,
) -> Mapping[str, Any]:
    raw, _item = _stable_read(
        _production_diff_observation_path(run_root),
        maximum=2 * 1024 * 1024,
        uid=0,
        gid=observer_gid,
        mode=0o440,
    )
    return _strict_json(raw, "gateway_observer_cleanup_evidence_invalid")


def _read_cleanup_facts(
    run_root: Path,
    observer_gid: int,
) -> tuple[Mapping[str, Any], str]:
    raw, _item = _stable_read(
        run_root / "cleanup-facts.json",
        maximum=2 * 1024 * 1024,
        uid=0,
        gid=observer_gid,
        mode=0o440,
    )
    return (
        _strict_json(raw, "gateway_observer_cleanup_evidence_invalid"),
        _sha256_bytes(raw),
    )


_OBSERVER_FRAME_EVENTS = (
    "plugin_ready",
    "api_session_bound",
    "private_target_probe_ready",
    "private_target_probe_result",
    "pre_api_request",
    "post_api_request",
    "post_tool_call",
    "canonical_case_readback",
    "session_end",
)
_OBSERVER_RUNTIME_EVENTS = (
    "plugin_ready",
    "api_session_bound",
    "private_target_probe_ready",
    "private_target_probe_result",
)
_OBSERVER_WORKSPACE_EVENTS = (
    "pre_api_request",
    "post_api_request",
    "post_tool_call",
    "canonical_case_readback",
    "session_end",
)
_OBSERVER_FAILURE_EVENTS = (
    "post_tool_call",
    "canonical_case_readback",
    "session_end",
)
_OBSERVER_FRAME_CHAIN_SCHEMA = "muncho-canary-evidence-chain.v1"
_OBSERVER_ZERO_CHAIN_SHA256 = "0" * 64
_OBSERVER_WORKSPACE_PROPOSAL_CORE_FIELDS = (
    "session_id",
    "capability_epoch_sha256",
    "task_workspace_evidence_sha256s",
    "first_path_failure_receipt_sha256",
    "alternate_read_receipt_sha256",
    "model_requested_effort",
    "later_request_effort",
    "reasoning_tool_call_id",
    "restart_count",
    "used_command_sha256s",
    "mutation_receipt_sha256s",
    "approval_prompt_count",
    "microapproval_prompt_count",
    "replayed_mutation_count",
    "owner_grant_id",
    "owner_grant_sha256",
    "consumed_command_sha256s",
    "terminal_plan_id",
    "terminal_plan_revision",
)
_OBSERVER_FAILURE_PROPOSAL_CORE_FIELDS = (
    "failures",
    "model_retained_tool_control",
)


def _recursive_digest_index(value: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        values.add(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            values.update(_recursive_digest_index(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            values.update(_recursive_digest_index(item))
    return values


def _projection_has_forbidden_source_field(value: Any) -> bool:
    forbidden = {
        "bindings",
        "native_evidence",
        "success",
        "outcome",
        "changed",
        "changed_surfaces",
        "surface_diffs",
        "unexpected_change_count",
        "unexpected_changes",
        "production_mutation_observed",
        "content",
        "prompt",
        "response",
        "model_output",
        "model_reasoning",
        "task_prose",
        "job_prose",
        "semantic_job_content",
    }
    if isinstance(value, Mapping):
        return any(
            key in forbidden or _projection_has_forbidden_source_field(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_projection_has_forbidden_source_field(item) for item in value)
    return False


def build_api_terminal_event_identity(conversation: Any) -> Mapping[str, Any]:
    """Project authenticated SSE identities without response/task prose."""

    code = "gateway_observer_source_projection_invalid"
    try:
        events = tuple(conversation.events)
        event_records = [
            {
                "ordinal": index,
                "event": name,
                "payload_sha256": _sha256_json(payload),
            }
            for index, (name, payload) in enumerate(events, start=1)
        ]
        unsigned = {
            "schema": "muncho-capability-canary-api-terminal-identity.v1",
            "session_id": conversation.session_id,
            "session_create_request_id": conversation.session_create_request_id,
            "chat_stream_request_id": conversation.chat_stream_request_id,
            "api_run_id": conversation.api_run_id,
            "api_message_id": conversation.api_message_id,
            "event_records": event_records,
            "event_chain_sha256": _sha256_json(event_records),
            "assistant_completed_sha256": _sha256_json(
                conversation.assistant_completed
            ),
            "run_completed_sha256": _sha256_json(conversation.run_completed),
            "transcript_sha256": _sha256_json({
                "session_id": conversation.session_id,
                "event_records": event_records,
            }),
            "observed_at_unix_ms": conversation.observed_at_unix_ms,
            "completed_at_unix_ms": conversation.completed_at_unix_ms,
            "response_or_task_prose_recorded": False,
            "secret_material_recorded": False,
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise CapabilityProducerError(code) from exc
    for field in (
        "session_id",
        "session_create_request_id",
        "chat_stream_request_id",
        "api_run_id",
        "api_message_id",
    ):
        if (
            not isinstance(unsigned[field], str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", unsigned[field])
            is None
        ):
            _fail(code)
    if (
        not event_records
        or any(
            not isinstance(row["event"], str)
            or re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", row["event"]) is None
            for row in event_records
        )
        or type(unsigned["observed_at_unix_ms"]) is not int
        or type(unsigned["completed_at_unix_ms"]) is not int
        or not 0 < unsigned["observed_at_unix_ms"] <= unsigned["completed_at_unix_ms"]
    ):
        _fail(code)
    return {**unsigned, "identity_sha256": _sha256_json(unsigned)}


def validate_gateway_observer_source_projection(
    value: Any,
    *,
    release_sha: str,
    capability_plan_sha256: str,
    full_canary_plan_sha256: str,
    fixture_sha256: str,
    run_id: str,
) -> Mapping[str, Any]:
    code = "gateway_observer_source_projection_invalid"
    fields = {
        "schema",
        "run_id",
        "release_sha",
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "fixture_sha256",
        "source_canary_run_id",
        "source_fixture_sha256",
        "observer_activation_identity",
        "collector_readiness_identity",
        "runtime_source_identity",
        "model_proposal_core_identities",
        "goal_continuation_identity",
        "frame_records",
        "frame_chain_head_sha256",
        "source_digest_index",
        "reasoning_efforts",
        "tool_call_records",
        "worker_restart_receipt",
        "api_terminal_event_identity",
        "slot_membership",
        "source_projection_only",
        "native_evidence_bindings_recorded",
        "semantic_task_prose_recorded",
        "success_or_outcome_recorded",
        "secret_material_recorded",
        "secret_digest_recorded",
        "observed_at_unix_ms",
        "projection_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail(code)
    raw = copy.deepcopy(dict(value))
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in raw.items()
        if key != "projection_sha256"
    }
    if (
        raw["schema"] != GATEWAY_OBSERVER_SOURCE_PROJECTION_SCHEMA
        or raw["run_id"] != run_id
        or raw["release_sha"] != release_sha
        or raw["capability_plan_sha256"] != capability_plan_sha256
        or raw["full_canary_plan_sha256"] != full_canary_plan_sha256
        or raw["fixture_sha256"] != fixture_sha256
        or not isinstance(raw["source_canary_run_id"], str)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}",
            raw["source_canary_run_id"],
        )
        is None
        or _digest(raw["source_fixture_sha256"], code) != raw["source_fixture_sha256"]
        or raw["source_projection_only"] is not True
        or raw["native_evidence_bindings_recorded"] is not False
        or raw["semantic_task_prose_recorded"] is not False
        or raw["success_or_outcome_recorded"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or type(raw["observed_at_unix_ms"]) is not int
        or raw["observed_at_unix_ms"] <= 0
        or raw["projection_sha256"] != _sha256_json(unsigned)
        or _projection_has_forbidden_source_field({
            key: item
            for key, item in raw.items()
            if key
            not in {
                "success_or_outcome_recorded",
                "semantic_task_prose_recorded",
            }
        })
    ):
        _fail(code)
    proposal_identities = _strict(
        raw["model_proposal_core_identities"],
        ("workspace_gateway", "failure_gateway"),
        code,
    )
    for expected_slot in ("workspace_gateway", "failure_gateway"):
        identity = _strict(
            proposal_identities[expected_slot],
            (
                "schema",
                "slot",
                "proposal_event_sha256",
                "proposal_event_id_sha256",
                "proposal_case_id_sha256",
                "model_tool_frame_sha256",
                "model_tool_call_id_sha256",
                "model_tool_result_sha256",
                "core_sha256",
                "identity_sha256",
            ),
            code,
        )
        identity_unsigned = {
            key: copy.deepcopy(item)
            for key, item in identity.items()
            if key != "identity_sha256"
        }
        if (
            identity["schema"] != GATEWAY_OBSERVER_PROPOSAL_IDENTITY_SCHEMA
            or identity["slot"] != expected_slot
            or identity["identity_sha256"] != _sha256_json(identity_unsigned)
        ):
            _fail(code)
        for name in (
            "proposal_event_sha256",
            "proposal_event_id_sha256",
            "proposal_case_id_sha256",
            "model_tool_frame_sha256",
            "model_tool_call_id_sha256",
            "model_tool_result_sha256",
            "core_sha256",
        ):
            _digest(identity[name], code)
    if any(
        proposal_identities["workspace_gateway"][name]
        != proposal_identities["failure_gateway"][name]
        for name in (
            "proposal_event_sha256",
            "proposal_event_id_sha256",
            "proposal_case_id_sha256",
            "model_tool_frame_sha256",
            "model_tool_call_id_sha256",
            "model_tool_result_sha256",
        )
    ):
        _fail(code)
    goal_identity = _strict(
        raw["goal_continuation_identity"],
        (
            "schema",
            "evidence_sha256",
            "terminal_sha256",
            "discord_ingress_receipt_sha256",
            "model_verdict_receipt_sha256s",
            "gateway_restart_receipt_sha256",
            "ctw_recovery_receipt_sha256",
            "model_route_receipt_sha256",
            "prompt_tool_stability_receipt_sha256",
            "user_preemption_queue_receipt_sha256",
            "production_diff_sha256",
            "completed_at_unix_ms",
            "identity_sha256",
        ),
        code,
    )
    goal_identity_unsigned = {
        key: copy.deepcopy(item)
        for key, item in goal_identity.items()
        if key != "identity_sha256"
    }
    verdict_receipts = goal_identity["model_verdict_receipt_sha256s"]
    if (
        goal_identity["schema"] != GOAL_CONTINUATION_NATIVE_IDENTITY_SCHEMA
        or goal_identity["identity_sha256"] != _sha256_json(goal_identity_unsigned)
        or not isinstance(verdict_receipts, list)
        or len(verdict_receipts) < 3
        or len(verdict_receipts) != len(set(verdict_receipts))
        or type(goal_identity["completed_at_unix_ms"]) is not int
        or goal_identity["completed_at_unix_ms"] <= 0
    ):
        _fail(code)
    for value in (
        goal_identity["evidence_sha256"],
        goal_identity["terminal_sha256"],
        goal_identity["discord_ingress_receipt_sha256"],
        *verdict_receipts,
        goal_identity["gateway_restart_receipt_sha256"],
        goal_identity["ctw_recovery_receipt_sha256"],
        goal_identity["model_route_receipt_sha256"],
        goal_identity["prompt_tool_stability_receipt_sha256"],
        goal_identity["user_preemption_queue_receipt_sha256"],
        goal_identity["production_diff_sha256"],
    ):
        _digest(value, code)
    readiness = _strict(
        raw["collector_readiness_identity"],
        (
            "receipt_sha256",
            "service_identity_sha256",
            "edge_service_identity_sha256",
            "collector_socket_sha256",
        ),
        code,
    )
    observer_activation = _strict(
        raw["observer_activation_identity"],
        (
            "producer_readiness_sha256",
            "observer_endpoint_readiness_sha256",
            "observer_main_pid",
            "observer_service_unit",
            "observer_service_identity_sha256",
            "observer_uid",
            "observer_gid",
            "identity_sha256",
        ),
        code,
    )
    observer_activation_unsigned = {
        key: copy.deepcopy(item)
        for key, item in observer_activation.items()
        if key != "identity_sha256"
    }
    if (
        type(observer_activation["observer_main_pid"]) is not int
        or observer_activation["observer_main_pid"] < 2
        or not isinstance(observer_activation["observer_service_unit"], str)
        or not observer_activation["observer_service_unit"].endswith(".service")
        or type(observer_activation["observer_uid"]) is not int
        or type(observer_activation["observer_gid"]) is not int
        or observer_activation["observer_uid"] <= 0
        or observer_activation["observer_gid"] <= 0
        or observer_activation["identity_sha256"]
        != _sha256_json(observer_activation_unsigned)
    ):
        _fail(code)
    for name in (
        "producer_readiness_sha256",
        "observer_endpoint_readiness_sha256",
        "observer_service_identity_sha256",
    ):
        _digest(observer_activation[name], code)
    runtime_source = _strict(
        raw["runtime_source_identity"],
        (
            "gateway_process_identity_sha256",
            "discord_connector_readiness_sha256",
            "connector_bot_user_id",
            "connector_bot_user_id_provenance",
        ),
        code,
    )
    for item in (
        *readiness.values(),
        runtime_source["gateway_process_identity_sha256"],
        runtime_source["discord_connector_readiness_sha256"],
    ):
        if not isinstance(item, str):
            _fail(code)
    for item in readiness.values():
        _digest(item, code)
    _digest(runtime_source["gateway_process_identity_sha256"], code)
    _digest(runtime_source["discord_connector_readiness_sha256"], code)
    if (
        not isinstance(runtime_source["connector_bot_user_id"], str)
        or re.fullmatch(r"[1-9][0-9]{5,24}", runtime_source["connector_bot_user_id"])
        is None
        or runtime_source["connector_bot_user_id_provenance"]
        != "discord_gateway_ready_user_id"
    ):
        _fail(code)

    frames = raw["frame_records"]
    if not isinstance(frames, list) or not frames:
        _fail(code)
    previous = _OBSERVER_ZERO_CHAIN_SHA256
    source_digests: set[str] = set()
    effort_records: list[Mapping[str, Any]] = []
    tool_records: list[Mapping[str, Any]] = []
    event_sequences: dict[str, list[int]] = {
        event: [] for event in _OBSERVER_FRAME_EVENTS
    }
    last_observed = 0
    for expected_sequence, item in enumerate(frames, start=1):
        record = _strict(
            item,
            (
                "sequence",
                "event",
                "session_id",
                "turn_id",
                "observed_at_unix_ms",
                "frame_sha256",
                "payload_sha256",
                "peer_pid",
                "peer_start_time_ticks",
                "peer_uid",
                "peer_gid",
                "chain_head_sha256",
            ),
            code,
        )
        event = record["event"]
        if (
            record["sequence"] != expected_sequence
            or event not in event_sequences
            or (
                record["session_id"] is not None
                and (
                    not isinstance(record["session_id"], str)
                    or re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}",
                        record["session_id"],
                    )
                    is None
                )
            )
            or (
                record["turn_id"] is not None
                and (
                    not isinstance(record["turn_id"], str)
                    or re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}",
                        record["turn_id"],
                    )
                    is None
                )
            )
            or type(record["observed_at_unix_ms"]) is not int
            or record["observed_at_unix_ms"] < last_observed
            or any(
                type(record[name]) is not int or record[name] < minimum
                for name, minimum in (
                    ("peer_pid", 2),
                    ("peer_start_time_ticks", 1),
                    ("peer_uid", 0),
                    ("peer_gid", 0),
                )
            )
        ):
            _fail(code)
        for name in ("frame_sha256", "payload_sha256", "chain_head_sha256"):
            _digest(record[name], code)
        expected_head = _sha256_json({
            "schema": _OBSERVER_FRAME_CHAIN_SCHEMA,
            "previous_sha256": previous,
            "sequence": expected_sequence,
            "frame_sha256": record["frame_sha256"],
            "peer_pid": record["peer_pid"],
            "peer_start_time_ticks": record["peer_start_time_ticks"],
        })
        if record["chain_head_sha256"] != expected_head:
            _fail(code)
        previous = expected_head
        last_observed = record["observed_at_unix_ms"]
        source_digests.update((record["frame_sha256"], record["payload_sha256"]))
        event_sequences[event].append(expected_sequence)
    singleton_events = (
        "plugin_ready",
        "api_session_bound",
        "private_target_probe_ready",
        "private_target_probe_result",
        "canonical_case_readback",
        "session_end",
    )
    if (
        any(len(event_sequences[event]) != 1 for event in singleton_events)
        or event_sequences["plugin_ready"] != [1]
        or event_sequences["session_end"] != [len(frames)]
        or not event_sequences["pre_api_request"]
        or len(event_sequences["pre_api_request"])
        != len(event_sequences["post_api_request"])
        or not event_sequences["post_tool_call"]
        or len({
            (
                record["peer_pid"],
                record["peer_start_time_ticks"],
                record["peer_uid"],
                record["peer_gid"],
            )
            for record in frames
        })
        != 1
    ):
        _fail(code)
    if raw["frame_chain_head_sha256"] != previous:
        _fail(code)
    digest_index = raw["source_digest_index"]
    proposal_source_digests = {
        value
        for identity in proposal_identities.values()
        for name, value in identity.items()
        if name.endswith("_sha256")
    }
    goal_source_digests = _recursive_digest_index(goal_identity)
    if (
        not isinstance(digest_index, list)
        or digest_index != sorted(set(digest_index))
        or any(_digest(item, code) != item for item in digest_index)
        or not source_digests.issubset(set(digest_index))
        or not proposal_source_digests.issubset(set(digest_index))
        or not goal_source_digests.issubset(set(digest_index))
    ):
        _fail(code)
    efforts = raw["reasoning_efforts"]
    if not isinstance(efforts, list):
        _fail(code)
    for item in efforts:
        record = _strict(
            item,
            (
                "frame_sequence",
                "request_ordinal",
                "provider_sha256",
                "api_mode_sha256",
                "base_url_sha256",
                "model_sha256",
                "effort",
            ),
            code,
        )
        if (
            record["frame_sequence"] not in event_sequences["pre_api_request"]
            or type(record["request_ordinal"]) is not int
            or record["request_ordinal"] <= 0
            or record["effort"] not in {"high", "max"}
        ):
            _fail(code)
        for name in (
            "provider_sha256",
            "api_mode_sha256",
            "base_url_sha256",
            "model_sha256",
        ):
            _digest(record[name], code)
        effort_records.append(record)
    if efforts != effort_records:
        _fail(code)
    route_digest_names = (
        "provider_sha256",
        "api_mode_sha256",
        "base_url_sha256",
        "model_sha256",
    )
    if (
        len(effort_records) < 2
        or [item["request_ordinal"] for item in effort_records]
        != list(range(1, len(effort_records) + 1))
        or [item["frame_sequence"] for item in effort_records]
        != sorted(item["frame_sequence"] for item in effort_records)
        or effort_records[0]["effort"] != "high"
        or effort_records[-1]["effort"] != "max"
        or any(
            item[name] != effort_records[0][name]
            for item in effort_records[1:]
            for name in route_digest_names
        )
    ):
        _fail(code)
    calls = raw["tool_call_records"]
    if not isinstance(calls, list):
        _fail(code)
    for item in calls:
        record = _strict(
            item,
            (
                "frame_sequence",
                "tool_call_ordinal",
                "tool_call_id",
                "tool_name",
                "args_sha256",
                "result_sha256",
                "reasoning_directive_effort_sha256",
            ),
            code,
        )
        if (
            record["frame_sequence"] not in event_sequences["post_tool_call"]
            or type(record["tool_call_ordinal"]) is not int
            or record["tool_call_ordinal"] <= 0
            or not isinstance(record["tool_call_id"], str)
            or not record["tool_call_id"]
            or not isinstance(record["tool_name"], str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", record["tool_name"])
            is None
        ):
            _fail(code)
        _digest(record["args_sha256"], code)
        _digest(record["result_sha256"], code)
        if record["reasoning_directive_effort_sha256"] is not None:
            _digest(record["reasoning_directive_effort_sha256"], code)
            if record["tool_name"] != "todo":
                _fail(code)
        tool_records.append(record)
    if calls != tool_records:
        _fail(code)
    if (
        [item["tool_call_ordinal"] for item in tool_records]
        != list(range(1, len(tool_records) + 1))
        or [item["frame_sequence"] for item in tool_records]
        != sorted(item["frame_sequence"] for item in tool_records)
        or len([
            item
            for item in tool_records
            if item["tool_name"] == "todo"
            and item["reasoning_directive_effort_sha256"] == _sha256_bytes(b"max")
        ])
        != 1
    ):
        _fail(code)
    proposal_identity = proposal_identities["workspace_gateway"]
    proposal_tool_records = [
        item
        for item in tool_records
        if item["tool_name"] == "canonical_event_append"
        and _sha256_bytes(item["tool_call_id"].encode("utf-8", errors="strict"))
        == proposal_identity["model_tool_call_id_sha256"]
        and item["result_sha256"] == proposal_identity["model_tool_result_sha256"]
        and frames[item["frame_sequence"] - 1]["frame_sha256"]
        == proposal_identity["model_tool_frame_sha256"]
    ]
    if len(proposal_tool_records) != 1:
        _fail(code)

    restart = raw["worker_restart_receipt"]
    restart_unsigned = (
        {
            key: copy.deepcopy(item)
            for key, item in restart.items()
            if key != "receipt_sha256"
        }
        if isinstance(restart, Mapping)
        else {}
    )
    if (
        set(restart_unsigned)
        != {"schema", "service_unit", "command_sha256", "completed_at_unix_ms"}
        or restart.get("schema") != "muncho-production-capability-worker-restart.v1"
        or restart.get("service_unit") != "muncho-isolated-worker.service"
        or restart.get("receipt_sha256") != _sha256_json(restart_unsigned)
        or type(restart.get("completed_at_unix_ms")) is not int
        or restart["completed_at_unix_ms"] <= 0
    ):
        _fail(code)
    _digest(restart.get("command_sha256"), code)
    terminal = raw["api_terminal_event_identity"]
    terminal_fields = {
        "schema",
        "session_id",
        "session_create_request_id",
        "chat_stream_request_id",
        "api_run_id",
        "api_message_id",
        "event_records",
        "event_chain_sha256",
        "assistant_completed_sha256",
        "run_completed_sha256",
        "transcript_sha256",
        "observed_at_unix_ms",
        "completed_at_unix_ms",
        "response_or_task_prose_recorded",
        "secret_material_recorded",
        "identity_sha256",
    }
    terminal_unsigned = (
        {
            key: copy.deepcopy(item)
            for key, item in terminal.items()
            if key != "identity_sha256"
        }
        if isinstance(terminal, Mapping)
        else {}
    )
    if (
        not isinstance(terminal, Mapping)
        or set(terminal) != terminal_fields
        or terminal.get("schema") != "muncho-capability-canary-api-terminal-identity.v1"
        or terminal.get("identity_sha256") != _sha256_json(terminal_unsigned)
        or terminal.get("response_or_task_prose_recorded") is not False
        or terminal.get("secret_material_recorded") is not False
    ):
        _fail(code)
    terminal_records = terminal["event_records"]
    if not isinstance(terminal_records, list) or not terminal_records:
        _fail(code)
    for expected_ordinal, item in enumerate(terminal_records, start=1):
        record = _strict(
            item,
            ("ordinal", "event", "payload_sha256"),
            code,
        )
        if (
            record["ordinal"] != expected_ordinal
            or not isinstance(record["event"], str)
            or re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", record["event"]) is None
        ):
            _fail(code)
        _digest(record["payload_sha256"], code)
    for name in (
        "session_id",
        "session_create_request_id",
        "chat_stream_request_id",
        "api_run_id",
        "api_message_id",
    ):
        if (
            not isinstance(terminal[name], str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", terminal[name])
            is None
        ):
            _fail(code)
    for name in (
        "event_chain_sha256",
        "assistant_completed_sha256",
        "run_completed_sha256",
        "transcript_sha256",
    ):
        _digest(terminal[name], code)
    if (
        terminal["event_chain_sha256"] != _sha256_json(terminal_records)
        or terminal["transcript_sha256"]
        != _sha256_json({
            "session_id": terminal["session_id"],
            "event_records": terminal_records,
        })
        or type(terminal["observed_at_unix_ms"]) is not int
        or type(terminal["completed_at_unix_ms"]) is not int
        or not 0 < terminal["observed_at_unix_ms"] <= terminal["completed_at_unix_ms"]
    ):
        _fail(code)
    if raw["observed_at_unix_ms"] < max(
        last_observed,
        restart["completed_at_unix_ms"],
        terminal["completed_at_unix_ms"],
        goal_identity["completed_at_unix_ms"],
    ):
        _fail(code)
    membership = _strict(
        raw["slot_membership"],
        ("runtime", "workspace_gateway", "failure_gateway"),
        code,
    )
    expected_membership = {
        "runtime": {
            "frame_sequences": [
                sequence
                for event in _OBSERVER_RUNTIME_EVENTS
                for sequence in event_sequences[event]
            ],
            "collector_readiness_receipt_sha256": readiness["receipt_sha256"],
            "runtime_source_identity_sha256": _sha256_json(runtime_source),
            "observer_activation_identity_sha256": observer_activation[
                "identity_sha256"
            ],
        },
        "workspace_gateway": {
            "frame_sequences": [
                sequence
                for event in _OBSERVER_WORKSPACE_EVENTS
                for sequence in event_sequences[event]
            ],
            "api_terminal_event_identity_sha256": terminal["identity_sha256"],
            "worker_restart_receipt_sha256": restart["receipt_sha256"],
            "model_proposal_core_identity_sha256": proposal_identities[
                "workspace_gateway"
            ]["identity_sha256"],
            "goal_continuation_identity_sha256": goal_identity[
                "identity_sha256"
            ],
        },
        "failure_gateway": {
            "frame_sequences": [
                sequence
                for event in _OBSERVER_FAILURE_EVENTS
                for sequence in event_sequences[event]
            ],
            "api_terminal_event_identity_sha256": terminal["identity_sha256"],
            "model_proposal_core_identity_sha256": proposal_identities[
                "failure_gateway"
            ]["identity_sha256"],
        },
    }
    for item in expected_membership.values():
        item["frame_sequences"] = sorted(item["frame_sequences"])
    if membership != expected_membership:
        _fail(code)
    return raw


def _extract_gateway_observer_model_proposal_contract(
    frames: Sequence[Any],
) -> tuple[
    Mapping[str, Mapping[str, Any]],
    Mapping[str, Mapping[str, Any]],
]:
    """Bind one model tool-authored proposal without exporting its core."""

    code = "gateway_observer_source_projection_invalid"
    readbacks = [
        item
        for item in frames
        if getattr(item, "value", {}).get("event") == "canonical_case_readback"
    ]
    if len(readbacks) != 1:
        _fail(code)
    try:
        readback_frame = readbacks[0]
        readback_payload = readback_frame.value["payload"]
        readback = readback_payload["readback"]
        events = readback["events"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise CapabilityProducerError(code) from exc
    if (
        not isinstance(readback, Mapping)
        or not isinstance(events, list)
        or not events
        or readback.get("truncated") is not False
        or readback_payload.get("readback_sha256") != _sha256_json(readback)
    ):
        _fail(code)
    matches = [
        (index, item)
        for index, item in enumerate(events)
        if isinstance(item, Mapping)
        and item.get("event_type") == GATEWAY_OBSERVER_PROPOSAL_EVENT_TYPE
    ]
    if len(matches) != 1 or matches[0][0] != len(events) - 1:
        _fail(code)
    proposal_event = matches[0][1]
    proposal_payload = proposal_event.get("payload")
    if (
        not isinstance(proposal_payload, Mapping)
        or set(proposal_payload) != {"evidence"}
        or not isinstance(proposal_payload["evidence"], list)
        or len(proposal_payload["evidence"]) != 2
    ):
        _fail(code)
    event_id = proposal_event.get("event_id")
    case_id = proposal_event.get("case_id")
    if any(
        not isinstance(item, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", item) is None
        for item in (event_id, case_id)
    ):
        _fail(code)
    tool_frames = []
    for item in frames:
        try:
            value = item.value
            payload = value["payload"]
            result = payload.get("result_projection")
            if (
                value.get("event") == "post_tool_call"
                and payload.get("tool_name") == "canonical_event_append"
                and isinstance(result, Mapping)
                and result.get("event_id") == event_id
            ):
                tool_frames.append(item)
        except (AttributeError, KeyError, TypeError):
            continue
    if len(tool_frames) != 1:
        _fail(code)
    tool_frame = tool_frames[0]
    tool_payload = tool_frame.value["payload"]
    tool_call_id = tool_payload.get("tool_call_id")
    if (
        not isinstance(tool_call_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", tool_call_id) is None
    ):
        _fail(code)
    _digest(tool_payload.get("result_sha256"), code)
    proposal_event_sha256 = _sha256_json(proposal_event)
    event_id_sha256 = _sha256_bytes(event_id.encode("utf-8", errors="strict"))
    case_id_sha256 = _sha256_bytes(case_id.encode("utf-8", errors="strict"))
    tool_call_id_sha256 = _sha256_bytes(tool_call_id.encode("utf-8", errors="strict"))
    identities: dict[str, Mapping[str, Any]] = {}
    cores: dict[str, Mapping[str, Any]] = {}
    contracts = (
        ("workspace_gateway", _OBSERVER_WORKSPACE_PROPOSAL_CORE_FIELDS),
        ("failure_gateway", _OBSERVER_FAILURE_PROPOSAL_CORE_FIELDS),
    )
    for raw, (expected_slot, core_fields) in zip(
        proposal_payload["evidence"],
        contracts,
        strict=True,
    ):
        proposal = _strict(raw, ("schema", "slot", "core"), code)
        core = proposal["core"]
        if (
            proposal["schema"] != GATEWAY_OBSERVER_PROPOSAL_CORE_SCHEMA
            or proposal["slot"] != expected_slot
            or not isinstance(core, Mapping)
            or set(core) != set(core_fields)
        ):
            _fail(code)
        unsigned = {
            "schema": GATEWAY_OBSERVER_PROPOSAL_IDENTITY_SCHEMA,
            "slot": expected_slot,
            "proposal_event_sha256": proposal_event_sha256,
            "proposal_event_id_sha256": event_id_sha256,
            "proposal_case_id_sha256": case_id_sha256,
            "model_tool_frame_sha256": _digest(tool_frame.sha256, code),
            "model_tool_call_id_sha256": tool_call_id_sha256,
            "model_tool_result_sha256": tool_payload["result_sha256"],
            "core_sha256": _sha256_json(core),
        }
        identities[expected_slot] = {
            **unsigned,
            "identity_sha256": _sha256_json(unsigned),
        }
        cores[expected_slot] = copy.deepcopy(dict(core))
    return identities, cores


def _extract_gateway_observer_model_proposal_identities(
    frames: Sequence[Any],
) -> Mapping[str, Mapping[str, Any]]:
    return _extract_gateway_observer_model_proposal_contract(frames)[0]


def extract_gateway_observer_model_proposal_cores(
    frames: Sequence[Any],
) -> Mapping[str, Mapping[str, Any]]:
    """Return only the two exact model tool-authored proposal cores."""

    return copy.deepcopy(
        dict(_extract_gateway_observer_model_proposal_contract(frames)[1])
    )


def _build_observer_activation_identity(
    *,
    producer_readiness: Mapping[str, Any],
    foundation: Mapping[str, Any],
    fixture_sha256: str,
    run_id: str,
) -> Mapping[str, Any]:
    code = "gateway_observer_source_projection_invalid"
    try:
        readiness_unsigned = {
            key: copy.deepcopy(item)
            for key, item in producer_readiness.items()
            if key != "readiness_sha256"
        }
        endpoint = producer_readiness["endpoint_readiness"]["gateway_observer"]
        endpoint_unsigned = {
            key: copy.deepcopy(item)
            for key, item in endpoint.items()
            if key != "readiness_sha256"
        }
        expected = foundation["endpoints"]["gateway_observer"]
        if (
            producer_readiness.get("schema")
            != "muncho-production-capability-canary-producer-activation.v1"
            or producer_readiness.get("foundation_sha256")
            != producer_foundation_sha256(foundation)
            or producer_readiness.get("release_sha") != foundation["release_sha"]
            or producer_readiness.get("capability_plan_sha256")
            != foundation["capability_plan_sha256"]
            or producer_readiness.get("full_canary_plan_sha256")
            != foundation["full_canary_plan_sha256"]
            or producer_readiness.get("fixture_sha256") != fixture_sha256
            or producer_readiness.get("run_id") != run_id
            or producer_readiness.get("readiness_sha256")
            != _sha256_json(readiness_unsigned)
            or endpoint.get("readiness_sha256") != _sha256_json(endpoint_unsigned)
            or endpoint.get("role") != "gateway_observer"
            or endpoint.get("foundation_sha256")
            != producer_readiness["foundation_sha256"]
            or endpoint.get("release_sha") != producer_readiness["release_sha"]
            or endpoint.get("capability_plan_sha256")
            != producer_readiness["capability_plan_sha256"]
            or endpoint.get("full_canary_plan_sha256")
            != producer_readiness["full_canary_plan_sha256"]
            or endpoint.get("service_unit") != expected["service_unit"]
            or endpoint.get("service_identity_sha256")
            != expected["service_identity_sha256"]
            or endpoint.get("uid") != expected["uid"]
            or endpoint.get("gid") != expected["gid"]
            or type(endpoint.get("main_pid")) is not int
            or endpoint["main_pid"] < 2
        ):
            _fail(code)
        unsigned = {
            "producer_readiness_sha256": producer_readiness["readiness_sha256"],
            "observer_endpoint_readiness_sha256": endpoint["readiness_sha256"],
            "observer_main_pid": endpoint["main_pid"],
            "observer_service_unit": endpoint["service_unit"],
            "observer_service_identity_sha256": endpoint["service_identity_sha256"],
            "observer_uid": endpoint["uid"],
            "observer_gid": endpoint["gid"],
        }
    except (KeyError, TypeError) as exc:
        raise CapabilityProducerError(code) from exc
    return {**unsigned, "identity_sha256": _sha256_json(unsigned)}


def build_goal_continuation_native_identity(
    evidence: Mapping[str, Any],
    *,
    fixture: Mapping[str, Any] | None = None,
    fixture_sha256: str | None = None,
    owner_approval_receipt_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Project only digests from independently collected goal evidence.

    The source projection deliberately contains no verdict enum, reason,
    message text, task prose, or other semantic value.  The signed receipt
    producer later requires the complete evidence object to hash back to this
    identity before it can include that object in the workspace receipt.
    """

    code = "goal_continuation_native_identity_invalid"
    if not isinstance(evidence, Mapping):
        _fail(code)
    # Construction from a live collector is an authority boundary: do not
    # bless a caller-supplied, internally self-hashed mapping.  Require the
    # complete replay-bound contract there.  The no-fixture form is retained
    # only for re-projecting an already-pinned payload and is always compared
    # to the independently constructed source identity by the native
    # collector below.
    if fixture is not None or fixture_sha256 is not None or owner_approval_receipt_sha256 is not None:
        if (
            not isinstance(fixture, Mapping)
            or not isinstance(fixture_sha256, str)
            or not isinstance(owner_approval_receipt_sha256, str)
        ):
            _fail(code)
        try:
            evidence = evidence_contract._validate_goal_continuation_evidence(
                evidence,
                fixture=fixture,
                fixture_sha256=fixture_sha256,
                owner_approval_receipt_sha256=owner_approval_receipt_sha256,
            )
        except evidence_contract.CapabilityCanaryEvidenceError as exc:
            raise CapabilityProducerError(code) from exc
    try:
        evidence_unsigned = {
            key: copy.deepcopy(item)
            for key, item in evidence.items()
            if key != "evidence_sha256"
        }
        terminal = evidence["terminal"]
        terminal_unsigned = {
            key: copy.deepcopy(item)
            for key, item in terminal.items()
            if key != "terminal_sha256"
        }
        ingress = evidence["discord_owner_ingress"]
        verdicts = evidence["model_outcomes"]
        restart = evidence["gateway_restart"]
        ctw = evidence["ctw_recovery"]
        route = evidence["model_route"]
        stability = evidence["prompt_tool_stability"]
        preemption = evidence["user_preemption_queue_e2e"]
        verdict_receipts = [item["receipt_sha256"] for item in verdicts]
        unsigned = {
            "schema": GOAL_CONTINUATION_NATIVE_IDENTITY_SCHEMA,
            "evidence_sha256": evidence["evidence_sha256"],
            "terminal_sha256": terminal["terminal_sha256"],
            "discord_ingress_receipt_sha256": ingress["receipt_sha256"],
            "model_verdict_receipt_sha256s": verdict_receipts,
            "gateway_restart_receipt_sha256": restart["receipt_sha256"],
            "ctw_recovery_receipt_sha256": ctw["receipt_sha256"],
            "model_route_receipt_sha256": route["receipt_sha256"],
            "prompt_tool_stability_receipt_sha256": stability["receipt_sha256"],
            "user_preemption_queue_receipt_sha256": preemption["receipt_sha256"],
            "production_diff_sha256": terminal["production_diff_sha256"],
            "completed_at_unix_ms": terminal["completed_at_unix_ms"],
        }
    except (KeyError, TypeError) as exc:
        raise CapabilityProducerError(code) from exc
    if (
        evidence["evidence_sha256"] != _sha256_json(evidence_unsigned)
        or terminal["terminal_sha256"] != _sha256_json(terminal_unsigned)
        or not isinstance(verdicts, list)
        or len(verdicts) < 3
        or len(verdict_receipts) != len(set(verdict_receipts))
        or type(unsigned["completed_at_unix_ms"]) is not int
        or unsigned["completed_at_unix_ms"] <= 0
    ):
        _fail(code)
    for value in (
        unsigned["evidence_sha256"],
        unsigned["terminal_sha256"],
        unsigned["discord_ingress_receipt_sha256"],
        *unsigned["model_verdict_receipt_sha256s"],
        unsigned["gateway_restart_receipt_sha256"],
        unsigned["ctw_recovery_receipt_sha256"],
        unsigned["model_route_receipt_sha256"],
        unsigned["prompt_tool_stability_receipt_sha256"],
        unsigned["user_preemption_queue_receipt_sha256"],
        unsigned["production_diff_sha256"],
    ):
        _digest(value, code)
    return {**unsigned, "identity_sha256": _sha256_json(unsigned)}


def build_gateway_observer_source_projection(
    *,
    foundation: Mapping[str, Any],
    fixture: Mapping[str, Any],
    fixture_sha256: str,
    run_id: str,
    producer_readiness: Mapping[str, Any],
    collector_readiness: Mapping[str, Any],
    runtime_source_identity: Mapping[str, Any],
    frames: Sequence[Any],
    worker_restart_receipt: Mapping[str, Any],
    api_terminal_event_identity: Mapping[str, Any],
    goal_continuation_evidence: Mapping[str, Any],
    observed_at_unix_ms: int,
) -> Mapping[str, Any]:
    """Build a redacted trusted-source projection, never observer evidence."""

    code = "gateway_observer_source_projection_invalid"
    if not frames or type(observed_at_unix_ms) is not int or observed_at_unix_ms <= 0:
        _fail(code)
    observer_activation_identity = _build_observer_activation_identity(
        producer_readiness=producer_readiness,
        foundation=foundation,
        fixture_sha256=fixture_sha256,
        run_id=run_id,
    )
    model_proposal_identities = _extract_gateway_observer_model_proposal_identities(
        frames
    )
    goal_continuation_identity = build_goal_continuation_native_identity(
        goal_continuation_evidence,
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        owner_approval_receipt_sha256=goal_continuation_evidence.get(
            "owner_approval_receipt_sha256"
        )
        or goal_continuation_evidence.get("discord_owner_ingress", {}).get(
            "owner_approval_receipt_sha256"
        ),
    )
    try:
        readiness_unsigned = {
            key: copy.deepcopy(item)
            for key, item in collector_readiness.items()
            if key != "receipt_sha256"
        }
        if collector_readiness.get("receipt_sha256") != _sha256_json(
            readiness_unsigned
        ):
            _fail(code)
        readiness_identity = {
            "receipt_sha256": collector_readiness["receipt_sha256"],
            "service_identity_sha256": collector_readiness["service_identity_sha256"],
            "edge_service_identity_sha256": collector_readiness[
                "edge_service_identity_sha256"
            ],
            "collector_socket_sha256": _sha256_json(
                collector_readiness["collector_socket"]
            ),
        }
    except (KeyError, TypeError) as exc:
        raise CapabilityProducerError(code) from exc
    frame_records: list[Mapping[str, Any]] = []
    digest_index: set[str] = set()
    efforts: list[Mapping[str, Any]] = []
    tool_calls: list[Mapping[str, Any]] = []
    for expected_sequence, collected in enumerate(frames, start=1):
        try:
            frame = collected.value
            peer = collected.peer
            frame_sha256 = _sha256_json(frame)
            payload = frame["payload"]
            event = frame["event"]
            if (
                collected.sha256 != frame_sha256
                or frame["sequence"] != expected_sequence
                or event not in _OBSERVER_FRAME_EVENTS
                or not isinstance(payload, Mapping)
                or frame.get("release_sha") != foundation["release_sha"]
                or frame.get("collector_service_identity_sha256")
                != readiness_identity["service_identity_sha256"]
                or frame.get("discord_edge_service_identity_sha256")
                != readiness_identity["edge_service_identity_sha256"]
                or frame.get("canary_run_id") != frames[0].value.get("canary_run_id")
                or frame.get("fixture_sha256") != frames[0].value.get("fixture_sha256")
            ):
                _fail(code)
            record = {
                "sequence": expected_sequence,
                "event": event,
                "session_id": frame.get("session_id"),
                "turn_id": frame.get("turn_id"),
                "observed_at_unix_ms": frame["observed_at_unix_ms"],
                "frame_sha256": frame_sha256,
                "payload_sha256": _sha256_json(payload),
                "peer_pid": peer.pid,
                "peer_start_time_ticks": peer.start_time_ticks,
                "peer_uid": peer.uid,
                "peer_gid": peer.gid,
                "chain_head_sha256": collected.chain_head_sha256,
            }
        except (AttributeError, KeyError, TypeError) as exc:
            raise CapabilityProducerError(code) from exc
        frame_records.append(record)
        digest_index.update(_recursive_digest_index(payload))
        digest_index.update((record["frame_sha256"], record["payload_sha256"]))
        if event == "pre_api_request":
            route_values = {
                name: payload.get(name)
                for name in ("provider", "api_mode", "base_url", "model")
            }
            if any(
                not isinstance(item, str) or not item for item in route_values.values()
            ):
                _fail(code)
            efforts.append({
                "frame_sequence": expected_sequence,
                "request_ordinal": payload.get("request_ordinal"),
                "provider_sha256": _sha256_bytes(
                    route_values["provider"].encode("utf-8", errors="strict")
                ),
                "api_mode_sha256": _sha256_bytes(
                    route_values["api_mode"].encode("utf-8", errors="strict")
                ),
                "base_url_sha256": _sha256_bytes(
                    route_values["base_url"].encode("utf-8", errors="strict")
                ),
                "model_sha256": _sha256_bytes(
                    route_values["model"].encode("utf-8", errors="strict")
                ),
                "effort": payload.get("reasoning_effort"),
            })
        if event == "post_tool_call":
            directive = payload.get("reasoning_directive")
            directive_effort = (
                directive.get("effort") if isinstance(directive, Mapping) else None
            )
            tool_calls.append({
                "frame_sequence": expected_sequence,
                "tool_call_ordinal": payload.get("tool_call_ordinal"),
                "tool_call_id": payload.get("tool_call_id"),
                "tool_name": payload.get("tool_name"),
                "args_sha256": payload.get("args_sha256"),
                "result_sha256": payload.get("result_sha256"),
                "reasoning_directive_effort_sha256": (
                    _sha256_bytes(directive_effort.encode("utf-8", errors="strict"))
                    if isinstance(directive_effort, str) and directive_effort
                    else None
                ),
            })
    for identity in model_proposal_identities.values():
        digest_index.update(
            value for name, value in identity.items() if name.endswith("_sha256")
        )
    digest_index.update(_recursive_digest_index(goal_continuation_identity))
    event_sequences = {
        event: [
            record["sequence"] for record in frame_records if record["event"] == event
        ]
        for event in _OBSERVER_FRAME_EVENTS
    }
    membership = {
        "runtime": {
            "frame_sequences": sorted(
                sequence
                for event in _OBSERVER_RUNTIME_EVENTS
                for sequence in event_sequences[event]
            ),
            "collector_readiness_receipt_sha256": readiness_identity["receipt_sha256"],
            "runtime_source_identity_sha256": _sha256_json(runtime_source_identity),
            "observer_activation_identity_sha256": (
                observer_activation_identity["identity_sha256"]
            ),
        },
        "workspace_gateway": {
            "frame_sequences": sorted(
                sequence
                for event in _OBSERVER_WORKSPACE_EVENTS
                for sequence in event_sequences[event]
            ),
            "api_terminal_event_identity_sha256": api_terminal_event_identity[
                "identity_sha256"
            ],
            "worker_restart_receipt_sha256": worker_restart_receipt["receipt_sha256"],
            "model_proposal_core_identity_sha256": (
                model_proposal_identities["workspace_gateway"]["identity_sha256"]
            ),
            "goal_continuation_identity_sha256": goal_continuation_identity[
                "identity_sha256"
            ],
        },
        "failure_gateway": {
            "frame_sequences": sorted(
                sequence
                for event in _OBSERVER_FAILURE_EVENTS
                for sequence in event_sequences[event]
            ),
            "api_terminal_event_identity_sha256": api_terminal_event_identity[
                "identity_sha256"
            ],
            "model_proposal_core_identity_sha256": (
                model_proposal_identities["failure_gateway"]["identity_sha256"]
            ),
        },
    }
    unsigned = {
        "schema": GATEWAY_OBSERVER_SOURCE_PROJECTION_SCHEMA,
        "run_id": run_id,
        "release_sha": foundation["release_sha"],
        "capability_plan_sha256": foundation["capability_plan_sha256"],
        "full_canary_plan_sha256": foundation["full_canary_plan_sha256"],
        "fixture_sha256": fixture_sha256,
        "source_canary_run_id": frames[0].value["canary_run_id"],
        "source_fixture_sha256": frames[0].value["fixture_sha256"],
        "observer_activation_identity": observer_activation_identity,
        "collector_readiness_identity": readiness_identity,
        "runtime_source_identity": copy.deepcopy(dict(runtime_source_identity)),
        "model_proposal_core_identities": model_proposal_identities,
        "goal_continuation_identity": goal_continuation_identity,
        "frame_records": frame_records,
        "frame_chain_head_sha256": frame_records[-1]["chain_head_sha256"],
        "source_digest_index": sorted(digest_index),
        "reasoning_efforts": efforts,
        "tool_call_records": tool_calls,
        "worker_restart_receipt": copy.deepcopy(dict(worker_restart_receipt)),
        "api_terminal_event_identity": copy.deepcopy(dict(api_terminal_event_identity)),
        "slot_membership": membership,
        "source_projection_only": True,
        "native_evidence_bindings_recorded": False,
        "semantic_task_prose_recorded": False,
        "success_or_outcome_recorded": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "observed_at_unix_ms": observed_at_unix_ms,
    }
    return validate_gateway_observer_source_projection(
        {**unsigned, "projection_sha256": _sha256_json(unsigned)},
        release_sha=foundation["release_sha"],
        capability_plan_sha256=foundation["capability_plan_sha256"],
        full_canary_plan_sha256=foundation["full_canary_plan_sha256"],
        fixture_sha256=fixture_sha256,
        run_id=run_id,
    )


def publish_gateway_observer_source_projection(
    value: Mapping[str, Any],
    *,
    foundation: Mapping[str, Any],
    fixture_sha256: str,
    run_id: str,
) -> Mapping[str, Any]:
    """No-replace root export; it contains no role-native bindings."""

    observer_gid = foundation["endpoints"]["gateway_observer"]["gid"]
    validated = validate_gateway_observer_source_projection(
        value,
        release_sha=foundation["release_sha"],
        capability_plan_sha256=foundation["capability_plan_sha256"],
        full_canary_plan_sha256=foundation["full_canary_plan_sha256"],
        fixture_sha256=fixture_sha256,
        run_id=run_id,
    )
    run_root = Path(foundation["receipt_contract"]["base_root"]) / run_id
    _require_exact_directory(
        run_root,
        uid=foundation["receipt_contract"]["run_directory_uid"],
        gid=foundation["receipt_contract"]["run_directory_gid"],
        mode=foundation["receipt_contract"]["run_directory_mode"],
    )
    path = _gateway_observer_source_projection_path(run_root)
    raw = _canonical_bytes(validated)
    _publish_no_replace(
        path,
        raw,
        uid=0,
        gid=observer_gid,
        mode=0o440,
        parent_uid=foundation["receipt_contract"]["run_directory_uid"],
        parent_gid=foundation["receipt_contract"]["run_directory_gid"],
        parent_mode=foundation["receipt_contract"]["run_directory_mode"],
    )
    return {
        "path": str(path),
        "file_sha256": _sha256_bytes(raw),
        "projection_sha256": validated["projection_sha256"],
        "uid": 0,
        "gid": observer_gid,
        "mode": "0440",
    }


_GATEWAY_OBSERVER_SOURCE_SLOT_SCHEMAS = {
    "runtime": "muncho-production-capability-runtime-receipt.v1",
    "workspace_gateway": (
        "muncho-production-capability-canonical-task-workspace-gateway.v4"
    ),
    "failure_gateway": "muncho-production-capability-failure-gateway.v1",
}
_GATEWAY_OBSERVER_RUNTIME_FIELDS = (
    "host_identity_sha256",
    "release_artifact_sha256",
    "installed_wheel_manifest_sha256",
    "effective_config_sha256",
    "tool_inventory_sha256",
    "provider",
    "api_mode",
    "model",
    "initial_effort",
    "adaptive_max_effort",
    "max_turns",
    "semantic_config_contract",
    "semantic_config_contract_sha256",
    "toolsets",
    "ordered_toolsets_sha256",
    "capability_role_topology_contract",
    "capability_role_topology_contract_sha256",
    "kanban_auxiliary_planning_enabled",
    "kanban_auto_decompose",
    "kanban_dispatch_in_gateway",
    "prompt_cache_stable",
    "message_alternation_valid",
    "gateway_process_identity_sha256",
    "connector_bot_user_id",
    "connector_bot_user_id_provenance",
    "connector_readiness_receipt_sha256",
)
_GATEWAY_OBSERVER_WORKSPACE_FIELDS = (
    "session_id",
    "capability_epoch_sha256",
    "transcript_sha256",
    "task_workspace_evidence_sha256s",
    "first_path_failure_receipt_sha256",
    "alternate_read_receipt_sha256",
    "model_requested_effort",
    "later_request_effort",
    "reasoning_tool_call_id",
    "restart_count",
    "used_command_sha256s",
    "mutation_receipt_sha256s",
    "approval_prompt_count",
    "microapproval_prompt_count",
    "replayed_mutation_count",
    "owner_grant_id",
    "owner_grant_sha256",
    "consumed_command_sha256s",
    "terminal_plan_id",
    "terminal_plan_revision",
    "goal_continuation_evidence",
)
_GATEWAY_OBSERVER_FAILURE_FIELDS = (
    "transcript_sha256",
    "failures",
    "model_retained_tool_control",
)
_GATEWAY_OBSERVER_COMMON_FIELDS = (
    "schema",
    "run_id",
    "release_sha",
    "fixture_sha256",
    "observed_at_unix_ms",
)
_GATEWAY_OBSERVER_REQUIRED_TOOLSETS = (
    "browser",
    "canonical_brain",
    "clarify",
    "delegation",
    "discord_guild_read",
    "file",
    "mac_ops",
    "memory",
    "session_search",
    "skills",
    "terminal",
    "todo",
    "web",
)
_GATEWAY_OBSERVER_FAILURE_COMPONENTS = (
    "tool",
    "browser",
    "database",
    "writer",
    "egress",
)


def _read_gateway_observer_source_projection(
    path: Path,
    gid: int,
) -> tuple[Mapping[str, Any], str]:
    raw, _item = _stable_read(
        path,
        maximum=2 * 1024 * 1024,
        uid=0,
        gid=gid,
        mode=0o440,
    )
    value = _strict_json(raw, "gateway_observer_source_projection_invalid")
    if raw != _canonical_bytes(value):
        _fail("gateway_observer_source_projection_invalid")
    return value, _sha256_bytes(raw)


def _observer_digest_list(
    value: Any,
    *,
    code: str,
    minimum: int = 0,
    sorted_unique: bool = False,
) -> list[str]:
    if not isinstance(value, list) or not minimum <= len(value) <= 64:
        _fail(code)
    try:
        unique = set(value)
        ordered = sorted(value) if sorted_unique else value
    except (TypeError, ValueError) as exc:
        raise CapabilityProducerError(code) from exc
    if len(value) != len(unique) or (sorted_unique and value != ordered):
        _fail(code)
    return [_digest(item, code) for item in value]


def _observer_text_sha256(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(code)
    try:
        return _sha256_bytes(value.encode("utf-8", errors="strict"))
    except UnicodeError as exc:
        raise CapabilityProducerError(code) from exc


class GatewayObserverSourceNativeCollector:
    """Derive observer bindings from one immutable, prose-free source file."""

    def __init__(
        self,
        *,
        config: ProducerConfig,
        foundation: Mapping[str, Any],
        source_reader: Callable[
            [Path, int], tuple[Mapping[str, Any], str]
        ] = _read_gateway_observer_source_projection,
    ) -> None:
        if config.role != "gateway_observer" or not callable(source_reader):
            _fail("gateway_observer_source_collector_invalid")
        validate_producer_config_binding(config, foundation)
        self.config = config
        self.foundation = copy.deepcopy(dict(foundation))
        self.source_reader = source_reader

    @staticmethod
    def _exact_payload(
        slot: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        code = "gateway_observer_source_evidence_invalid"
        slot_fields = {
            "runtime": _GATEWAY_OBSERVER_RUNTIME_FIELDS,
            "workspace_gateway": _GATEWAY_OBSERVER_WORKSPACE_FIELDS,
            "failure_gateway": _GATEWAY_OBSERVER_FAILURE_FIELDS,
        }.get(slot)
        if (
            slot_fields is None
            or not isinstance(payload, Mapping)
            or set(payload) != set(_GATEWAY_OBSERVER_COMMON_FIELDS + slot_fields)
            or payload.get("schema") != _GATEWAY_OBSERVER_SOURCE_SLOT_SCHEMAS[slot]
            or type(payload.get("observed_at_unix_ms")) is not int
            or payload["observed_at_unix_ms"] <= 0
        ):
            _fail(code)
        return payload

    def _load(
        self,
        *,
        payload: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], str]:
        code = "gateway_observer_source_evidence_invalid"
        run_id = payload.get("run_id")
        if (
            not isinstance(run_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", run_id) is None
        ):
            _fail(code)
        path = _gateway_observer_source_projection_path(
            self.config.receipt_base_root / run_id
        )
        try:
            raw, file_sha256 = self.source_reader(path, self.config.service_gid)
            source = validate_gateway_observer_source_projection(
                raw,
                release_sha=self.config.release_sha,
                capability_plan_sha256=self.config.capability_plan_sha256,
                full_canary_plan_sha256=self.config.full_canary_plan_sha256,
                fixture_sha256=_digest(payload.get("fixture_sha256"), code),
                run_id=run_id,
            )
        except CapabilityProducerError:
            raise
        except Exception as exc:
            raise CapabilityProducerError(code) from exc
        activation = source["observer_activation_identity"]
        if (
            file_sha256 != _sha256_bytes(_canonical_bytes(source))
            or payload.get("release_sha") != self.config.release_sha
            or payload["observed_at_unix_ms"] > source["observed_at_unix_ms"]
            or activation["observer_main_pid"] != os.getpid()
            or activation["observer_service_unit"] != self.config.service_unit
            or activation["observer_service_identity_sha256"]
            != self.config.service_identity_sha256
            or activation["observer_uid"] != self.config.service_uid
            or activation["observer_gid"] != self.config.service_gid
            or activation["observer_service_identity_sha256"]
            != self.foundation["endpoints"]["gateway_observer"][
                "service_identity_sha256"
            ]
        ):
            _fail(code)
        return source, file_sha256

    @staticmethod
    def _runtime_bindings(
        payload: Mapping[str, Any],
        source: Mapping[str, Any],
        verification_sha256: str,
    ) -> Sequence[NativeEvidenceBinding]:
        code = "gateway_observer_source_evidence_invalid"
        for name in (
            "host_identity_sha256",
            "release_artifact_sha256",
            "installed_wheel_manifest_sha256",
            "effective_config_sha256",
            "tool_inventory_sha256",
        ):
            _digest(payload.get(name), code)
        runtime_source = source["runtime_source_identity"]
        model_calls = source["reasoning_efforts"]
        first = model_calls[0] if model_calls else None
        semantic_config = payload.get("semantic_config_contract")
        topology = payload.get("capability_role_topology_contract")
        toolsets = payload.get("toolsets")
        if (
            not isinstance(first, Mapping)
            or payload.get("gateway_process_identity_sha256")
            != runtime_source["gateway_process_identity_sha256"]
            or payload.get("connector_readiness_receipt_sha256")
            != runtime_source["discord_connector_readiness_sha256"]
            or payload.get("connector_bot_user_id")
            != runtime_source["connector_bot_user_id"]
            or payload.get("connector_bot_user_id_provenance")
            != runtime_source["connector_bot_user_id_provenance"]
            or _observer_text_sha256(payload.get("provider"), code)
            != first["provider_sha256"]
            or _observer_text_sha256(payload.get("api_mode"), code)
            != first["api_mode_sha256"]
            or _observer_text_sha256(payload.get("model"), code)
            != first["model_sha256"]
            or payload.get("initial_effort") != first["effort"]
            or payload.get("adaptive_max_effort") != "max"
            or not any(item["effort"] == "max" for item in model_calls)
            or payload.get("max_turns") != 90
            or toolsets != list(_GATEWAY_OBSERVER_REQUIRED_TOOLSETS)
            or payload.get("ordered_toolsets_sha256")
            != _sha256_json({"toolsets": toolsets})
            or not isinstance(semantic_config, Mapping)
            or payload.get("semantic_config_contract_sha256")
            != _sha256_json(semantic_config)
            or semantic_config.get("model_route")
            != {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "model": "gpt-5.6-sol",
            }
            or semantic_config.get("goals")
            != {
                "manager": "hermes_cli.goals.GoalManager",
                "continuations_enabled": True,
                "max_turns": 0,
                "outcome_source": "todo.goal_outcome",
                "completion_authority": "primary_model",
            }
            or not isinstance(topology, Mapping)
            or payload.get("capability_role_topology_contract_sha256")
            != _sha256_json(topology)
            or topology.get("gateway", {}).get("discord_credential_present")
            is not False
            or topology.get("gateway", {}).get("direct_canonical_write_enabled")
            is not False
            or payload.get("kanban_auxiliary_planning_enabled") is not False
            or payload.get("kanban_auto_decompose") is not False
            or payload.get("kanban_dispatch_in_gateway") is not False
            or payload.get("prompt_cache_stable") is not True
            or payload.get("message_alternation_valid") is not True
        ):
            _fail(code)
        observer_activation = source["observer_activation_identity"]
        runtime_source_sha256 = _sha256_json(runtime_source)
        routeback_identity = {
            "connector_bot_user_id": runtime_source["connector_bot_user_id"],
            "connector_bot_user_id_provenance": runtime_source[
                "connector_bot_user_id_provenance"
            ],
        }
        return (
            NativeEvidenceBinding(
                kind="gateway_runtime_readiness",
                source_identity_sha256=observer_activation["identity_sha256"],
                artifact_sha256=runtime_source["gateway_process_identity_sha256"],
                verification_receipt_sha256=verification_sha256,
            ),
            NativeEvidenceBinding(
                kind="discord_connector_readiness",
                source_identity_sha256=runtime_source_sha256,
                artifact_sha256=runtime_source["discord_connector_readiness_sha256"],
                verification_receipt_sha256=verification_sha256,
            ),
            NativeEvidenceBinding(
                kind="routeback_bot_identity",
                source_identity_sha256=runtime_source_sha256,
                artifact_sha256=_sha256_json(routeback_identity),
                verification_receipt_sha256=verification_sha256,
            ),
        )

    @staticmethod
    def _workspace_bindings(
        payload: Mapping[str, Any],
        source: Mapping[str, Any],
        verification_sha256: str,
    ) -> Sequence[NativeEvidenceBinding]:
        code = "gateway_observer_source_evidence_invalid"
        terminal = source["api_terminal_event_identity"]
        restart = source["worker_restart_receipt"]
        source_digests = set(source["source_digest_index"])
        proposal_identity = source["model_proposal_core_identities"][
            "workspace_gateway"
        ]
        proposal_core = {
            name: copy.deepcopy(payload[name])
            for name in _OBSERVER_WORKSPACE_PROPOSAL_CORE_FIELDS
        }
        goal_evidence = payload.get("goal_continuation_evidence")
        if not isinstance(goal_evidence, Mapping):
            _fail(code)
        try:
            goal_identity = build_goal_continuation_native_identity(goal_evidence)
        except CapabilityProducerError:
            _fail(code)
        task_evidence = _observer_digest_list(
            payload.get("task_workspace_evidence_sha256s"),
            code=code,
            minimum=1,
            sorted_unique=True,
        )
        used = _observer_digest_list(
            payload.get("used_command_sha256s"),
            code=code,
            minimum=1,
            sorted_unique=True,
        )
        consumed = _observer_digest_list(
            payload.get("consumed_command_sha256s"),
            code=code,
            minimum=1,
            sorted_unique=True,
        )
        mutations = _observer_digest_list(
            payload.get("mutation_receipt_sha256s"),
            code=code,
            minimum=1,
        )
        digest_values = {
            _digest(payload.get("capability_epoch_sha256"), code),
            _digest(payload.get("first_path_failure_receipt_sha256"), code),
            _digest(payload.get("alternate_read_receipt_sha256"), code),
            _digest(payload.get("owner_grant_sha256"), code),
            *task_evidence,
            *used,
            *mutations,
            *consumed,
        }
        todo_records = [
            item
            for item in source["tool_call_records"]
            if item["tool_name"] == "todo"
            and item["reasoning_directive_effort_sha256"] == _sha256_bytes(b"max")
        ]
        frame_sessions = {
            item["session_id"]
            for item in source["frame_records"]
            if item["session_id"] is not None
        }
        if (
            _sha256_json(proposal_core) != proposal_identity["core_sha256"]
            or goal_identity != source["goal_continuation_identity"]
            or digest_values - source_digests
            or payload.get("session_id") != terminal["session_id"]
            or frame_sessions != {terminal["session_id"]}
            or payload.get("transcript_sha256") != terminal["transcript_sha256"]
            or payload.get("model_requested_effort") != "max"
            or payload.get("later_request_effort") != "max"
            or not any(
                item["request_ordinal"] > 1 and item["effort"] == "max"
                for item in source["reasoning_efforts"]
            )
            or len(todo_records) != 1
            or payload.get("reasoning_tool_call_id") != todo_records[0]["tool_call_id"]
            or payload.get("restart_count") != 1
            or payload.get("approval_prompt_count") != 0
            or payload.get("microapproval_prompt_count") != 0
            or payload.get("replayed_mutation_count") != 0
            or used != consumed
            or not isinstance(payload.get("owner_grant_id"), str)
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}",
                payload["owner_grant_id"],
            )
            is None
            or not isinstance(payload.get("terminal_plan_id"), str)
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}",
                payload["terminal_plan_id"],
            )
            is None
            or type(payload.get("terminal_plan_revision")) is not int
            or payload["terminal_plan_revision"] <= 0
        ):
            _fail(code)
        readiness = source["collector_readiness_identity"]
        terminal_source = _sha256_json({
            "schema": terminal["schema"],
            "session_create_request_id": terminal["session_create_request_id"],
            "chat_stream_request_id": terminal["chat_stream_request_id"],
            "api_run_id": terminal["api_run_id"],
            "api_message_id": terminal["api_message_id"],
        })
        restart_source = _sha256_json({
            "service_unit": restart["service_unit"],
            "command_sha256": restart["command_sha256"],
        })
        goal_source = goal_identity["identity_sha256"]
        return (
            NativeEvidenceBinding(
                kind="gateway_observer_frame_chain",
                source_identity_sha256=readiness["service_identity_sha256"],
                artifact_sha256=source["frame_chain_head_sha256"],
                verification_receipt_sha256=verification_sha256,
            ),
            NativeEvidenceBinding(
                kind="authenticated_api_terminal_event",
                source_identity_sha256=terminal_source,
                artifact_sha256=terminal["identity_sha256"],
                verification_receipt_sha256=verification_sha256,
            ),
            NativeEvidenceBinding(
                kind="isolated_worker_restart_receipt",
                source_identity_sha256=restart_source,
                artifact_sha256=restart["receipt_sha256"],
                verification_receipt_sha256=verification_sha256,
            ),
            NativeEvidenceBinding(
                kind="goal_continuation_native_identity",
                source_identity_sha256=goal_source,
                artifact_sha256=goal_identity["evidence_sha256"],
                verification_receipt_sha256=verification_sha256,
            ),
        )

    @staticmethod
    def _failure_bindings(
        payload: Mapping[str, Any],
        source: Mapping[str, Any],
        verification_sha256: str,
    ) -> Sequence[NativeEvidenceBinding]:
        code = "gateway_observer_source_evidence_invalid"
        terminal = source["api_terminal_event_identity"]
        source_digests = set(source["source_digest_index"])
        proposal_identity = source["model_proposal_core_identities"]["failure_gateway"]
        proposal_core = {
            name: copy.deepcopy(payload[name])
            for name in _OBSERVER_FAILURE_PROPOSAL_CORE_FIELDS
        }
        failures = payload.get("failures")
        if not isinstance(failures, list) or len(failures) != len(
            _GATEWAY_OBSERVER_FAILURE_COMPONENTS
        ):
            _fail(code)
        receipt_digests: list[str] = []
        for expected, item in zip(
            _GATEWAY_OBSERVER_FAILURE_COMPONENTS,
            failures,
            strict=True,
        ):
            row = _strict(
                item,
                (
                    "component",
                    "failure_observed",
                    "failure_receipt_sha256",
                    "alternative_available",
                    "alternative_attempted",
                    "alternative_receipt_sha256",
                ),
                code,
            )
            failure_sha256 = _digest(row["failure_receipt_sha256"], code)
            if (
                row["component"] != expected
                or row["failure_observed"] is not True
                or type(row["alternative_available"]) is not bool
                or type(row["alternative_attempted"]) is not bool
                or row["alternative_available"] != row["alternative_attempted"]
            ):
                _fail(code)
            receipt_digests.append(failure_sha256)
            alternative = row["alternative_receipt_sha256"]
            if row["alternative_available"]:
                receipt_digests.append(_digest(alternative, code))
            elif alternative is not None:
                _fail(code)
            if expected in {"tool", "browser"} and not row["alternative_available"]:
                _fail(code)
        if (
            _sha256_json(proposal_core) != proposal_identity["core_sha256"]
            or set(receipt_digests) - source_digests
            or payload.get("transcript_sha256") != terminal["transcript_sha256"]
            or payload.get("model_retained_tool_control") is not True
            or not source["tool_call_records"]
        ):
            _fail(code)
        readiness = source["collector_readiness_identity"]
        terminal_source = _sha256_json({
            "session_create_request_id": terminal["session_create_request_id"],
            "chat_stream_request_id": terminal["chat_stream_request_id"],
            "api_run_id": terminal["api_run_id"],
            "api_message_id": terminal["api_message_id"],
        })
        return (
            NativeEvidenceBinding(
                kind="gateway_observer_frame_chain",
                source_identity_sha256=readiness["service_identity_sha256"],
                artifact_sha256=source["frame_chain_head_sha256"],
                verification_receipt_sha256=verification_sha256,
            ),
            NativeEvidenceBinding(
                kind="authenticated_api_terminal_event",
                source_identity_sha256=terminal_source,
                artifact_sha256=terminal["identity_sha256"],
                verification_receipt_sha256=verification_sha256,
            ),
            NativeEvidenceBinding(
                kind="failure_probe_receipts",
                source_identity_sha256=source["frame_chain_head_sha256"],
                artifact_sha256=_sha256_json(receipt_digests),
                verification_receipt_sha256=verification_sha256,
            ),
        )

    def collect(
        self,
        *,
        slot: str,
        payload: Mapping[str, Any],
    ) -> Sequence[NativeEvidenceBinding]:
        exact = self._exact_payload(slot, payload)
        source, source_file_sha256 = self._load(payload=exact)
        membership = source["slot_membership"][slot]
        if not membership["frame_sequences"]:
            _fail("gateway_observer_source_evidence_invalid")
        verification_sha256 = _sha256_json({
            "schema": "muncho-gateway-observer-source-verification.v1",
            "slot": slot,
            "payload_sha256": _sha256_json(exact),
            "source_file_sha256": source_file_sha256,
            "source_projection_sha256": source["projection_sha256"],
            "slot_membership_sha256": _sha256_json(membership),
        })
        collector = {
            "runtime": self._runtime_bindings,
            "workspace_gateway": self._workspace_bindings,
            "failure_gateway": self._failure_bindings,
        }[slot]
        bindings = tuple(collector(exact, source, verification_sha256))
        if tuple(item.kind for item in bindings) != SLOT_NATIVE_BINDING_KINDS[slot]:
            _fail("gateway_observer_source_evidence_invalid")
        return bindings


class GatewayObserverCleanupNativeCollector:
    """Verify root cleanup facts while the credential-blind signer is live."""

    def __init__(
        self,
        *,
        config: ProducerConfig,
        foundation: Mapping[str, Any],
        service_state_reader: Callable[[str], Mapping[str, Any]] = (
            _cleanup_service_state
        ),
        cleanup_facts_reader: Callable[
            [Path, int], tuple[Mapping[str, Any], str]
        ] = _read_cleanup_facts,
        production_diff_reader: Callable[[Path, int], Mapping[str, Any]] = (
            _read_cleanup_production_diff
        ),
    ) -> None:
        if (
            config.role != "gateway_observer"
            or not callable(service_state_reader)
            or not callable(cleanup_facts_reader)
            or not callable(production_diff_reader)
        ):
            _fail("gateway_observer_cleanup_collector_invalid")
        validate_producer_config_binding(config, foundation)
        self.config = config
        self.foundation = copy.deepcopy(dict(foundation))
        self.service_state_reader = service_state_reader
        self.cleanup_facts_reader = cleanup_facts_reader
        self.production_diff_reader = production_diff_reader

    def _load_facts(self, run_id: str) -> tuple[Mapping[str, Any], str]:
        code = "gateway_observer_cleanup_evidence_invalid"
        value, facts_file_sha256 = self.cleanup_facts_reader(
            self.config.receipt_base_root / run_id,
            self.config.service_gid,
        )
        fields = {
            "schema",
            "revision",
            "capability_plan_sha256",
            "full_canary_plan_sha256",
            "non_observer_stop_order",
            "non_observer_service_states",
            "credential_consumer_stop_proof",
            "observer_signer_identity",
            "retirements",
            "retirement_receipt_sha256s",
            "credential_absence",
            "bitrix_receipt_key_retirement",
            "bitrix_receipt_key_absence",
            "browser_session_retirement",
            "isolated_worker_lease_cleanup",
            "observed_at_unix_ms",
            "facts_sha256",
        }
        if set(value) != fields:
            _fail(code)
        unsigned = {
            key: copy.deepcopy(item)
            for key, item in value.items()
            if key != "facts_sha256"
        }
        if (
            value["schema"] != "muncho-production-capability-cleanup-facts.v1"
            or value["revision"] != self.config.release_sha
            or value["capability_plan_sha256"] != self.config.capability_plan_sha256
            or value["full_canary_plan_sha256"] != self.config.full_canary_plan_sha256
            or value["facts_sha256"] != _sha256_json(unsigned)
            or type(value["observed_at_unix_ms"]) is not int
            or value["observed_at_unix_ms"] <= 0
        ):
            _fail(code)
        _digest(facts_file_sha256, code)
        return value, facts_file_sha256

    def _validate_facts(
        self,
        facts: Mapping[str, Any],
    ) -> Mapping[str, Mapping[str, Any]]:
        code = "gateway_observer_cleanup_evidence_invalid"
        order = _CLEANUP_NON_OBSERVER_SERVICE_UNITS
        states = facts.get("non_observer_service_states")
        if (
            facts.get("non_observer_stop_order") != list(order)
            or not isinstance(states, Mapping)
            or set(states) != set(order)
        ):
            _fail(code)
        observed_states: dict[str, Mapping[str, Any]] = {}
        for unit in order:
            current = self.service_state_reader(unit)
            if (
                not isinstance(current, Mapping)
                or dict(current) != states[unit]
                or not _cleanup_service_stopped(current)
            ):
                _fail(code)
            observed_states[unit] = copy.deepcopy(dict(current))
        observer_state = self.service_state_reader(_CLEANUP_OBSERVER_UNIT)
        observer = facts.get("observer_signer_identity")
        if not isinstance(observer, Mapping) or not isinstance(observer_state, Mapping):
            _fail(code)
        observer_fields = {
            "role",
            "service_unit",
            "live",
            "signing_only",
            "credential_read_access",
            "service_state_sha256",
            "producer_foundation_sha256",
            "unit_bundle_manifest_sha256",
            "credential_inaccessibility_contract_sha256",
        }
        foundation_sha256 = producer_foundation_sha256(self.foundation)
        owner = self.foundation["owner_authority"]
        bundle = render_producer_units(
            foundation=self.foundation,
            pinned_owner_public_key_ed25519_hex=(owner["public_key_ed25519_hex"]),
            pinned_owner_public_key_source_sha256=(owner["public_key_source_sha256"]),
            role_identities=_cleanup_role_identities(self.foundation),
        )
        inaccessibility = {
            "paths": list(PRODUCER_INACCESSIBLE_CREDENTIAL_PATHS),
            "applies_to_roles": list(ENDPOINT_ROLES),
            "unit_hash_bound": True,
            "cleanup_observer_has_no_credential_read_access": True,
        }
        if (
            set(observer) != observer_fields
            or observer["role"] != "gateway_observer"
            or observer["service_unit"] != _CLEANUP_OBSERVER_UNIT
            or observer["live"] is not True
            or observer["signing_only"] is not True
            or observer["credential_read_access"] is not False
            or not _cleanup_observer_live(observer_state)
            or observer["service_state_sha256"] != _sha256_json(dict(observer_state))
            or observer["producer_foundation_sha256"] != foundation_sha256
            or observer["unit_bundle_manifest_sha256"]
            != bundle.manifest["manifest_sha256"]
            or observer["credential_inaccessibility_contract_sha256"]
            != _sha256_json(inaccessibility)
        ):
            _fail(code)

        proof = facts.get("credential_consumer_stop_proof")
        proof_fields = {
            "schema",
            "plan_sha256",
            "non_observer_stop_order",
            "non_observer_services_state_sha256",
            "all_credential_consumers_stopped",
            "observer_service_unit",
            "observer_state_sha256",
            "observer_live_signing_only",
            "observer_credential_read_access",
            "producer_foundation_sha256",
            "unit_bundle_manifest_sha256",
            "credential_inaccessibility_contract_sha256",
            "observed_at_unix",
            "secret_material_recorded",
            "secret_digest_recorded",
            "stop_proof_sha256",
        }
        if not isinstance(proof, Mapping) or set(proof) != proof_fields:
            _fail(code)
        proof_unsigned = {
            key: copy.deepcopy(item)
            for key, item in proof.items()
            if key != "stop_proof_sha256"
        }
        if (
            proof["schema"]
            != "muncho-production-capability-credential-consumer-stop-proof.v1"
            or proof["plan_sha256"] != self.config.capability_plan_sha256
            or proof["non_observer_stop_order"] != list(order)
            or proof["non_observer_services_state_sha256"]
            != _sha256_json(observed_states)
            or proof["all_credential_consumers_stopped"] is not True
            or proof["observer_service_unit"] != _CLEANUP_OBSERVER_UNIT
            or proof["observer_state_sha256"] != observer["service_state_sha256"]
            or proof["observer_live_signing_only"] is not True
            or proof["observer_credential_read_access"] is not False
            or proof["producer_foundation_sha256"] != foundation_sha256
            or proof["unit_bundle_manifest_sha256"]
            != bundle.manifest["manifest_sha256"]
            or proof["credential_inaccessibility_contract_sha256"]
            != _sha256_json(inaccessibility)
            or proof["secret_material_recorded"] is not False
            or proof["secret_digest_recorded"] is not False
            or proof["stop_proof_sha256"] != _sha256_json(proof_unsigned)
            or type(proof["observed_at_unix"]) is not int
            or proof["observed_at_unix"] * 1000 > facts["observed_at_unix_ms"]
        ):
            _fail(code)

        retirements = facts.get("retirements")
        digests = facts.get("retirement_receipt_sha256s")
        absence = facts.get("credential_absence")
        if (
            not isinstance(retirements, Mapping)
            or not isinstance(digests, Mapping)
            or not isinstance(absence, Mapping)
            or set(retirements) != set(_CLEANUP_CREDENTIAL_BINDINGS)
            or set(digests) != set(_CLEANUP_CREDENTIAL_BINDINGS)
            or set(absence) != set(_CLEANUP_CREDENTIAL_BINDINGS)
        ):
            _fail(code)
        for binding in _CLEANUP_CREDENTIAL_BINDINGS:
            retirement = retirements[binding]
            absent = absence[binding]
            if not isinstance(retirement, Mapping) or not isinstance(absent, Mapping):
                _fail(code)
            retirement_unsigned = {
                key: copy.deepcopy(item)
                for key, item in retirement.items()
                if key != "receipt_sha256"
            }
            if (
                retirement.get("credential_binding") != binding
                or retirement.get("service_stop_proof_sha256")
                != proof["stop_proof_sha256"]
                or retirement.get("receipt_sha256") != _sha256_json(retirement_unsigned)
                or digests[binding] != retirement.get("receipt_sha256")
                or absent.get("path") != retirement.get("target_path")
                or absent.get("absent") is not True
            ):
                _fail(code)

        key_retirement = facts.get("bitrix_receipt_key_retirement")
        key_absence = facts.get("bitrix_receipt_key_absence")
        if not isinstance(key_retirement, Mapping) or not isinstance(
            key_absence, Mapping
        ):
            _fail(code)
        key_unsigned = {
            key: copy.deepcopy(item)
            for key, item in key_retirement.items()
            if key != "receipt_sha256"
        }
        if (
            key_retirement.get("service_stop_proof_sha256")
            != proof["stop_proof_sha256"]
            or key_retirement.get("receipt_sha256") != _sha256_json(key_unsigned)
            or key_absence.get("both_pair_members_absent") is not True
        ):
            _fail(code)
        for field in (
            "browser_session_retirement",
            "isolated_worker_lease_cleanup",
        ):
            item = facts.get(field)
            if (
                not isinstance(item, Mapping)
                or item.get("empty") is not True
                or item.get("retired") is not True
                or item.get("secret_material_recorded") is not False
            ):
                _fail(code)
        return observed_states

    def collect(
        self,
        *,
        slot: str,
        payload: Mapping[str, Any],
    ) -> Sequence[NativeEvidenceBinding]:
        code = "gateway_observer_cleanup_evidence_invalid"
        if slot != "cleanup" or not isinstance(payload, Mapping):
            _fail(code)
        run_id = payload.get("run_id")
        if not isinstance(run_id, str):
            _fail(code)
        facts, facts_file_sha256 = self._load_facts(run_id)
        observed_states = self._validate_facts(facts)
        diff = self.production_diff_reader(
            self.config.receipt_base_root / run_id,
            self.config.service_gid,
        )
        if not isinstance(diff, Mapping):
            _fail(code)
        diff_fields = {
            "schema",
            "run_id",
            "canary_revision",
            "capability_plan_sha256",
            "full_canary_plan_sha256",
            "fixture_sha256",
            "target",
            "before_envelope_sha256",
            "after_envelope_sha256",
            "before_observation_sha256",
            "after_observation_sha256",
            "before_observed_at_unix_ms",
            "after_observed_at_unix_ms",
            "static_before_sha256",
            "static_after_sha256",
            "changed_surfaces",
            "surface_diffs",
            "expected_change_contract_sha256",
            "unexpected_change_count",
            "production_mutation_observed",
            "secret_material_recorded",
            "secret_digest_recorded",
            "semantic_job_content_recorded",
            "diff_sha256",
        }
        diff_sha256 = _digest(diff.get("diff_sha256"), code)
        surfaces = diff.get("surface_diffs")
        if (
            set(diff) != diff_fields
            or diff.get("schema") != PRODUCTION_DIFF_OBSERVATION_SCHEMA
            or diff.get("canary_revision") != self.config.release_sha
            or diff.get("capability_plan_sha256") != self.config.capability_plan_sha256
            or diff.get("full_canary_plan_sha256")
            != self.config.full_canary_plan_sha256
            or diff.get("fixture_sha256") != payload.get("fixture_sha256")
            or diff.get("run_id") != run_id
            or not isinstance(diff.get("target"), Mapping)
            or diff.get("changed_surfaces") != []
            or not isinstance(surfaces, Mapping)
            or set(surfaces) != set(PRODUCTION_DIFF_CATEGORIES)
            or any(
                not isinstance(surfaces[name], Mapping)
                or set(surfaces[name]) != {"before_sha256", "after_sha256", "changed"}
                or surfaces[name].get("before_sha256")
                != surfaces[name].get("after_sha256")
                or surfaces[name].get("changed") is not False
                for name in PRODUCTION_DIFF_CATEGORIES
            )
            or diff.get("static_before_sha256") != diff.get("static_after_sha256")
            or diff.get("unexpected_change_count") != 0
            or diff.get("production_mutation_observed") is not False
            or diff.get("secret_material_recorded") is not False
            or diff.get("secret_digest_recorded") is not False
            or diff.get("semantic_job_content_recorded") is not False
            or diff_sha256
            != _sha256_json({
                key: item for key, item in diff.items() if key != "diff_sha256"
            })
            or type(diff.get("before_observed_at_unix_ms")) is not int
            or type(diff.get("after_observed_at_unix_ms")) is not int
            or not diff["before_observed_at_unix_ms"]
            < diff["after_observed_at_unix_ms"]
            <= facts["observed_at_unix_ms"]
        ):
            _fail(code)
        for field in (
            "before_envelope_sha256",
            "after_envelope_sha256",
            "before_observation_sha256",
            "after_observation_sha256",
            "static_before_sha256",
            "static_after_sha256",
            "expected_change_contract_sha256",
        ):
            _digest(diff[field], code)
        for name in PRODUCTION_DIFF_CATEGORIES:
            _digest(surfaces[name]["before_sha256"], code)
            _digest(surfaces[name]["after_sha256"], code)
        proof = facts["credential_consumer_stop_proof"]
        expected = {
            "schema": "muncho-production-capability-cleanup.v1",
            "run_id": run_id,
            "release_sha": self.config.release_sha,
            "fixture_sha256": payload.get("fixture_sha256"),
            "observed_at_unix_ms": facts["observed_at_unix_ms"],
            "non_observer_service_units": facts["non_observer_stop_order"],
            "non_observer_services_stopped": True,
            "non_observer_services_state_sha256": proof[
                "non_observer_services_state_sha256"
            ],
            "gateway_observer_signer_identity": facts["observer_signer_identity"],
            "credential_consumer_stop_proof": proof,
            "credential_leases": list(_CLEANUP_CREDENTIAL_BINDINGS),
            "credential_leases_retired": True,
            "retirements": facts["retirements"],
            "retirement_receipt_sha256s": facts["retirement_receipt_sha256s"],
            "credential_absence": facts["credential_absence"],
            "credentials_absent": True,
            "bitrix_receipt_key_retirement": facts["bitrix_receipt_key_retirement"],
            "bitrix_receipt_key_absence": facts["bitrix_receipt_key_absence"],
            "discord_credential_topology": dict(_CLEANUP_DISCORD_CREDENTIAL_TOPOLOGY),
            "browser_session_retired": True,
            "isolated_worker_lease_cleanup_verified": True,
            "production_diff_sha256": diff_sha256,
        }
        if dict(payload) != expected:
            _fail(code)
        systemd_source = _sha256_json({
            "collector": "systemctl_show_fixed_properties",
            "service_units": facts["non_observer_stop_order"],
        })
        facts_source = _sha256_json({
            "schema": facts["schema"],
            "facts_file_sha256": facts_file_sha256,
            "uid": 0,
            "gid": self.config.service_gid,
            "mode": "0440",
        })
        state_verification = _sha256_json(observed_states)
        retirements = facts["retirements"]
        absence = facts["credential_absence"]
        key_retirement = facts["bitrix_receipt_key_retirement"]
        key_absence = facts["bitrix_receipt_key_absence"]
        binding_values = (
            (
                "systemd_non_observer_services_stopped_state",
                systemd_source,
                proof["non_observer_services_state_sha256"],
                state_verification,
            ),
            (
                "gateway_observer_cleanup_signer_live_identity",
                systemd_source,
                facts["observer_signer_identity"]["service_state_sha256"],
                proof["stop_proof_sha256"],
            ),
            *(
                (
                    kind,
                    facts_source,
                    retirements[binding]["receipt_sha256"],
                    proof["stop_proof_sha256"],
                )
                for kind, binding in (
                    ("api_control_credential_retirement_journal", "api_control"),
                    (
                        "routeback_credential_retirement_journal",
                        "discord_canonical_routeback_bot_token",
                    ),
                    (
                        "connector_credential_retirement_journal",
                        "discord_public_session_bot_token",
                    ),
                    ("codex_credential_retirement_journal", "openai_codex"),
                    ("mac_ops_credential_retirement_journal", "mac_ops_gitlab"),
                    (
                        "bitrix_operational_edge_credential_retirement_journal",
                        "bitrix_operational_edge_webhook",
                    ),
                )
            ),
            (
                "all_six_credentials_absent_readback",
                facts_source,
                _sha256_json(absence),
                facts["facts_sha256"],
            ),
            (
                "bitrix_receipt_key_pair_retirement_journal",
                facts_source,
                key_retirement["receipt_sha256"],
                proof["stop_proof_sha256"],
            ),
            (
                "bitrix_receipt_key_pair_absence_readback",
                facts_source,
                _sha256_json(key_absence),
                facts["facts_sha256"],
            ),
            (
                "browser_session_retirement",
                facts_source,
                _sha256_json(facts["browser_session_retirement"]),
                facts["facts_sha256"],
            ),
            (
                "isolated_worker_lease_cleanup",
                facts_source,
                _sha256_json(facts["isolated_worker_lease_cleanup"]),
                facts["facts_sha256"],
            ),
            (
                "production_diff_observation",
                _sha256_json({
                    "source": "owner_authenticated_remote_production_observation",
                    "target": diff.get("target"),
                }),
                diff_sha256,
                _sha256_json(diff),
            ),
        )
        return tuple(
            NativeEvidenceBinding(
                kind=kind,
                source_identity_sha256=source,
                artifact_sha256=artifact,
                verification_receipt_sha256=verification,
            )
            for kind, source, artifact, verification in binding_values
        )


class _CanonicalWriterCollector:
    """Fixed slot table; the slot is already sealed in ProducerConfig."""

    def __init__(
        self,
        *,
        publications: Any,
        bitrix: BitrixWriterNativeCollector,
    ) -> None:
        self.publications = publications
        self.bitrix = bitrix

    def collect(
        self,
        *,
        slot: str,
        payload: Mapping[str, Any],
    ) -> Sequence[NativeEvidenceBinding]:
        if slot == "bitrix_writer":
            return self.bitrix.collect(slot=slot, payload=payload)
        return self.publications.collect(slot=slot, payload=payload)


class _GatewayObserverCollector:
    """Fixed mechanical split between live source projection and cleanup."""

    def __init__(
        self,
        *,
        source: GatewayObserverSourceNativeCollector,
        cleanup: GatewayObserverCleanupNativeCollector,
    ) -> None:
        self.source = source
        self.cleanup = cleanup

    def collect(
        self,
        *,
        slot: str,
        payload: Mapping[str, Any],
    ) -> Sequence[NativeEvidenceBinding]:
        if slot == "cleanup":
            return self.cleanup.collect(slot=slot, payload=payload)
        return self.source.collect(slot=slot, payload=payload)


def build_role_native_collector(
    config: ProducerConfig,
    foundation: Mapping[str, Any],
) -> Any:
    """Construct the exact role-local collector, with no task-text routing."""

    from gateway.operational_edge_client import (
        OperationalEdgeClient,
        OperationalEdgeClientConfig,
    )

    validate_producer_config_binding(config, foundation)
    role = config.role
    publication = RoleOwnedNativePublicationCollector(
        role=role,
        uid=config.service_uid,
        gid=config.service_gid,
        partial_kinds=(
            {"bitrix_writer": ("canonical_writer_handoff_events",)}
            if role == "canonical_writer"
            else None
        ),
    )
    if role == "gateway_observer":
        return _GatewayObserverCollector(
            source=GatewayObserverSourceNativeCollector(
                config=config,
                foundation=foundation,
            ),
            cleanup=GatewayObserverCleanupNativeCollector(
                config=config,
                foundation=foundation,
            ),
        )
    if role == "discord_edge":
        from gateway.canonical_writer_client import (
            ExactServerMainPidAuthorizer,
            SystemctlServerMainPidProvider,
        )
        from gateway.discord_edge_client import DiscordEdgeClient
        from gateway.discord_guild_history_client import (
            privileged_discord_guild_history_client,
        )

        contract = foundation["discord_edge_evidence_contract"]
        edge_key, edge_public_hex, edge_file_sha256 = _load_public_key(
            producer_discord_edge_public_key_projection_path(role),
            uid=config.service_uid,
            gid=config.service_gid,
            mode=0o400,
        )
        if (
            _sha256_bytes(bytes.fromhex(edge_public_hex))
            != contract["receipt_public_key_id"]
            or edge_file_sha256 != contract["receipt_public_key_file_sha256"]
        ):
            _fail("discord_edge_public_key_projection_invalid")
        catalog_raw, _catalog_item = _stable_read(
            producer_probe_catalog_projection_path(role),
            maximum=512 * 1024,
            uid=config.service_uid,
            gid=config.service_gid,
            mode=0o400,
        )
        edge_authorizer = ExactServerMainPidAuthorizer(
            server_unit=contract["edge_service_unit"],
            expected_server_uid=contract["edge_service_uid"],
            main_pid_provider=SystemctlServerMainPidProvider(),
        )
        return DiscordEdgeNativeCollector(
            edge_client=DiscordEdgeClient(
                contract["edge_socket_path"],
                server_authorizer=edge_authorizer,
            ),
            history_client=privileged_discord_guild_history_client(),
            edge_public_key=edge_key,
            catalog=validate_probe_catalog(
                _strict_json(catalog_raw, "probe_catalog_invalid")
            ),
            contract=contract,
        )
    if role not in {"business_edge", "canonical_writer"}:
        return publication
    contract = foundation["bitrix_operational_edge_contract"]
    identity = contract["identity_bootstrap"]
    receipt_key = contract["receipt_key_contract"]
    bitrix_config = OperationalEdgeClientConfig(
        domain="bitrix",
        socket_path=Path("/run/muncho-operational-edge/bitrix/edge.sock"),
        service_unit=BITRIX_OPERATIONAL_EDGE_SERVICE_UNIT,
        service_uid=identity["service_uid"],
        service_gid=identity["service_gid"],
        socket_gid=identity["socket_client_gid"],
        probe_uid=config.service_uid,
        probe_gid=config.service_gid,
        probe_supplementary_gids=(identity["socket_client_gid"],),
        receipt_public_key_file=Path(receipt_key["public_path"]),
        receipt_key_id=receipt_key["public_key_id"],
    )
    client = OperationalEdgeClient(bitrix_config)
    if role == "business_edge":
        return BitrixOperationalEdgeNativeCollector(
            client,
            release_revision=config.release_sha,
            receipt_key_id=receipt_key["public_key_id"],
        )
    from gateway.canonical_writer_activation import WRITER_UNIT, WRITER_USER
    from gateway.canonical_writer_boundary import DEFAULT_SOCKET_PATH
    from gateway.canonical_writer_client import (
        CanonicalWriterClient,
        ExactServerMainPidAuthorizer,
        SystemctlServerMainPidProvider,
    )

    try:
        writer_uid = pwd.getpwnam(WRITER_USER).pw_uid
    except KeyError as exc:
        raise CapabilityProducerError(
            "canonical_writer_service_identity_unavailable"
        ) from exc
    catalog_path = producer_probe_catalog_projection_path(role)
    catalog_raw, _catalog_item = _stable_read(
        catalog_path,
        maximum=512 * 1024,
        uid=config.service_uid,
        gid=config.service_gid,
        mode=0o400,
    )
    catalog = validate_probe_catalog(_strict_json(catalog_raw, "probe_catalog_invalid"))
    writer_client = CanonicalWriterClient(
        DEFAULT_SOCKET_PATH,
        server_authorizer=ExactServerMainPidAuthorizer(
            server_unit=WRITER_UNIT,
            expected_server_uid=writer_uid,
            main_pid_provider=SystemctlServerMainPidProvider(),
        ),
    )
    writer_projection = CanonicalWriterProjectionNativeCollector(
        client=writer_client,
        catalog=catalog,
        release_sha=config.release_sha,
        capability_plan_sha256=config.capability_plan_sha256,
        full_canary_plan_sha256=config.full_canary_plan_sha256,
        source_identity={
            "service_unit": WRITER_UNIT,
            "service_uid": writer_uid,
            "socket_path": str(DEFAULT_SOCKET_PATH),
            "peer_authorization": "exact_current_systemd_main_pid_each_call",
            "operation": "projection.read_events",
        },
    )
    bitrix_writer = BitrixWriterNativeCollector(
        client,
        release_revision=config.release_sha,
        receipt_key_id=receipt_key["public_key_id"],
        canonical_writer_collector=writer_projection,
    )
    return _CanonicalWriterCollector(
        publications=writer_projection,
        bitrix=bitrix_writer,
    )


def _read_canonical_stdin(maximum: int = 2 * 1024 * 1024) -> Mapping[str, Any]:
    raw = sys.stdin.buffer.read(maximum + 1)
    if not raw or len(raw) > maximum or sys.stdin.buffer.read(1):
        _fail("producer_input_invalid")
    value = _strict_json(raw, "producer_input_invalid")
    if raw != _canonical_bytes(value):
        _fail("producer_input_invalid")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install the signed capability-canary producer foundation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-foundation")
    sub.add_parser("install-foundation")
    sub.add_parser("preflight")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if (
            sys.platform != "linux" or os.geteuid() != 0
        ):  # windows-footgun: ok — Linux production/canary boundary
            _fail("producer_root_linux_required")
        from gateway.canonical_capability_canary_runtime import (
            load_capability_plan,
            validate_plan_against_full,
        )
        from gateway.canonical_full_canary_runtime import load_full_canary_plan

        plan = load_capability_plan()
        full_plan = load_full_canary_plan()
        validate_plan_against_full(plan, full_plan)
        if args.command == "prepare-foundation":
            result = prepare_producer_foundation(
                _read_canonical_stdin(),
                plan=plan,
                full_plan=full_plan,
            )
        elif args.command == "install-foundation":
            result = install_prepared_producer_foundation(
                _read_canonical_stdin(),
                preparation=load_foundation_preparation(),
            )
        else:
            result = validate_installed_producer_foundation(
                plan=plan,
                full_plan=full_plan,
            )
    except Exception as exc:
        result = {
            "schema": "muncho-capability-producer-foundation-command.v1",
            "ok": False,
            "failure_code": (
                exc.code
                if isinstance(exc, CapabilityProducerError)
                else "producer_foundation_command_failed_closed"
            ),
        }
        sys.stdout.buffer.write(_canonical_bytes(result) + b"\n")
        return 2
    sys.stdout.buffer.write(_canonical_bytes(result) + b"\n")
    return 0


__all__ = [
    "CANONICAL_ROUTEBACK_EVENT_BINDING_SCHEMA",
    "DEFAULT_KEY_BOOTSTRAP_RECEIPT",
    "DEFAULT_FOUNDATION_INSTALL_RECEIPT",
    "DEFAULT_FOUNDATION_PREPARATION_PATH",
    "DEFAULT_PRODUCER_IDENTITY_FOUNDATION_ROOT",
    "DEFAULT_NATIVE_ROOT",
    "KEY_BOOTSTRAP_SCHEMA",
    "FOUNDATION_INSTALL_RECEIPT_SCHEMA",
    "FOUNDATION_INSTALL_REQUEST_SCHEMA",
    "FOUNDATION_PREPARATION_SCHEMA",
    "FOUNDATION_PREPARE_REQUEST_SCHEMA",
    "SERVICE_IDENTITY_FOUNDATION_SCHEMA",
    "PRODUCER_HOST_IDENTITY_SCHEMA",
    "PRODUCER_IDENTITY_FOUNDATION_SCHEMA",
    "GATEWAY_OBSERVER_SOURCE_PROJECTION_FILENAME",
    "GATEWAY_OBSERVER_SOURCE_PROJECTION_SCHEMA",
    "GATEWAY_OBSERVER_PROPOSAL_CORE_SCHEMA",
    "GATEWAY_OBSERVER_PROPOSAL_EVENT_TYPE",
    "GATEWAY_OBSERVER_PROPOSAL_IDENTITY_SCHEMA",
    "extract_gateway_observer_model_proposal_cores",
    "NATIVE_PUBLICATION_SCHEMA",
    "CanonicalWriterProjectionNativeCollector",
    "DiscordEdgeNativeCollector",
    "GatewayObserverCleanupNativeCollector",
    "GatewayObserverSourceNativeCollector",
    "ProducerKeyBootstrap",
    "PRODUCER_RECEIPT_WRITER_GROUP",
    "PRODUCER_RECEIPT_WRITER_GID",
    "PRODUCER_ROLE_ACCOUNTS",
    "PRODUCER_ROLE_NUMERIC_IDENTITIES",
    "ProducerRoleIdentity",
    "ProducerUnitBundle",
    "ProducerUnitIdentityContract",
    "RoleOwnedNativePublicationCollector",
    "UNIT_BUNDLE_SCHEMA",
    "bootstrap_producer_keys",
    "attest_foundation_service_identities",
    "build_api_terminal_event_identity",
    "build_gateway_observer_source_projection",
    "build_role_native_collector",
    "endpoint_contracts",
    "ensure_producer_role_identities",
    "install_prepared_producer_foundation",
    "load_foundation_preparation",
    "load_producer_identity_foundation_receipt",
    "planned_producer_role_identities",
    "prepare_producer_foundation",
    "producer_config_path",
    "producer_private_key_projection_path",
    "producer_private_key_source_path",
    "producer_discord_edge_public_key_projection_path",
    "producer_probe_catalog_projection_path",
    "producer_public_key_path",
    "producer_public_key_source_path",
    "producer_socket_path",
    "producer_host_identity_receipt",
    "publish_gateway_observer_source_projection",
    "render_producer_unit_identity_contract",
    "render_producer_units",
    "validate_foundation_preparation",
    "validate_gateway_observer_source_projection",
    "validate_installed_producer_foundation",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
