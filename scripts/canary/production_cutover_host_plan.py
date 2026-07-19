#!/usr/bin/env python3
"""Fixed, package-bound host staging and plan production for cutover.

``stage`` is the only mutating operation.  It creates the already-approved,
inert host inputs below the fixed root-owned staging tree and cleanly resumes
only when every existing byte is exact.  ``collect`` is read-only: it combines
those bytes, live public pre-state, and the initial collector receipt into the
seven fields accepted by the host-authority collector.  Neither operation
accepts a caller-authored path, identity, topology, transition, or cron plan.
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
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gateway import canonical_writer_bootstrap
from gateway import canonical_writer_production_cutover as cutover
from gateway import production_model_sovereignty_runtime as gateway_runtime
from gateway import production_secret_stager
from gateway.mac_ops_edge_client import DEFAULT_SERVICE_UNIT
from gateway.operational_edge_bootstrap import (
    validate_operational_edge_key_foundation,
)
from gateway.production_capability_prerequisites import (
    API_APPROVAL_CREDENTIAL_PATH,
    API_SERVER_CREDENTIAL_PATH,
    CODEX_AUTH_PATH,
    MAC_OPS_CONFIG_PATH,
    MAC_OPS_CREDENTIAL_PATH,
    MAC_OPS_JOURNAL_PATH,
    MAC_OPS_SOCKET_PATH,
    MAC_OPS_UNIT,
    PHASE_B_RECEIPT_PATH,
    PHASE_B_UNIT,
    PREREQUISITE_PATH,
    PUBLIC_CONNECTOR_CONFIG_PATH,
    PUBLIC_CONNECTOR_CREDENTIAL_PATH,
    PUBLIC_CONNECTOR_READINESS_PATH,
    PUBLIC_CONNECTOR_SOCKET_PATH,
    PUBLIC_CONNECTOR_UNIT,
    ROUTEBACK_EDGE_CONFIG_PATH,
    ROUTEBACK_EDGE_CREDENTIAL_PATH,
    ROUTEBACK_EDGE_READINESS_PATH,
    ROUTEBACK_EDGE_SOCKET_PATH,
    ROUTEBACK_EDGE_UNIT,
    TOPOLOGY_SCHEMA,
    packaged_prerequisite_contract_sha256,
    validate_production_capability_topology,
)
from gateway.production_capability_units import (
    render_production_mac_ops_config,
    render_production_routeback_config,
)
from gateway.support_ops_team_registry import (
    SKYVISION_APPROVED_OPERATIONAL_CHANNEL_IDS,
    SKYVISION_CONTROL_TOWER_CHANNEL_ID,
    SKYVISION_GUILD_ID,
    SKYVISION_NASI_AI_OPS_CHANNEL_ID,
)
from scripts.canary import package_production_cutover_artifacts as package
from scripts.canary import production_cutover_host_authority as host_authority


STAGING_SCHEMA = "muncho-production-cutover-fixed-host-staging.v1"
STAGING_RECEIPT_PATH = (
    package.CUTOVER_STAGED_ROOT / "host-plan-staging-receipt.json"
)
HOST_STAGING_ROOT = package.CUTOVER_STAGED_ROOT / "host"
KEY_STAGING_ROOT = package.CUTOVER_STAGED_ROOT / "keys"
SOURCE_GATEWAY_CONFIG_PATH = Path(
    "/opt/adventico-ai-platform/hermes-home/config.yaml"
)
SOURCE_WRITER_CONFIG_PATH = Path(
    "/etc/muncho/writer-activation/staged/writer.json"
)
SOURCE_CONNECTOR_TOKEN_PATH = Path(
    "/opt/adventico-ai-platform/hermes-home/.discord-token"
)
REVIEWED_CONNECTOR_UNIT_TEMPLATE = Path(
    "ops/muncho/systemd/muncho-discord-connector.service.in"
)
REVIEWED_CONNECTOR_CONFIG_TEMPLATE = Path(
    "ops/muncho/systemd/discord-public-connector.json.in"
)
REVIEWED_GATEWAY_DROP_IN = Path(
    "ops/muncho/systemd/hermes-cloud-gateway.discord-connector.conf"
)
MAX_FILE = 4 * 1024 * 1024
MAX_INPUT = 16 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
OWNER_RUNTIME_ARTIFACT_NAMES = frozenset({
    "gateway_unit",
    "writer_unit",
    "connector_unit",
    "gateway_config",
    "writer_config",
    "connector_config",
    "routeback_config",
    "mac_ops_config",
})
ROOT_VERIFIER_ARTIFACT_NAMES = frozenset({
    "api_bearer_verifier",
    "api_approval_verifier",
})
REVIEWED_RELEASE_ARTIFACT_NAMES = frozenset({
    "gateway_connector_drop_in",
})
RELEASE_SEALED_ARTIFACT_NAMES = frozenset(
    name
    for name, (_target, binding) in package.HOST_ARTIFACT_TARGETS.items()
    if binding == "release_sealed_payload"
)
HOST_ARTIFACT_SOURCE_PARTITIONS = (
    RELEASE_SEALED_ARTIFACT_NAMES,
    REVIEWED_RELEASE_ARTIFACT_NAMES,
    OWNER_RUNTIME_ARTIFACT_NAMES,
    ROOT_VERIFIER_ARTIFACT_NAMES,
)


class HostPlanProducerError(RuntimeError):
    """Stable, secret-free fixed producer failure."""


def _canonical(value: Any) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise HostPlanProducerError("host_plan_json_invalid") from exc
    if len(raw) > MAX_INPUT:
        raise HostPlanProducerError("host_plan_json_oversized")
    return raw


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _decode(raw: bytes) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, item in items:
            if name in result:
                raise HostPlanProducerError("host_plan_json_duplicate_key")
            result[name] = item
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except HostPlanProducerError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise HostPlanProducerError("host_plan_json_invalid") from exc
    if not isinstance(value, Mapping) or raw != _canonical(value):
        raise HostPlanProducerError("host_plan_json_not_canonical")
    return value


def _physical(logical: Path, filesystem_root: Path) -> Path:
    if not logical.is_absolute() or ".." in logical.parts:
        raise HostPlanProducerError("host_plan_path_invalid")
    if filesystem_root == Path("/"):
        return logical
    try:
        root = filesystem_root.resolve(strict=True)
    except OSError as exc:
        raise HostPlanProducerError("host_plan_test_root_invalid") from exc
    if not root.is_dir():
        raise HostPlanProducerError("host_plan_test_root_invalid")
    return root.joinpath(*logical.parts[1:])


def _identity(item: os.stat_result) -> tuple[int, ...]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_uid,
        item.st_gid,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _read_regular(
    logical: Path,
    *,
    filesystem_root: Path,
    uid: int | None = None,
    gid: int | None = None,
    modes: frozenset[int] | None = None,
) -> tuple[bytes, os.stat_result]:
    path = _physical(logical, filesystem_root)
    descriptor: int | None = None
    try:
        before = os.lstat(path)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= MAX_FILE
            or (uid is not None and before.st_uid != uid)
            or (gid is not None and before.st_gid != gid)
            or (
                modes is not None
                and stat.S_IMODE(before.st_mode) not in modes
            )
        ):
            raise HostPlanProducerError("host_plan_file_identity_invalid")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = MAX_FILE + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        reachable = os.lstat(path)
    except HostPlanProducerError:
        raise
    except OSError as exc:
        raise HostPlanProducerError("host_plan_file_unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        len(raw) != before.st_size
        or len(raw) > MAX_FILE
        or _identity(before) != _identity(opened)
        or _identity(before) != _identity(after)
        or _identity(before) != _identity(reachable)
    ):
        raise HostPlanProducerError("host_plan_file_changed")
    return raw, before


def _ensure_private_directory(
    logical: Path,
    *,
    filesystem_root: Path,
    uid: int,
    gid: int,
) -> None:
    path = _physical(logical, filesystem_root)
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise HostPlanProducerError("host_plan_directory_invalid") from exc
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise HostPlanProducerError("host_plan_directory_invalid") from exc
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != uid
        or observed.st_gid != gid
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        raise HostPlanProducerError("host_plan_directory_invalid")


def _create_or_validate(
    logical: Path,
    payload: bytes,
    *,
    filesystem_root: Path,
    uid: int,
    gid: int,
) -> None:
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_FILE:
        raise HostPlanProducerError("host_plan_payload_invalid")
    path = _physical(logical, filesystem_root)
    if not os.path.lexists(path):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        created = False
        try:
            try:
                descriptor = os.open(path, flags, 0o400)
                created = True
            except FileExistsError:
                descriptor = None
            if descriptor is None:
                observed, _state = _read_regular(
                    logical,
                    filesystem_root=filesystem_root,
                    uid=uid,
                    gid=gid,
                    modes=frozenset({0o400}),
                )
                if observed != payload:
                    raise HostPlanProducerError("host_plan_staging_conflict")
                return
            os.fchown(descriptor, uid, gid)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short host-plan staging write")
                view = view[written:]
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
                descriptor = None
            if created:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)
        parent = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    observed, _state = _read_regular(
        logical,
        filesystem_root=filesystem_root,
        uid=uid,
        gid=gid,
        modes=frozenset({0o400}),
    )
    if observed != payload:
        raise HostPlanProducerError("host_plan_staging_conflict")


def _release_root(revision: str) -> Path:
    return cutover.PRODUCTION_RELEASE_BASE / f"hermes-agent-{revision[:12]}"


def _target_discord_policy() -> Mapping[str, Any]:
    return {
        "allowed_guild_ids": [SKYVISION_GUILD_ID],
        "allowed_channel_ids": sorted(SKYVISION_APPROVED_OPERATIONAL_CHANNEL_IDS),
        "allowed_user_ids": [],
        "allowed_role_ids": [],
        "allow_all_users": False,
        "allow_bot_authors": False,
        "require_mention": True,
        "auto_thread": True,
        "thread_require_mention": False,
        "discord_dm_allowed": False,
        "free_response_channel_ids": sorted(
            {SKYVISION_CONTROL_TOWER_CHANNEL_ID, SKYVISION_NASI_AI_OPS_CHANNEL_ID}
        ),
        "public_only": False,
        "author_policy": "guild_acl",
    }


def _legacy_discord_policy() -> Mapping[str, Any]:
    """Reviewed public legacy membership; never sourced from a secret file."""

    return {
        "allowed_guild_ids": ["1282725267068157972"],
        "allowed_channel_ids": sorted(
            {
                "1504852355588423801",
                "1504852408227069993",
                "1504852444407140402",
                "1504852485083496561",
                "1504852553031221391",
                "1504852628373373028",
                "1505499746939174993",
                "1507239177350283274",
                "1507239385010016308",
                "1507239516409167942",
                "1510888721614901358",
            }
        ),
        "allowed_user_ids": sorted(
            {"1279454038731264061", "1282938967888498720"}
        ),
        "allowed_role_ids": sorted(
            {"1282725267068157972", "1505077218374586468"}
        ),
        "allow_all_users": False,
        "allow_bot_authors": False,
        "require_mention": True,
        "auto_thread": True,
        "thread_require_mention": False,
        "discord_dm_allowed": False,
        "free_response_channel_ids": sorted(
            {"1504852355588423801", "1505499746939174993"}
        ),
        "public_only": False,
        "author_policy": "exact_ids_or_roles",
    }


def _validate_reconciliation_intent(
    inputs: Mapping[str, Any], *, revision: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    legacy = _legacy_discord_policy()
    target = _target_discord_policy()
    intent = inputs["discord_reconciliation_intent"]
    if (
        intent["release_revision"] != revision
        or intent["legacy_public_policy_sha256"] != _sha(_canonical(legacy))
        or intent["target_public_policy_sha256"] != _sha(_canonical(target))
        or intent["reviewed_reconciliation"] is not True
    ):
        raise HostPlanProducerError("host_plan_reconciliation_intent_mismatch")
    return legacy, target


def _render_connector_unit(template: bytes, revision: str) -> bytes:
    marker = b"@EXACT_12_CHAR_SHA@"
    if template.count(marker) != 3:
        raise HostPlanProducerError("host_plan_connector_template_invalid")
    rendered = template.replace(marker, revision[:12].encode("ascii"))
    if b"@" in rendered or not rendered.endswith(b"\n"):
        raise HostPlanProducerError("host_plan_connector_template_invalid")
    return rendered


def _render_connector_config(
    template: bytes,
    *,
    inputs: Mapping[str, Any],
    target_policy: Mapping[str, Any],
) -> bytes:
    replacements = {
        b"@GATEWAY_UID@": str(inputs["gateway"]["uid"]).encode("ascii"),
        b"@CONNECTOR_UID@": str(inputs["connector"]["uid"]).encode("ascii"),
        b"@CONNECTOR_GID@": str(inputs["connector"]["gid"]).encode("ascii"),
    }
    rendered = template
    for marker, replacement in replacements.items():
        if rendered.count(marker) != 1:
            raise HostPlanProducerError("host_plan_connector_template_invalid")
        rendered = rendered.replace(marker, replacement)
    try:
        value = json.loads(rendered.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise HostPlanProducerError("host_plan_connector_template_invalid") from exc
    discord = value.get("discord") if isinstance(value, Mapping) else None
    if not isinstance(discord, Mapping):
        raise HostPlanProducerError("host_plan_connector_policy_invalid")
    rendered_discord = copy.deepcopy(dict(discord))
    for name, item in target_policy.items():
        rendered_discord[name] = copy.deepcopy(item)
    rendered_value = copy.deepcopy(dict(value))
    rendered_value["discord"] = rendered_discord
    if any(
        rendered_discord.get(name) != target_policy[name]
        for name in (
            "allowed_guild_ids",
            "allowed_channel_ids",
            "allowed_user_ids",
            "allowed_role_ids",
            "allow_all_users",
            "allow_bot_authors",
            "require_mention",
            "auto_thread",
            "thread_require_mention",
            "discord_dm_allowed",
            "free_response_channel_ids",
            "public_only",
            "author_policy",
        )
    ):
        raise HostPlanProducerError("host_plan_connector_policy_invalid")
    return _canonical(rendered_value)


def _render_writer_config(
    source: bytes,
    *,
    inputs: Mapping[str, Any],
) -> bytes:
    try:
        value = json.loads(
            source.decode("utf-8", errors="strict"),
            object_pairs_hook=lambda pairs: _reject_duplicate_pairs(pairs),
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise HostPlanProducerError("host_plan_writer_source_invalid") from exc
    if not isinstance(value, Mapping):
        raise HostPlanProducerError("host_plan_writer_source_invalid")
    try:
        canonical_writer_bootstrap._reject_embedded_secrets(value)
    except ValueError as exc:
        raise HostPlanProducerError("host_plan_writer_source_invalid") from exc
    expected_root = {
        "service", "database", "privileges", "discord_edge_authority"
    }
    service = value.get("service")
    database = value.get("database")
    authority = value.get("discord_edge_authority")
    target = inputs["target"]
    if (
        set(value) != expected_root
        or not isinstance(service, Mapping)
        or not isinstance(database, Mapping)
        or authority != {"enabled": False}
        or database.get("host") != target["sql_host"]
        or database.get("tls_server_name") != target["tls_server_name"]
        or database.get("port") != target["port"]
        or database.get("database") != target["database"]
        or database.get("user") != target["writer_login"]
    ):
        raise HostPlanProducerError("host_plan_writer_source_invalid")
    rendered = copy.deepcopy(dict(value))
    rendered_service = dict(rendered["service"])
    rendered_service.update(
        {
            "gateway_uid": inputs["gateway"]["uid"],
            "writer_uid": inputs["writer"]["uid"],
            "writer_gid": inputs["writer"]["gid"],
            "socket_gid": inputs["writer_client_group"]["gid"],
            "projector_gid": inputs["projector"]["gid"],
            "owner_discord_user_ids": [cutover.OWNER_DISCORD_USER_ID],
            "connection_timeout_seconds": 30.0,
            "max_connections": 8,
        }
    )
    rendered["service"] = rendered_service
    rendered["discord_edge_authority"] = {
        "enabled": True,
        "capability_private_key_file": str(
            cutover.WRITER_CAPABILITY_PRIVATE_KEY_PATH
        ),
        "edge_receipt_public_key_file": str(
            cutover.EDGE_RECEIPT_PUBLIC_KEY_PATH
        ),
        "edge_receipt_public_key_id": inputs[
            "discord_edge_receipt_public_key_id"
        ],
        "request_timeout_seconds": 15,
    }
    return _canonical(rendered)


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("duplicate JSON key")
        value[name] = item
    return value


def _render_writer_unit(
    *,
    revision: str,
    inputs: Mapping[str, Any],
) -> bytes:
    release = _release_root(revision)
    interpreter = release / ".venv/bin/python"
    lines = [
        "# Exact production Canonical Writer service.",
        f"# ReleaseRevision={revision}",
        "[Unit]",
        "Description=Muncho production Canonical Writer",
        "After=network-online.target muncho-canonical-writer-phase-b-readiness.service",
        "Wants=network-online.target",
        "Requires=muncho-canonical-writer-phase-b-readiness.service",
        "Before=hermes-cloud-gateway.service",
        f"AssertPathExists={interpreter}",
        f"AssertPathExists={release / '.codex-source-commit'}",
        "AssertPathExists=/etc/muncho-canonical-writer/writer.json",
        "",
        "[Service]",
        "Type=notify",
        "NotifyAccess=main",
        f"User={inputs['writer']['user']}",
        f"Group={inputs['writer']['group']}",
        f"SupplementaryGroups={inputs['projector']['group']}",
        "RuntimeDirectory=muncho-canonical-writer",
        "RuntimeDirectoryMode=0750",
        "RuntimeDirectoryPreserve=no",
        "StateDirectory=muncho-canonical-writer",
        "StateDirectoryMode=0700",
        f"WorkingDirectory={release}",
        "Environment=PYTHONNOUSERSITE=1",
        f"Environment=PYTHONPATH={release}",
        (
            f"ExecStart={interpreter} -B -P -s -m "
            "gateway.canonical_writer_bootstrap --config "
            "/etc/muncho-canonical-writer/writer.json"
        ),
        "Restart=on-failure",
        "RestartSec=5s",
        "TimeoutStartSec=180s",
        "TimeoutStopSec=60s",
        "KillMode=mixed",
        "LimitCORE=0",
        "UMask=0077",
        "NoNewPrivileges=yes",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
        "LockPersonality=yes",
        "MemoryDenyWriteExecute=yes",
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
        "ProcSubset=pid",
        "RemoveIPC=yes",
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "RestrictNamespaces=yes",
        "RestrictRealtime=yes",
        "RestrictSUIDSGID=yes",
        "SystemCallArchitectures=native",
        "IPAddressDeny=any",
        f"IPAddressAllow={inputs['database_ip']}",
        f"ReadOnlyPaths={release}",
        "ReadOnlyPaths=/etc/muncho-canonical-writer/writer.json",
        "ReadOnlyPaths=/etc/muncho/keys/writer-capability-private.pem",
        "ReadOnlyPaths=/etc/muncho/keys/discord-edge-receipt-public.pem",
        "ReadWritePaths=/run/muncho-canonical-writer",
        "ReadWritePaths=/var/lib/muncho-canonical-writer",
        "StandardOutput=journal",
        "StandardError=journal",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _build_capability_topology(
    *,
    descriptor: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    connector_unit_sha256: str,
    routeback_config_sha256: str,
    mac_ops_config_sha256: str,
    inputs: Mapping[str, Any],
) -> Mapping[str, Any]:
    fragments = descriptor["topology_fragments"]
    capability = descriptor["capability_bundle"]
    topology = {
        "schema": TOPOLOGY_SCHEMA,
        "prerequisite_receipt_path": str(PREREQUISITE_PATH),
        "collector_contract_sha256": packaged_prerequisite_contract_sha256(),
        "isolated_worker": copy.deepcopy(fragments["isolated_worker"]),
        "browser": copy.deepcopy(fragments["browser"]),
        "mac_ops": {
            "unit": MAC_OPS_UNIT,
            "fragment_sha256": _sha(payloads["mac_ops_unit"]),
            "config_sha256": mac_ops_config_sha256,
            "config_path": str(MAC_OPS_CONFIG_PATH),
            "socket_path": str(MAC_OPS_SOCKET_PATH),
            "credential_path": str(MAC_OPS_CREDENTIAL_PATH),
            "journal_path": str(MAC_OPS_JOURNAL_PATH),
        },
        "routeback_edge": {
            "unit": ROUTEBACK_EDGE_UNIT,
            "fragment_sha256": _sha(payloads["routeback_unit"]),
            "config_sha256": routeback_config_sha256,
            "config_path": str(ROUTEBACK_EDGE_CONFIG_PATH),
            "socket_path": str(ROUTEBACK_EDGE_SOCKET_PATH),
            "credential_path": str(ROUTEBACK_EDGE_CREDENTIAL_PATH),
            "readiness_path": str(ROUTEBACK_EDGE_READINESS_PATH),
        },
        "public_connector": {
            "unit": PUBLIC_CONNECTOR_UNIT,
            "fragment_sha256": connector_unit_sha256,
            "config_path": str(PUBLIC_CONNECTOR_CONFIG_PATH),
            "socket_path": str(PUBLIC_CONNECTOR_SOCKET_PATH),
            "credential_path": str(PUBLIC_CONNECTOR_CREDENTIAL_PATH),
            "readiness_path": str(PUBLIC_CONNECTOR_READINESS_PATH),
        },
        "phase_b": {
            "unit": PHASE_B_UNIT,
            "fragment_sha256": _sha(payloads["phase_b_unit"]),
            "readiness_path": str(PHASE_B_RECEIPT_PATH),
        },
        "codex_auth_file": str(CODEX_AUTH_PATH),
        "api_control_credential_file": str(API_SERVER_CREDENTIAL_PATH),
        "api_approval_credential_file": str(API_APPROVAL_CREDENTIAL_PATH),
        "gateway_identity": {
            "uid": inputs["gateway"]["uid"],
            "gid": inputs["gateway"]["gid"],
        },
    }
    if capability["units"][MAC_OPS_UNIT] != topology["mac_ops"][
        "fragment_sha256"
    ]:
        raise HostPlanProducerError("host_plan_sealed_topology_invalid")
    try:
        return copy.deepcopy(dict(validate_production_capability_topology(topology)))
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HostPlanProducerError("host_plan_topology_invalid") from exc


def _secret_projection(
    value: Any,
    *,
    inputs: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HostPlanProducerError("host_plan_secret_foundation_invalid")
    expected_fields = {
        "schema",
        "bearer_verifier_path",
        "bearer_verifier_sha256",
        "approval_verifier_path",
        "approval_verifier_sha256",
        "writer_private_path",
        "writer_public_key_id",
        "edge_private_path",
        "edge_public_key_id",
        "operational_edge_key_foundation",
        "operational_edge_key_foundation_sha256",
        "operational_edge_receipt_public_key_ids",
        "created",
        "source_secrets_retained_for_cutover",
        "private_content_or_digest_recorded",
        "secret_material_recorded",
        "secret_digest_recorded",
        "receipt_sha256",
    }
    operational = value.get("operational_edge_key_foundation")
    operational_ids = value.get("operational_edge_receipt_public_key_ids")
    created = value.get("created")
    unsigned = {
        name: item for name, item in value.items() if name != "receipt_sha256"
    }
    if (
        set(value) != expected_fields
        or value.get("schema") != production_secret_stager.STAGING_SCHEMA
        or value.get("bearer_verifier_path")
        != str(production_secret_stager.STAGED_API_BEARER_VERIFIER_PATH)
        or value.get("approval_verifier_path")
        != str(production_secret_stager.STAGED_API_APPROVAL_VERIFIER_PATH)
        or value.get("writer_private_path")
        != str(production_secret_stager.STAGED_WRITER_PRIVATE_KEY_PATH)
        or value.get("edge_private_path")
        != str(production_secret_stager.STAGED_EDGE_PRIVATE_KEY_PATH)
        or value.get("writer_public_key_id")
        != inputs["writer_capability_public_key_id"]
        or value.get("edge_public_key_id")
        != inputs["discord_edge_receipt_public_key_id"]
        or value.get("operational_edge_key_foundation_sha256")
        != inputs["operational_edge_key_foundation_sha256"]
        or operational_ids
        != inputs["operational_edge_receipt_public_key_ids"]
        or not isinstance(created, Mapping)
        or set(created)
        != {
            "bearer_verifier",
            "approval_verifier",
            "writer_private_key",
            "edge_private_key",
        }
        or any(type(item) is not bool for item in created.values())
        or not isinstance(operational, Mapping)
        or operational.get("receipt_sha256")
        != value.get("operational_edge_key_foundation_sha256")
        or any(
            _SHA256.fullmatch(str(value.get(name))) is None
            for name in (
                "bearer_verifier_sha256",
                "approval_verifier_sha256",
                "writer_public_key_id",
                "edge_public_key_id",
                "operational_edge_key_foundation_sha256",
                "receipt_sha256",
            )
        )
        or not isinstance(operational_ids, Mapping)
        or any(_SHA256.fullmatch(str(item)) is None for item in operational_ids.values())
        or value.get("source_secrets_retained_for_cutover") is not True
        or value.get("private_content_or_digest_recorded") is not False
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or value.get("receipt_sha256") != _sha(_canonical(unsigned))
    ):
        raise HostPlanProducerError("host_plan_secret_foundation_invalid")
    try:
        validate_operational_edge_key_foundation(
            operational,
            expected_writer_public_key_id=inputs[
                "writer_capability_public_key_id"
            ],
            key_root=production_secret_stager.KEY_STAGING_ROOT,
            trust_root=production_secret_stager.KEY_STAGING_ROOT,
            expected_uid=0,
            expected_gid=0,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HostPlanProducerError(
            "host_plan_secret_foundation_invalid"
        ) from exc
    return {
        "schema": value["schema"],
        "bearer_verifier_path": value["bearer_verifier_path"],
        "bearer_verifier_sha256": value["bearer_verifier_sha256"],
        "approval_verifier_path": value["approval_verifier_path"],
        "approval_verifier_sha256": value["approval_verifier_sha256"],
        "writer_private_path": value["writer_private_path"],
        "writer_public_key_id": value["writer_public_key_id"],
        "edge_private_path": value["edge_private_path"],
        "edge_public_key_id": value["edge_public_key_id"],
        "operational_edge_key_foundation": copy.deepcopy(dict(operational)),
        "operational_edge_key_foundation_sha256": value[
            "operational_edge_key_foundation_sha256"
        ],
        "operational_edge_receipt_public_key_ids": copy.deepcopy(
            dict(operational_ids)
        ),
        "private_content_or_digest_recorded": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }


def _staged_rows(
    *,
    contract: Mapping[str, Any],
    filesystem_root: Path,
    uid: int,
    gid: int,
) -> Mapping[str, Any]:
    rows: dict[str, Any] = {}
    files = contract.get("files")
    if not isinstance(files, Mapping) or set(files) != set(
        package.HOST_ARTIFACT_TARGETS
    ):
        raise HostPlanProducerError("host_plan_contract_invalid")
    for name in sorted(files):
        item = files[name]
        if not isinstance(item, Mapping):
            raise HostPlanProducerError("host_plan_contract_invalid")
        raw, state = _read_regular(
            Path(str(item.get("staged_path"))),
            filesystem_root=filesystem_root,
            uid=uid,
            gid=gid,
            modes=frozenset({0o400}),
        )
        digest = _sha(raw)
        if item.get("package_sha256") is not None and item[
            "package_sha256"
        ] != digest:
            raise HostPlanProducerError("host_plan_package_payload_drifted")
        rows[name] = {
            "staged_path": item["staged_path"],
            "target_path": item["target_path"],
            "sha256": digest,
            "size": len(raw),
            "staged_uid": state.st_uid,
            "staged_gid": state.st_gid,
            "staged_mode": stat.S_IMODE(state.st_mode),
        }
    return rows


def _validate_staging_receipt(
    value: Any,
    *,
    revision: str,
    inputs: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected_fields = {
        "schema",
        "release_revision",
        "release_manifest_sha256",
        "host_artifact_contract_sha256",
        "unit_inputs_authority_plan_sha256",
        "unit_inputs_authority_approval_sha256",
        "source_gateway_config_sha256",
        "source_writer_config_sha256",
        "secret_foundation",
        "capability_topology",
        "staged_file_count",
        "staged_files",
        "staged_set_sha256",
        "secret_material_recorded",
        "secret_digest_recorded",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise HostPlanProducerError("host_plan_staging_receipt_invalid")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    staged = value.get("staged_files")
    if (
        value.get("schema") != STAGING_SCHEMA
        or value.get("release_revision") != revision
        or value.get("unit_inputs_authority_plan_sha256")
        != inputs["authority_plan_sha256"]
        or value.get("unit_inputs_authority_approval_sha256")
        != inputs["authority_approval_sha256"]
        or any(
            _SHA256.fullmatch(str(value.get(name))) is None
            for name in (
                "release_manifest_sha256",
                "host_artifact_contract_sha256",
                "source_gateway_config_sha256",
                "source_writer_config_sha256",
                "staged_set_sha256",
                "receipt_sha256",
            )
        )
        or not isinstance(staged, Mapping)
        or set(staged) != set(package.HOST_ARTIFACT_TARGETS)
        or value.get("staged_file_count") != len(package.HOST_ARTIFACT_TARGETS)
        or value.get("staged_set_sha256") != _sha(_canonical({"files": staged}))
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or value.get("receipt_sha256") != _sha(_canonical(unsigned))
    ):
        raise HostPlanProducerError("host_plan_staging_receipt_invalid")
    _secret_projection(value.get("secret_foundation"), inputs=inputs)
    try:
        validate_production_capability_topology(value["capability_topology"])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HostPlanProducerError("host_plan_staging_receipt_invalid") from exc
    return copy.deepcopy(dict(value))


def stage_fixed_host_artifacts(
    revision: str,
    *,
    release_root: Path | None = None,
    filesystem_root: Path = Path("/"),
    unit_inputs: Mapping[str, Any] | None = None,
    secret_stager: Callable[..., Mapping[str, Any]] = (
        production_secret_stager.stage_production_secret_foundation
    ),
    require_root: bool = True,
) -> Mapping[str, Any]:
    """Create or exactly re-observe every fixed host artifact."""

    if package.REVISION.fullmatch(revision or "") is None:
        raise HostPlanProducerError("host_plan_revision_invalid")
    fixed_release = _release_root(revision)
    release = fixed_release if release_root is None else release_root
    if require_root:
        if (
            not sys.platform.startswith("linux")
            or os.geteuid() != 0
            or os.getegid() != 0
            or release != fixed_release
            or filesystem_root != Path("/")
        ):
            raise HostPlanProducerError("host_plan_requires_linux_root")
        inputs = package.load_fixed_unit_inputs(revision=revision)
        trusted_uid = 0
        trusted_gid = 0
    else:
        if unit_inputs is None:
            raise HostPlanProducerError("host_plan_test_inputs_required")
        inputs = package._unit_inputs(unit_inputs, revision=revision)
        trusted_uid = os.geteuid()
        trusted_gid = os.getegid()
    source_root_uid = 0 if require_root else trusted_uid
    source_root_gid = 0 if require_root else trusted_gid
    legacy_policy, target_policy = _validate_reconciliation_intent(
        inputs, revision=revision
    )
    del legacy_policy
    try:
        sealed, descriptor, manifest = package.render_release_sealed_host_payloads(
            release_root=release,
            revision=revision,
            unit_inputs=inputs,
        )
    except (package.PackagingError, OSError) as exc:
        raise HostPlanProducerError("host_plan_release_invalid") from exc
    contract = manifest["host_artifact_contract"]

    gateway_source, _gateway_state = _read_regular(
        SOURCE_GATEWAY_CONFIG_PATH,
        filesystem_root=filesystem_root,
        uid=inputs["gateway"]["uid"],
        gid=inputs["gateway"]["gid"],
        modes=frozenset({0o600, 0o640}),
    )
    writer_source, _writer_state = _read_regular(
        SOURCE_WRITER_CONFIG_PATH,
        filesystem_root=filesystem_root,
        uid=source_root_uid,
        gid=source_root_gid,
        modes=frozenset({0o400}),
    )
    connector_unit_template, _ = _read_regular(
        release / REVIEWED_CONNECTOR_UNIT_TEMPLATE,
        filesystem_root=Path("/"),
    )
    connector_config_template, _ = _read_regular(
        release / REVIEWED_CONNECTOR_CONFIG_TEMPLATE,
        filesystem_root=Path("/"),
    )
    gateway_drop_in, _ = _read_regular(
        release / REVIEWED_GATEWAY_DROP_IN,
        filesystem_root=Path("/"),
    )
    connector_unit = _render_connector_unit(connector_unit_template, revision)
    connector_config = _render_connector_config(
        connector_config_template,
        inputs=inputs,
        target_policy=target_policy,
    )
    routeback_config = render_production_routeback_config(
        gateway_uid=inputs["gateway"]["uid"],
        routeback_uid=inputs["routeback"]["uid"],
        routeback_gid=inputs["routeback"]["gid"],
        writer_capability_public_key_id=inputs[
            "writer_capability_public_key_id"
        ],
        edge_receipt_public_key_id=inputs["discord_edge_receipt_public_key_id"],
        connection_timeout_seconds=10,
        max_connections=4,
        api_timeout_seconds=5,
        journal_busy_timeout_ms=5_000,
        max_proof_age_ms=10_000,
    )
    mac_ops_config = render_production_mac_ops_config(
        gateway_uid=inputs["gateway"]["uid"],
        socket_gid=inputs["mac_ops"]["gid"],
        service_identity_sha256=_sha(sealed["mac_ops_unit"]),
        max_connections=4,
        project_id=inputs["target"]["project"],
        timeout_seconds=20,
        journal_busy_timeout_ms=5_000,
    )
    topology = _build_capability_topology(
        descriptor=descriptor,
        payloads=sealed,
        connector_unit_sha256=_sha(connector_unit),
        routeback_config_sha256=_sha(routeback_config),
        mac_ops_config_sha256=_sha(mac_ops_config),
        inputs=inputs,
    )
    mac_ops_edge_config = {
        "enabled": True,
        "socket_path": str(MAC_OPS_SOCKET_PATH),
        "service_unit": DEFAULT_SERVICE_UNIT,
        "service_uid": inputs["mac_ops"]["uid"],
        "socket_gid": inputs["mac_ops"]["gid"],
        "service_identity_sha256": _sha(sealed["mac_ops_unit"]),
        "connect_timeout_seconds": 2.0,
        "request_timeout_seconds": 30.0,
    }
    try:
        gateway_contract = gateway_runtime.produce_production_gateway_contract(
            gateway_source,
            expected_source_sha256=_sha(gateway_source),
            revision=revision,
            gateway_user=inputs["gateway"]["user"],
            gateway_group=inputs["gateway"]["group"],
            topology=topology,
            mac_ops_edge_config=mac_ops_edge_config,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HostPlanProducerError("host_plan_gateway_render_failed") from exc
    writer_config = _render_writer_config(writer_source, inputs=inputs)
    writer_unit = _render_writer_unit(revision=revision, inputs=inputs)

    for directory in (
        package.CUTOVER_STAGED_ROOT.parent,
        package.CUTOVER_STAGED_ROOT,
        HOST_STAGING_ROOT,
        KEY_STAGING_ROOT,
    ):
        _ensure_private_directory(
            directory,
            filesystem_root=filesystem_root,
            uid=trusted_uid,
            gid=trusted_gid,
        )
    try:
        secret_receipt = (
            secret_stager()
            if require_root
            else secret_stager(
                filesystem_root=filesystem_root,
                inputs=inputs,
            )
        )
    except (TypeError, ValueError, RuntimeError, OSError) as exc:
        raise HostPlanProducerError("host_plan_secret_foundation_failed") from exc
    secret_foundation = _secret_projection(secret_receipt, inputs=inputs)

    payloads: dict[str, bytes] = {
        **sealed,
        "gateway_connector_drop_in": gateway_drop_in,
        "gateway_unit": gateway_contract.unit_bytes,
        "writer_unit": writer_unit,
        "connector_unit": connector_unit,
        "gateway_config": gateway_contract.config_bytes,
        "writer_config": writer_config,
        "connector_config": connector_config,
        "routeback_config": routeback_config,
        "mac_ops_config": mac_ops_config,
    }
    if (
        set().union(*HOST_ARTIFACT_SOURCE_PARTITIONS)
        != set(package.HOST_ARTIFACT_TARGETS)
        or sum(len(group) for group in HOST_ARTIFACT_SOURCE_PARTITIONS)
        != len(package.HOST_ARTIFACT_TARGETS)
    ):
        raise HostPlanProducerError("host_plan_source_partition_invalid")
    expected_payload_names = (
        set(package.HOST_ARTIFACT_TARGETS)
        - ROOT_VERIFIER_ARTIFACT_NAMES
    )
    if set(payloads) != expected_payload_names:
        raise HostPlanProducerError("host_plan_payload_set_invalid")
    contract_files = contract["files"]
    for name in sorted(payloads):
        _create_or_validate(
            Path(contract_files[name]["staged_path"]),
            payloads[name],
            filesystem_root=filesystem_root,
            uid=trusted_uid,
            gid=trusted_gid,
        )
    rows = _staged_rows(
        contract=contract,
        filesystem_root=filesystem_root,
        uid=trusted_uid,
        gid=trusted_gid,
    )
    unsigned = {
        "schema": STAGING_SCHEMA,
        "release_revision": revision,
        "release_manifest_sha256": manifest["manifest_sha256"],
        "host_artifact_contract_sha256": contract["contract_sha256"],
        "unit_inputs_authority_plan_sha256": inputs["authority_plan_sha256"],
        "unit_inputs_authority_approval_sha256": inputs[
            "authority_approval_sha256"
        ],
        "source_gateway_config_sha256": _sha(gateway_source),
        "source_writer_config_sha256": _sha(writer_source),
        "secret_foundation": secret_foundation,
        "capability_topology": topology,
        "staged_file_count": len(rows),
        "staged_files": rows,
        "staged_set_sha256": _sha(_canonical({"files": rows})),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    receipt = {**unsigned, "receipt_sha256": _sha(_canonical(unsigned))}
    _create_or_validate(
        STAGING_RECEIPT_PATH,
        _canonical(receipt),
        filesystem_root=filesystem_root,
        uid=trusted_uid,
        gid=trusted_gid,
    )
    return _validate_staging_receipt(receipt, revision=revision, inputs=inputs)


def _supplementary_groups(user: str, primary_gid: int) -> list[str]:
    try:
        gids = os.getgrouplist(user, primary_gid)
        names = {grp.getgrgid(gid).gr_name for gid in gids if gid != primary_gid}
    except (KeyError, OSError) as exc:
        raise HostPlanProducerError("host_plan_identity_prestate_invalid") from exc
    return sorted(names)


def _identity_foundation(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    user_targets: dict[str, Mapping[str, Any]] = {
        role: inputs[role]
        for role in (
            "gateway",
            "writer",
            "projector",
            "routeback",
            "connector",
            "mac_ops",
            "browser",
            "worker",
        )
    }
    group_targets: dict[str, Mapping[str, Any]] = dict(user_targets)
    group_targets["writer_client"] = inputs["writer_client_group"]
    group_targets["worker_client"] = inputs["worker_client_group"]
    for domain, item in sorted(inputs["operational_edge_identities"].items()):
        user_targets[f"operational_edge_{domain}"] = item
        group_targets[f"operational_edge_{domain}"] = item
        group_targets[f"operational_edge_{domain}_client"] = inputs[
            "operational_edge_socket_groups"
        ][domain]
    supplementary_roles: dict[str, list[str]] = {
        "gateway": [
            "browser",
            "connector",
            "mac_ops",
            "routeback",
            "worker_client",
            "writer_client",
            *(
                f"operational_edge_{domain}_client"
                for domain in sorted(inputs["operational_edge_identities"])
            ),
        ],
        "writer": ["projector"],
        "projector": [],
        "routeback": [],
        "connector": [],
        "mac_ops": [],
        "browser": [],
        "worker": [],
        **{
            f"operational_edge_{domain}": [
                f"operational_edge_{domain}_client"
            ]
            for domain in sorted(inputs["operational_edge_identities"])
        },
    }
    name_to_group_role = {
        item["group"]: role for role, item in group_targets.items()
    }
    target_members: dict[str, set[str]] = {
        role: set() for role in group_targets
    }
    users: dict[str, Any] = {}
    groups: dict[str, Any] = {}
    for role, target in group_targets.items():
        name = target["group"]
        try:
            observed = grp.getgrnam(name)
        except KeyError:
            try:
                occupied = grp.getgrgid(target["gid"])
            except KeyError:
                occupied = None
            if occupied is not None:
                raise HostPlanProducerError(
                    "host_plan_identity_prestate_invalid"
                )
            pre = {"state": "absent", "gid": None, "members": None}
        else:
            try:
                by_id = grp.getgrgid(target["gid"])
            except KeyError as exc:
                raise HostPlanProducerError(
                    "host_plan_identity_prestate_invalid"
                ) from exc
            if (
                observed.gr_gid != target["gid"]
                or by_id.gr_name != name
            ):
                raise HostPlanProducerError("host_plan_identity_prestate_invalid")
            pre = {
                "state": "present",
                "gid": observed.gr_gid,
                "members": sorted(set(observed.gr_mem)),
            }
        groups[role] = {
            "name": name,
            "gid": target["gid"],
            "members": [],
            "pre": pre,
        }
    for role, target in user_targets.items():
        name = target["user"]
        supplementary_names = sorted(
            group_targets[group_role]["group"]
            for group_role in supplementary_roles[role]
        )
        for group_name in supplementary_names:
            target_members[name_to_group_role[group_name]].add(name)
        try:
            observed = pwd.getpwnam(name)
        except KeyError:
            try:
                occupied = pwd.getpwuid(target["uid"])
            except KeyError:
                occupied = None
            if occupied is not None:
                raise HostPlanProducerError(
                    "host_plan_identity_prestate_invalid"
                )
            home = "/nonexistent"
            shell = "/usr/sbin/nologin"
            pre = {
                "state": "absent",
                "uid": None,
                "gid": None,
                "home": None,
                "shell": None,
                "supplementary_group_names": None,
            }
        else:
            try:
                by_id = pwd.getpwuid(target["uid"])
            except KeyError as exc:
                raise HostPlanProducerError(
                    "host_plan_identity_prestate_invalid"
                ) from exc
            home = observed.pw_dir
            shell = observed.pw_shell
            if (
                observed.pw_uid != target["uid"]
                or by_id.pw_name != name
                or observed.pw_gid != group_targets[role]["gid"]
                or (role != "gateway" and home != "/nonexistent")
                or (role != "gateway" and shell != "/usr/sbin/nologin")
            ):
                raise HostPlanProducerError("host_plan_identity_prestate_invalid")
            pre = {
                "state": "present",
                "uid": observed.pw_uid,
                "gid": observed.pw_gid,
                "home": observed.pw_dir,
                "shell": observed.pw_shell,
                "supplementary_group_names": _supplementary_groups(
                    name, observed.pw_gid
                ),
            }
        users[role] = {
            "name": name,
            "uid": target["uid"],
            "primary_group": role,
            "home": home,
            "shell": shell,
            "supplementary_groups": supplementary_names,
            "pre": pre,
        }
    for role, members in target_members.items():
        groups[role]["members"] = sorted(members)
    if (
        users["gateway"]["pre"]["state"] != "present"
        or groups["gateway"]["pre"]["state"] != "present"
    ):
        raise HostPlanProducerError("host_plan_gateway_identity_missing")
    unsigned = {
        "schema": cutover._IDENTITY_FOUNDATION_SCHEMA,
        "users": users,
        "groups": groups,
        "retain_created_dormant_on_rollback": True,
        "secret_material_recorded": False,
    }
    return {**unsigned, "foundation_sha256": _sha(_canonical(unsigned))}


def _discord_key_foundation(
    *,
    identity: Mapping[str, Any],
    secret_foundation: Mapping[str, Any],
    filesystem_root: Path,
) -> Mapping[str, Any]:
    expected_absent = (
        cutover.WRITER_CAPABILITY_PRIVATE_KEY_PATH,
        cutover.WRITER_CAPABILITY_PUBLIC_KEY_PATH,
        cutover.EDGE_RECEIPT_PRIVATE_KEY_PATH,
        cutover.EDGE_RECEIPT_PUBLIC_KEY_PATH,
    )
    for logical in expected_absent:
        if os.path.lexists(_physical(Path(logical), filesystem_root)):
            raise HostPlanProducerError("host_plan_discord_key_target_exists")
    unsigned = {
        "schema": cutover._DISCORD_KEY_FOUNDATION_SCHEMA,
        "writer": {
            "staged_private_path": secret_foundation["writer_private_path"],
            "private_path": str(cutover.WRITER_CAPABILITY_PRIVATE_KEY_PATH),
            "private_uid": identity["users"]["writer"]["uid"],
            "private_gid": identity["groups"]["writer"]["gid"],
            "private_mode": 0o400,
            "public_path": str(cutover.WRITER_CAPABILITY_PUBLIC_KEY_PATH),
            "public_uid": 0,
            "public_gid": identity["groups"]["routeback"]["gid"],
            "public_mode": 0o440,
            "public_key_id": secret_foundation["writer_public_key_id"],
        },
        "edge": {
            "staged_private_path": secret_foundation["edge_private_path"],
            "private_path": str(cutover.EDGE_RECEIPT_PRIVATE_KEY_PATH),
            "private_uid": 0,
            "private_gid": 0,
            "private_mode": 0o400,
            "public_path": str(cutover.EDGE_RECEIPT_PUBLIC_KEY_PATH),
            "public_uid": 0,
            "public_gid": identity["groups"]["writer"]["gid"],
            "public_mode": 0o440,
            "public_key_id": secret_foundation["edge_public_key_id"],
        },
        "pre_state": "absent",
        "keys_distinct": True,
        "private_content_or_digest_recorded": False,
        "secret_material_recorded": False,
    }
    return {**unsigned, "foundation_sha256": _sha(_canonical(unsigned))}


def _target_file_identity(
    name: str,
    *,
    inputs: Mapping[str, Any],
    pre: Mapping[str, Any],
) -> tuple[int, int, int]:
    if name.endswith("_unit") or name == "gateway_connector_drop_in":
        return 0, 0, 0o644
    if name == "gateway_config":
        mode = pre["mode"] if pre["state"] == "present" else 0o640
        if mode not in {0o600, 0o640}:
            raise HostPlanProducerError("host_plan_gateway_config_mode_invalid")
        return inputs["gateway"]["uid"], inputs["gateway"]["gid"], mode
    if name == "writer_config":
        return 0, inputs["writer"]["gid"], 0o440
    if name == "connector_config":
        return 0, inputs["connector"]["gid"], 0o440
    if name == "routeback_config":
        return 0, inputs["routeback"]["gid"], 0o440
    if name == "mac_ops_config":
        return 0, inputs["mac_ops"]["gid"], 0o440
    if name == "browser_config":
        return 0, inputs["browser"]["gid"], 0o440
    if name == "isolated_worker_config":
        return 0, inputs["worker"]["gid"], 0o440
    if name.startswith("operational_edge_config_"):
        return 0, 0, 0o400
    if name == "operational_edge_client_config":
        return 0, 0, 0o444
    if name in {"api_bearer_verifier", "api_approval_verifier"}:
        return 0, 0, 0o400
    raise HostPlanProducerError("host_plan_target_identity_missing")


def _metadata_only(
    logical: Path,
    *,
    filesystem_root: Path,
    uid: int,
    gid: int,
    mode: int,
) -> None:
    path = _physical(logical, filesystem_root)
    try:
        item = os.lstat(path)
    except OSError as exc:
        raise HostPlanProducerError("host_plan_secret_metadata_unavailable") from exc
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISREG(item.st_mode)
        or item.st_nlink != 1
        or item.st_uid != uid
        or item.st_gid != gid
        or stat.S_IMODE(item.st_mode) != mode
        or item.st_size <= 0
    ):
        raise HostPlanProducerError("host_plan_secret_metadata_invalid")


def _directory_prestate(
    logical: Path, *, filesystem_root: Path
) -> Mapping[str, Any]:
    path = _physical(logical, filesystem_root)
    try:
        item = os.lstat(path)
    except FileNotFoundError:
        return {"state": "absent", "uid": None, "gid": None, "mode": None}
    except OSError as exc:
        raise HostPlanProducerError("host_plan_directory_prestate_invalid") from exc
    if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
        raise HostPlanProducerError("host_plan_directory_prestate_invalid")
    return {
        "state": "present",
        "uid": item.st_uid,
        "gid": item.st_gid,
        "mode": stat.S_IMODE(item.st_mode),
    }


def collect_fixed_host_plan(
    revision: str,
    initial_receipt: Mapping[str, Any],
    *,
    release_root: Path | None = None,
    filesystem_root: Path = Path("/"),
    unit_inputs: Mapping[str, Any] | None = None,
    require_root: bool = True,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Read only and derive the exact seven host-authority plan fields."""

    if package.REVISION.fullmatch(revision or "") is None:
        raise HostPlanProducerError("host_plan_revision_invalid")
    fixed_release = _release_root(revision)
    release = fixed_release if release_root is None else release_root
    if require_root:
        if (
            not sys.platform.startswith("linux")
            or os.geteuid() != 0
            or os.getegid() != 0
            or release != fixed_release
            or filesystem_root != Path("/")
        ):
            raise HostPlanProducerError("host_plan_requires_linux_root")
        inputs = package.load_fixed_unit_inputs(revision=revision)
        trusted_uid = 0
        trusted_gid = 0
    else:
        if unit_inputs is None:
            raise HostPlanProducerError("host_plan_test_inputs_required")
        inputs = package._unit_inputs(unit_inputs, revision=revision)
        trusted_uid = os.geteuid()
        trusted_gid = os.getegid()
    source_root_uid = 0 if require_root else trusted_uid
    source_root_gid = 0 if require_root else trusted_gid
    try:
        from scripts.canary import production_cutover_owner_launcher as owner_launcher

        initial = owner_launcher.validate_initial_collector_receipt(
            initial_receipt,
            release_revision=revision,
            now_unix=now_unix,
        )
        manifest = package.verify_release_artifacts(
            release,
            revision,
            release_address=fixed_release,
            unit_inputs=inputs,
        )
    except (TypeError, ValueError, RuntimeError, package.PackagingError) as exc:
        raise HostPlanProducerError("host_plan_release_or_initial_invalid") from exc
    receipt_raw, _receipt_state = _read_regular(
        STAGING_RECEIPT_PATH,
        filesystem_root=filesystem_root,
        uid=trusted_uid,
        gid=trusted_gid,
        modes=frozenset({0o400}),
    )
    staging = _validate_staging_receipt(
        _decode(receipt_raw), revision=revision, inputs=inputs
    )
    gateway_source, _gateway_state = _read_regular(
        SOURCE_GATEWAY_CONFIG_PATH,
        filesystem_root=filesystem_root,
        uid=inputs["gateway"]["uid"],
        gid=inputs["gateway"]["gid"],
        modes=frozenset({0o600, 0o640}),
    )
    writer_source, _writer_state = _read_regular(
        SOURCE_WRITER_CONFIG_PATH,
        filesystem_root=filesystem_root,
        uid=source_root_uid,
        gid=source_root_gid,
        modes=frozenset({0o400}),
    )
    if (
        _sha(gateway_source) != staging["source_gateway_config_sha256"]
        or _sha(writer_source) != staging["source_writer_config_sha256"]
    ):
        raise HostPlanProducerError("host_plan_source_config_drifted")
    contract = manifest["host_artifact_contract"]
    if (
        staging["release_manifest_sha256"] != manifest["manifest_sha256"]
        or staging["host_artifact_contract_sha256"]
        != contract["contract_sha256"]
    ):
        raise HostPlanProducerError("host_plan_staging_release_drifted")
    observed_rows = _staged_rows(
        contract=contract,
        filesystem_root=filesystem_root,
        uid=trusted_uid,
        gid=trusted_gid,
    )
    if observed_rows != staging["staged_files"]:
        raise HostPlanProducerError("host_plan_staging_drifted")
    topology = staging["capability_topology"]
    identity = _identity_foundation(inputs)
    secret_foundation = staging["secret_foundation"]
    discord_keys = _discord_key_foundation(
        identity=identity,
        secret_foundation=secret_foundation,
        filesystem_root=filesystem_root,
    )

    files: dict[str, Any] = {}
    for name in sorted(package.HOST_ARTIFACT_TARGETS):
        row = observed_rows[name]
        pre = host_authority._target_pre_state(
            Path(row["target_path"]), filesystem_root=filesystem_root
        )
        uid, gid, mode = _target_file_identity(
            name, inputs=inputs, pre=pre
        )
        files[name] = {
            "staged_path": row["staged_path"],
            "target_path": row["target_path"],
            "sha256": row["sha256"],
            "uid": uid,
            "gid": gid,
            "mode": mode,
            "pre": pre,
        }
    gateway_target = {
        "name": cutover.GATEWAY_UNIT,
        "fragment_path": cutover.GATEWAY_FRAGMENT,
        "fragment_sha256": files["gateway_unit"]["sha256"],
        "load_state": "loaded",
        "unit_file_state": "enabled",
        "drop_in_paths": [cutover.GATEWAY_CONNECTOR_DROP_IN],
        "drop_in_sha256": {
            cutover.GATEWAY_CONNECTOR_DROP_IN: files[
                "gateway_connector_drop_in"
            ]["sha256"]
        },
        "need_daemon_reload": False,
        "triggered_by": [],
        "triggers": [],
    }
    writer_target = {
        "name": cutover.WRITER_UNIT,
        "fragment_path": cutover.WRITER_FRAGMENT,
        "fragment_sha256": files["writer_unit"]["sha256"],
        "load_state": "loaded",
        "unit_file_state": "enabled",
        "drop_in_paths": [],
        "drop_in_sha256": {},
        "need_daemon_reload": False,
        "triggered_by": [],
        "triggers": [],
    }
    connector_target = {
        "name": cutover.CONNECTOR_UNIT,
        "fragment_path": cutover.CONNECTOR_FRAGMENT,
        "fragment_sha256": files["connector_unit"]["sha256"],
        "load_state": "loaded",
        "unit_file_state": "enabled",
        "drop_in_paths": [],
        "drop_in_sha256": {},
        "need_daemon_reload": False,
        "triggered_by": [],
        "triggers": [],
    }
    _metadata_only(
        SOURCE_CONNECTOR_TOKEN_PATH,
        filesystem_root=filesystem_root,
        uid=inputs["gateway"]["uid"],
        gid=inputs["gateway"]["gid"],
        mode=0o400,
    )
    _metadata_only(
        cutover.STAGED_APPROVAL_PASSKEY_PATH,
        filesystem_root=filesystem_root,
        uid=0 if require_root else trusted_uid,
        gid=0 if require_root else trusted_gid,
        mode=0o400,
    )
    legacy_policy, target_policy = _validate_reconciliation_intent(
        inputs, revision=revision
    )
    continuity = cutover.build_discord_policy_continuity(
        source_evidence_sha256=_sha(_canonical(legacy_policy)),
        legacy_policy=legacy_policy,
        target_policy=target_policy,
    )
    transition_unsigned = {
        "schema": cutover._HOST_TRANSITION_SCHEMA,
        "files": files,
        "identity_foundation": identity,
        "discord_key_foundation": discord_keys,
        "operational_edge_key_foundation": copy.deepcopy(
            secret_foundation["operational_edge_key_foundation"]
        ),
        "operational_edge_key_foundation_sha256": secret_foundation[
            "operational_edge_key_foundation_sha256"
        ],
        "operational_edge_receipt_public_key_ids": copy.deepcopy(
            secret_foundation["operational_edge_receipt_public_key_ids"]
        ),
        "release_owner_uid": inputs["release_owner_uid"],
        "release_owner_gid": inputs["release_owner_gid"],
        "isolated_worker_lease_mountpoint": {
            "target_path": str(cutover.ISOLATED_WORKER_LEASE_BASE),
            "uid": 0,
            "gid": 0,
            "mode": 0o700,
            "pre": _directory_prestate(
                cutover.ISOLATED_WORKER_LEASE_BASE,
                filesystem_root=filesystem_root,
            ),
        },
        "connector_token": {
            "path": cutover.CONNECTOR_TOKEN_PATH,
            "uid": inputs["connector"]["uid"],
            "gid": inputs["connector"]["gid"],
            "mode": 0o400,
            "regular_one_link": True,
            "content_or_digest_recorded": False,
            "gateway_readable": False,
            "source_path": str(SOURCE_CONNECTOR_TOKEN_PATH),
            "source_uid": inputs["gateway"]["uid"],
            "source_gid": inputs["gateway"]["gid"],
            "source_mode": 0o400,
        },
        "gateway_retired_token_paths": [str(SOURCE_CONNECTOR_TOKEN_PATH)],
        "approval_passkey": {
            "path": str(cutover.API_APPROVAL_CREDENTIAL_PATH),
            "uid": 0,
            "gid": 0,
            "mode": 0o400,
            "regular_one_link": True,
            "content_or_digest_recorded": False,
            "gateway_readable": False,
            "source_path": str(cutover.STAGED_APPROVAL_PASSKEY_PATH),
            "source_uid": 0,
            "source_gid": 0,
            "source_mode": 0o400,
        },
        "retired_approval_passkey_paths": [
            str(cutover.STAGED_APPROVAL_PASSKEY_PATH)
        ],
        "routeback_token_paths": [str(ROUTEBACK_EDGE_CREDENTIAL_PATH)],
        "gateway_direct_discord_enabled": False,
        "gateway_relay_platforms": ["discord"],
        "connector_operation_class": "ordinary_guild_acl_session_only",
        "routeback_operation_class": (
            "canonical_guild_acl_routeback_rest_only"
        ),
        "discord_dm_allowed": False,
        "discord_policy_continuity": continuity,
        "secret_material_recorded": False,
    }
    host_transition = {
        **transition_unsigned,
        "manifest_sha256": _sha(_canonical(transition_unsigned)),
    }
    result = {
        "release_manifest_sha256": manifest["manifest_sha256"],
        "gateway_target_identity": gateway_target,
        "writer_target_identity": writer_target,
        "connector_target_identity": connector_target,
        "host_transition": host_transition,
        "capability_topology": copy.deepcopy(topology),
        "cron_continuity_plan": copy.deepcopy(
            initial["cron_continuity_plan"]
        ),
    }
    try:
        host_authority._validate_transition_and_plan(result, initial=initial)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HostPlanProducerError("host_plan_derived_contract_invalid") from exc
    if set(result) != {
        "release_manifest_sha256",
        "gateway_target_identity",
        "writer_target_identity",
        "connector_target_identity",
        "host_transition",
        "capability_topology",
        "cron_continuity_plan",
    }:
        raise HostPlanProducerError("host_plan_field_set_invalid")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage", "collect"))
    parser.add_argument("--revision", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "stage":
            if not sys.stdin.isatty():
                supplied = sys.stdin.buffer.read(1)
                if supplied:
                    raise HostPlanProducerError("host_plan_stage_input_forbidden")
            result = stage_fixed_host_artifacts(args.revision)
        else:
            raw = sys.stdin.buffer.read(MAX_INPUT + 1)
            if not raw or len(raw) > MAX_INPUT:
                raise HostPlanProducerError("host_plan_input_invalid")
            frame = raw[:-1] if raw.endswith(b"\n") else raw
            result = collect_fixed_host_plan(
                args.revision,
                _decode(frame),
            )
    except HostPlanProducerError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(_canonical(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HostPlanProducerError",
    "STAGING_RECEIPT_PATH",
    "collect_fixed_host_plan",
    "main",
    "stage_fixed_host_artifacts",
]
