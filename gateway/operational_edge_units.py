"""Render exact per-domain systemd units for operational edge services."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from gateway.operational_edge_catalog import (
    CANONICAL_BRAIN,
    CREDENTIALS_BY_DOMAIN,
    HERMES_HOME,
    asset_catalog,
    catalog_public_contract,
    operation_catalog,
)
from gateway.operational_edge_service import CONFIG_SCHEMA


UNIT_BUNDLE_SCHEMA = "muncho-operational-edge-unit-bundle.v2"
CLIENT_CONFIG_SCHEMA = "muncho-operational-edge-client-config.v3"

CONFIG_ROOT = Path("/etc/muncho/operational-edge")
CLIENT_CONFIG_PATH = Path("/etc/muncho/operational-edge-client.json")
KEY_ROOT = Path("/etc/muncho/keys")
TRUST_ROOT = CONFIG_ROOT / "trust"
WRITER_PUBLIC_KEY = KEY_ROOT / "writer-capability-public.pem"
SOCKET_ROOT = Path("/run/muncho-operational-edge")
STATE_ROOT = Path("/var/lib/muncho-operational-edge")
SUBPROCESS_HOME = Path("/opt/adventico-ai-platform")

OPERATIONAL_EDGE_MUTATION_USER = "ai-platform-brain"

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")


class OperationalEdgeUnitError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _catalog_sha256() -> str:
    return _sha256(_canonical(catalog_public_contract()))


def service_unit(domain: str) -> str:
    return f"muncho-operational-edge-{domain}.service"


def service_identity_name(domain: str) -> str:
    """Return the fixed, domain-exclusive service user and primary group."""

    return f"muncho-edge-{domain}"


def socket_group_name(domain: str) -> str:
    """Return the fixed client group for exactly one domain socket."""

    return f"muncho-edge-{domain}-c"


def service_config_path(domain: str) -> Path:
    return CONFIG_ROOT / f"{domain}.json"


def receipt_private_key_path(domain: str) -> Path:
    return KEY_ROOT / f"operational-edge-{domain}-receipt-private.pem"


def receipt_public_key_path(domain: str) -> Path:
    return TRUST_ROOT / f"{domain}-receipt-public.pem"


@dataclass(frozen=True)
class OperationalEdgeUnitBundle:
    revision: str
    units: Mapping[str, bytes]
    configs: Mapping[str, bytes]
    client_config: bytes
    manifest: Mapping[str, Any]


def _service_config(
    *,
    revision: str,
    release: Path,
    domain: str,
    release_owner_uid: int,
    release_owner_gid: int,
    service_uid: int,
    service_gid: int,
    socket_gid: int,
    read_peer_uids: Sequence[int],
    mutation_peer_uid: int,
    receipt_public_key_id: str,
    writer_key_id: str,
) -> bytes:
    unit = service_unit(domain)
    value = {
        "schema": CONFIG_SCHEMA,
        "domain": domain,
        "release_revision": revision,
        "release_root": str(release),
        "release_owner_uid": release_owner_uid,
        "release_owner_gid": release_owner_gid,
        "socket_path": str(SOCKET_ROOT / domain / "edge.sock"),
        "socket_gid": socket_gid,
        "service_uid": service_uid,
        "service_gid": service_gid,
        "allowed_read_peer_uids": sorted(set(read_peer_uids)),
        "mutation_peer_uid": mutation_peer_uid,
        "journal_path": str(STATE_ROOT / domain / "journal.sqlite3"),
        "subprocess_home": str(SUBPROCESS_HOME),
        "receipt_private_key_file": f"/run/credentials/{unit}/receipt-private-key",
        "receipt_key_id": receipt_public_key_id,
        "writer_public_key_file": (
            f"/run/credentials/{unit}/writer-public-key"
        ),
        "writer_key_id": writer_key_id,
        "maximum_output_bytes": 1024 * 1024,
        "maximum_connections": 16,
        "catalog_sha256": _catalog_sha256(),
    }
    return _canonical(value) + b"\n"


def _helper_binds(release: Path, domain: str) -> list[str]:
    assets = asset_catalog()
    pairs_by_domain = {
        "skyvision_email": (
            ("muncho_step_up_verify", HERMES_HOME / "bin/muncho_step_up_verify"),
            ("muncho_dangerous_action_guard", HERMES_HOME / "bin/muncho_dangerous_action_guard"),
        ),
        "adventico_email": (),
        "bitrix": (
            ("muncho_step_up_verify", HERMES_HOME / "bin/muncho_step_up_verify"),
            ("muncho_dangerous_action_guard", HERMES_HOME / "bin/muncho_dangerous_action_guard"),
        ),
        "skyvision_db": (
            ("ssh-alwyzon-phoenix", HERMES_HOME / "bin/ssh-alwyzon-phoenix"),
            ("muncho_step_up_verify", HERMES_HOME / "bin/muncho_step_up_verify"),
        ),
        "skyvision_panel": (
            ("ssh-alwyzon-phoenix", HERMES_HOME / "bin/ssh-alwyzon-phoenix"),
            ("muncho_step_up_verify", HERMES_HOME / "bin/muncho_step_up_verify"),
        ),
        "skyvision_gitlab": (
            ("muncho_step_up_verify", HERMES_HOME / "bin/muncho_step_up_verify"),
            ("muncho_dangerous_action_guard", HERMES_HOME / "bin/muncho_dangerous_action_guard"),
        ),
        "infrastructure": (
            ("contabo-api", SUBPROCESS_HOME / ".hermes/bin/contabo-api"),
            ("ssh-alwyzon-phoenix", SUBPROCESS_HOME / ".hermes/bin/ssh-alwyzon-phoenix"),
        ),
        "github": (("gh-hermes", HERMES_HOME / "bin/gh-hermes"),),
        "canonical": (),
    }
    pairs = pairs_by_domain.get(domain)
    if pairs is None:
        raise OperationalEdgeUnitError("operational edge helper domain unknown")
    return [
        f"BindReadOnlyPaths={release / assets[asset_id].packaged_relative}:{target}"
        for asset_id, target in pairs
    ]


def _service_unit(
    *,
    revision: str,
    release: Path,
    interpreter: Path,
    domain: str,
    service_user: str,
    service_group: str,
    release_owner_uid: int,
    release_owner_gid: int,
    service_uid: int,
    service_gid: int,
    socket_group: str,
    socket_gid: int,
) -> bytes:
    unit = service_unit(domain)
    config = service_config_path(domain)
    projected_config = Path("/run/credentials") / unit / "service-config"
    credential_lines = [
        f"LoadCredential={item.name}:{item.source_path}"
        for item in CREDENTIALS_BY_DOMAIN[domain]
    ]
    credential_binds = [
        f"BindReadOnlyPaths=/run/credentials/{unit}/{item.name}:{item.target_path}"
        for item in CREDENTIALS_BY_DOMAIN[domain]
    ]
    metadata_lines = (
        ["IPAddressAllow=169.254.169.254/32"]
        if domain == "canonical"
        else ["IPAddressDeny=169.254.169.254/32"]
    )
    audit_domains = {
        "skyvision_email", "bitrix", "skyvision_panel", "skyvision_gitlab"
    }
    if domain == "canonical":
        canonical_visibility = [
            f"ReadOnlyPaths={CANONICAL_BRAIN}",
            f"ReadWritePaths={CANONICAL_BRAIN / 'state'}",
        ]
    elif domain in audit_domains:
        reports = CANONICAL_BRAIN / "state/reports"
        canonical_visibility = [
            f"InaccessiblePaths=-{CANONICAL_BRAIN}",
            f"BindPaths={reports}",
            f"ReadWritePaths={reports}",
        ]
    else:
        canonical_visibility = [f"InaccessiblePaths=-{CANONICAL_BRAIN}"]
    lines = [
        "# Release-addressed credential-scoped operational edge.",
        f"# ReleaseRevision={revision}",
        f"# Domain={domain}",
        f"# PrincipalUID={service_uid}",
        f"# PrincipalGID={service_gid}",
        f"# ReleaseOwnerUID={release_owner_uid}",
        f"# ReleaseOwnerGID={release_owner_gid}",
        "[Unit]",
        f"Description=Muncho credential-scoped operational edge ({domain})",
        "After=network-online.target",
        "Wants=network-online.target",
        "Before=hermes-cloud-gateway.service",
        "StartLimitIntervalSec=300s",
        "StartLimitBurst=5",
        f"AssertPathExists={interpreter}",
        f"AssertPathExists={release / '.codex-source-commit'}",
        f"AssertPathExists={release / 'ops/muncho/runtime/operational-assets/manifest.json'}",
        f"AssertPathExists={config}",
        f"AssertPathExists={WRITER_PUBLIC_KEY}",
        f"AssertPathExists={receipt_private_key_path(domain)}",
        f"AssertPathExists={receipt_public_key_path(domain)}",
        *(f"AssertPathExists={item.source_path}" for item in CREDENTIALS_BY_DOMAIN[domain]),
        "",
        "[Service]",
        "Type=simple",
        f"User={service_user}",
        f"Group={service_group}",
        f"SupplementaryGroups={socket_group}",
        f"LoadCredential=receipt-private-key:{receipt_private_key_path(domain)}",
        f"LoadCredential=writer-public-key:{WRITER_PUBLIC_KEY}",
        f"LoadCredential=service-config:{config}",
        *credential_lines,
        f"RuntimeDirectory=muncho-operational-edge/{domain}",
        # Exact-path traversal is public; the socket itself remains 0660 and
        # group-scoped.  0711 also lets the root publisher's dropped collector
        # reach the socket without making the directory listable.
        "RuntimeDirectoryMode=0711",
        f"StateDirectory=muncho-operational-edge/{domain}",
        "StateDirectoryMode=0700",
        f"WorkingDirectory={release}",
        f"ExecStart={interpreter} -I -B -m gateway.operational_edge_service --config {projected_config}",
        "Restart=on-failure",
        "RestartSec=5s",
        "TimeoutStartSec=30s",
        "TimeoutStopSec=30s",
        "KillMode=control-group",
        "LimitCORE=0",
        "LimitNOFILE=4096",
        f"Environment=HOME={STATE_ROOT / domain}",
        "Environment=LANG=C.UTF-8",
        "Environment=LC_ALL=C.UTF-8",
        "Environment=PATH=/usr/bin:/bin",
        "Environment=PYTHONDONTWRITEBYTECODE=1",
        "Environment=PYTHONNOUSERSITE=1",
        "Environment=TZ=UTC",
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
        "UMask=0077",
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        *metadata_lines,
        f"ReadOnlyPaths={release}",
        *canonical_visibility,
        f"InaccessiblePaths=-{HERMES_HOME / 'secrets'}",
        f"InaccessiblePaths=-{SUBPROCESS_HOME / '.hermes/secrets'}",
        *_helper_binds(release, domain),
        *credential_binds,
        f"ReadWritePaths={STATE_ROOT / domain}",
        f"ReadWritePaths={SOCKET_ROOT / domain}",
        "StandardInput=null",
        "StandardOutput=journal",
        "StandardError=journal",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ]
    text = "\n".join(lines) + "\n"
    if (
        text.count("ExecStart=") != 1
        or "EnvironmentFile=" in text
        or "PassEnvironment=" in text
        or "LoadCredential=" not in text
        or f"SupplementaryGroups={socket_group}\n" not in text
        or "ProtectSystem=strict\n" not in text
        or "NoNewPrivileges=yes\n" not in text
    ):
        raise OperationalEdgeUnitError("operational edge unit safety drifted")
    return text.encode("utf-8")


def render_operational_edge_units(
    *,
    revision: str,
    service_identities: Mapping[str, Mapping[str, Any]],
    socket_groups: Mapping[str, Mapping[str, Any]],
    release_owner_uid: int,
    release_owner_gid: int,
    read_peer_uids: Sequence[int],
    mutation_peer_uid: int,
    mutation_peer_gid: int,
    receipt_public_key_ids: Mapping[str, str],
    writer_key_id: str,
) -> OperationalEdgeUnitBundle:
    domains = sorted({item.domain for item in operation_catalog().values()})
    services = (
        dict(service_identities)
        if isinstance(service_identities, Mapping)
        else {}
    )
    sockets = dict(socket_groups) if isinstance(socket_groups, Mapping) else {}
    service_rows_valid = set(services) == set(domains) and all(
        isinstance(row, Mapping)
        and set(row) == {"user", "group", "uid", "gid"}
        and row["user"] == service_identity_name(domain)
        and row["group"] == service_identity_name(domain)
        and _IDENTITY.fullmatch(str(row["user"])) is not None
        and _IDENTITY.fullmatch(str(row["group"])) is not None
        and type(row["uid"]) is int
        and row["uid"] > 0
        and type(row["gid"]) is int
        and row["gid"] > 0
        for domain, row in services.items()
    )
    socket_rows_valid = set(sockets) == set(domains) and all(
        isinstance(row, Mapping)
        and set(row) == {"group", "gid"}
        and row["group"] == socket_group_name(domain)
        and _IDENTITY.fullmatch(str(row["group"])) is not None
        and type(row["gid"]) is int
        and row["gid"] > 0
        for domain, row in sockets.items()
    )
    service_uids = [row["uid"] for row in services.values()] if service_rows_valid else []
    service_gids = [row["gid"] for row in services.values()] if service_rows_valid else []
    socket_gids = [row["gid"] for row in sockets.values()] if socket_rows_valid else []
    if (
        _REVISION.fullmatch(revision or "") is None
        or not service_rows_valid
        or not socket_rows_valid
        or any(
            type(value) is not int or value < 1
            for value in (
                release_owner_uid,
                release_owner_gid,
                mutation_peer_uid,
                mutation_peer_gid,
            )
        )
        or len(set(service_uids)) != len(domains)
        or len(set(service_gids)) != len(domains)
        or len(set(socket_gids)) != len(domains)
        or len(set(service_gids) | set(socket_gids)) != len(domains) * 2
        or not read_peer_uids
        or len(read_peer_uids) > 16
        or any(type(value) is not int or value < 1 for value in read_peer_uids)
        or tuple(read_peer_uids) != tuple(sorted(set(read_peer_uids)))
        or mutation_peer_uid not in read_peer_uids
        or set(read_peer_uids).intersection(service_uids)
        or mutation_peer_uid in set(service_uids)
        or mutation_peer_gid in set(service_gids) | set(socket_gids)
        or not isinstance(receipt_public_key_ids, Mapping)
        or set(receipt_public_key_ids) != set(domains)
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in receipt_public_key_ids.values()
        )
        or len(set(receipt_public_key_ids.values())) != len(domains)
        or not isinstance(writer_key_id, str)
        or _SHA256.fullmatch(writer_key_id) is None
    ):
        raise OperationalEdgeUnitError("operational edge identity input invalid")
    release = Path("/opt/adventico-ai-platform/hermes-agent-releases") / f"hermes-agent-{revision[:12]}"
    interpreter = release / ".venv/bin/python"
    if set(domains) != set(CREDENTIALS_BY_DOMAIN):
        raise OperationalEdgeUnitError("operational edge domain mapping incomplete")
    configs = {
        str(service_config_path(domain)): _service_config(
            revision=revision,
            release=release,
            domain=domain,
            release_owner_uid=release_owner_uid,
            release_owner_gid=release_owner_gid,
            service_uid=services[domain]["uid"],
            service_gid=services[domain]["gid"],
            socket_gid=sockets[domain]["gid"],
            read_peer_uids=read_peer_uids,
            mutation_peer_uid=mutation_peer_uid,
            receipt_public_key_id=receipt_public_key_ids[domain],
            writer_key_id=writer_key_id,
        )
        for domain in domains
    }
    units = {
        service_unit(domain): _service_unit(
            revision=revision, release=release, interpreter=interpreter,
            domain=domain, service_user=services[domain]["user"],
            service_group=services[domain]["group"],
            release_owner_uid=release_owner_uid,
            release_owner_gid=release_owner_gid,
            service_uid=services[domain]["uid"],
            service_gid=services[domain]["gid"],
            socket_group=sockets[domain]["group"],
            socket_gid=sockets[domain]["gid"],
        )
        for domain in domains
    }
    client_value = {
        "schema": CLIENT_CONFIG_SCHEMA,
        "domains": {
            domain: {
                "socket_path": str(SOCKET_ROOT / domain / "edge.sock"),
                "service_unit": service_unit(domain),
                "service_uid": services[domain]["uid"],
                "service_gid": services[domain]["gid"],
                "socket_gid": sockets[domain]["gid"],
                "probe_uid": mutation_peer_uid,
                "probe_gid": mutation_peer_gid,
                "probe_supplementary_gids": sorted(socket_gids),
                "receipt_public_key_file": str(receipt_public_key_path(domain)),
                "receipt_key_id": receipt_public_key_ids[domain],
            }
            for domain in domains
        },
    }
    client_config = _canonical(client_value) + b"\n"
    unsigned_manifest = {
        "schema": UNIT_BUNDLE_SCHEMA,
        "release_revision": revision,
        "release_root": str(release),
        "release_owner_uid": release_owner_uid,
        "release_owner_gid": release_owner_gid,
        "catalog_sha256": _catalog_sha256(),
        "domains": domains,
        "operation_count": len(operation_catalog()),
        "all_operations_implemented": True,
        "credentials_by_domain": {
            domain: [item.name for item in CREDENTIALS_BY_DOMAIN[domain]]
            for domain in domains
        },
        "credential_count": sum(
            len(CREDENTIALS_BY_DOMAIN[domain]) for domain in domains
        ),
        "credential_values_read": False,
        "receipt_public_key_ids": {
            domain: receipt_public_key_ids[domain] for domain in domains
        },
        "identity_contract": {
            "services": {
                domain: dict(services[domain]) for domain in domains
            },
            "probe_runner": {
                "user": OPERATIONAL_EDGE_MUTATION_USER,
                "uid": mutation_peer_uid,
                "gid": mutation_peer_gid,
                "supplementary_gids": sorted(socket_gids),
                "root_allowed": False,
            },
            "sockets": {
                domain: {**dict(sockets[domain]), "mode": "0660"}
                for domain in domains
            },
            "mutation_peer": {
                "user": OPERATIONAL_EDGE_MUTATION_USER,
                "uid": mutation_peer_uid,
                "gid": mutation_peer_gid,
                "distinct_from_services": True,
            },
            "allowed_read_peer_uids_by_domain": {
                domain: list(read_peer_uids) for domain in domains
            },
            "cross_domain_service_access": False,
            "root_socket_peer_allowed": False,
        },
        "units": {name: _sha256(value) for name, value in units.items()},
        "configs": {name: _sha256(value) for name, value in configs.items()},
        "client_config_path": str(CLIENT_CONFIG_PATH),
        "client_config_sha256": _sha256(client_config),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    manifest = {
        **unsigned_manifest,
        "bundle_sha256": _sha256(_canonical(unsigned_manifest)),
    }
    return OperationalEdgeUnitBundle(
        revision=revision,
        units=units,
        configs=configs,
        client_config=client_config,
        manifest=manifest,
    )


__all__ = [
    "CLIENT_CONFIG_PATH",
    "CLIENT_CONFIG_SCHEMA",
    "CONFIG_ROOT",
    "TRUST_ROOT",
    "OperationalEdgeUnitBundle",
    "OperationalEdgeUnitError",
    "OPERATIONAL_EDGE_MUTATION_USER",
    "render_operational_edge_units",
    "receipt_private_key_path",
    "receipt_public_key_path",
    "service_config_path",
    "service_identity_name",
    "service_unit",
    "socket_group_name",
]
