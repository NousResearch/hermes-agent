#!/usr/bin/env python3
"""Collect and sign the exact production network evidence for owner-gate.

All Cloud calls are fixed read-only inventory operations.  The resulting
evidence is signed by the release-bound network collector key held only in the
owner authority directory.  This module never creates a subnet, VM, firewall,
route, peering, address, or connector.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_owner_reauth as owner_reauth
from scripts.canary import owner_gate_trust_author as release_author
from scripts.canary import trusted_signer_author as signer_author


RunJson = Callable[[Sequence[str]], Any]

_CAPTURE_TIMEOUT_SECONDS = 60.0
_MAX_CAPTURE_BYTES = 16 * 1024 * 1024

_RESOURCE_PREFIX = "https://www.googleapis.com/compute/v1/"
_REGION = re.compile(r"^[a-z]+-[a-z0-9]+[0-9]$")
_NETWORK_NAME = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_MAX_REGIONS = 128
_MAX_ITEMS = 10_000
_RFC1918 = tuple(
    ipaddress.ip_network(cidr)
    for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


class OwnerGateNetworkEvidenceAuthorError(RuntimeError):
    """Stable, secret-free network evidence authoring failure."""


@dataclass(frozen=True)
class _CapturedJson:
    returncode: int
    stdout: bytes
    stderr: bytes


class _NetworkEvidenceRunner(Protocol):
    def run_capture(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> _CapturedJson: ...


class _SubprocessNetworkEvidenceRunner:
    """Bounded runner reachable only through the exact public capability."""

    def run_capture(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> _CapturedJson:
        try:
            completed = subprocess.run(
                tuple(argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(env),
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OwnerGateNetworkEvidenceAuthorError(
                "owner_gate_network_collection_failed"
            ) from None
        return _CapturedJson(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def inventory_commands() -> Mapping[str, tuple[str, ...]]:
    project = f"--project={foundation.PROJECT}"
    json_format = "--format=json"
    return {
        "instance": (
            "gcloud",
            "compute",
            "instances",
            "describe",
            foundation.PRODUCTION_SOURCE_VM,
            project,
            f"--zone={foundation.ZONE}",
            json_format,
        ),
        "subnets": (
            "gcloud",
            "compute",
            "networks",
            "subnets",
            "list",
            project,
            json_format,
        ),
        "routes": (
            "gcloud",
            "compute",
            "routes",
            "list",
            project,
            json_format,
        ),
        "peerings": (
            "gcloud",
            "compute",
            "networks",
            "peerings",
            "list",
            f"--network={foundation.NETWORK_NAME}",
            project,
            json_format,
        ),
        "private_service_ranges": (
            "gcloud",
            "compute",
            "addresses",
            "list",
            "--global",
            project,
            json_format,
        ),
        "regions": (
            "gcloud",
            "compute",
            "regions",
            "list",
            project,
            json_format,
        ),
        "iap_firewall": (
            "gcloud",
            "compute",
            "firewall-rules",
            "describe",
            "allow-iap-ssh",
            project,
            json_format,
        ),
    }


def connector_command(region: str) -> tuple[str, ...]:
    if _REGION.fullmatch(region or "") is None:
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_region_invalid"
        )
    return (
        "gcloud",
        "compute",
        "networks",
        "vpc-access",
        "connectors",
        "list",
        f"--region={region}",
        f"--project={foundation.PROJECT}",
        "--format=json",
    )


def _items(value: Any, *, label: str) -> list[Mapping[str, Any]]:
    if (
        not isinstance(value, list)
        or len(value) > _MAX_ITEMS
        or any(not isinstance(item, Mapping) for item in value)
    ):
        raise OwnerGateNetworkEvidenceAuthorError(
            f"owner_gate_network_{label}_invalid"
        )
    return [dict(item) for item in value]


def _resource(value: Any, *, suffix: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OwnerGateNetworkEvidenceAuthorError(
            f"owner_gate_network_{label}_invalid"
        )
    normalized = value.removeprefix(_RESOURCE_PREFIX)
    if normalized != suffix:
        raise OwnerGateNetworkEvidenceAuthorError(
            f"owner_gate_network_{label}_invalid"
        )
    return normalized


def _network(value: Any) -> str:
    return _resource(
        value,
        suffix=(
            f"projects/{foundation.PROJECT}/global/networks/"
            f"{foundation.NETWORK_NAME}"
        ),
        label="network",
    )


def _canonical_inventory(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_inventory(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        normalized = [_canonical_inventory(item) for item in value]
        return sorted(normalized, key=foundation.canonical_json_bytes)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise OwnerGateNetworkEvidenceAuthorError(
        "owner_gate_network_inventory_invalid"
    )


def _private_cidr(value: Any, *, label: str) -> ipaddress.IPv4Network:
    try:
        network = ipaddress.ip_network(str(value), strict=True)
    except (TypeError, ValueError) as exc:
        raise OwnerGateNetworkEvidenceAuthorError(
            f"owner_gate_network_{label}_invalid"
        ) from None
    if not isinstance(network, ipaddress.IPv4Network) or not any(
        network.subnet_of(parent) for parent in _RFC1918
    ):
        raise OwnerGateNetworkEvidenceAuthorError(
            f"owner_gate_network_{label}_invalid"
        )
    return network


def _collect_raw(run_json: RunJson) -> Mapping[str, Any]:
    if not callable(run_json):
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_runner_invalid"
        )
    commands = inventory_commands()
    raw = {name: run_json(argv) for name, argv in commands.items()}
    regions = _items(raw["regions"], label="regions")
    names = sorted({
        str(item.get("name"))
        for item in regions
        if item.get("status") in {"UP", None}
        and isinstance(item.get("name"), str)
        and _REGION.fullmatch(str(item["name"])) is not None
    })
    if not names or len(names) > _MAX_REGIONS:
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_regions_invalid"
        )
    raw["serverless_connectors"] = {
        region: run_json(connector_command(region)) for region in names
    }
    return raw


def _normalize_and_author(
    raw: Mapping[str, Any],
    *,
    release_revision: str,
    collected_at_unix: int,
) -> Mapping[str, Any]:
    instance = raw.get("instance")
    if not isinstance(instance, Mapping):
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_instance_invalid"
        )
    expected_instance = (
        f"projects/{foundation.PROJECT}/zones/{foundation.ZONE}/instances/"
        f"{foundation.PRODUCTION_SOURCE_VM}"
    )
    instance_self_link = _resource(
        instance.get("selfLink"),
        suffix=expected_instance,
        label="instance",
    )
    interfaces = instance.get("networkInterfaces")
    accounts = instance.get("serviceAccounts")
    if (
        instance.get("name") != foundation.PRODUCTION_SOURCE_VM
        or str(instance.get("id")) != foundation.PRODUCTION_SOURCE_VM_ID
        or not isinstance(interfaces, list)
        or len(interfaces) != 1
        or not isinstance(interfaces[0], Mapping)
        or not isinstance(accounts, list)
        or len(accounts) != 1
        or not isinstance(accounts[0], Mapping)
        or accounts[0].get("email")
        != foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT
    ):
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_instance_invalid"
        )
    interface = interfaces[0]
    network_self_link = _network(interface.get("network"))
    expected_subnet = (
        f"projects/{foundation.PROJECT}/regions/{foundation.REGION}/"
        f"subnetworks/{foundation.PRODUCTION_SUBNET_NAME}"
    )
    subnet_self_link = _resource(
        interface.get("subnetwork"),
        suffix=expected_subnet,
        label="subnet",
    )
    try:
        source_ip = ipaddress.ip_address(str(interface.get("networkIP")))
    except ValueError as exc:
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_instance_invalid"
        ) from None
    if not isinstance(source_ip, ipaddress.IPv4Address) or not any(
        source_ip in parent for parent in _RFC1918
    ):
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_instance_invalid"
        )

    subnets = _items(raw.get("subnets"), label="subnets")
    target_subnets = [
        item
        for item in subnets
        if str(item.get("selfLink", "")).removeprefix(_RESOURCE_PREFIX)
        == expected_subnet
    ]
    if len(target_subnets) != 1:
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_subnet_invalid"
        )
    target_subnet = target_subnets[0]
    _network(target_subnet.get("network"))
    production_cidr = _private_cidr(
        target_subnet.get("ipCidrRange"), label="subnet"
    )
    if source_ip not in production_cidr or type(
        target_subnet.get("privateIpGoogleAccess")
    ) is not bool:
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_subnet_invalid"
        )

    reserved: set[ipaddress.IPv4Network] = set()
    for subnet in subnets:
        try:
            if _network(subnet.get("network")) != network_self_link:
                continue
        except OwnerGateNetworkEvidenceAuthorError:
            continue
        reserved.add(_private_cidr(subnet.get("ipCidrRange"), label="subnet"))
        secondary = subnet.get("secondaryIpRanges", [])
        if not isinstance(secondary, list):
            raise OwnerGateNetworkEvidenceAuthorError(
                "owner_gate_network_subnets_invalid"
            )
        for item in secondary:
            if not isinstance(item, Mapping):
                raise OwnerGateNetworkEvidenceAuthorError(
                    "owner_gate_network_subnets_invalid"
                )
            reserved.add(
                _private_cidr(item.get("ipCidrRange"), label="subnet")
            )

    routes = _items(raw.get("routes"), label="routes")
    for route in routes:
        try:
            _network(route.get("network"))
        except OwnerGateNetworkEvidenceAuthorError:
            continue
        destination = route.get("destRange")
        try:
            route_network = ipaddress.ip_network(
                str(destination), strict=True
            )
        except ValueError as exc:
            raise OwnerGateNetworkEvidenceAuthorError(
                "owner_gate_network_route_invalid"
            ) from None
        if not isinstance(route_network, ipaddress.IPv4Network):
            raise OwnerGateNetworkEvidenceAuthorError(
                "owner_gate_network_route_invalid"
            )
        # Every route remains covered by the signed inventory digest.  Only
        # private IPv4 destinations can collide with the private owner-gate
        # subnet, so public/default routes are deliberately excluded from the
        # overlap set instead of making otherwise valid inventories unusable.
        if any(route_network.subnet_of(parent) for parent in _RFC1918):
            reserved.add(route_network)

    addresses = _items(
        raw.get("private_service_ranges"), label="private_service_ranges"
    )
    for address in addresses:
        if address.get("purpose") != "VPC_PEERING":
            continue
        address_network = address.get("network")
        address_network_prefix = (
            f"{_RESOURCE_PREFIX}projects/{foundation.PROJECT}/global/networks/"
        )
        if (
            not isinstance(address_network, str)
            or not address_network.startswith(address_network_prefix)
        ):
            raise OwnerGateNetworkEvidenceAuthorError(
                "owner_gate_network_private_service_ranges_invalid"
            )
        address_network_name = address_network[len(address_network_prefix) :]
        if (
            _NETWORK_NAME.fullmatch(address_network_name) is None
            or address_network
            != f"{address_network_prefix}{address_network_name}"
        ):
            raise OwnerGateNetworkEvidenceAuthorError(
                "owner_gate_network_private_service_ranges_invalid"
            )
        if address_network_name != foundation.NETWORK_NAME:
            continue
        try:
            prefix_value = address.get("prefixLength")
            if prefix_value is None:
                raise TypeError("missing prefixLength")
            prefix = int(prefix_value)
            network = ipaddress.ip_network(
                f"{address.get('address')}/{prefix}", strict=True
            )
        except (TypeError, ValueError) as exc:
            raise OwnerGateNetworkEvidenceAuthorError(
                "owner_gate_network_private_service_ranges_invalid"
            ) from None
        if not isinstance(network, ipaddress.IPv4Network) or not any(
            network.subnet_of(parent) for parent in _RFC1918
        ):
            raise OwnerGateNetworkEvidenceAuthorError(
                "owner_gate_network_private_service_ranges_invalid"
            )
        reserved.add(network)

    connectors_raw = raw.get("serverless_connectors")
    if not isinstance(connectors_raw, Mapping):
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_serverless_connectors_invalid"
        )
    connector_inventory: dict[str, list[Mapping[str, Any]]] = {}
    for region, items in sorted(connectors_raw.items()):
        if not isinstance(region, str) or _REGION.fullmatch(region) is None:
            raise OwnerGateNetworkEvidenceAuthorError(
                "owner_gate_network_serverless_connectors_invalid"
            )
        connector_inventory[region] = _items(
            items, label="serverless_connectors"
        )
        for connector in connector_inventory[region]:
            network_value = connector.get("network")
            if str(network_value).removeprefix(_RESOURCE_PREFIX) not in {
                network_self_link,
                foundation.NETWORK_NAME,
            }:
                continue
            reserved.add(
                _private_cidr(
                    connector.get("ipCidrRange"), label="serverless_connector"
                )
            )

    peerings = _items(raw.get("peerings"), label="peerings")
    firewall = raw.get("iap_firewall")
    if not isinstance(firewall, Mapping):
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_iap_firewall_invalid"
        )
    allowed = firewall.get("allowed")
    if (
        firewall.get("name") != "allow-iap-ssh"
        or firewall.get("direction") != "INGRESS"
        or firewall.get("disabled", False) is not False
        or firewall.get("sourceRanges") != [foundation.IAP_SOURCE_RANGE]
        or firewall.get("targetTags") != [foundation.IAP_NETWORK_TAG]
        or not isinstance(allowed, list)
        or allowed
        != [{"IPProtocol": "tcp", "ports": ["22"]}]
    ):
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_iap_firewall_invalid"
        )

    desired = ipaddress.ip_network(foundation.OWNER_GATE_SUBNET_CIDR, strict=True)
    if any(desired.overlaps(item) for item in reserved):
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_owner_subnet_overlap"
        )
    receipts = {
        "aggregate_subnets": foundation.sha256_json(
            _canonical_inventory(subnets)
        ),
        "routes": foundation.sha256_json(_canonical_inventory(routes)),
        "peerings": foundation.sha256_json(_canonical_inventory(peerings)),
        "private_service_ranges": foundation.sha256_json(
            _canonical_inventory(addresses)
        ),
        "serverless_connectors": foundation.sha256_json(
            _canonical_inventory(connector_inventory)
        ),
    }
    unsigned = {
        "schema": foundation.NETWORK_EVIDENCE_SCHEMA,
        "collected_at_unix": collected_at_unix,
        "project": foundation.PROJECT,
        "zone": foundation.ZONE,
        "source_instance": foundation.PRODUCTION_SOURCE_VM,
        "source_instance_id": foundation.PRODUCTION_SOURCE_VM_ID,
        "source_instance_self_link": instance_self_link,
        "source_service_account": foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT,
        "network_self_link": network_self_link,
        "subnetwork_self_link": subnet_self_link,
        "source_internal_ip": str(source_ip),
        "subnetwork_cidr": str(production_cidr),
        "reserved_network_ranges": sorted(
            (str(item) for item in reserved),
            key=lambda item: (
                int(ipaddress.ip_network(item).network_address),
                ipaddress.ip_network(item).prefixlen,
            ),
        ),
        "range_inventory_receipts": receipts,
        "private_google_access": target_subnet["privateIpGoogleAccess"],
        "iap_firewall_rule": "allow-iap-ssh",
        "iap_source_range": foundation.IAP_SOURCE_RANGE,
    }
    evidence_sha256 = foundation.sha256_json(unsigned)
    try:
        private_raw = release_author._read_exact_regular(
            signer_author._private_path(release_revision, "network"),
            size=32,
            modes=frozenset({0o600}),
            code="owner_gate_network_signing_key_invalid",
        )
        public_raw = release_author._read_exact_regular(
            signer_author._public_path(release_revision, "network"),
            size=32,
            modes=frozenset({0o400, 0o440, 0o444}),
            code="owner_gate_network_signing_key_invalid",
        )
    except release_author.OwnerGateTrustAuthorError as exc:
        raise OwnerGateNetworkEvidenceAuthorError(str(exc)) from None
    key = Ed25519PrivateKey.from_private_bytes(private_raw)
    key_id = hashlib.sha256(public_raw).hexdigest()
    if key.public_key().public_bytes_raw() != public_raw:
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_signing_key_invalid"
        )
    signed = {
        **unsigned,
        "evidence_sha256": evidence_sha256,
        "collector_public_key_id": key_id,
    }
    result = {
        **signed,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(
            key.sign(
                foundation.NETWORK_EVIDENCE_SIGNATURE_DOMAIN
                + foundation.canonical_json_bytes(signed)
            )
        )
        .rstrip(b"=")
        .decode("ascii"),
    }
    try:
        foundation.ProductionNetworkEvidence.from_mapping(
            result,
            public_key=key.public_key(),
            expected_public_key_id=key_id,
            now_unix=collected_at_unix,
        )
    except foundation.OwnerGateFoundationError as exc:
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_evidence_self_validation_failed"
        ) from None
    return result


def _collect_and_author_with_runner(
    *,
    release_revision: str,
    run_json: RunJson,
    collected_at_unix: int,
) -> Mapping[str, Any]:
    """Private deterministic seam used by tests after fixed live collection."""

    if (
        type(collected_at_unix) is not int
        or collected_at_unix <= 0
    ):
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_time_invalid"
        )
    signer_author._require_authority_directories(
        release_revision, create=False
    )
    return _normalize_and_author(
        _collect_raw(run_json),
        release_revision=release_revision,
        collected_at_unix=collected_at_unix,
    )


def _decode_captured_json(result: _CapturedJson) -> Any:
    if (
        type(result.returncode) is not int
        or result.returncode != 0
        or type(result.stdout) is not bytes
        or type(result.stderr) is not bytes
        or not result.stdout
        or len(result.stdout) > _MAX_CAPTURE_BYTES
        or len(result.stderr) > _MAX_CAPTURE_BYTES
    ):
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_collection_failed"
        )
    try:
        return json.loads(result.stdout.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_collection_invalid"
        ) from None


def _collect_and_author_with_runtime(
    *,
    release_revision: str,
    collected_at_unix: int,
    gcloud_executable: launcher.StableExecutable,
    gcloud_configuration: launcher.StableGcloudConfiguration,
    sealed_runtime_snapshot: Callable[[], Mapping[str, Any]],
    runner: _NetworkEvidenceRunner,
) -> Mapping[str, Any]:
    """Private runtime seam; public entry admits only exact pin classes."""

    try:
        before, prefix, environment = owner_reauth._trusted_snapshot(
            gcloud_executable,
            gcloud_configuration,
        )
        sealed_before = owner_reauth._validate_sealed_runtime_identity(
            sealed_runtime_snapshot(),
            expected_release_revision=release_revision,
            prefix=prefix,
        )
    except owner_reauth.OwnerGateOwnerReauthError as exc:
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_runtime_invalid"
        ) from None
    environment = dict(environment)
    environment["CLOUDSDK_CORE_DISABLE_PROMPTS"] = "1"
    forbidden = {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "PYTHONPATH",
        "PYTHONHOME",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
    }
    if forbidden & set(environment):
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_environment_invalid"
        )

    allowed_static = frozenset(inventory_commands().values())

    def run_json(logical_argv: Sequence[str]) -> Any:
        logical = tuple(logical_argv)
        connector = (
            len(logical) == 9
            and logical[:6]
            == (
                "gcloud",
                "compute",
                "networks",
                "vpc-access",
                "connectors",
                "list",
            )
            and logical[-2]
            == f"--project={foundation.PROJECT}"
            and logical[-1] == "--format=json"
            and logical[-3].startswith("--region=")
            and _REGION.fullmatch(logical[-3].removeprefix("--region="))
            is not None
        )
        if logical not in allowed_static and not connector:
            raise OwnerGateNetworkEvidenceAuthorError(
                "owner_gate_network_command_forbidden"
            )
        full_argv = (
            *prefix,
            *logical[1:],
            f"--account={owner_reauth.OWNER_ACCOUNT}",
            f"--configuration={owner_reauth.GCLOUD_CONFIGURATION}",
            "--quiet",
        )
        return _decode_captured_json(
            runner.run_capture(
                full_argv,
                env=environment,
                timeout_seconds=_CAPTURE_TIMEOUT_SECONDS,
            )
        )

    result = _collect_and_author_with_runner(
        release_revision=release_revision,
        run_json=run_json,
        collected_at_unix=collected_at_unix,
    )
    try:
        after, after_prefix, after_environment = owner_reauth._trusted_snapshot(
            gcloud_executable,
            gcloud_configuration,
        )
        sealed_after = owner_reauth._validate_sealed_runtime_identity(
            sealed_runtime_snapshot(),
            expected_release_revision=release_revision,
            prefix=after_prefix,
        )
    except owner_reauth.OwnerGateOwnerReauthError as exc:
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_runtime_invalid"
        ) from None
    after_environment = dict(after_environment)
    after_environment["CLOUDSDK_CORE_DISABLE_PROMPTS"] = "1"
    if (
        before != after
        or sealed_before != sealed_after
        or prefix != after_prefix
        or environment != after_environment
    ):
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_runtime_changed"
        )
    return result


def collect_and_author(
    *,
    release_revision: str,
    collected_at_unix: int,
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
) -> Mapping[str, Any]:
    """Collect and sign only through the sealed production owner runtime."""

    if (
        type(gcloud_executable) is not launcher.TrustedGcloudExecutable
        or type(gcloud_configuration) is not launcher.PinnedGcloudConfiguration
        or re.fullmatch(r"[0-9a-f]{40}", release_revision or "") is None
    ):
        raise OwnerGateNetworkEvidenceAuthorError(
            "owner_gate_network_runtime_invalid"
        )

    def sealed_snapshot() -> Mapping[str, Any]:
        try:
            return gcloud_executable.sealed_runtime_identity(
                expected_release_sha=release_revision,
            )
        except launcher.OwnerLauncherError as exc:
            raise OwnerGateNetworkEvidenceAuthorError(
                "owner_gate_network_runtime_invalid"
            ) from None

    return _collect_and_author_with_runtime(
        release_revision=release_revision,
        collected_at_unix=collected_at_unix,
        gcloud_executable=gcloud_executable,
        gcloud_configuration=gcloud_configuration,
        sealed_runtime_snapshot=sealed_snapshot,
        runner=_SubprocessNetworkEvidenceRunner(),
    )


__all__ = [
    "OwnerGateNetworkEvidenceAuthorError",
    "collect_and_author",
    "connector_command",
    "inventory_commands",
]
