#!/usr/bin/env python3
"""Fixed owner-side author for owner-gate Cloud observations.

The live boundary accepts only a validated Foundation apply chain, the exact
signed stage-zero streams, and the release-pinned IAP/gcloud capabilities.
Attached-service-account permissions are proved only by the fixed in-host
metadata-token observer carried by the immutable release.  Owner Cloud access
is a separate closed set of read-only GET and ``getIamPolicy`` requests
derived from the final release plan and its signed ancestry evidence.  There
is no caller-selected receipt, probe, URL, host, request body, output path,
proxy, redirect, custom CA, token, signer, or generic gcloud surface.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import math
import os
import re
import socket
import ssl
import stat
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, NoReturn, Sequence
from urllib.parse import urlsplit

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from scripts.canary import direct_iam_identity_author as direct_iam_author
from scripts.canary import direct_iam_identity_authority as direct_iam
from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_preflight as preflight
from scripts.canary import (
    owner_gate_production_ingress_observation as production_ingress,
)
from scripts.canary import owner_gate_project_ancestry as project_ancestry
from scripts.canary import owner_gate_trust_author as release_author
from scripts.canary import trusted_signer_author as signer_author


PHASES = frozenset({"inert", "post_iam"})
ATTACHED_SA_PROBE_SCHEMA = "muncho-owner-gate-attached-sa-permission-probe.v1"
OWNER_ACCOUNT = "lomliev@adventico.com"
HTTP_TIMEOUT_SECONDS = 10
MAX_HTTP_BODY_BYTES = 8 * 1024 * 1024
MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024
MAX_REQUESTS = 160
MAX_ITEMS = 10_000

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{5,30}$")
_REGION = re.compile(r"^[a-z]+(?:-[a-z0-9]+)+[0-9]$")
_RESOURCE_PREFIXES = (
    "https://www.googleapis.com/compute/v1/",
    "https://compute.googleapis.com/compute/v1/",
)
_ALLOWED_HOSTS = frozenset({
    "compute.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "vpcaccess.googleapis.com",
})
_FORBIDDEN_NETWORK_ENVIRONMENT = frozenset({
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "all_proxy",
    "https_proxy",
    "http_proxy",
    "no_proxy",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "SSLKEYLOGFILE",
    "OPENSSL_CONF",
    "OPENSSL_MODULES",
    "CURL_CA_BUNDLE",
    "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH",
})


class OwnerGateCloudObservationAuthorError(RuntimeError):
    """Stable, secret-free Cloud observation authoring failure."""


def _error(code: str, exc: BaseException | None = None) -> NoReturn:
    del exc
    raise OwnerGateCloudObservationAuthorError(code) from None


def _canonical(value: Any) -> bytes:
    try:
        return foundation.canonical_json_bytes(value)
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_cloud_observation_json_invalid", exc)


def _canonical_inventory(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_inventory(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [_canonical_inventory(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    _error("owner_gate_cloud_observation_json_invalid")


def _strict(value: Any, fields: set[str], code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _error(code)
    return value


def _resource(value: Any, *, expected: str, code: str) -> str:
    if not isinstance(value, str):
        _error(code)
    if value not in {
        expected,
        *[f"{prefix}{expected}" for prefix in _RESOURCE_PREFIXES],
    }:
        _error(code)
    return value


def _list_response(value: Any, *, code: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Mapping) or value.get("nextPageToken") not in {None, ""}:
        _error(code)
    items = value.get("items", [])
    if (
        not isinstance(items, list)
        or len(items) > MAX_ITEMS
        or any(not isinstance(item, Mapping) for item in items)
    ):
        _error(code)
    return [dict(item) for item in items]


def _named_list_response(
    value: Any,
    *,
    field: str,
    code: str,
) -> list[Mapping[str, Any]]:
    if not isinstance(value, Mapping) or value.get("nextPageToken") not in {None, ""}:
        _error(code)
    items = value.get(field, [])
    if (
        not isinstance(items, list)
        or len(items) > MAX_ITEMS
        or any(not isinstance(item, Mapping) for item in items)
    ):
        _error(code)
    return [dict(item) for item in items]


def _request_key(request: Mapping[str, str]) -> str:
    method = request.get("method")
    url = request.get("url")
    if method not in {"GET", "POST"} or not isinstance(url, str):
        _error("owner_gate_cloud_observation_request_inventory_invalid")
    return f"{method} {url}"


def _validated_context(
    *,
    plan: foundation.OwnerGateFoundationPlan,
    ancestry_evidence: project_ancestry.ProjectAncestryEvidence,
    phase: str,
) -> tuple[str, ...]:
    if (
        type(plan) is not foundation.OwnerGateFoundationPlan
        or type(ancestry_evidence) is not project_ancestry.ProjectAncestryEvidence
        or phase not in PHASES
    ):
        _error("owner_gate_cloud_observation_input_invalid")
    try:
        plan.spec.validate()
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_cloud_observation_input_invalid", exc)
    chain = tuple(
        str(node["resource_name"]) for node in ancestry_evidence.ordered_chain
    )
    if (
        not plan.spec.final_release_bound
        or ancestry_evidence.signed_evidence_sha256
        != plan.spec.ancestry_evidence_sha256
        or ancestry_evidence.organization_id != plan.spec.organization_id
        or len(chain) < 2
        or not chain[0].startswith("projects/")
        or chain[-1] != plan.spec.organization_resource
    ):
        _error("owner_gate_cloud_observation_input_invalid")
    return chain


def request_inventory(
    *,
    plan: foundation.OwnerGateFoundationPlan,
    ancestry_evidence: project_ancestry.ProjectAncestryEvidence,
    phase: str,
    _connector_regions: Sequence[str] = (),
) -> tuple[Mapping[str, str], ...]:
    """Return the exact request surface for one release and signed chain."""

    chain = _validated_context(
        plan=plan,
        ancestry_evidence=ancestry_evidence,
        phase=phase,
    )
    try:
        requests = preflight.read_only_cloud_requests(
            plan=plan,
            project_resource_name=chain[0],
            resource_ancestor_chain=chain[1:],
            connector_regions=_connector_regions,
        )
    except preflight.OwnerGatePreflightError as exc:
        _error("owner_gate_cloud_observation_request_inventory_invalid", exc)
    keys = tuple(_request_key(item) for item in requests)
    if not requests or len(requests) > MAX_REQUESTS or len(keys) != len(set(keys)):
        _error("owner_gate_cloud_observation_request_inventory_invalid")
    return requests


def _reject_ambient_network_environment() -> None:
    if any(os.environ.get(name) for name in _FORBIDDEN_NETWORK_ENVIRONMENT):
        _error("owner_gate_cloud_observation_tls_invalid")
    try:
        launcher._reject_custom_ca_environment()
    except Exception as exc:
        _error("owner_gate_cloud_observation_tls_invalid", exc)


def _default_connection(host: str) -> Any:
    if host not in _ALLOWED_HOSTS:
        _error("owner_gate_cloud_observation_request_forbidden")
    _reject_ambient_network_environment()
    try:
        context = launcher._pinned_system_tls_context()
    except Exception as exc:
        _error("owner_gate_cloud_observation_tls_invalid", exc)
    return http.client.HTTPSConnection(
        host,
        443,
        timeout=HTTP_TIMEOUT_SECONDS,
        context=context,
    )


def _bounded_body(response: Any) -> bytes:
    length_header = response.getheader("Content-Length")
    if length_header is not None:
        try:
            length = int(length_header)
        except (TypeError, ValueError) as exc:
            _error("owner_gate_cloud_observation_http_invalid", exc)
        if length < 1 or length > MAX_HTTP_BODY_BYTES:
            _error("owner_gate_cloud_observation_http_invalid")
    body = response.read(MAX_HTTP_BODY_BYTES + 1)
    if not isinstance(body, bytes) or not body or len(body) > MAX_HTTP_BODY_BYTES:
        _error("owner_gate_cloud_observation_http_invalid")
    return body


def _decode_json(raw: bytes) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if not isinstance(key, str) or key in result:
                _error("owner_gate_cloud_observation_json_invalid")
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        _error("owner_gate_cloud_observation_json_invalid")

    import json

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except OwnerGateCloudObservationAuthorError:
        raise
    except (UnicodeError, ValueError) as exc:
        _error("owner_gate_cloud_observation_json_invalid", exc)
    if not isinstance(value, Mapping):
        _error("owner_gate_cloud_observation_json_invalid")
    return dict(value)


def _json_content_type(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parts = tuple(item.strip().lower().replace(" ", "") for item in value.split(";"))
    return bool(
        parts
        and parts[0] == "application/json"
        and (len(parts) == 1 or (len(parts) == 2 and parts[1] == "charset=utf-8"))
    )


class _FixedCloudFactsReader:
    """Closed reader: the only public operation is two stable full snapshots."""

    def __init__(
        self,
        *,
        token: Any,
        plan: foundation.OwnerGateFoundationPlan,
        ancestry_evidence: project_ancestry.ProjectAncestryEvidence,
        phase: str,
        _connection_factories: Mapping[str, Callable[[], Any]] | None = None,
    ) -> None:
        if (
            type(token) is not direct_iam_author._GcloudAccessToken
            or len(token._raw) < 20
        ):
            _error("owner_gate_cloud_observation_capability_invalid")
        self._token = token
        self._base_requests = tuple(
            dict(item)
            for item in request_inventory(
                plan=plan,
                ancestry_evidence=ancestry_evidence,
                phase=phase,
            )
        )
        self._plan = plan
        self._ancestry_evidence = ancestry_evidence
        self._phase = phase
        self._requests = self._base_requests
        keys = tuple(_request_key(item) for item in self._base_requests)
        if (
            not self._base_requests
            or len(keys) > MAX_REQUESTS
            or len(keys) != len(set(keys))
        ):
            _error("owner_gate_cloud_observation_request_inventory_invalid")
        supplied = dict(_connection_factories or {})
        if set(supplied) - _ALLOWED_HOSTS or any(
            not callable(factory) for factory in supplied.values()
        ):
            _error("owner_gate_cloud_observation_capability_invalid")
        self._factories = {
            host: supplied.get(host, lambda host=host: _default_connection(host))
            for host in _ALLOWED_HOSTS
        }

    def _request_exact(
        self, request: Mapping[str, str]
    ) -> tuple[Mapping[str, Any], int]:
        if request not in self._requests:
            _error("owner_gate_cloud_observation_request_forbidden")
        parsed = urlsplit(str(request["url"]))
        if (
            parsed.scheme != "https"
            or parsed.hostname not in _ALLOWED_HOSTS
            or parsed.port not in {None, 443}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or not parsed.path.startswith("/")
        ):
            _error("owner_gate_cloud_observation_request_forbidden")
        method = request["method"]
        body_text = request.get("body")
        if (method == "GET" and body_text is not None) or (
            method == "POST" and not isinstance(body_text, str)
        ):
            _error("owner_gate_cloud_observation_request_forbidden")
        try:
            bearer = self._token._raw.decode("ascii", errors="strict")
        except UnicodeError as exc:
            _error("owner_gate_cloud_observation_capability_invalid", exc)
        headers = {
            "Authorization": f"Bearer {bearer}",
            "Accept": "application/json",
            "Connection": "close",
        }
        body = None if body_text is None else body_text.encode("ascii", errors="strict")
        if body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        connection = self._factories[str(parsed.hostname)]()
        try:
            target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            connection.request(method, target, body=body, headers=headers)
            response = connection.getresponse()
            response_body = _bounded_body(response)
            if (
                response.status != 200
                or response.getheader("Location") is not None
                or not _json_content_type(response.getheader("Content-Type"))
            ):
                _error("owner_gate_cloud_observation_resource_unavailable")
        except OwnerGateCloudObservationAuthorError:
            raise
        except (
            OSError,
            socket.timeout,
            ssl.SSLError,
            http.client.HTTPException,
        ) as exc:
            _error("owner_gate_cloud_observation_resource_unavailable", exc)
        finally:
            bearer = ""
            connection.close()
        return _decode_json(response_body), len(response_body)

    def _collect_once(self) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        total = 0
        for request in self._requests:
            value, size = self._request_exact(request)
            total += size
            if total > MAX_SNAPSHOT_BYTES:
                _error("owner_gate_cloud_observation_http_invalid")
            result[_request_key(request)] = value
        return result

    @staticmethod
    def _location_regions(value: Any) -> tuple[str, ...]:
        locations = _named_list_response(
            value,
            field="locations",
            code="owner_gate_cloud_observation_network_invalid",
        )
        regions: list[str] = []
        prefix = f"projects/{foundation.PROJECT}/locations/"
        for location in locations:
            name = location.get("name")
            location_id = location.get("locationId")
            if (
                not isinstance(name, str)
                or not isinstance(location_id, str)
                or name != f"{prefix}{location_id}"
                or _REGION.fullmatch(location_id) is None
                or location_id in regions
            ):
                _error("owner_gate_cloud_observation_network_invalid")
            regions.append(location_id)
        if not regions or len(regions) > 100:
            _error("owner_gate_cloud_observation_network_invalid")
        return tuple(sorted(regions))

    def collect(self) -> Mapping[str, Any]:
        location_url = _url(
            f"/v1/projects/{foundation.PROJECT}/locations?pageSize=100",
            host="vpcaccess.googleapis.com",
        )
        location_request = {"method": "GET", "url": location_url}
        if location_request not in self._base_requests:
            _error("owner_gate_cloud_observation_request_inventory_invalid")
        first_location, first_size = self._request_exact(location_request)
        second_location, second_size = self._request_exact(location_request)
        if first_size + second_size > MAX_SNAPSHOT_BYTES or _canonical(
            _canonical_inventory(first_location)
        ) != _canonical(_canonical_inventory(second_location)):
            _error("owner_gate_cloud_observation_facts_unstable")
        regions = self._location_regions(second_location)
        self._requests = tuple(
            dict(item)
            for item in request_inventory(
                plan=self._plan,
                ancestry_evidence=self._ancestry_evidence,
                phase=self._phase,
                _connector_regions=regions,
            )
        )
        first = self._collect_once()
        second = self._collect_once()
        if _canonical(_canonical_inventory(first)) != _canonical(
            _canonical_inventory(second)
        ):
            _error("owner_gate_cloud_observation_facts_unstable")
        return second


def _url(path: str, *, host: str = "compute.googleapis.com") -> str:
    return f"https://{host}{path}"


class _Facts:
    def __init__(
        self,
        *,
        raw: Mapping[str, Any],
        requests: Sequence[Mapping[str, str]],
    ) -> None:
        expected = {_request_key(item) for item in requests}
        if not isinstance(raw, Mapping) or set(raw) != expected:
            _error("owner_gate_cloud_observation_facts_invalid")
        self.raw = raw

    def get(self, method: str, url: str) -> Mapping[str, Any]:
        value = self.raw.get(f"{method} {url}")
        if not isinstance(value, Mapping):
            _error("owner_gate_cloud_observation_facts_invalid")
        return value


def _validate_live_ancestry(
    facts: _Facts,
    *,
    ancestry_evidence: project_ancestry.ProjectAncestryEvidence,
) -> None:
    for index, signed in enumerate(ancestry_evidence.ordered_chain):
        resource = signed.get("resource_name")
        resource_type = (
            "projects"
            if index == 0
            else "organizations"
            if index == len(ancestry_evidence.ordered_chain) - 1
            else "folders"
        )
        if not isinstance(resource, str):
            _error("owner_gate_cloud_observation_ancestry_invalid")
        raw = facts.get(
            "GET",
            _url(
                f"/v3/{resource}",
                host="cloudresourcemanager.googleapis.com",
            ),
        )
        try:
            normalized = project_ancestry._normalized_node(
                raw,
                expected_resource_name=resource,
                expected_type=resource_type,
            )
        except project_ancestry.OwnerGateProjectAncestryError as exc:
            _error("owner_gate_cloud_observation_ancestry_invalid", exc)
        if (index == 0 and raw.get("projectId") != foundation.PROJECT) or _canonical(
            normalized
        ) != _canonical(dict(signed)):
            _error("owner_gate_cloud_observation_ancestry_invalid")


def _instance_metadata(value: Any) -> Mapping[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        _error("owner_gate_cloud_observation_instance_invalid")
    items = value.get("items", [])
    if not isinstance(items, list) or len(items) > 256:
        _error("owner_gate_cloud_observation_instance_invalid")
    result: dict[str, str] = {}
    for item in items:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("key"), str)
            or not isinstance(item.get("value"), str)
            or item["key"] in result
            or len(item["value"]) > 1024 * 1024
        ):
            _error("owner_gate_cloud_observation_instance_invalid")
        result[item["key"]] = item["value"]
    return result


def _bool_metadata(metadata: Mapping[str, str], key: str, *, default: bool) -> bool:
    value = metadata.get(key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        _error("owner_gate_cloud_observation_instance_invalid")
    return normalized == "true"


def _single_interface(
    value: Mapping[str, Any],
    *,
    network: str,
    subnet: str,
    ip: str,
    code: str,
) -> Mapping[str, Any]:
    interfaces = value.get("networkInterfaces")
    if (
        not isinstance(interfaces, list)
        or len(interfaces) != 1
        or not isinstance(interfaces[0], Mapping)
    ):
        _error(code)
    interface = interfaces[0]
    _resource(interface.get("network"), expected=network, code=code)
    _resource(interface.get("subnetwork"), expected=subnet, code=code)
    if interface.get("networkIP") != ip:
        _error(code)
    return interface


def _source_projection(value: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = (
        f"projects/{foundation.PROJECT}/zones/{foundation.ZONE}/instances/"
        f"{foundation.PRODUCTION_SOURCE_VM}"
    )
    _resource(
        value.get("selfLink"),
        expected=expected,
        code="owner_gate_cloud_observation_source_invalid",
    )
    network = f"projects/{foundation.PROJECT}/global/networks/{foundation.NETWORK_NAME}"
    subnet = (
        f"projects/{foundation.PROJECT}/regions/{foundation.REGION}/subnetworks/"
        f"{foundation.PRODUCTION_SUBNET_NAME}"
    )
    interface = _single_interface(
        value,
        network=network,
        subnet=subnet,
        ip="10.80.0.2",
        code="owner_gate_cloud_observation_source_invalid",
    )
    accounts = value.get("serviceAccounts")
    if (
        value.get("name") != foundation.PRODUCTION_SOURCE_VM
        or str(value.get("id")) != foundation.PRODUCTION_SOURCE_VM_ID
        or not isinstance(accounts, list)
        or len(accounts) != 1
        or not isinstance(accounts[0], Mapping)
        or accounts[0].get("email") != foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT
    ):
        _error("owner_gate_cloud_observation_source_invalid")
    return {
        "name": foundation.PRODUCTION_SOURCE_VM,
        "numeric_id": foundation.PRODUCTION_SOURCE_VM_ID,
        "internal_ip": "10.80.0.2",
        "service_account": foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT,
        "network": interface["network"],
        "subnetwork": interface["subnetwork"],
    }


def _aggregated_subnets(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Mapping) or value.get("nextPageToken") not in {None, ""}:
        _error("owner_gate_cloud_observation_subnet_invalid")
    scopes = value.get("items", {})
    if not isinstance(scopes, Mapping) or len(scopes) > 256:
        _error("owner_gate_cloud_observation_subnet_invalid")
    result: list[Mapping[str, Any]] = []
    for scoped in scopes.values():
        if not isinstance(scoped, Mapping):
            _error("owner_gate_cloud_observation_subnet_invalid")
        items = scoped.get("subnetworks", [])
        if not isinstance(items, list) or any(
            not isinstance(item, Mapping) for item in items
        ):
            _error("owner_gate_cloud_observation_subnet_invalid")
        result.extend(dict(item) for item in items)
    if len(result) > MAX_ITEMS:
        _error("owner_gate_cloud_observation_subnet_invalid")
    return result


def _subnet_projection(
    subnet: Mapping[str, Any],
    *,
    aggregate: Sequence[Mapping[str, Any]],
    routes: Sequence[Mapping[str, Any]],
    addresses: Sequence[Mapping[str, Any]],
    connectors: Mapping[str, Sequence[Mapping[str, Any]]],
    peerings: Sequence[Mapping[str, Any]],
    expected_numeric_id: str,
) -> Mapping[str, Any]:
    network = f"projects/{foundation.PROJECT}/global/networks/{foundation.NETWORK_NAME}"
    resource = (
        f"projects/{foundation.PROJECT}/regions/{foundation.REGION}/subnetworks/"
        f"{foundation.OWNER_GATE_SUBNET_NAME}"
    )
    _resource(
        subnet.get("selfLink"),
        expected=resource,
        code="owner_gate_cloud_observation_subnet_invalid",
    )
    _resource(
        subnet.get("network"),
        expected=network,
        code="owner_gate_cloud_observation_subnet_invalid",
    )
    if (
        subnet.get("name") != foundation.OWNER_GATE_SUBNET_NAME
        or str(subnet.get("id")) != expected_numeric_id
        or subnet.get("ipCidrRange") != foundation.OWNER_GATE_SUBNET_CIDR
        or subnet.get("privateIpGoogleAccess") is not True
        or subnet.get("stackType", "IPV4_ONLY") != "IPV4_ONLY"
        or subnet.get("purpose", "PRIVATE") != "PRIVATE"
        or subnet.get("secondaryIpRanges", []) != []
    ):
        _error("owner_gate_cloud_observation_subnet_invalid")
    target = ipaddress.ip_network(foundation.OWNER_GATE_SUBNET_CIDR, strict=True)
    reserved: list[ipaddress.IPv4Network] = []
    exact_count = 0
    for item in aggregate:
        try:
            item_network = str(item.get("network", ""))
            normalized_network = item_network
            for prefix in _RESOURCE_PREFIXES:
                normalized_network = normalized_network.removeprefix(prefix)
            if normalized_network != network:
                continue
            cidr = ipaddress.ip_network(str(item.get("ipCidrRange")), strict=True)
        except (TypeError, ValueError):
            _error("owner_gate_cloud_observation_subnet_invalid")
        if not isinstance(cidr, ipaddress.IPv4Network) or not cidr.is_private:
            _error("owner_gate_cloud_observation_subnet_invalid")
        item_self = str(item.get("selfLink", ""))
        normalized_self = item_self
        for prefix in _RESOURCE_PREFIXES:
            normalized_self = normalized_self.removeprefix(prefix)
        if normalized_self == resource:
            if cidr != target:
                _error("owner_gate_cloud_observation_subnet_invalid")
            exact_count += 1
        else:
            reserved.append(cidr)
        secondary = item.get("secondaryIpRanges", [])
        if not isinstance(secondary, list):
            _error("owner_gate_cloud_observation_subnet_invalid")
        for secondary_range in secondary:
            try:
                secondary_cidr = ipaddress.ip_network(
                    str(secondary_range.get("ipCidrRange")), strict=True
                )
            except (AttributeError, ValueError):
                _error("owner_gate_cloud_observation_subnet_invalid")
            if (
                not isinstance(secondary_cidr, ipaddress.IPv4Network)
                or not secondary_cidr.is_private
            ):
                _error("owner_gate_cloud_observation_subnet_invalid")
            reserved.append(secondary_cidr)
    if exact_count != 1:
        _error("owner_gate_cloud_observation_subnet_invalid")
    owner_route_count = 0
    for route in routes:
        item_network = str(route.get("network", ""))
        for prefix in _RESOURCE_PREFIXES:
            item_network = item_network.removeprefix(prefix)
        if item_network != network:
            continue
        try:
            destination = ipaddress.ip_network(str(route.get("destRange")), strict=True)
        except ValueError:
            _error("owner_gate_cloud_observation_network_invalid")
        if not isinstance(destination, ipaddress.IPv4Network):
            _error("owner_gate_cloud_observation_network_invalid")
        next_hop_network = str(route.get("nextHopNetwork", ""))
        for prefix in _RESOURCE_PREFIXES:
            next_hop_network = next_hop_network.removeprefix(prefix)
        if destination == target:
            if (
                next_hop_network != network
                or route.get("routeType") != "SUBNET"
                or route.get("priority") != 0
            ):
                _error("owner_gate_cloud_observation_network_invalid")
            owner_route_count += 1
            continue
        if destination.is_private:
            reserved.append(destination)
    if owner_route_count != 1:
        _error("owner_gate_cloud_observation_network_invalid")
    for address in addresses:
        if address.get("purpose") != "VPC_PEERING":
            continue
        item_network = str(address.get("network", ""))
        for prefix in _RESOURCE_PREFIXES:
            item_network = item_network.removeprefix(prefix)
        if item_network != network:
            continue
        try:
            prefix = int(address["prefixLength"])
            address_range = ipaddress.ip_network(
                f"{address.get('address')}/{prefix}", strict=True
            )
        except (KeyError, TypeError, ValueError):
            _error("owner_gate_cloud_observation_network_invalid")
        if (
            not isinstance(address_range, ipaddress.IPv4Network)
            or not address_range.is_private
        ):
            _error("owner_gate_cloud_observation_network_invalid")
        reserved.append(address_range)
    connector_inventory: dict[str, list[Mapping[str, Any]]] = {}
    for region, values in sorted(connectors.items()):
        if _REGION.fullmatch(region) is None:
            _error("owner_gate_cloud_observation_network_invalid")
        connector_inventory[region] = [dict(item) for item in values]
        for connector in values:
            connector_network = str(connector.get("network", ""))
            for prefix in _RESOURCE_PREFIXES:
                connector_network = connector_network.removeprefix(prefix)
            if connector_network not in {network, foundation.NETWORK_NAME}:
                continue
            try:
                connector_range = ipaddress.ip_network(
                    str(connector.get("ipCidrRange")), strict=True
                )
            except ValueError:
                _error("owner_gate_cloud_observation_network_invalid")
            if (
                not isinstance(connector_range, ipaddress.IPv4Network)
                or not connector_range.is_private
            ):
                _error("owner_gate_cloud_observation_network_invalid")
            reserved.append(connector_range)
    if any(target.overlaps(item) for item in reserved):
        _error("owner_gate_cloud_observation_subnet_invalid")
    inventory = {
        "aggregate_subnets": list(aggregate),
        "routes": list(routes),
        "peerings": list(peerings),
        "private_service_ranges": list(addresses),
        "serverless_connectors": connector_inventory,
    }
    return {
        "name": foundation.OWNER_GATE_SUBNET_NAME,
        "network": subnet["network"],
        "cidr": foundation.OWNER_GATE_SUBNET_CIDR,
        "private_google_access": True,
        "stack_type": "IPV4_ONLY",
        "overlap_count": 0,
        "route_inventory_sha256": foundation.sha256_json(
            _canonical_inventory(inventory)
        ),
    }


def _owner_instance_projection(
    value: Mapping[str, Any],
    *,
    project_metadata: Mapping[str, str],
    expected_identity: Mapping[str, Any],
    boot_disk: Mapping[str, Any],
) -> Mapping[str, Any]:
    resource = (
        f"projects/{foundation.PROJECT}/zones/{foundation.ZONE}/instances/"
        f"{foundation.VM_NAME}"
    )
    _resource(
        value.get("selfLink"),
        expected=resource,
        code="owner_gate_cloud_observation_instance_invalid",
    )
    network = f"projects/{foundation.PROJECT}/global/networks/{foundation.NETWORK_NAME}"
    subnet = (
        f"projects/{foundation.PROJECT}/regions/{foundation.REGION}/subnetworks/"
        f"{foundation.OWNER_GATE_SUBNET_NAME}"
    )
    interface = _single_interface(
        value,
        network=network,
        subnet=subnet,
        ip=foundation.OWNER_GATE_PRIVATE_IP,
        code="owner_gate_cloud_observation_instance_invalid",
    )
    accounts = value.get("serviceAccounts")
    tags = (
        value.get("tags", {}).get("items", [])
        if isinstance(value.get("tags", {}), Mapping)
        else None
    )
    shielded = value.get("shieldedInstanceConfig")
    instance_metadata = _instance_metadata(value.get("metadata"))
    effective_metadata = {**project_metadata, **instance_metadata}
    access_configs = interface.get("accessConfigs", [])
    expected_sa = f"{foundation.SERVICE_ACCOUNT_NAME}@{foundation.PROJECT}.iam.gserviceaccount.com"
    expected_disk = (
        f"projects/{foundation.PROJECT}/zones/{foundation.ZONE}/disks/"
        f"{foundation.VM_NAME}"
    )
    attachments = value.get("disks")
    if (
        not isinstance(attachments, list)
        or len(attachments) != 1
        or not isinstance(attachments[0], Mapping)
    ):
        _error("owner_gate_cloud_observation_instance_invalid")
    attachment = attachments[0]
    _resource(
        attachment.get("source"),
        expected=expected_disk,
        code="owner_gate_cloud_observation_instance_invalid",
    )
    _resource(
        boot_disk.get("selfLink"),
        expected=expected_disk,
        code="owner_gate_cloud_observation_instance_invalid",
    )
    expected_disk_type = (
        f"projects/{foundation.PROJECT}/zones/{foundation.ZONE}/diskTypes/"
        f"{foundation.BOOT_DISK_TYPE}"
    )
    _resource(
        boot_disk.get("type"),
        expected=expected_disk_type,
        code="owner_gate_cloud_observation_instance_invalid",
    )
    if (
        value.get("name") != foundation.VM_NAME
        or str(value.get("id")) != str(expected_identity.get("numeric_id"))
        or value.get("status") != "RUNNING"
        or not isinstance(access_configs, list)
        or access_configs
        or not isinstance(accounts, list)
        or len(accounts) != 1
        or not isinstance(accounts[0], Mapping)
        or accounts[0].get("email") != expected_sa
        or sorted(accounts[0].get("scopes", []))
        != sorted(foundation.OWNER_GATE_OAUTH_SCOPES)
        or not isinstance(tags, list)
        or set(tags) != {foundation.IAP_NETWORK_TAG, foundation.OWNER_GATE_NETWORK_TAG}
        or not isinstance(shielded, Mapping)
        or shielded.get("enableSecureBoot") is not True
        or shielded.get("enableVtpm") is not True
        or shielded.get("enableIntegrityMonitoring") is not True
        or not str(value.get("machineType", "")).endswith(
            f"/machineTypes/{foundation.MACHINE_TYPE}"
        )
        or _bool_metadata(effective_metadata, "serial-port-enable", default=False)
        is not False
        or _bool_metadata(effective_metadata, "block-project-ssh-keys", default=False)
        is not True
        or _bool_metadata(effective_metadata, "enable-oslogin", default=False)
        is not True
        or any(
            key in effective_metadata
            for key in ("startup-script", "startup-script-url")
        )
        or boot_disk.get("name") != foundation.VM_NAME
        or str(boot_disk.get("id"))
        != str(expected_identity.get("boot_disk_numeric_id"))
        or str(boot_disk.get("sizeGb"))
        != str(expected_identity.get("boot_disk_size_gb"))
        or attachment.get("deviceName") != expected_identity.get("boot_disk_name")
        or attachment.get("boot") is not expected_identity.get("boot_disk_boot")
        or attachment.get("autoDelete")
        is not expected_identity.get("boot_disk_auto_delete")
        or attachment.get("mode") != expected_identity.get("boot_disk_mode")
        or attachment.get("interface") != expected_identity.get("boot_disk_interface")
        or attachment.get("type") != expected_identity.get("boot_disk_attachment_type")
        or attachment.get("index")
        != expected_identity.get("boot_disk_attachment_index")
    ):
        _error("owner_gate_cloud_observation_instance_invalid")
    return {
        "name": foundation.VM_NAME,
        "numeric_id": str(value["id"]),
        "status": "RUNNING",
        "network": interface["network"],
        "subnetwork": interface["subnetwork"],
        "internal_ip": foundation.OWNER_GATE_PRIVATE_IP,
        "access_config_count": 0,
        "service_accounts": [expected_sa],
        "oauth_scopes": list(foundation.OWNER_GATE_OAUTH_SCOPES),
        "tags": sorted({foundation.IAP_NETWORK_TAG, foundation.OWNER_GATE_NETWORK_TAG}),
        "shielded_secure_boot": True,
        "shielded_vtpm": True,
        "shielded_integrity_monitoring": True,
        "serial_port_enabled": False,
        "project_ssh_keys_blocked": True,
        "os_login_enabled": True,
        "startup_script_present": False,
    }


def _normalize_condition(value: Any) -> Mapping[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) - {
        "title",
        "description",
        "expression",
    }:
        _error("owner_gate_cloud_observation_iam_invalid")
    normalized = {
        "title": value.get("title"),
        "description": value.get("description", ""),
        "expression": value.get("expression"),
    }
    if any(not isinstance(item, str) for item in normalized.values()):
        _error("owner_gate_cloud_observation_iam_invalid")
    return normalized


def _policy_bindings(value: Any, *, resource: str) -> list[Mapping[str, Any]]:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("etag"), str)
        or not value.get("etag")
        or type(value.get("version", 1)) is not int
        or value.get("version", 1) not in {1, 3}
    ):
        _error("owner_gate_cloud_observation_iam_invalid")
    bindings = value.get("bindings", [])
    if not isinstance(bindings, list) or len(bindings) > 4096:
        _error("owner_gate_cloud_observation_iam_invalid")
    version = value.get("version", 1)
    normalized: list[Mapping[str, Any]] = []
    for binding in bindings:
        if (
            not isinstance(binding, Mapping)
            or set(binding) - {"role", "members", "condition"}
            or not isinstance(binding.get("role"), str)
            or not isinstance(binding.get("members"), list)
            or not binding["members"]
            or any(not isinstance(member, str) for member in binding["members"])
            or len(set(binding["members"])) != len(binding["members"])
        ):
            _error("owner_gate_cloud_observation_iam_invalid")
        normalized.append({
            "resource": resource,
            "role": binding["role"],
            "members": sorted(binding["members"]),
            "condition": _normalize_condition(binding.get("condition")),
        })
    if any(item["condition"] is not None for item in normalized) and version != 3:
        _error("owner_gate_cloud_observation_iam_invalid")
    normalized.sort(key=_canonical)
    return normalized


def _role_permissions(
    value: Any,
    *,
    name: str,
    title: str,
    description: str,
    permissions: Sequence[str],
) -> list[str]:
    if (
        not isinstance(value, Mapping)
        or value.get("name") != name
        or value.get("title") != title
        or value.get("description", "") != description
        or value.get("stage") != "GA"
        or value.get("deleted", False) is not False
        or not isinstance(value.get("includedPermissions"), list)
        or sorted(value["includedPermissions"]) != sorted(permissions)
        or len(value["includedPermissions"]) != len(set(value["includedPermissions"]))
    ):
        _error("owner_gate_cloud_observation_iam_invalid")
    return sorted(value["includedPermissions"])


def _iam_projection(
    *,
    facts: _Facts,
    plan: foundation.OwnerGateFoundationPlan,
    chain: Sequence[str],
    phase: str,
    expected_service_account_unique_id: str,
    permission_probe: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    project = foundation.PROJECT
    iam_host = "iam.googleapis.com"
    sa_email = f"{foundation.SERVICE_ACCOUNT_NAME}@{project}.iam.gserviceaccount.com"
    member = f"serviceAccount:{sa_email}"
    sa_resource = f"projects/{project}/serviceAccounts/{sa_email}"
    sa_path = f"/v1/projects/{project}/serviceAccounts/{foundation.SERVICE_ACCOUNT_NAME}%40{project}.iam.gserviceaccount.com"
    service_account = facts.get("GET", _url(sa_path, host=iam_host))
    if (
        service_account.get("name") != sa_resource
        or service_account.get("projectId") != project
        or service_account.get("email") != sa_email
        or service_account.get("disabled", False) is not False
        or str(service_account.get("uniqueId")) != expected_service_account_unique_id
    ):
        _error("owner_gate_cloud_observation_service_account_invalid")
    service_account_unique_id = str(service_account["uniqueId"])
    keys = _named_list_response(
        facts.get(
            "GET",
            _url(
                f"{sa_path}/keys?keyTypes=USER_MANAGED",
                host=iam_host,
            ),
        ),
        field="keys",
        code="owner_gate_cloud_observation_service_account_invalid",
    )
    if any(key.get("keyType") not in {None, "USER_MANAGED"} for key in keys):
        _error("owner_gate_cloud_observation_service_account_invalid")
    user_key_count = len(keys)
    if user_key_count:
        _error("owner_gate_cloud_observation_service_account_invalid")
    sa_policy_url = _url(
        f"{sa_path}:getIamPolicy?options.requestedPolicyVersion=3",
        host=iam_host,
    )
    sa_policy = facts.get("POST", sa_policy_url)
    if _policy_bindings(sa_policy, resource=sa_resource):
        _error("owner_gate_cloud_observation_service_account_invalid")

    policies: list[list[Mapping[str, Any]]] = []
    for index, resource in enumerate(chain):
        policies.append(
            _policy_bindings(
                facts.get(
                    "POST",
                    _url(
                        f"/v3/{resource}:getIamPolicy",
                        host="cloudresourcemanager.googleapis.com",
                    ),
                ),
                resource=(f"projects/{project}" if index == 0 else resource),
            )
        )

    project_read_role = plan.spec.read_only_iam_role
    mutation_role = plan.spec.custom_role
    ancestor_role = plan.spec.ancestor_read_only_iam_role
    project_permissions = _role_permissions(
        facts.get("GET", _url(f"/v1/{project_read_role}", host=iam_host)),
        name=project_read_role,
        title=foundation.PROJECT_READ_ROLE_TITLE,
        description=foundation.PROJECT_READ_ROLE_DESCRIPTION,
        permissions=foundation.READ_ONLY_IAM_PERMISSIONS,
    )
    mutation_permissions = _role_permissions(
        facts.get("GET", _url(f"/v1/{mutation_role}", host=iam_host)),
        name=mutation_role,
        title=foundation.MUTATION_ROLE_TITLE,
        description=foundation.MUTATION_ROLE_DESCRIPTION,
        permissions=foundation.MUTATION_PERMISSIONS,
    )
    ancestor_permissions = _role_permissions(
        facts.get("GET", _url(f"/v1/{ancestor_role}", host=iam_host)),
        name=ancestor_role,
        title=foundation.ANCESTOR_READ_ROLE_TITLE,
        description=foundation.ANCESTOR_READ_ROLE_DESCRIPTION,
        permissions=foundation.DIRECT_IAM_ANCESTOR_PERMISSIONS,
    )
    expected_bindings: list[dict[str, Any]] = [
        {
            "resource": f"projects/{project}",
            "role": project_read_role,
            "members": [member],
            "condition": None,
        },
        {
            "resource": chain[-1],
            "role": ancestor_role,
            "members": [member],
            "condition": None,
        },
    ]
    if phase == "post_iam":
        expected_bindings.append({
            "resource": f"projects/{project}",
            "role": mutation_role,
            "members": [member],
            "condition": {
                "title": foundation.MUTATION_CONDITION_TITLE,
                "description": foundation.MUTATION_CONDITION_DESCRIPTION,
                "expression": foundation._condition_expression(),
            },
        })

    def owner_gate_principal(candidate: str) -> bool:
        return candidate == member or candidate == (
            "principal://iam.googleapis.com/projects/-/serviceAccounts/"
            f"{service_account_unique_id}"
        )

    subject_bindings = [
        binding
        for policy in policies
        for binding in policy
        if any(owner_gate_principal(candidate) for candidate in binding["members"])
    ]
    if sorted(subject_bindings, key=_canonical) != sorted(
        expected_bindings, key=_canonical
    ):
        _error("owner_gate_cloud_observation_iam_invalid")

    mutation_present = phase == "post_iam"
    expected_probe = preflight.expected_effective_permission_probe(mutation_present)
    if not isinstance(permission_probe, Mapping) or _canonical(
        _canonical_inventory(permission_probe)
    ) != _canonical(expected_probe):
        _error("owner_gate_cloud_observation_permission_probe_invalid")
    service_projection = {
        "email": sa_email,
        "disabled": False,
        "user_managed_key_count": 0,
        "project_roles": sorted([
            project_read_role,
            *([mutation_role] if mutation_present else []),
        ]),
        "effective_sensitive_permissions": (
            sorted(foundation.EXECUTION_PERMISSIONS) if mutation_present else []
        ),
        "effective_permissions_probe_verified": True,
        "effective_permission_probe": expected_probe,
    }
    iam_projection = {
        "custom_role_permissions": mutation_permissions,
        "mutation_binding_present": mutation_present,
        "forbidden_roles": [],
        "condition_expression": foundation._condition_expression(),
        "read_only_role_permissions": project_permissions,
        "read_only_binding_present": True,
        "ancestor_read_only_permissions": ancestor_permissions,
    }
    return service_projection, iam_projection


def _allowed_tcp_port(rule: Mapping[str, Any], port: int) -> bool:
    allowed = rule.get("allowed", [])
    if not isinstance(allowed, list):
        _error("owner_gate_cloud_observation_firewall_invalid")
    for item in allowed:
        if not isinstance(item, Mapping):
            _error("owner_gate_cloud_observation_firewall_invalid")
        protocol = item.get("IPProtocol")
        ports = item.get("ports")
        if protocol in {"all", None}:
            return True
        if protocol != "tcp":
            continue
        if ports is None:
            return True
        if not isinstance(ports, list):
            _error("owner_gate_cloud_observation_firewall_invalid")
        for entry in ports:
            if not isinstance(entry, str):
                _error("owner_gate_cloud_observation_firewall_invalid")
            if entry.isdigit() and int(entry) == port:
                return True
            if "-" in entry:
                try:
                    low, high = (int(part) for part in entry.split("-", 1))
                except ValueError:
                    _error("owner_gate_cloud_observation_firewall_invalid")
                if low <= port <= high:
                    return True
    return False


def _public_owner_gate_rule(rule: Mapping[str, Any], *, owner_sa: str) -> bool:
    if (
        rule.get("direction", "INGRESS") != "INGRESS"
        or rule.get("disabled", False) is True
    ):
        return False
    target_sas = rule.get("targetServiceAccounts", [])
    target_tags = rule.get("targetTags", [])
    if not isinstance(target_sas, list) or not isinstance(target_tags, list):
        _error("owner_gate_cloud_observation_firewall_invalid")
    targeted = (
        (not target_sas and not target_tags)
        or owner_sa in target_sas
        or foundation.OWNER_GATE_NETWORK_TAG in target_tags
        or foundation.IAP_NETWORK_TAG in target_tags
    )
    if not targeted or not _allowed_tcp_port(rule, foundation.WEB_LISTEN_PORT):
        return False
    source_sas = rule.get("sourceServiceAccounts", [])
    source_tags = rule.get("sourceTags", [])
    if not isinstance(source_sas, list) or not isinstance(source_tags, list):
        _error("owner_gate_cloud_observation_firewall_invalid")
    if (
        source_sas == [foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT]
        and not source_tags
        and not rule.get("sourceRanges", [])
    ):
        return False
    ranges = rule.get("sourceRanges", ["0.0.0.0/0"])
    if not isinstance(ranges, list) or any(
        not isinstance(item, str) for item in ranges
    ):
        _error("owner_gate_cloud_observation_firewall_invalid")
    try:
        networks = [ipaddress.ip_network(item, strict=True) for item in ranges]
    except ValueError:
        _error("owner_gate_cloud_observation_firewall_invalid")
    return any(not network.is_private for network in networks)


def _exact_selectors(
    rule: Mapping[str, Any],
    *,
    source_ranges: Sequence[str] = (),
    source_service_accounts: Sequence[str] = (),
    target_tags: Sequence[str] = (),
    target_service_accounts: Sequence[str] = (),
) -> bool:
    expected = {
        "sourceRanges": list(source_ranges),
        "sourceServiceAccounts": list(source_service_accounts),
        "sourceTags": [],
        "targetTags": list(target_tags),
        "targetServiceAccounts": list(target_service_accounts),
        "destinationRanges": [],
    }
    return all(rule.get(name, []) == value for name, value in expected.items())


def _firewall_projection(
    *,
    iap: Mapping[str, Any],
    private_web: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    effective: Mapping[str, Any],
    expected_private_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    owner_sa = f"{foundation.SERVICE_ACCOUNT_NAME}@{foundation.PROJECT}.iam.gserviceaccount.com"
    expected_network = (
        f"projects/{foundation.PROJECT}/global/networks/{foundation.NETWORK_NAME}"
    )
    _resource(
        iap.get("network"),
        expected=expected_network,
        code="owner_gate_cloud_observation_firewall_invalid",
    )
    _resource(
        private_web.get("network"),
        expected=expected_network,
        code="owner_gate_cloud_observation_firewall_invalid",
    )
    _resource(
        private_web.get("selfLink"),
        expected=(
            f"projects/{foundation.PROJECT}/global/firewalls/"
            "muncho-owner-gate-web-from-production"
        ),
        code="owner_gate_cloud_observation_firewall_invalid",
    )
    expected_iap = {
        "name": "allow-iap-ssh",
        "source_ranges": [foundation.IAP_SOURCE_RANGE],
        "target_tags": [foundation.IAP_NETWORK_TAG],
        "tcp_ports": [22],
        "enabled": True,
    }
    if (
        iap.get("name") != "allow-iap-ssh"
        or iap.get("direction") != "INGRESS"
        or iap.get("disabled", False) is not False
        or iap.get("sourceRanges") != [foundation.IAP_SOURCE_RANGE]
        or iap.get("targetTags") != [foundation.IAP_NETWORK_TAG]
        or not _exact_selectors(
            iap,
            source_ranges=[foundation.IAP_SOURCE_RANGE],
            target_tags=[foundation.IAP_NETWORK_TAG],
        )
        or iap.get("allowed") != [{"IPProtocol": "tcp", "ports": ["22"]}]
    ):
        _error("owner_gate_cloud_observation_firewall_invalid")
    if (
        private_web.get("name") != "muncho-owner-gate-web-from-production"
        or private_web.get("direction") != "INGRESS"
        or private_web.get("disabled", False) is not False
        or str(private_web.get("id"))
        != str(expected_private_identity.get("numeric_id"))
        or private_web.get("priority") != 700
        or private_web.get("sourceServiceAccounts")
        != [foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT]
        or private_web.get("targetServiceAccounts") != [owner_sa]
        or not _exact_selectors(
            private_web,
            source_service_accounts=[foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT],
            target_service_accounts=[owner_sa],
        )
        or private_web.get("allowed")
        != [{"IPProtocol": "tcp", "ports": [str(foundation.WEB_LISTEN_PORT)]}]
        or not isinstance(private_web.get("logConfig"), Mapping)
        or private_web["logConfig"].get("enable") is not True
    ):
        _error("owner_gate_cloud_observation_firewall_invalid")
    public = sorted(
        str(rule.get("name"))
        for rule in inventory
        if _public_owner_gate_rule(rule, owner_sa=owner_sa)
    )
    if public:
        _error("owner_gate_cloud_observation_firewall_invalid")
    effective_rules = effective.get("firewalls")
    if (
        not isinstance(effective_rules, list)
        or any(not isinstance(rule, Mapping) for rule in effective_rules)
        or len(effective_rules) > MAX_ITEMS
        or len([
            rule for rule in effective_rules if rule.get("name") == "allow-iap-ssh"
        ])
        != 1
        or len([
            rule
            for rule in effective_rules
            if rule.get("name") == "muncho-owner-gate-web-from-production"
        ])
        != 1
        or any(
            _public_owner_gate_rule(rule, owner_sa=owner_sa) for rule in effective_rules
        )
    ):
        _error("owner_gate_cloud_observation_firewall_invalid")
    effective_iap = next(
        rule for rule in effective_rules if rule.get("name") == "allow-iap-ssh"
    )
    effective_private = next(
        rule
        for rule in effective_rules
        if rule.get("name") == "muncho-owner-gate-web-from-production"
    )
    if (
        not _exact_selectors(
            effective_iap,
            source_ranges=[foundation.IAP_SOURCE_RANGE],
            target_tags=[foundation.IAP_NETWORK_TAG],
        )
        or not _exact_selectors(
            effective_private,
            source_service_accounts=[foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT],
            target_service_accounts=[owner_sa],
        )
        or not _allowed_tcp_port(effective_iap, 22)
        or not _allowed_tcp_port(effective_private, foundation.WEB_LISTEN_PORT)
    ):
        _error("owner_gate_cloud_observation_firewall_invalid")
    return {
        "iap": expected_iap,
        "private_web": {
            "name": "muncho-owner-gate-web-from-production",
            "source_service_accounts": [foundation.PRODUCTION_SOURCE_SERVICE_ACCOUNT],
            "target_service_accounts": [owner_sa],
            "tcp_ports": [foundation.WEB_LISTEN_PORT],
            "enabled": True,
            "logging": True,
        },
        "public_owner_gate_rules": [],
        "effective_inventory_sha256": foundation.sha256_json(
            _canonical_inventory({
                "configured": list(inventory),
                "effective": list(effective_rules),
            })
        ),
        "effective_firewall_probe_verified": True,
    }


def _target_projection(
    instance: Mapping[str, Any],
    disk: Mapping[str, Any],
) -> Mapping[str, Any]:
    attachments = instance.get("disks")
    expected_disk = (
        f"projects/{foundation.PROJECT}/zones/{foundation.ZONE}/disks/"
        f"{foundation.TARGET_DISK}"
    )
    expected_instance = (
        f"projects/{foundation.PROJECT}/zones/{foundation.ZONE}/instances/"
        f"{foundation.TARGET_INSTANCE}"
    )
    _resource(
        instance.get("selfLink"),
        expected=expected_instance,
        code="owner_gate_cloud_observation_targets_invalid",
    )
    _resource(
        disk.get("selfLink"),
        expected=expected_disk,
        code="owner_gate_cloud_observation_targets_invalid",
    )
    matches = []
    if isinstance(attachments, list):
        for attachment in attachments:
            if not isinstance(attachment, Mapping):
                _error("owner_gate_cloud_observation_targets_invalid")
            source = str(attachment.get("source", ""))
            normalized = source
            for prefix in _RESOURCE_PREFIXES:
                normalized = normalized.removeprefix(prefix)
            if normalized == expected_disk:
                matches.append(attachment)
    if (
        instance.get("name") != foundation.TARGET_INSTANCE
        or str(instance.get("id")) != foundation.TARGET_INSTANCE_ID
        or disk.get("name") != foundation.TARGET_DISK
        or str(disk.get("id")) != foundation.TARGET_DISK_ID
        or len(matches) != 1
        or matches[0].get("deviceName") != foundation.TARGET_BOOT_DEVICE
        or matches[0].get("boot") is not True
    ):
        _error("owner_gate_cloud_observation_targets_invalid")
    return {
        "instance_name": foundation.TARGET_INSTANCE,
        "instance_numeric_id": foundation.TARGET_INSTANCE_ID,
        "disk_name": foundation.TARGET_DISK,
        "disk_numeric_id": foundation.TARGET_DISK_ID,
        "boot_device_name": foundation.TARGET_BOOT_DEVICE,
    }


_VERIFIED_PROBE_MARKER = object()


@dataclass(frozen=True, init=False)
class _VerifiedAttachedSaProbe:
    phase: str
    release_revision: str
    source_tree_oid: str
    package_sha256: str
    permission_probe: Mapping[str, Any]
    host_observation_report_sha256: str
    host_observation_binding_sha256: str
    production_ingress_observation_sha256: str
    attached_sa_permission_probe_report_sha256: str
    terminal_receipt_sha256: str
    cloud_signer_provisioning_receipt_sha256: str
    cloud_signer_readiness_sha256: str
    host_signer_provisioning_receipt_sha256: str
    host_signer_readiness_sha256: str
    _marker: object

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "_VerifiedAttachedSaProbe":
        _error("owner_gate_cloud_observation_probe_factory_required")

    @classmethod
    def _create(
        cls,
        *,
        phase: str,
        release_revision: str,
        source_tree_oid: str,
        package_sha256: str,
        permission_probe: Mapping[str, Any],
        host_observation_report_sha256: str,
        host_observation_binding_sha256: str,
        production_ingress_observation_sha256: str,
        attached_sa_permission_probe_report_sha256: str,
        terminal_receipt_sha256: str,
        cloud_signer_provisioning_receipt_sha256: str,
        cloud_signer_readiness_sha256: str,
        host_signer_provisioning_receipt_sha256: str,
        host_signer_readiness_sha256: str,
    ) -> "_VerifiedAttachedSaProbe":
        value = object.__new__(cls)
        for name, item in locals().copy().items():
            if name not in {"cls", "value"}:
                object.__setattr__(value, name, item)
        object.__setattr__(value, "_marker", _VERIFIED_PROBE_MARKER)
        return value


def _verified_probe_from_host_handoff(
    handoff: Any,
    *,
    plan: foundation.OwnerGateFoundationPlan,
    phase: str,
    identities: _FoundationIdentities,
    package: Mapping[str, Any],
    host_public_key: Ed25519PublicKey,
    production_ingress_observation_sha256: str,
) -> _VerifiedAttachedSaProbe:
    try:
        from scripts.canary import owner_gate_stage0_iap as stage0_iap

        handoff_type = stage0_iap.OwnerGateHostObservationHandoff
        marker = stage0_iap._HOST_OBSERVATION_HANDOFF_MARKER
    except (AttributeError, ImportError) as exc:
        _error("owner_gate_cloud_observation_host_composite_unavailable", exc)
    if (
        type(handoff) is not handoff_type
        or getattr(handoff, "_marker", None) is not marker
    ):
        _error("owner_gate_cloud_observation_host_handoff_invalid")
    terminal = getattr(handoff, "terminal_receipt", None)
    host_observation = getattr(handoff, "host_observation", None)
    if not isinstance(terminal, Mapping) or not isinstance(host_observation, Mapping):
        _error("owner_gate_cloud_observation_host_handoff_invalid")
    expected_chain = [
        str(item["resource_name"])
        for item in identities.ancestry_evidence.ordered_chain[1:]
    ]
    terminal_unsigned = {
        key: item for key, item in terminal.items() if key != "terminal_receipt_sha256"
    }
    terminal_sha256 = terminal.get("terminal_receipt_sha256")
    if (
        set(terminal)
        != {
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
        }
        or terminal.get("schema") != stage0_iap.INERT_CLOUD_BUNDLE_TERMINAL_SCHEMA
        or terminal.get("release_sha") != package.get("release_revision")
        or terminal.get("source_tree_oid") != package.get("source_tree_oid")
        or terminal.get("package_sha256") != package.get("package_sha256")
        or terminal.get("pre_foundation_authority_sha256")
        != identities.pre_foundation_authority_sha256
        or terminal.get("foundation_apply_receipt_sha256")
        != identities.foundation_apply_receipt_sha256
        or terminal.get("project_ancestry_evidence_sha256")
        != identities.ancestry_evidence.signed_evidence_sha256
        or terminal.get("project_ancestry_chain_sha256")
        != identities.ancestry_evidence.value["stable_chain_sha256"]
        or terminal.get("resource_ancestor_chain") != expected_chain
        or terminal.get("operation_order")
        != [
            "transport_exact_stage0_and_bundle",
            "cloud-verify",
            "cloud-preflight",
            "cloud-install",
        ]
        or terminal.get("inert_cloud_bundle_installed") is not True
        or terminal.get("host_filesystem_materialization_performed") is not True
        or terminal.get("current_release_selected") is not False
        or terminal.get("systemd_units_enabled") != []
        or terminal.get("service_activation_performed") is not False
        or terminal.get("activation_performed") is not False
        or terminal.get("activation_seal_created") is not False
        or terminal.get("iam_binding_created") is not False
        or terminal.get("caddy_cutover_performed") is not False
        or terminal.get("cloud_mutation_performed") is not False
        or terminal.get("cloud_control_plane_mutation_performed") is not False
        or any(
            _SHA256.fullmatch(str(terminal.get(name, ""))) is None
            for name in (
                "transport_receipt_sha256",
                "cloud_verify_receipt_sha256",
                "cloud_preflight_receipt_sha256",
                "cloud_install_receipt_sha256",
                "cloud_install_receipt_file_sha256",
            )
        )
        or _SHA256.fullmatch(str(terminal_sha256 or "")) is None
        or terminal_sha256 != foundation.sha256_json(terminal_unsigned)
    ):
        _error("owner_gate_cloud_observation_host_handoff_invalid")
    try:
        preflight._validate_host(
            host_observation,
            spec=plan.spec,
            plan_sha256=plan.sha256,
            public_key=host_public_key,
            expected_public_key_id=str(plan.spec.host_collector_public_key_id),
            mutation_binding_present=phase == "post_iam",
        )
    except preflight.OwnerGatePreflightError as exc:
        _error("owner_gate_cloud_observation_host_handoff_invalid", exc)
    release = host_observation.get("release")
    expected_probe = preflight.expected_effective_permission_probe(phase == "post_iam")
    if (
        not isinstance(release, Mapping)
        or host_observation.get("effective_permission_probe") != expected_probe
        or release.get("revision") != package.get("release_revision")
        or release.get("source_tree_oid") != package.get("source_tree_oid")
        or release.get("package_sha256") != package.get("package_sha256")
        or release.get("package_sha256") != terminal.get("package_sha256")
        or release.get("package_inventory_sha256") != plan.spec.package_inventory_sha256
        or release.get("pre_foundation_authority_sha256")
        != identities.pre_foundation_authority_sha256
        or release.get("foundation_apply_receipt_sha256")
        != identities.foundation_apply_receipt_sha256
        or release.get("project_ancestry_evidence_sha256")
        != identities.ancestry_evidence.signed_evidence_sha256
        or release.get("project_ancestry_chain_sha256")
        != identities.ancestry_evidence.value["stable_chain_sha256"]
        or release.get("resource_ancestor_chain") != expected_chain
        or _SHA256.fullmatch(
            str(host_observation.get("observation_binding_sha256", ""))
        )
        is None
        or host_observation.get("production_ingress_observation_sha256")
        != production_ingress_observation_sha256
        or _SHA256.fullmatch(production_ingress_observation_sha256) is None
        or any(
            _SHA256.fullmatch(str(release.get(name, ""))) is None
            for name in (
                "attached_sa_permission_probe_report_sha256",
                "cloud_signer_provisioning_receipt_sha256",
                "cloud_signer_readiness_sha256",
                "host_signer_provisioning_receipt_sha256",
                "host_signer_readiness_sha256",
            )
        )
    ):
        _error("owner_gate_cloud_observation_host_handoff_invalid")
    return _VerifiedAttachedSaProbe._create(
        phase=phase,
        release_revision=str(package["release_revision"]),
        source_tree_oid=str(package["source_tree_oid"]),
        package_sha256=str(package["package_sha256"]),
        permission_probe=expected_probe,
        host_observation_report_sha256=str(host_observation["report_sha256"]),
        host_observation_binding_sha256=str(
            host_observation["observation_binding_sha256"]
        ),
        production_ingress_observation_sha256=(
            production_ingress_observation_sha256
        ),
        attached_sa_permission_probe_report_sha256=str(
            release["attached_sa_permission_probe_report_sha256"]
        ),
        terminal_receipt_sha256=str(terminal_sha256),
        cloud_signer_provisioning_receipt_sha256=str(
            release["cloud_signer_provisioning_receipt_sha256"]
        ),
        cloud_signer_readiness_sha256=str(release["cloud_signer_readiness_sha256"]),
        host_signer_provisioning_receipt_sha256=str(
            release["host_signer_provisioning_receipt_sha256"]
        ),
        host_signer_readiness_sha256=str(release["host_signer_readiness_sha256"]),
    )


def _build_unsigned(
    *,
    plan: foundation.OwnerGateFoundationPlan,
    ancestry_evidence: project_ancestry.ProjectAncestryEvidence,
    phase: str,
    raw: Mapping[str, Any],
    collected_at_unix: int,
    package_sha256: str,
    foundation_identities: _FoundationIdentities,
    verified_probe: _VerifiedAttachedSaProbe,
) -> Mapping[str, Any]:
    chain = _validated_context(
        plan=plan,
        ancestry_evidence=ancestry_evidence,
        phase=phase,
    )
    if type(collected_at_unix) is not int or collected_at_unix <= 0:
        _error("owner_gate_cloud_observation_time_invalid")
    if (
        type(verified_probe) is not _VerifiedAttachedSaProbe
        or verified_probe._marker is not _VERIFIED_PROBE_MARKER
        or verified_probe.phase != phase
        or verified_probe.release_revision != plan.spec.release_revision
        or _REVISION.fullmatch(verified_probe.release_revision) is None
        or _REVISION.fullmatch(verified_probe.source_tree_oid) is None
        or _SHA256.fullmatch(package_sha256) is None
        or verified_probe.package_sha256 != package_sha256
        or _SHA256.fullmatch(
            verified_probe.production_ingress_observation_sha256
        )
        is None
        or _NUMERIC_ID.fullmatch(
            str(foundation_identities.owner_gate_vm.get("numeric_id", ""))
        )
        is None
        or _NUMERIC_ID.fullmatch(
            str(foundation_identities.service_account.get("unique_id", ""))
        )
        is None
    ):
        _error("owner_gate_cloud_observation_foundation_identity_invalid")
    locations_url = _url(
        f"/v1/projects/{foundation.PROJECT}/locations?pageSize=100",
        host="vpcaccess.googleapis.com",
    )
    if not isinstance(raw, Mapping):
        _error("owner_gate_cloud_observation_facts_invalid")
    connector_regions = _FixedCloudFactsReader._location_regions(
        raw.get(f"GET {locations_url}")
    )
    requests = request_inventory(
        plan=plan,
        ancestry_evidence=ancestry_evidence,
        phase=phase,
        _connector_regions=connector_regions,
    )
    facts = _Facts(raw=raw, requests=requests)
    _validate_live_ancestry(facts, ancestry_evidence=ancestry_evidence)
    project = foundation.PROJECT
    compute = "/compute/v1"
    project_fact = facts.get("GET", _url(f"{compute}/projects/{project}"))
    if (
        project_fact.get("name") != project
        or str(project_fact.get("id")) != ancestry_evidence.project_number
    ):
        _error("owner_gate_cloud_observation_project_invalid")
    project_metadata = _instance_metadata(project_fact.get("commonInstanceMetadata"))
    source = _source_projection(
        facts.get(
            "GET",
            _url(
                f"{compute}/projects/{project}/zones/{foundation.ZONE}/instances/{foundation.PRODUCTION_SOURCE_VM}"
            ),
        )
    )
    aggregate = _aggregated_subnets(
        facts.get(
            "GET",
            _url(f"{compute}/projects/{project}/aggregated/subnetworks"),
        )
    )
    routes = _list_response(
        facts.get("GET", _url(f"{compute}/projects/{project}/global/routes")),
        code="owner_gate_cloud_observation_subnet_invalid",
    )
    addresses = _list_response(
        facts.get("GET", _url(f"{compute}/projects/{project}/global/addresses")),
        code="owner_gate_cloud_observation_network_invalid",
    )
    connectors = {
        region: _named_list_response(
            facts.get(
                "GET",
                _url(
                    f"/v1/projects/{project}/locations/{region}/connectors?pageSize=100",
                    host="vpcaccess.googleapis.com",
                ),
            ),
            field="connectors",
            code="owner_gate_cloud_observation_network_invalid",
        )
        for region in connector_regions
    }
    network = facts.get(
        "GET",
        _url(f"{compute}/projects/{project}/global/networks/{foundation.NETWORK_NAME}"),
    )
    _resource(
        network.get("selfLink"),
        expected=f"projects/{project}/global/networks/{foundation.NETWORK_NAME}",
        code="owner_gate_cloud_observation_network_invalid",
    )
    if network.get("name") != foundation.NETWORK_NAME:
        _error("owner_gate_cloud_observation_network_invalid")
    peerings = network.get("peerings", [])
    if (
        not isinstance(peerings, list)
        or len(peerings) > MAX_ITEMS
        or any(not isinstance(item, Mapping) for item in peerings)
    ):
        _error("owner_gate_cloud_observation_network_invalid")
    subnet = _subnet_projection(
        facts.get(
            "GET",
            _url(
                f"{compute}/projects/{project}/regions/{foundation.REGION}/subnetworks/"
                f"{foundation.OWNER_GATE_SUBNET_NAME}"
            ),
        ),
        aggregate=aggregate,
        routes=routes,
        addresses=addresses,
        connectors=connectors,
        peerings=[dict(item) for item in peerings],
        expected_numeric_id=str(foundation_identities.subnet["numeric_id"]),
    )
    instance = _owner_instance_projection(
        facts.get(
            "GET",
            _url(
                f"{compute}/projects/{project}/zones/{foundation.ZONE}/instances/{foundation.VM_NAME}"
            ),
        ),
        project_metadata=project_metadata,
        expected_identity=foundation_identities.owner_gate_vm,
        boot_disk=facts.get(
            "GET",
            _url(
                f"{compute}/projects/{project}/zones/{foundation.ZONE}/disks/"
                f"{foundation.VM_NAME}"
            ),
        ),
    )
    service_account, iam = _iam_projection(
        facts=facts,
        plan=plan,
        chain=chain,
        phase=phase,
        expected_service_account_unique_id=str(
            foundation_identities.service_account["unique_id"]
        ),
        permission_probe=verified_probe.permission_probe,
    )
    firewalls = _firewall_projection(
        iap=facts.get(
            "GET", _url(f"{compute}/projects/{project}/global/firewalls/allow-iap-ssh")
        ),
        private_web=facts.get(
            "GET",
            _url(
                f"{compute}/projects/{project}/global/firewalls/muncho-owner-gate-web-from-production"
            ),
        ),
        inventory=_list_response(
            facts.get("GET", _url(f"{compute}/projects/{project}/global/firewalls")),
            code="owner_gate_cloud_observation_firewall_invalid",
        ),
        effective=facts.get(
            "GET",
            _url(
                f"{compute}/projects/{project}/zones/{foundation.ZONE}/instances/"
                f"{foundation.VM_NAME}/getEffectiveFirewalls"
            ),
        ),
        expected_private_identity=foundation_identities.private_web_firewall,
    )
    targets = _target_projection(
        facts.get(
            "GET",
            _url(
                f"{compute}/projects/{project}/zones/{foundation.ZONE}/instances/{foundation.TARGET_INSTANCE}"
            ),
        ),
        facts.get(
            "GET",
            _url(
                f"{compute}/projects/{project}/zones/{foundation.ZONE}/disks/{foundation.TARGET_DISK}"
            ),
        ),
    )
    return {
        "schema": preflight.CLOUD_OBSERVATION_SCHEMA,
        "collected_at_unix": collected_at_unix,
        "plan_sha256": plan.sha256,
        "project": project,
        "zone": foundation.ZONE,
        "source": source,
        "subnet": subnet,
        "instance": instance,
        "service_account": service_account,
        "iam": iam,
        "firewalls": firewalls,
        "targets": targets,
        "release_binding": {
            "phase": phase,
            "release_revision": verified_probe.release_revision,
            "source_tree_oid": verified_probe.source_tree_oid,
            "package_sha256": package_sha256,
            "package_inventory_sha256": plan.spec.package_inventory_sha256,
            "pre_foundation_authority_sha256": (
                foundation_identities.pre_foundation_authority_sha256
            ),
            "foundation_apply_receipt_sha256": (
                foundation_identities.foundation_apply_receipt_sha256
            ),
            "project_ancestry_evidence_sha256": (
                ancestry_evidence.signed_evidence_sha256
            ),
            "project_ancestry_chain_sha256": ancestry_evidence.value[
                "stable_chain_sha256"
            ],
            "resource_ancestor_chain": list(chain[1:]),
            "terminal_receipt_sha256": (verified_probe.terminal_receipt_sha256),
            "host_observation_report_sha256": (
                verified_probe.host_observation_report_sha256
            ),
            "host_observation_binding_sha256": (
                verified_probe.host_observation_binding_sha256
            ),
            "production_ingress_observation_sha256": (
                verified_probe.production_ingress_observation_sha256
            ),
            "attached_sa_permission_probe_report_sha256": (
                verified_probe.attached_sa_permission_probe_report_sha256
            ),
            "cloud_signer_provisioning_receipt_sha256": (
                verified_probe.cloud_signer_provisioning_receipt_sha256
            ),
            "cloud_signer_readiness_sha256": (
                verified_probe.cloud_signer_readiness_sha256
            ),
            "host_signer_provisioning_receipt_sha256": (
                verified_probe.host_signer_provisioning_receipt_sha256
            ),
            "host_signer_readiness_sha256": (
                verified_probe.host_signer_readiness_sha256
            ),
            "effective_permission_probe_sha256": foundation.sha256_json(
                verified_probe.permission_probe
            ),
        },
        "collector": "owner_read_only_rest_remote_executor_attested",
        "credential_values_read": False,
    }


@dataclass(frozen=True)
class _PublicSignerSnapshot:
    public_raw: bytes
    identity: tuple[int, int, int, int, int, int, int]


def _public_signer_snapshot(
    release_revision: str,
    *,
    role: str,
) -> _PublicSignerSnapshot:
    if role not in {"network", "cloud", "host"}:
        _error("owner_gate_cloud_observation_signer_invalid")
    try:
        signer_author._require_authority_directories(release_revision, create=False)
        path = signer_author._public_path(release_revision, role)
        before = path.lstat()
        public_raw = release_author._read_exact_regular(
            path,
            size=32,
            modes=frozenset({0o400, 0o440, 0o444}),
            code="owner_gate_cloud_observation_signer_invalid",
        )
        after = path.lstat()
    except (
        OSError,
        release_author.OwnerGateTrustAuthorError,
        signer_author.TrustedSignerAuthorError,
    ) as exc:
        _error("owner_gate_cloud_observation_signer_invalid", exc)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
        stat.S_IMODE(before.st_mode),
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
        stat.S_IMODE(after.st_mode),
    ):
        _error("owner_gate_cloud_observation_signer_invalid")
    return _PublicSignerSnapshot(public_raw, identity)


def _unsigned_from_raw(
    *,
    plan: foundation.OwnerGateFoundationPlan,
    ancestry_evidence: project_ancestry.ProjectAncestryEvidence,
    phase: str,
    raw: Mapping[str, Any],
    collected_at_unix: int,
    package_sha256: str,
    foundation_identities: _FoundationIdentities,
    verified_probe: _VerifiedAttachedSaProbe,
) -> Mapping[str, Any]:
    return _build_unsigned(
        plan=plan,
        ancestry_evidence=ancestry_evidence,
        phase=phase,
        raw=raw,
        collected_at_unix=collected_at_unix,
        package_sha256=package_sha256,
        foundation_identities=foundation_identities,
        verified_probe=verified_probe,
    )


@dataclass(frozen=True)
class _FoundationIdentities:
    ancestry_evidence: project_ancestry.ProjectAncestryEvidence
    owner_gate_vm: Mapping[str, Any]
    service_account: Mapping[str, Any]
    subnet: Mapping[str, Any]
    private_web_firewall: Mapping[str, Any]
    pre_foundation_plan: foundation.OwnerGateFoundationPlan
    network_evidence: foundation.ProductionNetworkEvidence
    network_collector_public_key: Ed25519PublicKey
    inert_plan_sha256: str
    foundation_source_revision: str
    foundation_source_tree_oid: str
    pre_foundation_authority_sha256: str
    foundation_apply_receipt_sha256: str


def _validated_foundation_identity(
    foundation_apply_chain: Any,
) -> _FoundationIdentities:
    from scripts.canary import owner_gate_foundation_apply as foundation_apply
    from scripts.canary import owner_gate_pre_foundation as pre_foundation

    if (
        type(foundation_apply_chain)
        is not foundation_apply.ValidatedFoundationApplyChain
        or getattr(foundation_apply_chain, "_marker", None)
        is not foundation_apply._CHAIN_MARKER
    ):
        _error("owner_gate_cloud_observation_foundation_identity_invalid")
    try:
        ancestry_evidence = foundation_apply_chain.foundation_a.ancestry_evidence
        vm_identity = dict(foundation_apply_chain.owner_gate_vm_identity)
        service_account_identity = dict(foundation_apply_chain.service_account_identity)
        subnet_identity = dict(foundation_apply_chain.subnet_identity)
        firewall_identity = dict(
            foundation_apply_chain.resource_identity(
                "allow_private_web_upstream_from_current_caddy_host"
            )
        )
        pre_plan = foundation_apply_chain.foundation_a.plan
        network_evidence = foundation_apply_chain.foundation_a.network_evidence
        network_collector_public_key = (
            foundation_apply_chain.foundation_a.network_collector_public_key
        )
        inert_plan_sha256 = pre_foundation.inert_plan_sha256(pre_plan)
    except (AttributeError, KeyError, TypeError) as exc:
        _error("owner_gate_cloud_observation_foundation_identity_invalid", exc)
    if (
        type(pre_plan) is not foundation.OwnerGateFoundationPlan
        or pre_plan.spec.final_release_bound
        or not pre_plan.spec.pre_foundation_bound
        or inert_plan_sha256
        != str(foundation_apply_chain.foundation_a.authority.get("inert_plan_sha256"))
        or _NUMERIC_ID.fullmatch(str(vm_identity.get("numeric_id", ""))) is None
        or _NUMERIC_ID.fullmatch(str(service_account_identity.get("unique_id", "")))
        is None
        or _NUMERIC_ID.fullmatch(str(subnet_identity.get("numeric_id", ""))) is None
        or _NUMERIC_ID.fullmatch(str(firewall_identity.get("numeric_id", ""))) is None
        or _NUMERIC_ID.fullmatch(str(vm_identity.get("boot_disk_numeric_id", "")))
        is None
        or vm_identity.get("name") != foundation.VM_NAME
        or service_account_identity.get("email")
        != (
            f"{foundation.SERVICE_ACCOUNT_NAME}@{foundation.PROJECT}."
            "iam.gserviceaccount.com"
        )
        or subnet_identity.get("name") != foundation.OWNER_GATE_SUBNET_NAME
        or str(vm_identity.get("subnetwork_numeric_id"))
        != str(subnet_identity.get("numeric_id"))
        or firewall_identity.get("name") != "muncho-owner-gate-web-from-production"
        or foundation_apply_chain.foundation_source_revision
        != pre_plan.spec.release_revision
        or foundation_apply_chain.foundation_source_tree_oid
        != pre_plan.spec.source_tree_oid
    ):
        _error("owner_gate_cloud_observation_foundation_identity_invalid")
    return _FoundationIdentities(
        ancestry_evidence=ancestry_evidence,
        owner_gate_vm=vm_identity,
        service_account=service_account_identity,
        subnet=subnet_identity,
        private_web_firewall=firewall_identity,
        pre_foundation_plan=pre_plan,
        network_evidence=network_evidence,
        network_collector_public_key=network_collector_public_key,
        inert_plan_sha256=inert_plan_sha256,
        foundation_source_revision=foundation_apply_chain.foundation_source_revision,
        foundation_source_tree_oid=foundation_apply_chain.foundation_source_tree_oid,
        pre_foundation_authority_sha256=(
            foundation_apply_chain.pre_foundation_authority_sha256
        ),
        foundation_apply_receipt_sha256=(
            foundation_apply_chain.foundation_apply_receipt_sha256
        ),
    )


def _validated_production_ingress_observation_sha256(
    value: Any,
    *,
    phase: str,
    release_revision: str,
    plan_sha256: str,
    release_public_key: Ed25519PublicKey,
    now_unix: int,
) -> str:
    """Validate the full signed envelope and return only its final digest."""

    try:
        checked = production_ingress.validate_signed_production_ingress_observation(
            value,
            phase=phase,
            release_revision=release_revision,
            plan_sha256=plan_sha256,
            release_public_key=release_public_key,
            now_unix=now_unix,
        )
    except production_ingress.ProductionIngressObservationError as exc:
        _error("owner_gate_cloud_observation_production_ingress_invalid", exc)
    digest = checked.get("envelope_sha256")
    if type(digest) is not str or _SHA256.fullmatch(digest) is None:
        _error("owner_gate_cloud_observation_production_ingress_invalid")
    return digest


def _validate_plan_from_exact_streams(
    *,
    plan: foundation.OwnerGateFoundationPlan,
    identities: _FoundationIdentities,
    final_network_evidence: foundation.ProductionNetworkEvidence,
    final_network_collector_public_key: Ed25519PublicKey,
    stage0_transport: Any,
    kit_stream: Any,
    bundle_stream: Any,
) -> Mapping[str, Any]:
    from scripts.canary import owner_gate_stage0 as cloud_stage0
    from scripts.canary import owner_gate_stage0_iap as stage0_iap

    if (
        type(stage0_transport) is not stage0_iap.OwnerGateStage0IapTransport
        or type(kit_stream) is not stage0_iap.PinnedExactTreeStream
        or type(bundle_stream) is not stage0_iap.PinnedExactTreeStream
    ):
        _error("owner_gate_cloud_observation_host_composite_invalid")
    try:
        binding = stage0_iap.OwnerGateStage0IapTransport._bind_inert_cloud_bundle(
            stage0_transport,
            kit_stream=kit_stream,
            bundle_stream=bundle_stream,
        )
        package_raw = bundle_stream.member("package-manifest.json")
        package = _decode_json(package_raw)
        direct_iam_raw = bundle_stream.member(
            "trust/direct-iam-identity-authority.json"
        )
        direct_iam_authority = direct_iam.decode_canonical(
            direct_iam_raw,
            release_revision=identities.foundation_source_revision,
        )
    except (
        launcher.OwnerLauncherError,
        OwnerGateCloudObservationAuthorError,
        direct_iam.DirectIamIdentityAuthorityError,
    ) as exc:
        _error("owner_gate_cloud_observation_plan_binding_invalid", exc)
    if (
        _canonical(package) != package_raw
        or set(package) != cloud_stage0.MANIFEST_FIELDS
    ):
        _error("owner_gate_cloud_observation_plan_binding_invalid")
    unsigned_package = {
        key: item for key, item in package.items() if key != "package_sha256"
    }
    inventory = {key: package[key] for key in cloud_stage0.INVENTORY_FIELDS}
    collectors = package.get("collector_public_key_ids")
    expected_chain = [
        str(item["resource_name"])
        for item in identities.ancestry_evidence.ordered_chain[1:]
    ]
    final_network_key_id = (
        hashlib.sha256(
            final_network_collector_public_key.public_bytes_raw()
        ).hexdigest()
        if isinstance(final_network_collector_public_key, Ed25519PublicKey)
        else None
    )
    if (
        package.get("schema") != cloud_stage0.PACKAGE_SCHEMA
        or package.get("package_sha256") != foundation.sha256_json(unsigned_package)
        or package.get("package_inventory_sha256") != foundation.sha256_json(inventory)
        or package.get("release_revision") != plan.spec.release_revision
        or package.get("release_revision")
        == identities.foundation_source_revision
        or package.get("foundation_source_revision")
        != identities.foundation_source_revision
        or package.get("foundation_source_tree_oid")
        != identities.foundation_source_tree_oid
        or package.get("source_tree_oid") != binding.source_tree_oid
        or package.get("package_sha256") != binding.package_sha256
        or hashlib.sha256(direct_iam_raw).hexdigest()
        != package.get("direct_iam_identity_authority_sha256")
        or direct_iam_authority.get("release_revision")
        != identities.foundation_source_revision
        or direct_iam_authority.get("pre_foundation_authority_sha256")
        != package.get("pre_foundation_authority_sha256")
        or direct_iam_authority.get("foundation_apply_receipt_sha256")
        != package.get("foundation_apply_receipt_sha256")
        or direct_iam_authority.get("resource_ancestor_chain")
        != package.get("resource_ancestor_chain")
        or package.get("pre_foundation_authority_sha256")
        != identities.pre_foundation_authority_sha256
        or package.get("foundation_apply_receipt_sha256")
        != identities.foundation_apply_receipt_sha256
        or package.get("project_ancestry_evidence_sha256")
        != identities.ancestry_evidence.signed_evidence_sha256
        or package.get("project_ancestry_chain_sha256")
        != identities.ancestry_evidence.value["stable_chain_sha256"]
        or package.get("resource_ancestor_chain") != expected_chain
        or not isinstance(collectors, Mapping)
        or set(collectors) != {"network", "cloud", "host"}
        or any(_SHA256.fullmatch(str(item)) is None for item in collectors.values())
        or len(set(collectors.values())) != 3
        or identities.pre_foundation_plan.spec.network_collector_public_key_id
        in set(collectors.values())
        or type(final_network_evidence)
        is not foundation.ProductionNetworkEvidence
        or final_network_key_id != collectors.get("network")
        or final_network_evidence.collector_public_key_id
        != collectors.get("network")
        or package.get("interpreter_sha256")
        != identities.pre_foundation_plan.spec.interpreter_sha256
    ):
        _error("owner_gate_cloud_observation_plan_binding_invalid")
    expected_spec = replace(
        identities.pre_foundation_plan.spec,
        release_revision=str(package["release_revision"]),
        network_collector_public_key_id=str(collectors["network"]),
        source_tree_oid=None,
        boot_image_numeric_id=None,
        package_inventory_sha256=str(package["package_inventory_sha256"]),
        cloud_collector_public_key_id=str(collectors["cloud"]),
        host_collector_public_key_id=str(collectors["host"]),
    )
    try:
        expected_plan = foundation.build_plan(
            spec=expected_spec,
            network_evidence=final_network_evidence,
            network_collector_public_key=final_network_collector_public_key,
            now_unix=final_network_evidence.collected_at_unix,
        )
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_cloud_observation_plan_binding_invalid", exc)
    if type(plan) is not foundation.OwnerGateFoundationPlan or _canonical(
        plan.report()
    ) != _canonical(expected_plan.report()):
        _error("owner_gate_cloud_observation_plan_binding_invalid")
    return dict(package)


_BOUND_OBSERVATION_PAIR_MARKER = object()


@dataclass(frozen=True, init=False)
class BoundObservationPair:
    """Opaque, single-use canonical CLOUD/HOST pair from one composite."""

    _cloud_raw: bytes
    _host_raw: bytes
    _plan_sha256: str
    _phase: str
    _cloud_report_sha256: str
    _host_report_sha256: str
    _host_binding_sha256: str
    _consumed: bool
    _marker: object

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "BoundObservationPair":
        _error("owner_gate_bound_observation_pair_factory_required")

    @classmethod
    def _create(
        cls,
        *,
        cloud_observation: Mapping[str, Any],
        host_observation: Mapping[str, Any],
        plan_sha256: str,
        phase: str,
    ) -> "BoundObservationPair":
        binding = cloud_observation.get("release_binding")
        cloud_report_sha256 = cloud_observation.get("report_sha256")
        host_report_sha256 = host_observation.get("report_sha256")
        host_binding_sha256 = host_observation.get("observation_binding_sha256")
        if (
            _SHA256.fullmatch(plan_sha256 or "") is None
            or phase not in PHASES
            or cloud_observation.get("plan_sha256") != plan_sha256
            or host_observation.get("plan_sha256") != plan_sha256
            or not isinstance(binding, Mapping)
            or binding.get("phase") != phase
            or host_observation.get("phase") != phase
            or _SHA256.fullmatch(str(cloud_report_sha256 or "")) is None
            or _SHA256.fullmatch(str(host_report_sha256 or "")) is None
            or _SHA256.fullmatch(str(host_binding_sha256 or "")) is None
            or binding.get("host_observation_report_sha256")
            != host_report_sha256
            or binding.get("host_observation_binding_sha256")
            != host_binding_sha256
        ):
            _error("owner_gate_bound_observation_pair_invalid")
        cloud_raw = _canonical(cloud_observation)
        host_raw = _canonical(host_observation)
        value = object.__new__(cls)
        object.__setattr__(value, "_cloud_raw", cloud_raw)
        object.__setattr__(value, "_host_raw", host_raw)
        object.__setattr__(value, "_plan_sha256", plan_sha256)
        object.__setattr__(value, "_phase", phase)
        object.__setattr__(value, "_cloud_report_sha256", cloud_report_sha256)
        object.__setattr__(value, "_host_report_sha256", host_report_sha256)
        object.__setattr__(value, "_host_binding_sha256", host_binding_sha256)
        object.__setattr__(value, "_consumed", False)
        object.__setattr__(value, "_marker", _BOUND_OBSERVATION_PAIR_MARKER)
        return value


def consume_bound_observation_pair(
    pair: BoundObservationPair,
    *,
    plan: foundation.OwnerGateFoundationPlan,
    phase: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Consume one factory-created pair exactly once for immediate preflight."""

    if (
        type(pair) is not BoundObservationPair
        or getattr(pair, "_marker", None) is not _BOUND_OBSERVATION_PAIR_MARKER
        or getattr(pair, "_consumed", True) is not False
        or type(plan) is not foundation.OwnerGateFoundationPlan
        or phase not in PHASES
        or pair._plan_sha256 != plan.sha256
        or pair._phase != phase
    ):
        _error("owner_gate_bound_observation_pair_invalid")
    try:
        cloud_observation = _decode_json(pair._cloud_raw)
        host_observation = _decode_json(pair._host_raw)
        binding = cloud_observation.get("release_binding")
        if (
            _canonical(cloud_observation) != pair._cloud_raw
            or _canonical(host_observation) != pair._host_raw
            or cloud_observation.get("report_sha256")
            != pair._cloud_report_sha256
            or host_observation.get("report_sha256") != pair._host_report_sha256
            or host_observation.get("observation_binding_sha256")
            != pair._host_binding_sha256
            or not isinstance(binding, Mapping)
            or binding.get("host_observation_report_sha256")
            != pair._host_report_sha256
            or binding.get("host_observation_binding_sha256")
            != pair._host_binding_sha256
        ):
            _error("owner_gate_bound_observation_pair_invalid")
    finally:
        object.__setattr__(pair, "_consumed", True)
    return cloud_observation, host_observation


def _collect_and_author_components(
    *,
    plan: foundation.OwnerGateFoundationPlan,
    foundation_apply_chain: Any,
    final_network_evidence: foundation.ProductionNetworkEvidence,
    final_network_collector_public_key: Ed25519PublicKey,
    production_ingress_observation: Mapping[str, Any],
    phase: str,
    collected_at_unix: int | None,
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
    owner_identity: launcher.GcloudOwnerAccessToken,
    stage0_transport: Any,
    kit_stream: Any,
    bundle_stream: Any,
) -> tuple[Mapping[str, Any], Any]:
    """Collect and remotely sign one inert or post-IAM observation."""

    # This entire lineage check is deliberately ahead of every gcloud runtime
    # check, owner credential operation, IAP exchange, and Cloud request.
    identities = _validated_foundation_identity(foundation_apply_chain)
    _validated_context(
        plan=plan,
        ancestry_evidence=identities.ancestry_evidence,
        phase=phase,
    )
    package = _validate_plan_from_exact_streams(
        plan=plan,
        identities=identities,
        final_network_evidence=final_network_evidence,
        final_network_collector_public_key=(
            final_network_collector_public_key
        ),
        stage0_transport=stage0_transport,
        kit_stream=kit_stream,
        bundle_stream=bundle_stream,
    )
    try:
        release_public_key = foundation_apply_chain.foundation_a.release_public_key
    except AttributeError as exc:
        _error("owner_gate_cloud_observation_production_ingress_invalid", exc)
    production_ingress_observation_sha256 = (
        _validated_production_ingress_observation_sha256(
            production_ingress_observation,
            phase=phase,
            release_revision=plan.spec.release_revision,
            plan_sha256=plan.sha256,
            release_public_key=release_public_key,
            now_unix=int(time.time()),
        )
    )
    from scripts.canary import owner_gate_stage0_iap as stage0_iap

    if (
        type(gcloud_executable) is not launcher.TrustedGcloudExecutable
        or type(gcloud_configuration) is not launcher.PinnedGcloudConfiguration
        or type(owner_identity) is not launcher.GcloudOwnerAccessToken
        or type(stage0_transport) is not stage0_iap.OwnerGateStage0IapTransport
        or owner_identity.gcloud_configuration is not gcloud_configuration
        or getattr(owner_identity, "_gcloud_executable", None) is not gcloud_executable
        or getattr(stage0_transport, "_owner_identity", None) is not owner_identity
        or getattr(stage0_transport, "_gcloud_executable", None)
        is not gcloud_executable
        or getattr(stage0_transport, "_gcloud_configuration", None)
        is not gcloud_configuration
        or getattr(stage0_transport, "_release_sha", None) != plan.spec.release_revision
    ):
        _error("owner_gate_cloud_observation_capability_invalid")
    try:
        host_identity_before = stage0_transport._host_identity.snapshot()
        projection = stage0_transport._foundation
    except Exception as exc:
        _error("owner_gate_cloud_observation_capability_invalid", exc)
    expected_chain = tuple(
        str(item["resource_name"])
        for item in identities.ancestry_evidence.ordered_chain[1:]
    )
    if (
        type(host_identity_before) is not launcher.OwnerGateHostIdentitySnapshot
        or host_identity_before.vm_numeric_id
        != str(identities.owner_gate_vm["numeric_id"])
        or getattr(projection, "foundation_source_revision", None)
        != identities.foundation_source_revision
        or getattr(projection, "foundation_source_tree_oid", None)
        != identities.foundation_source_tree_oid
        or getattr(projection, "pre_foundation_authority_sha256", None)
        != identities.pre_foundation_authority_sha256
        or getattr(projection, "foundation_apply_receipt_sha256", None)
        != identities.foundation_apply_receipt_sha256
        or getattr(projection, "project_ancestry_evidence_sha256", None)
        != identities.ancestry_evidence.signed_evidence_sha256
        or getattr(projection, "project_ancestry_chain_sha256", None)
        != identities.ancestry_evidence.value["stable_chain_sha256"]
        or tuple(getattr(projection, "resource_ancestor_chain", ())) != expected_chain
        or getattr(projection, "interpreter_sha256", None)
        != plan.spec.interpreter_sha256
    ):
        _error("owner_gate_cloud_observation_capability_invalid")
    composite_method = stage0_iap.OwnerGateStage0IapTransport.__dict__.get(
        "collect_owner_gate_host_observation"
    )
    signer_method = stage0_iap.OwnerGateStage0IapTransport.__dict__.get(
        "_sign_owner_gate_cloud_observation_on_target"
    )
    if not callable(composite_method) or not callable(signer_method):
        _error("owner_gate_cloud_observation_host_composite_unavailable")
    cloud_public_before = _public_signer_snapshot(
        plan.spec.release_revision,
        role="cloud",
    )
    host_public_before = _public_signer_snapshot(
        plan.spec.release_revision,
        role="host",
    )
    try:
        runtime_before = gcloud_executable.sealed_runtime_identity(
            expected_release_sha=plan.spec.release_revision
        )
        gcloud_configuration.assert_stable()
        if gcloud_configuration.account != OWNER_ACCOUNT:
            _error("owner_gate_cloud_observation_owner_identity_invalid")
        owner_identity.bind_approved_subject(
            hashlib.sha256(OWNER_ACCOUNT.encode("ascii")).hexdigest()
        )
        owner_identity.require_stable()
    except OwnerGateCloudObservationAuthorError:
        raise
    except Exception as exc:
        _error("owner_gate_cloud_observation_capability_invalid", exc)
    token: Any = None
    handoff: Any = None
    handoff_snapshot: bytes | None = None
    try:
        _reject_ambient_network_environment()
        try:
            handoff = composite_method(
                stage0_transport,
                phase=phase,
                plan=plan,
                final_network_evidence=final_network_evidence,
                final_network_collector_public_key=(
                    final_network_collector_public_key
                ),
                production_ingress_observation_sha256=(
                    production_ingress_observation_sha256
                ),
                kit_stream=kit_stream,
                bundle_stream=bundle_stream,
            )
        except launcher.OwnerLauncherError as exc:
            _error("owner_gate_cloud_observation_host_composite_failed", exc)
        verified_probe = _verified_probe_from_host_handoff(
            handoff,
            plan=plan,
            phase=phase,
            identities=identities,
            package=package,
            host_public_key=Ed25519PublicKey.from_public_bytes(
                host_public_before.public_raw
            ),
            production_ingress_observation_sha256=(
                production_ingress_observation_sha256
            ),
        )
        handoff_snapshot = _canonical({
            "terminal_receipt": handoff.terminal_receipt,
            "host_observation": handoff.host_observation,
        })
        observed_at = (
            int(time.time()) if collected_at_unix is None else collected_at_unix
        )
        if type(observed_at) is not int or observed_at <= 0:
            _error("owner_gate_cloud_observation_time_invalid")
        try:
            token = direct_iam_author.acquire_access_token(
                token_provider=owner_identity
            )
        except direct_iam_author.DirectIamIdentityAuthorError as exc:
            _error("owner_gate_cloud_observation_token_unavailable", exc)
        raw = _FixedCloudFactsReader(
            token=token,
            plan=plan,
            ancestry_evidence=identities.ancestry_evidence,
            phase=phase,
        ).collect()
        direct_iam_author.wipe_access_token(token)
        if any(token._raw):
            _error("owner_gate_cloud_observation_token_wipe_failed")
        unsigned = _unsigned_from_raw(
            plan=plan,
            ancestry_evidence=identities.ancestry_evidence,
            phase=phase,
            raw=raw,
            collected_at_unix=observed_at,
            package_sha256=str(package["package_sha256"]),
            foundation_identities=identities,
            verified_probe=verified_probe,
        )
        try:
            result = signer_method(
                stage0_transport,
                phase=phase,
                unsigned_observation=unsigned,
                terminal_binding=handoff,
            )
        except launcher.OwnerLauncherError as exc:
            _error("owner_gate_cloud_observation_remote_signer_failed", exc)
        returned_unsigned = (
            {
                name: item
                for name, item in result.items()
                if name not in {"report_sha256", "attestation"}
            }
            if isinstance(result, Mapping)
            else None
        )
        if (
            not isinstance(result, Mapping)
            or not isinstance(returned_unsigned, Mapping)
            or _canonical(returned_unsigned) != _canonical(unsigned)
        ):
            _error("owner_gate_cloud_observation_remote_signer_invalid")
        try:
            preflight._validate_cloud(
                result,
                plan_sha256=plan.sha256,
                public_key=Ed25519PublicKey.from_public_bytes(
                    cloud_public_before.public_raw
                ),
                expected_public_key_id=str(plan.spec.cloud_collector_public_key_id),
                mutation_binding_present=phase == "post_iam",
            )
        except preflight.OwnerGatePreflightError as exc:
            _error("owner_gate_cloud_observation_remote_signer_invalid", exc)
    finally:
        failures: list[BaseException] = []
        if token is not None:
            try:
                direct_iam_author.wipe_access_token(token)
            except BaseException as exc:
                failures.append(exc)
        runtime_after: Any = None
        cloud_public_after: _PublicSignerSnapshot | None = None
        host_public_after: _PublicSignerSnapshot | None = None
        for check in (
            _reject_ambient_network_environment,
            gcloud_configuration.assert_stable,
            owner_identity.require_stable,
            kit_stream.assert_stable,
            bundle_stream.assert_stable,
        ):
            try:
                check()
            except BaseException as exc:
                failures.append(exc)
        try:
            runtime_after = gcloud_executable.sealed_runtime_identity(
                expected_release_sha=plan.spec.release_revision
            )
        except BaseException as exc:
            failures.append(exc)
        try:
            cloud_public_after = _public_signer_snapshot(
                plan.spec.release_revision,
                role="cloud",
            )
        except BaseException as exc:
            failures.append(exc)
        try:
            host_public_after = _public_signer_snapshot(
                plan.spec.release_revision,
                role="host",
            )
        except BaseException as exc:
            failures.append(exc)
        try:
            host_identity_after = stage0_transport._host_identity.snapshot()
            if host_identity_after != host_identity_before:
                failures.append(
                    OwnerGateCloudObservationAuthorError(
                        "owner_gate_cloud_observation_host_identity_changed"
                    )
                )
        except BaseException as exc:
            failures.append(exc)
        if handoff_snapshot is not None:
            try:
                if handoff_snapshot != _canonical({
                    "terminal_receipt": handoff.terminal_receipt,
                    "host_observation": handoff.host_observation,
                }):
                    failures.append(
                        OwnerGateCloudObservationAuthorError(
                            "owner_gate_cloud_observation_handoff_changed"
                        )
                    )
            except BaseException as exc:
                failures.append(exc)
        if (
            failures
            or runtime_after != runtime_before
            or cloud_public_after != cloud_public_before
            or host_public_after != host_public_before
            or (token is not None and any(token._raw))
        ):
            _error("owner_gate_cloud_observation_capability_changed")
    return result, handoff


def collect_and_author(
    *,
    plan: foundation.OwnerGateFoundationPlan,
    foundation_apply_chain: Any,
    final_network_evidence: foundation.ProductionNetworkEvidence,
    final_network_collector_public_key: Ed25519PublicKey,
    production_ingress_observation: Mapping[str, Any],
    phase: str,
    collected_at_unix: int | None,
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
    owner_identity: launcher.GcloudOwnerAccessToken,
    stage0_transport: Any,
    kit_stream: Any,
    bundle_stream: Any,
) -> Mapping[str, Any]:
    """Collect one CLOUD report while preserving the established API shape."""

    cloud_observation, _handoff = _collect_and_author_components(
        plan=plan,
        foundation_apply_chain=foundation_apply_chain,
        final_network_evidence=final_network_evidence,
        final_network_collector_public_key=(
            final_network_collector_public_key
        ),
        production_ingress_observation=production_ingress_observation,
        phase=phase,
        collected_at_unix=collected_at_unix,
        gcloud_executable=gcloud_executable,
        gcloud_configuration=gcloud_configuration,
        owner_identity=owner_identity,
        stage0_transport=stage0_transport,
        kit_stream=kit_stream,
        bundle_stream=bundle_stream,
    )
    return cloud_observation


def collect_and_author_bound_pair(
    *,
    plan: foundation.OwnerGateFoundationPlan,
    foundation_apply_chain: Any,
    final_network_evidence: foundation.ProductionNetworkEvidence,
    final_network_collector_public_key: Ed25519PublicKey,
    production_ingress_observation: Mapping[str, Any],
    phase: str,
    collected_at_unix: int | None,
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
    owner_identity: launcher.GcloudOwnerAccessToken,
    stage0_transport: Any,
    kit_stream: Any,
    bundle_stream: Any,
) -> BoundObservationPair:
    """Return the opaque pair produced by one exact HOST/CLOUD composite."""

    cloud_observation, handoff = _collect_and_author_components(
        plan=plan,
        foundation_apply_chain=foundation_apply_chain,
        final_network_evidence=final_network_evidence,
        final_network_collector_public_key=(
            final_network_collector_public_key
        ),
        production_ingress_observation=production_ingress_observation,
        phase=phase,
        collected_at_unix=collected_at_unix,
        gcloud_executable=gcloud_executable,
        gcloud_configuration=gcloud_configuration,
        owner_identity=owner_identity,
        stage0_transport=stage0_transport,
        kit_stream=kit_stream,
        bundle_stream=bundle_stream,
    )
    host_observation = getattr(handoff, "host_observation", None)
    if not isinstance(host_observation, Mapping):
        _error("owner_gate_bound_observation_pair_invalid")
    return BoundObservationPair._create(
        cloud_observation=cloud_observation,
        host_observation=host_observation,
        plan_sha256=plan.sha256,
        phase=phase,
    )


__all__ = [
    "BoundObservationPair",
    "OwnerGateCloudObservationAuthorError",
    "PHASES",
    "collect_and_author",
    "collect_and_author_bound_pair",
    "consume_bound_observation_pair",
]
