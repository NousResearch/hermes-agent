"""Normalized route identity and canonical route registries."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import unicodedata
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit

from .validation import (
    DomainValidationError,
    ROUTE_CANONICALIZATION_VERSION,
    _HEX,
    _MAX_ROUTE_REGISTRY_ITEMS,
    _REQUIRED_ROUTE_FIELDS,
    _ROUTE_FIELDS,
    _SENSITIVE_VALUE_PATTERN,
    _UNRESERVED,
    _ascii_trimmed_nfc,
    _mapping_snapshot,
    _normalized_text,
    _reject_sensitive,
    _reject_sensitive_mapping_key,
    _validated_mapping_keys,
)
def _normalize_percent_path(path: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(path):
        character = path[index]
        if character != "%":
            output.append(character)
            index += 1
            continue
        if index + 2 >= len(path) or path[index + 1] not in _HEX or path[index + 2] not in _HEX:
            raise DomainValidationError(
                "route.endpoint_invalid",
                "endpoint path contains malformed percent-encoding",
            )
        byte = int(path[index + 1 : index + 3], 16)
        decoded = chr(byte)
        output.append(decoded if decoded in _UNRESERVED else f"%{byte:02X}")
        index += 3
    return "".join(output)

def _remove_dot_segments(path: str) -> str:
    """Apply RFC 3986 §5.2.4 while preserving non-dot empty segments."""

    input_buffer = path
    output_buffer = ""
    while input_buffer:
        if input_buffer.startswith("../"):
            input_buffer = input_buffer[3:]
        elif input_buffer.startswith("./"):
            input_buffer = input_buffer[2:]
        elif input_buffer.startswith("/./"):
            input_buffer = "/" + input_buffer[3:]
        elif input_buffer == "/.":
            input_buffer = "/"
        elif input_buffer.startswith("/../"):
            input_buffer = "/" + input_buffer[4:]
            output_buffer = output_buffer.rsplit("/", 1)[0]
        elif input_buffer == "/..":
            input_buffer = "/"
            output_buffer = output_buffer.rsplit("/", 1)[0]
        elif input_buffer in {".", ".."}:
            input_buffer = ""
        else:
            segment_end = (
                input_buffer.find("/", 1)
                if input_buffer.startswith("/")
                else input_buffer.find("/")
            )
            if segment_end == -1:
                segment_end = len(input_buffer)
            output_buffer += input_buffer[:segment_end]
            input_buffer = input_buffer[segment_end:]
    return output_buffer

def _normalize_host(host: str) -> str:
    normalized = unicodedata.normalize("NFC", host).casefold()
    if ":" in normalized:
        try:
            return ipaddress.IPv6Address(normalized).compressed
        except ValueError as exc:
            raise DomainValidationError(
                "route.endpoint_invalid",
                "endpoint contains an invalid IPv6 host",
            ) from exc
    normalized = normalized.rstrip(".")
    if not normalized:
        raise DomainValidationError(
            "route.endpoint_invalid",
            "endpoint host is required",
        )
    try:
        return normalized.encode("idna").decode("ascii").casefold()
    except UnicodeError as exc:
        raise DomainValidationError(
            "route.endpoint_invalid",
            "endpoint contains an invalid internationalized host",
        ) from exc

def _normalize_endpoint(value: object) -> str:
    raw = _ascii_trimmed_nfc(
        value,
        field_name="endpoint",
        code="route.endpoint_invalid",
    )
    authority = raw.partition("://")[2].partition("/")[0]
    if (
        "\\" in raw
        or "%" in authority
        or any(
            character.isspace()
            or unicodedata.category(character) in {"Cc", "Cf", "Cs"}
            or character in {'"', "<", ">", "{", "}", "|", "^", "`"}
            for character in raw
        )
    ):
        raise DomainValidationError(
            "route.endpoint_invalid",
            "endpoint contains forbidden URL characters or malformed authority encoding",
        )
    if "?" in raw or "#" in raw:
        raise DomainValidationError(
            "route.endpoint_invalid",
            "endpoint must not contain a query or fragment",
        )
    try:
        parsed = urlsplit(raw)
        port = parsed.port
        host = parsed.hostname
    except ValueError as exc:
        raise DomainValidationError("route.endpoint_invalid", "endpoint URL is malformed") from exc

    scheme = parsed.scheme.casefold()
    if (
        scheme not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.netloc.endswith(":")
    ):
        raise DomainValidationError(
            "route.endpoint_invalid",
            "endpoint must be an absolute http(s) URL without credentials, query or fragment",
        )

    normalized_host = _normalize_host(host)
    rendered_host = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    default_port = (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    netloc = rendered_host if port is None or default_port else f"{rendered_host}:{port}"

    percent_normalized = _normalize_percent_path(unicodedata.normalize("NFC", parsed.path or "/"))
    normalized_path = _remove_dot_segments(percent_normalized)
    if not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path
    if normalized_path == "/.":
        normalized_path = "/"
    if normalized_path != "/":
        normalized_path = normalized_path.rstrip("/")

    return urlunsplit((scheme, netloc, normalized_path, "", ""))

@dataclass(frozen=True)
class RouteV1:
    provider: str
    product: str
    surface: str
    account_id: str
    billing_pool_id: str
    quota_pool_id: str
    model: str
    endpoint: str
    region: str
    entitlement_id: str | None = None
    variant: str | None = None
    canonicalization_version: str = ROUTE_CANONICALIZATION_VERSION

    def __post_init__(self) -> None:
        for name in _REQUIRED_ROUTE_FIELDS:
            value = getattr(self, name)
            normalized = (
                _normalize_endpoint(value)
                if name == "endpoint"
                else _normalized_text(value, required=True, field_name=name)
            )
            object.__setattr__(self, name, normalized)
        for name in ("entitlement_id", "variant"):
            object.__setattr__(
                self,
                name,
                _normalized_text(getattr(self, name), field_name=name),
            )
        if self.canonicalization_version != ROUTE_CANONICALIZATION_VERSION:
            raise DomainValidationError(
                "route.canonicalization_unknown",
                "canonicalization_version must be route-v1",
            )
        if any(
            _SENSITIVE_VALUE_PATTERN.search(value)
            for value in self.canonical_object().values()
            if type(value) is str
        ):
            raise DomainValidationError(
                "decision.sensitive_field_prohibited",
                "sensitive value prohibited in route",
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "RouteV1":
        payload = _mapping_snapshot(
            payload,
            code="route.unexpected_field",
            location="route",
        )
        payload_keys = _validated_mapping_keys(
            payload,
            code="route.unexpected_field",
            location="route",
        )
        _reject_sensitive(payload, "route")
        unknown = payload_keys - set(_ROUTE_FIELDS)
        if unknown:
            raise DomainValidationError(
                "route.unexpected_field",
                f"unexpected route fields: {sorted(unknown)}",
            )
        values = {
            key: payload.get(key)
            for key in _ROUTE_FIELDS
            if key != "canonicalization_version"
        }
        if "canonicalization_version" in payload:
            values["canonicalization_version"] = payload["canonicalization_version"]
        return cls(**values)  # type: ignore[arg-type]

    def canonical_object(self) -> dict[str, str | None]:
        return {name: getattr(self, name) for name in _ROUTE_FIELDS}

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_object(),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def route_id(self) -> str:
        digest = hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
        return f"{ROUTE_CANONICALIZATION_VERSION}:{digest}"

def _revalidated_route(value: object, location: str) -> RouteV1:
    if type(value) is not RouteV1:
        raise DomainValidationError(
            "route.context_invalid",
            f"{location} requires a validated RouteV1",
        )
    payload = value.canonical_object()
    try:
        return RouteV1(**payload)  # type: ignore[arg-type]
    except DomainValidationError as exc:
        if exc.code != "decision.sensitive_field_prohibited":
            # RouteV1.from_mapping scans caller mappings before field
            # validation. Preserve that error precedence for a forged typed
            # route while avoiding the duplicate scan on the valid hot path.
            _reject_sensitive(payload, "route")
        raise

def _revalidated_route_registry(
    value: Mapping[str, RouteV1] | None,
    *,
    code: str,
    location: str,
) -> dict[str, RouteV1]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DomainValidationError(code, f"{location} must be a mapping")
    if len(value) > _MAX_ROUTE_REGISTRY_ITEMS:
        raise DomainValidationError(
            code,
            f"{location} must contain at most {_MAX_ROUTE_REGISTRY_ITEMS} routes",
        )
    validated: dict[str, RouteV1] = {}
    for key, route in value.items():
        _reject_sensitive_mapping_key(key, location)
        if type(key) is not str or type(route) is not RouteV1:
            raise DomainValidationError(code, f"{location} requires string keys and RouteV1 values")
        validated_route = _revalidated_route(route, location)
        if key != validated_route.route_id:
            raise DomainValidationError(code, f"{location} keys must match route identities")
        validated[key] = validated_route
    return validated
