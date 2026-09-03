"""Pinned Collective Wisdom wire contract and canonicalization helpers.

The Gateway remains authoritative for authorization and publication. Claims
decoded by this package are display hints only; every mutation is rechecked by
the Gateway.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


@dataclass(frozen=True)
class ContractPin:
    gateway_commit: str
    openapi_sha256: str
    manifest_schema_sha256: str
    canonical_vectors_sha256: str
    requirements_pr: str
    requirements_commit: str


CONTRACT_PIN = ContractPin(
    gateway_commit="abf016f25afc99a5af815057b41286b6f32ac2dc",
    openapi_sha256="1c35e8501d549ad30beb0666f99ce13b27b6fc78a5c5e7c4f044d2e806ba93a3",
    manifest_schema_sha256="64d0010eada1d79fa16309e9fd715faf77b6186360ea0b095182b2bdaeec5714",
    canonical_vectors_sha256="e2b28c708f69e99b342de1df48498d96efde68867857391590bc964a609a730b",
    requirements_pr="NousResearch/gateway-gateway#215",
    requirements_commit="d51078fd30ed461542dfbbf25a5b810b6b512036",
)

PACKAGE_MANIFEST_SCHEMA_VERSION = 1
AUTHOR_DESCRIPTION_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
FILESYSTEM_SAFE_ORG_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def org_directory_name(org_id: str) -> str:
    """Map an opaque organization id to a portable managed-tree segment."""
    if FILESYSTEM_SAFE_ORG_ID_RE.fullmatch(org_id):
        return org_id
    return "org-" + hashlib.sha256(org_id.encode("utf-8")).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HermesRequirement(StrictModel):
    minimum_version: str


class ModelRequirement(StrictModel):
    capabilities: list[str] = Field(default_factory=list, max_length=64)
    minimum_context_window: int | None = Field(
        default=None, ge=1, le=9_007_199_254_740_991
    )


class ToolRequirement(StrictModel):
    name: str
    minimum_version: str | None = None
    auto_install: Literal[False] = False
    requires_admin: bool = False


class PluginRequirement(StrictModel):
    id: str
    minimum_version: str | None = None
    required: bool = True


class FilesystemRequirement(StrictModel):
    read: list[str] = Field(default_factory=list, max_length=64)
    write: list[str] = Field(default_factory=list, max_length=64)


class NetworkRequirement(StrictModel):
    destinations: list[str] = Field(default_factory=list, max_length=64)


class RuntimeRequirement(StrictModel):
    shell: bool = False
    browser: bool = False
    code: bool = False
    sandbox: bool = True


class SystemSpecification(StrictModel):
    hermes: HermesRequirement
    platforms: list[str] = Field(default_factory=list, max_length=64)
    architectures: list[str] = Field(default_factory=list, max_length=64)
    model: ModelRequirement = Field(default_factory=ModelRequirement)
    tools: list[ToolRequirement] = Field(default_factory=list, max_length=64)
    plugins: list[PluginRequirement] = Field(default_factory=list, max_length=64)
    credentials: list[str] = Field(default_factory=list, max_length=64)
    connections: list[str] = Field(default_factory=list, max_length=64)
    filesystem: FilesystemRequirement = Field(default_factory=FilesystemRequirement)
    network: NetworkRequirement = Field(default_factory=NetworkRequirement)
    runtime: RuntimeRequirement = Field(default_factory=RuntimeRequirement)
    hardware: list[str] = Field(default_factory=list, max_length=64)
    known_limitations: list[str] = Field(default_factory=list, max_length=64)

    @field_validator(
        "platforms",
        "architectures",
        "credentials",
        "connections",
        "hardware",
        "known_limitations",
    )
    @classmethod
    def _bounded_strings(cls, values: list[str]) -> list[str]:
        if any(not value or len(value.encode("utf-8")) > 512 for value in values):
            raise ValueError("requirement values must be 1..512 UTF-8 bytes")
        return values


class PackageManifest(StrictModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=512)
    requirements: SystemSpecification


class ContentFile(StrictModel):
    path: str
    hash: str
    mode: Literal["file", "exec"]
    content_base64: str | None = None

    @field_validator("hash")
    @classmethod
    def _hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("invalid sha256 address")
        return value


def sha256_address(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_content_manifest(files: list[ContentFile]) -> bytes:
    lines = sorted(f"{item.path} {item.mode} {item.hash}\n" for item in files)
    return "".join(lines).encode("utf-8")


def derive_content_hash(files: list[ContentFile]) -> str:
    return sha256_address(canonical_content_manifest(files))


def sanitize_author_description(raw: str) -> str:
    # Mirror gateway descriptionSanitizer.ts. html.unescape is intentionally
    # not used: the Gateway strips tags but does not decode entities.
    value = re.sub(r"<[^>]*>", "", raw)
    value = "".join(
        ch
        for ch in value
        if ord(ch) in (0x09, 0x0A, 0x0D) or (ord(ch) > 0x1F and ord(ch) != 0x7F)
    )
    value = unicodedata.normalize(
        "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
    )
    value = "\n".join(line.rstrip() for line in value.split("\n")).strip()
    if not value or len(value.encode("utf-8")) > 4096:
        raise ValueError("author description must be 1..4096 canonical UTF-8 bytes")
    return value


def author_description_hash(canonical_description: str) -> str:
    return sha256_address(canonical_description.encode("utf-8"))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_number(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not permitted: {value}")


def parse_manifest_bytes(raw: bytes) -> PackageManifest:
    """Parse a manifest without JSON last-key-wins or non-finite values."""
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_number,
    )
    return PackageManifest.model_validate(value)


def load_manifest(path: Path) -> tuple[PackageManifest, bytes]:
    raw = path.read_bytes()
    parsed = parse_manifest_bytes(raw)
    # The bytes themselves are consented. We validate but never rewrite them.
    return parsed, raw


def escape_for_display(value: str) -> str:
    """Escape untrusted server/skill text for HTML-based surfaces."""
    return html.escape(value, quote=True)
