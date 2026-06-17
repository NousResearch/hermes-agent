"""Typed-agent spec preview DTOs and validation constants.

These models are read-only reporting structures. They do not enforce runtime
policy and must not be wired into agent startup without a separate review gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from hermes_constants import VALID_REASONING_EFFORTS

SCHEMA_VERSION = "hermes.agent_spec/v1alpha1"
SUPPORTED_SCHEMA_VERSIONS = {SCHEMA_VERSION, "v1alpha1"}
REASONING_EFFORTS = {"none", *VALID_REASONING_EFFORTS}
MCP_VALIDATION_STATES = {
    "known_in_catalog_and_configured",
    "known_in_catalog_but_not_configured_optional",
    "known_in_catalog_but_required_missing",
    "unknown_server_id",
    "tool_discovery_unavailable",
    "tool_not_in_catalog_or_discovery",
}
SANDBOX_ENFORCEMENT_STATUSES = {
    "declared_only",
    "partially_enforced_by_backend",
    "enforced",
    "not_supported_on_backend",
}


def is_valid_reasoning_effort(value: object) -> bool:
    return isinstance(value, str) and value in REASONING_EFFORTS


def validation_status(errors: list[Any], warnings: list[Any], *, strict: bool = False) -> str:
    if errors or (strict and warnings):
        return "fail"
    if warnings:
        return "warn"
    return "pass"


@dataclass
class SpecSource:
    kind: str
    path: str | None
    status: str
    precedence: int


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    field: str | None = None
    source: str | None = None


@dataclass
class AgentSpecDocument:
    source: SpecSource
    raw: dict[str, Any]
    body: str | None
    format: str
    parse_errors: list[ValidationIssue] = field(default_factory=list)


@dataclass
class McpServerPreview:
    server_id: str
    tool: str | None
    required: bool
    state: str
    configured: bool | None
    catalog_known: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    status: str
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    infos: list[ValidationIssue] = field(default_factory=list)
    sources: list[SpecSource] = field(default_factory=list)
    effective_preview: dict[str, Any] | None = None
    read_only_guarantee: bool = True

    def recompute_status(self, *, strict: bool = False) -> None:
        self.status = validation_status(self.errors, self.warnings, strict=strict)


def to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_plain(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [to_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    return value
