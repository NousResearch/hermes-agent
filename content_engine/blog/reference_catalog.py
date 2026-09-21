"""Provider-free, schema-v2 reference catalogue contract for P11.

The catalogue is deliberately provider-free. It validates the external reference
root, exposes the visually reviewed core set for planning, and refuses every
non-permitted input for real generation. It neither invokes a provider nor
writes reference data.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

__all__ = [
    "CatalogError",
    "CatalogIntegrityError",
    "CatalogLoadError",
    "GenerationEligibilityError",
    "ReferenceCatalog",
    "ReferenceRecord",
]

_SCHEMA_VERSION = "2"
_ALLOWED_CLASSIFICATIONS = {"permitted", "review-required", "blocked"}
_ALLOWED_PROVENANCE_CLASSES = {
    "sahil_curated",
    "baoyu_derived",
    "derived",
    "generated_output",
}
_ALLOWED_VISUAL_ROLES = {"layout", "style", "composition", "palette", "subject"}


class CatalogError(Exception):
    """Base class for reference-catalogue errors."""


class CatalogLoadError(CatalogError):
    """Raised when catalogue data cannot be read."""


class CatalogIntegrityError(CatalogError):
    """Raised when catalogue data violates the immutable contract."""


class GenerationEligibilityError(CatalogError):
    """Raised when a requested image input is not generation-eligible."""


@dataclass(frozen=True)
class ReferenceRecord:
    """One immutable reference record, optionally enriched by the core pack."""

    reference_id: str
    path: str
    sha256: str
    record_schema_version: str
    provenance_class: str
    ownership_or_usage_basis: str
    usage_classification: str
    allowed_roles: tuple[str, ...]
    parent_reference_id: Optional[str]
    collection: str = ""
    source: str = ""
    provenance_state: str = ""
    bytes_: int = 0
    core_role: str = ""
    core_tag: str = ""
    curation_status: str = ""
    blocked_roles: tuple[str, ...] = field(default_factory=tuple)
    visual_rationale: str = ""
    is_core: bool = False


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                value = raw.strip()
                if not value:
                    continue
                try:
                    row = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise CatalogLoadError(
                        f"{path.name}:{line_number}: invalid JSON ({exc.msg})"
                    ) from exc
                if not isinstance(row, dict):
                    raise CatalogIntegrityError(
                        f"{path.name}:{line_number}: row must be a JSON object"
                    )
                rows.append(row)
    except OSError as exc:
        raise CatalogLoadError(f"cannot read {path}: {exc}") from exc
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CatalogIntegrityError(f"cannot hash reference {path}: {exc}") from exc
    return digest.hexdigest()


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogIntegrityError("reference path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise CatalogIntegrityError(
            f"reference path must be relative and safe: {value!r}"
        )
    return path.as_posix()


def _resolved_reference_path(root: Path, relative_path: str) -> Path:
    """Resolve one reference and prove it remains physically beneath ``root``."""
    candidate = root / relative_path
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CatalogIntegrityError(
            f"reference path resolves outside catalogue root: {relative_path!r}"
        ) from exc
    if not resolved.is_file():
        raise CatalogIntegrityError(f"missing file for reference path: {relative_path!r}")
    return resolved


def _required_string(row: dict, key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CatalogIntegrityError(f"{context}: missing/invalid {key}")
    return value.strip()


def _roles(row: dict, context: str) -> tuple[str, ...]:
    raw_roles = row.get("allowed_roles")
    if not isinstance(raw_roles, list):
        raise CatalogIntegrityError(f"{context}: allowed_roles must be a list")
    roles = tuple(str(role).strip() for role in raw_roles if str(role).strip())
    if len(set(roles)) != len(roles):
        raise CatalogIntegrityError(f"{context}: duplicate allowed_roles")
    unknown = set(roles) - _ALLOWED_VISUAL_ROLES
    if unknown:
        raise CatalogIntegrityError(
            f"{context}: unknown allowed_roles {sorted(unknown)!r}"
        )
    return roles


def _blocked_roles(row: dict, context: str) -> tuple[str, ...]:
    raw_roles = row.get("blocked_roles", [])
    if not isinstance(raw_roles, list):
        raise CatalogIntegrityError(f"{context}: blocked_roles must be a list")
    return tuple(str(role).strip() for role in raw_roles if str(role).strip())


def _record_from_row(
    row: dict,
    *,
    root: Path,
    context: str,
    is_core: bool,
) -> ReferenceRecord:
    record_schema_version = _required_string(row, "record_schema_version", context)
    if record_schema_version != _SCHEMA_VERSION:
        raise CatalogIntegrityError(
            f"{context}: record_schema_version must be {_SCHEMA_VERSION!r}"
        )
    reference_id = _required_string(row, "reference_id", context)
    relative_path = _safe_relative_path(row.get("path"))
    expected_hash = _required_string(row, "sha256", context).lower()
    if len(expected_hash) != 64 or set(expected_hash) - set("0123456789abcdef"):
        raise CatalogIntegrityError(f"{context}: invalid sha256 for {reference_id!r}")
    provenance_class = _required_string(row, "provenance_class", context)
    if provenance_class not in _ALLOWED_PROVENANCE_CLASSES:
        raise CatalogIntegrityError(
            f"{context}: invalid provenance_class {provenance_class!r}"
        )
    ownership_or_usage_basis = _required_string(row, "ownership_or_usage_basis", context)
    usage_classification = _required_string(row, "usage_classification", context)
    if usage_classification not in _ALLOWED_CLASSIFICATIONS:
        raise CatalogIntegrityError(
            f"{context}: invalid usage_classification {usage_classification!r}"
        )
    parent = row.get("parent_reference_id")
    if parent is not None and (not isinstance(parent, str) or not parent.strip()):
        raise CatalogIntegrityError(f"{context}: invalid parent_reference_id")
    resolved = _resolved_reference_path(root, relative_path)
    actual_hash = _sha256(resolved)
    if actual_hash != expected_hash:
        raise CatalogIntegrityError(
            f"{context}: sha256 mismatch for {reference_id!r}: "
            f"declared={expected_hash!r} actual={actual_hash!r}"
        )
    return ReferenceRecord(
        reference_id=reference_id,
        path=relative_path,
        sha256=expected_hash,
        record_schema_version=record_schema_version,
        provenance_class=provenance_class,
        ownership_or_usage_basis=ownership_or_usage_basis,
        usage_classification=usage_classification,
        allowed_roles=_roles(row, context),
        parent_reference_id=parent.strip() if isinstance(parent, str) else None,
        collection=str(row.get("collection", "")),
        source=str(row.get("source", "")),
        provenance_state=str(row.get("provenance_state", "")),
        bytes_=int(row.get("bytes", 0) or 0),
        core_role=str(row.get("core_role", "")),
        core_tag=str(row.get("core_tag", "")),
        curation_status=str(row.get("curation_status", "")),
        blocked_roles=_blocked_roles(row, context),
        visual_rationale=str(row.get("visual_rationale", "")),
        is_core=is_core,
    )


@dataclass(frozen=True)
class ReferenceCatalog:
    """Validated reference root with baseline records and core enrichments."""

    root: Path
    baseline: dict[str, ReferenceRecord]
    core: dict[str, ReferenceRecord]

    def __iter__(self) -> Iterator[ReferenceRecord]:
        for reference_id in sorted(self.baseline):
            yield self.get(reference_id) or self.baseline[reference_id]

    def __len__(self) -> int:
        return len(self.baseline)

    def get(self, reference_id: str) -> Optional[ReferenceRecord]:
        return self.core.get(reference_id) or self.baseline.get(reference_id)

    def all_records(self) -> list[ReferenceRecord]:
        return list(self)

    def records_for_contract(self) -> list[ReferenceRecord]:
        """Return only visually reviewed core candidates, never generation approval."""
        return [self.core[reference_id] for reference_id in sorted(self.core)]

    def references_for_generation(self, reference_ids: list[str]) -> list[ReferenceRecord]:
        """Fail closed unless every requested reference is explicitly permitted."""
        if not isinstance(reference_ids, list) or not reference_ids:
            raise GenerationEligibilityError("reference_ids must be a non-empty list")
        selected: list[ReferenceRecord] = []
        for reference_id in reference_ids:
            record = self.core.get(reference_id)
            if record is None:
                if reference_id in self.baseline:
                    raise GenerationEligibilityError(
                        f"reference id is not a core record: {reference_id!r}"
                    )
                raise GenerationEligibilityError(f"unknown reference id: {reference_id!r}")
            if record.usage_classification != "permitted":
                raise GenerationEligibilityError(
                    f"reference id {reference_id!r} is not permitted for generation "
                    f"(usage_classification={record.usage_classification!r})"
                )
            selected.append(record)
        return selected

    @classmethod
    def load(cls, root: Path) -> "ReferenceCatalog":
        root = Path(root).resolve(strict=True)
        manifest_path = root / "manifest.jsonl"
        if not manifest_path.is_file():
            raise CatalogLoadError(f"manifest.jsonl not found under root {root}")

        baseline: dict[str, ReferenceRecord] = {}
        baseline_paths: set[str] = set()
        for index, row in enumerate(_read_jsonl(manifest_path), start=1):
            context = f"manifest row {index}"
            record = _record_from_row(row, root=root, context=context, is_core=False)
            if record.reference_id in baseline:
                raise CatalogIntegrityError(
                    f"{context}: duplicate reference_id {record.reference_id!r}"
                )
            if record.path in baseline_paths:
                raise CatalogIntegrityError(f"{context}: duplicate path {record.path!r}")
            baseline[record.reference_id] = record
            baseline_paths.add(record.path)

        for record in baseline.values():
            if record.parent_reference_id and record.parent_reference_id not in baseline:
                raise CatalogIntegrityError(
                    f"parent reference {record.parent_reference_id!r} is missing for "
                    f"{record.reference_id!r}"
                )

        core: dict[str, ReferenceRecord] = {}
        core_path = root / "core-pack.jsonl"
        for index, row in enumerate(_read_jsonl(core_path), start=1):
            context = f"core-pack row {index}"
            record = _record_from_row(row, root=root, context=context, is_core=True)
            if record.reference_id in core:
                raise CatalogIntegrityError(
                    f"{context}: duplicate core reference_id {record.reference_id!r}"
                )
            base = baseline.get(record.reference_id)
            if base is None:
                raise CatalogIntegrityError(
                    f"{context}: no baseline row for {record.reference_id!r}"
                )
            if (record.path, record.sha256) != (base.path, base.sha256):
                raise CatalogIntegrityError(
                    f"{context}: path/hash must match baseline for {record.reference_id!r}"
                )
            if (
                record.provenance_class,
                record.ownership_or_usage_basis,
                record.parent_reference_id,
            ) != (
                base.provenance_class,
                base.ownership_or_usage_basis,
                base.parent_reference_id,
            ):
                raise CatalogIntegrityError(
                    f"{context}: provenance fields must match baseline for "
                    f"{record.reference_id!r}"
                )
            if not set(record.allowed_roles).issubset(base.allowed_roles):
                raise CatalogIntegrityError(
                    f"{context}: allowed_roles must be a subset of baseline roles for "
                    f"{record.reference_id!r}"
                )
            core[record.reference_id] = record
        return cls(root=root, baseline=baseline, core=core)
