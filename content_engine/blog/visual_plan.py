"""Provider-free visual-plan contract for P11.

A visual plan chooses immutable, reviewed reference inputs before any image
backend is reached. It does not select a provider or generate an image.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from blog.reference_catalog import ReferenceCatalog

VISUAL_PLAN_VERSION = "1"
VALID_VISUAL_ROLES = frozenset({"layout", "style", "composition", "palette", "subject"})


class VisualPlanError(ValueError):
    """Raised when a visual plan is incomplete or inconsistent."""


@dataclass(frozen=True)
class ReferenceAssignment:
    """One selected core reference bound to the role it serves in an asset."""

    visual_role: str
    reference_id: str
    sha256: str
    provenance_class: str
    usage_classification: str
    parent_reference_id: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "parent_reference_id": self.parent_reference_id,
            "provenance_class": self.provenance_class,
            "reference_id": self.reference_id,
            "sha256": self.sha256,
            "usage_classification": self.usage_classification,
            "visual_role": self.visual_role,
        }


@dataclass(frozen=True)
class VisualAsset:
    role: str
    key: str
    layout: str
    style: str
    palette: str
    motif: str
    reference_assignments: tuple[ReferenceAssignment, ...]
    section_heading: str | None = None

    @property
    def reference_ids(self) -> tuple[str, ...]:
        return tuple(assignment.reference_id for assignment in self.reference_assignments)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": self.key,
            "layout": self.layout,
            "motif": self.motif,
            "palette": self.palette,
            "reference_assignments": [
                assignment.to_dict() for assignment in self.reference_assignments
            ],
            "role": self.role,
            "style": self.style,
        }
        if self.section_heading is not None:
            payload["section_heading"] = self.section_heading
        return payload


@dataclass(frozen=True)
class VisualPlan:
    version: str
    article_id: str
    art_brief: str
    assets: tuple[VisualAsset, ...]

    @property
    def style(self) -> str:
        return self.assets[0].style

    @property
    def palette(self) -> str:
        return self.assets[0].palette

    @property
    def motif(self) -> str:
        return self.assets[0].motif

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "art_brief": self.art_brief,
            "assets": [asset.to_dict() for asset in self.assets],
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VisualPlan":
        if not isinstance(data, Mapping):
            raise VisualPlanError("visual plan must be an object")
        version = _required_text(data, "version")
        if version != VISUAL_PLAN_VERSION:
            raise VisualPlanError(f"unsupported visual plan version {version!r}")
        article_id = _required_text(data, "article_id")
        art_brief = _required_text(data, "art_brief")
        raw_assets = data.get("assets")
        if not isinstance(raw_assets, list) or not raw_assets:
            raise VisualPlanError("assets must be a non-empty list")
        assets = tuple(_asset_from_persisted(item) for item in raw_assets)
        _validate_family(assets)
        return cls(version=version, article_id=article_id, art_brief=art_brief, assets=assets)


def _required_text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VisualPlanError(f"{key} is required")
    return value.strip()


def _assignment_from_catalogue(
    visual_role: str,
    reference_id: str,
    catalog: ReferenceCatalog,
) -> ReferenceAssignment:
    if visual_role not in VALID_VISUAL_ROLES:
        raise VisualPlanError(f"unknown visual role {visual_role!r}")
    record = catalog.get(reference_id)
    if record is None:
        raise VisualPlanError(f"unknown reference {reference_id!r}")
    if not record.is_core:
        raise VisualPlanError(f"reference {reference_id!r} is not a reviewed core record")
    if visual_role not in record.allowed_roles:
        raise VisualPlanError(
            f"reference {reference_id!r} is not allowed for visual role {visual_role!r}"
        )
    return ReferenceAssignment(
        visual_role=visual_role,
        reference_id=reference_id,
        sha256=record.sha256,
        provenance_class=record.provenance_class,
        usage_classification=record.usage_classification,
        parent_reference_id=record.parent_reference_id,
    )


def _assignments_from_input(raw: object, catalog: ReferenceCatalog) -> tuple[ReferenceAssignment, ...]:
    if not isinstance(raw, Mapping) or not raw:
        raise VisualPlanError("reference_assignments must be a non-empty object")
    assignments: list[ReferenceAssignment] = []
    seen_ids: set[str] = set()
    for role in sorted(raw):
        raw_ids = raw[role]
        if not isinstance(role, str):
            raise VisualPlanError("reference assignment role must be a string")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise VisualPlanError(f"reference assignments for {role!r} must be a non-empty list")
        for reference_id in raw_ids:
            if not isinstance(reference_id, str) or not reference_id.strip():
                raise VisualPlanError("reference assignment IDs must be non-empty strings")
            if reference_id in seen_ids:
                raise VisualPlanError(f"duplicate reference assignment {reference_id!r}")
            seen_ids.add(reference_id)
            assignments.append(_assignment_from_catalogue(role, reference_id, catalog))
    return tuple(assignments)


def _asset_from_input(raw: object, catalog: ReferenceCatalog) -> VisualAsset:
    if not isinstance(raw, Mapping):
        raise VisualPlanError("each asset must be an object")
    role = _required_text(raw, "role")
    if role not in {"hero", "section"}:
        raise VisualPlanError("asset role must be hero or section")
    key = _required_text(raw, "key")
    layout = _required_text(raw, "layout")
    style = _required_text(raw, "style")
    palette = _required_text(raw, "palette")
    motif = _required_text(raw, "motif")
    assignments = _assignments_from_input(raw.get("reference_assignments"), catalog)
    section_heading = raw.get("section_heading")
    if role == "hero":
        if section_heading is not None:
            raise VisualPlanError("hero asset must not have section_heading")
    else:
        if not isinstance(section_heading, str) or not section_heading.strip():
            raise VisualPlanError("section asset must have section_heading")
        section_heading = section_heading.strip()
    return VisualAsset(
        role=role,
        key=key,
        layout=layout,
        style=style,
        palette=palette,
        motif=motif,
        reference_assignments=assignments,
        section_heading=section_heading,
    )


def _assignment_from_persisted(raw: object) -> ReferenceAssignment:
    if not isinstance(raw, Mapping):
        raise VisualPlanError("reference assignment must be an object")
    visual_role = _required_text(raw, "visual_role")
    if visual_role not in VALID_VISUAL_ROLES:
        raise VisualPlanError(f"unknown visual role {visual_role!r}")
    parent = raw.get("parent_reference_id")
    if parent is not None and (not isinstance(parent, str) or not parent.strip()):
        raise VisualPlanError("parent_reference_id must be a string or null")
    return ReferenceAssignment(
        visual_role=visual_role,
        reference_id=_required_text(raw, "reference_id"),
        sha256=_required_text(raw, "sha256"),
        provenance_class=_required_text(raw, "provenance_class"),
        usage_classification=_required_text(raw, "usage_classification"),
        parent_reference_id=parent.strip() if isinstance(parent, str) else None,
    )


def _asset_from_persisted(raw: object) -> VisualAsset:
    if not isinstance(raw, Mapping):
        raise VisualPlanError("each asset must be an object")
    role = _required_text(raw, "role")
    key = _required_text(raw, "key")
    section_heading = raw.get("section_heading")
    assignments_raw = raw.get("reference_assignments")
    if not isinstance(assignments_raw, list) or not assignments_raw:
        raise VisualPlanError("reference_assignments must be a non-empty list")
    assignments = tuple(_assignment_from_persisted(item) for item in assignments_raw)
    return VisualAsset(
        role=role,
        key=key,
        layout=_required_text(raw, "layout"),
        style=_required_text(raw, "style"),
        palette=_required_text(raw, "palette"),
        motif=_required_text(raw, "motif"),
        reference_assignments=assignments,
        section_heading=section_heading.strip() if isinstance(section_heading, str) else None,
    )


def _validate_family(assets: tuple[VisualAsset, ...]) -> None:
    if not assets:
        raise VisualPlanError("assets must be a non-empty list")
    heroes = [asset for asset in assets if asset.role == "hero"]
    if len(heroes) != 1:
        raise VisualPlanError("plan must have exactly one hero")
    keys = [asset.key for asset in assets]
    if len(set(keys)) != len(keys):
        raise VisualPlanError("duplicate asset keys")
    layouts = [asset.layout for asset in assets]
    if len(assets) > 1 and len(set(layouts)) != len(layouts):
        raise VisualPlanError("multi-asset plan requires distinct layouts")
    first = next(iter(assets), None)
    if first is None:
        raise VisualPlanError("assets must be a non-empty list")
    for asset in assets:
        if asset.style != first.style:
            raise VisualPlanError("asset style does not match family style")
        if asset.palette != first.palette:
            raise VisualPlanError("asset palette does not match family palette")
        if asset.motif != first.motif:
            raise VisualPlanError("asset motif does not match family motif")


def build_visual_plan(
    *,
    article_id: str,
    art_brief: str,
    assets: Sequence[Mapping[str, Any]],
    catalog: ReferenceCatalog,
) -> VisualPlan:
    """Validate core-reference assignments into an immutable article plan."""
    if not isinstance(catalog, ReferenceCatalog):
        raise VisualPlanError("catalog must be a ReferenceCatalog")
    normalized_assets = tuple(_asset_from_input(asset, catalog) for asset in assets)
    _validate_family(normalized_assets)
    return VisualPlan(
        version=VISUAL_PLAN_VERSION,
        article_id=_required_text({"article_id": article_id}, "article_id"),
        art_brief=_required_text({"art_brief": art_brief}, "art_brief"),
        assets=normalized_assets,
    )


def save_visual_plan(plan: VisualPlan, output_path: Path) -> Path:
    """Atomically persist a deterministic plan to an explicit file path."""
    if not isinstance(plan, VisualPlan):
        raise VisualPlanError("plan must be a VisualPlan")
    output = Path(output_path)
    if output.exists() and output.is_dir():
        raise VisualPlanError("visual plan output must be a file path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(plan.to_json() + "\n", encoding="utf-8")
    temporary.replace(output)
    return output
