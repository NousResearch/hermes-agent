"""Immutable provider-free P11 asset-provenance and QA manifest contract."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

ASSET_MANIFEST_VERSION = "1"
_ASSET_STATES = {"planned", "generated", "qa-complete"}
_QA_STATES = {"pending", "approved", "rejected"}
_REVIEW_STATES = {"pending", "approved", "rejected"}
_FORBIDDEN_CREDENTIAL_FIELDS = {
    "api_key",
    "authorization",
    "credential",
    "password",
    "provider_token",
    "secret",
    "token",
}


class AssetManifestError(ValueError):
    """Raised when asset provenance cannot be represented safely."""


def _required_text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AssetManifestError(f"{key} is required")
    return value.strip()


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise AssetManifestError(f"{name} must be a sha256 digest")
    normalized = value.strip().lower()
    if len(normalized) != 64 or set(normalized) - set("0123456789abcdef"):
        raise AssetManifestError(f"{name} must be a 64-character sha256 digest")
    return normalized


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AssetManifestError("output_path is required")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise AssetManifestError("output_path must be relative and safe")
    return path.as_posix()


def _dimensions(value: object, *, name: str, allow_none: bool) -> dict[str, int] | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, Mapping):
        raise AssetManifestError(f"{name} must be an object")
    if set(value) != {"width", "height"}:
        raise AssetManifestError(f"{name} must contain exactly width and height")
    dimensions: dict[str, int] = {}
    for key in ("width", "height"):
        dimension = value[key]
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise AssetManifestError(f"{name}.{key} must be a positive integer")
        dimensions[key] = dimension
    return dimensions


@dataclass(frozen=True)
class ReferenceInput:
    reference_id: str
    sha256: str
    provenance_class: str
    visual_role: str

    def to_dict(self) -> dict[str, str]:
        return {
            "provenance_class": self.provenance_class,
            "reference_id": self.reference_id,
            "sha256": self.sha256,
            "visual_role": self.visual_role,
        }


def _reference_inputs(value: object) -> tuple[ReferenceInput, ...]:
    if not isinstance(value, list) or not value:
        raise AssetManifestError("reference_inputs must be a non-empty list")
    inputs: list[ReferenceInput] = []
    seen: set[tuple[str, str]] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise AssetManifestError("each reference input must be an object")
        reference = ReferenceInput(
            reference_id=_required_text(item, "reference_id"),
            sha256=_digest("reference_inputs.sha256", item.get("sha256")),
            provenance_class=_required_text(item, "provenance_class"),
            visual_role=_required_text(item, "visual_role"),
        )
        identity = (reference.reference_id, reference.visual_role)
        if identity in seen:
            raise AssetManifestError(f"duplicate reference input {identity!r}")
        seen.add(identity)
        inputs.append(reference)
    return tuple(inputs)


def _qa(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AssetManifestError("visual_qa must be an object")
    status = _required_text(value, "status")
    if status not in _QA_STATES:
        raise AssetManifestError(f"visual_qa.status must be one of {sorted(_QA_STATES)}")
    reasons = value.get("rejection_reasons")
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        raise AssetManifestError("visual_qa.rejection_reasons must be a list of strings")
    reasons = [reason.strip() for reason in reasons if reason.strip()]
    if status == "rejected" and not reasons:
        raise AssetManifestError("rejected visual_qa requires rejection_reasons")
    return {"status": status, "rejection_reasons": reasons}


def _text_ocr(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise AssetManifestError("text_ocr must be an object")
    return {"policy": _required_text(value, "policy"), "result": _required_text(value, "result")}


@dataclass(frozen=True)
class GeneratedAssetRecord:
    asset_key: str
    article_id: str
    state: str
    visual_plan_schema_version: str
    visual_plan_digest: str
    prompt: str
    prompt_digest: str
    reference_inputs: tuple[ReferenceInput, ...]
    provider: str | None
    model: str | None
    output_path: str
    output_digest: str | None
    requested_dimensions: dict[str, int]
    actual_dimensions: dict[str, int] | None
    generated_at: str | None
    text_ocr: dict[str, str]
    visual_qa: dict[str, Any]
    review_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "actual_dimensions": self.actual_dimensions,
            "article_id": self.article_id,
            "asset_key": self.asset_key,
            "generated_at": self.generated_at,
            "model": self.model,
            "output_digest": self.output_digest,
            "output_path": self.output_path,
            "prompt": self.prompt,
            "prompt_digest": self.prompt_digest,
            "provider": self.provider,
            "reference_inputs": [item.to_dict() for item in self.reference_inputs],
            "requested_dimensions": self.requested_dimensions,
            "review_status": self.review_status,
            "state": self.state,
            "text_ocr": self.text_ocr,
            "visual_plan_digest": self.visual_plan_digest,
            "visual_plan_schema_version": self.visual_plan_schema_version,
            "visual_qa": self.visual_qa,
        }


def _build_record(data: Mapping[str, Any]) -> GeneratedAssetRecord:
    if not isinstance(data, Mapping):
        raise AssetManifestError("record must be an object")
    forbidden = sorted(key for key in data if key.lower() in _FORBIDDEN_CREDENTIAL_FIELDS)
    if forbidden:
        raise AssetManifestError(f"credential fields are forbidden: {forbidden}")
    state = _required_text(data, "state")
    if state not in _ASSET_STATES:
        raise AssetManifestError(f"state must be one of {sorted(_ASSET_STATES)}")
    prompt = _required_text(data, "prompt")
    prompt_digest = _digest("prompt_digest", data.get("prompt_digest"))
    actual_prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if prompt_digest != actual_prompt_digest:
        raise AssetManifestError("prompt_digest does not match prompt")
    review_status = _required_text(data, "review_status")
    if review_status not in _REVIEW_STATES:
        raise AssetManifestError(f"review_status must be one of {sorted(_REVIEW_STATES)}")
    provider = data.get("provider")
    model = data.get("model")
    output_digest = data.get("output_digest")
    actual_dimensions = _dimensions(
        data.get("actual_dimensions"), name="actual_dimensions", allow_none=True
    )
    generated_at = data.get("generated_at")
    if generated_at is not None and (not isinstance(generated_at, str) or not generated_at.strip()):
        raise AssetManifestError("generated_at must be a non-empty timestamp or null")
    if state == "planned":
        if any(value is not None for value in (provider, model, output_digest, actual_dimensions, generated_at)):
            raise AssetManifestError("planned record must not contain generated output evidence")
    else:
        if not isinstance(provider, str) or not provider.strip():
            raise AssetManifestError("generated record requires provider")
        if not isinstance(model, str) or not model.strip():
            raise AssetManifestError("generated record requires model")
        if actual_dimensions is None:
            raise AssetManifestError("generated record requires actual_dimensions")
        if generated_at is None:
            raise AssetManifestError("generated record requires generated_at")
        output_digest = _digest("output_digest", output_digest)
    return GeneratedAssetRecord(
        asset_key=_required_text(data, "asset_key"),
        article_id=_required_text(data, "article_id"),
        state=state,
        visual_plan_schema_version=_required_text(data, "visual_plan_schema_version"),
        visual_plan_digest=_digest("visual_plan_digest", data.get("visual_plan_digest")),
        prompt=prompt,
        prompt_digest=prompt_digest,
        reference_inputs=_reference_inputs(data.get("reference_inputs")),
        provider=provider.strip() if isinstance(provider, str) else None,
        model=model.strip() if isinstance(model, str) else None,
        output_path=_relative_path(data.get("output_path")),
        output_digest=output_digest,
        requested_dimensions=_dimensions(
            data.get("requested_dimensions"), name="requested_dimensions", allow_none=False
        ) or {},
        actual_dimensions=actual_dimensions,
        generated_at=generated_at.strip() if isinstance(generated_at, str) else None,
        text_ocr=_text_ocr(data.get("text_ocr")),
        visual_qa=_qa(data.get("visual_qa")),
        review_status=review_status,
    )


@dataclass(frozen=True)
class AssetManifest:
    version: str
    article_id: str
    records: tuple[GeneratedAssetRecord, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "records": [record.to_dict() for record in self.records],
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssetManifest":
        if not isinstance(data, Mapping):
            raise AssetManifestError("manifest must be an object")
        version = _required_text(data, "version")
        if version != ASSET_MANIFEST_VERSION:
            raise AssetManifestError(f"unsupported manifest version {version!r}")
        article_id = _required_text(data, "article_id")
        raw_records = data.get("records")
        if not isinstance(raw_records, list):
            raise AssetManifestError("records must be a list")
        records = tuple(_build_record(record) for record in raw_records)
        _ensure_unique_keys(records)
        if any(record.article_id != article_id for record in records):
            raise AssetManifestError("all records must match manifest article_id")
        return cls(version=version, article_id=article_id, records=records)

    @classmethod
    def from_json(cls, text: str) -> "AssetManifest":
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as exc:
            raise AssetManifestError(f"invalid JSON: {exc.msg}") from exc


def _ensure_unique_keys(records: tuple[GeneratedAssetRecord, ...]) -> None:
    keys = [record.asset_key for record in records]
    if len(set(keys)) != len(keys):
        raise AssetManifestError("duplicate asset keys")


def build_asset_manifest(article_id: str, records: Sequence[Mapping[str, Any]]) -> AssetManifest:
    """Validate an immutable collection of planned/generated asset records."""
    if not isinstance(records, (list, tuple)):
        raise AssetManifestError("records must be a list")
    manifest = AssetManifest.from_dict(
        {
            "version": ASSET_MANIFEST_VERSION,
            "article_id": article_id,
            "records": records,
        }
    )
    return manifest


def save_asset_manifest(manifest: AssetManifest, output_path: Path) -> Path:
    """Atomically persist a deterministic manifest to an explicit file path."""
    if not isinstance(manifest, AssetManifest):
        raise AssetManifestError("manifest must be an AssetManifest")
    output = Path(output_path)
    if output.exists() and output.is_dir():
        raise AssetManifestError("asset manifest output must be a file path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(manifest.to_json() + "\n", encoding="utf-8")
    temporary.replace(output)
    return output
