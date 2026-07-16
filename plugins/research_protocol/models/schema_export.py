"""Deterministic JSON Schema export for persisted artifact contracts."""

import json
from pathlib import Path

from pydantic import BaseModel

from .evidence import EvidenceSnapshot, SourceQuality
from .plan import PlanV1
from .publication import FigureSpec
from .review import ConflictRecord, DedupCluster, VerdictRecord
from .run import ManifestRecord, RunConfig
from .source import SourceCandidate

SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "plan.v1.json": PlanV1,
    "run_config.v1.json": RunConfig,
    "manifest_record.v1.json": ManifestRecord,
    "source_candidate.v1.json": SourceCandidate,
    "evidence_snapshot.v1.json": EvidenceSnapshot,
    "source_quality.v1.json": SourceQuality,
    "conflict_record.v1.json": ConflictRecord,
    "dedup_cluster.v1.json": DedupCluster,
    "verdict_record.v1.json": VerdictRecord,
    "figure_spec.v1.json": FigureSpec,
}
BOUNDARY_COMMENT = (
    "JSON Schema validation is necessary but insufficient; artifact bytes must be "
    "validated with load_artifact_from_json_bytes and Pydantic model_validate_json."
)
SEMANTIC_COMMENTS = {
    "conflict_record.v1.json": (
        "Pydantic requires left_provenance_id and right_provenance_id to be distinct."
    ),
    "dedup_cluster.v1.json": (
        "Pydantic requires canonical_provenance_id to be a member of "
        "member_provenance_ids."
    ),
}


def _close_pattern_property_objects(node: object) -> None:
    """Mirror strict Pydantic mapping-key validation in exported schemas."""

    if isinstance(node, dict):
        if "patternProperties" in node:
            node["additionalProperties"] = False
        for value in node.values():
            _close_pattern_property_objects(value)
    elif isinstance(node, list):
        for value in node:
            _close_pattern_property_objects(value)


def render_schema_documents() -> dict[str, bytes]:
    rendered: dict[str, bytes] = {}
    for filename, model in SCHEMA_MODELS.items():
        schema = model.model_json_schema(mode="validation")
        _close_pattern_property_objects(schema)
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$comment"] = " ".join(
            part
            for part in (BOUNDARY_COMMENT, SEMANTIC_COMMENTS.get(filename))
            if part is not None
        )
        rendered[filename] = (
            json.dumps(
                schema,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    return rendered


def write_schema_documents(root: Path | None = None) -> None:
    target = root or Path(__file__).resolve().parents[1] / "schemas"
    target.mkdir(parents=True, exist_ok=True)
    for filename, payload in render_schema_documents().items():
        (target / filename).write_bytes(payload)


if __name__ == "__main__":
    write_schema_documents()
