"""Strict model contracts for Hermes Research Protocol v1 artifacts."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from plugins.research_protocol.models import (
    ArtifactStatus,
    BudgetLimits,
    Capability,
    CapabilityGrant,
    ConflictRecord,
    ConflictStatus,
    DedupCluster,
    EvidenceSnapshot,
    FigureKind,
    FigureSpec,
    InputReference,
    ManifestRecord,
    OutputRequirement,
    PlanV1,
    RunConfig,
    SourceCandidate,
    SourceIdentifier,
    SourceQuality,
    Verdict,
    VerdictRecord,
    load_artifact_from_json_bytes,
)

NOW = datetime(2026, 7, 13, 20, 0, tzinfo=UTC)
SHA_A = "a" * 64
SHA_B = "b" * 64
META = {
    "producer_version": "0.18.2",
    "run_id": "run-001",
    "created_at": NOW,
}


def _plan_payload() -> dict:
    return {
        "schema_version": "plan.v1",
        **META,
        "objective": "Produce a reproducible offline evidence report.",
        "constraints": ("offline", "no external publication"),
        "inputs": (
            InputReference(
                input_id="input-001",
                artifact_type="fixture",
                sha256=SHA_A,
            ),
        ),
        "outputs": (
            OutputRequirement(
                artifact_id="manifest-001",
                artifact_type="manifest",
            ),
        ),
        "budgets": BudgetLimits(
            max_duration_seconds=60,
            max_executions=1,
            max_records=100,
            max_bytes=1_000_000,
            max_external_calls=0,
        ),
        "capabilities": (
            CapabilityGrant(
                capability=Capability.RESEARCH_COLLECT,
                input_hashes=(SHA_A,),
                external_rights=(),
            ),
        ),
    }


def test_plan_v1_accepts_minimal_strict_contract():
    plan = PlanV1.model_validate(_plan_payload())

    assert plan.schema_version == "plan.v1"
    assert plan.capabilities[0].capability is Capability.RESEARCH_COLLECT
    assert plan.budgets.max_external_calls == 0

    encoded = plan.model_dump_json()
    assert PlanV1.model_validate_json(encoded) == plan


def test_plan_v1_rejects_unknown_fields():
    payload = _plan_payload()
    payload["sql"] = "select * from secrets"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PlanV1.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", " run-001"),
        ("producer_version", "0.18.2 "),
        ("objective", " Produce a reproducible offline evidence report."),
    ],
)
def test_strict_string_contracts_reject_instead_of_normalizing_whitespace(field, value):
    payload = _plan_payload()
    payload[field] = value

    with pytest.raises(ValidationError) as exc_info:
        PlanV1.model_validate(payload)

    assert {error["loc"] for error in exc_info.value.errors()} == {(field,)}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_artifact_models_reject_non_finite_numbers(value):
    with pytest.raises(ValidationError):
        SourceQuality(
            schema_version="source_quality.v1",
            **META,
            source_id="source-001",
            provenance_id="prov-001",
            score=value,
            dimensions={"authority": 0.8},
            rationale=("fixture",),
        )


def test_strict_mapping_contracts_reject_keys_outside_identifier_pattern():
    with pytest.raises(ValidationError) as exc_info:
        SourceQuality(
            schema_version="source_quality.v1",
            **META,
            source_id="source-001",
            provenance_id="prov-001",
            score=0.8,
            dimensions={"invalid key": 0.8},
            rationale=("fixture",),
        )

    assert {error["loc"] for error in exc_info.value.errors()} == {
        ("dimensions", "invalid key", "[key]")
    }


def test_run_and_manifest_models_include_provenance_metadata():
    run = RunConfig(
        schema_version="run_config.v1",
        **META,
        plan_artifact_id="plan-001",
        plan_sha256=SHA_A,
        capability=Capability.RESEARCH_COLLECT,
        budgets=_plan_payload()["budgets"],
        input_hashes=(SHA_A,),
        policy_hashes={"source_quality.v1": SHA_B},
        seed=7,
    )
    manifest = ManifestRecord(
        schema_version="manifest_record.v1",
        **META,
        artifact_id="evidence-001",
        artifact_type="evidence_snapshot",
        path_relative="evidence/evidence-001.json",
        sha256=SHA_B,
        byte_length=42,
        status=ArtifactStatus.SUCCESS,
        provenance_ids=("prov-001",),
    )

    assert run.plan_sha256 == SHA_A
    assert manifest.status is ArtifactStatus.SUCCESS
    assert manifest.provenance_ids == ("prov-001",)


def test_source_and_evidence_models_require_stable_provenance():
    source = SourceCandidate(
        schema_version="source_candidate.v1",
        **META,
        source_id="source-001",
        adapter_id="fixtures.v1",
        locator="fixture://source-001",
        title="Fixture source",
        identifiers=(SourceIdentifier(kind="fixture", value="source-001"),),
        authors=("Example Author",),
        published_at=NOW,
        license="CC0-1.0",
        provenance_id="prov-source-001",
    )
    evidence = EvidenceSnapshot(
        schema_version="evidence_snapshot.v1",
        **META,
        evidence_id="evidence-001",
        source_id=source.source_id,
        locator=source.locator,
        content_sha256=SHA_A,
        extract="A bounded fixture extract.",
        provenance_id="prov-evidence-001",
    )

    assert evidence.source_id == source.source_id
    assert evidence.provenance_id != source.provenance_id


def test_review_models_bind_bilateral_provenance_and_verdict_evidence():
    conflict = ConflictRecord(
        schema_version="conflict_record.v1",
        **META,
        conflict_id="conflict-001",
        claim_id="claim-001",
        left_provenance_id="prov-left",
        right_provenance_id="prov-right",
        conflict_type="direction",
        status=ConflictStatus.OPEN,
        rationale="Fixtures disagree on direction.",
    )
    cluster = DedupCluster(
        schema_version="dedup_cluster.v1",
        **META,
        cluster_id="cluster-001",
        canonical_provenance_id="prov-left",
        member_provenance_ids=("prov-left", "prov-duplicate"),
        method="exact",
        confidence=1.0,
    )
    verdict = VerdictRecord(
        schema_version="verdict_record.v1",
        **META,
        verdict_id="verdict-001",
        question_id="question-001",
        verdict=Verdict.MIXED,
        evidence_provenance_ids=(
            conflict.left_provenance_id,
            conflict.right_provenance_id,
        ),
        rationale="The bounded fixture evidence conflicts.",
        confidence=0.5,
    )

    assert cluster.canonical_provenance_id in cluster.member_provenance_ids
    assert verdict.verdict is Verdict.MIXED


def test_plan_rejects_duplicate_capability_grants():
    payload = _plan_payload()
    payload["capabilities"] = (
        CapabilityGrant(
            capability=Capability.RESEARCH_COLLECT,
            input_hashes=(SHA_A,),
        ),
        CapabilityGrant(
            capability=Capability.RESEARCH_COLLECT,
            input_hashes=(SHA_B,),
        ),
    )

    with pytest.raises(ValidationError, match="capabilities must be unique"):
        PlanV1.model_validate(payload)


def test_conflict_rejects_identical_provenance_references():
    with pytest.raises(ValidationError, match="provenance references must be distinct"):
        ConflictRecord(
            schema_version="conflict_record.v1",
            **META,
            conflict_id="conflict-001",
            claim_id="claim-001",
            left_provenance_id="prov-same",
            right_provenance_id="prov-same",
            conflict_type="direction",
            status=ConflictStatus.OPEN,
            rationale="A source cannot conflict with itself.",
        )


def test_dedup_cluster_rejects_duplicate_members():
    with pytest.raises(ValidationError, match="member_provenance_ids must be unique"):
        DedupCluster(
            schema_version="dedup_cluster.v1",
            **META,
            cluster_id="cluster-001",
            canonical_provenance_id="prov-same",
            member_provenance_ids=("prov-same", "prov-same"),
            method="exact",
            confidence=1.0,
        )


def test_dedup_cluster_requires_canonical_member():
    with pytest.raises(
        ValidationError, match="canonical provenance must be a cluster member"
    ):
        DedupCluster(
            schema_version="dedup_cluster.v1",
            **META,
            cluster_id="cluster-001",
            canonical_provenance_id="prov-canonical",
            member_provenance_ids=("prov-left", "prov-right"),
            method="semantic",
            confidence=0.9,
        )


def test_figure_spec_is_data_bound_not_an_opaque_image():
    figure = FigureSpec(
        schema_version="figure_spec.v1",
        **META,
        figure_id="figure-001",
        title="Fixture comparison",
        kind=FigureKind.BAR,
        data_artifact_id="manifest-001",
        data_sha256=SHA_A,
        x_field="source",
        y_fields=("score",),
        units={"score": "ratio"},
        labels={"score": "Quality score"},
        palette=("#3366CC",),
    )

    assert figure.data_sha256 == SHA_A
    assert figure.kind is FigureKind.BAR


@pytest.mark.parametrize(
    "path",
    [
        "..\\\\secret.json",
        "foo\\\\..\\\\secret.json",
        ".",
        "..",
        "/absolute/file.json",
        "C:/absolute/file.json",
        "foo//bar.json",
        "foo/./bar.json",
        "foo/../bar.json",
        "foo/bar.json/",
        "foo\x00bar.json",
        " evidence/evidence-001.json",
        "evidence/evidence-001.json ",
    ],
)
def test_manifest_rejects_every_unsafe_relative_path_form(path):
    with pytest.raises(ValidationError) as exc_info:
        ManifestRecord(
            schema_version="manifest_record.v1",
            **META,
            artifact_id="evidence-001",
            artifact_type="evidence_snapshot",
            path_relative=path,
            sha256=SHA_B,
            byte_length=42,
            status=ArtifactStatus.SUCCESS,
            provenance_ids=("prov-001",),
        )

    assert {error["loc"] for error in exc_info.value.errors()} == {("path_relative",)}


def test_manifest_rejects_path_exceeding_bounded_text_limit():
    with pytest.raises(ValidationError) as exc_info:
        ManifestRecord(
            schema_version="manifest_record.v1",
            **META,
            artifact_id="artifact-001",
            artifact_type="evidence_snapshot",
            path_relative="a" * 8193,
            sha256=SHA_B,
            byte_length=42,
            status=ArtifactStatus.SUCCESS,
            provenance_ids=("prov-001",),
        )

    assert {error["loc"] for error in exc_info.value.errors()} == {("path_relative",)}


def test_artifact_bytes_loader_round_trips_enum_and_aware_timestamp():
    plan = PlanV1.model_validate(_plan_payload())

    encoded = plan.to_canonical_json_bytes()
    loaded = load_artifact_from_json_bytes(PlanV1, encoded)

    assert loaded == plan
    assert loaded.capabilities[0].capability is Capability.RESEARCH_COLLECT
    assert loaded.created_at == NOW


def test_artifact_bytes_loader_is_bytes_only():
    plan = PlanV1.model_validate(_plan_payload())

    with pytest.raises(TypeError, match="bytes"):
        load_artifact_from_json_bytes(PlanV1, plan.to_canonical_json_bytes().decode())


def test_artifact_sequential_collections_are_deeply_immutable():
    plan = PlanV1.model_validate(_plan_payload())

    assert isinstance(plan.constraints, tuple)
    with pytest.raises(AttributeError):
        plan.constraints.append("mutated")


def test_artifact_mappings_are_deeply_immutable_and_serializable():
    run = RunConfig(
        schema_version="run_config.v1",
        **META,
        plan_artifact_id="plan-001",
        plan_sha256=SHA_A,
        capability=Capability.RESEARCH_COLLECT,
        budgets=_plan_payload()["budgets"],
        input_hashes=(SHA_A,),
        policy_hashes={"source_quality.v1": SHA_B},
        seed=7,
    )

    with pytest.raises(TypeError):
        run.policy_hashes["source_quality.v1"] = "not-a-sha"

    assert (
        load_artifact_from_json_bytes(RunConfig, run.to_canonical_json_bytes()) == run
    )
