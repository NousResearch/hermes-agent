from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import canonical_writer_production_readiness as readiness


REVISION = "3ec7cfb675e529473ab84c6dd0a7492153f71ffa"
DECISION_EVENT_ID = "11111111-1111-4111-8111-111111111111"


def _receipt() -> dict[str, object]:
    value: dict[str, object] = {
        "schema": readiness.PRODUCTION_PREFLIGHT_SCHEMA,
        "plan_sha256": "1" * 64,
        "artifact_sha256": "2" * 64,
        "final_snapshot_sha256": "3" * 64,
        "source_row_count": 18_928,
        "archive_row_count": 18_928,
        "canonical_row_count": 0,
        "archive_extended19_sha256": "4" * 64,
        "canonical14_sha256": "5" * 64,
        "relation_identity_sha256": "6" * 64,
        "acl_identity_sha256": "7" * 64,
        "index_identity_sha256": "8" * 64,
        "roles_acl_sha256": "9" * 64,
        "zero_canonical_writer_writes": True,
        "legacy_shape_restored": True,
        "ok": True,
        "legacy_truth_mode": "start_new_truth_epoch",
        "legacy_truth_decision_sha256": "a" * 64,
        "legacy_truth_decision_event_id": DECISION_EVENT_ID,
        "accepted_event_set_sha256": "b" * 64,
        "trusted_legacy_event_count": 0,
        "truth_epoch_sha256": "c" * 64,
        "secret_material_recorded": False,
    }
    value["receipt_sha256"] = hashlib.sha256(
        readiness._canonical_bytes(value)
    ).hexdigest()
    return value


def _fixture(tmp_path: Path) -> SimpleNamespace:
    release_base = tmp_path / "releases"
    release = release_base / f"hermes-agent-{REVISION[:12]}"
    module_path = (
        release / "gateway/canonical_writer_production_readiness.py"
    )
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# exact test module\n", encoding="utf-8")
    marker = release / ".codex-source-commit"
    marker.write_text(f"{REVISION}\n", encoding="ascii")
    interpreter = release / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"test interpreter")

    receipt_path = tmp_path / "phase-b/runtime-receipt.json"
    receipt_path.parent.mkdir()
    receipt_path.write_bytes(readiness._canonical_bytes(_receipt()))
    receipt_path.chmod(0o444)

    config_path = tmp_path / "writer.json"
    config = SimpleNamespace(
        writer_uid=os.getuid() + 100_000,
        database=SimpleNamespace(database=readiness.PRODUCTION_DATABASE),
        source_config_path=config_path,
        source_config_sha256="d" * 64,
        discord_edge_authority=SimpleNamespace(enabled=True),
    )
    return SimpleNamespace(
        config=config,
        config_path=config_path,
        release_base=release_base,
        release=release,
        module_path=module_path,
        interpreter=interpreter,
        receipt_path=receipt_path,
    )


def _validate(fixture: SimpleNamespace):
    return readiness._validate_production_phase_b_readiness_at(
        fixture.config,
        release_revision=REVISION,
        receipt_path=fixture.receipt_path,
        release_base=fixture.release_base,
        module_path=fixture.module_path,
        executable=str(fixture.interpreter),
        expected_receipt_uid=os.getuid(),
        expected_receipt_gid=os.getgid(),
        trusted_parent_uid=None,
        expected_config_path=fixture.config_path,
        expected_receipt_path=fixture.receipt_path,
    )


def _replace_receipt(
    fixture: SimpleNamespace,
    value: dict[str, object],
) -> None:
    fixture.receipt_path.chmod(0o644)
    fixture.receipt_path.write_bytes(readiness._canonical_bytes(value))
    fixture.receipt_path.chmod(0o444)


def test_accepts_exact_root_published_production_preflight(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    observed = _validate(fixture)

    assert observed["source_row_count"] == 18_928
    assert observed["legacy_shape_restored"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("legacy_shape_restored", False),
        ("zero_canonical_writer_writes", False),
        ("canonical_row_count", 1),
        ("secret_material_recorded", True),
    ),
)
def test_rejects_unsafe_or_postmigration_receipt_state(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    fixture = _fixture(tmp_path)
    receipt = _receipt()
    receipt[field] = value
    unsigned = {
        name: item
        for name, item in receipt.items()
        if name != "receipt_sha256"
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        readiness._canonical_bytes(unsigned)
    ).hexdigest()
    _replace_receipt(fixture, receipt)

    with pytest.raises(
        readiness.ProductionWriterReadinessError,
        match="production_writer_readiness_receipt_invalid",
    ):
        _validate(fixture)


def test_rejects_receipt_content_tampering(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    receipt = _receipt()
    receipt["source_row_count"] = 18_929
    _replace_receipt(fixture, receipt)

    with pytest.raises(
        readiness.ProductionWriterReadinessError,
        match="production_writer_readiness_receipt_invalid",
    ):
        _validate(fixture)


def test_rejects_release_marker_or_module_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture.release / ".codex-source-commit").write_text(
        f"{'f' * 40}\n",
        encoding="ascii",
    )

    with pytest.raises(
        readiness.ProductionWriterReadinessError,
        match="production_writer_readiness_release_mismatch",
    ):
        _validate(fixture)

    (fixture.release / ".codex-source-commit").write_text(
        f"{REVISION}\n",
        encoding="ascii",
    )
    with pytest.raises(
        readiness.ProductionWriterReadinessError,
        match="production_writer_readiness_module_origin_invalid",
    ):
        readiness._validate_production_phase_b_readiness_at(
            fixture.config,
            release_revision=REVISION,
            receipt_path=fixture.receipt_path,
            release_base=fixture.release_base,
            module_path=fixture.release / "gateway/other.py",
            executable=str(fixture.interpreter),
            expected_receipt_uid=os.getuid(),
            expected_receipt_gid=os.getgid(),
            trusted_parent_uid=None,
            expected_config_path=fixture.config_path,
            expected_receipt_path=fixture.receipt_path,
        )


def test_rejects_nonproduction_config_or_caller_selected_path(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture.config.database.database = "other"

    with pytest.raises(
        readiness.ProductionWriterReadinessError,
        match="production_writer_readiness_config_invalid",
    ):
        _validate(fixture)

    fixture.config.database.database = readiness.PRODUCTION_DATABASE
    with pytest.raises(
        readiness.ProductionWriterReadinessError,
        match="production_writer_readiness_config_invalid",
    ):
        readiness._validate_production_phase_b_readiness_at(
            fixture.config,
            release_revision=REVISION,
            receipt_path=fixture.receipt_path,
            release_base=fixture.release_base,
            module_path=fixture.module_path,
            executable=str(fixture.interpreter),
            expected_receipt_uid=os.getuid(),
            expected_receipt_gid=os.getgid(),
            trusted_parent_uid=None,
            expected_config_path=fixture.config_path,
            expected_receipt_path=tmp_path / "different.json",
        )
