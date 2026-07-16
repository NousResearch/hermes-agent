from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway import canonical_writer_production_cutover as cutover
from gateway import production_cron_continuity_package as continuity
from scripts.canary import stage_production_cron_continuity as stage
from tests.gateway.test_canonical_writer_production_cutover import (
    REVISION,
    Services,
    _freeze,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def test_legacy_authority_is_an_explicit_zero_mutation_noop(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    freeze = _freeze(Ed25519PrivateKey.generate(), Services())
    freeze_path = tmp_path / "freeze-plan.json"
    freeze_path.write_bytes(_canonical(freeze.to_mapping()) + b"\n")
    output_root = tmp_path / "cron-continuity"
    monkeypatch.setattr(cutover, "STAGED_FREEZE_PLAN_PATH", freeze_path)
    monkeypatch.setattr(stage, "DEFAULT_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(stage.os, "geteuid", lambda: 0)

    assert stage.main(["stage", "--revision", REVISION]) == 0

    receipt = json.loads(capsys.readouterr().out)
    unsigned = {
        name: item
        for name, item in receipt.items()
        if name != "receipt_sha256"
    }
    assert receipt["schema"] == stage.LEGACY_NOOP_SCHEMA
    assert receipt["freeze_plan_sha256"] == freeze.sha256
    assert receipt["legacy_noop"] is True
    assert receipt["artifacts_staged"] is False
    assert receipt["jobs_store_mutated"] is False
    assert receipt["receipt_sha256"] == hashlib.sha256(
        _canonical(unsigned)
    ).hexdigest()
    assert not output_root.exists()


def test_stage_rejects_caller_supplied_mechanical_package(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_path = tmp_path / "package.json"
    package_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(stage.os, "geteuid", lambda: 0)

    try:
        stage.main([
            "stage",
            "--revision",
            REVISION,
            "--mechanical-job-package",
            str(package_path),
        ])
    except RuntimeError as exc:
        assert str(exc) == "production_cron_stage_package_unexpected"
    else:
        raise AssertionError("caller-supplied package unexpectedly accepted")


def test_packaged_v4_authority_stages_the_bound_continuity_plan(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    freeze_path = tmp_path / "freeze-plan.json"
    freeze_path.write_text("{}", encoding="ascii")
    continuity_plan = {"schema": continuity.PLAN_SCHEMA, "plan_sha256": "a" * 64}
    mechanical_package = {"manifest_sha256": "b" * 64}
    freeze = SimpleNamespace(
        sha256="c" * 64,
        value={
            "release_revision": REVISION,
            "cutover_authority": {
                "cron_continuity_plan": continuity_plan,
                "mechanical_job_package": mechanical_package,
            },
        },
    )
    observed: dict[str, object] = {}

    def stage_bound(**kwargs):
        observed.update(kwargs)
        return {"schema": "staged-v4"}

    monkeypatch.setattr(cutover, "STAGED_FREEZE_PLAN_PATH", freeze_path)
    monkeypatch.setattr(
        cutover.FreezePlan,
        "from_mapping",
        lambda _value: freeze,
    )
    monkeypatch.setattr(stage, "stage_packaged_continuity_from_host", stage_bound)
    monkeypatch.setattr(stage.os, "geteuid", lambda: 0)

    assert stage.main(["stage", "--revision", REVISION]) == 0

    assert json.loads(capsys.readouterr().out) == {"schema": "staged-v4"}
    assert observed["revision"] == REVISION
    assert observed["mechanical_job_package"] is mechanical_package
    assert observed["expected_continuity_plan"] is continuity_plan


def test_unknown_continuity_schema_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    freeze_path = tmp_path / "freeze-plan.json"
    freeze_path.write_text("{}", encoding="ascii")
    freeze = SimpleNamespace(
        sha256="c" * 64,
        value={
            "release_revision": REVISION,
            "cutover_authority": {
                "cron_continuity_plan": {
                    "schema": "muncho-production-cron-packaged-continuity-plan.v999"
                },
                "mechanical_job_package": {},
            },
        },
    )
    monkeypatch.setattr(cutover, "STAGED_FREEZE_PLAN_PATH", freeze_path)
    monkeypatch.setattr(
        cutover.FreezePlan,
        "from_mapping",
        lambda _value: freeze,
    )
    monkeypatch.setattr(stage.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        stage,
        "stage_packaged_continuity_from_host",
        lambda **_kwargs: pytest.fail("unknown schema reached staging"),
    )

    with pytest.raises(RuntimeError, match="production_cron_stage_schema_unsupported"):
        stage.main(["stage", "--revision", REVISION])
