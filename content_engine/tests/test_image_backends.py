"""Provider plans must preserve Codex-primary and Local-manual policy."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_backends import BackendPlanError, ImageBackendRouter
from image_jobs import prepare_image_request
from image_reference_staging import StagedReference


def _staged_reference() -> StagedReference:
    return StagedReference(
        source_url="https://example.com/source",
        final_url="https://example.com/source",
        content_type="image/png",
        sha256="a" * 64,
        relative_path="job-001/reference-001-aaaaaaaaaaaaaaaa.png",
        requested_roles=frozenset({"written_inspiration", "visual_reference"}),
        realised_roles=frozenset({"visual_reference"}),
        text_excerpt="",
    )


def test_codex_plan_is_native_and_keeps_untrusted_prompt_out_of_executable_arguments(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    job_dir = root / "job-001"
    job_dir.mkdir(parents=True)
    request = prepare_image_request(
        prompt="A system map; touch /tmp/never-run",
        style="Data Atlas",
        references=("https://example.com/source",),
    )

    staged_reference = _staged_reference()
    staged_path = root / staged_reference.relative_path
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path.write_bytes(b"staged-reference")
    plan = ImageBackendRouter().plan(
        request,
        (staged_reference,),
        job_dir=job_dir,
        staging_root=root,
    )

    assert plan.backend == "codex"
    assert plan.execution_enabled is False
    assert not hasattr(plan, "argv")
    assert "native openai-codex" in plan.instruction_path.read_text().lower()
    assert "codex exec" not in plan.instruction_path.read_text().lower()
    assert plan.instruction_path.is_file()
    assert "touch /tmp/never-run" in plan.manifest_path.read_text()
    assert plan.manifest_path.is_file()
    manifest = json.loads(plan.manifest_path.read_text())
    assert manifest["backend"] == "codex"
    assert manifest["reference_sha256"] == ["a" * 64]


def test_local_plan_is_available_only_when_explicit_and_is_not_execution_enabled(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    job_dir = root / "job-local"
    job_dir.mkdir(parents=True)
    request = prepare_image_request(
        prompt="A local-quality evaluation sheet.",
        style="Technical Diorama",
        backend="local",
    )

    plan = ImageBackendRouter().plan(request, (), job_dir=job_dir, staging_root=root)

    assert plan.backend == "local"
    assert plan.execution_enabled is False
    assert not hasattr(plan, "argv")
    assert "manual quality" in plan.reason.lower()


def test_router_rejects_job_directory_outside_the_trusted_staging_root(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    request = prepare_image_request(prompt="A system map.", style="Data Atlas")

    with pytest.raises(BackendPlanError, match="trusted staging root"):
        ImageBackendRouter().plan(request, (), job_dir=outside, staging_root=root)
