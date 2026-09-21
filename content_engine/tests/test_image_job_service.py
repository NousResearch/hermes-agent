"""Provider-free end-to-end consumer for staged image job planning."""
from __future__ import annotations

import pytest
from pathlib import Path

from image_backends import BackendPlanError
from image_job_service import stage_and_plan_image_job
from image_jobs import prepare_image_request
from image_reference_staging import FetchedReference


PUBLIC_IP = "93.184.216.34"


def test_stage_and_plan_consumes_references_without_provider_execution(tmp_path: Path) -> None:
    request = prepare_image_request(
        prompt="A compact visual system map.",
        style="Data Atlas",
        references=("https://example.com/brief",),
    )

    result = stage_and_plan_image_job(
        request,
        staging_root=tmp_path / "staging",
        job_id="job-service",
        resolver=lambda hostname: (PUBLIC_IP,),
        fetcher=lambda url: FetchedReference(
            final_url=url,
            content_type="text/plain",
            body=b"Source material for the generated preview.",
            redirect_count=0,
        ),
    )

    assert result.plan.backend == "codex"
    assert result.plan.execution_enabled is False
    assert result.staged_references[0].text_excerpt == "Source material for the generated preview."
    assert (tmp_path / "staging" / "job-service" / "reference-manifest.json").is_file()


def test_stage_and_plan_removes_incomplete_job_when_backend_planning_fails(tmp_path: Path) -> None:
    class _RejectingRouter:
        def plan(self, *args, **kwargs):
            raise BackendPlanError("test-only backend planning failure")

    root = tmp_path / "staging"
    request = prepare_image_request(
        prompt="A compact visual system map.",
        style="Data Atlas",
        references=("https://example.com/brief",),
    )

    with pytest.raises(BackendPlanError, match="test-only"):
        stage_and_plan_image_job(
            request,
            staging_root=root,
            job_id="job-failed",
            resolver=lambda hostname: (PUBLIC_IP,),
            fetcher=lambda url: FetchedReference(
                final_url=url,
                content_type="text/plain",
                body=b"Source material.",
                redirect_count=0,
            ),
            router=_RejectingRouter(),
        )

    assert not (root / "job-failed").exists()
