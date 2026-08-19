"""Private plan seam for the converged image-generation workflow.

This module creates auditable plans only. It contains no subprocess, HTTP,
provider, ComfyUI or publishing call. The separate native executor consumes a
Codex plan and records the returned output's private completion provenance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from image_jobs import PreparedImageRequest
from image_reference_staging import StagedReference

class BackendPlanError(RuntimeError):
    """The selected backend cannot safely be planned or made ready."""


@dataclass(frozen=True)
class ImageBackendPlan:
    backend: str
    job_dir: Path
    instruction_path: Path
    manifest_path: Path
    execution_enabled: bool
    reason: str


class ImageBackendRouter:
    """Create one selected backend plan; never choose a fallback backend."""

    @staticmethod
    def _trusted_job_directory(job_dir: Path, staging_root: Path) -> tuple[Path, Path]:
        root = Path(staging_root)
        if root.is_symlink():
            raise BackendPlanError("trusted staging root must not be a symlink")
        try:
            resolved_root = root.resolve(strict=True)
            resolved_job = Path(job_dir).resolve(strict=True)
        except OSError as exc:
            raise BackendPlanError(f"trusted staging root and job directory must exist: {exc}") from exc
        if not resolved_root.is_dir() or not resolved_job.is_dir() or Path(job_dir).is_symlink():
            raise BackendPlanError("trusted staging root and job directory must be real directories")
        try:
            resolved_job.relative_to(resolved_root)
        except ValueError as exc:
            raise BackendPlanError("job directory must be beneath the trusted staging root") from exc
        return resolved_root, resolved_job

    @staticmethod
    def _write_private_json(path: Path, payload: dict) -> None:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        path.chmod(0o600)

    @staticmethod
    def _write_private_text(path: Path, text: str) -> None:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
        path.chmod(0o600)

    def plan(
        self,
        request: PreparedImageRequest,
        references: Iterable[StagedReference],
        *,
        job_dir: Path,
        staging_root: Path,
    ) -> ImageBackendPlan:
        """Persist private instructions and provenance for exactly one backend."""
        root, job = self._trusted_job_directory(job_dir, staging_root)
        staged = tuple(references)
        for reference in staged:
            try:
                (root / reference.relative_path).resolve(strict=True).relative_to(root)
            except (OSError, ValueError) as exc:
                raise BackendPlanError("staged reference escaped the trusted staging root") from exc

        manifest_path = job / "image-job.json"
        instruction_path = job / "backend-instructions.md"
        if manifest_path.exists() or instruction_path.exists():
            raise BackendPlanError("image job metadata already exists")
        self._write_private_json(
            manifest_path,
            {
                "schema_version": 1,
                "backend": request.backend,
                "prompt": request.prompt,
                "style_id": request.style_id,
                "registry_traits": request.registry_traits,
                "registry_slugs": list(request.registry_slugs),
                "registry_seed": request.registry_seed,
                "reference_sha256": [reference.sha256 for reference in staged],
                "references": [
                    {
                        "relative_path": reference.relative_path,
                        "sha256": reference.sha256,
                        "roles": sorted(reference.realised_roles),
                        "text_excerpt": reference.text_excerpt,
                    }
                    for reference in staged
                ],
                "preview_only": True,
            },
        )

        if request.backend == "codex":
            instruction = (
                "# Native openai-codex image job\n\n"
                "This job is staged for the explicit Hermes native `openai-codex` provider. "
                "A trusted executor must use the request manifest, pass only staged visual references, "
                "claim the returned provider output into this private job directory, and write "
                "`image-completion.json` with request and output SHA-256 values.\n\n"
                "Do not invoke Codex CLI, choose another provider, publish, deliver, queue, commit, push, "
                "edit source material or invoke a local image backend.\n"
            )
            self._write_private_text(instruction_path, instruction)
            return ImageBackendPlan(
                backend="codex",
                job_dir=job,
                instruction_path=instruction_path,
                manifest_path=manifest_path,
                execution_enabled=False,
                reason="native openai-codex execution is explicit and separate from planning",
            )

        if request.backend == "local":
            self._write_private_text(
                instruction_path,
                "# Local ComfyUI image job\n\n"
                "This request is recorded for manual quality evaluation only. No local workflow may be "
                "submitted until P12 GPU admission and explicit quality acceptance are complete.\n",
            )
            return ImageBackendPlan(
                backend="local",
                job_dir=job,
                instruction_path=instruction_path,
                manifest_path=manifest_path,
                execution_enabled=False,
                reason="Local execution is manual quality evaluation only and remains disabled",
            )

        raise BackendPlanError(f"unsupported backend: {request.backend}")
