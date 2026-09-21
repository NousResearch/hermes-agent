"""TDD contract for native Codex output claim and completion provenance."""
from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from image_job_service import (
    ImageExecutionError,
    execute_staged_image_job,
    stage_and_plan_image_job,
)
from image_jobs import prepare_image_request


_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


class _SuccessfulCodexProvider:
    def __init__(self, cache_image: Path, *, provider: str = "openai-codex") -> None:
        self._cache_image = cache_image
        self._provider = provider
        self.calls: list[dict[str, object]] = []

    def generate(self, prompt: str, aspect_ratio: str, *, reference_image_urls=None):
        self.calls.append(
            {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "reference_image_urls": list(reference_image_urls or ()),
            }
        )
        return {
            "success": True,
            "provider": self._provider,
            "model": "gpt-image-2-medium",
            "image": str(self._cache_image),
        }


def _staged_job(tmp_path: Path):
    request = prepare_image_request(
        prompt="A compact visual map of the native image pipeline.",
        style="Data Atlas",
    )
    root = tmp_path / "staging"
    staged = stage_and_plan_image_job(request, staging_root=root, job_id="native-job")
    return request, root, staged


def test_native_executor_claims_private_output_and_records_exact_provenance(tmp_path: Path) -> None:
    request, root, staged = _staged_job(tmp_path)
    cache_root = tmp_path / "cache" / "images"
    cache_root.mkdir(parents=True, mode=0o700)
    cache_image = cache_root / "provider-result.png"
    cache_image.write_bytes(_PNG_BYTES)
    cache_image.chmod(0o664)
    provider = _SuccessfulCodexProvider(cache_image)

    completed = execute_staged_image_job(
        request,
        staged,
        staging_root=root,
        job_id="native-job",
        provider=provider,
        provider_cache_root=cache_root,
        aspect_ratio="square",
    )

    assert provider.calls[0]["aspect_ratio"] == "square"
    assert provider.calls[0]["reference_image_urls"] == []
    assert "Data Atlas" in str(provider.calls[0]["prompt"])
    assert completed.output_path == root / "native-job" / "generated.png"
    assert completed.output_path.read_bytes() == _PNG_BYTES
    assert stat.S_IMODE(completed.output_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(cache_image.stat().st_mode) == 0o600
    assert completed.sha256 == hashlib.sha256(_PNG_BYTES).hexdigest()

    completion = json.loads(completed.completion_path.read_text())
    assert completion["backend"] == {"provider": "openai-codex", "model": "gpt-image-2-medium"}
    assert completion["job_id"] == "native-job"
    assert completion["output"]["relative_path"] == "native-job/generated.png"
    assert completion["output"]["sha256"] == completed.sha256
    assert completion["request_sha256"] == hashlib.sha256(staged.plan.manifest_path.read_bytes()).hexdigest()
    assert stat.S_IMODE(completed.completion_path.stat().st_mode) == 0o600


def test_native_executor_rejects_non_codex_result_without_claiming_output(tmp_path: Path) -> None:
    request, root, staged = _staged_job(tmp_path)
    cache_root = tmp_path / "cache" / "images"
    cache_root.mkdir(parents=True, mode=0o700)
    cache_image = cache_root / "provider-result.png"
    cache_image.write_bytes(_PNG_BYTES)

    with pytest.raises(ImageExecutionError, match="openai-codex"):
        execute_staged_image_job(
            request,
            staged,
            staging_root=root,
            job_id="native-job",
            provider=_SuccessfulCodexProvider(cache_image, provider="fal"),
            provider_cache_root=cache_root,
        )

    assert not (root / "native-job" / "generated.png").exists()
    assert not (root / "native-job" / "image-completion.json").exists()


def test_native_executor_rejects_output_outside_provider_cache(tmp_path: Path) -> None:
    request, root, staged = _staged_job(tmp_path)
    cache_root = tmp_path / "cache" / "images"
    cache_root.mkdir(parents=True, mode=0o700)
    outside = tmp_path / "outside.png"
    outside.write_bytes(_PNG_BYTES)

    with pytest.raises(ImageExecutionError, match="provider cache"):
        execute_staged_image_job(
            request,
            staged,
            staging_root=root,
            job_id="native-job",
            provider=_SuccessfulCodexProvider(outside),
            provider_cache_root=cache_root,
        )

    assert not (root / "native-job" / "generated.png").exists()
