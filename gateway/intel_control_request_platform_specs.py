"""Platform specs for oral intel-control request parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from gateway.config import Platform
from gateway.group_target_intents import extract_qq_group_target
from gateway.qq_intel_control_requests import (
    extract_qq_oral_intel_hire_objective,
    extract_qq_worker_name,
    looks_like_qq_intel_worker_context,
    match_qq_intel_control_request,
)


@dataclass(frozen=True)
class IntelControlRequestPlatformSpec:
    platform: Platform
    request_matcher: Callable[..., tuple[dict[str, object] | None, str | None]]
    worker_name_extractor: Callable[[str, Iterable[str]], str]
    worker_context_checker: Callable[[str], bool]
    target_extractor: Callable[[Any, str], str]
    hire_objective_extractor: Callable[..., str]


def build_qq_intel_control_request_platform_spec(
    *,
    request_matcher: Callable[..., tuple[dict[str, object] | None, str | None]] | None = None,
    worker_name_extractor: Callable[[str, Iterable[str]], str] | None = None,
    worker_context_checker: Callable[[str], bool] | None = None,
    target_extractor: Callable[[Any, str], str] | None = None,
    hire_objective_extractor: Callable[..., str] | None = None,
) -> IntelControlRequestPlatformSpec:
    return IntelControlRequestPlatformSpec(
        platform=Platform.QQ_NAPCAT,
        request_matcher=request_matcher or match_qq_intel_control_request,
        worker_name_extractor=worker_name_extractor or extract_qq_worker_name,
        worker_context_checker=worker_context_checker or looks_like_qq_intel_worker_context,
        target_extractor=target_extractor or extract_qq_group_target,
        hire_objective_extractor=hire_objective_extractor or extract_qq_oral_intel_hire_objective,
    )


QQ_INTEL_CONTROL_REQUEST_PLATFORM_SPEC = build_qq_intel_control_request_platform_spec()
