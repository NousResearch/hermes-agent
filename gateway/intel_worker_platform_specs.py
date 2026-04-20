"""Platform specs for intel-worker metadata loaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from gateway.config import Platform
from gateway.qq_intel_assignments import list_intel_workers


@dataclass(frozen=True)
class IntelWorkerPlatformSpec:
    platform: Platform
    list_workers: Callable[[], list[dict[str, Any]]]


def build_qq_intel_worker_platform_spec(
    *,
    list_workers_fn: Callable[[], list[dict[str, Any]]] | None = None,
) -> IntelWorkerPlatformSpec:
    return IntelWorkerPlatformSpec(
        platform=Platform.QQ_NAPCAT,
        list_workers=list_workers_fn or list_intel_workers,
    )


QQ_INTEL_WORKER_PLATFORM_SPEC = build_qq_intel_worker_platform_spec()


def load_known_intel_worker_names(spec: IntelWorkerPlatformSpec) -> set[str]:
    return {
        str(item.get("worker_name") or "").strip()
        for item in spec.list_workers()
        if isinstance(item, dict) and str(item.get("worker_name") or "").strip()
    }


def load_known_qq_intel_worker_names(
    *,
    spec: IntelWorkerPlatformSpec | None = None,
) -> set[str]:
    return load_known_intel_worker_names(spec or QQ_INTEL_WORKER_PLATFORM_SPEC)
