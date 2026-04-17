"""Typed payloads for sandbox providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IsolationProfile:
    """Resource and kernel isolation knobs (mapped to Docker / runc flags)."""

    network_policy: str = "bridge"
    cpu_quota: float = 1.0
    mem_limit_mb: int = 2048
    cap_drop: list[str] = field(default_factory=lambda: ["ALL"])
    seccomp_profile_ref: str = ""
    fs_mounts: list[str] = field(default_factory=list)


@dataclass
class SandboxExecResult:
    """Structured result from an isolated command."""

    stdout: str
    stderr: str
    exit_code: int
    artifacts: dict[str, Any] | None = None


@dataclass
class FSSnapshot:
    """Opaque snapshot handle for post-exec diff hooks."""

    snapshot_id: str
    root: str


def isolation_profile_from_config(sandbox_cfg: dict, profile_name: str) -> IsolationProfile:
    """Build :class:`IsolationProfile` from config ``sandbox.profiles``."""
    profiles = sandbox_cfg.get("profiles") or {}
    raw: dict[str, Any] = dict(profiles.get(profile_name) or profiles.get("default") or {})
    return IsolationProfile(
        network_policy=str(raw.get("network_policy", "bridge")),
        cpu_quota=float(raw.get("cpu_quota", 1.0)),
        mem_limit_mb=int(raw.get("mem_limit_mb", 2048)),
        cap_drop=list(raw.get("cap_drop") or ["ALL"]),
        seccomp_profile_ref=str(raw.get("seccomp_profile_ref", "")),
        fs_mounts=list(raw.get("fs_mounts") or []),
    )
