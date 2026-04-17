"""Host-specific hardening helpers (Docker flags, no native seccomp BPF here)."""

from __future__ import annotations

import platform
import sys
from pathlib import Path


def seccomp_security_opt(seccomp_profile_ref: str) -> list[str]:
    """Map a host seccomp JSON path to ``docker --security-opt`` fragments."""
    ref = (seccomp_profile_ref or "").strip()
    if not ref:
        return []
    path = Path(ref)
    if not path.is_file():
        return []
    if sys.platform == "win32":
        # Never claim seccomp is active on Windows bind mounts.
        return []
    try:
        resolved = path.resolve()
    except OSError:
        return []
    return ["--security-opt", f"seccomp={resolved}"]


def linux_kernel_major_minor() -> tuple[int, int] | None:
    """Best-effort kernel tuple on Linux; ``None`` elsewhere."""
    if platform.system() != "Linux":
        return None
    release = platform.release().split(".")
    try:
        return int(release[0]), int(release[1].split("-")[0])
    except (ValueError, IndexError):
        return None
