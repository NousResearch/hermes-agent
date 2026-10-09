"""Keep the canonical runtime Python package in Docker's source context.

The former runtime/ ignore rule hid every module before the Dockerfile COPY.
The image suite separately checks the actual filtered/published image.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_MODULES = {
    "__init__.py",
    "desktop_identity.py",
    "git_subprocess.py",
    "processes.py",
    "process_identity.py",
    "resource_limits.py",
    "sqlite_runtime.py",
    "stdio.py",
    "subprocess_compat.py",
}


def test_dockerignore_does_not_mask_canonical_runtime_package() -> None:
    """Reject root-wide exclusions even if local source imports still work."""
    patterns = {
        line.strip().removeprefix("./")
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "!"))
    }
    forbidden = {"runtime", "runtime/", "/runtime", "/runtime/", "runtime/**", "/runtime/**"}
    assert not (patterns & forbidden), "Docker context excludes the canonical runtime/ owner"
    for name in RUNTIME_MODULES:
        assert (ROOT / "runtime" / name).is_file(), name
