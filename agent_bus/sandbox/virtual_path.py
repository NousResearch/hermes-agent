"""Virtual path translation.

Agent sees:
    /mnt/user-data/workspace/...
    /mnt/user-data/uploads/...
    /mnt/user-data/outputs/...
    /mnt/skills/{public,custom}/...

Physical backing:
    ~/.hermes/threads/{thread_id}/user-data/{workspace,uploads,outputs}/...
    ~/.hermes/skills/{category}/...   (shared across threads)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

VIRTUAL_ROOT = "/mnt"
VIRTUAL_USER_DATA = f"{VIRTUAL_ROOT}/user-data"
VIRTUAL_SKILLS = f"{VIRTUAL_ROOT}/skills"

_USER_DATA_RE = re.compile(rf"^{re.escape(VIRTUAL_USER_DATA)}(/|$)")
_SKILLS_RE = re.compile(rf"^{re.escape(VIRTUAL_SKILLS)}(/|$)")


def _thread_base(thread_id: str) -> Path:
    base = Path(os.environ.get(
        "HERMES_THREADS_ROOT",
        str(Path.home() / ".hermes" / "threads"),
    )).expanduser()
    return base / thread_id


def _skills_base() -> Path:
    return Path(os.environ.get(
        "HERMES_SKILLS_ROOT",
        str(Path.home() / ".hermes" / "skills"),
    )).expanduser()


def translate_virtual_path(virtual_path: str, thread_id: str) -> Path:
    """Map a `/mnt/...` virtual path to a real host path for this thread.

    Raises ValueError if the path escapes or isn't recognized.
    """
    vp = virtual_path
    if not vp.startswith("/"):
        raise ValueError(f"virtual path must be absolute, got {vp!r}")

    # /mnt/user-data/...
    m = _USER_DATA_RE.match(vp)
    if m:
        suffix = vp[len(VIRTUAL_USER_DATA):].lstrip("/")
        real = _thread_base(thread_id) / "user-data" / suffix
        _ensure_within(real, _thread_base(thread_id))
        return real

    # /mnt/skills/...
    m = _SKILLS_RE.match(vp)
    if m:
        suffix = vp[len(VIRTUAL_SKILLS):].lstrip("/")
        real = _skills_base() / suffix
        _ensure_within(real, _skills_base())
        return real

    raise ValueError(f"unrecognized virtual path: {vp}")


def _ensure_within(candidate: Path, container: Path) -> None:
    """Guard against path escape via `..` etc."""
    try:
        candidate.resolve().relative_to(container.resolve())
    except ValueError as e:
        raise ValueError(f"path escape: {candidate} not under {container}") from e


def replace_virtual_paths_in_text(text: str, thread_id: str) -> str:
    """Replace `/mnt/user-data/...` substrings in `text` with host paths.

    Used when feeding agent commands to host tools.
    """
    if not text:
        return text
    # Replace /mnt/user-data/xxx
    text = re.sub(
        rf"{re.escape(VIRTUAL_USER_DATA)}(/[\w\-./]*)?",
        lambda m: str(_thread_base(thread_id) / "user-data") + (m.group(1) or ""),
        text,
    )
    text = re.sub(
        rf"{re.escape(VIRTUAL_SKILLS)}(/[\w\-./]*)?",
        lambda m: str(_skills_base()) + (m.group(1) or ""),
        text,
    )
    return text
