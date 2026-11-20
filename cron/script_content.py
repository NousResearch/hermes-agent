"""``cronjob_manage(script_content=...)``: save a gate script under ``<HERMES_HOME>/scripts/``.

The model can create a script-gated watcher in one call: the handler writes the body before the
usual ``script`` path validation runs, and the approval card shows it in full. A write never
clobbers someone else's file: an existing target with different content is refused unless it is
the updated job's own script. ``StagedScript.discard()`` undoes the write when the create or
update it belongs to fails afterwards.
"""

from __future__ import annotations

import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MAX_SCRIPT_BYTES = 64 * 1024
_BARE_SCRIPT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.(?:py|sh|bash)")


def is_bare_script_name(name: object) -> bool:
    return isinstance(name, str) and bool(_BARE_SCRIPT_NAME.fullmatch(name))


def derived_script_name(job_name: Optional[str]) -> str:
    """``cron_<slug of the job name>_<6 hex>.py`` for a script_content given without ``script``."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(job_name or "").lower()).strip("_")[:32] or "job"
    return f"cron_{slug}_{secrets.token_hex(3)}.py"


@dataclass
class StagedScript:
    path: Optional[Path] = None
    previous: Optional[bytes] = None  # the bytes an own-script overwrite replaced
    error: Optional[str] = None

    def discard(self) -> None:
        """Undo the write: restore an overwritten own script, remove a new file."""
        if self.path is None:
            return
        if self.previous is not None:
            self.path.write_bytes(self.previous)
        else:
            self.path.unlink(missing_ok=True)


def stage_script_content(content: object, name: str, own_script: Optional[str] = None) -> StagedScript:
    """Write ``content`` to ``<HERMES_HOME>/scripts/<name>`` (0600, UTF-8). ``own_script`` is the
    updated job's current ``script``: the one existing file it may overwrite."""
    if not isinstance(content, str) or not content.strip():
        return StagedScript(error="script_content must be the non-empty source of the script.")
    body = content.encode("utf-8")
    if len(body) > MAX_SCRIPT_BYTES:
        return StagedScript(error=f"script_content is over {MAX_SCRIPT_BYTES // 1024} KB.")
    if not is_bare_script_name(name):
        return StagedScript(error=(
            f"With script_content, script must be a bare filename ending in .py, .sh or .bash "
            f"(no directories); got {name!r}."))
    from hermes_constants import get_hermes_home

    scripts_dir = get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    path = scripts_dir / name
    previous = path.read_bytes() if path.is_file() else None
    if path.exists() and previous != body and name != (own_script or "").strip():
        return StagedScript(error=(
            f"{path} already exists with different content and is not this job's script; "
            "pick another script name."))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(body)
    os.chmod(path, 0o600)
    # Same bytes already on disk: nothing to undo either way.
    return StagedScript(path=None if previous == body else path, previous=previous)
