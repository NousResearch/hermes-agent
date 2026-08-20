"""Path and profile resolution for the Claude Code self-improvement pipeline.

Every path in this module resolves from two environment inputs so tests can
sandbox the whole pipeline without touching a real machine:

* ``HOME`` — the OS home directory. ``~/.claude`` and ``~/.claude-hermes``
  are always direct children of this, matching the real layout on George's
  Mac. Overridable per-root for cases where a profile keeps Claude Code state
  somewhere else (``CLAUDE_SELFIMPROVE_CLAUDE_DIR`` /
  ``CLAUDE_SELFIMPROVE_CLAUDE_HERMES_DIR``).
* ``HERMES_HOME`` — the active Hermes profile. Pipeline state (checkpoints,
  the candidate database, backups, the audit log) lives under
  ``{HERMES_HOME}/state/claude-selfimprove/``, next to
  ``self-improvement-notify`` state, so a second Hermes profile never shares
  state with the first. Resolution order mirrors ``hermes_constants
  .get_hermes_home()`` (override env var, then platform default) without
  importing the hermes-agent package, keeping this pipeline dependency-free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _home() -> Path:
    """The OS user's home directory, honoring ``$HOME`` for tests."""
    env_home = os.environ.get("HOME", "").strip()
    if env_home:
        return Path(env_home)
    return Path.home()


def hermes_home() -> Path:
    """The active Hermes profile directory.

    ``HERMES_HOME`` wins when set (matches every other Hermes automation
    script). Falls back to ``~/.hermes``.
    """
    env_home = os.environ.get("HERMES_HOME", "").strip()
    if env_home:
        return Path(env_home)
    return _home() / ".hermes"


def claude_home() -> Path:
    """Root of the primary Claude Code profile (``~/.claude`` by default)."""
    override = os.environ.get("CLAUDE_SELFIMPROVE_CLAUDE_DIR", "").strip()
    if override:
        return Path(override)
    return _home() / ".claude"


def claude_hermes_home() -> Path:
    """Root of the secondary (Hermes-flavored) Claude Code profile.

    On George's Mac this is ``~/.claude-hermes`` — a separate session/
    transcript root that shares most *configuration* with ``~/.claude`` via
    symlinks, but keeps its own ``projects/`` transcript directory. t3code
    worktree sessions run under this profile, so it must be scanned too.
    """
    override = os.environ.get("CLAUDE_SELFIMPROVE_CLAUDE_HERMES_DIR", "").strip()
    if override:
        return Path(override)
    return _home() / ".claude-hermes"


@dataclass(frozen=True)
class SourceRoot:
    """One Claude Code profile's transcript directory to scan."""

    name: str
    projects_dir: Path


def source_roots() -> list[SourceRoot]:
    """Every transcript root the pipeline scans, in a stable order.

    Only roots that exist are returned — a machine without a
    ``~/.claude-hermes`` profile scans just ``~/.claude`` instead of
    erroring.
    """
    candidates = [
        SourceRoot("claude", claude_home() / "projects"),
        SourceRoot("claude-hermes", claude_hermes_home() / "projects"),
    ]
    return [c for c in candidates if c.projects_dir.is_dir()]


def claude_md_path() -> Path:
    """The target CLAUDE.md file the pipeline's managed block lives in."""
    return claude_home() / "CLAUDE.md"


def rules_dir() -> Path:
    """The target directory for pipeline-owned global rule files."""
    return claude_home() / "rules"


def skills_dir() -> Path:
    """The target directory for pipeline-owned global skills."""
    return claude_home() / "skills"


def state_dir() -> Path:
    """Where this pipeline's own state lives: checkpoints, DB, backups, log."""
    return hermes_home() / "state" / "claude-selfimprove"


def checkpoints_path() -> Path:
    return state_dir() / "checkpoints.json"


def candidates_db_path() -> Path:
    return state_dir() / "candidates.db"


def audit_log_path() -> Path:
    return state_dir() / "audit.log.jsonl"


def backups_dir() -> Path:
    return state_dir() / "backups"


def run_log_path() -> Path:
    return state_dir() / "run.log"


def lock_dir_path() -> Path:
    return state_dir() / ".lock"


def self_improvement_queue_path() -> Path:
    """The existing Hermes Slack-notification queue this pipeline feeds.

    Shared with the ``self-improvement-notify`` plugin and drained by
    ``~/.hermes/scripts/self_improvement_watch.sh``. Writing compatible
    events here reuses that script's batching, de-duplication, and
    redaction instead of building a second Slack sender.
    """
    return hermes_home() / "state" / "self-improvement-notify" / "queue.jsonl"


def ensure_state_dirs() -> None:
    """Create every directory this pipeline writes to, if missing."""
    for path in (state_dir(), backups_dir()):
        path.mkdir(parents=True, exist_ok=True)
