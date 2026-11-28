"""Git-native version control for Hermes configuration (issue #12238).

Reddit asked for a "git-compatible version of all I do" so that when a machine
dies the same setup can be recreated. Hermes already ships ``hermes backup``
(zip snapshots) and curator snapshots for skills. This module adds the missing
piece: an explicit, local git history of the configuration files themselves.

Deliberately narrow:

* **Explicit only.** No background auto-commit, no daemon, no hidden state. The
  user runs ``hermes config snapshot`` (or a cron they control). This avoids the
  "non-deterministic self-modification" complaint about memory/SOUL writes.
* **Config files only.** Runtime trees (``state.db``, ``sessions``, ``cache``,
  ``installs``, ``tools``) and credential stores are never added. Secrets are
  excluded by path, not by content heuristics, so nothing is half-redacted.
* **Fails closed.** If the profile is not a git repo, ``init_repo`` creates one;
  every other command refuses to run until that is done. No implicit init on
  commit, so a first commit is always a deliberate act.

The module is pure orchestration around ``git`` and can be used by a CLI
subcommand without touching the backup code path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Files that make up a reproducible setup. This mirrors the default profile
# export allow-list (profiles._DEFAULT_EXPORT_INCLUDE_ROOT) minus large/volatile
# artifacts: sessions and cron history are data, not configuration.
CONFIG_FILES = (
    "config.yaml",
    "SOUL.md",
    "MEMORY.md",
    "USER.md",
    "AGENTS.md",
    "CLAUDE.md",
    ".cursorrules",
    "desktop.json",
    "mcp.json",
    "import-sync.json",
    "profile.yaml",
)

# Credential and runtime paths must never enter version control.
SECRET_PATHS = (
    ".env", ".env.local", "auth.json", "auth.json.bak",
    "vault", ".secrets", "secrets", "oauth", "runtime",
)


class NotARepo(RuntimeError):
    """Raised when an operation needs a git repo that has not been initialized."""


def _run(home: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(home), capture_output=True, text=True, check=check
    )


def is_repo(home: Path) -> bool:
    """Whether ``home`` is inside a git work tree."""
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(home), capture_output=True, text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def init_repo(home: Path) -> Path:
    """Initialize git in ``home`` and write a .gitignore for runtime/secret paths.

    Idempotent: an existing repo is left alone.
    """
    home = Path(home)
    if not is_repo(home):
        _run(home, "init")
    ignore = home / ".gitignore"
    if not ignore.exists():
        lines = [f"{p}/" for p in SECRET_PATHS] + [
            "state.db*", "sessions/", "backups/", "cache/", "installs/",
            "tools/", "checkpoints/", "state-snapshots/", "*.zip", "*.log",
        ]
        ignore.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return home


def _tracked_candidates(home: Path) -> list[str]:
    """Configuration files that actually exist under ``home``."""
    return [name for name in CONFIG_FILES if (home / name).exists()]


def snapshot(home: Path, message: str | None = None) -> str:
    """Stage config files and commit them. Returns the short commit hash.

    Refuses to run before :func:`init_repo`.
    """
    home = Path(home)
    if not is_repo(home):
        raise NotARepo(f"{home} is not a git repo; run 'hermes config init-repo' first")

    candidates = _tracked_candidates(home)
    if not candidates:
        return "nothing-to-commit"

    _run(home, "add", "--", *candidates)

    diff = _run(home, "diff", "--cached", "--quiet", check=False)
    if diff.returncode == 0:
        return "nothing-to-commit"

    commit_message = message or "hermes config snapshot"
    _run(home, "commit", "-m", commit_message)
    short = _run(home, "rev-parse", "--short", "HEAD")
    return short.stdout.strip()


def status(home: Path) -> str:
    """Human-readable git status for the config files, or a not-a-repo notice."""
    home = Path(home)
    if not is_repo(home):
        return "not-a-repo"
    candidates = _tracked_candidates(home)
    if not candidates:
        return "no-config-files"
    result = _run(home, "status", "--short", "--", *candidates, check=False)
    return result.stdout.strip() or "clean"
