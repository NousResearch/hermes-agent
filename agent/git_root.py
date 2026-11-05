"""Git-root validity — the one rule every repository-root resolver shares.

Three consumers walk up from a working directory looking for a ``.git`` entry:
``agent.coding_context._git_root``, ``agent.prompt_builder._find_git_root`` and
``hermes_cli.web_routers.files._fs_find_git_root``. Each originally tested the entry
with a bare ``.exists()``, which a *phantom* ``.git`` satisfies — an empty directory
left behind by tooling (this host has one at ``$HOME``: 0 entries, and
``git -C $HOME rev-parse`` says "not a git repository").

A phantom that wins the upward walk hijacks every project below it: AGENTS.md /
.hermes.md lookup chains climb to ``$HOME`` (so a home-level ``AGENTS.md`` would be
injected as project context for every non-git project under the home directory), the
coding posture's workspace snapshot resolves to the wrong root, and the web UI reports
the home directory as the "git root" for any file under it.

The rule therefore lives here once, rather than being re-derived per consumer. This
module is dependency-free (stdlib only), so prompt assembly, the coding posture and the
web layer can all import it without risking an import cycle. The walk loops stay with
their consumers — each has its own termination guard and error shape — and only the
validity test is shared.

Both layouts git itself writes are valid:

* ``.git`` is a **directory**: requires ``HEAD``, ``objects`` **and** ``refs`` — the same
  trio git's own ``is_git_directory`` demands. ``HEAD`` + ``objects`` alone is not enough:
  ``git rev-parse`` still answers "not a git repository" for it, so accepting it would be
  the phantom bug again in a slightly better disguise;
* ``.git`` is a **file**: requires a ``gitdir:`` pointer whose target directory exists
  (a linked worktree from ``git worktree add``, or a submodule working tree).

Everything doubtful is rejected so the caller keeps walking up (fail closed). No path
here raises: a parent the process may not stat is "not a repository", not a crash.
Filesystem checks only — no shelling out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_GITDIR_PREFIX = "gitdir:"


def gitdir_target(dot_git: Path) -> Optional[Path]:
    """The existing directory a ``.git`` *file* points at, or ``None`` (fail closed).

    ``git worktree add`` and submodule checkouts replace ``.git`` with a file holding a
    single ``gitdir: <path>`` line; a relative path is relative to that file's directory.
    """
    try:
        text = dot_git.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith(_GITDIR_PREFIX):
            continue
        raw = stripped[len(_GITDIR_PREFIX):].strip()
        if not raw:
            return None
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = dot_git.parent / target
        try:
            target = target.resolve()
        except OSError:
            return None
        return target if target.is_dir() else None
    return None


def is_valid_git_root(p: Path) -> bool:
    """Whether *p* really is a repository root — not merely a path with a ``.git`` entry.

    See the module docstring for why a bare ``.exists()`` is not enough. Never raises:
    an unreadable path is treated as "not a repository here" so the caller keeps walking.
    """
    dot_git = p / ".git"
    try:
        if dot_git.is_dir():
            # git's own `is_git_directory` requires all three; `HEAD` + `objects` without
            # `refs` is still "not a git repository" to `git rev-parse`, so rejecting it
            # here keeps this test no weaker than git's. `refs` may be a file (git accepts
            # a plain-file ref store layout), so test existence, not directory-ness.
            return (
                (dot_git / "HEAD").is_file()
                and (dot_git / "objects").exists()
                and (dot_git / "refs").exists()
            )
        if dot_git.is_file():
            return gitdir_target(dot_git) is not None
    except OSError:
        return False
    return False
