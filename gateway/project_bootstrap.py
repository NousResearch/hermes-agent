"""Project-bootstrap context loader for the gateway.

When a new session is created with a topic-route cwd (see
:mod:`gateway.topic_routing`), the agent in that session needs to know it's
working in a specific project — what files exist, what skills are scoped
to it, recent git activity, etc. Without this context the agent runs as
IRIS-default and the user has to manually say "I'm in project X, here's
what we're doing" on every topic-routed turn.

This module builds a single ``system_reminder`` message that the gateway
prepends to the first turn of any session whose cwd matches a project
root under the user's projects directory. The message is:

  - **invisible to the user** (displayed as a system-reminder, not a
    user/assistant bubble),
  - **read-only context** for the agent (no tool calls, just a prompt
    payload),
  - **generated once per session** (re-invocations of the same session
    re-use the cached reminder; the cache is keyed on session_id and
    stored in memory),
  - **bounded in size** (truncates README, file listings, etc. to a
    conservative budget so the context window is not blown out).

Project detection rules (``is_project_root``):
  - The path is an existing directory.
  - The path contains a ``README.md`` OR an ``AGENTS.md`` OR a
    ``pyproject.toml`` / ``package.json`` / ``Cargo.toml`` at the root.
  - The path lives directly under a configured projects root (default
    ``~/projects``).

The bootstrap is intentionally **non-fatal** at every layer:
  - A project root that fails to read still returns a minimal
    reminder ("Project: X (cwd set, bootstrap failed: <reason>)").
  - A missing README returns a directory listing instead of failing.
  - A missing git history returns "no git repo" instead of failing.

Failure modes are logged at INFO so a config issue is visible in the
gateway log without spamming it.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# Conservative budget for the entire reminder. The agent's first turn
# carries this verbatim into the system prompt, so a runaway project
# listing can blow out the context window. 8KB keeps the reminder
# useful but bounded — large enough to surface a 100-line README, a
# directory tree, and 5 git commits, small enough to be cheap.
_MAX_REMINDER_BYTES = 8192

# Top-of-tree depth for the directory listing. Depth 1 = project root
# only. Depth 2 = project root + immediate subdirs. Going deeper is
# rarely useful and balloons the listing.
_DIR_LISTING_DEPTH = 2

# Cap on number of project-local skills listed. The full content of each
# skill is NOT loaded here — only the names — so the agent knows they
# exist and can call skill_view to load them on demand.
_MAX_SKILLS_LISTED = 20

# README excerpt size. The first 100 lines is enough to surface project
# purpose, install instructions, and key conventions without dragging
# the whole file in.
_README_MAX_LINES = 100

# Number of recent git commits to include.
_GIT_LOG_COUNT = 5

# Files that mark a directory as a "project root" (heuristic — any of
# these present is sufficient).
_PROJECT_ROOT_MARKERS = (
    "README.md",
    "README.rst",
    "README.txt",
    "AGENTS.md",
    "CLAUDE.md",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Gemfile",
)


@dataclass
class ProjectContext:
    """The gathered context for one project root. Cheap to construct."""

    cwd: str
    name: str
    has_git: bool
    readme_excerpt: Optional[str] = None
    readme_path: Optional[str] = None
    agents_path: Optional[str] = None
    directory_listing: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    git_log: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def render(self) -> str:
        """Render the reminder message. Always succeeds — never raises.

        Output budget: the entire string is bounded by ``_MAX_REMINDER_BYTES``.
        Truncation appends a single ``[truncated]`` marker so the agent
        knows the reminder was cut.
        """
        lines: List[str] = []
        lines.append(f"<project_bootstrap cwd={self.cwd!r}>")
        lines.append(f"# Project: {self.name}")
        if self.error:
            lines.append("")
            lines.append(f"_Bootstrap warning: {self.error}_")
        if self.readme_path:
            lines.append("")
            lines.append(f"## README: {self.readme_path}")
            if self.readme_excerpt:
                lines.append(self.readme_excerpt)
        if self.agents_path:
            lines.append("")
            lines.append(f"## Project agent rules: {self.agents_path}")
        if self.directory_listing:
            lines.append("")
            lines.append("## Directory tree (depth=2)")
            lines.extend(self.directory_listing)
        if self.skills:
            lines.append("")
            lines.append(
                f"## Project-local skills ({len(self.skills)} found; "
                "use skill_view to load on demand)"
            )
            for s in self.skills[:_MAX_SKILLS_LISTED]:
                lines.append(f"- {s}")
            if len(self.skills) > _MAX_SKILLS_LISTED:
                lines.append(f"- ... and {len(self.skills) - _MAX_SKILLS_LISTED} more")
        if self.git_log:
            lines.append("")
            lines.append(f"## Recent git activity (last {_GIT_LOG_COUNT} commits)")
            lines.extend(self.git_log)
        lines.append("</project_bootstrap>")

        out = "\n".join(lines)
        if len(out.encode("utf-8")) > _MAX_REMINDER_BYTES:
            out = out[: _MAX_REMINDER_BYTES - 50] + "\n[truncated]\n</project_bootstrap>"
        return out


def is_project_root(path: Path, projects_root: Path) -> bool:
    """Return True iff ``path`` looks like a project root.

    Rules (loosened per #82888 follow-up):
      - ``path`` is an existing directory.
      - At least one of the project-root markers exists, OR the path is a
        direct child of ``projects_root`` with one of the heuristic markers
        (``.hermes/`` or ``code/``).
      - When ``path`` lives directly under ``projects_root``, it is treated
        as a project root as long as it has any project marker. This keeps
        the original narrow behavior for the canonical ``~/projects/<name>``
        layout while also accepting desktop sessions whose cwd happens to be
        a project root identified by marker files alone (e.g.
        ``/home/rashan/projects/foo-bar`` from a desktop session launched
        against that directory).
      - The strict parent == projects_root check is no longer the sole gate
        because the desktop launcher can bind a session to a project cwd
        without that cwd's parent being the projects_root (it may be a
        workspace symlink, a relocated checkout, etc).
    """
    if not path.is_dir():
        return False
    has_marker_file = False
    for marker in _PROJECT_ROOT_MARKERS:
        if (path / marker).is_file():
            has_marker_file = True
            break
    if has_marker_file:
        return True
    # Heuristic fallback: directory contains a ``.hermes/`` or ``code/``
    # subdir, both of which are project conventions in this workspace. Only
    # count as a project root if the path lives under ``projects_root`` —
    # this prevents a stray ``.hermes/`` in any random directory from
    # triggering bootstrap.
    has_heuristic_dir = (path / ".hermes").is_dir() or (path / "code").is_dir()
    if has_heuristic_dir:
        try:
            if path.parent.resolve() == projects_root.resolve():
                return True
        except OSError:
            return False
    return False


def _read_readme(project: Path) -> tuple[Optional[str], Optional[str]]:
    """Read the first ``_README_MAX_LINES`` of README.md if present.

    Returns (excerpt, relative_path) or (None, None) when no README exists.
    """
    for name in ("README.md", "README.rst", "README.txt"):
        readme = project / name
        if not readme.is_file():
            continue
        try:
            text = readme.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return None, str(readme.relative_to(project))
        lines = text.splitlines()[:_README_MAX_LINES]
        return "\n".join(lines), str(readme.relative_to(project))
    return None, None


def _read_agents(project: Path) -> Optional[str]:
    """Return the path to AGENTS.md / CLAUDE.md if present, else None."""
    for name in ("AGENTS.md", "CLAUDE.md"):
        agents = project / name
        if agents.is_file():
            return str(agents.relative_to(project))
    return None


def _directory_listing(project: Path, depth: int) -> List[str]:
    """Render a bounded tree-style directory listing up to ``depth``.

    Skips common noise directories (``.git``, ``__pycache__``, ``node_modules``,
    ``.venv``, ``venv``, ``.mypy_cache``, ``.pytest_cache``, ``dist``,
    ``build``, ``*.egg-info``).
    """
    skip_names = {
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        ".mypy_cache", ".pytest_cache", "dist", "build",
    }
    out: List[str] = []
    try:
        entries = sorted(project.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as e:
        return [f"_could not list directory: {e}_"]
    for entry in entries:
        if entry.name in skip_names or entry.name.startswith("."):
            # Keep .hermes, .gitignore visible (project-relevant)
            if entry.name not in (".hermes", ".gitignore"):
                continue
        suffix = "/" if entry.is_dir() else ""
        out.append(f"  {entry.name}{suffix}")
        if entry.is_dir() and depth > 1:
            try:
                children = sorted(entry.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                continue
            for child in children:
                if child.name in skip_names or (child.name.startswith(".") and child.name not in (".hermes", ".gitignore")):
                    continue
                c_suffix = "/" if child.is_dir() else ""
                out.append(f"    {child.name}{c_suffix}")
    return out


def _list_project_skills(project: Path) -> List[str]:
    """List project-local skill files under ``<project>/.hermes/skills/``.

    Returns names like ``reddit-radar-overview`` (no ``.md``). The agent
    uses these as keys to call skill_view.
    """
    skills_dir = project / ".hermes" / "skills"
    if not skills_dir.is_dir():
        return []
    out: List[str] = []
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        out.append(skill_md.parent.name)
    return out


def _git_log(project: Path, count: int) -> List[str]:
    """Return the last ``count`` git log lines, or empty if not a git repo."""
    git_dir = project / ".git"
    if not git_dir.exists():
        return []
    try:
        result = subprocess.run(
            ["git", "-C", str(project), "log", "--oneline", f"-{count}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


def build_project_context(cwd: str, projects_root: Optional[str] = None) -> ProjectContext:
    """Build a ``ProjectContext`` for the given cwd.

    ``projects_root`` defaults to ``~/projects``. Pass a different value
    in tests or for non-default workspace layouts.

    Returns a fully-populated ``ProjectContext`` (never raises). On any
    read failure, the relevant field is left None and the error is
    stored in ``ProjectContext.error`` so the reminder can still be
    delivered with a "bootstrap warning" annotation.
    """
    if projects_root is None:
        projects_root = os.path.expanduser("~/projects")
    project = Path(cwd)
    name = project.name or cwd
    ctx = ProjectContext(cwd=cwd, name=name, has_git=(project / ".git").exists())

    try:
        ctx.readme_excerpt, ctx.readme_path = _read_readme(project)
        ctx.agents_path = _read_agents(project)
        ctx.directory_listing = _directory_listing(project, depth=_DIR_LISTING_DEPTH)
        ctx.skills = _list_project_skills(project)
        ctx.git_log = _git_log(project, _GIT_LOG_COUNT)
    except Exception as e:  # belt-and-suspenders — bootstrap is never fatal
        ctx.error = f"{type(e).__name__}: {e}"
        logger.info("project_bootstrap: %s cwd=%s", ctx.error, cwd)

    return ctx
