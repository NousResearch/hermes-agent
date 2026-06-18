"""Durable `/loop` V1 engine — one public primitive for outcome-driven runs.

This module implements the conservative V1 slice of `/loop`: a durable,
profile-aware and repo-aware state store plus an explicit, command-driven
lifecycle. It does **not** launch background workers, run hidden agents, create
Kanban boards, schedule continuations, or auto-advance. Every step is taken by
an explicit `/loop` command.

State layout (repo-bound)::

    <repo>/.hermes/loops/<slug>/
        loop.json        # loop metadata + lifecycle status
        project.md
        docs.md
        decisions.md
        prd.md
        stories.json     # the story manifest
        status.md        # derived, human-readable
        runs/<id>/prompt.md
        runs/<id>/run.json
        reviews/
        closeout.md

Profile-global state (non-repo / gateway / desktop)::

    $HERMES_HOME/loops/<slug>/

The load-bearing rule of V1: a story may only become ``done`` with referenced
evidence. A model assertion is never evidence.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


_STATUS_ORDER = ("todo", "ready", "running", "blocked", "needs_review", "done", "archived")
_TERMINAL_STATUSES = {"done", "archived"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str, *, fallback: str = "loop") -> str:
    """Return a filesystem-safe, lowercase loop slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or fallback


def find_git_root(cwd: str | Path | None = None) -> Path | None:
    """Return the enclosing git worktree root, if any."""
    start = Path(cwd or Path.cwd()).expanduser()
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    root = proc.stdout.strip()
    return Path(root).resolve() if root else None


def workspace_root(cwd: str | Path | None = None) -> Path:
    """Return the evidence workspace root (git root or the working directory)."""
    return find_git_root(cwd) or Path(cwd or Path.cwd()).expanduser().resolve()


@dataclass(frozen=True)
class LoopLocation:
    """Resolved storage location for loop state."""

    base_dir: Path
    scope: str
    repo_root: Path | None = None

    @property
    def active_file(self) -> Path:
        return self.base_dir / ".active"


def resolve_loop_location(
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
    prefer_repo: bool = True,
) -> LoopLocation:
    """Resolve repo-local or profile-global loop storage.

    Repo-local state is used when *cwd* is inside a git worktree. Otherwise the
    active Hermes profile home is used via :func:`get_hermes_home`, so gateway
    and desktop callers never need an interactive repo cwd.
    """
    if prefer_repo:
        repo_root = find_git_root(cwd)
        if repo_root is not None:
            return LoopLocation(repo_root / ".hermes" / "loops", "repo", repo_root)
    home = Path(hermes_home).expanduser() if hermes_home is not None else get_hermes_home()
    return LoopLocation(home / "loops", "profile", None)


def loop_dir_for_slug(
    slug: str,
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> Path:
    loc = resolve_loop_location(cwd, hermes_home=hermes_home)
    return loc.base_dir / slugify(slug)


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _ensure_repo_loop_gitignore(base_dir: Path) -> None:
    """Write conservative ignore guidance for repo-local loop state."""
    gitignore = base_dir / ".gitignore"
    if gitignore.exists():
        return
    gitignore.write_text(
        "# Hermes /loop local operational state.\n"
        "# Review before committing project docs from individual loop folders.\n"
        "*/loop.json\n"
        "*/stories.json\n"
        "*/status.md\n"
        "*/runs/\n"
        "*/reviews/\n"
        ".active\n",
        encoding="utf-8",
    )


def create_loop(
    name_or_goal: str,
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> dict[str, Any]:
    """Create or load a loop and mark it active.

    The operation is idempotent for the same slug: existing metadata is loaded
    and left intact except for `last_opened_at` and the active pointer.
    """
    raw = (name_or_goal or "loop").strip()
    slug = slugify(raw)
    loc = resolve_loop_location(cwd, hermes_home=hermes_home)
    loop_dir = loc.base_dir / slug
    loop_dir.mkdir(parents=True, exist_ok=True)
    if loc.scope == "repo":
        _ensure_repo_loop_gitignore(loc.base_dir)

    state_path = loop_dir / "loop.json"
    created = False
    if state_path.exists():
        state = load_loop_state(loop_dir)
        state["last_opened_at"] = _now_iso()
    else:
        now = _now_iso()
        created = True
        state = {
            "name": raw,
            "slug": slug,
            "goal": raw,
            "status": "planning",
            "phase": "intake",
            "created_at": now,
            "updated_at": now,
            "last_opened_at": now,
            "scope": loc.scope,
            "repo_root": str(loc.repo_root) if loc.repo_root else None,
            "path": str(loop_dir),
            "docs": [],
            "blockers": [],
            "next_steps": ["Add docs or run a grill-with-docs planning pass."],
        }
        (loop_dir / "project.md").write_text(
            f"# {raw}\n\n"
            f"Goal: {raw}\n\n"
            "## Current state\n\n"
            "Planning gate created. No worker execution has started.\n",
            encoding="utf-8",
        )
    state["updated_at"] = _now_iso()
    state["path"] = str(loop_dir)
    _write_json_atomic(state_path, state)
    loc.active_file.write_text(slug + "\n", encoding="utf-8")
    return {"created": created, "location": loc, "state": state, "path": loop_dir}


def load_loop_state(loop_dir: str | Path) -> dict[str, Any]:
    path = Path(loop_dir) / "loop.json"
    with path.open("r", encoding="utf-8") as fh:
        state = json.load(fh)
    state.setdefault("path", str(Path(loop_dir)))
    return state


def save_loop_state(loop_dir: str | Path, state: dict[str, Any]) -> None:
    state = dict(state)
    state["updated_at"] = _now_iso()
    _write_json_atomic(Path(loop_dir) / "loop.json", state)


def list_loops(
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> list[dict[str, Any]]:
    loc = resolve_loop_location(cwd, hermes_home=hermes_home)
    loops: list[dict[str, Any]] = []
    if loc.base_dir.exists():
        for child in sorted(loc.base_dir.iterdir()):
            if child.is_dir() and (child / "loop.json").exists():
                try:
                    loops.append(load_loop_state(child))
                except (OSError, json.JSONDecodeError):
                    continue
    return loops


def resolve_loop(
    name: str | None = None,
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> tuple[Path, dict[str, Any]] | None:
    """Resolve an explicit loop name/path or the active loop."""
    candidates: list[Path] = []
    loc = resolve_loop_location(cwd, hermes_home=hermes_home)
    home_loc = LoopLocation((Path(hermes_home).expanduser() if hermes_home else get_hermes_home()) / "loops", "profile")

    if name:
        raw = Path(name).expanduser()
        if raw.exists():
            candidates.append(raw if raw.is_dir() else raw.parent)
        slug = slugify(name)
        candidates.extend([loc.base_dir / slug, home_loc.base_dir / slug])
    else:
        for active_file, base in ((loc.active_file, loc.base_dir), (home_loc.active_file, home_loc.base_dir)):
            try:
                active = active_file.read_text(encoding="utf-8").strip()
            except OSError:
                active = ""
            if active:
                candidates.append(base / slugify(active))
        loops = list_loops(cwd, hermes_home=hermes_home)
        if loops:
            latest = max(loops, key=lambda s: s.get("updated_at") or s.get("created_at") or "")
            if latest.get("path"):
                candidates.append(Path(latest["path"]))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve() if candidate.exists() else candidate
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "loop.json").exists():
            return candidate, load_loop_state(candidate)
    return None


def _load_stories(loop_dir: Path) -> list[dict[str, Any]]:
    path = loop_dir / "stories.json"
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, dict)]
    if isinstance(raw, dict) and isinstance(raw.get("stories"), list):
        return [s for s in raw["stories"] if isinstance(s, dict)]
    return []


def render_loop_status(
    name: str | None = None,
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> str:
    resolved = resolve_loop(name, cwd, hermes_home=hermes_home)
    if not resolved:
        return "No active loop found. Start one with `/loop start <name or goal>`."
    loop_dir, state = resolved
    stories = _load_stories(loop_dir)
    counts = {k: 0 for k in _STATUS_ORDER}
    for story in stories:
        status = str(story.get("status") or "todo").lower()
        counts[status] = counts.get(status, 0) + 1
    story_summary = "none"
    if stories:
        story_summary = ", ".join(f"{k}:{v}" for k, v in counts.items() if v)
    blockers = state.get("blockers") or []
    next_steps = state.get("next_steps") or []
    lines = [
        f"Loop: {state.get('name') or state.get('slug')}",
        f"Slug: {state.get('slug')}",
        f"Scope: {state.get('scope', 'unknown')}",
        f"Path: {loop_dir}",
        f"Status: {state.get('status', 'planning')}",
        f"Phase: {state.get('phase', 'intake')}",
        f"Stories: {story_summary}",
        f"Blockers: {', '.join(map(str, blockers)) if blockers else 'none recorded'}",
        f"Next: {next_steps[0] if next_steps else 'define PRD/stories or choose one next action'}",
    ]
    return "\n".join(lines)


def _render_loop_view(loop_dir: Path, state: dict[str, Any]) -> str:
    stories = _load_stories(loop_dir)
    done = [s for s in stories if str(s.get("status", "")).lower() in _TERMINAL_STATUSES]
    running = [s for s in stories if str(s.get("status", "")).lower() == "running"]
    blocked = [s for s in stories if str(s.get("status", "")).lower() == "blocked"]
    todo = [s for s in stories if str(s.get("status", "todo")).lower() in {"todo", "ready"}]
    checked: list[str] = ["Loop state created"]
    if (loop_dir / "docs.md").exists() or state.get("docs"):
        checked.append("Docs indexed")
    if (loop_dir / "decisions.md").exists():
        checked.append("Decisions captured")
    if (loop_dir / "prd.md").exists():
        checked.append("PRD drafted")
    if stories:
        checked.append(f"Story manifest present ({len(stories)} stories, {len(done)} done)")

    state_line = state.get("status", "planning")
    if running:
        state_line = "running: " + ", ".join(str(s.get("id") or s.get("title")) for s in running[:3])
    elif blocked:
        state_line = "blocked"
    elif stories:
        state_line = f"{len(done)}/{len(stories)} stories done"

    next_steps = [str(x) for x in (state.get("next_steps") or []) if str(x).strip()]
    default_intake_step = "Add docs or run a grill-with-docs planning pass."
    if stories and next_steps == [default_intake_step]:
        next_steps = []
    if not next_steps:
        if todo:
            first = todo[0]
            next_steps = [f"Run or specify {first.get('id', 'next story')}: {first.get('title', first.get('objective', 'next story'))}"]
        elif not stories:
            next_steps = ["Draft PRD and story manifest from the current goal."]
        else:
            next_steps = ["Close out acceptance, verification, and review summary."]

    blocker_lines = [str(x) for x in (state.get("blockers") or []) if str(x).strip()]
    blocker_lines.extend(
        str(s.get("blocked_reason") or s.get("title") or s.get("id")) for s in blocked
    )

    sections = {
        "Aim": str(state.get("goal") or state.get("name") or state.get("slug") or "Active loop"),
        "Checked off": "\n".join(f"- {item}" for item in checked),
        "Current state": state_line,
        "Next steps": "\n".join(f"- {item}" for item in next_steps[:5]),
        "Blocked/unclear": "\n".join(f"- {item}" for item in blocker_lines) if blocker_lines else "- None recorded",
        "Best next move": next_steps[0],
    }
    return "\n\n".join(f"## {heading}\n{body}" for heading, body in sections.items())


def _render_generic_repo_view(cwd: str | Path | None = None) -> str:
    root = find_git_root(cwd) or Path(cwd or Path.cwd()).resolve()
    plans = sorted((root / "docs" / "plans").glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True) if (root / "docs" / "plans").exists() else []
    aim = f"Repository: {root.name}"
    checked = ["Git/project context detected"] if find_git_root(cwd) else ["Working directory context detected"]
    if plans:
        checked.append(f"Plan file available: {plans[0].relative_to(root)}")
    next_step = "Start a loop with `/loop start <name or goal>` if this needs tracked project state."
    if plans:
        next_step = f"Review/update {plans[0].relative_to(root)} or start a loop for it."
    return "\n\n".join(
        [
            f"## Aim\n{aim}",
            "## Checked off\n" + "\n".join(f"- {item}" for item in checked),
            "## Current state\nNo active loop state found for this context.",
            f"## Next steps\n- {next_step}",
            "## Blocked/unclear\n- No active project authority selected.",
            f"## Best next move\n{next_step}",
        ]
    )


def render_global_view(
    project: str | None = None,
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> str:
    """Render a global project orientation view.

    Prefers explicit/active loop state. Falls back to a conservative repo
    orientation instead of inventing progress.
    """
    resolved = resolve_loop(project, cwd, hermes_home=hermes_home)
    if resolved:
        loop_dir, state = resolved
        return _render_loop_view(loop_dir, state)
    return _render_generic_repo_view(cwd)


_DOC_SKIP_NAMES = {
    ".env",
    ".env.local",
    ".envrc",
    "auth.json",
    "credentials.json",
    "secrets.json",
}
_DOC_SKIP_PARTS = {".git", "node_modules", "venv", ".venv", "__pycache__"}
_DEFAULT_DOC_GLOBS = (
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "SKILL.md",
    "docs/plans/*.md",
)


def _is_safe_doc_path(path: Path) -> bool:
    parts = set(path.parts)
    name = path.name.lower()
    if parts & _DOC_SKIP_PARTS:
        return False
    if name in _DOC_SKIP_NAMES or name.startswith(".env"):
        return False
    if any(token in name for token in ("secret", "credential", "token", "apikey", "api_key")):
        return False
    return path.is_file()


def _summarize_doc_text(text: str, *, limit: int = 260) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("#").strip()
        if line:
            lines.append(line)
        if len(lines) >= 2:
            break
    if lines:
        return " — ".join(lines)[:limit]
    return "No text content found."


def _collect_default_doc_paths(root: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in _DEFAULT_DOC_GLOBS:
        found.extend(root.glob(pattern))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen and _is_safe_doc_path(path):
            seen.add(resolved)
            unique.append(path)
    return unique


def index_docs(
    paths: list[str] | None = None,
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> str:
    """Index explicit docs, or safe default repo docs, into the active loop."""
    resolved = resolve_loop(cwd=cwd, hermes_home=hermes_home)
    if not resolved:
        return "No active loop found. Start one with `/loop start <name or goal>`."
    loop_dir, state = resolved
    root = find_git_root(cwd) or Path(cwd or Path.cwd()).resolve()
    candidates: list[Path] = []
    skipped: list[str] = []

    if paths:
        for raw in paths:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = root / path
            if path.is_dir():
                candidates.extend(p for p in path.rglob("*.md") if _is_safe_doc_path(p))
            elif _is_safe_doc_path(path):
                candidates.append(path)
            else:
                skipped.append(raw)
    else:
        candidates = _collect_default_doc_paths(root)

    sources: list[dict[str, str]] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved_path = path.resolve()
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            skipped.append(str(path))
            continue
        display = str(path)
        try:
            display = str(path.resolve().relative_to(root))
        except ValueError:
            pass
        sources.append(
            {
                "path": display,
                "authority": "explicit" if paths else "repo-default",
                "summary": _summarize_doc_text(text),
            }
        )

    docs_md = [f"# Docs Index — {state.get('name') or state.get('slug')}\n"]
    if sources:
        for item in sources:
            docs_md.append(
                f"- `{item['path']}` ({item['authority']}): {item['summary']}"
            )
    else:
        docs_md.append("- No safe docs found.")
    if skipped:
        docs_md.append("\n## Skipped")
        docs_md.extend(f"- `{item}`" for item in skipped)
    (loop_dir / "docs.md").write_text("\n".join(docs_md) + "\n", encoding="utf-8")
    state["docs"] = sources
    save_loop_state(loop_dir, state)
    return f"Indexed {len(sources)} docs into {loop_dir / 'docs.md'}."


def build_grill_prompt(
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> str:
    """Build a grounded grill-with-docs prompt for the active loop."""
    resolved = resolve_loop(cwd=cwd, hermes_home=hermes_home)
    if not resolved:
        return "No active loop found. Start one with `/loop start <name or goal>`."
    loop_dir, state = resolved
    docs_path = loop_dir / "docs.md"
    docs_summary = "No docs indexed yet. Run `/loop docs add` first if docs exist."
    if docs_path.exists():
        docs_summary = docs_path.read_text(encoding="utf-8", errors="replace")[:4000]
    decisions_path = loop_dir / "decisions.md"
    decisions = "No decisions captured yet."
    if decisions_path.exists():
        decisions = decisions_path.read_text(encoding="utf-8", errors="replace")[:2000]
    return (
        "Use grill-me-with-docs mode for this loop project.\n\n"
        f"Loop: {state.get('name') or state.get('slug')}\n"
        f"Goal: {state.get('goal') or state.get('name')}\n"
        f"Path: {loop_dir}\n\n"
        "Ask one question at a time. Ask only unresolved questions that materially "
        "change the PRD, story plan, safety boundaries, or verification gates. "
        "For each question include: why it matters, your inferred/default answer, "
        "and the consequence of accepting that default. If docs answer the question, "
        "do not ask it. Stop after the smallest useful set of questions.\n\n"
        "Docs summary:\n"
        f"{docs_summary}\n\n"
        "Known decisions:\n"
        f"{decisions}\n"
    )


_STORY_REQUIRED_FIELDS = (
    "id",
    "title",
    "status",
    "kind",
    "assignee",
    "model_hint",
    "depends_on",
    "workspace",
    "objective",
    "context",
    "allowed_paths",
    "acceptance",
    "verification",
    "review_required",
)
_STORY_STATUSES = {"todo", "ready", "running", "blocked", "needs_review", "done", "archived"}
_STORY_KINDS = {"implementation", "research", "review", "docs", "ops", "design", "writing", "code"}


def _read_loop_text(loop_dir: Path, filename: str, fallback: str, *, limit: int = 6000) -> str:
    path = loop_dir / filename
    if not path.exists():
        return fallback
    return path.read_text(encoding="utf-8", errors="replace")[:limit].strip() or fallback


def _story_template(
    story_id: str,
    title: str,
    objective: str,
    acceptance: list[str],
    verification: list[str],
    *,
    kind: str = "implementation",
    depends_on: list[str] | None = None,
    allowed_paths: list[str] | None = None,
    review_required: bool = False,
) -> dict[str, Any]:
    return {
        "id": story_id,
        "title": title,
        "status": "todo",
        "kind": kind,
        "assignee": "current",
        "model_hint": "current",
        "depends_on": depends_on or [],
        "workspace": "current",
        "objective": objective,
        "context": ["prd.md", "docs.md", "decisions.md"],
        "allowed_paths": allowed_paths or [],
        "acceptance": acceptance,
        "verification": verification,
        "review_required": review_required,
        "evidence_required": [],
        "evidence": [],
    }


def validate_story_manifest(manifest: Any) -> list[str]:
    """Validate the V1 story manifest shape and return human-readable errors."""
    errors: list[str] = []
    stories = manifest.get("stories") if isinstance(manifest, dict) else None
    if not isinstance(stories, list):
        return ["stories must be a list"]
    seen_ids: set[str] = set()
    for index, story in enumerate(stories, start=1):
        if not isinstance(story, dict):
            errors.append(f"story {index} must be an object")
            continue
        label = str(story.get("id") or f"story {index}")
        for field in _STORY_REQUIRED_FIELDS:
            if field not in story:
                errors.append(f"{label} missing required field: {field}")
        story_id = story.get("id")
        if isinstance(story_id, str):
            if story_id in seen_ids:
                errors.append(f"{label} duplicate story id")
            seen_ids.add(story_id)
        if "status" in story and str(story.get("status")).lower() not in _STORY_STATUSES:
            errors.append(f"{label} has invalid status: {story.get('status')}")
        if "kind" in story and str(story.get("kind")).lower() not in _STORY_KINDS:
            errors.append(f"{label} has invalid kind: {story.get('kind')}")
        for list_field in ("depends_on", "context", "allowed_paths", "acceptance", "verification"):
            if list_field in story and not isinstance(story.get(list_field), list):
                errors.append(f"{label} field must be a list: {list_field}")
        if "review_required" in story and not isinstance(story.get("review_required"), bool):
            errors.append(f"{label} review_required must be boolean")
    valid_ids = {s.get("id") for s in stories if isinstance(s, dict)}
    for story in stories:
        if not isinstance(story, dict):
            continue
        label = str(story.get("id") or "story")
        for dep in story.get("depends_on") or []:
            if dep not in valid_ids:
                errors.append(f"{label} depends on unknown story: {dep}")
    return errors


def generate_plan_scaffolds(
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> str:
    """Write deterministic V1 `prd.md` and `stories.json` scaffolds for the active loop."""
    resolved = resolve_loop(cwd=cwd, hermes_home=hermes_home)
    if not resolved:
        return "No active loop found. Start one with `/loop start <name or goal>`."
    loop_dir, state = resolved
    name = str(state.get("name") or state.get("slug") or "loop project")
    goal = str(state.get("goal") or name)
    docs_summary = _read_loop_text(
        loop_dir,
        "docs.md",
        "No docs indexed yet. Run `/loop docs add` before treating this PRD as grounded.",
    )
    decisions = _read_loop_text(loop_dir, "decisions.md", "No confirmed decisions captured yet.")
    prd_text = f"""# PRD — {name}

## Outcome
{goal}

## Source context
{docs_summary}

## Decisions
{decisions}

## Users / stakeholders
- Price: needs a visible, resumable project loop with honest status.
- Cody/Hermes: acts as the controller that keeps docs, decisions, PRD, stories, and execution state aligned.

## Non-goals
- Hidden autonomous worker fan-out.
- Kanban, reviewer, or swarm orchestration before foreground story execution is proven.
- Automatic commits, pushes, merges, spending, or irreversible actions.

## Workflows
1. Start or open the loop.
2. Index docs and capture decisions.
3. Review this PRD scaffold and edit it when needed.
4. Execute one small foreground story at a time.
5. Verify before marking a story done.

## Acceptance criteria
- The loop has a readable PRD grounded in available docs and decisions.
- `stories.json` contains small independently executable stories.
- Each story includes acceptance criteria and verification steps.
- `/loop status` and `/view` can explain where the project stands.

## Risks
- Under-specified requirements can create fake progress.
- Prompt-only path boundaries are advisory, not enforcement.
- Background or multi-agent execution can hide drift if added too early.

## Verification requirements
- Run targeted tests for changed loop behavior.
- Run lint/format checks for touched files.
- Record blocked reasons instead of claiming completion when verification is missing.
"""
    stories = [
        _story_template(
            "S1",
            "Review and tighten PRD",
            "Turn `prd.md` from scaffold into the project authority with explicit acceptance criteria and non-goals.",
            [
                "PRD outcome, non-goals, acceptance criteria, risks, and verification gates are explicit.",
                "Open questions are either answered, defaulted with consequences, or listed as blockers.",
            ],
            ["Review `prd.md` against `docs.md` and `decisions.md` for contradictions."],
            kind="docs",
            allowed_paths=["prd.md", "decisions.md"],
        ),
        _story_template(
            "S2",
            "Execute first foreground implementation slice",
            "Implement the smallest high-leverage slice from the PRD in the current session, with no hidden workers.",
            [
                "One story-sized implementation slice is complete or explicitly blocked.",
                "Changed files stay within the story scope.",
                "Verification command output is captured before marking done.",
            ],
            ["Run the targeted test command for the touched files."],
            depends_on=["S1"],
        ),
        _story_template(
            "S3",
            "Verify status and closeout readiness",
            "Prove status/view output reflects PRD and story progress before closeout.",
            [
                "`/loop status` summarizes story counts and next action.",
                "`/view` shows checked-off work, current state, blockers, and best next move.",
                "Closeout blockers or deferred items are visible.",
            ],
            ["Run loop/view tests and lint for touched files."],
            depends_on=["S1", "S2"],
            kind="review",
        ),
    ]
    manifest = {"version": 1, "loop": state.get("slug"), "generated_at": _now_iso(), "stories": stories}
    errors = validate_story_manifest(manifest)
    if errors:
        return "Story manifest validation failed:\n" + "\n".join(f"- {error}" for error in errors)
    (loop_dir / "prd.md").write_text(prd_text, encoding="utf-8")
    _write_json_atomic(loop_dir / "stories.json", manifest)
    state["phase"] = "planning"
    state["status"] = "planning"
    state["next_steps"] = ["Review prd.md and stories.json, then run the first foreground story."]
    save_loop_state(loop_dir, state)
    return f"Wrote PRD scaffold and 3-story manifest to {loop_dir}."


def _load_story_manifest(loop_dir: Path) -> dict[str, Any] | None:
    path = loop_dir / "stories.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(raw, dict) and isinstance(raw.get("stories"), list):
        return raw
    if isinstance(raw, list):
        return {"version": 1, "stories": raw}
    return None


def _find_story(stories: list[dict[str, Any]], story_id: str) -> dict[str, Any] | None:
    target = str(story_id).strip()
    for story in stories:
        if str(story.get("id")) == target:
            return story
    return None


def _select_next_runnable_story(stories: list[dict[str, Any]]) -> dict[str, Any] | None:
    done_ids = {str(story.get("id")) for story in stories if str(story.get("status", "")).lower() in _TERMINAL_STATUSES}
    for story in stories:
        status = str(story.get("status") or "todo").lower()
        if status not in {"todo", "ready"}:
            continue
        deps = [str(dep) for dep in (story.get("depends_on") or [])]
        if all(dep in done_ids for dep in deps):
            return story
    return None


def _bullet_list(items: Any, *, fallback: str = "None specified.") -> str:
    if not isinstance(items, list) or not items:
        return f"- {fallback}"
    return "\n".join(f"- {item}" for item in items)


def build_story_execution_prompt(
    loop_dir: Path,
    state: dict[str, Any],
    story: dict[str, Any],
    *,
    prd: str,
    docs: str,
    decisions: str,
) -> str:
    """Build the foreground story execution prompt for V1."""
    story_id = str(story.get("id") or "story")
    title = str(story.get("title") or story_id)
    return f"""Story execution prompt — {story_id}

Loop: {state.get('name') or state.get('slug')}
Goal: {state.get('goal') or state.get('name')}
Path: {loop_dir}

Story: {story_id} — {title}
Objective: {story.get('objective') or title}
Workspace: {story.get('workspace', 'current')}
Assignee: {story.get('assignee', 'current')}
Model hint: {story.get('model_hint', 'current')}
Allowed paths:
{_bullet_list(story.get('allowed_paths'), fallback='No explicit path boundary; stay within the smallest necessary scope.')}

Acceptance criteria:
{_bullet_list(story.get('acceptance'))}

Verification:
{_bullet_list(story.get('verification'), fallback='State the missing verification and mark the story blocked if it cannot be run.')}

PRD excerpt:
{prd[:3500]}

Docs excerpt:
{docs[:2500]}

Decisions excerpt:
{decisions[:2000]}

Instructions:
- Execute only this story in the foreground/current session.
- Do not launch hidden background workers.
- Do not claim done until verification is run or the story is blocked with a concrete reason.
- If blocked, run `/loop block {story_id} <reason>` with the exact next unblock action.
- On completion, run `/loop complete {story_id} --evidence <path>` referencing real evidence
  (command/test output, a changed-file diff summary, a screenshot, a log excerpt, or a review packet).
- A model assertion alone is not evidence.
"""


def _write_status(loop_dir: Path, state: dict[str, Any], stories: list[dict[str, Any]]) -> None:
    counts = {k: 0 for k in _STATUS_ORDER}
    for story in stories:
        status = str(story.get("status") or "todo").lower()
        counts[status] = counts.get(status, 0) + 1
    running = [story for story in stories if str(story.get("status") or "").lower() == "running"]
    next_story = _select_next_runnable_story(stories)
    lines = [
        f"# Status — {state.get('name') or state.get('slug')}",
        "",
        f"Phase: {state.get('phase', 'planning')}",
        "Stories: " + ", ".join(f"{k}:{v}" for k, v in counts.items() if v),
        "",
        "## Running",
    ]
    if running:
        lines.extend(f"- {story.get('id')} {story.get('title')}" for story in running)
    else:
        lines.append("- None")
    lines.extend(["", "## Next"])
    if next_story:
        lines.append(f"- {next_story.get('id')} {next_story.get('title')}")
    else:
        lines.append("- No runnable story found")
    loop_dir.joinpath("status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_dir(loop_dir: Path, story_id: str) -> Path:
    return loop_dir / "runs" / slugify(str(story_id), fallback="story")


def run_next_story(
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> str:
    """Mark the next runnable story running, persist run artifacts, return its prompt.

    Refusals (deterministic, human-readable):
    - no active loop / no manifest
    - closed loop
    - an existing running story (one story at a time)
    - no runnable story
    """
    resolved = resolve_loop(cwd=cwd, hermes_home=hermes_home)
    if not resolved:
        return "No active loop found. Start one with `/loop start <name or goal>`."
    loop_dir, state = resolved
    if str(state.get("status", "")).lower() == "closed":
        return "This loop is closed. Reopen with `/loop start` or start a new loop before running stories."
    manifest = _load_story_manifest(loop_dir)
    if not manifest:
        return "No story manifest found. Run `/loop plan` first."
    errors = validate_story_manifest(manifest)
    if errors:
        return "Story manifest validation failed:\n" + "\n".join(f"- {error}" for error in errors)
    stories = [story for story in manifest["stories"] if isinstance(story, dict)]
    running = [s for s in stories if str(s.get("status") or "").lower() == "running"]
    if running:
        current = running[0]
        return (
            f"A story is already running: {current.get('id')} {current.get('title')}.\n"
            f"Finish it with `/loop complete {current.get('id')} --evidence <path>` or "
            f"`/loop block {current.get('id')} <reason>` before starting another."
        )
    story = _select_next_runnable_story(stories)
    if not story:
        _write_status(loop_dir, state, stories)
        return "No runnable story found. Complete dependencies, unblock a story, or close out."
    story_id = str(story.get("id"))
    story["status"] = "running"
    story["started_at"] = _now_iso()
    story["attempts"] = int(story.get("attempts") or 0) + 1
    state["phase"] = "executing"
    state["status"] = "executing"
    state["next_steps"] = [f"Finish {story_id}: {story.get('title')}"]
    manifest["updated_at"] = _now_iso()

    prompt = build_story_execution_prompt(
        loop_dir,
        state,
        story,
        prd=_read_loop_text(loop_dir, "prd.md", "No PRD found."),
        docs=_read_loop_text(loop_dir, "docs.md", "No docs indexed."),
        decisions=_read_loop_text(loop_dir, "decisions.md", "No decisions captured."),
    )

    # Durable run artifacts must be written before returning the handoff prompt.
    run_dir = _run_dir(loop_dir, story_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    _write_json_atomic(
        run_dir / "run.json",
        {
            "story_id": story_id,
            "title": story.get("title"),
            "status": "running",
            "attempt": story.get("attempts"),
            "started_at": story.get("started_at"),
            "prompt_path": str((run_dir / "prompt.md")),
            "evidence": [],
            "verdict": None,
        },
    )

    _write_json_atomic(loop_dir / "stories.json", manifest)
    save_loop_state(loop_dir, state)
    _write_status(loop_dir, state, stories)
    return prompt


def _resolve_evidence_path(
    evidence: str,
    loop_dir: Path,
    cwd: str | Path | None,
) -> tuple[Path | None, str | None]:
    """Resolve and bound-check an evidence path.

    Returns ``(resolved_path, None)`` on success or ``(None, reason)`` on
    refusal. Evidence must exist and live inside either the loop directory or
    the workspace (git root / cwd) evidence boundary.
    """
    raw = (evidence or "").strip()
    if not raw:
        return None, "no evidence path provided"
    candidate = Path(raw).expanduser()
    roots = [loop_dir.resolve(), workspace_root(cwd)]
    search: list[Path] = []
    if candidate.is_absolute():
        search.append(candidate)
    else:
        search.extend(root / candidate for root in roots)
        search.append(Path(cwd or Path.cwd()).expanduser() / candidate)
    found: Path | None = None
    for path in search:
        if path.exists():
            found = path.resolve()
            break
    if found is None:
        return None, f"evidence path not found: {raw}"
    if not any(found == root or root in found.parents for root in roots):
        return None, (
            f"evidence path is outside the loop/workspace boundary: {found}"
        )
    return found, None


def complete_story(
    story_id: str,
    evidence: str,
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> str:
    """Complete a story, requiring referenced evidence that exists in-boundary.

    A story with ``review_required: true`` moves to ``needs_review`` rather than
    ``done``; ``/loop review`` then renders the packet.
    """
    resolved = resolve_loop(cwd=cwd, hermes_home=hermes_home)
    if not resolved:
        return "No active loop found. Start one with `/loop start <name or goal>`."
    loop_dir, state = resolved
    if str(state.get("status", "")).lower() == "closed":
        return "This loop is closed. Reopen before completing stories."
    if not str(story_id).strip():
        return "Usage: /loop complete <story-id> --evidence <path>"
    manifest = _load_story_manifest(loop_dir)
    if not manifest:
        return "No story manifest found. Run `/loop plan` first."
    stories = [s for s in manifest["stories"] if isinstance(s, dict)]
    story = _find_story(stories, story_id)
    if story is None:
        return f"Story not found: {story_id}"
    if str(story.get("status") or "").lower() in _TERMINAL_STATUSES:
        return f"Story {story_id} is already {story.get('status')}."

    evidence_path, reason = _resolve_evidence_path(str(evidence), loop_dir, cwd)
    if evidence_path is None:
        return (
            f"Refusing to complete {story_id}: {reason}.\n"
            "Completion requires real evidence (command/test output, changed-file diff summary, "
            "screenshot, log excerpt, or review packet). A model assertion is not evidence."
        )

    try:
        display = str(evidence_path.relative_to(loop_dir.resolve()))
    except ValueError:
        display = str(evidence_path)
    record = {"path": display, "abs_path": str(evidence_path), "recorded_at": _now_iso()}
    story.setdefault("evidence", [])
    if isinstance(story["evidence"], list):
        story["evidence"].append(record)
    else:
        story["evidence"] = [record]

    current_status = str(story.get("status") or "").lower()
    review_required = bool(story.get("review_required"))
    approving_review = current_status == "needs_review" and review_required
    new_status = "done" if approving_review or not review_required else "needs_review"
    story["status"] = new_status
    story["completed_at"] = _now_iso()
    if approving_review:
        story["review_approved_at"] = story["completed_at"]
    manifest["updated_at"] = _now_iso()

    run_dir = _run_dir(loop_dir, story_id)
    if (run_dir / "run.json").exists():
        try:
            run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            run = {"story_id": str(story_id)}
        run["status"] = new_status
        run["completed_at"] = story["completed_at"]
        run["evidence"] = story["evidence"]
        run["verdict"] = "needs_review" if review_required else "pass"
        _write_json_atomic(run_dir / "run.json", run)

    next_story = _select_next_runnable_story(stories)
    if new_status == "needs_review":
        state["next_steps"] = [f"Review {story_id} with `/loop review {story_id}`."]
    elif next_story:
        state["next_steps"] = [f"Run next story {next_story.get('id')}: {next_story.get('title')}"]
    else:
        state["next_steps"] = ["No runnable story remains. Run `/loop review` or `/loop close`."]
    state["status"] = "executing"

    _write_json_atomic(loop_dir / "stories.json", manifest)
    save_loop_state(loop_dir, state)
    _write_status(loop_dir, state, stories)
    if review_required and not approving_review:
        return (
            f"Story {story_id} marked needs_review (evidence: {display}).\n"
            f"Render the packet with `/loop review {story_id}` before it can become done."
        )
    if approving_review:
        return f"Story {story_id} marked done after review approval (evidence: {display})."
    return f"Story {story_id} marked done (evidence: {display})."


def block_story(
    story_id: str,
    reason: str,
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> str:
    """Record an explicit blocker on a story and the loop."""
    resolved = resolve_loop(cwd=cwd, hermes_home=hermes_home)
    if not resolved:
        return "No active loop found. Start one with `/loop start <name or goal>`."
    loop_dir, state = resolved
    if str(state.get("status", "")).lower() == "closed":
        return "This loop is closed. Reopen before blocking stories."
    if not str(story_id).strip() or not str(reason).strip():
        return "Usage: /loop block <story-id> <reason>"
    manifest = _load_story_manifest(loop_dir)
    if not manifest:
        return "No story manifest found. Run `/loop plan` first."
    stories = [s for s in manifest["stories"] if isinstance(s, dict)]
    story = _find_story(stories, story_id)
    if story is None:
        return f"Story not found: {story_id}"

    reason_text = str(reason).strip()
    story["status"] = "blocked"
    story["blocked_reason"] = reason_text
    story["blocked_at"] = _now_iso()
    manifest["updated_at"] = _now_iso()

    blockers = state.get("blockers")
    blocker_line = f"{story_id}: {reason_text}"
    if isinstance(blockers, list):
        if blocker_line not in blockers:
            blockers.append(blocker_line)
    else:
        blockers = [blocker_line]
    state["blockers"] = blockers
    state["next_steps"] = [f"Unblock {story_id}: {reason_text}"]

    run_dir = _run_dir(loop_dir, story_id)
    if (run_dir / "run.json").exists():
        try:
            run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            run = {"story_id": str(story_id)}
        run["status"] = "blocked"
        run["blocked_reason"] = reason_text
        run["blocked_at"] = story["blocked_at"]
        run["verdict"] = "block"
        _write_json_atomic(run_dir / "run.json", run)

    _write_json_atomic(loop_dir / "stories.json", manifest)
    save_loop_state(loop_dir, state)
    _write_status(loop_dir, state, stories)
    return f"Blocked {story_id}: {reason_text}"


def render_review_packet(
    story_id: str | None = None,
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> str:
    """Render a read-only review packet for a story (or all needs_review stories).

    This never mutates `stories.json` or loop state.
    """
    resolved = resolve_loop(cwd=cwd, hermes_home=hermes_home)
    if not resolved:
        return "No active loop found. Start one with `/loop start <name or goal>`."
    loop_dir, state = resolved
    stories = _load_stories(loop_dir)
    if not stories:
        return "No stories to review. Run `/loop plan` first."

    if story_id and str(story_id).strip():
        targets = [s for s in stories if str(s.get("id")) == str(story_id).strip()]
        if not targets:
            return f"Story not found: {story_id}"
    else:
        targets = [s for s in stories if str(s.get("status") or "").lower() == "needs_review"]
        if not targets:
            return "No stories are awaiting review."

    blocks: list[str] = [f"# Review packet — {state.get('name') or state.get('slug')}", ""]
    for story in targets:
        sid = str(story.get("id"))
        evidence = story.get("evidence") if isinstance(story.get("evidence"), list) else []
        evidence_lines = [f"- {e.get('path', e)}" for e in evidence] if evidence else ["- None recorded"]
        run_dir = _run_dir(loop_dir, sid)
        run_note = "present" if (run_dir / "run.json").exists() else "missing"
        blocks.extend(
            [
                f"## {sid} — {story.get('title')}",
                f"Status: {story.get('status')}",
                f"Kind: {story.get('kind', 'unknown')}",
                f"Objective: {story.get('objective', '')}",
                "",
                "Acceptance:",
                _bullet_list(story.get("acceptance")),
                "",
                "Verification:",
                _bullet_list(story.get("verification")),
                "",
                "Evidence:",
                "\n".join(evidence_lines),
                "",
                f"Run artifact: {run_note} ({run_dir})",
                "",
            ]
        )
    blocks.append(
        "This packet is read-only. After human/Cody review, mark approval with "
        "`/loop complete <story-id> --evidence <review-packet-or-artifact>`."
    )
    return "\n".join(blocks)


def close_loop(
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
    force: bool = False,
) -> str:
    """Close the active loop only when story state makes that honest, unless forced."""
    resolved = resolve_loop(cwd=cwd, hermes_home=hermes_home)
    if not resolved:
        return "No active loop found. Start one with `/loop start <name or goal>`."
    loop_dir, state = resolved
    manifest = _load_story_manifest(loop_dir) or {"stories": []}
    stories = [story for story in manifest.get("stories", []) if isinstance(story, dict)]
    runnable = []
    next_story = _select_next_runnable_story(stories)
    if next_story:
        runnable.append(next_story)
    pending_review = [s for s in stories if str(s.get("status") or "").lower() == "needs_review"]
    incomplete = [story for story in stories if str(story.get("status") or "todo").lower() not in _TERMINAL_STATUSES]
    if (runnable or pending_review) and not force:
        items = "\n".join(
            f"- {story.get('id')} {story.get('title')} [{story.get('status', 'todo')}]"
            for story in (runnable + pending_review)
        )
        return (
            "Refusing clean close. Runnable or review-pending stories remain:\n"
            f"{items}\n"
            "Run `/loop run next`, finish review, mark stories blocked, or use `/loop close --force`."
        )
    counts = {k: 0 for k in _STATUS_ORDER}
    for story in stories:
        key = str(story.get("status") or "todo").lower()
        counts[key] = counts.get(key, 0) + 1
    deferred_lines = [f"- {story.get('id')} {story.get('title')} [{story.get('status', 'todo')}]" for story in incomplete]
    closeout = [
        f"# Closeout — {state.get('name') or state.get('slug')}",
        "",
        f"Closed at: {_now_iso()}",
        f"Mode: {'Forced/deferred close' if force and incomplete else 'Clean close'}",
        "",
        "## Acceptance",
        "- Story acceptance criteria are tracked in `stories.json`.",
        "- Done/archive stories are treated as accepted; incomplete stories are listed below.",
        "",
        "## Verification",
        "- Verification commands/results are tracked per story when available.",
        "- Closeout does not imply unrun verification passed.",
        "",
        "## Story summary",
        "- " + ", ".join(f"{k}:{v}" for k, v in counts.items() if v) if stories else "- No stories found.",
        "",
        "## Blocked / deferred",
    ]
    closeout.extend(deferred_lines or ["- None"])
    closeout.extend(["", "## Artifacts", "- prd.md", "- stories.json", "- status.md", "- closeout.md"])
    loop_dir.joinpath("closeout.md").write_text("\n".join(closeout) + "\n", encoding="utf-8")
    state["status"] = "closed"
    state["phase"] = "closed"
    state["closed_at"] = _now_iso()
    state["next_steps"] = ["Loop closed. Reopen or start a new loop if more work appears."]
    save_loop_state(loop_dir, state)
    _write_status(loop_dir, state, stories)
    return f"Closed loop: {state.get('name') or state.get('slug')}\nPath: {loop_dir / 'closeout.md'}"


_USAGE = (
    "Usage: /loop start <name or goal> | /loop status [slug] | /loop docs add [paths...] | "
    "/loop grill | /loop plan | /loop run next | "
    "/loop complete <story-id> --evidence <path> | /loop block <story-id> <reason> | "
    "/loop review [story-id] | /loop close [--force]"
)


def _parse_complete_args(rest: str) -> tuple[str, str]:
    """Parse ``<story-id> --evidence <path>`` (order-tolerant)."""
    try:
        tokens = shlex.split(rest)
    except ValueError:
        tokens = rest.split()
    story_id = ""
    evidence = ""
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ("--evidence", "-e", "--evidence-path"):
            evidence = tokens[i + 1] if i + 1 < len(tokens) else ""
            i += 2
            continue
        if token.startswith("--evidence="):
            evidence = token.split("=", 1)[1]
            i += 1
            continue
        if not story_id:
            story_id = token
        i += 1
    return story_id, evidence


def handle_loop_command(
    command: str,
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> str:
    """Handle the V1 `/loop` subcommands and return display text.

    This is the single narrow waist used by every surface (CLI, gateway, TUI).
    """
    stripped = command.strip()
    # Drop a leading "/loop" (or bare "loop") token, keep the remainder intact.
    body = stripped
    for prefix in ("/loop", "loop"):
        if body == prefix:
            return _USAGE
        if body.lower().startswith(prefix + " "):
            body = body[len(prefix) + 1:].strip()
            break
    if not body:
        return _USAGE

    parts = body.split(None, 1)
    subcommand = parts[0].lower()
    remainder = parts[1].strip() if len(parts) > 1 else ""

    if subcommand == "start":
        if not remainder:
            return "Usage: /loop start <name or goal>"
        result = create_loop(remainder, cwd, hermes_home=hermes_home)
        state = result["state"]
        verb = "Created" if result["created"] else "Opened"
        return (
            f"{verb} loop: {state.get('name')}\n"
            f"Slug: {state.get('slug')}\n"
            f"Path: {result['path']}\n"
            "Next: add docs or run a grill-with-docs planning pass."
        )
    if subcommand == "status":
        return render_loop_status(remainder or None, cwd, hermes_home=hermes_home)
    if subcommand == "docs":
        doc_parts = remainder.split(None, 1)
        if doc_parts and doc_parts[0].lower() == "add":
            doc_paths = doc_parts[1].split() if len(doc_parts) > 1 else None
            return index_docs(doc_paths, cwd, hermes_home=hermes_home)
        return "Usage: /loop docs add [paths...]"
    if subcommand == "grill":
        return build_grill_prompt(cwd, hermes_home=hermes_home)
    if subcommand == "plan":
        return generate_plan_scaffolds(cwd, hermes_home=hermes_home)
    if subcommand == "run":
        run_arg = remainder.split(None, 1)[0].lower() if remainder else ""
        if run_arg == "next":
            return run_next_story(cwd, hermes_home=hermes_home)
        return "Usage: /loop run next"
    if subcommand == "complete":
        story_id, evidence = _parse_complete_args(remainder)
        if not story_id:
            return "Usage: /loop complete <story-id> --evidence <path>"
        return complete_story(story_id, evidence, cwd, hermes_home=hermes_home)
    if subcommand == "block":
        block_parts = remainder.split(None, 1)
        story_id = block_parts[0] if block_parts else ""
        reason = block_parts[1].strip() if len(block_parts) > 1 else ""
        if not story_id or not reason:
            return "Usage: /loop block <story-id> <reason>"
        return block_story(story_id, reason, cwd, hermes_home=hermes_home)
    if subcommand == "review":
        return render_review_packet(remainder or None, cwd, hermes_home=hermes_home)
    if subcommand == "close":
        force = "--force" in remainder.split()
        return close_loop(cwd, hermes_home=hermes_home, force=force)
    return "Unsupported V1 loop command.\n" + _USAGE


def handle_view_command(
    command: str,
    cwd: str | Path | None = None,
    *,
    hermes_home: str | Path | None = None,
) -> str:
    parts = command.strip().split(None, 1)
    project = parts[1].strip() if len(parts) > 1 else None
    return render_global_view(project, cwd, hermes_home=hermes_home)
