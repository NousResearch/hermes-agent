"""Matrix project routing backed by Hermes state_meta."""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# One-time legacy seed. Runtime lookups use the persisted registry exclusively.
_BOOTSTRAP_PROJECTS = {
    "newmoon": {
        "path": "/home/rle/projects/NewMoonNailsAndSpa",
        "metadata": {
            "display_name": "New Moon Nails",
            "aliases": ["new moon", "new moon nails", "nail site", "nails site"],
        },
    },
    "fivehours": {
        "path": "/home/rle/projects/savefivehours",
        "metadata": {
            "display_name": "Five Hours",
            "aliases": ["five hours", "save five hours"],
        },
    },
}
_META_PREFIX = "matrix_project_router:"
_REGISTRY_META_KEY = "matrix_project_router:registry"
_REGISTRY_VERSION = 2
# Relative `!project add` references are resolved only beneath this root.
# Keep this in one place so a future gateway configuration can override it.
DEFAULT_PROJECTS_ROOT = Path("/home/rle/projects")
_PROJECT_MARKERS = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "package.json",
    "tsconfig.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "Gemfile",
    "composer.json",
    "mix.exs",
    "pubspec.yaml",
    "CMakeLists.txt",
    "Makefile",
    "Dockerfile",
)
_CONTEXT_PATHS = (
    ("AGENTS.md", lambda root: (root / "AGENTS.md").is_file()),
    ("README*", lambda root: any(path.is_file() for path in root.glob("README*"))),
    ("CONTRIBUTING.md", lambda root: (root / "CONTRIBUTING.md").is_file()),
    ("package.json", lambda root: (root / "package.json").is_file()),
    ("pyproject.toml", lambda root: (root / "pyproject.toml").is_file()),
    ("Cargo.toml", lambda root: (root / "Cargo.toml").is_file()),
    ("go.mod", lambda root: (root / "go.mod").is_file()),
    ("docs/", lambda root: (root / "docs").is_dir()),
    ("docs/STATUS.md", lambda root: (root / "docs" / "STATUS.md").is_file()),
    ("docs/decisions/", lambda root: (root / "docs" / "decisions").is_dir()),
)


@dataclass(frozen=True)
class RegisteredProject:
    key: str
    path: Path
    context: tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class SetupRecommendation:
    """A proposed repository-context change for a future explicit apply step."""

    action: str
    target: str
    reason: str
    category: str


@dataclass(frozen=True)
class ProjectSetupPlan:
    """Read-only repository-context analysis; it deliberately holds no pending state."""

    key: str
    path: Path
    found: tuple[str, ...]
    recommendations: tuple[SetupRecommendation, ...]
    not_needed: tuple[str, ...]
    authoritative_sources: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ProjectSetupApplyResult:
    """Result of an explicit apply based only on a freshly analyzed plan."""

    key: str
    path: Path
    recommended_count: int
    created: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]
    had_unrelated_changes: bool
    plan: ProjectSetupPlan


def normalize_project_key(value: str) -> str:
    """Return a Matrix-friendly key derived from a directory or project name."""
    return re.sub(r"[^a-z0-9]+", "", (value or "").casefold())


def _normalize_routing_phrase(value: str) -> str:
    """Normalize routing evidence into a boundary-aware, comparable phrase."""
    return re.sub(r"[^a-z0-9]+", " ", (value or "").casefold()).strip()


def _default_display_name(value: str) -> str:
    words = _normalize_routing_phrase(value).split()
    return " ".join(word.capitalize() for word in words)


def _validated_metadata(metadata: object) -> dict:
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("project registry state is invalid")
    result = dict(metadata)
    display_name = result.get("display_name")
    if display_name is not None:
        if not isinstance(display_name, str) or not _normalize_routing_phrase(display_name):
            raise ValueError("project registry state is invalid")
        result["display_name"] = " ".join(display_name.split())
    aliases = result.get("aliases", [])
    if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
        raise ValueError("project registry state is invalid")
    normalized_aliases = sorted({_normalize_routing_phrase(alias) for alias in aliases})
    if any(not alias for alias in normalized_aliases):
        raise ValueError("project registry state is invalid")
    result["aliases"] = normalized_aliases
    return result


def _bootstrap_registry_value() -> dict:
    return {
        "version": _REGISTRY_VERSION,
        "projects": {
            key: {
                "path": project["path"],
                "metadata": {
                    "display_name": project["metadata"]["display_name"],
                    "aliases": list(project["metadata"]["aliases"]),
                },
            }
            for key, project in sorted(_BOOTSTRAP_PROJECTS.items())
        },
    }


def _migrate_registry(registry: dict) -> tuple[dict, bool]:
    if registry.get("version") not in {1, _REGISTRY_VERSION} or not isinstance(
        registry.get("projects"), dict
    ):
        raise ValueError("project registry state is invalid")
    migrated = dict(registry)
    if not all(isinstance(key, str) and isinstance(entry, dict) for key, entry in registry["projects"].items()):
        raise ValueError("project registry state is invalid")
    projects = {key: dict(entry) for key, entry in registry["projects"].items()}
    changed = registry.get("version") != _REGISTRY_VERSION
    for key, entry in projects.items():
        if not isinstance(key, str) or key != normalize_project_key(key):
            raise ValueError("project registry state is invalid")
        if not isinstance(entry.get("path"), str):
            raise ValueError("project registry state is invalid")
        original_metadata = entry.get("metadata")
        raw_metadata = original_metadata if isinstance(original_metadata, dict) else {}
        metadata = _validated_metadata(original_metadata)
        bootstrap = _BOOTSTRAP_PROJECTS.get(key, {}).get("metadata", {})
        for field, value in bootstrap.items():
            if field not in raw_metadata:
                metadata[field] = list(value) if isinstance(value, list) else value
        if "display_name" not in raw_metadata and "display_name" not in bootstrap:
            metadata["display_name"] = _default_display_name(Path(entry["path"]).name) or key
        if original_metadata != metadata:
            changed = True
        entry["metadata"] = metadata
    migrated["version"] = _REGISTRY_VERSION
    migrated["projects"] = projects
    return migrated, changed


def _load_registry(db) -> dict:
    raw = db.get_meta(_REGISTRY_META_KEY)
    if raw is None:
        registry = _bootstrap_registry_value()
        _save_registry(db, registry)
        return registry
    try:
        registry = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("project registry state is invalid") from exc
    registry, migrated = _migrate_registry(registry)
    if migrated:
        _save_registry(db, registry)
    return registry


def _save_registry(db, registry: dict) -> None:
    db.set_meta(_REGISTRY_META_KEY, json.dumps(registry, sort_keys=True))


def bootstrap_registry(db) -> None:
    """Create or migrate the registry without overwriting saved metadata."""
    _load_registry(db)


def _registry_projects(db) -> dict:
    return _load_registry(db)["projects"]


def project_keys(db) -> tuple[str, ...]:
    return tuple(sorted(_registry_projects(db)))


def project_path(db, key: str) -> Path | None:
    entry = _registry_projects(db).get(normalize_project_key(key))
    return Path(entry["path"]) if entry else None


def project_details(db, key: str) -> tuple[str, str, tuple[str, ...], Path]:
    """Return one project's normalized key, display metadata, and canonical path."""
    normalized_key = normalize_project_key(key)
    entry = _registry_projects(db).get(normalized_key)
    if entry is None:
        raise ValueError(
            f"unknown project '{normalized_key}'. Valid projects: {', '.join(project_keys(db))}"
        )
    metadata = entry["metadata"]
    return (
        normalized_key,
        metadata["display_name"],
        tuple(metadata["aliases"]),
        Path(entry["path"]),
    )


def registered_project_details(db) -> tuple[tuple[str, str, tuple[str, ...], Path], ...]:
    """Return every registered project in deterministic key and alias order."""
    return tuple(project_details(db, key) for key in project_keys(db))


def _routing_phrases(key: str, metadata: dict) -> tuple[str, ...]:
    phrases = {_normalize_routing_phrase(key)}
    display_name = metadata.get("display_name")
    if isinstance(display_name, str):
        phrases.add(_normalize_routing_phrase(display_name))
    phrases.update(metadata.get("aliases", []))
    return tuple(sorted(phrase for phrase in phrases if phrase))


def _contains_routing_phrase(text: str, phrase: str) -> bool:
    words = phrase.split()
    pattern = r"(?<!\w)" + r"[^a-z0-9]+".join(map(re.escape, words)) + r"(?!\w)"
    return re.search(pattern, text.casefold()) is not None


def resolve_project_reference(db, text: str) -> tuple[str, ...]:
    """Return every registered project with exact key/name/alias evidence."""
    if not isinstance(text, str) or not text.strip():
        return ()
    return tuple(
        key
        for key, entry in sorted(_registry_projects(db).items())
        if any(
            _contains_routing_phrase(text, phrase)
            for phrase in _routing_phrases(key, entry["metadata"])
        )
    )


def add_project_alias(db, key: str, alias: str) -> str:
    """Persist an alias only when it cannot route to another registered project."""
    normalized_key = normalize_project_key(key)
    normalized_alias = _normalize_routing_phrase(alias)
    if not normalized_alias:
        raise ValueError("project alias must contain at least one ASCII letter or number")
    registry = _load_registry(db)
    projects = registry["projects"]
    entry = projects.get(normalized_key)
    if entry is None:
        raise ValueError(
            f"unknown project '{normalized_key}'. Valid projects: {', '.join(sorted(projects))}"
        )
    for other_key, other_entry in sorted(projects.items()):
        if other_key != normalized_key and normalized_alias in _routing_phrases(
            other_key, other_entry["metadata"]
        ):
            raise ValueError(f"project alias '{normalized_alias}' conflicts with project '{other_key}'")
    aliases = entry["metadata"]["aliases"]
    if normalized_alias not in aliases:
        aliases.append(normalized_alias)
        aliases.sort()
        _save_registry(db, registry)
    return normalized_alias


def remove_project_alias(db, key: str, alias: str) -> str:
    """Remove one normalized alias without changing any other project metadata."""
    normalized_key = normalize_project_key(key)
    normalized_alias = _normalize_routing_phrase(alias)
    registry = _load_registry(db)
    entry = registry["projects"].get(normalized_key)
    if entry is None:
        raise ValueError(
            f"unknown project '{normalized_key}'. Valid projects: {', '.join(project_keys(db))}"
        )
    aliases = entry["metadata"]["aliases"]
    if normalized_alias not in aliases:
        raise ValueError(f"project alias '{normalized_alias}' is not registered for '{normalized_key}'")
    aliases.remove(normalized_alias)
    _save_registry(db, registry)
    return normalized_alias


def inspect_project_context(path: Path) -> tuple[tuple[str, bool], ...]:
    """Inspect bounded, static repository context without executing project code."""
    return tuple((label, present(path)) for label, present in _CONTEXT_PATHS)


def _existing_files(path: Path, pattern: str) -> tuple[Path, ...]:
    return tuple(sorted(candidate for candidate in path.glob(pattern) if candidate.is_file()))


def _has_static_content(path: Path) -> bool:
    """Check whether a text context file has content without executing it."""
    try:
        return bool(path.read_text(encoding="utf-8", errors="replace").strip())
    except OSError:
        return False


def _setup_adr_paths(path: Path) -> tuple[str, ...]:
    """Return established ADR/decision locations in deterministic preference order."""
    candidates = (
        "docs/decisions",
        "docs/adr",
        "docs/adrs",
        "decisions",
        "adr",
        "adrs",
        "docs/ADR.md",
        "ADR.md",
    )
    return tuple(
        candidate + "/" if (path / candidate).is_dir() else candidate
        for candidate in candidates
        if (path / candidate).is_dir() or (path / candidate).is_file()
    )


def analyze_project_setup(db, key: str) -> ProjectSetupPlan:
    """Analyze one registered repository using only static filesystem inspection.

    This function neither executes repository content nor writes repository or
    registry state. The structured recommendations are intentionally suitable
    for a later, separately authorized apply command.
    """
    normalized_key = normalize_project_key(key)
    path = project_path(db, normalized_key)
    if path is None:
        raise ValueError(
            f"unknown project '{normalized_key}'. Valid projects: {', '.join(project_keys(db))}"
        )
    if not path.is_dir():
        raise ValueError(f"configured project path does not exist: {path}")

    agents = path / "AGENTS.md"
    claude = path / "CLAUDE.md"
    contributing = path / "CONTRIBUTING.md"
    docs = path / "docs"
    status = docs / "STATUS.md"
    readmes = _existing_files(path, "README*")
    requirements = _existing_files(path, "requirements*.txt")
    manifests = tuple(
        candidate
        for candidate in (path / "package.json", path / "pyproject.toml", path / "Cargo.toml", path / "go.mod")
        if candidate.is_file()
    ) + requirements
    adr_paths = _setup_adr_paths(path)
    github_context = tuple(
        f".github/{candidate.name}"
        for candidate in _existing_files(path / ".github", "*.md")
        if candidate.name.upper() in {"CONTRIBUTING.MD", "INSTRUCTIONS.MD", "AGENTS.MD"}
    )
    github_agent_context = tuple(
        candidate
        for candidate in _existing_files(path / ".github", "*.md")
        if candidate.name.upper() in {"INSTRUCTIONS.MD", "AGENTS.MD"}
    )

    found: list[str] = []
    for candidate in (agents, *readmes, contributing, claude, *manifests):
        if candidate.is_file():
            found.append(candidate.relative_to(path).as_posix())
    if docs.is_dir():
        found.append("docs/")
    if status.is_file():
        found.append("docs/STATUS.md")
    found.extend(adr_paths)
    found.extend(github_context)

    recommendations: list[SetupRecommendation] = []
    not_needed: list[str] = []
    has_agent_convention = any(
        _has_static_content(candidate) for candidate in (agents, claude, *github_agent_context)
    )
    if has_agent_convention:
        not_needed.append("AGENTS.md — existing repository agent instructions are present")
    else:
        recommendations.append(
            SetupRecommendation(
                action="create",
                target="AGENTS.md",
                reason="minimal repo-specific agent operating context is absent",
                category="recommended",
            )
        )

    if status.is_file():
        not_needed.append("docs/STATUS.md — current-state context already exists")
    elif docs.is_dir():
        recommendations.append(
            SetupRecommendation(
                action="create",
                target="docs/STATUS.md",
                reason="concise current-state snapshot would aid ongoing work",
                category="recommended",
            )
        )
    else:
        not_needed.append("docs/STATUS.md — no documentation convention detected")

    if adr_paths:
        not_needed.append(f"docs/decisions/ — existing ADR convention: {adr_paths[0]}")
    elif docs.is_dir() and (contributing.is_file() or has_agent_convention) and manifests:
        recommendations.append(
            SetupRecommendation(
                action="create",
                target="docs/decisions/",
                reason="durable technical decisions are likely useful for this documented repository",
                category="recommended",
            )
        )
    elif docs.is_dir():
        not_needed.append("docs/decisions/ — no clear durable-decision need detected")
    else:
        not_needed.append("docs/decisions/ — no documentation or ADR convention detected")

    authoritative_sources: list[tuple[str, str]] = []
    for candidate, role in (
        (agents, "repository agent instructions"),
        (readmes[0] if readmes else None, "project overview"),
        (contributing, "contribution conventions"),
        (claude, "repository agent instructions"),
        (status, "current implementation state"),
    ):
        if candidate is not None and candidate.is_file() and _has_static_content(candidate):
            authoritative_sources.append((candidate.relative_to(path).as_posix(), role))

    return ProjectSetupPlan(
        key=normalized_key,
        path=path,
        found=tuple(found),
        recommendations=tuple(recommendations),
        not_needed=tuple(not_needed),
        authoritative_sources=tuple(authoritative_sources),
    )


def render_project_setup_plan(plan: ProjectSetupPlan) -> str:
    """Render a concise, deterministic Matrix-safe representation of a plan."""
    lines = [f"Project setup analysis: {plan.key}", f"Path: {plan.path}"]
    if plan.found:
        lines.extend(["", "Found:", *(f"- {item}" for item in plan.found)])
    if plan.recommendations:
        lines.extend(
            [
                "",
                "Recommended:",
                *(f"- {item.target} — {item.reason}" for item in plan.recommendations),
            ]
        )
    if plan.not_needed:
        lines.extend(["", "Not currently needed:", *(f"- {item}" for item in plan.not_needed)])
    if plan.authoritative_sources:
        lines.extend(
            [
                "",
                "Authoritative sources:",
                *(f"- {source} — {role}" for source, role in plan.authoritative_sources),
            ]
        )
    lines.extend(["", "No repository files were changed."])
    return "\n".join(lines)


def _agents_content(plan: ProjectSetupPlan) -> str:
    """Render compact instructions derived only from the current static plan."""
    sources = [f"- {source} — {role}" for source, role in plan.authoritative_sources]
    manifests = [
        item
        for item in plan.found
        if item in {"package.json", "pyproject.toml", "Cargo.toml", "go.mod"}
        or item.startswith("requirements")
    ]
    lines = ["# Agent Instructions", "", "## Project context"]
    if sources:
        lines.extend(sources)
    else:
        lines.append("- Confirm the repository's authoritative sources before making changes.")
    if manifests:
        lines.extend(["", "## Static project evidence", *(f"- {item}" for item in manifests)])
    lines.extend(
        [
            "",
            "## Working rules",
            "- Read the listed authoritative sources before changing repository behavior.",
            "- Keep changes scoped to the requested work and preserve existing conventions.",
            "",
            "## Validation",
            "- Validation commands must be confirmed from repository documentation or manifests before running them.",
            "",
            "## Safety",
            "- Do not overwrite existing repository context files without explicit approval.",
            "- Do not create branches, commit, or push unless explicitly requested.",
            "",
        ]
    )
    return "\n".join(lines)


def _status_content(plan: ProjectSetupPlan) -> str:
    """Render a static-evidence status note without inferring project state."""
    evidence = [item for item in plan.found if item not in {"docs/", "docs/STATUS.md"}]
    lines = [
        "# Current Status",
        "",
        "This snapshot records only static repository context observed during setup.",
        "Confirm current behavior, priorities, and validation before treating it as project status.",
    ]
    if evidence:
        lines.extend(["", "## Static evidence", *(f"- {item} is present." for item in evidence)])
    lines.append("")
    return "\n".join(lines)


def _decision_readme_content() -> str:
    return (
        "# Decision Records\n\n"
        "Use this directory for durable technical decisions when the repository needs them.\n"
        "No decision records were created automatically.\n"
    )


def _write_new_file(path: Path, content: str) -> None:
    """Atomically create a UTF-8 file without replacing an existing target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


def _recommended_writes(plan: ProjectSetupPlan) -> tuple[tuple[str, Path, str], ...]:
    """Map only known recommended actions to their v1 create-only file targets."""
    writes: list[tuple[str, Path, str]] = []
    for recommendation in plan.recommendations:
        if recommendation.category != "recommended":
            continue
        if recommendation.action != "create":
            raise ValueError(f"unsupported recommended setup action: {recommendation.action}")
        if recommendation.target == "AGENTS.md":
            writes.append(("AGENTS.md", plan.path / "AGENTS.md", _agents_content(plan)))
        elif recommendation.target == "docs/STATUS.md":
            writes.append(("docs/STATUS.md", plan.path / "docs" / "STATUS.md", _status_content(plan)))
        elif recommendation.target == "docs/decisions/":
            writes.append(
                (
                    "docs/decisions/README.md",
                    plan.path / "docs" / "decisions" / "README.md",
                    _decision_readme_content(),
                )
            )
        else:
            raise ValueError(f"unsupported recommended setup target: {recommendation.target}")
    return tuple(writes)


def _has_unrelated_worktree_changes(path: Path) -> bool:
    """Inspect Git status read-only; non-Git directories simply report false."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def apply_project_setup(db, key: str) -> ProjectSetupApplyResult:
    """Apply the current create-only recommendations without relying on saved plans."""
    current_plan = analyze_project_setup(db, key)
    had_unrelated_changes = _has_unrelated_worktree_changes(current_plan.path)
    writes = _recommended_writes(current_plan)
    created: list[str] = []
    skipped: list[tuple[str, str]] = []
    for label, target, content in writes:
        try:
            _write_new_file(target, content)
        except FileExistsError:
            skipped.append((label, "target already exists"))
        else:
            created.append(label)
    post_write_plan = analyze_project_setup(db, key)
    return ProjectSetupApplyResult(
        key=current_plan.key,
        path=current_plan.path,
        recommended_count=len(writes),
        created=tuple(created),
        skipped=tuple(skipped),
        had_unrelated_changes=had_unrelated_changes,
        plan=post_write_plan,
    )


def render_project_setup_apply_result(result: ProjectSetupApplyResult) -> str:
    """Render deterministic, explicit output for an authorized setup apply."""
    if not result.recommended_count:
        return (
            f"Project setup analysis: {result.key}\n\n"
            "No setup changes are currently recommended.\nNothing to apply.\n\n"
            "No repository files were changed."
        )
    lines = [f"Project setup applied: {result.key}", "", "Created:"]
    lines.extend(f"- {target}" for target in result.created) if result.created else lines.append("- none")
    lines.extend(["", "Skipped:"])
    lines.extend(f"- {target} — {reason}" for target, reason in result.skipped)
    if not result.skipped:
        lines.append("- none")
    lines.extend(["", "No existing files were overwritten."])
    if result.had_unrelated_changes:
        lines.append("Unrelated working-tree changes already existed and were preserved.")
    lines.append("Changes remain uncommitted for review.")
    return "\n".join(lines)


def _appears_to_be_project(path: Path) -> bool:
    return (path / ".git").exists() or any((path / marker).is_file() for marker in _PROJECT_MARKERS)


def _resolve_project_path(raw_path: str, projects_root: Path) -> Path:
    """Resolve an absolute path or an exact relative path beneath projects_root."""
    value = (raw_path or "").strip()
    if not value:
        raise ValueError("project path must not be empty")

    candidate = Path(value)
    if candidate.is_absolute():
        path = candidate.resolve()
    else:
        root = Path(projects_root).resolve()
        path = (root / candidate).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"relative project path must remain beneath projects root: {root}"
            ) from exc

    if not path.exists():
        raise ValueError(f"project path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"project path is not a directory: {path}")
    return path


def register_project(
    db, raw_path: str, *, key: str | None = None, projects_root: Path | None = None
) -> RegisteredProject:
    """Validate and persist a project without modifying its repository files.

    Absolute paths retain their existing behavior. Relative paths are exact,
    potentially nested references beneath projects_root (the default projects root
    in production), and cannot escape it after canonicalization.
    """
    path = _resolve_project_path(raw_path, projects_root or DEFAULT_PROJECTS_ROOT)
    if not _appears_to_be_project(path):
        raise ValueError(f"project path does not appear to be a project or repository: {path}")

    normalized_key = normalize_project_key(key if key is not None else path.name)
    if not normalized_key:
        raise ValueError("project key must contain at least one ASCII letter or number")

    registry = _load_registry(db)
    projects = registry["projects"]
    existing = projects.get(normalized_key)
    if existing:
        existing_path = Path(existing["path"])
        if existing_path == path:
            raise ValueError(f"project key '{normalized_key}' is already registered for this path")
        raise ValueError(
            f"project key '{normalized_key}' is already registered for: {existing_path}"
        )
    for existing_key, entry in projects.items():
        if Path(entry["path"]) == path:
            raise ValueError(f"project path is already registered as '{existing_key}'")

    context = inspect_project_context(path)
    projects[normalized_key] = {
        "path": str(path),
        "metadata": {"display_name": _default_display_name(path.name) or normalized_key, "aliases": []},
    }
    _save_registry(db, registry)
    return RegisteredProject(normalized_key, path, context)


def select_project(db, session_key: str, key: str) -> Path:
    normalized_key = normalize_project_key(key)
    path = project_path(db, normalized_key)
    if path is None:
        raise ValueError(
            f"unknown project '{normalized_key}'. Valid projects: {', '.join(project_keys(db))}"
        )
    if not path.is_dir():
        raise ValueError(f"configured project path does not exist: {path}")
    db.set_meta(_META_PREFIX + session_key, normalized_key)
    return path


def clear_project(db, session_key: str) -> None:
    db.delete_meta(_META_PREFIX + session_key)


def active_project(db, session_key: str) -> tuple[str, Path] | None:
    key = normalize_project_key(db.get_meta(_META_PREFIX + session_key) or "")
    if not key:
        return None
    path = project_path(db, key)
    return (key, path) if path and path.is_dir() else None


def active_project_path(db, session_key: str) -> Path | None:
    project = active_project(db, session_key)
    return project[1] if project else None
