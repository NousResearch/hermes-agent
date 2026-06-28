"""Developer authoring helpers for Hermes profile distributions.

This module is intentionally CLI-shaped rather than agent-tool-shaped. It helps
profile authors produce and check installable distribution repositories without
adding anything to the model tool schema.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes_cli.profile_distribution import (
    DEFAULT_DIST_OWNED,
    DistributionError,
    DistributionManifest,
    MANIFEST_FILENAME,
    USER_OWNED_EXCLUDE,
    read_manifest,
    write_manifest,
)

_PLACEHOLDER_RE = re.compile(r"{{[a-zA-Z0-9_]+}}")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_SECRET_PATTERNS = (
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(r"gho_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)
_FORBIDDEN_DIR_NAMES = USER_OWNED_EXCLUDE | {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    "dist",
    "build",
}
_FORBIDDEN_FILE_NAMES = {".coverage", "coverage.xml"}
_BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ico"}


@dataclass
class ValidationResult:
    """Profile distribution validation outcome."""

    root: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def slugify(value: str) -> str:
    """Return a distribution-safe kebab-case slug."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    if not value:
        raise ValueError("name must contain at least one alphanumeric character")
    return value


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise DistributionError("PyYAML is required for profile authoring") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dump_yaml(data: Any) -> str:
    import yaml
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _as_list(value: Any, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else list(default or [])


def _render_soul(display_name: str, description: str, params: dict[str, Any]) -> str:
    principles = _as_list(params.get("principles"), [
        "Use tools when they materially improve correctness.",
        "Keep user data private and never expose secrets.",
        "Verify important claims with evidence.",
    ])
    scope = _as_list(params.get("scope"), [description, "Produce clear, actionable outputs."])
    lines = [
        f"# {display_name}",
        "",
        f"You are {display_name}, a focused Hermes Agent profile.",
        "",
        "## Mission",
        "",
        description,
        "",
        "## First principles",
        "",
    ]
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(principles, 1))
    lines.extend(["", "## Scope", ""])
    lines.extend(f"- {item}" for item in scope)
    lines.extend([
        "",
        "## Safety",
        "",
        "Refuse credential theft, hidden persistence, secret exposure, fabricated facts, and unsafe destructive actions without explicit user approval.",
    ])
    return "\n".join(lines)


def _render_config(params: dict[str, Any]) -> str:
    return _dump_yaml({
        "model": {
            "provider": str(params.get("model_provider") or "openrouter"),
            "default": str(params.get("model_default") or "anthropic/claude-sonnet-4"),
        },
        "toolsets": _as_list(params.get("toolsets"), ["file", "terminal", "skills", "web"]),
        "agent": {"max_turns": int(params.get("max_turns") or 90), "tool_use_enforcement": True},
        "memory": {"memory_enabled": True, "user_profile_enabled": True},
        "security": {"redact_secrets": True},
        "approvals": {"mode": "manual"},
    })


def _render_env_example(params: dict[str, Any]) -> str:
    lines = ["# Copy this file to .env and fill in real values.", "# Never commit .env.", ""]
    env_requires = params.get("env_requires") or []
    if not env_requires:
        lines.append("# Add profile-specific environment variables here when needed.")
    for item in env_requires:
        if not isinstance(item, dict) or not item.get("name"):
            raise ValueError("each env_requires item must be a mapping with name")
        required = bool(item.get("required", True))
        lines.append(f"# {item.get('description') or 'Required by this profile'}")
        lines.append(f"# {'required' if required else 'optional'}")
        prefix = "" if required else "# "
        lines.append(f"{prefix}{item['name']}={item.get('default') or ''}")
        lines.append("")
    return "\n".join(lines)


def _render_readme(slug: str, display_name: str, description: str) -> str:
    return f"""# {display_name}

{description}

This is a Hermes Agent profile distribution. It can be installed with `hermes profile install` and updated from git.

## Install

```bash
hermes profile install github.com/YOUR_ORG/{slug} --alias
hermes -p {slug} chat
```

## Local authoring checks

```bash
hermes profile validate .
hermes profile install . --name {slug}-local --yes
```

## Safety

Do not commit `.env`, credentials, memories, sessions, logs, runtime databases, or user data.
"""


def scaffold_distribution(
    output: Path,
    *,
    name: str | None = None,
    description: str | None = None,
    display_name: str | None = None,
    author: str = "Hermes profile author",
    params_file: Path | None = None,
    force: bool = False,
) -> Path:
    """Create a starter profile distribution repository."""
    params: dict[str, Any] = {}
    if params_file is not None:
        loaded = _load_yaml(params_file)
        if not isinstance(loaded, dict):
            raise ValueError("params file must contain a YAML mapping")
        params.update(loaded)

    raw_name = name or params.get("name")
    if not raw_name:
        raise ValueError("profile scaffold requires a name or params.name")
    slug = slugify(str(raw_name))
    desc = str(description or params.get("description") or "").strip()
    if not desc:
        raise ValueError("profile scaffold requires --description or params.description")
    pretty = display_name or str(params.get("display_name") or slug.replace("-", " ").title())

    if output.exists():
        if not force:
            raise FileExistsError(f"output exists. Pass --force to overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    manifest = DistributionManifest(
        name=slug,
        version=str(params.get("version") or "0.1.0"),
        description=desc,
        hermes_requires=str(params.get("hermes_requires") or ">=0.12.0"),
        author=str(params.get("author") or author),
        license=str(params.get("license") or "MIT"),
        distribution_owned=list(DEFAULT_DIST_OWNED) + ["README.md", "AGENTS.md", "CONTRIBUTING.md", "SECURITY.md", ".env.EXAMPLE"],
    )
    env_requires = params.get("env_requires") or []
    if env_requires:
        from hermes_cli.profile_distribution import EnvRequirement
        manifest.env_requires = [EnvRequirement.from_dict(item) for item in env_requires]

    write_manifest(output, manifest)
    _write(output / "SOUL.md", _render_soul(pretty, desc, params))
    _write(output / "config.yaml", _render_config(params))
    _write(output / "mcp.json", json.dumps({"mcpServers": {}}, indent=2))
    _write(output / ".env.EXAMPLE", _render_env_example(params))
    _write(output / "README.md", _render_readme(slug, pretty, desc))
    _write(output / "AGENTS.md", "# Agent Instructions\n\nRun `hermes profile validate .` after substantive edits. Never commit secrets.")
    _write(output / "CONTRIBUTING.md", "# Contributing\n\nRun `hermes profile validate .` and a local install smoke test before publishing changes.")
    _write(output / "SECURITY.md", "# Security\n\nNever commit `.env`, `auth.json`, sessions, memories, logs, or runtime databases.")
    _write(output / ".gitignore", ".env\nauth.json\nstate.db*\nsessions/\nmemories/\nlogs/\nworkspace/\nplans/\nlocal/\ncache/\n__pycache__/\n*.pyc\n.DS_Store\n")
    _write(output / "skills" / "README.md", "# Bundled skills\n\nAdd profile-specific skills here. Each skill should live in a directory containing `SKILL.md`.")
    _write(output / "cron" / "README.md", "# Cron jobs\n\nAdd distribution-owned scheduled job definitions here when this profile needs them.")
    _write(output / "templates" / "profile.params.yaml", _dump_yaml({
        "name": slug,
        "display_name": pretty,
        "description": desc,
        "author": str(params.get("author") or author),
        "version": manifest.version,
        "license": manifest.license,
        "model_provider": params.get("model_provider") or "openrouter",
        "model_default": params.get("model_default") or "anthropic/claude-sonnet-4",
        "toolsets": _as_list(params.get("toolsets"), ["file", "terminal", "skills", "web"]),
        "env_requires": env_requires,
    }))
    return output


def _iter_validation_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts]


def validate_distribution(root: Path) -> ValidationResult:
    """Validate a profile distribution authoring tree."""
    root = root.resolve()
    result = ValidationResult(root=root)
    if not root.exists():
        result.errors.append(f"path does not exist: {root}")
        return result
    if not root.is_dir():
        result.errors.append(f"path is not a directory: {root}")
        return result

    try:
        raw_manifest = (
            _load_yaml(root / MANIFEST_FILENAME)
            if (root / MANIFEST_FILENAME).exists()
            else None
        )
        manifest = read_manifest(root)
    except Exception as exc:
        result.errors.append(f"invalid {MANIFEST_FILENAME}: {exc}")
        raw_manifest = None
        manifest = None
    if manifest is None:
        result.errors.append(f"missing required file: {MANIFEST_FILENAME}")
    else:
        if isinstance(raw_manifest, dict):
            for key in ("name", "version", "description"):
                if not str(raw_manifest.get(key) or "").strip():
                    result.errors.append(f"distribution.yaml missing required field: {key}")
        if not _NAME_RE.fullmatch(manifest.name):
            result.errors.append("distribution.yaml name must be lowercase kebab case")
        if not manifest.version.strip():
            result.errors.append("distribution.yaml missing required field: version")
        if not manifest.description.strip():
            result.errors.append("distribution.yaml missing required field: description")
        for rel in manifest.owned_paths():
            if not (root / rel.rstrip("/")).exists():
                result.errors.append(f"distribution_owned path does not exist: {rel}")
        env_example = (root / ".env.EXAMPLE").read_text(encoding="utf-8") if (root / ".env.EXAMPLE").exists() else ""
        for req in manifest.env_requires:
            if req.name not in env_example:
                result.errors.append(f"env var {req.name} is declared but missing from .env.EXAMPLE")

    for required in ("SOUL.md", "README.md", "config.yaml", ".env.EXAMPLE"):
        if not (root / required).is_file():
            result.errors.append(f"missing recommended distribution file: {required}")

    files = _iter_validation_files(root)
    for path in files:
        rel = path.relative_to(root)
        for parent in rel.parents:
            if str(parent) != "." and parent.name in _FORBIDDEN_DIR_NAMES:
                result.errors.append(f"runtime or cache directory must not be committed: {parent}")
        if path.name in _FORBIDDEN_FILE_NAMES or path.name in USER_OWNED_EXCLUDE:
            result.errors.append(f"user-owned runtime file must not be committed: {rel}")
        if path.suffix.lower() in {".json"}:
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                result.errors.append(f"invalid JSON in {rel}: {exc}")
        if path.suffix.lower() in {".yaml", ".yml"} or path.name == MANIFEST_FILENAME:
            try:
                _load_yaml(path)
            except Exception as exc:
                result.errors.append(f"invalid YAML in {rel}: {exc}")
        if path.suffix.lower() in _BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "templates" not in rel.parts and _PLACEHOLDER_RE.search(text):
            result.errors.append(f"unresolved template placeholder in {rel}")
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                result.errors.append(f"possible secret pattern in {rel}")

    skills_dir = root / "skills"
    if skills_dir.is_dir():
        for skill_md in skills_dir.rglob("SKILL.md"):
            rel = skill_md.relative_to(root)
            text = skill_md.read_text(encoding="utf-8")
            if not text.startswith("---\n"):
                result.errors.append(f"skill missing YAML frontmatter: {rel}")
                continue
            parts = text.split("---", 2)
            if len(parts) < 3:
                result.errors.append(f"skill frontmatter not closed: {rel}")
                continue
            try:
                import yaml
                meta = yaml.safe_load(parts[1]) or {}
            except Exception as exc:
                result.errors.append(f"invalid skill frontmatter in {rel}: {exc}")
                continue
            if not meta.get("name"):
                result.errors.append(f"skill {rel} missing frontmatter field: name")
            if not meta.get("description"):
                result.errors.append(f"skill {rel} missing frontmatter field: description")
    return result
