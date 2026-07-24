"""Native agent registry for Hermes.

Discovers and validates reusable named agent definitions from:
- Global: $HERMES_HOME/agents/*.md and $HERMES_HOME/agents/**/AGENT.md
- Project: <project>/.hermes/agents/*.md and <project>/.hermes/agents/**/AGENT.md

Project-local agents override global agents with the same name.
"""

from __future__ import annotations

import logging
import os
import re
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Public exceptions
# ─────────────────────────────────────────────────────────────────────────────


class AgentLoadError(ValueError):
    """Raised when an agent file fails validation."""

    pass


# ─────────────────────────────────────────────────────────────────────────────
# Secret-like field detection
# ─────────────────────────────────────────────────────────────────────────────

SECRET_LIKE_FIELD_NAMES: typing.Final[typing.Set[str]] = {
    "api_key",
    "token",
    "secret",
    "password",
    "api_key_base",
    "access_token",
    "refresh_token",
    "client_secret",
    "private_key",
}

_SECRET_LIKE_PATTERN = re.compile(
    r"^(.*[_-])?(api_key|token|secret|password|key|cred|credential)"
    r"|(api_key|token|secret|password|key|cred|credential)[_-]?.*$",
    re.IGNORECASE,
)


def _is_secret_like_field(name: str) -> bool:
    """Return True if the field name looks like it contains a secret value."""
    name_lower = name.lower()
    if name_lower in SECRET_LIKE_FIELD_NAMES:
        return True
    if _SECRET_LIKE_PATTERN.match(name_lower):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses — internal normalized types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AgentRouting:
    mode: Literal["inherit", "hermes", "acp"] = "inherit"
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_mode: str | None = None
    reasoning_effort: str | None = None
    acp_command: str | None = None
    acp_args: list[str] | None = None
    # runner is deferred to a future PR; reserve field for forward-compat
    runner_mode: str | None = None
    runner_name: str | None = None
    runner_continue: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentRouting:
        if data is None:
            return cls()
        return cls(
            mode=data.get("mode", "inherit"),
            provider=data.get("provider"),
            model=data.get("model"),
            base_url=data.get("base_url"),
            api_mode=data.get("api_mode"),
            reasoning_effort=data.get("reasoning_effort"),
            acp_command=data.get("acp_command"),
            acp_args=data.get("acp_args"),
            runner_mode=data.get("runner", {}).get("mode") if isinstance(data.get("runner"), dict) else None,
            runner_name=data.get("runner", {}).get("name") if isinstance(data.get("runner"), dict) else None,
            runner_continue=data.get("runner", {}).get("continue") if isinstance(data.get("runner"), dict) else None,
        )


@dataclass(frozen=True)
class AgentTools:
    mode: Literal["inherit", "restrict", "none"] = "inherit"
    allow_toolsets: list[str] | None = None
    deny_toolsets: list[str] | None = None
    allow_tools: list[str] | None = None
    deny_tools: list[str] | None = None
    inherit_mcp_toolsets: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AgentTools:
        if data is None:
            return cls()
        return cls(
            mode=data.get("mode", "inherit"),
            allow_toolsets=data.get("allow_toolsets"),
            deny_toolsets=data.get("deny_toolsets"),
            allow_tools=data.get("allow_tools"),
            deny_tools=data.get("deny_tools"),
            inherit_mcp_toolsets=data.get("inherit_mcp_toolsets", True),
        )


@dataclass(frozen=True)
class AgentSkills:
    preload: tuple[str, ...] = ()
    required: bool = False
    missing: Literal["warn", "error", "ignore"] = "warn"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AgentSkills:
        if data is None:
            return cls()
        preload_raw = data.get("preload", [])
        preload: tuple[str, ...] = tuple(preload_raw) if isinstance(preload_raw, list) else ()
        return cls(
            preload=preload,
            required=bool(data.get("required", False)),
            missing=data.get("missing", "warn"),
        )


@dataclass(frozen=True)
class AgentLimits:
    max_iterations: int | None = None
    timeout_seconds: int | None = None
    max_context_chars: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AgentLimits:
        if data is None:
            return cls()
        return cls(
            max_iterations=data.get("max_iterations"),
            timeout_seconds=data.get("timeout_seconds"),
            max_context_chars=data.get("max_context_chars"),
        )


@dataclass(frozen=True)
class AgentSecurity:
    require_approval: bool = False
    allow_side_effects: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AgentSecurity:
        if data is None:
            return cls()
        return cls(
            require_approval=bool(data.get("require_approval", False)),
            allow_side_effects=bool(data.get("allow_side_effects", False)),
        )


@dataclass(frozen=True)
class AgentCompatibility:
    platforms: tuple[str, ...] = ()
    min_hermes_version: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AgentCompatibility:
        if data is None:
            return cls()
        platforms_raw = data.get("platforms", [])
        platforms: tuple[str, ...] = tuple(platforms_raw) if isinstance(platforms_raw, list) else ()
        return cls(
            platforms=platforms,
            min_hermes_version=data.get("min_hermes_version"),
        )


@dataclass(frozen=True)
class AgentDefinition:
    """Normalized internal representation of a named agent definition."""

    schema_version: int
    name: str
    display_name: str
    description: str
    enabled: bool
    source: Literal["global", "project"]
    path: Path
    prompt: str

    tags: list[str] = field(default_factory=list)
    routing: AgentRouting = field(default_factory=AgentRouting)
    tools: AgentTools = field(default_factory=AgentTools)
    skills: AgentSkills = field(default_factory=AgentSkills)
    limits: AgentLimits = field(default_factory=AgentLimits)
    delegation_role: Literal["leaf", "orchestrator"] | None = None
    security: AgentSecurity = field(default_factory=AgentSecurity)
    compatibility: AgentCompatibility = field(default_factory=AgentCompatibility)
    extensions: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    # Metadata
    warnings: tuple[str, ...] = field(default_factory=tuple)
    shadowed: bool = False  # True when this is a lower-priority definition hidden by a higher-priority one

    def to_dict(self) -> dict[str, Any]:
        """Full dict representation including prompt."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "enabled": self.enabled,
            "source": self.source,
            "path": str(self.path),
            "prompt": self.prompt,
            "tags": list(self.tags),
            "routing": {
                "mode": self.routing.mode,
                "provider": self.routing.provider,
                "model": self.routing.model,
                "base_url": self.routing.base_url,
                "api_mode": self.routing.api_mode,
                "reasoning_effort": self.routing.reasoning_effort,
                "acp_command": self.routing.acp_command,
                "acp_args": self.routing.acp_args,
            },
            "tools": {
                "mode": self.tools.mode,
                "allow_toolsets": self.tools.allow_toolsets,
                "deny_toolsets": self.tools.deny_toolsets,
                "allow_tools": self.tools.allow_tools,
                "deny_tools": self.tools.deny_tools,
                "inherit_mcp_toolsets": self.tools.inherit_mcp_toolsets,
            },
            "skills": {
                "preload": list(self.skills.preload),
                "required": self.skills.required,
                "missing": self.skills.missing,
            },
            "limits": {
                "max_iterations": self.limits.max_iterations,
                "timeout_seconds": self.limits.timeout_seconds,
                "max_context_chars": self.limits.max_context_chars,
            },
            "delegation_role": self.delegation_role,
            "security": {
                "require_approval": self.security.require_approval,
                "allow_side_effects": self.security.allow_side_effects,
            },
            "compatibility": {
                "platforms": list(self.compatibility.platforms),
                "min_hermes_version": self.compatibility.min_hermes_version,
            },
            "extensions": dict(self.extensions),
            "warnings": list(self.warnings),
            "shadowed": self.shadowed,
        }

    def list_summary(self) -> dict[str, Any]:
        """Compact metadata suitable for agents_list output — no full prompt."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "tags": list(self.tags),
            "source": self.source,
            "path": str(self.path),
            "enabled": self.enabled,
            "routing": {
                "mode": self.routing.mode,
                "model": self.routing.model,
                "provider": self.routing.provider,
            },
            "toolsets": list(self.tools.allow_toolsets) if self.tools.allow_toolsets else None,
            "role": self.delegation_role,
            "shadowed": self.shadowed,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class AgentDir:
    """A discovered agent directory with its source classification."""
    path: Path
    source: Literal["global", "project"]


# ─────────────────────────────────────────────────────────────────────────────
# Name validation
# ─────────────────────────────────────────────────────────────────────────────

_NAME_RE = re.compile(r"^[a-z][a-z0-9\-]{0,63}$")

# Characters that are never allowed in agent names
_FORBIDDEN_NAME_CHARS = frozenset("/\\.. \t\n\r")


def validate_agent_name(name: str) -> str:
    """Validate an agent name and return it (normalized) or raise AgentLoadError.

    Rules:
    - Must match ^[a-z][a-z0-9-]{0,63}$ (1-64 chars, lowercase alphanumeric + hyphens, must start with a letter)
    - Cannot contain path separators, double-dots, spaces, or control chars
    - Cannot be empty or start with a dash
    """
    if not name:
        raise AgentLoadError("Agent name cannot be empty.")

    if len(name) > 64:
        raise AgentLoadError(f"Agent name '{name}' exceeds maximum length of 64 characters.")

    # Check for forbidden characters
    for char in _FORBIDDEN_NAME_CHARS:
        if char in name:
            raise AgentLoadError(
                f"Agent name '{name}' contains forbidden character '{char}'."
            )

    # Reject names starting with dash
    if name.startswith("-"):
        raise AgentLoadError(f"Agent name '{name}' cannot start with a dash.")

    # Reject hidden names (starting with dot)
    if name.startswith("."):
        raise AgentLoadError(f"Agent name '{name}' cannot start with a dot (hidden).")

    if not _NAME_RE.match(name):
        raise AgentLoadError(
            f"Agent name '{name}' must be lowercase, start with a letter, "
            "and may contain only letters, numbers, and hyphens (1-64 chars)."
        )

    return name


# ─────────────────────────────────────────────────────────────────────────────
# Project root discovery
# ─────────────────────────────────────────────────────────────────────────────

def _find_project_root(start: Path) -> Path | None:
    """Walk upward from start and return the first directory containing .git or .hermes.

    Returns None if no project root is found (filesystem root boundary).
    Does not treat HERMES_HOME itself as a project root to avoid double-scanning.
    """
    hermes_home = get_hermes_home()
    current: Path | None = start.resolve()

    while current is not None:
        # Stop at filesystem root
        if current.parent == current:
            return None

        # Do not treat HERMES_HOME as a project root
        if current == hermes_home:
            return None

        if (current / ".git").is_dir() or (current / ".hermes").is_dir():
            return current

        current = current.parent

    return None


def discover_agent_dirs(workdir: str | None = None) -> list[AgentDir]:
    """Discover all agent directories.

    Global: $HERMES_HOME/agents/
    Project: <project>/.hermes/agents/ (when workdir is inside a project)

    Returns a list of AgentDir objects. Global dirs are always included;
    project dirs are included when workdir resolves inside a project.
    """
    dirs: list[AgentDir] = []
    hermes_home = get_hermes_home()

    # Global agents directory — always include the path if HERMES_HOME itself exists
    # (the path may not exist yet; callers handle missing dirs gracefully)
    if hermes_home.exists():
        dirs.append(AgentDir(path=hermes_home / "agents", source="global"))

    # Project-local agents directory
    if workdir:
        project_root = _find_project_root(Path(workdir).resolve())
        if project_root is not None:
            project_agents = project_root / ".hermes" / "agents"
            dirs.append(AgentDir(path=project_agents, source="project"))

    return dirs


# ─────────────────────────────────────────────────────────────────────────────
# Frontmatter and body parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_yaml_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Returns (frontmatter_dict, body_str).
    Uses the same safe-loading approach as agent.skill_utils.parse_frontmatter.
    """
    import yaml

    # Import from agent.skill_utils if available (lazy to avoid circular deps)
    try:
        from agent.skill_utils import parse_frontmatter as skill_parse
        return skill_parse(content)
    except ImportError:
        pass

    # Inline fallback — same logic as skill_utils
    frontmatter: dict[str, Any] = {}
    body = content

    if not content.startswith("---"):
        return frontmatter, body

    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return frontmatter, body

    yaml_content = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3:]

    try:
        loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader
        parsed = yaml.load(yaml_content, Loader=loader)
        if isinstance(parsed, dict):
            frontmatter = parsed
    except Exception:
        # Fallback: simple key:value parsing
        for line in yaml_content.strip().split("\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()

    return frontmatter, body


# ─────────────────────────────────────────────────────────────────────────────
# Agent file loading and validation
# ─────────────────────────────────────────────────────────────────────────────

_MAX_FILE_SIZE = 128 * 1024  # 128 KiB


def load_agent_file(path: Path, source: Literal["global", "project"]) -> AgentDefinition:
    """Load and validate a single agent .md file.

    Returns a normalized AgentDefinition or raises AgentLoadError.
    """
    # Basic file checks
    if not path.is_file():
        raise AgentLoadError(f"Agent file '{path}' is not a regular file.")

    # Skip symlinks
    if path.is_symlink():
        raise AgentLoadError(f"Agent file '{path}' is a symlink — skipping.")

    # Bound file size
    try:
        size = path.stat().st_size
        if size > _MAX_FILE_SIZE:
            raise AgentLoadError(
                f"Agent file '{path.name}' exceeds maximum size: {size} bytes (max 128 KiB)."
            )
    except OSError as e:
        raise AgentLoadError(f"Cannot stat agent file '{path}': {e}")

    # Read content
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise AgentLoadError(f"Agent file '{path}' must be UTF-8 encoded: {e}.")
    except OSError as e:
        raise AgentLoadError(f"Cannot read agent file '{path}': {e}.")

    # Parse frontmatter
    try:
        frontmatter, body = _parse_yaml_frontmatter(content)
    except Exception as e:
        raise AgentLoadError(f"Failed to parse frontmatter in '{path}': {e}")

    if not isinstance(frontmatter, dict):
        raise AgentLoadError(
            f"Frontmatter in '{path.name}' must be a YAML dict, got {type(frontmatter).__name__}."
        )

    # Check for secret-like fields anywhere in frontmatter
    _check_no_secret_fields(frontmatter, path.name)

    # Validate required fields
    schema_version = frontmatter.get("schema_version")
    if schema_version is None:
        raise AgentLoadError(f"Missing required field 'schema_version' in '{path.name}'.")
    if not isinstance(schema_version, int):
        raise AgentLoadError(
            f"'schema_version' in '{path.name}' must be an integer, got {type(schema_version).__name__}."
        )
    if schema_version != 1:
        raise AgentLoadError(
            f"Unsupported schema_version {schema_version} in '{path.name}' — only schema_version: 1 is supported."
        )

    name = frontmatter.get("name")
    if name is None:
        raise AgentLoadError(f"Missing required field 'name' in '{path.name}'.")
    if not isinstance(name, str):
        raise AgentLoadError(f"'name' in '{path.name}' must be a string.")
    name = validate_agent_name(name)

    description = frontmatter.get("description")
    if description is None:
        raise AgentLoadError(f"Missing required field 'description' in '{path.name}'.")
    if not isinstance(description, str):
        raise AgentLoadError(f"'description' in '{path.name}' must be a string.")
    if not (1 <= len(description) <= 500):
        raise AgentLoadError(
            f"'description' in '{path.name}' must be 1-500 characters, got {len(description)}."
        )

    # Validate prompt body
    prompt = body.strip() if body else ""
    if not prompt:
        raise AgentLoadError(f"Agent '{name}' has an empty prompt body.")

    # Collect warnings for unknown fields
    warnings: list[str] = []
    known_top_fields = {
        "schema_version", "name", "display_name", "description", "enabled",
        "tags", "routing", "tools", "skills", "limits",
        "delegation", "security", "compatibility", "extensions",
    }
    for key in frontmatter:
        if key not in known_top_fields and not key.startswith("x-"):
            warnings.append(f"Unknown top-level field '{key}' — ignoring.")

    # Normalize optional fields
    enabled = bool(frontmatter.get("enabled", True))
    display_name = frontmatter.get("display_name") or name

    tags_raw = frontmatter.get("tags")
    tags: list[str] = []
    if isinstance(tags_raw, list):
        tags = [str(t) for t in tags_raw]

    routing = AgentRouting.from_dict(frontmatter.get("routing"))
    # Warn about unknown routing fields (non-secret only)
    routing_raw = frontmatter.get("routing")
    if isinstance(routing_raw, dict):
        known_routing_fields = {
            "mode", "provider", "model", "base_url", "api_mode",
            "reasoning_effort", "acp_command", "acp_args", "runner",
        }
        for key in routing_raw:
            if key not in known_routing_fields and not key.startswith("x-"):
                if not _is_secret_like_field(key):
                    warnings.append(f"Unknown routing field '{key}' — ignoring.")
    tools = AgentTools.from_dict(frontmatter.get("tools"))
    skills = AgentSkills.from_dict(frontmatter.get("skills"))
    limits = AgentLimits.from_dict(frontmatter.get("limits"))

    delegation_raw = frontmatter.get("delegation")
    delegation_role: Literal["leaf", "orchestrator"] | None = None
    if isinstance(delegation_raw, dict):
        delegation_role = delegation_raw.get("role")
        if delegation_role not in (None, "leaf", "orchestrator"):
            warnings.append(f"Unknown delegation.role '{delegation_role}' — ignoring.")
            delegation_role = None

    # Reject project-local ACP commands unconditionally (not gated on delegation block)
    if source == "project" and routing.acp_command:
        warnings.append("Project-local agents cannot set routing.acp_command — ignoring.")
        routing = AgentRouting(
            mode=routing.mode,
            provider=routing.provider,
            model=routing.model,
            base_url=routing.base_url,
            api_mode=routing.api_mode,
            reasoning_effort=routing.reasoning_effort,
            acp_command=None,
            acp_args=None,
        )

    security = AgentSecurity.from_dict(frontmatter.get("security"))
    # Warn about unknown security fields (non-secret only)
    security_raw = frontmatter.get("security")
    if isinstance(security_raw, dict):
        known_security_fields = {"require_approval", "allow_side_effects"}
        for key in security_raw:
            if key not in known_security_fields and not key.startswith("x-"):
                if not _is_secret_like_field(key):
                    warnings.append(f"Unknown security field '{key}' — ignoring.")
    compatibility = AgentCompatibility.from_dict(frontmatter.get("compatibility"))

    extensions_raw = frontmatter.get("extensions")
    extensions: tuple[tuple[str, Any], ...] = ()
    if isinstance(extensions_raw, dict):
        extensions = tuple(extensions_raw.items())

    return AgentDefinition(
        schema_version=schema_version,
        name=name,
        display_name=display_name,
        description=description,
        enabled=enabled,
        source=source,
        path=path,
        prompt=prompt,
        tags=tags,
        routing=routing,
        tools=tools,
        skills=skills,
        limits=limits,
        delegation_role=delegation_role,
        security=security,
        compatibility=compatibility,
        extensions=extensions,
        warnings=tuple(warnings),
    )


def _check_no_secret_fields(data: dict[str, Any], path_name: str) -> None:
    """Recursively check dict for secret-like field names and raise if found."""
    for key, value in data.items():
        if _is_secret_like_field(key):
            raise AgentLoadError(
                f"Agent '{path_name}' contains a secret-like field name '{key}' — rejecting load."
            )
        if isinstance(value, dict):
            _check_no_secret_fields(value, path_name)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _check_no_secret_fields(item, path_name)


# ─────────────────────────────────────────────────────────────────────────────
# Agent discovery and listing
# ─────────────────────────────────────────────────────────────────────────────

# Patterns for finding agent files
_AGENT_FILE_PATTERNS: typing.Final[list[str]] = [
    "*.md",
    "*/AGENT.md",
    "**/*.md",
]


def _iter_agent_files(dir_path: Path) -> typing.Iterator[Path]:
    """Walk a directory tree and yield candidate agent file paths.

    Skips symlinks and deduplicates — files matching both *.md and **/*.md
    are yielded only once.
    """
    seen: set[Path] = set()

    for pattern in _AGENT_FILE_PATTERNS:
        for match in dir_path.glob(pattern):
            if match in seen:
                continue
            if match.is_symlink():
                continue
            if not match.is_file():
                continue
            seen.add(match)
            yield match


def list_agents(
    workdir: str | None = None,
    include_disabled: bool = False,
    include_shadowed: bool = False,
) -> list[AgentDefinition]:
    """List all discovered agents with optional filtering.

    Project-local agents shadow global agents with the same name.
    By default, shadowed lower-priority entries are hidden and disabled agents are hidden.
    """
    dirs = discover_agent_dirs(workdir=workdir)

    # Collect all agents by name, tracking precedence
    # Higher source priority = later in list (we process global then project)
    agents_by_name: dict[str, list[AgentDefinition]] = {}

    for dir_info in dirs:
        try:
            for file_path in _iter_agent_files(dir_info.path):
                try:
                    agent = load_agent_file(file_path, dir_info.source)
                except AgentLoadError as e:
                    logger.warning("Skipping agent file '%s': %s", file_path, e)
                    continue

                name = agent.name
                if name not in agents_by_name:
                    agents_by_name[name] = []

                agents_by_name[name].append(agent)

        except Exception as e:
            logger.warning("Error scanning agent directory '%s': %s", dir_info.path, e)

    # Build result list
    result: list[AgentDefinition] = []
    for name, agent_list in agents_by_name.items():
        if len(agent_list) == 1:
            agent = agent_list[0]
            if not agent.enabled and not include_disabled:
                continue
            result.append(agent)
        else:
            # Multiple entries — determine precedence: project > global
            # Sort so project comes last (we want effective = last appended)
            sorted_agents = sorted(agent_list, key=lambda a: 0 if a.source == "global" else 1)

            # Mark all but the effective as shadowed
            effective = sorted_agents[-1]
            if not effective.enabled and not include_disabled:
                continue

            for agent in sorted_agents[:-1]:
                shadowed_agent = AgentDefinition(
                    schema_version=agent.schema_version,
                    name=agent.name,
                    display_name=agent.display_name,
                    description=agent.description,
                    enabled=agent.enabled,
                    source=agent.source,
                    path=agent.path,
                    prompt=agent.prompt,
                    tags=agent.tags,
                    routing=agent.routing,
                    tools=agent.tools,
                    skills=agent.skills,
                    limits=agent.limits,
                    delegation_role=agent.delegation_role,
                    security=agent.security,
                    compatibility=agent.compatibility,
                    extensions=agent.extensions,
                    warnings=agent.warnings,
                    shadowed=True,
                )
                if include_shadowed:
                    if shadowed_agent.enabled or include_disabled:
                        result.append(shadowed_agent)

            if include_shadowed or effective.enabled:
                if not effective.enabled and not include_disabled:
                    pass  # skip
                else:
                    result.append(effective)

    return result


def get_agent(
    name: str,
    workdir: str | None = None,
    source: Literal["global", "project"] | None = None,
) -> AgentDefinition | None:
    """Get a single agent by name.

    Returns the agent definition, or None if not found / disabled.
    When source is specified, returns only agents from that source.
    When source is specified, returns the agent even if it is shadowed.
    Default get_agent(name) returns the effective (non-shadowed) agent.
    """
    # Validate name first
    try:
        validate_agent_name(name)
    except AgentLoadError:
        return None

    agents = list_agents(workdir=workdir, include_disabled=True, include_shadowed=True)

    for agent in agents:
        if agent.name != name:
            continue
        if source is not None and agent.source != source:
            continue
        # When source is specified, return even shadowed entries (caller asked for that source)
        if agent.shadowed and source is None:
            continue
        # Return disabled agents too — caller checks .enabled if they care
        return agent

    return None