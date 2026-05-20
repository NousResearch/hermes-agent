"""Lightweight skill metadata utilities shared by prompt_builder and skills_tool.

This module intentionally avoids importing the tool registry, CLI config, or any
heavy dependency chain.  It is safe to import at module level without triggering
tool registration or provider resolution.
"""

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from hermes_constants import get_config_path, get_skills_dir

logger = logging.getLogger(__name__)

# ── Platform mapping ──────────────────────────────────────────────────────

PLATFORM_MAP = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}

EXCLUDED_SKILL_DIRS = frozenset((".git", ".github", ".hub", ".archive"))

# ── Lazy YAML loader ─────────────────────────────────────────────────────

_yaml_load_fn = None


def yaml_load(content: str):
    """Parse YAML with lazy import and CSafeLoader preference."""
    global _yaml_load_fn
    if _yaml_load_fn is None:
        import yaml

        loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader

        def _load(value: str):
            return yaml.load(value, Loader=loader)

        _yaml_load_fn = _load
    return _yaml_load_fn(content)


# ── Frontmatter parsing ──────────────────────────────────────────────────


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from a markdown string.

    Uses yaml with CSafeLoader for full YAML support (nested metadata, lists)
    with a fallback to simple key:value splitting for robustness.

    Returns:
        (frontmatter_dict, remaining_body)
    """
    frontmatter: Dict[str, Any] = {}
    body = content

    if not content.startswith("---"):
        return frontmatter, body

    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return frontmatter, body

    yaml_content = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :]

    try:
        parsed = yaml_load(yaml_content)
        if isinstance(parsed, dict):
            frontmatter = parsed
    except Exception:
        # Fallback: simple key:value parsing for malformed YAML
        for line in yaml_content.strip().split("\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()

    return frontmatter, body


# ── Platform matching ─────────────────────────────────────────────────────


def skill_matches_platform(frontmatter: Dict[str, Any]) -> bool:
    """Return True when the skill is compatible with the current OS.

    Skills declare platform requirements via a top-level ``platforms`` list
    in their YAML frontmatter::

        platforms: [macos]          # macOS only
        platforms: [macos, linux]   # macOS and Linux

    If the field is absent or empty the skill is compatible with **all**
    platforms (backward-compatible default).
    """
    platforms = frontmatter.get("platforms")
    if not platforms:
        return True
    if not isinstance(platforms, list):
        platforms = [platforms]
    current = sys.platform
    for platform in platforms:
        normalized = str(platform).lower().strip()
        mapped = PLATFORM_MAP.get(normalized, normalized)
        if current.startswith(mapped):
            return True
    return False


# ── Disabled skills ───────────────────────────────────────────────────────


def get_disabled_skill_names(platform: str | None = None) -> Set[str]:
    """Read disabled skill names from config.yaml.

    Args:
        platform: Explicit platform name (e.g. ``"telegram"``).  When
            *None*, resolves from ``HERMES_PLATFORM`` or
            ``HERMES_SESSION_PLATFORM`` env vars.  Falls back to the
            global disabled list when no platform is determined.

    Reads the config file directly (no CLI config imports) to stay
    lightweight.
    """
    config_path = get_config_path()
    if not config_path.exists():
        return set()
    try:
        parsed = yaml_load(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("Could not read skill config %s: %s", config_path, e)
        return set()
    if not isinstance(parsed, dict):
        return set()

    skills_cfg = parsed.get("skills")
    if not isinstance(skills_cfg, dict):
        return set()

    from gateway.session_context import get_session_env
    resolved_platform = (
        platform
        or os.getenv("HERMES_PLATFORM")
        or get_session_env("HERMES_SESSION_PLATFORM")
    )
    if resolved_platform:
        platform_disabled = (skills_cfg.get("platform_disabled") or {}).get(
            resolved_platform
        )
        if platform_disabled is not None:
            return _normalize_string_set(platform_disabled)
    return _normalize_string_set(skills_cfg.get("disabled"))


def _normalize_string_set(values) -> Set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    return {str(v).strip() for v in values if str(v).strip()}


# ── Registry entry dataclass ──────────────────────────────────────────────

# Valid scope values, ordered from most-specific (personal) to least (global).
# Resolution follows CSS specificity: personal overrides team overrides global.
SCOPE_ORDER: List[str] = ["personal", "team", "global"]


@dataclass
class RegistryEntry:
    """Represents a single entry in ``skills.registries`` config block.

    Carries all metadata needed for scope-aware resolution, Git sync, and
    write-gate enforcement.  Fields match the YAML config shape defined in
    Plan 003.

    Assumptions:
      - ``scope`` is one of "personal", "team", "global"; unknown scopes are
        treated as "global" (lowest precedence) so they never shadow personal.
      - ``path`` is always stored as an absolute resolved Path after construction.
      - ``writable`` defaults to True for personal, False for global; callers
        should not assume a default — they should read this field explicitly.
    """

    scope: Literal["personal", "team", "global"]
    name: str
    path: Path
    writable: bool = True
    remote: Optional[str] = None
    auto_sync: bool = False
    promote_requires_pr: bool = False
    # Optional channel tags for context-aware skill surfacing (ST-003-B.1).
    # Skills within this registry may declare ``channel_tags`` in their
    # frontmatter; this field is for registry-level defaults (not yet used
    # but wired so the dataclass doesn't need a breaking change later).
    channel_tags: List[str] = field(default_factory=list)


# ── Registry cache ─────────────────────────────────────────────────────────

# (config_path_str, mtime_ns) -> (registries_list, external_dirs_list).
# Same mtime-keyed strategy as _EXTERNAL_DIRS_CACHE to keep cold-start cheap.
_REGISTRY_CACHE: Dict[Tuple[str, int], Tuple[List[RegistryEntry], List[Path]]] = {}


def _registry_cache_clear() -> None:
    """Test hook — drop both caches at once."""
    _REGISTRY_CACHE.clear()
    _EXTERNAL_DIRS_CACHE.clear()


def _resolve_path(raw: str) -> Path:
    """Expand shell shortcuts and return an absolute Path."""
    expanded = os.path.expanduser(os.path.expandvars(str(raw).strip()))
    p = Path(expanded)
    if not p.is_absolute():
        from hermes_constants import get_hermes_home
        p = (get_hermes_home() / p).resolve()
    else:
        p = p.resolve()
    return p


def get_skill_registries() -> List[RegistryEntry]:
    """Parse ``skills.registries`` from config.yaml and return structured entries.

    Falls back gracefully:
      - If ``skills.registries`` is absent, returns an empty list (caller
        falls through to legacy ``external_dirs`` path).
      - Unknown ``scope`` values are coerced to ``"global"`` so they never
        silently override personal or team entries.
      - Paths that do not exist on disk are still returned — callers decide
        whether to skip non-existent dirs (scan paths) or create them (write
        paths).

    Cached in-process keyed on config.yaml mtime.  Same rationale as the
    existing ``_EXTERNAL_DIRS_CACHE``.
    """
    config_path = get_config_path()
    if not config_path.exists():
        return []

    try:
        stat = config_path.stat()
        cache_key: Tuple[str, int] = (str(config_path), stat.st_mtime_ns)
    except OSError:
        cache_key = None  # type: ignore[assignment]

    if cache_key is not None:
        cached = _REGISTRY_CACHE.get(cache_key)
        if cached is not None:
            regs, _ = cached
            return list(regs)

    try:
        parsed = yaml_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(parsed, dict):
        return []

    skills_cfg = parsed.get("skills")
    if not isinstance(skills_cfg, dict):
        return []

    raw_regs = skills_cfg.get("registries")
    if not raw_regs or not isinstance(raw_regs, list):
        # No registries block — cache empty result so we don't re-parse.
        if cache_key is not None:
            _REGISTRY_CACHE[cache_key] = ([], [])
        return []

    local_skills = get_skills_dir().resolve()
    seen_paths: Set[Path] = set()
    entries: List[RegistryEntry] = []

    for item in raw_regs:
        if not isinstance(item, dict):
            continue
        raw_path = item.get("path")
        if not raw_path:
            continue

        p = _resolve_path(str(raw_path))

        # Skip duplicates (same resolved path under different names is an error)
        if p in seen_paths:
            logger.debug(
                "skills.registries: duplicate path %s, skipping entry named %r",
                p, item.get("name"),
            )
            continue
        seen_paths.add(p)

        raw_scope = str(item.get("scope", "global")).strip().lower()
        if raw_scope not in SCOPE_ORDER:
            logger.debug(
                "skills.registries: unknown scope %r for %r, treating as 'global'",
                raw_scope, item.get("name"),
            )
            raw_scope = "global"

        # Global registries are always read-only regardless of config.
        # Personal and team registries default to writable=True unless
        # the user explicitly sets writable: false.
        if raw_scope == "global":
            writable = bool(item.get("writable", False))
        else:
            writable = bool(item.get("writable", True))

        remote = item.get("remote")
        if remote is not None:
            remote = str(remote).strip() or None

        channel_tags_raw = item.get("channel_tags", [])
        if isinstance(channel_tags_raw, str):
            channel_tags_raw = [channel_tags_raw]
        channel_tags = [str(t).strip() for t in channel_tags_raw if str(t).strip()]

        entry = RegistryEntry(
            scope=raw_scope,  # type: ignore[arg-type]
            name=str(item.get("name", p.name)),
            path=p,
            writable=writable,
            remote=remote,
            auto_sync=bool(item.get("auto_sync", False)),
            promote_requires_pr=bool(item.get("promote_requires_pr", False)),
            channel_tags=channel_tags,
        )
        entries.append(entry)

    # Sort into canonical resolution order: personal → team → global.
    # Within the same scope, preserve config.yaml declaration order (stable
    # sort keeps relative order for same-scope entries).
    entries.sort(key=lambda e: SCOPE_ORDER.index(e.scope))

    # Pre-compute external_dirs from this registry list for the combo cache.
    ext_dirs = [e.path for e in entries if e.path != local_skills and e.path.is_dir()]

    if cache_key is not None:
        _REGISTRY_CACHE[cache_key] = (list(entries), list(ext_dirs))

    return entries


def get_skill_scope_for_path(path: Path) -> str:
    """Return the scope label for a skill path by checking it against known registries.

    Used by the skills list and promotion CLI to annotate skills with their
    effective scope.  Falls back to ``"personal"`` for paths that live under
    the default ``~/.hermes/skills/`` dir (which is the implicit personal
    registry), and to ``"global"`` for bundled skills in the hermes-agent
    repo.  Returns ``"unknown"`` when no registry matches and the path is not
    under a known default location.

    Args:
        path: Absolute path to a skill directory (not the SKILL.md file itself).

    Returns:
        One of ``"personal"``, ``"team"``, ``"global"``, or ``"unknown"``.
    """
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path

    # Check structured registries first (most precise)
    for entry in get_skill_registries():
        try:
            resolved.relative_to(entry.path)
            return entry.scope
        except (ValueError, OSError):
            continue

    # Fallback heuristics for pre-registry installs
    local_skills = get_skills_dir().resolve()
    try:
        resolved.relative_to(local_skills)
        return "personal"
    except (ValueError, OSError):
        pass

    # Bundled skills shipped with hermes-agent are global
    try:
        import hermes_constants as _hc
        hermes_home = _hc.get_hermes_home().resolve()
        hermes_agent_dir = Path(__file__).resolve().parents[1]
        resolved.relative_to(hermes_agent_dir / "skills")
        return "global"
    except (ValueError, OSError, AttributeError):
        pass

    return "unknown"


def get_skill_registry_for_path(path: Path) -> Optional[RegistryEntry]:
    """Return the ``RegistryEntry`` that contains *path*, or ``None``.

    Used by the write path (skill_manage, git push after write) to know
    which remote to push to.
    """
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path

    for entry in get_skill_registries():
        try:
            resolved.relative_to(entry.path)
            return entry
        except (ValueError, OSError):
            continue
    return None


# ── External skills directories ──────────────────────────────────────────

# (config_path_str, mtime_ns) -> resolved external dirs list.  Keyed by
# mtime_ns so a config.yaml edit mid-run is picked up automatically;
# otherwise every call would re-read + re-YAML-parse the 15KB config,
# which becomes the dominant cost of ``hermes`` startup when ~120 skills
# each trigger a category lookup during banner construction (10+ seconds
# of pure waste).
_EXTERNAL_DIRS_CACHE: Dict[Tuple[str, int], List[Path]] = {}


def _external_dirs_cache_clear() -> None:
    """Test hook — drop the in-process cache."""
    _EXTERNAL_DIRS_CACHE.clear()


def get_external_skills_dirs() -> List[Path]:
    """Read ``skills.external_dirs`` from config.yaml and return validated paths.

    Each entry is expanded (``~`` and ``${VAR}``) and resolved to an absolute
    path.  Only directories that actually exist are returned.  Duplicates and
    paths that resolve to the local ``~/.hermes/skills/`` are silently skipped.

    Cached in-process, keyed on ``config.yaml`` mtime — the function is
    called once per skill during banner / tool-registry scans, and YAML
    parsing a non-trivial config dominates ``hermes`` cold-start time
    when the cache is absent.
    """
    config_path = get_config_path()
    if not config_path.exists():
        return []

    # Cache key: (absolute path, mtime_ns).  stat() is ~2us vs ~85ms for
    # the full YAML parse, so the fast path is nearly free.
    try:
        stat = config_path.stat()
        cache_key: Tuple[str, int] = (str(config_path), stat.st_mtime_ns)
    except OSError:
        cache_key = None  # type: ignore[assignment]

    if cache_key is not None:
        cached = _EXTERNAL_DIRS_CACHE.get(cache_key)
        if cached is not None:
            # Return a copy so callers can't mutate the cached list.
            return list(cached)

    try:
        parsed = yaml_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(parsed, dict):
        return []

    skills_cfg = parsed.get("skills")
    if not isinstance(skills_cfg, dict):
        return []

    raw_dirs = skills_cfg.get("external_dirs")
    if not raw_dirs:
        result: List[Path] = []
        if cache_key is not None:
            _EXTERNAL_DIRS_CACHE[cache_key] = list(result)
        return result
    if isinstance(raw_dirs, str):
        raw_dirs = [raw_dirs]
    if not isinstance(raw_dirs, list):
        return []

    from hermes_constants import get_hermes_home

    hermes_home = get_hermes_home()
    local_skills = get_skills_dir().resolve()
    seen: Set[Path] = set()
    result = []

    for entry in raw_dirs:
        entry = str(entry).strip()
        if not entry:
            continue
        # Expand ~ and environment variables
        expanded = os.path.expanduser(os.path.expandvars(entry))
        p = Path(expanded)
        # Resolve relative paths against HERMES_HOME, not cwd
        if not p.is_absolute():
            p = (hermes_home / p).resolve()
        else:
            p = p.resolve()
        if p == local_skills:
            continue
        if p in seen:
            continue
        if p.is_dir():
            seen.add(p)
            result.append(p)
        else:
            logger.debug("External skills dir does not exist, skipping: %s", p)

    if cache_key is not None:
        _EXTERNAL_DIRS_CACHE[cache_key] = list(result)
    return result


def get_all_skills_dirs() -> List[Path]:
    """Return all skill directories in resolution order: personal → team → global → legacy.

    Resolution order follows CSS specificity: more-specific scope wins when two
    registries contain a skill with the same name.  First-found-wins semantics
    are preserved — callers iterate and stop at the first match.

    Order:
      1. Directories from ``skills.registries`` sorted personal → team → global
         (already enforced by ``get_skill_registries()``).
      2. Legacy ``~/.hermes/skills/`` personal dir if NOT already covered by a
         registry entry — ensures backward compatibility for users without a
         ``registries`` block.
      3. Legacy ``skills.external_dirs`` entries appended last.

    The local ``~/.hermes/skills/`` directory is always included (even if it
    doesn't exist yet) so callers that create skills there see it immediately.
    """
    local_skills = get_skills_dir()
    local_skills_resolved = local_skills.resolve()

    # Step 1: registry-based dirs (already sorted personal→team→global)
    registry_entries = get_skill_registries()
    registry_paths: List[Path] = [e.path for e in registry_entries]
    registry_paths_resolved: Set[Path] = {p.resolve() for p in registry_paths}

    dirs: List[Path] = list(registry_paths)

    # Step 2: personal fallback — ensure ~/.hermes/skills/ is always present
    # even when no registries block exists.
    if local_skills_resolved not in registry_paths_resolved:
        dirs.insert(0, local_skills)

    # Step 3: legacy external_dirs (skip any already covered by a registry)
    seen: Set[Path] = {p.resolve() for p in dirs}
    for ext_dir in get_external_skills_dirs():
        try:
            ext_resolved = ext_dir.resolve()
        except OSError:
            ext_resolved = ext_dir
        if ext_resolved not in seen:
            dirs.append(ext_dir)
            seen.add(ext_resolved)

    return dirs


# ── Condition extraction ──────────────────────────────────────────────────


def extract_skill_conditions(frontmatter: Dict[str, Any]) -> Dict[str, List]:
    """Extract conditional activation fields from parsed frontmatter."""
    metadata = frontmatter.get("metadata")
    # Handle cases where metadata is not a dict (e.g., a string from malformed YAML)
    if not isinstance(metadata, dict):
        metadata = {}
    hermes = metadata.get("hermes") or {}
    if not isinstance(hermes, dict):
        hermes = {}
    return {
        "fallback_for_toolsets": hermes.get("fallback_for_toolsets", []),
        "requires_toolsets": hermes.get("requires_toolsets", []),
        "fallback_for_tools": hermes.get("fallback_for_tools", []),
        "requires_tools": hermes.get("requires_tools", []),
    }


# ── Skill config extraction ───────────────────────────────────────────────


def extract_skill_config_vars(frontmatter: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract config variable declarations from parsed frontmatter.

    Skills declare config.yaml settings they need via::

        metadata:
          hermes:
            config:
              - key: wiki.path
                description: Path to the LLM Wiki knowledge base directory
                default: "~/wiki"
                prompt: Wiki directory path

    Returns a list of dicts with keys: ``key``, ``description``, ``default``,
    ``prompt``.  Invalid or incomplete entries are silently skipped.
    """
    metadata = frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        return []
    hermes = metadata.get("hermes")
    if not isinstance(hermes, dict):
        return []
    raw = hermes.get("config")
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    result: List[Dict[str, Any]] = []
    seen: set = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        if not key or key in seen:
            continue
        # Must have at least key and description
        desc = str(item.get("description", "")).strip()
        if not desc:
            continue
        entry: Dict[str, Any] = {
            "key": key,
            "description": desc,
        }
        default = item.get("default")
        if default is not None:
            entry["default"] = default
        prompt_text = item.get("prompt")
        if isinstance(prompt_text, str) and prompt_text.strip():
            entry["prompt"] = prompt_text.strip()
        else:
            entry["prompt"] = desc
        seen.add(key)
        result.append(entry)
    return result


def discover_all_skill_config_vars() -> List[Dict[str, Any]]:
    """Scan all enabled skills and collect their config variable declarations.

    Walks every skills directory, parses each SKILL.md frontmatter, and returns
    a deduplicated list of config var dicts.  Each dict also includes a
    ``skill`` key with the skill name for attribution.

    Disabled and platform-incompatible skills are excluded.
    """
    all_vars: List[Dict[str, Any]] = []
    seen_keys: set = set()

    disabled = get_disabled_skill_names()
    for skills_dir in get_all_skills_dirs():
        if not skills_dir.is_dir():
            continue
        for skill_file in iter_skill_index_files(skills_dir, "SKILL.md"):
            try:
                raw = skill_file.read_text(encoding="utf-8")
                frontmatter, _ = parse_frontmatter(raw)
            except Exception:
                continue

            skill_name = frontmatter.get("name") or skill_file.parent.name
            if str(skill_name) in disabled:
                continue
            if not skill_matches_platform(frontmatter):
                continue

            config_vars = extract_skill_config_vars(frontmatter)
            for var in config_vars:
                if var["key"] not in seen_keys:
                    var["skill"] = str(skill_name)
                    all_vars.append(var)
                    seen_keys.add(var["key"])

    return all_vars


# Storage prefix: all skill config vars are stored under skills.config.*
# in config.yaml.  Skill authors declare logical keys (e.g. "wiki.path");
# the system adds this prefix for storage and strips it for display.
SKILL_CONFIG_PREFIX = "skills.config"


def _resolve_dotpath(config: Dict[str, Any], dotted_key: str):
    """Walk a nested dict following a dotted key.  Returns None if any part is missing."""
    parts = dotted_key.split(".")
    current = config
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def resolve_skill_config_values(
    config_vars: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve current values for skill config vars from config.yaml.

    Skill config is stored under ``skills.config.<key>`` in config.yaml.
    Returns a dict mapping **logical** keys (as declared by skills) to their
    current values (or the declared default if the key isn't set).
    Path values are expanded via ``os.path.expanduser``.
    """
    config_path = get_config_path()
    config: Dict[str, Any] = {}
    if config_path.exists():
        try:
            parsed = yaml_load(config_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                config = parsed
        except Exception:
            pass

    resolved: Dict[str, Any] = {}
    for var in config_vars:
        logical_key = var["key"]
        storage_key = f"{SKILL_CONFIG_PREFIX}.{logical_key}"
        value = _resolve_dotpath(config, storage_key)

        if value is None or (isinstance(value, str) and not value.strip()):
            value = var.get("default", "")

        # Expand ~ in path-like values
        if isinstance(value, str) and ("~" in value or "${" in value):
            value = os.path.expanduser(os.path.expandvars(value))

        resolved[logical_key] = value

    return resolved


# ── Description extraction ────────────────────────────────────────────────


def extract_skill_description(frontmatter: Dict[str, Any]) -> str:
    """Extract a truncated description from parsed frontmatter."""
    raw_desc = frontmatter.get("description", "")
    if not raw_desc:
        return ""
    desc = str(raw_desc).strip().strip("'\"")
    if len(desc) > 60:
        return desc[:57] + "..."
    return desc


# ── File iteration ────────────────────────────────────────────────────────


def iter_skill_index_files(skills_dir: Path, filename: str):
    """Walk skills_dir yielding sorted paths matching *filename*.

    Excludes ``.git``, ``.github``, ``.hub``, ``.archive`` directories.
    """
    matches = []
    for root, dirs, files in os.walk(skills_dir, followlinks=True):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_SKILL_DIRS]
        if filename in files:
            matches.append(Path(root) / filename)
    for path in sorted(matches, key=lambda p: str(p.relative_to(skills_dir))):
        yield path


# ── Namespace helpers for plugin-provided skills ───────────────────────────

_NAMESPACE_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def parse_qualified_name(name: str) -> Tuple[Optional[str], str]:
    """Split ``'namespace:skill-name'`` into ``(namespace, bare_name)``.

    Returns ``(None, name)`` when there is no ``':'``.
    """
    if ":" not in name:
        return None, name
    return tuple(name.split(":", 1))  # type: ignore[return-value]


def is_valid_namespace(candidate: Optional[str]) -> bool:
    """Check whether *candidate* is a valid namespace (``[a-zA-Z0-9_-]+``)."""
    if not candidate:
        return False
    return bool(_NAMESPACE_RE.match(candidate))
