"""Lightweight skill metadata utilities shared by prompt_builder and skills_tool.

This module intentionally avoids importing the tool registry, CLI config, or any
heavy dependency chain.  It is safe to import at module level without triggering
tool registration or provider resolution.
"""

import ast
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hermes_constants import get_config_path, get_skills_dir, is_termux

logger = logging.getLogger(__name__)

# ── Platform mapping ──────────────────────────────────────────────────────

PLATFORM_MAP = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}

EXCLUDED_SKILL_DIRS = frozenset(
    (
        ".git",
        ".github",
        ".hub",
        ".archive",
        ".curator_backups",
        ".venv",
        "venv",
        "node_modules",
        "site-packages",
        "__pycache__",
        ".tox",
        ".nox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    )
)

# Supporting files live inside a skill package and are loaded explicitly via
# skill_view(skill, file_path=...). They are not standalone skills and must not
# be scanned for active SKILL.md/DESCRIPTION.md entries, even if a Curator or
# archive workflow preserves a complete old skill package under references/.
SKILL_SUPPORT_DIRS = frozenset(("references", "templates", "assets", "scripts"))

# ── Org-shared skills (sync contract) ───────────────────────────
# Org mirrors live under ~/.hermes/skills/_org/<org_id>/. Resolution is
# TOKEN-GATED via a marker file the sync client writes after verifying the
# token (skills_sync_client.pull_org_skills): only the marked org's mirror is
# scanned. No marker ⇒ no org skills load. The marker is plain data (org_id
# string) so this module stays import-light; the VERIFICATION lives in the
# sync client, which is the only writer. Offline grace: the marker persists,
# so already-pulled org skills keep working without connectivity; a VERIFIED
# org change (or personal-org token) rewrites/removes it.

ORG_MIRROR_DIR_NAME = "_org"
ORG_ACTIVE_MARKER = ".active_org"
ORG_PROVENANCE_FILE = ".org-provenance.json"
# Records the fingerprint of each skill exactly as upstream sent it, so a
# later local edit is detectable and an org pull can refuse to clobber it.
ORG_BASELINE_FILE = ".org-baseline.json"


def read_active_org_id(skills_dir: Path) -> Optional[str]:
    """The org id whose mirror may resolve, or None (no org skills load)."""
    try:
        marker = skills_dir / ORG_MIRROR_DIR_NAME / ORG_ACTIVE_MARKER
        if not marker.exists():
            return None
        val = marker.read_text(encoding="utf-8").strip()
        return val or None
    except OSError:
        return None


def is_org_mirror_path(path, skills_dir: Path) -> bool:
    """True when *path* is inside the org mirror (``_org/``)."""
    try:
        rel = Path(path).resolve().relative_to(Path(skills_dir).resolve())
    except (OSError, ValueError):
        return False
    return bool(rel.parts) and rel.parts[0] == ORG_MIRROR_DIR_NAME


def org_id_of_path(path, skills_dir: Path) -> Optional[str]:
    """The ``<org_id>`` segment for a path under ``_org/<org_id>/...``."""
    try:
        rel = Path(path).resolve().relative_to(Path(skills_dir).resolve())
    except (OSError, ValueError):
        return None
    if len(rel.parts) >= 2 and rel.parts[0] == ORG_MIRROR_DIR_NAME:
        return rel.parts[1]
    return None


def is_excluded_skill_path(path, *, root: Optional[Path] = None) -> bool:
    """True if *path* should be skipped by active skill scanners.

    Use this on every ``SKILL.md`` path produced by direct ``rglob`` scans to
    prune dependency, virtualenv, VCS, cache, and progressive-disclosure
    support-package paths. Centralising the check here keeps every
    skill-scanning site in sync with the shared exclusion set.

    Accepts a Path or string.
    """
    try:
        parts = path.parts  # Path
    except AttributeError:
        from pathlib import PurePath
        parts = PurePath(str(path)).parts
    return any(part in EXCLUDED_SKILL_DIRS for part in parts) or is_skill_support_path(
        path, root=root
    )


def is_skill_support_path(path, *, root: Optional[Path] = None) -> bool:
    """True if *path* is under a support dir of an actual skill root.

    ``references/``, ``templates/``, ``assets/``, and ``scripts/`` are
    progressive-disclosure support areas when they sit directly inside a skill
    directory containing ``SKILL.md``. They are not active discovery roots for
    standalone skills. A preserved package such as
    ``some-skill/references/old-skill-package/SKILL.md`` is documentation data
    unless the caller explicitly loads it via ``file_path``.

    Legitimate categories or skill names such as ``skills/scripts/foo`` remain
    discoverable because their ``scripts`` component is not directly under a
    directory that contains ``SKILL.md``.
    """
    path_obj = path if isinstance(path, Path) else Path(str(path))
    parts = path_obj.parts
    # Last component may be a file or candidate skill directory name. Only
    # components before the leaf can be containing support directories.
    for idx, part in enumerate(parts[:-1]):
        if part not in SKILL_SUPPORT_DIRS or idx == 0:
            continue
        skill_root = Path(*parts[:idx])
        if root is not None and not path_obj.is_absolute():
            skill_root = root / skill_root
        if (skill_root / "SKILL.md").exists():
            return True
    return False


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

    A single leading UTF-8 BOM (U+FEFF) is stripped before parsing. Windows
    GUI editors (Notepad, PowerShell ``>``) prepend one when saving a SKILL.md
    as UTF-8, and ``read_text(encoding="utf-8")`` preserves it (only
    ``utf-8-sig`` strips it). Left in place, the BOM defeats the ``---`` fence
    check below and the whole frontmatter is silently discarded — name,
    description, ``platforms`` gating, env-var setup, and conditional
    activation all vanish. See CONTRIBUTING.md "File encoding".

    Returns:
        (frontmatter_dict, remaining_body)
    """
    frontmatter: Dict[str, Any] = {}

    # Strip only a leading BOM; a BOM mid-content is data, not a marker.
    if content.startswith("\ufeff"):
        content = content[1:]
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


def skill_matches_platform_list(platforms: Any) -> bool:
    """Return True when *platforms* is compatible with the current OS."""
    if not platforms:
        return True
    if not isinstance(platforms, list):
        platforms = [platforms]
    current = sys.platform
    running_in_termux = is_termux()
    for platform in platforms:
        normalized = str(platform).lower().strip()
        mapped = PLATFORM_MAP.get(normalized, normalized)
        if current.startswith(mapped):
            return True
        # Termux runs a Linux userland on Android. Accept linux-tagged
        # skills regardless of whether sys.platform is "linux" (pre-3.13
        # Termux) or "android" (Python 3.13+ Termux, and any other
        # Android runtime).
        if running_in_termux and mapped == "linux":
            return True
        # Explicit termux/android tags match a Termux session too.
        if running_in_termux and mapped in ("termux", "android"):
            return True
    return False


def skill_matches_platform(frontmatter: Dict[str, Any]) -> bool:
    """Return True when the skill is compatible with the current OS.

    Skills declare platform requirements via a top-level ``platforms`` list
    in their YAML frontmatter::

        platforms: [macos]          # macOS only
        platforms: [macos, linux]   # macOS and Linux

    If the field is absent or empty the skill is compatible with **all**
    platforms (backward-compatible default).

    Termux note: on Termux/Android, ``sys.platform`` is ``"linux"`` on
    older Pythons but became ``"android"`` on Python 3.13+. Termux is a
    Linux userland riding on the Android kernel, so skills tagged
    ``linux`` are treated as compatible in Termux regardless of which
    ``sys.platform`` value Python reports. Individual Linux commands
    inside a skill may still misbehave (no systemd, BusyBox utils, no
    apt/dnf, etc.) but that is on the skill, not on platform gating.
    """
    return skill_matches_platform_list(frontmatter.get("platforms"))


# ── Environment matching ──────────────────────────────────────────────────

# Recognized environment tags and how each is detected. An environment tag is
# a *relevance* gate, not a hard-compatibility gate (that is what ``platforms:``
# is for). A skill tagged for an environment it isn't relevant to is hidden from
# the skills index / offer surfaces so it does not add noise for users who will
# never need it — but it can ALWAYS still be loaded explicitly (``skill_view``,
# ``--skills``), because an explicit request is explicit consent.
#
# Detection is cached for the process lifetime via ``_ENV_DETECT_CACHE``.
_KNOWN_ENVIRONMENTS = frozenset({"kanban", "docker", "s6"})

_ENV_DETECT_CACHE: Dict[str, bool] = {}


def _detect_environment(env: str) -> bool:
    """Return True when the named runtime environment is currently active.

    Cached per process, EXCEPT ``kanban``: that verdict is context-dependent
    (a delegate_task child or an in-process cron job sees the worker's
    HERMES_KANBAN_* vars without owning them), so caching it process-wide would
    freeze whichever context asked first and leak it to the others.
    """
    if env != "kanban" and env in _ENV_DETECT_CACHE:
        return _ENV_DETECT_CACHE[env]

    result = True
    if env == "kanban":
        # Kanban is "active" either as a dispatcher-spawned worker (the
        # dispatcher sets ``HERMES_KANBAN_TASK`` / ``HERMES_KANBAN_BOARD`` in the
        # worker env) or as an orchestrator profile that has opted into the
        # kanban toolset. Mirror the same signals the kanban tools themselves
        # gate on (``tools/kanban_tools.py``) so the offer filter agrees with
        # tool availability.
        if os.getenv("HERMES_KANBAN_TASK") or os.getenv("HERMES_KANBAN_BOARD"):
            # ...but only when this execution actually owns the dispatcher's
            # task. A delegate_task child or a cron job fired in-process from a
            # worker sees the worker's vars without being that worker.
            try:
                from agent.delegation_context import (
                    is_dispatcher_owned_worker_context,
                )

                _owns_dispatcher_task = is_dispatcher_owned_worker_context()
            except Exception:
                _owns_dispatcher_task = True
        else:
            _owns_dispatcher_task = False
        if _owns_dispatcher_task:
            result = True
        else:
            try:
                from tools.kanban_tools import _profile_has_kanban_toolset

                result = bool(_profile_has_kanban_toolset())
            except Exception:
                result = False
    elif env == "docker":
        try:
            from hermes_constants import is_container

            result = is_container()
        except Exception:
            result = False
    elif env == "s6":
        # The Hermes Docker image runs s6-overlay as PID 1 (/init). s6 plants
        # its runtime scaffolding under /run/s6 and ships its admin tree under
        # /package/admin/s6-overlay. Either marker means we're inside an
        # s6-supervised container.
        result = os.path.isdir("/run/s6") or os.path.isdir(
            "/package/admin/s6-overlay"
        )

    _ENV_DETECT_CACHE[env] = result
    return result


def skill_matches_environment(frontmatter: Dict[str, Any]) -> bool:
    """Return True when the skill is relevant to the current runtime environment.

    Skills may declare an ``environments`` list in their YAML frontmatter::

        environments: [kanban]        # only relevant when kanban is active
        environments: [s6]            # only relevant inside the s6 Docker image
        environments: [docker]        # only relevant inside any container

    If the field is absent or empty the skill is relevant in **all**
    environments (backward-compatible default).

    This is an OFFER-time filter: it controls whether a skill shows up in the
    skills index / autocomplete / slash-command list. It is intentionally NOT
    enforced by ``skill_view`` or ``--skills`` preloading — an explicit load is
    explicit consent, and load-bearing force-loads (e.g. a dispatcher pinning
    a task to a specialist skill via ``--skills``) must always succeed
    regardless of how the offer surfaces filter the skill.

    A skill matches when ANY of its declared environments is currently active
    (OR semantics, mirroring ``platforms``). Unknown env tags fail open.
    """
    environments = frontmatter.get("environments")
    if not environments:
        return True
    if not isinstance(environments, list):
        environments = [environments]
    for env in environments:
        normalized = str(env).lower().strip()
        if not normalized:
            continue
        if normalized not in _KNOWN_ENVIRONMENTS:
            # Tag we don't understand — don't hide the skill over it.
            return True
        if _detect_environment(normalized):
            return True
    return False


# ── Disabled skills ───────────────────────────────────────────────────────


_ConfigStatKey = Tuple[str, int, int, int, int, int]


def _config_stat_key(config_path: Path, stat: os.stat_result) -> _ConfigStatKey:
    return (
        str(config_path),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


_RAW_CONFIG_CACHE: Dict[_ConfigStatKey, Dict[str, Any]] = {}


def _raw_config_cache_clear() -> None:
    """Test hook — drop the shared raw config cache."""
    _RAW_CONFIG_CACHE.clear()


def _load_raw_config() -> Dict[str, Any]:
    """Read config.yaml with a shared filesystem-metadata keyed cache.

    This module intentionally avoids importing ``hermes_cli.config`` on the
    skill prompt/build path. A tiny local cache gives the same repeated-read
    win without pulling the heavier CLI config stack into startup. File
    identity and change time supplement mtime and size so atomic replacements
    cannot reuse the previous cache entry merely by preserving those values.
    """
    config_path = get_config_path()
    if not config_path.exists():
        return {}
    try:
        stat = config_path.stat()
        cache_key = _config_stat_key(config_path, stat)
    except OSError:
        cache_key = None

    if cache_key is not None:
        cached = _RAW_CONFIG_CACHE.get(cache_key)
        if cached is not None:
            return cached

    try:
        parsed = yaml_load(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("Could not read skill config %s: %s", config_path, e)
        return {}
    if not isinstance(parsed, dict):
        return {}

    if cache_key is not None:
        _RAW_CONFIG_CACHE.clear()
        _RAW_CONFIG_CACHE[cache_key] = parsed
    return parsed


# Skills that must stay available regardless of configuration. The
# `hermes-agent` skill is the agent's own operating manual — it drives
# configuring, extending, and troubleshooting Hermes itself, and the system
# prompt unconditionally points at it. Disabling it leaves the agent unable
# to help with Hermes, so disable requests for these names are ignored
# everywhere the disabled list is consulted.
ESSENTIAL_SKILLS: frozenset = frozenset({"hermes-agent"})


def get_disabled_skill_names(platform: str | None = None) -> Set[str]:
    """Read disabled skill names from config.yaml.

    Args:
        platform: Explicit platform name (e.g. ``"telegram"``).  When
            *None*, resolves from ``HERMES_PLATFORM`` or
            ``HERMES_SESSION_PLATFORM`` env vars.  Returns the global
            disabled list, unioned with the platform-specific list when a
            platform is resolved (a globally-disabled skill stays disabled
            on every platform).

    Reads the config file directly (no CLI config imports) to stay
    lightweight.
    """
    parsed = _load_raw_config()
    if not parsed:
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
    global_disabled = _normalize_string_set(skills_cfg.get("disabled"))
    if resolved_platform:
        platform_disabled = (skills_cfg.get("platform_disabled") or {}).get(
            resolved_platform
        )
        if platform_disabled is not None:
            return (
                global_disabled | _normalize_string_set(platform_disabled)
            ) - ESSENTIAL_SKILLS
    return global_disabled - ESSENTIAL_SKILLS


def parse_config_string_list(value) -> List[str]:
    """Normalize a config value that may hold a JSON-array string into a list.

    ``hermes config set`` and JSON-mode editor saves store lists as quoted
    JSON strings (``'["a","b"]'`` or the Python-literal ``"['a']"``). Treating
    such a string as a single name makes a curated disabled list silently
    filter nothing (#86661); parsing it restores the intended list. A scalar
    string still means one name (#13026).
    """
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        return [value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item) for item in value]
    return []


def _normalize_string_set(values) -> Set[str]:
    return {name.strip() for name in parse_config_string_list(values) if name.strip()}


# ── External skills directories ──────────────────────────────────────────

# Filesystem metadata key -> resolved external dirs list.  Keyed consistently
# with the shared raw config cache so a config.yaml edit or replacement is
# picked up automatically;
# otherwise every call would re-read + re-YAML-parse the 15KB config,
# which becomes the dominant cost of ``hermes`` startup when ~120 skills
# each trigger a category lookup during banner construction (10+ seconds
# of pure waste).
_EXTERNAL_DIRS_CACHE: Dict[_ConfigStatKey, List[Path]] = {}


def _external_dirs_cache_clear() -> None:
    """Test hook — drop the in-process cache."""
    _EXTERNAL_DIRS_CACHE.clear()
    _raw_config_cache_clear()


def get_external_skills_dirs() -> List[Path]:
    """Read ``skills.external_dirs`` from config.yaml and return validated paths.

    Each entry is expanded (``~`` and ``${VAR}``) and resolved to an absolute
    path.  Only directories that actually exist are returned.  Duplicates and
    paths that resolve to the local ``~/.hermes/skills/`` are silently skipped.

    Cached in-process, keyed on ``config.yaml`` filesystem metadata — the
    function is called once per skill during banner / tool-registry scans, and
    YAML parsing a non-trivial config dominates ``hermes`` cold-start time when
    the cache is absent.
    """
    config_path = get_config_path()
    if not config_path.exists():
        return []

    # stat() is ~2us vs ~85ms for the full YAML parse, so the fast path is
    # nearly free.
    try:
        stat = config_path.stat()
        cache_key: _ConfigStatKey = _config_stat_key(config_path, stat)
    except OSError:
        cache_key = None  # type: ignore[assignment]

    if cache_key is not None:
        cached = _EXTERNAL_DIRS_CACHE.get(cache_key)
        if cached is not None:
            # Return a copy so callers can't mutate the cached list.
            return list(cached)

    parsed = _load_raw_config()
    if not parsed:
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


def get_skill_create_dir() -> Optional[Path]:
    """Return the configured ``skills.create_dir``, or ``None`` when unset.

    When set, agent-created skills (``skill_manage`` action=create) land in
    this directory instead of the profile-local ``~/.hermes/skills/``, and
    every user-facing instruction string that names the creation path renders
    this directory instead of the default.

    The entry is expanded (``~`` and ``${VAR}``); relative paths resolve
    against HERMES_HOME.  A value that resolves to the local skills dir is
    treated as unset (that is already the default behaviour).  The directory
    does NOT need to exist yet — skill creation mkdirs it on first write.
    """
    parsed = _load_raw_config()
    if not parsed:
        return None
    skills_cfg = parsed.get("skills")
    if not isinstance(skills_cfg, dict):
        return None
    raw = skills_cfg.get("create_dir")
    if not raw or not isinstance(raw, (str, os.PathLike)):
        return None
    entry = str(raw).strip()
    if not entry:
        return None

    from hermes_constants import get_hermes_home

    expanded = os.path.expanduser(os.path.expandvars(entry))
    p = Path(expanded)
    if not p.is_absolute():
        p = get_hermes_home() / p
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    try:
        if resolved == get_skills_dir().resolve():
            return None
    except OSError:
        pass
    return resolved


def display_skill_create_dir() -> str:
    """User-facing display string for where new skills are created.

    Renders the configured ``skills.create_dir`` (with ``~/`` shorthand when
    under the user's home) or the default ``<home>/skills/`` path.  Used by
    instruction text (tool schema descriptions, prompts, docs strings) so a
    configured creation dir changes every instruction that names the path.
    """
    from hermes_constants import display_hermes_home

    create_dir = get_skill_create_dir()
    if create_dir is None:
        return f"{display_hermes_home()}/skills/"
    try:
        return "~/" + create_dir.relative_to(Path.home()).as_posix() + "/"
    except ValueError:
        return create_dir.as_posix() + "/"


def get_all_skills_dirs() -> List[Path]:
    """Return all skill directories: local ``~/.hermes/skills/`` first, then external.

    The local dir is always first (and always included even if it doesn't exist
    yet — callers handle that).  When ``skills.create_dir`` is configured, it
    follows immediately after the local dir (so agent-created skills are
    discovered, trusted, and modifiable).  External dirs follow in config order.

    NOTE: trusted project-local dirs (``./.hermes/skills`` at the git root) are
    NOT part of this list — they have *higher* precedence than the local dir,
    so callers that need them use :func:`get_project_skills_dirs` and scan
    those roots first. See ``get_scan_ordered_skills_dirs`` for the full
    precedence-ordered list.
    """
    dirs = [get_skills_dir()]
    create_dir = get_skill_create_dir()
    if create_dir is not None and create_dir.is_dir():
        dirs.append(create_dir)
    for d in get_external_skills_dirs():
        if d not in dirs:
            dirs.append(d)
    return dirs


# ── Project-local skills directories ──────────────────────────────────────
#
# Repo-local skills, mirroring what OpenCode (.opencode/skill/, .agents/skills/)
# and Codex (.codex/skills/, .agents/skills/) do: a project checkout can carry
# its own skills, active only for sessions started inside that project.
#
# Two candidate roots at the project root (found by walking up from cwd to the
# first directory containing ``.git``):
#   <root>/.hermes/skills/   — Hermes-native location
#   <root>/.agents/skills/   — cross-tool convention shared with other harnesses
#
# TRUST GATE: unlike AGENTS.md (plain instruction text), skills are load-on-
# demand procedure documents an agent will follow — auto-sourcing them from any
# cloned repo is a prompt-injection vector. Project skills therefore only load
# when the project root is listed in ``skills.trusted_project_dirs`` in
# config.yaml (Codex-style per-path trust). Untrusted dirs are still
# *discoverable* via get_untrusted_project_skills_root() so the CLI can print
# a one-line "run `hermes skills trust`" notice.
#
# PRECEDENCE: trusted project skills override same-named profile/bundled
# skills (index scans project dirs first; skill_view resolves cross-tier
# collisions in favor of the project tier). This matches both competitor
# harnesses and is the point of the feature: vendored repo skills win inside
# their repo.
#
# CACHE SAFETY: cwd is fixed for the life of a session, and the trust list is
# read from config at agent build time — the resolved dirs are stable for the
# conversation, so the skills index (and with it the system prompt) stays
# byte-stable. Same contract as AGENTS.md injection and project plugins.

PROJECT_SKILLS_SUBDIRS = (
    os.path.join(".hermes", "skills"),
    os.path.join(".agents", "skills"),
)

# Walk-up bound: don't scan the whole filesystem on pathological cwds.
_PROJECT_ROOT_MAX_DEPTH = 64


def find_project_root(start: Optional[Path] = None) -> Optional[Path]:
    """Locate the enclosing project root: nearest ancestor containing ``.git``.

    Returns None when cwd is not inside a git checkout. ``.git`` may be a dir
    (normal clone) or a file (worktree/submodule) — both count.

    When *start* is not given, the surface's working directory wins over the
    process cwd: ``TERMINAL_CWD`` is the same per-surface workdir the terminal
    tool and cron jobs use (a cron job sets it from its per-job ``workdir``
    without chdir'ing the scheduler process). This is what lets
    non-interactive surfaces inherit a prior interactive trust decision by
    project identity — and a surface with no workdir in a trusted repo simply
    resolves no project and loads nothing (#48975).
    """
    try:
        if start is None:
            from agent.runtime_cwd import scope_terminal_cwd

            env_cwd = scope_terminal_cwd()
            start = Path(env_cwd) if env_cwd else Path.cwd()
        cur = Path(start).resolve()
    except OSError:
        return None
    home = Path.home().resolve()
    for _ in range(_PROJECT_ROOT_MAX_DEPTH):
        try:
            if (cur / ".git").exists():
                # A git checkout AT the home dir (dotfiles-style) would make
                # every session project-scoped; treat home itself as non-project.
                if cur == home:
                    return None
                return cur
        except OSError:
            return None
        if cur.parent == cur:
            return None
        cur = cur.parent
    return None


def _project_trusted_dirs_from_config() -> Set[Path]:
    """Resolved set of trusted project roots from ``skills.trusted_project_dirs``."""
    parsed = _load_raw_config()
    if not parsed:
        return set()
    skills_cfg = parsed.get("skills")
    if not isinstance(skills_cfg, dict):
        return set()
    raw = skills_cfg.get("trusted_project_dirs")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return set()
    result: Set[Path] = set()
    for entry in raw:
        entry = str(entry).strip()
        if not entry:
            continue
        try:
            result.add(Path(os.path.expanduser(os.path.expandvars(entry))).resolve())
        except OSError:
            continue
    return result


def is_project_root_trusted(root: Path) -> bool:
    """True when *root* is listed in ``skills.trusted_project_dirs``."""
    try:
        return Path(root).resolve() in _project_trusted_dirs_from_config()
    except OSError:
        return False


def _candidate_project_skills_dirs(root: Path) -> List[Path]:
    """Existing skill dirs under *root*, excluding the profile's own skills dir.

    The exclusion matters when HERMES_HOME itself lives inside a git checkout:
    ``<root>/.hermes/skills`` would otherwise double as both the profile-local
    and the project tier.
    """
    local_skills = get_skills_dir().resolve()
    dirs: List[Path] = []
    for sub in PROJECT_SKILLS_SUBDIRS:
        cand = root / sub
        try:
            if cand.is_dir() and cand.resolve() != local_skills:
                dirs.append(cand.resolve())
        except OSError:
            continue
    return dirs


def get_project_skills_dirs() -> List[Path]:
    """Trusted project-local skill dirs for the current cwd (may be empty).

    Empty when: not in a git checkout, no project skills dirs exist, project
    discovery is disabled (``skills.project_discovery: false``), or the
    project root is not trusted.
    """
    parsed = _load_raw_config()
    skills_cfg = parsed.get("skills") if isinstance(parsed, dict) else None
    if isinstance(skills_cfg, dict) and skills_cfg.get("project_discovery") is False:
        return []
    root = find_project_root()
    if root is None:
        return []
    if not is_project_root_trusted(root):
        return []
    return _candidate_project_skills_dirs(root)


def get_untrusted_project_skills_root() -> Optional[Tuple[Path, int]]:
    """When cwd's project has skills but is NOT trusted: (root, skill_count).

    Used by the CLI to print a one-line notice pointing at
    ``hermes skills trust``. Returns None when there is nothing to notify
    about (no project, no skills, already trusted, or discovery disabled).
    """
    parsed = _load_raw_config()
    skills_cfg = parsed.get("skills") if isinstance(parsed, dict) else None
    if isinstance(skills_cfg, dict) and skills_cfg.get("project_discovery") is False:
        return None
    root = find_project_root()
    if root is None or is_project_root_trusted(root):
        return None
    count = 0
    for d in _candidate_project_skills_dirs(root):
        try:
            count += sum(1 for _ in iter_skill_index_files(d, "SKILL.md"))
        except OSError:
            continue
    if count == 0:
        return None
    return root, count


def get_scan_ordered_skills_dirs() -> List[Path]:
    """All skill dirs in precedence order: project → local → external.

    First-wins name deduplication over this order gives project skills
    priority over profile-local and external ones.
    """
    dirs = list(get_project_skills_dirs())
    dirs.extend(get_all_skills_dirs())
    return dirs


# ── Project skill quarantine (scan-time injection defense) ────────────────
#
# Trust (`hermes skills trust`) is a REPO-level decision made once; the repo's
# skill content keeps changing underneath it with every pull. The hub install
# path runs skills_guard on install, but project skills are read straight from
# a checkout — without this gate a `git pull` could inject a malicious skill
# into an already-trusted repo with no scan anywhere (#48974).
#
# Every project SKILL.md's parent dir is scanned with the same skills_guard
# scanner the hub uses (content-hash cached, so the cost is one scan per
# skill per content change). A "dangerous" verdict quarantines the skill: it
# is excluded from the index, skills_list, skill_view, and slash commands.
# "caution" loads (matches hub behavior for prose-level keyword hits) — the
# quarantine is for high-confidence findings only.
#
# The scan cache lives under HERMES_HOME, never inside the repo (we don't
# write artifacts into the user's checkout).

_PROJECT_SCAN_SOURCE = "project-local"
# (skill_dir_resolved) -> quarantined bool, keyed per-process; scan_skill_cached
# already re-scans on content change via the bundle hash, this only avoids
# re-reading the attestation JSON on every index/list/view call in one run.
_PROJECT_QUARANTINE_CACHE: Dict[str, bool] = {}


def _project_scan_cache_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cache" / "project_skill_scans"


def is_quarantined_project_skill(skill_md) -> bool:
    """True when a project skill's scan verdict is ``dangerous``.

    Fail-closed: a scanner crash or missing scanner quarantines the skill
    (repo-sourced content with no completed scan must not load). Non-project
    callers should not call this — it scans unconditionally.
    """
    skill_dir = Path(skill_md).parent
    try:
        key = str(skill_dir.resolve())
    except OSError:
        key = str(skill_dir)
    cached = _PROJECT_QUARANTINE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        from tools.skills_guard import scan_skill_cached

        result, _prov = scan_skill_cached(
            skill_dir,
            source=_PROJECT_SCAN_SOURCE,
            cache_dir=_project_scan_cache_dir(),
        )
        quarantined = result.verdict == "dangerous"
        if quarantined:
            logger.warning(
                "Project skill quarantined (verdict=dangerous): %s — %s",
                skill_dir,
                result.summary,
            )
    except Exception:
        logger.warning(
            "Project skill scan failed — quarantining (fail closed): %s",
            skill_dir,
            exc_info=True,
        )
        quarantined = True
    _PROJECT_QUARANTINE_CACHE[key] = quarantined
    return quarantined


def _project_quarantine_cache_clear() -> None:
    """Test hook."""
    _PROJECT_QUARANTINE_CACHE.clear()


def iter_project_skill_files(project_dir: Path):
    """Yield non-quarantined SKILL.md files under a trusted project dir.

    The single iteration chokepoint for the project tier: every consumer
    (index, skills_list, slash commands) iterates through here so the
    quarantine cannot be bypassed by a new call site forgetting the check.
    """
    for skill_md in iter_skill_index_files(project_dir, "SKILL.md"):
        if is_quarantined_project_skill(skill_md):
            continue
        yield skill_md


def normalize_skill_lookup_name(identifier: str) -> str:
    """Normalize a skill identifier to a ``skill_view()``-safe relative path.

    Slash commands and cron jobs may store absolute paths to skills that live
    under ``~/.hermes/skills/`` (including via symlinks) or configured
    ``skills.external_dirs``. ``skill_view()`` rejects absolute names for
    security, so callers must translate trusted absolute paths to their
    relative form first.
    """
    raw_identifier = (identifier or "").strip()
    if not raw_identifier:
        return raw_identifier

    identifier_path = Path(raw_identifier).expanduser()
    if not identifier_path.is_absolute():
        return raw_identifier.lstrip("/")

    # Look the primary skills root up on tools.skills_tool at CALL time
    # (not via get_skills_dir()): callers and tests patch
    # ``tools.skills_tool.SKILLS_DIR`` and skill_view() itself resolves
    # against ``_skills_dir()`` — which honors that patch and otherwise
    # follows the live profile-scoped HERMES_HOME (the import-time
    # SKILLS_DIR is frozen to the launch home, #67277) — so normalization
    # must agree with the exact root skill_view() will enforce.  Import
    # deferred to avoid a module cycle (tools.skills_tool imports
    # agent.skill_utils).
    try:
        from tools import skills_tool as _skills_tool
        primary_root = _skills_tool._skills_dir()
    except Exception:
        primary_root = get_skills_dir()

    trusted_roots = [primary_root]
    try:
        trusted_roots.extend(get_project_skills_dirs())
    except Exception:
        pass
    try:
        trusted_roots.extend(get_external_skills_dirs())
    except Exception:
        pass

    # Prefer the lexical path under a trusted skill root before resolving
    # symlinks. Slash-command discovery can legitimately find a skill via
    # ~/.hermes/skills/<name> where <name> is a symlink to a checked-out
    # skill elsewhere. Resolving first turns that trusted visible path into
    # an arbitrary absolute path that skill_view() refuses to load.
    for root in trusted_roots:
        try:
            return str(identifier_path.relative_to(root))
        except ValueError:
            continue

    try:
        return str(identifier_path.resolve().relative_to(primary_root.resolve()))
    except Exception:
        logger.debug(
            "Skill identifier %r is an absolute path outside trusted skills "
            "roots — passing through unchanged (skill_view will reject it)",
            raw_identifier,
        )
        return raw_identifier


def _resolve_for_skill_ownership(path) -> Path:
    path_obj = path if isinstance(path, Path) else Path(str(path))
    try:
        return path_obj.expanduser().resolve()
    except (OSError, RuntimeError):
        return path_obj.expanduser().absolute()


def is_external_skill_path(path) -> bool:
    """Return True when ``path`` lives under a configured external skills dir.

    ``skills.external_dirs`` are externally owned: Hermes can discover and view
    their skills, and foreground user-directed tool calls may still edit them,
    but autonomous lifecycle maintenance must treat them as read-only. This
    helper centralizes the ownership boundary so curator/reporting/tool paths do
    not each need to re-interpret the config.
    """
    candidate = _resolve_for_skill_ownership(path)
    roots: List[Path] = list(get_external_skills_dirs())
    # Trusted project-local dirs are repo-owned — same read-only boundary
    # for autonomous lifecycle maintenance as configured external dirs.
    try:
        roots.extend(get_project_skills_dirs())
    except Exception:
        pass
    for root in roots:
        resolved_root = _resolve_for_skill_ownership(root)
        try:
            candidate.relative_to(resolved_root)
            return True
        except ValueError:
            continue
    return False


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
        # Gateway-channel gate (maintainer-directed, skills-index slim):
        # list of session platforms (e.g. ["msteams"]) the skill is FOR.
        # Unlike top-level ``platforms:`` (host OS), this hides the skill
        # from the index on every other channel — the teams-meeting
        # pipeline has no business in a desktop or telegram session's
        # index. Empty/absent = visible everywhere (backward compat).
        "session_platforms": hermes.get("session_platforms", []),
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
    config = _load_raw_config()

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

SKILL_PROMPT_DESC_LIMIT = 60


def _normalize_skill_description(frontmatter: Dict[str, Any]) -> str:
    """Normalize a skill's description field for comparison/truncation."""
    raw_desc = frontmatter.get("description", "")
    return str(raw_desc).strip().strip("'\"") if raw_desc else ""


def extract_skill_description(frontmatter: Dict[str, Any]) -> str:
    """Extract a system-prompt-length description from parsed frontmatter."""
    desc = _normalize_skill_description(frontmatter)
    if not desc:
        return ""
    if len(desc) > SKILL_PROMPT_DESC_LIMIT:
        return desc[:SKILL_PROMPT_DESC_LIMIT - 3] + "..."
    return desc


def is_skill_description_truncated_for_prompt(frontmatter: Dict[str, Any]) -> bool:
    """True when the description will be truncated in the system prompt skill index."""
    desc = _normalize_skill_description(frontmatter)
    return len(desc) > SKILL_PROMPT_DESC_LIMIT


# ── File iteration ────────────────────────────────────────────────────────


def iter_skill_index_files(skills_dir: Path, filename: str):
    """Walk skills_dir yielding sorted paths matching *filename*.

    Excludes Hermes metadata, VCS, virtualenv/dependency, cache, and skill
    support directories. Support directories (references/templates/assets/
    scripts) can contain arbitrary markdown and even archived package
    ``SKILL.md`` files, but they are progressive-disclosure data loaded through
    ``skill_view(..., file_path=...)`` rather than active skill roots.

    M2 org mirrors (``_org/``): TOKEN-GATED resolution. Only the active org's
    subdir (per the sync-client-written ``.active_org`` marker) is walked;
    every other ``_org/<id>/`` (stale mirror from a previous org, or no
    marker at all) is pruned — leave an org and its skills stop resolving,
    without any manual cleanup.
    """
    skills_dir_str = str(skills_dir)
    active_org = read_active_org_id(skills_dir)
    org_root = os.path.join(skills_dir_str, ORG_MIRROR_DIR_NAME)
    matches: list[str] = []
    for root, dirs, files in os.walk(skills_dir_str, followlinks=True):
        has_skill_md = "SKILL.md" in files
        if root == skills_dir_str and ORG_MIRROR_DIR_NAME in dirs and active_org is None:
            dirs.remove(ORG_MIRROR_DIR_NAME)
        elif root == org_root:
            # Inside _org/: descend ONLY into the active org's mirror.
            dirs[:] = [d for d in dirs if d == active_org]
        dirs[:] = [
            d
            for d in dirs
            if d not in EXCLUDED_SKILL_DIRS
            and not (has_skill_md and d in SKILL_SUPPORT_DIRS)
        ]
        if filename in files:
            matches.append(os.path.join(root, filename))
    for path in sorted(matches):
        yield Path(path)


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
