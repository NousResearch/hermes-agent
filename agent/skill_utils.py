"""Lightweight skill metadata utilities shared by prompt_builder and skills_tool.
Import-light by design: no tool registry, CLI config, or provider resolution."""

import ast
import errno as _errno
import hashlib
import logging
import os
import re
import stat as _stat
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from hermes_constants import (
    get_config_path,
    get_skills_dir,
    get_subprocess_home,
)

logger = logging.getLogger(__name__)

PLATFORM_MAP = {"macos": "darwin", "linux": "linux", "windows": "win32"}

# Where profile-distribution installs stage and set aside a dir they swap in (same parent, so the
# rename stays atomic); a backup a lock keeps there must never be discovered as a second copy.
DIST_SWAP_DIR = ".hermes-dist"

EXCLUDED_SKILL_DIRS = frozenset((
    ".git", ".github", ".hub", ".archive", ".curator_backups", ".locks", DIST_SWAP_DIR,
    ".venv", "venv", "node_modules", "site-packages", "__pycache__",
    ".tox", ".nox", ".pytest_cache", ".mypy_cache", ".ruff_cache",
))

# Progressive-disclosure support dirs inside a skill package: loaded explicitly
# via skill_view(skill, file_path=...), never scanned as standalone skills.
SKILL_SUPPORT_DIRS = frozenset(("references", "templates", "assets", "scripts"))

def is_excluded_skill_path(path, *, root: Optional[Path] = None) -> bool:
    """True if *path* should be skipped by skill scanners (VCS/dependency/cache
    dirs + support packages). Apply to every SKILL.md from a direct ``rglob``."""
    parts = PurePath(str(path)).parts
    return any(part in EXCLUDED_SKILL_DIRS for part in parts) or is_skill_support_path(path, root=root)


def is_skill_support_path(path, *, root: Optional[Path] = None) -> bool:
    """True if *path* is under a support dir sitting directly inside a skill root
    (``skills/scripts/foo`` stays discoverable: no ``SKILL.md`` above ``scripts``)."""
    path_obj = path if isinstance(path, Path) else Path(str(path))
    parts = path_obj.parts
    base = root if root is not None and not path_obj.is_absolute() else Path()
    # Only components before the leaf can be containing support directories.
    return any(
        part in SKILL_SUPPORT_DIRS and (base / Path(*parts[:idx]) / "SKILL.md").exists()
        for idx, part in enumerate(parts[:-1])
        if idx > 0
    )


_yaml_load_fn = None


def yaml_load(content: str):
    """Parse YAML with the shared safe loader, imported lazily."""
    global _yaml_load_fn
    if _yaml_load_fn is None:
        from hermes_yaml import safe_load
        _yaml_load_fn = safe_load
    return _yaml_load_fn(content)


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown; returns (frontmatter_dict, body).
    Malformed YAML falls back to key:value line splitting. A leading UTF-8 BOM
    (Windows editors) is stripped first or it would defeat the ``---`` fence check."""
    content = content.removeprefix("\ufeff")
    end_match = re.search(r"\n---\s*\n", content[3:]) if content.startswith("---") else None
    if not end_match:
        return {}, content
    yaml_content = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :]
    frontmatter: dict[str, Any] = {}
    try:
        parsed = yaml_load(yaml_content)
        if isinstance(parsed, dict):
            frontmatter = parsed
    except Exception:
        for line in yaml_content.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                frontmatter[key.strip()] = value.strip()
    return frontmatter, body


def skill_matches_platform_list(platforms: Any) -> bool:
    """Return True when *platforms* is compatible with the current OS."""
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


def skill_matches_platform(frontmatter: dict[str, Any]) -> bool:
    """True when the skill's ``platforms:`` list (absent = all) matches this OS."""
    return skill_matches_platform_list(frontmatter.get("platforms"))


# An ``environments:`` tag is a *relevance* gate for offer surfaces (index,
# autocomplete, slash commands), not a compatibility gate: an explicit load
# (skill_view, --skills) always succeeds. Detection is cached per process.

_ENV_DETECT_CACHE: dict[str, bool] = {}


def _detect_kanban() -> bool:
    # Mirror tools/kanban_tools.py: a dispatcher-spawned worker (env vars, but
    # only when this execution OWNS the task — delegate children / in-process
    # cron see the worker's vars) or a profile opted into the kanban toolset.
    if os.getenv("HERMES_KANBAN_TASK") or os.getenv("HERMES_KANBAN_BOARD"):
        try:
            from agent.delegation_context import is_dispatcher_owned_worker_context
            owned = is_dispatcher_owned_worker_context()
        except Exception:
            owned = True
        if owned:
            return True
    try:
        from tools.kanban_tools import _profile_has_kanban_toolset
        return bool(_profile_has_kanban_toolset())
    except Exception:
        return False


def _detect_docker() -> bool:
    try:
        from hermes_constants import is_container
        return is_container()
    except Exception:
        return False


_ENV_DETECTORS: dict[str, Callable[[], bool]] = {
    "kanban": _detect_kanban, "docker": _detect_docker,
    "s6": lambda: os.path.isdir("/run/s6") or os.path.isdir("/package/admin/s6-overlay"),  # s6-overlay is PID 1 in the image
}


def _detect_environment(env: str) -> bool:
    """True when the named runtime environment is active (unknown => True).
    Cached per process EXCEPT ``kanban``: that verdict is context-dependent
    (delegate children / in-process cron see the worker's vars), so a
    process-wide cache would leak the first asker's answer to the others."""
    if env != "kanban" and env in _ENV_DETECT_CACHE:
        return _ENV_DETECT_CACHE[env]
    detector = _ENV_DETECTORS.get(env)
    result = detector() if detector else True
    _ENV_DETECT_CACHE[env] = result
    return result


def skill_matches_environment(frontmatter: dict[str, Any]) -> bool:
    """True when ANY declared ``environments:`` tag is active (absent = all;
    unknown tags fail open). Offer-time filter only."""
    environments = frontmatter.get("environments")
    if not environments:
        return True
    tags = [str(env).lower().strip() for env in (environments if isinstance(environments, list) else [environments])]
    return any(_detect_environment(tag) for tag in tags if tag)


def skill_matches_apps(frontmatter: dict[str, Any]) -> bool:
    """True when every app named in ``requires_apps:`` has a registered declaration this host satisfies.

    Names resolve through ``hermes_platform.declaration`` (registered by whoever owns the server,
    e.g. the plugin loader); the check is the same ``availability()`` the MCP check_fn uses. An
    unknown name hides the skill (fail closed). Offer-time filter, like ``environments:``.
    """
    names = frontmatter.get("requires_apps")
    if not names:
        return True
    from hermes_platform import declaration
    from hermes_platform.resolver.availability import availability

    for name in names if isinstance(names, list) else [names]:
        decl = declaration.lookup(str(name).strip())
        if decl is None or decl.app is None:
            return False
        if not availability(decl).offerable:
            return False
    return True


_RAW_CONFIG_CACHE: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
# A multiplexed watcher rotates through up to ~32 published profile/project scopes plus
# the ambient one (#141610), so a miss must not drop the whole cache — only a stale
# signature entry. The bound retires entries a config's own edits leave behind.
_RAW_CONFIG_CACHE_MAX = 128
# Guards every cache read/publish/evict: concurrent tool threads share it, and an unlocked
# ``pop(next(iter(...)))`` races another thread's insert (RuntimeError) or pop (KeyError).
_RAW_CONFIG_CACHE_LOCK = threading.Lock()


def _raw_config_cache_clear() -> None:
    """Test hook — drop the shared raw config cache."""
    with _RAW_CONFIG_CACHE_LOCK:
        _RAW_CONFIG_CACHE.clear()


def _config_cache_key(config_path: Path) -> Optional[tuple[Any, ...]]:
    """``(path, *backend cache version)`` identity of config.yaml, or None when unreadable/absent."""
    from hermes_cli.config_backend import config_cache_version_or_none
    version = config_cache_version_or_none(config_path)
    return None if version is None else (str(config_path), *version)


def _load_raw_config() -> dict[str, Any]:
    """Read config.yaml with a version-keyed cache (no hermes_cli.config import)."""
    from agent.safe_worker_policy import worker_config_snapshot
    from hermes_cli.config_backend import config_exists, read_config_doc

    snapshot = worker_config_snapshot()
    if snapshot is not None:
        return snapshot
    config_path = get_config_path()
    if not config_exists(config_path):
        return {}
    cache_key = _config_cache_key(config_path)
    with _RAW_CONFIG_CACHE_LOCK:
        cached = _RAW_CONFIG_CACHE.get(cache_key) if cache_key is not None else None
    if cached is not None:
        return cached
    try:
        parsed = read_config_doc(config_path)
    except Exception as e:
        logger.debug("Could not read skill config %s: %s", config_path, e)
        return {}
    if not isinstance(parsed, dict):
        return {}
    if cache_key is not None:
        with _RAW_CONFIG_CACHE_LOCK:
            _RAW_CONFIG_CACHE[cache_key] = parsed
            while len(_RAW_CONFIG_CACHE) > _RAW_CONFIG_CACHE_MAX:
                _RAW_CONFIG_CACHE.pop(next(iter(_RAW_CONFIG_CACHE)))
    return parsed


def _skills_cfg() -> Optional[dict[str, Any]]:
    """The ``skills:`` mapping from config.yaml, or None when absent/malformed."""
    skills_cfg = _load_raw_config().get("skills")
    return skills_cfg if isinstance(skills_cfg, dict) else None


def _skills_cfg_get(key: str) -> Any:
    """``skills.<key>`` from config.yaml, or None when the section is absent/malformed."""
    skills_cfg = _skills_cfg()
    return skills_cfg.get(key) if skills_cfg is not None else None


def _expand_path(entry: str) -> Path:
    """Expand ``~`` and ``${VAR}`` in a config path entry."""
    return Path(os.path.expanduser(os.path.expandvars(entry)))


def _home_relative(p: Path) -> Path:
    """Anchor a relative config path at HERMES_HOME; absolute paths pass through."""
    from hermes_constants import get_hermes_home
    return p if p.is_absolute() else get_hermes_home() / p


# Never disableable: `hermes-agent` is the agent's own operating manual and the
# system prompt points at it unconditionally.
ESSENTIAL_SKILLS: frozenset = frozenset({"hermes-agent"})


def get_disabled_skill_names(platform: str | None = None) -> set[str]:
    """Disabled skill names from config.yaml: global list ∪ platform list
    (*platform* defaults to ``HERMES_PLATFORM`` / ``HERMES_SESSION_PLATFORM``)."""
    skills_cfg = _skills_cfg()
    if skills_cfg is None:
        return set()
    from gateway.session_context import get_session_env
    resolved_platform = platform or os.getenv("HERMES_PLATFORM") or get_session_env("HERMES_SESSION_PLATFORM")
    disabled = _normalize_string_set(skills_cfg.get("disabled"))
    platform_disabled = (skills_cfg.get("platform_disabled") or {}).get(resolved_platform) if resolved_platform else None
    if platform_disabled is not None:
        disabled |= _normalize_string_set(platform_disabled)
    return disabled - ESSENTIAL_SKILLS


def parse_config_string_list(value) -> list[str]:
    """Normalize a config value that may hold a JSON-array string into a list.
    ``hermes config set`` stores lists as quoted JSON/Python-literal strings;
    treating one as a single name would silently filter nothing. A scalar
    string still means one name.

    See #13026, #86661.
    """
    if isinstance(value, str):
        if value.strip().startswith("["):
            try:
                parsed = ast.literal_eval(value.strip())
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        return [value]
    return [str(item) for item in value] if isinstance(value, (list, tuple, set, frozenset)) else []


def _normalize_string_set(values) -> set[str]:
    return {name.strip() for name in parse_config_string_list(values) if name.strip()}


# config identity -> resolved external dirs. Called once per skill during
# banner / tool-registry scans; re-resolving each time dominated cold-start.
_EXTERNAL_DIRS_CACHE: dict[tuple[str, int, int, int, int], list[Path]] = {}


def _external_dirs_cache_clear() -> None:
    """Test hook — drop the in-process cache."""
    _EXTERNAL_DIRS_CACHE.clear()
    _raw_config_cache_clear()


def _config_str_list(raw) -> list[str]:
    """A scalar-or-list config entry as a list of stripped non-empty strings."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [e for e in (str(entry).strip() for entry in raw) if e]


def get_external_skills_dirs(*, strict: Optional[bool] = None) -> list[Path]:
    """Validated, deduplicated ``skills.external_dirs``.

    Entries are ``~``/``${VAR}`` expanded, relative to HERMES_HOME; the local skills dir is
    skipped. Fail-closed (B5-F R1): a configured entry that is *not* a usable directory is
    never dropped in silence — it is classified by errno, logged at WARNING and reported
    through :func:`external_root_problems`. With *strict* (default: ``skills.fail_closed``)
    an unreachable root raises instead of shrinking the effective skill set.
    """
    if strict is None:
        strict = fail_closed_enabled()
    from hermes_cli.config_backend import config_exists
    config_path = get_config_path()
    if not config_exists(config_path):
        return []
    full_key = _config_cache_key(config_path)
    cache_key = full_key
    cached = _EXTERNAL_DIRS_CACHE.get(cache_key) if cache_key is not None else None
    if cached is not None:
        return list(cached)  # copy so callers can't mutate the cache
    skills_cfg = _skills_cfg()
    if skills_cfg is None:
        return []
    local_skills = get_skills_dir().resolve()
    result: list[Path] = []
    problems: list[ExternalRootProblem] = []
    for entry in _config_str_list(skills_cfg.get("external_dirs")):
        p = _home_relative(_expand_path(entry))
        try:
            p = p.resolve()
        except OSError as e:
            # A root that cannot even be resolved is NOT a root we may forget about: it is a
            # configured entry whose skills silently vanished from the effective set (B5-F R1/N01).
            problems.append(ExternalRootProblem(p, "unresolved", e.errno, e.strerror or str(e)))
            continue
        if p == local_skills or p in result:
            continue
        status, err_no, err_text = classify_external_root(p)
        if status == "dir":
            result.append(p)
            continue
        # Fail-closed: every non-dir entry is reported at WARNING and kept out of any cache, so
        # a transient failure can never become a permanent silent drop (R1.1/R1.2/R1.4).
        problems.append(ExternalRootProblem(p, status, err_no, err_text))
    del _EXTERNAL_ROOT_PROBLEMS[:]
    _EXTERNAL_ROOT_PROBLEMS.extend(problems)
    for prob in problems:
        logger.warning("Configured external skills dir is unreachable (%s%s): %s — the skills it "
                       "provides are NOT part of the effective set", prob.status,
                       f" errno={prob.errno}" if prob.errno is not None else "", prob.path)
    if problems:
        # Negative cache would freeze the failure: next call must re-probe (R1.4).
        _EXTERNAL_DIRS_CACHE.pop(cache_key, None)
        if strict:
            # R1.3: leaving the effective set short of what was configured must not be a
            # silent success. The escape hatch is `skills.fail_closed: false`, not a debug log.
            raise SkillResolutionError(
                "Configured external skills dirs are unreachable: "
                + ", ".join(f"{p.path} ({p.status})" for p in problems),
                kind="external_root_unreachable")
        return result
    if cache_key is not None:
        _EXTERNAL_DIRS_CACHE[cache_key] = list(result)
    return result


def get_skill_create_dir() -> Optional[Path]:
    """Configured ``skills.create_dir`` (need not exist yet), or None when unset;
    relative to HERMES_HOME; a value equal to the local skills dir counts as unset."""
    raw = _skills_cfg_get("create_dir")
    entry = str(raw).strip() if raw and isinstance(raw, (str, os.PathLike)) else ""
    if not entry:
        return None
    p = _home_relative(_expand_path(entry))
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
    """User-facing path where new skills are created (``~/`` shorthand when
    possible); tool schema descriptions and prompts follow ``skills.create_dir``."""
    from hermes_constants import display_hermes_home
    create_dir = get_skill_create_dir()
    if create_dir is None:
        return f"{display_hermes_home()}/skills/"
    if create_dir.is_relative_to(Path.home()):
        return "~/" + create_dir.relative_to(Path.home()).as_posix() + "/"
    return create_dir.as_posix() + "/"


# Cross-directory precedence, lowest tier wins: trusted project > local profile > skills.create_dir >
# skills.external_dirs. Inside ONE tier two different skills sharing a name stay ambiguous — refused,
# never guessed (59da8ec4e) — while identical copies under one root resolve to the shallowest.
TIER_PROJECT, TIER_LOCAL, TIER_CREATE_DIR, TIER_EXTERNAL = range(4)
# Leading words of every same-tier refusal (skill_view error, preload/cron label) — one spelling.
AMBIGUOUS_SKILL_PREFIX = "Ambiguous skill name "
# (shadowed path, *sorted higher-tier paths) already judged: the identity check (it hashes both
# SKILL.md files) and its one-time warning run once per pairing, not on every catalog resolve.
_SHADOW_CHECKED: set[tuple[str, ...]] = set()


def get_skill_search_roots(local: Optional[Path] = None, *, include_project: bool = True) -> list[tuple[int, Path]]:
    """``(tier, dir)`` for every skill root in precedence order — the ONE ordering the skills list,
    prompt index, slash commands, skill_view, preload and cron share. *local* overrides the profile
    skills dir (skills_tool passes its live root); that entry is kept even when missing."""
    roots = [(TIER_PROJECT, d) for d in get_project_skills_dirs()] if include_project else []
    roots.append((TIER_LOCAL, Path(local) if local is not None else get_skills_dir()))
    create_dir = get_skill_create_dir()
    if create_dir is not None and create_dir.is_dir():
        roots.append((TIER_CREATE_DIR, create_dir))
    roots += [(TIER_EXTERNAL, d) for d in get_external_skills_dirs()]
    seen: set[Path] = set()
    return [(t, d) for t, d in roots if not (d in seen or seen.add(d))]


def get_all_skills_dirs() -> list[Path]:
    """Skill dirs: local ``~/.hermes/skills/`` first, then create_dir, then external.
    Trusted project dirs are NOT included (higher precedence; see get_project_skills_dirs)."""
    return [d for _tier, d in get_skill_search_roots(include_project=False)]


def provably_same_skill(skill_mds) -> bool:
    """True only when every path is the SAME skill: one resolved file (symlink view) or byte-identical
    content (copy). Anything else is two different skills sharing a name, and picking one by depth
    would let ``<root>/evil`` (``name: github``) shadow the real one."""
    try:
        if len({os.path.realpath(p) for p in skill_mds}) == 1:
            return True
        return len({hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in skill_mds}) == 1
    except OSError:
        return False


def skill_candidate_rank(skill_md, root) -> tuple[bool, int]:
    """Same-root order of identical copies: a real SKILL.md beats a legacy flat ``<name>.md``,
    then the shallower path wins (shared by skill_view and :func:`resolve_skill_catalog`)."""
    skill_md = Path(skill_md)
    return (skill_md.name != "SKILL.md", len(skill_md.relative_to(root).parts))


def pick_skill_candidate(candidates) -> tuple[Optional[int], list[int]]:
    """Winner index among one identifier's ``(tier, root, rank, skill_md)`` candidates, plus the
    winning tier's contender indexes. The lowest tier wins; inside it a lone candidate wins, identical
    copies under one root resolve to the strictly best ``rank``, anything else is ambiguous (None)."""
    top = min(c[0] for c in candidates)
    contenders = [i for i, c in enumerate(candidates) if c[0] == top]
    if len(contenders) > 1 and len({candidates[i][1] for i in contenders}) == 1 and provably_same_skill(
            [candidates[i][3] for i in contenders]):
        ranked = sorted(contenders, key=lambda i: candidates[i][2])
        if candidates[ranked[0]][2] != candidates[ranked[1]][2]:
            return ranked[0], contenders
    return (contenders[0] if len(contenders) == 1 else None), contenders


def is_disabled_entry(entry: dict[str, Any], disabled: set[str]) -> bool:
    """``skills.disabled`` matches a resolved catalog entry by its declared name OR its ``load_name`` —
    the exact path a same-tier duplicate's list/config/web row shows (``a/one``) and saves. A unique
    copy elsewhere that merely sits at the same relative path is not matched."""
    return not disabled.isdisjoint({str(entry["name"]), entry.get("load_name")} - {None})


def resolve_skill_catalog(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copies of scanned skills (each with ``name``, ``tier``, ``root`` and ``path`` = its SKILL.md)
    annotated with what skill_view() resolves, using its aliases (declared name, directory name, path
    relative to the root) and :func:`pick_skill_candidate`. Adds ``relative_path`` and ``status``:
    ``unique`` (``load_name`` = name), ``ambiguous`` (``load_name`` = the exact relative path, or None
    when even that is shared) or ``shadowed`` (a higher tier owns the name: hidden, warned once)."""
    out = [dict(e) for e in entries]
    owners: dict[str, list[int]] = {}
    for i, e in enumerate(out):
        skill_dir = Path(e["path"]).parent
        e["relative_path"] = skill_dir.relative_to(e["root"]).as_posix()
        for alias in {str(e["name"]), skill_dir.name, e["relative_path"]}:
            owners.setdefault(alias, []).append(i)
    winner: dict[str, Optional[int]] = {}
    for alias, idxs in owners.items():
        won, _ = pick_skill_candidate([
            (out[j]["tier"], str(out[j]["root"]), skill_candidate_rank(out[j]["path"], out[j]["root"]), out[j]["path"])
            for j in idxs])
        winner[alias] = None if won is None else idxs[won]
    for i, e in enumerate(out):
        name, rel = str(e["name"]), e["relative_path"]
        higher = [j for j in owners[name] if out[j]["tier"] < e["tier"]]
        if higher or winner[name] not in (None, i):  # lower tier, or an identical same-root copy
            e.update(status="shadowed", load_name=None)
            # A symlink view or byte-identical copy of the winner hides nothing worth a warning.
            key = (str(e["path"]), *sorted(str(out[j]["path"]) for j in higher))
            if higher and key not in _SHADOW_CHECKED:
                _SHADOW_CHECKED.add(key)
                if not any(provably_same_skill([e["path"], out[j]["path"]]) for j in higher):
                    logger.warning("Skill '%s' at %s is shadowed by a higher-precedence copy "
                                   "(project > local > create_dir > external_dirs)", name, e["path"])
        elif winner[name] == i:
            e.update(status="unique", load_name=name)
        else:
            e.update(status="ambiguous", load_name=rel if winner[rel] == i else None)
    return out


# Project-local skills (<root>/.hermes/skills, <root>/.agents/skills; root = nearest
# .git ancestor) are a prompt-injection vector if auto-sourced from any clone, so
# they load only when the root is in ``skills.trusted_project_dirs``; then they
# override same-named profile/bundled skills. cwd + trust list are session-fixed
# so the skills index stays byte-stable.

PROJECT_SKILLS_SUBDIRS = (os.path.join(".hermes", "skills"), os.path.join(".agents", "skills"))

_PROJECT_ROOT_MAX_DEPTH = 64  # walk-up bound for pathological cwds


def find_project_root(start: Optional[Path] = None) -> Optional[Path]:
    """Nearest ancestor containing ``.git`` (dir or worktree file), or None.
    Without *start*, the surface's effective working directory wins over the process cwd — the same
    ladder every other cwd consumer reads (``resolve_agent_cwd``: session-bound cwd, then the scope's
    ``TERMINAL_CWD``, then the process cwd). The session cwd comes first because a multi-session host
    (TUI/desktop gateway) pins each session's workspace there while its terminal scope resolves a
    placeholder ``terminal.cwd`` to ``$HOME``; reading only the scope made every project skill invisible
    on those surfaces (#114359). ``TERMINAL_CWD`` is the per-surface workdir the terminal tool and cron
    jobs use (a cron job sets it from its per-job ``workdir`` without chdir'ing the scheduler process),
    which lets non-interactive surfaces inherit a prior interactive trust decision by project identity —
    and a surface with no workdir in a trusted repo simply resolves no project and loads nothing (#48975).
    """
    try:
        if start is None:
            from agent.runtime_cwd import resolve_agent_cwd
            start = resolve_agent_cwd()
        cur = Path(start).resolve()
    except OSError:
        return None
    home = Path.home().resolve()
    try:
        for _ in range(_PROJECT_ROOT_MAX_DEPTH):
            if (cur / ".git").exists():
                # A dotfiles checkout AT home would make every session
                # project-scoped; treat home itself as non-project.
                return None if cur == home else cur
            if cur.parent == cur:
                return None
            cur = cur.parent
    except OSError:
        pass
    return None


def _project_trusted_dirs_from_config() -> set[Path]:
    """Resolved set of trusted project roots from ``skills.trusted_project_dirs``."""
    result: set[Path] = set()
    for entry in _config_str_list(_skills_cfg_get("trusted_project_dirs")):
        try:
            result.add(_expand_path(entry).resolve())
        except OSError:
            continue
    return result


def is_project_root_trusted(root: Path) -> bool:
    """True when *root* is listed in ``skills.trusted_project_dirs``."""
    try:
        return Path(root).resolve() in _project_trusted_dirs_from_config()
    except OSError:
        return False


def _candidate_project_skills_dirs(root: Path) -> list[Path]:
    """Existing skill dirs under *root*, excluding the profile's own skills dir
    (HERMES_HOME itself may live inside a git checkout)."""
    local_skills = get_skills_dir().resolve()
    dirs: list[Path] = []
    for cand in (root / sub for sub in PROJECT_SKILLS_SUBDIRS):
        try:
            if cand.is_dir() and cand.resolve() != local_skills:
                dirs.append(cand.resolve())
        except OSError:
            continue
    return dirs


def _current_project_root(trusted: bool) -> Optional[Path]:
    """cwd's project root when discovery is on and its trust state == *trusted*."""
    if _skills_cfg_get("project_discovery") is False:
        return None
    root = find_project_root()
    return root if root is not None and is_project_root_trusted(root) == trusted else None


def get_project_skills_dirs() -> list[Path]:
    """Trusted project-local skill dirs for the current cwd (may be empty)."""
    root = _current_project_root(trusted=True)
    return _candidate_project_skills_dirs(root) if root is not None else []


def get_untrusted_project_skills_root() -> Optional[tuple[Path, int]]:
    """(root, skill_count) when cwd's project has skills but is NOT trusted, else None."""
    root = _current_project_root(trusted=False)
    count = 0
    for d in _candidate_project_skills_dirs(root) if root is not None else ():
        try:
            count += sum(1 for _ in iter_skill_index_files(d, "SKILL.md"))
        except OSError:
            continue
    return (root, count) if count else None


# Scan-time injection defense: trust is a repo-level decision made once, but a
# `git pull` could inject a malicious skill into an already-trusted repo. Every
# project SKILL.md is scanned with the hub's skills_guard scanner (content-hash
# cached under HERMES_HOME, never inside the repo); "dangerous" excludes the
# skill from index, list, view and slash commands ("caution" loads, as on the hub).

# ── Project skill quarantine (scan-time injection defense) ──────────────── Trust (`hermes skills trust`)
# is a REPO-level decision made once; the repo's skill content keeps changing underneath it with every pull.
# The hub install path runs skills_guard on install, but project skills are read straight from a checkout —
# without this gate a `git pull` could inject a malicious skill into an already-trusted repo with no scan
# anywhere (#48974). Every project SKILL.md's parent dir is scanned with the same skills_guard scanner the
# hub uses (content-hash cached, so the cost is one scan per skill per content change). A "dangerous"
# verdict quarantines the skill: it is excluded from the index, skills_list, skill_view, and slash commands.
# "caution" loads (matches hub behavior for prose-level keyword hits) — the quarantine is for
# high-confidence findings only. The scan cache lives under HERMES_HOME, never inside the repo (we don't
# write artifacts into the user's checkout).
_PROJECT_SCAN_SOURCE = "project-local"
_PROJECT_QUARANTINE_CACHE: dict[str, bool] = {}  # skill_dir -> quarantined


def is_quarantined_project_skill(skill_md) -> bool:
    """True when a project skill's scan verdict is ``dangerous``. Fail-closed: a
    scanner crash or missing scanner quarantines the skill. Scans
    unconditionally — non-project callers should not call this."""
    skill_dir = Path(skill_md).parent
    try:
        key = str(skill_dir.resolve())
    except OSError:
        key = str(skill_dir)
    if key in _PROJECT_QUARANTINE_CACHE:
        return _PROJECT_QUARANTINE_CACHE[key]
    try:
        from tools.skills_guard import scan_skill_cached
        from hermes_constants import get_hermes_home
        cache_dir = get_hermes_home() / "cache" / "project_skill_scans"
        result, _prov = scan_skill_cached(skill_dir, source=_PROJECT_SCAN_SOURCE, cache_dir=cache_dir)
        quarantined = result.verdict == "dangerous"
        if quarantined:
            logger.warning("Project skill quarantined (verdict=dangerous): %s — %s", skill_dir, result.summary)
    except Exception:
        logger.warning("Project skill scan failed — quarantining (fail closed): %s", skill_dir, exc_info=True)
        quarantined = True
    _PROJECT_QUARANTINE_CACHE[key] = quarantined
    return quarantined


def iter_project_skill_files(project_dir: Path):
    """Yield non-quarantined SKILL.md files under a trusted project dir — the
    single iteration chokepoint for the project tier, so the quarantine cannot
    be bypassed by a call site forgetting the check."""
    yield from (p for p in iter_skill_index_files(project_dir, "SKILL.md") if not is_quarantined_project_skill(p))


def normalize_skill_lookup_name(identifier: str) -> str:
    """Translate a trusted absolute skill path (slash commands / cron may store
    them) to the relative form ``skill_view()`` accepts."""
    raw_identifier = (identifier or "").strip()
    if not raw_identifier:
        return raw_identifier
    identifier_path = Path(raw_identifier).expanduser()
    if not identifier_path.is_absolute():
        return raw_identifier.lstrip("/")
    # Resolve the primary root via tools.skills_tool at CALL time: tests patch
    # ``tools.skills_tool.SKILLS_DIR`` and skill_view() enforces ``_skills_dir()``
    # (which follows the live profile-scoped HERMES_HOME), so normalization
    # must agree with that exact root. Import deferred (cycle).
    try:
        # See #67277.
        from tools import skills_tool as _skills_tool
        primary_root = _skills_tool._skills_dir()
    except Exception:
        primary_root = get_skills_dir()
    trusted_roots = [primary_root]
    for getter in (get_project_skills_dirs, get_external_skills_dirs):
        try:
            trusted_roots.extend(getter())
        except Exception:
            pass
    # Prefer the lexical path under a trusted root before resolving symlinks:
    # ~/.hermes/skills/<name> may be a symlink to a checkout elsewhere, and
    # resolving first would turn that trusted path into one skill_view rejects.
    for root in trusted_roots:
        if identifier_path.is_relative_to(root):
            return str(identifier_path.relative_to(root))
    try:
        return str(identifier_path.resolve().relative_to(primary_root.resolve()))
    except Exception:
        logger.debug("Skill identifier %r is an absolute path outside trusted skills "
                     "roots — passing through unchanged (skill_view will reject it)", raw_identifier)
        return raw_identifier


def _resolve_for_skill_ownership(path) -> Path:
    path_obj = path if isinstance(path, Path) else Path(str(path))
    try:
        return path_obj.expanduser().resolve()
    except (OSError, RuntimeError):
        return path_obj.expanduser().absolute()


def is_external_skill_path(path) -> bool:
    """True when ``path`` lives under an external or trusted project skills dir.
    Those are externally owned: autonomous lifecycle maintenance treats them as
    read-only (user-directed tool calls may still edit them)."""
    candidate = _resolve_for_skill_ownership(path)
    roots: list[Path] = list(get_external_skills_dirs())
    try:
        roots.extend(get_project_skills_dirs())
    except Exception:
        pass
    return any(candidate.is_relative_to(_resolve_for_skill_ownership(root)) for root in roots)


def _hermes_metadata(frontmatter: dict[str, Any]) -> dict[str, Any]:
    """``metadata.hermes`` mapping from frontmatter, or ``{}`` when malformed."""
    metadata = frontmatter.get("metadata")
    hermes = metadata.get("hermes") if isinstance(metadata, dict) else None
    return hermes if isinstance(hermes, dict) else {}


# ``session_platforms`` is the gateway-channel gate: session platforms the skill
# is FOR (hidden from the index elsewhere), unlike ``platforms:`` (host OS).
_CONDITION_KEYS = ("fallback_for_toolsets", "requires_toolsets", "fallback_for_tools", "requires_tools", "session_platforms")


def extract_skill_conditions(frontmatter: dict[str, Any]) -> dict[str, list]:
    """Extract conditional activation fields from parsed frontmatter (absent = ``[]``)."""
    hermes = _hermes_metadata(frontmatter)
    return {key: hermes.get(key, []) for key in _CONDITION_KEYS}


def extract_skill_config_vars(frontmatter: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract ``metadata.hermes.config`` declarations (key/description/default/prompt).
    Entries missing ``key`` or ``description`` are skipped; ``prompt`` defaults to the description."""
    raw = _hermes_metadata(frontmatter).get("config")
    if isinstance(raw, dict):
        raw = [raw]
    if not raw or not isinstance(raw, list):
        return []
    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        desc = str(item.get("description", "")).strip()
        if not key or key in result or not desc:
            continue
        entry: dict[str, Any] = {"key": key, "description": desc}
        if item.get("default") is not None:
            entry["default"] = item["default"]
        prompt_text = item.get("prompt")
        entry["prompt"] = prompt_text.strip() if isinstance(prompt_text, str) and prompt_text.strip() else desc
        result[key] = entry
    return list(result.values())


def discover_all_skill_config_vars() -> list[dict[str, Any]]:
    """Config var declarations across all enabled, platform-compatible skills,
    deduplicated by key; each dict carries a ``skill`` attribution key."""
    all_vars: dict[str, dict[str, Any]] = {}
    disabled = get_disabled_skill_names()
    for skills_dir in get_all_skills_dirs():
        if not skills_dir.is_dir():
            continue
        for skill_file in iter_skill_index_files(skills_dir, "SKILL.md"):
            try:
                frontmatter, _ = parse_frontmatter(skill_file.read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            skill_name = str(frontmatter.get("name") or skill_file.parent.name)
            if skill_name in disabled or not skill_matches_platform(frontmatter):
                continue
            for var in extract_skill_config_vars(frontmatter):
                if var["key"] not in all_vars:
                    var["skill"] = skill_name
                    all_vars[var["key"]] = var
    return list(all_vars.values())


# Skill config vars are stored under skills.config.<logical key> in config.yaml.
SKILL_CONFIG_PREFIX = "skills.config"


def _resolve_dotpath(config: dict[str, Any], dotted_key: str):
    """Walk a nested dict following a dotted key; None if any part is missing."""
    current = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


_HOME_VAR_RE = re.compile(r"\$(?:\{HOME\}|HOME)(?=$|[/\\])")


def _expand_skill_config_path(value: str) -> str:
    """Expand ``~`` / ``$HOME`` against the HOME Hermes injects into tool subprocesses.

    Skill config defaults describe paths the agent hands to tools, so in a container where the
    control process HOME (``/opt/data``) differs from the tool HOME (``{HERMES_HOME}/home``) a
    plain ``expanduser`` pointed the prompt at a path no tool would ever read (#12260).
    """
    subprocess_home = get_subprocess_home()
    if subprocess_home:
        if value == "~" or value.startswith(("~/", "~\\")):
            value = subprocess_home + value[1:]
        # Callable replacement: a literal template would parse backslashes in the home path
        # as regex escapes.
        value = _HOME_VAR_RE.sub(lambda _m: subprocess_home, value)
    return os.path.expanduser(os.path.expandvars(value))


def resolve_skill_config_values(config_vars: list[dict[str, Any]]) -> dict[str, Any]:
    """Map logical skill config keys to current values (or declared defaults);
    path-like string values are ``~``/``$HOME``/``${VAR}`` expanded against the tool HOME."""
    config = _load_raw_config()
    resolved: dict[str, Any] = {}
    for var in config_vars:
        value = _resolve_dotpath(config, f"{SKILL_CONFIG_PREFIX}.{var['key']}")
        if value is None or (isinstance(value, str) and not value.strip()):
            value = var.get("default", "")
        if isinstance(value, str) and ("~" in value or "$" in value):
            value = _expand_skill_config_path(value)
        resolved[var["key"]] = value
    return resolved


# Max description chars shown per row of the system-prompt skill index. skill_manage rejects a new or changed
# description over it and tests/skills/test_authoring_standards.py holds shipped skills to it, so only legacy or
# hand-edited skills ever reach clip_skill_description().
SKILL_PROMPT_DESC_LIMIT = 60
_LAST_WHITESPACE_RE = re.compile(r"\s\S*$")
_CLIP_TRAIL_RE = re.compile(r"[\s.,;:\-\u2013\u2014(/]+$")


def _normalize_skill_description(frontmatter: dict[str, Any]) -> str:
    """Normalize a skill's description field for comparison/truncation."""
    raw_desc = frontmatter.get("description", "")
    return str(raw_desc).strip().strip("'\"") if raw_desc else ""


def clip_skill_description(desc: str) -> str:
    """*desc* fitted to SKILL_PROMPT_DESC_LIMIT, cut at a word boundary so the index never shows a half word."""
    if len(desc) <= SKILL_PROMPT_DESC_LIMIT:
        return desc
    head = desc[:SKILL_PROMPT_DESC_LIMIT - 2]  # one char past the budget: whitespace there means the last word is whole
    boundary = _LAST_WHITESPACE_RE.search(head)  # any whitespace (tab, NBSP, ...) separates words, not just " "
    if not boundary or boundary.start() == 0:  # a single word longer than the budget has no boundary to cut at
        return desc[:SKILL_PROMPT_DESC_LIMIT - 3] + "..."
    return _CLIP_TRAIL_RE.sub("", head[:boundary.start()]) + "..."


def extract_skill_description(frontmatter: dict[str, Any]) -> str:
    """Extract a system-prompt-length description from parsed frontmatter."""
    return clip_skill_description(_normalize_skill_description(frontmatter))


def is_skill_description_truncated_for_prompt(frontmatter: dict[str, Any]) -> bool:
    """True when the description will be truncated in the system prompt skill index."""
    return len(_normalize_skill_description(frontmatter)) > SKILL_PROMPT_DESC_LIMIT


# ===========================================================================
# Fail-closed resolver primitives (B5-F spec §7.4)
# ===========================================================================
# Every helper here answers the same question the resolver used to answer with a
# fallback: "can I use this root / this file?" The old answers silently shrank the
# effective skill set (DEBUG-only skips, ``except Exception: return compatible``,
# ``followlinks=True``). These return either *the bytes plus the identity they came
# from* or an explicit, unignorable error.

SKILL_FAIL_CLOSED_CONFIG_KEY = "fail_closed"


class SkillResolutionError(RuntimeError):
    """A skill root or skill file could not be resolved *reliably*.

    Raised instead of quietly reducing the skill set: the effective set must
    either match what the caller expected or the caller must see this error.
    """

    def __init__(self, message: str, *, path: Optional[str] = None,
                 errno: Optional[int] = None, kind: str = "unresolved") -> None:
        super().__init__(message)
        self.path = path
        self.errno = errno
        self.kind = kind

    def as_dict(self) -> dict[str, Any]:
        return {"error": str(self), "path": self.path, "errno": self.errno, "kind": self.kind}


@dataclass(frozen=True)
class ExternalRootProblem:
    """A configured ``skills.external_dirs`` entry that is not a usable root (R1)."""

    path: Path
    status: str            # missing | permission | not_a_dir | symlink | unresolved | other
    errno: Optional[int]
    strerror: str

    def as_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "status": self.status,
                "errno": self.errno, "strerror": self.strerror}


@dataclass(frozen=True)
class FileIdentity:
    """``(dev, ino)`` of the object whose bytes were actually read (R4).

    This — not the path — is what "the skill I validated is the skill I loaded"
    means: a swap re-creates the path with a new inode, and comparing paths would
    call that success.
    """

    dev: int
    ino: int
    size: int
    mtime_ns: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.dev, self.ino)

    @property
    def full(self) -> tuple[int, int, int, int]:
        """``(dev, ino, size, mtime_ns)`` — everything that must be unchanged for "same
        object" to mean "same *content*" too. A rewrite in place keeps the inode, so
        comparing ``(dev, ino)`` alone would accept a different skill at the same path."""
        return (self.dev, self.ino, self.size, self.mtime_ns)

    def same_as(self, other: "FileIdentity") -> bool:
        return self.full == other.full

    @classmethod
    def from_fd(cls, fd: int) -> "FileIdentity":
        st = os.fstat(fd)
        return cls(st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)

    def as_dict(self) -> dict[str, Any]:
        return {"dev": self.dev, "ino": self.ino, "size": self.size, "mtime_ns": self.mtime_ns}


# Problems from the most recent external-root resolution. Kept module-level (rather than
# threaded through every caller) so that a gate or worker can ask "is anything missing?"
# without re-implementing the scan — the resolver owns the answer, it just never told anyone.
_EXTERNAL_ROOT_PROBLEMS: list[ExternalRootProblem] = []
# Skill files that failed to load during the most recent scan (R2/R3 visibility).
SKILL_LOAD_ERRORS: list[dict[str, Any]] = []
_SKILL_LOAD_ERRORS_LOCK = threading.Lock()


def take_external_root_problems() -> list[ExternalRootProblem]:
    """Drain the unreachable-root report from the last :func:`get_external_skills_dirs`."""
    problems = list(_EXTERNAL_ROOT_PROBLEMS)
    del _EXTERNAL_ROOT_PROBLEMS[:]
    return problems


def external_root_problems() -> list[ExternalRootProblem]:
    """Non-destructive read of the last report (diagnostics / logging)."""
    return list(_EXTERNAL_ROOT_PROBLEMS)


def record_skill_load_error(error: SkillResolutionError) -> None:
    """Publish a load failure so prompt index, tools and gates all see it (R2/R3)."""
    with _SKILL_LOAD_ERRORS_LOCK:
        SKILL_LOAD_ERRORS.append(error.as_dict())


def take_skill_load_errors() -> list[dict[str, Any]]:
    """Drain the recorded skill-load failures."""
    with _SKILL_LOAD_ERRORS_LOCK:
        errors = list(SKILL_LOAD_ERRORS)
        del SKILL_LOAD_ERRORS[:]
    return errors


def reset_skill_load_errors() -> None:
    with _SKILL_LOAD_ERRORS_LOCK:
        del SKILL_LOAD_ERRORS[:]


def classify_external_root(path) -> tuple[str, Optional[int], str]:
    """``(status, errno, strerror)`` for a configured external root — **by errno, not by
    ``is_dir()``/``exists()``** (R1.1). ``exists()`` folds EACCES into False, which makes a
    permission failure indistinguishable from a directory that was never created.
    """
    try:
        st = os.lstat(str(path))
    except OSError as e:
        if e.errno == _errno.ENOENT:
            return "missing", e.errno, e.strerror or "no such file or directory"
        if e.errno == _errno.EACCES:
            return "permission", e.errno, e.strerror or "permission denied"
        if e.errno in (_errno.ELOOP, _errno.ENOTDIR, _errno.ENAMETOOLONG):
            return "unresolved", e.errno, e.strerror or str(e)
        return "other", e.errno, e.strerror or str(e)
    if _stat.S_ISDIR(st.st_mode):
        return "dir", None, ""
    if _stat.S_ISLNK(st.st_mode):
        return "symlink", None, "root is a symlink"
    return "not_a_dir", None, "entry is not a directory"


def require_external_dirs() -> None:
    """Fail when any configured external root is unreachable (R1.3)."""
    problems = external_root_problems()
    if not problems:
        return
    raise SkillResolutionError(
        "Configured external skills dirs are unreachable: "
        + ", ".join(f"{p.path} ({p.status})" for p in problems),
        kind="external_root_unreachable")


@dataclass(frozen=True)
class ParsedSkillFile:
    """Outcome of one SKILL.md read: the verdict plus the error that produced it.

    The *single* shape both consumers (prompt index and tool catalog) receive, so they
    cannot disagree about whether a skill loaded (B5-F R3).
    """

    is_compatible: bool
    frontmatter: dict
    description: str
    error: Optional[SkillResolutionError] = None

    def as_error_dict(self) -> Optional[dict]:
        return self.error.as_dict() if self.error is not None else None


def load_skill_file_checked(skill_file) -> ParsedSkillFile:
    """Read + parse one SKILL.md, fail-closed, and publish every failure (B5-F R2/R3).

    The old prompt-index helper ended in ``except Exception: return True, {}, ""``: an
    ``EACCES``/``ELOOP``/decode failure made the skill *compatible* and visible with an
    empty description. Here a failure yields ``is_compatible=False`` plus a structured
    error that is also recorded in :data:`SKILL_LOAD_ERRORS`, so prompt index, tool
    catalog and any gate see the same fact. Only I/O and decoding are handled; anything
    else is a bug in our own parser and propagates rather than degrading the index.
    """
    try:
        raw, _identity = read_verified_text(skill_file)
    except SkillResolutionError as e:
        logger.warning("Failed to read skill file %s: %s", skill_file, e)
        record_skill_load_error(e)
        return ParsedSkillFile(False, {}, "", e)
    try:
        frontmatter, _ = parse_frontmatter(raw)
        # Host-platform / runtime-environment gates are offer-time only; explicit loads bypass them.
        if not (skill_matches_platform(frontmatter) and skill_matches_environment(frontmatter)
                and skill_matches_apps(frontmatter)):
            return ParsedSkillFile(False, frontmatter, extract_skill_description(frontmatter))
        return ParsedSkillFile(True, frontmatter, extract_skill_description(frontmatter))
    except SkillResolutionError as e:
        logger.warning("Failed to parse skill file %s: %s", skill_file, e)
        record_skill_load_error(e)
        return ParsedSkillFile(False, {}, "", e)
    except UnicodeDecodeError as e:
        err = SkillResolutionError(f"skill file is not valid UTF-8: {skill_file}: {e}",
                                   path=str(skill_file), kind="decode_failed")
        logger.warning("Failed to decode skill file %s: %s", skill_file, e)
        record_skill_load_error(err)
        return ParsedSkillFile(False, {}, "", err)


def read_verified_bytes(path, *, no_follow: bool = False) -> tuple[bytes, FileIdentity]:
    """Read *path* once and return ``(bytes, identity-of-what-was-read)`` (R4.1).

    There is no separate ``stat`` followed by a later ``open``: the identity comes from the
    very fd the bytes were read through, and it is re-checked after the read, so a swap
    mid-read is detected rather than served.
    """
    target = str(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if no_follow:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(target, flags)
    except OSError as e:
        raise SkillResolutionError(f"cannot open skill file {target}: {e.strerror or e}",
                                   path=target, errno=e.errno,
                                   kind="open_failed") from e
    try:
        ident = FileIdentity.from_fd(fd)
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(fd, 1 << 16)
            except OSError as e:
                raise SkillResolutionError(f"read failed for {target}: {e.strerror or e}",
                                           path=target, errno=e.errno, kind="read_failed") from e
            if not chunk:
                break
            chunks.append(chunk)
        try:
            after = os.fstat(fd)
        except OSError as e:  # pragma: no cover - fstat after a successful read
            raise SkillResolutionError(f"cannot re-stat {target}: {e.strerror or e}",
                                       path=target, errno=e.errno, kind="restat_failed") from e
        if (after.st_dev, after.st_ino) != ident.key:
            raise SkillResolutionError(
                f"skill file replaced while being read: {target}", path=target,
                kind="identity_mismatch")
        data = b"".join(chunks)
        if len(data) != ident.size:
            raise SkillResolutionError(
                f"skill file changed size during read: {target} "
                f"(declared {ident.size}, read {len(data)})", path=target, kind="size_mismatch")
    finally:
        os.close(fd)
    return data, ident


def read_verified_text(path, *, encoding: str = "utf-8-sig",
                       no_follow: bool = False) -> tuple[str, FileIdentity]:
    """Text form of :func:`read_verified_bytes`; ``UnicodeDecodeError`` becomes an error too."""
    data, ident = read_verified_bytes(path, no_follow=no_follow)
    try:
        return data.decode(encoding), ident
    except UnicodeDecodeError as e:
        raise SkillResolutionError(f"skill file is not valid {encoding}: {path}: {e}",
                                   path=str(path), kind="decode_failed") from e


def assert_same_identity(path, ident: FileIdentity) -> FileIdentity:
    """Use-time recheck (R4.2/R4.3): the object behind *path* must still be the object
    whose bytes were validated. Raises :class:`SkillResolutionError` on any change —
    "same path" is not evidence.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError as e:
        raise SkillResolutionError(f"cannot re-verify {path}: {e.strerror or e}", path=str(path),
                                   errno=e.errno, kind="use_time_recheck_failed") from e
    try:
        current = FileIdentity.from_fd(fd)
    finally:
        os.close(fd)
    if not current.same_as(ident):
        raise SkillResolutionError(
            f"skill file identity changed between resolve and use: {path} "
            f"({ident.full} -> {current.full})", path=str(path), kind="identity_mismatch")
    return current


def content_sha256(path) -> str:
    """SHA-256 of what is actually on disk at *path* right now (R6-3 hash attestation)."""
    data, _ident = read_verified_bytes(path)
    return hashlib.sha256(data).hexdigest()


def iter_skill_index_files_strict(root, filename: str = "SKILL.md", *,
                                  allow_symlinks: bool = False) -> list[Path]:
    """Sorted skill-index files under *root*, **fail-closed** (R5/R7).

    Unlike :func:`iter_skill_index_files` this never follows links (``followlinks=True``
    let a link reach outside the root and let ``os.walk`` track loops), rejects any symlink
    component by ``lstat``, detects a scan that revisits an inode even when the kernel did
    not report ``ELOOP``, and turns an unreadable directory into an error instead of an
    empty root.
    """
    root = Path(root)
    matches: list[Path] = []
    chain: list[tuple[int, int]] = []

    def _walk_error(e: OSError) -> None:
        raise SkillResolutionError(f"cannot walk skill root {e.filename or root}: "
                                   f"{e.strerror or e}", path=str(e.filename or root),
                                   errno=e.errno, kind="walk_failed")

    try:
        root_lstat = os.lstat(str(root))
    except OSError as e:
        raise SkillResolutionError(f"cannot stat skill root {root}: {e.strerror or e}",
                                   path=str(root), errno=e.errno, kind="root_unreachable") from e
    if _stat.S_ISLNK(root_lstat.st_mode):
        raise SkillResolutionError(f"skill root is a symlink: {root}", path=str(root),
                                   kind="symlink_rejected")
    if not _stat.S_ISDIR(root_lstat.st_mode):
        raise SkillResolutionError(f"skill root is not a directory: {root}", path=str(root),
                                   kind="root_unreachable")

    real_root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(str(root), followlinks=False, onerror=_walk_error):
        this = Path(dirpath)
        try:
            dstat = os.lstat(dirpath)
        except OSError as e:
            raise SkillResolutionError(f"cannot stat {dirpath}: {e.strerror or e}", path=dirpath,
                                       errno=e.errno, kind="walk_failed") from e
        key = (dstat.st_dev, dstat.st_ino)
        if key in chain:
            raise SkillResolutionError(f"skill scan revisits an inode in {root}: {dirpath}",
                                       path=dirpath, errno=_errno.ELOOP, kind="loop_detected")
        chain.append(key)
        for name in list(dirnames):
            full = this / name
            try:
                st = os.lstat(str(full))
            except OSError as e:
                raise SkillResolutionError(f"cannot stat {full}: {e.strerror or e}", path=str(full),
                                           errno=e.errno, kind="walk_failed") from e
            if _stat.S_ISLNK(st.st_mode):
                if allow_symlinks:
                    dirnames.remove(name)  # never descend through a link into an unknown tree
                    continue
                raise SkillResolutionError(
                    f"symlink in skill root is not permitted: {full} -> {os.readlink(str(full))}",
                    path=str(full), errno=_errno.ELOOP, kind="symlink_rejected")
            if name in EXCLUDED_SKILL_DIRS:
                dirnames.remove(name)
        # §3: not one symlink anywhere in the view tree — references/, templates/, scripts/
        # and stray files included, not just SKILL.md. A single link is enough to redirect
        # whatever the resolver thought it attested to.
        for name in filenames:
            full = this / name
            try:
                st = os.lstat(str(full))
            except OSError as e:
                raise SkillResolutionError(f"cannot stat {full}: {e.strerror or e}", path=str(full),
                                           errno=e.errno, kind="stat_failed") from e
            if _stat.S_ISLNK(st.st_mode):
                if allow_symlinks:
                    continue
                raise SkillResolutionError(
                    f"symlink in skill root is not permitted: {full} -> {os.readlink(str(full))}",
                    path=str(full), errno=_errno.ELOOP, kind="symlink_rejected")
        if filename in filenames:
            full = this / filename
            try:
                st = os.lstat(str(full))
            except OSError as e:
                raise SkillResolutionError(f"cannot stat {full}: {e.strerror or e}", path=str(full),
                                           errno=e.errno, kind="stat_failed") from e
            if _stat.S_ISLNK(st.st_mode):
                if allow_symlinks:
                    continue
                raise SkillResolutionError(
                    f"symlink skill file is not permitted: {full} -> {os.readlink(str(full))}",
                    path=str(full), errno=_errno.ELOOP, kind="symlink_rejected")
            matches.append(full)
        # Containment: even without links, refuse anything that escaped the real root.
        try:
            Path(dirpath).resolve().relative_to(real_root)
        except (ValueError, OSError):
            raise SkillResolutionError(f"skill scan escaped its root {real_root}: {dirpath}",
                                       path=dirpath, kind="escape_detected")
    return sorted(matches)


@dataclass
class EffectiveSetReport:
    """Result of comparing the resolved skill set against what was expected (R6-3/R6-4)."""

    expected: dict[str, int] = field(default_factory=dict)
    actual: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    divergence: list[dict[str, Any]] = field(default_factory=list)
    load_errors: list[dict[str, Any]] = field(default_factory=list)
    root_problems: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.divergence or self.load_errors or self.root_problems)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "expected": dict(self.expected), "actual": dict(self.actual),
                "missing": list(self.missing), "unexpected": list(self.unexpected),
                "divergence": list(self.divergence), "load_errors": list(self.load_errors),
                "root_problems": list(self.root_problems)}


def effective_set_report(expected: dict[str, int], resolved: list[dict[str, Any]]) -> EffectiveSetReport:
    """Compare an expected ``{skill_name: tier}`` against a resolved catalog (R6-4/R8/R9).

    A skill resolved from a *different tier* than expected (a local copy shadowing the
    external one, say) counts as divergence rather than success: the name is present, but
    not the skill the caller meant.
    """
    actual: dict[str, int] = {}
    for entry in resolved or ():
        if not isinstance(entry, dict):
            continue
        name = entry.get("load_name") or entry.get("name")
        if name is None:
            continue
        tier = entry.get("tier")
        if tier is None:
            continue
        prev = actual.get(str(name))
        # Keep the winning (lowest) tier — the one skill_view() would load.
        actual[str(name)] = int(tier) if prev is None else min(prev, int(tier))
    report = EffectiveSetReport(expected={str(k): int(v) for k, v in (expected or {}).items()},
                                actual=actual,
                                load_errors=take_skill_load_errors(),
                                root_problems=[p.as_dict() for p in external_root_problems()])
    report.missing = sorted(n for n in report.expected if n not in actual)
    report.unexpected = sorted(n for n in actual if n not in report.expected)
    report.divergence = [
        {"name": n, "expected_tier": t, "actual_tier": actual[n]}
        for n, t in report.expected.items() if n in actual and actual[n] != t]
    return report


def require_effective_set(expected: dict[str, int], resolved: list[dict[str, Any]]) -> EffectiveSetReport:
    """Fail-closed wrapper: raise unless the resolved set *is* the expected set (R6-3)."""
    report = effective_set_report(expected, resolved)
    if not report.ok:
        raise SkillResolutionError(
            "effective skill set does not match expectation: "
            f"missing={report.missing} divergence={report.divergence} "
            f"load_errors={len(report.load_errors)} root_problems={len(report.root_problems)}",
            kind="effective_set_mismatch")
    return report


def fail_closed_enabled() -> bool:
    """``skills.fail_closed`` — True unless explicitly disabled (default fail-closed)."""
    raw = _skills_cfg_get(SKILL_FAIL_CLOSED_CONFIG_KEY)
    if raw is None:
        return True
    if isinstance(raw, str):
        return raw.strip().lower() not in ("false", "0", "no", "off")
    return bool(raw)


def exec_environment_report() -> dict[str, Any]:
    """Self-reported execution environment (R8): uid/gid/mount namespace/HERMES_HOME/cwd.

    A gate and a worker agreeing on the catalog means nothing if they resolved it in
    different namespaces; this is what lets a gate prove they did not.
    """
    try:
        from hermes_constants import get_hermes_home
        hermes_home = str(get_hermes_home())
    except Exception:  # pragma: no cover - config backend unavailable
        hermes_home = None
    try:
        mount_ns = os.readlink("/proc/self/ns/mnt")
    except OSError:
        mount_ns = None
    try:
        cwd = os.getcwd()
    except OSError:  # pragma: no cover - deleted cwd
        cwd = None
    return {"uid": os.getuid(), "gid": os.getgid(), "pid": os.getpid(),
            "mount_namespace": mount_ns, "hermes_home": hermes_home, "cwd": cwd}


def _mount_readonly(path) -> Optional[bool]:
    """``True``/``False`` when */proc/self/mountinfo* names the mount serving *path*."""
    try:
        target = os.path.realpath(str(path))
        rows: list[tuple[str, str]] = []
        with open("/proc/self/mountinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 7 and parts[4]:
                    opts = next((o for o in parts[6:] if o in ("ro", "rw")), None)
                    rows.append((parts[4], opts or ""))
        best: Optional[tuple[int, str]] = None
        for mountpoint, opts in rows:
            norm = mountpoint.rstrip("/") or "/"
            if target == norm or target.startswith(norm + "/"):
                if best is None or len(mountpoint) > best[0]:
                    best = (len(mountpoint), opts)
        return None if best is None else best[1] == "ro"
    except OSError:  # pragma: no cover - non-Linux / no procfs
        return None


def root_immutability_report(root) -> dict[str, Any]:
    """Owner/mode/mount-read-only for a resolved root — the immutability claim (R9)."""
    try:
        st = os.stat(str(root))
    except OSError as e:
        raise SkillResolutionError(f"cannot stat skill root {root}: {e.strerror or e}",
                                   path=str(root), errno=e.errno, kind="root_unreachable") from e
    return {"path": str(root), "owner_uid": st.st_uid, "gid": st.st_gid,
            "mode": oct(_stat.S_IMODE(st.st_mode)), "mount_readonly": _mount_readonly(root)}


def resolve_report(entries: list[dict[str, Any]], *, expected: Optional[dict[str, int]] = None,
                   roots=()) -> dict[str, Any]:
    """One resolution record: environment (R8) + root immutability (R9) + set check."""
    report: dict[str, Any] = {
        "environment": exec_environment_report(),
        "root_problems": [p.as_dict() for p in external_root_problems()],
        "load_errors": take_skill_load_errors(),
        "immutability": [],
    }
    for root in roots or ():
        try:
            report["immutability"].append(root_immutability_report(root))
        except SkillResolutionError as e:
            report["immutability"].append({"path": str(root), "error": e.as_dict()})
    if expected is not None:
        report["effective_set"] = effective_set_report(expected, entries).as_dict()
        report["effective_set_ok"] = report["effective_set"]["ok"]
    return report


def iter_skill_index_files(skills_dir: Path, filename: str):
    """Walk skills_dir yielding sorted paths matching *filename*; prunes
    EXCLUDED_SKILL_DIRS and support dirs of skill roots."""
    matches: list[str] = []
    for root, dirs, files in os.walk(str(skills_dir), followlinks=True):
        has_skill_md = "SKILL.md" in files
        dirs[:] = [d for d in dirs if d not in EXCLUDED_SKILL_DIRS and not (has_skill_md and d in SKILL_SUPPORT_DIRS)]
        if filename in files:
            matches.append(os.path.join(root, filename))
    yield from map(Path, sorted(matches))


# Namespace helpers for plugin-provided skills.
_NAMESPACE_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def parse_qualified_name(name: str) -> tuple[Optional[str], str]:
    """Split ``'namespace:skill-name'`` into ``(namespace, bare_name)``; ``(None, name)`` without ``':'``."""
    namespace, sep, bare = name.partition(":")
    return (namespace, bare) if sep else (None, name)


def is_valid_namespace(candidate: Optional[str]) -> bool:
    """Check whether *candidate* is a valid namespace (``[a-zA-Z0-9_-]+``)."""
    return bool(candidate) and bool(_NAMESPACE_RE.match(candidate))
