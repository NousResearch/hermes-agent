"""Lightweight skill metadata utilities shared by prompt_builder and skills_tool.
Import-light by design: no tool registry, CLI config, or provider resolution."""

import ast
import hashlib
import logging
import os
import re
import sys
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from hermes_constants import (
    get_config_path,
    get_skills_dir,
    get_subprocess_home,
)

logger = logging.getLogger(__name__)

PLATFORM_MAP = {"macos": "darwin", "linux": "linux", "windows": "win32"}

EXCLUDED_SKILL_DIRS = frozenset((
    ".git", ".github", ".hub", ".archive", ".curator_backups", ".locks",
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


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from markdown; returns (frontmatter_dict, body).
    Malformed YAML falls back to key:value line splitting. A leading UTF-8 BOM
    (Windows editors) is stripped first or it would defeat the ``---`` fence check."""
    content = content.removeprefix("\ufeff")
    end_match = re.search(r"\n---\s*\n", content[3:]) if content.startswith("---") else None
    if not end_match:
        return {}, content
    yaml_content = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :]
    frontmatter: Dict[str, Any] = {}
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


def skill_matches_platform(frontmatter: Dict[str, Any]) -> bool:
    """True when the skill's ``platforms:`` list (absent = all) matches this OS."""
    return skill_matches_platform_list(frontmatter.get("platforms"))


# An ``environments:`` tag is a *relevance* gate for offer surfaces (index,
# autocomplete, slash commands), not a compatibility gate: an explicit load
# (skill_view, --skills) always succeeds. Detection is cached per process.

_ENV_DETECT_CACHE: Dict[str, bool] = {}


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


_ENV_DETECTORS: Dict[str, Callable[[], bool]] = {
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


def skill_matches_environment(frontmatter: Dict[str, Any]) -> bool:
    """True when ANY declared ``environments:`` tag is active (absent = all;
    unknown tags fail open). Offer-time filter only."""
    environments = frontmatter.get("environments")
    if not environments:
        return True
    tags = [str(env).lower().strip() for env in (environments if isinstance(environments, list) else [environments])]
    return any(_detect_environment(tag) for tag in tags if tag)


def skill_matches_apps(frontmatter: Dict[str, Any]) -> bool:
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


_RAW_CONFIG_CACHE: Dict[Tuple[str, int, int, int, int], Dict[str, Any]] = {}


def _raw_config_cache_clear() -> None:
    """Test hook — drop the shared raw config cache."""
    _RAW_CONFIG_CACHE.clear()


def _config_cache_key(config_path: Path) -> Optional[Tuple[str, int, int, int, int]]:
    """``(path, *file_signature)`` identity of config.yaml, or None when unreadable/absent."""
    try:
        from utils import file_signature
        return (str(config_path), *file_signature(config_path.stat()))
    except OSError:
        return None


def _load_raw_config() -> Dict[str, Any]:
    """Read config.yaml with an mtime+size keyed cache (no hermes_cli.config import)."""
    config_path = get_config_path()
    if not config_path.exists():
        return {}
    cache_key = _config_cache_key(config_path)
    cached = _RAW_CONFIG_CACHE.get(cache_key) if cache_key is not None else None
    if cached is not None:
        return cached
    try:
        parsed = yaml_load(config_path.read_text(encoding="utf-8-sig"))
    except Exception as e:
        logger.debug("Could not read skill config %s: %s", config_path, e)
        return {}
    if not isinstance(parsed, dict):
        return {}
    if cache_key is not None:
        _RAW_CONFIG_CACHE.clear()
        _RAW_CONFIG_CACHE[cache_key] = parsed
    return parsed


def _skills_cfg() -> Optional[Dict[str, Any]]:
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


def get_disabled_skill_names(platform: str | None = None) -> Set[str]:
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


def parse_config_string_list(value) -> List[str]:
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


def _normalize_string_set(values) -> Set[str]:
    return {name.strip() for name in parse_config_string_list(values) if name.strip()}


# config identity -> resolved external dirs. Called once per skill during
# banner / tool-registry scans; re-resolving each time dominated cold-start.
_EXTERNAL_DIRS_CACHE: Dict[Tuple[str, int, int, int, int], List[Path]] = {}


def _external_dirs_cache_clear() -> None:
    """Test hook — drop the in-process cache."""
    _EXTERNAL_DIRS_CACHE.clear()
    _raw_config_cache_clear()


def _config_str_list(raw) -> List[str]:
    """A scalar-or-list config entry as a list of stripped non-empty strings."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [e for e in (str(entry).strip() for entry in raw) if e]


def get_external_skills_dirs() -> List[Path]:
    """Validated, deduplicated ``skills.external_dirs`` (existing dirs only). Entries
    are ``~``/``${VAR}`` expanded, relative to HERMES_HOME; the local skills dir is skipped."""
    config_path = get_config_path()
    if not config_path.exists():
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
    result: List[Path] = []
    for entry in _config_str_list(skills_cfg.get("external_dirs")):
        p = _home_relative(_expand_path(entry)).resolve()
        if p == local_skills or p in result:
            continue
        if p.is_dir():
            result.append(p)
        else:
            logger.debug("External skills dir does not exist, skipping: %s", p)
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
_SHADOW_CHECKED: Set[Tuple[str, ...]] = set()


def get_skill_search_roots(local: Optional[Path] = None, *, include_project: bool = True) -> List[Tuple[int, Path]]:
    """``(tier, dir)`` for every skill root in precedence order — the ONE ordering the skills list,
    prompt index, slash commands, skill_view, preload and cron share. *local* overrides the profile
    skills dir (skills_tool passes its live root); that entry is kept even when missing."""
    roots = [(TIER_PROJECT, d) for d in get_project_skills_dirs()] if include_project else []
    roots.append((TIER_LOCAL, Path(local) if local is not None else get_skills_dir()))
    create_dir = get_skill_create_dir()
    if create_dir is not None and create_dir.is_dir():
        roots.append((TIER_CREATE_DIR, create_dir))
    roots += [(TIER_EXTERNAL, d) for d in get_external_skills_dirs()]
    seen: Set[Path] = set()
    return [(t, d) for t, d in roots if not (d in seen or seen.add(d))]


def get_all_skills_dirs() -> List[Path]:
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


def skill_candidate_rank(skill_md, root) -> Tuple[bool, int]:
    """Same-root order of identical copies: a real SKILL.md beats a legacy flat ``<name>.md``,
    then the shallower path wins (shared by skill_view and :func:`resolve_skill_catalog`)."""
    skill_md = Path(skill_md)
    return (skill_md.name != "SKILL.md", len(skill_md.relative_to(root).parts))


def pick_skill_candidate(candidates) -> Tuple[Optional[int], List[int]]:
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


def is_disabled_entry(entry: Dict[str, Any], disabled: Set[str]) -> bool:
    """``skills.disabled`` matches a resolved catalog entry by its declared name OR its ``load_name`` —
    the exact path a same-tier duplicate's list/config/web row shows (``a/one``) and saves. A unique
    copy elsewhere that merely sits at the same relative path is not matched."""
    return not disabled.isdisjoint({str(entry["name"]), entry.get("load_name")} - {None})


def resolve_skill_catalog(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copies of scanned skills (each with ``name``, ``tier``, ``root`` and ``path`` = its SKILL.md)
    annotated with what skill_view() resolves, using its aliases (declared name, directory name, path
    relative to the root) and :func:`pick_skill_candidate`. Adds ``relative_path`` and ``status``:
    ``unique`` (``load_name`` = name), ``ambiguous`` (``load_name`` = the exact relative path, or None
    when even that is shared) or ``shadowed`` (a higher tier owns the name: hidden, warned once)."""
    out = [dict(e) for e in entries]
    owners: Dict[str, List[int]] = {}
    for i, e in enumerate(out):
        skill_dir = Path(e["path"]).parent
        e["relative_path"] = skill_dir.relative_to(e["root"]).as_posix()
        for alias in {str(e["name"]), skill_dir.name, e["relative_path"]}:
            owners.setdefault(alias, []).append(i)
    winner: Dict[str, Optional[int]] = {}
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


# Project-local skills (<root>/.hermes/skills, <root>/.agents/skills, <root>/.github/skills; root = nearest
# .git ancestor) are a prompt-injection vector if auto-sourced from any clone, so
# they load only when the root is in ``skills.trusted_project_dirs``; then they
# override same-named profile/bundled skills. cwd + trust list are session-fixed
# so the skills index stays byte-stable.

PROJECT_SKILLS_SUBDIRS = (
    os.path.join(".hermes", "skills"),
    os.path.join(".agents", "skills"),
    os.path.join(".github", "skills"),  # GitHub Copilot convention
)

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


def _project_trusted_dirs_from_config() -> Set[Path]:
    """Resolved set of trusted project roots from ``skills.trusted_project_dirs``."""
    result: Set[Path] = set()
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


def _candidate_project_skills_dirs(root: Path) -> List[Path]:
    """Existing skill dirs under *root*, excluding the profile's own skills dir
    (HERMES_HOME itself may live inside a git checkout)."""
    local_skills = get_skills_dir().resolve()
    dirs: List[Path] = []
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


def get_project_skills_dirs() -> List[Path]:
    """Trusted project-local skill dirs for the current cwd (may be empty)."""
    root = _current_project_root(trusted=True)
    return _candidate_project_skills_dirs(root) if root is not None else []


def get_untrusted_project_skills_root() -> Optional[Tuple[Path, int]]:
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
_PROJECT_QUARANTINE_CACHE: Dict[str, bool] = {}  # skill_dir -> quarantined


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
    roots: List[Path] = list(get_external_skills_dirs())
    try:
        roots.extend(get_project_skills_dirs())
    except Exception:
        pass
    return any(candidate.is_relative_to(_resolve_for_skill_ownership(root)) for root in roots)


def _hermes_metadata(frontmatter: Dict[str, Any]) -> Dict[str, Any]:
    """``metadata.hermes`` mapping from frontmatter, or ``{}`` when malformed."""
    metadata = frontmatter.get("metadata")
    hermes = metadata.get("hermes") if isinstance(metadata, dict) else None
    return hermes if isinstance(hermes, dict) else {}


# ``session_platforms`` is the gateway-channel gate: session platforms the skill
# is FOR (hidden from the index elsewhere), unlike ``platforms:`` (host OS).
_CONDITION_KEYS = ("fallback_for_toolsets", "requires_toolsets", "fallback_for_tools", "requires_tools", "session_platforms")


def extract_skill_conditions(frontmatter: Dict[str, Any]) -> Dict[str, List]:
    """Extract conditional activation fields from parsed frontmatter (absent = ``[]``)."""
    hermes = _hermes_metadata(frontmatter)
    return {key: hermes.get(key, []) for key in _CONDITION_KEYS}


def extract_skill_config_vars(frontmatter: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ``metadata.hermes.config`` declarations (key/description/default/prompt).
    Entries missing ``key`` or ``description`` are skipped; ``prompt`` defaults to the description."""
    raw = _hermes_metadata(frontmatter).get("config")
    if isinstance(raw, dict):
        raw = [raw]
    if not raw or not isinstance(raw, list):
        return []
    result: Dict[str, Dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        desc = str(item.get("description", "")).strip()
        if not key or key in result or not desc:
            continue
        entry: Dict[str, Any] = {"key": key, "description": desc}
        if item.get("default") is not None:
            entry["default"] = item["default"]
        prompt_text = item.get("prompt")
        entry["prompt"] = prompt_text.strip() if isinstance(prompt_text, str) and prompt_text.strip() else desc
        result[key] = entry
    return list(result.values())


def discover_all_skill_config_vars() -> List[Dict[str, Any]]:
    """Config var declarations across all enabled, platform-compatible skills,
    deduplicated by key; each dict carries a ``skill`` attribution key."""
    all_vars: Dict[str, Dict[str, Any]] = {}
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


def _resolve_dotpath(config: Dict[str, Any], dotted_key: str):
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


def resolve_skill_config_values(config_vars: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map logical skill config keys to current values (or declared defaults);
    path-like string values are ``~``/``$HOME``/``${VAR}`` expanded against the tool HOME."""
    config = _load_raw_config()
    resolved: Dict[str, Any] = {}
    for var in config_vars:
        value = _resolve_dotpath(config, f"{SKILL_CONFIG_PREFIX}.{var['key']}")
        if value is None or (isinstance(value, str) and not value.strip()):
            value = var.get("default", "")
        if isinstance(value, str) and ("~" in value or "$" in value):
            value = _expand_skill_config_path(value)
        resolved[var["key"]] = value
    return resolved


SKILL_PROMPT_DESC_LIMIT = 60


def _normalize_skill_description(frontmatter: Dict[str, Any]) -> str:
    """Normalize a skill's description field for comparison/truncation."""
    raw_desc = frontmatter.get("description", "")
    return str(raw_desc).strip().strip("'\"") if raw_desc else ""


def extract_skill_description(frontmatter: Dict[str, Any]) -> str:
    """Extract a system-prompt-length description from parsed frontmatter."""
    desc = _normalize_skill_description(frontmatter)
    return desc[:SKILL_PROMPT_DESC_LIMIT - 3] + "..." if len(desc) > SKILL_PROMPT_DESC_LIMIT else desc


def is_skill_description_truncated_for_prompt(frontmatter: Dict[str, Any]) -> bool:
    """True when the description will be truncated in the system prompt skill index."""
    return len(_normalize_skill_description(frontmatter)) > SKILL_PROMPT_DESC_LIMIT


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


def parse_qualified_name(name: str) -> Tuple[Optional[str], str]:
    """Split ``'namespace:skill-name'`` into ``(namespace, bare_name)``; ``(None, name)`` without ``':'``."""
    namespace, sep, bare = name.partition(":")
    return (namespace, bare) if sep else (None, name)


def is_valid_namespace(candidate: Optional[str]) -> bool:
    """Check whether *candidate* is a valid namespace (``[a-zA-Z0-9_-]+``)."""
    return bool(candidate) and bool(_NAMESPACE_RE.match(candidate))
