"""Profile distributions — shareable, packaged Hermes profiles via git.

Sources: a git URL (``github.com/user/repo``, ``https://...``, ``git@...``, ``ssh://``,
``git://``) or a local directory that already contains ``distribution.yaml`` (profile
development before the first push).
"""

from __future__ import annotations

import operator
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

import yaml

from hermes_cli._subprocess_compat import noninteractive_git_env


MANIFEST_FILENAME = "distribution.yaml"
ENV_TEMPLATE_FILENAME = ".env.template"
ENV_EXAMPLE_FILENAME = ".env.EXAMPLE"

# Default distribution-owned paths (relative to profile root). Authors may override via
# ``distribution_owned:``. config.yaml is dist-owned but preserved on update by default.
DEFAULT_DIST_OWNED: Tuple[str, ...] = ("SOUL.md", "config.yaml", "mcp.json", "skills", "cron", MANIFEST_FILENAME)

# Paths NEVER part of a distribution: user-owned, protected on update. Keep consistent with
# ``profiles.py`` export exclusions plus the ``local/`` convention for user customizations.
USER_OWNED_EXCLUDE: frozenset = frozenset({
    # Credentials & runtime secrets
    "auth.json", ".env",
    # Databases & runtime state
    "state.db", "state.db-shm", "state.db-wal",
    "hermes_state.db", "response_store.db",
    "response_store.db-shm", "response_store.db-wal",
    "gateway.pid", "gateway_state.json", "processes.json",
    "auth.lock", "active_profile", ".update_check",
    "errors.log", ".hermes_history",
    # User data
    "memories", "sessions", "logs", "plans", "workspace", "home",
    "image_cache", "audio_cache", "document_cache",
    "browser_screenshots", "checkpoints", "sandboxes",
    "backups", "cache",
    # Infrastructure
    "hermes-agent", ".worktrees", "profiles", "bin", "node_modules",
    # User customization namespace
    "local",
})


class DistributionError(Exception):
    """Raised for distribution install/update failures."""


# Manifest

def _str(data: dict, key: str, default: str = "") -> str:
    return str(data.get(key) or default)


@dataclass
class EnvRequirement:
    name: str
    description: str = ""
    required: bool = True
    default: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Any) -> "EnvRequirement":
        if not isinstance(data, dict):
            raise DistributionError(f"env_requires entry must be a mapping, got {type(data).__name__}")
        name = _str(data, "name").strip()
        if not name:
            raise DistributionError("env_requires entry missing 'name'")
        return cls(
            name=name, description=_str(data, "description"), required=bool(data.get("required", True)),
            default=data.get("default"),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "description": self.description}
        if not self.required:
            out["required"] = False
        if self.default is not None:
            out["default"] = self.default
        return out


@dataclass
class DistributionManifest:
    name: str
    version: str = "0.1.0"
    description: str = ""
    hermes_requires: str = ""
    author: str = ""
    license: str = ""
    env_requires: List[EnvRequirement] = field(default_factory=list)
    distribution_owned: List[str] = field(default_factory=list)
    # Tracked after install — where we pulled from, so ``update`` can re-pull.
    source: str = ""
    # ISO-8601 UTC timestamp written on install/update (empty in repo-shipped manifests).
    installed_at: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "DistributionManifest":
        if not isinstance(data, dict):
            raise DistributionError(f"{MANIFEST_FILENAME} must be a mapping, got {type(data).__name__}")
        name = _str(data, "name").strip()
        if not name:
            raise DistributionError(f"{MANIFEST_FILENAME} missing 'name'")
        env_raw = data.get("env_requires") or []
        if not isinstance(env_raw, list):
            raise DistributionError("env_requires must be a list")
        dist_owned_raw = data.get("distribution_owned") or []
        if dist_owned_raw and not isinstance(dist_owned_raw, list):
            raise DistributionError("distribution_owned must be a list")
        return cls(
            name=name, version=_str(data, "version", "0.1.0"), description=_str(data, "description"),
            hermes_requires=_str(data, "hermes_requires"), author=_str(data, "author"),
            license=_str(data, "license"), env_requires=[EnvRequirement.from_dict(e) for e in env_raw],
            distribution_owned=[str(p).strip().strip("/") for p in dist_owned_raw if str(p).strip()],
            source=_str(data, "source"), installed_at=_str(data, "installed_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "version": self.version}
        # Key order is the on-disk YAML order (write_manifest uses sort_keys=False).
        optional = (
            ("description", self.description), ("hermes_requires", self.hermes_requires),
            ("author", self.author), ("license", self.license),
            ("env_requires", [e.to_dict() for e in self.env_requires]),
            ("distribution_owned", self.distribution_owned), ("source", self.source),
            ("installed_at", self.installed_at),
        )
        out.update((k, v) for k, v in optional if v)
        return out


def read_manifest(profile_dir: Path) -> Optional[DistributionManifest]:
    """Return the manifest for *profile_dir*, or None if it isn't a distribution."""
    mf_path = profile_dir / MANIFEST_FILENAME
    if not mf_path.is_file():
        return None
    try:
        data = yaml.safe_load(mf_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DistributionError(f"Failed to parse {mf_path}: {exc}") from exc
    return DistributionManifest.from_dict(data or {})


def write_manifest(profile_dir: Path, manifest: DistributionManifest) -> Path:
    """Atomically write ``distribution.yaml``. A bare write_text() truncates before the dump
    lands and read_manifest() treats a missing/unparseable manifest as "not a distribution",
    so an interrupted install/update would silently demote the profile."""
    mf_path = profile_dir / MANIFEST_FILENAME
    from utils import atomic_yaml_write

    # create_mode=0o644: with an explicit `distribution_owned` allowlist that omits
    # distribution.yaml, _copy_dist_payload reaches here with no manifest on disk. It is a
    # shareable descriptor, not a secret — don't leave it at mkstemp's 0600. An existing
    # file's mode is preserved.
    atomic_yaml_write(mf_path, manifest.to_dict(), sort_keys=False, default_flow_style=False, create_mode=0o644)
    return mf_path


# Version check

_VERSION_OP_RE = re.compile(r"^\s*(>=|<=|==|!=|>|<)\s*(.+?)\s*$")
_VERSION_OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq, "!=": operator.ne, ">": operator.gt, "<": operator.lt}


def _parse_semver(v: str) -> Tuple[int, int, int]:
    """major.minor.patch only; pre-release / build metadata ("0.12.0-rc1+abc") stripped."""
    parts = re.split(r"[-+]", str(v).strip().lstrip("v"), 1)[0].split(".")
    parts += ["0"] * (3 - len(parts))
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise DistributionError(f"Unparseable version: {v!r}") from exc


def check_hermes_requires(spec: str, current_version: str) -> None:
    """Raise DistributionError if ``current_version`` does not satisfy ``spec`` (bare version = ``>=``)."""
    if not spec or not spec.strip():
        return
    m = _VERSION_OP_RE.match(spec)
    op, target = m.groups() if m else (">=", spec.strip())
    if not _VERSION_OPS[op](_parse_semver(current_version), _parse_semver(target)):
        raise DistributionError(f"This distribution requires Hermes {op}{target}, but you have {current_version}.")


def _env_template_from_manifest(manifest: DistributionManifest) -> str:
    """Generate a ``.env.template`` body from env_requires."""
    lines = [
        "# Environment variables required by this Hermes distribution.",
        "# Copy to `.env` and fill in your own values before running.", "",
    ]
    for req in manifest.env_requires:
        if req.description:
            lines.append(f"# {req.description}")
        default_val = req.default if req.default is not None else ""
        if req.required:
            lines += ["# (required)", f"{req.name}={default_val}", ""]
        else:
            lines += ["# (optional)", f"# {req.name}={default_val}", ""]
    return "\n".join(lines).rstrip() + "\n"


# Source staging — git clone or local directory

_GITHUB_SHORTHAND_RE = re.compile(r"^github\.com/[\w.-]+/[\w.-]+/?$")


def _looks_like_git_url(s: str) -> bool:
    """Any http(s) URL is a git repo — git is the only remote transport (no tar.gz URLs)."""
    s = s.strip()
    return (
        s.endswith(".git")
        or s.startswith(("git@", "ssh://", "git://", "http://", "https://"))
        or bool(_GITHUB_SHORTHAND_RE.match(s))
    )


def _git_clone(url: str, dest: Path) -> None:
    if _GITHUB_SHORTHAND_RE.match(url):
        url = f"https://{url.rstrip('/')}"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)], check=True, capture_output=True,
            stdin=subprocess.DEVNULL, env=noninteractive_git_env(),
        )
    except FileNotFoundError as exc:
        raise DistributionError("git is required for git-URL installs") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise DistributionError(f"git clone failed: {stderr.strip()}") from exc


def _stage_source(source: str, workdir: Path) -> Tuple[Path, str]:
    """Resolve *source* to ``(staged_dir, provenance)``: git URLs are shallow-cloned into
    *workdir* (``.git`` removed); a local directory is used in place."""
    src_str = source.strip()
    if _looks_like_git_url(src_str):
        staged, provenance = workdir / "clone", src_str
        _git_clone(src_str, staged)
        shutil.rmtree(staged / ".git", ignore_errors=True)
        missing = (
            f"No {MANIFEST_FILENAME} at the root of {src_str!r}. "
            "This repository is not a Hermes profile distribution."
        )
    elif (path_guess := Path(src_str).expanduser()).is_dir():
        staged = path_guess.resolve()
        provenance = str(staged)
        missing = (
            f"No {MANIFEST_FILENAME} in {path_guess}. "
            "A local-directory source must contain a distribution.yaml at its root."
        )
    else:
        raise DistributionError(
        f"Cannot resolve distribution source: {source!r}. "
        "Expected a git URL (e.g. github.com/user/repo) or a local directory."
    )
    if not (staged / MANIFEST_FILENAME).is_file():
        raise DistributionError(missing)
    return staged, provenance


def _reject_distribution_symlinks(staged: Path) -> None:
    """Reject symlinks before reading or copying distribution files."""
    for entry in staged.rglob("*"):
        if not entry.is_symlink():
            continue
        try:
            rel = entry.relative_to(staged)
        except ValueError:
            rel = entry
        raise DistributionError(f"Profile distributions cannot contain symlinks: {rel}")


# Install

@dataclass
class InstallPlan:
    """Summary of what an install will do, surfaced for user confirmation."""
    manifest: DistributionManifest
    staged_dir: Path
    provenance: str
    target_dir: Path
    existing: bool  # True if target profile already exists (update path)
    preserves_config: bool = True
    has_cron: bool = False


def _has_cron_jobs(staged: Path) -> bool:
    cron_dir = staged / "cron"
    return cron_dir.is_dir() and (any(cron_dir.rglob("*.json")) or any(cron_dir.rglob("*.yaml")))


def plan_install(source: str, workdir: Path, override_name: Optional[str] = None) -> InstallPlan:
    """Stage *source* and produce a plan describing what install would do."""
    from hermes_cli.profiles import _canon_valid, get_profile_dir
    from hermes_cli import __version__ as hermes_version
    staged, provenance = _stage_source(source, workdir)
    _reject_distribution_symlinks(staged)
    manifest = read_manifest(staged)
    if manifest is None:
        raise DistributionError(
            f"No {MANIFEST_FILENAME} found at the distribution root — this source is not a Hermes distribution."
        )
    check_hermes_requires(manifest.hermes_requires, hermes_version)  # fail fast
    canon = _canon_valid(override_name or manifest.name)
    if canon == "default":
        raise DistributionError(
            "Cannot install a distribution as 'default' — that is the built-in "
            "root profile (~/.hermes).  Pass --name <name> to install under a new profile."
        )
    manifest.name = canon
    manifest.source = provenance
    # Stamped once here so both fresh install and update propagate a fresh timestamp.
    manifest.installed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    target_dir = get_profile_dir(canon)
    existing = target_dir.is_dir()
    return InstallPlan(
        manifest=manifest, staged_dir=staged, provenance=provenance, target_dir=target_dir, existing=existing,
        preserves_config=existing, has_cron=_has_cron_jobs(staged),
    )


def _owned_entries(staged: Path, manifest: DistributionManifest):
    """Yield ``(src, rel_parts)`` for every staged path the distribution owns."""
    explicit_owned = [p for p in (p.strip().strip("/") for p in manifest.distribution_owned) if p]
    if not explicit_owned:
        # Legacy: no allowlist means the whole payload (minus USER_OWNED_EXCLUDE) is owned.
        # Do NOT narrow to DEFAULT_DIST_OWNED — existing distributions ship arbitrary extra
        # top-level paths without declaring them.
        for entry in staged.iterdir():
            if entry.name not in USER_OWNED_EXCLUDE:
                yield entry, (entry.name,)
        return
    # Path-aware allowlist: copy exactly the declared paths.
    for rel in explicit_owned:
        rel_parts = PurePosixPath(rel).parts
        if not rel_parts or rel_parts[0] in USER_OWNED_EXCLUDE:
            continue
        if ".." in rel_parts or PurePosixPath(rel).is_absolute():
            continue
        src = staged.joinpath(*rel_parts)
        if src.exists():
            yield src, rel_parts


def _copy_dist_payload(staged: Path, target: Path, manifest: DistributionManifest, preserve_config: bool) -> None:
    """Copy distribution-owned files (see ``_owned_entries``) from *staged* into *target*.

    User-owned paths are never touched. ``config.yaml`` is replaced only when
    ``preserve_config`` is False (fresh install / ``--force-config``). ``.env.template`` lands
    as ``.env.EXAMPLE`` so it never shadows a real ``.env``."""
    target.mkdir(parents=True, exist_ok=True)
    staged_resolved = staged.resolve()

    def _ignore_user_owned(d, names):
        # Only the staged root's direct children are filtered.
        return [n for n in names if n in USER_OWNED_EXCLUDE] if Path(d).resolve() == staged_resolved else []

    for src, rel_parts in _owned_entries(staged, manifest):
        if len(rel_parts) == 1:
            name = rel_parts[0]
            if name == ENV_TEMPLATE_FILENAME:
                shutil.copy2(src, target / ENV_EXAMPLE_FILENAME)
                continue
            if name == "config.yaml" and preserve_config and (target / "config.yaml").exists():
                continue
        dest = target.joinpath(*rel_parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest, ignore=_ignore_user_owned)
        else:
            shutil.copy2(src, dest)

    # Emit .env.EXAMPLE from manifest if the staged tree didn't ship one
    if manifest.env_requires and not (target / ENV_EXAMPLE_FILENAME).exists():
        (target / ENV_EXAMPLE_FILENAME).write_text(_env_template_from_manifest(manifest), encoding="utf-8")

    # Make sure the manifest on disk reflects resolved name + source
    write_manifest(target, manifest)


def _bootstrap_user_dirs(target: Path) -> None:
    """Create the bootstrap dirs a fresh profile expects (same set as ``create_profile``)."""
    from hermes_cli.profiles import _PROFILE_DIRS
    for d in _PROFILE_DIRS:
        (target / d).mkdir(parents=True, exist_ok=True)


def install_distribution(
    source: str, name: Optional[str] = None, force: bool = False, create_alias: bool = False
) -> InstallPlan:
    """Install a distribution from *source* into a new profile; returns the resolved plan.
    Use :func:`plan_install` first to preview + prompt."""
    from hermes_cli.profiles import check_alias_collision, create_wrapper_script
    with tempfile.TemporaryDirectory(prefix="hermes_dist_install_") as tmp:
        plan = plan_install(source, Path(tmp), override_name=name)
        if plan.existing and not force:
            raise DistributionError(
                f"Profile '{plan.manifest.name}' already exists at {plan.target_dir}. "
                "Use `hermes profile update` to upgrade in place, or pass --force to overwrite."
            )

        # Fresh install: config.yaml comes from the distribution.
        _bootstrap_user_dirs(plan.target_dir)
        _copy_dist_payload(plan.staged_dir, plan.target_dir, plan.manifest, preserve_config=False)
        if create_alias and check_alias_collision(plan.manifest.name) is None:
            create_wrapper_script(plan.manifest.name)
        return plan


def _existing_profile(profile_name: str) -> Tuple[str, Path]:
    """Return ``(canonical_name, profile_dir)`` or raise if the profile doesn't exist."""
    from hermes_cli.profiles import _existing_profile_dir

    try:
        return _existing_profile_dir(profile_name)
    except FileNotFoundError as exc:
        raise DistributionError(str(exc)) from exc


def update_distribution(profile_name: str, force_config: bool = False) -> InstallPlan:
    """Re-pull from the installed manifest's ``source:`` and apply: dist-owned files
    overwritten, user data never touched, ``config.yaml`` preserved unless ``force_config``."""
    canon, target = _existing_profile(profile_name)
    existing_manifest = read_manifest(target)
    if existing_manifest is None:
        raise DistributionError(
            f"Profile '{canon}' is not a distribution (no {MANIFEST_FILENAME}). "
            "Only profiles installed via `hermes profile install` can be updated."
        )
    if not existing_manifest.source:
        raise DistributionError(
            f"Profile '{canon}' has no recorded source.  Re-install with "
            "`hermes profile install <source> --name {canon} --force`."
        )
    with tempfile.TemporaryDirectory(prefix="hermes_dist_update_") as tmp:
        plan = plan_install(existing_manifest.source, Path(tmp), override_name=canon)
        plan.preserves_config = not force_config
        _copy_dist_payload(plan.staged_dir, plan.target_dir, plan.manifest, preserve_config=plan.preserves_config)
        return plan


def describe_distribution(profile_name: str) -> Dict[str, Any]:
    """Return a structured view of a profile's distribution metadata ({} if not a distribution)."""
    manifest = read_manifest(_existing_profile(profile_name)[1])
    return {} if manifest is None else manifest.to_dict()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'is_excluded_skill_path': ('agent.skill_utils', 'is_excluded_skill_path'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
