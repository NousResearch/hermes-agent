"""Profile distributions — shareable, packaged Hermes profiles via git.

A distribution is a Hermes profile published as a git repository (or
installed from a local directory for development). Install with one command
from a git URL, update in place, and keep your local memories / sessions /
credentials untouched.

Where this fits relative to the existing pieces:

* ``hermes profile export/import`` — local backup / restore for a profile
  on your own machine. NOT a distribution format. Stays as-is.
* ``hermes skills install <url>`` — the URL install pattern we're mirroring,
  but at the profile granularity.

Subcommands (all live under ``hermes profile``, not a parallel tree):

    hermes profile install <source> [--name N] [--alias] [--force] [--yes]
    hermes profile update  <name>  [--force-config] [--yes]
    hermes profile info    <name>

``<source>`` is one of:

* A git URL (``github.com/user/repo``, ``https://github.com/...``, ``git@...``,
  ``ssh://``, ``git://``), optionally with ``#<ref>`` to pin a tag / branch /
  commit SHA.
* A local directory that already contains ``distribution.yaml`` — used
  during profile development before the first push.

Manifest format (``distribution.yaml`` at the profile root)::

    name: telemetry
    version: 0.1.0
    description: "Compliance monitoring harness"
    hermes_requires: ">=0.12.0"
    author: "..."
    license: "..."
    env_requires:
      - name: OPENAI_API_KEY
        description: "OpenAI API key"
        required: true
      - name: GRAPHITI_MCP_URL
        description: "Memory graph URL"
        required: false
        default: "http://127.0.0.1:8000/sse"
    distribution_owned:      # optional; sensible defaults apply
      - SOUL.md
      - skills/
      - cron/
      - mcp.json

Update semantics:

* Distribution-owned paths (SOUL.md, mcp.json, skills/, cron/,
  distribution.yaml) are replaced from the new source.
* ``config.yaml`` is distribution-owned but preserved on update unless
  ``--force-config`` is passed (user overrides typically live here).
* User-owned paths (memories/, sessions/, state.db, auth.json, .env,
  logs/, workspace/, home/, plans/, *_cache/, and anything under
  ``local/``) are never touched.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = "distribution.yaml"
ENV_TEMPLATE_FILENAME = ".env.template"
ENV_EXAMPLE_FILENAME = ".env.EXAMPLE"

# Default distribution-owned paths (relative to profile root).  Authors may
# override via ``distribution_owned:`` in the manifest.  config.yaml is
# distribution-owned but treated specially on update (see _is_config_like).
DEFAULT_DIST_OWNED: Tuple[str, ...] = (
    "SOUL.md",
    "config.yaml",
    "mcp.json",
    "skills",
    "cron",
    MANIFEST_FILENAME,
)

# Paths that are NEVER part of a distribution. These are user-owned and are
# protected on update. Must stay consistent with
# ``profiles.py::_DEFAULT_EXPORT_EXCLUDE_ROOT`` plus the ``local/``
# convention for user customizations.
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


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DistributionError(Exception):
    """Raised for distribution install/update failures."""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class EnvRequirement:
    name: str
    description: str = ""
    required: bool = True
    default: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Any) -> "EnvRequirement":
        if not isinstance(data, dict):
            raise DistributionError(
                f"env_requires entry must be a mapping, got {type(data).__name__}"
            )
        name = str(data.get("name") or "").strip()
        if not name:
            raise DistributionError("env_requires entry missing 'name'")
        return cls(
            name=name,
            description=str(data.get("description") or ""),
            required=bool(data.get("required", True)),
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
    # ISO-8601 UTC timestamp written on install / update, so ``info`` and
    # ``list`` can show when a distribution landed on disk.  Empty for
    # manifests that ship in a repo (authors don't populate this).
    installed_at: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "DistributionManifest":
        if not isinstance(data, dict):
            raise DistributionError(
                f"{MANIFEST_FILENAME} must be a mapping, got {type(data).__name__}"
            )
        name = str(data.get("name") or "").strip()
        if not name:
            raise DistributionError(f"{MANIFEST_FILENAME} missing 'name'")
        env_raw = data.get("env_requires") or []
        if not isinstance(env_raw, list):
            raise DistributionError("env_requires must be a list")
        env_requires = [EnvRequirement.from_dict(e) for e in env_raw]
        dist_owned_raw = data.get("distribution_owned") or []
        if dist_owned_raw and not isinstance(dist_owned_raw, list):
            raise DistributionError("distribution_owned must be a list")
        distribution_owned = [str(p).strip().strip("/") for p in dist_owned_raw if str(p).strip()]
        return cls(
            name=name,
            version=str(data.get("version") or "0.1.0"),
            description=str(data.get("description") or ""),
            hermes_requires=str(data.get("hermes_requires") or ""),
            author=str(data.get("author") or ""),
            license=str(data.get("license") or ""),
            env_requires=env_requires,
            distribution_owned=distribution_owned,
            source=str(data.get("source") or ""),
            installed_at=str(data.get("installed_at") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
        }
        if self.description:
            out["description"] = self.description
        if self.hermes_requires:
            out["hermes_requires"] = self.hermes_requires
        if self.author:
            out["author"] = self.author
        if self.license:
            out["license"] = self.license
        if self.env_requires:
            out["env_requires"] = [e.to_dict() for e in self.env_requires]
        if self.distribution_owned:
            out["distribution_owned"] = self.distribution_owned
        if self.source:
            out["source"] = self.source
        if self.installed_at:
            out["installed_at"] = self.installed_at
        return out

    def owned_paths(self) -> List[str]:
        """Resolve which paths count as distribution-owned."""
        if self.distribution_owned:
            return list(self.distribution_owned)
        return list(DEFAULT_DIST_OWNED)


def _load_yaml(text: str) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover — pyyaml is a hard dep
        raise DistributionError("PyYAML is required for distribution manifests") from exc
    return yaml.safe_load(text)


def _dump_yaml(data: Any) -> str:
    import yaml

    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def read_manifest(profile_dir: Path) -> Optional[DistributionManifest]:
    """Return the manifest for *profile_dir*, or None if it isn't a distribution."""
    mf_path = profile_dir / MANIFEST_FILENAME
    if not mf_path.is_file():
        return None
    try:
        data = _load_yaml(mf_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DistributionError(f"Failed to parse {mf_path}: {exc}") from exc
    return DistributionManifest.from_dict(data or {})


def write_manifest(profile_dir: Path, manifest: DistributionManifest) -> Path:
    mf_path = profile_dir / MANIFEST_FILENAME
    mf_path.write_text(_dump_yaml(manifest.to_dict()), encoding="utf-8")
    return mf_path


# ---------------------------------------------------------------------------
# Version check
# ---------------------------------------------------------------------------


_VERSION_OP_RE = re.compile(r"^\s*(>=|<=|==|!=|>|<)\s*(.+?)\s*$")


def _parse_semver(v: str) -> Tuple[int, int, int]:
    """Very small semver parser — major.minor.patch only.  Extra labels stripped."""
    s = str(v).strip().lstrip("v")
    # Strip any pre-release / build metadata (e.g. "0.12.0-rc1+abc")
    s = re.split(r"[-+]", s, 1)[0]
    parts = s.split(".")
    while len(parts) < 3:
        parts.append("0")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise DistributionError(f"Unparseable version: {v!r}") from exc


def check_hermes_requires(spec: str, current_version: str) -> None:
    """Raise DistributionError if ``current_version`` does not satisfy ``spec``.

    ``spec`` accepts a single comparator (``>=0.12.0``, ``==0.12.0``, etc.).
    Empty or blank spec is a no-op — no requirement.
    """
    if not spec or not spec.strip():
        return
    m = _VERSION_OP_RE.match(spec)
    if not m:
        # Bare version → treat as ``>=``
        op, target = ">=", spec.strip()
    else:
        op, target = m.group(1), m.group(2)
    cur = _parse_semver(current_version)
    tgt = _parse_semver(target)
    ok = {
        ">=": cur >= tgt,
        "<=": cur <= tgt,
        "==": cur == tgt,
        "!=": cur != tgt,
        ">":  cur > tgt,
        "<":  cur < tgt,
    }[op]
    if not ok:
        raise DistributionError(
            f"This distribution requires Hermes {op}{target}, "
            f"but you have {current_version}."
        )


# ---------------------------------------------------------------------------
# Env var template helper
# ---------------------------------------------------------------------------


def _env_template_from_manifest(manifest: DistributionManifest) -> str:
    """Generate a ``.env.template`` body from env_requires."""
    lines = [
        "# Environment variables required by this Hermes distribution.",
        "# Copy to `.env` and fill in your own values before running.",
        "",
    ]
    for req in manifest.env_requires:
        if req.description:
            lines.append(f"# {req.description}")
        status = "required" if req.required else "optional"
        lines.append(f"# ({status})")
        default_val = req.default if req.default is not None else ""
        prefix = "" if req.required else "# "
        lines.append(f"{prefix}{req.name}={default_val}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Source staging — git clone or local directory
# ---------------------------------------------------------------------------


def _looks_like_git_url(s: str) -> bool:
    s = s.strip()
    if s.endswith(".git"):
        return True
    if s.startswith(("git@", "ssh://", "git://")):
        return True
    if s.startswith(("http://", "https://")):
        # Any http(s) URL is treated as a git repo.  We no longer accept
        # tar.gz URLs — git is the only remote transport.
        return True
    # Bare github.com/user/repo shorthand
    if re.match(r"^github\.com/[\w.-]+/[\w.-]+/?$", s):
        return True
    return False


def _git_clone(url: str, dest: Path) -> None:
    # Normalize github.com/user/repo shorthand
    if re.match(r"^github\.com/[\w.-]+/[\w.-]+/?$", url):
        url = f"https://{url.rstrip('/')}"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise DistributionError("git is required for git-URL installs") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise DistributionError(f"git clone failed: {stderr.strip()}") from exc


def _stage_source(source: str, workdir: Path) -> Tuple[Path, str]:
    """Resolve *source* to a local directory containing distribution.yaml.

    Returns ``(staged_dir, provenance)`` where ``provenance`` is stored in the
    installed manifest's ``source:`` field so ``hermes profile update`` can
    re-pull from the same place.

    Accepts:
      * A git URL (https / ssh / git@ / bare github.com shorthand) — cloned
        into a temp directory; ``.git`` removed after clone.
      * A local directory already containing ``distribution.yaml``.
    """
    src_str = source.strip()

    # Git URL
    if _looks_like_git_url(src_str):
        cloned = workdir / "clone"
        _git_clone(src_str, cloned)
        # Remove .git to keep the staged tree clean
        shutil.rmtree(cloned / ".git", ignore_errors=True)
        if not (cloned / MANIFEST_FILENAME).is_file():
            raise DistributionError(
                f"No {MANIFEST_FILENAME} at the root of {src_str!r}. "
                "This repository is not a Hermes profile distribution."
            )
        return cloned, src_str

    # Local directory
    path_guess = Path(src_str).expanduser()
    if path_guess.is_dir():
        if not (path_guess / MANIFEST_FILENAME).is_file():
            raise DistributionError(
                f"No {MANIFEST_FILENAME} in {path_guess}. "
                "A local-directory source must contain a distribution.yaml at its root."
            )
        return path_guess.resolve(), str(path_guess.resolve())

    raise DistributionError(
        f"Cannot resolve distribution source: {source!r}. "
        "Expected a git URL (e.g. github.com/user/repo) or a local directory."
    )


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


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
    has_skills: bool = False


def _has_cron_jobs(staged: Path) -> bool:
    cron_dir = staged / "cron"
    if not cron_dir.is_dir():
        return False
    for _ in cron_dir.rglob("*.json"):
        return True
    for _ in cron_dir.rglob("*.yaml"):
        return True
    return False


def _count_skills(staged: Path) -> int:
    skills_dir = staged / "skills"
    if not skills_dir.is_dir():
        return 0
    return sum(1 for _ in skills_dir.rglob("SKILL.md"))


def plan_install(
    source: str,
    workdir: Path,
    override_name: Optional[str] = None,
) -> InstallPlan:
    """Stage *source* and produce a plan describing what install would do."""
    from hermes_cli.profiles import (
        get_profile_dir,
        normalize_profile_name,
        validate_profile_name,
    )
    from hermes_cli import __version__ as hermes_version

    staged, provenance = _stage_source(source, workdir)
    manifest = read_manifest(staged)
    if manifest is None:
        raise DistributionError(
            f"No {MANIFEST_FILENAME} found at the distribution root — "
            "this source is not a Hermes distribution."
        )

    # Version check up-front so we fail fast
    check_hermes_requires(manifest.hermes_requires, hermes_version)

    # Resolve target profile name
    target_name = override_name or manifest.name
    canon = normalize_profile_name(target_name)
    validate_profile_name(canon)
    if canon == "default":
        raise DistributionError(
            "Cannot install a distribution as 'default' — that is the built-in "
            "root profile (~/.hermes).  Pass --name <name> to install under a "
            "new profile."
        )
    manifest.name = canon
    manifest.source = provenance
    # Stamped once here so plan_install() callers (both fresh install and
    # update) propagate a freshly-minted timestamp through _copy_dist_payload.
    manifest.installed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    target_dir = get_profile_dir(canon)
    existing = target_dir.is_dir()
    has_cron = _has_cron_jobs(staged)
    skill_count = _count_skills(staged)

    return InstallPlan(
        manifest=manifest,
        staged_dir=staged,
        provenance=provenance,
        target_dir=target_dir,
        existing=existing,
        preserves_config=existing,
        has_cron=has_cron,
        has_skills=skill_count > 0,
    )


def _copy_dist_payload(
    staged: Path,
    target: Path,
    manifest: DistributionManifest,
    preserve_config: bool,
) -> None:
    """Copy distribution-owned files from *staged* into *target*.

    User-owned paths are never touched.  ``config.yaml`` is replaced only when
    ``preserve_config`` is False (fresh install or ``--force-config`` update).
    ``.env.template`` is renamed to ``.env.EXAMPLE`` in the target to avoid
    shadowing a real ``.env``.
    """
    target.mkdir(parents=True, exist_ok=True)

    for entry in staged.iterdir():
        name = entry.name

        if name in USER_OWNED_EXCLUDE:
            continue
        if name == ENV_TEMPLATE_FILENAME:
            shutil.copy2(entry, target / ENV_EXAMPLE_FILENAME)
            continue
        if name == "config.yaml" and preserve_config and (target / "config.yaml").exists():
            # Leave user's config.yaml alone on update
            continue

        dest = target / name
        if entry.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(
                entry,
                dest,
                ignore=lambda d, names: [n for n in names if n in USER_OWNED_EXCLUDE],
            )
        else:
            shutil.copy2(entry, dest)

    # Emit .env.EXAMPLE from manifest if the staged tree didn't ship one
    if manifest.env_requires and not (target / ENV_EXAMPLE_FILENAME).exists():
        (target / ENV_EXAMPLE_FILENAME).write_text(
            _env_template_from_manifest(manifest), encoding="utf-8"
        )

    # Make sure the manifest on disk reflects resolved name + source
    write_manifest(target, manifest)


def _bootstrap_user_dirs(target: Path) -> None:
    """Create the bootstrap dirs a fresh profile expects."""
    for d in ("memories", "sessions", "skills", "skins", "logs",
              "plans", "workspace", "cron", "home"):
        (target / d).mkdir(parents=True, exist_ok=True)


def install_distribution(
    source: str,
    name: Optional[str] = None,
    force: bool = False,
    create_alias: bool = False,
) -> InstallPlan:
    """Install a distribution from *source* into a new profile.

    Returns the resolved :class:`InstallPlan`.  Use :func:`plan_install`
    first if you want to preview + prompt the user before calling this.
    """
    from hermes_cli.profiles import (
        check_alias_collision,
        create_wrapper_script,
    )

    with tempfile.TemporaryDirectory(prefix="hermes_dist_install_") as tmp:
        plan = plan_install(source, Path(tmp), override_name=name)

        if plan.existing and not force:
            raise DistributionError(
                f"Profile '{plan.manifest.name}' already exists at {plan.target_dir}. "
                "Use `hermes profile update` to upgrade in place, "
                "or pass --force to overwrite."
            )

        # Fresh install: config.yaml comes from the distribution.
        _bootstrap_user_dirs(plan.target_dir)
        _copy_dist_payload(
            plan.staged_dir,
            plan.target_dir,
            plan.manifest,
            preserve_config=False,
        )

        if create_alias:
            collision = check_alias_collision(plan.manifest.name)
            if collision is None:
                create_wrapper_script(plan.manifest.name)

        return plan


def update_distribution(
    profile_name: str,
    force_config: bool = False,
) -> InstallPlan:
    """Re-pull the distribution for an existing profile and apply updates.

    The source is read from the installed profile's ``distribution.yaml``
    ``source:`` field.  Distribution-owned files are overwritten; user-owned
    data (memories, sessions, auth) is never touched.  ``config.yaml`` is
    preserved unless ``force_config`` is True.
    """
    from hermes_cli.profiles import (
        get_profile_dir,
        normalize_profile_name,
        validate_profile_name,
    )

    canon = normalize_profile_name(profile_name)
    validate_profile_name(canon)
    target = get_profile_dir(canon)
    if not target.is_dir():
        raise DistributionError(f"Profile '{canon}' does not exist.")

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
        plan = plan_install(
            existing_manifest.source,
            Path(tmp),
            override_name=canon,
        )
        plan.preserves_config = not force_config

        _copy_dist_payload(
            plan.staged_dir,
            plan.target_dir,
            plan.manifest,
            preserve_config=plan.preserves_config,
        )
        return plan


# ---------------------------------------------------------------------------
# Info — render a manifest summary
# ---------------------------------------------------------------------------


def describe_distribution(profile_name: str) -> Dict[str, Any]:
    """Return a structured view of a profile's distribution metadata.

    Returns an empty dict if the profile exists but has no manifest.
    Raises DistributionError if the profile itself doesn't exist.
    """
    from hermes_cli.profiles import (
        get_profile_dir,
        normalize_profile_name,
        validate_profile_name,
    )

    canon = normalize_profile_name(profile_name)
    validate_profile_name(canon)
    target = get_profile_dir(canon)
    if not target.is_dir():
        raise DistributionError(f"Profile '{canon}' does not exist.")
    manifest = read_manifest(target)
    if manifest is None:
        return {}
    return manifest.to_dict()
