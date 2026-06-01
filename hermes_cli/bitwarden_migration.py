"""Safe Bitwarden migration helpers for profile-local ``.env`` files.

These helpers are deliberately read-only until ``apply_prune_plan`` runs.
They never touch ``config.yaml`` beyond reading the Bitwarden settings that
control verification.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import yaml

from agent.secret_sources.bitwarden import fetch_bitwarden_secrets
from hermes_cli.profiles import get_active_profile_name, list_profiles, resolve_profile_env
from hermes_constants import get_hermes_home
from utils import atomic_replace, is_truthy_value

ENV_ASSIGNMENT_RE = re.compile(
    r"^\s*(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*="
)
ENV_SURFACE = ".env"

BOOTSTRAP_CLASS = "bootstrap-secret"
SECRET_CLASS = "secret"
NON_SECRET_CLASS = "non-secret"

_SECRET_MARKERS = (
    "API_KEY",
    "ACCESS_TOKEN",
    "REFRESH_TOKEN",
    "PRIVATE_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "_KEY",
)


@dataclass(frozen=True)
class BitwardenSettings:
    enabled: bool
    access_token_env: str
    project_id: str
    server_url: str


@dataclass(frozen=True)
class InventoryRow:
    profile: str
    surface: str
    env_path: Path
    key: str
    classification: str
    state: str


@dataclass(frozen=True)
class PruneRow:
    profile: str
    surface: str
    env_path: Path
    key: str
    classification: str
    state: str
    action: str
    reason: str
    bws_resolved: bool | None = None


@dataclass(frozen=True)
class PrunePlan:
    profile: str
    profile_home: Path
    env_path: Path
    config_path: Path
    settings: BitwardenSettings
    rows: list[PruneRow]
    warnings: list[str]
    verification_error: str | None = None

    def removable_keys(self) -> list[str]:
        return [row.key for row in self.rows if row.action == "remove"]


@dataclass(frozen=True)
class PruneResult:
    profile: str
    env_path: Path
    backup_path: Path | None
    removed_keys: list[str]
    changed: bool


def classify_env_key(key: str, *, bootstrap_key: str = "BWS_ACCESS_TOKEN") -> str:
    """Return a conservative secret classification for a single env var name."""
    if key == bootstrap_key:
        return BOOTSTRAP_CLASS

    upper = key.upper()
    if any(marker in upper for marker in _SECRET_MARKERS):
        return SECRET_CLASS
    return NON_SECRET_CLASS


def _read_yaml_mapping(config_path: Path) -> tuple[dict[str, object], list[str]]:
    warnings: list[str] = []
    if not config_path.exists():
        return {}, warnings

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - surfaced as a warning for the CLI
        return {}, [f"could not read {config_path.name}: {exc}"]

    if loaded is None:
        return {}, warnings
    if not isinstance(loaded, dict):
        return {}, [f"{config_path.name} did not contain a mapping"]
    return loaded, warnings


def load_bitwarden_settings(profile_home: Path) -> tuple[BitwardenSettings, list[str]]:
    """Read the Bitwarden config for one profile without mutating anything."""
    config_path = profile_home / "config.yaml"
    config, warnings = _read_yaml_mapping(config_path)
    secrets = config.get("secrets")
    bws = secrets.get("bitwarden") if isinstance(secrets, dict) else None
    if not isinstance(bws, dict):
        return (
            BitwardenSettings(
                enabled=False,
                access_token_env="BWS_ACCESS_TOKEN",
                project_id="",
                server_url="",
            ),
            warnings,
        )

    access_token_env = str(bws.get("access_token_env") or "BWS_ACCESS_TOKEN").strip()
    if not access_token_env:
        access_token_env = "BWS_ACCESS_TOKEN"

    return (
        BitwardenSettings(
            enabled=is_truthy_value(bws.get("enabled"), default=False),
            access_token_env=access_token_env,
            project_id=str(bws.get("project_id") or "").strip(),
            server_url=str(bws.get("server_url") or "").strip(),
        ),
        warnings,
    )


def _read_env_assignments(env_path: Path) -> list[tuple[str, str]]:
    """Parse a .env file into ordered ``(key, value)`` assignments.

    The parser is intentionally conservative: it only recognizes top-level
    assignments (with optional ``export``) and treats everything after the first
    ``=`` as value text.  That is enough for inventory and pruning without
    needing to load the file into ``os.environ``.
    """
    if not env_path.exists():
        return []

    assignments: list[tuple[str, str]] = []
    for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ENV_ASSIGNMENT_RE.match(line)
        if not match:
            continue
        key = match.group("key")
        raw_value = line[match.end() :]
        assignments.append((key, _normalize_env_value(raw_value)))
    return assignments


def _normalize_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""

    # Strip one layer of matching quotes if present.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]

    # Drop a simple inline comment for unquoted values.
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def _assignment_map(assignments: Iterable[tuple[str, str]]) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in assignments:
        values[key] = value
    return values


def _profile_targets(
    profile_name: str | None = None,
    *,
    all_profiles: bool = False,
) -> list[tuple[str, Path]]:
    if all_profiles:
        targets: list[tuple[str, Path]] = []
        for profile in list_profiles():
            targets.append((profile.name, profile.path))
        return targets

    if profile_name:
        return [(profile_name, Path(resolve_profile_env(profile_name)))]

    return [(get_active_profile_name() or "default", get_hermes_home())]


def inventory_rows_for_profile(profile_name: str, profile_home: Path) -> list[InventoryRow]:
    env_path = profile_home / ".env"
    if not env_path.exists():
        return []

    settings, _warnings = load_bitwarden_settings(profile_home)
    assignments = _read_env_assignments(env_path)
    values = _assignment_map(assignments)
    rows: list[InventoryRow] = []
    seen: set[str] = set()
    for key, _value in assignments:
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            InventoryRow(
                profile=profile_name,
                surface=ENV_SURFACE,
                env_path=env_path,
                key=key,
                classification=classify_env_key(
                    key,
                    bootstrap_key=settings.access_token_env,
                ),
                state="set" if values.get(key, "").strip() else "blank",
            )
        )
    return rows


def collect_inventory_rows(
    profile_name: str | None = None,
    *,
    all_profiles: bool = False,
) -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    for name, home in _profile_targets(profile_name, all_profiles=all_profiles):
        rows.extend(inventory_rows_for_profile(name, home))
    return rows


def prune_plan_for_profile(
    profile_name: str,
    profile_home: Path,
    *,
    fetcher: Callable[..., tuple[dict[str, str], list[str]]] | None = None,
) -> PrunePlan:
    """Build a safe, read-only plan for pruning plaintext secrets."""
    env_path = profile_home / ".env"
    config_path = profile_home / "config.yaml"
    settings, warnings = load_bitwarden_settings(profile_home)
    assignments = _read_env_assignments(env_path)
    values = _assignment_map(assignments)
    fetcher = fetcher or fetch_bitwarden_secrets

    rows: list[PruneRow] = []
    verification_error: str | None = None
    secret_names: set[str] = set()

    if not env_path.exists():
        return PrunePlan(
            profile=profile_name,
            profile_home=profile_home,
            env_path=env_path,
            config_path=config_path,
            settings=settings,
            rows=[],
            warnings=warnings,
            verification_error=None,
        )

    if not settings.enabled:
        verification_error = "Bitwarden is disabled in config.yaml"
    else:
        token = values.get(settings.access_token_env, "").strip()
        if not token:
            verification_error = (
                f"{settings.access_token_env} is not set in {env_path.name}"
            )
        elif not settings.project_id:
            verification_error = "Bitwarden project_id is not configured"
        else:
            try:
                fetched, fetch_warnings = fetcher(
                    access_token=token,
                    project_id=settings.project_id,
                    server_url=settings.server_url,
                    use_cache=False,
                    home_path=profile_home,
                )
                warnings.extend(fetch_warnings)
                secret_names = set(fetched)
            except Exception as exc:  # noqa: BLE001 - surfaced to the CLI
                verification_error = f"Bitwarden verification failed: {exc}"

    seen: set[str] = set()
    for key, value in assignments:
        if key in seen:
            continue
        seen.add(key)
        classification = classify_env_key(
            key,
            bootstrap_key=settings.access_token_env,
        )
        state = "set" if value.strip() else "blank"
        if classification == BOOTSTRAP_CLASS:
            rows.append(
                PruneRow(
                    profile=profile_name,
                    surface=ENV_SURFACE,
                    env_path=env_path,
                    key=key,
                    classification=classification,
                    state=state,
                    action="keep",
                    reason="bootstrap token is kept locally",
                    bws_resolved=None,
                )
            )
            continue

        if classification == NON_SECRET_CLASS:
            rows.append(
                PruneRow(
                    profile=profile_name,
                    surface=ENV_SURFACE,
                    env_path=env_path,
                    key=key,
                    classification=classification,
                    state=state,
                    action="keep",
                    reason="non-secret config stays in .env",
                    bws_resolved=None,
                )
            )
            continue

        if verification_error is not None:
            rows.append(
                PruneRow(
                    profile=profile_name,
                    surface=ENV_SURFACE,
                    env_path=env_path,
                    key=key,
                    classification=classification,
                    state=state,
                    action="keep",
                    reason=verification_error,
                    bws_resolved=None,
                )
            )
            continue

        if not state == "set":
            rows.append(
                PruneRow(
                    profile=profile_name,
                    surface=ENV_SURFACE,
                    env_path=env_path,
                    key=key,
                    classification=classification,
                    state=state,
                    action="keep",
                    reason="blank secret entry is left untouched",
                    bws_resolved=key in secret_names,
                )
            )
            continue

        if key in secret_names:
            rows.append(
                PruneRow(
                    profile=profile_name,
                    surface=ENV_SURFACE,
                    env_path=env_path,
                    key=key,
                    classification=classification,
                    state=state,
                    action="remove",
                    reason="verified in Bitwarden project",
                    bws_resolved=True,
                )
            )
        else:
            rows.append(
                PruneRow(
                    profile=profile_name,
                    surface=ENV_SURFACE,
                    env_path=env_path,
                    key=key,
                    classification=classification,
                    state=state,
                    action="keep",
                    reason="not found in Bitwarden project",
                    bws_resolved=False,
                )
            )

    return PrunePlan(
        profile=profile_name,
        profile_home=profile_home,
        env_path=env_path,
        config_path=config_path,
        settings=settings,
        rows=rows,
        warnings=warnings,
        verification_error=verification_error,
    )


def prune_plan_for_current_profile(
    *,
    fetcher: Callable[..., tuple[dict[str, str], list[str]]] | None = None,
) -> PrunePlan:
    profile_name = get_active_profile_name() or "default"
    return prune_plan_for_profile(
        profile_name,
        get_hermes_home(),
        fetcher=fetcher,
    )


def apply_prune_plan(plan: PrunePlan) -> PruneResult:
    """Apply a prune plan to ``plan.env_path`` and create a local backup."""
    remove_keys = {row.key for row in plan.rows if row.action == "remove"}
    if not remove_keys:
        return PruneResult(
            profile=plan.profile,
            env_path=plan.env_path,
            backup_path=None,
            removed_keys=[],
            changed=False,
        )

    original_lines = plan.env_path.read_text(encoding="utf-8", errors="replace").splitlines(
        keepends=True
    )
    kept_lines: list[str] = []
    removed_keys: list[str] = []
    for line in original_lines:
        match = ENV_ASSIGNMENT_RE.match(line)
        if match and match.group("key") in remove_keys:
            removed_keys.append(match.group("key"))
            continue
        kept_lines.append(line)

    if not removed_keys:
        return PruneResult(
            profile=plan.profile,
            env_path=plan.env_path,
            backup_path=None,
            removed_keys=[],
            changed=False,
        )

    backup_path = _make_backup_path(plan.env_path)
    shutil.copy2(plan.env_path, backup_path)
    _tighten_mode(backup_path)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(plan.env_path.parent),
        prefix=f".{plan.env_path.name}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("".join(kept_lines))
            fh.flush()
            os.fsync(fh.fileno())
        atomic_replace(tmp_path, plan.env_path)
        _tighten_mode(plan.env_path)
    except Exception:
        raise

    return PruneResult(
        profile=plan.profile,
        env_path=plan.env_path,
        backup_path=backup_path,
        removed_keys=removed_keys,
        changed=True,
    )


def _make_backup_path(env_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    candidate = env_path.with_name(
        f"{env_path.name}.bak-pre-bitwarden-prune-{stamp}"
    )
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = env_path.with_name(
            f"{env_path.name}.bak-pre-bitwarden-prune-{stamp}-{suffix}"
        )
    return candidate


def _tighten_mode(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
