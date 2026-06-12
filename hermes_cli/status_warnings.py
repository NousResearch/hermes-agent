"""Ops-safe Hermes status/doctor drift diagnostics.

These helpers intentionally return only configuration metadata and credential
health summaries. They must never surface access tokens, refresh tokens, API
keys, or raw exception messages that can contain secrets.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

from hermes_cli.config import check_config_version, get_hermes_home, load_config
from hermes_constants import get_default_hermes_root

_SAFE_CREDENTIAL_STATUS_KEYS = (
    "last_status",
    "last_error_code",
    "last_error_reason",
)

_SECRETISH_KEYS = frozenset({
    "access_token",
    "refresh_token",
    "api_key",
    "token",
    "key",
    "secret",
    "client_secret",
})


RunCallable = Callable[..., subprocess.CompletedProcess[str]]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _auth_store(home: Path) -> dict[str, Any]:
    return _read_json(home / "auth.json")


def _provider_pool_entries(store: dict[str, Any], provider: str) -> list[dict[str, Any]]:
    pool = store.get("credential_pool")
    if not isinstance(pool, dict):
        return []
    entries = pool.get(provider)
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _credential_pool_source(
    provider: str,
    *,
    hermes_home: Path | None = None,
    default_root: Path | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Return provider-slice pool source plus entries without changing auth semantics."""
    home = hermes_home or get_hermes_home()
    root = default_root or get_default_hermes_root()
    local_entries = _provider_pool_entries(_auth_store(home), provider)
    if local_entries:
        return "local", local_entries

    try:
        same_home = home.resolve(strict=False) == root.resolve(strict=False)
    except Exception:
        same_home = home == root
    if not same_home:
        global_entries = _provider_pool_entries(_auth_store(root), provider)
        if global_entries:
            return "global-fallback", global_entries
    return "none", []


def _sanitize_pool_status(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for entry in entries:
        status = {
            key: entry.get(key)
            for key in _SAFE_CREDENTIAL_STATUS_KEYS
            if entry.get(key) is not None
        }
        if status:
            statuses.append(status)
    return statuses


def _profile_name_for_home(home: Path, root: Path) -> str:
    try:
        if home.resolve(strict=False) == root.resolve(strict=False):
            return "default"
        profiles_root = root / "profiles"
        rel = home.resolve(strict=False).relative_to(profiles_root.resolve(strict=False))
        return rel.parts[0] if rel.parts else home.name
    except Exception:
        return home.name or "default"


def collect_provider_auth_diagnostic(
    *,
    config: dict[str, Any] | None = None,
    hermes_home: Path | None = None,
    default_root: Path | None = None,
) -> dict[str, Any]:
    """Collect non-secret auth/config drift diagnostics for the default provider."""
    cfg = config if config is not None else load_config()
    raw_model_cfg = cfg.get("model")
    model_cfg = raw_model_cfg if isinstance(raw_model_cfg, dict) else {}
    config_provider = str(model_cfg.get("provider") or "").strip().lower()
    home = hermes_home or get_hermes_home()
    root = default_root or get_default_hermes_root()
    auth_store = _auth_store(home)
    active_provider = auth_store.get("active_provider")
    if isinstance(active_provider, str):
        active_provider = active_provider.strip().lower() or None
    else:
        active_provider = None

    source, entries = _credential_pool_source(
        config_provider,
        hermes_home=home,
        default_root=root,
    ) if config_provider else ("none", [])

    profile_name = _profile_name_for_home(home, root)
    warnings: list[dict[str, Any]] = []
    if config_provider == "openai-codex" and active_provider != config_provider:
        warnings.append({
            "code": "auth_active_provider_drift",
            "severity": "warning",
            "message": (
                "model.provider is openai-codex but auth.active_provider is "
                f"{active_provider or 'unset'}"
            ),
            "config_provider": config_provider,
            "auth_active_provider": active_provider,
            "credential_pool_source": source,
        })
    if profile_name != "default" and config_provider and source == "global-fallback":
        warnings.append({
            "code": "profile_global_auth_fallback",
            "severity": "warning",
            "message": (
                f"profile {profile_name} is using global-root credential_pool "
                f"fallback for default provider {config_provider}"
            ),
            "profile": profile_name,
            "config_provider": config_provider,
            "credential_pool_source": source,
        })

    return {
        "profile": profile_name,
        "config_provider": config_provider or None,
        "auth_active_provider": active_provider,
        "credential_pool": {
            "provider": config_provider or None,
            "source": source,
            "entries": len(entries),
            "entry_statuses": _sanitize_pool_status(entries),
        },
        "warnings": warnings,
    }


def _profile_config_paths(default_root: Path) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    default_config = default_root / "config.yaml"
    if default_config.exists():
        paths.append(("default", default_config))
    profiles_root = default_root / "profiles"
    if profiles_root.is_dir():
        for entry in sorted(profiles_root.iterdir()):
            if not entry.is_dir() or entry.name == "default":
                continue
            config_path = entry / "config.yaml"
            if config_path.exists():
                paths.append((entry.name, config_path))
    return paths


def collect_config_version_drift(
    *,
    default_root: Path | None = None,
    latest_config_version: int | None = None,
) -> dict[str, Any]:
    """Collect profile config-version drift without reading secrets."""
    root = default_root or get_default_hermes_root()
    if latest_config_version is None:
        _, resolved_latest = check_config_version()
        latest_config_version = int(resolved_latest)

    profiles: list[dict[str, Any]] = []
    behind: list[dict[str, Any]] = []
    for name, config_path in _profile_config_paths(root):
        raw = _read_yaml(config_path)
        version = raw.get("_config_version", 0)
        try:
            version_int = int(version or 0)
        except (TypeError, ValueError):
            version_int = 0
        item = {"profile": name, "config_version": version_int}
        profiles.append(item)
        if version_int < latest_config_version:
            behind.append({
                "profile": name,
                "config_version": version_int,
                "latest_config_version": latest_config_version,
            })

    warning = None
    if behind:
        warning = {
            "code": "profile_config_version_drift",
            "severity": "warning",
            "message": (
                f"{len(behind)} profile(s) behind latest_config_version="
                f"{latest_config_version}"
            ),
            "behind": behind,
        }
    return {
        "latest_config_version": latest_config_version,
        "profiles": profiles,
        "behind": behind,
        "warning": warning,
    }


def _parse_launchctl_list(stdout: str) -> dict[str, int | None]:
    pid_match = re.search(r'"PID"\s*=\s*(\d+)', stdout)
    last_exit_match = re.search(r'"LastExitStatus"\s*=\s*(-?\d+)', stdout)
    return {
        "pid": int(pid_match.group(1)) if pid_match else None,
        "last_exit_status": int(last_exit_match.group(1)) if last_exit_match else None,
    }


def collect_launchd_gateway_warning(
    *,
    gateway_pid: int | None,
    run: RunCallable = subprocess.run,
) -> dict[str, Any] | None:
    """Warn when launchd status disagrees with a live gateway PID.

    Read-only: does not restart, bootout, bootstrap, or kickstart services.
    """
    if sys.platform != "darwin" or gateway_pid is None:
        return None
    try:
        from hermes_cli.gateway import get_launchd_label

        label = get_launchd_label()
        result = run(
            ["launchctl", "list", label],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return None

    parsed = _parse_launchctl_list(result.stdout or "")
    last_exit = parsed.get("last_exit_status")
    if result.returncode == 0 and (last_exit is None or last_exit == 0):
        return None
    return {
        "code": "launchd_gateway_state_mismatch",
        "severity": "warning",
        "message": (
            "gateway PID is running but launchd reports a nonzero "
            "status/LastExitStatus"
        ),
        "label": label,
        "gateway_pid": gateway_pid,
        "launchctl_status": result.returncode,
        "launchd_pid": parsed.get("pid"),
        "last_exit_status": last_exit,
    }


def collect_status_warnings(*, gateway_pid: int | None = None) -> dict[str, Any]:
    """Aggregate all ops-safe warning families for doctor/dashboard status."""
    provider_auth = collect_provider_auth_diagnostic()
    config_versions = collect_config_version_drift()
    warnings: list[dict[str, Any]] = []
    warnings.extend(provider_auth["warnings"])
    if config_versions["warning"]:
        warnings.append(config_versions["warning"])
    launchd_warning = collect_launchd_gateway_warning(gateway_pid=gateway_pid)
    if launchd_warning:
        warnings.append(launchd_warning)
    return {
        "warnings": warnings,
        "provider_auth": provider_auth,
        "config_versions": config_versions,
    }


def assert_no_secretish_keys(payload: Any) -> None:
    """Test helper: raise when a diagnostic payload exposes secret-shaped keys."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in _SECRETISH_KEYS:
                raise AssertionError(f"secret key leaked in diagnostic: {key}")
            assert_no_secretish_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            assert_no_secretish_keys(item)
