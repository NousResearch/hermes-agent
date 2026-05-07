"""Shared file safety rules used by both tools and ACP shims."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


def _hermes_home_path() -> Path:
    """Resolve the active HERMES_HOME (profile-aware) without circular imports."""
    try:
        from hermes_constants import get_hermes_home  # local import to avoid cycles
        return get_hermes_home()
    except Exception:
        return Path(os.path.expanduser("~/.hermes"))


def _load_security_policy_config() -> dict[str, Any]:
    """Read the security policy block from active config.yaml.

    This module sits under ``agent/`` and is imported by low-level file tools,
    so avoid importing the full config loader here. A tiny raw YAML read keeps
    the write guard profile-aware without introducing a circular dependency.
    """
    config_path = _hermes_home_path() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    security = data.get("security")
    if not isinstance(security, dict):
        return {}
    policy = security.get("policy")
    return policy if isinstance(policy, dict) else {}


def _load_security_policy_filesystem_config() -> dict[str, Any]:
    """Read the legacy Hermes filesystem policy block from active config.yaml."""
    filesystem = _load_security_policy_config().get("filesystem")
    return filesystem if isinstance(filesystem, dict) else {}


def _load_openshell_filesystem_policy_config() -> dict[str, Any]:
    """Read the OpenShell-shaped filesystem_policy block from config.yaml."""
    filesystem = _load_security_policy_config().get("filesystem_policy")
    return filesystem if isinstance(filesystem, dict) else {}


def _as_config_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _deny_hermes_control_plane_enabled() -> bool:
    fs_cfg = _load_security_policy_filesystem_config()
    if "deny_hermes_control_plane" in fs_cfg:
        return _as_config_bool(fs_cfg.get("deny_hermes_control_plane"), True)
    openshell_policy = _load_openshell_filesystem_policy_config()
    return _as_config_bool(openshell_policy.get("deny_hermes_control_plane"), True)


def _configured_safe_write_root() -> str:
    fs_cfg = _load_security_policy_filesystem_config()
    raw = fs_cfg.get("write_safe_root", "")
    return str(raw).strip() if raw is not None else ""


def _configured_safe_write_roots() -> list[str]:
    roots: list[str] = []
    legacy_root = _configured_safe_write_root()
    if legacy_root:
        roots.append(legacy_root)

    openshell_policy = _load_openshell_filesystem_policy_config()
    read_write = openshell_policy.get("read_write", [])
    if isinstance(read_write, list):
        roots.extend(str(path).strip() for path in read_write if str(path).strip())

    return roots


def build_write_denied_paths(home: str) -> set[str]:
    """Return exact sensitive paths that must never be written."""
    hermes_home = _hermes_home_path()
    paths = [
        os.path.join(home, ".ssh", "authorized_keys"),
        os.path.join(home, ".ssh", "id_rsa"),
        os.path.join(home, ".ssh", "id_ed25519"),
        os.path.join(home, ".ssh", "config"),
        str(hermes_home / ".env"),
        os.path.join(home, ".bashrc"),
        os.path.join(home, ".zshrc"),
        os.path.join(home, ".profile"),
        os.path.join(home, ".bash_profile"),
        os.path.join(home, ".zprofile"),
        os.path.join(home, ".netrc"),
        os.path.join(home, ".pgpass"),
        os.path.join(home, ".npmrc"),
        os.path.join(home, ".pypirc"),
        "/etc/sudoers",
        "/etc/passwd",
        "/etc/shadow",
    ]

    if _deny_hermes_control_plane_enabled():
        paths.extend(
            [
                str(hermes_home / "auth.json"),
                str(hermes_home / "config.yaml"),
                str(hermes_home / "gateway.json"),
                str(hermes_home / "webhook_subscriptions.json"),
                str(hermes_home / ".anthropic_oauth.json"),
            ]
        )

    return {os.path.realpath(p) for p in paths}


def build_write_denied_prefixes(home: str) -> list[str]:
    """Return sensitive directory prefixes that must never be written."""
    hermes_home = _hermes_home_path()
    prefixes = [
        os.path.realpath(p) + os.sep
        for p in [
            os.path.join(home, ".ssh"),
            os.path.join(home, ".aws"),
            os.path.join(home, ".gnupg"),
            os.path.join(home, ".kube"),
            "/etc/sudoers.d",
            "/etc/systemd",
            os.path.join(home, ".docker"),
            os.path.join(home, ".azure"),
            os.path.join(home, ".config", "gh"),
        ]
    ]
    if _deny_hermes_control_plane_enabled():
        prefixes.extend(
            os.path.realpath(p) + os.sep
            for p in [
                str(hermes_home / "auth"),
                str(hermes_home / "mcp-tokens"),
            ]
        )
    return prefixes


def get_safe_write_roots() -> list[str]:
    """Return resolved write-safe roots, or an empty list if unset.

    ``HERMES_WRITE_SAFE_ROOT`` wins for backwards compatibility. Otherwise
    Hermes accepts the legacy ``security.policy.filesystem.write_safe_root``
    and the OpenShell-shaped ``security.policy.filesystem_policy.read_write``
    list from config.yaml.
    """
    root = os.getenv("HERMES_WRITE_SAFE_ROOT", "").strip()
    roots = [root] if root else _configured_safe_write_roots()
    resolved: list[str] = []
    seen: set[str] = set()
    for item in roots:
        if not item:
            continue
        try:
            real = os.path.realpath(os.path.expandvars(os.path.expanduser(item)))
        except Exception:
            continue
        if real not in seen:
            resolved.append(real)
            seen.add(real)
    return resolved


def get_safe_write_root() -> Optional[str]:
    """Return the first resolved write-safe root for legacy callers."""
    roots = get_safe_write_roots()
    if not roots:
        return None
    return roots[0]


def _is_under_any_root(resolved: str, roots: list[str]) -> bool:
    for safe_root in roots:
        if resolved == safe_root or resolved.startswith(safe_root + os.sep):
            return True
    return False


def is_write_denied(path: str) -> bool:
    """Return True if path is blocked by the write denylist or safe root."""
    home = os.path.realpath(os.path.expanduser("~"))
    resolved = os.path.realpath(os.path.expanduser(str(path)))

    if resolved in build_write_denied_paths(home):
        return True
    for prefix in build_write_denied_prefixes(home):
        if resolved.startswith(prefix):
            return True

    safe_roots = get_safe_write_roots()
    if safe_roots and not _is_under_any_root(resolved, safe_roots):
        return True

    return False


def get_read_block_error(path: str) -> Optional[str]:
    """Return an error message when a read targets internal Hermes cache files."""
    resolved = Path(path).expanduser().resolve()
    hermes_home = _hermes_home_path().resolve()
    blocked_dirs = [
        hermes_home / "skills" / ".hub" / "index-cache",
        hermes_home / "skills" / ".hub",
    ]
    for blocked in blocked_dirs:
        try:
            resolved.relative_to(blocked)
        except ValueError:
            continue
        return (
            f"Access denied: {path} is an internal Hermes cache file "
            "and cannot be read directly to prevent prompt injection. "
            "Use the skills_list or skill_view tools instead."
        )
    return None
