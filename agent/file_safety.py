"""Shared file safety rules used by both tools and ACP shims."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _hermes_home_path() -> Path:
    """Resolve the active HERMES_HOME (profile-aware) without circular imports."""
    try:
        from hermes_constants import get_hermes_home  # local import to avoid cycles
        return get_hermes_home()
    except Exception:
        return Path(os.path.expanduser("~/.hermes"))


def build_write_denied_paths(home: str) -> set[str]:
    """Return exact sensitive paths that must never be written."""
    hermes_home = _hermes_home_path()
    return {
        os.path.realpath(p)
        for p in [
            os.path.join(home, ".ssh", "authorized_keys"),
            os.path.join(home, ".ssh", "id_rsa"),
            os.path.join(home, ".ssh", "id_ed25519"),
            os.path.join(home, ".ssh", "config"),
            os.path.join(home, ".hermes", ".env"),
            str(hermes_home / ".env"),
            str(hermes_home / "config.yaml"),
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
    }


def build_write_denied_prefixes(home: str) -> list[str]:
    """Return sensitive directory prefixes that must never be written."""
    return [
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


def get_safe_write_root() -> Optional[str]:
    """Return the resolved HERMES_WRITE_SAFE_ROOT path, or None if unset."""
    root = os.getenv("HERMES_WRITE_SAFE_ROOT", "")
    if not root:
        return None
    try:
        return os.path.realpath(os.path.expanduser(root))
    except Exception:
        return None


def _is_outside_root(candidate: str, root: str) -> bool:
    """Return True when candidate is outside root (or incomparable)."""
    try:
        return os.path.commonpath([candidate, root]) != root
    except ValueError:
        # Mixed drives / invalid roots are always treated as outside.
        return True


def is_write_denied(
    path: str,
    home: str | None = None,
    base_dir: str | None = None,
) -> bool:
    """Return True if path is blocked by denylist or root constraints.

    Enforcement order is additive (most restrictive wins):
    1) static denylist/prefixes
    2) optional call-site ``base_dir`` sandbox
    3) optional ``HERMES_WRITE_SAFE_ROOT`` sandbox

    ``home`` selects which user-home denylist paths are evaluated.
    """
    home = os.path.realpath(os.path.expanduser(home or "~"))
    base_root = (
        os.path.realpath(os.path.expanduser(str(base_dir)))
        if base_dir
        else None
    )
    path_str = os.path.expanduser(str(path))
    if os.path.isabs(path_str):
        resolved = os.path.realpath(path_str)
    elif base_root:
        resolved = os.path.realpath(os.path.join(base_root, path_str))
        if _is_outside_root(resolved, base_root):
            logger.debug(
                "Denied write path outside base_dir: path=%r base_dir=%r",
                path,
                base_dir,
            )
            return True
    else:
        resolved = os.path.realpath(path_str)

    if resolved in build_write_denied_paths(home):
        return True
    for prefix in build_write_denied_prefixes(home):
        if resolved.startswith(prefix):
            return True

    # Absolute paths skip the relative-join check above, so enforce base_root
    # containment for both absolute and relative inputs here.
    if base_root and _is_outside_root(resolved, base_root):
        return True

    safe_root = get_safe_write_root()
    if safe_root and _is_outside_root(resolved, safe_root):
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
