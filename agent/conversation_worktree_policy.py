"""Platform-neutral configuration policy for conversation worktree isolation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any


class ConversationWorktreePolicyError(ValueError):
    """Raised when conversation worktree configuration is unsafe or incomplete."""


@dataclass(frozen=True)
class ConversationWorktreePolicy:
    enabled: bool
    source_worktree: Path | None
    worktree_root: Path | None
    branch_prefix: str = "hermes/session"
    bootstrap: bool = False
    bootstrap_command: tuple[str, ...] = ()
    bootstrap_timeout: float = 300.0
    create_timeout: float = 60.0
    retain_until_explicit_cleanup: bool = True
    legacy_location: bool = False


_BRANCH_FORBIDDEN = frozenset(" ~^:?*[\\")


def _section(config: Mapping[str, object]) -> tuple[Mapping[str, object], bool]:
    if "conversation_worktree" in config:
        section = config["conversation_worktree"]
        if section is None:
            # ``DEFAULT_CONFIG`` uses None as a presence sentinel. Deep merge
            # preserves it only when no user top-level policy exists, so the
            # legacy desktop block remains observable during the migration.
            pass
        elif not isinstance(section, Mapping):
            raise ConversationWorktreePolicyError("conversation_worktree must be a mapping")
        else:
            return section, False

    desktop = config.get("desktop", {})
    if not isinstance(desktop, Mapping):
        raise ConversationWorktreePolicyError("desktop must be a mapping")
    section = desktop.get("conversation_worktree", {})
    if not isinstance(section, Mapping):
        raise ConversationWorktreePolicyError("desktop.conversation_worktree must be a mapping")
    return section, bool(section)


def _boolean(section: Mapping[str, object], field: str, default: bool) -> bool:
    value = section.get(field, default)
    if not isinstance(value, bool):
        raise ConversationWorktreePolicyError(f"{field} must be a boolean")
    return value


def _absolute_path(section: Mapping[str, object], field: str) -> Path | None:
    value = section.get(field)
    if value is None or value == "":
        return None
    if not isinstance(value, (str, Path)):
        raise ConversationWorktreePolicyError(f"{field} must be an absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ConversationWorktreePolicyError(f"{field} must be an absolute path")
    return path.resolve()


def _positive_timeout(section: Mapping[str, object], field: str, default: float) -> float:
    value = section.get(field, default)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConversationWorktreePolicyError(f"{field} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ConversationWorktreePolicyError(f"{field} must be a finite positive number")
    return value


def _branch_prefix(section: Mapping[str, object]) -> str:
    value = section.get("branch_prefix", "hermes/session")
    if not isinstance(value, str):
        raise ConversationWorktreePolicyError("branch_prefix must be a non-empty safe branch prefix")
    prefix = value.strip()
    components = prefix.split("/")
    if (
        not prefix
        or prefix.startswith("/")
        or prefix.endswith("/")
        or "//" in prefix
        or ".." in prefix
        or "@{" in prefix
        or prefix.endswith(".")
        or prefix.endswith(".lock")
        or prefix == "@"
        or any(component.startswith(".") or component.endswith(".lock") for component in components)
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in prefix)
        or any(char in _BRANCH_FORBIDDEN or char.isspace() for char in prefix)
    ):
        raise ConversationWorktreePolicyError("branch_prefix must be a non-empty safe branch prefix")
    return prefix


def _bootstrap_command(section: Mapping[str, object]) -> tuple[str, ...]:
    value: Any = section.get("bootstrap_command", [])
    if not isinstance(value, list) or any(not isinstance(arg, str) or not arg for arg in value):
        raise ConversationWorktreePolicyError("bootstrap_command must be a list of non-empty strings")
    return tuple(value)


def resolve_conversation_worktree_policy(config: Mapping[str, object]) -> ConversationWorktreePolicy:
    """Resolve top-level policy, falling back to desktop.conversation_worktree."""
    if not isinstance(config, Mapping):
        raise ConversationWorktreePolicyError("configuration must be a mapping")

    section, legacy_location = _section(config)
    enabled = _boolean(section, "enabled", False)
    source_worktree = _absolute_path(section, "source_worktree")
    worktree_root = _absolute_path(section, "worktree_root")
    retain_until_explicit_cleanup = _boolean(section, "retain_until_explicit_cleanup", True)
    bootstrap = _boolean(section, "bootstrap", False)
    bootstrap_command = _bootstrap_command(section)

    if enabled and source_worktree is None:
        raise ConversationWorktreePolicyError("source_worktree is required when enabled")
    if enabled and worktree_root is None:
        raise ConversationWorktreePolicyError("worktree_root is required when enabled")
    if enabled and not retain_until_explicit_cleanup:
        raise ConversationWorktreePolicyError(
            "retain_until_explicit_cleanup must be true when enabled"
        )
    if bootstrap and not bootstrap_command:
        raise ConversationWorktreePolicyError(
            "bootstrap_command must be non-empty when bootstrap is enabled"
        )

    return ConversationWorktreePolicy(
        enabled=enabled,
        source_worktree=source_worktree,
        worktree_root=worktree_root,
        branch_prefix=_branch_prefix(section),
        bootstrap=bootstrap,
        bootstrap_command=bootstrap_command,
        bootstrap_timeout=_positive_timeout(section, "bootstrap_timeout", 300.0),
        create_timeout=_positive_timeout(section, "create_timeout", 60.0),
        retain_until_explicit_cleanup=retain_until_explicit_cleanup,
        legacy_location=legacy_location,
    )
