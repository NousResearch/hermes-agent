"""Pure profile naming and validation primitives."""

from __future__ import annotations

import re

from hermes_constants import PROFILE_ID_RE

_PROFILE_ID_RE = PROFILE_ID_RE
_RESERVED_NAMES = frozenset({"hermes", "default", "test", "tmp", "root", "sudo"})
_HERMES_SUBCOMMANDS = frozenset({
    "chat", "model", "gateway", "setup", "whatsapp", "login", "logout",
    "status", "cron", "doctor", "dump", "config", "pairing", "skills", "tools",
    "mcp", "sessions", "insights", "version", "update", "uninstall", "profile",
    "plugins", "honcho", "acp",
})
_PROFILE_NAME_RULE = (
    "Use lowercase letters, numbers, '-' or '_', starting with a letter or number, "
    "up to 64 characters"
)


def _missing_profile_error(canon: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"Profile '{canon}' does not exist. Create it with: hermes profile create {canon}"
    )


def _unknown_profile_error(canon: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"No profile named '{canon}'. See your profiles with: hermes profile list"
    )


def _suggest_profile_name(name: str) -> str:
    candidate = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("-_")[:64]
    return candidate if _PROFILE_ID_RE.match(candidate) else "my-work"


def _invalid_profile_name_error(name: str) -> ValueError:
    suggestion = _suggest_profile_name(name)
    return ValueError(
        f"{name!r} is not a valid profile name. {_PROFILE_NAME_RULE} "
        f"(for example: {suggestion}). Then run `hermes profile create {suggestion}`."
    )


def normalize_profile_name(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    stripped = name.strip()
    if not stripped:
        raise ValueError("profile name cannot be empty")
    if stripped.casefold() == "default":
        return "default"
    return stripped.lower()


def validate_profile_name(name: str) -> None:
    if name == "default":
        return
    if not _PROFILE_ID_RE.match(name):
        raise _invalid_profile_name_error(name)
    if name in _RESERVED_NAMES:
        raise ValueError(
            f"Profile name {name!r} is reserved — it collides with either "
            f"the Hermes installation itself or a common system binary.  "
            f"Pick a different name."
        )


def validate_alias_name(name: str) -> None:
    if not _PROFILE_ID_RE.match(name):
        raise ValueError(f"Invalid alias name {name!r}. {_PROFILE_NAME_RULE}.")


def _canon_valid(name: str) -> str:
    canon = normalize_profile_name(name)
    validate_profile_name(canon)
    return canon
