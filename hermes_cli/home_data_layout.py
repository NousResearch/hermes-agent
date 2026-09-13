"""Machine-specific state excluded from portable home transfers."""
from __future__ import annotations


# Dependency generations, downloaded tools, and partial transfers belong to this machine.
PM_RUNTIME_ROOT_DIRS = frozenset({"installs", "tools", "cache"})


def profile_root_entry(parts: tuple[str, ...]) -> str | None:
    """Locate a home-root entry without excluding same-named skill/plugin data."""
    if len(parts) >= 3 and parts[0] == "profiles":
        return parts[2]
    return parts[0] if parts else None
