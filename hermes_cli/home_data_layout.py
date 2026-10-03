"""Machine-specific state excluded from portable home transfers."""
from __future__ import annotations


# Dependency generations, downloaded tools, and partial transfers belong to this machine.
# ``build-tools`` is the native-dep build cache (vcpkg + static OpenSSL for win-arm64
# ``cryptography``, ~5 GiB / 65k files) — regenerable, machine-local, and must never be
# walked into by backups or cloned into a profile (#125237).
PM_RUNTIME_ROOT_DIRS = frozenset({"installs", "tools", "cache", "build-tools"})


def profile_root_entry(parts: tuple[str, ...]) -> str | None:
    """Locate a home-root entry without excluding same-named skill/plugin data."""
    if len(parts) >= 3 and parts[0] == "profiles":
        return parts[2]
    return parts[0] if parts else None
