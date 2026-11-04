"""Machine-specific state excluded from portable home transfers."""
from __future__ import annotations


# Dependency generations, downloaded tools, and partial transfers belong to this machine.
# ``build-tools`` is the native-dep build cache (vcpkg + static OpenSSL for win-arm64
# ``cryptography``, ~5 GiB / 65k files) — regenerable, machine-local, and must never be
# walked into by backups or cloned into a profile (#125237).
PM_RUNTIME_ROOT_DIRS = frozenset({"installs", "tools", "cache", "build-tools"})

# Bitwarden / 1Password secret-value caches and the Photon sidecar auth token, as POSIX paths relative to
# a home. They sit at its top, where a sandbox can hide a file by path (a writable folder above a hidden
# file can be renamed). Hermes regenerates them on use, so backups leave them out.
SECRET_CACHE_FILES = ("bws_cache.json", "bws_cache.enc.json", "op_cache.json", "photon-sidecar.json")
# Where earlier releases kept the same files, in folders the agent can write. Hermes deletes a copy
# there unread on first use.
LEGACY_SECRET_CACHE_PATHS = (
    "cache/bws_cache.json", "cache/bws_cache.enc.json", "cache/op_cache.json", "runtime/photon-sidecar.json",
)
# Credential stores for every guard that lists them.
SECRET_CACHE_STORES = (*SECRET_CACHE_FILES, *LEGACY_SECRET_CACHE_PATHS)


def profile_root_entry(parts: tuple[str, ...]) -> str | None:
    """Locate a home-root entry without excluding same-named skill/plugin data."""
    if len(parts) >= 3 and parts[0] == "profiles":
        return parts[2]
    return parts[0] if parts else None
