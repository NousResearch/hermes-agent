"""Machine-specific state excluded from portable home transfers."""
from __future__ import annotations


# Dependency generations, downloaded tools, and partial transfers belong to this machine.
PM_RUNTIME_ROOT_DIRS = frozenset({"installs", "tools", "cache"})

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
# Home subpaths the agent must not rewrite with generic file tools or destroy from the shell without a
# prompt. Session transcripts (state.db, sessions/) are application-owned state whose rewrite can
# falsify history and break resume/compression; mcp-tokens/, pairing/, vault/ (key + ciphertext side
# by side), browser-profile/ (copied cookies / Login Data) and webapp/ (the Webapp's browser
# credential and its grants to native apps) hold credential material. skills/ and memories/ are
# deliberately NOT here: they are user-editable content, and write approval for them (memory/skills
# ``write_approval``) is enforced by the memory/skill subsystem tools, not the generic write path
# (#60440 tracks closing that gap for write_file/patch). Control files (auth.json,
# config.yaml, webhook_subscriptions.json) are deliberately NOT here (#45947): read-denied, but the
# user may ask to edit them.
HERMES_PROTECTED_SUBPATHS = (
    "state.db", "sessions", "mcp-tokens", "pairing", "vault", "browser-profile", "webapp")


def profile_root_entry(parts: tuple[str, ...]) -> str | None:
    """Locate a home-root entry without excluding same-named skill/plugin data."""
    if len(parts) >= 3 and parts[0] == "profiles":
        return parts[2]
    return parts[0] if parts else None
