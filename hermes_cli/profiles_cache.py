"""The per-profile derived-value cache behind ``list_profiles`` (``hermes_cli.profiles``)."""

from pathlib import Path
from typing import Dict, Optional

from hermes_cli.config_backend import config_version


# (path, kind) -> (file signature, the small derived value). `list_profiles` re-reads three YAML
# files PER PROFILE, and it is the shared body of `GET /api/profiles` and `profiles.list`, which the
# Bots roster polls every 5s per connection — so an installer-seeded config.yaml (the annotated
# template, ~119KB) was re-parsed for every bot every five seconds to yield the same two strings.
# Only DERIVED values are cached, never a document a caller could write back: the raw readers
# (`read_user_config_raw`, `_load_yaml_dict`) keep their uncached contract. See #117378.
_PROFILE_FILE_CACHE: Dict[tuple, tuple] = {}
_PROFILE_FILE_CACHE_MAX = 512


def _profile_file_signature(path: Path) -> Optional[tuple]:
    """``(mtime_ns, size, inode)``, or None when the file is absent. The atomic writers rename a
    temp file into place, so a rewrite always lands a new inode even within one mtime tick."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


def _config_version_or_none(path: Path) -> Optional[tuple]:
    """The config backend's version of a profile's config.yaml (None when absent): the user layer
    changes through the backend, not necessarily on local disk."""
    try:
        return config_version(path)
    except OSError:
        return None


def _cached_profile_read(path: Path, kind: str, compute, signature_of=_profile_file_signature):
    """``compute()``'s value, reused while ``signature_of(path)`` is unchanged. A missing file is never
    cached: reading it costs nothing, and one created later must be picked up."""
    signature = signature_of(path)
    if signature is None:
        return compute()
    key = (str(path), kind)
    cached = _PROFILE_FILE_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    value = compute()
    if len(_PROFILE_FILE_CACHE) >= _PROFILE_FILE_CACHE_MAX:
        _PROFILE_FILE_CACHE.clear()
    _PROFILE_FILE_CACHE[key] = (signature, value)
    return value
