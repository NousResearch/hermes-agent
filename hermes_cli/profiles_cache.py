"""The per-profile derived-value cache behind ``list_profiles`` (``hermes_cli.profiles``)."""

import json
import math
from pathlib import Path
from typing import Optional



# (path, kind) -> (file signature, the small derived value). `list_profiles` re-reads three YAML
# files PER PROFILE, and it is the shared body of `GET /api/profiles` and `profiles.list`, which the
# Bots roster polls every 5s per connection — so an installer-seeded config.yaml (the annotated
# template, ~119KB) was re-parsed for every bot every five seconds to yield the same two strings.
# Only DERIVED values are cached, never a document a caller could write back: the raw readers
# (`read_user_config_raw`, `_load_yaml_dict`) keep their uncached contract. See #117378.
_PROFILE_FILE_CACHE: dict[tuple, tuple] = {}
_PROFILE_FILE_CACHE_MAX = 512


def _profile_file_signature(path: Path) -> Optional[tuple]:
    """``(mtime_ns, size, inode)``, or None when the file is absent. The atomic writers rename a
    temp file into place, so a rewrite always lands a new inode even within one mtime tick."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


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


def yaml_scalar_to_json(value):
    """``json.dumps`` default for YAML-only scalars (datetime/date → ISO 8601, else ``str``)."""
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def json_safe_value(value):
    """Deep copy of *value* with non-finite floats mapped to ``None``.

    ``json.dumps`` serializes ``inf``/``-inf``/``nan`` as the bare tokens ``Infinity``/``NaN``, which
    are not JSON: strict clients (Swift, Java, Go) reject the whole frame that carries one. YAML is the
    usual source — YAML 1.1 resolves an id like ``20260819_045052_92e993`` as a float (underscore digit
    separators plus an ``e`` exponent) and overflows to ``inf``. See #159335.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    return value


def json_safe_dumps(value) -> str:
    """``value`` as strict JSON (a cacheable, per-caller-copyable derived value), or ``"{}"`` when it
    cannot be one (unhashable YAML keys): one hand-edited profile.yaml must not fail a list.
    Non-finite floats are nulled rather than dropped with the rest of the document (#159335)."""
    try:
        return json.dumps(json_safe_value(value), default=yaml_scalar_to_json, allow_nan=False)
    except (TypeError, ValueError):
        return "{}"
