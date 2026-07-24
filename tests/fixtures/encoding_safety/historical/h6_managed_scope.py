"""Historical instance #6 — managed_scope._cached_read (parked residual).

Admin-managed config.yaml / .env read with plain encoding="utf-8". A UTF-8
BOM on the managed .env renames the first key; managed policy silently fails
to apply for that key. Scope-guarded (not filed) at #65124 time.
"""
from pathlib import Path
from typing import Dict
import copy


_ENV_CACHE: Dict[str, tuple] = {}
_CACHE_LOCK = __import__("threading").Lock()


def _parse_env(f):
    return {}


def _cached_read(path: Path, cache: Dict[str, tuple], parse):
    try:
        st = path.stat()
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size)
    path_key = str(path)
    with _CACHE_LOCK:
        hit = cache.get(path_key)
        if hit is not None and hit[:2] == key:
            return copy.deepcopy(hit[2])
    try:
        with open(path, encoding="utf-8") as f:
            parsed = parse(f)
    except Exception:
        return None
    with _CACHE_LOCK:
        cache[path_key] = (key[0], key[1], copy.deepcopy(parsed))
    return parsed


def load_managed_env(managed_dir: Path) -> dict:
    return _cached_read(managed_dir / ".env", _ENV_CACHE, _parse_env) or {}


def load_managed_config(managed_dir: Path) -> dict:
    return (
        _cached_read(managed_dir / "config.yaml", _ENV_CACHE, lambda f: {})
        or {}
    )
