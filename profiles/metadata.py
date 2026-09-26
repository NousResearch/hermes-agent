"""Generic profile.yaml metadata primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

SETUP_ROLE = "setup"
PROFILE_ROLES = frozenset({SETUP_ROLE})

_PROFILE_FILE_CACHE: dict[tuple, tuple] = {}
_PROFILE_FILE_CACHE_MAX = 512


def _load_yaml_dict(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _profile_file_signature(path: Path) -> Optional[tuple]:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


def _cached_profile_read(path: Path, kind: str, compute):
    signature = _profile_file_signature(path)
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


def _clean_previous_names(raw) -> list[str]:
    if not isinstance(raw, list):
        return []
    cleaned: list[str] = []
    seen = set()
    for item in raw:
        name = str(item or "").strip()
        if name and name not in seen:
            seen.add(name)
            cleaned.append(name)
    return cleaned


def read_profile_meta(profile_dir: Path) -> dict:
    profile_dir = Path(profile_dir)

    def _read() -> dict:
        data = _load_yaml_dict(profile_dir / "profile.yaml") or {}
        ui_meta = data.get("ui_meta")
        bot_title = ""
        if isinstance(ui_meta, dict):
            hermes_bots = ui_meta.get("hermes-bots")
            if isinstance(hermes_bots, dict):
                bot_title = str(hermes_bots.get("title") or "").strip()
        return {
            "description": str(data.get("description") or "").strip(),
            "description_auto": bool(data.get("description_auto", False)),
            "display_name": str(data.get("display_name") or "").strip(),
            "bot_title": bot_title,
            "previous_names": _clean_previous_names(data.get("previous_names")),
            "role": data.get("role") if data.get("role") in PROFILE_ROLES else None,
        }

    meta = dict(_cached_profile_read(profile_dir / "profile.yaml", "profile-meta", _read))
    meta["previous_names"] = list(meta["previous_names"])
    return meta


def write_profile_meta(
    profile_dir: Path,
    *,
    description: Optional[str] = None,
    description_auto: Optional[bool] = None,
    display_name: Optional[str] = None,
    previous_names: Optional[list[str]] = None,
    role: Optional[str] = None,
) -> None:
    profile_dir = Path(profile_dir)
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"profile directory does not exist: {profile_dir}")
    if role is not None and role not in PROFILE_ROLES:
        raise ValueError(f"unknown profile role: {role!r}")
    path = profile_dir / "profile.yaml"
    existing: dict = _load_yaml_dict(path) or {}
    if role is not None:
        existing["role"] = role
    if description is not None:
        existing["description"] = description.strip()
    if description_auto is not None:
        existing["description_auto"] = bool(description_auto)
    if display_name is not None:
        if display_name.strip():
            existing["display_name"] = display_name.strip()
        else:
            existing.pop("display_name", None)
    if previous_names is not None:
        cleaned = _clean_previous_names(previous_names)
        if cleaned:
            existing["previous_names"] = cleaned
        else:
            existing.pop("previous_names", None)
    from utils import atomic_yaml_write

    atomic_yaml_write(path, existing, sort_keys=False)


def drop_profile_role(profile_dir: Path) -> None:
    path = Path(profile_dir) / "profile.yaml"
    existing = _load_yaml_dict(path)
    if not existing or "role" not in existing:
        return
    existing.pop("role")
    from utils import atomic_yaml_write

    atomic_yaml_write(path, existing, sort_keys=False)


def format_profile_label(name: str, display_name: Optional[str]) -> str:
    dn = (display_name or "").strip()
    return f"{dn} ({name})" if dn and dn != name else name
