"""Desired profile-serving policy for the host Gateway."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from profiles.current import get_active_profile_name
from profiles.paths import _get_default_hermes_home, get_profile_dir
from profiles.registry import _iter_named_profile_dirs

logger = logging.getLogger(__name__)

_STANDALONE_MEMO: Dict[str, Tuple[Optional[tuple], Optional[bool]]] = {}
_STANDALONE_WARNED = False
_STANDALONE_DEFAULT_WARNING = (
    "gateway.standalone is ignored on the default profile: it is the host gateway"
)
_parked_default_warned: set[Path] = set()


def profile_is_standalone(home: Path) -> bool:
    global _STANDALONE_WARNED
    from yaml import YAMLError
    from utils import file_signature

    home = Path(home)
    cfg_path = home / "config.yaml"
    key = str(home)
    try:
        signature = file_signature(cfg_path.stat())
    except FileNotFoundError:
        signature = None
    except OSError as exc:
        signature = ("stat-error", exc.errno)
        if _STANDALONE_MEMO.get(key) != (signature, False):
            logger.warning(
                "Cannot read gateway.standalone from %s (%s); treating as not standalone",
                cfg_path,
                type(exc).__name__,
            )
        _STANDALONE_MEMO[key] = (signature, False)
        return False

    cached = _STANDALONE_MEMO.get(key)
    if cached is not None and cached[0] == signature and cached[1] is not None:
        return cached[1]

    value = None
    if signature is not None:
        from hermes_cli.config import read_user_config_raw

        try:
            cfg = read_user_config_raw(cfg_path) or {}
        except (YAMLError, OSError, UnicodeError) as exc:
            if cached is None or cached[0] != signature:
                logger.warning(
                    "Cannot read gateway.standalone from %s (%s); treating as not standalone",
                    cfg_path,
                    type(exc).__name__,
                )
            _STANDALONE_MEMO[key] = (
                signature,
                False if isinstance(exc, YAMLError) else None,
            )
            return False
        if isinstance(cfg.get("gateway"), dict):
            value = cfg["gateway"].get("standalone")

    result = _standalone_truthy(value)
    if home == _get_default_hermes_home():
        if result and not _STANDALONE_WARNED:
            logger.warning(_STANDALONE_DEFAULT_WARNING)
            _STANDALONE_WARNED = True
        result = False
    _STANDALONE_MEMO[key] = (signature, result)
    return result


def _standalone_truthy(value: object) -> bool:
    from gateway.config import _bool_token

    if isinstance(value, str):
        return _bool_token(value) is True
    return bool(value)


def parked_marker_path(home: Path) -> Path:
    return Path(home) / "gateway.parked"


def profile_is_parked(home: Path) -> bool:
    return parked_marker_path(home).exists()


def profiles_to_serve(
    multiplex: bool,
    *,
    include_standalone: bool = False,
    include_parked: bool = False,
) -> List[Tuple[str, Path]]:
    active = get_active_profile_name() or "default"
    default = _get_default_hermes_home()
    if profile_is_parked(default) and default not in _parked_default_warned:
        logger.warning(
            "Ignoring gateway.parked for the default profile; stop the host gateway instead"
        )
        _parked_default_warned.add(default)
    if not multiplex:
        return [(active, get_profile_dir(active))]
    serve: List[Tuple[str, Path]] = [("default", default)]
    serve.extend(
        (entry.name, entry)
        for entry in _iter_named_profile_dirs()
        if (include_standalone or not profile_is_standalone(entry))
        and (include_parked or not profile_is_parked(entry))
    )
    return serve
