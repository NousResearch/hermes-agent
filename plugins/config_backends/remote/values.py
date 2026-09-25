"""Client-side value rules of the config plane wire contract.

- :func:`to_wire` — YAML-derived Python values → JSON values (contract.md §4.1, §4.2). Runs on the
  whole new user-layer document BEFORE the diff, so a conversion never shows up as a change (§4.3).
- :func:`secret_literal_path` — the secret-shaped-key rule (§14, D17), driven by the vendored
  ``config-secret-keys.json`` fixture so the agent and the plane use one list. The plane re-checks
  every write; this check exists so a literal is never sent.
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import re
from functools import lru_cache
from pathlib import Path as _FsPath
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

from hermes_cli.config_backend import ConfigValueError

from .paths import Path, encode

logger = logging.getLogger(__name__)

MAX_SAFE_INTEGER = 2**53 - 1
VENDOR_DIR = _FsPath(__file__).resolve().parent / "vendor"
SECRET_KEYS_FIXTURE = VENDOR_DIR / "conformance" / "config-secret-keys.json"


def to_wire(value: Any, path: Path = ()) -> Any:
    """``value`` converted per §4.1/§4.2; raises :class:`ConfigValueError` naming the path."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER:
            return value
        logger.warning("config path %s: integer %d is outside ±(2^53-1); sent as a decimal string",
                       encode(path) or "<root>", value)
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ConfigValueError(f"{encode(path) or '<root>'}: NaN/Infinity cannot be stored in Remote Config")
        return value
    if isinstance(value, datetime.datetime | datetime.date):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return [to_wire(v, path) for v in value]
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            key = _wire_key(k, path)
            if key in out:
                raise ConfigValueError(
                    f"{encode(path + (key,))}: two keys become the same string key {key!r}")
            out[key] = to_wire(v, path + (key,))
        return out
    raise ConfigValueError(
        f"{encode(path) or '<root>'}: a {type(value).__name__} value cannot be stored in Remote Config")


def _wire_key(key: Any, path: Path) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, int) and not isinstance(key, bool):
        return str(key)
    raise ConfigValueError(
        f"{encode(path) or '<root>'}: a {type(key).__name__} mapping key ({key!r}) cannot be stored "
        "in Remote Config; quote it in YAML")


# --- secret-shaped keys (§14) -----------------------------------------------------------------

@lru_cache(maxsize=1)
def _secret_rules() -> Dict[str, Any]:
    data = json.loads(SECRET_KEYS_FIXTURE.read_text(encoding="utf-8-sig"))
    return {
        "secret_keys": frozenset(data["secretKeys"]),
        "secret_suffixes": tuple(data["secretKeySuffixes"]),
        "non_secret_suffixes": tuple(data["nonSecretKeySuffixes"]),
        "env_keys": frozenset(data["envConfigKeys"]),
        "env_suffixes": tuple(data["envConfigKeySuffixes"]),
        "env_prefixes": tuple(data["envConfigKeyPrefixes"]),
        # fullmatch: Python's ``$`` also matches before a trailing newline, so ``re.match`` with the
        # fixture pattern would accept ``"${X}\n"`` (a fixture vector the plane rejects).
        "placeholder": re.compile(data["placeholderPattern"]),
        "non_secret_paths": frozenset(tuple(p) for p in data["nonSecretPaths"]),
    }


def _is_env_config_key(key: str, rules: Dict[str, Any]) -> bool:
    if "." in key:
        return False
    up = key.upper()
    return up in rules["env_keys"] or up.endswith(rules["env_suffixes"]) or up.startswith(rules["env_prefixes"])


def is_secret_path(segments: Sequence[str]) -> bool:
    """§14.2 on decoded segments (list indices already dropped)."""
    rules = _secret_rules()
    if tuple(segments) in rules["non_secret_paths"]:
        return False
    last = segments[-1]
    leaf = last.lower().replace("-", "_")
    if len(segments) == 1 and _is_env_config_key(last, rules):
        return not leaf.endswith(rules["non_secret_suffixes"])
    return leaf in rules["secret_keys"] or leaf.endswith(rules["secret_suffixes"])


def _walk(value: Any, path: Path) -> Iterator[Tuple[Path, Any]]:
    yield path, value
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _walk(v, path + (str(k),))
    elif isinstance(value, list):
        for v in value:  # list elements inherit the list's path (§14.3)
            yield from _walk(v, path)


def secret_literal_path(path: Path, value: Any) -> Optional[Path]:
    """The first path under ``set path = value`` that holds a secret literal (§14.3), or None."""
    placeholder = _secret_rules()["placeholder"]
    for node_path, node in _walk(value, path):
        if not node_path or not is_secret_path(node_path):
            continue
        if isinstance(node, bool) or node is None:
            continue
        if isinstance(node, str) and (node == "" or placeholder.fullmatch(node)):
            continue
        if isinstance(node, str | int | float):
            return node_path
    return None

