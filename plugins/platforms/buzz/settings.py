"""Live mention policy, adapted from installed Buzz settings (5e9f111020).

Mention and authorization policies share exact-scope acquisition, not caches.
Raw policy reads intentionally avoid merged defaults and fail-open managed reads.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import re
import threading
from typing import Any, Callable

_POLICY_FIELDS = ("require_mention", "thread_require_mention")
_DEFAULT_POLICY = dict.fromkeys(_POLICY_FIELDS, True)
_ENV_KEYS = {field: "BUZZ_" + field.upper() for field in _POLICY_FIELDS}

from gateway.policy_environment import PolicyEnvironmentError, owner_environment_getter

def _merge_policy_candidate(merged: dict[str, Any], candidate: Any, fields=_POLICY_FIELDS) -> None:
    if not isinstance(candidate, dict):
        return
    for field in fields:
        if field in candidate:
            merged[field] = candidate[field]
    extra = candidate.get("extra")
    if isinstance(extra, dict):
        for field in fields:
            if field in extra:
                merged[field] = extra[field]

def _explicit_policy_values(config: Any, fields=_POLICY_FIELDS) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if not isinstance(config, dict):
        return merged
    gateway = config.get("gateway")
    gateway_platforms = gateway.get("platforms") if isinstance(gateway, dict) else None
    if isinstance(gateway_platforms, dict):
        _merge_policy_candidate(merged, gateway_platforms.get("buzz"), fields)
    platforms = config.get("platforms")
    if isinstance(platforms, dict):
        _merge_policy_candidate(merged, platforms.get("buzz"), fields)
    if isinstance(gateway, dict):
        _merge_policy_candidate(merged, gateway.get("buzz"), fields)
    _merge_policy_candidate(merged, config.get("buzz"), fields)
    return merged

def _configured_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Buzz {field} must be a boolean")

def _expand_scoped_env_vars(obj: Any, getenv: Callable[[str], str | None]) -> Any:
    """Expand config env refs through one authoritative profile getter."""

    from hermes_cli.config import _env_ref_var_name

    if isinstance(obj, str):

        def replace(match: re.Match[str]) -> str:
            name = _env_ref_var_name(match.group(1))
            if name is None:
                return match.group(0)
            value = getenv(name)
            return match.group(0) if value is None else value

        return re.sub(r"\${([^}]+)}", replace, obj)
    if isinstance(obj, dict):
        return {
            key: _expand_scoped_env_vars(value, getenv) for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_expand_scoped_env_vars(value, getenv) for value in obj]
    return obj

def _profile_home(profile: str | None) -> Path:
    if profile is None:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    from hermes_cli.profiles import resolve_profile_env

    return Path(resolve_profile_env(profile))

def _scoped_environment_value(name: str) -> str | None:
    """Read one active-profile value without multiplex cross-profile fallback."""

    try:
        from agent.secret_scope import UnscopedSecretError, get_secret
    except ImportError as exc:
        raise PolicyEnvironmentError("secret_scope_unavailable") from exc
    try:
        value = get_secret(name)
    except UnscopedSecretError as exc:
        raise PolicyEnvironmentError("secret_scope_unavailable") from exc
    except Exception as exc:
        raise PolicyEnvironmentError("secret_scope_unavailable") from exc
    try:
        return None if value is None else str(value)
    except Exception as exc:
        raise PolicyEnvironmentError("secret_scope_unavailable") from exc

def _read_policy_file(path: Path) -> dict:
    from hermes_cli.config import fast_safe_load
    try:
        with path.open(encoding="utf-8") as stream:
            value = fast_safe_load(stream)
    except FileNotFoundError:
        return {}
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Buzz policy document must be a mapping")
    # Validate supported containers before applying precedence. A broken alias
    # must not silently expose a more permissive lower-precedence value.
    def mapping(parent, key):
        if key not in parent:
            return {}
        node = parent[key]
        if not isinstance(node, dict):
            raise ValueError("Buzz policy container must be a mapping")
        return node
    gateway = mapping(value, "gateway")
    for parent in (mapping(gateway, "platforms"), mapping(value, "platforms"), gateway, value):
        candidate = mapping(parent, "buzz")
        mapping(candidate, "extra")
    return value


def policy_from_config(config: Any) -> dict[str, bool]:
    policy = deepcopy(_DEFAULT_POLICY)
    for field, value in _explicit_policy_values(config).items():
        policy[field] = _configured_bool(value, field=field)
    return policy


def _environment_getter(profile: str | None, home: Path | None = None):
    return owner_environment_getter(profile, home, _scoped_environment_value)


class RuntimePolicyLoader:
    """Last-valid raw policy keyed by exact owner and managed paths."""
    def __init__(self, *, fields=_POLICY_FIELDS, defaults=_DEFAULT_POLICY, parser=policy_from_config):
        self._fields, self._defaults, self._parser = fields, defaults, parser
        self._last_valid = {}
        self._lock = threading.RLock()

    def load(self, profile: str | None = None, *, home: Path | None = None):
        from hermes_cli import managed_scope
        try:
            owner = home if home is not None else _profile_home(profile)
            path = owner / "config.yaml"
            managed_dir = managed_scope.get_managed_dir()
            key = (str(path.resolve()), str((managed_dir / "config.yaml").resolve()) if managed_dir else "")
        except Exception:
            return deepcopy(self._defaults)
        with self._lock:
            try:
                explicit = _explicit_policy_values(_read_policy_file(path), self._fields)
                if managed_dir is not None:
                    # Resolve aliases within each layer before overlaying; managed
                    # canonical policy must outrank a local legacy alias.
                    explicit.update(_explicit_policy_values(_read_policy_file(managed_dir / "config.yaml"), self._fields))
            except Exception:
                return deepcopy(self._last_valid.get(key, self._defaults))
            try:
                expanded = _expand_scoped_env_vars(explicit, _environment_getter(profile, home))
                policy = self._parser({"buzz": expanded})
            except Exception:
                policy = deepcopy(self._defaults)
            self._last_valid[key] = deepcopy(policy)
            return policy


_RUNTIME_POLICY_LOADER = RuntimePolicyLoader()


# Authorization has a separate expanded last-good cache so malformed access
# settings cannot change mention-policy retention (or vice versa).
_AUTH_FIELDS = ("allowed_users", "allow_all_users")
_AUTH_DEFAULTS = {"allowed_users": [], "allow_all_users": False}
_BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

def _bech32_polymod(values: list[int]) -> int:
    checksum = 1
    generators = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)
    for value in values:
        top = checksum >> 25
        checksum = ((checksum & 0x1FFFFFF) << 5) ^ value
        for index, generator in enumerate(generators):
            if (top >> index) & 1:
                checksum ^= generator
    return checksum


def _convert_bits(data: list[int], from_bits: int, to_bits: int) -> list[int] | None:
    accumulator = 0
    bit_count = 0
    converted: list[int] = []
    mask = (1 << to_bits) - 1
    for value in data:
        if value < 0 or value >> from_bits:
            return None
        accumulator = (accumulator << from_bits) | value
        bit_count += from_bits
        while bit_count >= to_bits:
            bit_count -= to_bits
            converted.append((accumulator >> bit_count) & mask)
    if bit_count >= from_bits or ((accumulator << (to_bits - bit_count)) & mask):
        return None
    return converted


def normalize_user_ref(value: str) -> str | None:
    """Normalize a hex or npub Nostr identity to lowercase 32-byte hex."""

    normalized = str(value or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized):
        return normalized
    if not normalized.startswith("npub1"):
        return None
    try:
        data = [_BECH32_CHARSET.index(char) for char in normalized[5:]]
    except ValueError:
        return None
    hrp = [ord(char) >> 5 for char in "npub"] + [0]
    hrp.extend(ord(char) & 31 for char in "npub")
    if len(data) < 7 or _bech32_polymod(hrp + data) != 1:
        return None
    decoded = _convert_bits(data[:-6], 5, 8)
    if decoded is None or len(decoded) != 32:
        return None
    return bytes(decoded).hex()



def authorization_from_config(config):
    values = _explicit_policy_values(config, _AUTH_FIELDS)
    policy = deepcopy(_AUTH_DEFAULTS)
    if "allowed_users" in values:
        raw = values["allowed_users"]
        if isinstance(raw, str):
            raw = raw.split(",")
        if not isinstance(raw, (list, tuple)):
            raise ValueError("Buzz allowed_users must be a list")
        for item in raw:
            if not isinstance(item, str) or (identity := normalize_user_ref(item)) is None:
                raise ValueError("Buzz allowed_users contains an invalid public key")
            if identity not in policy["allowed_users"]:
                policy["allowed_users"].append(identity)
    if "allow_all_users" in values:
        policy["allow_all_users"] = _configured_bool(values["allow_all_users"], field="allow_all_users")
    return policy


_AUTH_POLICY_LOADER = RuntimePolicyLoader(
    fields=_AUTH_FIELDS, defaults=_AUTH_DEFAULTS, parser=authorization_from_config,
)


def effective_authorization_policy(profile=None, *, home=None):
    """Effective live access fields; explicit-empty environment revokes YAML."""
    try:
        policy = _AUTH_POLICY_LOADER.load(profile, home=home)
        getenv = _environment_getter(profile, home)
        allowed = getenv("BUZZ_ALLOWED_USERS")
        if allowed is not None:
            policy["allowed_users"] = [item.strip() for item in allowed.split(",") if item.strip()]
        allow_all = getenv("BUZZ_ALLOW_ALL_USERS")
        if allow_all is not None:
            policy["allow_all_users"] = allow_all.strip().lower() in {"true", "1", "yes", "on"}
        return authorization_from_config({"buzz": policy})
    except Exception:
        return deepcopy(_AUTH_DEFAULTS)


def effective_runtime_policy(profile: str | None = None, *, home: Path | None = None):
    try:
        policy = _RUNTIME_POLICY_LOADER.load(profile, home=home)
        getenv = _environment_getter(profile, home)
        for field, name in _ENV_KEYS.items():
            value = getenv(name)
            if value is not None:
                policy[field] = _configured_bool(value, field=name)
        return policy
    except Exception:
        return deepcopy(_DEFAULT_POLICY)
