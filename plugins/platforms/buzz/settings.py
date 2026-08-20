"""Live, profile-scoped access policy for the Buzz plugin."""

from __future__ import annotations

from collections.abc import Iterator, Set
from copy import deepcopy
from pathlib import Path
import re
import threading
from typing import Any, Callable, cast

import yaml


_BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
_POLICY_FIELDS = (
    "allowed_users",
    "allow_all_users",
    "require_mention",
    "thread_require_mention",
)
_DEFAULT_POLICY: dict[str, Any] = {
    "allowed_users": [],
    "allow_all_users": False,
    "require_mention": True,
    "thread_require_mention": True,
}
_ENV_KEYS = {
    "allowed_users": "BUZZ_ALLOWED_USERS",
    "allow_all_users": "BUZZ_ALLOW_ALL_USERS",
    "require_mention": "BUZZ_REQUIRE_MENTION",
    "thread_require_mention": "BUZZ_THREAD_REQUIRE_MENTION",
}


class PolicyEnvironmentError(RuntimeError):
    """A higher-precedence policy environment source could not be acquired."""


class EnvironmentOverrideFields(Set[str]):
    """Privacy-minimal override names plus a non-sensitive acquisition status."""

    def __init__(self, names: set[str] | None = None, *, error: str | None = None):
        self._names = frozenset(names or ())
        self.error = error

    def __contains__(self, name: object) -> bool:
        return name in self._names

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        return (
            f"EnvironmentOverrideFields(names={sorted(self._names)!r}, "
            f"error={self.error!r})"
        )


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


def _merge_policy_candidate(merged: dict[str, Any], candidate: Any) -> None:
    if not isinstance(candidate, dict):
        return
    for field in _POLICY_FIELDS:
        if field in candidate:
            merged[field] = candidate[field]
    extra = candidate.get("extra")
    if isinstance(extra, dict):
        for field in _POLICY_FIELDS:
            if field in extra:
                merged[field] = extra[field]


def _explicit_policy_values(config: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if not isinstance(config, dict):
        return merged
    gateway = config.get("gateway")
    gateway_platforms = gateway.get("platforms") if isinstance(gateway, dict) else None
    if isinstance(gateway_platforms, dict):
        _merge_policy_candidate(merged, gateway_platforms.get("buzz"))
    platforms = config.get("platforms")
    if isinstance(platforms, dict):
        _merge_policy_candidate(merged, platforms.get("buzz"))
    if isinstance(gateway, dict):
        _merge_policy_candidate(merged, gateway.get("buzz"))
    _merge_policy_candidate(merged, config.get("buzz"))
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


def policy_from_config(config: Any) -> dict[str, Any]:
    """Resolve explicit Buzz configuration over fail-closed defaults."""

    policy = deepcopy(_DEFAULT_POLICY)
    configured = _explicit_policy_values(config)
    if "allowed_users" in configured:
        raw = configured["allowed_users"]
        if isinstance(raw, str):
            raw = raw.split(",")
        if not isinstance(raw, (list, tuple)):
            raise ValueError("Buzz allowed_users must be a list")
        allowed_users: list[str] = []
        for entry in raw:
            if not isinstance(entry, str):
                raise ValueError("Buzz allowed_users entries must be text")
            identity = normalize_user_ref(entry)
            if identity is None:
                raise ValueError("Buzz allowed_users contains an invalid public key")
            if identity not in allowed_users:
                allowed_users.append(identity)
        policy["allowed_users"] = allowed_users
    for field in ("allow_all_users", "require_mention", "thread_require_mention"):
        if field in configured:
            policy[field] = _configured_bool(configured[field], field=field)
    return policy


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


class RuntimePolicyLoader:
    """Resolve and retain last-valid Buzz policy per exact profile scope."""

    def __init__(self) -> None:
        self._last_valid: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = threading.RLock()

    def _scope_key(
        self, config_path: Path, managed_dir: Path | None
    ) -> tuple[str, str]:
        config_key = str(config_path.expanduser().resolve(strict=False))
        managed_key = ""
        if managed_dir is not None:
            managed_key = str(
                (managed_dir / "config.yaml").expanduser().resolve(strict=False)
            )
        return config_key, managed_key

    def load(self, profile: str | None = None) -> dict[str, Any]:
        from hermes_cli import managed_scope
        from hermes_cli.config import _deep_merge, read_user_config_raw

        config_path = _profile_home(profile) / "config.yaml"
        managed_dir = managed_scope.get_managed_dir()
        scope_key = self._scope_key(config_path, managed_dir)
        with self._lock:
            try:
                config = read_user_config_raw(config_path)
                if managed_dir is not None:
                    managed_path = managed_dir / "config.yaml"
                    try:
                        managed_text = managed_path.read_text(encoding="utf-8")
                    except FileNotFoundError:
                        managed_text = ""
                    if managed_text:
                        parsed: Any = yaml.safe_load(managed_text) or {}
                        if not isinstance(parsed, dict):
                            raise ValueError("managed config root must be a mapping")
                        config = _deep_merge(config, cast(dict[str, Any], parsed))
            except Exception:
                return deepcopy(self._last_valid.get(scope_key, _DEFAULT_POLICY))
            try:
                explicit = _explicit_policy_values(config)
                expanded = _expand_scoped_env_vars(
                    explicit, _environment_getter(profile)
                )
                policy = policy_from_config({"buzz": {"extra": expanded}})
            except Exception:
                policy = deepcopy(_DEFAULT_POLICY)
                self._last_valid[scope_key] = deepcopy(policy)
                return policy
            self._last_valid[scope_key] = deepcopy(policy)
            return policy


_RUNTIME_POLICY_LOADER = RuntimePolicyLoader()


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


def _strict_policy_env(path: Path) -> dict[str, str]:
    """Read dotenv without collapsing absence and acquisition failure."""

    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as exc:
        raise PolicyEnvironmentError("environment_unavailable") from exc

    from hermes_cli.config import _parse_env_value

    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            raise PolicyEnvironmentError("environment_unavailable")
        key, _, raw_value = line.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in values:
            raise PolicyEnvironmentError("environment_unavailable")
        value = raw_value.strip()
        if value[:1] in {"'", '"'}:
            quote = value[0]
            escaped = False
            close = None
            for index, char in enumerate(value[1:], 1):
                if quote == '"' and char == "\\" and not escaped:
                    escaped = True
                    continue
                if char == quote and not escaped:
                    close = index
                    break
                escaped = False
            if close is None or (
                value[close + 1 :].strip()
                and not value[close + 1 :].strip().startswith("#")
            ):
                raise PolicyEnvironmentError("environment_unavailable")
            value = value[: close + 1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        try:
            values[key] = _parse_env_value(value)
        except Exception as exc:
            raise PolicyEnvironmentError("environment_unavailable") from exc
    return values


def _policy_multiplex_active() -> bool:
    try:
        from agent.secret_scope import is_multiplex_active

        return bool(is_multiplex_active())
    except Exception as exc:
        raise PolicyEnvironmentError("secret_scope_unavailable") from exc


def _environment_getter(profile: str | None) -> Callable[[str], str | None]:
    from hermes_cli import managed_scope

    managed_dir = managed_scope.get_managed_dir()
    managed = (
        _strict_policy_env(managed_dir / ".env") if managed_dir is not None else {}
    )
    if profile is None:
        values = (
            {}
            if _policy_multiplex_active()
            else _strict_policy_env(_profile_home(None) / ".env")
        )
        return lambda name: (
            managed[name]
            if name in managed
            else values[name]
            if name in values
            else _scoped_environment_value(name)
        )
    values = _strict_policy_env(_profile_home(profile) / ".env")
    return lambda name: managed[name] if name in managed else values.get(name)


def environment_override_fields(
    profile: str | None = None,
) -> EnvironmentOverrideFields:
    """Return only override names and non-sensitive acquisition status."""

    try:
        getenv = _environment_getter(profile)
        fields = {
            field for field, name in _ENV_KEYS.items() if getenv(name) is not None
        }
        if getenv("GATEWAY_ALLOW_ALL_USERS") is not None:
            fields.add("allow_all_users")
        return EnvironmentOverrideFields(fields)
    except Exception:
        return EnvironmentOverrideFields(error="environment_unavailable")


def with_environment_overrides(
    policy: dict[str, Any],
    getenv: Callable[[str], str | None] = _scoped_environment_value,
) -> dict[str, Any]:
    """Apply explicit environment policy over a resolved config policy."""

    effective = deepcopy(policy)
    allowed = getenv(_ENV_KEYS["allowed_users"])
    if allowed is not None:
        normalized: list[str] = []
        for entry in (part.strip() for part in allowed.split(",")):
            if not entry:
                continue
            identity = normalize_user_ref(entry)
            if identity is None:
                raise ValueError("BUZZ_ALLOWED_USERS contains an invalid public key")
            if identity not in normalized:
                normalized.append(identity)
        effective["allowed_users"] = normalized

    allow_all = getenv(_ENV_KEYS["allow_all_users"])
    if allow_all is not None:
        effective["allow_all_users"] = str(allow_all).strip().lower() in {
            "true",
            "1",
            "yes",
            "on",
        }
    for field in ("require_mention", "thread_require_mention"):
        value = getenv(_ENV_KEYS[field])
        if value is not None:
            effective[field] = _configured_bool(value, field=_ENV_KEYS[field])
    global_allow = getenv("GATEWAY_ALLOW_ALL_USERS")
    if global_allow is not None and str(global_allow).strip().lower() in {
        "true",
        "1",
        "yes",
        "on",
    }:
        effective["allow_all_users"] = True
    return effective


def effective_policy_with_environment(
    policy: dict[str, Any],
    getenv: Callable[[str], str | None] = _scoped_environment_value,
) -> dict[str, Any]:
    """Apply environment policy, failing closed on malformed explicit values."""

    try:
        return with_environment_overrides(policy, getenv)
    except Exception:
        return deepcopy(_DEFAULT_POLICY)


def effective_runtime_policy(profile: str | None = None) -> dict[str, Any]:
    """Return the current effective Buzz policy for one profile."""

    policy = _RUNTIME_POLICY_LOADER.load(profile)
    try:
        getenv = _environment_getter(profile)
    except Exception:
        return deepcopy(_DEFAULT_POLICY)
    return effective_policy_with_environment(policy, getenv)


def effective_authorization_policy(profile: str | None = None) -> dict[str, Any]:
    """Return only fields accepted by the central authorization resolver."""

    policy = effective_runtime_policy(profile)
    return {
        "allowed_users": policy["allowed_users"],
        "allow_all_users": policy["allow_all_users"],
    }
