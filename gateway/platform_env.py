"""Declarative env-credential bridge a platform plugin attaches to its ``PlatformEntry``.

A platform declares ``env_overrides`` (ordered ``step(config)`` callables, usually :class:`EnvCred`,
:func:`home_step`, :func:`reply_mode_step`) and ``env_enable_credentials`` (the env names its
explicit-disable WARNING cites). ``gateway.config_env._apply_env_overrides`` runs the steps of every
platform with activation evidence, so the data lives with the adapter instead of in a core table.

Import-light on purpose (only ``gateway.config``): adapters import it at module load.
"""

import contextlib
import functools
import importlib.util
import json
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig, _getenv_str

# Logger name parity with the origin module: records stay under "gateway.config".
logger = logging.getLogger("gateway.config")

getenv = _getenv_str


# --- small value parsers -----------------------------------------------------

def csv_list(value: str) -> list:
    return [part.strip() for part in value.split(",") if part.strip()]


def int_or(default: int) -> Callable[[str], int]:
    """``int(raw.strip(), 10)`` with *default* on malformed/blank input."""
    def parse(raw: str) -> int:
        try:
            return int(str(raw).strip(), 10)
        except (TypeError, ValueError):
            return default
    return parse


def strip_slash(value: str) -> str:
    return value.rstrip("/")


def strip_lower(value: str) -> str:
    return value.strip().lower()


def truthy_token(value: str) -> bool:
    return value.lower() in {"true", "1", "yes", "on"}


def mention_patterns(value: str) -> Any:
    """JSON list, else comma/newline separated."""
    try:
        return json.loads(value)
    except ValueError:  # json.JSONDecodeError
        return csv_list(value.replace("\n", ","))


def csv_extras(extra: dict[str, Any], spec) -> None:
    """``extra[key] = csv_list(raw)`` for each ``(key, raw)`` whose list is non-empty."""
    for key, raw in spec:
        items = csv_list(raw)
        if items:
            extra[key] = items


def env_first(envs) -> str:
    """First truthy value among *envs* (a single name or a tuple of alternatives)."""
    if isinstance(envs, str):
        return getenv(envs)
    return next((v for v in map(getenv, envs) if v), "")


# --- reusable steps -----------------------------------------------------------

INT = object()  # spec marker: ``int(value)``, silently skipped when malformed


def env_extras(extra: dict[str, Any], spec, *, strip: bool = False) -> None:
    """``extra[key] = fn(value)`` for each ``(key, env[, fn])`` whose env value is truthy.

    ``strip=True`` strips BEFORE the truthiness check. ``fn=INT`` parses an int
    and silently ignores a non-integer value.
    """
    for key, env, *fn in spec:
        value = getenv(env)
        if strip:
            value = value.strip()
        if not value:
            continue
        if fn and fn[0] is INT:
            with contextlib.suppress(ValueError):
                extra[key] = int(value)
        else:
            extra[key] = fn[0](value) if fn else value


def env_home_channel(config: GatewayConfig, platform: Platform, env_base: str, *, strip: bool = False) -> None:
    """Set ``home_channel`` from ``<env_base>`` (+``_NAME``/``_THREAD_ID``) when the platform is configured."""
    chat_id = getenv(env_base)
    if strip:
        chat_id = chat_id.strip()
    if chat_id and platform in config.platforms:
        config.platforms[platform].home_channel = HomeChannel(
            platform=platform, chat_id=chat_id,
            name=getenv(f"{env_base}_NAME", "Home"), thread_id=getenv(f"{env_base}_THREAD_ID") or None,
        )


def env_reply_mode(config: GatewayConfig, platform: Platform, env: str) -> None:
    mode = getenv(env).lower()
    if mode in {"off", "first", "all"}:
        config.platforms.setdefault(platform, PlatformConfig()).reply_to_mode = mode


# ``partial`` (not a closure) so ``hermes_cli.profile_channels`` can read the env names a step owns
# from ``step.keywords`` (``env`` / ``env_base``).
def home_step(platform: Platform, env_base: str, *, strip: bool = False):
    return partial(env_home_channel, platform=platform, env_base=env_base, strip=strip)


def reply_mode_step(platform: Platform, env: str):
    return partial(env_reply_mode, platform=platform, env=env)


def enable_from_env(config: GatewayConfig, platform: Platform) -> PlatformConfig:
    """Enable *platform* on env credentials unless config.yaml explicitly disabled it (warns once).
    Delegates to the core gate, which owns the explicit-disable marker and the one-time WARNING."""
    from gateway.config_env import _enable_from_env  # lazy: config_env imports this module
    return _enable_from_env(config, platform)


@dataclass(frozen=True)
class EnvCred:
    """Credential-gated platform enable, applied as ``step(config)``.

    ``creds``: env names that must ALL be truthy (inner tuple = ANY of). ``token``: env stored as
    ``PlatformConfig.token`` even when yaml disables the adapter (sending skills use it). ``fixed``:
    ``(extra_key, env[, default[, fn]])`` always written once enabled. ``optional*``: ``env_extras``
    specs. ``warn_missing``: ``(env, msg)`` logged BEFORE enabling when blank. ``then``: tail
    ``fn(config, platform_config)``. ``home``: ``env_home_channel`` env base applied only when the gate passed.
    """
    platform: Platform
    creds: tuple
    token: Optional[str] = None
    fixed: tuple = ()
    optional: tuple = ()
    optional_stripped: tuple = ()
    warn_missing: Optional[tuple] = None
    then: Optional[Callable[[GatewayConfig, PlatformConfig], None]] = None
    home: Optional[str] = None
    home_strip: bool = False

    def __call__(self, config: GatewayConfig) -> None:
        if not all(env_first(group) for group in self.creds):
            return
        if self.warn_missing and not getenv(self.warn_missing[0]):
            logger.warning(self.warn_missing[1])
        platform_config = enable_from_env(config, self.platform)
        if self.token and (token := getenv(self.token)):
            platform_config.token = token
        extra = platform_config.extra
        for key, env, *rest in self.fixed:
            default = rest[0] if rest else ""
            value = env_first(env) or default if isinstance(env, tuple) else getenv(env, default)
            extra[key] = rest[1](value) if len(rest) > 1 else value
        env_extras(extra, self.optional)
        env_extras(extra, self.optional_stripped, strip=True)
        if self.then is not None:
            self.then(config, platform_config)
        if self.home:
            env_home_channel(config, self.platform, self.home, strip=self.home_strip)


# --- bundled rows without the plugin -------------------------------------------

# Entries a bundled plugin registers under a name other than its directory, with the export prefix.
_BUNDLED_ENV_SIBLINGS = {"wecom_callback": ("wecom", "CALLBACK_")}


@functools.cache
def bundled_sibling(directory: str, stem: str = "gateway_env") -> Any:
    """``plugins/platforms/<directory>/<stem>.py`` (a light declaration module importing only core)
    executed from its file under a private name: the plugin package (``__init__``, adapter,
    ``register()``) is never imported. None when absent or *directory* is not a bundled plugin."""
    from gateway.config import Platform
    if directory not in Platform._scan_bundled_plugin_platforms()[0]:
        return None
    path = Path(__file__).resolve().parent.parent / "plugins" / "platforms" / directory / f"{stem}.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(f"_hermes_bundled_{stem}_{directory}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bundled_env_rows(name: str) -> Optional[tuple]:
    """``(env_overrides, env_enable_credentials)`` a BUNDLED platform plugin declares, read without
    the plugin. For when its registry entry is absent: discovery suppressed (a parked profile's
    config read, ``HERMES_SAFE_MODE``) or the plugin disabled. These rows were core's own table, and
    the duplicate-credential guard and the explicit-disable warning still need them then."""
    directory, prefix = _BUNDLED_ENV_SIBLINGS.get(name, (name, ""))
    module = bundled_sibling(directory)
    if module is None:
        return None
    return (tuple(getattr(module, f"{prefix}ENV_OVERRIDES", ()) or ()),
            tuple(getattr(module, f"{prefix}ENV_ENABLE_CREDENTIALS", ()) or ()))


def bundled_authz(name: str) -> dict:
    """``AUTHZ`` (allowlist / allow-all / allow-bots / own-policy env names) a bundled platform
    plugin declares in its ``gateway_env.py``, read without the plugin; {} for other names."""
    directory, prefix = _BUNDLED_ENV_SIBLINGS.get(name, (name, ""))
    module = bundled_sibling(directory)
    return dict(getattr(module, f"{prefix}AUTHZ", None) or {}) if module is not None else {}


def authz_field(name: str, field: str, entry: Any = None) -> Any:
    """*entry*'s authorization env field, else the bundled plugin's declaration. A user plugin
    overriding a bundled platform, safe mode or a disabled plugin then still gate on the names core
    always enforced for it; never looser than the bundled adapter."""
    value = getattr(entry, field, None) if entry is not None else None
    return value or bundled_authz(str(name or "").strip().lower()).get(field)
