"""Small policy-gated credential broker for tool subprocesses.

This module is intentionally narrow: it lets tools request named credentials
from existing Hermes sources without teaching tools to read ~/.hermes/.env or
other secret stores directly. The first supported source is ``env`` so the
feature can be adopted incrementally without introducing a new secret backend.
"""

from __future__ import annotations

import logging
import os
import shlex
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BrokeredCredential:
    """A resolved credential value plus non-secret metadata."""

    name: str
    value: str
    env_name: str
    source: str = "env"

    @property
    def forced_env_name(self) -> str:
        """Return the internal env override name used by terminal backends."""

        return f"_HERMES_FORCE_{self.env_name}"


def _load_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        return load_config() or {}
    except Exception:
        logger.debug("credential broker: failed to load config", exc_info=True)
        return {}


def _broker_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(config or _load_config())
    credentials = cfg.get("credentials") or {}
    if not isinstance(credentials, Mapping):
        return {}
    broker = credentials.get("broker") or {}
    return dict(broker) if isinstance(broker, Mapping) else {}


def is_enabled(config: Mapping[str, Any] | None = None) -> bool:
    """Return whether the optional credential broker is enabled."""

    broker = _broker_config(config)
    return bool(broker.get("enabled", False))


def _requested_executable(command: str | None) -> str:
    if not command:
        return ""
    try:
        parts = shlex.split(command, posix=(os.name != "nt"))
    except ValueError:
        return ""
    if not parts:
        return ""
    return os.path.basename(parts[0])


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if item]
    return []


def _is_allowed(secret_cfg: Mapping[str, Any], *, requester: str, command: str | None) -> bool:
    allow = secret_cfg.get("allow") or {}
    if not isinstance(allow, Mapping) or not allow:
        return False

    tools = set(_as_str_list(allow.get("tools")))
    commands = set(_as_str_list(allow.get("commands")))
    if not tools and not commands:
        return False

    if tools and requester not in tools:
        return False

    if commands:
        executable = _requested_executable(command)
        if executable not in commands:
            return False

    return True


class CredentialBrokerError(RuntimeError):
    """Raised when brokered credential resolution is denied or unavailable."""


def resolve(
    name: str,
    *,
    requester: str,
    command: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> BrokeredCredential:
    """Resolve one named credential if broker config and policy allow it.

    Supported config shape::

        credentials:
          broker:
            enabled: true
            secrets:
              github_token:
                source: env
                name: GITHUB_TOKEN
                allow:
                  tools: [terminal]
                  commands: [gh]
    """

    broker = _broker_config(config)
    if not broker.get("enabled", False):
        raise CredentialBrokerError("credential broker is disabled")

    secrets = broker.get("secrets") or {}
    if not isinstance(secrets, Mapping) or name not in secrets:
        raise CredentialBrokerError(f"credential {name!r} is not configured")

    secret_cfg = secrets[name]
    if not isinstance(secret_cfg, Mapping):
        raise CredentialBrokerError(f"credential {name!r} has invalid configuration")

    source = str(secret_cfg.get("source") or "env")
    if source != "env":
        raise CredentialBrokerError(
            f"credential {name!r} uses unsupported source {source!r}; only 'env' is supported"
        )

    env_name = str(secret_cfg.get("name") or "").strip()
    if not env_name:
        raise CredentialBrokerError(f"credential {name!r} is missing env variable name")

    if not _is_allowed(secret_cfg, requester=requester, command=command):
        raise CredentialBrokerError(f"credential {name!r} is not allowed for {requester}")

    try:
        from hermes_cli.config import get_env_value

        value = get_env_value(env_name)
    except Exception:
        value = os.environ.get(env_name)
    if not value:
        raise CredentialBrokerError(f"credential {name!r} source env:{env_name} is not set")

    return BrokeredCredential(name=name, value=value, env_name=env_name, source=source)


def resolve_env_overrides(
    names: Sequence[str] | None,
    *,
    requester: str,
    command: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Resolve names into internal one-command environment overrides."""

    overrides: dict[str, str] = {}
    for name in names or []:
        credential = resolve(str(name), requester=requester, command=command, config=config)
        overrides[credential.forced_env_name] = credential.value
    return overrides


@contextmanager
def scoped_env_overrides(env: dict[str, str], overrides: Mapping[str, str]) -> Iterator[None]:
    """Temporarily add brokered env overrides to an environment object."""

    if not overrides:
        yield
        return

    sentinel = object()
    previous: dict[str, str | object] = {key: env.get(key, sentinel) for key in overrides}
    env.update({str(key): str(value) for key, value in overrides.items()})
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is sentinel:
                env.pop(key, None)
            else:
                env[key] = str(old_value)
