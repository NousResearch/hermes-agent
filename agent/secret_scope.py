"""Profile-scoped credential resolution for multi-profile gateway multiplexing.

The multiplexing gateway serves many profiles from one process; each profile's
``.env`` keys **cannot** be unioned into ``os.environ`` (profile A's keys would
leak into profile B's turns and subprocesses). This module is a fail-closed,
context-local secret scope: ``set_secret_scope(mapping)`` installs the active
profile's secrets for the current task (a contextvar, so it propagates into the
agent's worker thread via ``copy_context()``); ``get_secret(name)`` reads from
it and, when multiplexing is active with no scope set, RAISES rather than
falling back to ``os.environ``. Design: ``docs/design/multiplexing-gateway.md``.
"""
from __future__ import annotations

import os
import re
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Dict, Mapping, Optional


# Process-global (describes the deployment mode, not a per-task value): set once
# at gateway startup when gateway.multiplex_profiles is true.
_MULTIPLEX_ACTIVE: bool = False


def set_multiplex_active(active: bool) -> None:
    """Mark whether the process is a profile multiplexer (get_secret fails closed)."""
    global _MULTIPLEX_ACTIVE
    _MULTIPLEX_ACTIVE = bool(active)


def is_multiplex_active() -> bool:
    return _MULTIPLEX_ACTIVE


_SECRET_SCOPE: ContextVar[Optional[Mapping[str, str]]] = ContextVar("_SECRET_SCOPE", default=None)


class UnscopedSecretError(RuntimeError):
    """A secret was read in multiplex mode with no scope installed.

    The fix is to wrap the call path in ``set_secret_scope(...)`` (the per-turn
    / per-adapter profile scope), not to widen the global allowlist.
    """


def set_secret_scope(secrets: Optional[Mapping[str, str]]) -> Token:
    """Install the active profile's secret mapping; ``None`` clears. Returns a reset token."""
    return _SECRET_SCOPE.set(secrets)


def reset_secret_scope(token: Token) -> None:
    _SECRET_SCOPE.reset(token)


def current_secret_scope() -> Optional[Mapping[str, str]]:
    """The active secret mapping, or None when no scope is installed."""
    return _SECRET_SCOPE.get()


# Genuinely-global env vars: process/deployment settings, NOT profile secrets.
# They keep reading os.environ even in multiplex mode (routing them through the
# fail-closed path would wrongly crash). Keep this tight — when in doubt a
# value is a profile secret. Membership is exact name OR prefix.
_GLOBAL_ENV_EXACT = frozenset({
    # Hermes runtime / deployment
    "HERMES_HOME", "HERMES_PROFILE", "HERMES_GATEWAY_LOCK_DIR",
    "HERMES_MAX_ITERATIONS", "HERMES_API_TIMEOUT",
    "HERMES_REDACT_SECRETS", "HERMES_NOUS_TIMEOUT_SECONDS",
    "_HERMES_GATEWAY",
    # OS / interpreter
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TZ", "PWD", "SHELL", "TMPDIR",
    "VIRTUAL_ENV", "PYTHONPATH", "SSL_CERT_FILE",
    # Kanban paths (per-board, not per-profile-secret)
    "HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_BOARD",
    # API-server LISTENER settings — deployment config (compose/systemd env),
    # which the scoped runner reload must keep seeing or containers silently
    # lose the api_server platform. API_SERVER_KEY is a credential: NOT here.
    # See #64674, #69379.
    "API_SERVER_ENABLED", "API_SERVER_HOST", "API_SERVER_PORT",
    "API_SERVER_CORS_ORIGINS",
    # Relay-connector ROUTING stamps injected by managed deploys. Every reader
    # (gateway.config, relay_url()/registration/self-provision) must resolve
    # the SAME value or the adapter registers while the platform is absent
    # from config. GATEWAY_RELAY_SECRET/_ID/_DELIVERY_KEY and IDP_* are auth
    # material and deliberately stay profile-scoped.
    "GATEWAY_RELAY_URL", "GATEWAY_RELAY_ENDPOINT",
    "GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS",
    "GATEWAY_RELAY_PLATFORMS", "GATEWAY_RELAY_BOT_IDS",
    "GATEWAY_RELAY_ROUTE_KEYS", "GATEWAY_RELAY_INSTANCE_ID",
    "GATEWAY_RELAY_WAKE_URL", "GATEWAY_RELAY_DISPLAY_NAME",
})
_GLOBAL_ENV_PREFIXES = (
    "HERMES_KANBAN_",
    "HERMES_TELEGRAM_",   # tuning knobs (batch delays, fallback toggles) — NOT the token
    "TERMINAL_",          # terminal/sandbox backend settings
)


def _is_global_env(name: str) -> bool:
    """True for genuinely process-global (non-profile-secret) env vars."""
    return name in _GLOBAL_ENV_EXACT or name.startswith(_GLOBAL_ENV_PREFIXES)


def _environ_or(name: str, default: Optional[str]) -> Optional[str]:
    val = os.environ.get(name)
    return val if val is not None else default


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a credential by env-var name, honoring the active profile scope.

    Global vars always read ``os.environ``. With a scope installed, a miss returns
    ``default`` under multiplexing (never another profile's ``os.environ`` value)
    but falls through to ``os.environ`` otherwise — single-profile deployments
    inject credentials via the process env (systemd, ``op run``), so the scope
    must stay a ``.env`` overlay, not a blindfold (otherwise cron 401s). With no
    scope: multiplex INACTIVE reads ``os.environ``; ACTIVE raises (fail closed).
    """
    if _is_global_env(name):
        return _environ_or(name, default)
    scope = _SECRET_SCOPE.get()
    if scope is not None:
        val = scope.get(name)
        if val is not None:
            return val
        return default if _MULTIPLEX_ACTIVE else _environ_or(name, default)
    if _MULTIPLEX_ACTIVE:
        raise UnscopedSecretError(
            f"get_secret({name!r}) called with no profile secret scope active "
            f"while multiplexing is on. This credential read must run inside a "
            f"set_secret_scope(...) block (the per-turn / per-adapter profile "
            f"scope). Reading os.environ here would risk leaking another "
            f"profile's value. See docs/design/multiplexing-gateway.md "
            f"(Workstream A)."
        )
    return _environ_or(name, default)


def _strip_inline_comment(value: str) -> str:
    """Strip a dotenv-style inline comment (python-dotenv semantics): quoted values
    scan to the matching close quote (backslash-aware for double quotes) and drop a
    trailing ``# ...``, else stay untouched; unquoted values truncate only at a
    ``#`` PRECEDED BY WHITESPACE (``foo#bar`` survives, ``value # c`` → ``value``)."""
    value = value.strip()
    if not value:
        return value
    quote = value[0]
    if quote in ("'", '"'):
        i = 1
        while i < len(value):
            ch = value[i]
            if quote == '"' and ch == "\\":
                i += 2  # skip the escaped character
                continue
            if ch == quote:
                return value[: i + 1] if value[i + 1:].lstrip().startswith("#") else value
            i += 1
        return value  # unterminated quote: leave as-is
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def load_env_file(env_path: Path, *, fail_closed: bool = False) -> Dict[str, str]:
    """Parse a ``.env`` file into a dict WITHOUT touching ``os.environ``: ``export``
    prefix, ``#`` comments, and the writer's quote escapes reversed via the canonical
    ``_parse_env_value``. ``utf-8-sig`` so a BOM doesn't prefix the first key."""
    secrets: Dict[str, str] = {}
    try:
        text = env_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return secrets
    except (OSError, UnicodeDecodeError) as exc:
        if fail_closed:
            raise RuntimeError(f"profile dotenv unavailable: {env_path.name}") from exc
        return secrets

    from hermes_cli.config import _parse_env_value

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if sep and key:
            secrets[key] = _parse_env_value(_strip_inline_comment(value))
    return secrets


def _profile_external_secret_values(home: Path, *, fail_closed: bool) -> Dict[str, str]:
    """Read the existing per-home external-source cache, without ambient fallback."""
    try:
        from hermes_cli.env_loader import get_secret_source_values
        return get_secret_source_values(home)
    except Exception:
        if fail_closed:
            raise
        return {}


def build_profile_secret_scope(
    hermes_home: Path, *, fail_closed_external: bool = False,
) -> Dict[str, str]:
    """Build the profile overlay; subprocess boundaries opt into fail-closed resolution.

    Dotenv wins over the optional .op.env bootstrap; cached external sources win
    over both, matching env_loader. Ordinary scope callers retain fail-open behavior.
    """
    home = Path(hermes_home)
    secrets = load_env_file(home / ".op.env", fail_closed=fail_closed_external)
    secrets.update(load_env_file(home / ".env", fail_closed=fail_closed_external))
    external = _profile_external_secret_values(home, fail_closed=fail_closed_external)
    secrets.update((k, v) for k, v in external.items() if not _is_global_env(k))
    return secrets


_PROFILE_OWNED_NAMES: dict[Path, set[str]] = {}
_PROFILE_OWNED_NAMES_LOCK = Lock()


def record_profile_owned_secret_names(home: str | os.PathLike, names) -> frozenset[str]:
    """Retain observed source names while their values may outlive a dotenv/cache reload.

    Only source ingestion and boundary capture add evidence. Never infer ownership
    from arbitrary ambient exports, and never clear it merely because a source resets.
    """
    key = Path(home).resolve()
    with _PROFILE_OWNED_NAMES_LOCK:
        owned = _PROFILE_OWNED_NAMES.setdefault(key, set())
        owned.update(name for name in names if not _is_global_env(name))
        return frozenset(owned)


def get_profile_owned_secret_names(
    hermes_home: str | os.PathLike, *, fail_closed_external: bool = False,
) -> frozenset[str]:
    """Exact dotenv/bootstrap/cached-external ownership, not credential-name heuristics."""
    return record_profile_owned_secret_names(hermes_home, build_profile_secret_scope(
        Path(hermes_home), fail_closed_external=fail_closed_external))


def profile_env_name(name: str) -> str:
    """Unwrap nested child-environment carriers before checking profile ownership."""
    prefixes = ("_HERMES_FORCE_", "APPTAINERENV_", "SINGULARITYENV_")
    while prefix := next((p for p in prefixes if name.startswith(p)), None):
        name = name[len(prefix):]
    return name


@dataclass(frozen=True)
class ProfileEnvBoundary:
    """Immutable source/target provenance; this is not an OS isolation boundary."""

    source_home: Path
    target_home: Path
    source_owned_names: frozenset[str]
    target_values: Mapping[str, str]

    @property
    def identity(self) -> str:
        return str(self.target_home)

    def sanitize(self, env: Mapping[str, str]) -> dict[str, str]:
        """Replace source-owned names with target values; drop their transport aliases."""
        result = dict(env)
        if self.source_home == self.target_home:
            return result
        owned = {profile_env_name(name) for name in self.source_owned_names}
        for name in tuple(result):
            effective = profile_env_name(name)
            if effective in owned and (name != effective or not _is_global_env(effective)):
                result.pop(name)
        for name in self.source_owned_names:
            if name in self.target_values:
                result[name] = self.target_values[name]
            else:
                result.pop(name, None)
        return result


def build_profile_env_boundary(
    source_home: str | os.PathLike | None = None,
    target_home: str | os.PathLike | None = None,
) -> ProfileEnvBoundary:
    """Capture launch ownership and the context-local (or explicit worker) target."""
    from hermes_constants import get_hermes_home_override, get_process_hermes_home

    source = Path(source_home if source_home is not None else get_process_hermes_home()).resolve()
    target = Path(target_home if target_home is not None else get_hermes_home_override() or source).resolve()
    return ProfileEnvBoundary(
        source_home=source, target_home=target,
        source_owned_names=get_profile_owned_secret_names(source, fail_closed_external=True),
        target_values=MappingProxyType(build_profile_secret_scope(target, fail_closed_external=True)),
    )
