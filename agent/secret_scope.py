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
from types import MappingProxyType
from typing import Dict, Mapping, Optional


# Process-global (describes the deployment mode, not a per-task value): set once
# at gateway startup when gateway.multiplex_profiles is true.
_MULTIPLEX_ACTIVE: bool = False
_ENV_KEYS_CASE_INSENSITIVE: bool = os.name == "nt"
_FORWARDED_ENV_PREFIXES = ("APPTAINERENV_", "SINGULARITYENV_")


def _env_name_key(name: str) -> str:
    value = str(name)
    changed = True
    while changed:
        changed = False
        upper = value.upper()
        for prefix in _FORWARDED_ENV_PREFIXES:
            if upper.startswith(prefix):
                value = value[len(prefix):]
                changed = True
                break
    return value.upper() if _ENV_KEYS_CASE_INSENSITIVE else value


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
    "HERMES_MAX_ITERATIONS", "HERMES_MAX_TOKENS", "HERMES_API_TIMEOUT",
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


def load_env_file(env_path: Path) -> Dict[str, str]:
    """Parse a ``.env`` file into a dict WITHOUT touching ``os.environ``: ``export``
    prefix, ``#`` comments, and the writer's quote escapes reversed via the canonical
    ``_parse_env_value``. ``utf-8-sig`` so a BOM doesn't prefix the first key."""
    secrets: Dict[str, str] = {}
    try:
        text = env_path.read_text(encoding="utf-8-sig")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
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


def build_profile_secret_scope(hermes_home: Path) -> Dict[str, str]:
    """Build a profile's secret mapping from ``<home>/.env`` plus its external
    secret sources. Global vars are NOT copied in — ``get_secret`` reads those
    from ``os.environ`` — so the scope holds only profile secrets."""
    secrets = load_env_file(Path(hermes_home) / ".env")
    try:
        from hermes_cli.env_loader import get_secret_source_values
        external_secrets = get_secret_source_values(Path(hermes_home))
    except Exception:
        external_secrets = {}
    secrets.update((k, v) for k, v in external_secrets.items() if not _is_global_env(k))
    return secrets


@dataclass(frozen=True)
class ProfileEnvBoundary:
    """Immutable source/target ownership boundary for a child environment.

    ``source_owned_names`` is deliberately name-based provenance from the
    launch/source profile, not a heuristic over variable spelling or a global
    value-equality scan. ``target_values`` contains only the target profile's
    values for those names, so an absent target value is removed rather than
    inherited from ambient ``os.environ``.
    """

    source_home: Path
    target_home: Path
    source_owned_names: frozenset[str]
    target_values: Mapping[str, str]

    @property
    def identity(self) -> str:
        """Stable target identity used by snapshot owners and diagnostics."""
        return str(self.target_home)

    def sanitize(self, env: Mapping[str, str]) -> dict[str, str]:
        """Return *env* with source-profile-owned names isolated to the target."""
        result = dict(env)
        if self.source_home == self.target_home:
            return result

        result_keys = {_env_name_key(key): key for key in result}
        target_values = {
            _env_name_key(key): (key, value)
            for key, value in self.target_values.items()
        }
        for name in self.source_owned_names:
            normalized = _env_name_key(name)
            existing_key = result_keys.get(normalized)
            target = target_values.get(normalized)
            if target is not None:
                target_key, value = target
                output_key = existing_key or target_key
                result[output_key] = value
                result_keys[normalized] = output_key
            elif existing_key is not None:
                result.pop(existing_key, None)
                result_keys.pop(normalized, None)
        return result


def get_profile_owned_secret_names(
    hermes_home: str | os.PathLike,
    *,
    fail_closed_external: bool = False,
) -> frozenset[str]:
    """Return exact secret names owned by one profile, without reading values.

    The profile's dotenv files and the external-source provenance snapshot are
    the ownership sources. Ordinary shell exports are intentionally excluded:
    they are user/process state, not profile-owned credentials.
    """
    home = Path(hermes_home)
    names = set(load_env_file(home / ".op.env"))
    names.update(load_env_file(home / ".env"))
    names.update(
        _profile_external_secret_values(
            home,
            fail_closed=fail_closed_external,
        )
    )
    return frozenset(name for name in names if not _is_global_env(name))


def build_profile_env_boundary(
    source_home: str | os.PathLike | None = None,
    target_home: str | os.PathLike | None = None,
) -> ProfileEnvBoundary:
    """Capture source/target profile identity and ownership for one execution.

    When homes are omitted, the source is the process launch home and the
    target is the context-local ``HERMES_HOME`` override, if present. Callers
    such as standalone Kanban pass both homes explicitly and therefore do not
    depend on gateway multiplex state.
    """
    if source_home is None:
        from hermes_constants import get_process_hermes_home

        source_home = get_process_hermes_home()
    if target_home is None:
        try:
            from hermes_constants import get_hermes_home_override

            target_home = get_hermes_home_override() or source_home
        except Exception as exc:
            if _MULTIPLEX_ACTIVE:
                raise RuntimeError(
                    "target profile home could not be resolved while multiplexing"
                ) from exc
            target_home = source_home
    source = Path(source_home).resolve()
    target = Path(target_home).resolve()
    target_values = build_profile_secret_scope(
        target,
        fail_closed_external=True,
    )
    return ProfileEnvBoundary(
        source_home=source,
        target_home=target,
        source_owned_names=get_profile_owned_secret_names(
            source,
            fail_closed_external=True,
        ),
        target_values=MappingProxyType(dict(target_values)),
    )


def sanitize_profile_owned_env(
    env: Mapping[str, str],
    boundary: ProfileEnvBoundary | None = None,
) -> dict[str, str]:
    """Apply a captured profile boundary without changing single-profile mode."""
    if boundary is None:
        return dict(env)
    return boundary.sanitize(env)
