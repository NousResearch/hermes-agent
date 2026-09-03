"""Profile-scoped credential resolution for multi-profile gateway multiplexing.

The multiplexing gateway serves many profiles from one process. Each profile
has its own ``.env`` with its own provider keys and platform tokens, so we
**cannot** union them into the process-global ``os.environ`` (that would leak
profile A's keys to profile B's turns, and to every subprocess spawned with
``env=dict(os.environ)``).

This module provides a fail-closed, context-local secret scope:

- ``set_secret_scope(mapping)`` installs the active profile's secrets for the
  current task (a contextvar, so it propagates into the agent's worker thread
  via ``copy_context()`` exactly like the HERMES_HOME override).
- ``get_secret(name)`` reads from that scope. When multiplexing is **active**
  and no scope is set, it RAISES rather than silently falling back to
  ``os.environ`` — an un-migrated or newly-added call site fails loud at that
  exact line instead of leaking another profile's value. When multiplexing is
  **off** (the default), it transparently reads ``os.environ`` so the
  single-profile gateway and every non-gateway caller behave exactly as before.

Design rationale lives in ``docs/design/multiplexing-gateway.md`` (Workstream A).
"""
from __future__ import annotations

import os
import re
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Mapping, Optional


# ── multiplex-active flag ────────────────────────────────────────────────
# Process-global: set once at gateway startup when gateway.multiplex_profiles
# is true. Governs whether get_secret() fails closed on an unscoped read.
# A plain module global (not a contextvar): it describes the deployment mode,
# not a per-task value.
_MULTIPLEX_ACTIVE: bool = False


def set_multiplex_active(active: bool) -> None:
    """Mark whether the process is running as a profile multiplexer.

    Called once at gateway startup. When True, ``get_secret`` fails closed on
    an unscoped read instead of falling back to ``os.environ``.
    """
    global _MULTIPLEX_ACTIVE
    _MULTIPLEX_ACTIVE = bool(active)


def is_multiplex_active() -> bool:
    """Return whether the process is running as a profile multiplexer."""
    return _MULTIPLEX_ACTIVE


# ── the secret scope contextvar ──────────────────────────────────────────
_SECRET_SCOPE: ContextVar[Optional[Mapping[str, str]]] = ContextVar(
    "_SECRET_SCOPE", default=None
)


class UnscopedSecretError(RuntimeError):
    """Raised when a secret is read in multiplex mode with no scope installed.

    This is the fail-closed signal: it means a credential read reached
    ``get_secret`` without a profile scope active, which in a multiplexer would
    otherwise leak whichever profile's value happened to be in ``os.environ``.
    The fix is to wrap the call path in ``set_secret_scope(...)`` (the per-turn
    / per-adapter profile scope), not to widen the allowlist.
    """


def set_secret_scope(secrets: Optional[Mapping[str, str]]) -> Token:
    """Install the active profile's secret mapping for the current context.

    Returns a token for ``reset_secret_scope``. Pass ``None`` to clear.
    """
    return _SECRET_SCOPE.set(secrets)


def reset_secret_scope(token: Token) -> None:
    """Restore the previous secret scope."""
    _SECRET_SCOPE.reset(token)


def current_secret_scope() -> Optional[Mapping[str, str]]:
    """Return the active secret mapping, or None when no scope is installed."""
    return _SECRET_SCOPE.get()


# ── genuinely-global env vars (NOT per-profile secrets) ──────────────────
# These are process/deployment-level settings, not profile credentials. They
# legitimately live in os.environ and must keep reading from it even in
# multiplex mode — routing them through the fail-closed path would wrongly
# crash. Anything matching is read from os.environ regardless of scope.
#
# Membership test is by exact name OR prefix (see _is_global_env). Keep this
# list tight: when in doubt a value is a profile secret, not a global.
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
    # API-server LISTENER settings — deployment config (Docker compose
    # ``environment:`` block, systemd ``Environment=``), not profile secrets.
    # The scoped runner reload (#64674) must keep seeing them or container
    # deployments silently lose the api_server platform (#69379). NOTE:
    # API_SERVER_KEY is deliberately NOT here — it IS a credential and stays
    # profile-scoped.
    "API_SERVER_ENABLED", "API_SERVER_HOST", "API_SERVER_PORT",
    "API_SERVER_CORS_ORIGINS",
    # Relay-connector ROUTING stamps — deployment config injected into the
    # container/process env by managed deploys (the same shape as the
    # API_SERVER listener settings above). The scoped runner reload and the
    # relay-exclusive sweep in gateway/config.py must keep seeing them, and
    # every reader (gateway.config, gateway.relay.relay_url()/registration/
    # self-provision) must resolve the SAME value — a scope-dependent split
    # leaves the adapter registered but the platform absent from config (or
    # vice versa). Mirrors the non-secret/secret line drawn by the terminal
    # env blocklist (tools/environments/local.py): routing hints are global;
    # GATEWAY_RELAY_SECRET / GATEWAY_RELAY_ID / GATEWAY_RELAY_DELIVERY_KEY
    # and the IDP_* credentials are auth material and deliberately NOT here —
    # they stay profile-scoped with the fail-closed multiplex guard.
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
    """Return True for genuinely process-global (non-profile-secret) env vars."""
    if name in _GLOBAL_ENV_EXACT:
        return True
    return any(name.startswith(p) for p in _GLOBAL_ENV_PREFIXES)


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a credential by env-var name, honoring the active profile scope.

    Resolution order:

    1. Genuinely-global vars (``_is_global_env``) always read ``os.environ`` —
       they are deployment settings, not profile secrets.
    2. When a secret scope is installed (multiplexed turn), read from it. Under
       multiplexing the scope is authoritative — an absent key returns
       ``default`` and we do NOT fall through to ``os.environ``, because in a
       multiplexer ``os.environ`` may hold another profile's value. When
       multiplexing is OFF, a scope miss falls through to ``os.environ``:
       single-profile deployments legitimately provide credentials via the
       process environment (systemd ``Environment=``, secret-manager wrappers
       like ``pass-cli run`` / ``op run``, plain shell exports) rather than
       ``<home>/.env``, and the scope — installed unconditionally around e.g.
       every cron job — must stay a ``.env`` overlay, not a blindfold.
    3. No scope installed:
       - multiplex INACTIVE (default deployment): read ``os.environ`` —
         identical to the legacy ``os.getenv`` behavior every caller had before.
       - multiplex ACTIVE: FAIL CLOSED. Raise ``UnscopedSecretError`` so the
         missing scope is caught loudly instead of leaking a cross-profile value.
    """
    if _is_global_env(name):
        val = os.environ.get(name)
        return val if val is not None else default

    scope = _SECRET_SCOPE.get()
    if scope is not None:
        val = scope.get(name)
        if val is not None:
            return val
        if _MULTIPLEX_ACTIVE:
            return default
        # Multiplex off: the scope is an overlay over the process environment,
        # not an isolation boundary — there is no other profile to leak from.
        # Without this fallthrough, credentials injected only into the process
        # environment vanish inside any set_secret_scope(...) block (the cron
        # scheduler installs one around every job), so cron jobs send a
        # placeholder API key and 401 while interactive turns keep working.
        val = os.environ.get(name)
        return val if val is not None else default

    if _MULTIPLEX_ACTIVE:
        raise UnscopedSecretError(
            f"get_secret({name!r}) called with no profile secret scope active "
            f"while multiplexing is on. This credential read must run inside a "
            f"set_secret_scope(...) block (the per-turn / per-adapter profile "
            f"scope). Reading os.environ here would risk leaking another "
            f"profile's value. See docs/design/multiplexing-gateway.md "
            f"(Workstream A)."
        )

    val = os.environ.get(name)
    return val if val is not None else default


def _strip_inline_comment(value: str) -> str:
    """Strip a dotenv-style inline comment from a raw ``.env`` value.

    Mirrors python-dotenv (1.2.2) semantics, verified empirically:

    - Quoted values: scan for the matching close quote
      (backslash-escape-aware for double quotes, since ``save_env_value``
      writes ``\\"``/``\\\\`` escapes). Everything through the close quote is
      kept; a trailing ``# ...`` remainder after it is discarded, so
      ``KEY="has # inside" # trailing`` yields ``has # inside``. Non-comment
      trailing junk leaves the value untouched (lenient, unlike dotenv's
      hard parse error).
    - Unquoted values: truncate only at a ``#`` PRECEDED BY WHITESPACE, so
      ``KEY=foo#bar`` keeps ``foo#bar`` while ``KEY=value # comment`` keeps
      ``value``. A value that *starts* with ``#`` (``KEY=#leading``) is kept.
    """
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
                remainder = value[i + 1:].lstrip()
                if remainder.startswith("#"):
                    return value[: i + 1]
                return value
            i += 1
        return value  # unterminated quote: leave as-is
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def load_env_file(env_path: Path) -> Dict[str, str]:
    """Parse a ``.env`` file into a plain dict WITHOUT touching ``os.environ``.

    Used to load a profile's secrets into an isolated mapping for
    ``set_secret_scope``. Parses the small KEY=VALUE subset Hermes writes
    itself (``export`` prefix, ``#`` comments — full-line and
    dotenv-compatible inline, matching quotes with the
    writer's ``\\"``/``\\\\`` escapes reversed — the same semantics as
    ``hermes_cli.config._parse_env_value``) but never mutates the process
    environment — that isolation is the whole point.

    Encoding is ``utf-8-sig`` so a leading UTF-8 BOM (Windows Notepad /
    PowerShell ``Set-Content -Encoding UTF8``) does not prefix the first
    key as ``\\ufeffNAME`` and make ``get_secret('NAME')`` miss under scope.
    """
    secrets: Dict[str, str] = {}
    try:
        text = env_path.read_text(encoding="utf-8-sig")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return secrets

    # Parse values with the canonical Hermes parser: save_env_value
    # escapes " and \ inside double quotes, and every other reader
    # (load_env, python-dotenv) reverses those escapes. Stripping only
    # the outer quotes here would corrupt credentials containing "
    # or \ — they work interactively but fail in scoped (cron /
    # multiplex) resolution.
    from hermes_cli.config import _parse_env_value

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        secrets[key] = _parse_env_value(_strip_inline_comment(value))

    return secrets


def _profile_external_secret_values(
    home: Path,
    *,
    fail_closed: bool,
) -> Dict[str, str]:
    """Read cached external-source values with caller-selected failure policy."""
    try:
        from hermes_cli.env_loader import get_secret_source_values

        return get_secret_source_values(home)
    except Exception:
        if fail_closed:
            raise
        return {}


def build_profile_secret_scope(
    hermes_home: Path,
    *,
    fail_closed_external: bool = False,
) -> Dict[str, str]:
    """Build a profile's secret mapping from its ``.env`` and optional ``.op.env``.

    Returns a fresh dict (safe to install via ``set_secret_scope``). Genuinely
    global vars are intentionally NOT copied in — ``get_secret`` reads those
    from ``os.environ`` directly, so the scope holds only profile secrets.
    External-source failures preserve the historical fail-open behavior unless
    a subprocess security boundary explicitly requests fail-closed resolution.
    """
    home = Path(hermes_home)
    # ``.env`` wins over the optional bootstrap file, matching env_loader's
    # profile hydration contract. Both files are profile-owned inputs.
    secrets = load_env_file(home / ".op.env")
    secrets.update(load_env_file(home / ".env"))
    external_secrets = _profile_external_secret_values(
        home,
        fail_closed=fail_closed_external,
    )

    for key, value in external_secrets.items():
        if _is_global_env(key):
            continue
        secrets[key] = value

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
        for name in self.source_owned_names:
            if name in self.target_values:
                result[name] = self.target_values[name]
            else:
                result.pop(name, None)
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
        except Exception:
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
