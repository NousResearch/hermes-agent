"""Profile-scoped credential resolution for multi-profile gateway multiplexing.

The multiplexing gateway serves many profiles from one process; each profile's
``.env`` keys **cannot** be unioned into ``os.environ`` (profile A's keys would
leak into profile B's turns and subprocesses). This module is a fail-closed,
context-local secret scope: ``set_secret_scope(mapping)`` installs the active
profile's secrets for the current task (a contextvar, so it propagates into the
agent's worker thread via ``copy_context()``); ``get_secret(name)`` reads from
it and, when multiplexing is active with no scope set, RAISES rather than
falling back to ``os.environ``. Design: ``website/docs/developer-guide/multiplexing-gateway.md``.
"""
from __future__ import annotations

import codecs
import os
import re
import threading
from collections import OrderedDict
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Dict, Mapping, NamedTuple, Optional, Tuple

from utils import file_signature


# Process-global (describes the deployment mode, not a per-task value): set once
# at gateway startup when gateway.multiplex_profiles is true.
_MULTIPLEX_ACTIVE: bool = False
# Launch home pinned by set_multiplex_active(True) itself (None: no auto-pin outstanding).
_AUTO_PINNED_HOME = None


def set_multiplex_active(active: bool) -> None:
    """Mark whether the process is a profile multiplexer (get_secret fails closed).

    Activation also pins the launch home for routed-profile decisions
    (``hermes_constants.pin_process_hermes_home``) unless an embedding host already pinned one:
    from here on "is this task routed" compares the override against the home the process was
    launched with, not against whatever a host later mirrors into ``os.environ["HERMES_HOME"]``.
    Deactivation releases only the pin activation itself created — a transient toggle
    (``gateway_migrate._multiplex_read_mode``, a cron worker restoring the caller's mode) must not
    drop the host's explicit pin (#119242)."""
    global _MULTIPLEX_ACTIVE, _AUTO_PINNED_HOME
    from hermes_constants import (
        get_routing_process_hermes_home,
        pin_process_hermes_home,
        process_hermes_home_is_pinned,
    )
    _MULTIPLEX_ACTIVE = bool(active)
    if _MULTIPLEX_ACTIVE:
        if not process_hermes_home_is_pinned():
            _AUTO_PINNED_HOME = get_routing_process_hermes_home()
            pin_process_hermes_home(_AUTO_PINNED_HOME)
    elif _AUTO_PINNED_HOME is not None:
        if get_routing_process_hermes_home() == _AUTO_PINNED_HOME:
            pin_process_hermes_home(None)
        _AUTO_PINNED_HOME = None


def is_multiplex_active() -> bool:
    return _MULTIPLEX_ACTIVE


class _BoundScope(NamedTuple):
    """An installed secret scope plus the home it was built for, when the binder
    declared one — the provenance ``serves_routed_profile`` needs when the binding
    deliberately skips the HERMES_HOME override (kanban spawn-env builds, MCP
    owner scopes)."""

    mapping: Mapping[str, str]
    profile_home: Optional[str]


def serves_routed_profile() -> bool:
    """True when the current task runs for a profile other than the process's own: always under
    multiplexing, else when a HERMES_HOME override names another home (dashboard/desktop backend,
    per-profile cron ticker) or a secret scope stamped with a foreign home is bound. The MCP
    registry scope and the check_fn cache key both follow this predicate so a served profile's
    view never aliases the launch profile's (#111151). A host that mirrors the turn's profile into
    ``HERMES_HOME`` pins its own home with ``hermes_constants.pin_process_hermes_home`` so the
    mirror cannot flip this predicate."""
    if is_multiplex_active():
        return True
    from hermes_constants import get_hermes_home_override, get_routing_process_hermes_home, hermes_home_key
    own = hermes_home_key(get_routing_process_hermes_home())
    bound = _SECRET_SCOPE.get()
    if bound is not None and bound.profile_home and hermes_home_key(bound.profile_home) != own:
        return True
    override = get_hermes_home_override()
    return override is not None and hermes_home_key(override) != own


_SECRET_SCOPE: ContextVar[Optional[_BoundScope]] = ContextVar("_SECRET_SCOPE", default=None)


class UnscopedSecretError(RuntimeError):
    """A secret was read in multiplex mode with no scope installed.

    The fix is to wrap the call path in ``set_secret_scope(...)`` (the per-turn
    / per-adapter profile scope), not to widen the global allowlist.

    ``str(exc)`` is the ONE sentence an end user can act on; the developer diagnosis
    (which secret, which doc) rides ``__notes__`` so tracebacks and logs keep it.
    """

    def __init__(self, secret_name: str = "", developer_detail: str = ""):
        # Older callers passed the whole developer sentence positionally
        # (``UnscopedSecretError("get_secret('X') called with no scope ...")``); a secret
        # name never contains whitespace, so treat such a string as the detail.
        if secret_name and not developer_detail and any(ch.isspace() for ch in secret_name):
            secret_name, developer_detail = "", secret_name
        what = f"this profile's {secret_name}" if secret_name else "this profile's API key"
        super().__init__(
            f"Hermes could not read {what} (an internal profile-scoping bug on the multiplexed "
            "gateway, not your configuration). Run `hermes gateway restart`; if it keeps happening, "
            "report it with `hermes debug share`."
        )
        self.secret_name = secret_name
        self.developer_detail = developer_detail
        if developer_detail:
            self.add_note(developer_detail)


def set_secret_scope(secrets: Optional[Mapping[str, str]], *, profile_home: Optional[str] = None) -> Token:
    """Install the active profile's secret mapping; ``None`` clears. Returns a reset token.

    ``profile_home`` stamps the home the mapping was built for so
    ``serves_routed_profile`` detects a foreign-home scope even when the binder
    deliberately skips the HERMES_HOME override."""
    if secrets is None:
        return _SECRET_SCOPE.set(None)
    return _SECRET_SCOPE.set(_BoundScope(secrets, str(profile_home) if profile_home else None))


def reset_secret_scope(token: Token) -> None:
    _SECRET_SCOPE.reset(token)


def current_secret_scope() -> Optional[Mapping[str, str]]:
    """The active secret mapping, or None when no scope is installed."""
    bound = _SECRET_SCOPE.get()
    return bound.mapping if bound is not None else None


def current_secret_scope_home() -> Optional[str]:
    """The home the active scope was stamped with, or None when unstamped/unbound."""
    bound = _SECRET_SCOPE.get()
    return bound.profile_home if bound is not None else None


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
    bound = _SECRET_SCOPE.get()
    if bound is not None:
        val = bound.mapping.get(name)
        if val is not None:
            return val
        return default if (_MULTIPLEX_ACTIVE or serves_routed_profile()) else _environ_or(name, default)
    if _MULTIPLEX_ACTIVE:
        raise UnscopedSecretError(
            name,
            f"get_secret({name!r}) called with no profile secret scope active "
            f"while multiplexing is on. This credential read must run inside a "
            f"set_secret_scope(...) block (the per-turn / per-adapter profile "
            f"scope). Reading os.environ here would risk leaking another "
            f"profile's value. See website/docs/developer-guide/multiplexing-gateway.md "
            f"(Workstream A).",
        )
    return _environ_or(name, default)


def get_secret_str(name: str, default: str = "") -> str:
    """``get_secret`` for callers that want a ``str``: ``default`` only when the secret is genuinely
    unset. Still raises ``UnscopedSecretError`` — swallowing it hides a spawn-site bug."""
    val = get_secret(name, default)
    return default if val is None else val


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


def _parse_env_value(raw_value: str) -> str:
    """Parse the small .env value subset Hermes writes itself (bare, 'single', or "double" with
    ``\\"`` / ``\\\\`` escapes)."""
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        quoted = value[1:-1]
        parsed: list[str] = []
        i = 0
        while i < len(quoted):
            escaped = quoted[i] == "\\" and quoted[i + 1:i + 2] in ('"', "\\")
            parsed.append(quoted[i + 1] if escaped else quoted[i])
            i += 2 if escaped else 1
        return "".join(parsed)
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value


# Per-path memo of parsed ``.env`` files. ``build_profile_secret_scope()`` runs on every gateway
# turn, cron fire, MCP/browser adoption and housekeeping drain, and each call used to re-read and
# re-parse the whole file.
#
# FRESHNESS: every call still OPENS the file and keys on ``utils.file_signature`` of that
# descriptor's ``fstat`` (mtime_ns, size, inode, ctime_ns — ctime can't be backdated, so a pinned-
# timestamp rewrite is still seen), re-checked after the read. The open keeps close-to-open
# revalidation on NFS, a vanished/unreadable file fails the open and is never cached (a transient
# EACCES must not become "this profile has no secrets"), and the descriptor pins one inode so a
# symlink repointed mid-read can't file one file's contents under another's identity.
# ``invalidate_env_file_cache()`` is the explicit knob; ``hermes_cli.config.invalidate_env_cache()``
# calls it for Hermes's own .env writers.
_ENV_FILE_CACHE: "OrderedDict[str, Tuple[tuple, Dict[str, str]]]" = OrderedDict()
_ENV_FILE_CACHE_LOCK = threading.Lock()
_ENV_FILE_CACHE_MAX = 64  # one entry per profile home in practice


def invalidate_env_file_cache(env_path: Optional[Path] = None) -> None:
    """Drop one path from the ``load_env_file()`` memo, or all of them."""
    with _ENV_FILE_CACHE_LOCK:
        if env_path is None:
            _ENV_FILE_CACHE.clear()
        else:
            _ENV_FILE_CACHE.pop(str(env_path), None)


def _decode_env_bytes(raw: bytes) -> str:
    """BOM stripped; invalid UTF-8 falls back to latin-1 exactly as
    ``env_loader._load_dotenv_with_fallback`` installs it into ``os.environ``."""
    if raw.startswith(codecs.BOM_UTF8):
        raw = raw[len(codecs.BOM_UTF8):]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _parse_env_text(text: str) -> Dict[str, str]:
    """Tokenize already-read ``.env`` text. See :func:`load_env_file`."""
    secrets: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if sep and key:
            secrets[key] = _parse_env_value(_strip_inline_comment(value))
    return secrets


def load_env_file(env_path: Path, *, fail_closed: bool = False) -> Dict[str, str]:
    """THE ``.env`` tokenizer: every reader (profile scope, ``hermes_cli.config.load_env``, the dashboard
    scrub, skill secret capture, managed .env, setup prompts) parses through here so no two boundaries
    disagree on which keys/values a file defines. Dict only — never touches ``os.environ``. ``export``
    prefix, ``#`` comments, quote escapes reversed; a BOM is stripped so it doesn't prefix the first key.
    Invalid UTF-8 decodes as latin-1, exactly like ``env_loader._load_dotenv_with_fallback`` installs it
    into ``os.environ``. Absent/unreadable → ``{}``.

    Memoised per path on the open descriptor's stat identity (see the cache comment above). Always
    returns a fresh dict: callers mutate what they get back (``build_profile_secret_scope`` layers
    external secrets over it).
    """
    key = str(env_path)
    try:
        with open(env_path, "rb") as handle:
            fingerprint = file_signature(os.fstat(handle.fileno()))
            with _ENV_FILE_CACHE_LOCK:
                cached = _ENV_FILE_CACHE.get(key)
                if cached is not None and cached[0] == fingerprint:
                    _ENV_FILE_CACHE.move_to_end(key)
                    return dict(cached[1])
            raw = handle.read()
            # Same descriptor: a rewrite that landed between the fstat and the read is parsed but not
            # stored under the pre-write fingerprint.
            settled = file_signature(os.fstat(handle.fileno())) == fingerprint
    except FileNotFoundError:
        invalidate_env_file_cache(env_path)
        return {}
    except OSError as exc:
        # Gone or unreadable: drop any entry so a stale map cannot outlive the file.
        invalidate_env_file_cache(env_path)
        if fail_closed:
            raise RuntimeError(f"profile dotenv unavailable: {env_path.name}") from exc
        return {}

    secrets = _parse_env_text(_decode_env_bytes(raw))
    if settled:
        with _ENV_FILE_CACHE_LOCK:
            _ENV_FILE_CACHE[key] = (fingerprint, dict(secrets))
            _ENV_FILE_CACHE.move_to_end(key)
            while len(_ENV_FILE_CACHE) > _ENV_FILE_CACHE_MAX:
                _ENV_FILE_CACHE.popitem(last=False)
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


def _is_process_home(hermes_home: Path) -> bool:
    """True when *hermes_home* is the launch home, not a mirrored routed home."""
    from hermes_constants import get_routing_process_hermes_home
    try:
        return Path(hermes_home).resolve() == get_routing_process_hermes_home().resolve()
    except OSError:
        return False


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
    # The DEFAULT profile's config.yaml allow_all_users grant lives only in os.environ (bridged by
    # gateway.config_loader); scoped gate readers under multiplex never fall to os.environ, so seed it
    # into that profile's own mapping. A secondary never inherits it (#80099 class).
    from gateway.config_loader import bridged_allow_all_users
    bridged = bridged_allow_all_users()
    if bridged is not None and _is_process_home(home):
        secrets.setdefault("GATEWAY_ALLOW_ALL_USERS", bridged)
    record_profile_owned_secret_names(home, secrets)
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
        """Remove source/target ambient collisions; only replace shared source names.

        Target-only declarations need the child policy's explicit forwarding grant.
        """
        result = dict(env)
        if self.source_home == self.target_home:
            return result
        owned = {profile_env_name(name) for name in self.source_owned_names | self.target_values.keys()}
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
    from hermes_constants import get_hermes_home_override, get_routing_process_hermes_home

    source = Path(source_home if source_home is not None else get_routing_process_hermes_home()).resolve()
    target = Path(target_home if target_home is not None else get_hermes_home_override() or source).resolve()
    return ProfileEnvBoundary(
        source_home=source, target_home=target,
        source_owned_names=get_profile_owned_secret_names(source, fail_closed_external=True),
        target_values=MappingProxyType(build_profile_secret_scope(target, fail_closed_external=True)),
    )
