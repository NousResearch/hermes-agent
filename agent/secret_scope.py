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
import hashlib
import hmac
import json
import threading
from collections import OrderedDict
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Dict, Mapping, NamedTuple, Optional, Tuple

from utils import file_signature


# Process-global (describes the deployment mode, not a per-task value): set once
# at gateway startup when gateway.multiplex_profiles is true.
_MULTIPLEX_ACTIVE: bool = False
_ENV_KEYS_CASE_INSENSITIVE: bool = os.name == "nt"
_FORWARDED_ENV_PREFIXES = ("APPTAINERENV_", "SINGULARITYENV_")
_PROFILE_OWNED_NAME_HISTORY: dict[str, set[str]] = {}
_PROFILE_OWNED_NAME_HISTORY_LOCK = RLock()
# Scope generations may be surfaced in diagnostics, so bind credential values
# without publishing a reusable unsalted hash of those values.  Generations are
# process-local authority epochs; they are not a durable cross-process ID.
_SCOPE_VALUE_DIGEST_KEY = os.urandom(32)


@dataclass(frozen=True)
class _EnvCarrier:
    """Physical environment spelling plus its effective carried name."""

    physical_name: str
    effective_name: str
    channels: tuple[str, ...]

    @property
    def forwarded(self) -> bool:
        return bool(self.channels)


def _env_carrier(name: str) -> _EnvCarrier:
    physical = str(name)
    value = physical
    channels: list[str] = []
    while True:
        upper = value.upper()
        matched = False
        for prefix in _FORWARDED_ENV_PREFIXES:
            if upper.startswith(prefix):
                channels.append(prefix[:-1].lower())
                value = value[len(prefix):]
                matched = True
                break
        if not matched:
            break
    effective = value.upper() if _ENV_KEYS_CASE_INSENSITIVE else value
    return _EnvCarrier(physical, effective, tuple(channels))


def _env_name_key(name: str) -> str:
    return _env_carrier(name).effective_name


def _direct_env_name_key(name: str) -> str:
    """Normalize a physical direct name without granting wrapper equivalence."""
    value = str(name)
    return value.upper() if _ENV_KEYS_CASE_INSENSITIVE else value
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

    mapping: ProfileSecretScope
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

@dataclass(frozen=True, eq=False)
class ProfileSecretScope(Mapping[str, str]):
    """Immutable, identity-bound profile secret view for one generation."""

    profile_home: Path | None
    data: Mapping[str, str]
    owned_names: frozenset[str]
    generation: str
    source_status: str
    digest: str
    external_generation: int = 0

    def __getitem__(self, key: str) -> str:
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return NotImplemented


@dataclass(frozen=True)
class EnvFileSnapshot:
    """Non-mutating dotenv parse result with an explicit source state."""

    path: Path
    data: Mapping[str, str]
    status: str
    error_kind: str | None = None


def _scope_generation(
    profile_home: Path | None,
    values: Mapping[str, str],
    source_status: str,
    external_generation: int = 0,
) -> tuple[str, str]:
    files: list[tuple[str, int, int]] = []
    if profile_home is not None:
        for name in (".op.env", ".env", "config.yaml"):
            path = profile_home / name
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((name, stat.st_mtime_ns, stat.st_size))
    value_material = json.dumps(
        sorted((str(name), str(value)) for name, value in values.items()),
        separators=(",", ":"),
    ).encode()
    values_digest = hmac.new(
        _SCOPE_VALUE_DIGEST_KEY,
        value_material,
        hashlib.sha256,
    ).hexdigest()
    material = {
        "profile_home": str(profile_home) if profile_home is not None else None,
        "names": sorted(str(name) for name in values),
        "values_digest": values_digest,
        "files": files,
        "external_generation": int(external_generation),
        "source_status": source_status,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return f"profile-scope-v1:{digest}", digest


def _immutable_scope(
    values: Mapping[str, str],
    *,
    profile_home: Path | None,
    source_status: str,
    external_generation: int = 0,
) -> ProfileSecretScope:
    profile_home = profile_home.resolve() if profile_home is not None else None
    copied = {str(key): str(value) for key, value in values.items()}
    generation, digest = _scope_generation(
        profile_home,
        copied,
        source_status,
        external_generation,
    )
    return ProfileSecretScope(
        profile_home=profile_home.resolve() if profile_home is not None else None,
        data=MappingProxyType(copied),
        owned_names=frozenset(copied),
        generation=generation,
        source_status=source_status,
        digest=digest,
        external_generation=external_generation,
    )


def set_secret_scope(secrets: Optional[Mapping[str, str]], *, profile_home: Optional[str] = None) -> Token:
    """Install the active profile's secret mapping; ``None`` clears. Returns a reset token.

    ``profile_home`` stamps the home the mapping was built for so
    ``serves_routed_profile`` detects a foreign-home scope even when the binder
    deliberately skips the HERMES_HOME override."""
    if secrets is None:
        return _SECRET_SCOPE.set(None)
    if isinstance(secrets, ProfileSecretScope):
        scope = secrets
        if (profile_home is not None and scope.profile_home is not None
                and Path(profile_home).resolve() != scope.profile_home.resolve()):
            raise RuntimeError("profile secret scope home does not match requested profile home")
        stamped_home = profile_home or (str(scope.profile_home) if scope.profile_home else None)
    else:
        scope = _immutable_scope(
            secrets,
            profile_home=Path(profile_home) if profile_home else None,
            source_status="legacy_mapping",
        )
        stamped_home = profile_home
    return _SECRET_SCOPE.set(_BoundScope(scope, str(stamped_home) if stamped_home else None))


def reset_secret_scope(token: Token) -> None:
    _SECRET_SCOPE.reset(token)


def current_secret_scope() -> Optional[ProfileSecretScope]:
    """The active secret mapping, or None when no scope is installed."""
    bound = _SECRET_SCOPE.get()
    return bound.mapping if bound is not None else None


def current_secret_scope_home() -> Optional[str]:
    """The home the active scope was stamped with, or None when unstamped/unbound."""
    bound = _SECRET_SCOPE.get()
    return bound.profile_home if bound is not None else None


def add_secret_scope_defaults(defaults: Mapping[str, str], *, profile_home: str | os.PathLike) -> None:
    """Publish request-local defaults without mutating or relabelling the bound authority."""
    scope = current_secret_scope()
    if scope is None:
        raise UnscopedSecretError("", "scoped defaults require a bound profile scope")
    stamped_home = current_secret_scope_home()
    owner = stamped_home or scope.profile_home
    if owner is not None and Path(owner).resolve() != Path(profile_home).resolve():
        raise RuntimeError("profile defaults home does not match active profile scope")
    values = dict(scope)
    for name, value in defaults.items():
        values.setdefault(name, value)
    if values == dict(scope):
        return
    replacement = _immutable_scope(
        values, profile_home=scope.profile_home, source_status=scope.source_status,
        external_generation=scope.external_generation,
    )
    _SECRET_SCOPE.set(_BoundScope(replacement, stamped_home))


def update_secret_scope(
    name: str, value: Optional[str], *, profile_home: str | os.PathLike,
) -> bool:
    """Publish a persisted change without retaining its old source generation."""
    scope = current_secret_scope()
    stamped_home = current_secret_scope_home()
    if scope is None:
        return False
    if stamped_home is not None:
        if Path(stamped_home).resolve() != Path(profile_home).resolve():
            raise RuntimeError("persisted environment home does not match active profile scope")
        # A dotenv write may also invalidate the external-source snapshot. Rebuild
        # through the owner so the next child sees the complete new generation;
        # merely copying the old external_generation makes our own write stale.
        rebuilt = build_profile_secret_scope(Path(stamped_home), fail_closed_external=True)
        _SECRET_SCOPE.set(_BoundScope(rebuilt, stamped_home))
        return True
    values = dict(scope)
    if value is None:
        values.pop(name, None)
    else:
        values[name] = str(value)
    rebuilt = _immutable_scope(
        values,
        profile_home=None,
        source_status="legacy_mapping",
    )
    _SECRET_SCOPE.set(_BoundScope(rebuilt, None))
    return True


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
    # Explicit non-secret terminal coordinate. Keep this exact: a broad
    # TERMINAL_* grant would also authorize future credential/control names.
    "TERMINAL_CWD",
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
# Prefix-wide process authority is intentionally forbidden. Terminal, Kanban,
# and Telegram settings can vary by profile; globally safe coordinates must be
# named explicitly in _GLOBAL_ENV_EXACT so a new secret/control key cannot gain
# authority merely by choosing a trusted-looking prefix.
_GLOBAL_ENV_PREFIXES: tuple[str, ...] = ()


def _is_global_env(name: str) -> bool:
    """True for genuinely process-global (non-profile-secret) env vars."""
    candidate = _direct_env_name_key(name)
    if candidate in {_direct_env_name_key(item) for item in _GLOBAL_ENV_EXACT}:
        return True
    return any(
        candidate.startswith(_direct_env_name_key(prefix))
        for prefix in _GLOBAL_ENV_PREFIXES
    )


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
_ENV_FILE_CACHE: "OrderedDict[str, Tuple[tuple, Dict[str, str], bool]]" = OrderedDict()
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


def load_env_file_snapshot(env_path: Path, *, strict_decode: bool = True) -> EnvFileSnapshot:
    """One descriptor-bound tokenizer/cache, with explicit child-boundary failure states."""
    env_path = Path(env_path)
    key = str(env_path)
    try:
        with open(env_path, "rb") as handle:
            fingerprint = file_signature(os.fstat(handle.fileno()))
            with _ENV_FILE_CACHE_LOCK:
                cached = _ENV_FILE_CACHE.get(key)
                if cached is not None and cached[0] == fingerprint:
                    _ENV_FILE_CACHE.move_to_end(key)
                    if strict_decode and not cached[2]:
                        return EnvFileSnapshot(env_path, MappingProxyType({}), "failed", "decode")
                    secrets = dict(cached[1])
                    return EnvFileSnapshot(env_path, MappingProxyType(secrets), "ready" if secrets else "empty")
            raw = handle.read()
            settled = file_signature(os.fstat(handle.fileno())) == fingerprint
    except FileNotFoundError:
        invalidate_env_file_cache(env_path)
        return EnvFileSnapshot(env_path, MappingProxyType({}), "absent")
    except OSError as exc:
        invalidate_env_file_cache(env_path)
        return EnvFileSnapshot(env_path, MappingProxyType({}), "failed", type(exc).__name__)
    if strict_decode and not settled:
        invalidate_env_file_cache(env_path)
        return EnvFileSnapshot(env_path, MappingProxyType({}), "failed", "source_changed_during_read")
    try:
        text = raw.decode("utf-8-sig")
        valid_utf8 = True
    except UnicodeDecodeError:
        if strict_decode:
            return EnvFileSnapshot(env_path, MappingProxyType({}), "failed", "decode")
        text = _decode_env_bytes(raw)
        valid_utf8 = False
    secrets = _parse_env_text(text)
    if settled:
        with _ENV_FILE_CACHE_LOCK:
            _ENV_FILE_CACHE[key] = (fingerprint, dict(secrets), valid_utf8)
            _ENV_FILE_CACHE.move_to_end(key)
            while len(_ENV_FILE_CACHE) > _ENV_FILE_CACHE_MAX:
                _ENV_FILE_CACHE.popitem(last=False)
    return EnvFileSnapshot(env_path, MappingProxyType(secrets), "ready" if secrets else "empty")


def _parse_env_text(text: str) -> Dict[str, str]:
    """Tokenize already-read ``.env`` text. See :func:`load_env_file`."""
    secrets: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            secrets[key] = _parse_env_value(_strip_inline_comment(value))
    return secrets


def load_env_file(env_path: Path) -> Dict[str, str]:
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
    return dict(load_env_file_snapshot(env_path, strict_decode=False).data)


def _profile_external_secret_snapshot(home: Path, *, fail_closed: bool, hydrate: bool = True):
    try:
        from hermes_cli import env_loader

        snapshot = env_loader.get_external_secret_snapshot(home, wait=hydrate)
        initial_status = snapshot.status
        if hydrate and (initial_status in {"not_hydrated", "stale", "failed"} or (
            initial_status == "degraded" and snapshot.generation > 0
        )):
            # Generation-zero legacy projections are not retryable authority.
            # Private hydration revokes startup error-suppression leases, so
            # the next boundary construction is the retry point for a
            # transient source/config outage.  Do not retry
            # twice inside one construction: a not_hydrated/stale refresh that
            # records failure must still fail this attempt.
            env_loader.hydrate_profile_secret_sources(home)
            snapshot = env_loader.get_external_secret_snapshot(home)
        if fail_closed and snapshot.status in {
            "degraded",
            "failed",
            "not_hydrated",
            "stale",
        }:
            raise RuntimeError(
                f"external secret snapshot is {snapshot.status}; refusing boundary"
            )
        return snapshot
    except Exception:
        if fail_closed:
            raise
        from types import SimpleNamespace

        return SimpleNamespace(data={}, status="failed", generation=0)


def _profile_external_secret_values(home: Path, *, fail_closed: bool) -> Dict[str, str]:
    return dict(_profile_external_secret_snapshot(home, fail_closed=fail_closed).data)


def build_profile_secret_scope(
    hermes_home: Path,
    *,
    fail_closed_external: bool = False,
    hydrate_external: bool = True,
) -> ProfileSecretScope:
    """Build an immutable identity-bound profile secret scope."""
    home = Path(hermes_home)
    op_snapshot = load_env_file_snapshot(home / ".op.env")
    env_snapshot = load_env_file_snapshot(home / ".env")
    if fail_closed_external:
        failed = [s for s in (op_snapshot, env_snapshot) if s.status == "failed"]
        if failed:
            details = ", ".join(
                f"{snapshot.path.name}:{snapshot.error_kind or 'read'}"
                for snapshot in failed
            )
            raise RuntimeError(
                f"profile dotenv snapshot unavailable ({details}); refusing boundary"
            )
    secrets = dict(op_snapshot.data)
    secrets.update(env_snapshot.data)
    external_snapshot = _profile_external_secret_snapshot(
        home, fail_closed=fail_closed_external, hydrate=hydrate_external
    )
    secrets.update(
        (key, value)
        for key, value in external_snapshot.data.items()
        if not _is_global_env(key)
    )
    # ``GATEWAY_ALLOW_ALL_USERS`` is a launch-home deployment grant bridged from
    # config into the process environment. It belongs only to that profile's
    # immutable scope; routed profiles must never inherit it.
    from gateway.config_loader import bridged_allow_all_users

    bridged = bridged_allow_all_users()
    if bridged is not None and _is_process_home(home):
        secrets.setdefault("GATEWAY_ALLOW_ALL_USERS", bridged)
    return _immutable_scope(
        secrets,
        profile_home=home,
        source_status=(
            f"dotenv:{op_snapshot.status}/{env_snapshot.status};"
            f"external:{external_snapshot.status}"
        ),
        external_generation=int(external_snapshot.generation),
    )


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
    target_generation: str = ""
    target_status: str = ""

    @property
    def identity(self) -> str:
        """Stable target identity used by snapshot owners and diagnostics."""
        return str(self.target_home)

    def compiled_target_values(self) -> dict[str, str]:
        """Return deterministic target declarations, rejecting conflicts."""
        declarations: dict[str, list[tuple[_EnvCarrier, str]]] = {}
        for key, value in self.target_values.items():
            carrier = _env_carrier(key)
            declarations.setdefault(carrier.effective_name, []).append(
                (carrier, value)
            )

        compiled: dict[str, str] = {}
        for effective_name, candidates in declarations.items():
            if len(candidates) > 1:
                raise RuntimeError(
                    f"conflicting target environment declarations for {effective_name!r}"
                )
            carrier, value = candidates[0]
            compiled[carrier.physical_name] = value
        return compiled

    def sanitize(self, env: Mapping[str, str]) -> dict[str, str]:
        """Return *env* with source-profile-owned names isolated to the target."""
        result = dict(env)
        if self.source_home == self.target_home:
            return result

        result_carriers: dict[str, list[_EnvCarrier]] = {}
        for key in result:
            carrier = _env_carrier(key)
            result_carriers.setdefault(carrier.effective_name, []).append(carrier)

        target_declarations: dict[str, tuple[_EnvCarrier, str]] = {}
        for key, value in self.compiled_target_values().items():
            carrier = _env_carrier(key)
            target_declarations[carrier.effective_name] = (carrier, value)

        source_carriers: dict[str, list[_EnvCarrier]] = {}
        for name in self.source_owned_names:
            carrier = _env_carrier(name)
            source_carriers.setdefault(carrier.effective_name, []).append(carrier)

        for effective_name in source_carriers.keys() | target_declarations.keys():
            owned_carriers = source_carriers.get(effective_name, ())
            # Ambient collisions with target-only ownership must be removed too;
            # the child policy separately grants target-only materialization.
            target_declaration = target_declarations.get(effective_name) if owned_carriers else None

            # Direct process globals (PATH, HOME, terminal coordinates) remain
            # operational baseline when the source owned only a forwarded
            # container carrier.  For ordinary profile-owned names, or when the
            # source owned the direct spelling, every equivalent carrier is
            # removed before a target declaration is materialized.
            source_has_direct = any(not carrier.forwarded for carrier in owned_carriers)
            direct_global = _is_global_env(effective_name)
            removable: list[_EnvCarrier] = []
            for carrier in result_carriers.get(effective_name, []):
                if source_has_direct or not direct_global or carrier.forwarded:
                    removable.append(carrier)

            for carrier in removable:
                result.pop(carrier.physical_name, None)

            if target_declaration is not None:
                target_carrier, target_value = target_declaration
                for existing in list(result):
                    if _direct_env_name_key(existing) == _direct_env_name_key(
                        target_carrier.physical_name
                    ):
                        result.pop(existing, None)
                result[target_carrier.physical_name] = target_value
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
    op_snapshot = load_env_file_snapshot(home / ".op.env")
    env_snapshot = load_env_file_snapshot(home / ".env")
    if fail_closed_external:
        failed = [
            snapshot
            for snapshot in (op_snapshot, env_snapshot)
            if snapshot.status == "failed"
        ]
        if failed:
            details = ", ".join(
                f"{snapshot.path.name}:{snapshot.error_kind or 'read'}"
                for snapshot in failed
            )
            raise RuntimeError(
                f"profile dotenv ownership unavailable ({details}); refusing boundary"
            )
    observed_names = set(op_snapshot.data)
    observed_names.update(env_snapshot.data)
    observed_names.update(
        _profile_external_secret_values(
            home,
            fail_closed=fail_closed_external,
        )
    )
    return record_profile_owned_secret_names(home, observed_names)


def record_profile_owned_secret_names(
    hermes_home: str | os.PathLike, names,
) -> frozenset[str]:
    """Retain names at profile-source ingestion, before ambient values outlive it."""
    observed_names = {name for name in names if not _is_global_env(name)}

    # Ownership is monotonic for the life of the process. A profile can remove
    # a name from its current sources while its old value remains in
    # ``os.environ``; forgetting the prior ownership would reclassify that stale
    # value as ambient and allow it to cross a later profile boundary. Clearing
    # this history is legal only after a separately verified process-global
    # removal event, which Hermes does not currently expose.
    history_key = str(Path(hermes_home).resolve())
    with _PROFILE_OWNED_NAME_HISTORY_LOCK:
        history = _PROFILE_OWNED_NAME_HISTORY.setdefault(history_key, set())
        history.update(observed_names)
        return frozenset(history)


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
        from hermes_constants import get_routing_process_hermes_home

        source_home = get_routing_process_hermes_home()
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
    active_scope = current_secret_scope()
    active_scope_home = current_secret_scope_home()
    if active_scope is not None and active_scope_home is not None:
        if Path(active_scope_home).resolve() != target:
            raise RuntimeError(
                "active profile secret scope does not match target profile home"
            )
        current_scope = build_profile_secret_scope(
            target,
            fail_closed_external=True,
        )
        if not isinstance(active_scope, ProfileSecretScope):
            raise RuntimeError("active profile secret scope is not identity-bound")
        if active_scope.generation != current_scope.generation:
            raise RuntimeError(
                "active profile secret scope is stale for target profile generation"
            )
        target_scope = active_scope
    else:
        target_scope = build_profile_secret_scope(
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
        target_values=MappingProxyType(dict(target_scope)),
        target_generation=target_scope.generation,
        target_status=target_scope.source_status,
    )


def sanitize_profile_owned_env(
    env: Mapping[str, str],
    boundary: ProfileEnvBoundary | None = None,
) -> dict[str, str]:
    """Apply a captured profile boundary without changing single-profile mode."""
    if boundary is None:
        return dict(env)
    return boundary.sanitize(env)


def _is_process_home(hermes_home: Path) -> bool:
    """Is *hermes_home* the profile this process serves as its own? Same launch-home identity as
    ``serves_routed_profile()``: a host that mirrors a served profile into ``HERMES_HOME`` would
    otherwise seed the launch profile's bridged allow-all grant into that profile's scope."""
    from hermes_constants import get_routing_process_hermes_home
    try:
        return Path(hermes_home).resolve() == get_routing_process_hermes_home().resolve()
    except OSError:
        return False
