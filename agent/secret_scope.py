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
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple


# Process-global (describes the deployment mode, not a per-task value): set once
# at gateway startup when gateway.multiplex_profiles is true.
_MULTIPLEX_ACTIVE: bool = False


def set_multiplex_active(active: bool) -> None:
    """Mark whether the process is a profile multiplexer (get_secret fails closed)."""
    global _MULTIPLEX_ACTIVE
    _MULTIPLEX_ACTIVE = bool(active)


def is_multiplex_active() -> bool:
    return _MULTIPLEX_ACTIVE


def serves_routed_profile() -> bool:
    """True when the current task runs for a profile other than the process's own: always under
    multiplexing, else when a HERMES_HOME override names another home (dashboard/desktop backend,
    per-profile cron ticker). The MCP registry scope and the check_fn cache key both follow this
    predicate so a served profile's view never aliases the launch profile's (#111151)."""
    if is_multiplex_active():
        return True
    from hermes_constants import get_hermes_home_override, get_process_hermes_home, hermes_home_key
    override = get_hermes_home_override()
    return override is not None and hermes_home_key(override) != hermes_home_key(get_process_hermes_home())


_SECRET_SCOPE: ContextVar[Optional[Mapping[str, str]]] = ContextVar("_SECRET_SCOPE", default=None)


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


# Per-path memo of parsed ``.env`` files. ``load_env_file()`` is THE tokenizer:
# the profile scope, ``hermes_cli.config.load_env``, the dashboard scrub, skill
# secret capture, managed .env and the setup prompts all parse through it, and
# ``build_profile_secret_scope()`` puts it on the gateway's hottest paths -
# every turn, every cron job, every MCP/browser lifecycle adoption and the 60s
# housekeeping delivery drain each re-read and re-parsed the whole file.
#
# FRESHNESS CONTRACT. Every call still OPENS the file, and the cache key is the
# ``fstat`` of that descriptor, not a ``stat`` of the path. That matters three
# ways, each of which a path-stat design gets wrong:
#   * the open is what drives close-to-open revalidation on NFS, so a rotation
#     on another client is seen here exactly as often as it was before;
#   * an unreadable or vanished file fails the open, so it can never be cached
#     (a transient EACCES used to be memoised as "this profile has no secrets");
#   * the descriptor pins one inode, so a symlink repointed mid-read cannot get
#     its contents filed under the original file's identity.
# The key is (mtime_ns, size, inode, device), re-checked after the read so a
# rewrite through the same inode is not stored under the pre-read fingerprint.
# What remains uncovered, and is accepted: a writer that changes contents while
# keeping length, inode AND nanosecond mtime identical.
# ``invalidate_env_file_cache()`` is the explicit knob for that, and
# ``hermes_cli.config.invalidate_env_cache()`` calls it, so the writers that
# already invalidate (save_env_value / remove_env_value / sanitize_env_file)
# clear both memos.
_ENV_FILE_CACHE: "OrderedDict[str, Tuple[tuple, Dict[str, str]]]" = OrderedDict()
_ENV_FILE_CACHE_LOCK = threading.Lock()
# Bumped by every invalidation. A reader captures it before opening and stores
# only if it has not moved, so an invalidation cannot be undone by a slow reader
# that started earlier and is still holding the previous contents.
_ENV_FILE_CACHE_GENERATION = 0
# One entry per profile home in practice; the cap only bounds a pathological
# caller that walks arbitrary paths.
_ENV_FILE_CACHE_MAX = 64


def _fd_fingerprint(fileno: int) -> tuple:
    """Identity of the bytes behind an OPEN descriptor."""
    st = os.fstat(fileno)
    return (st.st_mtime_ns, st.st_size, st.st_ino, st.st_dev)


def _fd_fingerprint_or_none(fileno: int) -> Optional[tuple]:
    """Fingerprint, or None when the metadata is unavailable.

    Unreadable metadata means "cannot cache", never "cannot read": CPython
    tolerates fstat failures on some filesystems (VirtualBox shared folders,
    for one), and a file whose bytes we can read must still be parsed and
    returned rather than downgraded to an empty credential map.
    """
    try:
        return _fd_fingerprint(fileno)
    except OSError:
        return None


def invalidate_env_file_cache(env_path: Optional[Path] = None) -> None:
    """Drop one path from the ``load_env_file()`` memo, or all of them."""
    global _ENV_FILE_CACHE_GENERATION
    with _ENV_FILE_CACHE_LOCK:
        _ENV_FILE_CACHE_GENERATION += 1
        if env_path is None:
            _ENV_FILE_CACHE.clear()
        else:
            _ENV_FILE_CACHE.pop(str(env_path), None)


def _decode_env_bytes(raw: bytes) -> str:
    """Decode ``.env`` bytes: BOM stripped, invalid UTF-8 falls back to latin-1
    exactly as ``env_loader._load_dotenv_with_fallback`` installs it into
    ``os.environ``. Never raises, so a decode can't be mistaken for "no file"."""
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


def _parse_env_file(env_path: Path) -> Dict[str, str]:
    """Uncached read plus parse, preserved as the fail-soft reference path."""
    try:
        raw = env_path.read_bytes()
    except OSError:
        return {}
    return _parse_env_text(_decode_env_bytes(raw))


def load_env_file(env_path: Path) -> Dict[str, str]:
    """THE ``.env`` tokenizer: every reader (profile scope, ``hermes_cli.config.load_env``, the dashboard
    scrub, skill secret capture, managed .env, setup prompts) parses through here so no two boundaries
    disagree on which keys/values a file defines. Dict only — never touches ``os.environ``. ``export``
    prefix, ``#`` comments, quote escapes reversed; a BOM is stripped so it doesn't prefix the first key.
    Invalid UTF-8 decodes as latin-1, exactly like ``env_loader._load_dotenv_with_fallback`` installs it
    into ``os.environ``. Absent/unreadable → ``{}``.

    Memoised per path on the open descriptor's stat identity, so an unchanged
    ``.env`` costs an open and an fstat instead of a full read and parse. A file
    that cannot be opened or read yields ``{}`` and is never cached, so a
    permission blip or a deletion cannot be remembered as "no secrets". Always
    returns a fresh dict: callers mutate what they get back
    (``build_profile_secret_scope`` layers external secrets over it).
    """
    key = str(env_path)
    generation = _ENV_FILE_CACHE_GENERATION
    try:
        handle = open(env_path, "rb")
    except OSError:
        # Gone or unreadable: drop any entry so a stale map cannot outlive the
        # file, and leave the caller on the old fail-soft empty result.
        invalidate_env_file_cache(env_path)
        return {}

    try:
        with handle:
            fingerprint = _fd_fingerprint_or_none(handle.fileno())
            if fingerprint is not None:
                with _ENV_FILE_CACHE_LOCK:
                    cached = _ENV_FILE_CACHE.get(key)
                    if cached is not None and cached[0] == fingerprint:
                        _ENV_FILE_CACHE.move_to_end(key)
                    else:
                        cached = None
                if cached is not None:
                    # Copied outside the lock: a cached entry is never mutated
                    # in place, so readers hold only the reference under it.
                    return dict(cached[1])
            raw = handle.read()
            # Same descriptor, so this compares the bytes actually parsed. No
            # fingerprint at all means this read is served but not stored.
            settled = (
                fingerprint is not None
                and _fd_fingerprint_or_none(handle.fileno()) == fingerprint
            )
    except OSError:
        invalidate_env_file_cache(env_path)
        return {}

    secrets = _parse_env_text(_decode_env_bytes(raw))
    if settled:
        with _ENV_FILE_CACHE_LOCK:
            if _ENV_FILE_CACHE_GENERATION == generation:
                _ENV_FILE_CACHE[key] = (fingerprint, dict(secrets))
                _ENV_FILE_CACHE.move_to_end(key)
                while len(_ENV_FILE_CACHE) > _ENV_FILE_CACHE_MAX:
                    _ENV_FILE_CACHE.popitem(last=False)
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
    # The DEFAULT profile's config.yaml allow_all_users grant lives only in os.environ (bridged by
    # gateway.config_loader); scoped gate readers under multiplex never fall to os.environ, so seed it
    # into that profile's own mapping. A secondary never inherits it (#80099 class).
    from gateway.config_loader import bridged_allow_all_users
    bridged = bridged_allow_all_users()
    if bridged is not None and _is_process_home(hermes_home):
        secrets.setdefault("GATEWAY_ALLOW_ALL_USERS", bridged)
    return secrets


def _is_process_home(hermes_home: Path) -> bool:
    from hermes_constants import get_process_hermes_home
    try:
        return Path(hermes_home).resolve() == get_process_hermes_home().resolve()
    except OSError:
        return False
