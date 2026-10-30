"""Cross-adapter helpers shared by gateway/platforms/* and plugins/platforms/*.

Kept dependency-light (stdlib + ``agent.secret_scope``) so every adapter can
import it at module top level without cycles.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

# Profile-scoped secret reader for multiplexing support (PR #50094)
from agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from agent.secret_scope import get_secret as _scoped_get_secret

logger = logging.getLogger(__name__)


def get_scoped_secret(name: str, default: Any = None, *, external_fallback: bool = False) -> Any:
    """Scope-aware credential read with the default-profile startup fallback.

    An installed profile secret scope is authoritative: a scoped miss returns
    ``default`` (never borrow another profile's value from ``os.environ``).
    The DEFAULT profile constructs and sends *unscoped* under multiplexing,
    where a bare ``get_secret`` raises ``UnscopedSecretError``; there
    ``os.environ`` is that profile's own value, so fall back to it.

    ``external_fallback`` adds one rung for startup gates that run before any
    scope exists: an unscoped miss consults a one-shot build of the profile's
    own secret scope, so externally managed credentials (Bitwarden etc., only
    ``BWS_ACCESS_TOKEN`` in ``.env``) are visible to ``check_requirements``.
    """
    try:
        val = _scoped_get_secret(name, None)
    except _UnscopedSecretError:
        val = os.getenv(name)
    if val is None and external_fallback and _current_scope() is None:
        val = _unscoped_profile_secrets().get(name)
    return val if val is not None else default


def _current_scope():
    from agent.secret_scope import current_secret_scope
    return current_secret_scope()


_UNSCOPED_PROFILE_SECRETS: Optional[dict] = None


def _unscoped_profile_secrets() -> dict:
    """Process-cached profile secret mapping (external resolvers are slow); failures degrade to {}."""
    global _UNSCOPED_PROFILE_SECRETS
    if _UNSCOPED_PROFILE_SECRETS is None:
        try:
            from agent.secret_scope import build_profile_secret_scope
            from hermes_constants import get_hermes_home
            _UNSCOPED_PROFILE_SECRETS = dict(build_profile_secret_scope(get_hermes_home()))
        except Exception:
            logger.warning(
                "Could not build the profile secret scope; externally managed credentials will not be "
                "visible to the startup gate (#95216)", exc_info=True)
            _UNSCOPED_PROFILE_SECRETS = {}
    return _UNSCOPED_PROFILE_SECRETS


def platform_gate_env(name: str, default: str = "") -> str:
    """Allow/deny gate env read with per-profile isolation, always stripped.

    With a profile secret scope installed AND multiplexing active, a scoped miss returns ``default``
    instead of falling through to ``os.environ``, which may hold ANOTHER profile's first-writer
    bridged value (the YAML→env bridges are first-writer-wins; allowlist leak, #72348).
    Single-profile deployments behave exactly like ``os.getenv``.
    """
    if not name:
        return default
    with contextlib.suppress(Exception):
        from agent.secret_scope import current_secret_scope, is_multiplex_active

        scope = current_secret_scope()
        if scope is not None and is_multiplex_active():
            val = scope.get(name)
            return default if val is None else str(val).strip()
    return (os.getenv(name) or default).strip()


def decode_json_list_literal(raw):
    """Decode a JSON-encoded allowlist written by ``hermes config set``.

    String-typed defaults keep list literals verbatim on write (``allowed_chats`` is
    declared as ``""``), so the config can hold ``'["-100","-200"]'`` as a string.
    Malformed JSON passes through unchanged and keeps the legacy comma-split path.
    """
    if isinstance(raw, str) and raw.lstrip()[:1] == "[":
        try:
            loaded = json.loads(raw)
        except ValueError:
            return raw
        if isinstance(loaded, list):
            return loaded
    return raw


def extra_or_secret(extra: Optional[dict], key: str, env: str, default: Any = "",
                    *, blank_is_unset: bool = True) -> Any:
    """The ONE per-profile setting reader: explicit env ``env`` → the profile's YAML
    ``config.extra[key]`` → ``default``.

    The env rung is the owning profile's, read through ``get_scoped_secret``: a secondary
    multiplex profile sees its own ``.env`` and a miss falls to ITS YAML, never to the launch
    process's ``os.environ`` (which holds the default profile's bridged values); single-profile
    and default-profile installs read ``os.environ`` there, keeping the documented env-over-YAML
    contract (an explicit ``DISCORD_ALLOW_MENTION_EVERYONE=false`` beats ``everyone: true``,
    #108440; ``TELEGRAM_REACTIONS=true`` beats the stock ``reactions: false``, #109032). A blank
    env value is unset. An explicit ``False``/``0`` in YAML is a real value (``require_mention:
    false`` must not fall to ``default``). A blank YAML string is unset by default; readers whose
    YAML key means "clear it" (``allowed_channels: "" `` = no whitelist) pass
    ``blank_is_unset=False`` so only a missing/``None`` key falls through.
    """
    if env:
        env_value = get_scoped_secret(env, None)
        if env_value is not None and str(env_value).strip():
            return env_value
    value = (extra or {}).get(key)
    if value is None or (blank_is_unset and isinstance(value, str) and not value.strip()):
        return default
    return value


def profile_scoped() -> bool:
    """True when running inside a multiplexed secondary profile's scope.

    Secondary-profile adapters are constructed/connected inside
    ``_profile_runtime_scope`` (secret scope installed + multiplex active).
    The DEFAULT profile under multiplexing runs unscoped and keeps the legacy
    ``os.environ`` precedence, so YAML->env bridges must skip only when True.
    """
    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active
        return bool(is_multiplex_active() and current_secret_scope() is not None)
    except Exception:
        return False


# --------------------------------------------------------------------------- YAML → env config bridge
# (apply_yaml_config_fn, #25443)
# ---------------------------------------------------------------------------

# Marker prefix for values this bridge itself wrote into ``os.environ``. A CLI-started gateway
# hands its environment to the restart child, so a bridged value outlives the YAML read that
# produced it; without the marker the bridge would skip on the next start ("env is set") and the
# edited YAML would never take effect (#109032's mirror image).
BRIDGED_ENV_MARKER_PREFIX = "HERMES_BRIDGED_"


def bridged_env_marker(name: str) -> str:
    return f"{BRIDGED_ENV_MARKER_PREFIX}{name}"


def _supplied_by_operator(name: str) -> bool:
    """True when ``name`` came from the launch profile's ``.env``: a deliberate value that wins (#109032).

    A ``.env`` entry can hold exactly the value a previous bridge wrote, so value comparison alone
    cannot tell it from the bridge's own leftover — this is the one case it must not clobber.
    """
    try:
        from hermes_cli.env_loader import has_nonblank_dotenv_value
    except ImportError:  # bridging must not fail because the loader is unavailable
        return False
    return has_nonblank_dotenv_value(name)


@dataclass
class _YamlEnvLoad:
    seen: set[str] = field(default_factory=set)
    failed: bool = False


# Loads are synchronous; copied contexts must not inherit another thread's partial seen-set.
_YAML_ENV_LOAD = threading.local()
_YAML_ENV_LOCK = threading.RLock()


def cancel_yaml_env_refresh() -> None:
    """A swallowed discovery/hook failure is not evidence that settings were deleted."""
    load = getattr(_YAML_ENV_LOAD, "current", None)
    if load is not None:
        load.failed = True


@contextlib.contextmanager
def yaml_env_refresh():
    """Retract only unchanged bridge-owned leftovers after one complete, successful YAML load."""
    if profile_scoped():
        yield
        return
    with _YAML_ENV_LOCK:
        if getattr(_YAML_ENV_LOAD, "current", None) is not None:
            yield  # A nested load participates in the outer load, never clears its partial writes.
            return
        before = {key[len(BRIDGED_ENV_MARKER_PREFIX):]: value for key, value in os.environ.items()
                  if key.startswith(BRIDGED_ENV_MARKER_PREFIX) and key != BRIDGED_ENV_MARKER_PREFIX}
        load = _YamlEnvLoad()
        _YAML_ENV_LOAD.current = load
        try:
            yield
            if not load.failed:
                for name, value in before.items():
                    marker = bridged_env_marker(name)
                    if (name not in load.seen and os.getenv(name) == value
                            and os.getenv(marker) == value and not _supplied_by_operator(name)):
                        os.environ.pop(name, None)
                        os.environ.pop(marker, None)
        finally:
            del _YAML_ENV_LOAD.current


def yaml_env_setter(*, fallback: bool = False) -> Callable[[str, Any], None]:
    """``set_env(name, value)`` for ``apply_yaml_config_fn`` hooks: writes ``os.environ[name]``
    unless the operator set that var themselves, and NEVER while a multiplexed secondary
    profile's scope is active — the gateway loads every secondary's config inside
    ``_profile_runtime_scope``, so a write there would pin that profile's policy process-wide and the
    default profile's adapters would read it as their own (first-writer-wins poisoning, #80099).
    Hooks seed the same values into the returned ``extra`` so each profile's adapter reads its own.
    Lists are comma-joined; ``None`` is skipped.

    Precedence is per *value*, not just per name: a var whose current value is the one this bridge
    wrote last time is the bridge's own leftover (a restarted gateway inherits it) and is refreshed
    from YAML, while any other preset value is an operator's and wins.
    """
    skip = profile_scoped()

    def set_env(name: str, value: Any) -> None:
        if value is None or skip:
            return
        with _YAML_ENV_LOCK:
            load = getattr(_YAML_ENV_LOAD, "current", None)
            if load is not None:
                if fallback and name in load.seen:
                    return  # The plugin already resolved authored/managed precedence this load.
                load.seen.add(name)
            bridged = ",".join(str(v) for v in value) if isinstance(value, list) else str(value)
            marker = bridged_env_marker(name)
            current = os.getenv(name)
            if current is not None and current.strip():
                if current != os.getenv(marker) or _supplied_by_operator(name):
                    return  # the operator's value, not this bridge's leftover
            os.environ[name] = bridged
            os.environ[marker] = bridged

    return set_env


def send_error(message: Any) -> dict:
    """Standalone-sender failure envelope with vendor exception text redacted (the same helper
    ``send_message`` uses), so a token or signed URL in an httpx/aiohttp error never reaches the
    model transcript."""
    from tools.send_message_senders import _error
    return _error(str(message))


# kind -> (applies-when predicate over (cfg, key), env encoder). "lower"/"json" bridge whenever the key
# is present (YAML ``none`` still writes "none"); "str" skips null/blank, "csv" skips null and leaves
# lists to the setter's comma join.
_YAML_KINDS: dict[str, tuple[Callable[[dict, str], bool], Callable[[Any], Any]]] = {
    "lower": (lambda cfg, key: key in cfg, lambda v: str(v).lower()),
    "str": (lambda cfg, key: cfg.get(key) not in (None, ""), str),
    "csv": (lambda cfg, key: cfg.get(key) is not None, lambda v: v),
    "json": (lambda cfg, key: key in cfg, json.dumps),
}


def apply_yaml_bridge(cfg: dict, spec: Iterable[tuple[str, str, str]]) -> dict | None:
    """Table-driven ``apply_yaml_config_fn`` body: for each ``(yaml_key, ENV_VAR, kind)`` row seed the
    original YAML value into the returned ``extra`` and bridge it to env through ``yaml_env_setter``
    (env wins; skipped under a secondary profile's scope). ``None`` when nothing matched.
    """
    set_env = yaml_env_setter()
    seeded: dict = {}
    for key, env, kind in spec:
        applies, encode = _YAML_KINDS[kind]
        if applies(cfg, key):
            seeded[key] = cfg[key]
            set_env(env, encode(cfg[key]))
    return seeded or None


# --------------------------------------------------------------------------- env → extra seeding
# (env_enablement_fn)
# ---------------------------------------------------------------------------

def seed_extra_from_env(spec: Iterable[tuple[str, str, Callable[[str], Any] | None]], *,
                        home_env: str | None = None, home_default: str = "") -> dict:
    """Table-driven ``env_enablement_fn`` body: for each ``(ENV_VAR, extra_key, conv)`` row read the
    profile-scoped value, skip it when blank or when ``conv`` raises ``ValueError``, else seed
    ``conv(value)`` (``None`` conv keeps the stripped string).

    ``home_env`` seeds ``home_channel`` as ``{"chat_id", "name"}`` from ``<home_env>`` (else
    ``home_default``) with ``<home_env>_NAME`` naming it — the core hook lifts that dict into a
    ``HomeChannel``. The name falls back to the literal ``"Home"``, the same rule the built-in
    platforms use.
    """
    seed: dict = {}
    for env, key, conv in spec:
        raw = str(get_scoped_secret(env, "") or "").strip()
        if not raw:
            continue
        with contextlib.suppress(ValueError):
            seed[key] = conv(raw) if conv else raw
    if home_env:
        home = str(get_scoped_secret(home_env, "") or "").strip() or home_default
        if home:
            seed["home_channel"] = {"chat_id": home, "name": get_scoped_secret(f"{home_env}_NAME", "Home")}
    return seed


def env_is_connected(*names: str) -> Callable[[Any], bool]:
    """``is_connected`` for platforms whose only configuration is env: True when every ``names`` var is
    non-blank. Resolves ``hermes_cli.gateway.get_env_value`` at call time (scope-aware, .env-backed) so
    setup-status tests that patch it see the same value."""

    def is_connected(config: Any) -> bool:
        import hermes_cli.gateway as gateway_mod
        return all((gateway_mod.get_env_value(name) or "").strip() for name in names)

    return is_connected


def config_connected_extra(*keys: str) -> Callable[[Any], bool]:
    """``PlatformEntry.config_connected_fn``: connected when the merged PlatformConfig carries a
    token/api_key or every ``extra`` key (no env fallback; that is ``is_connected``'s job)."""

    def config_connected(config: Any) -> bool:
        return bool(config.token or config.api_key) or all(config.extra.get(k) for k in keys)

    return config_connected


def coerce_port(value: Any, default: int) -> int:
    """``int(value)`` or ``default`` when unparseable (incl. non-finite YAML like ``.inf``)."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default
