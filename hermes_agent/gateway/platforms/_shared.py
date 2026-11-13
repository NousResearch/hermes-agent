"""Cross-adapter helpers shared by hermes_agent/gateway/platforms/* and plugins/platforms/*.

Kept dependency-light (stdlib + ``hermes_agent.agent.secret_scope``) so every adapter can
import it at module top level without cycles.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any, Callable, Iterable, Optional

# Profile-scoped secret reader for multiplexing support (PR #50094)
from hermes_agent.agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from hermes_agent.agent.secret_scope import get_secret as _scoped_get_secret

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
    from hermes_agent.agent.secret_scope import current_secret_scope
    return current_secret_scope()


_UNSCOPED_PROFILE_SECRETS: Optional[dict] = None


def _unscoped_profile_secrets() -> dict:
    """Process-cached profile secret mapping (external resolvers are slow); failures degrade to {}."""
    global _UNSCOPED_PROFILE_SECRETS
    if _UNSCOPED_PROFILE_SECRETS is None:
        try:
            from hermes_agent.agent.secret_scope import build_profile_secret_scope
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
        from hermes_agent.agent.secret_scope import current_secret_scope, is_multiplex_active

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
        from hermes_agent.agent.secret_scope import current_secret_scope, is_multiplex_active
        return bool(is_multiplex_active() and current_secret_scope() is not None)
    except Exception:
        return False


# --------------------------------------------------------------------------- YAML → env config bridge
# (apply_yaml_config_fn, #25443)
# ---------------------------------------------------------------------------

def yaml_env_setter() -> Callable[[str, Any], None]:
    """``set_env(name, value)`` for ``apply_yaml_config_fn`` hooks: writes ``os.environ[name]`` only
    when the var is unset (explicit env wins over YAML) and NEVER while a multiplexed secondary
    profile's scope is active — the gateway loads every secondary's config inside
    ``_profile_runtime_scope``, so a write there would pin that profile's policy process-wide and the
    default profile's adapters would read it as their own (first-writer-wins poisoning, #80099).
    Hooks seed the same values into the returned ``extra`` so each profile's adapter reads its own.
    Lists are comma-joined; ``None`` is skipped.
    """
    skip = profile_scoped()

    def set_env(name: str, value: Any) -> None:
        if value is None or skip or os.getenv(name):
            return
        os.environ[name] = ",".join(str(v) for v in value) if isinstance(value, list) else str(value)

    return set_env


def send_error(message: Any, *, nothing_sent: bool = False) -> dict:
    """Standalone-sender failure envelope with vendor exception text redacted (the same helper
    ``send_message`` uses), so a token or signed URL in an httpx/aiohttp error never reaches the
    model transcript.

    ``nothing_sent=True`` adds ``"retryable": True``: the sender PROVES nothing reached the chat
    (no credential, or the connection failed before the request left). Only that marker lets cron
    queue the payload for a post-reconnect replay; any other failure may already be on screen, so
    a replay would post it twice."""
    from hermes_agent.tools.send_message_senders import _error
    return {**_error(str(message)), **({"retryable": True} if nothing_sent else {})}


# Transport errors raised before a request is written: the connection (or a pool slot) was never
# obtained. Matched by class name so this module stays free of httpx/aiohttp imports. A read or
# write timeout, a reset, or any server answer is NOT here: the request may have been delivered.
_NEVER_LEFT_ERRORS = frozenset({
    "ConnectError", "ConnectTimeout", "PoolTimeout",  # httpx
    "ClientConnectorError", "ClientConnectorDNSError", "ClientProxyConnectionError",  # aiohttp
    "ConnectionTimeoutError", "ConnectionRefusedError"})


def request_never_left(exc: BaseException) -> bool:
    """True when ``exc`` (or a cause it wraps) proves the request was never sent."""
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if any(cls.__name__ in _NEVER_LEFT_ERRORS for cls in type(cur).__mro__):
            return True
        stack.extend(x for x in (cur.__cause__, cur.__context__) if x is not None)
    return False


# What a standalone send put in the chat, as ``delivery_state`` reads it off a result dict. Only
# NONE licenses a whole-payload replay; PARTIAL carries the ids that landed in
# ``delivered_message_ids``; UNCERTAIN is a request that may have landed without a receipt.
DELIVERED, PARTIAL, UNCERTAIN, NONE = "delivered", "partial", "uncertain", "none"


class DeliveryOutcome:
    """The delivery record of ONE logical standalone send, across every attempt, endpoint and
    chunk. A sender marks each receipt (``landed``) and each attempt that may have reached the chat
    without one (``ambiguous``, e.g. a 5xx it then retried); ``failure`` reads the whole history, so
    a final connect error can never erase an earlier possibly delivered attempt."""

    def __init__(self) -> None:
        self.message_ids: list[str] = []
        self.has_landed = False
        self.is_ambiguous = False

    def landed(self, message_id: Any = None) -> None:
        self.has_landed = True
        if message_id is not None:
            self.message_ids.append(str(message_id))

    def ambiguous(self) -> None:
        self.is_ambiguous = True

    def split_fields(self) -> dict:
        """Success-envelope fields of a send split into several messages: every landed id."""
        return {"delivered_message_ids": list(self.message_ids)} if len(self.message_ids) > 1 else {}

    def failure(self, error: dict, *, never_sent: bool = False) -> dict:
        """``error`` stamped with this send's state. ``never_sent`` = the FINAL failure proves its own
        request never left or was refused; it means NONE only when nothing earlier landed or may have.
        An ``error`` already stamped by an inner sender keeps what IT saw landing or maybe landing."""
        inner = error.get("delivery")
        ids = list(dict.fromkeys([*self.message_ids, *delivered_message_ids(error)]))
        if self.has_landed or ids or inner == PARTIAL:
            state = PARTIAL
        else:
            state = UNCERTAIN if self.is_ambiguous or inner == UNCERTAIN or not (never_sent or inner == NONE) else NONE
        out = {k: v for k, v in error.items() if k not in ("retryable", "delivered_message_ids")}
        out["delivery"] = state
        if ids:
            out["delivered_message_ids"] = ids
        return {**out, "retryable": True} if state == NONE else out


def delivery_state(result: Any) -> str:
    """DELIVERED / PARTIAL / UNCERTAIN / NONE for a standalone sender's result dict. Senders that
    predate ``DeliveryOutcome`` are read by their envelope: success, an explicit ``retryable``
    (``send_error(nothing_sent=True)``: nothing sent), else a failure that may have landed."""
    if not isinstance(result, dict) or not result.get("error"):
        return DELIVERED
    if result.get("delivery") in (PARTIAL, UNCERTAIN, NONE):
        return result["delivery"]
    return NONE if result.get("retryable") is True else UNCERTAIN


def delivered_message_ids(result: Any) -> list[str]:
    """Ids a standalone send put in the chat (every landed chunk), ``[]`` when none are known."""
    ids = result.get("delivered_message_ids") if isinstance(result, dict) else None
    return [str(i) for i in ids or () if i is not None]


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
    non-blank. Resolves ``hermes_agent.hermes_cli.gateway.get_env_value`` at call time (scope-aware, .env-backed) so
    setup-status tests that patch it see the same value."""

    def is_connected(config: Any) -> bool:
        import hermes_agent.hermes_cli.gateway as gateway_mod
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
