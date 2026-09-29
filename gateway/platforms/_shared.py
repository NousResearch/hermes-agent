"""Cross-adapter helpers shared by gateway/platforms/* and plugins/platforms/*.

Kept dependency-light (stdlib + ``agent.secret_scope``) so every adapter can
import it at module top level without cycles.
"""

from __future__ import annotations

import contextlib
import contextvars
import hmac
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

# Profile-scoped secret reader for multiplexing support (PR #50094)
from agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from agent.secret_scope import get_secret as _scoped_get_secret
from gateway.bridged_env import BRIDGED_ENV_MARKER_PREFIX, bridged_env_digest
from gateway.bridged_env import YAML_ENV_LOCK as _YAML_ENV_LOCK

logger = logging.getLogger(__name__)

# How far a signed webhook/callback timestamp may drift from now (either way) before the request is
# treated as a replay. Providers retry within seconds; per-message dedup caches expire long before
# a captured request stops verifying, so the signature alone never bounds replay.
WEBHOOK_REPLAY_WINDOW_SECONDS = 300


def secrets_match(provided: Any, expected: Any) -> bool:
    """Timing-safe equality of a remote-supplied secret or signature against the configured one.

    Fails closed on a missing/non-str value or an unset ``expected``. Compared as UTF-8 bytes with
    ``surrogatepass``: ``compare_digest`` raises TypeError on non-ASCII str, and aiohttp
    surrogate-escapes undecodable header bytes, which a plain ``.encode()`` turns into a 500.
    """
    if not isinstance(provided, str) or not isinstance(expected, str) or not expected:
        return False
    return hmac.compare_digest(provided.encode("utf-8", "surrogatepass"), expected.encode("utf-8", "surrogatepass"))


def timestamp_fresh(raw: Any, window: float = WEBHOOK_REPLAY_WINDOW_SECONDS) -> bool:
    """True when epoch-seconds *raw* lies within *window* of now; unparseable → False."""
    try:
        return abs(time.time() - int(raw)) <= window
    except (TypeError, ValueError, OverflowError):
        return False


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
    """True when running inside a profile's bound scope under multiplexing.

    Secondary-profile adapters are constructed/connected inside
    ``_profile_runtime_scope`` (secret scope installed + multiplex active). The launch profile's
    own bodies are bound too once the process multiplexes (``launch_profile_scope_if_multiplexed``);
    its YAML->env bridge then runs only for the live allowlist reload (``bridge_env_skipped``).
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
# edited YAML would never take effect (#109032's mirror image). The marker holds a digest, never
# the value: bridged names include credential-bearing ones (``TELEGRAM_PROXY``) and allowlists the
# subprocess scrub strips, and ownership only needs equality. The scrub also drops the markers
# (``local_env_policy._is_hermes_internal_secret``); only a same-profile restart child gets them back. The prefix and digest live in the
# stdlib-only ``gateway.bridged_env`` so the spawn path never imports ``gateway.platforms``.


def bridged_env_marker(name: str) -> str:
    return f"{BRIDGED_ENV_MARKER_PREFIX}{name}"


def _is_bridge_leftover(name: str, marker_value: Optional[str]) -> bool:
    return marker_value is not None and bridged_env_digest(os.getenv(name)) == marker_value


# Set while ``refresh_live_gate_extra`` reloads the LAUNCH profile's own config: its bridged lists live
# in ``os.environ``, so that load must refresh them even under the launch profile's bound scope.
_LAUNCH_GATE_RELOAD: contextvars.ContextVar[bool] = contextvars.ContextVar("_LAUNCH_GATE_RELOAD", default=False)


def _foreign_home_load() -> bool:
    from agent.secret_scope import current_secret_scope_home
    from hermes_constants import get_hermes_home_override, get_routing_process_hermes_home, hermes_home_key
    own = hermes_home_key(get_routing_process_hermes_home())
    return any(home and hermes_home_key(home) != own
               for home in (get_hermes_home_override(), current_secret_scope_home()))


def bridge_env_skipped() -> bool:
    """True when this load must not touch the launch profile's ``os.environ`` bridge values.

    A multiplexed secondary's scope (``profile_scoped``), and also any load for a home other than the
    process's own: a HERMES_HOME override or a secret scope stamped with another home. Without the
    second half a cross-home load with no bound secret scope would refresh or retract the launch
    profile's allowlist with another profile's YAML. The default profile's own unscoped loads under
    multiplex still bridge, as on main, and so does the live allowlist reload of the launch profile's
    own config under its bound scope (``_LAUNCH_GATE_RELOAD``).
    """
    if _LAUNCH_GATE_RELOAD.get():
        return _foreign_home_load()
    return profile_scoped() or _foreign_home_load()


def _supplied_by_operator(name: str) -> bool:
    """True when ``name`` came from the launch profile's ``.env`` or an external secret source (vault,
    1Password...): a deliberate value that wins (#109032).

    Either can hold exactly the value a previous bridge wrote (a proxy URL moved from YAML into the
    vault), so value comparison alone cannot tell it from the bridge's own leftover — the one case it
    must not clobber or retract.
    """
    try:
        from hermes_cli.env_loader import get_secret_source_values, has_nonblank_dotenv_value
    except ImportError:  # bridging must not fail because the loader is unavailable
        return False
    from hermes_constants import get_routing_process_hermes_home
    # The launch home's CURRENT source snapshot, never the lifetime all-homes name history
    # (``source_supplied_names``): a mapping removed since, or another profile's vault, owns nothing here.
    return has_nonblank_dotenv_value(name) or name in get_secret_source_values(get_routing_process_hermes_home())


@dataclass
class _YamlEnvLoad:
    seen: set[str] = field(default_factory=set)
    failed: bool = False
    # name -> (value, marker) before this load's first write: a load that raises restores them, so the
    # adapter (last good lists) and the runner (process env) never check different policies.
    prior: dict[str, tuple[Optional[str], Optional[str]]] = field(default_factory=dict)
    undo: list[Callable[[], None]] = field(default_factory=list)  # ownership state kept outside the env
    before: dict[str, str] = field(default_factory=dict)  # bridge markers present when the load began
    in_layer: bool = False
    committed: bool = False

    def retract_leftovers(self) -> None:
        """Drop unchanged bridge-owned values this load's YAML no longer sets. Their priors are
        recorded first, so a later failure in the same load restores them with its writes."""
        self.committed = True
        if self.failed:
            return
        for name, digest in self.before.items():
            marker = bridged_env_marker(name)
            if (name not in self.seen and os.getenv(marker) == digest
                    and _is_bridge_leftover(name, digest) and not _supplied_by_operator(name)):
                self.prior.setdefault(name, (os.getenv(name), digest))
                os.environ.pop(name, None)
                os.environ.pop(marker, None)

    def restore(self) -> None:
        for name, values in self.prior.items():
            for key, value in zip((name, bridged_env_marker(name)), values):
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        for undo in self.undo:
            undo()


# Loads are synchronous; copied contexts must not inherit another thread's partial seen-set.
_YAML_ENV_LOAD = threading.local()


def cancel_yaml_env_refresh() -> None:
    """A swallowed discovery/hook failure is not evidence that settings were deleted."""
    load = getattr(_YAML_ENV_LOAD, "current", None)
    if load is not None:
        load.failed = True


def abort_yaml_env_refresh() -> None:
    """Undo this load's bridged writes now: its caller caught a failure and continues on a fallback
    config, so neither the writes nor a retraction of missing settings may be published."""
    with _YAML_ENV_LOCK:
        load = getattr(_YAML_ENV_LOAD, "current", None)
        if load is not None:
            load.restore()
            load.prior.clear()
            load.undo.clear()
            load.failed = True


@contextlib.contextmanager
def yaml_env_transaction():
    """One load's rollback scope: everything bridged or retracted inside it is restored if it raises.

    ``load_gateway_config`` holds it across typed construction and env overrides, so a load that fails
    after the YAML layer publishes nothing; the retraction itself happens at the YAML layer's end
    (:func:`yaml_env_refresh`), before anything reads the env it cleans."""
    if bridge_env_skipped():
        yield
        return
    with _YAML_ENV_LOCK:
        if getattr(_YAML_ENV_LOAD, "current", None) is not None:
            yield  # A nested load participates in the outer load, never clears its partial writes.
            return
        load = _YamlEnvLoad(before={
            key[len(BRIDGED_ENV_MARKER_PREFIX):]: value for key, value in os.environ.items()
            if key.startswith(BRIDGED_ENV_MARKER_PREFIX) and key != BRIDGED_ENV_MARKER_PREFIX})
        _YAML_ENV_LOAD.current = load
        try:
            try:
                yield
            except BaseException:
                load.restore()
                raise
            if not load.committed:
                load.retract_leftovers()
        finally:
            del _YAML_ENV_LOAD.current


@contextlib.contextmanager
def yaml_env_refresh():
    """The YAML layer: retract only unchanged bridge-owned leftovers after one complete, successful
    overlay, before the caller builds a config from the env (env overrides read the same names)."""
    if bridge_env_skipped():
        yield
        return
    with yaml_env_transaction(), _YAML_ENV_LOCK:
        load = _YAML_ENV_LOAD.current
        if load.in_layer or load.committed:
            yield  # a nested or later layer: only the transaction's first layer decides retraction
            return
        load.in_layer = True
        try:
            yield
            load.retract_leftovers()
        finally:
            load.in_layer = False


def record_yaml_env_prior(name: str, undo: Callable[[], None]) -> None:
    """Record *name* before a bridge writes it outside :func:`yaml_env_setter` (the allow-all grant),
    so a load that raises restores it and runs *undo* to roll back that writer's ownership flag."""
    with _YAML_ENV_LOCK:
        load = getattr(_YAML_ENV_LOAD, "current", None)
        if load is not None and name not in load.prior:
            load.prior[name] = (os.getenv(name), os.getenv(bridged_env_marker(name)))
            load.undo.append(undo)


def yaml_env_setter(*, fallback: bool = False) -> Callable[[str, Any], None]:
    """``set_env(name, value)`` for ``apply_yaml_config_fn`` hooks: writes ``os.environ[name]``
    unless the operator set that var themselves, and NEVER while a multiplexed secondary
    profile's scope (or any other home's load, ``bridge_env_skipped``) is active — the gateway loads every secondary's config inside
    ``_profile_runtime_scope``, so a write there would pin that profile's policy process-wide and the
    default profile's adapters would read it as their own (first-writer-wins poisoning, #80099).
    Hooks seed the same values into the returned ``extra`` so each profile's adapter reads its own.
    Lists are comma-joined; ``None`` is skipped.

    Precedence is per *value*, not just per name: a var whose current value is the one this bridge
    wrote last time is the bridge's own leftover (a restarted gateway inherits it) and is refreshed
    from YAML, while any other preset value is an operator's and wins.
    """
    skip = bridge_env_skipped()

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
                if not _is_bridge_leftover(name, os.getenv(marker)) or _supplied_by_operator(name):
                    return  # the operator's value, not this bridge's leftover
            if load is not None:
                load.prior.setdefault(name, (current, os.getenv(marker)))
            os.environ[name] = bridged
            os.environ[marker] = bridged_env_digest(bridged)

    return set_env


def bridge_owns_env(name: str) -> bool:
    """True when ``os.environ[name]`` is this bridge's own value (YAML-derived, maybe canonicalized
    by :func:`replace_bridged_env`), not an operator's."""
    return _is_bridge_leftover(name, os.getenv(bridged_env_marker(name))) and not _supplied_by_operator(name)


def replace_bridged_env(name: str, value: str) -> None:
    """Rewrite ``os.environ[name]`` (an adapter canonicalizing a list it reads, e.g. Discord usernames
    resolved to IDs) without turning a bridge-owned value into one that looks operator-set: the marker
    follows the new value, so the next YAML load still refreshes or retracts it."""
    with _YAML_ENV_LOCK:
        marker = bridged_env_marker(name)
        owned = _is_bridge_leftover(name, os.getenv(marker))
        os.environ[name] = value
        if owned:
            os.environ[marker] = bridged_env_digest(value) or ""


# ---------------------------------------------------------------------------
# Live allowlists: a config.yaml edit reaches the next inbound event, no reconnect
# ---------------------------------------------------------------------------
# Adapters copy their gate lists at connect time (per-profile isolation, #72348). These helpers keep
# that copy and refresh it when the owning profile's config.yaml changes. An unchanged file costs a
# stat; a changed one costs one ``load_gateway_config()`` (~20 ms on a 1,000-line config), which also
# refreshes the launch profile's bridged env, so an operator's own env/.env value still wins (#109032).

_LIVE_GATE_LOCK = threading.RLock()


def gate_config_version(home: Any = None) -> tuple:
    """Change signature of *home*'s config.yaml (default: the active home) plus the managed overlay's."""
    from pathlib import Path

    from hermes_cli.config_backend import config_cache_version_or_none, config_version_or_none, get_config_backend
    from hermes_cli.managed_scope import get_managed_dir
    from hermes_constants import get_hermes_home

    home = get_hermes_home() if home is None else Path(home)
    managed = get_managed_dir()
    managed_sig = None
    if managed is not None and get_config_backend().honors_managed_config():
        managed_sig = config_version_or_none(managed / "config.yaml")
    return str(home), config_cache_version_or_none(home / "config.yaml"), managed_sig


def mark_live_gate_baseline(adapter: Any) -> None:
    """Record, on the adapter's first connect, which config the connect-time gate copy came from.
    A reconnect keeps the first baseline so an edit made before it is still picked up. The
    baseline's home is the adapter's OWNING profile: every later refresh reads that home.

    Only a config ``load_gateway_config()`` stamped has a known source. Any other (``gateway
    --config <file>``, a programmatic config) keeps its loaded gates: refreshing it from a home
    config.yaml would replace the operator's chosen policy with a file it never came from."""
    if getattr(adapter, "_live_gate_version", None) is None:
        loaded = getattr(getattr(adapter, "config", None), "loaded_version", None)
        adapter._live_gate_version = loaded if loaded is not None else ()


@contextlib.contextmanager
def _owner_scope(home: str):
    """Bind the owning profile's scope when the ambient one is another home's: a routed command
    (``/voice join`` served by another profile's runtime) starts adapter tasks in that runtime's
    scope, and its home's lists and .env must never become this adapter's gate. A secret scope
    stamped with another home counts as foreign even with no home override: its mapping, not the
    owner's, is what the gate would read."""
    from agent.secret_scope import current_secret_scope_home
    from hermes_constants import get_hermes_home, hermes_home_key

    stamped = current_secret_scope_home()
    if str(get_hermes_home()) == home and (stamped is None or hermes_home_key(stamped) == hermes_home_key(home)):
        yield
        return
    from pathlib import Path

    from gateway.run import _profile_runtime_scope
    with _profile_runtime_scope(Path(home)):
        yield


def refresh_live_gate_extra(adapter: Any, keys: Iterable[str], on_reload: Optional[Callable[[], None]] = None) -> bool:
    """Bring ``adapter.config.extra[keys]`` up to date with the OWNING profile's config.yaml when it changed.

    The owner is the home the connect-time baseline was read from, never the ambient scope: a task
    started under another profile's scope still reads (and re-derives from) the owner's file and
    .env, bound for the reload; ``on_reload`` re-derives the adapter's gate sets inside that bind.
    Returns True when the config was reloaded. An adapter that never connected (owner unknown)
    keeps its state. A config that fails to parse keeps the last good lists rather than dropping them.
    """
    baseline = getattr(adapter, "_live_gate_version", None)
    if not baseline or not baseline[0]:
        return False
    home = baseline[0]
    version = gate_config_version(home)
    if version == adapter._live_gate_version:
        return False
    with _LIVE_GATE_LOCK:
        if version == adapter._live_gate_version:
            return False
        from hermes_constants import get_routing_process_hermes_home, hermes_home_key
        launch = hermes_home_key(home) == hermes_home_key(get_routing_process_hermes_home())
        with _owner_scope(home):
            return _reload_live_gate_extra(adapter, keys, version, on_reload, launch=launch)


def _reload_live_gate_extra(adapter: Any, keys: Iterable[str], version: tuple,
                            on_reload: Optional[Callable[[], None]], launch: bool = False) -> bool:
    from gateway.config import PlatformConfig, load_gateway_config
    # The launch profile's YAML lists live in the process env: its load bridges them even under its
    # bound scope, and the installed launch mapping (frozen env + .env) then takes them, so the gate
    # below and the runner's env check both read the edited lists (multiplexed: launch_secret_scope).
    token = _LAUNCH_GATE_RELOAD.set(launch)
    try:  # ONE checked load: the fallback env-only config would drop the YAML-held lists
        loaded = load_gateway_config(strict=True)
    except Exception:
        if getattr(adapter, "_live_gate_failed_version", None) != version:  # once per edit, not per event
            adapter._live_gate_failed_version = version
            logger.warning("[%s] config.yaml changed but cannot be loaded; keeping the current allowlists",
                           getattr(adapter, "name", "?"), exc_info=True)
        return False
    finally:
        _LAUNCH_GATE_RELOAD.reset(token)
    if launch:  # _owner_scope bound the launch home's own scope, so this load bridged
        from tui_gateway.launch_profile_policy import follow_launch_bridged_env
        follow_launch_bridged_env()
    current = loaded.platforms.get(adapter.platform) or PlatformConfig()
    extra = adapter.config.extra
    for key in keys:
        if key in current.extra:
            extra[key] = current.extra[key]
        else:
            extra.pop(key, None)
    if on_reload is not None:
        on_reload()
    # Retry this same edit after a transient read failure, without requiring a second write.
    adapter._live_gate_version = version
    return True


def send_error(message: Any, *, nothing_sent: bool = False) -> dict:
    """Standalone-sender failure envelope with vendor exception text redacted (the same helper
    ``send_message`` uses), so a token or signed URL in an httpx/aiohttp error never reaches the
    model transcript.

    ``nothing_sent=True`` adds ``"retryable": True``: the sender PROVES nothing reached the chat
    (no credential, or the connection failed before the request left). Only that marker lets cron
    queue the payload for a post-reconnect replay; any other failure may already be on screen, so
    a replay would post it twice."""
    from tools.send_message_senders import _error
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
