"""One frozen routing identity per inbound gateway event.

A multiplexed gateway answers three questions about every event, and until now answered them in
three places that only agreed because they read the same fallback chain: WHICH bot received it
(``_transport_owner``), WHO may admit it (``_authorization_home_for_source``) and WHERE the turn
runs (``_resolve_profile_home_for_source`` / ``_session_key_profile``). :func:`resolve_identity`
answers all three once and pins the result on the source as a wire-invisible dynamic attribute
(like ``_transport_adapter_ref``); the existing helpers read it when present and keep their
fallback chain when absent, so a source built outside the runner still resolves as before.

``SessionSource.profile`` stays the serialized runtime profile — ``None`` on the wire means the
receiving bot's own profile — so nothing here changes the wire format or any historical
``agent:main`` key.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from contextlib import suppress
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from gateway.session import SessionSource

logger = logging.getLogger(__name__)

_IDENTITY_ATTR = "_identity"
# Wire-invisible provenance copied alongside the identity when a source is duplicated.
_PROVENANCE_ATTRS = ("_transport_adapter_ref", "_authorization_profile_home", _IDENTITY_ATTR)


class IdentityUnresolved(RuntimeError):
    """Under multiplexing the event's runtime profile could not be established (an explicit
    ``profile_routes`` entry targets a profile this gateway does not serve). Callers drop the
    event; it must never fall through to the default profile."""


@dataclass(frozen=True)
class RoutingIdentity:
    """Everything a turn needs to know about who it is, resolved once at ingress.

    ``transport_profile`` owns the receiving adapter (its credential and allowlist);
    ``runtime_profile`` is the profile that executes the turn — the same name unless a
    ``profile_routes`` entry re-homed the event. Both are explicit (``"default"`` is spelled out);
    ``None`` never means default here. ``multiplexed`` is False for a standalone gateway, whose
    keys stay in the legacy ``agent:main`` namespace whatever profile it was launched with.
    """

    transport_profile: str
    runtime_profile: str
    authorization_home: Path
    runtime_home: Path
    multiplexed: bool = True
    # Receiving adapter; None for restored/synthetic sources (no live provenance → fail closed).
    # Provenance, not identity: two events from the same bot share one identity.
    transport: Optional[weakref.ref] = field(default=None, compare=False, hash=False)
    # True when nothing named the receiving bot (no live adapter, no persisted transport_profile,
    # no explicit hint) and ``transport_profile`` is the primary by default. A hand-built or
    # pre-column source. Delivery may still fall back to the runtime profile's unique adapter for
    # these; an identity whose transport is KNOWN (live or restored) never does.
    transport_inferred: bool = field(default=False, compare=False, hash=False)

    @property
    def namespace(self) -> str:
        """``agent:<ns>`` prefix for this identity's session keys — byte-identical to
        :func:`gateway.session._session_key_namespace` for every historical key."""
        from gateway.session import _session_key_namespace
        return _session_key_namespace(self.session_key_profile)

    @property
    def store_path(self) -> Path:
        return self.runtime_home / "state.db"

    @property
    def session_key_profile(self) -> Optional[str]:
        """The ``profile=`` argument :func:`gateway.session.build_session_key` expects for this
        identity: the runtime profile under multiplexing, else ``None`` (legacy namespace)."""
        return self.runtime_profile if self.multiplexed else None

    def adapter(self) -> Any:
        """The live receiving adapter, or None when it is gone or was never known."""
        return self.transport() if self.transport is not None else None


def identity_of(source: Any) -> Optional[RoutingIdentity]:
    """The identity pinned on *source* by :func:`resolve_identity`, if any."""
    identity = getattr(source, _IDENTITY_ATTR, None)
    return identity if isinstance(identity, RoutingIdentity) else None


def clear_identity(source: Any) -> None:
    """Drop the pinned identity AND the routed runtime profile so the next :func:`canonical_identity`
    re-routes from scratch — for sources a caller reuses across events whose routing may differ (a
    guild's cached voice source is shared by every speaker, and ``profile_routes[].user_id`` routes
    per speaker). ``source.profile`` is the previous event's routing *result*; left in place it
    short-circuits the route and the new speaker runs as the old one."""
    for attr in (_IDENTITY_ATTR, "profile_route_rejected", "_authorization_profile_home"):
        with suppress(AttributeError):
            delattr(source, attr)
    with suppress(AttributeError):
        source.profile = None


def transport_profile_of(source: Any) -> Optional[str]:
    """The receiving bot's profile to persist alongside a routing entry (``SessionEntry.transport_profile``);
    None outside multiplexing or when nothing resolved the source (an unknown transport is never guessed)."""
    identity = identity_of(source)
    return identity.transport_profile if identity is not None and identity.multiplexed else None


def replace_source(source: "SessionSource", **changes: Any) -> "SessionSource":
    """:func:`dataclasses.replace` that keeps the wire-invisible provenance (transport ref,
    authorization home, identity). A plain ``replace`` silently produces a source the runner
    can only route through heuristics."""
    copied = dataclasses.replace(source, **changes)
    for name in _PROVENANCE_ATTRS:
        value = getattr(source, name, None)
        if value is not None:
            setattr(copied, name, value)
    return copied


def _name(value: Any) -> Optional[str]:
    text = value.strip() if isinstance(value, str) else ""
    return text or None


def canonical_identity(
    source: "SessionSource", *, runner: Any, adapter: Any = None,
    transport_profile: Optional[str] = None, primary_home: Optional[Path] = None,
) -> Optional[RoutingIdentity]:
    """The identity already pinned on *source*, else :func:`resolve_identity` — the one call every
    ingress path makes FIRST, before any key is derived. ``None`` = unresolved under multiplexing
    (``source.profile_route_rejected`` is set): the caller drops the event and says so once; it
    must never fall through to ``agent:main``."""
    identity = identity_of(source)
    if identity is not None:
        return identity
    try:
        return resolve_identity(
            source, runner=runner, adapter=adapter, transport_profile=transport_profile,
            primary_home=primary_home)
    except IdentityUnresolved as exc:
        logger.debug("identity unresolved: %s", exc)
        return None


def restore_identity(
    source: "SessionSource", *, runner: Any, transport_profile: Optional[str],
) -> Optional[RoutingIdentity]:
    """Pin the identity of a source rebuilt from durable state (``SessionEntry.origin``, a
    ``sessions`` row, a cached copy) — no live adapter, so ``transport=None``: the restored row of the
    transport matrix, where delivery goes through the persisted transport owner or fails closed.

    *transport_profile* is what the routing index persisted at ingress (``SessionEntry.transport_profile``);
    ``None`` = a row written before the column existed, whose transport is unknown → nothing is pinned
    and the legacy heuristics (``_is_shared_bot_satellite``) keep deciding. Standalone gateways have
    nothing to restore (one bot, one home).
    """
    transport_name = _name(transport_profile)
    if transport_name is None:
        return None
    if not bool(getattr(getattr(runner, "config", None), "multiplex_profiles", False)):
        return None
    existing = identity_of(source)
    if existing is not None:
        return existing
    from profiles.paths import get_profile_dir
    from hermes_constants import get_process_hermes_home

    primary_profile = _name(getattr(runner, "_primary_profile_name", None)) or "default"
    runtime_name = _name(getattr(source, "profile", None)) or primary_profile
    authorization_home = (
        Path(get_process_hermes_home()) if transport_name == primary_profile
        else get_profile_dir(transport_name))
    runtime_home = (
        authorization_home if runtime_name == transport_name
        else Path(runner._resolve_profile_home_for_source(source)))
    source._authorization_profile_home = authorization_home
    identity = RoutingIdentity(
        transport_profile=transport_name, runtime_profile=runtime_name,
        authorization_home=authorization_home, runtime_home=runtime_home,
        multiplexed=True, transport=None)
    setattr(source, _IDENTITY_ATTR, identity)
    return identity


def resolve_identity(
    source: "SessionSource", *, runner: Any, adapter: Any = None,
    transport_profile: Optional[str] = None, primary_home: Optional[Path] = None,
) -> RoutingIdentity:
    """Resolve and pin the :class:`RoutingIdentity` of an inbound *source*.

    *adapter* is the receiving adapter when the caller holds it; otherwise the source's own
    transport provenance is consulted. *transport_profile* names the receiving bot's owning
    profile when the caller knows it by construction (the runner's per-profile handlers);
    ``None`` = derive it from the adapter registry, primary when unknown. *primary_home* is the
    primary bot's home for authorization (default: the process home, never a per-turn override).

    Stamps ``source.profile`` the way the ingress handlers always did (routed name, else a
    secondary's own name; ``None`` stays ``None`` for the primary so the wire is unchanged) and
    ``_authorization_profile_home`` for the existing authorization readers.

    Raises :class:`IdentityUnresolved` under multiplexing when the route is rejected.
    """
    from gateway.profile_routing import ProfileRouteRejected
    from hermes_constants import get_hermes_home, get_process_hermes_home

    multiplexed = bool(getattr(getattr(runner, "config", None), "multiplex_profiles", False))
    primary_profile = _name(getattr(runner, "_primary_profile_name", None))
    if primary_profile is None:
        active = getattr(runner, "_active_profile_name", None)
        primary_profile = (_name(active()) if callable(active) else None) or "default"
    platform = getattr(source, "platform", None)

    owner_profile: Optional[str] = None
    if adapter is None:
        owner = runner._transport_owner(source)
        if owner is not None:
            adapter, owner_profile = owner
    else:
        if getattr(source, "_transport_adapter_ref", None) is None:
            source._transport_adapter_ref = weakref.ref(adapter)
        _registered, owner_profile = runner._owning_profile(adapter, platform)
    transport_name = _name(transport_profile) or _name(owner_profile) or primary_profile
    transport_inferred = adapter is None and _name(transport_profile) is None and _name(owner_profile) is None
    transport_ref = weakref.ref(adapter) if adapter is not None else None

    if not multiplexed:
        home = Path(get_hermes_home())
        identity = RoutingIdentity(
            transport_profile=primary_profile, runtime_profile=primary_profile,
            authorization_home=home, runtime_home=home, multiplexed=False, transport=transport_ref)
        setattr(source, _IDENTITY_ATTR, identity)
        return identity

    if transport_name == primary_profile:
        authorization_home = Path(primary_home) if primary_home is not None else Path(get_process_hermes_home())
    else:
        from profiles.paths import get_profile_dir
        authorization_home = get_profile_dir(transport_name)
    source._authorization_profile_home = authorization_home

    where = f"{getattr(platform, 'value', platform)}/{getattr(source, 'chat_id', '')}"
    if getattr(source, "profile_route_rejected", False) is True:
        raise IdentityUnresolved(f"{where}: profile route rejected")
    if _name(getattr(source, "profile", None)) is None:
        adapter_profile = None if transport_name == primary_profile else transport_name
        try:
            # The primary keeps the historical one-argument call (its adapter profile is None).
            routed = (
                runner._profile_name_for_source(source) if adapter_profile is None
                else runner._profile_name_for_source(source, adapter_profile=adapter_profile))
        except ProfileRouteRejected as exc:
            source.profile_route_rejected = True
            raise IdentityUnresolved(f"{where}: {exc}") from exc
        source.profile = routed or adapter_profile

    runtime_name = _name(source.profile) or primary_profile
    # A routed runtime goes through the runner's resolver (missing-profile fallback + warning);
    # a bot serving its own profile runs where it authorizes.
    runtime_home = (
        authorization_home if runtime_name == transport_name
        else Path(runner._resolve_profile_home_for_source(source)))
    identity = RoutingIdentity(
        transport_profile=transport_name, runtime_profile=runtime_name,
        authorization_home=authorization_home, runtime_home=runtime_home,
        multiplexed=True, transport=transport_ref, transport_inferred=transport_inferred)
    setattr(source, _IDENTITY_ATTR, identity)
    return identity


def authenticated_subject(identity):
    """Keep private native receipts stable; namespace every dashboard identity.

    Only the private bootstrap stamps profile/instance/capabilities. A dashboard
    provider named 'local' (or a subject resembling a UID) is not native provenance.
    Historical remote bare-subject receipts cannot be assigned to a first caller.
    """
    subject = str(identity.get('user_id') or 'unbound')
    if (identity.get('provider') == 'local' and identity.get('profile_id')
            and identity.get('instance_id') and 'capabilities' in identity):
        return subject
    return 'auth:v1:' + json.dumps(
        [identity.get('provider') or '', identity.get('issuer') or '', subject],
        separators=(',', ':'), ensure_ascii=True)
