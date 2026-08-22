"""Canonical routing identity for multiplexed gateway turns.

Issue #88715 exposed that Hermes used one overloaded ``SessionSource.profile``
value for three independent questions:

* which live credential/adapter received and must deliver the turn;
* which runtime profile owns config, tools, memory, and the session namespace;
* which profile home physically stores the session.

Those values are often equal, but profile routes deliberately allow a shared
credential to serve a different runtime profile.  Keeping them independent is
therefore a correctness requirement, not optional metadata.

This module has no gateway imports and is safe to use from adapters, session
storage, callbacks, and worker threads.  ``ContextVar`` propagation covers
normal asyncio task creation and ``asyncio.to_thread``.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from functools import wraps
import logging
from typing import Any, Awaitable, Callable, Iterable, Iterator, Mapping, Optional, TypeVar

DEFAULT_PROFILE = "default"
LEGACY_SESSION_NAMESPACE = "main"
logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Awaitable[Any]])
_SOURCE_PROFILE_UNSET = object()


class RoutingIdentityError(RuntimeError):
    """Base class for fail-closed routing identity failures."""


class RoutingIdentityRejected(RoutingIdentityError):
    """An explicit route or ownership claim cannot be served safely."""


class RoutingIdentityConflict(RoutingIdentityRejected):
    """Two authoritative identity claims disagree."""


def normalize_profile(value: object, *, default: str = DEFAULT_PROFILE) -> str:
    """Return the canonical internal profile name.

    ``None``, empty strings, and the legacy session namespace ``main`` all
    denote the default profile.  The external session-key format remains
    unchanged: :func:`session_namespace` maps the canonical default back to
    ``main``.
    """

    if value is None:
        return default
    if not isinstance(value, str):
        # Non-string values (e.g. MagicMock attributes on test doubles) are
        # not profile claims. Coercing a repr would fabricate a profile name
        # that collides with a real one; treat them as absent instead.
        return default
    text = value.strip()
    if not text or text in {DEFAULT_PROFILE, LEGACY_SESSION_NAMESPACE}:
        return default
    return text


def session_namespace(profile: object) -> str:
    """Return the public ``agent:<namespace>`` slot for *profile*."""

    canonical = normalize_profile(profile)
    return LEGACY_SESSION_NAMESPACE if canonical == DEFAULT_PROFILE else canonical


def _normalized_set(values: Optional[Iterable[object]]) -> Optional[frozenset[str]]:
    if values is None:
        return None
    return frozenset(normalize_profile(value) for value in values)


@dataclass(frozen=True, slots=True)
class RoutingIdentity:
    """Canonical ownership of one gateway turn.

    ``transport_profile``
        Owns the live credential and every outbound branch for the turn.

    ``runtime_profile``
        Owns config, tools, memory, callbacks, and the public session-key
        namespace.

    ``persistence_profile``
        Owns the physical ``state.db`` and routing index.  Under the architecture
        selected by merged PR #88734 this must equal ``runtime_profile``.
    """

    transport_profile: str
    runtime_profile: str
    persistence_profile: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "transport_profile", normalize_profile(self.transport_profile)
        )
        object.__setattr__(self, "runtime_profile", normalize_profile(self.runtime_profile))
        object.__setattr__(
            self, "persistence_profile", normalize_profile(self.persistence_profile)
        )
        if self.persistence_profile != self.runtime_profile:
            raise RoutingIdentityConflict(
                "physical session ownership must equal runtime ownership under "
                "the per-profile-store invariant"
            )

    @property
    def key_namespace(self) -> str:
        return session_namespace(self.runtime_profile)

    @property
    def is_default_runtime(self) -> bool:
        return self.runtime_profile == DEFAULT_PROFILE

    def to_persistence_dict(self) -> dict[str, str]:
        """Trusted routing-index representation.

        This object belongs beside a persisted ``SessionEntry``.  It must not be
        copied into the wire-serializable ``SessionSource.to_dict`` payload,
        because a remote relay peer must not be able to claim a local credential
        owner.
        """

        return {
            "transport_profile": self.transport_profile,
            "runtime_profile": self.runtime_profile,
            "persistence_profile": self.persistence_profile,
        }

    @classmethod
    def from_persistence_dict(cls, value: Mapping[str, object]) -> "RoutingIdentity":
        if not isinstance(value, Mapping):
            raise RoutingIdentityRejected("persisted routing identity is not a mapping")
        required = {
            "transport_profile",
            "runtime_profile",
            "persistence_profile",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise RoutingIdentityRejected(
                f"persisted routing identity is missing {', '.join(missing)}"
            )
        return cls(
            transport_profile=str(value["transport_profile"]),
            runtime_profile=str(value["runtime_profile"]),
            persistence_profile=str(value["persistence_profile"]),
        )


def canonicalize_routing_identity(
    *,
    route_profile: object = None,
    source_profile: object = None,
    credential_owner: object = None,
    restored_transport_profile: object = None,
    restored_persistence_profile: object = None,
    active_profile: object = DEFAULT_PROFILE,
    served_profiles: Optional[Iterable[object]] = None,
    route_rejected: bool = False,
) -> RoutingIdentity:
    """Resolve one canonical identity or reject the turn.

    Precedence is route first, then an explicit/restored source profile, then
    credential ownership, then the process's active profile.  Independent
    authoritative claims are checked before precedence is applied; conflicting
    claims never silently collapse into ``default``.

    Duplicate-credential enforcement (one credential, one profile) is a
    startup invariant enforced by the multiplexer's adapter-install path
    (``GatewayRunner._configure_profile_adapter`` ``claimed`` map / ``_adapter_credential_fingerprint``),
    not a per-turn concern here.
    """

    if route_rejected:
        raise RoutingIdentityRejected("matched profile route was explicitly rejected")

    served = _normalized_set(served_profiles)

    route = normalize_profile(route_profile) if route_profile not in (None, "") else None
    source = (
        normalize_profile(source_profile)
        if source_profile not in (None, "")
        else None
    )
    owner = (
        normalize_profile(credential_owner)
        if credential_owner not in (None, "")
        else None
    )
    restored_owner = (
        normalize_profile(restored_transport_profile)
        if restored_transport_profile not in (None, "")
        else None
    )
    restored_store = (
        normalize_profile(restored_persistence_profile)
        if restored_persistence_profile not in (None, "")
        else None
    )
    active = normalize_profile(active_profile)

    if route is not None and source is not None and route != source:
        raise RoutingIdentityConflict(
            f"route profile {route!r} conflicts with stamped source profile {source!r}"
        )
    if owner is not None and restored_owner is not None and owner != restored_owner:
        raise RoutingIdentityConflict(
            f"live credential owner {owner!r} conflicts with restored owner "
            f"{restored_owner!r}"
        )

    runtime = route or source or owner or active
    transport = owner or restored_owner or runtime
    persistence = restored_store or runtime

    if persistence != runtime:
        raise RoutingIdentityConflict(
            f"restored persistence profile {persistence!r} conflicts with runtime "
            f"profile {runtime!r}"
        )

    if served is not None:
        for label, profile in (
            ("runtime", runtime),
            ("transport", transport),
            ("persistence", persistence),
        ):
            if profile not in served:
                raise RoutingIdentityRejected(
                    f"{label} profile {profile!r} is not served by this gateway"
                )

    return RoutingIdentity(
        transport_profile=transport,
        runtime_profile=runtime,
        persistence_profile=persistence,
    )


_CURRENT_ROUTING_IDENTITY: ContextVar[Optional[RoutingIdentity]] = ContextVar(
    "gateway_routing_identity", default=None
)


def current_routing_identity() -> Optional[RoutingIdentity]:
    return _CURRENT_ROUTING_IDENTITY.get()


def require_routing_identity() -> RoutingIdentity:
    identity = current_routing_identity()
    if identity is None:
        raise RoutingIdentityRejected("no canonical routing identity is active")
    return identity


@contextmanager
def routing_identity_scope(identity: RoutingIdentity) -> Iterator[RoutingIdentity]:
    token: Token[Optional[RoutingIdentity]] = _CURRENT_ROUTING_IDENTITY.set(identity)
    try:
        yield identity
    finally:
        _CURRENT_ROUTING_IDENTITY.reset(token)


def attach_identity_to_source(source: Any, identity: RoutingIdentity) -> Any:
    """Attach trusted identity without exposing transport ownership on the wire."""

    # ``profile`` is the existing public/runtime field.  Keep default as None
    # for byte-compatible legacy behavior on non-multiplexed/default routes.
    source.profile = None if identity.is_default_runtime else identity.runtime_profile
    source._routing_identity = identity
    source._transport_profile = identity.transport_profile
    source._persistence_profile = identity.persistence_profile
    # Dataclass-aware internal carrier: unlike an ad-hoc attribute this is
    # preserved by dataclasses.replace once SessionSource declares the field.
    # SessionSource.to_dict deliberately does not serialize it.
    try:
        source.routing_identity = identity.to_persistence_dict()
    except Exception:
        pass
    return source


def identity_from_source(
    source: Any,
    *,
    route_profile: object = None,
    source_profile: object = _SOURCE_PROFILE_UNSET,
    active_profile: object = DEFAULT_PROFILE,
    served_profiles: Optional[Iterable[object]] = None,
    credential_owner: object = None,
) -> RoutingIdentity:
    """Recover a trusted identity from a live or restored source.

    A previously attached ``RoutingIdentity`` wins after consistency checks.
    Legacy sources without private identity metadata are reconstructed from the
    public runtime profile and the supplied live credential owner.
    """

    effective_source_profile = (
        getattr(source, "profile", None)
        if source_profile is _SOURCE_PROFILE_UNSET
        else source_profile
    )

    attached = getattr(source, "_routing_identity", None)
    if isinstance(attached, RoutingIdentity):
        if route_profile not in (None, ""):
            routed = normalize_profile(route_profile)
            if routed != attached.runtime_profile:
                raise RoutingIdentityConflict(
                    f"resolved route {routed!r} conflicts with attached runtime "
                    f"{attached.runtime_profile!r}"
                )
        if effective_source_profile not in (None, ""):
            claimed_runtime = normalize_profile(effective_source_profile)
            if claimed_runtime != attached.runtime_profile:
                raise RoutingIdentityConflict(
                    f"source profile {claimed_runtime!r} conflicts with attached "
                    f"runtime {attached.runtime_profile!r}"
                )
        if credential_owner not in (None, ""):
            live_owner = normalize_profile(credential_owner)
            if live_owner != attached.transport_profile:
                raise RoutingIdentityConflict(
                    f"live credential owner {live_owner!r} conflicts with attached "
                    f"owner {attached.transport_profile!r}"
                )
        served = _normalized_set(served_profiles)
        if served is not None:
            missing = {
                attached.transport_profile,
                attached.runtime_profile,
                attached.persistence_profile,
            }.difference(served)
            if missing:
                raise RoutingIdentityRejected(
                    "attached identity references unserved profile(s): "
                    + ", ".join(sorted(missing))
                )
        return attached

    carried = getattr(source, "routing_identity", None)
    if isinstance(carried, Mapping):
        restored = RoutingIdentity.from_persistence_dict(carried)
        if route_profile not in (None, ""):
            routed = normalize_profile(route_profile)
            if routed != restored.runtime_profile:
                raise RoutingIdentityConflict(
                    f"resolved route {routed!r} conflicts with restored runtime "
                    f"{restored.runtime_profile!r}"
                )
        if effective_source_profile not in (None, ""):
            claimed_runtime = normalize_profile(effective_source_profile)
            if claimed_runtime != restored.runtime_profile:
                raise RoutingIdentityConflict(
                    f"source profile {claimed_runtime!r} conflicts with restored "
                    f"runtime {restored.runtime_profile!r}"
                )
        if credential_owner not in (None, ""):
            live_owner = normalize_profile(credential_owner)
            if live_owner != restored.transport_profile:
                raise RoutingIdentityConflict(
                    f"live credential owner {live_owner!r} conflicts with carried "
                    f"owner {restored.transport_profile!r}"
                )
        served = _normalized_set(served_profiles)
        if served is not None:
            missing = {
                restored.transport_profile,
                restored.runtime_profile,
                restored.persistence_profile,
            }.difference(served)
            if missing:
                raise RoutingIdentityRejected(
                    "restored identity references unserved profile(s): "
                    + ", ".join(sorted(missing))
                )
        attach_identity_to_source(source, restored)
        return restored

    return canonicalize_routing_identity(
        route_profile=route_profile,
        source_profile=effective_source_profile,
        credential_owner=credential_owner,
        restored_transport_profile=getattr(source, "_transport_profile", None),
        restored_persistence_profile=getattr(source, "_persistence_profile", None),
        active_profile=active_profile,
        served_profiles=served_profiles,
        route_rejected=getattr(source, "profile_route_rejected", False) is True,
    )


def persistence_payload_for_source(source: Any) -> Optional[dict[str, str]]:
    """Return the trusted routing-index payload for *source*, when available."""

    attached = getattr(source, "_routing_identity", None)
    if isinstance(attached, RoutingIdentity):
        return attached.to_persistence_dict()
    carried = getattr(source, "routing_identity", None)
    if isinstance(carried, Mapping):
        return RoutingIdentity.from_persistence_dict(carried).to_persistence_dict()

    transport = getattr(source, "_transport_profile", None)
    persistence = getattr(source, "_persistence_profile", None)
    runtime = getattr(source, "profile", None)
    # A public runtime profile alone is not enough to reconstruct the
    # credential owner.  Persist only identity that was established by the
    # local adapter/ingress seam; never manufacture transport=runtime while
    # serialising a restored or wire-originated source.
    if transport in (None, "") or persistence in (None, ""):
        return None
    identity = canonicalize_routing_identity(
        source_profile=runtime,
        restored_transport_profile=transport,
        restored_persistence_profile=persistence,
    )
    return identity.to_persistence_dict()


def restore_identity_on_source(
    source: Any,
    payload: Optional[Mapping[str, object]],
) -> Optional[RoutingIdentity]:
    """Restore trusted routing-index metadata onto a deserialized source."""

    if not payload:
        return None
    identity = RoutingIdentity.from_persistence_dict(payload)
    attach_identity_to_source(source, identity)
    return identity


def _credential_owner_for_adapter(
    runner: Any,
    adapter: Any,
    source: Any,
) -> Optional[str]:
    """Resolve the profile that owns *adapter*'s live credential.

    Adapter-local intake reaches :meth:`BasePlatformAdapter.handle_message`
    before the shared runner handler.  Resolve ownership from the concrete
    adapter first so session-keying, topic recovery, busy policy, callbacks,
    and the background task all enter the same canonical scope.
    """

    carried = getattr(source, "_transport_profile", None)
    if carried not in (None, ""):
        return normalize_profile(carried)

    # main's set_owner_profile (_configure_profile_adapter, #89860) stores the
    # canonical credential owner on the adapter. Consume that seam rather than
    # a duplicate private field.
    explicit = getattr(adapter, "_owner_profile", None)
    if explicit not in (None, ""):
        return normalize_profile(explicit)

    platform = getattr(source, "platform", None) or getattr(adapter, "platform", None)
    if adapter is (getattr(runner, "adapters", None) or {}).get(platform):
        return normalize_profile(
            getattr(runner, "_primary_profile_name", None) or DEFAULT_PROFILE
        )
    for profile, adapters in (
        getattr(runner, "_profile_adapters", None) or {}
    ).items():
        if adapter is (adapters or {}).get(platform):
            return normalize_profile(profile)
    return None


def _credential_owner_for_source(runner: Any, source: Any) -> Optional[str]:
    """Resolve the process-local credential owner without runtime fallback."""

    carried = getattr(source, "_transport_profile", None)
    if carried not in (None, ""):
        return normalize_profile(carried)

    adapter = None
    registered = getattr(runner, "_registered_transport_adapter", None)
    if callable(registered):
        try:
            adapter = registered(source)
        except Exception:
            adapter = None
    if adapter is not None:
        return _credential_owner_for_adapter(runner, adapter, source)
    return None


def _served_profiles_for_runner(runner: Any) -> Optional[frozenset[str]]:
    """Best-effort authoritative served-profile set for one runner.

    Production multiplexed runners can enumerate profiles from the same
    ``profiles_to_serve`` helper used by route validation.  Process-local maps
    are included as corroborating evidence and keep partial test fixtures
    useful.  ``None`` means the set could not be established; callers still
    enforce route and credential consistency, but do not invent a served set.
    """

    profiles: set[str] = set()
    primary = getattr(runner, "_primary_profile_name", None)
    if primary not in (None, ""):
        profiles.add(normalize_profile(primary))

    for attribute in ("_profile_adapters", "pairing_stores"):
        mapping = getattr(runner, attribute, None)
        if isinstance(mapping, Mapping):
            profiles.update(normalize_profile(name) for name in mapping)

    config = getattr(runner, "config", None)
    if bool(getattr(config, "multiplex_profiles", False)):
        try:
            from hermes_cli.profiles import profiles_to_serve

            allowlist = getattr(config, "multiplex_profile_allowlist", None)
            for name, _home in profiles_to_serve(
                multiplex=True, profile_allowlist=allowlist
            ):
                profiles.add(normalize_profile(name))
        except Exception:
            # The local maps above may still be sufficient.  Do not turn an
            # enumeration failure into a guessed singleton set.
            pass

    return frozenset(profiles) if profiles else None


def _route_profile_for_source(
    runner: Any,
    source: Any,
    *,
    multiplexed: bool,
) -> Optional[str]:
    """Resolve an explicit route or raise a fail-closed identity error."""

    if not multiplexed:
        return None
    if getattr(source, "profile_route_rejected", False) is True:
        raise RoutingIdentityRejected("matched profile route was explicitly rejected")

    resolver = getattr(runner, "_profile_name_for_source", None)
    if not callable(resolver):
        return None
    try:
        routed = resolver(source)
    except Exception as exc:
        try:
            from gateway.profile_routing import ProfileRouteRejected
        except Exception:  # pragma: no cover - stripped scaffold
            ProfileRouteRejected = ()  # type: ignore[assignment]
        if ProfileRouteRejected and isinstance(exc, ProfileRouteRejected):
            source.profile_route_rejected = True
            raise RoutingIdentityRejected("matched profile route is not served") from exc
        raise RoutingIdentityRejected("profile route resolution failed") from exc
    return normalize_profile(routed) if routed not in (None, "") else None


def resolve_identity_for_runner_source(
    runner: Any,
    source: Any,
    *,
    adapter: Any = None,
) -> RoutingIdentity:
    """Resolve and attach the canonical identity before any adapter-local work."""

    config = getattr(runner, "config", None)
    multiplexed = bool(getattr(config, "multiplex_profiles", False))
    routed_profile = _route_profile_for_source(
        runner, source, multiplexed=multiplexed
    )
    primary = normalize_profile(
        getattr(runner, "_primary_profile_name", None) or DEFAULT_PROFILE
    )
    owner = (
        _credential_owner_for_adapter(runner, adapter, source)
        if adapter is not None
        else _credential_owner_for_source(runner, source)
    )

    source_claim = getattr(source, "profile", None)
    attached = getattr(source, "_routing_identity", None)
    carried = getattr(source, "routing_identity", None)
    has_trusted_carrier = (
        isinstance(attached, RoutingIdentity)
        or isinstance(carried, Mapping)
        or (
            getattr(source, "_transport_profile", None) not in (None, "")
            and getattr(source, "_persistence_profile", None) not in (None, "")
        )
    )

    # A callback or child task may construct a fresh source while already
    # executing inside a proven turn. Inherit that immutable identity only
    # when every explicit runtime claim agrees; otherwise reject.
    inherited = current_routing_identity()
    if owner is None and not has_trusted_carrier and inherited is not None:
        if routed_profile is not None and routed_profile != inherited.runtime_profile:
            raise RoutingIdentityConflict(
                f"resolved route {routed_profile!r} conflicts with inherited runtime "
                f"{inherited.runtime_profile!r}"
            )
        if source_claim not in (None, ""):
            claimed_runtime = normalize_profile(source_claim)
            if claimed_runtime != inherited.runtime_profile:
                raise RoutingIdentityConflict(
                    f"source profile {claimed_runtime!r} conflicts with inherited "
                    f"runtime {inherited.runtime_profile!r}"
                )
        served = _served_profiles_for_runner(runner)
        if served is not None:
            missing = {
                inherited.transport_profile,
                inherited.runtime_profile,
                inherited.persistence_profile,
            }.difference(served)
            if missing:
                raise RoutingIdentityRejected(
                    "inherited identity references unserved profile(s): "
                    + ", ".join(sorted(missing))
                )
        attach_identity_to_source(source, inherited)
        return inherited

    # In multiplex mode a public runtime stamp is not credential provenance.
    # Restored/synthetic sources with neither a live adapter owner nor trusted
    # local routing metadata must fail closed instead of guessing that the
    # runtime profile owns the outbound credential.
    if multiplexed and owner is None and not has_trusted_carrier:
        raise RoutingIdentityRejected(
            "multiplexed source has no trusted transport binding"
        )

    # Legacy secondary-profile handlers stamped the credential owner into
    # source.profile. A verified route is authoritative over that legacy stamp,
    # while an unrelated source claim remains an explicit conflict.
    if (
        routed_profile is not None
        and owner not in (None, "")
        and source_claim not in (None, "")
        and normalize_profile(source_claim) == normalize_profile(owner)
    ):
        source_claim = None

    identity = identity_from_source(
        source,
        route_profile=routed_profile,
        source_profile=source_claim,
        active_profile=primary,
        served_profiles=_served_profiles_for_runner(runner),
        credential_owner=owner,
    )
    attach_identity_to_source(source, identity)
    return identity


async def _invoke_in_routing_scope(
    runner: Any,
    source: Any,
    identity: RoutingIdentity,
    function: Callable[..., Awaitable[Any]],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Invoke *function* inside identity and profile-home scope."""

    multiplexed = bool(
        getattr(getattr(runner, "config", None), "multiplex_profiles", False)
    )
    if multiplexed:
        resolver = getattr(runner, "_resolve_profile_home_for_source", None)
        if not callable(resolver):
            raise RoutingIdentityRejected(
                "multiplexed runner has no profile-home resolver"
            )
        try:
            profile_home = resolver(source)
        except Exception as exc:
            raise RoutingIdentityRejected(
                "runtime profile home could not be resolved"
            ) from exc

        # Lazy import avoids a cycle while gateway.run is defining the
        # decorated class method. The wrapper executes after module load.
        from gateway.run import _profile_runtime_scope

        with routing_identity_scope(identity), _profile_runtime_scope(profile_home):
            return await function(*args, **kwargs)

    with routing_identity_scope(identity):
        return await function(*args, **kwargs)


def _reject_source(source: Any, message: str, exc: BaseException) -> None:
    try:
        source.profile_route_rejected = True
    except Exception:
        pass
    logger.warning("Dropping event with %s: %s", message, exc)


def routing_identity_adapter_entrypoint(function: _F) -> _F:
    """Scope adapter-local intake before topic recovery and session keying.

    ``BasePlatformAdapter.handle_message`` performs topic recovery, derives the
    adapter guard key, dispatches busy/clarify/control paths, and creates the
    long-running processing task before the shared runner handler is necessarily
    reached. Decorating that adapter entrypoint makes the identity active for
    every one of those branches; the background task inherits both ContextVars.
    """

    @wraps(function)
    async def wrapped(adapter: Any, event: Any, *args: Any, **kwargs: Any) -> Any:
        source = getattr(event, "source", None)
        runner = getattr(adapter, "gateway_runner", None)
        if source is None or runner is None:
            return await function(adapter, event, *args, **kwargs)
        try:
            identity = resolve_identity_for_runner_source(
                runner, source, adapter=adapter
            )
            return await _invoke_in_routing_scope(
                runner,
                source,
                identity,
                function,
                adapter,
                event,
                *args,
                **kwargs,
            )
        except RoutingIdentityRejected as exc:
            _reject_source(source, "invalid adapter routing identity", exc)
            return None

    return wrapped  # type: ignore[return-value]


def routing_identity_entrypoint(function: _F) -> _F:
    """Decorate the shared gateway ingress with canonical identity scope.

    This is the second proof gate. Adapter-local intake already enters the scope
    when available, while restored/internal events can reach the runner directly;
    both paths are resolved and checked against live credential ownership here.
    """

    @wraps(function)
    async def wrapped(runner: Any, event: Any, *args: Any, **kwargs: Any) -> Any:
        source = getattr(event, "source", None)
        if source is None:
            return await function(runner, event, *args, **kwargs)
        try:
            identity = resolve_identity_for_runner_source(runner, source)
            return await _invoke_in_routing_scope(
                runner,
                source,
                identity,
                function,
                runner,
                event,
                *args,
                **kwargs,
            )
        except RoutingIdentityRejected as exc:
            _reject_source(source, "invalid runner routing identity", exc)
            return None

    return wrapped  # type: ignore[return-value]
