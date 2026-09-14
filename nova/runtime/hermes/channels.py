"""Compiling NOVA's channel declaration into the runtime's gateway configuration.

This module is the whole of NOVA's runtime-specific channel knowledge, and it is small on
purpose. The audit's finding was that the runtime already routes a conversation to a profile,
already scopes credentials per profile, already verifies webhooks and already has a durable
outbound ledger. None of that is reimplemented here. What happens here is a translation:

    NOVA                              runtime config.yaml
    ────                              ───────────────────
    channel connection        ->      gateway.platforms.<provider>.enabled
    route (conversation)      ->      gateway.profile_routes[].chat_id
    route (workspace)         ->      gateway.profile_routes[].guild_id
    route (thread)            ->      gateway.profile_routes[].thread_id
    route (agent)             ->      gateway.profile_routes[].profile
    allowed_agents (union)    ->      gateway.multiplex_profile_allowlist  (where supported)

**What actually enforces a revoked grant, stated in order of strength.**

*First, and always:* ``profile_routes`` is **replaced**, not merged. A channel removed from
the declaration loses its route, and an inbound conversation with no route does not reach
the agent. This is NOVA writing the runtime's own routing table, and it holds on every
runtime version.

*Second, where the runtime still offers it:* the served-profile allowlist. The runtime
rejects a route whose target profile is not served (``gateway/run.py``: *"Rejecting profile
route %r: target profile %r is not served"*), and that check is still there — but **what
counts as "served" is no longer NOVA's to set on every version.** Newer Hermes computes it
in one chokepoint, ``hermes_cli.profiles.profiles_to_serve``, as *default plus every live
named profile directory*; ``gateway.multiplex_profile_allowlist`` was removed from
``GatewayConfig`` and is now inert. Since a NOVA agent id *is* a Hermes profile name and
NOVA materialises one profile per agent, on those versions every agent is served and the
allowlist narrows nothing.

So this module **probes** (:func:`supports_serve_allowlist`) rather than assuming. Where the
key exists it is written and the backstop is real; where it does not, the key is omitted —
writing an ignored key would leave a reader believing in a control that is not running — and
the plan carries a warning saying so. NOVA does not claim the second tier on a runtime that
no longer implements it.

**The write is a merge, not a replacement.** The gateway's ``config.yaml`` at the home root
is operator territory: it carries their bind addresses, their session limits, their
quick-commands. NOVA replaces exactly the keys it owns and leaves every other byte alone,
because an operator who loses a setting to an apply will not trust the next one.

**NOVA never writes a credential here.** Not a token, not an api_key. The provider's adapter
reads fixed variable names and the values live in the agent's own ``.env``, which is on
``materialize.NEVER_WRITE``. This module refuses ``token`` and ``api_key`` explicitly, so a
future settings passthrough cannot reintroduce them by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from nova.channels.spec import ChannelSpec
from nova.errors import RuntimeAdapterError

#: Keys NOVA compiles itself. A settings passthrough that set one of these could silently
#: undo the grant or hand the runtime a literal secret.
REFUSED_SETTINGS = frozenset({"token", "api_key", "enabled", "profile_routes"})

#: The config keys this module owns. Everything outside this set is preserved byte-for-byte
#: on a merge, and this tuple is what the test asserts against so the promise cannot drift.
OWNED_GATEWAY_KEYS = (
    "platforms",
    "profile_routes",
    "multiplex_profiles",
    "multiplex_profile_allowlist",
)

#: NOVA provider id -> the runtime's platform name. Equal for most; WhatsApp is the case
#: that needs it, because the runtime ships two WhatsApp transports and NOVA offers the
#: Cloud API one by name.
PLATFORM_NAMES: Mapping[str, str] = {
    "telegram": "telegram",
    "slack": "slack",
    "discord": "discord",
    "whatsapp": "whatsapp_cloud",
    "email": "email",
    "web": "api_server",
}


@dataclass(frozen=True)
class ChannelPlan:
    """What applying a channel declaration would change, before anything is written."""

    platforms: Mapping[str, Any] = field(default_factory=dict)
    routes: tuple[Mapping[str, Any], ...] = ()
    served_agents: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "platforms": dict(self.platforms),
            "routes": [dict(r) for r in self.routes],
            "served_agents": list(self.served_agents),
            "warnings": list(self.warnings),
        }


def platform_name(provider_id: str) -> str:
    try:
        return PLATFORM_NAMES[provider_id]
    except KeyError:
        raise RuntimeAdapterError(
            f"this runtime adapter has no platform for provider {provider_id!r}"
        ) from None


def plan(
    channels: Sequence[ChannelSpec], derivations: Sequence[Any] = ()
) -> ChannelPlan:
    """Translate a declaration into runtime configuration, without writing anything.

    ``derivations`` are the channel-scoped agent variants (see ``nova/channels/derive.py``).
    A route to an agent that has a variant on this channel is pointed at the variant, which
    is how a per-channel approval requirement reaches an enforcement hook that is never told
    which channel it is serving.
    """
    from nova.channels.derive import route_target
    platforms: dict[str, Any] = {}
    routes: list[dict[str, Any]] = []
    served: list[str] = []
    warnings: list[str] = []

    for channel in channels:
        name = platform_name(channel.provider)
        refused = sorted(set(channel.settings) & REFUSED_SETTINGS)
        if refused:
            raise RuntimeAdapterError(
                f"channel {channel.id!r} sets {', '.join(refused)} in settings — NOVA "
                f"compiles those, and a credential never belongs in a declared file"
            )

        if not channel.enabled:
            warnings.append(f"{channel.id}: disabled in its declaration; no routes compiled")
            # Still surfaced as a platform key so the operator sees it was considered, but
            # left disabled rather than dropped: a silently absent channel reads as a bug.
            platforms.setdefault(name, {"enabled": False})
            continue

        existing = platforms.get(name, {})
        platforms[name] = {**existing, **dict(channel.settings), "enabled": True}
        # The grant, as the runtime will enforce it: where a variant exists it is the thing
        # that must be served, and the base agent is not reachable over this channel at all.
        served.extend(
            route_target(derivations, channel.id, agent) for agent in channel.allowed_agents
        )

        for index, route in enumerate(channel.routes):
            entry: dict[str, Any] = {
                "name": route.name or f"{channel.id}:{index}",
                "platform": name,
                "profile": route_target(derivations, channel.id, route.agent),
            }
            # Quoted as strings deliberately: the runtime warns that an unquoted numeric id
            # loaded as an int "can never match an inbound id".
            if route.conversation:
                entry["chat_id"] = str(route.conversation)
            if route.workspace:
                entry["guild_id"] = str(route.workspace)
            if route.thread:
                entry["thread_id"] = str(route.thread)
            routes.append(entry)

        if channel.enabled and not channel.routes:
            warnings.append(
                f"{channel.id}: connected but has no routes, so inbound messages fall to the "
                f"runtime's default profile rather than a granted agent"
            )

    # Most specific first, matching the runtime's own ordering, so what an operator reads in
    # the plan is the order that will actually apply.
    routes.sort(
        key=lambda r: 8 * ("thread_id" in r) + 4 * ("chat_id" in r) + 2 * ("guild_id" in r),
        reverse=True,
    )
    return ChannelPlan(
        platforms=platforms,
        routes=tuple(routes),
        served_agents=tuple(sorted(dict.fromkeys(served))),
        warnings=tuple(warnings),
    )


def supports_serve_allowlist() -> bool:
    """Whether this runtime still lets configuration narrow the served-profile set.

    Probed from the runtime's own config object rather than from a version number: a
    downstream fork that keeps the field keeps the control, and a version check would
    mis-answer for both. Import failure counts as unsupported — the honest direction, since
    the alternative is claiming an enforcement path that could not even be loaded.
    """
    try:
        from gateway.config import GatewayConfig
    except Exception:  # pragma: no cover — runtime absent or unimportable
        return False
    fields = getattr(GatewayConfig, "__dataclass_fields__", None)
    if fields is not None:
        return "multiplex_profile_allowlist" in fields
    return hasattr(GatewayConfig, "multiplex_profile_allowlist")


def merge_into(
    existing: Optional[Mapping[str, Any]],
    plan_: ChannelPlan,
    *,
    serve_allowlist: Optional[bool] = None,
) -> dict[str, Any]:
    """The new config document: NOVA's keys replaced, everything else untouched.

    Replacement rather than a deep merge for the keys NOVA owns. A route removed from the
    declaration must disappear from the runtime, and a merge that only ever adds would leave
    a revoked channel delivering — which is the one outcome this layer must never produce.

    ``serve_allowlist`` says whether to write ``multiplex_profile_allowlist``. ``None``
    probes the runtime. Pass it explicitly to compile for a runtime other than the one this
    process imports.
    """
    document = {k: v for k, v in (existing or {}).items()}
    gateway = dict(document.get("gateway") or {})

    platforms = dict(gateway.get("platforms") or {})
    for name, settings in plan_.platforms.items():
        platforms[name] = {**dict(platforms.get(name) or {}), **dict(settings)}
        platforms[name].pop("token", None)
        platforms[name].pop("api_key", None)
    if platforms:
        gateway["platforms"] = platforms

    if serve_allowlist is None:
        serve_allowlist = supports_serve_allowlist()

    gateway["profile_routes"] = [dict(route) for route in plan_.routes]
    if plan_.served_agents:
        gateway["multiplex_profiles"] = True
        if serve_allowlist:
            gateway["multiplex_profile_allowlist"] = list(plan_.served_agents)
        else:
            # Removed upstream and now inert. Leaving a stale one behind would be worse than
            # omitting it: a reader would take it for a live restriction.
            gateway.pop("multiplex_profile_allowlist", None)
    else:
        # No channels: hand multiplexing back rather than leaving a stale allowlist that
        # would keep serving agents no connection grants.
        gateway.pop("multiplex_profile_allowlist", None)
        gateway["profile_routes"] = []

    document["gateway"] = gateway
    return document


def apply(
    home: Path,
    channels: Sequence[ChannelSpec],
    *,
    derivations: Sequence[Any] = (),
    dry_run: bool = False,
) -> ChannelPlan:
    """Write the gateway configuration for a channel declaration. Returns the plan."""
    from nova.runtime.hermes import materialize as _materialize

    plan_ = plan(channels, derivations)
    if dry_run:
        return plan_

    import yaml

    path = Path(home) / "config.yaml"
    existing: Mapping[str, Any] = {}
    if path.is_file():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeAdapterError(
                f"{path} could not be read as YAML, so NOVA will not overwrite it: {exc}"
            ) from exc
        if loaded is not None and not isinstance(loaded, Mapping):
            raise RuntimeAdapterError(f"{path} is not a YAML mapping; refusing to overwrite it")
        existing = loaded or {}

    serve_allowlist = supports_serve_allowlist()
    if plan_.served_agents and not serve_allowlist:
        plan_ = replace(
            plan_,
            warnings=plan_.warnings
            + (
                "this runtime computes its served-profile set from the profile directory "
                "(hermes_cli.profiles.profiles_to_serve) and no longer reads "
                "gateway.multiplex_profile_allowlist, so a channel grant is enforced by the "
                "routing table alone. Removing a connection still removes its route; it does "
                "not additionally un-serve the agent.",
            ),
        )
    document = merge_into(existing, plan_, serve_allowlist=serve_allowlist)
    _materialize.atomic_write(
        path, yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    )
    return plan_


def readiness(
    channels: Sequence[ChannelSpec], *, home: Path, derivations: Sequence[Any] = ()
) -> list[dict[str, Any]]:
    """Which credential variables each connection still needs, per granted agent.

    Reported rather than resolved: NOVA reads variable *names* from the agent's ``.env`` and
    never their values, for the same reason it never writes them. A connection whose
    credential is missing is not broken configuration — it is configuration waiting for the
    operator step NOVA deliberately cannot take.

    Checked against the profile that will actually run. Where a channel tightened approval
    the conversation reaches a variant, and the variant's ``.env`` is the one the adapter
    reads — reporting the base agent's would tell an operator the credential was in place
    while every message failed.
    """
    from nova._env import read_env_file
    from nova.channels.derive import route_target

    rows: list[dict[str, Any]] = []
    for channel in channels:
        required = list(channel.required_env)
        per_agent: dict[str, list[str]] = {}
        for agent in channel.allowed_agents:
            profile = route_target(derivations, channel.id, agent)
            present = read_env_file(Path(home) / "profiles" / profile / ".env")
            per_agent[profile] = [name for name in required if name not in present]
        rows.append(
            {
                "id": channel.id,
                "provider": channel.provider,
                "required_env": required,
                "missing_by_agent": per_agent,
                "ready": all(not missing for missing in per_agent.values()),
            }
        )
    return rows
