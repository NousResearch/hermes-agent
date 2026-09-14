"""What a customer declares when they connect a channel.

The audit found the runtime already routes a chat to a profile. What it does not have — and
what a commercial product cannot ship without — is a statement that *this connection may only
reach these agents*. A route says where a message goes; it never says where a message is
allowed to go, and those become different the moment somebody edits a route.

So the grant is the centre of this module. ``allowed_agents`` is declared per connection,
every route is checked against it, and a route naming an agent outside the grant is refused
at parse time rather than compiled into infrastructure and discovered later.

**No secret is ever expressible here.** A channel declaration carries provider ids, chat ids,
agent names and display text. The credential variable names come from the provider catalogue,
which NOVA does not let an author override — an author who could name the variable could name
it something the adapter never reads, and the failure would look like a broken channel rather
than a misconfiguration. Values live in the agent's own ``.env``, which NOVA never writes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from nova._fields import Doc
from nova.channels.providers import Provider, get_provider
from nova.errors import SpecError

#: The file a tenant bundle declares its channels in.
CHANNELS_FILE = "channels.yaml"

#: A connection id: a name in a URL, a log field and an audit subject.
CONNECTION_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,46}[a-z0-9]$")

#: Keys that would carry a credential value. Refused outright: a token in this file is in
#: version control, in every clone, and in the history after it is removed. Refusing at parse
#: time is the only moment where saying no still helps.
FORBIDDEN_KEYS = frozenset({"token", "api_key", "secret", "password", "credential", "app_secret"})


@dataclass(frozen=True)
class ChannelRoute:
    """One inbound routing rule: which conversation reaches which agent.

    The discriminators are the runtime's own — a chat, a group, a thread — because inventing
    a different vocabulary would mean translating a customer's Slack channel id into
    something else and back, and the translation is where the bugs live.
    """

    agent: str
    #: A specific conversation: a Slack channel, a Telegram chat, an email address.
    conversation: str = ""
    #: The workspace/guild/server the conversation lives in, where the provider has one.
    workspace: str = ""
    #: A thread within the conversation.
    thread: str = ""
    name: str = ""

    @property
    def specificity(self) -> int:
        """Higher is more specific. Mirrors the runtime's own ordering so that what a
        customer sees in the dashboard is the order the runtime will actually apply."""
        return 4 * bool(self.conversation) + 2 * bool(self.workspace) + 8 * bool(self.thread)

    @property
    def is_catch_all(self) -> bool:
        return not (self.conversation or self.workspace or self.thread)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"agent": self.agent}
        for key, value in (
            ("name", self.name), ("conversation", self.conversation),
            ("workspace", self.workspace), ("thread", self.thread),
        ):
            if value:
                out[key] = value
        return out


@dataclass(frozen=True)
class ChannelApproval:
    """Business actions this channel escalates to a human, on top of the agent's own.

    Additive only, and deliberately so. A channel cannot *remove* an approval the tenant
    policy or the agent already requires: a permission that could be widened by connecting
    a messaging app would make the channel list part of the security review, and the whole
    point of declaring approvals in the policy is that they hold everywhere.
    """

    required_for: tuple[str, ...] = ()

    @property
    def declared(self) -> bool:
        return bool(self.required_for)

    def to_dict(self) -> dict[str, Any]:
        return {"required_for": list(self.required_for)} if self.required_for else {}


@dataclass(frozen=True)
class ChannelSpec:
    """One connected channel, as the customer declared it."""

    id: str
    provider: str
    display_name: str = ""
    enabled: bool = True
    #: **The grant.** Agents this connection may reach — nothing else, whatever a route says.
    allowed_agents: tuple[str, ...] = ()
    routes: tuple[ChannelRoute, ...] = ()
    #: Extra human-approval requirements for anything reached over this channel.
    approval: ChannelApproval = field(default_factory=ChannelApproval)
    #: Non-secret provider settings passed through to the runtime's own platform config.
    #: Guarded by :data:`FORBIDDEN_KEYS`.
    settings: Mapping[str, Any] = field(default_factory=dict)
    source: Optional[Path] = None

    @property
    def catalogue(self) -> Provider:
        return get_provider(self.provider)

    @property
    def required_env(self) -> tuple[str, ...]:
        return self.catalogue.required_env

    def agents_reached(self) -> tuple[str, ...]:
        """Every agent any route on this connection can deliver to."""
        return tuple(dict.fromkeys(route.agent for route in self.routes))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "provider": self.provider,
            "enabled": self.enabled,
            "allowed_agents": list(self.allowed_agents),
            "routes": [route.to_dict() for route in self.routes],
        }
        if self.display_name:
            out["display_name"] = self.display_name
        if self.approval.declared:
            out["approval"] = self.approval.to_dict()
        if self.settings:
            out["settings"] = dict(self.settings)
        return out


def _check_no_secrets(data: Any, *, where: str, source: Optional[Path]) -> None:
    """Refuse anything that looks like it carries a credential value."""
    if isinstance(data, Mapping):
        for key, value in data.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in FORBIDDEN_KEYS):
                raise SpecError(
                    f"{key!r} would carry a credential. A channel declaration names agents "
                    f"and conversations, never secrets: the provider's credential variables "
                    f"are fixed by its adapter and their values belong in the agent's .env, "
                    f"which NOVA never writes",
                    field=where,
                    source=source,
                )
            _check_no_secrets(value, where=f"{where}.{key}", source=source)
    elif isinstance(data, (list, tuple)):
        for index, item in enumerate(data):
            _check_no_secrets(item, where=f"{where}[{index}]", source=source)


def _conversation_id(raw: Any, *, field_path: str, source: Optional[Path]) -> str:
    """Normalise a conversation / workspace / thread id to a string.

    A Telegram group id is ``-1001234567890`` and YAML loads that unquoted as an ``int``, so
    refusing non-strings here would reject the most natural thing a customer can write. The
    runtime hit this exact problem first — ``gateway/profile_routing.py::_coerce_route_id``
    exists because an unquoted numeric id "can never match an inbound id" — and reached the
    same answer: coerce integers, refuse everything else rather than stringify it into
    something (``"123.0"``) that silently never matches.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, int) and not isinstance(raw, bool):
        return str(raw)
    raise SpecError(
        f"{raw!r} is not a conversation id. Use a string (quote it in YAML if it is "
        f"numeric); a {type(raw).__name__} would be turned into text that can never match "
        f"an inbound message",
        field=field_path,
        source=source,
    )


def _parse_route(doc: Doc, *, prefix: str, source: Optional[Path]) -> ChannelRoute:
    agent = doc.str_("agent")
    if not agent:
        raise SpecError(
            "needs an agent — a route with no destination cannot be compiled, and the only "
            "thing it could mean is 'anywhere'",
            field=f"{prefix}.agent",
            source=source,
        )
    route = ChannelRoute(
        agent=agent,
        conversation=_conversation_id(
            doc._raw("conversation"), field_path=f"{prefix}.conversation", source=source
        ),
        workspace=_conversation_id(
            doc._raw("workspace"), field_path=f"{prefix}.workspace", source=source
        ),
        thread=_conversation_id(
            doc._raw("thread"), field_path=f"{prefix}.thread", source=source
        ),
        name=doc.str_("name"),
    )
    doc.reject_unknown()
    if route.thread and not route.conversation:
        raise SpecError(
            "names a thread without the conversation it lives in, which can never match: "
            "the runtime matches a thread inside its parent conversation",
            field=f"{prefix}.thread",
            source=source,
        )
    return route


def parse_channels(
    data: Any,
    *,
    source: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
) -> tuple[ChannelSpec, ...]:
    """Parse and fully validate a channel declaration."""
    if data is None:
        return ()
    if isinstance(data, Mapping):
        data = data.get("channels")
    if data is None:
        return ()
    if not isinstance(data, (list, tuple)):
        raise SpecError("must be a list", field="channels", source=source)

    _check_no_secrets(data, where="channels", source=source)

    out: list[ChannelSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(data):
        prefix = f"channels[{index}]"
        doc = Doc(entry or {}, source=source, prefix=prefix, env=env)

        connection_id = doc.str_("id")
        if not CONNECTION_ID.match(connection_id or ""):
            raise SpecError(
                f"{connection_id!r} is not a connection id: lowercase letters, digits and "
                f"hyphens, 3-48 characters. It appears in URLs, logs and the audit trail",
                field=f"{prefix}.id",
                source=source,
            )
        if connection_id in seen:
            raise SpecError(
                f"{connection_id!r} is declared twice. Two connections with one id would "
                f"compile to one, and the second set of routes would vanish silently",
                field=f"{prefix}.id",
                source=source,
            )
        seen.add(connection_id)

        provider_id = doc.str_("provider")
        if not provider_id:
            raise SpecError("needs a provider", field=f"{prefix}.provider", source=source)
        get_provider(provider_id)  # refuses with the list of what exists

        allowed = tuple(doc.str_list("allowed_agents", unique=True))
        if not allowed:
            raise SpecError(
                "needs allowed_agents. A connection that grants nothing is not a safe "
                "default to infer in either direction: empty could mean 'no agent' or "
                "'every agent', and one of those is a channel into the whole workforce",
                field=f"{prefix}.allowed_agents",
                source=source,
            )

        raw_routes = doc._raw("routes") or []
        if not isinstance(raw_routes, (list, tuple)):
            raise SpecError("must be a list", field=f"{prefix}.routes", source=source)
        routes = tuple(
            _parse_route(
                Doc(item or {}, source=source, prefix=f"{prefix}.routes[{n}]", env=env),
                prefix=f"{prefix}.routes[{n}]",
                source=source,
            )
            for n, item in enumerate(raw_routes)
        )

        settings = doc._raw("settings") or {}
        if settings and not isinstance(settings, Mapping):
            raise SpecError("must be a mapping", field=f"{prefix}.settings", source=source)

        approval_doc = doc.child("approval")
        approval = ChannelApproval()
        if approval_doc is not None:
            approval = ChannelApproval(
                required_for=tuple(approval_doc.str_list("required_for", unique=True))
            )
            approval_doc.reject_unknown()

        spec = ChannelSpec(
            id=connection_id,
            provider=provider_id,
            display_name=doc.str_("display_name"),
            enabled=doc.bool_("enabled", default=True),
            allowed_agents=allowed,
            routes=routes,
            approval=approval,
            settings=dict(settings),
            source=source,
        )
        doc.reject_unknown()

        # The grant, enforced. A route outside allowed_agents is refused here rather than
        # compiled into a routing table where it would work perfectly.
        outside = sorted(set(spec.agents_reached()) - set(allowed))
        if outside:
            raise SpecError(
                f"routes to {', '.join(outside)}, which {connection_id!r} does not grant. "
                f"Either add them to allowed_agents deliberately, or fix the route — a "
                f"connection reaching an agent nobody authorised is the failure this "
                f"declaration exists to prevent",
                field=f"{prefix}.routes",
                source=source,
            )

        catch_alls = [r for r in spec.routes if r.is_catch_all]
        if len(catch_alls) > 1:
            raise SpecError(
                "has more than one catch-all route. Only the first could ever match, so the "
                "others are silently dead",
                field=f"{prefix}.routes",
                source=source,
            )
        out.append(spec)
    return tuple(out)


def check_agents_exist(
    channels: Sequence[ChannelSpec], known_agents: Sequence[str]
) -> None:
    """Refuse a channel naming an agent this bundle does not contain.

    Almost always a rename that left a grant behind. A grant with nobody to use it survives
    review by looking like something somebody meant.
    """
    known = set(known_agents)
    for channel in channels:
        unknown = sorted(set(channel.allowed_agents) - known)
        if unknown:
            raise SpecError(
                f"grants {', '.join(unknown)}, which this bundle does not contain. Either "
                f"the agent was renamed and this grant was left behind, or it does not exist "
                f"yet — both are worth fixing before a customer message arrives",
                field=f"channels[{channel.id}].allowed_agents",
                source=channel.source,
            )


def load_channels(
    bundle_root: Path, *, env: Optional[Mapping[str, str]] = None
) -> tuple[ChannelSpec, ...]:
    """Read ``channels.yaml``. Absent means no channels, which is the default and safe one."""
    import yaml

    path = bundle_root / CHANNELS_FILE
    if not path.is_file():
        return ()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpecError(f"could not be read: {exc.strerror or exc}", source=path) from exc
    except yaml.YAMLError as exc:
        raise SpecError(f"is not valid YAML: {exc}", source=path) from exc
    return parse_channels(data or {}, source=path, env=env)
