"""What NOVA knows about each channel provider, and how sure it is.

Every entry here was read out of the runtime's own plugin manifests and adapters during the
Phase 9 audit (``docs/NOVA_CHANNEL_AUDIT.md``). None of it was invented, and none of it is a
marketing claim: the :class:`Verification` on each provider says how the entry was
established, and the honest answer today is "we read the source" for all of them.

That field exists because the failure mode here is specific and expensive. A channel list is
the most quotable page in a product — a customer reads "WhatsApp ✓" and signs a contract on
it. If the tick means "a plugin directory exists", the first real deployment discovers what
the tick actually meant. So the catalogue carries its own evidence level, the dashboard shows
it, and nothing is promoted to ``FIELD_VALIDATED`` without a real provider connection.

**NOVA does not choose credential variable names.** The runtime's adapters read fixed names
(``TELEGRAM_BOT_TOKEN``), so NOVA reports *which names a provider needs* and never invents an
indirection the adapter would not honour. The values live in the agent's own ``.env``, which
NOVA is forbidden from writing — the same rule the model-provider seam runs on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Optional


class Verification(str, Enum):
    """How a claim in this catalogue was established. Ordered weakest to strongest."""

    #: The runtime bundles a manifest for it. Nobody has read the adapter, let alone run
    #: it. Weakest rung, and the right one for a provider discovered but never opened.
    DECLARED = "declared"
    #: Read in the runtime's source. Nothing was connected.
    SOURCE_READ = "source_read"
    #: Exercised end to end against a scripted or local stand-in for the provider.
    LOCAL_VALIDATED = "local_validated"
    #: Exercised against the real provider with real credentials.
    FIELD_VALIDATED = "field_validated"


class Transport(str, Enum):
    """How inbound messages reach the runtime. Decides what a deployment must expose."""

    #: The runtime polls the provider. Needs outbound network only.
    POLL = "poll"
    #: A persistent socket the runtime opens. Outbound only.
    SOCKET = "socket"
    #: The provider calls in. **Needs a publicly reachable HTTPS endpoint**, which is a
    #: deployment decision with a security boundary, not a checkbox.
    WEBHOOK = "webhook"
    #: A separate long-running bridge process beside the runtime.
    BRIDGE = "bridge"
    #: Not stated. A ``plugin.yaml`` does not declare how its adapter connects, and
    #: inferring it from the adapter's source is a claim, not a reading. Rendered as
    #: unknown rather than guessed.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Capability:
    """One provider capability and the evidence behind it.

    ``supported=None`` means unknown — which is a different and more useful statement than
    ``False``, and the audit's rule was that unknown is never rendered as yes.
    """

    supported: Optional[bool]
    verification: Verification = Verification.SOURCE_READ
    note: str = ""

    def to_dict(self) -> dict:
        out: dict = {"supported": self.supported, "verification": self.verification.value}
        if self.note:
            out["note"] = self.note
        return out


UNKNOWN = Capability(supported=None)


def _known(note: str = "") -> Capability:
    return Capability(supported=True, note=note)


@dataclass(frozen=True)
class Provider:
    """A channel NOVA can offer, described in the customer's vocabulary."""

    id: str
    label: str
    transport: Transport
    #: Environment variable names the runtime's adapter reads. NOVA reports these and writes
    #: none of their values, ever.
    required_env: tuple[str, ...] = ()
    optional_env: tuple[str, ...] = ()
    #: Per-capability evidence. Absent key means the question was never asked.
    capabilities: Mapping[str, Capability] = field(default_factory=dict)
    #: Overall evidence level for "this provider works at all".
    verification: Verification = Verification.SOURCE_READ
    #: Where the implementation lives, so a reviewer can check this entry themselves.
    implementation: str = ""
    #: Stated plainly on the connect screen rather than discovered during a deployment.
    caveat: str = ""
    #: The manifest's own prose. Empty for an annotation-only provider.
    description: str = ""
    #: Full credential metadata from the manifest: name, description, prompt, url, secret,
    #: required. This is what a connect form renders — and ``secret`` is what it masks.
    #: NOVA reports these names and writes none of their values, ever.
    credentials: tuple[dict, ...] = ()

    @property
    def needs_public_endpoint(self) -> bool:
        return self.transport is Transport.WEBHOOK

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "transport": self.transport.value,
            "required_env": list(self.required_env),
            "optional_env": list(self.optional_env),
            "capabilities": {k: v.to_dict() for k, v in sorted(self.capabilities.items())},
            "verification": self.verification.value,
            "implementation": self.implementation,
            "caveat": self.caveat,
            "description": self.description,
            "credentials": [dict(c) for c in self.credentials],
            "needs_public_endpoint": self.needs_public_endpoint,
        }


def _caps(**kwargs: Capability) -> dict[str, Capability]:
    base = {"attachments": UNKNOWN, "groups": UNKNOWN, "threads": UNKNOWN}
    base.update(kwargs)
    return base


#: Hand-written **annotations**, not the catalogue.
#:
#: These six were opened, driven and written up, so each carries transport, capability
#: notes and caveats that no manifest states. They are layered onto whatever the runtime
#: reports (see :func:`catalogue`) rather than being the list itself — a hand-kept list
#: beside a runtime that bundles twenty-two adapters drifts, and did: the Control Centre
#: offered Telegram and nothing else while the runtime could reach twenty-two platforms.
ANNOTATIONS: tuple[Provider, ...] = (
    Provider(
        id="telegram",
        label="Telegram",
        transport=Transport.POLL,
        required_env=("TELEGRAM_BOT_TOKEN",),
        optional_env=("TELEGRAM_ALLOWED_USERS", "TELEGRAM_HOME_CHANNEL"),
        capabilities=_caps(
            groups=_known("group chats and forum topics are first-class in routing"),
            threads=_known("topics carried as thread_id"),
        ),
        implementation="plugins/platforms/telegram/",
    ),
    Provider(
        id="slack",
        label="Slack",
        transport=Transport.SOCKET,
        required_env=("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"),
        optional_env=("SLACK_ALLOWED_USERS", "SLACK_HOME_CHANNEL"),
        capabilities=_caps(
            groups=_known("channels"),
            threads=_known("native Slack threads"),
        ),
        implementation="plugins/platforms/slack/",
        caveat="Socket Mode: needs an app-level token with connections:write.",
    ),
    Provider(
        id="discord",
        label="Discord",
        transport=Transport.SOCKET,
        required_env=("DISCORD_BOT_TOKEN",),
        optional_env=("DISCORD_ALLOWED_USERS", "DISCORD_HOME_CHANNEL"),
        capabilities=_caps(
            groups=_known("guilds and channels"),
            threads=_known("threads and forum posts, via parent_chat_id"),
        ),
        implementation="plugins/platforms/discord/",
    ),
    Provider(
        id="whatsapp",
        label="WhatsApp (Business Cloud API)",
        transport=Transport.WEBHOOK,
        required_env=("WHATSAPP_CLOUD_TOKEN", "WHATSAPP_CLOUD_PHONE_NUMBER_ID"),
        optional_env=("WHATSAPP_CLOUD_APP_SECRET", "WHATSAPP_CLOUD_VERIFY_TOKEN"),
        capabilities=_caps(groups=UNKNOWN, threads=UNKNOWN),
        implementation="gateway/platforms/whatsapp_cloud.py",
        caveat=(
            "Meta enforces a 24-hour customer service window: outside it only approved "
            "template messages send. Needs a publicly reachable HTTPS webhook."
        ),
    ),
    Provider(
        id="email",
        label="Email",
        transport=Transport.POLL,
        required_env=("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST"),
        optional_env=("EMAIL_IMAP_PORT", "EMAIL_SMTP_PORT"),
        capabilities=_caps(threads=_known("reply threading by subject/references")),
        implementation="plugins/platforms/email/",
        caveat="IMAP polling, so inbound latency is the poll interval, not instant.",
    ),
    Provider(
        id="web",
        label="Web chat",
        transport=Transport.SOCKET,
        required_env=("API_SERVER_KEY",),
        capabilities=_caps(groups=Capability(supported=False, note="one conversation per caller")),
        implementation="gateway/platforms/api_server.py",
    ),
)

ANNOTATIONS_BY_ID: Mapping[str, Provider] = {p.id: p for p in ANNOTATIONS}

#: Kept as a name because callers import it. It is the annotated subset, and any caller
#: that wants "everything this deployment can connect" should call :func:`catalogue`.
PROVIDERS: tuple[Provider, ...] = ANNOTATIONS
PROVIDERS_BY_ID: Mapping[str, Provider] = ANNOTATIONS_BY_ID


#: Set by the runtime adapter at import. Returns manifest rows; see
#: ``nova/runtime/hermes/catalogue.py``. Injected rather than imported because this module
#: may not name the runtime — ``tests/platform/test_boundaries.py`` holds that line.
_discovery: Optional[Callable[[], tuple[dict, ...]]] = None
_cache: Optional[tuple[Provider, ...]] = None


def set_discovery(fn: Optional[Callable[[], tuple[dict, ...]]]) -> None:
    """Install the runtime's channel discovery, clearing any cached catalogue."""
    global _discovery, _cache
    _discovery, _cache = fn, None


def _from_manifest(row: dict) -> Provider:
    """One discovered platform, with an annotation layered on where one exists.

    The manifest wins on identity and credentials — it is what the adapter actually reads.
    The annotation wins on transport, capabilities and caveats, because a manifest states
    none of those and a guess would be exactly the invented claim this catalogue exists to
    avoid. Undiscovered-by-annotation providers are therefore ``UNKNOWN`` across the board,
    which the dashboard already knows how to render.
    """
    note = ANNOTATIONS_BY_ID.get(row["id"])
    required = tuple(entry["name"] for entry in row.get("required_env", ()))
    optional = tuple(entry["name"] for entry in row.get("optional_env", ()))
    return Provider(
        id=row["id"],
        label=row.get("label") or (note.label if note else row["id"]),
        transport=note.transport if note else Transport.UNKNOWN,
        required_env=required or (note.required_env if note else ()),
        optional_env=optional or (note.optional_env if note else ()),
        capabilities=note.capabilities if note else _caps(),
        verification=note.verification if note else Verification.DECLARED,
        implementation=row.get("implementation", "") or (note.implementation if note else ""),
        caveat=note.caveat if note else "",
        description=row.get("description", ""),
        credentials=tuple(row.get("required_env", ())) + tuple(row.get("optional_env", ())),
    )


def catalogue() -> tuple[Provider, ...]:
    """Every provider this deployment can actually connect.

    The runtime's bundled platform plugins when a runtime is present, falling back to the
    annotated set when NOVA is used as a library with no adapter imported. Cached, because
    it reads a directory of manifests and the answer cannot change while the process runs.
    """
    global _cache
    if _cache is not None:
        return _cache
    if _discovery is None:
        _cache = ANNOTATIONS
        return _cache
    try:
        rows = _discovery() or ()
    except Exception:
        # Discovery is best-effort: a runtime that cannot be read must not take the
        # control plane down, and the annotated set is still true.
        _cache = ANNOTATIONS
        return _cache
    discovered = tuple(_from_manifest(row) for row in rows)
    if not discovered:
        _cache = ANNOTATIONS
        return _cache
    # Annotated providers the runtime does not bundle are dropped, not kept: the runtime
    # decides what can connect, and offering something it cannot load is the failure mode
    # this whole module exists to remove.
    _cache = tuple(sorted(discovered, key=lambda p: p.label.lower()))
    return _cache


def catalogue_by_id() -> Mapping[str, Provider]:
    return {p.id: p for p in catalogue()}


def get_provider(provider_id: str) -> Provider:
    """Look up a provider, or refuse with the list of ones that exist."""
    from nova.errors import SpecError

    known = catalogue_by_id()
    try:
        return known[provider_id]
    except KeyError:
        listed = ", ".join(sorted(known))
        source = (
            "the runtime's bundled platform plugins"
            if _discovery is not None
            else "NOVA's annotated set (no runtime adapter is loaded, so the full list "
                 "could not be read)"
        )
        raise SpecError(
            f"{provider_id!r} is not a channel this deployment can connect. Available, "
            f"from {source}: {listed}"
        ) from None
