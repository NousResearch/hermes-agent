"""Declarative configuration schema for desktop memory providers.

Each memory provider *declares* its configurable surface here — the fields, their
types, which values are secrets, and (for selects) the allowed options. A single
generic renderer in the desktop UI and a single generic ``GET/PUT
/api/memory/providers/{name}/config`` endpoint pair drive the whole experience,
so adding a new provider is pure declaration with no bespoke UI components.

This module is intentionally pure data: it imports nothing from the config/env
layer. ``web_server`` owns the generic read/write logic that interprets these
declarations, dispatching on ``MemoryProvider.storage`` to the matching backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

# Field kinds understood by the generic renderer.
KIND_TEXT = "text"
KIND_SELECT = "select"
KIND_SECRET = "secret"
KIND_BOOL = "bool"
KIND_NUMBER = "number"
KIND_JSON = "json"

# Storage backends understood by web_server. ``flat_json`` persists non-secret
# fields to ``<hermes_home>/<provider>/config.json``; ``honcho_host_block``
# targets Honcho's real config (honcho.json, scoped to the active profile host).
STORAGE_FLAT_JSON = "flat_json"
STORAGE_HONCHO_HOST_BLOCK = "honcho_host_block"


@dataclass(frozen=True)
class ProviderFieldOption:
    """A single choice for a ``select`` field."""

    value: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class ProviderField:
    """One configurable field on a memory provider.

    A field is stored in exactly one place, decided by ``kind``:

    * non-secret kinds — persisted to the provider's config via its storage
      backend under ``key``.
    * ``secret`` — persisted to the env store under ``env_key`` and never read
      back out over the API (only an ``is_set`` flag is surfaced).

    ``aliases`` and ``env_fallbacks`` let a field read legacy values written by
    earlier CLI/env setup without re-introducing per-provider code. ``inline``
    marks the curated subset shown in the compact panel; the rest surface only
    in the full-config modal. ``group`` buckets fields within that modal.
    """

    key: str
    label: str
    kind: str = KIND_TEXT
    default: str = ""
    description: str = ""
    placeholder: str = ""
    options: tuple[ProviderFieldOption, ...] = ()
    env_key: str | None = None
    aliases: tuple[str, ...] = ()
    env_fallbacks: tuple[str, ...] = ()
    inline: bool = False
    group: str = ""
    # Where a host-block backend stores the field: "host" (per-profile host
    # block) or "root" (config root). Ignored by the flat-json backend.
    scope: str = "host"

    @property
    def is_secret(self) -> bool:
        return self.kind == KIND_SECRET

    def allowed_values(self) -> set[str]:
        return {opt.value for opt in self.options}


@dataclass(frozen=True)
class MemoryProvider:
    """A declared memory provider and its configurable fields."""

    name: str
    label: str
    storage: str = STORAGE_FLAT_JSON
    # Optional link to the provider's config docs, shown in the full-config modal.
    docs_url: str = ""
    fields: tuple[ProviderField, ...] = dataclass_field(default_factory=tuple)

    def inline_fields(self) -> tuple[ProviderField, ...]:
        return tuple(f for f in self.fields if f.inline)


# Reasoning effort levels shared by dialectic-related selects.
_REASONING_LEVELS = (
    ProviderFieldOption("minimal", "Minimal"),
    ProviderFieldOption("low", "Low"),
    ProviderFieldOption("medium", "Medium"),
    ProviderFieldOption("high", "High"),
    ProviderFieldOption("max", "Max"),
)


HONCHO = MemoryProvider(
    name="honcho",
    label="Honcho",
    storage=STORAGE_HONCHO_HOST_BLOCK,
    docs_url="https://docs.honcho.dev/v3/guides/integrations/hermes",
    fields=(
        # — Connection —
        ProviderField(
            key="apiKey",
            label="API key",
            kind=KIND_SECRET,
            env_key="HONCHO_API_KEY",
            description="Authenticate with Honcho Cloud. Not needed for a self-hosted base URL.",
            placeholder="Enter Honcho API key",
            inline=True,
            group="Connection",
        ),
        ProviderField(
            key="baseUrl",
            label="Base URL",
            kind=KIND_TEXT,
            aliases=("base_url",),
            env_fallbacks=("HONCHO_BASE_URL",),
            description="Self-hosted Honcho URL. Overrides the environment when set.",
            placeholder="https://… (self-hosted)",
            inline=True,
            group="Connection",
            scope="root",
        ),
        ProviderField(
            key="environment",
            label="Environment",
            kind=KIND_SELECT,
            default="production",
            env_fallbacks=("HONCHO_ENVIRONMENT",),
            description="Honcho environment. Ignored when a base URL is set.",
            options=(
                ProviderFieldOption("production", "Cloud"),
                ProviderFieldOption("local", "Local"),
            ),
            inline=True,
            group="Connection",
        ),
        ProviderField(
            key="workspace",
            label="Workspace",
            kind=KIND_TEXT,
            description="Honcho workspace ID. Defaults to the profile host.",
            inline=True,
            group="Connection",
        ),
        # — Identity —
        ProviderField(
            key="peerName",
            label="Peer name",
            kind=KIND_TEXT,
            description="Your stable user peer. Unifies memory across platforms for single-user setups.",
            placeholder="e.g. eri",
            inline=True,
            group="Identity",
        ),
        ProviderField(
            key="aiPeer",
            label="AI peer",
            kind=KIND_TEXT,
            description="The AI-side peer name. Defaults to the profile host.",
            inline=True,
            group="Identity",
        ),
        # — Session —
        ProviderField(
            key="sessionStrategy",
            label="Session strategy",
            kind=KIND_SELECT,
            default="per-directory",
            description="How conversations map to Honcho sessions.",
            options=(
                ProviderFieldOption("per-directory", "Per directory"),
                ProviderFieldOption("per-repo", "Per repo"),
                ProviderFieldOption("per-session", "Per session"),
                ProviderFieldOption("global", "Global"),
            ),
            inline=True,
            group="Session",
        ),
        # —————— Full-config-only fields below (inline=False) ——————
        # — Connection —
        ProviderField(
            key="timeout",
            label="Request timeout",
            kind=KIND_NUMBER,
            aliases=("requestTimeout",),
            env_fallbacks=("HONCHO_TIMEOUT",),
            description="Request timeout in seconds for Honcho HTTP calls. Blank uses the default.",
            placeholder="30",
            group="Connection",
            scope="root",
        ),
        # — Identity —
        ProviderField(
            key="pinUserPeer",
            label="Pin user peer",
            kind=KIND_BOOL,
            default="false",
            aliases=("pinPeerName",),
            description="Pin the user peer to the peer name, ignoring gateway runtime identity. Unifies memory for single-user setups.",
            group="Identity",
        ),
        ProviderField(
            key="runtimePeerPrefix",
            label="Runtime peer prefix",
            kind=KIND_TEXT,
            description="Prefix applied to unknown gateway runtime user IDs.",
            placeholder="e.g. telegram_",
            group="Identity",
        ),
        ProviderField(
            key="userPeerAliases",
            label="User peer aliases",
            kind=KIND_JSON,
            description="Map gateway runtime user IDs to stable Honcho peers.",
            placeholder='{"telegram_123": "eri"}',
            group="Identity",
        ),
        # — Session —
        ProviderField(
            key="sessionPeerPrefix",
            label="Session peer prefix",
            kind=KIND_BOOL,
            default="false",
            description="Prefix session peer names with the host.",
            group="Session",
        ),
        ProviderField(
            key="sessions",
            label="Session overrides",
            kind=KIND_JSON,
            description="Explicit session ID overrides keyed by resolver.",
            placeholder='{"key": "session-id"}',
            group="Session",
            scope="root",
        ),
        # — Message writing —
        ProviderField(
            key="saveMessages",
            label="Save messages",
            kind=KIND_BOOL,
            default="true",
            description="Persist conversation messages to Honcho.",
            group="Message writing",
        ),
        ProviderField(
            key="writeFrequency",
            label="Write frequency",
            kind=KIND_TEXT,
            default="async",
            description="When to flush messages: async, turn, session, or every N turns.",
            placeholder="async | turn | session | N",
            group="Message writing",
        ),
        # — Dialectic —
        ProviderField(
            key="dialecticReasoningLevel",
            label="Reasoning level",
            kind=KIND_SELECT,
            default="low",
            description="Reasoning effort for dialectic (peer.chat) calls.",
            options=_REASONING_LEVELS,
            group="Dialectic",
        ),
        ProviderField(
            key="dialecticDynamic",
            label="Dynamic reasoning",
            kind=KIND_BOOL,
            default="true",
            description="Let the model override the reasoning level per call.",
            group="Dialectic",
        ),
        ProviderField(
            key="dialecticMaxChars",
            label="Max result chars",
            kind=KIND_NUMBER,
            description="Max chars of dialectic result injected into the system prompt.",
            placeholder="1200",
            group="Dialectic",
        ),
        ProviderField(
            key="dialecticDepth",
            label="Depth",
            kind=KIND_NUMBER,
            description="Dialectic passes per cycle (1–3).",
            placeholder="1",
            group="Dialectic",
        ),
        ProviderField(
            key="dialecticDepthLevels",
            label="Per-pass levels",
            kind=KIND_JSON,
            description="Reasoning level per pass; array length matches depth.",
            placeholder='["low", "medium"]',
            group="Dialectic",
        ),
        ProviderField(
            key="dialecticMaxInputChars",
            label="Max input chars",
            kind=KIND_NUMBER,
            description="Max chars of query input sent to peer.chat().",
            placeholder="10000",
            group="Dialectic",
        ),
        # — Reasoning —
        ProviderField(
            key="reasoningHeuristic",
            label="Reasoning heuristic",
            kind=KIND_BOOL,
            default="true",
            description="Scale the reasoning level up on longer queries.",
            group="Reasoning",
        ),
        ProviderField(
            key="reasoningLevelCap",
            label="Reasoning level cap",
            kind=KIND_SELECT,
            default="high",
            description="Ceiling for the heuristic-selected reasoning level.",
            options=_REASONING_LEVELS,
            group="Reasoning",
        ),
        # — Recall —
        ProviderField(
            key="recallMode",
            label="Recall mode",
            kind=KIND_SELECT,
            default="hybrid",
            description="How memory retrieval works: hybrid, context-only, or tools-only.",
            options=(
                ProviderFieldOption("hybrid", "Hybrid"),
                ProviderFieldOption("context", "Context only"),
                ProviderFieldOption("tools", "Tools only"),
            ),
            group="Recall",
        ),
        ProviderField(
            key="contextTokens",
            label="Context token cap",
            kind=KIND_NUMBER,
            description="Cap on auto-injected context tokens. Blank leaves it uncapped.",
            placeholder="(uncapped)",
            group="Recall",
        ),
        ProviderField(
            key="initOnSessionStart",
            label="Eager init",
            kind=KIND_BOOL,
            default="false",
            description="Initialize the session eagerly in tools mode instead of on first tool call.",
            group="Recall",
        ),
        # — Limits —
        ProviderField(
            key="messageMaxChars",
            label="Message max chars",
            kind=KIND_NUMBER,
            description="Max chars per message sent to Honcho.",
            placeholder="25000",
            group="Limits",
        ),
        # — Observation —
        ProviderField(
            key="observationMode",
            label="Observation mode",
            kind=KIND_SELECT,
            default="directional",
            description="Per-peer observation preset. Directional observes all directions; unified shares one view.",
            options=(
                ProviderFieldOption("directional", "Directional"),
                ProviderFieldOption("unified", "Unified"),
            ),
            group="Observation",
        ),
    ),
)


HINDSIGHT = MemoryProvider(
    name="hindsight",
    label="Hindsight",
    fields=(
        ProviderField(
            key="mode",
            label="Mode",
            kind=KIND_SELECT,
            default="cloud",
            description="How Hermes connects to Hindsight.",
            options=(
                ProviderFieldOption(
                    "cloud",
                    "Cloud",
                    "Hindsight Cloud API (lightweight, just needs an API key)",
                ),
                ProviderFieldOption(
                    "local_external",
                    "Local External",
                    "Connect to an existing Hindsight instance",
                ),
            ),
            inline=True,
        ),
        ProviderField(
            key="api_key",
            label="API key",
            kind=KIND_SECRET,
            env_key="HINDSIGHT_API_KEY",
            description="Used to authenticate with the Hindsight API.",
            placeholder="Enter Hindsight API key",
            inline=True,
        ),
        ProviderField(
            key="api_url",
            label="API URL",
            kind=KIND_TEXT,
            default="https://api.hindsight.vectorize.io",
            aliases=("apiUrl",),
            env_fallbacks=("HINDSIGHT_API_URL",),
            inline=True,
        ),
        ProviderField(
            key="bank_id",
            label="Bank ID",
            kind=KIND_TEXT,
            default="hermes",
            aliases=("bankId",),
            inline=True,
        ),
        ProviderField(
            key="recall_budget",
            label="Recall budget",
            kind=KIND_SELECT,
            default="mid",
            aliases=("budget",),
            options=(
                ProviderFieldOption("low", "low"),
                ProviderFieldOption("mid", "mid"),
                ProviderFieldOption("high", "high"),
            ),
            inline=True,
        ),
    ),
)


# Registry of providers that expose a desktop config surface. Providers without
# an entry here (e.g. ``builtin``) simply render no config panel. Honcho leads.
MEMORY_PROVIDERS: dict[str, MemoryProvider] = {
    HONCHO.name: HONCHO,
    HINDSIGHT.name: HINDSIGHT,
}


def get_memory_provider(name: str) -> MemoryProvider | None:
    """Return the declared provider for ``name``, or ``None`` if undeclared."""

    return MEMORY_PROVIDERS.get(name)
