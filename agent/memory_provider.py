"""Abstract base class for pluggable memory providers.

Plugins ship in ``plugins/memory/<name>/``, activated via ``memory.provider`` (ONE external
provider at a time). Lifecycle, driven by MemoryManager: initialize -> system_prompt_block /
prefetch / sync_turn per turn -> tool dispatch -> shutdown, plus optional ``on_*`` hooks.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

# v1 = best-effort on_pre_compress() with the raw message list; v2 = opt-in fail-closed
# checkpoint (normalized evidence handoff + strict-mode failure propagation).
PRE_COMPRESS_CHECKPOINT_API_VERSION = 2

# Default glyph for recall indicators; providers may use their own brand mark.
INDICATOR_GLYPH = "🧠"


@dataclass(frozen=True)
class RecallStatus:
    """What the last prefetch injected, for the deterministic recall indicator
    (``MemoryManager.describe_recall``). ``count == 0`` means content without a
    discrete count (e.g. a synthesized reflect answer) and renders generically."""

    provider_label: str
    count: int
    glyph: str = INDICATOR_GLYPH


# ---------------------------------------------------------------------------
# Structured recall-provenance envelope (tier B of #84251)
# ---------------------------------------------------------------------------
#
# Recalled memory used to reach the host as an opaque, already-formatted
# ``str`` (see ``prefetch`` below). Flattening to a string that early destroyed
# every provenance signal — which provider produced a line, whether the host or
# a provider authored a "trust" framing, where a fact came from — so the host
# could not reason about, or defend against, recalled content. ``RecallItem``
# is the structured envelope that carries that provenance from a provider up to
# (and only up to) ``build_memory_context_block()``, which renders it back to a
# single ``str``. Nothing downstream of that render changes type.


class RecallTrust(Enum):
    """Host-controlled trust level for a recalled item.

    Only the HOST sets this. Provider-supplied text or metadata must never be
    able to elevate an item's trust — the aggregation path in
    ``MemoryManager`` re-stamps every item ``UNTRUSTED`` regardless of what the
    provider returned. ``TRUSTED`` exists for a future host-authored
    elevation mechanism; nothing in tier B ever sets it.
    """

    UNTRUSTED = "untrusted"
    TRUSTED = "trusted"


class RecallSensitivity(Enum):
    """Optional, diagnostic sensitivity hint for a recalled item.

    Sensitivity classification POLICY (what to do with a sensitive item) is
    explicitly out of scope for tier B; this enum only gives providers a place
    to record a hint. It carries no host behavior yet.
    """

    NORMAL = "normal"
    SENSITIVE = "sensitive"


@dataclass(frozen=True)
class RecallItem:
    """A single recalled memory plus its host-verifiable provenance.

    ``text`` and ``provider`` are required. ``provider`` and ``trust`` are
    HOST-STAMPED during aggregation — a provider cannot spoof either. The
    remaining fields are optional provenance; ``metadata`` is diagnostic only
    (never authoritative, never trusted).

    Deliberately has NO ``__str__``: an implicit string conversion is exactly
    the flattening that erased provenance in the first place, so callers must
    render items explicitly (via ``build_memory_context_block``) and can never
    accidentally interpolate a bare item into a prompt.
    """

    text: str
    provider: str
    trust: RecallTrust = RecallTrust.UNTRUSTED
    source: Optional[str] = None
    writer: Optional[str] = None
    sensitivity: Optional[RecallSensitivity] = None
    verified: bool = False
    record_id: Optional[str] = None
    occurred_at: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)


# Prompts with no semantic signal; single source of truth for the core prefetch gate and
# provider-side classifiers. Anchored and followed only by whitespace/punctuation, so
# "k8s"/"yolo"/"note" do NOT match while "hi!"/"thanks :)"/"done???" do.
TRIVIAL_PROMPT_RE = re.compile(
    r'^(yes|no|ok|okay|sure|thanks|thank you|y|n|yep|nope|yeah|nah|'
    r'hi|hey|hello|yo|sup|'
    r'continue|go ahead|do it|proceed|got it|cool|nice|great|done|next|lgtm|k)'
    r'[\s!?.:;,"' + "'" + r'~\u2018\u2019\u201c\u201d\u2014\u2013\u2026()\[\]{}<>*&^%$#@!+=`\u00a0]*$',
    re.IGNORECASE,
)


def is_trivial_prompt(text: Optional[str]) -> bool:
    """True for empty input, slash commands and bare greetings/acknowledgements (skipping
    recall saves a round-trip and keeps stale context from derailing one-word replies)."""
    stripped = (text or "").strip()
    if not stripped or stripped.startswith("/"):
        return True
    return bool(TRIVIAL_PROMPT_RE.match(stripped))


class MemoryProvider(ABC):
    """Abstract base class for memory providers."""

    # Providers that durably checkpoint every successful on_pre_compress() set this to
    # PRE_COMPRESS_CHECKPOINT_API_VERSION; 1 = best-effort legacy.
    pre_compress_checkpoint_api_version = 1

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider (e.g. 'builtin', 'honcho', 'hindsight')."""

    # -- Core lifecycle (implement these) ------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        """Configured, credentialed and ready? Gates activation; check config/deps only, no network."""

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize once at agent startup (connections, resources, threads).

        kwargs always include ``hermes_home`` (profile-scoped storage; never hardcode
        ``~/.hermes``) and ``platform``; may include ``agent_context`` ("primary" |
        "subagent" | "cron" | "flush" — skip writes for non-primary contexts),
        ``agent_identity``, ``agent_workspace``, ``parent_session_id``, ``user_id``, ``user_id_alt``.
        """

    def unavailable_reason(self) -> str:
        """User-facing hint for the "provider unavailable" warning (``initialize()`` never runs then)."""
        return ""

    def system_prompt_block(self) -> str:
        """STATIC system-prompt text; "" to skip. Recalled context goes through prefetch(), not here."""
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Formatted recall context for the upcoming turn ("" if none). Must be fast — recall
        in the background and return cached results; ``session_id`` scopes concurrent sessions."""
        return ""

    def prefetch_items(
        self, query: str, *, session_id: str = ""
    ) -> Optional[List["RecallItem"]]:
        """Structured recall hook — return provenance-bearing ``RecallItem``s.

        This is the tier-B (#84251) opt-in companion to :meth:`prefetch`.
        Return a list of :class:`RecallItem` to hand the host structured recall
        it can frame per-item (provider, trust, source). Returning an empty
        list means "implemented, but nothing to recall this turn".

        Return ``None`` (the default) to signal "not implemented" — the host
        then falls back to the legacy string :meth:`prefetch` and wraps its
        output as a single untrusted item. Providers overriding this should be
        just as fast as ``prefetch`` (return cached results; do the real recall
        on a background thread).

        The host ALWAYS re-stamps ``provider`` and ``trust`` on every returned
        item, so a provider cannot elevate its own trust or impersonate another
        provider by populating those fields.
        """
        return None

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background recall after each turn; prefetch() consumes it next turn."""

    def recall_status(self) -> Optional[RecallStatus]:
        """What the most recent :meth:`prefetch` injected (``None`` = no indicator). Must reflect
        only the LAST prefetch, never a stale prior count."""
        return None

    def sync_turn(
        self, user_content: str, assistant_content: str, *,
        session_id: str = "", messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist a completed turn (non-blocking). ``messages`` is the OpenAI-style list so far."""

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """OpenAI function-calling schemas ({"name", "description", "parameters"}); [] if none."""

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle one of this provider's tools; must return a JSON string."""
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def shutdown(self) -> None:
        """Clean shutdown — flush queues, close connections."""

    # -- Optional hooks (override to opt in) ---------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Per-turn tick. kwargs may include remaining_tokens, model, platform, tool_count."""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """End-of-session extraction; fires only at real session boundaries, never per-turn."""

    def on_session_switch(
        self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, rewound: bool = False, **kwargs,
    ) -> None:
        """session_id reassigned mid-process (/resume, /branch, /reset, /new, compression)
        without teardown: rebind per-session state so later writes land in the right record.
        ``reset`` is True only for a genuinely new conversation (flush buffers); ``rewound``:
        same id but the transcript was truncated."""

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract insights from ``messages`` about to be compressed, fed into the summary prompt."""
        return ""

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        """PARENT-side observation of a completed delegation (the subagent has no provider session)."""

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Setup fields for ``hermes memory setup`` ([] if none): ``key``, ``description``,
        optional ``secret`` (goes to .env), ``required``, ``default``, ``choices``, ``type``
        (text | integer | number | boolean), ``minimum``/``maximum``/``step``, ``url``,
        ``env_var`` (explicit secret env var; default auto-generated)."""
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write non-secret setup ``values`` to the provider's native config. Plugins MUST either
        override this or use only env vars (every schema field carrying ``env_var``)."""

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror a built-in memory-tool write (``action``: add | replace | remove; ``target``:
        memory | user; ``metadata``: provenance such as write_origin, session_id, tool_name)."""

    def backup_paths(self) -> List[str]:
        """Absolute paths of provider state OUTSIDE HERMES_HOME for ``hermes backup``/``import``
        (paths outside the home dir are skipped). MUST work without ``initialize()`` or network."""
        return []
