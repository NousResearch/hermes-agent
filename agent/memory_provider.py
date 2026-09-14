"""Abstract base class for pluggable memory providers.

Plugins ship in ``plugins/memory/<name>/``, activated via ``memory.provider`` (ONE external
provider at a time). Lifecycle, driven by MemoryManager: initialize -> system_prompt_block /
prefetch / sync_turn per turn -> tool dispatch -> shutdown, plus optional ``on_*`` hooks.
"""

from __future__ import annotations

import contextvars
import logging
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def ctx_bound(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Bind ``fn`` to the CALLER's contextvars for another thread/executor. Profile isolation
    is a ContextVar-scoped HERMES_HOME override plus the per-turn secret scope; a worker started
    with an empty context silently lands on the default profile (or fails closed on secrets)."""
    ctx = contextvars.copy_context()
    return lambda *args, **kwargs: ctx.run(fn, *args, **kwargs)


def spawn_context_thread(target: Callable[..., Any], *, name: str, daemon: bool = True,
                         args: tuple = ()) -> threading.Thread:
    """Unstarted thread running *target* under the spawner's contextvars (see :func:`ctx_bound`).
    Every memory-provider background job (prefetch, sync, writer loops) must go through this."""
    return threading.Thread(target=ctx_bound(target), args=args, name=name, daemon=daemon)

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

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background recall after each turn; prefetch() consumes it next turn."""

    def recall_status(self) -> Optional[RecallStatus]:
        """What the most recent :meth:`prefetch` injected (``None`` = no indicator). Must reflect
        only the LAST prefetch, never a stale prior count."""
        return None

    def sync_turn(
        self, user_content: str, assistant_content: str, *,
        session_id: str = "", messages: Optional[List[Dict[str, Any]]] = None,
        turn_author: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a completed turn (non-blocking). ``messages`` is the OpenAI-style list so far.
        ``turn_author`` (``{"id", "name", "is_bot"}``) is who wrote the user side; the manager sends it only to signatures that accept it."""

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
        """Per-turn tick. kwargs may include remaining_tokens, model, platform, tool_count, author_id, author_name,
        author_is_bot. The author trio names who wrote THIS turn (None, None, False without one): a shared session
        carries several participants, so a provider keying durable state on identity must read it per turn."""

    def identity_signature(self) -> Dict[str, Any]:
        """Identity-mapping values that must bust a cached gateway agent when they change (writer identity, alias
        tables, session-name prefixing). Provider-namespaced keys, JSON-serializable values. The gateway calls this
        on an uninitialized instance on every inbound message, so keep it cheap and read-only."""
        return {}

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

    def journey_cards(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Return durable memory entries for the learning-journey graph.

        The journey surfaces (``hermes journey``, the TUI ``/journey``
        overlay, and the desktop Star Map) render what the agent has learned
        over time. By default that graph only sees built-in memory
        (``MEMORY.md`` / ``USER.md``); implementing this hook lets an external
        provider's durable facts appear alongside them as first-class
        memory nodes.

        Each card is a dict with:

        - ``body`` (str, required): the fact/entry text.
        - ``title`` (str, optional): short label; defaults to the body's
          first line.
        - ``timestamp`` (optional): unix seconds, ISO-8601 string, or a
          ``datetime`` — when the entry was learned. ``None`` is allowed
          (the node renders without a date bucket).
        - ``session_id`` (str, optional): the provider-side session this
          entry was derived from, when known. Journey surfaces use it two
          ways: resolving the originating Hermes conversation (when the
          provider's session naming maps onto Hermes session ids) and
          fetching the source corpus via ``journey_session_messages``.

        Contract:

        - MUST be callable without ``initialize()`` — journey views run
          outside any chat session. Resolve config/credentials yourself.
        - MUST be best-effort and non-fatal: backend down, not configured,
          or any error → return ``[]`` (never raise). Journey rendering
          must never break because a memory backend is unreachable.
        - SHOULD be fast: called from interactive UI paths. Cap work at
          ``limit`` entries and use short network timeouts.
        - Cards are READ-ONLY in the journey: edit/delete for provider
          nodes is refused and points users at the provider's own tools.

        Default returns an empty list (provider contributes nothing).
        """
        return []

    def journey_session_messages(
        self, session_id: str, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Return the raw source messages of one provider-side session.

        Journey surfaces call this when the user asks to read the original
        corpus a ``journey_cards`` entry was derived from (the card's
        ``session_id``). This covers provider-only history — e.g. imported
        conversations that never existed as Hermes sessions — where the
        provider backend is the only place the source data lives.

        Each message is a dict with:

        - ``content`` (str, required): the message text.
        - ``peer`` (str, optional): who authored it (provider peer id).
        - ``role`` (str, optional): ``"user"`` or ``"assistant"`` when the
          provider can tell which peer is the human. Used when the corpus is
          materialized back into a Hermes session; consumers fall back to a
          first-peer-is-user heuristic when absent.
        - ``timestamp`` (optional): unix seconds, ISO-8601 string, or a
          ``datetime`` — when the message was created.

        Contract mirrors ``journey_cards``:

        - MUST be callable without ``initialize()`` — resolve config and
          credentials yourself.
        - MUST be best-effort and non-fatal: unknown session, backend down,
          or any error → return ``[]`` (never raise).
        - SHOULD be fast and bounded: return at most ``limit`` messages in
          chronological order and use short network timeouts.

        Default returns an empty list (no corpus available).
        """
        return []
