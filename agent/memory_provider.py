"""Abstract base class for pluggable memory providers.

Memory providers give the agent persistent recall across sessions.
The MemoryManager enforces a one-external-provider limit to prevent
tool schema bloat and conflicting memory backends.

External providers (Honcho, Hindsight, Mem0, etc.) are registered
and managed via MemoryManager. Only one external provider runs at a
time.

Registration:
  Plugins ship in plugins/memory/<name>/ and are activated via
  the memory.provider config key.

Lifecycle (called by MemoryManager, wired in run_agent.py):
  initialize()          — connect, create resources, warm up
  system_prompt_block()  — static text for the system prompt
  prefetch(query)        — background recall before each turn
  sync_turn(user, asst)  — async write after each turn
  get_tool_schemas()     — tool schemas to expose to the model
  handle_tool_call()     — dispatch a tool call
  shutdown()             — clean exit

Optional hooks (override to opt in):
  on_turn_start(turn, message, **kwargs) — per-turn tick with runtime context
  on_session_end(messages)               — end-of-session extraction
  on_session_switch(new_session_id, **kwargs) — mid-process session_id rotation
  on_pre_compress(messages) -> str       — extract before context compression
  on_memory_write(action, target, content, metadata=None) — mirror built-in memory writes
  on_delegation(task, result, **kwargs)  — parent-side observation of subagent work
  backup_paths() -> list[str]            — extra on-disk paths to include in `hermes backup`
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

# Default glyph for the deterministic memory indicators. Providers override
# per-status with their own brand mark (e.g. Hindsight uses "👁️").
INDICATOR_GLYPH = "🧠"


@dataclass(frozen=True)
class RecallStatus:
    """Summary of what a provider's most recent prefetch injected this turn.

    Returned by :meth:`MemoryProvider.recall_status` so the agent can emit a
    deterministic, model-independent "memory was used" indicator (see
    ``MemoryManager.describe_recall``). ``count`` is the number of discrete
    memories injected; ``0`` means content was injected but has no discrete
    count (e.g. a synthesized reflect answer), which the indicator renders
    generically rather than as "0 memories". ``glyph`` is the brand mark the
    indicator leads with.
    """

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


# Prompts that carry no semantic signal — trivial acknowledgements, greetings,
# slash commands, empty input. Single source of truth shared by the core
# per-turn prefetch gate (agent/turn_context.py, run_agent.py) and provider-
# side classifiers (plugins/memory/honcho) so the two can never drift apart.
# The alternation is anchored and may only be followed by whitespace or
# punctuation, so words that merely START with a trivial word ("k8s", "yolo",
# "note", "hindsight") do NOT match, while trailing-punctuation variants
# ("hi!", "hey.", "thanks :)", "done???") do.
TRIVIAL_PROMPT_RE = re.compile(
    r'^(yes|no|ok|okay|sure|thanks|thank you|y|n|yep|nope|yeah|nah|'
    r'hi|hey|hello|yo|sup|'
    r'continue|go ahead|do it|proceed|got it|cool|nice|great|done|next|lgtm|k)'
    r'[\s!?.:;,"' + "'" + r'~\u2018\u2019\u201c\u201d\u2014\u2013\u2026()\[\]{}<>*&^%$#@!+=`\u00a0]*$',
    re.IGNORECASE,
)


def is_trivial_prompt(text: Optional[str]) -> bool:
    """Return True if a user prompt is too trivial to warrant memory recall.

    Empty/whitespace-only input, slash commands, and bare greetings or
    acknowledgements (with optional trailing punctuation) all count as
    trivial. Callers use this to skip memory-provider prefetch/injection
    on turns that carry no semantic signal — saving a blocking network
    round-trip and preventing stale user-model context from derailing
    one-word replies.
    """
    if not text:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.startswith("/"):
        return True
    return bool(TRIVIAL_PROMPT_RE.match(stripped))


class MemoryProvider(ABC):
    """Abstract base class for memory providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider (e.g. 'builtin', 'honcho', 'hindsight')."""

    # -- Core lifecycle (implement these) ------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider is configured, has credentials, and is ready.

        Called during agent init to decide whether to activate the provider.
        Should not make network calls — just check config and installed deps.
        """

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize for a session.

        Called once at agent startup. May create resources (banks, tables),
        establish connections, start background threads, etc.

        kwargs always include:
          - hermes_home (str): The active HERMES_HOME directory path. Use this
            for profile-scoped storage instead of hardcoding ``~/.hermes``.
          - platform (str): "cli", "telegram", "discord", "cron", etc.

        kwargs may also include:
          - agent_context (str): "primary", "subagent", "cron", or "flush".
            Providers should skip writes for non-primary contexts (cron system
            prompts would corrupt user representations).
          - agent_identity (str): Profile name (e.g. "coder"). Use for
            per-profile provider identity scoping.
          - agent_workspace (str): Shared workspace name (e.g. "hermes").
          - parent_session_id (str): For subagents, the parent's session_id.
          - user_id (str): Platform user identifier (gateway sessions).
          - user_id_alt (str): Optional alternate stable platform user identifier.
        """

    def unavailable_reason(self) -> str:
        """Actionable reason this provider reports unavailable, for the caller.

        ``is_available()`` gates initialization, so a provider that reports
        unavailable is never initialized — any diagnostic it would log from
        ``initialize()`` is unreachable. Return a short, user-facing hint here
        (e.g. which package to install) so the caller's "provider unavailable"
        warning can surface it. Empty string (the default) adds nothing.
        """
        return ""

    def system_prompt_block(self) -> str:
        """Return text to include in the system prompt.

        Called during system prompt assembly. Return empty string to skip.
        This is for STATIC provider info (instructions, status). Prefetched
        recall context is injected separately via prefetch().
        """
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant context for the upcoming turn.

        Called before each API call. Return formatted text to inject as
        context, or empty string if nothing relevant. Implementations
        should be fast — use background threads for the actual recall
        and return cached results here.

        session_id is provided for providers serving concurrent sessions
        (gateway group chats, cached agents). Providers that don't need
        per-session scoping can ignore it.
        """
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
        """Queue a background recall for the NEXT turn.

        Called after each turn completes. The result will be consumed
        by prefetch() on the next turn. Default is no-op — providers
        that do background prefetching should override this.
        """

    def recall_status(self) -> Optional[RecallStatus]:
        """Describe what the most recent :meth:`prefetch` injected, for the UI.

        Called by the agent right after prefetch, on the same (single) turn
        thread, so it can surface a deterministic "👁️ recalled N memories"
        status line that does not depend on the model choosing to mention it.

        Return ``None`` (the default) when this provider injected nothing this
        turn or does not want a visible indicator. Providers that override it
        must reflect only the LAST prefetch — never a stale prior count.
        """
        return None

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist a completed turn to the backend.

        Called after each turn. Should be non-blocking — queue for
        background processing if the backend has latency.

        ``messages`` is the OpenAI-style conversation message list as of the
        completed turn, including any assistant tool calls and tool results.
        Providers that do not need raw turn context can ignore it.
        """

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas this provider exposes.

        Each schema follows the OpenAI function calling format:
        {"name": "...", "description": "...", "parameters": {...}}

        Return empty list if this provider has no tools (context-only).
        """

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a tool call for one of this provider's tools.

        Must return a JSON string (the tool result).
        Only called for tool names returned by get_tool_schemas().
        """
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def shutdown(self) -> None:
        """Clean shutdown — flush queues, close connections."""

    # -- Optional hooks (override to opt in) ---------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Called at the start of each turn with the user message.

        Use for turn-counting, scope management, periodic maintenance.

        kwargs may include: remaining_tokens, model, platform, tool_count.
        Providers use what they need; extras are ignored.
        """

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Called when a session ends (explicit exit or timeout).

        Use for end-of-session fact extraction, summarization, etc.
        messages is the full conversation history.

        NOT called after every turn — only at actual session boundaries
        (CLI exit, /reset, gateway session expiry).
        """

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """Called when the agent switches session_id mid-process.

        Fires on ``/resume``, ``/branch``, ``/reset``, ``/new`` (CLI), the
        gateway equivalents, and context compression — any path that
        reassigns ``AIAgent.session_id`` without tearing the provider down.

        Providers that cache per-session state in ``initialize()``
        (``_session_id``, ``_document_id``, accumulated turn buffers,
        counters) should update or reset that state here so subsequent
        writes land in the correct session's record.

        Parameters
        ----------
        new_session_id:
            The session_id the agent just switched to.
        parent_session_id:
            The previous session_id, if meaningful — set for ``/branch``
            (fork lineage), context compression (continuation lineage),
            and ``/resume`` (the session we're leaving). Empty string
            when no lineage applies.
        reset:
            ``True`` when this is a genuinely new conversation, not a
            resumption of an existing one. Fired by ``/reset`` / ``/new``.
            Providers should flush accumulated per-session buffers
            (``_session_turns``, ``_turn_counter``, etc.) when this is
            set. ``False`` for ``/resume`` / ``/branch`` / compression
            where the logical conversation continues under the new id.
        rewound:
            ``True`` if session_id is unchanged but the transcript was
            truncated; providers caching per-turn document state should
            invalidate.

        Default is no-op for backward compatibility.
        """

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Called before context compression discards old messages.

        Use to extract insights from messages about to be compressed.
        messages is the list that will be summarized/discarded.

        Return text to include in the compression summary prompt so the
        compressor preserves provider-extracted insights. Return empty
        string for no contribution (backwards-compatible default).
        """
        return ""

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Called on the PARENT agent when a subagent completes.

        The parent's memory provider gets the task+result pair as an
        observation of what was delegated and what came back. The subagent
        itself has no provider session (skip_memory=True).

        task: the delegation prompt
        result: the subagent's final response
        child_session_id: the subagent's session_id
        """

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Return config fields this provider needs for setup.

        Used by 'hermes memory setup' to walk the user through configuration.
        Each field is a dict with:
          key:         config key name (e.g. 'api_key', 'mode')
          description: human-readable description
          secret:      True if this should go to .env (default: False)
          required:    True if required (default: False)
          default:     default value (optional)
          choices:     list of valid values (optional)
          type:        text, integer, number, or boolean (optional)
          minimum:     numeric lower bound for integer/number fields (optional)
          maximum:     numeric upper bound for integer/number fields (optional)
          step:        numeric input step for Dashboard rendering (optional)
          url:         URL where user can get this credential (optional)
          env_var:     explicit env var name for secrets (default: auto-generated)

        Return empty list if no config needed (e.g. local-only providers).
        """
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write non-secret config to the provider's native location.

        Called by 'hermes memory setup' after collecting user inputs.
        ``values`` contains only non-secret fields (secrets go to .env).
        ``hermes_home`` is the active HERMES_HOME directory path.

        Providers with native config files (JSON, YAML) should override
        this to write to their expected location. Providers that use only
        env vars can leave the default (no-op).

        All new memory provider plugins MUST implement either:
        - save_config() for native config file formats, OR
        - use only env vars (in which case get_config_schema() fields
          should all have ``env_var`` set and this method stays no-op).
        """

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called when the built-in memory tool writes an entry.

        action: 'add', 'replace', or 'remove'
        target: 'memory' or 'user'
        content: the entry content
        metadata: structured provenance for the write, when available. Common
          keys include ``write_origin``, ``execution_context``, ``session_id``,
          ``parent_session_id``, ``platform``, and ``tool_name``.

        Use to mirror built-in memory writes to your backend.
        """

    def backup_paths(self) -> List[str]:
        """Return extra on-disk paths this provider stores OUTSIDE HERMES_HOME.

        ``hermes backup`` only walks HERMES_HOME, so any provider state kept
        under ``~/.honcho``, ``~/.hindsight``, ``~/.openviking``, etc. is lost
        across a backup/import cycle unless it's declared here.

        Return a list of absolute path strings (files or directories). The
        backup command resolves each, captures the ones that exist and live
        under the user's home directory into a reserved ``_external/`` subtree
        of the archive, and ``hermes import`` restores them to their original
        locations. Paths outside the home directory are skipped for safety.

        MUST be callable without ``initialize()`` and without network — resolve
        from config/env only. Default returns an empty list (nothing external).
        """
        return []
