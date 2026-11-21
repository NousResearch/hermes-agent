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
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def ctx_bound(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Bind ``fn`` to the CALLER's contextvars for another thread/executor. Profile isolation
    is a ContextVar-scoped HERMES_HOME override plus the per-turn secret scope; a worker started
    with an empty context silently lands on the default profile (or fails closed on secrets)."""
    ctx = contextvars.copy_context()
    return lambda *args, **kwargs: ctx.run(fn, *args, **kwargs)


def spawn_context_thread(target: Callable[..., Any], *, name: str, daemon: bool = True,
                         args: tuple = (), kwargs: Optional[dict[str, Any]] = None) -> threading.Thread:
    """Unstarted thread running *target* under the spawner's contextvars (see :func:`ctx_bound`).
    Every memory-provider background job (prefetch, sync, writer loops) must go through this."""
    return threading.Thread(target=ctx_bound(target), args=args, kwargs=kwargs, name=name, daemon=daemon)

# v1 = best-effort on_pre_compress() with the raw message list; v2 = opt-in fail-closed
# checkpoint (normalized evidence handoff + strict-mode failure propagation).
PRE_COMPRESS_CHECKPOINT_API_VERSION = 2

# Default glyph for recall indicators; providers may use their own brand mark.
INDICATOR_GLYPH = "🧠"

# ``memory.provider`` values that mean "the built-in store, no external plugin". The built-in
# store is core: doctor, migration and dependency refresh must never look these up as plugins.
CORE_MEMORY_PROVIDER_SENTINELS = frozenset({"", "default", "builtin", "built-in", "none"})


def is_core_memory_provider(name: Optional[str]) -> bool:
    """True when ``memory.provider`` selects the built-in store rather than an external plugin."""
    return str(name or "").strip().lower() in CORE_MEMORY_PROVIDER_SENTINELS


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


_TRIVIAL_SENDER_PREFIX_MAX_CHARS = 60


def _normalize_trivial_acknowledgement(text: str) -> str:
    """Normalize a short acknowledgement while preserving all interior words."""
    value = text.strip()
    if value.startswith("["):
        closing = value.find("]")
        if 1 <= closing <= _TRIVIAL_SENDER_PREFIX_MAX_CHARS and (
            closing + 1 == len(value) or value[closing + 1].isspace()
        ):
            value = value[closing + 1:].lstrip()
    value = unicodedata.normalize("NFD", value.lower())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = "".join(" " if char in ",;:!?" else char for char in value)
    value = " ".join(value.split())
    while value and not (value[0].isalnum() or value[0] == "_"):
        value = value[1:]
    while value and not (value[-1].isalnum() or value[-1] == "_"):
        value = value[:-1]
    return value.strip()


_TRIVIAL_ACK_FILE_MAX_BYTES = 16_384
_TRIVIAL_ACK_FILE_MAX_PHRASES = 256
_TRIVIAL_ACK_PHRASE_MAX_CHARS = 60
_TRIVIAL_ACK_PHRASE_MAX_WORDS = 4


def _is_recordable_trivial_ack(phrase: str) -> bool:
    """Guard the phrase file against regexes, sentences and look-alike words."""
    return bool(
        phrase
        and len(phrase) <= _TRIVIAL_ACK_PHRASE_MAX_CHARS
        and len(phrase.split()) <= _TRIVIAL_ACK_PHRASE_MAX_WORDS
        and all(char.isalpha() or char.isspace() for char in phrase)
    )


def _trivial_ack_phrase_path() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "state" / "trivial_ack_phrases.txt"


def _mutable_trivial_acknowledgements() -> frozenset[str]:
    """Read this profile's small phrase list; never evaluate regex/code from it."""
    try:
        path = _trivial_ack_phrase_path()
        try:
            with path.open("rb") as stream:
                raw = stream.read(_TRIVIAL_ACK_FILE_MAX_BYTES + 1)
        except FileNotFoundError:
            return frozenset()
        if len(raw) > _TRIVIAL_ACK_FILE_MAX_BYTES:
            return frozenset()
        lines = raw.decode("utf-8-sig").splitlines()
        parsed = set()
        for line in lines[:_TRIVIAL_ACK_FILE_MAX_PHRASES]:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            phrase = _normalize_trivial_acknowledgement(line)
            if _is_recordable_trivial_ack(phrase):
                parsed.add(phrase)
        return frozenset(parsed)
    except Exception:
        logger.debug("Could not load mutable trivial-ack phrases; using built-ins only", exc_info=True)
        return frozenset()


def record_trivial_acknowledgement(text: str) -> bool:
    """Grow the profile's phrase list so future turns skip memory prefetch.

    The text is normalized and validated exactly like a file line, so a regex
    (``^.*$``), a line with digits, or a long sentence is refused instead of
    poisoning the gate. A stored phrase matches only when it is the whole
    message, so even a recorded fragment cannot hide a longer request.
    """
    phrase = _normalize_trivial_acknowledgement(text or "")
    if not _is_recordable_trivial_ack(phrase):
        return False
    try:
        if phrase in _mutable_trivial_acknowledgements():
            return True
        path = _trivial_ack_phrase_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(f"{phrase}\n")
        return True
    except Exception:
        logger.debug("Could not record trivial-ack phrase", exc_info=True)
        return False


def is_trivial_prompt(text: Optional[str]) -> bool:
    """True for empty input, slash commands and bare acknowledgements/continuations.

    This only skips memory prefetch for a message with no semantic content; it
    does not suppress the user turn or its normal persistence.
    """
    stripped = (text or "").strip()
    if not stripped or stripped.startswith("/"):
        return True
    if TRIVIAL_PROMPT_RE.match(stripped):
        return True
    normalized = _normalize_trivial_acknowledgement(stripped)
    if not normalized:
        return False
    return bool(
        TRIVIAL_PROMPT_RE.match(normalized)
        or normalized in _mutable_trivial_acknowledgements()
    )


class MemoryProvider(ABC):
    """Abstract base class for memory providers.

    Multiplexing contract (#140058). One gateway process can serve several profiles, and each agent's
    ``MemoryManager`` owns its own provider instance, initialized for one profile. A provider must:

    - hand every load a NEW instance (``register(ctx)`` / the loader constructs one per agent). The
      manager refuses an instance another profile's manager already bound, rather than rebinding it;
    - bind every store, path, client, bank or tenant from ``initialize()``'s ``hermes_home`` (and
      scoped secrets read during ``initialize``) and keep that binding on the instance;
    - keep no profile-specific mutable state at module or class level, and never "rebind" a shared
      binding per call: two profiles' calls interleave on different threads.

    The manager runs tool calls and lifecycle hooks only from the profile the instance was initialized
    for (refusing callers that name another profile, or none on a multiplexed process), runs its
    background hooks under that profile's ``HERMES_HOME``, and passes ``hermes_home`` /
    ``agent_identity`` / ``caller_gsk`` to ``handle_tool_call`` when it accepts ``**kwargs``. Those
    keywords name the same profile the instance is bound to; they are not a reason to rebind. Core
    cannot see inside a provider: one that keeps a process-wide binding still breaks this contract.
    """

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
        Agent builds also pass ``resumed`` (bool: an earlier turn of this session id ran) and
        ``message_count`` (int: messages already stored for it, which on TUI/Desktop can include the
        prompt being submitted). A new session id (subagent, cron run, fresh chat) or no session
        store gives ``False`` and a count of 0 or 1; a ``/branch`` child seeded with copied history
        reports the copy. Read both with ``kwargs.get``: older Hermes omits them. A mid-process
        ``/resume`` rebinds through :meth:`on_session_switch` instead.
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
        session_id: str = "", messages: Optional[list[dict[str, Any]]] = None,
        turn_author: Optional[dict[str, Any]] = None,
    ) -> None:
        """Persist a completed turn (non-blocking). ``messages`` is the OpenAI-style list so far.
        ``turn_author`` (``{"id", "name", "is_bot"}``) is who wrote the user side; the manager sends it only to signatures that accept it."""

    @abstractmethod
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """OpenAI function-calling schemas ({"name", "description", "parameters"}); [] if none."""

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        """Handle one of this provider's tools; must return a JSON string."""
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def shutdown(self) -> None:
        """Clean shutdown — flush queues, close connections."""

    # -- Optional hooks (override to opt in) ---------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Per-turn tick. kwargs may include remaining_tokens, model, platform, tool_count, author_id, author_name,
        author_is_bot, previous_message. ``model`` is the turn's primary model id at turn start ("" when unset; a
        preset name under mixture-of-agents). The author trio names who wrote THIS turn (None, None, False without
        one): a shared session carries several participants, so a provider keying durable state on identity must read
        it per turn. ``previous_message`` is the secret-scrubbed text of the person's last message before this turn
        ("" on a first turn or an image-only one), from the transcript Hermes holds, so a resumed process can redo
        what a live one queued at the end of that turn. Context Hermes appended to an image turn (recalled memory,
        plugin and gateway notes) is not part of it. Best-effort: known runtime user-role rows and /steer messages
        are skipped, but a notice a gateway replays after a restart (cron delivery, process heartbeat, watch-disabled
        notice) can come through, and so can the notes on an image turn an older Hermes stored or that repair or
        compaction later merged with another turn."""

    def identity_signature(self) -> dict[str, Any]:
        """Identity-mapping values that must bust a cached gateway agent when they change (writer identity, alias
        tables, session-name prefixing). Provider-namespaced keys, JSON-serializable values. The gateway calls this
        on an uninitialized instance on every inbound message, so keep it cheap and read-only."""
        return {}

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """End-of-session extraction; fires only at real session boundaries, never per-turn."""

    def on_session_switch(
        self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, rewound: bool = False, **kwargs,
    ) -> None:
        """session_id reassigned mid-process (/resume, /branch, /reset, /new, compression)
        without teardown: rebind per-session state so later writes land in the right record.
        ``reset`` is True only for a genuinely new conversation (flush buffers); ``rewound``:
        same id but the transcript was truncated."""

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Extract insights from ``messages`` about to be compressed, fed into the summary prompt."""
        return ""

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        """PARENT-side observation of a completed delegation (the subagent has no provider session)."""

    def get_config_schema(self) -> list[dict[str, Any]]:
        """Setup fields for ``hermes memory setup`` ([] if none): ``key``, ``description``,
        optional ``secret`` (goes to .env), ``required``, ``default``, ``choices``, ``type``
        (text | integer | number | boolean), ``minimum``/``maximum``/``step``, ``url``,
        ``env_var`` (explicit secret env var; default auto-generated)."""
        return []

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        """Write non-secret setup ``values`` to the provider's native config. Plugins MUST either
        override this or use only env vars (every schema field carrying ``env_var``)."""

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Mirror a built-in memory-tool write (``action``: add | replace | remove; ``target``:
        memory | user; ``metadata``: provenance such as write_origin, session_id, tool_name).
        For replace/remove, ``metadata["previous_content"]`` is the full entry selected
        under the native-store lock. A memory-tool patch arrives as a replace whose
        ``content`` is the whole entry after the edit. Notifications follow a successful complete write
        or batch; each batch operation sees the preceding operation's result. Older
        callers may omit this field: ``old_text`` alone is not authoritative identity.
        """

    def backup_paths(self) -> list[str]:
        """Absolute paths of provider state OUTSIDE HERMES_HOME for ``hermes backup``/``import``
        (paths outside the home dir are skipped). MUST work without ``initialize()`` or network."""
        return []

    # -- Profile lifecycle (uninitialized instance; that profile's home and secrets bound; errors are
    # logged). Keep them fast: the dashboard and Desktop run them inside the create/rename request,
    # so put network work (creating a remote peer) on a background thread. --

    def on_profile_created(self, profile_name: str, hermes_home: str) -> None:
        """A profile whose config selects this provider was just created by ``create_profile`` (CLI,
        dashboard, Desktop/TUI; a clone carries ``memory.provider``). Not called for profile import.
        Set up per-profile provider state here (a peer, a host entry)."""

    def on_profile_renamed(self, old_name: str, new_name: str, hermes_home: str) -> None:
        """A profile whose config selects this provider was renamed; ``hermes_home`` is its new
        directory. Re-key anything the provider stores under the old profile name."""

    # -- Dashboard config panel for a ``config_schema.py`` declaring ``storage=STORAGE_PROVIDER`` --

    def read_config_fields(self, hermes_home: str) -> dict[str, Any]:
        """``{"values": {field key: stored native value}, "placeholders": {field key: text}}``. A key
        absent from ``values`` reads as unset; a placeholder shows what a blank field resolves to.
        A stored secret may be included: it only marks the field as set and is never sent out."""
        return {"values": {}, "placeholders": {}}

    def write_config_fields(self, values: dict[str, Any], hermes_home: str) -> None:
        """Persist the submitted fields (only those present): natives already coerced, ``None`` clears a
        key, non-blank secrets included (core saves each to ``.env`` under its ``env_key`` once this
        returns). Raise ``ValueError`` (a subclass such as ``json.JSONDecodeError`` counts, also under
        ``plugins.isolation: host``) to refuse the save (the dashboard shows it as a 400)."""
        raise NotImplementedError(f"memory provider {self.name} declares storage=provider but cannot write it")
