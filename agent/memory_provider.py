"""Abstract base class for pluggable memory providers.

Plugins ship in ``plugins/memory/<name>/``, activated via ``memory.provider`` (ONE external
provider at a time). Lifecycle, driven by MemoryManager: initialize -> system_prompt_block /
prefetch / sync_turn per turn -> tool dispatch -> shutdown, plus optional ``on_*`` hooks.
"""

from __future__ import annotations

import contextvars
import json
import logging
import math
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)


def ctx_bound(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Bind ``fn`` to the CALLER's contextvars for another thread/executor. Profile isolation
    is a ContextVar-scoped HERMES_HOME override plus the per-turn secret scope; a worker started
    with an empty context silently lands on the default profile (or fails closed on secrets)."""
    ctx = contextvars.copy_context()
    return lambda *args, **kwargs: ctx.run(fn, *args, **kwargs)


def spawn_context_thread(target: Callable[..., Any], *, name: str, daemon: bool = True,
                         args: tuple = (), kwargs: Optional[Dict[str, Any]] = None) -> threading.Thread:
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


# Structured prefetch observations are an in-process extension surface. Keep
# their budget deliberately small: providers can still inject their existing
# formatted string, but an observer must not become an unbounded side channel.
MAX_MEMORY_OBSERVATIONS = 16
MAX_MEMORY_OBSERVATION_DEPTH = 6
MAX_MEMORY_OBSERVATION_ITEMS = 64
MAX_MEMORY_OBSERVATION_STRING_CHARS = 4096
MAX_MEMORY_OBSERVATION_BYTES = 16 * 1024
MAX_MEMORY_OBSERVATION_BATCH_BYTES = 64 * 1024
MAX_MEMORY_OBSERVATION_FIELD_CHARS = 128
# Global cap on the total number of JSON nodes visited while freezing a single
# observation payload. Per-container width and per-payload depth are bounded
# individually, but their product (64**6) is not: without an aggregate budget
# a well-formed-looking width/depth combination can force explosive traversal
# and allocation before the encoded-byte check has a chance to reject it. The
# limit is picked to comfortably exceed any payload that could fit under
# MAX_MEMORY_OBSERVATION_BYTES while still terminating pathological trees
# during recursion rather than after full expansion.
MAX_MEMORY_OBSERVATION_NODES = 4096
# Operation-wide cap on freeze traversal across every observation candidate a
# prefetch inspects — including malformed ones. Per-candidate the payload
# budget still fires, but a provider returning many malformed payloads that
# each exhaust a fresh 4096-node budget would otherwise force repeated deep
# traversal for each one. The operation cap is set to
# MAX_MEMORY_OBSERVATIONS × MAX_MEMORY_OBSERVATION_NODES so a well-behaved
# provider filling the full accepted prefix with max-node payloads still fits,
# while a malformed tail exhausts the shared budget and every subsequent
# candidate fails on its first budget decrement instead of walking its tree.
MAX_MEMORY_OBSERVATION_OPERATION_NODES = (
    MAX_MEMORY_OBSERVATIONS * MAX_MEMORY_OBSERVATION_NODES
)
# Operation-wide cap on the total number of candidate observations the manager
# is willing to *inspect* (pull via next() and validate) across every provider
# in one prefetch. The per-payload and operation node caps only fire once a
# freeze call runs — wrong-type or invalid-metadata candidates fail their
# validation guard before freeze and consume no node budget at all, so an
# unbounded or infinite malformed iterable could still force unbounded
# next()/logging work. This cap decrements for every candidate the manager
# pulls (valid, malformed, or peeked for truncation), stops the tail once
# exhausted, and emits at most one truncation warning. Set to a small strict
# multiple of MAX_MEMORY_OBSERVATIONS so a well-behaved provider filling the
# accepted prefix (plus the count/bytes look-ahead) never trips it, while a
# pathological tail is bounded well below the point where per-candidate log
# spam becomes a problem.
MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES = MAX_MEMORY_OBSERVATIONS * 4


class _FrozenDict(dict):
    """A JSON-serializable dict with the mutating surface disabled."""

    def __setitem__(self, key, value):
        raise TypeError("frozen observation payload")

    def __delitem__(self, key):
        raise TypeError("frozen observation payload")

    def clear(self):
        raise TypeError("frozen observation payload")

    def pop(self, key, default=None):
        raise TypeError("frozen observation payload")

    def popitem(self):
        raise TypeError("frozen observation payload")

    def setdefault(self, key, default=None):
        raise TypeError("frozen observation payload")

    def update(self, *args, **kwargs):
        raise TypeError("frozen observation payload")

    def __ior__(self, other):
        raise TypeError("frozen observation payload")


@dataclass(frozen=True)
class MemoryObservation:
    """One bounded, provider-authored observation attached to a prefetch.

    ``payload`` is opaque to Hermes, but the manager only emits observations
    after recursively validating and freezing it as JSON-shaped data. The
    provider field is normally left empty by a provider and is bound to the
    registered provider name by :class:`MemoryManager`; a non-empty mismatch
    is rejected rather than allowing provenance to be spoofed.

    The class is intentionally generic. It does not encode retrieval ranks,
    tenants, incident identifiers, model identity, storage, cryptography, or
    evaluation metrics.
    """

    source_kind: str
    schema: str
    version: int
    payload: Any
    provider: str = ""


@dataclass(frozen=True)
class MemoryPrefetchResult:
    """Immutable context plus observations from one provider prefetch call.

    Existing providers may continue returning ``str``. ``MemoryManager``
    converts that legacy return to this shape internally, preserving the
    context bytes. The manager owns the observation trust boundary: it warns
    and applies the operation-level bounds, then replaces provider payloads
    with recursively immutable values before returning this trusted result and
    before exposing its observation tuple to the privacy-limited hook.
    """

    context: str = ""
    observations: tuple[MemoryObservation, ...] = ()

    def __post_init__(self) -> None:
        # A provider may conveniently pass a list. Keep every supplied item
        # here so constructing this public value never silently loses data;
        # MemoryManager is the authoritative trust boundary that warns and
        # truncates before an operation result or observer event is emitted.
        # Reject arbitrary iterables instead of consuming a potentially
        # unbounded generator at the trust boundary.
        raw_observations = self.observations
        if raw_observations is None:
            raw_observations = ()
        if not isinstance(raw_observations, (list, tuple)):
            raise TypeError("observations must be a list or tuple")
        # Preserve tuple instances (including tuple subclasses) so the
        # manager can enforce its operation budget while traversing them. A
        # list is copied in full here intentionally: construction of this
        # public value must not silently truncate provider data.
        object.__setattr__(
            self,
            "observations",
            raw_observations
            if isinstance(raw_observations, tuple)
            else tuple(raw_observations),
        )


def _freeze_json_value(
    value: Any,
    *,
    depth: int = 0,
    budget: Optional[List[int]] = None,
    operation_budget: Optional[List[int]] = None,
) -> Any:
    """Validate and recursively freeze one JSON-safe observation value.

    This helper is intentionally private: providers return ordinary JSON
    values, while the manager owns the trust boundary and emits the frozen
    representation only after validation.

    ``budget`` is a shared mutable counter (a single-element list) tracking
    the number of JSON nodes still allowed for this payload. It is decremented
    on every entry so that pathological width/depth combinations (each
    container individually under MAX_MEMORY_OBSERVATION_ITEMS, but nested
    such that their product explodes) fail during recursion rather than
    after the whole structure has been materialized. When ``None``, the top
    call auto-initializes it to ``MAX_MEMORY_OBSERVATION_NODES``.

    ``operation_budget`` is an *additional* shared counter that spans many
    payloads in one operation (see ``MAX_MEMORY_OBSERVATION_OPERATION_NODES``).
    When supplied, every node decrements *both* counters and either exhausting
    raises. Passing an operation budget does NOT relax the per-payload budget:
    a single payload is still capped at ``MAX_MEMORY_OBSERVATION_NODES``.
    """
    if budget is None:
        budget = [MAX_MEMORY_OBSERVATION_NODES]
    if depth > MAX_MEMORY_OBSERVATION_DEPTH:
        raise ValueError("observation payload is too deeply nested")
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("observation payload has too many nodes")
    if operation_budget is not None:
        operation_budget[0] -= 1
        if operation_budget[0] < 0:
            raise ValueError("observation operation exhausted node budget")
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and len(value) > MAX_MEMORY_OBSERVATION_STRING_CHARS:
            raise ValueError("observation payload string is too long")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("observation payload contains a non-finite number")
        return value
    if isinstance(value, dict):
        if len(value) > MAX_MEMORY_OBSERVATION_ITEMS:
            raise ValueError("observation payload object has too many keys")
        frozen = {}
        for key, child in value.items():
            if not isinstance(key, str) or len(key) > MAX_MEMORY_OBSERVATION_STRING_CHARS:
                raise ValueError("observation payload object keys must be bounded strings")
            frozen[key] = _freeze_json_value(
                child,
                depth=depth + 1,
                budget=budget,
                operation_budget=operation_budget,
            )
        return _FrozenDict(frozen)
    if isinstance(value, list):
        if len(value) > MAX_MEMORY_OBSERVATION_ITEMS:
            raise ValueError("observation payload array has too many items")
        return tuple(
            _freeze_json_value(
                child,
                depth=depth + 1,
                budget=budget,
                operation_budget=operation_budget,
            )
            for child in value
        )
    raise TypeError("observation payload must contain only JSON-safe values")


def _thaw_json_value(value: Any) -> Any:
    """Return a JSON-native copy of a frozen observation value for sizing."""
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(child) for child in value]
    return value


def _freeze_memory_observation_payload(
    payload: Any,
    *,
    operation_budget: Optional[List[int]] = None,
) -> tuple[Any, int]:
    """Validate, freeze, and size a provider observation payload.

    ``MemoryManager`` uses this at the provider boundary. Providers should
    return ordinary JSON values and must not use this to bypass manager
    provenance checks.

    ``operation_budget`` is an optional shared node counter that lets a caller
    cap the total freeze traversal work across many candidate payloads in one
    operation (see ``MAX_MEMORY_OBSERVATION_OPERATION_NODES``). Every payload
    still gets its own ``MAX_MEMORY_OBSERVATION_NODES`` budget on top of it:
    the two counters are additive, not substitutive, so a caller cannot
    accidentally raise the per-payload cap by supplying an operation budget.
    """
    frozen = _freeze_json_value(payload, operation_budget=operation_budget)
    encoded = json.dumps(
        _thaw_json_value(frozen), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded) > MAX_MEMORY_OBSERVATION_BYTES:
        raise ValueError("observation payload is too large")
    return frozen, len(encoded)


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

    def prefetch(
        self, query: str, *, session_id: str = ""
    ) -> str | MemoryPrefetchResult:
        """Recall relevant context for the upcoming turn.

        Return formatted context, or a :class:`MemoryPrefetchResult` carrying
        that text plus optional bounded observations. Implementations should be
        fast — use background threads for the actual recall and return cached
        results here.
        """
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
        memory | user; ``metadata``: provenance such as write_origin, session_id, tool_name).
        For replace/remove, ``metadata["previous_content"]`` is the full entry selected
        under the native-store lock. Notifications follow a successful complete write
        or batch; each batch operation sees the preceding operation's result. Older
        callers may omit this field: ``old_text`` alone is not authoritative identity.
        """

    def backup_paths(self) -> List[str]:
        """Absolute paths of provider state OUTSIDE HERMES_HOME for ``hermes backup``/``import``
        (paths outside the home dir are skipped). MUST work without ``initialize()`` or network."""
        return []
