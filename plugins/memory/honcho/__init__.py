"""Honcho memory plugin — MemoryProvider for Honcho AI-native memory.

Cross-session user modeling with dialectic Q&A, semantic search, peer cards and
persistent conclusions; five tools (profile, search, reasoning, context, conclude).
Config chain: $HERMES_HOME/honcho.json -> ~/.honcho/config.json -> env vars.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from agent.memory_manager import sanitize_context
from agent.memory_provider import MemoryProvider, is_trivial_prompt
from plugins.memory.honcho.client import spawn_context_thread
from plugins.memory.honcho.dialectic import DialecticMixin
from plugins.memory.honcho.tool_schemas import ALL_TOOL_SCHEMAS
from tools.registry import tool_error

logger = logging.getLogger(__name__)


# Gateway-internal notifications arrive through the same user-role channel as genuine
# user messages; they are execution metadata and must never become durable memory.
# Deliberately anchored: a human discussing one of these strings mid-message is valid input.
_INTERNAL_GATEWAY_TURN_RE = re.compile(
    r"^\s*(?:"
    r"\[ASYNC (?:DELEGATION )?(?:BATCH )?COMPLETE[^\]]*\]|"
    r"\[CONTEXT COMPACTION[^\]]*\]|"
    r"\[CONTEXT SUMMARY\]:?|"
    r"\[PRIOR CONTEXT[^\]]*\]|"
    r"\[Your active task list was preserved across context compression\]|"
    r"\[IMPORTANT: Background process \d+ matched watch pattern[^\n]*|"
    r"A background fan-out of \d+ subagent\(s\) you dispatched earlier has finished\.|"
    r"A background subagent you dispatched earlier has finished\."
    r")",
    re.IGNORECASE,
)


def _is_internal_gateway_turn(text: str) -> bool:
    """Return True for machine-generated gateway/delegation notifications."""
    return bool(_INTERNAL_GATEWAY_TURN_RE.match(text or ""))


def _cfg_usable(cfg) -> bool:
    """Enabled with a credential or a self-hosted URL to talk to."""
    return bool(cfg.enabled and (cfg.api_key or cfg.base_url))


# Static per-mode system prompt text (prompt-cache friendly: never changes between turns).
_TOOL_GUIDE = (
    "Use honcho_profile for a quick factual snapshot, "
    "honcho_search for raw excerpts, honcho_context for raw peer context, "
    "honcho_reasoning for synthesized answers (pass reasoning_level "
    "minimal/low/medium/high/max — you pick the depth per call), "
    "honcho_conclude to save facts about the user."
)
_PROMPT_HEADERS = {
    "context": (
        "# Honcho Memory\nActive (context-injection mode). Relevant user context is automatically "
        "injected before each turn. No memory tools are available — context is managed automatically."
    ),
    "tools": (
        f"# Honcho Memory\nActive (tools-only mode). {_TOOL_GUIDE} "
        "No automatic context injection — you must use tools to access memory."
    ),
    "hybrid": (
        "# Honcho Memory\nActive (hybrid mode). Relevant context is auto-injected AND memory tools "
        f"are available. {_TOOL_GUIDE}"
    ),
}

# (context key, section header) for the injected base-context block, in display order.
_CONTEXT_SECTIONS = (
    ("summary", "Session Summary"),
    ("representation", "User Representation"),
    ("card", "User Peer Card"),
    ("ai_representation", "AI Self-Representation"),
    ("ai_card", "AI Identity Card"),
)

_PREWARM_QUERY = "Summarize what you know about this user. Focus on preferences, current projects, and working style."


class HonchoMemoryProvider(DialecticMixin, MemoryProvider):
    """Honcho AI-native memory with dialectic Q&A and persistent user modeling."""

    def backup_paths(self) -> List[str]:
        """Whole ~/.honcho dir (peer/session config when no profile-local honcho.json exists)."""
        try:
            from .client import resolve_global_config_path
            return [str(resolve_global_config_path().parent)]
        except Exception:
            return []

    def __init__(self, query_rewriter: Optional[Callable[[str], str]] = None):
        self._manager = None   # HonchoSessionManager
        self._config = None    # HonchoClientConfig
        self._session_key = ""
        self._query_rewriter = query_rewriter
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._memwrite_thread: Optional[threading.Thread] = None
        self._recall_mode = "hybrid"  # "context", "tools", or "hybrid"
        # Base context cache — refreshed on context_cadence, not frozen.
        self._base_context_cache: Optional[str] = None
        self._base_context_lock = threading.Lock()

        # Recall cadence state (overwritten from config in initialize()).
        self._turn_count = 0
        self._query_rewrite_enabled = False
        self._injection_frequency = "every-turn"  # or "first-turn"
        self._context_cadence = 1   # minimum turns between context API calls
        self._dialectic_cadence = 1  # backwards-compat fallback; wizard writes 2 on new configs
        self._dialectic_depth = 1   # .chat() calls per dialectic cycle (1-3)
        self._dialectic_depth_levels: list[str] | None = None  # per-pass reasoning levels
        self._reasoning_heuristic: bool = True  # scale base level by query length
        self._reasoning_level_cap: str = "high"  # ceiling for auto-selected level
        self._last_context_turn = self._last_dialectic_turn = -999
        # Liveness: monotonic start of the current prefetch thread, the turn the pending
        # result was fired at, and consecutive empty dialectic returns (drives backoff).
        self._prefetch_thread_started_at: float = 0.0
        self._prefetch_result_fired_at: int = -999
        self._dialectic_empty_streak: int = 0

        # Tools-only mode may defer session initialization until a tool call.
        self._session_initialized = False
        self._lazy_init_kwargs: Optional[dict] = None
        self._lazy_init_session_id: Optional[str] = None
        self._init_thread: Optional[threading.Thread] = None
        self._init_lock = threading.Lock()
        # Init auth failures live here because the failed manager is discarded.
        self._init_auth_failure: Optional[str] = None
        self._init_auth_notice_emitted = False
        self._cron_skipped = False  # cron and flush contexts disable the plugin entirely

    @property
    def name(self) -> str:
        return "honcho"

    def is_available(self) -> bool:
        """Check if Honcho is configured. No network calls."""
        try:
            from plugins.memory.honcho.client import HonchoClientConfig
            return _cfg_usable(HonchoClientConfig.from_global_config())
        except Exception:
            return False

    def save_config(self, values, hermes_home):
        """Merge ``values`` into $HERMES_HOME/honcho.json (Honcho SDK native format)."""
        from pathlib import Path
        from utils import atomic_json_write
        from plugins.memory.honcho.client import _read_config
        config_path = Path(hermes_home) / "honcho.json"
        try:
            existing = _read_config(config_path)
        except Exception:
            existing = {}
        atomic_json_write(config_path, {**existing, **values}, mode=0o600)

    def get_config_schema(self):
        return [
            {"key": "api_key", "description": "Honcho API key", "secret": True, "env_var": "HONCHO_API_KEY", "url": "https://app.honcho.dev"},
            {"key": "baseUrl", "description": "Honcho base URL (for self-hosted)"},
        ]

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Run the full Honcho setup wizard after provider selection."""
        import types
        from plugins.memory.honcho.cli import cmd_setup
        cmd_setup(types.SimpleNamespace())

    # ----- Session lifecycle -----

    def initialize(self, session_id: str, **kwargs) -> None:
        """Configure recall settings and start (or defer) Honcho session creation."""
        try:
            agent_context, platform = kwargs.get("agent_context", ""), kwargs.get("platform", "cli")
            if agent_context in {"cron", "flush"} or platform == "cron":
                logger.debug("Honcho skipped: cron/flush context (agent_context=%s, platform=%s)",
                             agent_context, platform)
                self._cron_skipped = True
                return

            from plugins.memory.honcho.client import HonchoClientConfig, get_honcho_client  # noqa: F401 — ImportError probe
            from plugins.memory.honcho.session import HonchoSessionManager  # noqa: F401

            cfg = HonchoClientConfig.from_global_config()
            if not _cfg_usable(cfg):
                logger.debug("Honcho not configured — plugin inactive")
                return

            self._config = cfg
            self._recall_mode = cfg.recall_mode
            logger.debug("Honcho recall_mode: %s", self._recall_mode)
            for name in ("injection_frequency", "context_cadence", "dialectic_cadence",
                         "dialectic_depth_levels", "reasoning_heuristic"):
                setattr(self, f"_{name}", getattr(cfg, name))
            self._query_rewrite_enabled = cfg.query_rewrite
            self._FIRST_TURN_BASE_TIMEOUT = cfg.first_turn_base_wait
            self._FIRST_TURN_DIALECTIC_CAP = cfg.first_turn_dialectic_wait
            self._dialectic_depth = max(1, min(cfg.dialectic_depth, 3))
            if cfg.reasoning_level_cap in self._LEVEL_ORDER:
                self._reasoning_level_cap = cfg.reasoning_level_cap

            # aiPeer comes from honcho.json only; SOUL.md is persona content, not identity config.
            self._lazy_init_kwargs = dict(kwargs)
            self._lazy_init_session_id = session_id
            self._session_key = self._resolve_session_key(cfg, session_id, **kwargs)

            # Session creation can block on Honcho/DB outages, so context/hybrid startup
            # fails open in a background thread. Tools-only mode has an explicit contract:
            # init_on_session_start=False stays lazy until the first tool call, True is eager.
            if self._recall_mode != "tools":
                self._start_session_init_background(wait_timeout=0.1)
            elif cfg.init_on_session_start:
                self._ensure_session()
            else:
                logger.debug("Honcho tools-only mode — deferring session init until first tool call")
        except ImportError:
            logger.debug("honcho-ai package not installed — plugin inactive")
        except Exception as e:
            logger.warning("Honcho init failed: %s", e)
            self._manager = None

    def _resolve_session_key(self, cfg, session_id: str, **kwargs) -> str:
        """Resolve the Honcho session key without touching the network."""
        return cfg.resolve_session_name(
            session_title=kwargs.get("session_title"), session_id=session_id,
            gateway_session_key=kwargs.get("gateway_session_key"),
        ) or session_id or "hermes-default"

    def _can_start_init(self) -> bool:
        return not (self._cron_skipped or self._session_initialized) and bool(self._config) and self._lazy_init_kwargs is not None

    def _run_session_init(self, label: str) -> bool:
        """Run _do_session_init with the deferred kwargs; on failure discard the manager
        and (for auth failures) keep the detail for the one-time notice."""
        from plugins.memory.honcho.session import HonchoAuthError

        init_kwargs = self._lazy_init_kwargs
        if init_kwargs is None:  # another init path already consumed the deferred kwargs
            return self._manager is not None
        try:
            self._do_session_init(self._config, self._lazy_init_session_id or "hermes-default", **dict(init_kwargs))
        except Exception as e:
            self._manager = None
            self._session_initialized = False
            detail: object = e
            if isinstance(e, HonchoAuthError):
                # Keep the auth detail so the one-time notice survives the manager discard.
                self._init_auth_failure = str(e)
                detail = "authentication rejected"
            logger.warning("Honcho %s session init failed: %s", label, detail)
            return False
        self._lazy_init_kwargs = self._lazy_init_session_id = None
        if self._init_auth_failure is not None:
            self._init_auth_failure = None
            self._init_auth_notice_emitted = False
        return True

    def _start_session_init_background(self, *, wait_timeout: float = 0.0) -> None:
        """Start session initialization in a daemon thread so a slow/down Honcho can't
        block agent construction or first prompt assembly. ``wait_timeout`` lets fast
        (mock) initializations finish before returning."""
        if not self._can_start_init():
            return
        with self._init_lock:
            if not self._can_start_init() or (self._init_thread and self._init_thread.is_alive()):
                return
            self._init_thread = spawn_context_thread(lambda: self._run_session_init("background"), name="honcho-session-init")
            self._init_thread.start()
            if wait_timeout > 0:
                self._init_thread.join(timeout=wait_timeout)

    def _ensure_session(self) -> bool:
        """Lazily initialize the Honcho session (tools-only mode). True when the manager is ready."""
        if self._manager and self._session_initialized:
            return True
        if not self._can_start_init() or (self._init_thread and self._init_thread.is_alive()):
            return False
        return self._run_session_init("lazy") and self._manager is not None

    def _do_session_init(self, cfg, session_id: str, **kwargs) -> None:
        """Shared session initialization for both eager and lazy paths."""
        from plugins.memory.honcho.client import get_honcho_client
        from plugins.memory.honcho.session import HonchoSessionManager

        self._manager = HonchoSessionManager(
            honcho=get_honcho_client(cfg), config=cfg, context_tokens=cfg.context_tokens,
            runtime_user_peer_name=kwargs.get("user_id") or None,
            runtime_user_peer_name_alt=kwargs.get("user_id_alt") or None,
        )
        self._session_key = self._resolve_session_key(cfg, session_id, **kwargs)
        logger.debug("Honcho session key resolved: %s", self._session_key)

        # The provider is not "ready" until this method returns: background startup sets
        # _manager before get_or_create/migration/prewarm finish, and lifecycle hooks must
        # not treat that partially initialized state as usable.
        session = self._manager.get_or_create(self._session_key)

        # Per-session strategy creates a fresh Honcho session every run, so a per-run
        # MEMORY.md/USER.md/SOUL.md upload would flood the backend with duplicates.
        if cfg.session_strategy == "per-session":
            logger.debug("Honcho memory file migration skipped: per-session strategy creates a fresh session per run (%s)",
                         self._session_key)
        elif not session.messages:
            try:
                from hermes_constants import get_hermes_home
                self._manager.migrate_memory_files(self._session_key, str(get_hermes_home() / "memories"))
                logger.debug("Honcho memory file migration attempted for new session: %s", self._session_key)
            except Exception as e:
                logger.debug("Honcho memory file migration skipped: %s", e)

        # Generic dialectic prewarm is incompatible with latest-message query rewriting,
        # which needs the first substantive user message.
        if self._recall_mode in {"context", "hybrid"}:
            if self._query_rewriter is None or not self._query_rewrite_enabled:
                self._spawn_dialectic(_PREWARM_QUERY, thread_name="honcho-prewarm-dialectic", fired_at=0,
                                      log_label="dialectic prewarm", use_query_rewrite=False)
                logger.debug("Honcho dialectic prewarm started for session: %s", self._session_key)
            else:
                logger.debug("Honcho generic dialectic prewarm skipped: awaiting first user message")

        self._session_initialized = True

    def _session_ready(self) -> bool:
        """Whether the manager/session key can be used safely. Background init sets
        ``_manager`` before get-or-create completes, so ``_session_initialized`` is the real
        guard; tests/legacy construction may inject a ready manager without the flag —
        allowed only with no init thread in flight."""
        if not self._manager or not self._session_key:
            return False
        return self._session_initialized or not (self._init_thread and self._init_thread.is_alive())

    def _writes_enabled(self) -> bool:
        """``saveMessages`` is the operator's hard write gate for every Honcho mutation path."""
        return not self._cron_skipped and getattr(self._config, "save_messages", True)

    def _ready_or_kick_init(self) -> bool:
        """True when writes may proceed; otherwise (outside tools mode) start background init."""
        if self._session_ready():
            return True
        if self._recall_mode != "tools":
            self._start_session_init_background()
        return False

    # ----- Prompt / prefetch -----

    def _format_first_turn_context(self, ctx: dict) -> str:
        """Format the prefetch context dict into a readable system prompt block."""
        return "\n\n".join(f"## {header}\n{ctx.get(key, '')}" for key, header in _CONTEXT_SECTIONS if ctx.get(key, ""))

    def system_prompt_block(self) -> str:
        """Static mode header + tool instructions (prompt-cache friendly).
        Live context (representation, card) is injected via prefetch()."""
        if self._cron_skipped or not (self._config or (self._manager and self._session_key)):
            return ""
        return _PROMPT_HEADERS.get(self._recall_mode, _PROMPT_HEADERS["hybrid"])

    def _first_turn_wait(self, base: float) -> float:
        """Turn-1 wait budget: a short request timeout may tighten, but never expand, it."""
        request_timeout = getattr(self._config, "timeout", None)
        return max(0.0, base if request_timeout is None else min(base, max(0.0, request_timeout)))

    def _fetch_base_context_layer(self, query: str, first_turn_base_deadline: float | None) -> str:
        """Layer 1: representation + card. The first fetch gets the remaining turn-1 budget;
        later turns consume the refresh queued by the previous turn."""
        with self._base_context_lock:
            first_base_fetch = self._base_context_cache is None
            if first_base_fetch:
                self._base_context_cache = ""
                self._last_context_turn = self._turn_count
            base_context = self._base_context_cache

        if not self._manager:
            return base_context

        def _adopt(ctx: dict) -> str:
            """Cache a fresh context dict's formatted block; keep the old text if it formats empty."""
            formatted = self._format_first_turn_context(ctx)
            if formatted:
                with self._base_context_lock:
                    self._base_context_cache = formatted
            return formatted or base_context

        if not first_base_fetch:
            fresh_ctx = self._manager.pop_context_result(self._session_key)
            return _adopt(fresh_ctx) if fresh_ctx else base_context

        ctx_holder: dict[str, dict] = {}

        def _fetch_base() -> None:
            ctx_holder["ctx"] = ctx = self._manager.get_prefetch_context(self._session_key, query or None) or {}
            if ctx:
                self._manager.set_context_result(self._session_key, ctx)

        bt = self._spawn_write(_fetch_base, "honcho-base-first", "Honcho first-turn base context failed: %s")
        base_wait = max(0.0, first_turn_base_deadline - time.monotonic()) if first_turn_base_deadline is not None else 0.0
        bt.join(timeout=base_wait)
        if ctx := ctx_holder.get("ctx"):
            self._manager.pop_context_result(self._session_key)
            return _adopt(ctx)
        if bt.is_alive():
            logger.debug("Honcho first-turn base context still running after %.1fs — will surface on next turn", base_wait)
        return base_context

    def _first_turn_dialectic_wait(self, query: str) -> None:
        """Turn 1 only: reuse an in-flight prewarm or start one dialectic, then wait briefly.
        Unfinished work stays async and surfaces on a later turn."""
        with self._prefetch_lock:
            prewarm_landed = bool(self._prefetch_result)
        if prewarm_landed and self._last_dialectic_turn == -999:
            self._last_dialectic_turn = self._turn_count
        if self._last_dialectic_turn != -999 or not query:
            return

        dia_wait = self._first_turn_wait(self._FIRST_TURN_DIALECTIC_CAP)
        if not self._thread_is_live():
            self._spawn_dialectic(query, thread_name="honcho-prefetch-first", fired_at=self._turn_count,
                                  log_label="first-turn dialectic")
        if (live := self._prefetch_thread) is not None:
            live.join(timeout=dia_wait)
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            logger.debug("Honcho first-turn dialectic still running after %.1fs — will surface on next turn", dia_wait)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Base context (representation + card, refreshed on context_cadence) plus the
        dialectic supplement (refreshed on dialectic_cadence), within the context budget.
        Empty in tools-only mode."""
        if self._cron_skipped or self._recall_mode == "tools":
            return ""

        first_turn_base_deadline = (time.monotonic() + self._first_turn_wait(self._FIRST_TURN_BASE_TIMEOUT)
                                    if self._turn_count <= 1 else None)

        if not self._session_ready():
            # Only turn 1 may wait for session init; later turns fail open.
            self._start_session_init_background()
            if first_turn_base_deadline is not None and self._init_thread is not None:
                self._init_thread.join(timeout=max(0.0, first_turn_base_deadline - time.monotonic()))
            if not self._session_ready():
                # A failed auth init still owes the user the one-time notice.
                return self._pop_auth_notice()

        # Trivial turns start no work, but may consume a ready pending result.
        if self._is_trivial_prompt(query):
            ready = self._consume_pending_dialectic()
            return self._truncate_to_budget(ready) if ready else ""

        # One-time notice, relayed by the model, that auth is dead and memory is paused.
        parts = [self._pop_auth_notice()]
        # First-turn mode suppresses only the base layer; dialectic is independent.
        if not (self._injection_frequency == "first-turn" and self._turn_count > 1):
            parts.append(self._fetch_base_context_layer(query, first_turn_base_deadline))
        self._first_turn_dialectic_wait(query)
        # Consume only results that are already ready; later turns never wait.
        parts.append(self._consume_pending_dialectic())
        parts = [p for p in parts if p and p.strip()]
        return self._truncate_to_budget("\n\n".join(parts)) if parts else ""

    def _pop_auth_notice(self) -> str:
        """One-time model-facing notice that Honcho auth expired and memory is paused."""
        # getattr (not a direct call): test fakes install minimal managers without pop_auth_notice.
        pop = getattr(self._manager, "pop_auth_notice", None)
        msg = pop() if callable(pop) else None
        if not isinstance(msg, str) or not msg:
            # Init failures discard the manager; the provider kept the detail.
            if self._init_auth_failure is None or self._init_auth_notice_emitted:
                return ""
            self._init_auth_notice_emitted = True
            msg = self._init_auth_failure
        return ("[Honcho memory status] Authentication with the Honcho memory backend has expired and automatic "
                f"token refresh failed, so memory sync and recall are paused. Reason: {msg}\n"
                "Tell the user (once) that Honcho memory is paused and that running 'hermes honcho setup' "
                "to re-authenticate will restore it.")

    def _truncate_to_budget(self, text: str) -> str:
        """Truncate text to the context_tokens budget (≈4 chars/token) at a word boundary."""
        if not self._config or not self._config.context_tokens:
            return text
        budget_chars = self._config.context_tokens * 4
        if len(text) <= budget_chars:
            return text
        truncated = text[:budget_chars]
        last_space = truncated.rfind(" ")
        return (truncated[:last_space] if last_space > budget_chars * 0.8 else truncated) + " …"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Fire background prefetch threads for the upcoming turn.
        Context and dialectic refreshes have independent cadence controls."""
        if self._cron_skipped or self._recall_mode == "tools":
            return
        if not self._session_ready() or not query:
            self._start_session_init_background()
            return
        # Trivial prompts don't warrant either a context refresh or a dialectic call.
        if self._is_trivial_prompt(query):
            return

        # First-turn-only base context never needs a later refresh.
        context_due = self._context_cadence <= 1 or (self._turn_count - self._last_context_turn) >= self._context_cadence
        if self._injection_frequency != "first-turn" and context_due:
            self._last_context_turn = self._turn_count
            try:
                self._manager.prefetch_context(self._session_key, query)
            except Exception as e:
                logger.debug("Honcho context prefetch failed: %s", e)

        # Dialectic layer: a hung call older than timeout × multiplier counts as dead.
        if self._thread_is_live():
            logger.debug("Honcho dialectic prefetch skipped: prior thread still running")
            return
        # Cadence gate, widened by the empty-streak backoff so a persistently silent
        # backend doesn't retry every turn forever.
        effective = self._effective_cadence()
        if (self._turn_count - self._last_dialectic_turn) < effective:
            logger.debug("Honcho dialectic prefetch skipped: effective cadence %d (base %d, empty streak %d), turns since last: %d",
                         effective, self._dialectic_cadence, self._dialectic_empty_streak,
                         self._turn_count - self._last_dialectic_turn)
            return
        self._spawn_dialectic(query, thread_name="honcho-prefetch", fired_at=self._turn_count, log_label="prefetch")

    # Shared with the core prefetch gate so the two classifiers can never drift apart.
    _is_trivial_prompt = staticmethod(is_trivial_prompt)

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Track turn count for cadence and injection_frequency logic."""
        self._turn_count = turn_number

    # ----- Writes -----

    @staticmethod
    def _chunk_message(content: str, limit: int) -> list[str]:
        """Split content to fit the Honcho message limit, cutting at paragraph, then
        sentence, then word boundaries; continuation chunks get a "[continued] " prefix
        so Honcho's representation engine can reconstruct the full message."""
        if len(content) <= limit:
            return [content]

        prefix = "[continued] "
        chunks: list[str] = []
        remaining, first = content, True
        while remaining:
            effective = limit if first else limit - len(prefix)
            if len(remaining) <= effective:
                chunks.append(remaining if first else prefix + remaining)
                break

            segment = remaining[:effective]
            # Paragraph, then sentence (keeping ". "), then word boundary; else a hard cut.
            for sep in ("\n\n", ". ", " "):
                cut = segment.rfind(sep)
                if cut >= 0 and sep == ". ":
                    cut += 2
                if cut >= effective * 0.3:
                    break
            else:
                cut = effective

            chunk = remaining[:cut].rstrip()
            remaining = remaining[cut:].lstrip()
            chunks.append(chunk if first else prefix + chunk)
            first = False

        return chunks

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Record the conversation turn in Honcho (non-blocking), chunking messages that
        exceed the Honcho API limit. Honors saveMessages: false."""
        if not self._writes_enabled():
            return
        if _is_internal_gateway_turn(user_content):
            logger.debug("Honcho sync skipped machine-generated gateway turn")
            return
        if not self._ready_or_kick_init():
            return

        msg_limit = self._config.message_max_chars if self._config else 25000
        clean_user_content = sanitize_context(user_content or "").strip()
        clean_assistant_content = sanitize_context(assistant_content or "").strip()
        # Skip only when the whole turn is empty: an interrupted or tool-only turn can have
        # an empty assistant side, and the user's message must still be persisted.
        if not clean_user_content and not clean_assistant_content:
            return

        def _sync():
            session = self._manager.get_or_create(self._session_key)
            for role, content in (("user", clean_user_content), ("assistant", clean_assistant_content)):
                for chunk in self._chunk_message(content, msg_limit) if content else ():
                    session.add_message(role, chunk)
            # save() (not _flush_session) so writeFrequency batching is honored.
            self._manager.save(session)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = self._spawn_write(_sync, "honcho-sync", "Honcho sync_turn failed: %s")

    @staticmethod
    def _spawn_write(fn: Callable[[], None], name: str, fail_msg: str) -> threading.Thread:
        """Run a Honcho write off-thread; failures are debug-logged, never raised into the turn."""
        def _run():
            try:
                fn()
            except Exception as e:
                logger.debug(fail_msg, e)

        thread = spawn_context_thread(_run, name=name)
        thread.start()
        return thread

    def on_memory_write(
        self, action: str, target: str, content: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in user-profile writes as Honcho conclusions (``metadata`` accepted
        for interface compatibility, not yet threaded into the conclusion payload)."""
        if action != "add" or target != "user" or not content:
            return
        if not self._writes_enabled() or not self._ready_or_kick_init():
            return
        self._memwrite_thread = self._spawn_write(lambda: self._manager.create_conclusion(self._session_key, content),
                                                  "honcho-memwrite", "Honcho memory mirror failed: %s")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Flush all pending messages to Honcho on session end."""
        if not self._writes_enabled() or not self._manager:
            return
        if not self._session_initialized and self._init_thread and self._init_thread.is_alive():
            return
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)
        try:
            self._manager.flush_all()
        except Exception as e:
            logger.debug("Honcho session-end flush failed: %s", e)

    # ----- Tools -----

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Tool schemas by recall_mode; context-only mode exposes no Honcho tools."""
        if self._cron_skipped or self._recall_mode == "context":
            return []
        return list(ALL_TOOL_SCHEMAS)

    def _empty_profile_hint(self, peer: str) -> Dict[str, Any]:
        """Diagnostic hint for an empty honcho_profile card, so the model can explain WHY
        instead of surfacing a cryptic "no facts" to the user. Likely causes, in order:
        observation disabled for the peer; card not accumulated yet (fresh peer / few
        dialectic cycles); self-hosted Honcho < 3.x without peer-card support."""
        cfg = self._config
        reasons: List[str] = []
        kind = "user" if peer == "user" else "ai"
        if cfg is not None and not (getattr(cfg, f"{kind}_observe_me", True) or getattr(cfg, f"{kind}_observe_others", True)):
            reasons.append(f"observation is disabled for peer '{peer}' (user_observe_me/ai_observe_me in config)")
        cadence, turn = self._dialectic_cadence, self._turn_count
        if turn < max(2, cadence):
            reasons.append(f"this session has only {turn} turn(s); peer cards accumulate as the dialectic "
                           f"layer reasons over conversation history (cadence every {cadence} turn(s))")
        if not reasons:
            reasons.append("peer card has no facts yet — Honcho's dialectic layer builds this over time from "
                           "observed turns; self-hosted Honcho < 3.x does not support peer cards at all")
        return {
            "result": "No profile facts available yet.",
            "hint": ("This is not an error.  " + "; ".join(reasons)
                     + ".  Try honcho_reasoning for a synthesized answer, or honcho_search to query raw conversation excerpts."),
        }

    def _tool_profile(self, args: dict) -> str:
        peer = args.get("peer", "user")
        if card_update := args.get("card"):
            result = self._manager.set_peer_card(self._session_key, card_update, peer=peer)
            if result is None:
                return tool_error("Failed to update peer card.")
            return json.dumps({"result": f"Peer card updated ({len(result)} facts).", "card": result})
        card = self._manager.get_peer_card(self._session_key, peer=peer)
        return json.dumps({"result": card} if card else self._empty_profile_hint(peer))

    def _tool_search(self, args: dict) -> str:
        if not (query := (args.get("query") or "").strip()):
            return tool_error("Missing required parameter: query")
        max_tokens = min(int(args.get("max_tokens", 800)), 2000)
        result = self._manager.search_context(self._session_key, query, max_tokens=max_tokens, peer=args.get("peer", "user"))
        return json.dumps({"result": result or "No relevant context found."})

    def _tool_reasoning(self, args: dict) -> str:
        from plugins.memory.honcho.session import HonchoAuthError

        if not (query := (args.get("query") or "").strip()):
            return tool_error("Missing required parameter: query")
        try:
            # Explicit reasoning bypasses the automatic-injection cap, and surfaces
            # timeouts/server errors as errors rather than an indistinguishable "no result".
            result = self._manager.dialectic_query(
                self._session_key, query, reasoning_level=args.get("reasoning_level"),
                peer=args.get("peer", "user"), apply_injection_cap=False, raise_errors=True,
            )
        except HonchoAuthError:
            raise  # rendered by handle_tool_call's auth-specific handler
        except Exception as e:
            logger.warning("honcho_reasoning failed: %s", e)
            return tool_error(
                f"Honcho reasoning query failed ({e}). This is a backend error, not an empty result — "
                "the peer may still have relevant context. Slow dialectic calls at higher reasoning levels "
                "can exceed the configured timeout; consider a lower reasoning_level or raising the "
                "'timeout' value in honcho.json."
            )
        # Auto-injection respects the cadence gap after an explicit call.
        self._last_dialectic_turn = self._turn_count
        return json.dumps({"result": result or "No result from Honcho."})

    def _tool_context(self, args: dict) -> str:
        ctx = self._manager.get_session_context(self._session_key, peer=args.get("peer", "user"))
        if not ctx:
            return json.dumps({"result": "No context available yet."})
        parts = [f"## {header}\n{ctx[key]}"
                 for key, header in (("summary", "Summary"), ("representation", "Representation"), ("card", "Card"))
                 if ctx.get(key)]
        if recent := ctx.get("recent_messages"):
            parts.append("## Recent messages\n" + "\n".join(f"  [{m['role']}] {m['content'][:200]}" for m in recent[-5:]))
        return json.dumps({"result": "\n\n".join(parts) or "No context available."})

    def _tool_conclude(self, args: dict) -> str:
        delete_id = (args.get("delete_id") or "").strip()
        conclusion = args.get("conclusion", "").strip()
        list_mode = bool(args.get("list"))
        peer = args.get("peer", "user")
        if sum([bool(delete_id), bool(conclusion), list_mode]) != 1:
            return tool_error("Exactly one of conclusion, delete_id, or list must be provided.")
        query = (args.get("query") or "").strip()
        if query and not list_mode:
            return tool_error("query is only valid when list is true.")

        if list_mode:
            return json.dumps({"conclusions": self._manager.list_conclusions(self._session_key, query=query or None, peer=peer)})
        if delete_id:
            if self._manager.delete_conclusion(self._session_key, delete_id, peer=peer):
                return json.dumps({"result": f"Conclusion {delete_id} deleted."})
            return tool_error(f"Failed to delete conclusion {delete_id}.")
        if self._manager.create_conclusion(self._session_key, conclusion, peer=peer):
            return json.dumps({"result": f"Conclusion saved for {peer}: {conclusion}"})
        return tool_error("Failed to save conclusion.")

    _TOOL_HANDLERS = {
        "honcho_profile": _tool_profile,
        "honcho_search": _tool_search,
        "honcho_reasoning": _tool_reasoning,
        "honcho_context": _tool_context,
        "honcho_conclude": _tool_conclude,
    }

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        """Dispatch a Honcho tool call, lazily initializing the session in tools-only mode."""
        from plugins.memory.honcho.session import HonchoAuthError

        if self._cron_skipped:
            return tool_error("Honcho is not active (cron context).")
        if not self._session_initialized:
            if self._init_thread and self._init_thread.is_alive():
                return tool_error("Honcho session is still initializing; try again shortly.")
            if not self._ensure_session():
                return tool_error(f"Honcho memory authentication failed: {self._init_auth_failure}"
                                  if self._init_auth_failure else "Honcho session could not be initialized.")
        if not self._manager or not self._session_key:
            return tool_error("Honcho is not active for this session.")
        if (handler := self._TOOL_HANDLERS.get(tool_name)) is None:
            return tool_error(f"Unknown tool: {tool_name}")
        try:
            return handler(self, args)
        except HonchoAuthError as e:
            # Never report an auth failure as an empty result; the model would read it as "no memory".
            logger.error("Honcho tool %s failed: authentication rejected", tool_name)
            return tool_error(f"Honcho memory authentication failed: {e}")
        except Exception as e:
            logger.error("Honcho tool %s failed: %s", tool_name, e)
            return tool_error(f"Honcho {tool_name} failed: {e}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread, self._memwrite_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        manager = self._manager
        if not manager or (self._init_thread and self._init_thread.is_alive() and not self._session_initialized):
            return
        # saveMessages: false skips persistence, but the async-writer thread must still
        # be joined so daemon threads aren't left blocked in httpx I/O at interpreter exit.
        with contextlib.suppress(Exception):
            if getattr(self._config, "save_messages", True):
                manager.shutdown()  # flush_all() + join the writer
            else:
                manager.stop_async_writer()


def register(ctx) -> None:
    """Register Honcho as a memory provider plugin."""
    from plugins.memory.query_rewrite import rewrite_memory_query

    ctx.register_memory_provider(HonchoMemoryProvider(query_rewriter=rewrite_memory_query))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

CONCLUDE_SCHEMA = {
    "name": "honcho_conclude",
    "description": (
        "Write, delete, or list CONCLUSIONS — persistent, derived facts about a peer that "
        "feeds their long-term profile (card + representation). Use this to record "
        "something durable you've learned about the peer (a stable preference, a "
        "correction, a standing constraint) so future sessions carry it forward. "
        "You MUST pass exactly one of `conclusion` (to create), `delete_id` (to "
        "delete), or `list` (to list/search); any other combination is an error. "
        "A deletion ID is an opaque server-generated string: first call with `list=true` "
        "and optionally `query`, then pass the returned ID as `delete_id`. "
        "Deletion exists only for "
        "PII removal — for merely wrong facts, write a corrected conclusion instead; "
        "Honcho self-heals contradictions over time. This is a WRITE tool: to read "
        "the profile use honcho_profile / honcho_context, and to search what was "
        "said use honcho_search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {
                "type": "string",
                "description": "A factual statement to persist. Provide this when creating a conclusion. Do not send it together with delete_id or list.",
            },
            "delete_id": {
                "type": "string",
                "description": "Conclusion ID to delete for PII removal. Provide this when deleting a conclusion. Do not send it together with conclusion or list. Get this id from a prior `list` call — never guess it.",
            },
            "list": {
                "type": "boolean",
                "description": "Set to true to list or search stored conclusions (with their ids) instead of creating or deleting one. Do not send together with conclusion or delete_id.",
            },
            "query": {
                "type": "string",
                "description": "Optional semantic search query, used only when `list` is true. Omit to list the most recent conclusions instead of searching.",
            },
            "peer": {
                "type": "string",
                "description": "The peer the conclusion is ABOUT. Built-in aliases: 'user' (default), 'ai'. Or pass any peer ID from this workspace.",
            },
        },
        "required": [],
    },
}

CONTEXT_SCHEMA = {
    "name": "honcho_context",
    "description": (
        "Retrieve the standing SNAPSHOT Honcho holds for the current session — "
        "session summary, the peer's representation, the peer card, and the most "
        "recent messages — in one call. No query, no LLM synthesis (cheaper than "
        "honcho_reasoning). Use it to orient yourself on what Honcho currently "
        "knows about this conversation and peer. This is a fixed snapshot, not a "
        "search: to look up a specific past fact use honcho_search; to ask a "
        "question and get a synthesized answer use honcho_reasoning; for just the "
        "compact card use honcho_profile."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "peer": {
                "type": "string",
                "description": "Peer to query. Built-in aliases: 'user' (default), 'ai'. Or pass any peer ID from this workspace.",
            },
        },
        "required": [],
    },
}

PROFILE_SCHEMA = {
    "name": "honcho_profile",
    "description": (
        "Read or write a peer's CARD — a short, curated list of standing facts "
        "about that peer (name, role, preferences, communication style, recurring "
        "patterns). This is the cheapest, fastest Honcho call: no query, no LLM, "
        "just the current card. Pass `card` to overwrite it; omit `card` to read. "
        "An empty read returns a `hint` explaining why (observation disabled, fresh "
        "peer, representation still warming up) — that is NOT an error; the card "
        "accumulates over time from observed conversation. "
        "Related tools: honcho_context for the fuller standing snapshot (card + "
        "representation + summary + recent messages); honcho_search to find "
        "specific things that were actually said; honcho_reasoning for a "
        "synthesized answer to a question."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "peer": {
                "type": "string",
                "description": "Peer to query. Built-in aliases: 'user' (default), 'ai'. Or pass any peer ID from this workspace.",
            },
            "card": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New peer card as a list of fact strings. Omit to read the current card.",
            },
        },
        "required": [],
    },
}

REASONING_SCHEMA = {
    "name": "honcho_reasoning",
    "description": (
        "Ask Honcho's dialectic agent a natural-language question about a peer and "
        "get back a SYNTHESIZED answer. This is the only Honcho tool that runs an "
        "LLM: it agentically searches both raw messages and derived conclusions, "
        "reasons over them, and writes a prose answer — so it is the slowest and "
        "most expensive call (seconds + tokens). Reach for it for nuanced or "
        "open-ended questions ('how does this person prefer to receive feedback?', "
        "'what's their relationship to project X?') where you want Honcho to do the "
        "synthesis. For a specific fact that was stated, prefer honcho_search "
        "(cheap, raw excerpts, you synthesize). For standing profile facts, prefer "
        "honcho_profile / honcho_context (no LLM). "
        "Pass reasoning_level to control depth: minimal (fast/cheap), low (default), "
        "medium, high, max (deep/expensive). Omit for the configured default."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A natural language question.",
            },
            "reasoning_level": {
                "type": "string",
                "description": (
                    "Override the default reasoning depth. "
                    "Omit to use the configured default (typically low).\n"
                    "reasoning_level parameter guide:\n"
                    "- minimal: use ONLY for a single quick factual lookup (e.g. "
                    "'what is the user's name'). Honcho hard-caps this tier's output "
                    "at 250 tokens combined with the model's own hidden reasoning "
                    "tokens — a multi-part answer can get cut off mid-thought before "
                    "it even reaches the final-answer phase, especially on models "
                    "with reasoning/thinking enabled.\n"
                    "- low/medium/high/max: use for anything requiring a synthesized, "
                    "multi-fact, or summary-style answer (e.g. 'summarize known facts "
                    "about this peer', 'what are their communication preferences'). "
                    "These tiers have no output-token cap of their own (fall back to "
                    "Honcho's 8192-token global default), so they don't have "
                    "minimal's cutoff failure mode.\n"
                    "  - low: straightforward questions with clear answers\n"
                    "  - medium: multi-aspect questions requiring synthesis across observations\n"
                    "  - high: complex behavioral patterns, contradictions, deep analysis\n"
                    "  - max: thorough audit-level analysis, leave no stone unturned\n"
                    "Default to at least 'low' unless the query is genuinely a single "
                    "fact lookup."
                ),
                "enum": ["minimal", "low", "medium", "high", "max"],
            },
            "peer": {
                "type": "string",
                "description": "Peer to query. Built-in aliases: 'user' (default), 'ai'. Or pass any peer ID from this workspace.",
            },
        },
        "required": ["query"],
    },
}

SEARCH_SCHEMA = {
    "name": "honcho_search",
    "description": (
        "Hybrid (semantic + keyword) search over a peer's actual message "
        "history across ALL past sessions they took part in — not just the "
        "current one. Returns RRF-ranked raw message excerpts (what was "
        "literally said, including the assistant's own messages about the "
        "peer), no LLM synthesis. Cheaper and faster than honcho_reasoning. "
        "Use this to recall specific past facts — 'what did I say about X', "
        "'what was the regimen/decision/config we settled on' — and reason "
        "over the excerpts yourself. For nuanced questions needing synthesis, "
        "use honcho_reasoning instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look for — a topic, keyword, name, or natural-language description of the fact you're trying to recall.",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Approximate budget for returned excerpts (default 800, max 2000). Larger budgets return more/longer ranked snippets.",
            },
            "peer": {
                "type": "string",
                "description": "Whose history to search. Built-in aliases: 'user' (default), 'ai'. Or pass any peer ID from this workspace. Spans every session that peer took part in.",
            },
        },
        "required": ["query"],
    },
}


_PLUGIN_COMPAT_LAZY = {
    'TRIVIAL_PROMPT_RE': ('agent.memory_provider', 'TRIVIAL_PROMPT_RE'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
