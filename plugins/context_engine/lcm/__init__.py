"""LCM — Lossless Context Management engine plugin.

Three-layer memory hierarchy for hermes-agent context compression:

  L1 Hot   — LLM summarization (in-turn, soft/hard thresholds)
  L2 Warm  — Dense Associative Memory (Modern Hopfield network, per-session)
  L3 Cold  — HRR persistent knowledge store (cross-session)

Activation: set ``context.engine: lcm`` in config.yaml.

All 7 LCM tools (lcm_expand, lcm_pin, lcm_forget, lcm_search, lcm_budget,
lcm_toc, lcm_focus) are exposed via the ContextEngine ABC's get_tool_schemas()
and handle_tool_call() — no run_agent.py patches needed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

from plugins.context_engine.lcm.config import LcmConfig
from plugins.context_engine.lcm.engine import LcmEngine, CompactionAction, ContextEntry
from plugins.context_engine.lcm.dag import SummaryDag, SummaryNode, MessageId
from plugins.context_engine.lcm.store import ImmutableStore
from plugins.context_engine.lcm.tools import LCM_TOOL_SCHEMAS
try:
    from plugins.context_engine.lcm.hrr.tools import MEMORY_TOOL_HANDLERS
    from plugins.context_engine.lcm.hrr.schemas import MEMORY_TOOL_SCHEMAS
except (ImportError, Exception):
    MEMORY_TOOL_HANDLERS: dict = {}  # type: ignore[assignment]
    MEMORY_TOOL_SCHEMAS: dict = {}  # type: ignore[assignment]
from plugins.context_engine.lcm.summarizer import Summarizer, SummarizerConfig
from plugins.context_engine.lcm.tokens import TokenEstimator, TokenEstimatorConfig, estimate_messages_tokens_rough
from plugins.context_engine.lcm.semantic import SemanticIndex, SemanticIndexConfig, NoOpSemanticIndex, create_semantic_index

logger = logging.getLogger(__name__)


class LcmContextEngine(ContextEngine):
    """LCM adapter that implements the ContextEngine ABC.

    Delegates all real work to an internal LcmEngine instance.
    The ABC's compress() method is the main entry point — it handles
    both ingestion of new messages and compaction.
    """
    def __init__(self, **kwargs: Any) -> None:
        # Extract config from kwargs (passed by plugin loader or agent init)
        lcm_cfg = kwargs.get("lcm_config", {})
        model = kwargs.get("model", "")
        provider = kwargs.get("provider", "")
        context_length = kwargs.get("context_length", 128_000)

        if isinstance(lcm_cfg, dict):
            self._config = LcmConfig.from_dict(lcm_cfg)
        elif isinstance(lcm_cfg, LcmConfig):
            self._config = lcm_cfg
        else:
            self._config = LcmConfig()

        # Honour lcm.enabled=false — build a stub engine only so other methods
        # (reset, get_status) remain safe to call, but skip all heavy init and
        # suppress tool surface.
        self._disabled: bool = not self._config.enabled

        self._engine = LcmEngine(
            config=self._config,
            model=model,
            provider=provider,
            context_length=context_length,
        )

        if not self._disabled:
            # Try to init HRR persistent store (L3)
            try:
                from plugins.context_engine.lcm.hrr.store import MemoryStore as _HrrStore
                self._engine.hrr_store = _HrrStore()
            except (ImportError, Exception):
                pass

        # Track ingestion state — how many messages we've ingested so far
        self._ingested_count: int = 0

        # Map ABC attributes to engine state
        self.context_length = context_length
        self.threshold_percent = self._config.tau_soft
        self.protect_last_n = self._config.protect_last_n
        self.threshold_tokens = int(context_length * self._config.tau_soft)

        if not self._disabled:
            # Try to init DAM retriever (L2)
            self._try_init_dam()

    def _try_init_dam(self) -> None:
        """Best-effort DAM initialization (requires numpy)."""
        try:
            from plugins.context_engine.lcm.dam import DenseAssociativeMemory, MessageEncoder, DAMRetriever
            self._engine.retriever = DAMRetriever(
                net=DenseAssociativeMemory(nv=2048, nh=64),
                enc=MessageEncoder(nv=2048),
            )
        except (ImportError, Exception):
            pass

    # ── Identity ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "lcm"

    # ── Core ABC interface ────────────────────────────────────────────────

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update token tracking from API response usage."""
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total = prompt_tokens + completion_tokens

        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens
        self.last_total_tokens = total

        # Also update the engine's token estimator if it tracks real usage
        self._engine.token_estimator._last_prompt_tokens = prompt_tokens

    def should_compress(self, prompt_tokens: int = None) -> bool:  # type: ignore[override]
        """Check if compaction should fire (covers both async and blocking).

        Uses the real token count from the API response (prompt_tokens) when
        available, falling back to the engine's internal active_tokens(). The
        engine's active list may be empty before the first compress() call
        (messages are only ingested during compress()), so relying solely on
        active_tokens() creates a chicken-and-egg where compression never fires.
        """
        if self._disabled:
            return False

        if prompt_tokens is not None:
            self.last_prompt_tokens = prompt_tokens

        # Use real API-reported token count if available — it reflects the
        # actual context size including messages the engine hasn't ingested yet.
        real_tokens = prompt_tokens if prompt_tokens and prompt_tokens > 0 else None

        if real_tokens is not None:
            budget = self._engine.context_length
            ratio = real_tokens / budget if budget > 0 else 0.0
            return ratio >= self._config.tau_soft

        # Fallback: check engine's internal state (works after first compress())
        action = self._engine.check_thresholds()
        return action != CompactionAction.NONE

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        """Ingest any new messages, then compact if needed.

        This is the main entry point called by run_agent.py.
        We ingest messages the engine hasn't seen yet, then
        run compaction and return the active message list.
        """
        if self._disabled:
            return messages

        # Ingest new messages since last call
        new_count = len(messages) - self._ingested_count
        if new_count > 0:
            for msg in messages[self._ingested_count:]:
                self._engine.ingest(msg)
            self._ingested_count = len(messages)

        # Check if compaction should fire
        action = self._engine.check_thresholds()
        if action == CompactionAction.ASYNC:
            self._engine.async_compact()
        elif action == CompactionAction.BLOCKING:
            self._engine.auto_compact()
            self.compression_count += 1

        # Return the engine's active message list
        result = self._engine.active_messages()

        # Update token tracking
        active_tokens = self._engine.active_tokens()
        self.last_prompt_tokens = active_tokens
        self.threshold_tokens = int(self.context_length * self._config.tau_soft)

        return result

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """Quick estimate before the API call.

        Checks real messages since the engine's active list may be empty
        before the first compress() call.
        """
        # If engine has ingested messages, use its active token count
        if self._engine.active:
            active_tokens = self._engine.active_tokens()
            return active_tokens > int(self.context_length * self._config.tau_soft)

        # Otherwise estimate from the raw messages passed in
        estimated = self._engine.token_estimator.estimate(messages)
        return estimated > int(self.context_length * self._config.tau_soft)

    # ── Session lifecycle ─────────────────────────────────────────────────

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Load persisted state and config for the session."""
        # Try to load LCM config from config.yaml
        try:
            from hermes_cli.config import load_config
            _config = load_config()
            _lcm_cfg = _config.get("lcm", {})
            if isinstance(_lcm_cfg, dict) and _lcm_cfg:
                self._config = LcmConfig.from_dict(_lcm_cfg)
                self.threshold_percent = self._config.tau_soft
                self.protect_last_n = self._config.protect_last_n
                # Propagate the reloaded config to the live engine so compaction
                # reads the user-supplied thresholds (tau_soft, tau_hard, etc.)
                # rather than the constructor-time defaults.
                self._engine.config = self._config
        except Exception:
            pass  # Config not available, use defaults

        # Update context length from kwargs or config
        context_length = kwargs.get("context_length")
        if context_length:
            self.context_length = context_length
            self._engine.token_estimator.config.context_length = context_length

        self.threshold_tokens = int(self.context_length * self._config.tau_soft)

        # Distinguish fresh session start from compression-driven rollover.
        # During compression rollover run_agent.py passes boundary_reason="compression"
        # so we preserve existing engine state instead of re-initializing.
        boundary_reason = kwargs.get("boundary_reason", "")
        is_compression_rollover = boundary_reason == "compression"

        # Try session rebuild from persisted state
        hermes_home = kwargs.get("hermes_home")
        if hermes_home and session_id:
            try:
                import json
                from pathlib import Path
                session_file = Path(hermes_home) / "sessions" / f"session_{session_id}.json"
                if session_file.exists():
                    session_data = json.loads(session_file.read_text(encoding="utf-8"))
                    self._engine = LcmEngine.rebuild_from_session(
                        session_data,
                        self._config,
                        model=self._engine.model,
                        provider=self._engine.provider,
                        context_length=self.context_length,
                    )
                    # Re-attach L3 HRR store — rebuild() returns a fresh LcmEngine
                    # that has no hrr_store; without this, resumed sessions lose all
                    # cross-session memory (same root cause as config not propagating
                    # after rebuild, fixed in round-3).
                    try:
                        from plugins.context_engine.lcm.hrr.store import MemoryStore as _HrrStore
                        self._engine.hrr_store = _HrrStore()
                    except (ImportError, Exception):
                        pass
                    # Re-attach L2 DAM retriever for the same reason.
                    self._try_init_dam()
                    # Sync ingested count to the number of input messages that were
                    # originally fed into this engine instance.  Using active list
                    # length would be wrong: after compaction the active list is
                    # shorter than the number of messages that were ingested, causing
                    # compress() to re-ingest already-stored messages on the next call.
                    lcm_meta = session_data.get("lcm") or {}
                    original = lcm_meta.get("original_messages")
                    if original:
                        self._ingested_count = len(original)
                    else:
                        # Legacy session (no original_messages key): rebuild() appended
                        # each raw active message to the store.  Seed from store size so
                        # compress() doesn't re-ingest everything that was just restored.
                        self._ingested_count = len(self._engine.store)
                elif not is_compression_rollover:
                    # No save file and not a compression rollover — fresh session,
                    # nothing ingested yet.  Compression rollovers keep existing
                    # engine state so already-compacted messages are not re-ingested.
                    self._ingested_count = 0
            except Exception as e:
                logger.warning("LCM session rebuild failed (starting fresh): %s", e)
                if not is_compression_rollover:
                    self._ingested_count = 0

        # Update model info if provided
        model = kwargs.get("model")
        if model:
            self._engine.model = model

        logger.info(
            "LCM session started (session=%s, context_length=%d, tau_soft=%.2f)",
            session_id, self.context_length, self._config.tau_soft,
        )

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Flush state, persist DAG/HRR for next session."""
        # Ingest any final messages
        new_count = len(messages) - self._ingested_count
        if new_count > 0:
            for msg in messages[self._ingested_count:]:
                self._engine.ingest(msg)
            self._ingested_count = len(messages)

        # Flush HRR store
        if hasattr(self._engine, "hrr_store") and self._engine.hrr_store:
            try:
                self._engine.hrr_store.flush()
            except Exception as e:
                logger.debug("HRR flush failed: %s", e)

    def on_session_reset(self) -> None:
        """Reset per-session state."""
        self._engine.reset()
        self._ingested_count = 0
        super().on_session_reset()

    # ── Tools ─────────────────────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return LCM and memory_* tool schemas.

        Combines the 7 lcm_* schemas with the 5 memory_* schemas so the agent
        has access to all unified memory tools.  The memory_* schemas are
        imported at module load time and fall back to an empty dict when the
        HRR sub-package is not installed.
        """
        if self._disabled:
            return []
        return list(LCM_TOOL_SCHEMAS.values()) + list(MEMORY_TOOL_SCHEMAS.values())

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch LCM and memory_* tool calls."""
        import json as _json

        if self._disabled:
            return _json.dumps({"error": "LCM is disabled"})

        # Route to the internal tool handlers via the engine
        from plugins.context_engine.lcm import tools as lcm_tools

        # Ensure the engine is registered for tool handlers (used by both lcm_*
        # and memory_* handlers — memory handlers call get_engine() internally)
        lcm_tools.set_engine(self._engine)

        handlers = {
            "lcm_expand": lcm_tools.handle_lcm_expand,
            "lcm_pin": lcm_tools.handle_lcm_pin,
            "lcm_forget": lcm_tools.handle_lcm_forget,
            "lcm_search": lcm_tools.handle_lcm_search,
            "lcm_budget": lcm_tools.handle_lcm_budget,
            "lcm_toc": lcm_tools.handle_lcm_toc,
            "lcm_focus": lcm_tools.handle_lcm_focus,
        }

        # Merge memory_* handlers (imported at module level; empty dict when HRR
        # sub-package is unavailable so this is always safe)
        handlers.update(MEMORY_TOOL_HANDLERS)

        handler = handlers.get(name)
        if handler is None:
            return _json.dumps({"error": f"Unknown LCM tool: {name}"})

        return handler(args)

    # ── Status / display ──────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Extended status including store/DAG info."""
        status = super().get_status()
        status.update({
            "store_count": len(self._engine.store),
            "active_entries": len(self._engine.active),
            "pinned_ids": sorted(self._engine._pinned_ids),
            "tau_soft": self._config.tau_soft,
            "tau_hard": self._config.tau_hard,
        })
        return status

    # ── Model switch support ──────────────────────────────────────────────

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        """Propagate model switch to the engine internals."""
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self._config.tau_soft)
        self._engine.model = model
        self._engine.provider = provider
        self._engine.token_estimator.config.model = model
        self._engine.token_estimator.config.provider = provider
        self._engine.token_estimator.config.context_length = context_length
        if self._config.summary_model:
            self._engine.summarizer.config.model = self._config.summary_model
        else:
            self._engine.summarizer.config.model = model

    # ── Availability check ────────────────────────────────────────────────

    @classmethod
    def is_available(cls) -> bool:
        """LCM is always available — it has no required external deps."""
        return True


# ── Plugin registration ───────────────────────────────────────────────────

def register(ctx, **kwargs) -> None:
    """Plugin entry point — register LCM as the context engine.

    The plugin system or plugins/context_engine/ loader calls this.
    We create an LcmContextEngine instance and register it.
    kwargs are forwarded to the engine constructor (e.g. lcm_config, model,
    provider, context_length).
    """
    engine = LcmContextEngine(**kwargs)
    if hasattr(ctx, "register_context_engine"):
        ctx.register_context_engine(engine)
    else:
        logger.warning("LCM plugin: context engine registration not supported by this plugin context")
