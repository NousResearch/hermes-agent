"""RecursiveIntell context-governor compressor for Hermes.

Drop-in replacement for ``ContextCompressor`` that uses the Rust
``context-governor`` crate via PyO3 for deterministic first-pass
compaction. When deterministic savings fall below the configured
threshold, falls back to the built-in LLM summarizer with a special
system prompt that preserves receipts for exact fallback.

Usage (in agent_init or run_agent)::

    from agent.transports.ri_context_compressor import RiContextCompressor
    agent.context_compressor = RiContextCompressor(
        token_budget=8000, name="legacy-ri-context-compressor",
        fallback_compressor=builtin_compressor,
    )
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)

_NATIVE_AVAILABLE = False
try:
    from context_governor._native import compact as _native_compact

    _NATIVE_AVAILABLE = True
except ImportError:
    logger.debug("context-governor native extension not available")

# Minimum token savings ratio to avoid LLM fallback.
# If the Rust path saves less than this fraction of original tokens,
# delegate to the LLM summarizer with receipt preservation.
_DIMINISHING_RETURNS_RATIO = 0.15  # 15% savings threshold

# ── Real CEA graph lane (advisory, fail-open) ──────────────────────────────
# The real causal graph lane expands protect_last_n when the graph predicts
# a risky edit among older messages. This is a SEPARATE lane from synthetic
# telemetry: it reads the forge-engine CEA database via the cea-graph binary,
# never the telemetry store, and never writes anything.
CEA_GRAPH_BIN = os.environ.get(
    "CEA_GRAPH_BIN", str(Path.home() / ".cargo/bin/cea-graph")
)
CEA_GRAPH_DB = os.environ.get(
    "CEA_GRAPH_DB", str(Path.home() / ".recall/forge/forge.db")
)
CEA_GRAPH_TIMEOUT = int(os.environ.get("CEA_GRAPH_TIMEOUT", "5"))
# A message whose edit risk prediction carries at least one risk flag with
# this confidence is considered causally important and protected from
# compression (when it falls outside the default protect window).
CEA_RISK_PROTECT_CONFIDENCE = float(
    os.environ.get("CEA_RISK_PROTECT_CONFIDENCE", "0.5")
)
# Tool names that indicate a file-edit operation worth querying the graph for.
_CEA_EDIT_TOOLS = {"patch", "write_file"}
# Matches a plausible source file path (absolute or relative with extension).
_CEA_PATH_RE = re.compile(r"(?:/[A-Za-z0-9_./~-]+|(?<![\w.])[\w./-]+)\.([A-Za-z0-9]+)")


def _cea_build_signature(file_path: str) -> Dict[str, Any]:
    """Build a minimal EditOpSignature for a file path (conservative defaults)."""
    ext = Path(file_path).suffix.lstrip(".").lower()
    return {
        "op_kind": "replace",
        "anchor_kind": "range",
        "lines_added": 1,
        "lines_removed": 0,
        "context_hash": "",
        "file_extension": ext,
        "scope_tag": "unknown",
        "op_index": 0,
        "file_index": 0,
    }


def _cea_query_risk(signatures: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Query cea-graph predict; return the prediction dict or None on failure."""
    request = {"signatures": signatures, "db_path": CEA_GRAPH_DB}
    try:
        proc = subprocess.run(
            [CEA_GRAPH_BIN, "predict"],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=CEA_GRAPH_TIMEOUT,
        )
        if proc.returncode != 0:
            return None
        payload = json.loads(proc.stdout)
        return payload.get("prediction")
    except Exception:  # noqa: BLE001 — fail open
        return None


def _cea_extract_edit_paths(content: str) -> List[str]:
    """Extract plausible file paths from a message's content text.

    Works on both structured tool-call JSON (tool_input.path) and plain
    text. Returns unique paths whose extension matches an edit tool use.
    """
    if not content:
        return []
    paths: List[str] = []
    # Structured tool-call JSON (Hermes tool messages often embed this).
    for match in re.finditer(r'"path"\s*:\s*"([^"]+)"', content):
        candidate = match.group(1)
        if Path(candidate).suffix:
            paths.append(candidate)
    # Plain text file-path mentions.
    for match in _CEA_PATH_RE.finditer(content):
        candidate = match.group(0)
        if Path(candidate).suffix:
            paths.append(candidate)
    # Dedupe preserving order.
    seen = set()
    return [p for p in paths if not (p in seen or seen.add(p))]


class RiContextCompressor(ContextEngine):
    """Two-stage compressor: Rust context-governor first, LLM fallback if needed.

    Stage 1 (deterministic): ``context_governor.compact()`` handles tool
    result pruning, head/tail protection, and structured summarization
    without an LLM call. Produces receipt-backed output.

    Stage 2 (LLM fallback): when Rust savings fall below
    ``diminishing_returns_ratio``, delegates to the fallback compressor
    (typically the built-in ``ContextCompressor``) which calls an
    auxiliary LLM with a special system prompt that preserves receipts
    for exact fallback references.
    """

    def __init__(
        self,
        token_budget: int = 8000,
        name: str = "legacy-ri-context-compressor",
        fallback_compressor: Any = None,
        diminishing_returns_ratio: float = _DIMINISHING_RETURNS_RATIO,
    ):
        if name == "ri-context-governor":
            raise ValueError(
                "'ri-context-governor' is reserved for the certified CLI-backed "
                "context-engine plugin; use 'legacy-ri-context-compressor' explicitly"
            )
        self.token_budget = token_budget
        self._name = name
        self._fallback = fallback_compressor
        self._diminishing_ratio = diminishing_returns_ratio
        self._last_summary_dropped_count = 0
        self._last_summary_fallback_used = False
        self._last_summary_error: Optional[str] = None
        self._last_compress_aborted = False
        self._last_compression_made_progress = False
        self._last_aux_model_failure_error: Optional[str] = None
        self._last_aux_model_failure_model: Optional[str] = None
        self.last_real_prompt_tokens = 0
        self.last_compression_rough_tokens = 0
        self.last_rough_tokens_when_real_prompt_fit = 0
        self._verify_compaction_cleared_threshold = False
        self._context_probed = False
        self._context_probe_persistable = False
        self.quiet_mode = False
        self.abort_on_summary_failure = False

    @property
    def name(self) -> str:
        """Stable context-engine identifier exposed to Hermes."""
        return self._name

    def _fallback_value(self, name: str, default: Any) -> Any:
        """Read shared context-engine state from the stock implementation."""
        if self._fallback is not None:
            return getattr(self._fallback, name, default)
        return default

    def _set_fallback_value(self, name: str, value: Any) -> None:
        """Keep host-managed context state on the stock implementation."""
        if self._fallback is not None:
            setattr(self._fallback, name, value)
        else:
            # ``update_model()`` normally creates the fallback before Hermes
            # writes these fields. Retain an early write defensively so a
            # standalone caller does not lose it.
            setattr(self, f"_pending_{name}", value)

    @property
    def context_length(self) -> int:
        return self._fallback_value("context_length", 0)

    @context_length.setter
    def context_length(self, value: int) -> None:
        self._set_fallback_value("context_length", value)

    @property
    def threshold_tokens(self) -> int:
        return self._fallback_value("threshold_tokens", 0)

    @threshold_tokens.setter
    def threshold_tokens(self, value: int) -> None:
        self._set_fallback_value("threshold_tokens", value)

    @property
    def threshold_percent(self) -> float:
        return self._fallback_value("threshold_percent", 0.50)

    @threshold_percent.setter
    def threshold_percent(self, value: float) -> None:
        self._set_fallback_value("threshold_percent", value)

    @property
    def protect_first_n(self) -> int:
        return self._fallback_value("protect_first_n", 3)

    @protect_first_n.setter
    def protect_first_n(self, value: int) -> None:
        self._set_fallback_value("protect_first_n", value)

    @property
    def protect_last_n(self) -> int:
        return self._fallback_value("protect_last_n", 6)

    @protect_last_n.setter
    def protect_last_n(self, value: int) -> None:
        self._set_fallback_value("protect_last_n", value)

    @property
    def compression_count(self) -> int:
        return self._fallback_value("compression_count", 0)

    @compression_count.setter
    def compression_count(self, value: int) -> None:
        self._set_fallback_value("compression_count", value)

    @property
    def last_prompt_tokens(self) -> int:
        return self._fallback_value("last_prompt_tokens", 0)

    @last_prompt_tokens.setter
    def last_prompt_tokens(self, value: int) -> None:
        self._set_fallback_value("last_prompt_tokens", value)

    @property
    def last_completion_tokens(self) -> int:
        return self._fallback_value("last_completion_tokens", 0)

    @last_completion_tokens.setter
    def last_completion_tokens(self, value: int) -> None:
        self._set_fallback_value("last_completion_tokens", value)

    @property
    def last_total_tokens(self) -> int:
        return self._fallback_value("last_total_tokens", 0)

    @last_total_tokens.setter
    def last_total_tokens(self, value: int) -> None:
        self._set_fallback_value("last_total_tokens", value)

    @property
    def last_compression_rough_tokens(self) -> int:
        return self._fallback_value("last_compression_rough_tokens", 0)

    @last_compression_rough_tokens.setter
    def last_compression_rough_tokens(self, value: int) -> None:
        self._set_fallback_value("last_compression_rough_tokens", value)

    @property
    def awaiting_real_usage_after_compression(self) -> bool:
        return self._fallback_value("awaiting_real_usage_after_compression", False)

    @awaiting_real_usage_after_compression.setter
    def awaiting_real_usage_after_compression(self, value: bool) -> None:
        self._set_fallback_value("awaiting_real_usage_after_compression", value)

    @property
    def available(self) -> bool:
        return _NATIVE_AVAILABLE

    def is_available(self) -> bool:
        """Plugin discovery contract — returns True when native extension is present."""
        return _NATIVE_AVAILABLE

    @property
    def fallback_available(self) -> bool:
        return self._fallback is not None and hasattr(self._fallback, "compress")

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Compress conversation messages with two-stage Rust+LLM pipeline.

        Stage 1: deterministic Rust compaction via context-governor.
        Stage 2: LLM-based summarization if Rust savings are insufficient.
        """
        # Match the stock compressor's per-attempt reporting contract. The
        # host inspects these fields after ``compress()`` to decide whether it
        # may rotate the session or should preserve the original transcript.
        self._last_summary_dropped_count = 0
        self._last_summary_fallback_used = False
        self._last_summary_error = None
        self._last_compress_aborted = False
        self._last_compression_made_progress = False

        if not _NATIVE_AVAILABLE:
            return self._fallback_compress(
                messages, current_tokens, focus_topic, force, memory_context
            )

        if len(messages) < 4:
            return messages

        try:
            # ── Stage 1: deterministic Rust compaction ────────────────
            msg_dicts = [
                {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
                for m in messages
            ]
            messages_json = json.dumps(msg_dicts)
            session_id = f"ctxr_{uuid.uuid4().hex[:12]}"

            # ── Real CEA graph lane ───────────────────────────────────
            # Advisory only: if the causal graph predicts risky edits in
            # older messages, expand protect_last_n to keep those messages.
            # Fail-open: any error leaves the default window unchanged.
            cea_protect = self._cea_protect_last_n(msg_dicts)
            logger.info(
                "ri-context-governor CEA lane: protect_last_n=%s (base=%s)",
                cea_protect if cea_protect is not None else "default",
                self.protect_last_n,
            )
            kwargs = {}
            if cea_protect is not None:
                kwargs["protect_last_n"] = cea_protect

            result_json = _native_compact(
                messages_json, session_id, self.token_budget, **kwargs
            )
            result = json.loads(result_json)

            compacted = result.get("compacted_messages", [])
            if not compacted:
                logger.warning("context-governor returned no compacted messages")
                self._last_compress_aborted = True
                return self._fallback_compress(
                    messages, current_tokens, focus_topic, force, memory_context
                )

            savings = result.get("token_savings_estimate", 0)
            original_tokens = result.get("original_approx_tokens", len(messages) * 100)
            savings_ratio = savings / max(original_tokens, 1)

            logger.info(
                "context-governor stage-1: %d→%d msgs, %d→%d tokens (%.1f%% saved, receipt=%s)",
                len(messages),
                len(compacted),
                original_tokens,
                result.get("compacted_approx_tokens", 0),
                savings_ratio * 100,
                result.get("receipt_id", "?"),
            )

            # ── Diminishing returns check ─────────────────────────────
            if savings_ratio < self._diminishing_ratio:
                logger.info(
                    "context-governor savings %.1f%% below threshold %.1f%% — "
                    "falling back to LLM summarizer with receipt preservation",
                    savings_ratio * 100,
                    self._diminishing_ratio * 100,
                )
                self._last_summary_fallback_used = True
                # Pass the compacted (not original) messages to the LLM
                # so it can build on the deterministic work. The compacted
                # messages already carry receipt references.
                return self._fallback_compress(
                    compacted, current_tokens, focus_topic, force, memory_context
                )

            # ── Sufficient savings — return deterministic result ──────
            self._last_compression_made_progress = len(compacted) < len(messages)
            self._last_summary_fallback_used = False
            if self._last_compression_made_progress:
                self.compression_count += 1
            return compacted

        except Exception as exc:
            logger.error("RiContextCompressor stage-1 failed: %s", exc, exc_info=True)
            self._last_summary_error = str(exc)
            return self._fallback_compress(
                messages, current_tokens, focus_topic, force, memory_context
            )

    def _cea_protect_last_n(self, msg_dicts: List[Dict[str, Any]]) -> Optional[int]:
        """Compute an expanded protect_last_n from the real CEA graph.

        Scans messages for file-edit tool calls, queries cea-graph predict,
        and if a risky edit is predicted in a message OUTSIDE the default
        protect window, expands protect_last_n to cover it. Returns None
        (fail-open) on any error or when no causal protection is warranted.
        """
        try:
            # Find messages that look like file-edit tool calls.
            risky_indices: List[int] = []
            signatures: List[Dict[str, Any]] = []
            index_by_signature: Dict[str, List[int]] = {}
            for idx, msg in enumerate(msg_dicts):
                content = str(msg.get("content", ""))
                paths = _cea_extract_edit_paths(content)
                if not paths:
                    continue
                # Only treat messages mentioning an edit tool as edit calls;
                # plain file-path mentions alone are not edits.
                content_lower = content.lower()
                if not any(tool in content_lower for tool in _CEA_EDIT_TOOLS):
                    continue
                for path in paths:
                    sig = _cea_build_signature(path)
                    sig_key = json.dumps(sig, sort_keys=True)
                    signatures.append(sig)
                    index_by_signature.setdefault(sig_key, []).append(idx)

            if not signatures:
                return None

            prediction = _cea_query_risk(signatures)
            if prediction is None:
                return None

            risk_flags = prediction.get("risk_flags") or []
            if not risk_flags:
                return None

            # A risk flag with confidence above the threshold marks its
            # associated edit messages as causally important.
            for flag in risk_flags:
                confidence = flag.get("confidence", 0.0)
                if confidence < CEA_RISK_PROTECT_CONFIDENCE:
                    continue
                sig_json = json.dumps(
                    {
                        "op_kind": flag.get("op_signature", {}).get(
                            "op_kind", "replace"
                        ),
                        "anchor_kind": flag.get("op_signature", {}).get(
                            "anchor_kind", "range"
                        ),
                        "lines_added": flag.get("op_signature", {}).get(
                            "lines_added", 0
                        ),
                        "lines_removed": flag.get("op_signature", {}).get(
                            "lines_removed", 0
                        ),
                        "context_hash": flag.get("op_signature", {}).get(
                            "context_hash", ""
                        ),
                        "file_extension": flag.get("op_signature", {}).get(
                            "file_extension", ""
                        ),
                        "scope_tag": flag.get("op_signature", {}).get(
                            "scope_tag", "unknown"
                        ),
                        "op_index": 0,
                        "file_index": 0,
                    },
                    sort_keys=True,
                )
                risky_indices.extend(index_by_signature.get(sig_json, []))

            if not risky_indices:
                return None

            total = len(msg_dicts)
            default_cutoff = total - self.protect_last_n
            # Earliest causally-important message outside the default window.
            outside = [i for i in risky_indices if i < default_cutoff]
            if not outside:
                return None
            earliest = min(outside)
            expanded = total - earliest
            return max(expanded, self.protect_last_n)
        except Exception:  # noqa: BLE001 — fail open
            logger.debug("CEA graph lane failed open", exc_info=True)
            return None

    def _fallback_compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int],
        focus_topic: Optional[str],
        force: bool,
        memory_context: str,
    ) -> List[Dict[str, Any]]:
        """Delegate to the built-in LLM summarizer with receipt preservation."""
        if not self.fallback_available:
            logger.warning(
                "RiContextCompressor: no fallback compressor available, "
                "returning messages unchanged"
            )
            self._last_compress_aborted = True
            return messages

        try:
            logger.info("RiContextCompressor: delegating to fallback LLM summarizer")
            compacted = self._fallback.compress(
                messages,
                current_tokens=current_tokens,
                focus_topic=focus_topic,
                force=force,
                memory_context=memory_context,
            )
            for attr in (
                "_last_summary_dropped_count",
                "_last_summary_fallback_used",
                "_last_summary_error",
                "_last_compress_aborted",
                "_last_compression_made_progress",
            ):
                setattr(self, attr, getattr(self._fallback, attr, getattr(self, attr)))
            return compacted
        except Exception as exc:
            logger.error("RiContextCompressor fallback failed: %s", exc, exc_info=True)
            self._last_summary_error = str(exc)
            self._last_compress_aborted = True
            return messages

    def __repr__(self) -> str:
        status = (
            "native+llm"
            if self.available and self.fallback_available
            else ("native" if self.available else "unavailable")
        )
        return f"RiContextCompressor(budget={self.token_budget}, {status})"

    # ── Hermes context-engine contract ────────────────────────────
    # Called by agent_init after plugin registration to inject model
    # parameters. We use these to construct a fallback ContextCompressor
    # for the LLM stage when the Rust path hits diminishing returns.

    def update_model(
        self,
        model: str = "",
        context_length: int = 0,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
        max_tokens: Optional[int] = None,
    ) -> None:
        """Called by Hermes to inject model parameters.

        Constructs a fallback ContextCompressor for LLM-based
        summarization when the Rust deterministic path yields
        insufficient savings.
        """
        if self._fallback is not None:
            # Fallback already configured with the dedicated summarization
            # model — do not overwrite with the session's active model.
            return

        try:
            from agent.context_compressor import ContextCompressor

            # Use gpt-5.6-luna via openai-codex for summarization,
            # regardless of the session's active model. This is a
            # dedicated summarization model — faster, cheaper, and
            # more reliable than using the session model for compaction.
            summary_model = "gpt-5.6-luna"
            summary_provider = "openai-codex"

            self._fallback = ContextCompressor(
                model=summary_model,
                base_url=base_url,
                api_key=api_key,
                provider=summary_provider,
                api_mode=api_mode,
                config_context_length=context_length,
                quiet_mode=True,
                threshold_percent=0.50,
                summary_target_ratio=0.20,
            )
            for name in (
                "context_length",
                "threshold_tokens",
                "threshold_percent",
                "protect_first_n",
                "protect_last_n",
                "compression_count",
                "last_prompt_tokens",
                "last_completion_tokens",
                "last_total_tokens",
                "last_compression_rough_tokens",
                "awaiting_real_usage_after_compression",
            ):
                pending_name = f"_pending_{name}"
                if hasattr(self, pending_name):
                    setattr(self._fallback, name, getattr(self, pending_name))
                    delattr(self, pending_name)
            logger.info(
                "RiContextCompressor: fallback LLM summarizer constructed "
                "(model=%s, ctx_len=%d)",
                model,
                context_length,
            )
        except Exception as exc:
            logger.warning("RiContextCompressor: could not construct fallback: %s", exc)

    def bind_session_state(self, session_db: Any = None, session_id: str = "") -> None:
        """Forward session binding to fallback compressor."""
        if self._fallback is not None and hasattr(self._fallback, "bind_session_state"):
            self._fallback.bind_session_state(
                session_db=session_db, session_id=session_id
            )

    # ── Stock ContextEngine contract ---------------------------------------
    # Hermes owns token accounting, preflight gating, lifecycle notifications,
    # and context-engine tool registration.  Delegate those policy/state
    # methods to the bundled compressor so this adapter remains compatible as
    # the host grows, while ``compress`` above remains Rust-first.

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """The governor currently exposes no model-callable recovery tools."""
        return []

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        if self._fallback is not None:
            self._fallback.update_from_response(usage)

    def should_compress(self, prompt_tokens: Optional[int] = None) -> bool:
        return bool(
            self._fallback is not None and self._fallback.should_compress(prompt_tokens)
        )

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        return bool(
            self._fallback is not None
            and self._fallback.should_defer_preflight_to_real_usage(rough_tokens)
        )

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        return bool(
            self._fallback is None or self._fallback.has_content_to_compress(messages)
        )

    def on_session_start(self, session_id: str, **kwargs) -> None:
        super().on_session_start(session_id, **kwargs)
        if self._fallback is not None:
            self._fallback.on_session_start(session_id, **kwargs)

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self._last_summary_dropped_count = 0
        self._last_summary_fallback_used = False
        self._last_summary_error = None
        self._last_compress_aborted = False
        self._last_compression_made_progress = False
        self._last_aux_model_failure_error = None
        self._last_aux_model_failure_model = None
        self._context_probed = False
        self._context_probe_persistable = False
        self.last_real_prompt_tokens = 0
        self.last_compression_rough_tokens = 0
        self.last_rough_tokens_when_real_prompt_fit = 0
        self._verify_compaction_cleared_threshold = False
        if self._fallback is not None:
            self._fallback.on_session_end(session_id, messages)

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._last_summary_dropped_count = 0
        self._last_summary_fallback_used = False
        self._last_summary_error = None
        self._last_compress_aborted = False
        self._last_compression_made_progress = False
        self._last_aux_model_failure_error = None
        self._last_aux_model_failure_model = None
        self._context_probed = False
        self._context_probe_persistable = False
        self.last_real_prompt_tokens = 0
        self.last_compression_rough_tokens = 0
        self.last_rough_tokens_when_real_prompt_fit = 0
        self._verify_compaction_cleared_threshold = False
        if self._fallback is not None:
            self._fallback.on_session_reset()
            # Sync token-tracking values back from the fallback after reset
            self.compression_count = self._fallback.compression_count

    def record_completed_compaction(self, *, used_fallback: bool = False) -> None:
        if self._fallback is not None:
            self._fallback.record_completed_compaction(
                used_fallback=used_fallback or self._last_summary_fallback_used
            )
