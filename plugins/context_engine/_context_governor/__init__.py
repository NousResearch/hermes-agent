"""Shared Hermes adapter implementation for the `context-governor` CLI.

This private module is intentionally thin: context-governor owns deterministic
compaction/receipt logic; Hermes owns host contracts such as preserving the
latest user message as the final active instruction and falling back safely on
adapter errors. The discoverable ``ri-context-governor`` plugin is the sole
selection owner and constructs this implementation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine
from agent.redact import redact_sensitive_text
from hermes_constants import get_hermes_home
from plugins.context_engine._context_governor.key_state import (
    ContextGovernorKeyError,
    ContextGovernorKeyState,
    GovernedKeyBinding,
)

logger = logging.getLogger(__name__)

# The provenance manifest is authenticated receipt metadata, not prompt-visible
# context. Keep it bounded, but large enough for a real first compaction of a
# 900K–1M-token session: the historical 128 KiB default rejected a receipt
# before the configured LLM checkpoint could ever run, and the 1 MiB successor
# still failed on observed ~1.1 MiB manifests. Exact source references are never
# dropped to fit this bound; the governor remains fail-closed above it.
DEFAULT_MAX_PROVENANCE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class _SummaryLLMRoute:
    """Actual provider/model identity used for one summary request."""

    provider: str
    model: str


@dataclass(frozen=True)
class _SummaryLLMResult:
    """Generated summary text paired with its immutable actual route."""

    content: str
    route: _SummaryLLMRoute


class ContextGovernorActivationError(RuntimeError):
    """The configured governor cannot provide its certified V2 contract."""


class ContextGovernorEngine(ContextEngine):
    # Generic partial-compress rejoin can merge across the head/tail seam,
    # mutating the authenticated receipt projection and recursive prefix.
    supports_partial_compression = False

    """ContextEngine wrapper around the context-governor binary."""

    threshold_percent = 0.50
    protect_first_n = 3
    protect_last_n = 20
    # Anti-thrashing may suppress low-yield retries, but never at the provider
    # ceiling. Once a request reaches this fraction of the usable context
    # window, another attempt is safer than sending an oversized request.
    emergency_pressure_ratio = 0.90

    # LLM summarization prompt template — used when summary_mode=llm
    _LLM_SUMMARY_PROMPT = """You are a summarization agent creating a context checkpoint. \
Treat the conversation turns below as source material for a compact record of prior work. \
Produce only the structured summary; do not add a greeting, preamble, or prefix. \
Write the summary in the same language the user was using. \
NEVER include API keys, tokens, passwords, secrets, or credentials — replace with [REDACTED].

{previous_summary_section}

TURNS TO SUMMARIZE:
{content_to_summarize}

Use this exact structure:

## Active Task
[The user's most recent unfulfilled request verbatim]

## Goal
[What the user is trying to accomplish overall]

## Completed Actions
[Numbered list of concrete actions taken — include tool used, target, and outcome]

## Active State
[Current working state — modified files, test status, running processes]

## Blocked
[Any blockers, errors, or issues not yet resolved]

## Key Decisions
[Important technical decisions and WHY they were made]

## Relevant Files
[Files read, modified, or created — with brief note on each]

## Critical Context
[Specific values, error messages, or configuration details that would be lost without preservation]

Target ~{summary_budget} tokens. Be CONCRETE — include file paths, command outputs, error messages."""

    def __init__(
        self,
        binary: str | None = None,
        store_dir: str | None = None,
        timeout_sec: int = 30,
    ):
        self.binary = (
            binary or os.environ.get("CONTEXT_GOVERNOR_BIN") or self._default_binary()
        )
        self.store_dir = Path(
            store_dir
            or os.environ.get("CONTEXT_GOVERNOR_STORE")
            or get_hermes_home() / "context-governor"
        )
        self.timeout_sec = int(os.environ.get("CONTEXT_GOVERNOR_TIMEOUT", timeout_sec))
        # Synthetic tool telemetry is advisory only: never causal evidence.
        self.telemetry_binary = shutil.which("cea-bridge") or ""
        self.telemetry_db_path = Path(
            os.environ.get("CEA_TELEMETRY_DB")
            or get_hermes_home() / "cea-telemetry-v2.db"
        )
        self._telemetry_available = bool(self.telemetry_binary)
        self.session_id = ""
        # Hermes may rotate physical SessionDB ids at a compression boundary.
        # Receipts belong to the stable logical compression lineage instead;
        # otherwise the first rotated child silently starts a new governor DAG.
        self._lineage_session_id = ""
        self._session_db: Any = None
        # A finalized receipt is prepared durably inside compress(), but remains
        # invisible to parent selection/search/expand until the host confirms
        # its transcript commit through commit_pending_compression().
        self._pending_admission: dict[str, Any] | None = None
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 0
        self.context_length = 0
        self.compression_count = 0
        self.last_receipt_id: str | None = None
        self.last_error: str | None = None
        self.last_warning: str | None = None
        self.last_summary_safety: dict[str, Any] | None = None
        self.last_compaction_metrics: dict[str, Any] | None = None
        self.fallback_event_count = 0
        self.max_tokens: int | None = None
        # Manual-compression callers use the built-in compressor's abort
        # fields as their provider-neutral failure contract.  Implement that
        # contract here too so a failed governor call cannot be presented as a
        # successful "No changes" result.
        self._last_compress_aborted = False
        self._last_summary_error: str | None = None
        self._last_summary_fallback_used = False
        self._last_compression_made_progress = False

        # Host contract attributes — read by agent/turn_context.py and
        # agent/conversation_compression.py without getattr defaults. Must
        # exist from construction so multi-turn sessions never AttributeError
        # when the host reads them before update_from_response fires.
        self.last_real_prompt_tokens = 0
        self.last_compression_rough_tokens = 0
        self.awaiting_real_usage_after_compression = False

        # Anti-thrashing state
        self._ineffective_compression_count = 0
        self._last_compression_savings_pct = 100.0

        # Deferred preflight state. The adapter tracks the baseline
        # internally as ``_last_rough_tokens_when_real_fit`` (legacy
        # name, kept for backward compat) and also exposes the host's
        # public name ``last_rough_tokens_when_real_prompt_fit`` so
        # future host code can read either engine uniformly. The two
        # are kept in sync via ``_set_defer_baseline`` below.
        self._set_defer_baseline(0)

        # Iterative summary state — persists across compaction cycles
        self._previous_summary: str | None = None

        # LLM summarization config
        self._summary_mode = "extractive"  # "extractive" or "llm"
        self._summary_model = ""
        self._summary_provider = ""
        self._summary_api_key = ""
        self._summary_base_url = ""
        self._llm_checkpoint_count = 0
        # Ares owns lifecycle under the profile root. No config-supplied key
        # path is ever a canonical signing authority.
        self._key_state = ContextGovernorKeyState(get_hermes_home(), self.binary)
        self._key_binding: GovernedKeyBinding | None = None
        self._unsafe_configured_hmac_path: str | None = None
        self._capabilities: dict[str, Any] | None = None
        # Runtime model credentials are refreshed by update_model.  Keep them
        # separately from optional summary overrides so LLM summaries inherit
        # the active agent route when no override is configured.
        self.model = ""
        self.base_url = ""
        self.api_key = ""
        self.provider = ""
        self.api_mode = ""

        # Config-driven policy with safe defaults
        self._policy = {
            "budget_mode": "soft_warn",
            "allocator": "deterministic_v1",
            "semantic_memory_enabled": False,
            "archive_memory_enabled": False,
            "summary_max_chars": 8000,
            "token_counter": "approx_chars",
            "unsafe_summary_policy": "fallback_extract",
            "checkpoint_strategy": "after_n:2",
            "max_checkpoints": 10,
            "token_budget": None,
            "protect_first_n": self.protect_first_n,
            "protect_last_n": self.protect_last_n,
            "telemetry_max_additional_protected_messages": 8,
            # Bounded V2 is an admission contract, not advisory telemetry.
            # These values prevent provenance from consuming the context it is
            # supposed to save; the Rust owner enforces them before issuing a
            # receipt.
            "max_lineage_generation": 32,
            # Receipt provenance is durable store metadata; render-prompt-v2
            # separately bounds the prompt-visible projection to four refs.
            # Sixteen MiB admits a real 900K–1M-token initial transcript and
            # observed tool-heavy recursive suffixes without making the
            # manifest unbounded across the 32-generation ceiling. Explicit
            # config values still override this default.
            "max_provenance_bytes": DEFAULT_MAX_PROVENANCE_BYTES,
            "min_net_savings_tokens": 128,
        }
        # Override from config if available
        self._load_policy_from_config()

    @staticmethod
    def _default_binary() -> str:
        return (
            shutil.which("context-governor")
            or "/home/sikmindz/.local/bin/context-governor"
            or "context-governor"
        )

    def _load_policy_from_config(self) -> None:
        """Load policy overrides from config.yaml context.governor section."""
        try:
            from hermes_cli.config import load_config

            cfg = load_config()
            ctx_cfg = cfg.get("context", {}).get("governor", {})
            for key in self._policy:
                if key in ctx_cfg:
                    self._policy[key] = ctx_cfg[key]
            # Older adapters exposed the safety key under this name. Keep one
            # explicit migration alias, but emit only the Rust owner's
            # ``unsafe_summary_policy`` field from now on.
            if (
                "unsafe_summary_policy" not in ctx_cfg
                and "summary_safety_policy" in ctx_cfg
            ):
                self._policy["unsafe_summary_policy"] = ctx_cfg["summary_safety_policy"]
            self.protect_first_n = max(0, int(self._policy.get("protect_first_n") or 0))
            self.protect_last_n = max(0, int(self._policy.get("protect_last_n") or 0))
            # LLM summary mode config
            self._summary_mode = ctx_cfg.get("summary_mode", "extractive")
            self._summary_model = ctx_cfg.get("summary_model", "")
            self._summary_provider = ctx_cfg.get("summary_provider", "")
            self._summary_api_key = ctx_cfg.get("summary_api_key", "")
            self._summary_base_url = ctx_cfg.get("summary_base_url", "")
            hmac_path = ctx_cfg.get("receipt_hmac_key_path", "")
            if isinstance(hmac_path, str) and hmac_path.strip():
                self._unsafe_configured_hmac_path = hmac_path
        except Exception:
            pass  # Use defaults if config unavailable

    @property
    def name(self) -> str:
        return "context_governor"

    def __deepcopy__(self, memo):
        clone = type(self)(str(self.binary), str(self.store_dir), self.timeout_sec)
        clone.session_id = self.session_id
        clone._lineage_session_id = self._lineage_session_id
        clone._session_db = self._session_db
        clone._pending_admission = copy.deepcopy(self._pending_admission)
        clone.last_prompt_tokens = self.last_prompt_tokens
        clone.last_completion_tokens = self.last_completion_tokens
        clone.last_total_tokens = self.last_total_tokens
        clone.threshold_tokens = self.threshold_tokens
        clone.context_length = self.context_length
        clone.compression_count = self.compression_count
        clone.last_receipt_id = self.last_receipt_id
        clone.last_error = self.last_error
        clone.last_warning = self.last_warning
        clone.last_summary_safety = copy.deepcopy(self.last_summary_safety)
        clone.last_compaction_metrics = copy.deepcopy(self.last_compaction_metrics)
        clone.fallback_event_count = self.fallback_event_count
        clone.max_tokens = self.max_tokens
        clone._last_compress_aborted = self._last_compress_aborted
        clone._last_summary_error = self._last_summary_error
        clone._last_summary_fallback_used = self._last_summary_fallback_used
        clone._last_compression_made_progress = self._last_compression_made_progress
        clone.last_real_prompt_tokens = self.last_real_prompt_tokens
        clone.last_compression_rough_tokens = self.last_compression_rough_tokens
        clone.awaiting_real_usage_after_compression = (
            self.awaiting_real_usage_after_compression
        )
        clone._ineffective_compression_count = self._ineffective_compression_count
        clone._last_compression_savings_pct = self._last_compression_savings_pct
        clone._set_defer_baseline(self.last_rough_tokens_when_real_prompt_fit)
        clone._previous_summary = self._previous_summary
        clone._summary_mode = self._summary_mode
        clone._summary_model = self._summary_model
        clone._summary_provider = self._summary_provider
        clone._summary_api_key = self._summary_api_key
        clone._summary_base_url = self._summary_base_url
        clone._llm_checkpoint_count = self._llm_checkpoint_count
        clone._unsafe_configured_hmac_path = self._unsafe_configured_hmac_path
        clone.model = self.model
        clone.base_url = self.base_url
        clone.api_key = self.api_key
        clone.provider = self.provider
        clone.api_mode = self.api_mode
        clone._policy = dict(self._policy)
        return clone

    def is_available(self) -> bool:
        try:
            self.probe_activation()
            return True
        except ContextGovernorActivationError:
            return False

    def probe_activation(self) -> dict[str, Any]:
        """Prove binary, protocol, receipt schema, and integrity readiness.

        This runs during strict selection, before Hermes accepts the engine as
        a replacement for stock compression. It deliberately refuses a V2
        runtime without a configured readable HMAC key.
        """
        if self._unsafe_configured_hmac_path:
            raise ContextGovernorActivationError(
                "ConfigurationPathOutsideCanonicalState: receipt_hmac_key_path is not accepted"
            )
        binary = str(self.binary)
        resolved_text = (
            str(Path(binary)) if Path(binary).is_file() else shutil.which(binary)
        )
        if not resolved_text:
            raise ContextGovernorActivationError(
                f"governor binary unavailable: {binary}"
            )
        resolved = Path(resolved_text)
        self._key_state = ContextGovernorKeyState(get_hermes_home(), str(resolved))
        try:
            binding = self._key_state.active_binding()
            binding.close()
        except ContextGovernorKeyError as exc:
            raise ContextGovernorActivationError(str(exc)) from exc
        try:
            proc = subprocess.run(
                [str(resolved), "capabilities"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_sec,
                check=False,
            )
        except OSError as exc:
            raise ContextGovernorActivationError(
                f"could not execute governor binary {resolved}: {exc}"
            ) from exc
        if proc.returncode != 0:
            raise ContextGovernorActivationError(
                (proc.stderr or proc.stdout or "capability probe failed").strip()
            )
        try:
            capabilities = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ContextGovernorActivationError(
                "governor capability probe returned invalid JSON"
            ) from exc
        required = {
            "schema": "ContextGovernorCapabilitiesV1",
            "engine": "ri-context-governor",
            "receipt_schema": "ContextCompactionReceiptV2",
            "integrity": "hmac-sha256-canonical-json-v1",
            "exactness_scope": "canonical_utf8_text_v1",
        }
        if (
            not isinstance(capabilities, dict)
            or any(capabilities.get(key) != value for key, value in required.items())
            or not capabilities.get("supports_recursive_lineage")
            or not capabilities.get("supports_certified_receipt_store")
        ):
            raise ContextGovernorActivationError(
                f"governor capability mismatch: {capabilities!r}"
            )
        self.binary = str(resolved)
        self._capabilities = capabilities
        self.set_activation_status(
            configured_engine=self.name,
            discovered=True,
            version_compatible=True,
            capability_compatible=True,
            instantiated=True,
            observed_live_engine=None,
            state="ready",
            detail="binary and certified V2 capability probe passed",
        )
        return capabilities

    def _certified_store_args(self) -> list[str]:
        try:
            if self._key_binding is not None:
                self._key_binding.close()
            binding = self._key_state.active_binding()
        except ContextGovernorKeyError as exc:
            raise ContextGovernorActivationError(str(exc)) from exc
        self._key_binding = binding
        return binding.command_args()

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
        max_tokens: int | None = None,
        threshold_percent: float | None = None,
        protect_first_n: int | None = None,
        protect_last_n: int | None = None,
    ) -> None:
        # Persist the active agent route.  The optional summary-specific fields
        # override these only when explicitly configured.
        self.model = str(model or "")
        self.base_url = str(base_url or "")
        self.api_key = str(api_key or "")
        self.provider = str(provider or "")
        self.api_mode = str(api_mode or "")
        if threshold_percent is not None:
            self.threshold_percent = float(threshold_percent)
        if protect_first_n is not None:
            self.protect_first_n = int(protect_first_n)
        if protect_last_n is not None:
            self.protect_last_n = int(protect_last_n)
        self.context_length = int(context_length or 0)
        self.max_tokens = (
            int(max_tokens) if max_tokens and int(max_tokens) > 0 else None
        )
        # Account for output reservation in effective input budget
        effective_window = self.context_length - (self.max_tokens or 0)
        if effective_window <= 0:
            effective_window = self.context_length
        self.threshold_tokens = (
            int(effective_window * self.threshold_percent) if effective_window else 0
        )

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = int(
            usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        )
        self.last_completion_tokens = int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
        self.last_total_tokens = int(
            usage.get("total_tokens")
            or (self.last_prompt_tokens + self.last_completion_tokens)
        )
        # Mirror the built-in contract: last_real_prompt_tokens tracks the
        # most recent non-zero provider-reported prompt count, separate from
        # last_prompt_tokens (which can be -1 after a deferred preflight).
        if self.last_prompt_tokens > 0:
            self.last_real_prompt_tokens = self.last_prompt_tokens
            if self.last_prompt_tokens < self.threshold_tokens:
                if (
                    self.awaiting_real_usage_after_compression
                    and self.last_compression_rough_tokens > 0
                ):
                    self._set_defer_baseline(self.last_compression_rough_tokens)
            else:
                self._set_defer_baseline(0)
        self.awaiting_real_usage_after_compression = False

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = int(
            prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens or 0
        )
        if not self.threshold_tokens or tokens < self.threshold_tokens:
            return False
        # Anti-thrashing is a normal-band guard only. Once the request is near
        # the usable provider ceiling, refusing to compact guarantees an
        # oversized API request; safety takes precedence over another
        # low-savings warning.
        emergency_threshold = self._emergency_pressure_threshold()
        if self._ineffective_compression_count >= 2 and (
            emergency_threshold <= 0 or tokens < emergency_threshold
        ):
            logger.warning(
                "Compression skipped — last %d compressions saved <10%% each. "
                "Consider /new to start a fresh session, or /compress <topic> "
                "for focused compression.",
                self._ineffective_compression_count,
            )
            return False
        if self._ineffective_compression_count >= 2:
            logger.warning(
                "Emergency compression: ~%s tokens reached the %s%% context safety band "
                "despite ineffective prior passes.",
                f"{tokens:,}",
                int(self.emergency_pressure_ratio * 100),
            )
        return True

    def _emergency_pressure_threshold(self) -> int:
        """Return the request-token band where anti-thrashing must yield."""
        if not self.context_length:
            return 0
        effective_window = self.context_length - (self.max_tokens or 0)
        if effective_window <= 0:
            effective_window = self.context_length
        return max(1, int(effective_window * self.emergency_pressure_ratio))

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        if not self.threshold_tokens:
            return False
        rough = sum(
            max(1, len(self._content_to_text(m.get("content"))) // 4)
            for m in messages
            if isinstance(m, dict)
        )
        return rough >= self.threshold_tokens

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        """Return True only while a high rough estimate is known-noisy.

        Mirrors the built-in ContextCompressor contract. The rough preflight
        estimator includes tool/schema overhead and can overestimate immediately
        after compaction; provider-reported real usage is a better signal only
        for a bounded growth window. Once rough growth exceeds tolerance,
        preflight must compress again instead of letting the session creep toward
        the hard context limit.
        """
        if not self.threshold_tokens or rough_tokens < self.threshold_tokens:
            return False
        # The one-turn post-compaction defer avoids a duplicate pass while the
        # next provider usage measurement arrives. It is not permission to send
        # a request at the usable context ceiling: a protected tail, tool schema
        # growth, or late hook context can make the actual next request unsafe.
        # At emergency pressure, fall through so should_compress() forces a
        # safety retry before the provider sees the oversized request.
        emergency_threshold = self._emergency_pressure_threshold()
        if self.awaiting_real_usage_after_compression and (
            emergency_threshold <= 0 or rough_tokens < emergency_threshold
        ):
            return True
        # Futility deferral is a normal-band optimization, not a safety gate.
        # At emergency pressure the caller must reach should_compress(), whose
        # emergency override can force a compaction before the provider rejects
        # the request. Keeping this unconditional used to make the governor
        # look completely dead after two low-yield passes: both automatic
        # preflight paths returned here before should_compress() was evaluated.
        if self._ineffective_compression_count >= 2 and (
            emergency_threshold <= 0 or rough_tokens < emergency_threshold
        ):
            return True
        if self.last_real_prompt_tokens <= 0:
            return False
        if self.last_real_prompt_tokens >= self.threshold_tokens:
            return False
        baseline = (
            self.last_rough_tokens_when_real_prompt_fit
            or self.last_compression_rough_tokens
        )
        if baseline <= 0:
            return False
        growth = max(0, rough_tokens - baseline)
        tolerated = max(4096, int(self.threshold_tokens * 0.05))
        if growth > tolerated:
            return False
        self._set_defer_baseline(max(baseline, rough_tokens))
        return True

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        non_system = [
            m for m in messages if isinstance(m, dict) and m.get("role") != "system"
        ]
        return len(non_system) > (self.protect_first_n + self.protect_last_n)

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self.session_id = str(session_id or self.session_id or "default")
        session_db = kwargs.get("session_db")
        if session_db is not None:
            self.bind_session_state(session_db=session_db, session_id=self.session_id)
        elif kwargs.get("boundary_reason") != "compression":
            self._lineage_session_id = self.session_id
        elif not self._lineage_session_id:
            # A boundary callback without a bound SessionDB is still one
            # logical conversation. Preserve the old physical id as its stable
            # governor key rather than rebasing the receipt DAG to the child.
            self._lineage_session_id = str(
                kwargs.get("old_session_id") or self.session_id
            )
        # Cross-session context transfer: if receipts exist from prior sessions,
        # load the most recent one as initial _previous_summary for context continuity.
        # This is a lightweight bootstrap — the model can use context_search to
        # find specific omitted content from prior sessions.
        if self._previous_summary is None:
            self._load_prior_session_context()

    def _load_prior_session_context(self) -> None:
        """Load the most recent receipt from the store as prior context.

        This bootstraps a new session with relevant compacted content from
        prior sessions without loading full transcripts. The model can then
        use context_search/context_expand to recover specific details.
        """
        try:
            receipt_ids = self._run_json(
                [
                    "search",
                    "--dir",
                    str(self.store_dir),
                    "--query",
                    "",
                    "--top-k",
                    "1",
                    *self._certified_store_args(),
                ],
                {},
            )
            # The search command returns a list — if we got results, use the
            # most recent receipt's compacted messages as a prior summary.
            if isinstance(receipt_ids, list) and receipt_ids:
                # Just note that prior receipts exist — don't load full content
                # to avoid bloating the new session's context.
                logger.debug(
                    "context-governor: %d prior receipts available in store %s",
                    len(receipt_ids),
                    self.store_dir,
                )
        except Exception:
            pass  # No prior receipts or binary unavailable — fine for new session

    def _set_defer_baseline(self, value: int) -> None:
        """Set the deferred-preflight baseline.

        The baseline lives under the internal name
        ``_last_rough_tokens_when_real_fit`` (legacy). The host's
        public name ``last_rough_tokens_when_real_prompt_fit`` is a
        property that reads from the internal name and writes back
        here — so any assignment to either name converges on the
        same internal value. This is the only direct write site.
        """
        self._last_rough_tokens_when_real_fit = int(value or 0)

    @property
    def last_rough_tokens_when_real_prompt_fit(self) -> int:
        """Public host-name alias for the defer baseline.

        Reads return the value held under the internal
        ``_last_rough_tokens_when_real_fit`` (legacy name). Writes
        route through ``_set_defer_baseline`` so both names stay
        in sync — external code can assign to either name and the
        other is updated atomically. This matches the built-in
        ``ContextCompressor`` host contract
        (``tests/run_agent/test_413_compression.py:581,622``,
        ``tests/agent/test_context_compressor.py:60,72,75,80,87``).
        """
        return self._last_rough_tokens_when_real_fit

    @last_rough_tokens_when_real_prompt_fit.setter
    def last_rough_tokens_when_real_prompt_fit(self, value: int) -> None:
        self._set_defer_baseline(value)

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self.last_receipt_id = None
        self.last_error = None
        self.last_warning = None
        self.last_summary_safety = None
        self.last_compaction_metrics = None
        self.fallback_event_count = 0
        self._llm_checkpoint_count = 0
        self._last_compress_aborted = False
        self._last_summary_error = None
        self._last_summary_fallback_used = False
        self._last_compression_made_progress = False
        # Reset host-contract attributes (matches built-in
        # ContextCompressor.on_session_reset).
        self.last_real_prompt_tokens = 0
        self.last_compression_rough_tokens = 0
        self.awaiting_real_usage_after_compression = False
        self._ineffective_compression_count = 0
        self._last_compression_savings_pct = 100.0
        self._set_defer_baseline(0)
        self._previous_summary = None
        self._lineage_session_id = ""
        self._session_db = None
        # A reset is a real user boundary. Any in-process prepared receipt was
        # not accepted by the old host transcript and must not leak into the
        # next logical conversation.
        self.discard_pending_compression(reason="session_reset")

    def bind_session_state(
        self,
        session_db: Any = None,
        session_id: str = "",
    ) -> None:
        """Bind SessionDB, restore the logical lineage, and reconcile pending.

        Resume-time provider repair is not durable, so recovery reads the raw
        active SessionDB projection. A pending receipt is activated only when
        its authenticated compacted projection is an exact normalized prefix
        of a durable in-place row set or compression-continuation tip.
        """
        if session_id:
            self.session_id = str(session_id)
        self._session_db = session_db
        self._lineage_session_id = self._compression_lineage_root(
            session_db,
            self.session_id or "default",
        )
        if session_db is not None:
            self._reconcile_pending_receipts(session_db, self.session_id)

    @staticmethod
    def _compression_lineage_root(session_db: Any, session_id: str) -> str:
        """Return the stable root following compression edges only."""
        current = str(session_id or "default")
        getter = getattr(type(session_db), "get_session", None)
        if session_db is None or not callable(getter):
            return current
        seen: set[str] = set()
        for _ in range(100):
            if not current or current in seen:
                break
            seen.add(current)
            row = getter(session_db, current)
            if not isinstance(row, dict):
                break
            parent_id = str(row.get("parent_session_id") or "")
            if not parent_id:
                break
            parent = getter(session_db, parent_id)
            if not isinstance(parent, dict) or parent.get("end_reason") != "compression":
                break
            current = parent_id
        return current or str(session_id or "default")

    def _governor_session_id(self) -> str:
        return self._lineage_session_id or self.session_id or "hermes-session"

    def get_active_compression_failure_cooldown(
        self,
    ) -> Optional[Dict[str, Any]]:
        """Return the live compression-failure cooldown, or None.

        The host reads this from agent/turn_context.py:368 to decide
        whether to skip preflight compression. The adapter's Rust binary
        tracks its own failure state internally; for the moment the
        adapter exposes no durable cooldown (the next turn's compact
        call will surface any failure via last_error / last_warning on
        get_status). Returns None to mean "no active cooldown, proceed".
        """
        return None

    # ------------------------------------------------------------------
    # Tool schemas — expose context_expand, context_search, context_status
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "context_expand",
                    "description": (
                        "Recover exact canonical UTF-8 text from a context-governor compaction receipt. "
                        "Use when the compacted summary references a receipt_id and item_id "
                        "that you need the full original text for. Provider-native multimodal "
                        "payloads are not reconstructed by this tool."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "receipt_id": {
                                "type": "string",
                                "description": "The receipt ID from the compaction summary",
                            },
                            "item_id": {
                                "type": "string",
                                "description": (
                                    "The exact item/source ID to expand "
                                    "(ctxi_... for local/legacy items or ctxs_... "
                                    "for transitive V2 sources)"
                                ),
                            },
                            "max_chars": {
                                "type": "integer",
                                "description": "Maximum characters to return",
                                "default": 100000,
                            },
                        },
                        "required": ["receipt_id", "item_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "context_search",
                    "description": (
                        "Search across all stored context-governor compaction receipts for "
                        "omitted content. Returns matching snippets from exact_store, "
                        "compacted_messages, and receipts."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Text to search for",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Maximum results",
                                "default": 10,
                            },
                            "scope": {
                                "type": "string",
                                "enum": ["all", "exact", "summary", "receipt"],
                                "default": "all",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "context_status",
                    "description": (
                        "Show context-governor engine status: compression count, "
                        "last receipt, last error, stored receipts."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "causal_provenance",
                    "description": (
                        "Compatibility alias for advisory synthetic tool telemetry history. "
                        "It does not provide causal provenance or proof and must not be "
                        "used to authorize skipped checks."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Tool name to look up in advisory telemetry history."
                                ),
                            },
                            "depth": {
                                "type": "integer",
                                "description": "Maximum history entries (default 5).",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if name == "context_expand":
                result = self._run_json(
                    [
                        "expand",
                        "--dir",
                        str(self.store_dir),
                        "--receipt",
                        args["receipt_id"],
                        "--item",
                        args["item_id"],
                        "--max-chars",
                        str(args.get("max_chars", 100000)),
                        *self._certified_store_args(),
                    ],
                    {},
                )
                # ``expand`` verifies the stored receipt and requested exact
                # bytes before returning. Record the event as a projection;
                # the Rust receipt/store remains the authority.
                self.fallback_event_count += 1
                if self.last_compaction_metrics is not None:
                    self.last_compaction_metrics["fallback_events"] = (
                        self.fallback_event_count
                    )
                    self.last_compaction_metrics["integrity_result"] = (
                        "exact_expand_verified"
                    )
                return json.dumps(result)
            elif name == "context_search":
                scope = args.get("scope", "all")
                cmd = ["search", "--dir", str(self.store_dir), "--query", args["query"]]
                cmd.extend(["--top-k", str(args.get("top_k", 10))])
                cmd.extend(self._certified_store_args())
                if scope != "all":
                    cmd.extend(["--scope", scope])
                result = self._run_json(cmd, {})
                return json.dumps(result)
            elif name == "context_status":
                return json.dumps(self.get_status())
            elif name == "causal_provenance":
                if not self._telemetry_available:
                    return json.dumps({
                        "error": "advisory telemetry unavailable: cea-bridge not found",
                        "history": [],
                        "evidence_kind": "synthetic_telemetry",
                        "causal_claim": False,
                    })
                query_payload = {
                    "query": args.get("query", ""),
                    "depth": args.get("depth", 5),
                }
                result = self._run_telemetry_json(["query-provenance"], query_payload)
                return json.dumps(result)
            else:
                return json.dumps({"error": f"Unknown context-governor tool: {name}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    # ------------------------------------------------------------------
    # Core compaction
    # ------------------------------------------------------------------

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        self._last_compress_aborted = False
        self._last_summary_error = None
        self._last_summary_fallback_used = False
        self._last_compression_made_progress = False

        if not messages:
            return messages
        if self._pending_admission is not None:
            error = (
                "a prior context-governor receipt is still pending host "
                "commit; refusing to start a competing compaction"
            )
            self.last_error = error
            self._last_compress_aborted = True
            self._last_summary_error = error
            return messages

        started = time.monotonic()
        before_bytes = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
        before_tokens = sum(
            max(1, len(self._content_to_text(message.get("content"))) // 4)
            for message in messages
            if isinstance(message, dict)
        )
        target_tokens = self._target_tokens(current_tokens)
        metrics: dict[str, Any] = {
            "before_tokens": before_tokens,
            "before_bytes": before_bytes,
            "before_messages": len(messages),
            "target_tokens": target_tokens,
            "deterministic_after_tokens": None,
            "deterministic_reduction_tokens": None,
            "llm_call": False,
            "llm_call_reason": "not_evaluated",
            "llm_retry": False,
            "llm_retry_reason": None,
            "summarizer_model": None,
            "summarizer_provider": None,
            "llm_latency_ms": None,
            "passes": 0,
            "after_tokens": before_tokens,
            "after_bytes": before_bytes,
            "after_messages": len(messages),
            "protected_region_size": self.protect_first_n + self.protect_last_n,
            "summary_id": None,
            "covered_source_hashes": [],
            "exact_fallback_available": False,
            "fallback_events": self.fallback_event_count,
            "integrity_result": "not_persisted",
            "elapsed_ms": None,
        }
        self.last_compaction_metrics = metrics

        # Capture immutable original tool telemetry before pruning/summarization.
        # This is fail-open and cannot alter compaction behavior.
        self._record_tool_telemetry(messages)

        # The Rust engine owns both prompt reduction and exact fallback. Do not
        # replace old tool output before handing it the transcript: doing so
        # stores only an adapter-generated one-line placeholder and makes the
        # original unrecoverable from its receipt. The core's allocator decides
        # what to omit/quarantine while retaining the authoritative exact copy.
        # Hermes appends its todo snapshot *after* an engine returns and before
        # the compacted transcript is persisted. That host-owned synthetic block
        # is refreshed at every boundary, so it is not part of the governor's
        # receipt projection. Strip the stale copy before recursive compaction;
        # otherwise the Rust store correctly rejects the child because the
        # parent's final user message is no longer an exact prefix.
        source_messages = self._without_host_todo_snapshots(messages)

        # Advisory telemetry can conservatively protect a bounded few messages;
        # it never creates causal claims or authorizes check skipping.
        telemetry_scores = self._score_telemetry_relevance(source_messages, focus_topic)
        telemetry_protect_last_n = self._advisory_protect_last_n(
            len(source_messages), telemetry_scores
        )
        metrics["protected_region_size"] = min(
            len(source_messages), self.protect_first_n + telemetry_protect_last_n
        )

        certified_args = self._certified_store_args()
        assert self._key_binding is not None
        governor_messages = [
            self._message_to_governor(m, i)
            for i, m in enumerate(source_messages)
            if isinstance(m, dict)
        ]
        # Session resume prepares a provider-safe replay by merging consecutive
        # assistant/user messages and dropping orphaned tool results.  That
        # request-only repair must not replace the authenticated transcript
        # identity owned by context-governor. Rehydrate the exact parent prefix
        # whenever the current prefix is provably the host-repaired projection
        # of a stored receipt. Older Ares archives may also be missing the
        # provider-facing ``name`` fields, which the same narrow bridge covers.
        governor_messages = self._rehydrate_legacy_parent_prefix(governor_messages)

        request = {
            "session_id": self._governor_session_id(),
            "messages": governor_messages,
            "policy": {
                "target_tokens": target_tokens,
                "protect_first_n": self.protect_first_n,
                "protect_last_n": telemetry_protect_last_n,
                "summary_max_chars": self._policy["summary_max_chars"],
                "allocator": self._policy["allocator"],
                "semantic_memory_enabled": self._policy["semantic_memory_enabled"],
                "archive_memory_enabled": self._policy["archive_memory_enabled"],
                "budget_mode": self._policy["budget_mode"],
                "token_counter": self._policy["token_counter"],
                "unsafe_summary_policy": self._policy["unsafe_summary_policy"],
                "checkpoint": {
                    "strategy": self._checkpoint_strategy_json(),
                    "max_checkpoints_per_session": self._max_checkpoints(),
                },
                "max_lineage_generation": self._policy["max_lineage_generation"],
                "max_provenance_bytes": self._policy["max_provenance_bytes"],
                "min_net_savings_tokens": self._request_min_net_savings_tokens(),
            },
            "focus": focus_topic,
        }
        try:
            response = self._run_json(
                [
                    "compact-v2",
                    "--dir",
                    str(self.store_dir),
                    *certified_args,
                ],
                request,
            )
            pending_receipt = response.get("receipt") or {}
            if pending_receipt.get("schema") != "ContextCompactionReceiptV2":
                raise ValueError(
                    "compact-v2 returned a non-V2 receipt; recursive provenance is required"
                )
            pending_receipt_id = pending_receipt.get("receipt_id")
            if not isinstance(pending_receipt_id, str) or not pending_receipt_id:
                raise ValueError("compact returned no receipt_id")
            raw_compacted = response.get("compacted_messages") or []
            expected_summary_id = self._expected_summary_message_id(response)
            summary_matches = 0
            compacted = []
            for raw_message in raw_compacted:
                if not isinstance(raw_message, dict):
                    continue
                host_message = self._message_from_governor(raw_message)
                if self._is_exact_summary_message(raw_message, expected_summary_id):
                    summary_matches += 1
                    # Transient adapter identity. _message_to_governor ignores
                    # underscore-prefixed host fields, so this can select the
                    # Rust-owned summary without becoming provider input or
                    # receipt metadata.
                    host_message["_context_governor_summary_id"] = expected_summary_id
                compacted.append(host_message)
            if summary_matches > 1:
                raise ValueError(
                    "compact-v2 returned multiple messages for one allocation-plan summary"
                )
            compacted = self._ensure_latest_user_last(source_messages, compacted)
            compacted = self._sanitize_dangling_tool_messages(compacted)
            compacted = self._sanitize_tool_pairs(compacted)
            # Tool-pair repair may insert a synthetic result after the active
            # instruction. Reassert the host contract before finalization.
            compacted = self._ensure_latest_user_last(source_messages, compacted)
            compacted = self._preserve_multimodal_tail(source_messages, compacted)

            deterministic_tokens = sum(
                max(1, len(self._content_to_text(message.get("content"))) // 4)
                for message in compacted
                if isinstance(message, dict)
            )
            metrics["passes"] = 1
            metrics["deterministic_after_tokens"] = deterministic_tokens
            metrics["deterministic_reduction_tokens"] = max(
                0, before_tokens - deterministic_tokens
            )

            # Deterministic compaction owns the fast path. An LLM may replace
            # the extractive summary only at a receipt-proven fixed point or
            # after the deterministic pass has reached diminishing returns.
            checkpoint, checkpoint_reason = self._llm_checkpoint_decision(
                response,
                target_tokens=target_tokens,
            )
            metrics["llm_call_reason"] = checkpoint_reason
            llm_checkpoint_applied = False
            pending_previous_summary: str | None = None
            if checkpoint:
                metrics["passes"] = 2
                deterministic_projection = copy.deepcopy(compacted)
                candidate = self._enhance_with_llm_summary(
                    compacted, source_messages, response, focus_topic
                )
                if candidate != deterministic_projection:
                    # Only Rust's configured token counter can decide whether
                    # an LLM projection satisfies the same budget as the
                    # deterministic pass. Finalize first, then accept or
                    # re-finalize the unchanged deterministic projection.
                    try:
                        candidate_response = self._finalize_response(
                            response, candidate
                        )
                    except Exception as exc:
                        warning = (
                            "LLM summary could not be finalized by the governor; "
                            f"using extractive summary: {exc}"
                        )
                        self._record_summary_fallback(warning)
                        logger.warning("context-governor: %s", warning)
                        compacted = deterministic_projection
                        metrics["llm_call_reason"] = (
                            f"{checkpoint_reason}:fallback_extract"
                        )
                    else:
                        candidate_receipt = candidate_response.get("receipt") or {}
                        candidate_tokens = candidate_receipt.get(
                            "compacted_approx_tokens"
                        )
                        if (
                            isinstance(candidate_tokens, int)
                            and not isinstance(candidate_tokens, bool)
                            and candidate_tokens <= target_tokens
                        ):
                            # The candidate receipt serves only the
                            # token-budget check below. The authoritative
                            # binding happens once, after host-equivalent
                            # alternation repair (see finalize below), so the
                            # receipt describes the exact transcript the host
                            # persists.
                            compacted = candidate
                            llm_checkpoint_applied = True
                            pending_previous_summary = self._summary_message_content(
                                candidate
                            )
                            metrics["llm_call_reason"] = (
                                f"{checkpoint_reason}:prepared"
                            )
                        else:
                            observed = (
                                str(candidate_tokens)
                                if isinstance(candidate_tokens, int)
                                and not isinstance(candidate_tokens, bool)
                                else "invalid"
                            )
                            warning = (
                                "LLM summary exceeded target under the governor "
                                f"token counter ({observed} > {target_tokens} tokens); "
                                "using extractive summary"
                            )
                            self._record_summary_fallback(warning)
                            logger.warning("context-governor: %s", warning)
                            compacted = deterministic_projection
                            metrics["llm_call_reason"] = (
                                f"{checkpoint_reason}:fallback_extract"
                            )
                elif metrics.get("llm_call_reason") == checkpoint_reason:
                    metrics["llm_call_reason"] = f"{checkpoint_reason}:fallback_extract"

            # Sanitation and the audited LLM checkpoint can both mutate the
            # emitted transcript. Rebind hashes/counts to that final adapter
            # output before persistence; never store the stale core response.
            # The host repairs role-alternation in memory immediately after
            # compress() returns. Apply the same repair BEFORE finalize so
            # the receipt describes the exact transcript the host persists;
            # otherwise the next compaction's input no longer exactly
            # prefixes the stored parent and recursive lineage rejects it —
            # which made deterministic compaction one-shot per session.
            compacted = self._repair_for_host_alternation(compacted)
            compacted = self._ensure_latest_user_last(source_messages, compacted)
            compacted = self._preserve_multimodal_tail(source_messages, compacted)
            response = self._finalize_response(response, compacted)
            finalized_messages = response.get("compacted_messages")
            if not isinstance(finalized_messages, list):
                raise ValueError("finalize returned no compacted_messages list")
            compacted = [
                self._message_from_governor(message)
                for message in finalized_messages
                if isinstance(message, dict)
            ]
            latest_user = next(
                (
                    message
                    for message in reversed(source_messages)
                    if isinstance(message, dict) and message.get("role") == "user"
                ),
                None,
            )
            if latest_user is not None and (
                not compacted
                or compacted[-1].get("role") != "user"
                or compacted[-1].get("content") != latest_user.get("content")
            ):
                raise ValueError(
                    "finalize changed or reordered the latest user message"
                )
            receipt = response.get("receipt") or {}
            exact_refs = receipt.get("covered_original_sources") or []
            metrics["summary_id"] = pending_receipt_id
            metrics["covered_source_hashes"] = [
                str(ref.get("content_blake3"))
                for ref in exact_refs
                if isinstance(ref, dict) and ref.get("content_blake3")
            ]
            metrics["exact_fallback_available"] = False
            metrics["integrity_result"] = "pending_host_commit"

            # Track compression effectiveness for anti-thrashing
            original_tokens = sum(
                max(1, len(self._content_to_text(m.get("content"))) // 4)
                for m in source_messages
                if isinstance(m, dict)
            )
            compacted_tokens = sum(
                max(1, len(self._content_to_text(m.get("content"))) // 4)
                for m in compacted
                if isinstance(m, dict)
            )
            savings_pct = (
                (original_tokens - compacted_tokens) / max(1, original_tokens)
            ) * 100
            metrics["after_tokens"] = compacted_tokens
            metrics["after_bytes"] = len(
                json.dumps(compacted, ensure_ascii=False).encode("utf-8")
            )
            metrics["after_messages"] = len(compacted)
            metrics["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)

            self._last_compression_made_progress = compacted != messages

            pending_info = self._prepare_response(response)
            self._pending_admission = {
                "receipt_id": pending_receipt_id,
                "response": copy.deepcopy(response),
                "pending_info": copy.deepcopy(pending_info),
                "llm_checkpoint_applied": llm_checkpoint_applied,
                "previous_summary": pending_previous_summary,
                "checkpoint_reason": checkpoint_reason,
                "savings_pct": savings_pct,
                "exact_fallback_available": bool(exact_refs),
                "physical_session_id": self.session_id,
                "lineage_session_id": self._governor_session_id(),
            }
            self.last_error = None

            return compacted or messages
        except Exception as exc:
            if self._pending_admission is not None:
                try:
                    self.discard_pending_compression(reason="adapter_exception")
                except Exception:
                    logger.warning(
                        "context-governor: failed to discard receipt after "
                        "adapter exception; recovery will retry",
                        exc_info=True,
                    )
            self.last_error = str(exc)
            self._last_compress_aborted = True
            self._last_summary_error = str(exc)
            self._last_compression_made_progress = False
            failure_type = self._classify_subprocess_error(exc)
            if failure_type == "auth":
                logger.error("context-governor auth failure: %s", exc)
            elif failure_type == "network":
                logger.warning(
                    "context-governor network failure (will retry next turn): %s", exc
                )
            elif failure_type == "timeout":
                logger.warning("context-governor subprocess timeout: %s", exc)
            else:
                logger.warning(
                    "context-governor compaction failed; keeping original messages: %s",
                    exc,
                )
            metrics["llm_call_reason"] = metrics.get("llm_call_reason") or "failed"
            metrics["integrity_result"] = "failed_closed_to_authoritative_source"
            metrics["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
            return messages

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update({
            "engine": self.name,
            "binary": str(self.binary),
            "available": self.is_available(),
            "store_dir": str(self.store_dir),
            "last_receipt_id": self.last_receipt_id,
            "last_error": self.last_error,
            "last_warning": self.last_warning,
            "last_summary_safety": self.last_summary_safety,
            "last_compaction_metrics": copy.deepcopy(self.last_compaction_metrics),
            "fallback_event_count": self.fallback_event_count,
            "llm_checkpoint_count": self._llm_checkpoint_count,
            "compression_count": self.compression_count,
            "ineffective_compression_count": self._ineffective_compression_count,
            "last_savings_pct": round(self._last_compression_savings_pct, 1),
            "policy": dict(self._policy),
            "telemetry_advisory": True,
            "telemetry_available": self._telemetry_available,
            "telemetry_binary": self.telemetry_binary or None,
            "telemetry_db_path": str(self.telemetry_db_path),
            "telemetry_max_additional_protected_messages": self._policy[
                "telemetry_max_additional_protected_messages"
            ],
        })
        return status

    # ------------------------------------------------------------------
    # Tool output pruning pre-pass (cheap, no LLM call)
    # ------------------------------------------------------------------

    def _prune_old_tool_results(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Prune old tool results: dedupe, summarize, strip images, truncate args.

        Walks backward from the end, protecting the most recent messages
        within protect_last_n. Older tool results get replaced with
        informative 1-line summaries.
        """
        if not messages:
            return messages

        result = [m.copy() if isinstance(m, dict) else m for m in messages]
        pruned = 0

        # Build index: tool_call_id -> (tool_name, arguments_json)
        call_id_to_tool: Dict[str, tuple] = {}
        for msg in result:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        cid = tc.get("id", "")
                        fn = tc.get("function", {})
                        call_id_to_tool[cid] = (
                            fn.get("name", "unknown"),
                            fn.get("arguments", ""),
                        )

        prune_boundary = max(0, len(result) - self.protect_last_n)

        # Pass 1: Deduplicate identical tool results (keep newest)
        content_hashes: dict = {}
        for i in range(len(result) - 1, -1, -1):
            msg = result[i]
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            content = msg.get("content") or ""
            if not isinstance(content, str) or len(content) < 200:
                continue
            h = hash(content) & 0xFFFFFFFF
            if h in content_hashes:
                result[i] = {
                    **msg,
                    "content": "[Duplicate tool output — same content as a more recent call]",
                }
                pruned += 1
            else:
                content_hashes[h] = i

        # Pass 2: Replace old tool results with informative summaries
        for i in range(prune_boundary):
            msg = result[i]
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            # Multimodal content: strip image payloads
            if isinstance(content, list):
                stripped = self._strip_image_parts(content)
                if stripped is not None:
                    result[i] = {**msg, "content": stripped}
                    pruned += 1
                continue
            if not isinstance(content, str) or not content:
                continue
            if len(content) <= 200:
                continue
            # Generate informative summary
            call_id = msg.get("tool_call_id", "")
            tool_name, tool_args = call_id_to_tool.get(call_id, ("unknown", ""))
            summary = self._summarize_tool_result(tool_name, tool_args, content)
            result[i] = {**msg, "content": summary}
            pruned += 1

        # Pass 3: Truncate large tool_call arguments in assistant messages
        for i in range(prune_boundary):
            msg = result[i]
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            if not msg.get("tool_calls"):
                continue
            new_tcs = []
            modified = False
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict):
                    args_str = tc.get("function", {}).get("arguments", "")
                    if len(args_str) > 500:
                        # Truncate but keep valid JSON-ish
                        new_args = args_str[:400] + "...[truncated]"
                        tc = {
                            **tc,
                            "function": {**tc["function"], "arguments": new_args},
                        }
                        modified = True
                new_tcs.append(tc)
            if modified:
                result[i] = {**msg, "tool_calls": new_tcs}

        return result

    @staticmethod
    def _strip_image_parts(content: list) -> str | None:
        """Strip image parts from multimodal content, return text summary."""
        text_parts = []
        had_image = False
        for part in content:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif part.get("type") in {"image", "image_url", "input_image"}:
                    had_image = True
        if had_image:
            return "[image removed] " + " ".join(text_parts)[:200]
        return None

    @staticmethod
    def _summarize_tool_result(tool_name: str, tool_args: str, content: str) -> str:
        """Generate an informative 1-line summary of a tool result."""
        lines = content.strip().split("\n")
        line_count = len(lines)

        # Try to extract exit code
        exit_code = None
        for line in lines[-5:]:
            if "exit_code" in line.lower():
                try:
                    import re

                    m = re.search(r'exit_code["\']?\s*[:=]\s*["\']?(\d+)', line)
                    if m:
                        exit_code = m.group(1)
                        break
                except Exception:
                    pass

        # Try to parse tool args for context
        args_preview = ""
        if tool_args:
            try:
                parsed = json.loads(tool_args)
                if isinstance(parsed, dict):
                    for key in ("command", "path", "file_path", "query"):
                        if key in parsed:
                            args_preview = (
                                f" `{parsed[key]}`"
                                if len(str(parsed[key])) < 80
                                else ""
                            )
                            break
            except Exception:
                pass

        parts = [f"[{tool_name}]{args_preview}"]
        if exit_code is not None:
            parts.append(f"exit {exit_code}")
        parts.append(f"{line_count} lines output")
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Message format preservation
    # ------------------------------------------------------------------

    @staticmethod
    def _without_host_todo_snapshots(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return a copy without host-refreshed todo snapshot suffixes."""
        from tools.todo_tool import TODO_INJECTION_HEADER

        normalized = copy.deepcopy(messages)
        for message in normalized:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                marker = content.find(TODO_INJECTION_HEADER)
                if marker >= 0:
                    message["content"] = content[:marker].rstrip()
            elif isinstance(content, list):
                message["content"] = [
                    part
                    for part in content
                    if not (
                        isinstance(part, dict)
                        and str(part.get("text") or "")
                        .lstrip()
                        .startswith(TODO_INJECTION_HEADER)
                    )
                ]
        return normalized

    def _rehydrate_legacy_parent_prefix(
        self,
        governor_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Restore a receipt prefix transformed for provider-safe replay.

        Live session restore repairs role alternation on an in-memory copy;
        older Ares builds also dropped ``name`` while archiving compacted
        messages. Both transformations can make a resumed transcript differ
        from its authenticated parent receipt. This is deliberately narrow: a
        candidate is accepted only when applying the same deterministic host
        repair and removing the known non-durable fields yields an otherwise
        exact prefix match. The core subsequently verifies the receipt
        signature and complete lineage as usual.
        """
        governor_session_id = self._governor_session_id()
        if not governor_session_id or not governor_messages:
            return governor_messages

        candidates: list[tuple[int, str, List[Dict[str, Any]]]] = []
        try:
            for path in self.store_dir.glob("ctxr_*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    receipt = payload.get("receipt") or {}
                    compacted = payload.get("compacted_messages") or []
                    if (
                        receipt.get("session_id") != governor_session_id
                        or not isinstance(compacted, list)
                        or not compacted
                        or any(not isinstance(message, dict) for message in compacted)
                    ):
                        continue
                    candidates.append((
                        int(receipt.get("generation") or 0),
                        str(receipt.get("created_utc") or ""),
                        compacted,
                    ))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
        except OSError:
            return governor_messages

        def legacy_projection(message: Dict[str, Any]) -> Dict[str, Any]:
            projection = copy.deepcopy(message)
            projection.pop("name", None)
            # Hermes never had a durable generic message-id column.  The
            # adapter intentionally retains tool-call identity, but assistant
            # summary ids from old receipts were not persisted.
            if projection.get("role") != "tool":
                projection.pop("id", None)
            return projection

        def repaired_projection(
            compacted: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            host_messages = [
                self._message_from_governor(message) for message in compacted
            ]
            try:
                from agent.agent_runtime_helpers import repair_message_sequence

                repair_message_sequence(None, host_messages)
            except Exception:
                return []
            return [
                self._message_to_governor(message, i)
                for i, message in enumerate(host_messages)
            ]

        for _generation, _created, compacted in sorted(candidates, reverse=True):
            durable_projection = [
                self._message_to_governor(self._message_from_governor(message), i)
                for i, message in enumerate(compacted)
            ]
            projections = [durable_projection]
            repaired = repaired_projection(compacted)
            if repaired and repaired != durable_projection:
                projections.append(repaired)
            for projection in projections:
                if len(projection) > len(governor_messages):
                    continue
                current_prefix = governor_messages[: len(projection)]
                if [legacy_projection(message) for message in projection] != [
                    legacy_projection(message) for message in current_prefix
                ]:
                    continue
                if durable_projection == current_prefix:
                    continue
                logger.info(
                    "context-governor: rehydrated authenticated receipt prefix "
                    "for session %s (host_messages=%d canonical_messages=%d)",
                    governor_session_id,
                    len(projection),
                    len(compacted),
                )
                return copy.deepcopy(compacted) + governor_messages[len(projection) :]
        return governor_messages

    def _message_to_governor(self, msg: Dict[str, Any], idx: int) -> Dict[str, Any]:
        role = msg.get("role") or "assistant"
        if role not in {"system", "user", "assistant", "tool"}:
            role = "assistant"
        out = {
            "role": role,
            "content": self._content_to_text(msg.get("content")),
        }
        # Hermes durably stores tool-call identity but has no generic provider
        # message-id column. Keep only the identity that survives an in-place
        # archive/reload so the next receipt can prove an exact parent prefix.
        message_id = msg.get("tool_call_id") if role == "tool" else None
        if message_id:
            out["id"] = str(message_id)
        message_name = msg.get("name") or msg.get("tool_name")
        if message_name:
            out["name"] = str(message_name)
        # Preserve OpenAI-specific fields in metadata for roundtrip
        metadata = {}
        if isinstance(msg.get("metadata"), dict):
            metadata["hermes_metadata"] = copy.deepcopy(msg["metadata"])
        if isinstance(msg.get("content"), list):
            # Rust classifies the text projection, while the canonical receipt
            # retains the provider-native parts for lossless host roundtrip.
            metadata["hermes_content"] = copy.deepcopy(msg["content"])
        if msg.get("tool_calls"):
            metadata["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id"):
            metadata["tool_call_id"] = msg["tool_call_id"]
        if metadata:
            out["metadata"] = metadata
        return out

    @staticmethod
    def _message_from_governor(msg: Dict[str, Any]) -> Dict[str, Any]:
        metadata = msg.get("metadata") or {}
        content = msg.get("content") or ""
        if isinstance(metadata, dict) and isinstance(
            metadata.get("hermes_content"), list
        ):
            content = copy.deepcopy(metadata["hermes_content"])
        out = {"role": msg.get("role") or "assistant", "content": content}
        if msg.get("name"):
            # ``name`` is the provider-facing shape; ``tool_name`` is Hermes'
            # durable SessionDB column. Carry both so the in-memory transcript
            # and a resumed transcript normalize to the same receipt message.
            out["name"] = msg.get("name")
            out["tool_name"] = msg.get("name")
        if msg.get("id") and msg.get("role") == "tool":
            out["id"] = msg.get("id")
        # Restore OpenAI-specific fields from metadata
        if isinstance(metadata, dict):
            if isinstance(metadata.get("hermes_metadata"), dict):
                out["metadata"] = copy.deepcopy(metadata["hermes_metadata"])
            if metadata.get("tool_calls"):
                out["tool_calls"] = metadata["tool_calls"]
            if metadata.get("tool_call_id"):
                out["tool_call_id"] = metadata["tool_call_id"]
        return out

    @staticmethod
    def _expected_summary_message_id(response: dict[str, Any]) -> str | None:
        """Return the one Rust allocation-plan summary identity, if valid."""
        plan = response.get("allocation_plan") or {}
        receipt = response.get("receipt") or {}
        if not isinstance(plan, dict) or not isinstance(receipt, dict):
            return None
        plan_id = plan.get("plan_id")
        if (
            not isinstance(plan_id, str)
            or not re.fullmatch(r"ctxp_[A-Za-z0-9_-]+", plan_id)
            or receipt.get("allocation_plan_id") != plan_id
        ):
            return None
        return f"summary_{plan_id}"

    @staticmethod
    def _is_exact_summary_message(
        message: dict[str, Any], expected_summary_id: str | None
    ) -> bool:
        """Match every core-owned summary discriminator, never user markers."""
        metadata = message.get("metadata")
        return bool(
            expected_summary_id
            and message.get("id") == expected_summary_id
            and message.get("name") == "context_governor"
            and message.get("role") in {"assistant", "user"}
            and isinstance(metadata, dict)
            and metadata.get("compressed_summary") is True
        )

    def _sanitize_dangling_tool_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert dangling tool messages to assistant text to avoid provider rejections.

        A dangling tool message is one without a preceding assistant message
        containing tool_calls. Providers reject these.
        """
        result = []
        prev_had_tool_call = False
        for msg in messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "tool"
                and not prev_had_tool_call
            ):
                # Dangling tool message — convert to assistant text
                content = msg.get("content", "")
                tool_call_id = msg.get("tool_call_id", "")
                text = (
                    f"[Tool result {tool_call_id}]: {content}"
                    if tool_call_id
                    else f"[Tool result]: {content}"
                )
                result.append({"role": "assistant", "content": text})
            else:
                result.append(msg)
            # Track whether this message had tool_calls
            prev_had_tool_call = bool(isinstance(msg, dict) and msg.get("tool_calls"))
        return result

    def _sanitize_tool_pairs(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fix orphaned tool_call / tool_result pairs after compression.

        Two failure modes:
        1. A tool result references a call_id whose assistant tool_call was
           removed. The API rejects this.
        2. An assistant message has tool_calls whose results were dropped.
           The API rejects this because every tool_call must be followed by
           a tool result with the matching call_id.

        This method removes orphaned results and inserts stub results for
        orphaned calls so the message list is always well-formed.
        """
        # Collect surviving call IDs
        surviving_call_ids: set = set()
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        cid = tc.get("id", "") or tc.get("call_id", "")
                        if cid:
                            surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                cid = msg.get("tool_call_id", "")
                if cid:
                    result_call_ids.add(cid)

        # 1. Remove tool results whose call_id has no matching assistant tool_call
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m
                for m in messages
                if not (
                    isinstance(m, dict)
                    and m.get("role") == "tool"
                    and m.get("tool_call_id") in orphaned_results
                )
            ]
            logger.debug(
                "context-governor: removed %d orphaned tool result(s)",
                len(orphaned_results),
            )

        # 2. Add stub results for assistant tool_calls whose results were dropped
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        if isinstance(tc, dict):
                            cid = tc.get("id", "") or tc.get("call_id", "")
                            if cid in missing_results:
                                patched.append({
                                    "role": "tool",
                                    "content": "[Result from earlier conversation — see context summary]",
                                    "tool_call_id": cid,
                                })
            messages = patched
            logger.debug(
                "context-governor: added %d stub tool result(s)", len(missing_results)
            )

        return messages

    def _repair_for_host_alternation(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply the host's role-alternation repair before receipt finalize.

        compact-v2 deliberately emits compacted tool-result notes as
        ``assistant`` messages (they are inline notes, not provider tool
        results), which can leave consecutive-assistant runs. Hermes'
        conversation loop repairs those in memory immediately after
        ``compress()`` returns. If the receipt instead describes the
        un-repaired transcript, the next compaction's child input no longer
        exactly prefixes the stored parent compacted transcript and the
        Rust recursive-lineage check rejects it — which made deterministic
        compaction effectively one-shot per session (every later attempt
        failed, and the LLM checkpoint could never reach its generation
        boundary). Running the identical repair here, before finalize,
        binds the receipt to the exact transcript the host persists.
        """
        if not messages:
            return messages
        try:
            from agent.agent_runtime_helpers import repair_message_sequence

            repairs = repair_message_sequence(None, messages)
        except Exception as exc:
            logger.warning(
                "context-governor: host alternation repair unavailable (%s); "
                "receipt may diverge from the live transcript",
                exc,
            )
            return messages
        if repairs:
            logger.info(
                "context-governor: applied host-equivalent alternation repair "
                "(%d merge/drop) before receipt finalize",
                repairs,
            )
        return messages

    def _preserve_multimodal_tail(
        self, original: List[Dict[str, Any]], compacted: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Preserve original multimodal content for messages in the protected tail.

        The Rust crate only handles text. For messages that survive into the
        compacted output, restore any original multimodal content (images, etc.)
        from the original message list.
        """
        # Build a lookup from original messages by content text
        original_by_text: dict = {}
        for msg in original:
            if isinstance(msg, dict):
                text = self._content_to_text(msg.get("content"))
                if text and text not in original_by_text:
                    original_by_text[text] = msg

        result = []
        for msg in compacted:
            if isinstance(msg, dict):
                text = msg.get("content", "")
                if isinstance(text, str) and text in original_by_text:
                    orig = original_by_text[text]
                    # If the original had multimodal content, restore it
                    orig_content = orig.get("content")
                    if isinstance(orig_content, list) and msg.get("role") == orig.get(
                        "role"
                    ):
                        result.append({**msg, "content": orig_content})
                        continue
            result.append(msg)
        return result

    # ------------------------------------------------------------------
    # LLM summary enhancement
    # ------------------------------------------------------------------

    def _checkpoint_strategy_json(self) -> Any:
        """Normalize the configured host checkpoint policy to Rust serde JSON."""
        raw = self._policy.get("checkpoint_strategy", "after_n:2")
        if isinstance(raw, dict) and len(raw) == 1:
            key, value = next(iter(raw.items()))
            if (
                key == "after_n"
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
            ):
                return {"after_n": value}
            if (
                key == "threshold_pct"
                and isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= 100
            ):
                return {"threshold_pct": value}
            return "off"
        value = str(raw or "").strip().lower()
        if value in {"off", "ineffective_only"}:
            return value
        match = re.fullmatch(r"after_n:(\d+)", value)
        if match and int(match.group(1)) > 0:
            return {"after_n": int(match.group(1))}
        match = re.fullmatch(r"threshold_pct:(\d+)", value)
        if match and 0 <= int(match.group(1)) <= 100:
            return {"threshold_pct": int(match.group(1))}
        return "off"

    def _max_checkpoints(self) -> int | None:
        raw = self._policy.get("max_checkpoints", 10)
        if raw is None:
            return None
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    def _llm_checkpoint_decision(
        self,
        response: dict[str, Any],
        *,
        target_tokens: int,
    ) -> tuple[bool, str]:
        if self._summary_mode != "llm":
            return False, "summary_mode_extractive"
        strategy = self._checkpoint_strategy_json()
        if strategy == "off":
            return False, "checkpoint_strategy_off_or_invalid"
        receipt = self._validated_checkpoint_receipt(response)
        if receipt is None:
            return False, "checkpoint_receipt_invalid"

        due = False
        reason = "checkpoint_strategy_unrecognized"
        if isinstance(strategy, dict) and "after_n" in strategy:
            every = int(strategy["after_n"])
            # Receipt generation is the durable compaction ordinal. Process
            # counters reset whenever Desktop/gateway restarts and therefore
            # cannot own a deterministic checkpoint schedule.
            ordinal = int(receipt["generation"])
            due = ordinal % every == 0
            reason = f"after_n:{every}:ordinal:{ordinal}"
        elif strategy == "ineffective_only":
            due = self._deterministic_summary_checkpoint_ready(
                response, target_tokens=target_tokens
            )
            reason = (
                "deterministic_ineffective"
                if due
                else "deterministic_boundary_not_reached"
            )
        elif isinstance(strategy, dict) and "threshold_pct" in strategy:
            original = int(receipt.get("original_approx_tokens") or 0)
            savings = int(receipt.get("token_savings_estimate") or 0)
            savings_pct = (savings / original * 100.0) if original > 0 else 100.0
            threshold = int(strategy["threshold_pct"])
            due = savings_pct <= threshold
            reason = f"threshold_pct:{threshold}:observed:{savings_pct:.3f}"

        if not due:
            return False, reason

        maximum = self._max_checkpoints()
        if maximum is not None:
            if maximum == 0:
                return False, "checkpoint_limit_reached"
            durable_count = self._durable_llm_checkpoint_count(maximum)
            if durable_count is None:
                return False, "checkpoint_history_unavailable"
            observed_checkpoints = max(self._llm_checkpoint_count, durable_count)
            if observed_checkpoints >= maximum:
                return False, "checkpoint_limit_reached"
        return True, reason

    def _durable_llm_checkpoint_count(self, maximum: int) -> int | None:
        """Count applied checkpoints through the verified receipt search path.

        Every accepted LLM carrier names its own receipt. A later deterministic
        summary may quote an older marker, so a hit counts only when the marker
        receipt equals the verified hit receipt. ``search`` loads and verifies
        each signed lineage before returning it.
        """
        governor_session_id = self._governor_session_id()
        if not governor_session_id:
            return 0
        session_sha256 = hashlib.sha256(
            governor_session_id.encode("utf-8")
        ).hexdigest()
        session_marker = f"llm_checkpoint_session_sha256={session_sha256}"
        try:
            result = self._run_json(
                [
                    "search",
                    "--dir",
                    str(self.store_dir),
                    "--query",
                    session_marker,
                    "--scope",
                    "summary",
                    "--top-k",
                    str(min(1024, max(64, maximum + 1))),
                    *self._certified_store_args(),
                ],
                {},
            )
        except Exception as exc:
            warning = (
                "could not verify durable LLM checkpoint history; "
                "skipping checkpoint: "
                f"{self._safe_summary_diagnostic(exc)}"
            )
            self._record_summary_warning("checkpoint_history_unavailable", warning)
            logger.warning("context-governor: %s", warning)
            return None
        if not isinstance(result, list):
            warning = (
                "could not verify durable LLM checkpoint history; skipping "
                "checkpoint: search returned an invalid response"
            )
            self._record_summary_warning("checkpoint_history_unavailable", warning)
            logger.warning("context-governor: %s", warning)
            return None
        seen: set[str] = set()
        for row in result:
            if not isinstance(row, dict):
                continue
            receipt_id = row.get("receipt_id")
            hit = row.get("hit")
            snippet = hit.get("snippet") if isinstance(hit, dict) else None
            if not isinstance(receipt_id, str) or not isinstance(snippet, str):
                continue
            if (
                re.fullmatch(r"ctxr_[0-9a-f]{32}", receipt_id)
                and session_marker in snippet
                and f"llm_checkpoint_receipt={receipt_id}" in snippet
            ):
                seen.add(receipt_id)
        return len(seen)

    @staticmethod
    def _validated_checkpoint_receipt(
        response: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return receipt evidence suitable for deterministic scheduling."""
        receipt = response.get("receipt") or {}
        if not isinstance(receipt, dict):
            return None
        hashes = (
            receipt.get("original_transcript_blake3"),
            receipt.get("compacted_transcript_blake3"),
        )
        if not all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in hashes
        ):
            return None
        generation = receipt.get("generation")
        original_tokens = receipt.get("original_approx_tokens")
        compacted_tokens = receipt.get("compacted_approx_tokens")
        savings = receipt.get("token_savings_estimate")
        numeric = (generation, original_tokens, compacted_tokens, savings)
        if not all(
            isinstance(value, int) and not isinstance(value, bool) for value in numeric
        ):
            return None
        if (
            generation < 1
            or original_tokens <= 0
            or compacted_tokens < 0
            or savings < 0
            or compacted_tokens > original_tokens
            or savings != original_tokens - compacted_tokens
        ):
            return None
        return receipt

    @staticmethod
    def _deterministic_summary_checkpoint_ready(
        response: dict[str, Any], *, target_tokens: int | None = None
    ) -> bool:
        """Return true only with explicit fixed-point/diminishing-return evidence.

        The Rust receipt is the authority for both transcript identity and token
        savings. Missing or malformed evidence fails closed to the deterministic
        extractive summary.
        """
        receipt = ContextGovernorEngine._validated_checkpoint_receipt(response)
        if receipt is None:
            return False
        original_hash = receipt["original_transcript_blake3"]
        compacted_hash = receipt["compacted_transcript_blake3"]
        original_tokens = receipt.get("original_approx_tokens")
        compacted_tokens = receipt.get("compacted_approx_tokens")
        savings = receipt.get("token_savings_estimate")
        # ``ineffective_only`` describes the deterministic pass's marginal
        # utility, not whether hard-cascade happened to fit the target. Requiring
        # ``compacted_tokens > target`` made the strategy unreachable: a real
        # hard-cascade response either fits, or removes the summary carrier when
        # the protected structural floor itself exceeds the target.
        if original_hash == compacted_hash:
            return compacted_tokens == original_tokens and savings == 0
        diminishing_return_tokens = max(1, int(original_tokens * 0.10))
        return savings <= diminishing_return_tokens

    def _finalize_response(
        self,
        response: dict[str, Any],
        compacted: List[Dict[str, Any]],
    ) -> dict[str, Any]:
        """Authenticate the candidate and bind its host-normalized projection."""
        payload = {
            "candidate": copy.deepcopy(response),
            "compacted_messages": [
                self._message_to_governor(message, i)
                for i, message in enumerate(compacted)
                if isinstance(message, dict)
            ],
        }
        finalized = self._run_json(
            ["finalize-v2", *self._certified_store_args()],
            payload,
        )
        if not isinstance(finalized, dict):
            raise ValueError("finalize returned a non-object response")
        return finalized

    def _enhance_with_llm_summary(
        self,
        compacted: List[Dict[str, Any]],
        original_messages: List[Dict[str, Any]],
        response: dict[str, Any],
        focus_topic: str | None,
    ) -> List[Dict[str, Any]]:
        """Enhance the extractive summary with an LLM-generated one.

        If the LLM call fails for any reason, the original extractive summary
        is preserved — this path is strictly additive, never worse.
        """
        # Find only the deterministic summary carrier explicitly tagged while
        # translating this exact compact-v2 response. User/system content that
        # merely resembles a summary must never be overwritten.
        summary_idx = self._summary_message_index(compacted)
        if summary_idx is None:
            if self.last_compaction_metrics is not None:
                self.last_compaction_metrics["llm_call_reason"] = (
                    "summary_projection_unavailable"
                )
            self._record_summary_fallback(
                "deterministic summary carrier was unavailable; retaining the "
                "receipt-backed deterministic projection"
            )
            return compacted

        # Serialize the summarized turns for the LLM
        receipt = response.get("receipt") or {}
        plan = response.get("allocation_plan") or {}
        summarized_ids = set(plan.get("summarized_item_ids") or [])
        items = plan.get("items") or []

        # Build content to summarize from the items that were summarized/omitted
        turns_to_summarize = []
        for item in items:
            if item.get("item_id") in summarized_ids:
                idx = item.get("start_index", 0)
                if idx < len(original_messages):
                    msg = original_messages[idx]
                    if isinstance(msg, dict):
                        turns_to_summarize.append(msg)

        if not turns_to_summarize:
            # A deterministic fixed point commonly has no *new* summarized
            # item IDs: the governed projection already contains the prior
            # extractive summary. That is precisely when the secondary LLM is
            # useful. Audit against the full current projection while the Rust
            # renderer supplies the bounded, receipt-aware summary prompt.
            turns_to_summarize = [
                msg for msg in original_messages if isinstance(msg, dict)
            ]

        try:
            configured_summary_chars = max(
                256, int(self._policy.get("summary_max_chars") or 8000)
            )
        except (TypeError, ValueError):
            configured_summary_chars = 8000
        configured_summary_tokens = max(256, configured_summary_chars // 4)
        summary_budget = min(
            int(self.context_length * 0.05) if self.context_length else 4000,
            8000,
            configured_summary_tokens,
        )

        # The Rust renderer is the canonical specialized prompt: it carries
        # exact-fallback refs, plan/step lineage, loss reports, and the strict
        # output contract developed for repeated compaction cycles.  Do not
        # substitute the legacy adapter-local generic template here.
        try:
            rendered = self._run_json(["render-prompt-v2"], response)
            system_prompt = str(rendered.get("system") or "")
            prompt = str(rendered.get("user") or "")
            if not system_prompt or not prompt:
                raise ValueError(
                    "render-prompt returned an empty system or user prompt"
                )
        except Exception as exc:
            warning = f"specialized summary prompt unavailable; using extractive summary: {exc}"
            self._record_summary_fallback(warning)
            logger.warning("context-governor: %s", warning)
            return compacted

        # Codex Responses intentionally does not expose a provider output-token
        # parameter on every route. Keep the receipt renderer authoritative, but
        # add a model-visible and host-enforced bound so a reasoning model cannot
        # spend minutes producing a summary that Rust must reject anyway.
        system_prompt = (
            f"{system_prompt.rstrip()}\n\n"
            "HARD OUTPUT LIMIT: Return at most "
            f"{configured_summary_chars} Unicode characters total, including all "
            "required section markers. Be concise; never reproduce the source "
            "transcript or add extra sections."
        )

        if focus_topic:
            prompt += f"\n\n=== FOCUS OVERRIDE ===\n{focus_topic}"

        llm_started = time.monotonic()
        try:
            metrics = self.last_compaction_metrics
            if metrics is not None:
                metrics["llm_call"] = True
            llm_result = self._call_summary_llm(
                prompt, summary_budget, system_prompt=system_prompt
            )
            if llm_result is None:
                warning = "secondary summary model returned no usable text; using extractive summary"
                self._record_summary_fallback(warning)
                logger.warning("context-governor: %s", warning)
                return compacted

            # Normalize and validate the first draft before deciding whether it
            # deserves the one allowed shrink retry.  This keeps malformed or
            # carrier-spoofing drafts away from a second model call and ensures
            # the retry sees only the normalized structured draft, never a
            # model preamble or the source transcript.
            llm_summary = self._normalize_llm_summary_output(llm_result.content)
            if not llm_summary:
                warning = "LLM summary violated the structured output contract; using extractive summary"
                self._record_summary_fallback(warning)
                logger.warning("context-governor: %s", warning)
                return compacted
            reserved_syntax = self._reserved_carrier_syntax_outside_fallback(
                llm_summary
            )
            if reserved_syntax:
                warning = (
                    "LLM summary placed host-reserved receipt syntax outside "
                    "EXACT FALLBACK REFS; using extractive summary"
                )
                self._record_summary_fallback(warning)
                logger.warning(
                    "context-governor: %s (syntax=%s)", warning, reserved_syntax
                )
                return compacted

            if len(llm_summary) > configured_summary_chars:
                if metrics is not None:
                    metrics["llm_retry"] = True
                    metrics["llm_retry_reason"] = "hard_character_limit"
                retry_result = self._retry_oversized_llm_summary(
                    llm_summary,
                    configured_summary_chars,
                    summary_budget,
                    pinned_route=llm_result.route,
                )
                llm_summary = (
                    self._normalize_llm_summary_output(retry_result.content)
                    if retry_result is not None
                    else None
                )
                if llm_summary:
                    reserved_syntax = self._reserved_carrier_syntax_outside_fallback(
                        llm_summary
                    )
                    if reserved_syntax:
                        warning = (
                            "LLM length retry placed host-reserved receipt syntax "
                            "outside EXACT FALLBACK REFS; using extractive summary"
                        )
                        self._record_summary_fallback(warning)
                        logger.warning(
                            "context-governor: %s (syntax=%s)",
                            warning,
                            reserved_syntax,
                        )
                        return compacted
            if metrics is not None:
                metrics["llm_latency_ms"] = round(
                    (time.monotonic() - llm_started) * 1000, 3
                )
            if not llm_summary:
                warning = (
                    "secondary summary model returned malformed or empty text "
                    "after its bounded length retry; using extractive summary"
                )
                self._record_summary_fallback(warning)
                logger.warning("context-governor: %s", warning)
                return compacted
            if len(llm_summary) > configured_summary_chars:
                warning = (
                    "secondary summary exceeded its hard character limit after "
                    "one bounded retry "
                    f"({len(llm_summary)} > {configured_summary_chars}); using "
                    "extractive summary"
                )
                self._record_summary_fallback(warning)
                logger.warning("context-governor: %s", warning)
                return compacted

            # Boundary evidence always comes from the original transcript
            # messages selected by the receipt, even after a shrink retry. The
            # generated first draft is never substituted as audit provenance.
            audit = self._audit_compression_boundary(
                copy.deepcopy(turns_to_summarize), llm_summary
            )
            self.last_summary_safety = audit
            safe = (
                bool(audit.get("safe_to_reinject", True))
                if isinstance(audit, dict)
                else True
            )
            if not safe:
                policy = str(
                    self._policy.get("unsafe_summary_policy") or "fallback_extract"
                )
                warning = (
                    "LLM summary failed compression-boundary safety audit; "
                    f"policy={policy}"
                )
                logger.warning("context-governor: %s", warning)
                if policy != "warn":
                    self._record_summary_fallback(
                        f"{warning}; using extractive summary"
                    )
                    if policy == "freeze":
                        self.last_error = warning
                    return compacted
                # ``warn`` deliberately permits the audited text, but it still
                # passes through the immutable carrier binder below.
                self._record_summary_warning("boundary_audit_unsafe_warn", warning)
            else:
                self.last_warning = None

            llm_summary = self._bind_receipt_carrier(llm_summary, response)
            candidate = list(compacted)
            metadata = candidate[summary_idx].get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            receipt_id = receipt.get("receipt_id")
            generation = receipt.get("generation")
            candidate[summary_idx] = {
                **candidate[summary_idx],
                "content": llm_summary,
                "metadata": {
                    **metadata,
                    "compressed_summary": True,
                    "llm_checkpoint": True,
                    "receipt_id": receipt_id,
                    "receipt_generation": generation,
                },
            }
            # The caller finalizes this candidate with Rust and accepts it only
            # if the governor's own token counter meets the target.
            compacted = candidate
            logger.debug("context-governor: LLM summary enhancement applied")
        except Exception as exc:
            if self.last_compaction_metrics is not None:
                self.last_compaction_metrics["llm_latency_ms"] = round(
                    (time.monotonic() - llm_started) * 1000, 3
                )
            logger.warning(
                "context-governor: LLM summary enhancement failed, using extractive: %s",
                exc,
            )
            self._record_summary_fallback(
                f"LLM summary enhancement failed; using extractive summary: {exc}"
            )

        return compacted

    def _retry_oversized_llm_summary(
        self,
        summary: str,
        max_chars: int,
        original_token_budget: int,
        *,
        pinned_route: _SummaryLLMRoute,
    ) -> _SummaryLLMResult | None:
        """Give an otherwise structured summary one bounded shrink attempt.

        The retry receives only generated text, never the source transcript.
        Structure auditing, boundary scanning, immutable carrier binding, Rust
        finalization, and the governor token-budget check still run afterward.
        """
        markers = (
            "=== ACTIVE TASK ===",
            "=== ACCEPTANCE GATES ===",
            "=== EXACT FALLBACK REFS ===",
            "=== SUMMARY LOSSES ===",
            "=== PRIOR CONTEXT SUMMARY ===",
        )
        marker_contract = "\n".join(markers)
        system_prompt = (
            "You are a bounded context-summary editor. Condense the supplied "
            "summary without adding facts. Return only these five sections, "
            "exactly once and in this order:\n"
            f"{marker_contract}\n\n"
            f"HARD OUTPUT LIMIT: at most {max_chars} Unicode characters total, "
            "including the section markers. Preserve task state, acceptance "
            "gates, decisions, unresolved risks, and identifiers. Keep hashes "
            "verbatim when present. Do not reproduce dialogue or explain your "
            "editing."
        )
        prompt = (
            f"Condense the following structured summary to at most {max_chars} "
            "Unicode characters while obeying the system contract.\n\n"
            "=== SUMMARY TO CONDENSE ===\n"
            f"{summary}"
        )
        retry_budget = max(256, min(original_token_budget, max_chars // 5))
        return self._call_summary_llm(
            prompt,
            retry_budget,
            system_prompt=system_prompt,
            pinned_route=pinned_route,
        )

    @staticmethod
    def _summary_message_index(messages: List[Dict[str, Any]]) -> int | None:
        """Locate only the summary tagged from this compact-v2 response."""
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            marker = message.get("_context_governor_summary_id")
            if isinstance(marker, str) and marker.startswith("summary_ctxp_"):
                return index
        return None

    def _record_summary_fallback(self, error: str) -> None:
        """Expose a successful deterministic fallback through the host contract."""
        message = self._safe_summary_diagnostic(
            error or "secondary summary unavailable"
        )
        self.last_warning = message
        self._last_summary_error = message
        self._last_summary_fallback_used = True
        if self.last_compaction_metrics is not None:
            self.last_compaction_metrics["summary_fallback_reason"] = message
            self.last_compaction_metrics["summary_fallback"] = {
                "code": "secondary_summary_fallback",
                "message": message,
            }

    def _record_summary_warning(self, code: str, warning: str) -> None:
        """Expose a non-fallback summary warning without leaking credentials."""
        message = self._safe_summary_diagnostic(warning or "summary warning")
        self.last_warning = message
        if self.last_compaction_metrics is not None:
            self.last_compaction_metrics["summary_warning"] = {
                "code": str(code or "summary_warning"),
                "message": message,
            }

    @staticmethod
    def _safe_summary_diagnostic(value: Any) -> str:
        """Redact exception/warning text before storing it in host-visible state."""
        return redact_sensitive_text(str(value or "").strip(), force=True)

    @staticmethod
    def _reserved_carrier_syntax_outside_fallback(summary: str) -> str | None:
        """Return the first host-reserved syntax found outside its owned section.

        The model may echo or propose fallback references inside
        ``EXACT FALLBACK REFS`` because that entire body is replaced by the
        authenticated host carrier. Everywhere else, receipt-looking syntax is
        rejected so a generated section cannot impersonate provenance or
        inflate the durable checkpoint counter.
        """
        exact_marker = "=== EXACT FALLBACK REFS ==="
        losses_marker = "=== SUMMARY LOSSES ==="
        try:
            exact_start = summary.index(exact_marker) + len(exact_marker)
            exact_end = summary.index(losses_marker, exact_start)
        except ValueError:
            return "section_contract"
        outside = summary[:exact_start] + "\n" + summary[exact_end:]
        patterns = (
            (
                "llm_checkpoint_marker",
                r"(?i)\bllm_checkpoint_(?:receipt|session_sha256)\s*=",
            ),
            ("receipt_id", r"(?i)(?<![\w])receipt_id\s*="),
            ("generation", r"(?i)(?<![\w])generation\s*="),
            (
                "lineage_hash",
                r"(?i)(?<![\w])lineage_(?:blake3|sha256)\s*=",
            ),
            (
                "original_transcript_hash",
                r"(?i)(?<![\w])original_transcript_(?:blake3|sha256)\s*=",
            ),
            (
                "ctxs_hash_line",
                r"(?i)\bctxs_[a-z0-9_-]+\s*\|\s*blake3\s*:[^\n|]*"
                r"\|\s*sha256\s*:",
            ),
        )
        for name, pattern in patterns:
            if re.search(pattern, outside):
                return name
        return None

    @classmethod
    def _summary_message_content(cls, messages: List[Dict[str, Any]]) -> str | None:
        index = cls._summary_message_index(messages)
        if index is None:
            return None
        content = messages[index].get("content")
        return content if isinstance(content, str) else None

    @staticmethod
    def _bind_receipt_carrier(
        summary: str,
        response: dict[str, Any],
    ) -> str:
        """Replace model-written fallback refs with an immutable host carrier.

        Receipt identity and the final compacted transcript hash are excluded:
        both bind the projection containing this text and would be
        self-referential. The lineage/original hashes and exact source hashes
        are stable across ``finalize-v2`` and remain independently verified by
        ``store-v2``.
        """
        receipt = response.get("receipt") or {}
        if not isinstance(receipt, dict):
            raise ValueError("LLM carrier response omitted its receipt")
        receipt_id = str(receipt.get("receipt_id") or "")
        generation = receipt.get("generation")
        session_id = str(receipt.get("session_id") or "")
        if not re.fullmatch(r"ctxr_[0-9a-f]{32}", receipt_id):
            raise ValueError("LLM carrier receipt_id is invalid")
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 1
        ):
            raise ValueError("LLM carrier generation is invalid")
        if not session_id:
            raise ValueError("LLM carrier session_id is missing")
        stable_hash_fields = (
            "lineage_blake3",
            "lineage_sha256",
            "original_transcript_blake3",
            "original_transcript_sha256",
        )
        for field in stable_hash_fields:
            value = receipt.get(field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"LLM carrier {field} is invalid")
        session_sha256 = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        sources = receipt.get("covered_original_sources") or []
        if not isinstance(sources, list):
            raise ValueError("LLM carrier covered_original_sources is invalid")
        valid_sources = []
        seen_source_ids: set[str] = set()
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("LLM carrier covered source is invalid")
            source_id = str(source.get("source_id") or "")
            blake3_hash = str(source.get("content_blake3") or "")
            sha256_hash = str(source.get("content_sha256") or "")
            if not re.fullmatch(r"ctxs_[0-9a-f]{64}", source_id):
                raise ValueError("LLM carrier source_id is invalid")
            if not re.fullmatch(r"[0-9a-f]{64}", blake3_hash):
                raise ValueError("LLM carrier source BLAKE3 is invalid")
            if not re.fullmatch(r"[0-9a-f]{64}", sha256_hash):
                raise ValueError("LLM carrier source SHA-256 is invalid")
            if source_id in seen_source_ids:
                raise ValueError("LLM carrier contained a duplicate source_id")
            seen_source_ids.add(source_id)
            valid_sources.append(source)
        carrier = [
            f"llm_checkpoint_receipt={receipt_id}",
            f"llm_checkpoint_session_sha256={session_sha256}",
            f"receipt_id={receipt_id}",
            f"generation={generation}",
            f"lineage_blake3={receipt['lineage_blake3']}",
            f"lineage_sha256={receipt['lineage_sha256']}",
            f"original_transcript_blake3={receipt['original_transcript_blake3']}",
            f"original_transcript_sha256={receipt['original_transcript_sha256']}",
            f"covered_sources={len(valid_sources)} (full manifest in verified receipt store)",
            "recover=context_search(query=..., scope=exact), then "
            f"context_expand(receipt_id={receipt_id}, item_id=<ctxs_id>)",
        ]
        for source in valid_sources[:4]:
            source_id = str(source.get("source_id") or "")
            blake3_hash = str(source.get("content_blake3") or "")
            sha256_hash = str(source.get("content_sha256") or "")
            carrier.append(f"{source_id} | blake3:{blake3_hash} | sha256:{sha256_hash}")
        carrier_text = "\n".join(carrier)

        markers = (
            "=== ACTIVE TASK ===",
            "=== ACCEPTANCE GATES ===",
            "=== EXACT FALLBACK REFS ===",
            "=== SUMMARY LOSSES ===",
            "=== PRIOR CONTEXT SUMMARY ===",
        )
        if any(summary.count(marker) != 1 for marker in markers):
            raise ValueError(
                "LLM summary contained missing or duplicate section markers"
            )
        positions = [summary.index(marker) for marker in markers]
        if positions != sorted(positions):
            raise ValueError("LLM summary section markers were out of order")
        reserved_syntax = (
            ContextGovernorEngine._reserved_carrier_syntax_outside_fallback(summary)
        )
        if reserved_syntax:
            raise ValueError(
                "LLM summary contained host-reserved receipt syntax outside "
                f"EXACT FALLBACK REFS ({reserved_syntax})"
            )
        start_marker = markers[2]
        end_marker = markers[3]
        start = summary.index(start_marker)
        end = summary.index(end_marker)
        content_start = start + len(start_marker)
        bound = summary[:content_start] + "\n" + carrier_text + "\n\n" + summary[end:]
        unique_lines = (
            f"llm_checkpoint_receipt={receipt_id}",
            f"llm_checkpoint_session_sha256={session_sha256}",
            f"receipt_id={receipt_id}",
            f"generation={generation}",
            f"lineage_blake3={receipt['lineage_blake3']}",
            f"lineage_sha256={receipt['lineage_sha256']}",
            f"original_transcript_blake3={receipt['original_transcript_blake3']}",
            f"original_transcript_sha256={receipt['original_transcript_sha256']}",
        )
        if any(bound.splitlines().count(line) != 1 for line in unique_lines):
            raise ValueError("LLM host receipt carrier was not unique after binding")
        return bound

    def _audit_compression_boundary(
        self,
        source_messages: List[Dict[str, Any]],
        compressed_summary: str,
    ) -> dict[str, Any]:
        """Run context-governor's compression-boundary safety audit.

        LLM summaries are generated text, so they are treated as untrusted until
        the Rust boundary scanner says they are safe to reinject. If the scanner
        itself fails, fail open to the extractive summary by returning unsafe.
        """
        source_fragments = [
            self._content_to_text(msg.get("content"))
            for msg in source_messages
            if isinstance(msg, dict)
            and self._content_to_text(msg.get("content")).strip()
        ]
        payload = {
            "source_fragments": source_fragments,
            "compressed_summary": compressed_summary,
        }
        try:
            return self._run_json(["boundary-audit"], payload)
        except Exception as exc:
            safe_error = self._safe_summary_diagnostic(exc)
            warning = (
                "compression-boundary audit failed; using extractive summary: "
                f"{safe_error}"
            )
            self._record_summary_warning("boundary_audit_unavailable", warning)
            logger.warning("context-governor: %s", warning)
            return {
                "schema": "CompressionBoundaryAuditV1",
                "safe_to_reinject": False,
                "relinking_risk": "unknown",
                "adapter_error": safe_error,
            }

    def _serialize_for_summary(self, turns: List[Dict[str, Any]]) -> str:
        """Serialize conversation turns into labeled text for the summarizer."""
        parts = []
        for msg in turns:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        text_parts.append(part["text"])
                    elif isinstance(part, dict) and part.get("type") in {
                        "image",
                        "image_url",
                    }:
                        text_parts.append("[image]")
                content = "\n".join(text_parts)
            if not isinstance(content, str):
                content = str(content)
            # Truncate long content
            if len(content) > 6000:
                content = content[:4000] + "\n...[truncated]...\n" + content[-1500:]

            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
            elif role == "assistant":
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    tc_parts = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = fn.get("arguments", "")[:1200]
                            tc_parts.append(f"  {name}({args})")
                    if tc_parts:
                        content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
            else:
                parts.append(f"[{role.upper()}]: {content}")
        return "\n\n".join(parts)

    @staticmethod
    def _normalize_llm_summary_output(content: str) -> str | None:
        """Accept only the specialized prompt's structured output.

        Reasoning-capable models sometimes prepend analysis despite the output
        contract. Strip a leading preamble only when the structured response
        that follows is complete; otherwise retain the extractive summary
        rather than feeding an unstable free-form summary into the next
        compaction cycle.

        The render-prompt system prompt declares that anything before
        ``=== ACTIVE TASK ===`` causes rejection. We honor this by stripping
        preamble (rather than discarding the whole output) only when the
        complete structured response is present — ``=== ACTIVE TASK ===``
        through ``=== PRIOR CONTEXT SUMMARY ===``. A free-form response with
        no structural markers is rejected outright.
        """
        markers = (
            "=== ACTIVE TASK ===",
            "=== ACCEPTANCE GATES ===",
            "=== EXACT FALLBACK REFS ===",
            "=== SUMMARY LOSSES ===",
            "=== PRIOR CONTEXT SUMMARY ===",
        )
        text = str(content or "").strip()
        if any(text.count(marker) != 1 for marker in markers):
            return None
        positions = [text.index(marker) for marker in markers]
        if positions != sorted(positions):
            return None
        start = positions[0]
        # Strip preamble (text before the first structural marker).
        if start > 0:
            text = text[start:]
        return text

    def _resolve_moa_runtime(self) -> tuple[str, str, str, str]:
        """If an explicit summary provider is MoA, resolve its real aggregator.

        With no explicit governor summary override, return empty values so
        ``auxiliary.compression`` remains the canonical secondary-model route.
        If that route itself is ``auto``, ``call_llm`` receives the live main
        runtime separately as its ordinary fallback context.

        Returns ``(model, provider, base_url, api_key)`` for an explicit
        override and empty strings when task routing should decide.

        ``_resolve_auto`` in ``auxiliary_client`` already does this for the
        auto-detection path, but ``_call_summary_llm`` passes explicit args
        which bypass auto-detection.  We resolve here so the explicit args
        carry a real HTTP provider instead of the virtual MoA facade.
        """
        model = self._summary_model
        provider = self._summary_provider
        base_url = self._summary_base_url
        api_key = self._summary_api_key

        if not any((model, provider, base_url, api_key)):
            return "", "", "", ""

        if provider != "moa":
            return model, provider, base_url, api_key

        # Resolve MoA preset → aggregator's real provider+model.
        # Drop the virtual base_url/api_key — the aggregator resolves
        # through its own provider credentials.
        try:
            from hermes_cli.config import load_config
            from hermes_cli.moa_config import resolve_moa_preset

            cfg = load_config()
            preset = resolve_moa_preset(cfg.get("moa") or {}, model)
            agg = preset.get("aggregator") or {}
            agg_provider = str(agg.get("provider") or "").strip()
            agg_model = str(agg.get("model") or "").strip()
            if agg_provider and agg_model and agg_provider.lower() != "moa":
                logger.info(
                    "context-governor: resolving MoA preset '%s' → "
                    "aggregator %s/%s for LLM summary",
                    model,
                    agg_provider,
                    agg_model,
                )
                return agg_model, agg_provider, "", ""
        except Exception:
            logger.debug(
                "context-governor: MoA resolution failed, falling through",
                exc_info=True,
            )

        # If resolution failed, return empty strings so call_llm uses
        # auto-detection rather than the virtual MoA values.
        return "", "", "", ""

    def _call_summary_llm(
        self,
        prompt: str,
        max_tokens: int,
        *,
        system_prompt: str = "",
        pinned_route: _SummaryLLMRoute | None = None,
    ) -> _SummaryLLMResult | None:
        """Call the summarizer and return text with its actual route identity.

        A shrink retry supplies ``pinned_route`` from the first call. Explicit
        provider/model arguments then outrank any concurrently changed
        ``auxiliary.compression`` config, and the post-call route check rejects
        provider fallbacks rather than silently changing summarizers mid-pair.
        """
        try:
            # Hermes centralizes auxiliary requests here.  The old
            # ``model_router`` imports never existed in the live package, which
            # made ``summary_mode: llm`` silently fall back to extractive mode.
            from agent.auxiliary_client import (
                _resolve_task_provider_model,
                call_llm,
            )
        except ImportError:
            logger.debug(
                "context-governor: auxiliary call_llm not available, skipping LLM summary"
            )
            return None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        call_kwargs = {
            "task": "compression",
            "messages": messages,
            "max_tokens": int(max_tokens * 1.3),
            "main_runtime": {
                "model": self.model,
                "provider": self.provider,
                "base_url": self.base_url,
                "api_key": self.api_key,
                "api_mode": self.api_mode,
            },
        }

        # Resolve MoA virtual provider to the aggregator's real provider.
        # Without this, passing provider="moa" / model="<preset>" as explicit
        # args bypasses _resolve_auto's MoA handling in auxiliary_client,
        # and call_llm fails to create a client for the virtual MoA endpoint.
        model, provider, base_url, api_key = self._resolve_moa_runtime()
        if pinned_route is not None:
            provider = pinned_route.provider
            model = pinned_route.model
        if model:
            call_kwargs["model"] = model
        if provider:
            call_kwargs["provider"] = provider
        if base_url:
            call_kwargs["base_url"] = base_url
        if api_key:
            call_kwargs["api_key"] = api_key

        expected_provider, expected_model, _, _, _ = _resolve_task_provider_model(
            "compression",
            provider or None,
            model or None,
            base_url or None,
            api_key or None,
        )

        route_info: dict[str, str] = {}
        call_kwargs["route_info"] = route_info

        response = call_llm(**call_kwargs)
        metrics = self.last_compaction_metrics
        if metrics is not None:
            metrics["summarizer_model"] = route_info.get("model") or model or None
            metrics["summarizer_provider"] = (
                route_info.get("provider") or provider or None
            )
        actual_provider = str(route_info.get("provider") or "").strip().lower()
        actual_model = str(route_info.get("model") or "").strip()
        expected_provider = str(expected_provider or "").strip().lower()
        expected_model = str(expected_model or "").strip()
        if (
            expected_provider not in {"", "auto"}
            and actual_provider != expected_provider
        ):
            raise RuntimeError(
                "configured compression summarizer route changed during fallback "
                f"({expected_provider}/{expected_model or 'default'} -> "
                f"{actual_provider or 'unknown'}/{actual_model or 'unknown'})"
            )
        if expected_model and actual_model != expected_model:
            raise RuntimeError(
                "configured compression summarizer model changed during fallback "
                f"({expected_model} -> {actual_model or 'unknown'})"
            )
        if actual_provider in {"", "auto", "unknown"} or actual_model in {
            "",
            "auto",
            "default",
            "unknown",
        }:
            raise RuntimeError(
                "compression summarizer did not expose an exact actual "
                "provider/model route"
            )
        actual_route = _SummaryLLMRoute(actual_provider, actual_model)
        if pinned_route is not None and actual_route != pinned_route:
            raise RuntimeError(
                "compression summary retry route changed "
                f"({pinned_route.provider}/{pinned_route.model} -> "
                f"{actual_route.provider}/{actual_route.model})"
            )
        if metrics is not None and pinned_route is not None:
            metrics["summarizer_retry_provider"] = actual_route.provider
            metrics["summarizer_retry_model"] = actual_route.model
        content = response.choices[0].message.content
        if not isinstance(content, str):
            content = str(content) if content else ""
        if not content.strip():
            return None
        return _SummaryLLMResult(content.strip(), actual_route)

    # ------------------------------------------------------------------
    # Failure classification
    # ------------------------------------------------------------------

    def _classify_subprocess_error(self, exc: Exception) -> str:
        """Classify subprocess errors for appropriate fallback behavior."""
        msg = str(exc).lower()
        if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg:
            return "auth"
        if (
            "connection" in msg
            or "timeout" in msg
            or "reset" in msg
            or "broken pipe" in msg
        ):
            return "network"
        if isinstance(exc, subprocess.TimeoutExpired):
            return "timeout"
        return "transient"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request_min_net_savings_tokens(self) -> int | None:
        """Keep the deterministic no-benefit gate off for hybrid checkpoints.

        A low-savings deterministic pass is exactly the input an LLM
        checkpoint may improve. Rejecting it in Rust before the host can
        evaluate ``after_n``/``ineffective_only`` makes the configured hybrid
        path unreachable. The adapter still finalizes against the target,
        persists an exact receipt, and uses anti-thrashing after the attempt.
        """
        if self._summary_mode == "llm" and self._checkpoint_strategy_json() != "off":
            return 0
        raw = self._policy.get("min_net_savings_tokens")
        if raw is None:
            return None
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return None

    def _target_tokens(self, current_tokens: int | None) -> int:
        explicit = self._policy.get("token_budget")
        try:
            if explicit is not None and int(explicit) > 0:
                return max(512, int(explicit))
        except (TypeError, ValueError):
            pass
        if self.context_length:
            return max(512, int(self.context_length * 0.20))
        if current_tokens:
            return max(512, int(current_tokens * 0.20))
        return 8000

    def _run_json(self, args: list[str], payload: dict[str, Any]) -> dict[str, Any]:
        binding = self._key_binding
        try:
            proc = subprocess.run(
                [str(self.binary), *args],
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_sec,
                pass_fds=binding.pass_fds if binding else (),
                check=False,
            )
        finally:
            if binding is not None:
                binding.close()
                self._key_binding = None
        if proc.returncode != 0:
            raise RuntimeError(
                (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
            )
        return json.loads(proc.stdout)

    def _prepare_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Durably stage a verified receipt without publishing a lineage tip."""
        self.store_dir.mkdir(parents=True, exist_ok=True)
        receipt = response.get("receipt") or {}
        if receipt.get("schema") != "ContextCompactionReceiptV2":
            raise ValueError("refusing to prepare a non-V2 Context Governor receipt")
        receipt_id = str(receipt.get("receipt_id") or "")
        result = self._run_json(
            [
                "prepare-v2",
                "--dir",
                str(self.store_dir),
                *self._certified_store_args(),
            ],
            response,
        )
        try:
            if not isinstance(result, dict):
                raise ValueError("prepare-v2 returned a non-object result")
            expected_messages = response.get("compacted_messages")
            checks = (
                result.get("schema") == "PendingReceiptInfoV2",
                result.get("verified") is True,
                result.get("receipt_id") == receipt_id,
                result.get("session_id") == receipt.get("session_id"),
                result.get("generation") == receipt.get("generation"),
                result.get("expected_compacted_message_count")
                == len(expected_messages or []),
                result.get("expected_compacted_transcript_blake3")
                == receipt.get("compacted_transcript_blake3"),
                result.get("expected_compacted_transcript_sha256")
                == receipt.get("compacted_transcript_sha256"),
                result.get("expected_compacted_messages") == expected_messages,
            )
            if not all(checks):
                raise ValueError("prepare-v2 result did not match the finalized receipt")
        except Exception:
            # Preparation may already have atomically published the pending
            # file. Remove that non-authoritative record before surfacing an
            # adapter validation failure.
            if receipt_id:
                try:
                    self._discard_pending_receipt(receipt_id)
                except Exception:
                    logger.warning(
                        "context-governor: failed to compensate invalid pending receipt %s",
                        receipt_id,
                        exc_info=True,
                    )
            raise
        return result

    def _discard_pending_receipt(self, receipt_id: str) -> dict[str, Any]:
        result = self._run_json(
            [
                "discard-v2",
                "--dir",
                str(self.store_dir),
                "--receipt",
                receipt_id,
                *self._certified_store_args(),
            ],
            {},
        )
        if (
            not isinstance(result, dict)
            or result.get("schema") != "ReceiptDiscardResultV2"
            or result.get("receipt_id") != receipt_id
            or result.get("discarded") is not True
        ):
            raise ValueError("discard-v2 did not confirm the requested receipt")
        return result

    def _committed_pending_projection(
        self,
        committed_messages: List[Dict[str, Any]],
        expected_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]] | None:
        """Prove the authenticated governor projection is a host commit prefix."""
        normalized = self._without_host_todo_snapshots(committed_messages)
        actual = [
            self._message_to_governor(message, index)
            for index, message in enumerate(normalized)
            if isinstance(message, dict)
        ]
        # SessionDB intentionally has no generic message-id/metadata columns.
        # Compare with the deterministic durable host projection, then return
        # the original authenticated objects for Rust's exact activation check.
        durable_expected = [
            self._message_to_governor(
                self._message_from_governor(message),
                index,
            )
            for index, message in enumerate(expected_messages)
            if isinstance(message, dict)
        ]
        if (
            expected_messages
            and len(durable_expected) == len(expected_messages)
            and actual[: len(durable_expected)] == durable_expected
        ):
            return copy.deepcopy(expected_messages)
        return None

    def commit_pending_compression(
        self,
        committed_messages: List[Dict[str, Any]],
        **kwargs,
    ) -> bool:
        """Activate the prepared receipt only after the host accepts its prefix."""
        pending = self._pending_admission
        if pending is None:
            return True
        info = pending.get("pending_info") or {}
        expected = info.get("expected_compacted_messages") or []
        projection = self._committed_pending_projection(
            committed_messages,
            expected,
        )
        if projection is None:
            raise ValueError(
                "host-committed transcript does not contain the authenticated "
                "pending governor projection as an exact prefix"
            )
        receipt_id = str(pending.get("receipt_id") or "")
        result = self._run_json(
            [
                "activate-v2",
                "--dir",
                str(self.store_dir),
                *self._certified_store_args(),
            ],
            {"receipt_id": receipt_id, "committed_messages": projection},
        )
        if (
            not isinstance(result, dict)
            or result.get("schema") != "ReceiptActivationResultV2"
            or result.get("receipt_id") != receipt_id
            or result.get("activated") is not True
            or result.get("verified") is not True
        ):
            raise ValueError("activate-v2 did not verify the prepared receipt")

        # Rust has atomically activated the receipt. Clear local pending state
        # before best-effort bookkeeping so a later Python exception cannot
        # make the host try to discard an already-active receipt.
        self._pending_admission = None
        try:
            generation = info.get("generation")
            if isinstance(generation, int) and not isinstance(generation, bool):
                self.compression_count = max(self.compression_count + 1, generation)
            else:
                self.compression_count += 1
            self.last_receipt_id = receipt_id
            if pending.get("llm_checkpoint_applied"):
                self._llm_checkpoint_count += 1
                previous = pending.get("previous_summary")
                if isinstance(previous, str) and previous:
                    self._previous_summary = previous
                if self.last_compaction_metrics is not None:
                    reason = str(pending.get("checkpoint_reason") or "checkpoint")
                    self.last_compaction_metrics["llm_call_reason"] = (
                        f"{reason}:applied"
                    )
            savings_pct = pending.get("savings_pct")
            if isinstance(savings_pct, (int, float)) and not isinstance(
                savings_pct, bool
            ):
                self._last_compression_savings_pct = float(savings_pct)
                if float(savings_pct) < 10:
                    self._ineffective_compression_count += 1
                else:
                    self._ineffective_compression_count = 0
            if self.last_compaction_metrics is not None:
                self.last_compaction_metrics["exact_fallback_available"] = bool(
                    pending.get("exact_fallback_available")
                )
                self.last_compaction_metrics["integrity_result"] = (
                    "host_commit_activation_verified"
                )
            self.last_error = None
            self.set_activation_status(
                configured_engine=self.name,
                discovered=True,
                version_compatible=True,
                capability_compatible=True,
                instantiated=True,
                observed_live_engine=self.name,
                state="observed_live",
                detail=f"verified receipt {receipt_id} activated after host commit",
            )
        except Exception:
            logger.warning(
                "context-governor: receipt %s activated but local bookkeeping failed",
                receipt_id,
                exc_info=True,
            )
        return True

    def validate_pending_compression(
        self,
        committed_messages: List[Dict[str, Any]],
        **kwargs,
    ) -> bool:
        """Prove host durability will preserve the authenticated projection."""
        pending = self._pending_admission
        if pending is None:
            return True
        info = pending.get("pending_info") or {}
        expected = info.get("expected_compacted_messages") or []
        return self._committed_pending_projection(committed_messages, expected) is not None

    def discard_pending_compression(self, **kwargs) -> bool:
        """Discard an in-process receipt when the host rejects its boundary."""
        pending = self._pending_admission
        if pending is None:
            return True
        receipt_id = str(pending.get("receipt_id") or "")
        if not receipt_id:
            self._pending_admission = None
            return True
        self._discard_pending_receipt(receipt_id)
        self._pending_admission = None
        if self.last_compaction_metrics is not None:
            self.last_compaction_metrics["integrity_result"] = "pending_discarded"
        return True

    def _reconcile_pending_receipts(self, session_db: Any, session_id: str) -> None:
        """Recover a receipt prepared before a process/desktop crash."""
        try:
            records = self._run_json(
                [
                    "pending-v2",
                    "--dir",
                    str(self.store_dir),
                    *self._certified_store_args(),
                ],
                {},
            )
        except Exception as exc:
            self._record_summary_warning(
                "pending_recovery_unavailable",
                f"pending receipt recovery unavailable: {exc}",
            )
            return
        if not isinstance(records, list):
            return
        getter = getattr(type(session_db), "get_messages_as_conversation", None)
        tip_getter = getattr(type(session_db), "get_compression_tip", None)
        if not callable(getter):
            return
        candidates = [str(session_id or "")]
        if callable(tip_getter):
            try:
                tip = str(tip_getter(session_db, self._governor_session_id()) or "")
                if tip and tip not in candidates:
                    candidates.append(tip)
            except Exception:
                pass
        durable_candidates: list[List[Dict[str, Any]]] = []
        for candidate_id in candidates:
            if not candidate_id:
                continue
            try:
                durable = getter(
                    session_db,
                    candidate_id,
                    repair_alternation=False,
                )
            except TypeError:
                durable = getter(session_db, candidate_id)
            except Exception:
                continue
            if isinstance(durable, list) and durable:
                durable_candidates.append(durable)

        for info in records:
            if (
                not isinstance(info, dict)
                or info.get("schema") != "PendingReceiptInfoV2"
                or info.get("verified") is not True
                or info.get("session_id") != self._governor_session_id()
            ):
                continue
            expected = info.get("expected_compacted_messages") or []
            for durable in durable_candidates:
                if self._committed_pending_projection(durable, expected) is None:
                    continue
                receipt_id = str(info.get("receipt_id") or "")
                summary_text = "\n".join(
                    str(message.get("content") or "")
                    for message in expected
                    if isinstance(message, dict)
                )
                self._pending_admission = {
                    "receipt_id": receipt_id,
                    "pending_info": copy.deepcopy(info),
                    "llm_checkpoint_applied": (
                        f"llm_checkpoint_receipt={receipt_id}" in summary_text
                    ),
                    "previous_summary": summary_text or None,
                    "checkpoint_reason": "recovered_checkpoint",
                    "savings_pct": None,
                    "exact_fallback_available": True,
                    "physical_session_id": session_id,
                    "lineage_session_id": self._governor_session_id(),
                }
                try:
                    self.commit_pending_compression(durable)
                except Exception:
                    self._pending_admission = None
                    self._record_summary_warning(
                        "pending_recovery_activation_failed",
                        "authenticated pending receipt could not be activated; "
                        "it remains staged for a later recovery attempt",
                    )
                    logger.warning(
                        "context-governor: pending receipt recovery activation failed",
                        exc_info=True,
                    )
                    continue
                return

    # ------------------------------------------------------------------
    # Advisory synthetic telemetry integration
    # ------------------------------------------------------------------

    def _run_telemetry_json(
        self, args: list[str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Run the isolated telemetry bridge. Callers remain fail-open."""
        if not self._telemetry_available:
            return {}
        proc = subprocess.run(
            [self.telemetry_binary, *args, "--db", str(self.telemetry_db_path)],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_sec,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
            )
        return json.loads(proc.stdout)

    def _score_telemetry_relevance(
        self, messages: List[Dict[str, Any]], focus_topic: str | None
    ) -> Dict[int, float]:
        """Return advisory telemetry-assisted rankings, never causal scores.

        Fail-open: unavailable/broken telemetry leaves compaction unchanged.
        """
        if not self._telemetry_available:
            return {}

        msg_infos: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "unknown")
            tool_name = None
            # Only names are sent for ranking; raw args and content stay local.
            if role == "assistant" and isinstance(msg.get("tool_calls"), list):
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                        tool_name = tc["function"].get("name")
                        break

            # Extract tool name from tool messages (role=tool)
            if role == "tool" and isinstance(msg.get("name"), str):
                tool_name = msg["name"]

            msg_infos.append({
                "index": i,
                "tool_name": tool_name,
            })

        if not msg_infos:
            return {}

        payload = {
            "messages": msg_infos,
            "focus": focus_topic or "",
        }

        try:
            results: Any = self._run_telemetry_json(["score-relevance"], payload)
            if isinstance(results, list):
                scores: Dict[int, float] = {}
                for r in results:
                    if isinstance(r, dict) and "index" in r and "relevance_score" in r:
                        scores[int(r["index"])] = float(r["relevance_score"])
                return scores
        except Exception as exc:
            logger.debug("telemetry score-relevance failed (fail-open): %s", exc)

        return {}

    def _advisory_protect_last_n(self, total: int, scores: Dict[int, float]) -> int:
        """Bound ranking-based protection; telemetry never changes safety gates."""
        limit = max(
            0, int(self._policy.get("telemetry_max_additional_protected_messages", 8))
        )
        protected = self.protect_last_n
        if not scores or limit == 0:
            return protected
        cutoff = max(0, total - protected)
        selected = [
            idx for idx, score in scores.items() if idx < cutoff and score > 0.7
        ]
        if selected:
            protected = max(protected, total - min(selected))
        return min(total, min(protected, self.protect_last_n + limit))

    def _record_tool_telemetry(self, messages: List[Dict[str, Any]]) -> None:
        """Capture hashed tool telemetry from original messages before degradation."""
        if not self._telemetry_available:
            return

        # Build a map of tool_call_id → tool_name + args
        tool_calls: dict[str, dict[str, Any]] = {}
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant" and isinstance(
                msg.get("tool_calls"), list
            ):
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                        call_id = tc.get("id", "")
                        tool_calls[call_id] = {
                            "name": tc["function"].get("name", ""),
                            "args": tc["function"].get("arguments", ""),
                        }

        # For each result, record one idempotent telemetry event.
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "tool":
                continue
            call_id = msg.get("tool_call_id", "")
            if call_id not in tool_calls:
                continue

            tc_info = tool_calls[call_id]
            tool_name = tc_info["name"]
            tool_args = tc_info["args"]

            content = self._content_to_text(msg.get("content", ""))
            outcome, error_class = self._classify_tool_outcome(content)
            telemetry_payload = {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_call_id": call_id,
                "result_digest": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "outcome": outcome,
                "error_class": error_class,
                "session_id": self.session_id or "hermes",
            }

            try:
                self._run_telemetry_json(["record-telemetry"], telemetry_payload)
            except Exception as exc:
                logger.debug("telemetry record failed (fail-open): %s", exc)

    def _classify_tool_outcome(self, content: str) -> tuple[str, str | None]:
        """Classify only explicit signals; ambiguity is deliberately unknown."""
        parsed: Any = None
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            pass
        if isinstance(parsed, dict):
            exit_code = parsed.get("exit_code")
            if parsed.get("success") is True or exit_code == 0:
                return "success", None
            if parsed.get("success") is False or (
                isinstance(exit_code, int) and exit_code != 0
            ):
                return "error", self._extract_error_class(content)
            if parsed.get("error") or parsed.get("exception"):
                return "error", self._extract_error_class(content)
        lowered = content.lower()
        if "traceback" in lowered or re.search(
            r"\b(exit[_ ]?code|status)\s*[:=]\s*[1-9]\d*", lowered
        ):
            return "error", self._extract_error_class(content)
        return "unknown", None

    @staticmethod
    def _extract_error_class(content: str) -> str | None:
        """Extract a rough error class from tool output content."""
        content_lower = content.lower()
        if "compile_error" in content_lower or "error[e" in content_lower:
            return "compile_error"
        if "timeout" in content_lower:
            return "timeout"
        if "not found" in content_lower or "no such file" in content_lower:
            return "not_found"
        if "permission" in content_lower:
            return "permission_denied"
        return "general_error"

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if isinstance(part.get("text"), str):
                        parts.append(part["text"])
                    elif part.get("type") in {"image", "image_url", "input_image"}:
                        parts.append("[image]")
                else:
                    parts.append(str(part))
            return "\n".join(parts)
        return str(content)

    def _ensure_latest_user_last(
        self, original: List[Dict[str, Any]], compacted: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        latest = None
        for msg in original:
            if isinstance(msg, dict) and msg.get("role") == "user":
                latest = copy.deepcopy(msg)
        if latest is None:
            return compacted
        latest_text = self._content_to_text(latest.get("content"))
        filtered = [
            m
            for m in compacted
            if not (
                isinstance(m, dict)
                and m.get("role") == "user"
                and self._content_to_text(m.get("content")) == latest_text
            )
        ]
        filtered.append(latest)
        return filtered


def register(ctx) -> None:
    ctx.register_context_engine(ContextGovernorEngine())
