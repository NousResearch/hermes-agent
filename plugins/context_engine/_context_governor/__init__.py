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
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine
from hermes_constants import get_hermes_home
from plugins.context_engine._context_governor.key_state import (
    ContextGovernorKeyError,
    ContextGovernorKeyState,
    GovernedKeyBinding,
)

logger = logging.getLogger(__name__)


class ContextGovernorActivationError(RuntimeError):
    """The configured governor cannot provide its certified V2 contract."""


class ContextGovernorEngine(ContextEngine):
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
            "max_provenance_bytes": 131072,
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
        # Reset host-contract attributes (matches built-in
        # ContextCompressor.on_session_reset).
        self.last_real_prompt_tokens = 0
        self.last_compression_rough_tokens = 0
        self.awaiting_real_usage_after_compression = False
        self._ineffective_compression_count = 0
        self._last_compression_savings_pct = 100.0
        self._set_defer_baseline(0)
        self._previous_summary = None

    def bind_session_state(
        self,
        session_db: Any = None,
        session_id: str = "",
    ) -> None:
        """Bind the current session row to this engine instance.

        The host calls this from session-reset paths (run_agent.py:709)
        to allow the engine to rebind durable per-session state (e.g.
        summary-failure cooldowns). The context-governor adapter is
        stateless across sessions — the Rust binary owns receipt
        persistence and the active binary call is the source of truth —
        so this is a deliberate no-op. The method exists so the host's
        ``hasattr(engine, "bind_session_state")`` guard passes and any
        future per-session adapter state has a documented hook.

        See tests/agent/test_context_engine_host_contract.py for the
        contract this method satisfies.
        """
        return None

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
                                "description": "The item ID to expand (e.g. ctxi_0001_abcdef123456)",
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
        if not messages:
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
        source_messages = messages

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
        request = {
            "session_id": self.session_id or "hermes-session",
            "messages": [
                self._message_to_governor(m, i)
                for i, m in enumerate(source_messages)
                if isinstance(m, dict)
            ],
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
                "min_net_savings_tokens": self._policy["min_net_savings_tokens"],
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
            compacted = response.get("compacted_messages") or []
            compacted = [
                self._message_from_governor(m) for m in compacted if isinstance(m, dict)
            ]
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
            if checkpoint:
                metrics["llm_call"] = True
                metrics["passes"] = 2
                deterministic_projection = copy.deepcopy(compacted)
                compacted = self._enhance_with_llm_summary(
                    compacted, source_messages, response, focus_topic
                )
                if compacted != deterministic_projection:
                    self._llm_checkpoint_count += 1
                    metrics["llm_call_reason"] = f"{checkpoint_reason}:applied"
                else:
                    metrics["llm_call_reason"] = f"{checkpoint_reason}:fallback_extract"

            # Sanitation and the audited LLM checkpoint can both mutate the
            # emitted transcript. Rebind hashes/counts to that final adapter
            # output before persistence; never store the stale core response.
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
            store_result = self._store_response(response)
            if not store_result.get("verified"):
                raise ValueError(
                    "store-v2 did not report verified durable receipt admission"
                )
            self.set_activation_status(
                configured_engine=self.name,
                discovered=True,
                version_compatible=True,
                capability_compatible=True,
                instantiated=True,
                observed_live_engine=self.name,
                state="observed_live",
                detail=f"verified receipt {pending_receipt_id} persisted after compaction",
            )
            self.last_receipt_id = pending_receipt_id
            self.compression_count += 1
            self.last_error = None

            receipt = response.get("receipt") or {}
            exact_refs = receipt.get("covered_original_sources") or []
            metrics["summary_id"] = pending_receipt_id
            metrics["covered_source_hashes"] = [
                str(ref.get("content_blake3"))
                for ref in exact_refs
                if isinstance(ref, dict) and ref.get("content_blake3")
            ]
            metrics["exact_fallback_available"] = bool(exact_refs)
            metrics["integrity_result"] = "store_admission_verified"

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
            self._last_compression_savings_pct = savings_pct
            if savings_pct < 10:
                self._ineffective_compression_count += 1
            else:
                self._ineffective_compression_count = 0

            metrics["after_tokens"] = compacted_tokens
            metrics["after_bytes"] = len(
                json.dumps(compacted, ensure_ascii=False).encode("utf-8")
            )
            metrics["after_messages"] = len(compacted)
            metrics["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)

            return compacted or messages
        except Exception as exc:
            self.last_error = str(exc)
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

    def _message_to_governor(self, msg: Dict[str, Any], idx: int) -> Dict[str, Any]:
        role = msg.get("role") or "assistant"
        if role not in {"system", "user", "assistant", "tool"}:
            role = "assistant"
        out = {
            "role": role,
            "content": self._content_to_text(msg.get("content")),
        }
        message_id = msg.get("id") or msg.get("tool_call_id")
        if message_id:
            out["id"] = str(message_id)
        if msg.get("name"):
            out["name"] = str(msg.get("name"))
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
            out["name"] = msg.get("name")
        if msg.get("id"):
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
        if not self._deterministic_summary_checkpoint_ready(
            response, target_tokens=target_tokens
        ):
            return False, "deterministic_boundary_not_reached"
        maximum = self._max_checkpoints()
        if maximum is not None and self._llm_checkpoint_count >= maximum:
            return False, "checkpoint_limit_reached"
        strategy = self._checkpoint_strategy_json()
        if strategy == "off":
            return False, "checkpoint_strategy_off_or_invalid"
        if strategy == "ineffective_only":
            return True, "deterministic_ineffective"
        if isinstance(strategy, dict) and "after_n" in strategy:
            every = int(strategy["after_n"])
            ordinal = self.compression_count + 1
            return (
                ordinal % every == 0,
                f"after_n:{every}:ordinal:{ordinal}",
            )
        if isinstance(strategy, dict) and "threshold_pct" in strategy:
            receipt = response.get("receipt") or {}
            original = int(receipt.get("original_approx_tokens") or 0)
            savings = int(receipt.get("token_savings_estimate") or 0)
            savings_pct = (savings / original * 100.0) if original > 0 else 100.0
            threshold = int(strategy["threshold_pct"])
            return (
                savings_pct <= threshold,
                f"threshold_pct:{threshold}:observed:{savings_pct:.3f}",
            )
        return False, "checkpoint_strategy_unrecognized"

    @staticmethod
    def _deterministic_summary_checkpoint_ready(
        response: dict[str, Any], *, target_tokens: int | None = None
    ) -> bool:
        """Return true only with explicit fixed-point/diminishing-return evidence.

        The Rust receipt is the authority for both transcript identity and token
        savings. Missing or malformed evidence fails closed to the deterministic
        extractive summary.
        """
        receipt = response.get("receipt") or {}
        if not isinstance(receipt, dict):
            return False
        original_hash = receipt.get("original_transcript_blake3")
        compacted_hash = receipt.get("compacted_transcript_blake3")
        if not (
            isinstance(original_hash, str)
            and isinstance(compacted_hash, str)
            and re.fullmatch(r"[0-9a-f]{64}", original_hash)
            and re.fullmatch(r"[0-9a-f]{64}", compacted_hash)
        ):
            return False
        original_tokens = receipt.get("original_approx_tokens")
        compacted_tokens = receipt.get("compacted_approx_tokens")
        savings = receipt.get("token_savings_estimate")
        if (
            not isinstance(original_tokens, int)
            or isinstance(original_tokens, bool)
            or not isinstance(compacted_tokens, int)
            or isinstance(compacted_tokens, bool)
            or not isinstance(savings, int)
            or isinstance(savings, bool)
        ):
            return False
        if original_tokens <= 0 or compacted_tokens < 0 or savings < 0:
            return False
        if (
            compacted_tokens > original_tokens
            or savings != original_tokens - compacted_tokens
        ):
            return False
        # A deterministic fixed point is only an LLM boundary while its output
        # still exceeds the intended budget. If the deterministic projection
        # already fits, an auxiliary summary is unnecessary work and introduces
        # avoidable approximation.
        if target_tokens is not None and compacted_tokens <= max(0, int(target_tokens)):
            return False
        if original_hash == compacted_hash:
            return compacted_tokens == original_tokens and savings == 0
        diminishing_return_tokens = max(1, int(original_tokens * 0.10))
        return savings <= diminishing_return_tokens

    def _finalize_response(
        self,
        response: dict[str, Any],
        compacted: List[Dict[str, Any]],
    ) -> dict[str, Any]:
        """Bind the receipt to the normalized messages Hermes will receive."""
        payload = copy.deepcopy(response)
        payload["compacted_messages"] = [
            self._message_to_governor(message, i)
            for i, message in enumerate(compacted)
            if isinstance(message, dict)
        ]
        finalized = self._run_json(["finalize-v2"], payload)
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
        # Find the summary message in compacted output
        summary_idx = None
        for i, msg in enumerate(compacted):
            if isinstance(msg, dict) and msg.get("name") == "context_governor":
                summary_idx = i
                break
        if summary_idx is None:
            return compacted

        extractive_summary = compacted[summary_idx].get("content", "")

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
            return compacted  # Nothing to enhance

        summary_budget = min(
            int(self.context_length * 0.05) if self.context_length else 4000,
            8000,
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
            self.last_warning = warning
            logger.warning("context-governor: %s", warning)
            return compacted

        if focus_topic:
            prompt += f"\n\n=== FOCUS OVERRIDE ===\n{focus_topic}"

        llm_started = time.monotonic()
        try:
            metrics = self.last_compaction_metrics
            if metrics is not None:
                model, provider, _base_url, _api_key = self._resolve_moa_runtime()
                metrics["summarizer_model"] = model or None
                metrics["summarizer_provider"] = provider or None
            llm_summary = self._call_summary_llm(
                prompt, summary_budget, system_prompt=system_prompt
            )
            if metrics is not None:
                metrics["llm_latency_ms"] = round(
                    (time.monotonic() - llm_started) * 1000, 3
                )
            if llm_summary and llm_summary.strip():
                llm_summary = self._normalize_llm_summary_output(llm_summary)
                if not llm_summary:
                    warning = "LLM summary violated the structured output contract; using extractive summary"
                    self.last_warning = warning
                    logger.warning("context-governor: %s", warning)
                    return compacted
                audit = self._audit_compression_boundary(
                    turns_to_summarize, llm_summary
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
                        f"policy={policy}; using extractive summary"
                    )
                    self.last_warning = warning
                    logger.warning("context-governor: %s", warning)
                    if policy == "warn":
                        self._previous_summary = llm_summary
                        compacted[summary_idx] = {
                            **compacted[summary_idx],
                            "content": llm_summary,
                        }
                    elif policy == "freeze":
                        self.last_error = warning
                    return compacted

                self.last_warning = None
                candidate = list(compacted)
                candidate[summary_idx] = {
                    **candidate[summary_idx],
                    "content": llm_summary,
                }
                candidate_tokens = sum(
                    max(1, len(self._content_to_text(message.get("content"))) // 4)
                    for message in candidate
                    if isinstance(message, dict)
                )
                target_tokens = self._target_tokens(
                    int(
                        (response.get("receipt") or {}).get("original_approx_tokens")
                        or 0
                    )
                )
                if candidate_tokens > target_tokens:
                    warning = (
                        "LLM summary exceeded target "
                        f"({candidate_tokens} > {target_tokens} tokens); using extractive summary"
                    )
                    self.last_warning = warning
                    logger.warning("context-governor: %s", warning)
                    return compacted
                # Store for iterative updates on next compaction
                self._previous_summary = llm_summary
                # Replace the extractive summary with the LLM one
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
            # Keep the extractive summary — never worse

        return compacted

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
            warning = (
                f"compression-boundary audit failed; using extractive summary: {exc}"
            )
            self.last_warning = warning
            logger.warning("context-governor: %s", warning)
            return {
                "schema": "CompressionBoundaryAuditV1",
                "safe_to_reinject": False,
                "relinking_risk": "unknown",
                "adapter_error": str(exc),
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
        marker = "=== ACTIVE TASK ==="
        required_tail = "=== PRIOR CONTEXT SUMMARY ==="
        text = str(content or "").strip()
        start = text.find(marker)
        if start < 0:
            return None
        # Strip preamble (text before the first structural marker).
        if start > 0:
            text = text[start:]
        if required_tail not in text:
            return None
        return text

    def _resolve_moa_runtime(self) -> tuple[str, str, str, str]:
        """If the active provider is MoA, resolve to the aggregator's real
        provider/model/credentials.  Returns ``(model, provider, base_url,
        api_key)``.  When the main provider is not MoA, returns the current
        runtime values unchanged.

        ``_resolve_auto`` in ``auxiliary_client`` already does this for the
        auto-detection path, but ``_call_summary_llm`` passes explicit args
        which bypass auto-detection.  We resolve here so the explicit args
        carry a real HTTP provider instead of the virtual MoA facade.
        """
        model = self._summary_model or self.model
        provider = self._summary_provider or self.provider
        base_url = self._summary_base_url or self.base_url
        api_key = self._summary_api_key or self.api_key

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
        self, prompt: str, max_tokens: int, *, system_prompt: str = ""
    ) -> str | None:
        """Call an LLM for summary generation. Returns None on failure."""
        try:
            # Hermes centralizes auxiliary requests here.  The old
            # ``model_router`` imports never existed in the live package, which
            # made ``summary_mode: llm`` silently fall back to extractive mode.
            from agent.auxiliary_client import call_llm
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
        }

        # Resolve MoA virtual provider to the aggregator's real provider.
        # Without this, passing provider="moa" / model="<preset>" as explicit
        # args bypasses _resolve_auto's MoA handling in auxiliary_client,
        # and call_llm fails to create a client for the virtual MoA endpoint.
        model, provider, base_url, api_key = self._resolve_moa_runtime()
        if model:
            call_kwargs["model"] = model
        if provider:
            call_kwargs["provider"] = provider
        if base_url:
            call_kwargs["base_url"] = base_url
        if api_key:
            call_kwargs["api_key"] = api_key

        response = call_llm(**call_kwargs)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            content = str(content) if content else ""
        if not content.strip():
            return None
        return content.strip()

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

    def _store_response(self, response: dict[str, Any]) -> dict[str, Any]:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        receipt = response.get("receipt") or {}
        if receipt.get("schema") != "ContextCompactionReceiptV2":
            raise ValueError("refusing to persist a non-V2 Context Governor receipt")
        command = [str(self.binary), "store-v2", "--dir", str(self.store_dir)]
        command.extend(self._certified_store_args())
        binding = self._key_binding
        try:
            proc = subprocess.run(
                command,
                input=json.dumps(response, ensure_ascii=False),
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
                (
                    proc.stderr or proc.stdout or f"store exited {proc.returncode}"
                ).strip()
            )
        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("store-v2 returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise RuntimeError("store-v2 returned a non-object result")
        return result

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
