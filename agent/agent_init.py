"""Implementation of :meth:`AIAgent.__init__` as ``init_agent(agent, ...)``.

``init_agent`` is a thin, ordered orchestrator over ``_init_*`` / ``_build_*`` phase
helpers (routing → callbacks → client → tools → session → config sections → compression →
context engine). Phase ORDER is load-bearing: later phases read attributes earlier ones set.
Symbols that tests patch on ``run_agent.*`` (``OpenAI``, ``get_tool_definitions``,
``logger``, …) are resolved through :func:`_ra` so the patch contract is preserved.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import suppress
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse, urlunparse

from agent.context_compressor import ContextCompressor
from agent.agent_runtime_helpers import _ra
from agent.iteration_budget import IterationBudget
from agent.memory_manager import StreamingContextScrubber
from agent.session_activity import ActivityProvenance
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH, fetch_model_metadata, is_local_endpoint, query_ollama_num_ctx
)
from agent.process_bootstrap import _install_safe_stdio
from agent.subdirectory_hints import SubdirectoryHintTracker
from agent.think_scrubber import StreamingThinkScrubber
from agent.tool_guardrails import (
    ToolCallGuardrailConfig, ToolCallGuardrailController
)
from hermes_cli.config import cfg_get
from hermes_cli.route_identity import normalize_route_base_url
from hermes_cli.timeouts import get_provider_request_timeout
from hermes_constants import get_hermes_home
from utils import base_url_host_matches, is_truthy_value

# Same logger name as run_agent so caplog/patches on "run_agent" see our records.
logger = logging.getLogger("run_agent")


# Deduped: the gateway builds a fresh AIAgent per message, so it would warn every turn.
_warned_unavailable_providers: set[str] = set()


def _warn_memory_provider_unavailable(name: str, reason: str = "") -> None:
    """Warn once per provider that a configured memory provider is unavailable.

    ``is_available()`` is a side-effect-free hot-path check and can't log itself; without this
    the provider is silently dropped. ``reason`` (the provider's ``unavailable_reason()`` hint)
    can only reach the user here, so it is appended when present.
    """
    if name in _warned_unavailable_providers:
        return
    _warned_unavailable_providers.add(name)
    logger.warning(
        "Memory provider %r is selected but reports unavailable — external memory "
        "is disabled for this session (built-in memory still works). Check the "
        "provider's credentials/config with 'hermes memory status'. Note: "
        "systemd/gateway services do not inherit ~/.hermes/.env automatically; set "
        "any required variables in the service environment.%s",
        name,
        f" {reason}" if reason else "",
    )


def _provider_default_routes(provider: str) -> set[str]:
    """Return known exact default routes for a canonical provider id."""
    routes: set[str] = set()

    def add(value):
        route = normalize_route_base_url(value)
        if route:
            routes.add(route)

    with suppress(Exception):
        from hermes_cli.providers import HERMES_OVERLAYS, get_provider
        overlay = HERMES_OVERLAYS.get(provider)
        provider_def = get_provider(provider, allow_network=False)
        add(getattr(overlay, "base_url_override", ""))
        add(getattr(provider_def, "base_url", ""))

    with suppress(Exception):
        from providers import get_provider_profile
        add(getattr(get_provider_profile(provider), "base_url", ""))

    with suppress(Exception):
        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.models import normalize_provider as normalize_model_provider
        from hermes_cli.providers import normalize_provider as normalize_registry_provider
        for provider_id, config in PROVIDER_REGISTRY.items():
            if normalize_registry_provider(normalize_model_provider(provider_id)) == provider:
                add(getattr(config, "inference_base_url", ""))

    if provider == "gemini":
        routes.update(f"{route.rstrip('/')}/openai" for route in list(routes))
    return routes


def _context_route_mismatch(
    configured_base_url: Any, active_base_url: Any, configured_provider: Any, active_provider: Any,
    *, already_normalized: bool = False,
) -> bool:
    """Return whether a context pin's configured route differs from runtime."""
    _norm = (lambda v: str(v or "")) if already_normalized else normalize_route_base_url
    configured_route, active_route = _norm(configured_base_url), _norm(active_base_url)
    if configured_route:
        return configured_route != active_route

    configured_provider = str(configured_provider or "").strip()
    active_provider = str(active_provider or "").strip()
    if not configured_provider:
        return False
    try:
        from hermes_cli.models import normalize_provider as normalize_model_provider
        configured_provider = normalize_model_provider(configured_provider)
        active_provider = normalize_model_provider(active_provider)
    except Exception:
        configured_provider = configured_provider.lower()
        active_provider = active_provider.lower()
    with suppress(Exception):
        from hermes_cli.providers import normalize_provider as normalize_registry_provider
        configured_provider = normalize_registry_provider(configured_provider)
        active_provider = normalize_registry_provider(active_provider)

    if active_route:
        configured_routes = _provider_default_routes(configured_provider)
        if configured_routes:
            return active_route not in configured_routes
        # Named/custom providers have no catalog default routes: an empty configured URL
        # with a matching provider identity is the same route (gateway display paths
        # compare the raw empty model.base_url and must not drop model.context_length).
        return not (active_provider and configured_provider == active_provider)
    return bool(
        configured_provider and active_provider and configured_provider != active_provider
    )


def _normalize_custom_provider_name(value: Any) -> str:
    """Mirror runtime normalization for a requested custom-provider identity."""
    return str(value or "").strip().lower().replace(" ", "-")


def _custom_provider_runtime_ids(value: Any) -> set[str]:
    """Return raw/menu identities that runtime accepts for a configured name."""
    normalized = _normalize_custom_provider_name(value)
    if not normalized:
        return set()
    return {normalized, f"custom:{normalized}"}


def _build_codex_gpt5_autoraise_notice(
    autoraise: Dict[str, Any], context_length: Optional[int] = None
) -> str:
    """One-time notice when Codex gpt-5.x raises compaction (``autoraise``: model/from/to).

    ``context_length`` is the live-resolved window (Codex's catalog shifts server-side) so the
    banner reports what this session got. Printed for CLI and replayed via status_callback for
    gateway users, so it must be self-contained and include the exact opt-back-out command.
    """
    model = str(autoraise.get("model") or "gpt-5.4/5.5").strip().lower().rsplit("/", 1)[-1]
    if isinstance(context_length, int) and context_length > 0:
        cap = f"{round(context_length / 1000)}K"
    else:
        # Static fallback: codex-spark is natively 128K; gpt-5.4/5.5/5.6 are capped at 272K.
        cap = "128K" if model.startswith("gpt-5.3-codex-spark") else "272K"
    from_pct = int(round(autoraise["from"] * 100))
    to_pct = int(round(autoraise["to"] * 100))
    return (
        f"ℹ Codex {model} caps context at {cap}, so auto-compaction was raised "
        f"to {to_pct}% (from {from_pct}%) to use more of the window before "
        f"summarizing.\n"
        f"  Opt back out: hermes config set compression.codex_gpt55_autoraise false"
    )


def _resolve_compression_threshold(
    global_threshold: float, model_cthresh: Optional[float], *, model: Optional[str] = None,
    is_codex_autoraise: bool,
) -> tuple[float, Optional[Dict[str, Any]]]:
    """Global compaction threshold merged with a per-model override.

    Returns ``(threshold, autoraise_notice)``; the notice is set only when a Codex autoraise
    actually RAISES the threshold — it never lowers a higher user value (the user deliberately
    keeps more raw context). Other overrides (Arcee Trinity) stay unconditional.
    """
    if model_cthresh is None:
        return global_threshold, None
    if is_codex_autoraise:
        if model_cthresh <= global_threshold + 1e-9:
            return global_threshold, None
        return model_cthresh, {"model": model, "from": global_threshold, "to": model_cthresh}
    return model_cthresh, None


def _codex_gpt55_autoraise_notice_marker():
    """Per-profile marker path (``$HERMES_HOME`` is profile-scoped; not a config key)."""
    return get_hermes_home() / ".codex_gpt55_autoraise_notice"


def _codex_gpt55_autoraise_notice_state(autoraise: Dict[str, Any]) -> str:
    """Notice identity keyed on what it displays (model + from→to percentages).

    An unchanged threshold stays silent across restarts; a changed global threshold or a
    different autoraised Codex model re-notifies once.
    """
    model = str(autoraise.get("model") or "").strip().lower().rsplit("/", 1)[-1]
    from_pct = int(round(float(autoraise["from"]) * 100))
    to_pct = int(round(float(autoraise["to"]) * 100))
    return f"{model}:{from_pct}:{to_pct}"


def _codex_gpt55_autoraise_notice_seen(autoraise: Dict[str, Any]) -> bool:
    """True if this exact notice was already shown for this profile (unreadable = unseen)."""
    try:
        current = _codex_gpt55_autoraise_notice_state(autoraise)
        return _codex_gpt55_autoraise_notice_marker().read_text(
            encoding="utf-8"
        ).strip() == current
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _record_codex_gpt55_autoraise_notice(autoraise: Dict[str, Any]) -> None:
    """Persist that the notice was shown. Best-effort: a failure only re-shows it later."""
    with suppress(OSError, KeyError, TypeError, ValueError):
        marker = _codex_gpt55_autoraise_notice_marker()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(_codex_gpt55_autoraise_notice_state(autoraise), encoding="utf-8")


def _normalized_custom_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def _custom_provider_model_matches(agent_model: str, entry: Dict[str, Any]) -> bool:
    agent_model_norm = str(agent_model or "").strip().lower()
    # Multi-model entries (`providers.<name>.models` mapping / legacy `models:` list):
    # matching ANY catalog entry counts, else a provider whose `model` differs from the
    # session model drops its extra_body (e.g. OpenAI service_tier) → wrong billing tier.
    models = entry.get("models")
    catalog = [str(m).strip().lower() for m in models] if isinstance(models, (dict, list, tuple)) else []
    if catalog and agent_model_norm in catalog:
        return True
    provider_model = str(entry.get("model", "") or "").strip().lower()
    return (not provider_model and not catalog) or provider_model == agent_model_norm


def _custom_provider_extra_body_for_agent(
    *, provider: str, model: str, base_url: str, custom_providers: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    provider_norm = (provider or "").strip().lower()
    if provider_norm != "custom" and not provider_norm.startswith("custom:"):
        return None
    provider_key_filter = provider_norm.partition(":")[2].strip()
    target_url = _normalized_custom_base_url(base_url)
    if not target_url:
        return None

    fallback: Optional[Dict[str, Any]] = None
    for entry in custom_providers or []:
        if not isinstance(entry, dict):
            continue
        entry_keys = {
            str(entry.get("provider_key", "") or "").strip().lower(),
            str(entry.get("name", "") or "").strip().lower(),
        }
        if provider_key_filter and provider_key_filter not in entry_keys:
            continue
        if _normalized_custom_base_url(entry.get("base_url")) != target_url:
            continue
        extra_body = entry.get("extra_body")
        if not isinstance(extra_body, dict) or not extra_body:
            continue
        if str(entry.get("model", "") or "").strip():
            if _custom_provider_model_matches(model, entry):
                return dict(extra_body)
        elif fallback is None:
            fallback = dict(extra_body)
    return fallback


def _merge_custom_provider_extra_body(agent, custom_providers: List[Dict[str, Any]]) -> None:
    extra_body = _custom_provider_extra_body_for_agent(
        provider=agent.provider, model=agent.model, base_url=agent.base_url,
        custom_providers=custom_providers,
    )
    if not extra_body:
        return
    overrides = dict(getattr(agent, "request_overrides", {}) or {})
    merged_extra_body = dict(extra_body)
    existing_extra_body = overrides.get("extra_body")
    if isinstance(existing_extra_body, dict):
        merged_extra_body.update(existing_extra_body)
    overrides["extra_body"] = merged_extra_body
    agent.request_overrides = overrides


def _normalize_run_budget_seconds(value) -> Optional[float]:
    """Positive float or None (feature off). ``bool`` rejected: YAML ``true`` → 1s budget."""
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None  # NaN compares False → None


def _refuse_checkpoint_required_on_codex_app_server(
    checkpoint_required: bool, api_mode: Optional[str]
) -> None:
    """Fail closed at init: the codex app-server compacts its own thread without a truthful
    pre-compaction boundary (default "native" mode), so a required checkpoint can't be
    guaranteed — the compress_context() guard alone cannot cover native turns."""
    if checkpoint_required and api_mode == "codex_app_server":
        raise RuntimeError(
            "BLOCKED_MISSING_PREREQUISITE: compression.checkpoint_required "
            "is incompatible with the codex_app_server API mode: the codex "
            "agent compacts its own thread without a truthful pre-compaction "
            "transcript boundary, so a required pre-compress checkpoint "
            "cannot be guaranteed. Disable compression.checkpoint_required "
            "or use a non-app-server API mode."
        )


def _parse_config_int(raw: Any, default: int) -> int:
    """Strict int coercion: rejects bool (YAML ``true`` → 1) and fractional floats."""
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() else default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _cfg_flag(cfg: Dict[str, Any], key: str, default: bool) -> bool:
    """Legacy string-set truthiness used by the ``compression`` section."""
    return str(cfg.get(key, default)).lower() in {"true", "1", "yes"}


def _cfg_dict(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    """``cfg[key]`` if it is a mapping, else ``{}`` (malformed sections are ignored)."""
    section = cfg.get(key, {})
    return section if isinstance(section, dict) else {}


class CompressionSettings(SimpleNamespace):
    """Parsed ``compression`` config section (see ``_parse_compression_config``)."""


_EXPLICIT_API_MODES = {
    "chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse",
    "codex_app_server", "claude_agent_sdk",
}


def _resolve_api_mode(agent, api_mode, provider_name, base_url):
    """Set ``agent.api_mode`` (and provider rewrites) — ordered ladder, first match wins."""
    host, url = agent._base_url_hostname, agent._base_url_lower
    if api_mode in _EXPLICIT_API_MODES:
        agent.api_mode = api_mode
    elif agent.provider in {"openai-codex", "xai", "xai-oauth"}:
        agent.api_mode = "codex_responses"
    elif provider_name is None and host == "chatgpt.com" and "/backend-api/codex" in url:
        agent.api_mode = "codex_responses"
        agent.provider = "openai-codex"
    elif provider_name is None and host == "api.x.ai":
        agent.api_mode = "codex_responses"
        agent.provider = "xai"
    elif agent.provider == "anthropic" or (provider_name is None and host == "api.anthropic.com"):
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
    elif url.rstrip("/").endswith("/anthropic"):
        # Third-party Anthropic-compatible endpoints (MiniMax, DashScope) end in /anthropic.
        agent.api_mode = "anthropic_messages"
    elif agent.provider == "bedrock" or (
        host.startswith("bedrock-runtime.") and base_url_host_matches(url, "amazonaws.com")
    ):
        agent.api_mode = "bedrock_converse"
    elif agent.provider in {"nous", "nous-portal", "nousresearch"}:
        # Portal is dual-wire (anthropic/* → Messages, else chat_completions); covers direct
        # AIAgent construction without a resolved runtime.
        from hermes_cli.providers import nous_api_mode
        agent.api_mode = nous_api_mode(agent.model)
    else:
        # Host-mandated wire check — LAST, so the provider-slug rewrites above always win.
        # Covers api.meta.ai → codex_responses (prompt caching: 0% on chat vs 93-99%).
        # URL-driven, not provider-name-driven: `providers.meta` may point anywhere.
        try:
            # Note: provider="meta" without an api.meta.ai base_url (or with a non-api.meta.ai base_url)
            # intentionally falls through to chat_completions here. The wire protocol for Meta is URL-driven
            # BY DESIGN, not provider-name-driven, because user config `providers.meta` may point at any
            # OpenAI-compatible endpoint, and forcing `codex_responses` on the provider name alone would
            # break custom endpoints named "meta" that do not host the Responses API. See #63425.
            from hermes_cli.providers import host_mandated_api_mode as _host_mandated_api_mode
            _mandated = _host_mandated_api_mode(base_url or "")
        except Exception:
            _mandated = None
        agent.api_mode = _mandated if _mandated is not None else "chat_completions"


def _finalize_routing(agent, api_mode, credential_pool):
    # Credential-pool validation runs AFTER provider auto-detection so a pool scoped to
    # "anthropic" isn't rejected for provider=None + anthropic.com URL.
    # Regression from #63048 which placed this check before the URL-based auto-detection block above (fixed
    # #63425).
    if credential_pool is not None:
        try:
            from agent.credential_pool import credential_pool_matches_provider
            if not credential_pool_matches_provider(
                credential_pool, agent.provider, base_url=agent.base_url,
            ):
                agent._credential_pool = None
        except Exception:
            agent._credential_pool = None

    # Warm the transport cache so import errors surface at init (non-fatal: some modes lack one).
    with suppress(Exception):
        agent._get_transport()

    with suppress(Exception):
        from hermes_cli.model_normalize import (
            _AGGREGATOR_PROVIDERS, normalize_model_for_provider
        )

        if agent.provider not in _AGGREGATOR_PROVIDERS:
            agent.model = normalize_model_for_provider(agent.model, agent.provider)

    # Auto-upgrade to Responses for GPT-5.x-style models and direct OpenAI URLs, unless
    # api_mode was explicit, the runtime is ACP (`acp://` clients route themselves, no
    # Responses surface) or Azure OpenAI (gpt-5.x on /chat/completions only). Provider
    # exceptions live in _provider_model_requires_responses_api.
    _base_lower = str(agent.base_url or "").lower()
    if (
        # GPT-5.x models usually require the Responses API path, but some providers have exceptions (for
        # example Copilot's gpt-5-mini still uses chat completions). ACP runtimes are excluded: an ACP
        # client handles its own routing and does not implement the Responses API surface. Keyed on the
        # `acp://` scheme, not one vendor, so every ACP client is covered. When api_mode was explicitly
        # provided, respect it — the user knows what their endpoint supports (#10473). Exception: Azure
        # OpenAI serves gpt-5.x on /chat/completions and does NOT support the Responses API — skip the
        # upgrade for Azure (openai.azure.com), even though it looks OpenAI-compatible.
        api_mode is None
        and agent.api_mode == "chat_completions"
        and agent.provider != "copilot-acp"
        and not _base_lower.startswith(("acp://", "acp+tcp://"))
        and not agent._is_azure_openai_url()
        and (
            agent._is_direct_openai_url()
            or agent._provider_model_requires_responses_api(agent.model, provider=agent.provider)
        )
    ):
        agent.api_mode = "codex_responses"
        # Invalidate the eager-warmed transport cache — api_mode changed after the warm.
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()

    # Pre-warm the OpenRouter metadata cache (1h TTL) off-thread so the first pricing estimate
    # doesn't block. Process-level Event guard: an unguarded spawn leaks a thread per message.
    if (agent.provider == "openrouter" or agent._is_openrouter_url()) and \
            not _ra()._openrouter_prewarm_done.is_set():
        _ra()._openrouter_prewarm_done.set()
        threading.Thread(
            target=fetch_model_metadata, daemon=True, name="openrouter-prewarm",
        ).start()


def _set_defaults(agent, table: Dict[str, Any]) -> None:
    """Assign each ``name -> value`` on ``agent``; callables are factories (fresh per agent)."""
    for name, value in table.items():
        setattr(agent, name, value() if callable(value) else value)


# Control-flow state (interrupts / steer / redirect / delegation / background review).
_CONTROL_STATE: Dict[str, Any] = {
    "_executing_tools": False,  # lets _vprint print while tools run with stream consumers on
    "_tool_guardrails": ToolCallGuardrailController,
    "_tool_guardrail_halt_decision": None,
    # Interrupts. Hard cancellation is separate from redirect/message state; the Event makes
    # the cause atomic for auxiliary stream pollers.
    "_interrupt_requested": False,
    "_interrupt_message": None,  # optional message that triggered the interrupt
    "_hard_interrupt_requested": threading.Event,
    "_execution_thread_id": None,  # set at run_conversation() start
    "_interrupt_thread_signal_pending": False,
    "_client_lock": threading.RLock,
    "_model_request_active": threading.Event,
    "_supports_active_turn_redirect": True,
    # /steer: the drain hook appends the note to the last tool result after the current
    # batch — no interrupt, no new user turn (role alternation preserved).
    "_pending_steer": None,
    "_pending_steer_lock": threading.Lock,
    # Active-turn redirect: keep the valid turn prefix, cancel only the in-flight request,
    # rebuild the tail with the correction. Drained at a role-safe boundary.
    "_pending_redirect": None,
    "_pending_redirect_lock": threading.Lock,
    # Concurrent-tool worker tids: `_set_interrupt` on `_execution_thread_id` alone doesn't
    # reach ThreadPoolExecutor workers, so interrupt()/clear_interrupt() fan out to these.
    "_tool_worker_threads": set,
    "_tool_worker_threads_lock": threading.Lock,
    # Subagent delegation: depth (0 = top-level) and running children (interrupt propagation).
    "_delegate_depth": 0,
    "_active_children": list,
    "_active_children_lock": threading.Lock,
    # Background review (agent/background_review.py): the run is installed before the worker
    # starts and fences its first provider phase; the agent pointer enables interrupt fan-out.
    "_background_review_agent": None,
    "_background_review_run": None,
    "_background_review_lock": threading.Lock,
}

# Per-turn bookkeeping: budgets, activity tracking, rate-limit/credits telemetry.
_TURN_STATE: Dict[str, Any] = {
    # Iteration budget: notify the LLM only on exhaustion (one message, one grace call, then
    # a forced summary) — intermediate pressure warnings made models give up early.
    "_budget_exhausted_injected": False,
    "_budget_grace_call": False,
    "_run_budget_started_at": None,  # set by turn_context.prepare_turn when a budget is active
    "_run_budget_wrapup_injected": False,  # one-shot latch for the 80% wrap-up notice
    # Activity tracking (API call / tool / stream chunk) for the gateway timeout handler and
    # "still working" notifications. Named provenances are stamped only by compression writers.
    "_last_activity_ts": lambda: time.time(),
    "_last_activity_desc": "initializing",
    "_last_activity_provenance": ActivityProvenance.UNKNOWN,
    "_session_activity_last_persist_mono": 0.0,  # rate-limits durable SessionDB stamps
    "_current_tool": None,
    "_api_call_count": 0,
    # Opt-out for the between-turns MCP refresh; set on forks that need byte-identical tools[].
    "_skip_mcp_refresh": False,
    # Registry generation of the tool snapshot (set in _load_tools): a late refresh rejects
    # a stale rebuild instead of clobbering a newer one.
    "_tool_snapshot_generation": 0,
    "_rate_limit_state": None,  # from x-ratelimit-* headers; read by /usage
    # Credits tracking (dev-only, HERMES_DEV_CREDITS) from x-nous-credits-* headers; session
    # start is latched on the first header so cumulative spend can be reported.
    "_credits_state": None,
    "_credits_session_start_micros": None,
    "_or_cache_hits": 0,  # X-OpenRouter-Cache-Status: HIT count
}

# Session persistence state.
_SESSION_STATE: Dict[str, Any] = {
    "_session_messages": list,
    # Responses encrypted-reasoning replay: routes that 400 with ``invalid_encrypted_content``
    # make the loop disable it for the session (stateless continuity).
    "_codex_reasoning_replay_enabled": True,
    "_memory_write_origin": "assistant_tool",
    "_memory_write_context": "foreground",
    # Cached system prompt (built once, rebuilt on compression) + its cross-session-stable
    # prefix, kept separately only to place an early cache marker.
    "_cached_system_prompt": None,
    "_cached_system_prompt_static": None,
    # Whether close() also closes _session_db. False: a caller-supplied handle is usually the
    # SHARED launch handle; callers handing over a DEDICATED handle set True.
    "_owns_session_db": False,
    # Close flush and turn-start flush can overlap; the durable marker lives on each message
    # dict, so its test-and-append is serialized per agent.
    "_session_persist_lock": threading.RLock,
    # CLI's just-accepted user dict, reused by turn setup so its durable marker survives a
    # close-persistence race.
    "_pending_cli_user_message": None,
    "_last_flushed_db_idx": 0,  # DB-write cursor (prevents duplicate writes)
    "_session_db_created": False,  # DB row deferred to run_conversation()
    # False on helper agents (compression / hygiene / review forks) that hand the session to
    # a continuation row that must stay open.
    "_end_session_on_close": True,
    # True on the background review fork: never persist, so its harness turn can't hijack
    # the live session.
    "_persist_disabled": False,
}

# Streaming delivery state.
_STREAM_STATE: Dict[str, Any] = {
    "_stream_callback": None,  # streaming TTS; set early so _vprint can reference it
    "_stream_needs_break": False,  # one "\n\n" before the next real text delta after tools
    # Stateful scrubbers: <memory-context> / thinking spans split across deltas defeat
    # per-delta regexes (both tags must be in one string).
    "_stream_context_scrubber": StreamingContextScrubber,
    "_stream_think_scrubber": StreamingThinkScrubber,
    "_current_streamed_assistant_text": "",  # so a later completed interim isn't re-sent
    "_delivered_interim_texts": set,  # interims this user turn (spans Codex continuations)
    # Single-writer guard for the delta sink: each attempt claims a monotonic writer token and
    # the sink drops chunks from threads holding a stale one, so a superseded stream can't
    # interleave with the retry's. Threads that never claimed are never fenced.
    "_stream_writer_lock": threading.Lock,
    "_stream_writer_token": 0,
    "_stream_writer_tls": threading.local,
    "_stream_writer_dropped": 0,
    # API-facing user message override when it differs from the persisted transcript (voice).
    "_persist_user_message_idx": None,
    "_persist_user_message_override": None,
    "_persist_user_message_timestamp": None,
    # Image-to-text fallbacks cached per payload/URL so one tool loop doesn't re-run vision.
    "_anthropic_image_fallback_cache": dict,
}


def _init_prompt_cache_config(agent):
    # Anthropic prompt caching (~75% input savings): auto-enabled for Claude on native
    # Anthropic, OpenRouter and anthropic_messages gateways. See _anthropic_prompt_cache_policy.
    agent._use_prompt_caching, agent._use_native_cache_layout = (
        agent._anthropic_prompt_cache_policy()
    )
    agent._cache_disabled = False
    # cache_ttl: "5m" (default) or "1h" (2x write cost; pays off with >5-minute pauses);
    # unknown values keep "5m". A falsy/off value disables caching entirely (OAuth plans
    # billing cache writes, proxies adding their own cache_control); the disable survives
    # /model switches and fallback re-derivation.
    # Anthropic supports "5m" (default) and "1h" cache TTL tiers. Read from config.yaml under
    # prompt_caching.cache_ttl; unknown values keep "5m". 1h tier costs 2x on write vs 1.25x for 5m, but
    # amortizes across long sessions with >5-minute pauses between turns (#14971). This is useful for OAuth
    # subscription users where cache writes bill against "extra usage" or for third-party proxies that
    # inject their own cache_control markers (#13477).
    agent._cache_ttl = "5m"
    with suppress(Exception):
        from hermes_cli.config import load_config_readonly as _load_pc_cfg
        from agent.agent_runtime_helpers import cache_ttl_means_disabled
        _pc_cfg = _load_pc_cfg().get("prompt_caching", {}) or {}
        _ttl = _pc_cfg.get("cache_ttl", "5m")
        if _ttl in {"5m", "1h"}:
            agent._cache_ttl = _ttl
        elif cache_ttl_means_disabled(_ttl):
            agent._use_prompt_caching = False
            agent._use_native_cache_layout = False
            agent._cache_ttl = None
            agent._cache_disabled = True


def _init_turn_state(agent, run_budget_seconds):
    _set_defaults(agent, _TURN_STATE)
    # Wall-clock run budget per turn: constructor arg wins, else agent.run_budget_seconds
    # (in _apply_agent_section). None = fully off (no clock reads, injection, or capping).
    agent.run_budget_seconds = _normalize_run_budget_seconds(run_budget_seconds)
    from agent.credits_tracker import new_credits_latch
    agent._credits_latch = new_credits_latch()  # threshold-notice latch (sticky keys + gates)


def _setup_logging(agent):
    # agent.log (INFO+) + errors.log (WARNING+); idempotent so per-message gateway agents
    # don't duplicate handlers.
    from hermes_logging import setup_logging, setup_verbose_logging
    setup_logging(hermes_home=_ra()._hermes_home)

    if agent.verbose_logging:
        setup_verbose_logging()
        _ra().logger.info("Verbose logging enabled (third-party library logs suppressed)")
    # Quiet mode must NOT raise per-logger levels: isEnabledFor() runs before propagation and
    # would starve the root file handlers. Noise reduction belongs in hermes_logging.


def _bedrock_region_from_url(base_url) -> str:
    """AWS region from a bedrock-runtime.<region>.amazonaws.com URL (default us-east-1)."""
    m = re.search(r"bedrock-runtime\.([a-z0-9-]+)\.", base_url or "")
    return m.group(1) if m else "us-east-1"


def _print_key_banner(key, label: str, warn_missing: bool = False) -> None:
    """Masked credential line. ``key`` may be a callable Entra ID bearer provider (Azure
    Foundry) — never invoke or inspect it. Keys ≤ 12 chars (incl. "dummy-key") are not shown."""
    from agent.azure_identity_adapter import is_token_provider
    if is_token_provider(key):
        print("🔑 Using credentials: Microsoft Entra ID")
    elif isinstance(key, str) and len(key) > 12:
        print(f"🔑 Using {label}: {key[:8]}...{key[-4:]}")
    elif warn_missing:
        print("⚠️  Warning: API key appears invalid or missing")


def _init_anthropic_client(agent, api_key, base_url, _provider_timeout):
    """anthropic_messages: native Anthropic SDK (or AnthropicBedrock for Bedrock+Claude)."""
    from agent.anthropic_adapter import build_anthropic_client
    from agent.anthropic_credentials import resolve_anthropic_token
    agent.client = None
    agent._client_kwargs = {}
    agent._anthropic_base_url = base_url
    if agent.provider == "bedrock":
        # AnthropicBedrock SDK for full feature parity (prompt caching, thinking budgets).
        from agent.anthropic_adapter import build_anthropic_bedrock_client
        _br_region = agent._bedrock_region = _bedrock_region_from_url(base_url)
        agent._anthropic_client = build_anthropic_bedrock_client(_br_region)
        agent._anthropic_api_key = "aws-sdk"
        agent._is_anthropic_oauth = False
        agent.api_key = "aws-sdk"
        if not agent.quiet_mode:
            print(f"🤖 AI Agent initialized with model: {agent.model} (AWS Bedrock + AnthropicBedrock SDK, {_br_region})")
        return
    # ANTHROPIC_TOKEN fallback only for native Anthropic — other anthropic_messages providers
    # must use their own key or Anthropic credentials leak to third-party endpoints.
    # Falling back would send Anthropic credentials to third-party endpoints (Fixes #1739, #minimax-401).
    _is_native_anthropic = agent.provider == "anthropic"
    effective_key = api_key or (resolve_anthropic_token() if _is_native_anthropic else None) or ""

    # MiniMax OAuth tokens live ~15 min and the SDK freezes api_key at construction, so use a
    # callable provider: build_anthropic_client mints a fresh bearer per request (re-reading
    # auth.json, so other processes' refreshes are seen).
    if agent.provider == "minimax-oauth" and isinstance(effective_key, str) and effective_key:
        try:
            from hermes_cli.auth import build_minimax_oauth_token_provider
            effective_key = build_minimax_oauth_token_provider()
        except Exception as _mm_exc:  # noqa: BLE001 — never block startup on this
            logging.getLogger(__name__).warning(
                "MiniMax OAuth: failed to install per-request token provider "
                "(%s); falling back to static bearer that will expire ~15min in.",
                _mm_exc,
            )

    agent.api_key = effective_key
    agent._anthropic_api_key = effective_key
    # OAuth only for native Anthropic: third-party anthropic_messages providers must never
    # trip OAuth paths — those inject Claude-Code identity headers → 401/403.
    # Only mark the session as OAuth-authenticated when the token genuinely belongs to native Anthropic.
    # Third-party providers (MiniMax, Kimi, GLM, LiteLLM proxies) that accept the Anthropic protocol must
    # never trip OAuth code paths — doing so injects Claude-Code identity headers and system prompts that
    # cause 401/403 on their endpoints. See #1739.
    from agent.anthropic_credentials import _is_oauth_token as _is_oat
    agent._is_anthropic_oauth = _is_oat(effective_key) if (_is_native_anthropic and isinstance(effective_key, str)) else False
    agent._anthropic_client = build_anthropic_client(effective_key, base_url, timeout=_provider_timeout)
    if not agent.quiet_mode:
        print(f"🤖 AI Agent initialized with model: {agent.model} (Anthropic native)")
        _print_key_banner(effective_key, "token")


def _init_moa_client(agent, api_key):
    """provider == "moa": virtual Mixture-of-Agents facade, no real HTTP client."""
    from agent.moa_loop import build_moa_facade
    agent.api_mode = "chat_completions"

    # build_moa_facade relays "moa.*" events through tool_progress_callback so every surface
    # shows each reference's answer before the aggregator acts. Display-only; shared with
    # fallback-restore so a restored facade keeps emitting.
    # build_moa_facade wires the reference relay that routes reference-model outputs to the agent's
    # tool_progress_callback so every surface that already consumes it (CLI spinner/scrollback, TUI,
    # desktop, gateway) can show each reference's answer as a labelled block before the aggregator acts. The
    # facade emits "moa.reference", "moa.progress", "moa.phase", and "moa.aggregating" events, forwarded
    # through the same callback the tool lifecycle uses. Best-effort and cache-safe — display-only events,
    # they never touch the message history. See #53802.
    agent.client = build_moa_facade(agent, agent.model)
    agent._client_kwargs = {}
    agent.api_key = api_key or "moa-virtual-provider"
    agent.base_url = "moa://local"
    if not agent.quiet_mode:
        print(f"🤖 AI Agent initialized with MoA preset: {agent.model}")


def _init_bedrock_client(agent, base_url):
    """bedrock_converse: boto3 directly, no OpenAI client."""
    agent._bedrock_region = _bedrock_region_from_url(base_url)
    # Guardrail config — read from config.yaml at init time.
    agent._bedrock_guardrail_config = None
    with suppress(Exception):
        from hermes_cli.config import load_config_readonly as _load_br_cfg
        _gr = _load_br_cfg().get("bedrock", {}).get("guardrail", {})
        if _gr.get("guardrail_identifier") and _gr.get("guardrail_version"):
            agent._bedrock_guardrail_config = {
                "guardrailIdentifier": _gr["guardrail_identifier"],
                "guardrailVersion": _gr["guardrail_version"],
            }
            for _src, _dst in (("stream_processing_mode", "streamProcessingMode"), ("trace", "trace")):
                if _gr.get(_src):
                    agent._bedrock_guardrail_config[_dst] = _gr[_src]
    agent.client = None
    agent._client_kwargs = {}
    if not agent.quiet_mode:
        _gr_label = " + Guardrails" if agent._bedrock_guardrail_config else ""
        print(f"🤖 AI Agent initialized with model: {agent.model} (AWS Bedrock, {agent._bedrock_region}{_gr_label})")


def _explicit_client_kwargs(agent, api_key, base_url, _provider_timeout) -> Dict[str, Any]:
    """OpenAI-client kwargs from explicit CLI/gateway credentials (auth already resolved)."""
    _parsed_url = urlparse(base_url)
    client_kwargs = {"api_key": api_key, "base_url": base_url}
    if _parsed_url.query:
        client_kwargs["base_url"] = urlunparse(_parsed_url._replace(query=""))
        client_kwargs["default_query"] = {k: v[0] for k, v in parse_qs(_parsed_url.query).items()}
    if _provider_timeout is not None:
        client_kwargs["timeout"] = _provider_timeout
    if agent.provider == "copilot-acp":
        client_kwargs["command"] = agent.acp_command
        client_kwargs["args"] = agent.acp_args
    # OpenCode Zen free tier is served ANONYMOUSLY and 401s any bearer (incl. our keyless
    # placeholder): send an empty Authorization header to override the SDK's "Bearer <key>".
    with suppress(Exception):
        from hermes_cli.models import (
            OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER, opencode_zen_free_headers
        )
        if api_key == OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER:
            client_kwargs["default_headers"] = opencode_zen_free_headers()
    _headers_for = _host_default_headers_factory(base_url)
    if _headers_for is not None:
        client_kwargs["default_headers"] = _headers_for(api_key, base_url)
    elif "default_headers" not in client_kwargs:
        # Fall back to profile.default_headers for providers that declare custom headers
        # (Vercel AI Gateway attribution, Kimi User-Agent on non-kimi.com endpoints).
        with suppress(Exception):
            from providers import get_provider_profile as _gpf
            _ph = _gpf(agent.provider)
            if _ph and _ph.default_headers:
                client_kwargs["default_headers"] = dict(_ph.default_headers)
    return client_kwargs


def _routed_client_kwargs(agent, fallback_model, _provider_timeout) -> Dict[str, Any]:
    """OpenAI-client kwargs via the centralized provider router (no explicit creds).

    Falls through to the init-time fallback chain, then raises with the missing-key /
    no-provider diagnostic.
    """
    from agent.auxiliary_client import resolve_provider_client
    _routed_client, _ = resolve_provider_client(
        agent.provider or "auto", model=agent.model, raw_codex=True)
    if _routed_client is not None:
        return _client_kwargs_from_routed(_routed_client, _provider_timeout)
    # No credentials: try the fallback chain BEFORE failing (an exhausted single-entry pool
    # must not die with a misleading "No LLM provider configured"); only explicitly named
    # providers keep the missing-key diagnostic.
    # An exhausted single-entry pool (typically ``openrouter`` under free-tier daily quotas) must still
    # reach the chain instead of dying at init with a misleading "No LLM provider configured" error. See
    # #17929.
    _explicit = (agent.provider or "").strip().lower()
    for _fb in _fallback_entries(fallback_model):
        try:
            from hermes_cli.fallback_config import resolve_entry_api_key
            _fb_explicit_key = resolve_entry_api_key(_fb)
            _fb_client, _fb_model = resolve_provider_client(
                _fb["provider"], model=_fb["model"], raw_codex=True,
                explicit_base_url=_fb.get("base_url"), explicit_api_key=_fb_explicit_key,
            )
        except Exception as _fb_exc:
            logger.debug("Init-time fallback entry %s failed: %s", _fb.get("provider"), _fb_exc)
            continue
        if _fb_client is not None:
            agent.provider = _fb["provider"]
            agent.model = _fb_model or _fb["model"]
            agent._fallback_activated = True
            return _client_kwargs_from_routed(_fb_client, _provider_timeout)
    if _explicit and _explicit not in {"auto", "openrouter", "custom"}:
        # Explicit non-OpenRouter provider with no creds and no usable fallback: fail fast.
        # Use the provider's real env var name (alibaba → DASHSCOPE_API_KEY).
        _env_hint = f"{_explicit.upper()}_API_KEY"
        with suppress(Exception):
            from hermes_cli.auth import PROVIDER_REGISTRY
            _pcfg = PROVIDER_REGISTRY.get(_explicit)
            if _pcfg and _pcfg.api_key_env_vars:
                _env_hint = _pcfg.api_key_env_vars[0]
        raise RuntimeError(
            f"Provider '{_explicit}' is set in config.yaml but no API key "
            f"was found. Set the {_env_hint} environment "
            f"variable, or switch to a different provider with `hermes model`."
        )
    raise RuntimeError(
        "No LLM provider configured. Run `hermes model` to "
        "select a provider, or run `hermes setup` for first-time "
        "configuration."
    )


_FINE_GRAINED_BETA = "fine-grained-tool-streaming-2025-05-14"


def _apply_openai_header_policy(agent, client_kwargs: Dict[str, Any]) -> None:
    """Mutate ``client_kwargs`` (== ``agent._client_kwargs``) with header/TLS policy, in order:
    OpenRouter Claude beta header → model.default_headers → custom-provider TLS/extra_headers."""
    # Fine-grained tool streaming for Claude on OpenRouter: without the beta header
    # Anthropic buffers the whole tool call and OpenRouter's proxy times out.
    _effective_base = str(client_kwargs.get("base_url", "")).lower()
    if base_url_host_matches(_effective_base, "openrouter.ai") and "claude" in (agent.model or "").lower():
        headers = client_kwargs.get("default_headers") or {}
        existing_beta = headers.get("x-anthropic-beta", "")
        if _FINE_GRAINED_BETA not in existing_beta:
            headers["x-anthropic-beta"] = ",".join(filter(None, (existing_beta, _FINE_GRAINED_BETA)))
            client_kwargs["default_headers"] = headers
    # model.default_headers override provider/SDK defaults (WAFs rejecting SDK headers).
    agent._apply_user_default_headers()
    try:
        from hermes_cli.config import (
            apply_custom_provider_extra_headers_to_client_kwargs,
            apply_custom_provider_tls_to_client_kwargs, get_compatible_custom_providers,
            load_config,
        )
        _cp_entries = get_compatible_custom_providers(load_config())
        _cp_base_url = str(client_kwargs.get("base_url") or agent.base_url or "")
        apply_custom_provider_tls_to_client_kwargs(client_kwargs, _cp_base_url, _cp_entries)
        # Per-provider extra_headers applied last so the most specific config level wins.
        # SECURITY: values may carry credentials — never log them.
        apply_custom_provider_extra_headers_to_client_kwargs(client_kwargs, _cp_base_url, _cp_entries)
    except Exception:
        logger.debug("custom-provider TLS resolution skipped", exc_info=True)


def _init_openai_client(agent, api_key, base_url, fallback_model, _provider_timeout):
    """OpenAI-wire client: resolve kwargs, apply header/TLS policy, construct."""
    if api_key and base_url:
        client_kwargs = _explicit_client_kwargs(agent, api_key, base_url, _provider_timeout)
    else:
        client_kwargs = _routed_client_kwargs(agent, fallback_model, _provider_timeout)
    try:
        from agent.bedrock_adapter import configure_bedrock_openai_client_kwargs
        configure_bedrock_openai_client_kwargs(client_kwargs, timeout=_provider_timeout)
    except Exception:
        if agent.provider == "bedrock" and "bedrock-mantle." in str(client_kwargs.get("base_url", "")):
            raise

    agent._client_kwargs = client_kwargs  # stored for rebuilding after interrupt
    _apply_openai_header_policy(agent, client_kwargs)
    agent.api_key = client_kwargs.get("api_key", "")
    agent.base_url = client_kwargs.get("base_url", agent.base_url)
    try:
        from agent.ssl_guard import verify_ca_bundle
        verify_ca_bundle()
        agent.client = agent._create_openai_client(client_kwargs, reason="agent_init", shared=True)
        if not agent.quiet_mode:
            print(f"🤖 AI Agent initialized with model: {agent.model}")
            if base_url:
                print(f"🔗 Using custom base URL: {base_url}")
            _print_key_banner(client_kwargs.get("api_key", "none"), "API key", warn_missing=True)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI client: {e}")


def _init_claude_agent_sdk_client(agent, api_key, base_url):
    """The Agent SDK owns its subprocess and subscription OAuth: no OpenAI-compatible client or
    HTTP endpoint exists on this route, so the agent carries no client and an empty base URL."""
    agent.client = None
    agent._client_kwargs = {}
    agent.api_key = api_key or "claude-subscription-oauth"
    agent.base_url = base_url or ""
    if not agent.quiet_mode:
        print(f"🤖 AI Agent initialized with model: {agent.model} (Claude Agent SDK, subscription)")


# Wire-mode client initializers keyed by ``api_mode``; ``chat_completions``/``codex_responses`` and every
# unknown mode take the OpenAI-compatible client. Each receives (agent, api_key, base_url, timeout).
_CLIENT_INITIALIZERS = {
    "anthropic_messages": lambda agent, api_key, base_url, timeout: _init_anthropic_client(agent, api_key, base_url, timeout),
    "bedrock_converse": lambda agent, api_key, base_url, timeout: _init_bedrock_client(agent, base_url),
    "claude_agent_sdk": lambda agent, api_key, base_url, timeout: _init_claude_agent_sdk_client(agent, api_key, base_url),
}


def _build_client(agent, api_key, base_url, fallback_model):
    # LLM client per wire mode (raw_codex=True: the main agent needs direct
    # responses.stream()). One provider/model timeout up front so every path applies it.
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    _provider_timeout = get_provider_request_timeout(agent.provider, agent.model)
    initializer = _CLIENT_INITIALIZERS.get(agent.api_mode)
    # The MoA virtual provider owns its client unless the wire mode is native Anthropic (which wins).
    if agent.provider == "moa" and agent.api_mode != "anthropic_messages":
        _init_moa_client(agent, api_key)
    elif initializer is not None:
        initializer(agent, api_key, base_url, _provider_timeout)
    else:
        _init_openai_client(agent, api_key, base_url, fallback_model, _provider_timeout)


def _lazy_headers(module: str, name: str, pass_key: bool = False, pass_base: bool = False):
    """Header factory ``(api_key, base_url) -> dict`` importing ``module.name`` at call time.
    ``pass_key`` forwards ``(key, base_url=base)``; ``pass_base`` forwards ``(base)``."""
    def factory(key, base):
        import importlib
        fn = getattr(importlib.import_module(module), name)
        if pass_key:
            return fn(key, base_url=base)
        return fn(base) if pass_base else fn()
    return factory


# Host → default_headers factory for explicit base_url client construction. Ordered: first
# host match wins; no match falls back to the provider profile's declared headers.
_HOST_DEFAULT_HEADERS: List[tuple[str, Callable[[Any, str], Dict[str, str]]]] = [
    ("openrouter.ai", _lazy_headers("agent.auxiliary_client", "build_or_headers")),
    ("integrate.api.nvidia.com",
     _lazy_headers("agent.auxiliary_client", "build_nvidia_nim_headers", pass_base=True)),
    ("api.routermint.com", _lazy_headers("agent.client_lifecycle", "_routermint_headers")),
    ("githubcopilot.com", _lazy_headers("hermes_cli.models", "copilot_default_headers")),
    ("api.kimi.com", lambda _k, _b: {"User-Agent": "claude-code/0.1.0"}),
    ("portal.qwen.ai", _lazy_headers("agent.client_lifecycle", "_qwen_portal_headers")),
    ("chatgpt.com", _lazy_headers("agent.codex_headers", "codex_cloudflare_headers", pass_key=True)),
    ("x.ai", _lazy_headers("tools.xai_http", "hermes_xai_default_headers")),
]


def _host_default_headers_factory(base_url: str):
    for host, factory in _HOST_DEFAULT_HEADERS:
        if base_url_host_matches(base_url, host):
            return factory
    return None


def _client_kwargs_from_routed(client, timeout) -> Dict[str, Any]:
    """OpenAI-client kwargs mirroring a router-resolved client, keeping its provider headers
    (SDK stores them in ``_custom_headers``; older/mocked clients expose ``default_headers``)."""
    kwargs = {"api_key": client.api_key, "base_url": str(client.base_url)}
    if timeout is not None:
        kwargs["timeout"] = timeout
    headers = (
        getattr(client, "_custom_headers", None)
        or getattr(client, "default_headers", None)
        or getattr(client, "_default_headers", None)
    )
    if headers:
        kwargs["default_headers"] = dict(headers)
    return kwargs


def _fallback_entries(fallback_model) -> List[Dict[str, Any]]:
    """Normalize legacy single-dict ``fallback_model`` / list ``fallback_providers``."""
    if isinstance(fallback_model, dict):
        fallback_model = [fallback_model]
    if not isinstance(fallback_model, list):
        return []
    return [
        f for f in fallback_model if isinstance(f, dict) and f.get("provider") and f.get("model")
    ]


def _init_fallback_chain(agent, fallback_model):
    # Stable pool-entry identity: OAuth refreshes can replace the token before a failed
    # request is recovered, so the key value alone can't attribute the failure.
    from agent.agent_runtime_helpers import sync_credential_pool_entry_id
    sync_credential_pool_entry_id(agent)

    # Ordered backups tried when the primary is exhausted (legacy single-dict or list).
    agent._fallback_chain = _fallback_entries(fallback_model)
    agent._fallback_index = 0
    agent._fallback_activated = getattr(agent, "_fallback_activated", False)
    # Legacy attribute kept for backward compat (tests, external callers)
    agent._fallback_model = agent._fallback_chain[0] if agent._fallback_chain else None
    chain = agent._fallback_chain
    if chain and not agent.quiet_mode:
        labels = [f"{f['model']} ({f['provider']})" for f in chain]
        if len(chain) == 1:
            print(f"🔄 Fallback model: {labels[0]}")
        else:
            print(f"🔄 Fallback chain ({len(chain)} providers): " + " → ".join(labels))


def _load_tools(agent, enabled_toolsets, disabled_toolsets):
    # A multiplexed gateway may have switched HERMES_HOME since model_tools was imported;
    # make sure this profile's plugins are discovered before the tool snapshot.
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
    except Exception:
        logger.warning("Plugin discovery failed during agent setup", exc_info=True)

    # Capture the registry generation FIRST so a concurrent refresh can detect staleness.
    try:
        from tools.registry import registry as _snapshot_registry
        agent._tool_snapshot_generation = _snapshot_registry._generation
    except Exception:
        agent._tool_snapshot_generation = 0
    import model_tools
    agent.tools = model_tools.get_tool_definitions(
        enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets,
        quiet_mode=agent.quiet_mode,
    )

    agent.valid_tool_names = {tool["function"]["name"] for tool in agent.tools} if agent.tools else set()
    # Kanban guidance is session-static (kanban_show iff HERMES_KANBAN_TASK); resolve once.
    from agent.prompt_builder import KANBAN_GUIDANCE
    agent._kanban_worker_guidance = (
        KANBAN_GUIDANCE if "kanban_show" in agent.valid_tool_names else ""
    )
    if agent.quiet_mode:
        return
    if agent.tools:
        print(f"🛠️  Loaded {len(agent.tools)} tools: {', '.join(sorted(agent.valid_tool_names))}")
        if enabled_toolsets:
            print(f"   ✅ Enabled toolsets: {', '.join(enabled_toolsets)}")
        if disabled_toolsets:
            print(f"   ❌ Disabled toolsets: {', '.join(disabled_toolsets)}")
        import model_tools
        requirements = model_tools.check_toolset_requirements()
        missing_reqs = [name for name, available in requirements.items() if not available]
        if missing_reqs:
            print(f"⚠️  Some tools may not work due to missing requirements: {missing_reqs}")
    else:
        print("🛠️  No tools loaded (all tools filtered out or unavailable)")
    if agent.save_trajectories:
        print("📝 Trajectory saving enabled")
    if agent.ephemeral_system_prompt:
        prompt_preview = agent.ephemeral_system_prompt[:60] + "..." if len(agent.ephemeral_system_prompt) > 60 else agent.ephemeral_system_prompt
        print(f"🔒 Ephemeral system prompt: '{prompt_preview}' (not saved to trajectories)")
    if agent._use_prompt_caching:
        if agent._use_native_cache_layout and agent.provider == "anthropic":
            source = "native Anthropic"
        elif agent._use_native_cache_layout:
            source = "Anthropic-compatible endpoint"
        else:
            source = "Claude via OpenRouter"
        print(f"💾 Prompt caching: ENABLED ({source}, {agent._cache_ttl} TTL)")


def _publish_session_id(session_id: str) -> None:
    """Expose the session ID to tools via ContextVar (+ legacy os.environ fallback).

    If the ContextVar bridge fails to import, keep the root-agent env fallback but never let
    delegated construction publish a child ID process-wide.
    """
    try:
        from gateway.session_context import set_current_session_id
        set_current_session_id(session_id)
    except Exception:
        try:
            from agent.delegation_context import is_delegated_child_context
            delegated_child = is_delegated_child_context()
        except Exception:
            delegated_child = False
        if not delegated_child:
            os.environ["HERMES_SESSION_ID"] = session_id


def _init_session_state(agent, session_id, session_db, parent_session_id, reasoning_config, max_tokens,
    checkpoints_enabled, checkpoint_max_snapshots, checkpoint_max_total_size_mb, checkpoint_max_file_size_mb):
    agent.session_start = datetime.now()
    agent.session_id = session_id or (
        f"{agent.session_start.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    _publish_session_id(agent.session_id)

    # ~/.hermes/sessions/ — kept unconditionally for request_dump_*.json debug breadcrumbs.
    agent.logs_dir = get_hermes_home() / "sessions"
    agent.logs_dir.mkdir(parents=True, exist_ok=True)
    # Per-session JSON snapshot is opt-in (sessions.write_json_snapshots); state.db is canonical.
    agent._session_json_enabled = False
    with suppress(Exception):
        from hermes_cli.config import load_config_readonly as _load_sess_cfg
        _sess_cfg = (_load_sess_cfg().get("sessions") or {})
        agent._session_json_enabled = bool(_sess_cfg.get("write_json_snapshots", False))

    _set_defaults(agent, _SESSION_STATE)

    # Filesystem checkpoint manager (transparent — not a tool)
    from tools.checkpoint_manager import CheckpointManager
    agent._checkpoint_mgr = CheckpointManager(
        enabled=checkpoints_enabled, max_snapshots=checkpoint_max_snapshots,
        max_total_size_mb=checkpoint_max_total_size_mb,
        max_file_size_mb=checkpoint_max_file_size_mb,
    )

    agent._session_db = session_db  # optional SQLite store (CLI/gateway-provided)
    agent._parent_session_id = parent_session_id
    agent._session_init_model_config = {
        "max_iterations": agent.max_iterations,
        "reasoning_config": reasoning_config,
        "max_tokens": max_tokens,
    }
    # Process-scoped --yolo is persisted so `hermes --resume` restores the bypass
    # (SessionDB.session_yolo_enabled); session-scoped /yolo toggles persist separately.
    with suppress(Exception):
        from tools.approval import _YOLO_MODE_FROZEN
        if _YOLO_MODE_FROZEN:
            agent._session_init_model_config["yolo_mode"] = True

    # In-memory todo list for task planning (one per agent/session)
    from tools.todo_tool import TodoStore
    agent._todo_store = TodoStore()


def _apply_display_config(agent, _agent_cfg, platform):
    # show_commentary: Codex phase=commentary → interim path (true) or reasoning channel.
    agent.show_commentary = bool(_cfg_dict(_agent_cfg, "display").get("show_commentary", True))

    # Window (seconds) for the bounded /fast auto|cold modes (agent.fast_mode).
    agent.fast_auto_seconds = (_agent_cfg.get("agent") or {}).get("fast_auto_seconds", 60)

    # lmstudio_load_mode: "explicit" (preload via management API) or "jit" (Auto-Evict path).
    _model_section = _cfg_dict(_agent_cfg, "model")
    _load_mode = str(_model_section.get("lmstudio_load_mode", "explicit") or "explicit").strip().lower()
    agent.lmstudio_load_mode = _load_mode if _load_mode in {"explicit", "jit"} else "explicit"
    if agent.lmstudio_load_mode != _load_mode:
        logger.warning(
            "Invalid model.lmstudio_load_mode=%r; expected 'explicit' or 'jit'. Using explicit.",
            _model_section.get("lmstudio_load_mode"),
        )

    # model.streaming=false seeds _disable_streaming (the loop's runtime fallback) for
    # backends with broken streaming tool calls. Session-scoped; orthogonal to display.streaming.
    _streaming = str(_model_section.get("streaming", "true")).strip().lower()
    agent._disable_streaming = _streaming in {"false", "0", "no", "off"}
    if not agent._disable_streaming and _streaming not in {"true", "1", "yes", "on"}:
        logger.warning(
            "Invalid model.streaming=%r; expected a boolean. Using streaming (default).",
            _model_section.get("streaming"),
        )

    try:
        agent._tool_guardrails = ToolCallGuardrailController(
            ToolCallGuardrailConfig.from_mapping(
                _agent_cfg.get("tool_loop_guardrails", {}), platform=platform,
            )
        )
    except Exception as _tlg_err:
        _ra().logger.warning("Tool loop guardrail config ignored: %s", _tlg_err)


def _memory_provider_init_kwargs(agent, platform) -> Dict[str, Any]:
    """Scoping kwargs for ``MemoryManager.initialize_all`` (status_callback is CLI-only:
    gateway status travels a different path and the indicator no-ops without it)."""
    kwargs = {
        "session_id": agent.session_id,
        "platform": platform or "cli",
        "hermes_home": str(get_hermes_home()),
        "agent_context": "primary",
    }
    if kwargs["platform"] == "cli":
        kwargs["warning_callback"] = agent._emit_warning
        kwargs["status_callback"] = agent._emit_status
    # Session title (e.g. honcho derives chat-scoped session keys from it).
    if agent._session_db:
        with suppress(Exception):
            _st = agent._session_db.get_session_title(agent.session_id)
            if _st:
                kwargs["session_title"] = _st
    # Gateway user/chat identity for per-user scoping (gateway_session_key: stable per-chat
    # Honcho session isolation).
    for _ident in _GATEWAY_IDENTITY_PARAMS:
        _val = getattr(agent, f"_{_ident}")
        if _val:
            kwargs[_ident] = _val
    # Profile identity for per-profile provider scoping
    with suppress(Exception):
        from hermes_cli.profiles import get_active_profile_name
        kwargs["agent_identity"] = get_active_profile_name()
        kwargs["agent_workspace"] = "hermes"
    return kwargs


def _init_memory(agent, _agent_cfg, skip_memory, platform):
    # Persistent memory (MEMORY.md + USER.md) — loaded from disk
    agent._memory_store = None
    agent._memory_enabled = False
    agent._user_profile_enabled = False
    agent._memory_nudge_interval = 10
    agent._turns_since_memory = 0
    agent._iters_since_skill = 0
    # skip_memory skips the external *provider*; enabled_toolsets=["memory"] still gets the
    # built-in store so the memory tool never sees store=None.
    # Flush/background agents can still pass enabled_toolsets=["memory"] so the built-in file store exists
    # and the memory tool does not fail with store=None (#65429). A toolset on disabled_toolsets is not a
    # request: a caller that denylists memory while its default toolset still names it must not get
    # MEMORY.md loaded by an enabled-only check. (Cron agents now run with skip_memory=False and take the
    # normal path here.)
    _memory_toolset_requested = (
        "memory" in (agent.enabled_toolsets or [])
        and "memory" not in (agent.disabled_toolsets or [])
    )
    if not skip_memory or _memory_toolset_requested:
        # Memory is optional — don't break agent init
        with suppress(Exception):
            from tools.memory_tool import (
                MemoryStore, get_builtin_memory_config, get_builtin_memory_store_flags,
            )
            mem_config = get_builtin_memory_config(_agent_cfg)
            agent._memory_enabled, agent._user_profile_enabled = get_builtin_memory_store_flags(
                _agent_cfg
            )
            agent._memory_nudge_interval = int(mem_config.get("nudge_interval", 10))
            if agent._memory_enabled or agent._user_profile_enabled:
                agent._memory_store = MemoryStore(
                    memory_char_limit=mem_config.get("memory_char_limit", 2200),
                    user_char_limit=mem_config.get("user_char_limit", 1375),
                    memory_enabled=agent._memory_enabled,
                    user_profile_enabled=agent._user_profile_enabled,
                )
                agent._memory_store.load_from_disk()

    # External memory provider plugin (one at a time, alongside built-in): memory.provider.
    agent._memory_manager = None
    if not skip_memory:
        try:
            _mem_provider_name = mem_config.get("provider", "") if mem_config else ""
            if _mem_provider_name and _mem_provider_name.strip():
                from agent.memory_manager import MemoryManager as _MemoryManager
                from plugins.memory import load_memory_provider as _load_mem
                agent._memory_manager = _MemoryManager()
                _mp = _load_mem(_mem_provider_name)
                if _mp and _mp.is_available():
                    agent._memory_manager.add_provider(_mp)
                elif _mp is not None and _mem_provider_name not in _warned_unavailable_providers:
                    # unavailable_reason() reads config/probes importlib — skip it once warned.
                    _unavailable_reason = ""
                    with suppress(Exception):
                        _unavailable_reason = _mp.unavailable_reason()
                    _warn_memory_provider_unavailable(_mem_provider_name, _unavailable_reason)
                if agent._memory_manager.providers:
                    agent._memory_manager.initialize_all(**_memory_provider_init_kwargs(agent, platform))
                    _ra().logger.info("Memory provider '%s' activated", _mem_provider_name)
                else:
                    _ra().logger.debug("Memory provider '%s' not found or not available", _mem_provider_name)
                    agent._memory_manager = None
        except Exception as _mpe:
            _ra().logger.warning("Memory provider plugin init failed: %s", _mpe)
            agent._memory_manager = None

    from agent.memory_manager import inject_memory_provider_tools
    inject_memory_provider_tools(agent)


def _apply_agent_section(agent, _agent_cfg):
    # Skills config: nudge interval for skill creation reminders
    agent._skill_nudge_interval = 10
    with suppress(Exception):
        agent._skill_nudge_interval = int(_agent_cfg.get("skills", {}).get("creation_nudge_interval", 10))

    _agent_section = _cfg_dict(_agent_cfg, "agent")
    # Both: "auto" (model-list match), true, false, or list of model substrings; independent
    # of each other (gates in agent/system_prompt.py).
    agent._tool_use_enforcement = _agent_section.get("tool_use_enforcement", "auto")
    agent._execution_guidance = _agent_section.get("execution_guidance", "auto")

    # Wall-clock run budget from config — only when the constructor arg was not given.
    if agent.run_budget_seconds is None:
        agent.run_budget_seconds = _normalize_run_budget_seconds(
            _agent_section.get("run_budget_seconds")
        )

    # Empty-response guard: a malformed section falls back to schema defaults (on, $0.25).
    from agent.empty_response_guard import resolve_guard_settings
    (
        agent._empty_guard_enabled, agent._empty_guard_cost_threshold_usd
    ) = resolve_guard_settings(_agent_section.get("empty_response_guard"))

    # "auto" (codex_responses only), true (all api_modes), false, or model substrings.
    agent._intent_ack_continuation = _agent_section.get("intent_ack_continuation", "auto")

    # Default-on boolean gates: anti-stall guards (notice-only), universal guidance toggles
    # (ALL models, unlike enforcement), the local toolchain probe, Bot Mode protocol section.
    for _key in (
        "stall_guards", "task_completion_guidance", "parallel_tool_call_guidance",
        "environment_probe", "bot_mode_protocol",
    ):
        setattr(agent, f"_{_key}", bool(_agent_section.get(_key, True)))
    # Warm the probe (~0.5s of subprocesses) off-thread so the first prompt build finds it cached.
    if agent._environment_probe:
        with suppress(Exception):
            from tools.env_probe import warm_environment_probe_async
            warm_environment_probe_async()

    # "Bot Chat" gate hint for hosts that defer the DB title write past the first prompt build.
    agent._session_title_hint = None

    # platform_hints: <platform>: {append|replace}, stored verbatim (agent/system_prompt.py).
    agent._platform_hint_overrides = _cfg_dict(_agent_cfg, "platform_hints")

    # App-level API retry count (wraps each model API call). Default 3; 1 = single attempt.
    try:
        _api_retries = max(int(_agent_section.get("api_max_retries", 3)), 1)
    except (TypeError, ValueError):
        _api_retries = 3
    agent._api_max_retries = _api_retries


def _positive_int(raw: Any, *, reject: tuple = ()) -> Optional[int]:
    """``int(raw)`` when positive, else None. ``reject`` lists types refused outright (bool, float)."""
    if reject and isinstance(raw, reject):
        return None
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _compression_threshold(agent, cfg: Dict[str, Any]) -> tuple[float, bool]:
    """Global threshold merged with the per-model override; stashes the autoraise notice.
    Codex gpt-5.4/5.5 raise to 85% (272K cap → 50% would compact at ~136K); the opt-out flag
    restores the global value, and the notice has its own display gate."""
    threshold = float(cfg.get("threshold", 0.50))
    autoraise = _cfg_flag(cfg, "codex_gpt55_autoraise", True)
    notice_enabled = _cfg_flag(cfg, "codex_gpt55_autoraise_notice", True)
    agent._compression_threshold_autoraised = None
    with suppress(Exception):
        from agent.auxiliary_client import (
            _compression_threshold_for_model as _cthresh_fn,
            _is_codex_gpt54_or_gpt55 as _is_codex_gpt54_or_gpt55_fn,
            _is_codex_spark as _is_codex_spark_fn,
        )
        _model_cthresh = _cthresh_fn(
            agent.model, agent.provider, allow_codex_gpt55_autoraise=autoraise,
        )
        # Codex autoraises apply only when they RAISE; Arcee Trinity keeps its
        # unconditional override.
        threshold, agent._compression_threshold_autoraised = _resolve_compression_threshold(
            threshold,
            _model_cthresh,
            model=agent.model,
            is_codex_autoraise=(
                _is_codex_gpt54_or_gpt55_fn(agent.model, agent.provider)
                or _is_codex_spark_fn(agent.model, agent.provider)
            ),
        )
    return threshold, notice_enabled


def _compression_codex_settings(cfg: Dict[str, Any]) -> tuple[str, bool, Optional[int]]:
    """``codex_app_server_auto`` / ``codex_responses_native`` / ``codex_responses_compact_threshold``."""
    app_server_auto = str(cfg.get("codex_app_server_auto", "native") or "native").lower()
    if app_server_auto not in {"native", "hermes", "off"}:
        _ra().logger.warning(
            "Invalid compression.codex_app_server_auto=%r; using 'native'. "
            "Valid values are: native, hermes, off.",
            app_server_auto,
        )
        app_server_auto = "native"
    # Native Responses server-side compaction (opt-in; gate in agent/native_compaction.py).
    # Truthy coercion so "false"/"off" strings stay disabled.
    responses_native = is_truthy_value(cfg.get("codex_responses_native", False))
    _raw = cfg.get("codex_responses_compact_threshold")
    compact_threshold = None
    if _raw is not None:
        compact_threshold = _positive_int(_raw, reject=(bool, float))
        if compact_threshold is None:
            _ra().logger.warning(
                "Invalid compression.codex_responses_compact_threshold=%r; "
                "using the automatic threshold derived from local compression.",
                _raw,
            )
    return app_server_auto, responses_native, compact_threshold


def _parse_compression_config(agent, _agent_cfg) -> CompressionSettings:
    """Parse the ``compression`` section. Defaults here MUST match DEFAULT_CONFIG."""
    cfg = _cfg_dict(_agent_cfg, "compression")
    threshold, autoraise_notice_enabled = _compression_threshold(agent, cfg)
    # Plain int()/float() coercions raise on garbage; evaluated up front, in config order.
    target_ratio = float(cfg.get("target_ratio", 0.20))
    protect_last = int(cfg.get("protect_last_n", 20))
    # max_attempts: retry rounds before "max compression attempts reached"; some sessions
    # need >3 (incompressible tool schemas). Default 3, floor 1, cap 10.
    max_attempts = _parse_config_int(cfg.get("max_attempts", 3), 3)
    if max_attempts < 1:
        max_attempts = 3
    # threshold_tokens: absolute cap (lower of ratio threshold and this); clamped to the
    # window at apply-time.
    threshold_tokens = cfg.get("threshold_tokens")
    if threshold_tokens is not None:
        threshold_tokens = _positive_int(threshold_tokens)
    # Non-system head messages to protect (system prompt is always protected); 0 is a
    # legitimate "system prompt + summary + tail".
    protect_first = max(0, int(cfg.get("protect_first_n", 3)))
    checkpoint_required = is_truthy_value(cfg.get("checkpoint_required"), default=False)
    _refuse_checkpoint_required_on_codex_app_server(
        checkpoint_required, getattr(agent, "api_mode", None)
    )
    app_server_auto, responses_native, compact_threshold = _compression_codex_settings(cfg)
    # Opt-in idle compaction: compact up front when a session resumes after this many
    # seconds idle (0 = disabled). Consumed by build_turn_context().
    idle_compact_after_seconds = max(0, int(cfg.get("idle_compact_after_seconds", 0)))
    return CompressionSettings(
        threshold=threshold,
        autoraise_notice_enabled=autoraise_notice_enabled,
        enabled=_cfg_flag(cfg, "enabled", True),
        target_ratio=target_ratio,
        protect_last=protect_last,
        # "lean" keeps a clamped 2.5%/10K-25K verbatim tail (continuity rides the summary);
        # "legacy" restores the 0.20*threshold tail. Unknown → lean inside the compressor.
        tail_mode=str(cfg.get("tail_mode", "lean")).strip().lower(),
        # Actionable user messages guaranteed to survive in the tail (default 1, floor 1).
        min_tail_users=max(1, _parse_config_int(cfg.get("min_tail_user_messages", 1), 1)),
        max_attempts=min(max_attempts, 10),
        # Opt-in proactive tool-result prune trigger (0 = disabled; negatives = disabled).
        proactive_prune_tokens=max(0, _parse_config_int(cfg.get("proactive_prune_tokens", 0), 0)),
        proactive_prune_min_chars=_parse_config_int(
            cfg.get("proactive_prune_min_result_chars", 8000), 8000
        ),
        proactive_prune_min_reclaim=max(
            0, _parse_config_int(cfg.get("proactive_prune_min_reclaim_tokens", 4096), 4096)
        ),
        protect_first=protect_first,
        abort_on_summary_failure=_cfg_flag(cfg, "abort_on_summary_failure", False),
        # Per-model threshold overrides: keys substring-matched against the model name
        # (longest match wins); {} = global threshold for all models.
        model_thresholds={
            str(k): float(v) for k, v in _cfg_dict(cfg, "model_thresholds").items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        },
        threshold_tokens=threshold_tokens,
        checkpoint_required=checkpoint_required,
        # In-place compaction: no session-id rotation. default=True MUST match DEFAULT_CONFIG
        # (a False default flipped agents into rotation mode when the key was omitted).
        in_place=is_truthy_value(cfg.get("in_place"), default=True),
        # Opt-in: micro-compaction rewrites sent history per turn (breaks the cache prefix).
        micro_compact=is_truthy_value(cfg.get("micro_compact"), default=False),
        # Pass cadence in completed turns; each pass costs one prompt-cache break (>= 1).
        micro_compact_every_n_turns=max(
            1, _parse_config_int(cfg.get("micro_compact_every_n_turns", 1), 1)
        ),
        # Rolling-summary defrag threshold, in tokens.
        micro_compact_defrag_tokens=max(
            1, _parse_config_int(cfg.get("micro_compact_defrag_threshold_tokens", 2000), 2000)
        ),
        codex_app_server_auto=app_server_auto,
        codex_responses_native=responses_native,
        codex_responses_compact_threshold=compact_threshold,
        idle_compact_after_seconds=idle_compact_after_seconds,
    )


def _warn_invalid_config_int(
    what: str, value: Any, requirement: str, fallback: str, print_fallback: str = "",
) -> None:
    """Log + stderr-print an invalid integer config value (``print_fallback``: user-facing
    wording where it differs from the log line)."""
    _ra().logger.warning(
        "Invalid %s: %r — %s. Falling back to %s.", what, value, requirement, fallback,
    )
    print(
        f"\n⚠ Invalid {what}: {value!r}\n"
        f"  {requirement[0].upper() + requirement[1:]}.\n"
        f"  Falling back to {print_fallback or fallback}.\n",
        file=sys.stderr,
    )


def _custom_provider_configured_base_url(
    _configured_provider: str, _agent_cfg, _custom_providers
) -> str:
    """Base URL of a named custom provider (``providers.<name>`` first, then
    ``custom_providers``), normalized for route comparison; "" if unknown.
    Disabled ``providers.*`` entries also mask their ``custom_providers`` twin.
    """
    _wanted = _normalize_custom_provider_name(_configured_provider)
    _user_providers = _agent_cfg.get("providers")
    _disabled_ids: set[str] = set()
    if isinstance(_user_providers, dict):
        from hermes_cli.config import is_provider_enabled
        for _key, _entry in _user_providers.items():
            if not isinstance(_entry, dict):
                continue
            _ids = _custom_provider_runtime_ids(_key) | _custom_provider_runtime_ids(_entry.get("name"))
            if not is_provider_enabled(_entry):
                _disabled_ids.update(_ids)
                continue
            if _wanted in _ids:
                _url = normalize_route_base_url(
                    _entry.get("api") or _entry.get("url") or _entry.get("base_url")
                )
                if _url:
                    return _url
    for _entry in _custom_providers:
        if not isinstance(_entry, dict):
            continue
        _key_ids = _custom_provider_runtime_ids(_entry.get("provider_key"))
        if _key_ids & _disabled_ids:
            continue
        if _wanted in _key_ids | _custom_provider_runtime_ids(_entry.get("name")):
            _url = normalize_route_base_url(_entry.get("base_url"))
            if _url:
                return _url
    return ""


# Provider ids whose runtime is resolved first-hand (never a named custom provider).
_RUNTIME_FIRST_PROVIDER_IDS = {
    "auto", "moa", "vertex", "google-vertex", "vertex-ai", "gcp-vertex", "vertexai",
}


def _configured_default_base_url(_agent_cfg, _model_cfg, _custom_providers) -> str:
    """Normalized route of the configured default model (``model.base_url``, else the named
    custom provider's URL when ``model.provider`` is not a first-class/auth provider)."""
    _configured_base_url = normalize_route_base_url(_model_cfg.get("base_url"))
    _configured_provider = str(_model_cfg.get("provider") or "").strip()
    _norm = _normalize_custom_provider_name(_configured_provider)
    _custom_provider_candidate = bool(_norm)
    if _norm in _RUNTIME_FIRST_PROVIDER_IDS:
        _custom_provider_candidate = False
    elif _custom_provider_candidate and _norm != "custom" and not _norm.startswith("custom:"):
        with suppress(Exception):
            from hermes_cli.auth import resolve_provider as resolve_auth_provider
            _custom_provider_candidate = (
                str(resolve_auth_provider(_norm) or "").strip().lower() != _norm
            )
    if not _configured_base_url and _custom_provider_candidate:
        _configured_base_url = _custom_provider_configured_base_url(
            _configured_provider, _agent_cfg, _custom_providers
        )
    return _configured_base_url


def _active_route_url(agent, base_url) -> str:
    """The runtime route, keeping the requested URL's query string when it is the same route."""
    _active = str(agent.base_url or "")
    _requested = str(base_url or "")
    if "?" in _requested.split("#", 1)[0]:
        with suppress(TypeError, ValueError):
            _without_query = urlunparse(urlparse(_requested)._replace(query=""))
            if normalize_route_base_url(_without_query) == normalize_route_base_url(_active):
                _active = _requested
    return normalize_route_base_url(_active)


def _scope_context_length_to_default_runtime(
    agent, _agent_cfg, _model_cfg, _custom_providers, _config_context_length, base_url
) -> Optional[int]:
    """Return ``model.context_length`` only if it describes the active runtime.

    It describes the configured default model; a ``--model`` launch has already replaced
    ``agent.model``, so carrying the default's window into that runtime is stale. Live
    switch/fallback paths already clear it — direct-start stays consistent with them.
    """
    _default = _model_cfg.get("default")
    if isinstance(_default, dict):
        from hermes_cli.config import split_model_config_default
        _default, _ = split_model_config_default(_default)
    _configured_default_model = str(_default or "").strip()
    _configured_default_runtime_model = _configured_default_model
    _active_runtime_model = agent.model
    if _configured_default_model:
        with suppress(Exception):
            from hermes_cli.model_normalize import normalize_model_for_provider
            _configured_default_runtime_model = normalize_model_for_provider(
                _configured_default_model, agent.provider
            )
            _active_runtime_model = normalize_model_for_provider(agent.model, agent.provider)
    _configured_base_url = _configured_default_base_url(_agent_cfg, _model_cfg, _custom_providers)
    _active_base_url = _active_route_url(agent, base_url)
    _route_mismatch = _context_route_mismatch(
        _configured_base_url, _active_base_url, str(_model_cfg.get("provider") or "").strip(),
        agent.provider, already_normalized=True,
    )
    _model_mismatch = bool(
        _configured_default_runtime_model
        and _configured_default_runtime_model != _active_runtime_model
    )
    if _model_mismatch or _route_mismatch:
        _ra().logger.debug(
            "Ignoring model.context_length=%s for startup runtime %s at %s "
            "(configured default is %s at %s)",
            _config_context_length,
            agent.model,
            _active_base_url or agent.provider,
            _configured_default_model,
            _configured_base_url or _model_cfg.get("provider"),
        )
        return None
    return _config_context_length


_CTX_LEN_REQUIREMENT = "must be a positive integer (e.g. 256000, not '256K')"


def _warn_invalid_custom_provider_context_length(agent, _custom_providers) -> None:
    """Surface a context_length the helper silently skipped (not a positive int)."""
    _target = normalize_route_base_url(agent.base_url)
    if not _target:
        return
    for _cp_entry in _custom_providers:
        if not isinstance(_cp_entry, dict):
            continue
        if normalize_route_base_url(_cp_entry.get("base_url")) != _target:
            continue
        _cp_models = _cp_entry.get("models", {})
        _cp_model_cfg = _cp_models.get(agent.model, {}) if isinstance(_cp_models, dict) else None
        _cp_ctx = _cp_model_cfg.get("context_length") if isinstance(_cp_model_cfg, dict) else None
        if _cp_ctx is not None and _positive_int(_cp_ctx) is None:
            _warn_invalid_config_int(
                f"context_length for model {agent.model!r} in custom_providers",
                _cp_ctx, _CTX_LEN_REQUIREMENT, "auto-detection", "auto-detected context window",
            )
        return


def _resolve_context_length(agent, _agent_cfg, base_url):
    # Aux compression model context_length hint (custom endpoints often can't report it).
    try:
        _aux_cfg = cfg_get(_agent_cfg, "auxiliary", "compression", default={})
    except Exception:
        _aux_cfg = {}
    _aux_ctx = _aux_cfg.get("context_length") if isinstance(_aux_cfg, dict) else None
    try:
        agent._aux_compression_context_length_config = int(_aux_ctx) if _aux_ctx is not None else None
    except (TypeError, ValueError):
        agent._aux_compression_context_length_config = None

    # model.max_tokens from config when the caller did not pass one.
    _model_cfg = _agent_cfg.get("model", {})
    _model_section = _model_cfg if isinstance(_model_cfg, dict) else {}
    _config_max_tokens = _model_section.get("max_tokens")
    if agent.max_tokens is None and _config_max_tokens is not None:
        agent.max_tokens = _positive_int(_config_max_tokens, reject=(bool,))
        if agent.max_tokens is None:
            _warn_invalid_config_int(
                "model.max_tokens in config.yaml", _config_max_tokens,
                "must be a positive integer (e.g. 4096)", "provider default",
            )
    agent._session_init_model_config["max_tokens"] = agent.max_tokens

    _config_context_length = _model_section.get("context_length")
    if _config_context_length is not None:
        try:
            _config_context_length = int(_config_context_length)
        except (TypeError, ValueError):
            _warn_invalid_config_int(
                "model.context_length in config.yaml", _config_context_length,
                "must be a plain integer (e.g. 256000, not '256K')",
                "auto-detection", "auto-detected context window",
            )
            _config_context_length = None

    # Resolve custom_providers before route-scoping: a named provider may keep its URL here.
    try:
        from hermes_cli.config import get_compatible_custom_providers
        _custom_providers = get_compatible_custom_providers(_agent_cfg)
    except Exception:
        _custom_providers = _agent_cfg.get("custom_providers")
        if not isinstance(_custom_providers, list):
            _custom_providers = []

    # ``model.context_length`` describes the configured default model; drop it when the
    # startup runtime (model or route) differs from that default.
    if _config_context_length is not None:
        _config_context_length = _scope_context_length_to_default_runtime(
            agent, _agent_cfg, _model_section, _custom_providers, _config_context_length, base_url
        )

    # Reused by _check_compression_model_feasibility (aux compression model detection).
    agent._custom_providers = _custom_providers
    _merge_custom_provider_extra_body(agent, _custom_providers)

    if _config_context_length is None and _custom_providers:
        with suppress(Exception):
            from hermes_cli.config import get_custom_provider_context_length
            _cp_ctx_resolved = get_custom_provider_context_length(
                model=agent.model, base_url=agent.base_url, custom_providers=_custom_providers
            )
            if _cp_ctx_resolved:
                _config_context_length = int(_cp_ctx_resolved)
        if _config_context_length is None:
            _warn_invalid_custom_provider_context_length(agent, _custom_providers)

    # Persisted for switch_model / fallback AFTER the custom_providers branch (per-model overrides).
    agent._config_context_length = _config_context_length

    _lmstudio_runtime_context_length = agent._ensure_lmstudio_runtime_loaded(_config_context_length)
    if agent._lmstudio_load_was_unverified(_lmstudio_runtime_context_length):
        _ra().logger.warning(
            "LM Studio model activation was rejected or completed without a "
            "verifiable active context length; falling back to configured context"
        )
    _effective_context_length = agent._effective_lmstudio_context_length(
        _config_context_length, _lmstudio_runtime_context_length,
    )
    return _config_context_length, _custom_providers, _effective_context_length, _model_cfg


def _select_context_engine(_agent_cfg):
    """Config-driven context engine: ``context.engine`` → plugins/context_engine/<name>/ →
    general plugin system → None (built-in ContextCompressor)."""
    _engine_name = "compressor"
    with suppress(Exception):
        _ctx_cfg = _agent_cfg.get("context", {}) if isinstance(_agent_cfg, dict) else {}
        _engine_name = _ctx_cfg.get("engine", "compressor") or "compressor"
    if _engine_name == "compressor":
        return None  # built-in; don't auto-activate plugins
    _selected_engine = None
    _copy_failed = False
    try:
        from plugins.context_engine import load_context_engine
        _selected_engine = load_context_engine(_engine_name)
    except Exception as _ce_load_err:
        _ra().logger.debug("Context engine load from plugins/context_engine/: %s", _ce_load_err)

    if _selected_engine is None:
        try:
            from hermes_cli.plugins import get_plugin_context_engine
            _candidate = get_plugin_context_engine()
        except Exception:
            _candidate = None
        if _candidate is not None and _candidate.name == _engine_name:
            # Deep-copy the shared singleton so a child's update_model() can't mutate the
            # parent's. Uncopyable state (locks, DB conns) → built-in with an ACCURATE message.
            import copy
            try:
                # Copy can fail for engines holding uncopyable state (locks, DB connections, clients); in
                # that case fall back to the built-in compressor with an ACCURATE message rather than
                # silently mislabelling it "not found". See #42449.
                _selected_engine = copy.deepcopy(_candidate)
            except Exception as _copy_err:
                _copy_failed = True
                _ra().logger.warning(
                    "Context engine '%s' could not be safely copied for this "
                    "agent (%s) — falling back to built-in compressor. Plugin "
                    "engines that hold uncopyable state (locks, DB connections) "
                    "should implement __deepcopy__ to copy only mutable budget "
                    "state.",
                    _engine_name, _copy_err,
                )

    if _selected_engine is None and not _copy_failed:
        _ra().logger.warning(
            "Context engine '%s' not found — falling back to built-in compressor", _engine_name
        )
    return _selected_engine


def _compressor_max_tokens(agent):
    """``agent.max_tokens``, or the native-Gemini adapter default when unset: generateContent
    still sends maxOutputTokens=65,535 and the threshold is pct×(window − max_tokens), so
    reserving 0 let the provider 400 before compaction fired."""
    if agent.max_tokens is not None:
        return agent.max_tokens
    with suppress(Exception):
        from agent.gemini_native_adapter import (
            GEMINI_DEFAULT_MAX_OUTPUT_TOKENS, is_native_gemini_base_url
        )
        _gemini_provider = str(agent.provider or "").strip().lower() in {
            "gemini", "google", "google-gemini", "google-ai-studio",
        }
        if _gemini_provider or is_native_gemini_base_url(agent.base_url):
            return GEMINI_DEFAULT_MAX_OUTPUT_TOKENS
    return None


def _build_context_engine(agent, _agent_cfg, cs, _custom_providers, _effective_context_length, session_db):
    _selected_engine = _select_context_engine(_agent_cfg)
    if _selected_engine is not None:
        agent.context_compressor = _selected_engine
        # External engines own compaction policy — the host threshold (and its Codex
        # autoraise) never reaches the plugin, so drop the notice.
        agent._compression_threshold_autoraised = None
        # External engines own compaction policy: the host compression threshold (including the Codex
        # gpt-5.5 autoraise above) only configures the built-in ContextCompressor and never reaches the
        # plugin, so the autoraise notice would announce a change that does not apply. (#44439)
        from agent.model_metadata import get_model_context_length
        _plugin_ctx_len = get_model_context_length(
            agent.model, base_url=agent.base_url, api_key=getattr(agent, "api_key", ""),
            config_context_length=_effective_context_length, provider=agent.provider,
            custom_providers=_custom_providers,
        )
        # Per-model overrides BEFORE the initial update_model() so the first threshold
        # resolution already sees them.
        if cs.model_thresholds:
            agent.context_compressor.model_thresholds = cs.model_thresholds
        agent.context_compressor.update_model(
            model=agent.model, context_length=_plugin_ctx_len, base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""), provider=agent.provider, api_mode=agent.api_mode,
        )
        if not agent.quiet_mode:
            _ra().logger.info("Using context engine: %s", _selected_engine.name)
    else:
        agent.context_compressor = ContextCompressor(
            model=agent.model, threshold_percent=cs.threshold, protect_first_n=cs.protect_first,
            protect_last_n=cs.protect_last, summary_target_ratio=cs.target_ratio,
            summary_model_override=None, quiet_mode=agent.quiet_mode, base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""), config_context_length=_effective_context_length,
            provider=agent.provider, api_mode=agent.api_mode,
            abort_on_summary_failure=cs.abort_on_summary_failure,
            max_tokens=_compressor_max_tokens(agent), model_thresholds=cs.model_thresholds,
            threshold_tokens_cap=cs.threshold_tokens,
            proactive_prune_tokens=cs.proactive_prune_tokens,
            proactive_prune_min_result_chars=cs.proactive_prune_min_chars,
            proactive_prune_min_reclaim_tokens=cs.proactive_prune_min_reclaim,
            min_tail_user_messages=cs.min_tail_users, tail_mode=cs.tail_mode,
        )
    _bind_session_state = getattr(agent.context_compressor, "bind_session_state", None)
    if callable(_bind_session_state):
        with suppress(Exception):
            _bind_session_state(session_db=session_db, session_id=agent.session_id)
    agent.compression_enabled = cs.enabled
    agent.compression_in_place = cs.in_place
    _cc = agent.context_compressor
    # Micro-compaction has no pre-compress checkpoint hook; suppress it while the gate is
    # armed (mirrors native_compaction.py).
    if cs.checkpoint_required and cs.micro_compact:
        logger.warning(
            "compression.checkpoint_required is enabled: post-turn "
            "micro-compaction is disabled for this agent so every lossy "
            "rewrite passes through the checkpoint-gated compressor."
        )
        cs.micro_compact = False
    for _attr, _value in (
        ("_micro_compact_enabled", cs.micro_compact),
        ("_micro_compact_every_n_turns", cs.micro_compact_every_n_turns),
        ("_micro_compact_defrag_threshold_tokens", cs.micro_compact_defrag_tokens),
    ):
        if hasattr(_cc, _attr):
            setattr(_cc, _attr, _value)
    agent.compression_checkpoint_required = cs.checkpoint_required
    agent.codex_app_server_auto_compaction = cs.codex_app_server_auto
    agent.codex_responses_native_compaction = cs.codex_responses_native
    agent.codex_responses_compact_threshold = cs.codex_responses_compact_threshold
    from agent.native_compaction import resolve_native_compaction_capabilities
    agent.runtime_capabilities = resolve_native_compaction_capabilities(
        model=agent.model, base_url=agent.base_url, provider=agent.provider,
        is_codex_backend=(agent.provider or "").strip().lower() == "openai-codex",
    )
    agent.max_compression_attempts = cs.max_attempts
    agent.compression_idle_compact_after_seconds = cs.idle_compact_after_seconds


def _enforce_minimum_context(agent):
    # Reject windows below the 64K floor needed for reliable tool-calling; an explicit
    # positive model.context_length on LM Studio is allowed below the floor.
    _ctx = getattr(agent.context_compressor, "context_length", 0)
    _allow_lmstudio_explicit_below_floor = (
        str(agent.provider or "").strip().lower() == "lmstudio"
        and isinstance(agent._config_context_length, int)
        and not isinstance(agent._config_context_length, bool)
        and agent._config_context_length > 0
    )
    if _ctx and _ctx < MINIMUM_CONTEXT_LENGTH and not _allow_lmstudio_explicit_below_floor:
        raise ValueError(
            f"Model {agent.model} has a context window of {_ctx:,} tokens, "
            f"which is below the minimum {MINIMUM_CONTEXT_LENGTH:,} required "
            f"by Hermes Agent.  Choose a model with at least "
            f"{MINIMUM_CONTEXT_LENGTH // 1000}K context.  If your server "
            f"reports a window smaller than the model's true window, set "
            f"model.context_length in config.yaml to the real value "
            f"(this must be at least {MINIMUM_CONTEXT_LENGTH // 1000}K)."
        )


def _warn_nonagentic_hermes_model(agent):
    # Nous Hermes 3/4 are chat models, not tool-call-tuned. cli.py show_banner() already
    # warns on the CLI, so skip platform=="cli"; non-quiet non-CLI surfaces still get it.
    if agent.quiet_mode or (agent.platform or "cli") == "cli":
        return
    with suppress(Exception):
        from hermes_cli.model_switch import _check_hermes_model_warning
        _hermes_warn = _check_hermes_model_warning(agent.model or "")
        if _hermes_warn:
            _user_msg = (
                "⚠ Nous Research Hermes 3 & 4 models are NOT agentic — they "
                "lack reliable tool-calling for agent workflows (delegation, "
                "cron, proactive tools). Consider an agentic model instead "
                "(Claude, GPT, Gemini, Qwen-Coder, etc.)."
            )
            agent._emit_warning(_user_msg)
            _ra().logger.warning(_hermes_warn)


def _inject_context_engine_tools(agent):
    # Context engine tool schemas (lcm_*), deduped against existing names (plugins may
    # register the same schemas; duplicates 400 provider-side) and gated on enabled_toolsets
    # so `platform_toolsets: telegram: []` can't leak them.
    # Skip names that are already present — the model_tools.get_tool_definitions() quiet_mode cache returned a
    # shared list pre-#17335, so a stray mutation here would poison subsequent agent inits in the same
    # Gateway process and trip provider-side 'duplicate tool name' errors. Even with the cache fix, dedup is
    # the right defense against plugin paths that may register the same schemas via ctx.register_tool().
    # Mirrors the memory tools dedup above. Respect the platform's enabled_toolsets configuration (#5544):
    # context engine tools follow the same gating pattern as memory provider tools — without the gate,
    # `platform_toolsets: telegram: []` would still leak lcm_* tools into the tool surface and incur the
    # same local-model latency penalty.
    agent._context_engine_tool_names: set = set()
    if (
        agent.context_compressor
        and agent.tools is not None
        and (agent.enabled_toolsets is None or "context_engine" in agent.enabled_toolsets)
    ):
        _existing_tool_names = {
            t.get("function", {}).get("name") for t in agent.tools if isinstance(t, dict)
        }
        from agent.memory_manager import normalize_tool_schema
        for _raw_schema in agent.context_compressor.get_tool_schemas():
            _schema = normalize_tool_schema(_raw_schema)
            if _schema is None:
                # A nameless tool makes strict providers 400 and disables the whole toolset.
                _ra().logger.warning(
                    # Skip it. See #47707.
                    "Context engine returned a tool schema with no resolvable "
                    "name; skipping to avoid poisoning the request (%r)",
                    _raw_schema,
                )
                continue
            _tname = _schema["name"]
            if _tname in _existing_tool_names:
                continue  # already registered via plugin/cache path
            agent.tools.append({"type": "function", "function": _schema})
            for _names in (agent.valid_tool_names, agent._context_engine_tool_names, _existing_tool_names):
                _names.add(_tname)

    if agent.context_compressor:
        try:
            agent.context_compressor.on_session_start(
                agent.session_id, hermes_home=str(get_hermes_home()),
                platform=agent.platform or "cli", model=agent.model,
                context_length=getattr(agent.context_compressor, "context_length", 0),
                conversation_id=getattr(agent, "_gateway_session_key", None),
            )
        except Exception as _ce_err:
            _ra().logger.debug("Context engine on_session_start: %s", _ce_err)


def _configure_ollama_num_ctx(agent, _model_cfg, _config_context_length):
    # Ollama defaults num_ctx to 2048, so detect the max window and send num_ctx per request.
    # model.ollama_num_ctx overrides; model.context_length caps the detected value (VRAM).
    agent._ollama_num_ctx: int | None = None
    _override = _model_cfg.get("ollama_num_ctx") if isinstance(_model_cfg, dict) else None
    if _override is not None:
        try:
            agent._ollama_num_ctx = int(_override)
        except (TypeError, ValueError):
            _ra().logger.debug("Invalid ollama_num_ctx config value: %r", _override)
    if agent._ollama_num_ctx is None and agent.base_url and is_local_endpoint(agent.base_url):
        try:
            # api_key may be a callable (Entra token provider); detection needs a string.
            _key = agent.api_key if isinstance(agent.api_key, str) else ""
            _detected = query_ollama_num_ctx(agent.model, agent.base_url, api_key=_key or "")
            if _detected and _detected > 0:
                agent._ollama_num_ctx = _detected
        except Exception as exc:
            _ra().logger.debug("Ollama num_ctx detection failed: %s", exc)
    # Cap auto-detected num_ctx to the explicit context_length (GGUF metadata can advertise
    # 256K+ and Ollama would allocate that much VRAM); never override an explicit num_ctx.
    if (
        agent._ollama_num_ctx
        and _config_context_length
        and _override is None
        and agent._ollama_num_ctx > _config_context_length
    ):
        _ra().logger.info(
            "Ollama num_ctx capped: %d -> %d (model.context_length override)",
            agent._ollama_num_ctx, _config_context_length,
        )
        agent._ollama_num_ctx = _config_context_length
    if agent._ollama_num_ctx and not agent.quiet_mode:
        _ra().logger.info(
            "Ollama num_ctx: will request %d tokens (model max from /api/show)",
            agent._ollama_num_ctx,
        )
    # Recalibrate the compressor to the served window: every request runs at num_ctx, so a
    # trigger derived from the probed model window could sit above it and never fire.
    # A config that sets only model.ollama_num_ctx (without model.context_length) previously left the
    # compressor targeting the probed window while the server truncated/rejected at num_ctx — the compaction
    # trigger could sit several times ABOVE the real served window and never fire. Clamp the compressor's
    # window to the effective num_ctx so threshold math operates on the context the server actually serves.
    # (Overlaps #60103's silent-clamp dead zone; this is the init-order half.)
    _cc_window = getattr(agent.context_compressor, "context_length", 0) or 0
    if agent._ollama_num_ctx and agent._ollama_num_ctx > 0 and _cc_window and agent._ollama_num_ctx < _cc_window:
        _ra().logger.info(
            "Compressor window clamped to Ollama num_ctx: %d -> %d",
            _cc_window, agent._ollama_num_ctx,
        )
        agent.context_compressor.update_model(
            model=agent.model, context_length=agent._ollama_num_ctx, base_url=agent.base_url,
            api_key=getattr(agent, "api_key", ""), provider=agent.provider, api_mode=agent.api_mode,
        )


def _emit_compression_summary(agent, cs):
    # Codex autoraise notice: once per profile/config state (persisted marker; the gateway
    # rebuilds the agent per message). The display gate hides the banner, not the autoraise.
    _autoraise = agent._compression_threshold_autoraised or {}
    _autoraise_notice = None
    if (
        # A change in the raised threshold (or the autoraised model) updates the marker state and
        # re-notifies once. The config display gate (compression.codex_gpt55_autoraise_notice) still
        # suppresses the banner entirely without disabling the threshold autoraise. See #54432.
        bool(_autoraise)
        and cs.enabled
        and cs.autoraise_notice_enabled
        and not _codex_gpt55_autoraise_notice_seen(_autoraise)
    ):
        _autoraise_notice = _build_codex_gpt5_autoraise_notice(
            _autoraise, context_length=getattr(agent.context_compressor, "context_length", None)
        )

    if not agent.quiet_mode:
        _cc = agent.context_compressor
        if cs.enabled:
            # The active engine's own threshold — a plugin's differs from cs.threshold.
            _pct = getattr(_cc, "threshold_percent", cs.threshold)
            _cap = getattr(_cc, "threshold_tokens_cap", None)
            _cap_note = f" (capped at {_cap:,} tokens)" if _cap and _cap > 0 else ""
            print(f"📊 Context limit: {_cc.context_length:,} tokens (compress at {int(_pct*100)}% = {_cc.threshold_tokens:,}{_cap_note})")
        else:
            print(f"📊 Context limit: {_cc.context_length:,} tokens (auto-compression disabled)")
        # Gateway users get the same text via _compression_warning on turn 1.
        if _autoraise_notice:
            print(_autoraise_notice)

    # status_callback isn't wired yet: stash for replay on the first turn; mark shown so
    # repeated inits stay silent.
    agent._compression_warning = _autoraise_notice
    if _autoraise_notice:
        _record_codex_gpt55_autoraise_notice(_autoraise)
    # Feasibility check deferred to the first turn near threshold (eager costs ~400ms cold).
    agent._compression_feasibility_checked = False


def _snapshot_primary_runtime(agent):
    # Per-turn restoration snapshot: after a fallback, the next turn restores these so the
    # preferred model gets a fresh attempt.
    _cc = agent.context_compressor
    agent._primary_runtime = {
        "model": agent.model,
        "provider": agent.provider,
        "requested_provider": agent.requested_provider,
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "api_key": getattr(agent, "api_key", ""),
        "request_overrides": dict(getattr(agent, "request_overrides", {}) or {}),
        "client_kwargs": dict(agent._client_kwargs),
        "use_prompt_caching": agent._use_prompt_caching,
        "use_native_cache_layout": agent._use_native_cache_layout,
        "reasoning_echo_flag": getattr(agent, "_reasoning_echo_flag", False),
        # Engine state _try_activate_fallback() overwrites (getattr: plugin engines may lack them).
        "compressor_model": getattr(_cc, "model", agent.model),
        "compressor_base_url": getattr(_cc, "base_url", agent.base_url),
        "compressor_api_key": getattr(_cc, "api_key", ""),
        "compressor_provider": getattr(_cc, "provider", agent.provider),
        "compressor_context_length": _cc.context_length,
        "compressor_threshold_tokens": _cc.threshold_tokens,
    }
    if agent.api_mode == "anthropic_messages":
        agent._primary_runtime.update({
            "anthropic_api_key": agent._anthropic_api_key,
            "anthropic_base_url": agent._anthropic_base_url,
            "is_anthropic_oauth": agent._is_anthropic_oauth,
        })


def _init_usage_state(agent):
    from agent.runtime_cwd import scope_terminal_cwd
    agent._subdirectory_hints = SubdirectoryHintTracker(working_dir=scope_terminal_cwd() or None)
    _set_defaults(agent, _USAGE_STATE)


# Per-session usage accounting.
_USAGE_STATE: Dict[str, Any] = {
    "_user_turn_count": 0,
    "_is_user_initiated_turn": False,  # Copilot x-initiator: first call of a user turn = "user"
    # Usage anchors (agent/model_metadata.py): last response's exact usage + transcript
    # snapshot; invalidated on compaction/session switch so stale anchors never suppress compression.
    "_usage_anchor": None,
    "_turn_base_usage_anchor": None,
    # Cumulative token usage for the session
    "session_prompt_tokens": 0,
    "session_completion_tokens": 0,
    "session_total_tokens": 0,
    "session_api_calls": 0,
    "session_input_tokens": 0,
    "session_output_tokens": 0,
    "session_cache_read_tokens": 0,
    "session_cache_write_tokens": 0,
    "session_reasoning_tokens": 0,
    "session_estimated_cost_usd": 0.0,
    "session_cost_status": "unknown",
    "session_cost_source": "none",
    # Status-bar latency/velocity history (last 10 calls), shared by loop + codex_runtime.
    "_api_latency_history": lambda: deque(maxlen=10),
    "_api_output_history": lambda: deque(maxlen=10),
}

# Constructor params stored verbatim under the same name.
_PASSTHROUGH_PARAMS = (
    "model", "max_iterations", "save_trajectories", "verbose_logging", "quiet_mode",
    "tool_progress_mode", "ephemeral_system_prompt", "platform", "skip_context_files",
    "load_soul_identity", "pass_session_id", "log_prefix_chars",
    # OpenRouter provider preferences
    "providers_allowed", "providers_ignored", "providers_order", "provider_sort",
    "provider_require_parameters", "provider_data_collection", "openrouter_min_coding_score",
    # Toolset filtering
    "enabled_toolsets", "disabled_toolsets",
    # Model response configuration (None = provider/model default)
    "max_tokens", "reasoning_config", "service_tier",
)
# Gateway identity params stored as ``agent._<name>``. gateway_session_key is the stable
# per-chat key (e.g. agent:main:telegram:dm:123).
_GATEWAY_IDENTITY_PARAMS = (
    "user_id", "user_id_alt", "user_name", "chat_id", "chat_name", "chat_type", "thread_id",
    "gateway_session_key",
)
_CALLBACK_PARAMS = (
    "tool_progress_callback", "tool_start_callback", "tool_complete_callback",
    "thinking_callback", "reasoning_callback", "clarify_callback",
    "read_terminal_callback", "read_preview_callback", "drive_preview_callback",
    "read_window_below_callback", "setup_mcp_callback", "tour_callback",
    "step_callback", "stream_delta_callback", "interim_assistant_callback",
    "status_callback", "notice_callback", "notice_clear_callback",
    "event_callback", "reaction_callback", "tool_gen_callback",
)


def init_agent(
    agent, base_url: str = None, api_key: str = None, provider: str = None, api_mode: str = None,
    acp_command: str = None, acp_args: list[str] | None = None, command: str = None,
    args: list[str] | None = None, model: str = "", max_iterations: int = sys.maxsize,
    enabled_toolsets: List[str] = None, disabled_toolsets: List[str] = None,
    save_trajectories: bool = False, verbose_logging: bool = False, quiet_mode: bool = False,
    tool_progress_mode: str = "all", ephemeral_system_prompt: str = None,
    log_prefix_chars: int = 100, log_prefix: str = "", providers_allowed: List[str] = None,
    providers_ignored: List[str] = None, providers_order: List[str] = None,
    provider_sort: str = None, provider_require_parameters: bool = False,
    provider_data_collection: str = None, openrouter_min_coding_score: Optional[float] = None,
    session_id: str = None, tool_progress_callback: callable = None,
    tool_start_callback: callable = None, tool_complete_callback: callable = None,
    thinking_callback: callable = None, reasoning_callback: callable = None,
    clarify_callback: callable = None, read_terminal_callback: callable = None,
    read_preview_callback: callable = None, drive_preview_callback: callable = None,
    read_window_below_callback: callable = None, setup_mcp_callback: callable = None,
    tour_callback: callable = None, step_callback: callable = None,
    stream_delta_callback: callable = None, interim_assistant_callback: callable = None,
    tool_gen_callback: callable = None, status_callback: callable = None,
    notice_callback: callable = None, notice_clear_callback: callable = None,
    event_callback: Optional[Callable[[str, dict], None]] = None,
    reaction_callback: Optional[Callable[[str], None]] = None, max_tokens: int = None,
    reasoning_config: Dict[str, Any] = None, service_tier: str = None,
    request_overrides: Dict[str, Any] = None, prefill_messages: List[Dict[str, Any]] = None,
    platform: str = None, user_id: str = None, user_id_alt: str = None, user_name: str = None,
    chat_id: str = None, chat_name: str = None, chat_type: str = None, thread_id: str = None,
    gateway_session_key: str = None, skip_context_files: bool = False,
    load_soul_identity: bool = False, skip_memory: bool = False,
    skip_background_review: bool = False, session_db=None, parent_session_id: str = None,
    iteration_budget: "IterationBudget" = None, run_budget_seconds: Optional[float] = None,
    fallback_model: Dict[str, Any] = None, credential_pool=None, checkpoints_enabled: bool = False,
    checkpoint_max_snapshots: int = 20, checkpoint_max_total_size_mb: int = 500,
    checkpoint_max_file_size_mb: int = 10, pass_session_id: bool = False,
    requested_provider: str = None, capabilities: Optional[Dict[str, bool]] = None,
):
    """Initialize the AI Agent (body of :meth:`AIAgent.__init__`).

    Non-obvious parameters:
      max_iterations: default unlimited (sys.maxsize); the budget is shared with subagents.
      requested_provider: provider identity before runtime canonicalization.
      openrouter_min_coding_score: coding-score floor for ``openrouter/pareto-code`` only.
      clarify_callback: ``(question, choices) -> str``; None → the clarify tool errors.
      reasoning_config: None → ``{"enabled": True, "effort": "medium"}`` on OpenRouter.
      prefill_messages: priming history. Anthropic Sonnet/Opus 4.6+ 400 on a trailing
        assistant message — use structured outputs there instead.
      skip_context_files: skip SOUL.md/.hermes.md/AGENTS.md/CLAUDE.md/.cursorrules injection;
        load_soul_identity keeps ~/.hermes/SOUL.md as identity regardless.
    """
    _install_safe_stdio()

    _params = locals()
    for _name in _PASSTHROUGH_PARAMS:
        setattr(agent, _name, _params[_name])
    for _name in _GATEWAY_IDENTITY_PARAMS:
        setattr(agent, f"_{_name}", _params[_name])
    # Shared iteration budget: parent creates, children inherit.
    agent.iteration_budget = iteration_budget or IterationBudget(max_iterations)
    # CLI replaces this with _cprint so raw ANSI status lines go through prompt_toolkit's
    # renderer (StdoutProxy would mangle them). None = builtins.print.
    agent._print_fn = None
    agent.background_review_callback = None  # Optional sync callback for gateway delivery
    agent.memory_notifications = "on"  # Memory update notifications: "off", "on", "verbose"
    # Skips the end-of-turn review fork (~30K tokens/event); one switch for both review paths.
    agent.skip_background_review = bool(skip_background_review)
    agent.log_prefix = f"{log_prefix} " if log_prefix else ""
    # Effective base URL for feature detection (prompt caching, reasoning, etc.)
    agent.base_url = base_url or ""
    provider_name = provider.strip().lower() if isinstance(provider, str) and provider.strip() else None
    agent.provider = provider_name or ""
    agent.requested_provider = (
        requested_provider.strip().lower()
        if isinstance(requested_provider, str) and requested_provider.strip()
        else agent.provider
    )
    agent.capabilities = {
        key: value for key, value in (capabilities or {}).items()
        if isinstance(key, str) and isinstance(value, bool)
    }
    agent._credential_pool = credential_pool
    agent.acp_command = acp_command or command
    agent.acp_args = list(acp_args or args or [])
    _resolve_api_mode(agent, api_mode, provider_name, base_url)
    _finalize_routing(agent, api_mode, credential_pool)

    # Platform callbacks are stored under their parameter names verbatim.
    for _cb in _CALLBACK_PARAMS:
        setattr(agent, _cb, _params[_cb])
    agent.suppress_status_output = False

    _set_defaults(agent, _CONTROL_STATE)

    # reasoning_content echo opt-in; switch_model / fallback / restore keep it in sync.
    agent._reasoning_echo_flag = agent._read_reasoning_echo_from_config()
    agent.request_overrides = dict(request_overrides or {})
    agent.prefill_messages = prefill_messages or []  # Prefilled conversation turns
    agent._force_ascii_payload = False

    _init_prompt_cache_config(agent)
    _init_turn_state(agent, run_budget_seconds)
    _setup_logging(agent)
    _set_defaults(agent, _STREAM_STATE)
    _build_client(agent, api_key, base_url, fallback_model)
    _init_fallback_chain(agent, fallback_model)
    _load_tools(agent, enabled_toolsets, disabled_toolsets)
    _init_session_state(
        agent, session_id, session_db, parent_session_id, reasoning_config, max_tokens,
        checkpoints_enabled, checkpoint_max_snapshots, checkpoint_max_total_size_mb, checkpoint_max_file_size_mb,
    )

    # Load config once for memory, skills, and compression sections
    try:
        from hermes_cli.config import load_config_readonly as _load_agent_config
        _agent_cfg = _load_agent_config()
    except Exception:
        _agent_cfg = {}

    _apply_display_config(agent, _agent_cfg, platform)
    _init_memory(agent, _agent_cfg, skip_memory, platform)
    _apply_agent_section(agent, _agent_cfg)
    cs = _parse_compression_config(agent, _agent_cfg)
    _config_context_length, _custom_providers, _effective_context_length, _model_cfg = _resolve_context_length(
        agent, _agent_cfg, base_url
    )
    _build_context_engine(agent, _agent_cfg, cs, _custom_providers, _effective_context_length, session_db)
    _enforce_minimum_context(agent)
    _warn_nonagentic_hermes_model(agent)
    _inject_context_engine_tools(agent)
    _init_usage_state(agent)
    _configure_ollama_num_ctx(agent, _model_cfg, _config_context_length)
    _emit_compression_summary(agent, cs)
    _snapshot_primary_runtime(agent)


__all__ = ["init_agent"]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'ToolGuardrailDecision': ('agent.tool_guardrails', 'ToolGuardrailDecision'),
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
